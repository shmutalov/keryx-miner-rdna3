#![cfg_attr(all(test, feature = "bench"), feature(test))]

use std::env::consts::DLL_EXTENSION;
use std::env::current_exe;
use std::error::Error as StdError;
use std::ffi::OsStr;

use clap::{App, FromArgMatches, IntoApp};
use keryx_miner::PluginManager;
use log::{error, info};
use rand::{thread_rng, RngCore};
use std::fs;
use std::sync::atomic::AtomicU16;
use std::sync::Arc;
use std::time::Duration;

use crate::cli::Opt;
use crate::client::grpc::KeryxdHandler;
use crate::client::stratum::StratumHandler;
use crate::client::Client;
use crate::miner::MinerManager;
use crate::target::Uint256;

mod cli;
mod client;
mod escrow;
mod ipfs;
mod keryxd_messages;
mod miner;
mod pow;
mod target;
mod watch;

// RDNA3 fork ships NO cdylib PoW plugins — PoW/PoM run in the in-process Vulkan worker
// (see vulkan_worker.rs, registered in real_main). The whitelist is intentionally EMPTY so a
// stray official-miner plugin left in the install dir is NEVER dlopen'd. This matters because
// the fork's binary is `keryx-miner-rdna3` but users sometimes unpack it over an existing
// official `keryx-miner` folder (e.g. /hive/miners/custom/keryx-miner), leaving a stale
// `libkeryxopencl.so`/`libkeryxcuda.so` behind. Such a plugin is built against a different
// clap/plugin ABI and aborts the process on a TypeId downcast mismatch the moment it loads.
const WHITELIST: [&str; 0] = [];

pub mod proto {
    #![allow(clippy::derive_partial_eq_without_eq)]
    tonic::include_proto!("protowire");
    // include!("protowire.rs"); // FIXME: https://github.com/intellij-rust/intellij-rust/issues/6579
}

pub type Error = Box<dyn StdError + Send + Sync + 'static>;

type Hash = Uint256;

#[cfg(target_os = "windows")]
fn adjust_console() -> Result<(), Error> {
    let console = win32console::console::WinConsole::input();
    let mut mode = console.get_mode()?;
    mode = (mode & !win32console::console::ConsoleMode::ENABLE_QUICK_EDIT_MODE)
        | win32console::console::ConsoleMode::ENABLE_EXTENDED_FLAGS;
    console.set_mode(mode)?;
    Ok(())
}

/// Install a SIGINT/SIGTERM (Ctrl-C on Windows) handler so the miner shuts down cleanly when a
/// supervisor (HiveOS / systemd / start-miner.bat) stops it, instead of only dying on SIGKILL. The
/// process holds long GPU dispatches and a resident PoM weight blob; a prompt `exit(0)` lets the OS
/// reclaim the Vulkan context and returns success so supervisors don't record a crash. `exit` skips
/// `WeightIndex::drop` (the pom-tree cleanup), but `build_from_gguf` already sweeps every stale
/// `pom-tree-*.bin` on the next start, so a skipped Drop does not permanently leak a tree.
fn spawn_shutdown_handler() {
    tokio::spawn(async {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut term = match signal(SignalKind::terminate()) {
                Ok(s) => s,
                Err(e) => {
                    log::warn!("Could not install SIGTERM handler: {} — only SIGKILL will stop the miner.", e);
                    return;
                }
            };
            tokio::select! {
                _ = tokio::signal::ctrl_c() => info!("Received SIGINT (Ctrl-C) — shutting down."),
                _ = term.recv() => info!("Received SIGTERM — shutting down."),
            }
        }
        #[cfg(not(unix))]
        {
            if tokio::signal::ctrl_c().await.is_err() {
                log::warn!("Could not listen for the Ctrl-C shutdown signal.");
                return;
            }
            info!("Received Ctrl-C — shutting down.");
        }
        std::process::exit(0);
    });
}

fn filter_plugins(dirname: &str) -> Vec<String> {
    match fs::read_dir(dirname) {
        Ok(readdir) => readdir
            .map(|entry| entry.unwrap().path())
            .filter(|fname| {
                fname.is_file()
                    && fname.extension().is_some()
                    && fname.extension().and_then(OsStr::to_str).unwrap_or_default().starts_with(DLL_EXTENSION)
            })
            .filter(|fname| WHITELIST.iter().any(|lib| *lib == fname.file_stem().and_then(OsStr::to_str).unwrap()))
            .map(|path| path.to_str().unwrap().to_string())
            .collect::<Vec<String>>(),
        _ => Vec::<String>::new(),
    }
}

/// Warn if GPU 0's VRAM is too small for the selected model tier (queried via Vulkan — the same
/// device the miner mines/serves on). Non-fatal: a host/CPU path can still serve it, so warn
/// rather than error. The capability gate (`filter_specs_by_vram`) does the announce-time drop;
/// this is just an upfront, tier-labelled heads-up.
///
/// VRAM requirements (GGUF weights only, not counting GPU workspace):
///   Qwen3-8B-ablit. →  ~4.6 GB  (Q4_K_S — H5 tier 0, requires ≥6 GB card)
///   Mistral-7B-v0.3 →  ~5.9 GB  (Q6_K — requires ≥8 GB card)
///   GLM-4-9B-0414   →  ~8.3 GB  (Q6_K — requires ≥12 GB card)
///   Qwen3.6-27B     → ~16.5 GB  (requires ≥24 GB card)
///   Kimi-Linear-48B → ~29.7 GB  (requires ≥32 GB card)
fn check_gpu_vram_for_tier(needs_high: bool, needs_very_high: bool) {
    let Some(vram_mb) = query_vram_mb() else { return };

    let (model_label, min_vram_mb): (&str, u64) = if needs_very_high {
        ("Kimi-Linear-48B (--very-high)", 30_000)
    } else if needs_high {
        ("Qwen3.6-27B (--high)", 24_000)
    } else {
        ("GLM-4-9B-0414 (default)", 12_000)
    };

    if vram_mb < min_vram_mb {
        log::warn!(
            "⚠  {} needs ≥{} GB VRAM but only {} GB on this GPU — GPU inference for this tier \
             will OOM. Use a smaller tier (--light Mistral-7B / --very-light Qwen3-8B) or \
             serve it via a host/CPU path.",
            model_label,
            min_vram_mb / 1024,
            vram_mb / 1024,
        );
    } else {
        log::info!("GPU: {} MB VRAM — ready for {}", vram_mb, model_label);
    }
}

/// Total VRAM (MB) of the INFERENCE Vulkan device (the one llama-server is pinned to on
/// multi-GPU rigs), or None when no usable Vulkan device is present. Memoized: the probe spins up
/// a transient Vulkan device, so the result is cached across the two lineup capability-gate calls.
fn query_vram_mb() -> Option<u64> {
    static VRAM_MB: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *VRAM_MB.get_or_init(|| keryx_vulkan::probe_vram_mb_for(keryx_vulkan::inference_device_index()))
}

/// OPoI capability gate (layer A): drop the models this machine cannot actually
/// serve on GPU 0, so the `ai:cap` announcement never promises a model the miner
/// would fail to load. Skipped when no Vulkan device can be queried (CPU-fallback
/// setups keep working), or when `KERYX_SKIP_VRAM_GATE=1` forces every staged model
/// through (testing: the download + possession-index build/R_T check work regardless
/// of VRAM, but an over-VRAM model still fails at engine load and cannot mine).
fn filter_specs_by_vram(
    specs: &'static [&'static keryx_miner::models::ModelSpec],
) -> &'static [&'static keryx_miner::models::ModelSpec] {
    if std::env::var("KERYX_SKIP_VRAM_GATE").ok().as_deref() == Some("1") {
        log::warn!(
            "KERYX_SKIP_VRAM_GATE=1 — VRAM capability gate disabled; announcing + downloading all \
             staged models. An over-VRAM model will OOM at engine load and cannot mine its tier."
        );
        return specs;
    }
    let Some(gpu0_mb) = query_vram_mb() else {
        log::warn!("Cannot query GPU VRAM (no Vulkan device) — skipping the model capability gate.");
        return specs;
    };
    let kept: Vec<&'static keryx_miner::models::ModelSpec> = specs
        .iter()
        .copied()
        .filter(|spec| {
            // Gate on the ASSIGNMENT floor, not `min_vram_mb`: the H5 tier-0 Qwen3-8B loads in
            // ~5,409 MiB, so a 6 GB card reporting slightly under its 6000 MB min_vram must still
            // keep tier 0 rather than be dropped into "cannot mine at all".
            let floor = keryx_miner::models::pom_assignment_floor_mb(spec);
            if floor <= gpu0_mb {
                true
            } else {
                log::warn!(
                    "✗  '{}' needs ≥{} MB VRAM (assignment floor {} MB) but only {} MB on GPU 0 — model NOT announced (ai:cap) and not downloaded.",
                    spec.name,
                    spec.min_vram_mb,
                    floor,
                    gpu0_mb,
                );
                false
            }
        })
        .collect();
    if kept.len() == specs.len() {
        specs
    } else {
        // Leaked once at startup to keep the &'static API of init_supported.
        Box::leak(kept.into_boxed_slice())
    }
}

async fn get_client(
    keryxd_address: String,
    mining_address: String,
    worker: String,
    password: String,
    mine_when_not_synced: bool,
    block_template_ctr: Arc<AtomicU16>,
    escrow_privkey: Option<String>,
    escrow_state_file: String,
    ipfs_url: String,
) -> Result<Box<dyn Client + 'static>, Error> {
    if keryxd_address.starts_with("stratum+tcp://") {
        let (_schema, address) = keryxd_address.split_once("://").unwrap();
        Ok(StratumHandler::connect(
            address.to_string().clone(),
            mining_address.clone(),
            worker,
            password,
            mine_when_not_synced,
            Some(block_template_ctr.clone()),
            ipfs_url.clone(),
        )
        .await?)
    } else if keryxd_address.starts_with("grpc://") {
        Ok(KeryxdHandler::connect(
            keryxd_address.clone(),
            mining_address.clone(),
            mine_when_not_synced,
            Some(block_template_ctr.clone()),
            escrow_privkey,
            escrow_state_file,
            ipfs_url,
        )
        .await?)
    } else {
        Err("Did not recognize pool/grpc address schema".into())
    }
}

async fn client_main(
    opt: &Opt,
    block_template_ctr: Arc<AtomicU16>,
    plugin_manager: &PluginManager,
    escrow_privkey: Option<String>,
) -> Result<(), Error> {
    // IPFS is only needed to serve/fetch OPoI model files; skip it in PoW-only test mode.
    if !keryx_miner::pow_only() {
        let ipfs_url = opt.ipfs_url.clone();
        tokio::task::spawn_blocking(move || crate::ipfs::ensure_daemon(&ipfs_url)).await.ok();
    }

    let mut client = get_client(
        opt.keryxd_address.clone(),
        opt.mining_address.clone().unwrap_or_default(),
        opt.worker.clone(),
        opt.password.clone(),
        opt.mine_when_not_synced,
        block_template_ctr.clone(),
        escrow_privkey,
        opt.escrow_state_file.clone(),
        opt.ipfs_url.clone(),
    )
    .await?;

    if opt.devfund_percent > 0 {
        client.add_devfund(opt.devfund_address.clone(), opt.devfund_percent);
    }
    client.register().await?;
    let mut miner_manager = MinerManager::new(client.get_block_channel(), opt.num_threads, plugin_manager);
    client.listen(&mut miner_manager).await?;
    drop(miner_manager);
    Ok(())
}

/// Tokio async worker count. The miner's async workload is tiny (one gRPC/stratum connection +
/// a few tasks and timers), so we cap workers instead of spawning one per logical CPU — dozens of
/// idle executor threads on a many-core rig are pure scheduler overhead. Override with
/// KERYX_ASYNC_WORKERS.
fn tokio_worker_threads() -> usize {
    std::env::var("KERYX_ASYNC_WORKERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2)
        .clamp(1, 8)
}

/// Optional cap for the `spawn_blocking` pool (SLM inference, IPFS upload, model prefetch). Only
/// applied when KERYX_BLOCKING_THREADS is set: the blocking pool spawns lazily and idles out, so
/// tokio's default costs nothing at rest and capping it low would bottleneck parallel multi-model
/// prefetch on multi-GPU rigs.
fn tokio_blocking_threads() -> Option<usize> {
    std::env::var("KERYX_BLOCKING_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|n| n.clamp(2, 64))
}

fn main() -> Result<(), Error> {
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.worker_threads(tokio_worker_threads()).enable_all();
    if let Some(n) = tokio_blocking_threads() {
        builder.max_blocking_threads(n);
    }
    let rt = builder.build()?;
    rt.block_on(run())
}

async fn run() -> Result<(), Error> {
    #[cfg(target_os = "windows")]
    adjust_console().unwrap_or_else(|e| {
        eprintln!("WARNING: Failed to protect console ({}). Any selection in console will freeze the miner.", e)
    });
    let mut path = current_exe().unwrap_or_default();
    path.pop(); // Getting the parent directory
    let plugins = filter_plugins(path.to_str().unwrap_or("."));
    let (app, mut plugin_manager): (App, PluginManager) = keryx_miner::load_plugins(Opt::into_app(), &plugins)?;

    let matches = app.get_matches();

    // RDNA3 fork: register the in-process Vulkan PoW worker (no cdylib CUDA/OpenCL plugins shipped).
    plugin_manager.register(Box::new(keryx_miner::vulkan_worker::VulkanPlugin::new()));

    let worker_count = plugin_manager.process_options(&matches)?;
    let mut opt: Opt = Opt::from_arg_matches(&matches)?;
    opt.process()?;
    // GPU-only by default: the 7900 XT does PoW/PoM via Vulkan, so launch no CPU mining threads
    // unless the user explicitly asks for them with --mining-threads.
    if opt.num_threads.is_none() {
        opt.num_threads = Some(0);
    }
    // Model storage root is configurable: an explicit --models-dir wins over an inherited
    // KERYX_MODELS_DIR (h-run.sh exports it on HiveOS so models live outside the package dir
    // and survive upgrades). slm::model_dir reads the env var.
    if let Some(dir) = opt.models_dir.as_ref() {
        std::env::set_var("KERYX_MODELS_DIR", dir);
    }
    env_logger::builder().filter_level(opt.log_level()).parse_default_env().init();
    // Shut down cleanly on SIGINT/SIGTERM (Ctrl-C on Windows) — supervisors stop the miner with
    // SIGTERM, which previously had no handler and required SIGKILL.
    spawn_shutdown_handler();
    info!("=================================================================================");
    info!("              Keryx-Miner-RDNA3 GPU {}", env!("CARGO_PKG_VERSION"));
    info!(" Mining for: {}", opt.mining_address.as_deref().unwrap_or("(recovery mode)"));
    info!("=================================================================================");
    if let Ok(dir) = std::env::var("KERYX_MODELS_DIR") {
        info!("Models directory: {}", dir);
    }

    // Recovery mode: rebuild escrow_state.json from the Keryx public API, then exit.
    // Must run before escrow key loading to avoid creating a new random key on disk.
    // Uses escrow.key to derive the pubkey — only claimable UTXOs are returned.
    if opt.recover_escrow {
        let escrow_privkey = match escrow::load_key(&opt.escrow_key_file) {
            Ok(k) => k,
            Err(e) => {
                error!("{}", e);
                return Err(e.into());
            }
        };
        let pubkey_hex = match escrow::pubkey_hex_from_privkey(&escrow_privkey) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to derive pubkey from escrow key: {}", e);
                return Err(e.into());
            }
        };
        let url = format!("{}/api/v1/escrow/{}", opt.recover_escrow_api.trim_end_matches('/'), pubkey_hex);
        info!("Querying escrow UTXOs from {}", url);

        #[derive(serde::Deserialize)]
        struct ApiEscrowEntry {
            coinbase_txid: String,
            block_hash: String,
            confirm_daa: i64,
            amount_sompi: i64,
            output_index: i64,
        }

        let url_clone = url.clone();
        let api_entries: Vec<ApiEscrowEntry> = tokio::task::spawn_blocking(move || {
            let response = ureq::get(&url_clone)
                .call()
                .map_err(|e| format!("HTTP request failed: {}", e))?;
            serde_json::from_reader::<_, Vec<ApiEscrowEntry>>(response.into_reader())
                .map_err(|e| format!("JSON parse error: {}", e))
        })
        .await
        .map_err(|e| format!("spawn_blocking failed: {}", e))??;

        let entries: Vec<escrow::EscrowEntry> = api_entries
            .into_iter()
            .map(|a| escrow::EscrowEntry {
                coinbase_txid: a.coinbase_txid,
                block_hash: a.block_hash,
                confirm_daa: a.confirm_daa as u64,
                amount_sompi: a.amount_sompi as u64,
                output_index: a.output_index as u32,
                claimed: false,
                slashed: false,
                orphan_slashed: false,
                orphan_retries: 0,
                orphan_retry_after_daa: None,
                submit_retries: 0,
                batch_cap: 0,
                is_inference: false,
            })
            .collect();

        let total_sompi: u64 = entries.iter().map(|e| e.amount_sompi).sum();
        let count = entries.len();
        let state = escrow::EscrowState { entries };
        let json = serde_json::to_string_pretty(&state)?;
        fs::write(&opt.escrow_state_file, &json)?;

        info!(
            "Recovered {} escrow entries — claimable: {:.4} KRX",
            count,
            total_sompi as f64 / 1e8
        );
        info!("State saved to '{}'.", opt.escrow_state_file);
        return Ok(());
    }

    // Resolve OPoI escrow private key (once, before the reconnect loop).
    let escrow_privkey: Option<String> = match escrow::load_or_generate_key(&opt.escrow_key_file) {
        Ok(k) => {
            info!("OPoI: escrow key loaded from '{}'.", opt.escrow_key_file);
            Some(k)
        }
        Err(e) => {
            error!("Failed to load/generate OPoI escrow key: {}", e);
            return Err(e.into());
        }
    };

    // Phase-3 OPoI / PoM: load inference models before mining starts. Under PoM each tier
    // mines AND serves exactly ONE model (1 GPU = 1 tier); multi-tier coverage is a network
    // property, not a per-GPU one. H4 lineup:
    //   --very-light → Qwen3-8B-ablit.  (PoM tier 0, H5)
    //   --light      → Mistral-7B-v0.3  (tier 1)
    //   (no flag)    → GLM-4-9B-0414    (tier 2) [default]
    //   --high       → Qwen3.6-27B      (tier 3)
    //   --very-high  → Kimi-Linear-48B  (tier 4)

    // Warn if GPU 0's VRAM is too small for the selected model tier (Vulkan-queried).
    check_gpu_vram_for_tier(opt.high || opt.very_high, opt.very_high);

    let tier = if opt.very_high {
        info!("--very-high mode: top tier — mines Kimi-Linear-48B under PoM.");
        keryx_miner::models::Tier::VeryHigh
    } else if opt.high {
        info!("--high mode: high tier — mines Qwen3.6-27B under PoM.");
        keryx_miner::models::Tier::High
    } else if opt.light {
        info!("--light mode: light tier — mines Mistral-7B-v0.3 under PoM.");
        keryx_miner::models::Tier::Light
    } else if opt.very_light {
        info!("--very-light mode: entry tier — mines Qwen3-8B-abliterated under PoM.");
        keryx_miner::models::Tier::VeryLight
    } else {
        info!("default mode: mines GLM-4-9B-0414 under PoM.");
        keryx_miner::models::Tier::Default
    };
    // H4-only binary: stage, announce, and prefetch exactly the one H4 model for the selected
    // hardware tier, filtered by hardware capability. Below the H4 flip this binary refuses to
    // mine (`pom_tier_index` returns None), so no pre-H4 lineup is ever staged.
    let specs_v2 = filter_specs_by_vram(keryx_miner::models::specs_for_tier(tier));
    // PoM: pick the highest tier this miner serves that has a pinned R_T (the model it will
    // mine under possession). Captured before `specs_v2` is consumed; the index is built after
    // prefetch (below). `&'static ModelSpec` is Copy so this survives the moves.
    let pom_spec = if keryx_miner::pom::pom_activation_daa() != u64::MAX {
        specs_v2
            .iter()
            .copied()
            .filter(|s| keryx_miner::models::is_pom_model(&s.model_id))
            .max_by_key(|s| s.min_vram_mb)
    } else {
        None
    };
    // Announce the H4 lineup from the start (static — no crossing swap left).
    keryx_miner::slm::init_supported(specs_v2);
    log::debug!(
        "OPoI Phase-3 active — {} uncensored model(s) staged (H4-only lineup).",
        specs_v2.len(),
    );
    // Block until the uncensored lineup is fully downloaded before mining: never start hashing
    // while a model this miner will serve is still downloading.
    if keryx_miner::pow_only() {
        info!("PoW-only mode (KERYX_POW_ONLY): skipping OPoI model prefetch + inference probe.");
    } else {
    match tokio::task::spawn_blocking(move || keryx_miner::slm::prefetch_models(specs_v2)).await {
        Ok(Ok(())) => info!("Model files ready — starting mining."),
        Ok(Err(e)) => {
            error!("OPoI v2 prefetch failed — refusing to mine without the post-hardfork lineup: {}", e);
            return Err(e.into());
        }
        Err(e) => {
            error!("Model prefetch task panicked: {}", e);
            return Err(e.into());
        }
    }
    // PoM possession setup is fully LAZY: nothing GPU- or host-heavy happens at boot. During the
    // pre-PoM legacy phase the GPU + host stay free for the legacy lineup (mining + inference start
    // immediately). The possession index AND the GPU walk are built by the mining loop the first
    // time PoM is active (DAA >= POM_ACTIVATION_DAA). Here we only record cheap config.
    if let Some(spec) = pom_spec {
        let gpath = keryx_miner::slm::gguf_path_for(spec).to_string_lossy().into_owned();
        // Record the mining MODEL so the walk can be built on demand (zero-dup: over the
        // in-process engine's resident weights on the inference GPU). The PoM tier INDEX is
        // computed per block from the block DAA (`pom_gpu::current_tier`), not frozen here —
        // the H4 gate makes it None below the flip, so a startup-frozen value would be wrong.
        keryx_miner::pom_gpu::set_mining_tier(spec.model_id, gpath);
        info!("PoM: configured to mine {} under possession; index + GPU walk load lazily when PoM activates (DAA {}).",
            spec.dir_name, keryx_miner::pom::pom_activation_daa());
    }

    // Verify the Vulkan inference backend before mining. OPoI challenges are mandatory, so a miner
    // that cannot run inference must fail fast with a clear message rather than spam panics.
    info!("Probing Vulkan inference backend before mining…");
    match tokio::task::spawn_blocking(keryx_miner::slm::probe_gpu_inference).await {
        Ok(keryx_miner::slm::GpuProbe::Ok) => info!("Vulkan inference backend verified."),
        Ok(keryx_miner::slm::GpuProbe::NoDevice) => {
            error!("No Vulkan device detected — OPoI inference and GPU mining require a Vulkan GPU (RDNA3). Cannot mine.");
            return Err("No Vulkan device — cannot start OPoI mining".into());
        }
        Err(e) => {
            error!("Inference probe task panicked: {}", e);
            return Err(e.into());
        }
    }
    } // end: not PoW-only
    info!("Found plugins: {:?}", plugins);
    info!("Plugins found {} workers", worker_count);
    if worker_count == 0 && opt.num_threads.unwrap_or(0) == 0 {
        error!("No workers specified");
        return Err("No workers specified".into());
    }

    let block_template_ctr = Arc::new(AtomicU16::new((thread_rng().next_u64() % 10_000u64) as u16));
    if opt.devfund_percent > 0 {
        info!(
            "devfund enabled, mining {}.{}% of the time to devfund address: {} ",
            opt.devfund_percent / 100,
            opt.devfund_percent % 100,
            opt.devfund_address
        );
    }
    // Reconnect with exponential backoff so a pool that drops the socket right after connect
    // (rate-limit, restart, transient blip) doesn't turn into a tight reconnect storm that
    // hammers the pool and gets the IP throttled/banned. A connection that stays up for a while
    // resets the delay back to the minimum.
    let mut backoff = Duration::from_secs(1);
    const MAX_BACKOFF: Duration = Duration::from_secs(30);
    const HEALTHY_SESSION: Duration = Duration::from_secs(60);
    // Some pools (e.g. suprnova) reject an in-process reconnect while the previous worker session
    // is still registered, yet accept a brand-new process — the OS hard-closes the old socket on
    // exit. So after a few short-lived sessions in a row, exit(1) and let the supervisor
    // (start-miner.bat / HiveOS) relaunch us clean, instead of looping uselessly.
    const MAX_SHORT_SESSIONS: u32 = 3;
    let mut short_sessions: u32 = 0;
    loop {
        let started = std::time::Instant::now();
        match client_main(&opt, block_template_ctr.clone(), &plugin_manager, escrow_privkey.clone()).await {
            Ok(_) => info!("Client closed gracefully"),
            Err(e) => error!("Client closed with error {:?}", e),
        }
        if started.elapsed() >= HEALTHY_SESSION {
            backoff = Duration::from_secs(1); // a long, healthy session → reset
            short_sessions = 0;
        } else {
            short_sessions += 1;
            if short_sessions >= MAX_SHORT_SESSIONS {
                error!(
                    "{} short-lived sessions in a row — the pool is refusing in-process reconnects. \
                     Exiting for a clean supervised restart.",
                    short_sessions
                );
                std::process::exit(1);
            }
        }
        info!("Client closed, reconnecting in {}s", backoff.as_secs());
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(MAX_BACKOFF);
    }
}
