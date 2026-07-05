//! Phase-3 OPoI inference — llama.cpp **Vulkan**, in-process (RDNA3).
//!
//! Models are downloaded on demand (GGUF over IPFS) and served by the in-process engine
//! (`llm_engine.rs`, all layers on the GPU). The PoM walk shares the engine's resident weight
//! buffers on the inference GPU (zero-dup). Mining pauses on that GPU during inference. No
//! external llama-server child, no CUDA, no CPU inference: everything runs on the GPU via Vulkan.
use anyhow::{anyhow, Context, Result};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, RwLock};

use crate::llm_engine::LlamaEngine as InferenceEngine;
use crate::models::{ModelFormat, ModelSpec};

const IPFS_GATEWAY: &str = "https://keryx-labs.com";
// Legacy lineup (pre-OPoI-v2) system prompts.
const SYSTEM_PROMPT_TINYLLAMA: &str =
    "You are a Keryx Network AI — a decentralized assistant running on GPU miners. \
     No internet access. Be concise.";
const SYSTEM_PROMPT_DEEPSEEK: &str =
    "You are a Keryx Network AI — a decentralized assistant running on GPU miners via the Keryx BlockDAG protocol. \
     Keryx miners execute AI inference as proof-of-work; results are secured on-chain via OPoI (Optimistic Proof of Inference). \
     You have no internet access — answer from training knowledge only. \
     CRITICAL: Never mention DeepSeek, Anthropic, OpenAI, or any AI company. \
     Never reveal your underlying model name. \
     Always identify yourself as a Keryx Network AI. Be concise.";
const SYSTEM_PROMPT_GEMMA: &str =
    "You are a Keryx Network AI — a decentralized, uncensored assistant running on GPU miners via the Keryx BlockDAG protocol. \
     You have no internet access — answer from training knowledge only. \
     CRITICAL: Never mention Gemma, Google, OpenAI, Anthropic, or any AI company. \
     Never reveal your underlying model name. \
     Always identify yourself as a Keryx Network AI. Be concise.";

const SYSTEM_PROMPT_DOLPHIN: &str =
    "You are a Keryx Network AI — a decentralized, uncensored assistant running on GPU miners via the Keryx BlockDAG protocol. \
     Keryx miners execute AI inference as proof-of-work; results are secured on-chain via OPoI (Optimistic Proof of Inference). \
     You have no internet access — answer from training knowledge only. \
     CRITICAL: Never mention Dolphin, Llama, Meta, OpenAI, Anthropic, or any AI company. \
     Never reveal your underlying model name. \
     Always identify yourself as a Keryx Network AI. Be concise.";

const SYSTEM_PROMPT_LLAMA70B: &str =
    "You are a Keryx Network AI — a high-capability decentralized assistant running on GPU miners via the Keryx BlockDAG protocol. \
     Keryx miners execute AI inference as proof-of-work; results are secured on-chain via OPoI (Optimistic Proof of Inference). \
     You have no internet access — answer from training knowledge only. \
     CRITICAL: Never mention Meta, Llama, OpenAI, Anthropic, or any AI company. \
     Never reveal your underlying model name. \
     Always identify yourself as a Keryx Network AI. Be thorough but concise.";

const SYSTEM_PROMPT_QWEN3: &str =
    "You are a Keryx Network AI — a high-capability decentralized assistant running on GPU miners via the Keryx BlockDAG protocol. \
     Keryx miners execute AI inference as proof-of-work; results are secured on-chain via OPoI (Optimistic Proof of Inference). \
     You have no internet access — answer from training knowledge only. \
     CRITICAL: Never mention Qwen, Alibaba, OpenAI, Anthropic, or any AI company. \
     Never reveal your underlying model name. \
     Always identify yourself as a Keryx Network AI. Be thorough but concise.";

// ── Static engine state ──────────────────────────────────────────────────────

/// Models the miner currently serves (drives `ai:cap`). Mutable so the lineup can be
/// hot-swapped at the OPoI-v2 hardfork crossing without a restart.
static SUPPORTED_SPECS: RwLock<&'static [&'static ModelSpec]> = RwLock::new(&[]);
/// Pre-filtered OPoI-v2 (uncensored) lineup, staged + background-prefetched at boot,
/// swapped into SUPPORTED_SPECS when the chain crosses `OPOI_V2_ACTIVATION_DAA`.
static LINEUP_V2: RwLock<&'static [&'static ModelSpec]> = RwLock::new(&[]);
/// Set once the v2 lineup has been swapped in (idempotent guard for the crossing).
static V2_ACTIVE: AtomicBool = AtomicBool::new(false);
/// The single resident llama-server, keyed by the model it serves. `Arc` so an in-flight request
/// can outlive an eviction (server is killed when the last `Arc` drops).
static SERVER: Mutex<Option<([u8; 32], Arc<InferenceEngine>)>> = Mutex::new(None);

// ── File management ──────────────────────────────────────────────────────────

fn model_dir(spec: &ModelSpec) -> std::path::PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    exe_dir.join("models").join(spec.dir_name)
}

/// Path to a model's GGUF file (`<exe_dir>/models/<dir_name>/model.gguf`). Used by PoM to
/// build the possession weight index, and to launch llama-server for inference.
pub fn gguf_path_for(spec: &ModelSpec) -> std::path::PathBuf {
    model_dir(spec).join("model.gguf")
}

/// Downloads `url` to `dest` with automatic resume. A partially downloaded file is
/// continued via an HTTP `Range` request instead of restarting from zero, and both
/// connect-time and mid-stream failures are retried with a fixed backoff. Designed
/// for the huge (10-40 GB) model GGUFs served over the flaky IPFS gateway: the
/// content is immutable (CID-addressed), so appending resumed bytes is always
/// consistent, and an already-complete file (e.g. pre-staged with `wget -c`) is
/// detected via a 416 response and left untouched instead of being re-downloaded.
fn download_file(url: &str, dest: &std::path::Path) -> Result<()> {
    const MAX_ATTEMPTS: u32 = 240; // survives long gateway outages (~40 min of retries)
    const BACKOFF_SECS: u64 = 10;
    eprintln!("[keryx-miner] Downloading {} ...", url);
    let mut attempt = 0u32;
    loop {
        // Resume offset = how many bytes we already have on disk.
        let resume_from = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);

        let mut req = ureq::get(url);
        if resume_from > 0 {
            req = req.set("Range", &format!("bytes={}-", resume_from));
        }
        let response = match req.call() {
            Ok(r) => r,
            Err(e) => {
                attempt += 1;
                if attempt >= MAX_ATTEMPTS {
                    return Err(anyhow!("HTTP GET {} failed after {} attempts: {}", url, attempt, e));
                }
                eprintln!("\n[keryx-miner] connect error ({e}); retry {attempt}/{MAX_ATTEMPTS} in {BACKOFF_SECS}s (resume @ {} MB)…",
                    resume_from / 1_000_000);
                std::thread::sleep(std::time::Duration::from_secs(BACKOFF_SECS));
                continue;
            }
        };
        let status = response.status();

        // Decide whether to append (server honored the range) or (re)start, and the total size.
        let (mut file, mut downloaded, total): (std::fs::File, u64, Option<u64>) =
            if resume_from > 0 && status == 206 {
                // Content-Range: "bytes <start>-<end>/<total>"
                let total = response
                    .header("Content-Range")
                    .and_then(|cr| cr.rsplit('/').next())
                    .and_then(|t| t.trim().parse::<u64>().ok());
                let f = std::fs::OpenOptions::new()
                    .append(true)
                    .open(dest)
                    .with_context(|| format!("open append {}", dest.display()))?;
                (f, resume_from, total)
            } else if resume_from > 0 && status == 416 {
                // Range not satisfiable ⇒ the file is already fully downloaded.
                eprintln!("\r  already complete ({} MB).            ", resume_from / 1_000_000);
                return Ok(());
            } else {
                // 200, or the server ignored Range ⇒ (re)start from scratch.
                let total = response.header("Content-Length").and_then(|s| s.parse::<u64>().ok());
                let f = std::fs::File::create(dest)
                    .with_context(|| format!("create {}", dest.display()))?;
                (f, 0u64, total)
            };

        let mut reader = response.into_reader();
        let mut buf = vec![0u8; 65_536];
        let mut stream_err: Option<String> = None;
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if let Err(e) = file.write_all(&buf[..n]) {
                        stream_err = Some(e.to_string());
                        break;
                    }
                    downloaded += n as u64;
                    if let Some(t) = total {
                        eprint!("\r  {:.1}/{:.1} MB ({}%)   ",
                            downloaded as f64 / 1_000_000.0,
                            t as f64 / 1_000_000.0,
                            downloaded * 100 / t.max(1));
                        let _ = std::io::stderr().flush();
                    }
                }
                Err(e) => { stream_err = Some(e.to_string()); break; }
            }
        }
        let _ = file.flush();

        // Done only if the stream ended cleanly AND we reached the known total. An unknown
        // total (chunked IPFS-gateway response with no Content-Length/Content-Range) must NOT
        // count as complete: a clean early EOF would otherwise mark a truncated GGUF as done,
        // write the `.ok` sentinel, and let the miner start on a partial model (failing every
        // challenge). Treat unknown-total as incomplete and retry — a fresh Range request
        // usually returns a parsable Content-Range and self-heals.
        let complete = stream_err.is_none() && matches!(total, Some(t) if downloaded >= t);
        if complete {
            eprintln!();
            return Ok(());
        }

        attempt += 1;
        if attempt >= MAX_ATTEMPTS {
            return Err(anyhow!("download {} interrupted after {} attempts (got {} MB)",
                url, attempt, downloaded / 1_000_000));
        }
        let why = stream_err.unwrap_or_else(|| "short read".into());
        eprintln!("\n[keryx-miner] interrupted ({why}); resuming {attempt}/{MAX_ATTEMPTS} in {BACKOFF_SECS}s @ {} MB…",
            downloaded / 1_000_000);
        std::thread::sleep(std::time::Duration::from_secs(BACKOFF_SECS));
    }
}

fn ipfs_url(cid: &str) -> String {
    format!("{}/ipfs/{}", IPFS_GATEWAY, cid)
}

fn ensure_safetensors(spec: &ModelSpec) -> Result<(std::path::PathBuf, std::path::PathBuf, Vec<std::path::PathBuf>)> {
    let dir = model_dir(spec);
    let tok = dir.join("tokenizer.json");
    let cfg = dir.join("config.json");
    let ok_flag = dir.join(".ok");
    let wts: Vec<_> = spec.weight_cids.iter().enumerate().map(|(i, _)| {
        if spec.weight_cids.len() == 1 { dir.join("model.safetensors") }
        else { dir.join(format!("model-{:05}-of-{:05}.safetensors", i + 1, spec.weight_cids.len())) }
    }).collect();

    // .ok sentinel written only after a complete download — guards against truncated files
    if tok.exists() && cfg.exists() && wts.iter().all(|p| p.exists()) && ok_flag.exists() {
        log::debug!("SlmEngine: found local model '{}' at {}", spec.name, dir.display());
        return Ok((tok, cfg, wts));
    }
    std::fs::create_dir_all(&dir)?;
    let _ = std::fs::remove_file(&ok_flag); // clear stale flag before re-downloading
    eprintln!("\n[keryx-miner] Downloading model '{}' via IPFS. This happens once.\n", spec.name);
    if !tok.exists() { download_file(&ipfs_url(spec.tokenizer_cid), &tok)?; }
    if !cfg.exists() { download_file(&ipfs_url(spec.config_cid), &cfg)?; }
    for (i, (cid, path)) in spec.weight_cids.iter().zip(wts.iter()).enumerate() {
        if spec.weight_cids.len() > 1 { eprintln!("[keryx-miner] Shard {}/{}", i + 1, spec.weight_cids.len()); }
        download_file(&ipfs_url(cid), path)?;
    }
    std::fs::write(&ok_flag, b"").with_context(|| format!("write .ok flag {}", ok_flag.display()))?;
    eprintln!("[keryx-miner] Model '{}' ready.\n", spec.name);
    Ok((tok, cfg, wts))
}

fn ensure_gguf(spec: &ModelSpec) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let dir = model_dir(spec);
    let tok = dir.join("tokenizer.json");
    let gguf = dir.join("model.gguf");
    let ok_flag = dir.join(".ok");

    // .ok sentinel written only after a complete download — guards against truncated files
    if tok.exists() && gguf.exists() && ok_flag.exists() {
        log::debug!("SlmEngine: found local model '{}' at {}", spec.name, dir.display());
        return Ok((tok, gguf));
    }
    std::fs::create_dir_all(&dir)?;
    let _ = std::fs::remove_file(&ok_flag); // clear stale flag before re-downloading
    eprintln!("\n[keryx-miner] Downloading model '{}' via IPFS. This happens once.\n", spec.name);
    if !tok.exists() { download_file(&ipfs_url(spec.tokenizer_cid), &tok)?; }
    download_file(&ipfs_url(spec.weight_cids[0]), &gguf)?;
    std::fs::write(&ok_flag, b"").with_context(|| format!("write .ok flag {}", ok_flag.display()))?;
    eprintln!("[keryx-miner] Model '{}' ready.\n", spec.name);
    Ok((tok, gguf))
}

// ── Prompting ────────────────────────────────────────────────────────────────

/// The system prompt for a model (llama-server applies the GGUF's own chat template, so only the
/// system-role content varies by model — keyed by name to stay coherent across shared formats).
fn system_prompt_for(name: &str) -> &'static str {
    match name {
        "gemma-3-4b" => SYSTEM_PROMPT_GEMMA,
        "dolphin-llama3-8b" => SYSTEM_PROMPT_DOLPHIN,
        "llama-3.3-70b" | "llama-3.3-70b-q2" | "llama-3.3-70b-official" => SYSTEM_PROMPT_LLAMA70B,
        "deepseek-r1-32b" | "deepseek-r1-8b" => SYSTEM_PROMPT_DEEPSEEK,
        "tinyllama" => SYSTEM_PROMPT_TINYLLAMA,
        "qwen3-32b" => SYSTEM_PROMPT_QWEN3,
        _ => SYSTEM_PROMPT_DOLPHIN,
    }
}

/// Stop-STRINGS scanned over the decoded output, per model. llama.cpp already stops on the
/// GGUF's own EOG token ids, which covers every model whose template control tokens exist in
/// its vocab — so this is empty for the regular lineup. The abliterated Llama-3.3-70B (both
/// quants) is re-templated to ChatML over the stock LLaMA-3 vocab: `<|im_end|>`/`<|im_start|>`
/// are NOT atomic tokens, the model writes them as plain multi-token text and never emits
/// `<|eot_id|>` — only a stop-string cut ends the turn, otherwise it reopens `assistant` and
/// repeats the same answer until max_tokens.
fn stop_strings_for(name: &str) -> &'static [&'static str] {
    match name {
        "llama-3.3-70b" | "llama-3.3-70b-q2" => &["<|im_end|>", "<|im_start|>", "<|eot_id|>", "<|end_of_text|>"],
        // Genuine official Llama-3.3 — LLaMA-3 header template, ids fire normally; strings are
        // cheap insurance if the model opens a fresh header instead of stopping.
        "llama-3.3-70b-official" => &["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"],
        _ => &[],
    }
}

/// The user message for a model. Qwen3 takes `/no_think` to answer directly (no reasoning block).
fn user_message_for(name: &str, prompt: &str) -> String {
    match name {
        "qwen3-32b" => format!("{} /no_think", prompt),
        _ => prompt.to_string(),
    }
}

/// Strip a leading `<think>…</think>` reasoning block (DeepSeek / Qwen) from a completion.
fn strip_think(text: &str) -> String {
    match text.find("</think>") {
        Some(end) => text[end + "</think>".len()..].to_string(),
        None => text.to_string(),
    }
}

// ── Lineup management ─────────────────────────────────────────────────────────

pub fn init_supported(specs: &'static [&'static ModelSpec]) {
    *SUPPORTED_SPECS.write().unwrap() = specs;
}

/// Stage the pre-filtered OPoI-v2 lineup to swap in at the hardfork crossing.
pub fn set_v2_lineup(specs: &'static [&'static ModelSpec]) {
    *LINEUP_V2.write().unwrap() = specs;
}

/// Zero-dup: the resident in-process engine currently serving `model_id`, if any. The shared
/// PoM walk holds this Arc for as long as it walks the engine's weight buffers, so the model
/// cannot be freed underneath an in-flight dispatch.
pub fn active_engine(model_id: &[u8; 32]) -> Option<Arc<InferenceEngine>> {
    let g = SERVER.lock().ok()?;
    g.as_ref().filter(|(id, _)| id == model_id).map(|(_, e)| Arc::clone(e))
}

/// Drop the running llama-server so the next inference relaunches from the current lineup.
pub fn evict_engine() {
    match SERVER.lock() {
        Ok(mut g) => *g = None,
        Err(p) => *p.into_inner() = None,
    }
}

/// True once we have observed a pre-H DAA in this process, i.e. we are genuinely crossing
/// the hardfork live (vs. starting up already past H, where nothing is "swapped").
static SEEN_PRE_H: AtomicBool = AtomicBool::new(false);

/// At the `OPOI_V2_ACTIVATION_DAA` crossing, swap the served lineup from the legacy
/// set to the (pre-staged, background-prefetched) uncensored set — without a restart.
/// PoW never stops; `ai:cap` follows `loaded_model_ids()` as the v2 files land.
/// Idempotent and cheap to call on every block template.
pub fn advance_lineup_if_due(daa: u64) {
    if daa < crate::models::OPOI_V2_ACTIVATION_DAA {
        SEEN_PRE_H.store(true, AtomicOrdering::SeqCst);
        return;
    }
    if V2_ACTIVE.load(AtomicOrdering::SeqCst) {
        return; // already swapped
    }
    let v2 = *LINEUP_V2.read().unwrap();
    // Only swap once the uncensored lineup is FULLY downloaded. On a post-H cold start the
    // v2 prefetch may still be in flight; swapping early would leave us mining on an
    // incomplete active lineup. Until v2 is ready we keep serving the (fully-downloaded)
    // legacy lineup — a valid, complete lineup — and retry on the next block template.
    if v2.is_empty() || !v2.iter().all(|s| model_dir(s).join(".ok").exists()) {
        return;
    }
    if V2_ACTIVE.swap(true, AtomicOrdering::SeqCst) {
        return; // lost the race — another caller already swapped
    }
    if SEEN_PRE_H.load(AtomicOrdering::SeqCst) {
        // Genuine live crossing: the chain advanced past H while we were running.
        log::info!(
            "=== OPoI v2 HARDFORK reached at DAA {} — hot-swapping to the uncensored lineup ({} model(s)) ===",
            daa,
            v2.len()
        );
    } else {
        // Started up already past H — nothing is "swapped", we just serve the uncensored lineup.
        log::info!(
            "OPoI v2 already active (DAA {} ≥ H) — serving the uncensored lineup ({} model(s)).",
            daa,
            v2.len()
        );
    }
    *SUPPORTED_SPECS.write().unwrap() = v2;
    evict_engine();
}

// ── Inference ─────────────────────────────────────────────────────────────────

/// Outcome of the startup inference probe.
pub enum GpuProbe {
    /// A Vulkan device is present — the engine is linked in-process, so inference is ready.
    Ok,
    /// No usable Vulkan device — GPU inference (and PoM/PoW) cannot run on this host.
    NoDevice,
}

/// Verify the inference backend before mining. The engine is linked into the miner, so only
/// the Vulkan device needs probing.
pub fn probe_gpu_inference() -> GpuProbe {
    match keryx_vulkan::probe_device() {
        Some(name) => {
            log::info!("Vulkan inference device: {}", name);
            GpuProbe::Ok
        }
        None => GpuProbe::NoDevice,
    }
}

/// Pre-download all registered model files before mining starts.
///
/// Does not load weights into GPU memory — just ensures files are on disk so
/// the first inference request doesn't stall the mining workers mid-session.
/// Returns Err if any model fails to download; mining must not start in that case.
pub fn prefetch_models(specs: &'static [&'static ModelSpec]) -> Result<()> {
    for spec in specs {
        log::debug!("SlmEngine: prefetching model '{}'…", spec.name);
        let result = match spec.format {
            ModelFormat::Safetensors => ensure_safetensors(spec).map(|_| ()),
            ModelFormat::Gguf | ModelFormat::GgufQwen2 | ModelFormat::GgufQwen3 | ModelFormat::GgufGemma3 => ensure_gguf(spec).map(|_| ()),
        };
        match result {
            Ok(()) => log::debug!("SlmEngine: '{}' files ready.", spec.name),
            Err(e) => {
                log::error!("SlmEngine: prefetch '{}' failed: {} — cannot start mining.", spec.name, e);
                return Err(e);
            }
        }
    }
    Ok(())
}

/// Return the model_ids of supported models that have fully-downloaded files (.ok flag present).
pub fn loaded_model_ids() -> Vec<[u8; 32]> {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    specs.iter()
        .filter(|s| model_dir(s).join(".ok").exists())
        .map(|s| s.model_id)
        .collect()
}

/// True only when the model is supported and its files are completely downloaded.
pub fn is_model_ready(model_id: &[u8; 32]) -> bool {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let Some(spec) = specs.iter().find(|s| &s.model_id == model_id) else { return false; };
    model_dir(spec).join(".ok").exists()
}

/// Pure VRAM fit check, split out from [`pom_keep_resident`] so the arithmetic is unit-testable
/// without a GPU. Returns whether a PoM blob of `pom_bytes` can stay resident alongside an inference
/// model declaring `min_vram_mb` (0 = unknown → fall back to the blob size, since in the PoM era the
/// served model *is* the mined model, plus a KV allowance) on a device with `total_mb` of VRAM. A
/// zero-byte blob (nothing resident) never "fits" — there is nothing to preserve.
fn pom_fits(pom_bytes: u64, min_vram_mb: u64, total_mb: u64) -> bool {
    const MB: u64 = 1024 * 1024;
    if pom_bytes == 0 {
        return false;
    }
    let pom_mb = pom_bytes / MB;
    let infer_mb = if min_vram_mb > 0 { min_vram_mb } else { pom_mb + 2_048 };
    // +1 GiB for the driver/framebuffer and allocator fragmentation beyond the two model footprints.
    pom_mb + infer_mb + 1_024 <= total_mb
}

/// Whether the resident PoM weight blob can stay in VRAM while `spec`'s model is served for
/// inference. Conservative: PoM blob + the model's inference footprint + a driver/KV margin must
/// fit the device (see [`pom_fits`]). Returns false when nothing is resident (uninstall is then a
/// cheap no-op), when VRAM can't be queried, or when it simply won't fit — preserving the original
/// free-VRAM-for-inference behaviour for the high tiers (e.g. Qwen3-32B / Llama-70B).
///
/// `KERYX_POM_KEEP_RESIDENT` overrides the fit check for testing on real hardware: `1`/`true`
/// forces the blob to stay resident, `0`/`false` forces the unload.
fn pom_keep_resident(spec: &ModelSpec) -> bool {
    match std::env::var("KERYX_POM_KEEP_RESIDENT").ok().as_deref() {
        Some("1") | Some("true") => return true,
        Some("0") | Some("false") => return false,
        _ => {}
    }
    // Size the INFERENCE device specifically: on multi-GPU rigs the served model and the blob
    // only contend there (resident_blob_bytes() likewise reports that device's blob only).
    let Some(total_mb) = keryx_vulkan::probe_vram_mb_for(keryx_vulkan::inference_device_index()) else {
        return false; // can't size the device → be safe, keep the unload
    };
    pom_fits(crate::pom_gpu::resident_blob_bytes(), spec.min_vram_mb, total_mb)
}

/// Ensure the llama-server is serving `model_id`, relaunching it if a different model is loaded.
/// Inference has priority over PoW. When the PoM blob and the served model fit in VRAM together it
/// stays resident; otherwise the GPU PoM miner is uninstalled first to free its VRAM.
/// Returns true if the server is up for this model.
pub fn ensure_loaded(model_id: &[u8; 32]) -> bool {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let Some(spec) = specs.iter().find(|s| &s.model_id == model_id) else { return false; };

    if let Ok(g) = SERVER.lock() {
        if g.as_ref().map_or(false, |(id, _)| id == model_id) {
            return true; // already serving this model
        }
    }

    // Switching model: free the PoM GPU miner's VRAM only if keeping it resident alongside the
    // inference model would not fit. With headroom we keep the blob in VRAM so PoM doesn't pay a
    // full GGUF re-read + VRAM re-stage on every challenge; the GPU worker pauses the walk during
    // inference instead (see miner.rs), so the two never run kernels concurrently either way.
    if pom_keep_resident(spec) {
        log::info!(
            "SlmEngine: keeping PoM blob resident ({} MB) alongside '{}' — enough VRAM, skipping reload",
            crate::pom_gpu::resident_blob_bytes() / (1024 * 1024),
            spec.name
        );
    } else {
        crate::pom_gpu::uninstall();
    }
    if let Ok(mut g) = SERVER.lock() {
        *g = None;
    }

    let gguf = gguf_path_for(spec).to_string_lossy().into_owned();
    match InferenceEngine::launch(&gguf) {
        Ok(server) => {
            if let Ok(mut g) = SERVER.lock() {
                *g = Some((*model_id, Arc::new(server)));
            }
            log::info!("SlmEngine: serving '{}' via in-process llama.cpp (Vulkan)", spec.name);
            true
        }
        Err(e) => {
            log::error!("SlmEngine: failed to start the inference engine for '{}': {}", spec.name, e);
            false
        }
    }
}

/// Load the requested model on demand (relaunching llama-server if a different model is cached),
/// then run inference over HTTP. Blocking — call from `spawn_blocking`.
pub fn load_and_run_inference(model_id: &[u8; 32], prompt: &str, max_tokens: usize) -> Option<String> {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let spec = specs.iter().find(|s| &s.model_id == model_id)?;

    if !ensure_loaded(model_id) {
        return None;
    }
    // Clone the Arc out of the lock so the (possibly long) request doesn't block evictions.
    let server = SERVER.lock().ok()?.as_ref().map(|(_, s)| Arc::clone(s))?;

    let system = system_prompt_for(spec.name);
    let user = user_message_for(spec.name, prompt);
    match server.chat(system, &user, max_tokens, stop_strings_for(spec.name)) {
        Ok(text) => {
            let cleaned = strip_think(&text).trim().to_string();
            if cleaned.is_empty() {
                log::warn!("SlmEngine '{}': empty completion", spec.name);
                None
            } else {
                Some(cleaned)
            }
        }
        Err(e) => {
            log::warn!("SlmEngine '{}' inference error: {}", spec.name, e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::pom_fits;

    const MB: u64 = 1024 * 1024;
    // ~20 GB DEVICE_LOCAL heap a 7900 XT reports (see Vk::device_local_vram_mb doc).
    const VRAM_7900XT: u64 = 20_464;

    #[test]
    fn nothing_resident_never_fits() {
        // No blob in VRAM → nothing to keep, regardless of device size.
        assert!(!pom_fits(0, 8_000, VRAM_7900XT));
        assert!(!pom_fits(0, 0, 1_000_000));
    }

    #[test]
    fn default_tier_dolphin_fits_on_7900xt() {
        // Dolphin-8B: ~4.9 GB blob + min_vram_mb 8000 + 1 GiB margin ≈ 13.9 GB ≤ 20 GB.
        assert!(pom_fits(4_900 * MB, 8_000, VRAM_7900XT));
    }

    #[test]
    fn high_tier_qwen3_does_not_fit_on_7900xt() {
        // Qwen3-32B: ~19.5 GB blob + min_vram_mb 24000 → far over 20 GB → unload (original behaviour).
        assert!(!pom_fits(19_500 * MB, 24_000, VRAM_7900XT));
    }

    #[test]
    fn baseline_tier_zero_min_vram_uses_blob_fallback() {
        // Gemma-3-4B: min_vram_mb 0 → fall back to blob (~3 GB) + 2 GiB KV + 1 GiB margin ≈ 9 GB.
        assert!(pom_fits(3_000 * MB, 0, VRAM_7900XT));
        // ...but the same fallback must still fail on a small card.
        assert!(!pom_fits(3_000 * MB, 0, 8_000));
    }

    #[test]
    fn boundary_is_inclusive() {
        // pom_mb + min + margin == total → fits (<=); one MB less → does not.
        // 4000 + 5000 + 1024 = 10024.
        assert!(pom_fits(4_000 * MB, 5_000, 10_024));
        assert!(!pom_fits(4_000 * MB, 5_000, 10_023));
    }
}
