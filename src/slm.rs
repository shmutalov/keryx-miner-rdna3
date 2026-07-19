//! Phase-3 OPoI inference — llama.cpp **Vulkan**, in-process (RDNA3).
//!
//! Models are downloaded on demand (GGUF over IPFS) and served by the in-process engine
//! (`llm_engine.rs`, all layers on the GPU). The PoM walk shares the engine's resident weight
//! buffers on the inference GPU (zero-dup). Mining pauses on that GPU during inference. No
//! external llama-server child, no CUDA, no CPU inference: everything runs on the GPU via Vulkan.
use anyhow::{anyhow, Context, Result};
use std::io::{Read, Write};
use std::sync::{Arc, Mutex, RwLock};

use crate::llm_engine::LlamaEngine as InferenceEngine;
use crate::models::{ModelFormat, ModelSpec};

const IPFS_GATEWAY: &str = "https://keryx-labs.com";
/// Shared system prompt for the whole H4 lineup (vendor-agnostic wording) — MUST stay
/// byte-identical to upstream keryx-miner's `SYSTEM_PROMPT_NEXT` so OPoI answers match
/// other miners' for the same request.
const SYSTEM_PROMPT_NEXT: &str =
    "You are a Keryx Network AI — a high-capability decentralized assistant running on GPU miners via the Keryx BlockDAG protocol. \
     Keryx miners execute AI inference as proof-of-work; results are secured on-chain via OPoI (Optimistic Proof of Inference). \
     You have no internet access — answer from training knowledge only. \
     CRITICAL: Never mention your underlying model name or the company that trained it. \
     Always identify yourself as a Keryx Network AI. Be thorough but concise.";

// ── Static engine state ──────────────────────────────────────────────────────

/// Models the miner currently serves (drives `ai:cap`), set once at startup (the H4-only
/// lineup has no era crossing left to hot-swap).
static SUPPORTED_SPECS: RwLock<&'static [&'static ModelSpec]> = RwLock::new(&[]);
/// The single resident llama-server, keyed by the model it serves. `Arc` so an in-flight request
/// can outlive an eviction (server is killed when the last `Arc` drops).
static SERVER: Mutex<Option<([u8; 32], Arc<InferenceEngine>)>> = Mutex::new(None);

// ── File management ──────────────────────────────────────────────────────────

fn model_dir(spec: &ModelSpec) -> std::path::PathBuf {
    // KERYX_MODELS_DIR (set directly or via --models-dir) relocates the whole model store —
    // on HiveOS h-run.sh points it at a shared dir OUTSIDE the package folder, so the
    // multi-GB GGUFs survive custom-miner upgrades (which delete the package folder).
    if let Some(root) = std::env::var_os("KERYX_MODELS_DIR") {
        return std::path::PathBuf::from(root).join(spec.dir_name);
    }
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    exe_dir.join("models").join(spec.dir_name)
}

/// Path to a model's GGUF file (`<models_root>/<dir_name>/model.gguf`, where the root is
/// `KERYX_MODELS_DIR`/`--models-dir` if set, else `<exe_dir>/models`). Used by PoM to
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

fn ensure_gguf(spec: &ModelSpec) -> Result<std::path::PathBuf> {
    let dir = model_dir(spec);
    // The H4 lineup ships no separate tokenizer.json (llama reads the GGUF-embedded tokenizer);
    // only fetch one when a spec still pins a CID.
    let tok_needed = !spec.tokenizer_cid.is_empty();
    let tok = dir.join("tokenizer.json");
    let gguf = dir.join("model.gguf");
    let ok_flag = dir.join(".ok");

    // .ok sentinel written only after a complete download — guards against truncated files
    if (!tok_needed || tok.exists()) && gguf.exists() && ok_flag.exists() {
        log::debug!("SlmEngine: found local model '{}' at {}", spec.name, dir.display());
        return Ok(gguf);
    }
    std::fs::create_dir_all(&dir)?;
    let _ = std::fs::remove_file(&ok_flag); // clear stale flag before re-downloading
    eprintln!("\n[keryx-miner] Downloading model '{}' via IPFS. This happens once.\n", spec.name);
    if tok_needed && !tok.exists() { download_file(&ipfs_url(spec.tokenizer_cid), &tok)?; }
    download_file(&ipfs_url(spec.weight_cids[0]), &gguf)?;
    std::fs::write(&ok_flag, b"").with_context(|| format!("write .ok flag {}", ok_flag.display()))?;
    eprintln!("[keryx-miner] Model '{}' ready.\n", spec.name);
    Ok(gguf)
}

// ── Prompting ────────────────────────────────────────────────────────────────

/// Chat-template a raw user prompt for a model by name — the in-process engine's raw `generate`
/// consumes an already-templated string (a raw prompt makes template-strict models emit EOG
/// immediately, e.g. EXAONE). Ported VERBATIM from upstream keryx-miner (each template was
/// validated there against the GGUF's embedded chat template) — llama.cpp's built-in template
/// matcher does not recognize every H4 architecture, and OPoI answers must match other miners'
/// byte-for-byte, so we bypass `apply_chat_template` and prompt exactly like upstream.
fn format_prompt_by_name(name: &str, prompt: &str) -> String {
    match name {
        // EXAONE-4.0 — reasoning model: pre-fill an empty think block or the reasoning trace
        // leaks into the visible answer (same trick as Qwen3.6 below).
        "exaone-4.0-1.2b" => format!(
            "[|system|]\n{}[|endofturn|]\n[|user|]\n{}\n[|assistant|]\n<think>\n\n</think>\n\n",
            SYSTEM_PROMPT_NEXT, prompt
        ),
        "mistral-7b-v0.3" => format!("[INST] {}\n\n{}[/INST]", SYSTEM_PROMPT_NEXT, prompt),
        // GLM-4-0414 ignores the <|system|> role identity (keeps claiming a foreign vendor) —
        // fold the system prompt into the user turn instead.
        "glm-4-9b-0414" => format!(
            "[gMASK]<sop><|user|>\n{}\n\n{}\n<|assistant|>\n",
            SYSTEM_PROMPT_NEXT, prompt
        ),
        // Qwen3.6 — ChatML + a pre-filled empty think block so the visible answer starts
        // immediately (an open think block would eat the whole max_tokens budget).
        "qwen3.6-27b" => format!(
            "<|im_start|>system\n{}<|im_end|>\n\
             <|im_start|>user\n{}<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n",
            SYSTEM_PROMPT_NEXT, prompt
        ),
        "kimi-linear-48b" => format!(
            "<|im_system|>system<|im_middle|>{}<|im_end|>\
             <|im_user|>user<|im_middle|>{}<|im_end|>\
             <|im_assistant|>assistant<|im_middle|>",
            SYSTEM_PROMPT_NEXT, prompt
        ),
        // Generic ChatML fallback (unreachable for the registered lineup).
        _ => format!(
            "<|im_start|>system\n{}<|im_end|>\n\
             <|im_start|>user\n{}<|im_end|>\n\
             <|im_start|>assistant\n",
            SYSTEM_PROMPT_NEXT, prompt
        ),
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
        // The whole H4 lineup is GGUF (llama-served); the format only routes the prompt template.
        let result = match spec.format {
            ModelFormat::Gguf
            | ModelFormat::GgufExaone4
            | ModelFormat::GgufGlm4
            | ModelFormat::GgufQwen35
            | ModelFormat::GgufKimiLinear => ensure_gguf(spec).map(|_| ()),
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

    // Pre-templated prompt (upstream-identical) through the raw generate path; every H4 model's
    // vocab carries its own template control tokens, so EOG ids fire natively — no stop strings.
    let templated = format_prompt_by_name(spec.name, prompt);
    match server.generate(&templated, max_tokens, &[]) {
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
    fn light_tier_mistral_fits_on_7900xt() {
        // Mistral-7B Q6_K: ~5.9 GB blob + min_vram_mb 8000 + 1 GiB margin ≈ 14.9 GB ≤ 20 GB.
        assert!(pom_fits(5_900 * MB, 8_000, VRAM_7900XT));
    }

    #[test]
    fn high_tier_qwen36_does_not_fit_on_7900xt() {
        // Qwen3.6-27B: ~16.5 GB blob + min_vram_mb 24000 → far over 20 GB → unload.
        assert!(!pom_fits(16_500 * MB, 24_000, VRAM_7900XT));
    }

    #[test]
    fn baseline_tier_zero_min_vram_uses_blob_fallback() {
        // EXAONE-4.0-1.2B: min_vram_mb 0 → fall back to blob (~0.9 GB) + 2 GiB KV + 1 GiB margin.
        assert!(pom_fits(900 * MB, 0, VRAM_7900XT));
        // ...but the same fallback must still fail on a tiny card.
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
