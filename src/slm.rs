//! Phase-3 OPoI inference — llama.cpp **Vulkan**, in-process (RDNA3).
//!
//! Models are downloaded on demand (GGUF over IPFS) and served by the in-process engine
//! (`llm_engine.rs`, all layers on the GPU). The PoM walk shares the engine's resident weight
//! buffers on the inference GPU (zero-dup). Mining pauses on that GPU during inference. No
//! external llama-server child, no CUDA, no CPU inference: everything runs on the GPU via Vulkan.
use anyhow::{anyhow, Context, Result};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use crate::llm_engine::LlamaEngine as InferenceEngine;
use crate::models::{ModelFormat, ModelSpec};

const IPFS_GATEWAY: &str = "https://keryx-labs.com";
/// Shared system prompt for the whole lineup (vendor-agnostic wording) — MUST stay byte-identical
/// to upstream keryx-miner's `SYSTEM_PROMPT_NEXT` so answers match other miners' for the same
/// request. Points the model at web results / facts attached to the request (private-inference
/// clients send them) and forbids claiming a search the request does not include.
const SYSTEM_PROMPT_NEXT: &str =
    "You are a Keryx Network AI, a decentralized assistant served by the GPU miners of the Keryx network. \
     Answer in the language of the user's message. \
     Do not introduce yourself or describe Keryx unless the user asks about it. If asked: Keryx is a proof-of-work BlockDAG \
     derived from Kaspa (about 10 blocks per second) where each mining GPU proves on every block that it holds a model in VRAM \
     (Proof-of-Model), so mining and inference are the same job; requests are on-chain transactions paid in KRX. \
     Web search results, facts or earlier messages included in the request are your sources for recent or specific information: \
     use them. Without them, answer from your training knowledge and say when it may be out of date; \
     never claim to have searched the web or cite sources that are not included in the request. \
     Never mention your underlying model name or the company that trained it: if asked, you are a Keryx Network AI. \
     Be thorough but concise.";

/// Answer sent when a request's prompt does not fit the model's context window — upstream's text.
pub const PROMPT_TOO_LONG_ANSWER: &str =
    "This request could not be served: the prompt is longer than the model's context window. Send a shorter message or less conversation history.";

// ── Static engine state ──────────────────────────────────────────────────────

/// Models the miner stages (downloaded, servable), set once at startup: the tier's model for every
/// era the chain can still reach — two for tier 3 until the H14 gate, one otherwise. Which of them
/// is announced (`ai:cap`) and served follows the chain DAA ([`note_chain_daa`]).
static SUPPORTED_SPECS: RwLock<&'static [&'static ModelSpec]> = RwLock::new(&[]);
/// Highest chain DAA score seen from the node or pool; 0 until the first template/job.
static CHAIN_DAA: AtomicU64 = AtomicU64::new(0);
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
    exe_dir().join("models").join(spec.dir_name)
}

/// The directory of the miner executable (`.` if it cannot be determined).
fn exe_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
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
/// immediately). Ported VERBATIM from upstream keryx-miner (each template was validated there
/// against the GGUF's embedded chat template) — llama.cpp's built-in template matcher does not
/// recognize every lineup architecture, and answers must match other miners' byte-for-byte, so we
/// bypass `apply_chat_template` and prompt exactly like upstream.
fn format_prompt_by_name(name: &str, prompt: &str) -> String {
    match name {
        // GLM-4-0414 ignores the <|system|> role identity (keeps claiming a foreign vendor) —
        // fold the system prompt into the user turn instead.
        "glm-4-9b-0414" => format!(
            "[gMASK]<sop><|user|>\n{}\n\n{}\n<|assistant|>\n",
            SYSTEM_PROMPT_NEXT, prompt
        ),
        // Qwen3 family — ChatML + a pre-filled empty think block so the visible answer starts
        // immediately (an open think block would eat the whole max_tokens budget). This is the
        // `enable_thinking = false` branch of their embedded template, verbatim.
        "qwen3.8-27b" | "qwen3.6-27b" | "qwen3.5-9b-abliterated" => format!(
            "<|im_start|>system\n{}<|im_end|>\n\
             <|im_start|>user\n{}<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n",
            SYSTEM_PROMPT_NEXT, prompt
        ),
        // Gemma 4 is NOT the classic <start_of_turn> Gemma: turns are `<|turn>role … <turn|>`,
        // and its generation prompt carries an empty thought channel when thinking is off — the
        // same role as Qwen's empty think block. Without it the model opens its own and the
        // channel markers leak into the answer. BOS is omitted on purpose: this GGUF sets
        // add_bos_token, so the tokenizer prepends it.
        "gemma-4-12b-abliterated" => format!(
            "<|turn>system\n{}<turn|>\n\
             <|turn>user\n{}<turn|>\n\
             <|turn>model\n<|channel>thought\n<channel|>",
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

/// The chat template of an audited model split around the request-bound prefix: the head the
/// prefix tokens follow, and the rest after them. `None` for a model without a prefix slot.
/// Verbatim from upstream: the head's token count is the prefix offset the consensus pins.
pub(crate) fn format_prompt_parts(name: &str, prompt: &str) -> Option<(String, String)> {
    Some(match name {
        "glm-4-9b-0414" => (
            "[gMASK]<sop><|user|>\nRequest id:".to_string(),
            format!("\n{}\n\n{}\n<|assistant|>\n", SYSTEM_PROMPT_NEXT, prompt),
        ),
        "qwen3.8-27b" | "qwen3.6-27b" | "qwen3.5-9b-abliterated" => (
            "<|im_start|>system\nRequest id:".to_string(),
            format!(
                "\n{}<|im_end|>\n\
                 <|im_start|>user\n{}<|im_end|>\n\
                 <|im_start|>assistant\n<think>\n\n</think>\n\n",
                SYSTEM_PROMPT_NEXT, prompt
            ),
        ),
        "gemma-4-12b-abliterated" => (
            "<|turn>system\nRequest id:".to_string(),
            format!(
                "\n{}<turn|>\n\
                 <|turn>user\n{}<turn|>\n\
                 <|turn>model\n<|channel>thought\n<channel|>",
                SYSTEM_PROMPT_NEXT, prompt
            ),
        ),
        "kimi-linear-48b" => (
            "<|im_system|>system<|im_middle|>Request id:".to_string(),
            format!(
                "\n{}<|im_end|>\
                 <|im_user|>user<|im_middle|>{}<|im_end|>\
                 <|im_assistant|>assistant<|im_middle|>",
                SYSTEM_PROMPT_NEXT, prompt
            ),
        ),
        _ => return None,
    })
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

/// Record a chain DAA score seen from the node (template) or the pool (job).
pub fn note_chain_daa(daa: u64) {
    let before = CHAIN_DAA.fetch_max(daa, Ordering::AcqRel);
    let gate = crate::pom::private_inference_activation_daa();
    if before < gate && daa >= gate {
        log::info!("SlmEngine: private-inference gate reached — answers travel inline, IPFS no longer required");
    }
}

/// True once the chain reached the private-inference gate: answers travel inline in the
/// AiResponse and this miner publishes nothing to IPFS.
pub fn inline_answers() -> bool {
    CHAIN_DAA.load(Ordering::Acquire) >= crate::pom::private_inference_activation_daa()
}

/// Whether a staged model is mined and served in the chain's current era. Every staged model
/// counts until the chain DAA is known; after that a model of an era the chain has left (tier 3's
/// Qwen3.6-27B past H14) or not yet reached (Qwen3.8-27B before it) is neither announced nor served,
/// and neither is the `--h15-tier` model on the other side of the H15 gate.
fn served_now(spec: &ModelSpec) -> bool {
    match CHAIN_DAA.load(Ordering::Acquire) {
        0 => true,
        daa => match crate::pom_gpu::mining_model(daa) {
            Some(mined) => mined.model_id == spec.model_id,
            None => crate::models::pom_tier_index(&spec.model_id, daa).is_some(),
        },
    }
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
        // The whole H6 lineup is GGUF (llama-served); the format only routes the prompt template.
        let result = match spec.format {
            ModelFormat::Gguf
            | ModelFormat::GgufGlm4
            | ModelFormat::GgufQwen35
            | ModelFormat::GgufKimiLinear
            | ModelFormat::GgufGemma4 => ensure_gguf(spec).map(|_| ()),
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
        .filter(|s| served_now(s) && model_dir(s).join(".ok").exists())
        .map(|s| s.model_id)
        .collect()
}

/// True only when the model is supported and its files are completely downloaded.
pub fn is_model_ready(model_id: &[u8; 32]) -> bool {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let Some(spec) = specs.iter().find(|s| &s.model_id == model_id) else { return false; };
    served_now(spec) && model_dir(spec).join(".ok").exists()
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

/// The file every inference's prompt and answer is appended to, from `KERYX_LOG_AI` (which
/// `--log-ai[=FILE]` sets). None — the default — when unset, empty or `0`: private requests are
/// logged as opened, in plaintext.
pub fn ai_log_path() -> Option<&'static Path> {
    static PATH: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| parse_ai_log_setting(&std::env::var("KERYX_LOG_AI").ok()?)).as_deref()
}

/// `KERYX_LOG_AI`: `1` = `ai-requests.log` next to the exe, any other value but empty or `0` is
/// the file's path.
fn parse_ai_log_setting(value: &str) -> Option<PathBuf> {
    match value.trim() {
        "" | "0" => None,
        "1" => Some(exe_dir().join("ai-requests.log")),
        path => Some(PathBuf::from(path)),
    }
}

/// Append one entry to the AI log. The file is opened on first use; if it cannot be, the miner
/// says so once and runs on without the log.
fn ai_log(request: &str, entry: std::fmt::Arguments) {
    static FILE: std::sync::OnceLock<Option<Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    let Some(path) = ai_log_path() else { return };
    let file = FILE.get_or_init(|| match std::fs::OpenOptions::new().create(true).append(true).open(path) {
        Ok(file) => Some(Mutex::new(file)),
        Err(e) => {
            log::error!("AI log: cannot open {}: {e} — not logging AI requests", path.display());
            None
        }
    });
    let Some(file) = file else { return };
    let mut file = file.lock().unwrap_or_else(|p| p.into_inner());
    if let Err(e) = write_ai_log_entry(&mut *file, time::OffsetDateTime::now_utc(), request, entry) {
        log::warn!("AI log: write to {} failed: {e}", path.display());
    }
}

/// One AI log line, written in a single call so entries never interleave.
fn write_ai_log_entry(
    out: &mut impl Write,
    at: time::OffsetDateTime,
    request: &str,
    entry: std::fmt::Arguments,
) -> std::io::Result<()> {
    let at = at
        .format(time::macros::format_description!("[year]-[month]-[day] [hour]:[minute]:[second]Z"))
        .unwrap_or_default();
    out.write_all(format!("{at} [{request}] {entry}\n").as_bytes())
}

/// Load the requested model on demand (swapping out a different resident model), then run
/// inference in-process. Blocking — call from `spawn_blocking`. `request` names the request in the
/// `--log-ai` file; prompt and answer are Debug-escaped there, so each entry stays one line and a
/// prompt cannot forge entries.
pub fn load_and_run_inference(model_id: &[u8; 32], prompt: &str, max_tokens: usize, request: &str) -> Option<String> {
    if ai_log_path().is_none() {
        return run_inference(model_id, prompt, max_tokens);
    }
    let model = SUPPORTED_SPECS
        .read()
        .ok()
        .and_then(|specs| specs.iter().find(|s| &s.model_id == model_id).map(|s| s.name))
        .unwrap_or("unknown model");
    ai_log(request, format_args!("prompt ({model}, {}, max_tokens={max_tokens}): {prompt:?}", inference_gpu_label()));
    // Total time of this inference, including a model (re)load when the model was not resident.
    let started = std::time::Instant::now();
    let answer = run_inference(model_id, prompt, max_tokens);
    let elapsed = started.elapsed().as_secs_f64();
    match &answer {
        Some(text) => ai_log(request, format_args!("answer ({elapsed:.2} s): {text:?}")),
        None => ai_log(request, format_args!("no answer ({elapsed:.2} s)")),
    }
    answer
}

/// The inference GPU as the AI log names it (`GPU 0 AMD Radeon RX 7900 XT`). Resolved once: the
/// inference device is fixed for the process, and naming it opens a Vulkan instance.
fn inference_gpu_label() -> &'static str {
    static LABEL: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    LABEL.get_or_init(|| {
        let index = keryx_vulkan::inference_device_index();
        match keryx_vulkan::enumerate_devices().into_iter().find(|d| d.index == index) {
            Some(device) => format!("GPU {index} {}", device.name),
            None => format!("GPU {index}"),
        }
    })
}

fn run_inference(model_id: &[u8; 32], prompt: &str, max_tokens: usize) -> Option<String> {
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
        // Upstream answers an oversized prompt with a fixed text instead of dropping it, so the
        // requester learns why rather than waiting for an answer that never comes.
        Err(e) if e.is::<crate::llm_engine::PromptTooLong>() => {
            log::warn!("SlmEngine '{}': {} — answering with the fixed error text", spec.name, e);
            Some(PROMPT_TOO_LONG_ANSWER.to_string())
        }
        Err(e) => {
            log::warn!("SlmEngine '{}' inference error: {}", spec.name, e);
            None
        }
    }
}

// ── Verifiable inference (H15) ───────────────────────────────────────────────
//
// From H15 every answer commits a stage-2 trace of its forward pass (see `trace.rs`) and every
// block commits the trace of its window's block task. The trace is recorded on the GPU by the
// patched ggml-vulkan backend (`llm_engine`); an opening replays the evaluation to capture the
// drawn leaves and is checked here exactly as the node checks it before it leaves the miner.

/// A generated answer and, when its tier is audited, the trace an opening needs.
pub struct Generated {
    pub text: String,
    pub trace: Option<crate::trace::ResponseTrace>,
}

/// The served engine for `model_id` (loaded on demand) and its spec.
fn engine_for(model_id: &[u8; 32]) -> Option<(&'static ModelSpec, Arc<InferenceEngine>)> {
    let specs = *SUPPORTED_SPECS.read().unwrap();
    let spec = *specs.iter().find(|s| &s.model_id == model_id)?;
    if !ensure_loaded(model_id) {
        return None;
    }
    let engine = SERVER.lock().ok()?.as_ref().map(|(_, s)| Arc::clone(s))?;
    Some((spec, engine))
}

/// The possession index of `spec`: the mined one, or built from the GGUF.
fn weight_index(spec: &ModelSpec) -> Option<&'static crate::pom::WeightIndex> {
    let gguf = gguf_path_for(spec).to_string_lossy().into_owned();
    if !crate::pom::get_or_build_index(&spec.model_id, move || crate::pom::WeightIndex::build_from_gguf(&gguf)) {
        log::warn!("SlmEngine '{}': no possession index for an opening", spec.name);
        return None;
    }
    crate::pom::index_for(&spec.model_id)
}

/// Context of a traced answer: the request's prompt and budget, as `llm_engine` sizes requests.
/// A replay recomputes it from the same two numbers.
fn answer_ctx(engine: &InferenceEngine, n_prompt: usize, max_tokens: usize) -> u32 {
    crate::llm_engine::request_ctx(n_prompt, max_tokens, engine.ctx_cap()).unwrap_or(engine.ctx_cap())
}

/// Engines (model, stage-2 kind) whose self-check passed, and those whose self-check failed.
static TRACE_CHECKED: Mutex<Vec<([u8; 32], &'static str)>> = Mutex::new(Vec::new());
static TRACE_FAILED: Mutex<Vec<([u8; 32], &'static str)>> = Mutex::new(Vec::new());

/// Replays the stage-2 trace `trace` for every leaf the draw of `seed` opens and builds the
/// opening body; the replay must reproduce every committed leaf.
#[allow(clippy::too_many_arguments)]
fn open_chain(
    engine: &InferenceEngine,
    tier: u8,
    m: &crate::trace::TierManifest,
    pctx: &keryx_verifiable_inference::prefix::PrefixCtx,
    trace: &crate::trace::ResponseTrace,
    seed: &[u8; 32],
    request_hash: &[u8; 32],
    response_txid: [u8; 32],
    index: &crate::pom::WeightIndex,
    n_ctx: u32,
) -> std::result::Result<Vec<u8>, String> {
    use crate::trace::{captures_match, chain_opening_body, draw_chain, leaf_cap, Captures, TierManifest};
    let raw = TierManifest::raw_chain(tier).ok_or("no stage-2 manifest")?;
    let rec = &trace.record;
    let d = draw_chain(m, pctx, rec, seed).ok_or("no stage-2 draw")?;
    let wanted = d.captures(m.n_products as u64);
    let slot = wanted.iter().map(|&(_, l)| leaf_cap(m, l)).max().unwrap_or(0).div_ceil(4) * 4;
    let bytes = engine.replay_multi(raw, &trace.tokens, rec.n_prompt, &wanted, slot, n_ctx).map_err(|e| e.to_string())?;
    let caps: Captures = wanted.into_iter().zip(bytes).collect();
    if !captures_match(rec, &caps) {
        return Err("the replay does not reproduce the committed trace".to_string());
    }
    chain_opening_body(m, pctx, rec, &trace.tokens, seed, request_hash, response_txid, &d, &caps, index).ok_or_else(|| "no stage-2 opening".to_string())
}

/// Traces a fixed prompt carrying a known request id under the stage-2 manifest, and a block task,
/// then opens and verifies both under two seeds exactly as the node would.
fn chain_self_check(spec: &ModelSpec, engine: &InferenceEngine, tier: u8) -> std::result::Result<(), String> {
    use crate::trace::{chain_ctx, prefix_tokens, verify_chain_opening, ResponseTrace, TierManifest, TraceRecord};
    use keryx_verifiable_inference::block_task::{draw_seed, task_bytes, task_prefix, task_tokens, verify_block_task, TASK_LEN};
    let m = TierManifest::for_chain(tier).ok_or("no stage-2 manifest")?;
    let raw = TierManifest::raw_chain(tier).ok_or("no stage-2 manifest")?;
    let (ctx, pctx) = chain_ctx(tier).ok_or("no stage-2 context")?;
    let index = weight_index(spec).ok_or("no possession index")?;
    let tier_ctx = m.tier_ctx(index);

    let (head, rest) = format_prompt_parts(spec.name, "Reply with one word.").ok_or("no prefix template")?;
    let request_hash: [u8; 32] = std::array::from_fn(|i| (i * 8 + 3) as u8);
    let tokens = prefix_tokens(&pctx, &request_hash);
    let prefix = crate::llm_engine::PromptPrefix { tokens: &tokens, head_tokens: pctx.offset, rest: &rest };
    let max_tokens = 4;
    let g = engine
        .generate_traced(raw, &head, Some(prefix), max_tokens, 0, 0, |n| answer_ctx(engine, n, max_tokens))
        .map_err(|e| e.to_string())?;
    let (lo, hi) = (pctx.offset as usize, pctx.offset as usize + tokens.len());
    if g.tokens.get(lo..hi) != Some(tokens.as_slice()) {
        return Err("the fed prompt does not carry the request prefix".to_string());
    }
    let n_ctx = answer_ctx(engine, g.n_prompt as usize, max_tokens);
    let record = TraceRecord::new_chain(&m, g.n_prompt, g.ubatch, g.leaves, &g.tokens).ok_or("incomplete stage-2 trace")?;
    let trace = ResponseTrace { record, tokens: g.tokens };
    for seed in [[0x11u8; 32], [0x77u8; 32]] {
        let body = open_chain(engine, tier, &m, &pctx, &trace, &seed, &request_hash, [0u8; 32], index, n_ctx)?;
        verify_chain_opening(&tier_ctx, &ctx, &pctx, &trace.record, &seed, &request_hash, &body)?;
    }

    let window_seed = [0x5au8; 32];
    let task: Vec<i32> = task_tokens(&pctx.table, &window_seed).into_iter().map(|t| t as i32).collect();
    let traced = engine.trace_tokens(raw, &task, crate::llm_engine::TASK_CONTEXT_TOKENS).map_err(|e| e.to_string())?;
    let record = TraceRecord::new_chain(&m, TASK_LEN, traced.ubatch, traced.leaves, &task).ok_or("incomplete block task trace")?;
    let task = ResponseTrace { record, tokens: task };
    for block_hash in [[0x21u8; 32], [0x9cu8; 32]] {
        let seed = draw_seed(&block_hash);
        let body = open_chain(engine, tier, &m, &task_prefix(&pctx), &task, &seed, &task_bytes(&window_seed), [0u8; 32], index, crate::llm_engine::TASK_CONTEXT_TOKENS)?;
        let opening = keryx_verifiable_inference::opening::Opening::decode(&body).ok_or("block task opening does not decode")?;
        verify_block_task(&tier_ctx, &ctx, &pctx, &task.record.trace_root, &window_seed, &block_hash, &opening).map_err(|e| format!("block task: {e:?}"))?;
    }
    Ok(())
}

/// Arms the stage-2 trace of `tier` on `engine` for evaluations of up to `max_pos` positions. The
/// first time per model it runs the self-check; a model that fails it is never traced (its
/// answers and blocks would be refused), and the failure is logged once.
fn arm_trace(spec: &ModelSpec, engine: &InferenceEngine, tier: u8, max_pos: u32) -> bool {
    const KIND: &str = "stage-2";
    let Some(raw) = crate::trace::TierManifest::raw_chain(tier) else { return false };
    if TRACE_FAILED.lock().unwrap_or_else(|p| p.into_inner()).contains(&(spec.model_id, KIND)) {
        return false;
    }
    // The self-check traces answer contexts too: arm room for both.
    let room = max_pos.max(2_048);
    if let Err(e) = engine.trace_configure(raw, room) {
        log::error!("SlmEngine '{}': the canonical trace cannot be configured ({e}) — no traced answers or blocks", spec.name);
        TRACE_FAILED.lock().unwrap_or_else(|p| p.into_inner()).push((spec.model_id, KIND));
        return false;
    }
    if TRACE_CHECKED.lock().unwrap_or_else(|p| p.into_inner()).contains(&(spec.model_id, KIND)) {
        return true;
    }
    let t0 = std::time::Instant::now();
    match chain_self_check(spec, engine, tier) {
        Ok(()) => {
            TRACE_CHECKED.lock().unwrap_or_else(|p| p.into_inner()).push((spec.model_id, KIND));
            log::info!("SlmEngine '{}': stage-2 trace self-check passed ({:.1} s) — openings verify as the node does", spec.name, t0.elapsed().as_secs_f64());
            true
        }
        Err(e) => {
            TRACE_FAILED.lock().unwrap_or_else(|p| p.into_inner()).push((spec.model_id, KIND));
            log::error!(
                "SlmEngine '{}': the stage-2 trace self-check FAILED on this GPU ({e}) — no traced answers or block tasks: past H15 this miner cannot produce valid blocks",
                spec.name
            );
            false
        }
    }
}

/// The block task trace of `tier` for the window seeded by `window_seed`, on the engine serving
/// `model_id`. Blocking.
pub fn run_block_task(model_id: &[u8; 32], tier: u8, window_seed: &[u8; 32]) -> Option<crate::trace::ResponseTrace> {
    use keryx_verifiable_inference::block_task::{task_tokens, TASK_LEN};
    let (spec, engine) = engine_for(model_id)?;
    if !arm_trace(spec, &engine, tier, crate::llm_engine::TASK_CONTEXT_TOKENS) {
        return None;
    }
    let m = crate::trace::TierManifest::for_chain(tier)?;
    let raw = crate::trace::TierManifest::raw_chain(tier)?;
    let (_, pctx) = crate::trace::chain_ctx(tier)?;
    let tokens: Vec<i32> = task_tokens(&pctx.table, window_seed).into_iter().map(|t| t as i32).collect();
    let t0 = std::time::Instant::now();
    let traced = match engine.trace_tokens(raw, &tokens, crate::llm_engine::TASK_CONTEXT_TOKENS) {
        Ok(t) => t,
        Err(e) => {
            log::warn!("SlmEngine '{}': block task not traced ({e})", spec.name);
            return None;
        }
    };
    let record = crate::trace::TraceRecord::new_chain(&m, TASK_LEN, traced.ubatch, traced.leaves, &tokens);
    if record.is_none() {
        log::warn!("SlmEngine '{}': incomplete block task trace", spec.name);
    }
    log::debug!("SlmEngine '{}': block task traced in {:.0} ms", spec.name, t0.elapsed().as_secs_f64() * 1e3);
    Some(crate::trace::ResponseTrace { record: record?, tokens })
}

/// The task opening of the block `block_hash` from the task trace of its window, checked as the
/// node will check it. Blocking.
pub fn open_block_task(model_id: &[u8; 32], tier: u8, trace: &crate::trace::ResponseTrace, window_seed: &[u8; 32], block_hash: &[u8; 32]) -> Option<Vec<u8>> {
    use keryx_verifiable_inference::block_task::{draw_seed, task_bytes, task_prefix, verify_block_task};
    let (spec, engine) = engine_for(model_id)?;
    if !arm_trace(spec, &engine, tier, crate::llm_engine::TASK_CONTEXT_TOKENS) {
        return None;
    }
    let index = weight_index(spec)?;
    let m = crate::trace::TierManifest::for_chain(tier)?;
    let (ctx, pctx) = crate::trace::chain_ctx(tier)?;
    let body = match open_chain(
        &engine,
        tier,
        &m,
        &task_prefix(&pctx),
        trace,
        &draw_seed(block_hash),
        &task_bytes(window_seed),
        [0u8; 32],
        index,
        crate::llm_engine::TASK_CONTEXT_TOKENS,
    ) {
        Ok(b) => b,
        Err(e) => {
            log::warn!("SlmEngine '{}': block task opening failed ({e})", spec.name);
            return None;
        }
    };
    let opening = keryx_verifiable_inference::opening::Opening::decode(&body)?;
    if let Err(e) = verify_block_task(&m.tier_ctx(index), &ctx, &pctx, &trace.record.trace_root, window_seed, block_hash, &opening) {
        log::error!("SlmEngine '{}': block task opening fails its self-check ({e:?})", spec.name);
        return None;
    }
    Some(body)
}

/// Cuts `text` to the longest answer the request's token budget allows, on a char boundary.
fn truncate_answer(mut text: String, max_tokens: usize) -> String {
    let cap = keryx_verifiable_inference::shape::max_answer_len(max_tokens.min(u32::MAX as usize) as u32).min(usize::MAX as u64) as usize;
    if text.len() > cap {
        let mut end = cap;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
    }
    text
}

/// Traced inference for an audited `tier`: the answer and its stage-2 trace, with the request id
/// `request_hash` in the prompt prefix. `prompt_len` is the request's plaintext prompt length (the
/// shape rule pads the traced prompt to it). Blocking. `request` names it in the `--log-ai` file.
#[allow(clippy::too_many_arguments)]
pub fn load_and_run_traced_inference(
    model_id: &[u8; 32],
    prompt: &str,
    max_tokens: usize,
    tier: u8,
    prompt_len: u32,
    request_hash: &[u8; 32],
    request: &str,
) -> Option<Generated> {
    let started = std::time::Instant::now();
    if ai_log_path().is_some() {
        let model = SUPPORTED_SPECS
            .read()
            .ok()
            .and_then(|specs| specs.iter().find(|s| &s.model_id == model_id).map(|s| s.name))
            .unwrap_or("unknown model");
        ai_log(request, format_args!("prompt ({model}, {}, max_tokens={max_tokens}, traced): {prompt:?}", inference_gpu_label()));
    }
    let answer = run_traced_inference(model_id, prompt, max_tokens, tier, prompt_len, request_hash);
    let elapsed = started.elapsed().as_secs_f64();
    match &answer {
        Some(g) => ai_log(request, format_args!("answer ({elapsed:.2} s, {}): {:?}", if g.trace.is_some() { "traced" } else { "untraced" }, g.text)),
        None => ai_log(request, format_args!("no answer ({elapsed:.2} s)")),
    }
    answer
}

fn run_traced_inference(model_id: &[u8; 32], prompt: &str, max_tokens: usize, tier: u8, prompt_len: u32, request_hash: &[u8; 32]) -> Option<Generated> {
    use crate::trace::{chain_ctx, prefix_tokens, ResponseTrace, TierManifest, TraceRecord};
    let (spec, engine) = engine_for(model_id)?;
    if !arm_trace(spec, &engine, tier, engine.ctx_cap()) {
        return None;
    }
    let m = TierManifest::for_chain(tier)?;
    let raw = TierManifest::raw_chain(tier)?;
    let (_, pctx) = chain_ctx(tier)?;
    let Some((head, rest)) = format_prompt_parts(spec.name, prompt) else {
        log::error!("SlmEngine '{}': no prefix template — a traced answer cannot be built", spec.name);
        return None;
    };
    let tokens = prefix_tokens(&pctx, request_hash);
    let prefix = crate::llm_engine::PromptPrefix { tokens: &tokens, head_tokens: pctx.offset, rest: &rest };
    let min_prompt = keryx_verifiable_inference::shape::min_prompt_positions(prompt_len);
    let per_position = keryx_verifiable_inference::shape::ANSWER_BYTES_PER_POSITION;
    let g = match engine.generate_traced(raw, &head, Some(prefix), max_tokens, min_prompt, per_position, |n| answer_ctx(&engine, n, max_tokens)) {
        Ok(g) => g,
        Err(e) if e.is::<crate::llm_engine::PromptTooLong>() => {
            log::warn!("SlmEngine '{}': {e} — response dropped", spec.name);
            return None;
        }
        Err(e) => {
            log::warn!("SlmEngine '{}': traced generation failed ({e}) — response dropped", spec.name);
            return None;
        }
    };
    let (lo, hi) = (pctx.offset as usize, pctx.offset as usize + tokens.len());
    if g.tokens.get(lo..hi) != Some(tokens.as_slice()) {
        log::warn!("SlmEngine '{}': the fed prompt does not carry the request prefix — response dropped", spec.name);
        return None;
    }
    let record = TraceRecord::new_chain(&m, g.n_prompt, g.ubatch, g.leaves, &g.tokens);
    if record.is_none() {
        log::warn!("SlmEngine '{}': incomplete trace — answering without a commitment", spec.name);
    }
    let text = strip_think(&g.text).trim().to_string();
    if text.is_empty() {
        log::warn!("SlmEngine '{}': empty completion", spec.name);
        return None;
    }
    Some(Generated { text: truncate_answer(text, max_tokens), trace: record.map(|record| ResponseTrace { record, tokens: g.tokens }) })
}

/// The opening of an audited stage-2 response (`max_tokens` is its request's budget, which sized
/// the context it ran in), replayed on the engine serving `model_id` and checked as the node would.
#[allow(clippy::too_many_arguments)]
pub fn open_trace(
    model_id: &[u8; 32],
    tier: u8,
    trace: &crate::trace::ResponseTrace,
    seed: &[u8; 32],
    response_txid: [u8; 32],
    request_hash: &[u8; 32],
    max_tokens: u32,
) -> Option<Vec<u8>> {
    use crate::trace::{chain_ctx, verify_chain_opening, TierManifest, STAGE_CHAIN};
    if trace.record.stage != STAGE_CHAIN {
        log::warn!("SlmEngine: only stage-2 traces are opened by this miner");
        return None;
    }
    let (spec, engine) = engine_for(model_id)?;
    let n_ctx = answer_ctx(&engine, trace.record.n_prompt as usize, max_tokens as usize);
    if !arm_trace(spec, &engine, tier, n_ctx.max(engine.ctx_cap())) {
        return None;
    }
    let index = weight_index(spec)?;
    let m = TierManifest::for_chain(tier)?;
    let (ctx, pctx) = chain_ctx(tier)?;
    let body = match open_chain(&engine, tier, &m, &pctx, trace, seed, request_hash, response_txid, index, n_ctx) {
        Ok(b) => b,
        Err(e) => {
            log::warn!("SlmEngine '{}': stage-2 opening failed ({e})", spec.name);
            return None;
        }
    };
    if let Err(e) = verify_chain_opening(&m.tier_ctx(index), &ctx, &pctx, &trace.record, seed, request_hash, &body) {
        log::error!("SlmEngine '{}': stage-2 opening fails its self-check ({e})", spec.name);
        return None;
    }
    Some(body)
}

#[cfg(test)]
mod tests {
    use super::{exe_dir, parse_ai_log_setting, pom_fits, write_ai_log_entry};
    use std::path::PathBuf;

    #[test]
    fn ai_log_setting() {
        assert_eq!(parse_ai_log_setting(""), None);
        assert_eq!(parse_ai_log_setting("0"), None);
        assert_eq!(parse_ai_log_setting("1"), Some(exe_dir().join("ai-requests.log")));
        assert_eq!(parse_ai_log_setting("D:/logs/ai.log"), Some(PathBuf::from("D:/logs/ai.log")));
    }

    #[test]
    fn ai_log_entry_is_one_escaped_line() {
        let at = time::macros::datetime!(2026-10-17 08:30:05 UTC);
        let mut out = Vec::new();
        let prompt = "line one\nline two \"quoted\" — Ўзбек";
        write_ai_log_entry(&mut out, at, "pool chat 7", format_args!("prompt (max_tokens=8): {prompt:?}")).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "2026-10-17 08:30:05Z [pool chat 7] prompt (max_tokens=8): \"line one\\nline two \\\"quoted\\\" — Ўзбек\"\n"
        );
    }

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

    /// The tier and model the GPU trace tests run: `KERYX_TEST_TIER` (default 0), whose model
    /// `KERYX_TEST_GGUF` must be.
    fn test_tier() -> (u8, &'static crate::models::ModelSpec) {
        use crate::models::Tier;
        let tier: u8 = std::env::var("KERYX_TEST_TIER").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let t = [Tier::VeryLight, Tier::Light, Tier::Default, Tier::High, Tier::VeryHigh][tier as usize];
        (tier, crate::models::pom_model_for_tier(u64::MAX, t).expect("the tier's model"))
    }

    /// The stage-2 self-check the miner runs before it traces: a traced answer and a block task on
    /// this GPU, replayed, opened under two seeds and verified exactly as the node verifies. Run with:
    ///   [KERYX_TEST_TIER=<0..4>] KERYX_TEST_GGUF=<that tier's model.gguf> cargo test --release --lib -- --ignored --nocapture stage2_trace_self_check
    #[test]
    #[ignore = "loads a tier model onto the GPU; set KERYX_TEST_GGUF (and KERYX_TEST_TIER) to run"]
    fn stage2_trace_self_check() {
        let _ = env_logger::builder().is_test(true).filter_level(log::LevelFilter::Info).parse_default_env().try_init();
        let Ok(gguf) = std::env::var("KERYX_TEST_GGUF") else {
            eprintln!("SKIP: KERYX_TEST_GGUF not set");
            return;
        };
        let (tier, spec) = test_tier();
        let path = gguf.clone();
        assert!(crate::pom::get_or_build_index(&spec.model_id, move || crate::pom::WeightIndex::build_from_gguf(&path)));
        let engine = super::InferenceEngine::launch(&gguf).expect("engine launch");
        let raw = crate::trace::TierManifest::raw_chain(tier).expect("the tier's manifest");
        engine.trace_configure(raw, 2_048).expect("trace configure");
        let t0 = std::time::Instant::now();
        super::chain_self_check(spec, &engine, tier).expect("stage-2 self-check");
        eprintln!("stage-2 self-check passed in {:.1} s", t0.elapsed().as_secs_f64());
    }

    /// Openings under many seeds, the way production traces: a shape-padded answer to a long prompt
    /// (so the draws reach generated positions, the head and the padding) and the block task. Run with:
    ///   [KERYX_TEST_TIER=<0..4>] KERYX_TEST_GGUF=<that tier's model.gguf> [KERYX_TEST_SEEDS=48] cargo test --release --lib -- --ignored --nocapture stage2_openings_many_seeds
    #[test]
    #[ignore = "loads a tier model onto the GPU; set KERYX_TEST_GGUF (and KERYX_TEST_TIER) to run"]
    fn stage2_openings_many_seeds() {
        use crate::trace::{chain_ctx, prefix_tokens, verify_chain_opening, ResponseTrace, TierManifest, TraceRecord};
        use keryx_verifiable_inference::block_task::{draw_seed, task_bytes, task_prefix, task_tokens, verify_block_task, TASK_LEN};
        use keryx_verifiable_inference::shape;
        let _ = env_logger::builder().is_test(true).filter_level(log::LevelFilter::Info).parse_default_env().try_init();
        let Ok(gguf) = std::env::var("KERYX_TEST_GGUF") else {
            eprintln!("SKIP: KERYX_TEST_GGUF not set");
            return;
        };
        let n_seeds: u32 = std::env::var("KERYX_TEST_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(48);
        let (tier, spec) = test_tier();
        let path = gguf.clone();
        assert!(crate::pom::get_or_build_index(&spec.model_id, move || crate::pom::WeightIndex::build_from_gguf(&path)));
        let index = crate::pom::index_for(&spec.model_id).unwrap();
        let engine = super::InferenceEngine::launch(&gguf).expect("engine launch");
        let raw = TierManifest::raw_chain(tier).unwrap();
        let m = TierManifest::for_chain(tier).unwrap();
        let (ctx, pctx) = chain_ctx(tier).unwrap();
        let tier_ctx = m.tier_ctx(index);
        engine.trace_configure(raw, 4_096).expect("trace configure");

        let prompt = "List three facts about the Moon, one short sentence each. ".repeat(12);
        let max_tokens = 40usize;
        let (head, rest) = super::format_prompt_parts(spec.name, &prompt).unwrap();
        let request_hash: [u8; 32] = *blake3::hash(b"many-seeds request").as_bytes();
        let tokens = prefix_tokens(&pctx, &request_hash);
        let prefix = crate::llm_engine::PromptPrefix { tokens: &tokens, head_tokens: pctx.offset, rest: &rest };
        let min_prompt = shape::min_prompt_positions(prompt.len() as u32);
        let g = engine
            .generate_traced(raw, &head, Some(prefix), max_tokens, min_prompt, shape::ANSWER_BYTES_PER_POSITION, |n| super::answer_ctx(&engine, n, max_tokens))
            .expect("traced generation");
        let text = super::truncate_answer(super::strip_think(&g.text).trim().to_string(), max_tokens);
        eprintln!("answer ({} prompt + {} positions, min prompt {min_prompt}): {text:?}", g.n_prompt, g.n_pos - g.n_prompt);
        assert!(shape::shape_ok(prompt.len() as u32, text.len() as u32, max_tokens as u32, g.n_prompt, g.n_pos), "shape rule");
        let n_ctx = super::answer_ctx(&engine, g.n_prompt as usize, max_tokens);
        let record = TraceRecord::new_chain(&m, g.n_prompt, g.ubatch, g.leaves, &g.tokens).expect("complete trace");
        let trace = ResponseTrace { record, tokens: g.tokens };
        let t0 = std::time::Instant::now();
        let mut failed = Vec::new();
        for s in 0..n_seeds {
            let seed = *blake3::hash(&s.to_le_bytes()).as_bytes();
            let res = super::open_chain(&engine, tier, &m, &pctx, &trace, &seed, &request_hash, [0u8; 32], index, n_ctx)
                .and_then(|body| verify_chain_opening(&tier_ctx, &ctx, &pctx, &trace.record, &seed, &request_hash, &body));
            if let Err(e) = res {
                eprintln!("answer seed {s}: {e}");
                failed.push(s);
            }
        }
        eprintln!("answer: {} of {n_seeds} openings verify ({:.1} s)", n_seeds as usize - failed.len(), t0.elapsed().as_secs_f64());

        let window_seed = *blake3::hash(b"many-seeds window").as_bytes();
        let task: Vec<i32> = task_tokens(&pctx.table, &window_seed).into_iter().map(|t| t as i32).collect();
        let traced = engine.trace_tokens(raw, &task, crate::llm_engine::TASK_CONTEXT_TOKENS).expect("task trace");
        let record = TraceRecord::new_chain(&m, TASK_LEN, traced.ubatch, traced.leaves, &task).expect("complete task trace");
        let task = ResponseTrace { record, tokens: task };
        let t0 = std::time::Instant::now();
        let mut task_failed = Vec::new();
        for s in 0..n_seeds {
            let block_hash = *blake3::hash(&(s + 1_000_000).to_le_bytes()).as_bytes();
            let res = super::open_chain(
                &engine,
                tier,
                &m,
                &task_prefix(&pctx),
                &task,
                &draw_seed(&block_hash),
                &task_bytes(&window_seed),
                [0u8; 32],
                index,
                crate::llm_engine::TASK_CONTEXT_TOKENS,
            )
            .and_then(|body| {
                let opening = keryx_verifiable_inference::opening::Opening::decode(&body).ok_or("no decode")?;
                verify_block_task(&tier_ctx, &ctx, &pctx, &task.record.trace_root, &window_seed, &block_hash, &opening).map_err(|e| format!("{e:?}"))
            });
            if let Err(e) = res {
                eprintln!("task block {s}: {e}");
                task_failed.push(s);
            }
        }
        eprintln!("block task: {} of {n_seeds} openings verify ({:.1} s)", n_seeds as usize - task_failed.len(), t0.elapsed().as_secs_f64());
        assert!(failed.is_empty() && task_failed.is_empty(), "answer seeds {failed:?}, task blocks {task_failed:?}");
    }

    /// Diagnostic: every plain/gated product at one position of the block task, x and y captured
    /// by a replay and the y of sampled rows recomputed from the committed x and the weights.
    #[test]
    #[ignore = "loads a tier model onto the GPU; set KERYX_TEST_GGUF (and KERYX_TEST_TIER) to run"]
    fn stage2_product_scan() {
        use keryx_verifiable_inference::{dot::canonical_dot, manifest, q8};
        let _ = env_logger::builder().is_test(true).filter_level(log::LevelFilter::Info).parse_default_env().try_init();
        let Ok(gguf) = std::env::var("KERYX_TEST_GGUF") else {
            eprintln!("SKIP: KERYX_TEST_GGUF not set");
            return;
        };
        let t: u32 = std::env::var("KERYX_TEST_POS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
        let (tier, spec) = test_tier();
        let path = gguf.clone();
        assert!(crate::pom::get_or_build_index(&spec.model_id, move || crate::pom::WeightIndex::build_from_gguf(&path)));
        let index = crate::pom::index_for(&spec.model_id).unwrap();
        let engine = super::InferenceEngine::launch(&gguf).expect("engine launch");
        let raw = crate::trace::TierManifest::raw_chain(tier).expect("the tier's manifest");
        let m = crate::trace::TierManifest::for_chain(tier).unwrap();
        engine.trace_configure(raw, 2_048).expect("trace configure");
        let (_, pctx) = crate::trace::chain_ctx(tier).unwrap();
        let task: Vec<i32> = keryx_verifiable_inference::block_task::task_tokens(&pctx.table, &[0x5au8; 32]).into_iter().map(|t| t as i32).collect();
        let n_ctx = crate::llm_engine::TASK_CONTEXT_TOKENS;
        let row_bytes = |e: &manifest::Entry, row: u64| -> Vec<u8> {
            let (first, last, off) = e.row_chunks(0, row).unwrap();
            let mut b = Vec::new();
            for c in first..=last {
                b.extend_from_slice(&index.read_chunk_bytes(c));
            }
            b[off..off + e.row_bytes as usize].to_vec()
        };
        let np = m.n_products;
        let (cctx, _) = crate::trace::chain_ctx(tier).unwrap();
        // Output-only products (the head, a trimmed last layer) have no leaves at a task position.
        let ranks: Vec<u32> = (0..np)
            .filter(|&r| {
                let e = m.entry(r as u64);
                let last = matches!(e.chain, manifest::Chain::Product(pl) if cctx.trim_last && pl.layer + 1 == cctx.n_layers);
                e.role == manifest::ROLE_PRODUCT && !last
            })
            .collect();
        eprintln!("SCAN {} of {np} products, trim_last {}", ranks.len(), cctx.trim_last);
        for group in ranks.chunks(16) {
            // The up half of a gated pair carries no output leaf of its own (the GLU is the gate's).
            let group: Vec<u32> = group.iter().copied().filter(|&r| m.entry(r as u64).glu_form != manifest::FORM_PAIR_UP).collect();
            let group = &group[..];
            let caps: Vec<(u32, u32)> = group.iter().flat_map(|&r| [(t, 2 * r), (t, 2 * r + 1)]).collect();
            let slot = caps.iter().map(|&(_, l)| crate::trace::leaf_cap(&m, l)).max().unwrap().div_ceil(4) * 4;
            let bytes = match engine.replay_multi(raw, &task, task.len() as u32, &caps, slot, n_ctx) {
                Ok(b) => b,
                Err(err) => {
                    eprintln!("SCAN group {group:?}: {err}");
                    for &r in group {
                        for leaf in [2 * r, 2 * r + 1] {
                            let one = engine.replay_multi(raw, &task, task.len() as u32, &[(t, leaf)], slot, n_ctx);
                            if let Err(e) = one {
                                eprintln!("SCAN r{r:3} {:32} leaf {leaf}: {e}", m.entry(r as u64).name);
                            }
                        }
                    }
                    continue;
                }
            };
            for (i, &r) in group.iter().enumerate() {
                let e = m.entry(r as u64);
                let (x, y) = (&bytes[2 * i], &bytes[2 * i + 1]);
                let yv: Vec<f32> = y.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                if e.n_exp != 1 || e.glu_form == manifest::FORM_PAIR_UP {
                    eprintln!("SCAN r{r:3} {:32} form {} n_exp {} x {} y {} (skipped)", e.name, e.glu_form, e.n_exp, x.len(), y.len());
                    continue;
                }
                let m_out = if e.glu_form == manifest::FORM_SPLIT { e.m / 2 } else { e.m };
                let mut worst = 0f64;
                let mut sample = String::new();
                for s in 0..16u64 {
                    let row = (s * 2654435761) % m_out;
                    let (y_hat, scale) = match e.glu_form {
                        manifest::FORM_NONE => {
                            let d = canonical_dot(e.wtype, &row_bytes(e, row), e.k as usize, x, q8::MMVQ).unwrap();
                            // A biased product commits its output after the bias add.
                            let b = match e.bias {
                                manifest::NONE => 0.0,
                                bi => {
                                    let be = m.entry(bi as u64);
                                    let c = index.read_chunk_bytes(be.chunk_base + row * 4 / 32);
                                    let o = (row * 4 % 32) as usize;
                                    f32::from_le_bytes(c[o..o + 4].try_into().unwrap()) as f64
                                }
                            };
                            (d.y_hat + b, d.scale + b.abs())
                        }
                        _ => {
                            let up = if e.glu_form == manifest::FORM_SPLIT { (e, row + m_out) } else { (m.entry(e.partner as u64), row) };
                            let g = canonical_dot(e.wtype, &row_bytes(e, row), e.k as usize, x, q8::MMVQ).unwrap();
                            let u = canonical_dot(up.0.wtype, &row_bytes(up.0, up.1), e.k as usize, x, q8::MMVQ).unwrap();
                            let f = g.y_hat / (1.0 + (-g.y_hat).exp());
                            (f * u.y_hat, g.scale.max(u.scale))
                        }
                    };
                    let got = yv.get(row as usize).copied().unwrap_or(f32::NAN) as f64;
                    let rel = (got - y_hat).abs() / scale.max(1e-30);
                    worst = worst.max(rel);
                    if s < 2 {
                        sample += &format!(" [{row}: {got:.5} vs {y_hat:.5}]");
                    }
                }
                eprintln!(
                    "SCAN r{r:3} {:32} wt {:2} k {:5} m {:6} form {} x {} y {} worst {worst:.2e}{}{sample}",
                    e.name,
                    e.wtype,
                    e.k,
                    e.m,
                    e.glu_form,
                    x.len(),
                    y.len(),
                    if worst <= 1e-5 { " ok" } else { " BAD" }
                );
            }
        }
    }
}
