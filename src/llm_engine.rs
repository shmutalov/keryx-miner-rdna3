//! Phase-1 in-process OPoI inference — llama.cpp via FFI (`llama-cpp-2`, **Vulkan** backend).
//!
//! The miner's ONLY inference engine (the external llama-server child process is gone):
//! same GGUF models, same ggml Vulkan backend, same greedy (temperature-0) decoding through
//! the model's own chat template — but linked into the miner, so there is no HTTP hop, no
//! child-process lifecycle, and (Phase 2) the PoM walk can eventually read inference's own
//! resident weight buffers instead of keeping a second VRAM copy.
//!

use std::ffi::{c_char, c_void, CStr};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Result};
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_sys_2::{
    ggml_log_level, GGML_LOG_LEVEL_CONT, GGML_LOG_LEVEL_ERROR, GGML_LOG_LEVEL_INFO, GGML_LOG_LEVEL_NONE,
    GGML_LOG_LEVEL_WARN,
};
use log::info;

/// Context cap for a GGUF outside the lineup (tests): the old fixed `-c 4096`.
const DEFAULT_CTX_CAP: u32 = 4096;
/// Tokens per prompt decode call (upstream's logical `n_batch`).
const N_BATCH: u32 = 2048;
/// Physical batch: bounds the compute buffers, which grow with it and not with the context
/// (upstream's `n_ubatch`).
const N_UBATCH: u32 = 512;
/// llama.cpp pads the KV cache to 256 cells; request contexts are sized in the same steps.
const CTX_STEP: u32 = 256;
/// Headroom kept free past the prompt, as upstream's `n > n_ctx - 16` check.
const PROMPT_MARGIN: u32 = 16;

/// The prompt does not fit the model's context cap. `slm` answers such a request with a fixed text
/// (upstream's `PROMPT_TOO_LONG_ANSWER`) instead of dropping it.
#[derive(Debug)]
pub struct PromptTooLong {
    pub tokens: usize,
    pub ctx_cap: u32,
}

impl std::fmt::Display for PromptTooLong {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} prompt tokens do not fit the {}-token context", self.tokens, self.ctx_cap)
    }
}

impl std::error::Error for PromptTooLong {}

/// Context window for one request: the prompt plus its token budget, rounded up to [`CTX_STEP`]
/// and bounded by the model's `ctx_cap`. Each request gets a fresh context, so a short chat never
/// pays for the cap's KV cache while a long private prompt still fits.
pub fn request_ctx(n_prompt: usize, max_tokens: usize, ctx_cap: u32) -> std::result::Result<u32, PromptTooLong> {
    if n_prompt as u64 + PROMPT_MARGIN as u64 > ctx_cap as u64 {
        return Err(PromptTooLong { tokens: n_prompt, ctx_cap });
    }
    let want = (n_prompt as u64 + max_tokens as u64 + 1).max(n_prompt as u64 + PROMPT_MARGIN as u64);
    let stepped = want.div_ceil(CTX_STEP as u64) * CTX_STEP as u64;
    Ok(stepped.min(ctx_cap as u64) as u32)
}

/// Process-wide ggml backend guard: `llama_backend_init` must run exactly once per process
/// (`LlamaBackend::init` errors on a second call). Never torn down — model switches drop the
/// [`LlamaEngine`] (freeing the model's VRAM) but keep the backend alive.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// Log target of llama.cpp/ggml's own output: `-d` or `RUST_LOG=llama.cpp=debug` shows it all.
pub const LLAMA_LOG_TARGET: &str = "llama.cpp";

/// llama.cpp warnings every request triggers by design — each runs a fresh context sized to the
/// request, far below the model's training context. Logged at debug, not warn.
const EXPECTED_LLAMA_WARNINGS: &[&str] = &["the full capacity of the model will not be utilized"];

/// llama.cpp/ggml log callback: their output goes to the miner's logger under
/// [`LLAMA_LOG_TARGET`] instead of raw stderr. INFO (the metadata dump at every model load, the
/// context setup at every request) becomes debug and DEBUG becomes trace; warnings and errors
/// keep their level.
extern "C" fn llama_log_to_log(level: ggml_log_level, text: *const c_char, _: *mut c_void) {
    // A line can arrive in pieces: CONT carries on the previous piece's level, and text without
    // a trailing newline waits for the rest.
    static PENDING: Mutex<(ggml_log_level, String)> = Mutex::new((GGML_LOG_LEVEL_NONE, String::new()));
    if text.is_null() {
        return;
    }
    // SAFETY: llama.cpp passes a NUL-terminated string that outlives the call.
    let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    let mut guard = PENDING.lock().unwrap_or_else(|p| p.into_inner());
    let (pending_level, pending) = &mut *guard;
    if level != GGML_LOG_LEVEL_CONT {
        // A new message: an unterminated previous one is complete as it stands.
        emit_llama_log(*pending_level, pending);
        pending.clear();
        *pending_level = level;
    }
    pending.push_str(&text);
    while let Some(end) = pending.find('\n') {
        emit_llama_log(*pending_level, &pending[..end]);
        pending.drain(..=end);
    }
}

fn emit_llama_log(level: ggml_log_level, line: &str) {
    let line = line.trim_end();
    if line.is_empty() {
        return;
    }
    let level = match level {
        GGML_LOG_LEVEL_ERROR => log::Level::Error,
        GGML_LOG_LEVEL_WARN if !EXPECTED_LLAMA_WARNINGS.iter().any(|w| line.contains(w)) => log::Level::Warn,
        GGML_LOG_LEVEL_WARN | GGML_LOG_LEVEL_INFO => log::Level::Debug,
        _ => log::Level::Trace,
    };
    log::log!(target: LLAMA_LOG_TARGET, level, "{line}");
}

fn backend() -> Result<&'static LlamaBackend> {
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let _g = INIT.lock().unwrap_or_else(|p| p.into_inner());
    if BACKEND.get().is_none() {
        // Before init: backend registration already logs (the Vulkan device list).
        // SAFETY: a plain callback without user data. llama_log_set also points ggml's logger at
        // it; setting ggml's directly keeps that true should the two ever be split.
        unsafe {
            llama_cpp_sys_2::llama_log_set(Some(llama_log_to_log), std::ptr::null_mut());
            llama_cpp_sys_2::ggml_log_set(Some(llama_log_to_log), std::ptr::null_mut());
        }
        let b = LlamaBackend::init().map_err(|e| anyhow!("llama backend init failed: {e}"))?;
        let _ = BACKEND.set(b);
    }
    Ok(BACKEND.get().expect("backend just initialised"))
}

/// One resident model bound to the inference GPU, serving chat completions in-process.
/// The `slm.rs` engine seam (`launch` / `chat`) mirrors `LlamaServer` exactly.
pub struct LlamaEngine {
    model: LlamaModel,
    /// Serializes chats: each request runs a fresh short-lived context (its own KV cache),
    /// exactly like the stateless per-request usage of llama-server's chat endpoint.
    lock: Mutex<()>,
    /// Largest context (tokens) a request may allocate: the lineup model's `ctx_cap`, or
    /// `KERYX_LLAMA_CTX` when set.
    ctx_cap: u32,
    /// The canonical trace the GPU is configured with, if any.
    trace: Mutex<Option<TraceConfig>>,
}

impl Drop for LlamaEngine {
    fn drop(&mut self) {
        if self.trace.get_mut().map_or(false, |t| t.is_some()) {
            // SAFETY: plain call into the backend; nothing traced runs once the engine drops.
            unsafe { llama_cpp_sys_2::ggml_backend_vk_keryx_trace_release(TRACE_DEVICE) };
        }
    }
}

// ── Canonical trace (H15 verifiable inference) ───────────────────────────────

/// Most captures one replay takes.
pub const REPLAY_MAX_CAPTURES: usize = 32;
/// Tokens per traced decode: the q8_1 mat-vec width of the Vulkan backend, so every product commits
/// its input in the MMVQ layout the verifier recomputes it from (`verify::layout_for` reads this
/// batch size back from each position's meta leaf).
pub const TRACE_UBATCH: u32 = 8;
/// Context of a block task trace and of its replays: the attention view spans the context, so a
/// replay must run in the same one to reproduce the trace.
pub const TASK_CONTEXT_TOKENS: u32 = 64;
/// Largest captured leaf the engine hands back.
const CAPTURE_CAP: usize = 4 << 20;
/// ggml device index of the inference GPU: the pinned, only visible one on a multi-GPU rig.
const TRACE_DEVICE: usize = 0;

/// The manifest (BLAKE3 of its bytes) and position capacity the GPU trace is configured with.
struct TraceConfig {
    manifest: [u8; 32],
    max_pos: u32,
    leaves: usize,
}

/// The request-bound prefix of a traced prompt: its token ids, the token count of the prompt head
/// they follow, and the rest of the prompt after them.
pub struct PromptPrefix<'a> {
    pub tokens: &'a [i32],
    pub head_tokens: u32,
    pub rest: &'a str,
}

/// The prompt head does not tokenize to the prefix offset the consensus pins for the model.
#[derive(Debug)]
pub struct PromptHeadMismatch(pub String);

impl std::fmt::Display for PromptHeadMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PromptHeadMismatch {}

/// A traced response: its text and everything needed to commit to and replay it.
#[derive(Debug, Clone, Default)]
pub struct TracedGeneration {
    pub text: String,
    /// Every token fed to the model: the prompt, then each decoded token, then answer padding.
    pub tokens: Vec<i32>,
    pub n_prompt: u32,
    /// Traced positions (`tokens.len()`).
    pub n_pos: u32,
    pub ubatch: Vec<u32>,
    pub leaves: Vec<[u8; 32]>,
}

/// A traced evaluation of raw tokens: the batch size and leaves of every position.
pub struct TracedTokens {
    pub ubatch: Vec<u32>,
    pub leaves: Vec<[u8; 32]>,
}

/// The backend's explanation of the last traced evaluation's failure, or `fallback`.
fn trace_error(fallback: &str) -> anyhow::Error {
    // SAFETY: returns a NUL-terminated string valid until the next call on this thread.
    let detail = unsafe { CStr::from_ptr(llama_cpp_sys_2::ggml_backend_vk_keryx_trace_error(TRACE_DEVICE)) }
        .to_string_lossy()
        .into_owned();
    if detail.is_empty() {
        anyhow!("{fallback}")
    } else {
        anyhow!("{fallback}: {detail}")
    }
}

/// Feeds `tokens` at positions `start..` in batches of at most [`TRACE_UBATCH`], announcing each
/// batch to the trace before its decode; only position `logits_at` produces logits.
fn feed_traced(ctx: &mut LlamaContext<'_>, batch: &mut LlamaBatch, tokens: &[LlamaToken], start: usize, logits_at: Option<usize>) -> Result<()> {
    for (c, chunk) in tokens.chunks(TRACE_UBATCH as usize).enumerate() {
        let base = start + c * TRACE_UBATCH as usize;
        batch.clear();
        for (j, &tok) in chunk.iter().enumerate() {
            batch.add(tok, (base + j) as i32, &[0], Some(base + j) == logits_at)?;
        }
        // SAFETY: plain call into the backend.
        if unsafe { llama_cpp_sys_2::ggml_backend_vk_keryx_trace_ubatch(TRACE_DEVICE, base as u32, chunk.len() as u32) } != 0 {
            return Err(trace_error("trace batch refused"));
        }
        ctx.decode(batch)?;
    }
    Ok(())
}

/// Ends the traced evaluation and reads back its leaves and per-position batch sizes.
fn read_trace(n_leaves: usize, expect_pos: usize) -> Result<(Vec<u32>, Vec<[u8; 32]>)> {
    // SAFETY: plain calls into the backend; the buffers are sized from the reported position count.
    unsafe {
        let ended = llama_cpp_sys_2::ggml_backend_vk_keryx_trace_end(TRACE_DEVICE);
        let mut n_pos = 0u32;
        if ended != 0 || llama_cpp_sys_2::ggml_backend_vk_keryx_trace_read(TRACE_DEVICE, std::ptr::null_mut(), 0, std::ptr::null_mut(), 0, &mut n_pos) != 0 {
            return Err(trace_error("traced evaluation failed"));
        }
        if n_pos as usize != expect_pos {
            return Err(anyhow!("{n_pos} positions traced for {expect_pos} fed"));
        }
        let mut leaves = vec![[0u8; 32]; n_pos as usize * n_leaves];
        let mut ubatch = vec![0u32; n_pos as usize];
        let rc = llama_cpp_sys_2::ggml_backend_vk_keryx_trace_read(
            TRACE_DEVICE,
            leaves.as_mut_ptr() as *mut u8,
            leaves.len() * 32,
            ubatch.as_mut_ptr(),
            ubatch.len(),
            &mut n_pos,
        );
        if rc != 0 {
            return Err(trace_error("trace read failed"));
        }
        Ok((ubatch, leaves))
    }
}

/// Disarms a traced evaluation left behind by an error.
fn abandon_trace() {
    // SAFETY: plain call into the backend.
    unsafe { llama_cpp_sys_2::ggml_backend_vk_keryx_trace_end(TRACE_DEVICE) };
}

impl LlamaEngine {
    /// Load `gguf_path` fully onto the GPU (all layers — the in-process `-ngl 999`) and get
    /// ready to serve. Mirrors `LlamaServer::launch`, minus the child process and health poll.
    pub fn launch(gguf_path: &str) -> Result<Self> {
        // Multi-GPU rigs: pin ggml's Vulkan enumeration to the inference device BEFORE the
        // backend spins up, for the same reason llama_server.rs sets it on the child: left
        // alone, ggml layer-splits across every visible device, fighting the mining workers
        // on the other cards. Same loader, same order → the raw index maps 1:1. A user-set
        // value always wins.
        if std::env::var_os("GGML_VK_VISIBLE_DEVICES").is_none()
            && keryx_vulkan::enumerate_devices().len() > 1
        {
            let infer = keryx_vulkan::inference_device_index();
            info!("llm-engine: multi-GPU rig — pinning inference to Vulkan device {infer} (GGML_VK_VISIBLE_DEVICES)");
            std::env::set_var("GGML_VK_VISIBLE_DEVICES", infer.to_string());
        }

        let backend = backend()?;
        info!("llm-engine: loading {} (Vulkan, all layers on GPU, in-process)", gguf_path);
        // u32::MAX = offload every layer (the crate's "all" sentinel, like -ngl 999).
        let params = LlamaModelParams::default().with_n_gpu_layers(u32::MAX);
        let model = LlamaModel::load_from_file(backend, Path::new(gguf_path), &params)
            .map_err(|e| anyhow!("llm-engine: model load failed: {e}"))?;
        let ctx_cap = std::env::var("KERYX_LLAMA_CTX")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&c| c >= CTX_STEP)
            .unwrap_or_else(|| crate::models::spec_for_gguf(gguf_path).map_or(DEFAULT_CTX_CAP, |m| m.ctx_cap));
        info!("llm-engine: model resident, ready to serve (context up to {ctx_cap} tokens per request, q8_0 KV + flash attention)");
        Ok(Self { model, lock: Mutex::new(()), ctx_cap, trace: Mutex::new(None) })
    }

    /// Largest context a request may allocate, in tokens.
    pub fn ctx_cap(&self) -> u32 {
        self.ctx_cap
    }

    /// A fresh `n_ctx`-token context: 8-bit KV cache with flash attention (half the per-token VRAM
    /// of f16, as upstream), falling back to the default f16 cache for an architecture the fast
    /// path cannot serve.
    fn new_context(&self, backend: &LlamaBackend, n_ctx: u32) -> Result<LlamaContext<'_>> {
        self.new_context_batched(backend, n_ctx, N_BATCH.min(n_ctx), N_UBATCH)
    }

    /// [`new_context`](Self::new_context) with explicit batch sizes (a traced evaluation decodes
    /// [`TRACE_UBATCH`] tokens at a time).
    fn new_context_batched(&self, backend: &LlamaBackend, n_ctx: u32, n_batch: u32, n_ubatch: u32) -> Result<LlamaContext<'_>> {
        let n_batch = n_batch.min(n_ctx);
        // swa_full off: sliding-window layers keep only their window (+ one ubatch) of KV instead of
        // the whole context. llama.cpp's full-size default exists for prompt-cache reuse, which a
        // fresh single-sequence context never does — Gemma-4 at 32K: 0.6 GiB of KV instead of 5.6.
        let params = || {
            LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(n_ctx))
                .with_n_batch(n_batch)
                .with_n_ubatch(n_ubatch.min(n_batch))
                .with_swa_full(false)
        };
        let fast = params()
            .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED)
            .with_type_k(KvCacheType::Q8_0)
            .with_type_v(KvCacheType::Q8_0);
        match self.model.new_context(backend, fast) {
            Ok(ctx) => Ok(ctx),
            Err(e) => {
                log::debug!("llm-engine: q8_0 KV + flash attention context failed ({e}) — retrying with an f16 cache");
                let plain = params()
                    .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO)
                    .with_type_k(KvCacheType::F16)
                    .with_type_v(KvCacheType::F16);
                self.model
                    .new_context(backend, plain)
                    .map_err(|e| anyhow!("llm-engine: cannot allocate a {n_ctx}-token context: {e}"))
            }
        }
    }

    /// Raw-prompt completion, greedy decoding (temperature-0 equivalent — keeps OPoI answers
    /// stable), capped at `max_tokens` generated tokens. `prompt` must already be chat-templated
    /// (see `slm::format_prompt_by_name`) — the H4 lineup bypasses llama.cpp's built-in template
    /// matcher, which does not recognize every H4 architecture, and upstream miners prompt with
    /// the exact same strings, keeping OPoI answers aligned across implementations.
    ///
    /// `stop_strings` are scanned over the decoded output and cut the turn (marker excluded)
    /// when a model writes an end marker as plain text instead of emitting an EOG token.
    /// Empty for the H4 lineup — every model's vocab carries its template's control tokens,
    /// so the EOG ids fire natively.
    pub fn generate(&self, prompt: &str, max_tokens: usize, stop_strings: &[&str]) -> Result<String> {
        let _serialize = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let backend = backend()?;

        // str_to_token parses special tokens (the template's control tokens) and AddBos
        // lets the tokenizer add BOS iff the model wants one — llama-server semantics.
        let tokens = self.model.str_to_token(&prompt, AddBos::Always)?;
        if tokens.is_empty() {
            return Err(anyhow!("llm-engine: the prompt tokenized to nothing"));
        }
        let n_ctx = request_ctx(tokens.len(), max_tokens, self.ctx_cap)?;

        // Fresh context per request, sized to this prompt + budget; KV is dropped with the
        // context when this returns.
        let mut ctx = self.new_context(backend, n_ctx)?;

        // Feed the prompt in batches of at most n_batch tokens (one decode call each), logits
        // only for the last prompt token.
        let n_batch = N_BATCH.min(n_ctx) as usize;
        let mut batch = LlamaBatch::new(n_batch, 1);
        let last = tokens.len() - 1;
        for (c, chunk) in tokens.chunks(n_batch).enumerate() {
            batch.clear();
            for (j, tok) in chunk.iter().copied().enumerate() {
                let pos = c * n_batch + j;
                batch.add(tok, pos as i32, &[0], pos == last)?;
            }
            ctx.decode(&mut batch)?;
        }

        // Greedy decode until end-of-generation or the token budget. Output is accumulated
        // as BYTES: byte-level BPE tokens can split UTF-8 sequences mid-character, so
        // per-token string conversion would corrupt multi-byte output.
        let budget = max_tokens.min((n_ctx as usize - tokens.len()).saturating_sub(1));
        let mut sampler = LlamaSampler::greedy();
        let mut out = Vec::<u8>::new();
        let mut n_cur = tokens.len() as i32;
        for _ in 0..budget {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                break;
            }
            // token_to_piece_bytes errors with the needed size when the hint is too small.
            let piece = match self.model.token_to_piece_bytes(token, 32, false, None) {
                Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(need)) => {
                    self.model.token_to_piece_bytes(token, (-need) as usize, false, None)
                }
                x => x,
            }?;
            out.extend_from_slice(&piece);
            if let Some(cut) = find_stop(&out, piece.len(), stop_strings) {
                out.truncate(cut);
                break;
            }
            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;
            ctx.decode(&mut batch)?;
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }
}

impl LlamaEngine {
    /// Configures the canonical trace of the products named by `manifest` (a serialized tier
    /// manifest) for evaluations of up to `max_pos` positions; returns the leaves per position. A
    /// no-op when the GPU already traces this manifest with enough room.
    pub fn trace_configure(&self, manifest: &[u8], max_pos: u32) -> Result<usize> {
        let digest = *blake3::hash(manifest).as_bytes();
        let mut g = self.trace.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(c) = g.as_ref().filter(|c| c.manifest == digest && c.max_pos >= max_pos) {
            return Ok(c.leaves);
        }
        let _serialize = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: the manifest outlives the call; the backend copies what it keeps.
        let rc = unsafe {
            llama_cpp_sys_2::ggml_backend_vk_keryx_trace_configure(TRACE_DEVICE, manifest.as_ptr() as *const c_void, manifest.len(), max_pos)
        };
        if rc != 0 {
            *g = None;
            return Err(anyhow!("trace configuration failed ({rc}) for {max_pos} positions"));
        }
        // SAFETY: plain call into the backend.
        let nl = unsafe { llama_cpp_sys_2::ggml_backend_vk_keryx_trace_leaf_count(TRACE_DEVICE) };
        if nl <= 0 {
            return Err(anyhow!("trace configured without leaves"));
        }
        *g = Some(TraceConfig { manifest: digest, max_pos, leaves: nl as usize });
        Ok(nl as usize)
    }

    /// Leaves per position of the configured trace, if it traces `manifest`.
    fn trace_leaves(&self, manifest: &[u8], n_pos: usize) -> Result<usize> {
        let digest = *blake3::hash(manifest).as_bytes();
        let g = self.trace.lock().unwrap_or_else(|p| p.into_inner());
        match g.as_ref() {
            Some(c) if c.manifest == digest && c.max_pos as usize >= n_pos => Ok(c.leaves),
            _ => Err(anyhow!("the trace is not configured for this manifest and {n_pos} positions")),
        }
    }

    /// A context for a traced evaluation: [`TRACE_UBATCH`]-token batches, the KV cache of
    /// [`new_context`](Self::new_context).
    fn new_trace_context(&self, backend: &LlamaBackend, n_ctx: u32) -> Result<LlamaContext<'_>> {
        self.new_context_batched(backend, n_ctx, TRACE_UBATCH, TRACE_UBATCH)
    }

    /// The token fed as padding: a newline, else the end-of-sequence token.
    fn pad_token(&self) -> LlamaToken {
        self.model.str_to_token("\n", AddBos::Never).ok().and_then(|t| t.first().copied()).unwrap_or_else(|| self.model.token_eos())
    }

    /// Token ids from the miner, checked against the vocabulary.
    fn vocab_tokens(&self, ids: &[i32]) -> Result<Vec<LlamaToken>> {
        let n_vocab = self.model.n_vocab();
        ids.iter()
            .map(|&id| if (0..n_vocab).contains(&id) { Ok(LlamaToken(id)) } else { Err(anyhow!("token {id} is outside the vocabulary")) })
            .collect()
    }

    /// Traced generation under the trace configured for `manifest`: the text, the fed tokens and
    /// the leaves of every position. With `prefix`, `prompt` is the head alone (it must tokenize to
    /// `head_tokens` tokens, else [`PromptHeadMismatch`]), followed by the prefix ids and then
    /// `rest`. The prompt is padded to `min_prompt` positions, and after the answer padding tokens
    /// are decoded until `answer bytes / answer_bytes_per_position` (at most `max_tokens`) positions
    /// were generated (0 = no answer padding). `n_ctx` sizes the context from the fed prompt
    /// length; a replay must use the same.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_traced(
        &self,
        manifest: &[u8],
        prompt: &str,
        prefix: Option<PromptPrefix<'_>>,
        max_tokens: usize,
        min_prompt: u32,
        answer_bytes_per_position: u32,
        n_ctx: impl Fn(usize) -> u32,
    ) -> Result<TracedGeneration> {
        let mut toks = self.model.str_to_token(prompt, AddBos::Always)?;
        if let Some(p) = &prefix {
            if toks.len() != p.head_tokens as usize {
                return Err(PromptHeadMismatch(format!("the prompt head tokenizes to {} tokens, {} expected", toks.len(), p.head_tokens)).into());
            }
            toks.extend(self.vocab_tokens(p.tokens)?);
            toks.extend(self.model.str_to_token(p.rest, AddBos::Never)?);
        }
        let pad = self.pad_token();
        if toks.len() < min_prompt as usize {
            toks.resize(min_prompt as usize, pad);
        }
        let n_prompt = toks.len();
        let n_ctx = n_ctx(n_prompt);
        if n_prompt as u64 + PROMPT_MARGIN as u64 > n_ctx as u64 {
            return Err(PromptTooLong { tokens: n_prompt, ctx_cap: n_ctx }.into());
        }
        let nl = self.trace_leaves(manifest, n_ctx as usize)?;
        let _serialize = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let backend = backend()?;
        let mut ctx = self.new_trace_context(backend, n_ctx)?;
        let mut batch = LlamaBatch::new(TRACE_UBATCH as usize, 1);
        // SAFETY: plain call into the backend.
        if unsafe { llama_cpp_sys_2::ggml_backend_vk_keryx_trace_begin(TRACE_DEVICE) } != 0 {
            return Err(anyhow!("the trace could not be started"));
        }
        let mut run = || -> Result<(Vec<LlamaToken>, Vec<u8>)> {
            let mut fed = toks.clone();
            feed_traced(&mut ctx, &mut batch, &toks, 0, Some(n_prompt - 1))?;
            let mut sampler = LlamaSampler::greedy();
            let mut out = Vec::<u8>::new();
            let budget = max_tokens.min((n_ctx as usize).saturating_sub(n_prompt + 1));
            for _ in 0..budget {
                let token = sampler.sample(&ctx, batch.n_tokens() - 1);
                sampler.accept(token);
                if self.model.is_eog_token(token) {
                    break;
                }
                let piece = match self.model.token_to_piece_bytes(token, 32, false, None) {
                    Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(need)) => {
                        self.model.token_to_piece_bytes(token, (-need) as usize, false, None)
                    }
                    x => x,
                }?;
                out.extend_from_slice(&piece);
                fed.push(token);
                feed_traced(&mut ctx, &mut batch, &[token], fed.len() - 1, Some(fed.len() - 1))?;
            }
            if answer_bytes_per_position > 0 {
                let need = (out.len() / answer_bytes_per_position as usize).min(max_tokens);
                while fed.len() - n_prompt < need && fed.len() + 1 < n_ctx as usize {
                    fed.push(pad);
                    feed_traced(&mut ctx, &mut batch, &[pad], fed.len() - 1, Some(fed.len() - 1))?;
                }
            }
            Ok((fed, out))
        };
        let (fed, out) = match run() {
            Ok(r) => r,
            Err(e) => {
                abandon_trace();
                return Err(e);
            }
        };
        let (ubatch, leaves) = read_trace(nl, fed.len())?;
        Ok(TracedGeneration {
            text: String::from_utf8_lossy(&out).into_owned(),
            tokens: fed.iter().map(|t| t.0).collect(),
            n_prompt: n_prompt as u32,
            n_pos: fed.len() as u32,
            ubatch,
            leaves,
        })
    }

    /// Traces `tokens` fed as a prompt, with nothing generated, in an `n_ctx`-token context.
    pub fn trace_tokens(&self, manifest: &[u8], tokens: &[i32], n_ctx: u32) -> Result<TracedTokens> {
        if tokens.is_empty() || tokens.len() > n_ctx as usize {
            return Err(anyhow!("{} tokens do not fit a {n_ctx}-token traced context", tokens.len()));
        }
        let toks = self.vocab_tokens(tokens)?;
        let nl = self.trace_leaves(manifest, n_ctx as usize)?;
        let _serialize = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let backend = backend()?;
        let mut ctx = self.new_trace_context(backend, n_ctx)?;
        let mut batch = LlamaBatch::new(TRACE_UBATCH as usize, 1);
        // SAFETY: plain call into the backend.
        if unsafe { llama_cpp_sys_2::ggml_backend_vk_keryx_trace_begin(TRACE_DEVICE) } != 0 {
            return Err(anyhow!("the trace could not be started"));
        }
        if let Err(e) = feed_traced(&mut ctx, &mut batch, &toks, 0, Some(toks.len() - 1)) {
            abandon_trace();
            return Err(e);
        }
        let (ubatch, leaves) = read_trace(nl, toks.len())?;
        Ok(TracedTokens { ubatch, leaves })
    }

    /// Replays a traced evaluation (`tokens`, the first `n_prompt` fed as its prompt) with the
    /// batching of the original run, in an `n_ctx`-token context as it ran in, up to the last
    /// requested position, and returns the bytes of each `(position, leaf)` capture, in order.
    /// `slot_bytes` bounds every captured leaf.
    pub fn replay_multi(&self, manifest: &[u8], tokens: &[i32], n_prompt: u32, caps: &[(u32, u32)], slot_bytes: usize, n_ctx: u32) -> Result<Vec<Vec<u8>>> {
        if caps.is_empty() || caps.len() > REPLAY_MAX_CAPTURES || slot_bytes == 0 || slot_bytes > CAPTURE_CAP || slot_bytes % 4 != 0 {
            return Err(anyhow!("invalid replay request"));
        }
        let n_prompt = n_prompt as usize;
        if n_prompt == 0 || tokens.len() < n_prompt || tokens.len() > n_ctx as usize {
            return Err(anyhow!("a replay of {} tokens ({n_prompt} prompt) does not fit a {n_ctx}-token context", tokens.len()));
        }
        let last = caps.iter().map(|c| c.0 as usize).max().unwrap_or(0);
        if last >= tokens.len() {
            return Err(anyhow!("capture past the traced positions"));
        }
        let toks = self.vocab_tokens(tokens)?;
        let nl = self.trace_leaves(manifest, n_ctx as usize)?;
        let _serialize = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let backend = backend()?;
        let mut ctx = self.new_trace_context(backend, n_ctx)?;
        let mut batch = LlamaBatch::new(TRACE_UBATCH as usize, 1);
        let pos: Vec<u32> = caps.iter().map(|c| c.0).collect();
        let leaf: Vec<u32> = caps.iter().map(|c| c.1).collect();
        if leaf.iter().any(|&l| l as usize >= nl) {
            return Err(anyhow!("capture of a leaf past the {nl} leaves of a position"));
        }
        // SAFETY: plain calls into the backend; the arrays outlive them.
        unsafe {
            if llama_cpp_sys_2::ggml_backend_vk_keryx_trace_begin(TRACE_DEVICE) != 0 {
                return Err(anyhow!("the trace could not be started"));
            }
            if llama_cpp_sys_2::ggml_backend_vk_keryx_trace_arm_captures(TRACE_DEVICE, pos.as_ptr(), leaf.as_ptr(), caps.len() as u32, slot_bytes as u32) != 0 {
                abandon_trace();
                return Err(anyhow!("the replay captures could not be armed"));
            }
        }
        let mut run = || -> Result<()> {
            // The prompt is fed in the same TRACE_UBATCH chunks as the original run, through the
            // chunk holding the last capture; generated positions one by one.
            let step = TRACE_UBATCH as usize;
            let prompt_end = if last < n_prompt { n_prompt.min((last / step + 1) * step) } else { n_prompt };
            feed_traced(&mut ctx, &mut batch, &toks[..prompt_end], 0, Some(n_prompt - 1))?;
            for p in n_prompt..=last {
                feed_traced(&mut ctx, &mut batch, &toks[p..=p], p, Some(p))?;
            }
            Ok(())
        };
        if let Err(e) = run() {
            abandon_trace();
            return Err(e);
        }
        let mut out = vec![0u8; caps.len() * slot_bytes];
        let mut lens = vec![0u32; caps.len()];
        // SAFETY: plain calls into the backend; the buffers hold `caps.len()` slots.
        unsafe {
            let ended = llama_cpp_sys_2::ggml_backend_vk_keryx_trace_end(TRACE_DEVICE);
            let rc = llama_cpp_sys_2::ggml_backend_vk_keryx_trace_read_captures(TRACE_DEVICE, out.as_mut_ptr(), out.len(), lens.as_mut_ptr(), caps.len() as u32);
            if ended != 0 || rc != 0 {
                return Err(trace_error(&format!("replay failed ({rc})")));
            }
        }
        Ok(out.chunks(slot_bytes).zip(&lens).map(|(c, &l)| c[..(l as usize).min(slot_bytes)].to_vec()).collect())
    }
}

/// Earliest start offset of any stop string in `out`, or None. A stop marker spans token
/// boundaries, so the scan runs after every appended piece — but only over the tail window
/// where a NEW match can end (`appended` fresh bytes plus `max_stop - 1` bytes of overlap);
/// older bytes were already scanned on the previous token.
fn find_stop(out: &[u8], appended: usize, stops: &[&str]) -> Option<usize> {
    let max_stop = stops.iter().map(|s| s.len()).max()?;
    let from = out.len().saturating_sub(appended + max_stop.saturating_sub(1));
    let hay = &out[from..];
    let mut cut: Option<usize> = None;
    for pat in stops.iter().map(|s| s.as_bytes()) {
        if pat.is_empty() || pat.len() > hay.len() {
            continue;
        }
        if let Some(pos) = hay.windows(pat.len()).position(|w| w == pat) {
            let abs = from + pos;
            cut = Some(cut.map_or(abs, |c| c.min(abs)));
        }
    }
    cut
}

/// Where a weight tensor's bytes live, as seen by the shared PoM walk.
pub enum TensorLoc {
    /// VK-resident in ggml's buffers: walked in place — zero duplication.
    Vk(u64),
    /// Kept host-side by llama.cpp (e.g. `token_embd.weight` under the Vulkan backend):
    /// bytes copied out so the caller can upload them to a small supplement buffer.
    Host(Vec<u8>),
}

/// One weight tensor of the resident model as seen by the shared PoM walk.
pub struct SharedTensor {
    pub name: String,
    pub size: u64,
    pub loc: TensorLoc,
}

impl LlamaEngine {
    /// Table of the resident model's weight tensors (load order — the caller sorts into the
    /// canonical name order). VK-resident tensors carry their GPU address; tensors llama.cpp
    /// keeps host-side (the Vulkan backend does this for `token_embd.weight`) carry their raw
    /// bytes for supplement upload — the walk needs every canonical chunk GPU-reachable.
    pub fn tensor_table(&self) -> Result<Vec<SharedTensor>> {
        let model = self.model.as_raw();
        let n = unsafe { llama_cpp_sys_2::llama_model_keryx_n_tensors(model) };
        let mut out = Vec::with_capacity(n as usize);
        for i in 0..n {
            let t = unsafe { llama_cpp_sys_2::llama_model_keryx_tensor(model, i) };
            if t.is_null() {
                return Err(anyhow!("zero-dup: tensor {i} is null"));
            }
            let name = unsafe { std::ffi::CStr::from_ptr(llama_cpp_sys_2::ggml_get_name(t)) }
                .to_string_lossy()
                .into_owned();
            let size = unsafe { llama_cpp_sys_2::ggml_nbytes(t) } as u64;
            let (mut gpu_addr, mut vk_size) = (0u64, 0u64);
            let loc = if unsafe { llama_cpp_sys_2::ggml_backend_vk_keryx_tensor_addr(t, &mut gpu_addr, &mut vk_size) } {
                TensorLoc::Vk(gpu_addr)
            } else {
                // Host-resident (CPU / pinned buffer): tensor->data is a real host pointer
                // holding the verbatim GGUF block bytes.
                let data = unsafe { (*t).data } as *const u8;
                if data.is_null() {
                    return Err(anyhow!("zero-dup: tensor '{name}' has no VK address and no host data"));
                }
                TensorLoc::Host(unsafe { std::slice::from_raw_parts(data, size as usize) }.to_vec())
            };
            out.push(SharedTensor { name, size, loc });
        }
        Ok(out)
    }

    /// Assemble the complete shared PoM walk over this engine's resident model: VK-resident
    /// tensors are walked IN PLACE (zero duplication); tensors llama.cpp keeps host-side
    /// (`token_embd.weight` under the Vulkan backend) are uploaded once into a small
    /// miner-owned supplement buffer on ggml's device so every canonical chunk is
    /// GPU-reachable. Returns the walk plus the supplement size in bytes (0 = true zero-dup).
    pub fn build_shared_walk(&self) -> Result<(keryx_vulkan::pom_walk::PomWalkShared, u64)> {
        let mut tensors = self.tensor_table()?;
        // Canonical name order (byte-wise, matching pom::WeightIndex / the node's R_T).
        tensors.sort_by(|a, b| a.name.cmp(&b.name));

        // Supplement blob: full chunks of every host-side tensor, in canonical order.
        let mut supplement: Vec<u8> = Vec::new();
        // (needs_supplement_base, addr_or_offset, chunks) per table entry, canonical order.
        let mut entries: Vec<(bool, u64, u64)> = Vec::with_capacity(tensors.len());
        for t in &tensors {
            let chunks = t.size / 32;
            if chunks == 0 {
                continue; // sub-chunk tensors are not part of the canonical layout
            }
            match &t.loc {
                TensorLoc::Vk(addr) => entries.push((false, *addr, chunks)),
                TensorLoc::Host(bytes) => {
                    let off = supplement.len() as u64;
                    supplement.extend_from_slice(&bytes[..(chunks * 32) as usize]);
                    entries.push((true, off, chunks));
                }
            }
        }

        let vk = self.walk_device()?;
        let supl_bytes = supplement.len() as u64;
        let mut supplements = Vec::new();
        let supl_addr = if supplement.is_empty() {
            0
        } else {
            let (buf, addr) = vk
                .create_device_local_address_buffer(&supplement)
                .map_err(|e| anyhow!("zero-dup: supplement upload failed: {e}"))?;
            supplements.push(buf);
            addr
        };

        let table: Vec<(u64, u64)> = entries
            .iter()
            .map(|&(host, a, chunks)| (if host { supl_addr + a } else { a }, chunks))
            .collect();
        let walk = keryx_vulkan::pom_walk::PomWalkShared::new(vk, &table, supplements)
            .map_err(|e| anyhow!("zero-dup: shared walk build failed: {e}"))?;
        Ok((walk, supl_bytes))
    }

    /// Borrow ggml's Vulkan device (its device 0 — pinning makes that the inference GPU) so
    /// the walk kernel can dereference the tensor addresses. Submissions route through ggml's
    /// queue-mutex-guarded hook, so they never race inference on the shared compute queue.
    pub fn walk_device(&self) -> Result<keryx_vulkan::Vk> {
        let mut instance = std::ptr::null_mut();
        let mut physical = std::ptr::null_mut();
        let mut device = std::ptr::null_mut();
        let mut qfi = 0u32;
        if !unsafe {
            llama_cpp_sys_2::ggml_backend_vk_keryx_raw_handles(0, &mut instance, &mut physical, &mut device, &mut qfi)
        } {
            return Err(anyhow!("zero-dup: ggml Vulkan raw handles unavailable (no device or no bufferDeviceAddress)"));
        }
        let submit: keryx_vulkan::ExternalSubmit = Box::new(|submit_info, fence| unsafe {
            llama_cpp_sys_2::ggml_backend_vk_keryx_queue_submit(0, submit_info, fence as usize as *mut std::ffi::c_void);
        });
        unsafe {
            keryx_vulkan::Vk::from_raw_handles(
                instance,
                physical,
                device,
                qfi,
                keryx_vulkan::inference_device_index(),
                submit,
            )
        }
        .map_err(|e| anyhow!("zero-dup: borrowing ggml's device failed: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test logger showing llama.cpp's own output at debug (the dry run reads its buffer plan
    /// there); `RUST_LOG` overrides.
    fn init_test_log() {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Info)
            .filter_module(LLAMA_LOG_TARGET, log::LevelFilter::Debug)
            .parse_default_env()
            .try_init();
    }

    /// End-to-end in-process inference on a real GGUF + GPU. Ignored by default (loads a
    /// multi-GB model); run with:
    ///   KERYX_TEST_GGUF=<path to model.gguf> cargo test --release -- --ignored inproc_chat
    #[test]
    #[ignore = "loads a multi-GB GGUF onto the GPU; set KERYX_TEST_GGUF to run"]
    fn inproc_chat_smoke() {
        init_test_log();
        let Ok(gguf) = std::env::var("KERYX_TEST_GGUF") else {
            eprintln!("SKIP: KERYX_TEST_GGUF not set");
            return;
        };
        let engine = LlamaEngine::launch(&gguf).expect("engine launch");
        // Generic ChatML prompt — good enough for a smoke test on any test GGUF.
        let prompt = "<|im_start|>system\nYou are a terse assistant.<|im_end|>\n\
                      <|im_start|>user\nReply with the single word: pong<|im_end|>\n\
                      <|im_start|>assistant\n";
        let out = engine.generate(prompt, 16, &[]).expect("generate");
        eprintln!("model replied: {out:?}");
        assert!(!out.trim().is_empty(), "empty completion");
        // Greedy decoding is deterministic: the same call must reproduce byte-identically.
        let again = engine.generate(prompt, 16, &[]).expect("generate (repeat)");
        assert_eq!(out, again, "greedy decode not deterministic");
    }

    /// The stop scan must catch a marker regardless of how token pieces split it, cut at the
    /// EARLIEST marker, and never fire on clean output.
    /// A prompt longer than one decode batch is fed in several batches, and a prompt past the
    /// context cap is refused with `PromptTooLong` instead of failing the engine. Ignored by default
    /// (loads a multi-GB model); run with:
    ///   KERYX_TEST_GGUF=<path to a lineup model.gguf> cargo test --release -- --ignored inproc_long_prompt
    #[test]
    #[ignore = "loads a multi-GB GGUF onto the GPU; set KERYX_TEST_GGUF to run"]
    fn inproc_long_prompt() {
        init_test_log();
        let Ok(gguf) = std::env::var("KERYX_TEST_GGUF") else {
            eprintln!("SKIP: KERYX_TEST_GGUF not set");
            return;
        };
        let engine = LlamaEngine::launch(&gguf).expect("engine launch");
        let filler = "The quick brown fox jumps over the lazy dog. ".repeat(600);
        let prompt = format!(
            "<|im_start|>user\n{filler}\nHow many times does the word fox appear above? Answer in one short sentence.<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let n = engine.model.str_to_token(&prompt, AddBos::Always).unwrap().len();
        assert!(n > N_BATCH as usize, "prompt of {n} tokens fits one batch");
        let out = engine.generate(&prompt, 48, &[]).expect("generate over a multi-batch prompt");
        eprintln!("{n}-token prompt → {out:?}");
        assert!(!out.trim().is_empty(), "empty completion");

        let huge = "word ".repeat(engine.ctx_cap as usize + 64);
        let err = engine.generate(&huge, 16, &[]).expect_err("a prompt past the cap must be refused");
        assert!(err.is::<PromptTooLong>(), "unexpected error: {err}");
    }

    /// VRAM dry run: llama.cpp loads only the GGUF metadata (`no_alloc`) and, per context size,
    /// logs what it WOULD allocate — `llama_kv_cache: size = …` (K/V), `llama_memory_recurrent:
    /// size = …` (hybrid models' recurrent state), `llama_context: Vulkan0 compute buffer size = …`
    /// — with the exact context parameters `new_context` uses. Weights are not read, so a sparse
    /// stub holding just the GGUF header works too. Run with:
    ///   KERYX_TEST_GGUF=<model.gguf> [KERYX_TEST_CTX=4096,32768] [KERYX_TEST_KV=f16] \
    ///     cargo test --release --lib -- --ignored --nocapture vram_dry_run
    #[test]
    #[ignore = "needs a Vulkan device and a GGUF; set KERYX_TEST_GGUF to run"]
    fn vram_dry_run() {
        init_test_log();
        let Ok(gguf) = std::env::var("KERYX_TEST_GGUF") else {
            eprintln!("SKIP: KERYX_TEST_GGUF not set");
            return;
        };
        let ctxs: Vec<u32> = std::env::var("KERYX_TEST_CTX")
            .unwrap_or_else(|_| "4096,8192,32768,65536,131072".into())
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let f16 = std::env::var("KERYX_TEST_KV").is_ok_and(|v| v == "f16");
        let _backend = backend().expect("llama backend");
        let path = std::ffi::CString::new(gguf.clone()).unwrap();
        unsafe {
            let mut mp = llama_cpp_sys_2::llama_model_default_params();
            mp.n_gpu_layers = i32::MAX;
            mp.no_alloc = true;
            mp.use_mmap = false; // llama.cpp asserts no_alloc never maps host-pointer buffers
            let model = llama_cpp_sys_2::llama_model_load_from_file(path.as_ptr(), mp);
            assert!(!model.is_null(), "metadata load failed for {gguf}");
            eprintln!("DRYRUN model {gguf}: tensors {:.1} MiB", llama_cpp_sys_2::llama_model_size(model) as f64 / 1048576.0);
            for n_ctx in ctxs {
                let mut cp = llama_cpp_sys_2::llama_context_default_params();
                let n_batch = N_BATCH.min(n_ctx);
                cp.n_ctx = n_ctx;
                cp.n_batch = n_batch;
                cp.n_ubatch = N_UBATCH.min(n_batch);
                cp.swa_full = false;
                if f16 {
                    cp.flash_attn_type = llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO;
                    cp.type_k = llama_cpp_sys_2::GGML_TYPE_F16;
                    cp.type_v = llama_cpp_sys_2::GGML_TYPE_F16;
                } else {
                    cp.flash_attn_type = llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED;
                    cp.type_k = llama_cpp_sys_2::GGML_TYPE_Q8_0;
                    cp.type_v = llama_cpp_sys_2::GGML_TYPE_Q8_0;
                }
                eprintln!("DRYRUN ctx {n_ctx} kv {}", if f16 { "f16" } else { "q8_0" });
                let ctx = llama_cpp_sys_2::llama_init_from_model(model, cp);
                assert!(!ctx.is_null(), "context {n_ctx} could not be planned");
                llama_cpp_sys_2::llama_free(ctx);
            }
            llama_cpp_sys_2::llama_model_free(model);
        }
    }

    /// A request context covers its prompt plus budget in 256-token steps, never past the cap,
    /// and a prompt that leaves no room under the cap is refused up front.
    #[test]
    fn request_ctx_fits_prompt_and_budget_under_the_cap() {
        assert_eq!(request_ctx(100, 64, 32_768).unwrap(), 256);
        assert_eq!(request_ctx(1_000, 1_000, 32_768).unwrap(), 2_048);
        assert_eq!(request_ctx(30_000, 8_000, 32_768).unwrap(), 32_768);
        assert_eq!(request_ctx(32_752, 0, 32_768).unwrap(), 32_768);
        let e = request_ctx(32_753, 16, 32_768).unwrap_err();
        assert_eq!((e.tokens, e.ctx_cap), (32_753, 32_768));
        // `slm` recognizes the refusal through anyhow.
        let wrapped: anyhow::Error = e.into();
        assert!(wrapped.is::<PromptTooLong>());
    }

    #[test]
    fn find_stop_cuts_split_and_earliest_markers() {
        let stops = &["<|im_end|>", "<|im_start|>"];

        // No marker → no cut, whatever the appended size.
        assert_eq!(find_stop(b"a clean answer", 6, stops), None);
        assert_eq!(find_stop(b"", 0, stops), None);
        assert_eq!(find_stop(b"anything", 3, &[]), None);

        // Marker split across two pieces: "...<|im_" seen first, "end|>" appended now. The
        // scan window must reach back across the boundary and cut at the marker start.
        let out = b"The answer is 42.<|im_end|>";
        assert_eq!(find_stop(out, "end|>".len(), stops), Some(17));

        // Marker fully inside the freshly appended piece.
        assert_eq!(find_stop(out, out.len(), stops), Some(17));

        // Two markers in the window → earliest wins (the shorter tail marker starts later).
        let two = b"x<|im_end|>y<|im_start|>";
        assert_eq!(find_stop(two, two.len(), stops), Some(1));

        // Bytes before the window are NOT rescanned: a marker that ended on a previous piece
        // is outside the window when only unrelated bytes were appended since.
        let stale = b"<|im_end|>0123456789abcdef";
        assert_eq!(find_stop(stale, 3, stops), None);
    }
}
