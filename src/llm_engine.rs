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
use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
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
const EXPECTED_LLAMA_WARNINGS: &[&str] = &["the full capacity of the model will not be utilized", "attention rotation force disabled"];

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

/// Sets a variable llama.cpp/ggml read through the C runtime's `getenv`. On Windows `set_var`
/// only updates the OS environment block, and the C runtime keeps its own copy (upstream keryx
/// 25fa0eb), so the variable is put there too.
fn set_llama_env(name: &str, value: &str) {
    std::env::set_var(name, value);
    #[cfg(windows)]
    {
        extern "C" {
            fn _putenv_s(name: *const c_char, value: *const c_char) -> std::os::raw::c_int;
        }
        let (Ok(n), Ok(v)) = (std::ffi::CString::new(name), std::ffi::CString::new(value)) else { return };
        // SAFETY: two NUL-terminated strings that outlive the call.
        unsafe {
            _putenv_s(n.as_ptr(), v.as_ptr());
        }
    }
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

/// GLM-4-0414 rotates 64 of its 128 head dims; a GGUF without `glm4.rope.dimension_count` (the
/// lineup's tier-1 GLM-4-9B is one) makes llama.cpp rotate all of them, which garbles positions
/// and makes the model loop. Upstream's `keryx_needs_glm4_rope_fix` (keryx-miner v0.5.7), same rule.
fn needs_glm4_rope_fix(gguf_path: &str) -> bool {
    let Ok(path) = std::ffi::CString::new(gguf_path) else {
        return false;
    };
    // SAFETY: a metadata-only read (no tensor data, no ggml context), freed before returning; the
    // value string is read while the context is alive.
    unsafe {
        let params = llama_cpp_sys_2::gguf_init_params { no_alloc: true, ctx: std::ptr::null_mut() };
        let g = llama_cpp_sys_2::gguf_init_from_file(path.as_ptr(), params);
        if g.is_null() {
            return false;
        }
        let arch = llama_cpp_sys_2::gguf_find_key(g, c"general.architecture".as_ptr());
        let fix = arch >= 0
            && llama_cpp_sys_2::gguf_get_kv_type(g, arch) == llama_cpp_sys_2::GGUF_TYPE_STRING
            && std::ffi::CStr::from_ptr(llama_cpp_sys_2::gguf_get_val_str(g, arch)).to_bytes() == b"glm4"
            && llama_cpp_sys_2::gguf_find_key(g, c"glm4.rope.dimension_count".as_ptr()) < 0;
        llama_cpp_sys_2::gguf_free(g);
        fix
    }
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

/// One layer of the sequence layout: (layer, kind, slot, heads, head_dim); kind 1 = grouped-query
/// attention, 2 = latent attention, 3/4 = gated/key-wise delta-rule recurrence.
pub type SeqLayer = [u32; 5];

/// A captured span of one sequence leaf: the whole chunks covering it and the chaining value of
/// every chunk of the leaf.
#[derive(Clone, Debug, Default)]
pub struct SeqCapture {
    pub bytes: Vec<u8>,
    pub cvs: Vec<[u8; 32]>,
}

/// The manifest (BLAKE3 of its bytes) and position capacity the GPU trace is configured with.
struct TraceConfig {
    manifest: [u8; 32],
    max_pos: u32,
    leaves: usize,
    /// Sequence leaves committed beside the position leaves (attention rule).
    seq: Option<SeqSpec>,
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
    /// Sequence leaf hashes, under the attention rule.
    pub seq: Option<Vec<[u8; 32]>>,
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
/// batch to the trace before its decode; only position `logits_at` produces logits. Each batch is
/// committed as the node expects the run of a prompt of `n_prompt` positions to be batched
/// (`verify::expected_ubatch`: 512-token prompt batches, CUDA's MMQ layouts past 8 tokens, one token
/// per generated position). A batch never straddles a 512-position boundary: batches start on
/// multiples of [`TRACE_UBATCH`], which divides 512, and the prompt is fed apart from the answer.
///
/// With `rec`, the recurrent records of each batch are taken right after its decode.
fn feed_traced(
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch,
    tokens: &[LlamaToken],
    start: usize,
    logits_at: Option<usize>,
    n_prompt: usize,
    mut rec: Option<&mut RecurSink>,
) -> Result<()> {
    let n_prompt = n_prompt.min(u32::MAX as usize) as u32;
    for (c, chunk) in tokens.chunks(TRACE_UBATCH as usize).enumerate() {
        let base = start + c * TRACE_UBATCH as usize;
        batch.clear();
        for (j, &tok) in chunk.iter().enumerate() {
            batch.add(tok, (base + j) as i32, &[0], Some(base + j) == logits_at)?;
        }
        let declared = keryx_verifiable_inference::verify::expected_ubatch(base as u64, n_prompt);
        let last = keryx_verifiable_inference::verify::expected_ubatch((base + chunk.len() - 1) as u64, n_prompt);
        if last != declared || (declared as usize) < chunk.len() {
            return Err(anyhow!("a traced batch at {base} straddles a committed batch"));
        }
        // SAFETY: plain call into the backend.
        if unsafe { llama_cpp_sys_2::ggml_backend_vk_keryx_trace_ubatch_as(TRACE_DEVICE, base as u32, chunk.len() as u32, declared) } != 0 {
            return Err(trace_error("trace batch refused"));
        }
        ctx.decode(batch)?;
        if let Some(r) = rec.as_deref_mut() {
            r.after_batch(chunk.len())?;
        }
    }
    Ok(())
}

// -- Sequence leaves (H15 attention rule) ---------------------------------------------------------
//
// Under the attention rule a response also commits, per attention layer and key/value head, its
// q8_0 key and value cache rows of every position end to end, and the log softmax mass of every
// query head at every position. The cache rows come out of the context's own KV cache (the
// sequence state of llama.cpp: the rows exactly as attention read them); the masses come from the
// backend, which recomputes them after every flash-attention node of a traced evaluation
// (`keryx_lse.comp`).

/// One attention layer of the sequence layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqAttn {
    pub layer: u32,
    pub slot: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
}

/// One recurrent (gated delta-rule) layer of the sequence layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqRecur {
    pub layer: u32,
    pub slot: u32,
    /// Sequence layer kind: 3 = one decay per head, 4 = one per key dimension.
    pub kind: u32,
    pub n_v_heads: u32,
    pub head_dim: u32,
}

/// Sequence layer kind of a delta-rule layer with one decay per head (`RECUR_GDA`).
pub const SEQ_KIND_GDA: u32 = 3;

/// The sequence leaves a manifest under the attention rule commits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeqSpec {
    pub n_attn: u32,
    pub n_recur: u32,
    /// Heads per layer in the leaf layout (`seq_heads`).
    pub heads: u32,
    pub attn: Vec<SeqAttn>,
    pub recur: Vec<SeqRecur>,
    /// Positions between recurrent state checkpoints.
    pub interval: u32,
}

impl SeqSpec {
    pub fn n_leaves(&self) -> usize {
        let h = self.heads as usize;
        2 * h * (self.n_attn + self.n_recur) as usize + h * self.n_attn as usize
    }

    /// Layers and query heads of the mass capture.
    fn lse_shape(&self) -> (u32, u32) {
        (self.attn.iter().map(|a| a.layer + 1).max().unwrap_or(0), self.attn.iter().map(|a| a.n_head).max().unwrap_or(0))
    }

    /// Bytes of one record of a recurrent head (`seq::record_len` of a one-decay layer).
    fn record_len(&self) -> usize {
        self.recur.first().map_or(0, |r| 8 * r.head_dim as usize + 8)
    }

    /// The backend's rows of the recurrent layers: (layer, kind, slot, heads, head_dim).
    fn recur_rows(&self) -> Vec<SeqLayer> {
        self.recur.iter().map(|r| [r.layer, r.kind, r.slot, r.n_v_heads, r.head_dim]).collect()
    }

    /// Whether this engine can commit the layout: grouped-query attention layers, and delta-rule
    /// layers with one decay per head, alike in heads and dimension (the backend's record
    /// capture), checkpointed on traced batch ends.
    pub fn check(&self) -> Result<()> {
        if self.attn.len() != self.n_attn as usize || self.recur.len() != self.n_recur as usize || self.heads == 0 {
            return Err(anyhow!("inconsistent sequence layout"));
        }
        for (i, a) in self.attn.iter().enumerate() {
            if a.slot as usize != i || a.n_head_kv == 0 || a.n_head_kv > self.heads || a.n_head % a.n_head_kv != 0 || a.head_dim % 32 != 0 {
                return Err(anyhow!("unsupported attention layer {}", a.layer));
            }
        }
        if let Some(first) = self.recur.first() {
            if self.interval == 0 || self.interval % TRACE_UBATCH != 0 {
                return Err(anyhow!("recurrent checkpoints every {} positions fall inside traced batches", self.interval));
            }
            for (i, r) in self.recur.iter().enumerate() {
                if r.slot as usize != i || r.kind != SEQ_KIND_GDA || r.n_v_heads != first.n_v_heads || r.head_dim != first.head_dim {
                    return Err(anyhow!("unsupported recurrent layer {}", r.layer));
                }
            }
            if first.n_v_heads == 0 || first.n_v_heads > self.heads || first.head_dim % 32 != 0 || first.head_dim > 128 {
                return Err(anyhow!("unsupported recurrent heads ({} of {})", first.n_v_heads, first.head_dim));
            }
        }
        Ok(())
    }
}

/// The recurrent sequence leaves of a traced evaluation, built batch by batch from the records
/// the backend takes after every gated-delta-net node (`keryx_rec.comp`): each head's records of
/// every position end to end, and its states after every whole checkpoint interval. Every leaf is
/// hashed as it grows; the leaves named at creation are also kept whole (a replay's spans).
struct RecurSink {
    n_recur: usize,
    heads: usize,
    seq_heads: usize,
    n_attn: usize,
    rec_len: usize,
    state_len: usize,
    interval: usize,
    rec: Vec<blake3::Hasher>,
    ckpt: Vec<blake3::Hasher>,
    keep: Vec<(u32, Vec<u8>)>,
    buf_rec: Vec<u8>,
    buf_ckpt: Vec<u8>,
    pos: usize,
    n_ckpt: usize,
}

impl RecurSink {
    /// A sink for `spec`'s recurrent leaves (`None` without any), keeping the leaves `keep` whole.
    fn new(spec: &SeqSpec, keep: &[u32]) -> Option<Self> {
        let first = *spec.recur.first()?;
        let (n_recur, heads, hd) = (spec.recur.len(), first.n_v_heads as usize, first.head_dim as usize);
        let rec_len = spec.record_len();
        let state_len = hd * hd * 4;
        let mut sink = Self {
            n_recur,
            heads,
            seq_heads: spec.heads as usize,
            n_attn: spec.n_attn as usize,
            rec_len,
            state_len,
            interval: spec.interval as usize,
            rec: vec![blake3::Hasher::new(); n_recur * heads],
            ckpt: vec![blake3::Hasher::new(); n_recur * heads],
            keep: Vec::new(),
            buf_rec: vec![0u8; n_recur * heads * TRACE_UBATCH as usize * rec_len],
            buf_ckpt: vec![0u8; n_recur * heads * state_len],
            pos: 0,
            n_ckpt: 0,
        };
        let mut leaves: Vec<u32> = keep.iter().copied().filter(|&l| sink.slot_head(l).is_some()).collect();
        leaves.sort_unstable();
        leaves.dedup();
        sink.keep = leaves.into_iter().map(|l| (l, Vec::new())).collect();
        Some(sink)
    }

    /// The (record?, slot * heads + head) of sequence leaf `leaf`, if it is a written recurrent leaf.
    fn slot_head(&self, leaf: u32) -> Option<(bool, usize)> {
        let base = 2 * self.seq_heads * self.n_attn;
        let i = (leaf as usize).checked_sub(base)?;
        let (sh, ck) = (i / 2, i % 2 == 1);
        let (slot, h) = (sh / self.seq_heads, sh % self.seq_heads);
        (slot < self.n_recur && h < self.heads).then_some((!ck, slot * self.heads + h))
    }

    /// Takes the records (and checkpoint) of the batch of `n_tokens` just decoded.
    fn after_batch(&mut self, n_tokens: usize) -> Result<()> {
        let (mut n, mut has_ckpt) = (0u32, 0u32);
        // SAFETY: the buffers hold a whole batch of records and one state of every head.
        let rc = unsafe {
            llama_cpp_sys_2::ggml_backend_vk_keryx_recur_read(
                TRACE_DEVICE,
                self.buf_rec.as_mut_ptr(),
                self.buf_rec.len(),
                self.buf_ckpt.as_mut_ptr(),
                self.buf_ckpt.len(),
                &mut n,
                &mut has_ckpt,
            )
        };
        if rc != 0 {
            return Err(trace_error(&format!("recurrent records unavailable ({rc})")));
        }
        if n as usize != n_tokens {
            return Err(anyhow!("{n} recurrent records for a batch of {n_tokens}"));
        }
        self.pos += n_tokens;
        let due = self.pos % self.interval == 0;
        if (has_ckpt != 0) != due {
            return Err(anyhow!("recurrent checkpoint {} at position {}", if due { "missing" } else { "unexpected" }, self.pos));
        }
        let per = n_tokens * self.rec_len;
        for (i, hasher) in self.rec.iter_mut().enumerate() {
            hasher.update(&self.buf_rec[i * per..(i + 1) * per]);
        }
        if due {
            self.n_ckpt += 1;
            for (i, hasher) in self.ckpt.iter_mut().enumerate() {
                hasher.update(&self.buf_ckpt[i * self.state_len..(i + 1) * self.state_len]);
            }
        }
        for k in 0..self.keep.len() {
            let (is_rec, i) = self.slot_head(self.keep[k].0).expect("kept leaves are recurrent");
            let bytes = if is_rec {
                &self.buf_rec[i * per..(i + 1) * per]
            } else if due {
                &self.buf_ckpt[i * self.state_len..(i + 1) * self.state_len]
            } else {
                continue;
            };
            self.keep[k].1.extend_from_slice(bytes);
        }
        Ok(())
    }

    /// Checks the evaluation covered `n_pos` positions.
    fn check_end(&self, n_pos: usize) -> Result<()> {
        if self.pos != n_pos || self.n_ckpt != n_pos / self.interval {
            return Err(anyhow!("recurrent records of {} positions ({} checkpoints) for {n_pos}", self.pos, self.n_ckpt));
        }
        Ok(())
    }

    /// The hashes of the recurrent leaves into `leaves` (every sequence leaf, in order).
    fn hashes_into(&self, leaves: &mut [[u8; 32]]) {
        let base = 2 * self.seq_heads * self.n_attn;
        for slot in 0..self.n_recur {
            for h in 0..self.heads {
                let (i, leaf) = (slot * self.heads + h, base + 2 * (slot * self.seq_heads + h));
                leaves[leaf] = *self.rec[i].finalize().as_bytes();
                leaves[leaf + 1] = *self.ckpt[i].finalize().as_bytes();
            }
        }
    }

    /// The whole bytes of kept leaf `leaf`.
    fn kept(&self, leaf: u32) -> Option<&[u8]> {
        self.keep.iter().find(|(l, _)| *l == leaf).map(|(_, b)| b.as_slice())
    }
}

/// The q8_0 key and value cache rows of one layer, every position end to end.
struct LayerCache {
    k: Vec<u8>,
    v: Vec<u8>,
    k_row: usize,
    v_row: usize,
}

/// Parses the llama.cpp state of sequence 0 (`llama_kv_cache::state_write`): one stream, one range
/// of cells holding positions `0..n_pos` in order, unrotated q8_0 rows, value rows untransposed.
/// The cache holds the attention layers only, in layer order. With `hybrid` the recurrent memory's
/// state follows (`llama_memory_hybrid::state_write`) and is skipped. A cell of an M-RoPE model
/// carries its 2D position and token (`llama_kv_cell_ext`, 12 bytes) after its sequence count; the
/// state does not say so, and only one of the two readings holds positions `0..n_pos` in order.
fn parse_kv_state(b: &[u8], n_pos: usize, hybrid: bool) -> Result<Vec<LayerCache>> {
    parse_kv_state_ext(b, n_pos, hybrid, 0).or_else(|e| parse_kv_state_ext(b, n_pos, hybrid, 12).map_err(|_| e))
}

fn parse_kv_state_ext(b: &[u8], n_pos: usize, hybrid: bool, cell_ext: usize) -> Result<Vec<LayerCache>> {
    let mut o = 0usize;
    let mut take = |n: usize| -> Result<&[u8]> {
        let s = b.get(o..o + n).ok_or_else(|| anyhow!("the cache state ends early"))?;
        o += n;
        Ok(s)
    };
    let u32_at = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap());
    // `llama_context::state_seq_get_data`: its magic and the sequence id, then the cache's state.
    let (magic, seq) = (u32_at(take(4)?), u32_at(take(4)?));
    if magic != 0xaf14_3cd8 || seq != 0 {
        return Err(anyhow!("not a state of sequence 0 (magic {magic:#x}, sequence {seq})"));
    }
    if u32_at(take(4)?) != 1 {
        return Err(anyhow!("the cache has several streams"));
    }
    let cells = u32_at(take(4)?) as usize;
    if cells != n_pos {
        return Err(anyhow!("the cache holds {cells} cells for {n_pos} positions"));
    }
    for i in 0..cells {
        let pos = u32_at(take(4)?);
        let n_seq = u32_at(take(4)?);
        take(cell_ext)?;
        let seq = if n_seq == 1 { u32_at(take(4)?) } else { u32::MAX };
        if pos as usize != i || seq != 0 {
            return Err(anyhow!("cache cell {i} holds position {pos} of {n_seq} sequences"));
        }
    }
    let v_trans = u32_at(take(4)?);
    let n_layer = u32_at(take(4)?) as usize;
    let (rot_k, rot_v) = (u32_at(take(4)?), u32_at(take(4)?));
    if v_trans != 0 || rot_k != 0 || rot_v != 0 {
        return Err(anyhow!("the cache is transposed or rotated (v_trans {v_trans}, rotation {rot_k}/{rot_v})"));
    }
    let mut layers: Vec<LayerCache> = Vec::with_capacity(n_layer);
    for which in 0..2 {
        for l in 0..n_layer {
            let ty = u32_at(take(4)?);
            let row = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
            if ty != llama_cpp_sys_2::GGML_TYPE_Q8_0 as u32 {
                return Err(anyhow!("cache layer {l} is not q8_0 (type {ty})"));
            }
            let rows = take(row * cells)?.to_vec();
            if which == 0 {
                layers.push(LayerCache { k: rows, v: Vec::new(), k_row: row, v_row: 0 });
            } else {
                layers[l].v = rows;
                layers[l].v_row = row;
            }
        }
    }
    if o != b.len() && !hybrid {
        return Err(anyhow!("{} bytes past the cache state", b.len() - o));
    }
    Ok(layers)
}

/// The bytes of every sequence leaf of `spec`, in leaf order, from the cache rows and the masses
/// (`[layers][n_pos][heads]`, as the backend reads them back).
fn seq_leaf_bytes(spec: &SeqSpec, cache: &[LayerCache], lse: &[f32], n_pos: usize) -> Result<Vec<Vec<u8>>> {
    let (lse_layers, lse_heads) = spec.lse_shape();
    let (lse_layers, lse_heads) = (lse_layers as usize, lse_heads as usize);
    if lse.len() != lse_layers * n_pos * lse_heads {
        return Err(anyhow!("{} masses for {lse_layers} layers of {n_pos} positions", lse.len()));
    }
    let heads = spec.heads as usize;
    let mut out = vec![Vec::new(); spec.n_leaves()];
    // The cache holds the attention layers in layer order: the rank of a layer among them is its slot.
    let by_slot = cache.len() == spec.attn.len();
    for a in &spec.attn {
        let at = if by_slot { a.slot } else { a.layer } as usize;
        let lc = cache.get(at).ok_or_else(|| anyhow!("no cache for layer {}", a.layer))?;
        let hb = a.head_dim as usize / 32 * 34;
        let nkv = a.n_head_kv as usize;
        if lc.k_row != nkv * hb || lc.v_row != nkv * hb {
            return Err(anyhow!("cache rows of layer {} are {}/{} bytes for {nkv} heads of {hb}", a.layer, lc.k_row, lc.v_row));
        }
        let g = (a.n_head / a.n_head_kv) as usize;
        let slot = a.slot as usize;
        for h in 0..nkv {
            let kl = 2 * (slot * heads + h);
            let mut k = Vec::with_capacity(n_pos * hb);
            let mut v = Vec::with_capacity(n_pos * hb);
            let mut m = Vec::with_capacity(n_pos * g * 4);
            for t in 0..n_pos {
                k.extend_from_slice(&lc.k[t * lc.k_row + h * hb..t * lc.k_row + (h + 1) * hb]);
                v.extend_from_slice(&lc.v[t * lc.v_row + h * hb..t * lc.v_row + (h + 1) * hb]);
                for qi in 0..g {
                    let x = lse[(a.layer as usize * n_pos + t) * lse_heads + h * g + qi];
                    if !x.is_finite() {
                        return Err(anyhow!("no softmax mass for layer {} position {t} head {}", a.layer, h * g + qi));
                    }
                    m.extend_from_slice(&x.to_le_bytes());
                }
            }
            out[kl] = k;
            out[kl + 1] = v;
            out[2 * heads * (spec.n_attn + spec.n_recur) as usize + slot * heads + h] = m;
        }
    }
    Ok(out)
}

/// Every sequence leaf of the traced evaluation that just ran in `ctx` (its trace ended).
fn read_seq(ctx: &LlamaContext<'_>, spec: &SeqSpec, n_pos: usize) -> Result<Vec<Vec<u8>>> {
    use llama_cpp_2::context::session::LlamaStateSeqFlags;
    let size = ctx.state_seq_get_size_ext(0, LlamaStateSeqFlags::empty());
    let mut state = vec![0u8; size];
    // SAFETY: the buffer holds the size llama.cpp reported for this sequence's state.
    let n = unsafe { ctx.state_seq_get_data_ext(state.as_mut_ptr(), 0, LlamaStateSeqFlags::empty()) };
    if n != size {
        return Err(anyhow!("cache state of {n} bytes for {size}"));
    }
    let cache = parse_kv_state(&state, n_pos, spec.n_recur > 0)?;
    let (layers, heads) = spec.lse_shape();
    let mut lse = vec![0f32; layers as usize * n_pos * heads as usize];
    // SAFETY: the buffer holds `layers * n_pos * heads` floats, as the backend writes them.
    let rc = unsafe { llama_cpp_sys_2::ggml_backend_vk_keryx_lse_read(TRACE_DEVICE, lse.as_mut_ptr(), lse.len(), n_pos as u32) };
    if rc != 0 {
        return Err(trace_error(&format!("softmax masses unavailable ({rc})")));
    }
    seq_leaf_bytes(spec, &cache, &lse, n_pos)
}

/// The capture of `[lo, hi)` of a sequence leaf: those bytes and the chaining value of every chunk.
pub fn seq_capture(leaf: &[u8], lo: u64, hi: u64) -> Result<SeqCapture> {
    use keryx_verifiable_inference::merkle::{b3_chunk_cv, B3_CHUNK};
    let bytes = leaf.get(lo as usize..hi as usize).ok_or_else(|| anyhow!("span {lo}..{hi} past a leaf of {} bytes", leaf.len()))?.to_vec();
    let cvs = if leaf.len() <= B3_CHUNK { vec![*blake3::hash(leaf).as_bytes()] } else { leaf.chunks(B3_CHUNK).enumerate().map(|(i, c)| b3_chunk_cv(c, i as u64)).collect() };
    Ok(SeqCapture { bytes, cvs })
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
            set_llama_env("GGML_VK_VISIBLE_DEVICES", &infer.to_string());
        }

        // The attention rule commits the q8_0 cache rows as written and the node reads them as the
        // plain quantized keys and values: llama.cpp's rotation of a quantized cache (0.6.0) stays
        // off, as upstream keryx turns it off. Set before any context exists; a user value wins.
        if std::env::var_os("LLAMA_ATTN_ROT_DISABLE").is_none() {
            set_llama_env("LLAMA_ATTN_ROT_DISABLE", "1");
        }
        let backend = backend()?;
        info!("llm-engine: loading {} (Vulkan, all layers on GPU, in-process)", gguf_path);
        // u32::MAX = offload every layer (the crate's "all" sentinel, like -ngl 999). MTP
        // (next-token prediction) layers are loaded too: llama.cpp 0.6.0 skips them by default,
        // but the possession index covers every tensor of the GGUF, so the zero-dup walk over
        // the engine's resident weights (Qwen3.8-27B, tier 3) needs them resident.
        let mut params = std::pin::pin!(LlamaModelParams::default().with_n_gpu_layers(u32::MAX).with_load_mtp(true));
        if needs_glm4_rope_fix(gguf_path) {
            info!("llm-engine: the GGUF has no glm4.rope.dimension_count — rotating 64 of the 128 head dims, as GLM-4-0414 does");
            params.as_mut().append_kv_override(c"glm4.rope.dimension_count", ParamOverrideValue::Int(64));
        }
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

        let tokens = self.tokenize(prompt, true);
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

        // Decode until end-of-generation or the token budget. Output is accumulated as BYTES:
        // byte-level BPE tokens can split UTF-8 sequences mid-character, so per-token string
        // conversion would corrupt multi-byte output. `sample` also accepts the token.
        let budget = max_tokens.min((n_ctx as usize - tokens.len()).saturating_sub(1));
        let mut sampler = self.answer_sampler();
        let mut out = Vec::<u8>::new();
        let mut n_cur = tokens.len() as i32;
        for _ in 0..budget {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            if self.model.vocab().is_eog(token) {
                break;
            }
            let piece = self.model.vocab().token_to_piece(token, false, None);
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
    pub fn trace_configure(&self, manifest: &[u8], max_pos: u32, seq: Option<&SeqSpec>) -> Result<usize> {
        let digest = *blake3::hash(manifest).as_bytes();
        let mut g = self.trace.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(c) = g.as_ref().filter(|c| c.manifest == digest && c.max_pos >= max_pos && c.seq.as_ref() == seq) {
            return Ok(c.leaves);
        }
        if let Some(spec) = seq {
            spec.check()?;
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
        let (layers, heads) = seq.map_or((0, 0), |s| s.lse_shape());
        // SAFETY: plain call into the backend.
        let rc = unsafe { llama_cpp_sys_2::ggml_backend_vk_keryx_lse_configure(TRACE_DEVICE, layers, heads) };
        if rc != 0 {
            *g = None;
            return Err(anyhow!("softmax mass capture unavailable ({rc}) for {layers} layers of {heads} heads"));
        }
        let rows = seq.map_or(Vec::new(), |s| s.recur_rows());
        let interval = seq.map_or(0, |s| s.interval);
        // SAFETY: `rows` outlives the call; the backend copies what it keeps.
        let rc = unsafe { llama_cpp_sys_2::ggml_backend_vk_keryx_recur_configure(TRACE_DEVICE, rows.as_ptr() as *const u32, rows.len() as u32, interval) };
        if rc != 0 {
            *g = None;
            return Err(anyhow!("recurrent record capture unavailable ({rc}) for {} layers", rows.len()));
        }
        *g = Some(TraceConfig { manifest: digest, max_pos, leaves: nl as usize, seq: seq.cloned() });
        Ok(nl as usize)
    }

    /// Leaves per position of the configured trace, if it traces `manifest`, and its sequence layout.
    fn trace_leaves(&self, manifest: &[u8], n_pos: usize) -> Result<(usize, Option<SeqSpec>)> {
        let digest = *blake3::hash(manifest).as_bytes();
        let g = self.trace.lock().unwrap_or_else(|p| p.into_inner());
        match g.as_ref() {
            Some(c) if c.manifest == digest && c.max_pos as usize >= n_pos => Ok((c.leaves, c.seq.clone())),
            _ => Err(anyhow!("the trace is not configured for this manifest and {n_pos} positions")),
        }
    }

    /// A context for a traced evaluation: [`TRACE_UBATCH`]-token batches, the KV cache of
    /// [`new_context`](Self::new_context). With `q8_cache` (sequence leaves commit the cache rows as
    /// q8_0) the f16 fallback is refused.
    fn new_trace_context(&self, backend: &LlamaBackend, n_ctx: u32, q8_cache: bool) -> Result<LlamaContext<'_>> {
        if !q8_cache {
            return self.new_context_batched(backend, n_ctx, TRACE_UBATCH, TRACE_UBATCH);
        }
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(TRACE_UBATCH)
            .with_n_ubatch(TRACE_UBATCH)
            .with_swa_full(false)
            .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED)
            .with_type_k(KvCacheType::Q8_0)
            .with_type_v(KvCacheType::Q8_0);
        self.model.new_context(backend, params).map_err(|e| anyhow!("llm-engine: no {n_ctx}-token q8_0 flash-attention context for a traced evaluation: {e}"))
    }

    /// `text` as tokens: special tokens parsed (the templates' control tokens), and with `bos` a BOS
    /// added iff the model wants one — llama-server semantics.
    fn tokenize(&self, text: &str, bos: bool) -> Vec<LlamaToken> {
        self.model.vocab().tokenize(text.as_bytes(), bos, true)
    }

    /// The sampler of an answer, upstream's (`keryx_llama.cpp`, since keryx-miner v0.4.5): DRY,
    /// a flat repeat penalty, top-p 0.9, temperature 0.7, seed 42. Greedy decoding loops on these
    /// models ("achieving the goal of achieving the goal of …" on GLM-4-9B), and a flat penalty
    /// alone barely charges a repeated sentence; DRY penalises continuing an already-seen
    /// sequence. Built per request, which restarts the seed as upstream's per-request reset
    /// does. The text is not consensus-relevant: a trace commits whatever tokens were sampled.
    fn answer_sampler(&self) -> LlamaSampler {
        // Upstream scans the whole context (-1); llama.cpp 0.6.0 clamps a negative DRY window to
        // 0, which turns DRY off, so the window is the training context instead.
        let window = i32::try_from(self.model.n_ctx_train()).unwrap_or(i32::MAX);
        LlamaSampler::chain_simple([
            // Breakers keep normal structure (newlines, list markers, quotes) from counting as repetition.
            LlamaSampler::dry(&self.model, 0.8, 1.75, 2, window, ["\n", ":", "\"", "*"]),
            LlamaSampler::penalties(self.model.n_vocab(), 256, 1.10, 0.0, 0.0),
            LlamaSampler::top_p(0.9, 1),
            LlamaSampler::temp(0.7),
            LlamaSampler::dist(42),
        ])
    }

    /// The token fed as padding: a newline, else the end-of-sequence token.
    fn pad_token(&self) -> LlamaToken {
        self.tokenize("\n", false).first().copied().unwrap_or_else(|| self.model.vocab().eos())
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
        let mut toks = self.tokenize(prompt, true);
        if let Some(p) = &prefix {
            if toks.len() != p.head_tokens as usize {
                return Err(PromptHeadMismatch(format!("the prompt head tokenizes to {} tokens, {} expected", toks.len(), p.head_tokens)).into());
            }
            toks.extend(self.vocab_tokens(p.tokens)?);
            toks.extend(self.tokenize(p.rest, false));
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
        let (nl, seq) = self.trace_leaves(manifest, n_ctx as usize)?;
        let _serialize = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let backend = backend()?;
        let mut ctx = self.new_trace_context(backend, n_ctx, seq.is_some())?;
        let mut batch = LlamaBatch::new(TRACE_UBATCH as usize, 1);
        // SAFETY: plain call into the backend.
        if unsafe { llama_cpp_sys_2::ggml_backend_vk_keryx_trace_begin(TRACE_DEVICE) } != 0 {
            return Err(anyhow!("the trace could not be started"));
        }
        let mut rec = seq.as_ref().and_then(|s| RecurSink::new(s, &[]));
        let mut run = || -> Result<(Vec<LlamaToken>, Vec<u8>)> {
            let mut fed = toks.clone();
            feed_traced(&mut ctx, &mut batch, &toks, 0, Some(n_prompt - 1), n_prompt, rec.as_mut())?;
            // The traced answer is sampled like any other (the trace commits the sampled tokens).
            let mut sampler = self.answer_sampler();
            let mut out = Vec::<u8>::new();
            let budget = max_tokens.min((n_ctx as usize).saturating_sub(n_prompt + 1));
            for _ in 0..budget {
                let token = sampler.sample(&ctx, batch.n_tokens() - 1);
                if self.model.vocab().is_eog(token) {
                    break;
                }
                let piece = self.model.vocab().token_to_piece(token, false, None);
                out.extend_from_slice(&piece);
                fed.push(token);
                feed_traced(&mut ctx, &mut batch, &[token], fed.len() - 1, Some(fed.len() - 1), n_prompt, rec.as_mut())?;
            }
            if answer_bytes_per_position > 0 {
                let need = (out.len() / answer_bytes_per_position as usize).min(max_tokens);
                while fed.len() - n_prompt < need && fed.len() + 1 < n_ctx as usize {
                    fed.push(pad);
                    feed_traced(&mut ctx, &mut batch, &[pad], fed.len() - 1, Some(fed.len() - 1), n_prompt, rec.as_mut())?;
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
        let seq = match &seq {
            Some(spec) => {
                let mut hashes: Vec<[u8; 32]> = read_seq(&ctx, spec, fed.len())?.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
                if let Some(r) = &rec {
                    r.check_end(fed.len())?;
                    r.hashes_into(&mut hashes);
                }
                Some(hashes)
            }
            None => None,
        };
        Ok(TracedGeneration {
            text: String::from_utf8_lossy(&out).into_owned(),
            tokens: fed.iter().map(|t| t.0).collect(),
            n_prompt: n_prompt as u32,
            n_pos: fed.len() as u32,
            ubatch,
            leaves,
            seq,
        })
    }

    /// Traces `tokens` fed as a prompt, with nothing generated, in an `n_ctx`-token context.
    pub fn trace_tokens(&self, manifest: &[u8], tokens: &[i32], n_ctx: u32) -> Result<TracedTokens> {
        if tokens.is_empty() || tokens.len() > n_ctx as usize {
            return Err(anyhow!("{} tokens do not fit a {n_ctx}-token traced context", tokens.len()));
        }
        let toks = self.vocab_tokens(tokens)?;
        let (nl, seq) = self.trace_leaves(manifest, n_ctx as usize)?;
        let _serialize = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let backend = backend()?;
        let mut ctx = self.new_trace_context(backend, n_ctx, seq.is_some())?;
        let mut batch = LlamaBatch::new(TRACE_UBATCH as usize, 1);
        // SAFETY: plain call into the backend.
        if unsafe { llama_cpp_sys_2::ggml_backend_vk_keryx_trace_begin(TRACE_DEVICE) } != 0 {
            return Err(anyhow!("the trace could not be started"));
        }
        if let Err(e) = feed_traced(&mut ctx, &mut batch, &toks, 0, Some(toks.len() - 1), toks.len(), None) {
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
        let (nl, seq) = self.trace_leaves(manifest, n_ctx as usize)?;
        let _serialize = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let backend = backend()?;
        let mut ctx = self.new_trace_context(backend, n_ctx, seq.is_some())?;
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
            feed_traced(&mut ctx, &mut batch, &toks[..prompt_end], 0, Some(n_prompt - 1), n_prompt, None)?;
            for p in n_prompt..=last {
                feed_traced(&mut ctx, &mut batch, &toks[p..=p], p, Some(p), n_prompt, None)?;
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

    /// Replays a traced evaluation whole, as it ran (see [`replay_multi`](Self::replay_multi)), and
    /// returns the bytes of the sequence leaves `leaves` (every leaf when empty and the layout has
    /// no recurrent layer), in that order.
    pub fn replay_seq(&self, manifest: &[u8], tokens: &[i32], n_prompt: u32, n_ctx: u32, leaves: &[u32]) -> Result<Vec<Vec<u8>>> {
        let n_prompt = n_prompt as usize;
        if n_prompt == 0 || tokens.len() < n_prompt || tokens.len() > n_ctx as usize {
            return Err(anyhow!("a replay of {} tokens ({n_prompt} prompt) does not fit a {n_ctx}-token context", tokens.len()));
        }
        let toks = self.vocab_tokens(tokens)?;
        let (_, seq) = self.trace_leaves(manifest, n_ctx as usize)?;
        let spec = seq.ok_or_else(|| anyhow!("the trace commits no sequence leaves"))?;
        if leaves.iter().any(|&l| l as usize >= spec.n_leaves()) {
            return Err(anyhow!("replay of a sequence leaf past the {}", spec.n_leaves()));
        }
        let _serialize = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let backend = backend()?;
        let mut ctx = self.new_trace_context(backend, n_ctx, true)?;
        let mut batch = LlamaBatch::new(TRACE_UBATCH as usize, 1);
        // SAFETY: plain call into the backend.
        if unsafe { llama_cpp_sys_2::ggml_backend_vk_keryx_trace_begin(TRACE_DEVICE) } != 0 {
            return Err(anyhow!("the trace could not be started"));
        }
        let mut rec = RecurSink::new(&spec, leaves);
        let mut run = || -> Result<()> {
            feed_traced(&mut ctx, &mut batch, &toks[..n_prompt], 0, Some(n_prompt - 1), n_prompt, rec.as_mut())?;
            for p in n_prompt..toks.len() {
                feed_traced(&mut ctx, &mut batch, &toks[p..=p], p, Some(p), n_prompt, rec.as_mut())?;
            }
            Ok(())
        };
        if let Err(e) = run() {
            abandon_trace();
            return Err(e);
        }
        // SAFETY: plain call into the backend; ends the evaluation (every dispatch landed).
        if unsafe { llama_cpp_sys_2::ggml_backend_vk_keryx_trace_end(TRACE_DEVICE) } != 0 {
            return Err(trace_error("replay failed"));
        }
        let all = read_seq(&ctx, &spec, toks.len())?;
        if let Some(r) = &rec {
            r.check_end(toks.len())?;
        }
        if leaves.is_empty() {
            if spec.n_recur > 0 {
                return Err(anyhow!("a replay keeps only the recurrent leaves it names"));
            }
            return Ok(all);
        }
        // A recurrent leaf comes whole from the records the replay took; the rest from the cache.
        // A leaf may be named twice (two spans of one checkpoint leaf).
        Ok(leaves.iter().map(|&l| rec.as_ref().and_then(|r| r.kept(l)).unwrap_or(&all[l as usize]).to_vec()).collect())
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
        // Each request restarts the sampler's seed: the same call must reproduce byte-identically.
        let again = engine.generate(prompt, 16, &[]).expect("generate (repeat)");
        assert_eq!(out, again, "decode not deterministic");
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
        let n = engine.tokenize(&prompt, true).len();
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
            mp.load_mtp = true; // as the engine loads it
            mp.no_alloc = true;
            mp.load_mode = llama_cpp_sys_2::LLAMA_LOAD_MODE_NONE; // llama.cpp asserts no_alloc never maps host-pointer buffers
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

    /// A llama.cpp sequence state of `cells` positions and the given layers' K and V rows.
    fn kv_state(cells: u32, layers: &[(Vec<u8>, Vec<u8>)], row: u64) -> Vec<u8> {
        kv_state_cells(cells, layers, row, false)
    }

    /// [`kv_state`] whose cells carry the M-RoPE extension (2D position, token) when `ext`.
    fn kv_state_cells(cells: u32, layers: &[(Vec<u8>, Vec<u8>)], row: u64, ext: bool) -> Vec<u8> {
        let mut b = Vec::new();
        for v in [0xaf14_3cd8u32, 0, 1, cells] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for i in 0..cells {
            let cell: &[u32] = if ext { &[i, 1, 0, 0, 1000 + i, 0] } else { &[i, 1, 0] };
            for v in cell {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in [0u32, layers.len() as u32, 0, 0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for which in 0..2 {
            for (k, v) in layers {
                b.extend_from_slice(&(llama_cpp_sys_2::GGML_TYPE_Q8_0 as u32).to_le_bytes());
                b.extend_from_slice(&row.to_le_bytes());
                b.extend_from_slice(if which == 0 { k } else { v });
            }
        }
        b
    }

    /// Sequence leaves are the per-head cache rows of every position end to end and the masses of
    /// each head's query group, in the node's leaf order.
    #[test]
    fn sequence_leaves_follow_the_node_layout() {
        let (n_pos, hb) = (3usize, 34usize);
        let spec = SeqSpec { n_attn: 1, n_recur: 0, heads: 2, attn: vec![SeqAttn { layer: 0, slot: 0, n_head: 4, n_head_kv: 2, head_dim: 32 }], recur: vec![], interval: 0 };
        spec.check().unwrap();
        let k: Vec<u8> = (0..n_pos * 2 * hb).map(|i| i as u8).collect();
        let v: Vec<u8> = (0..n_pos * 2 * hb).map(|i| (i as u8).wrapping_mul(7)).collect();
        let state = kv_state(n_pos as u32, &[(k.clone(), v.clone())], 2 * hb as u64);
        let cache = parse_kv_state(&state, n_pos, false).unwrap();
        assert!(parse_kv_state(&state, n_pos + 1, false).is_err());
        assert!(parse_kv_state(&state[..state.len() - 1], n_pos, false).is_err());
        // A hybrid memory's recurrent state follows the cache.
        let mut hybrid = state.clone();
        hybrid.extend_from_slice(&[1, 2, 3]);
        assert!(parse_kv_state(&hybrid, n_pos, false).is_err());
        assert_eq!(parse_kv_state(&hybrid, n_pos, true).unwrap().len(), 1);
        // An M-RoPE cache's cells carry their 2D position and token.
        let mrope = kv_state_cells(n_pos as u32, &[(k.clone(), v.clone())], 2 * hb as u64, true);
        let ext = parse_kv_state(&mrope, n_pos, false).unwrap();
        assert_eq!((ext[0].k.as_slice(), ext[0].v.as_slice()), (cache[0].k.as_slice(), cache[0].v.as_slice()));
        let lse: Vec<f32> = (0..n_pos * 4).map(|i| i as f32 * 0.5).collect();
        let leaves = seq_leaf_bytes(&spec, &cache, &lse, n_pos).unwrap();
        assert_eq!(leaves.len(), spec.n_leaves());
        let layout = keryx_verifiable_inference::seq::SeqLayout { n_attn: 1, n_recur: 0, heads: 2 };
        for h in 0..2usize {
            let want_k: Vec<u8> = (0..n_pos).flat_map(|t| k[t * 2 * hb + h * hb..t * 2 * hb + (h + 1) * hb].to_vec()).collect();
            let want_v: Vec<u8> = (0..n_pos).flat_map(|t| v[t * 2 * hb + h * hb..t * 2 * hb + (h + 1) * hb].to_vec()).collect();
            assert_eq!(leaves[layout.kcache(0, h as u64).unwrap() as usize], want_k);
            assert_eq!(leaves[layout.vcache(0, h as u64).unwrap() as usize], want_v);
            let want_m: Vec<u8> = (0..n_pos).flat_map(|t| (0..2).flat_map(move |qi| (((t * 4 + h * 2 + qi) as f32) * 0.5).to_le_bytes())).collect();
            assert_eq!(leaves[layout.lse(0, h as u64).unwrap() as usize], want_m);
        }
        // An unwritten mass is an error, never a value.
        let mut holes = lse.clone();
        holes[5] = f32::NAN;
        assert!(seq_leaf_bytes(&spec, &cache, &holes, n_pos).is_err());
        // A rotated cache is refused.
        let mut rotated = state.clone();
        let rot = 16 + 12 * n_pos + 8;
        rotated[rot..rot + 4].copy_from_slice(&64u32.to_le_bytes());
        assert!(parse_kv_state(&rotated, n_pos, false).is_err());
    }

    /// Recurrent leaves land where the node's layout puts each head's records and checkpoints,
    /// sized as it decodes them; layouts the record capture cannot take are refused.
    #[test]
    fn recurrent_leaves_follow_the_node_layout() {
        use keryx_verifiable_inference::manifest::RECUR_GDA;
        use keryx_verifiable_inference::seq::{record_len, SeqLayout};
        let recur = |layer, slot| SeqRecur { layer, slot, kind: SEQ_KIND_GDA, n_v_heads: 3, head_dim: 32 };
        let spec = SeqSpec {
            n_attn: 1,
            n_recur: 2,
            heads: 4,
            attn: vec![SeqAttn { layer: 1, slot: 0, n_head: 4, n_head_kv: 2, head_dim: 32 }],
            recur: vec![recur(0, 0), recur(2, 1)],
            interval: 16,
        };
        spec.check().unwrap();
        assert_eq!(spec.record_len(), record_len(RECUR_GDA, 32));
        assert_eq!(spec.recur_rows(), vec![[0, 3, 0, 3, 32], [2, 3, 1, 3, 32]]);
        let layout = SeqLayout { n_attn: 1, n_recur: 2, heads: 4 };
        let sink = RecurSink::new(&spec, &[layout.checkpoint(1, 2).unwrap() as u32, layout.kcache(0, 0).unwrap() as u32]).unwrap();
        assert_eq!(sink.keep.len(), 1, "only recurrent leaves are kept");
        for slot in 0..2u64 {
            for h in 0..4u64 {
                let (r, c) = (layout.record(slot, h).unwrap() as u32, layout.checkpoint(slot, h).unwrap() as u32);
                let i = (slot * 3 + h) as usize;
                let want = |rec: bool| (h < 3).then_some((rec, i));
                assert_eq!((sink.slot_head(r), sink.slot_head(c)), (want(true), want(false)), "slot {slot} head {h}");
            }
        }
        assert_eq!(sink.slot_head(layout.vcache(0, 3).unwrap() as u32), None);
        assert_eq!(sink.slot_head(layout.lse(0, 0).unwrap() as u32), None);
        // Nothing recorded yet: every written recurrent leaf is the hash of nothing; the others stay.
        let mut leaves = vec![[7u8; 32]; spec.n_leaves()];
        sink.hashes_into(&mut leaves);
        let empty = *blake3::hash(b"").as_bytes();
        assert_eq!(leaves[layout.record(1, 2).unwrap() as usize], empty);
        assert_eq!(leaves[layout.checkpoint(0, 0).unwrap() as usize], empty);
        assert_eq!(leaves[layout.record(0, 3).unwrap() as usize], [7u8; 32]);
        assert!(sink.check_end(0).is_ok() && sink.check_end(1).is_err());
        // Checkpoints inside a traced batch, key-wise decays, uneven layers: refused.
        assert!(SeqSpec { interval: 12, ..spec.clone() }.check().is_err());
        assert!(SeqSpec { recur: vec![recur(0, 0), SeqRecur { kind: 4, ..recur(2, 1) }], ..spec.clone() }.check().is_err());
        assert!(SeqSpec { recur: vec![recur(0, 0), SeqRecur { n_v_heads: 2, ..recur(2, 1) }], ..spec.clone() }.check().is_err());
        assert!(SeqSpec { n_recur: 3, ..spec.clone() }.check().is_err());
    }

    /// A capture holds the asked bytes and the chaining value of every chunk of its leaf, which
    /// the opening's range proof is built from.
    #[test]
    fn sequence_captures_prove_against_the_leaf() {
        use keryx_verifiable_inference::merkle::b3_root;
        let leaf: Vec<u8> = (0..5000u32).map(|i| (i * 13) as u8).collect();
        let c = seq_capture(&leaf, 1024, 3072).unwrap();
        assert_eq!(c.bytes, leaf[1024..3072]);
        assert_eq!(c.cvs.len(), 5);
        assert_eq!(b3_root(&c.cvs), *blake3::hash(&leaf).as_bytes());
        let small = seq_capture(&leaf[..100], 0, 100).unwrap();
        assert_eq!(small.cvs, vec![*blake3::hash(&leaf[..100]).as_bytes()]);
        assert!(seq_capture(&leaf, 4096, 6000).is_err());
    }

    /// Traced batches commit the batch size the node expects: 512-token prompt batches, the rest
    /// of the prompt as one batch, one position per generated token.
    #[test]
    fn traced_batches_commit_the_expected_batch_size() {
        use keryx_verifiable_inference::verify::expected_ubatch;
        assert_eq!(expected_ubatch(0, 970), 512);
        assert_eq!(expected_ubatch(511, 970), 512);
        assert_eq!(expected_ubatch(512, 970), 458);
        assert_eq!(expected_ubatch(969, 970), 458);
        assert_eq!(expected_ubatch(970, 970), 1);
        assert_eq!(expected_ubatch(512, 517), 5);
        // TRACE_UBATCH divides 512, so no 8-token batch straddles a committed batch.
        assert_eq!(512 % TRACE_UBATCH, 0);
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
