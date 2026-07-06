//! Phase-1 in-process OPoI inference — llama.cpp via FFI (`llama-cpp-2`, **Vulkan** backend).
//!
//! The miner's ONLY inference engine (the external llama-server child process is gone):
//! same GGUF models, same ggml Vulkan backend, same greedy (temperature-0) decoding through
//! the model's own chat template — but linked into the miner, so there is no HTTP hop, no
//! child-process lifecycle, and (Phase 2) the PoM walk can eventually read inference's own
//! resident weight buffers instead of keeping a second VRAM copy.
//!

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use log::info;

/// Context window — matches the `-c 4096` the llama-server launch used.
const CTX_SIZE: u32 = 4096;

/// Process-wide ggml backend guard: `llama_backend_init` must run exactly once per process
/// (`LlamaBackend::init` errors on a second call). Never torn down — model switches drop the
/// [`LlamaEngine`] (freeing the model's VRAM) but keep the backend alive.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn backend() -> Result<&'static LlamaBackend> {
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let _g = INIT.lock().unwrap_or_else(|p| p.into_inner());
    if BACKEND.get().is_none() {
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
        info!("llm-engine: model resident, ready to serve");
        Ok(Self { model, lock: Mutex::new(()) })
    }

    /// Chat completion through the GGUF's own chat template, greedy decoding (temperature-0
    /// equivalent — keeps OPoI answers stable), capped at `max_tokens` generated tokens.
    ///
    /// `stop_strings` are scanned over the decoded output and cut the turn (marker excluded)
    /// when the model writes an end marker as plain text instead of emitting an EOG token.
    /// Needed for GGUFs whose chat template was swapped over a vocab that lacks the template's
    /// control tokens (abliterated Llama-3.3-70B: ChatML over stock LLaMA-3 — `<|im_end|>`
    /// tokenizes as text, `<|eot_id|>` is never emitted, and without the cut the model reopens
    /// `assistant` and repeats its answer until `max_tokens`). Empty for well-formed models.
    pub fn chat(&self, system: &str, user: &str, max_tokens: usize, stop_strings: &[&str]) -> Result<String> {
        let _serialize = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let backend = backend()?;

        // The model's baked-in chat template — the same one llama-server applies.
        let tmpl = self
            .model
            .chat_template(None)
            .map_err(|e| anyhow!("llm-engine: model has no usable chat template: {e}"))?;
        let msgs = vec![
            LlamaChatMessage::new("system".to_string(), system.to_string())?,
            LlamaChatMessage::new("user".to_string(), user.to_string())?,
        ];
        // add_ass = true: end with the assistant header so generation starts the reply.
        let prompt = self.model.apply_chat_template(&tmpl, &msgs, true)?;

        // str_to_token parses special tokens (the template's control tokens) and AddBos
        // lets the tokenizer add BOS iff the model wants one — llama-server semantics.
        let tokens = self.model.str_to_token(&prompt, AddBos::Always)?;
        if tokens.len() as u32 >= CTX_SIZE {
            return Err(anyhow!("llm-engine: prompt ({} tokens) exceeds the {} context", tokens.len(), CTX_SIZE));
        }

        // Fresh context per request: n_batch = CTX_SIZE so the whole prompt decodes in one
        // batch; KV is dropped with the context when this returns.
        let mut ctx = self.model.new_context(
            backend,
            LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(CTX_SIZE))
                .with_n_batch(CTX_SIZE),
        )?;

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        let last = tokens.len() as i32 - 1;
        for (i, tok) in (0_i32..).zip(tokens.iter().copied()) {
            batch.add(tok, i, &[0], i == last)?; // logits only for the last prompt token
        }
        ctx.decode(&mut batch)?;

        // Greedy decode until end-of-generation or the token budget. Output is accumulated
        // as BYTES: byte-level BPE tokens can split UTF-8 sequences mid-character, so
        // per-token string conversion would corrupt multi-byte output.
        let budget = max_tokens.min((CTX_SIZE as usize - tokens.len()).saturating_sub(1));
        let mut sampler = LlamaSampler::greedy();
        let mut out = Vec::<u8>::new();
        let mut n_cur = batch.n_tokens();
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

    /// End-to-end in-process inference on a real GGUF + GPU. Ignored by default (loads a
    /// multi-GB model); run with:
    ///   KERYX_TEST_GGUF=<path to model.gguf> cargo test --release -- --ignored inproc_chat
    #[test]
    #[ignore = "loads a multi-GB GGUF onto the GPU; set KERYX_TEST_GGUF to run"]
    fn inproc_chat_smoke() {
        let Ok(gguf) = std::env::var("KERYX_TEST_GGUF") else {
            eprintln!("SKIP: KERYX_TEST_GGUF not set");
            return;
        };
        let engine = LlamaEngine::launch(&gguf).expect("engine launch");
        let out = engine
            .chat("You are a terse assistant.", "Reply with the single word: pong", 16, &[])
            .expect("chat");
        eprintln!("model replied: {out:?}");
        assert!(!out.trim().is_empty(), "empty completion");
        // Greedy decoding is deterministic: the same call must reproduce byte-identically.
        let again = engine
            .chat("You are a terse assistant.", "Reply with the single word: pong", 16, &[])
            .expect("chat (repeat)");
        assert_eq!(out, again, "greedy decode not deterministic");
    }

    /// The stop scan must catch a marker regardless of how token pieces split it, cut at the
    /// EARLIEST marker, and never fire on clean output.
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
