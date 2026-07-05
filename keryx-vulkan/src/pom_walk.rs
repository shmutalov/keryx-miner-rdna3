//! GPU Proof-of-Model walk: dispatch the `pom_walk` compute shader over a resident weight blob to
//! find the lowest nonce in a batch whose `pom_pow_value <= target`. The folds are byte-identical
//! to `src/pom.rs`, so a nonce found here builds a `PomProof` the node accepts.

use crate::{GpuBuffer, Kernel, Vk};
use std::io::Cursor;

/// SPIR-V for the PoM walk, compiled from `shaders/pom_walk.comp` by build.rs.
const POM_WALK_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_walk.spv"));

/// POM_WALK_STEPS — must match `pom::POM_WALK_STEPS` and the node.
pub const POM_WALK_STEPS: u32 = 256;

/// Push-constant block — layout MUST match the `Push` block in `pom_walk.comp` (std430: twelve u64
/// at 0..96, then three u32 at 96, 100, 104; total 112 bytes incl. tail pad). The weight blob is
/// split into power-of-two-sized shards (each a separate device-address buffer, ≤ the 2 GiB
/// `maxMemoryAllocationSize`); the shader maps a chunk to its shard via `shard_shift`/`shard_mask`
/// and reads the shard's GPU address from a small bound address table.
#[repr(C)]
#[derive(Clone, Copy)]
struct PomPush {
    p: [u64; 4],
    t: [u64; 4],
    timestamp: u64,
    n_chunks: u64,
    start_nonce: u64,
    shard_mask: u64, // chunks_per_shard - 1
    k: u32,
    batch: u32,
    shard_shift: u32, // log2(chunks_per_shard)
}

const NO_WINNER: u32 = 0xFFFF_FFFF;

/// Chunks per shard: 2^25 × 32 B = 1 GiB, comfortably under the AMD 2 GiB `maxMemoryAllocationSize`
/// single-allocation cap, with headroom for driver overhead. Power of two so the shader maps a
/// chunk to (shard, offset) with a shift + mask instead of 64-bit divide.
const SHARD_CHUNKS: u64 = 1 << 25;

/// Max nonces per GPU dispatch. The walk is latency-bound (256 dependent reads/nonce), so a full
/// 1<<20 batch in one dispatch can exceed the Windows TDR watchdog (~2 s) and lose the device.
/// 65,536 keeps each dispatch to a few ms on device-local VRAM while staying far above launch cost.
const MAX_DISPATCH_NONCES: u32 = 1 << 16;

/// Resident GPU PoM miner: the weight blob lives in a storage buffer; `mine` re-dispatches the
/// walk over nonce batches. Build once per mining tier (the weight blob is large).
pub struct PomWalkGpu {
    vk: Vk,
    kernel: Kernel,
    shards: Vec<GpuBuffer>, // weight blob split into ≤1 GiB device-address buffers
    addr_table: GpuBuffer,  // bound SSBO: one u64 GPU address per shard
    winner: GpuBuffer,
    n_chunks: u64,
    shard_chunks: u64,
}

impl PomWalkGpu {
    /// Upload the canonical weight blob (`weight_words` = the model's quant bytes as little-endian
    /// u64 words, `n_chunks * 4` of them) and compile the walk kernel on the RDNA3 GPU.
    pub fn new(weight_words: &[u64], n_chunks: u64) -> Result<Self, String> {
        Self::new_sharded(weight_words, n_chunks, SHARD_CHUNKS)
    }

    /// Like [`new`](Self::new) but with an explicit shard size (chunks per shard, power of two).
    /// Lets tests force a multi-shard layout without multi-GiB allocations.
    pub fn new_sharded(weight_words: &[u64], n_chunks: u64, shard_chunks: u64) -> Result<Self, String> {
        if weight_words.len() as u64 != n_chunks * 4 {
            return Err(format!(
                "weight blob size mismatch: {} words for {} chunks (expected {})",
                weight_words.len(),
                n_chunks,
                n_chunks * 4
            ));
        }
        Self::new_streamed_sharded(None, n_chunks, shard_chunks, &mut |first_chunk, out| {
            let first_word = (first_chunk * 4) as usize;
            out.copy_from_slice(&words_as_bytes(weight_words)[first_word * 8..first_word * 8 + out.len()]);
            Ok(())
        })
    }

    /// Zero-dup constructor: build the resident blob on `device_index` (`None` = the historical
    /// auto pick) by pulling bytes through `source(first_chunk, out)` — `out` is always a multiple
    /// of 32 B (whole chunks). The miner streams straight from the GGUF on disk, so the full packed
    /// host blob (`Vec<u64>`, ~1x model size) never exists.
    pub fn new_streamed(
        device_index: Option<usize>,
        n_chunks: u64,
        source: &mut dyn FnMut(u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<Self, String> {
        Self::new_streamed_sharded(device_index, n_chunks, SHARD_CHUNKS, source)
    }

    /// [`new_streamed`](Self::new_streamed) with an explicit shard size, for multi-shard tests.
    pub fn new_streamed_sharded(
        device_index: Option<usize>,
        n_chunks: u64,
        shard_chunks: u64,
        source: &mut dyn FnMut(u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<Self, String> {
        if n_chunks == 0 {
            return Err("weight blob is empty (0 chunks)".into());
        }
        if !shard_chunks.is_power_of_two() {
            return Err(format!("shard_chunks must be a power of two, got {shard_chunks}"));
        }
        let vk = Vk::new_for_device(device_index)?;
        let spirv = ash::util::read_spv(&mut Cursor::new(POM_WALK_SPV)).map_err(|e| e.to_string())?;
        // Two descriptor bindings: the winner buffer and the shard address table. The (large) weight
        // shards are reached by device address, not bound as descriptors.
        let kernel = vk.make_kernel(&spirv, 2, std::mem::size_of::<PomPush>() as u32)?;

        // Split the blob on chunk boundaries into device-address shards; collect their GPU addresses.
        let n_shards = n_chunks.div_ceil(shard_chunks);
        let mut shards: Vec<GpuBuffer> = Vec::with_capacity(n_shards as usize);
        let mut addrs: Vec<u64> = Vec::with_capacity(n_shards as usize);
        for s in 0..n_shards {
            let first_chunk = s * shard_chunks;
            let shard_bytes = (((s + 1) * shard_chunks).min(n_chunks) - first_chunk) * 32;
            // Device-local VRAM (staged copy): the walk's random reads are ~100x faster here than
            // host-visible memory — host-visible overran the TDR watchdog → DEVICE_LOST. The shard
            // is filled through a bounded staging window; offsets are chunk-aligned by construction
            // (the window size and shard size are both multiples of 32).
            let (buf, addr) = vk.create_device_local_address_buffer_streamed(shard_bytes, &mut |off, out| {
                debug_assert!(off % 32 == 0 && out.len() % 32 == 0);
                source(first_chunk + off / 32, out)
            })?;
            shards.push(buf);
            addrs.push(addr);
        }

        // Address table (tiny — one u64 per shard) bound as a normal storage buffer at binding 1.
        let addr_table = vk.create_buffer((addrs.len() * 8) as u64)?;
        vk.write_buffer(&addr_table, words_as_bytes(&addrs));
        let winner = vk.create_buffer(4)?;

        Ok(Self { vk, kernel, shards, addr_table, winner, n_chunks, shard_chunks })
    }

    /// Name of the GPU the miner is running on.
    pub fn device_name(&self) -> &str {
        self.vk.device_name()
    }

    /// Raw enumeration index of the GPU the blob is resident on.
    pub fn device_index(&self) -> usize {
        self.vk.device_index()
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_chunks
    }

    /// Search nonces `[start, start + batch)`. Returns the lowest winning nonce, or None.
    ///
    /// `pph_words` are the pre_pow_hash's 4 LE u64 words, already salted for the block's era by
    /// the caller (H3 XORs `POM_H3_PPH_SALT` in — see `pom::pph_words_for_era`). The kernel is
    /// era-agnostic: it folds whatever words it receives, so no shader change at the H3 gate.
    ///
    /// The batch is ground in `MAX_DISPATCH_NONCES`-sized sub-dispatches, in increasing nonce order,
    /// so no single GPU dispatch runs long enough to trip the Windows TDR watchdog (DEVICE_LOST).
    /// Sub-batches are ascending, so the first one with any winner holds the global lowest nonce —
    /// returning there is identical to grinding the whole batch, and skips the rest.
    pub fn mine(&self, pph_words: &[u64; 4], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u32) -> Option<u64> {
        let mut done: u32 = 0;
        while done < batch {
            let sub = (batch - done).min(MAX_DISPATCH_NONCES);
            self.vk.write_buffer(&self.winner, &NO_WINNER.to_le_bytes());
            let push = PomPush {
                p: *pph_words,
                t: words4(target_le),
                timestamp,
                n_chunks: self.n_chunks,
                start_nonce: start + done as u64,
                shard_mask: self.shard_chunks - 1,
                k: POM_WALK_STEPS,
                batch: sub,
                shard_shift: self.shard_chunks.trailing_zeros(),
            };
            let groups = sub.div_ceil(64); // local_size_x = 64
            self.vk.dispatch(&self.kernel, &[&self.winner, &self.addr_table], push_bytes(&push), groups);

            let mut out = [0u8; 4];
            self.vk.read_buffer(&self.winner, &mut out);
            if let offset @ 0..=0xFFFF_FFFE = u32::from_le_bytes(out) {
                return Some(start + done as u64 + offset as u64);
            }
            done += sub;
        }
        None
    }
}

impl Drop for PomWalkGpu {
    fn drop(&mut self) {
        self.vk.destroy_buffer(&self.winner);
        self.vk.destroy_buffer(&self.addr_table);
        for shard in &self.shards {
            self.vk.destroy_buffer(shard);
        }
        self.vk.destroy_kernel(&self.kernel);
    }
}

/// SPIR-V for the zero-dup prefix-table walk + the chunk-fetch verifier.
const POM_WALK_PREFIX_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_walk_prefix.spv"));
const POM_FETCH_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_fetch.spv"));

/// Push-constant block for `pom_walk_prefix.comp` (std430: eleven u64 at 0..88, three u32 at
/// 88..100; padded to 104).
#[repr(C)]
#[derive(Clone, Copy)]
struct PomPrefixPush {
    p: [u64; 4],
    t: [u64; 4],
    timestamp: u64,
    n_chunks: u64,
    start_nonce: u64,
    k: u32,
    batch: u32,
    n_tensors: u32,
    _pad: u32,
}

/// Push-constant block for `pom_fetch.comp`.
#[repr(C)]
#[derive(Clone, Copy)]
struct FetchPush {
    chunk: u64,
    n_tensors: u32,
    _pad: u32,
}

/// ZERO-DUP resident PoM miner: walks the inference engine's own weight tensors (ggml-vulkan
/// buffers, reached by buffer device address) via a per-tensor prefix table — no miner-owned
/// weight copy exists. Built over a [`Vk`] wrapping ggml's device (`Vk::from_raw_handles`);
/// every dispatch routes through ggml's guarded queue hook. Must be dropped BEFORE the
/// inference engine unloads the model that owns the tensors.
pub struct PomWalkShared {
    vk: Vk,
    walk: Kernel,
    fetch: Kernel,
    prefix: GpuBuffer, // binding 1: n_tensors + 1 cumulative chunk starts (u64)
    addrs: GpuBuffer,  // binding 2: n_tensors GPU virtual addresses (u64)
    winner: GpuBuffer,
    out32: GpuBuffer, // 32-byte readback target for the chunk-fetch verifier
    /// Miner-owned supplement buffers for tensors the engine keeps host-side (their table
    /// addresses point in here); owned so they live exactly as long as the walk.
    supplements: Vec<GpuBuffer>,
    n_chunks: u64,
    n_tensors: u32,
}

impl PomWalkShared {
    /// Build over `tensors`: one `(gpu_addr, n_chunks)` per tensor in CANONICAL (name-sorted)
    /// order, zero-chunk tensors already skipped — the same layout `pom::WeightIndex` indexes,
    /// so canonical chunk `i` resolves to the identical bytes the node pinned in `R_T`.
    /// `supplements`: buffers (created on the same `vk`) that back any non-engine-resident
    /// table entries; ownership transfers here.
    pub fn new(vk: Vk, tensors: &[(u64, u64)], supplements: Vec<GpuBuffer>) -> Result<Self, String> {
        if tensors.is_empty() {
            return Err("shared PoM walk: empty tensor table".into());
        }
        let n_tensors = u32::try_from(tensors.len()).map_err(|_| "tensor table too large")?;
        let mut prefix: Vec<u64> = Vec::with_capacity(tensors.len() + 1);
        let mut addrs: Vec<u64> = Vec::with_capacity(tensors.len());
        let mut total: u64 = 0;
        for &(addr, chunks) in tensors {
            if addr == 0 || chunks == 0 {
                return Err("shared PoM walk: null address or empty tensor in table".into());
            }
            prefix.push(total);
            addrs.push(addr);
            total += chunks;
        }
        prefix.push(total); // sentinel: prefix[n_tensors] == n_chunks
        if total == 0 {
            return Err("shared PoM walk: table produced 0 chunks".into());
        }

        let walk_spirv = ash::util::read_spv(&mut Cursor::new(POM_WALK_PREFIX_SPV)).map_err(|e| e.to_string())?;
        let walk = vk.make_kernel(&walk_spirv, 3, std::mem::size_of::<PomPrefixPush>() as u32)?;
        let fetch_spirv = ash::util::read_spv(&mut Cursor::new(POM_FETCH_SPV)).map_err(|e| e.to_string())?;
        let fetch = vk.make_kernel(&fetch_spirv, 3, std::mem::size_of::<FetchPush>() as u32)?;

        let prefix_buf = vk.create_buffer((prefix.len() * 8) as u64)?;
        vk.write_buffer(&prefix_buf, words_as_bytes(&prefix));
        let addrs_buf = vk.create_buffer((addrs.len() * 8) as u64)?;
        vk.write_buffer(&addrs_buf, words_as_bytes(&addrs));
        let winner = vk.create_buffer(4)?;
        let out32 = vk.create_buffer(32)?;

        Ok(Self {
            vk,
            walk,
            fetch,
            prefix: prefix_buf,
            addrs: addrs_buf,
            winner,
            out32,
            supplements,
            n_chunks: total,
            n_tensors,
        })
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_chunks
    }

    pub fn device_name(&self) -> &str {
        self.vk.device_name()
    }

    /// Fetch canonical chunk `off` through the exact table + BDA path the walk uses — the
    /// host samples random chunks against the GGUF-backed index before mining is allowed.
    pub fn read_chunk(&self, off: u64) -> [u8; 32] {
        assert!(off < self.n_chunks, "chunk out of range");
        let push = FetchPush { chunk: off, n_tensors: self.n_tensors, _pad: 0 };
        self.vk.dispatch(&self.fetch, &[&self.out32, &self.prefix, &self.addrs], fetch_push_bytes(&push), 1);
        let mut out = [0u8; 32];
        self.vk.read_buffer(&self.out32, &mut out);
        out
    }

    /// Search nonces `[start, start + batch)`. Identical sub-dispatch grinding (TDR-bounded)
    /// and lowest-winner semantics as [`PomWalkGpu::mine`], including the caller-salted
    /// `pph_words` era contract.
    pub fn mine(&self, pph_words: &[u64; 4], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u32) -> Option<u64> {
        let mut done: u32 = 0;
        while done < batch {
            let sub = (batch - done).min(MAX_DISPATCH_NONCES);
            self.vk.write_buffer(&self.winner, &NO_WINNER.to_le_bytes());
            let push = PomPrefixPush {
                p: *pph_words,
                t: words4(target_le),
                timestamp,
                n_chunks: self.n_chunks,
                start_nonce: start + done as u64,
                k: POM_WALK_STEPS,
                batch: sub,
                n_tensors: self.n_tensors,
                _pad: 0,
            };
            let groups = sub.div_ceil(64); // local_size_x = 64
            self.vk.dispatch(&self.walk, &[&self.winner, &self.prefix, &self.addrs], prefix_push_bytes(&push), groups);

            let mut out = [0u8; 4];
            self.vk.read_buffer(&self.winner, &mut out);
            if let offset @ 0..=0xFFFF_FFFE = u32::from_le_bytes(out) {
                return Some(start + done as u64 + offset as u64);
            }
            done += sub;
        }
        None
    }
}

impl Drop for PomWalkShared {
    fn drop(&mut self) {
        for b in &self.supplements {
            self.vk.destroy_buffer(b);
        }
        self.vk.destroy_buffer(&self.out32);
        self.vk.destroy_buffer(&self.winner);
        self.vk.destroy_buffer(&self.addrs);
        self.vk.destroy_buffer(&self.prefix);
        self.vk.destroy_kernel(&self.fetch);
        self.vk.destroy_kernel(&self.walk);
    }
}

fn prefix_push_bytes(p: &PomPrefixPush) -> &[u8] {
    unsafe { std::slice::from_raw_parts(p as *const PomPrefixPush as *const u8, std::mem::size_of::<PomPrefixPush>()) }
}

fn fetch_push_bytes(p: &FetchPush) -> &[u8] {
    unsafe { std::slice::from_raw_parts(p as *const FetchPush as *const u8, std::mem::size_of::<FetchPush>()) }
}

/// 32 LE bytes → 4 u64 words (matches `pom::pph_words` / `words4`).
pub fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

fn words_as_bytes(words: &[u64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words)) }
}

fn push_bytes(p: &PomPush) -> &[u8] {
    unsafe { std::slice::from_raw_parts(p as *const PomPush as *const u8, std::mem::size_of::<PomPush>()) }
}
