//! Proof-of-Model — miner-side possession proof builder (build order §6).
//!
//! Byte-exact mirror of the node's verifier (`keryx-node-hardfork consensus/core/src/pom.rs`)
//! and the canonical reference (`pom-core`). The miner runs the memory-hard walk over its
//! resident weight blob; once a winning nonce is found, `build_proof` re-walks (recording the
//! trace), commits it, and opens the `t` Fiat-Shamir-selected steps with Merkle paths to the
//! tier root `R_T` and the trace root.
//!
//! The `PomProof`/`PomOpening` structs MUST keep the exact field order/types of the node's
//! (borsh wire format), and the primitives MUST stay bit-identical (the node re-derives the
//! same challenges and recomputes the same transitions). See POM_CONSENSUS_SPEC.md.

use borsh::{BorshDeserialize, BorshSerialize};
use candle_core::quantized::gguf_file;
use candle_core::Device;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::PathBuf;

fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(target_family = "unix")]
    {
        use std::os::unix::fs::FileExt;
        return file.read_exact_at(buf, offset);
    }
    #[cfg(target_family = "windows")]
    {
        use std::os::windows::fs::FileExt;
        let mut pos = 0usize;
        while pos < buf.len() {
            let n = file.seek_read(&mut buf[pos..], offset + pos as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "read_exact_at: eof"));
            }
            pos += n;
        }
        return Ok(());
    }
}
use std::sync::{Mutex, OnceLock};

pub const CHUNK_WORDS: usize = 4; // 32 B chunk
const SEED_SALT: u64 = 0x4B65727978500; // "KeryxP"

/// Walk length / opening count — MUST match the node's `POM_WALK_STEPS` / `POM_OPENINGS`.
/// K=256 — chosen compromise (~25 MH/s on a 3090, solid possession).
pub const POM_WALK_STEPS: u32 = 256;
pub const POM_OPENINGS: usize = 32;

/// Merkle tree checkpoint interval: store every K-th level on disk (level 0 never stored —
/// recomputed from the GGUF on demand; root always stored).
const CHECKPOINT_INTERVAL: u32 = 6;

// --- wire structs (field order == node's PomOpening/PomProof) ---

#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomOpening {
    pub state_before: u64,
    pub chunk: [u8; 32],
    pub weight_path: Vec<[u8; 32]>,
    pub trace_path_before: Vec<[u8; 32]>,
    pub trace_path_after: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomProof {
    pub tier: u8,
    pub trace_root: [u8; 32],
    pub pow_value: [u8; 32],
    pub final_state: u64,
    pub initial_trace_path: Vec<[u8; 32]>,
    pub final_trace_path: Vec<[u8; 32]>,
    pub openings: Vec<PomOpening>,
}

// --- byte-exact primitives (mirror node) ---

#[inline]
pub fn blake(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

#[inline]
pub fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

#[inline]
pub fn seed_state(pow_seed: u64) -> u64 {
    mix64(pow_seed ^ SEED_SALT)
}

#[inline]
pub fn transition(state: u64, chunk: &[u64; CHUNK_WORDS]) -> u64 {
    let mut h = state;
    for &w in chunk.iter() {
        h ^= w;
    }
    mix64(h)
}

#[inline]
pub fn chunk_to_words(c: &[u8; 32]) -> [u64; CHUNK_WORDS] {
    let mut w = [0u64; CHUNK_WORDS];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

#[inline]
pub fn words_to_bytes(w: &[u64; CHUNK_WORDS]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for (i, wi) in w.iter().enumerate() {
        b[i * 8..i * 8 + 8].copy_from_slice(&wi.to_le_bytes());
    }
    b
}

#[inline]
fn trace_leaf(state: u64) -> [u8; 32] {
    blake(&state.to_le_bytes())
}

fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(left);
    buf[32..].copy_from_slice(right);
    blake(&buf)
}

pub fn le_leq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if a[i] < b[i] {
            return true;
        }
        if a[i] > b[i] {
            return false;
        }
    }
    true
}

#[inline]
fn pph_words(pre_pow_hash: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(pre_pow_hash[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

/// H3 domain salt applied to the pre_pow_hash words feeding both PoM folds at/after
/// `POM_LEVEL_ACTIVATION_DAA`. Forced-update mechanism: every walk trajectory and pow value
/// changes at the gate, so pre-H3 binaries produce proofs the node rejects. The Vulkan walk
/// kernel is unchanged — the host salts the pph words before they enter the push constants
/// (`pom_gpu::mine`). Derivation: sha256("keryx-h3-pom-pph-salt") read as 4 little-endian
/// u64 words. MUST equal the node's `POM_H3_PPH_SALT`.
pub const POM_H3_PPH_SALT: [u64; 4] = [0x7C99D381176D4EC4, 0xC2E28E3E28118C36, 0xD496CE1B129B76CA, 0x47CF0979FA580BCE];

/// pph words for the era selected by `h3` (raw pre-H3, salted at/after the H3 gate).
#[inline]
pub fn pph_words_for_era(pre_pow_hash: &[u8; 32], h3: bool) -> [u64; 4] {
    let mut w = pph_words(pre_pow_hash);
    if h3 {
        for (wi, si) in w.iter_mut().zip(POM_H3_PPH_SALT.iter()) {
            *wi ^= si;
        }
    }
    w
}

#[inline]
fn pom_block_seed_from_words(p: &[u64; 4], timestamp: u64, nonce: u64) -> u64 {
    let mut s = mix64(nonce ^ 0x4B65727978531);
    s = mix64(s ^ timestamp);
    s = mix64(s ^ p[0]);
    s = mix64(s ^ p[1]);
    s = mix64(s ^ p[2]);
    s = mix64(s ^ p[3]);
    s
}

/// Canonical block seed = initial walk state. mix64-fold of (nonce, time, pre_pow_hash).
/// BYTE-IDENTICAL to the Vulkan walk kernel's seed fold and the node's `pom_block_seed`(`_h3`).
pub fn pom_block_seed(pre_pow_hash: &[u8; 32], timestamp: u64, nonce: u64, h3: bool) -> u64 {
    pom_block_seed_from_words(&pph_words_for_era(pre_pow_hash, h3), timestamp, nonce)
}

/// Canonical pow value (256-bit LE) = mix64-fold of (final_state, pre_pow_hash).
/// BYTE-IDENTICAL to the Vulkan walk kernel's pow fold and the node's `pom_pow_value`(`_h3`).
pub fn pom_pow_value(final_state: u64, pre_pow_hash: &[u8; 32], h3: bool) -> [u8; 32] {
    let p = pph_words_for_era(pre_pow_hash, h3);
    let o0 = mix64(final_state ^ p[0] ^ 0x9E3779B97F4A7C15);
    let o1 = mix64(o0 ^ p[1] ^ 0xC2B2AE3D27D4EB4F);
    let o2 = mix64(o1 ^ p[2] ^ 0x165667B19E3779F9);
    let o3 = mix64(o2 ^ p[3] ^ 0xD6E8FEB86659FD93);
    let mut out = [0u8; 32];
    out[0..8].copy_from_slice(&o0.to_le_bytes());
    out[8..16].copy_from_slice(&o1.to_le_bytes());
    out[16..24].copy_from_slice(&o2.to_le_bytes());
    out[24..32].copy_from_slice(&o3.to_le_bytes());
    out
}

pub fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    assert!(!leaves.is_empty(), "merkle_root: empty leaves");
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            next.push(hash_pair(&level[i], &r));
            i += 2;
        }
        level = next;
    }
    level[0]
}

pub fn merkle_proof(leaves: &[[u8; 32]], index: usize) -> Vec<[u8; 32]> {
    let mut path = Vec::new();
    let mut level = leaves.to_vec();
    let mut idx = index;
    while level.len() > 1 {
        let sib_idx = if idx & 1 == 0 { idx + 1 } else { idx - 1 };
        let sib = if sib_idx < level.len() { level[sib_idx] } else { level[idx] };
        path.push(sib);
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            next.push(hash_pair(&level[i], &r));
            i += 2;
        }
        idx >>= 1;
        level = next;
    }
    path
}

fn verify_merkle(leaf: [u8; 32], index: u64, path: &[[u8; 32]], root: &[u8; 32]) -> bool {
    let mut acc = leaf;
    let mut idx = index;
    for sib in path {
        acc = if idx & 1 == 0 { hash_pair(&acc, sib) } else { hash_pair(sib, &acc) };
        idx >>= 1;
    }
    &acc == root
}

/// Fiat-Shamir challenge step-indices — byte-layout identical to node/pom-core.
pub fn challenges(pre_pow_hash: &[u8; 32], nonce: u64, trace_root: &[u8; 32], pow_value: &[u8; 32], t: usize, k: u32) -> Vec<u32> {
    let mut fs = [0u8; 104];
    fs[..32].copy_from_slice(pre_pow_hash);
    fs[32..40].copy_from_slice(&nonce.to_le_bytes());
    fs[40..72].copy_from_slice(trace_root);
    fs[72..104].copy_from_slice(pow_value);
    let seed = blake(&fs);
    let mut out = Vec::with_capacity(t);
    for j in 0..t as u64 {
        let mut buf = [0u8; 40];
        buf[..32].copy_from_slice(&seed);
        buf[32..].copy_from_slice(&j.to_le_bytes());
        let d = blake(&buf);
        let v = u64::from_le_bytes(d[..8].try_into().unwrap());
        out.push((v % k as u64) as u32);
    }
    out
}

/// The hot search walk: K data-dependent reads, returns only `state[K]` (no trace recording).
/// This is the per-nonce work; on GPU (slice 3b) this becomes the kernel over VRAM weights.
pub fn walk_final<F: Fn(u64) -> [u64; CHUNK_WORDS]>(seed: u64, n_chunks: u64, k: u32, read_chunk: F) -> u64 {
    let mut state = seed;
    let mut off = state % n_chunks;
    for _ in 0..k {
        state = transition(state, &read_chunk(off));
        off = state % n_chunks;
    }
    state
}

/// CPU Proof-of-Model mining (slice 3a — functional, slow). Searches nonces in
/// `nonce_start..nonce_start+max_nonces`; on the first whose `pom_pow_value <= target`,
/// re-walks to build the full `PomProof`. GPU fast-path is slice 3b. Returns the winning
/// nonce + proof, or None if the range is exhausted.
#[allow(clippy::too_many_arguments)]
pub fn mine_pom(
    index: &WeightIndex,
    tier: u8,
    pre_pow_hash: &[u8; 32],
    timestamp: u64,
    target: &[u8; 32],
    k: u32,
    t: usize,
    nonce_start: u64,
    max_nonces: u64,
    h3: bool,
) -> Option<(u64, PomProof)> {
    for nonce in nonce_start..nonce_start.saturating_add(max_nonces) {
        let seed = pom_block_seed(pre_pow_hash, timestamp, nonce, h3);
        let final_state = walk_final(seed, index.n_chunks, k, |o| index.read_chunk(o));
        if le_leq(&pom_pow_value(final_state, pre_pow_hash, h3), target) {
            let proof = build_proof(tier, pre_pow_hash, nonce, seed, index.n_chunks, k, t, |o| index.read_chunk(o), |o| index.merkle_path(o), h3);
            return Some((nonce, proof));
        }
    }
    None
}

/// PROVER. Re-walk the (already-won) nonce recording the trace, commit it, and open the
/// `t` FS-selected steps. `read_chunk(off)` reads the 32 B chunk at canonical chunk index
/// `off` from the resident weight blob; `weight_leaves` is the precomputed per-chunk leaf
/// set (`blake(chunk_bytes)`) over the canonical layout, used to produce weight Merkle paths.
#[allow(clippy::too_many_arguments)]
pub fn build_proof<F, WP>(
    tier: u8,
    pre_pow_hash: &[u8; 32],
    nonce: u64,
    seed: u64,
    n_chunks: u64,
    k: u32,
    t: usize,
    read_chunk: F,
    weight_path: WP,
    h3: bool,
) -> PomProof
where
    F: Fn(u64) -> [u64; CHUNK_WORDS],
    WP: Fn(u64) -> Vec<[u8; 32]>,
{
    let mut trace = Vec::with_capacity(k as usize + 1);
    let mut state = seed;
    trace.push(state);
    let mut off = state % n_chunks;
    for _ in 0..k {
        state = transition(state, &read_chunk(off));
        trace.push(state);
        off = state % n_chunks;
    }
    let trace_leaves: Vec<[u8; 32]> = trace.iter().map(|&s| trace_leaf(s)).collect();
    let trace_root = merkle_root(&trace_leaves);
    let final_state = trace[k as usize];
    let pow_value = pom_pow_value(final_state, pre_pow_hash, h3);

    let chs = challenges(pre_pow_hash, nonce, &trace_root, &pow_value, t, k);
    let openings = chs
        .iter()
        .map(|&i| {
            let i = i as usize;
            let sb = trace[i];
            let off = sb % n_chunks;
            PomOpening {
                state_before: sb,
                chunk: words_to_bytes(&read_chunk(off)),
                weight_path: weight_path(off),
                trace_path_before: merkle_proof(&trace_leaves, i),
                trace_path_after: merkle_proof(&trace_leaves, i + 1),
            }
        })
        .collect();

    PomProof {
        tier,
        trace_root,
        pow_value,
        final_state,
        initial_trace_path: merkle_proof(&trace_leaves, 0),
        final_trace_path: merkle_proof(&trace_leaves, k as usize),
        openings,
    }
}

/// Self-check a built proof before submit (same logic the node runs). Cheap insurance
/// against emitting a block the node will reject.
#[allow(clippy::too_many_arguments)]
pub fn verify_proof(pre_pow_hash: &[u8; 32], nonce: u64, seed: u64, proof: &PomProof, n_chunks: u64, k: u32, t: usize, r_t: &[u8; 32], target: &[u8; 32], h3: bool) -> bool {
    if proof.openings.len() != t {
        return false;
    }
    if pom_pow_value(proof.final_state, pre_pow_hash, h3) != proof.pow_value {
        return false;
    }
    if !le_leq(&proof.pow_value, target) {
        return false;
    }
    if !verify_merkle(trace_leaf(seed), 0, &proof.initial_trace_path, &proof.trace_root) {
        return false;
    }
    if !verify_merkle(trace_leaf(proof.final_state), k as u64, &proof.final_trace_path, &proof.trace_root) {
        return false;
    }
    let chs = challenges(pre_pow_hash, nonce, &proof.trace_root, &proof.pow_value, t, k);
    for (op, &i) in proof.openings.iter().zip(chs.iter()) {
        let i = i as u64;
        if !verify_merkle(trace_leaf(op.state_before), i, &op.trace_path_before, &proof.trace_root) {
            return false;
        }
        let off = op.state_before % n_chunks;
        if !verify_merkle(blake(&op.chunk), off, &op.weight_path, r_t) {
            return false;
        }
        let state_after = transition(op.state_before, &chunk_to_words(&op.chunk));
        if !verify_merkle(trace_leaf(state_after), i + 1, &op.trace_path_after, &proof.trace_root) {
            return false;
        }
    }
    true
}

/// Source of the raw 32 B canonical chunks for `read_chunk`.
enum ChunkSource {
    /// In-RAM chunks for the synthetic test helper (`synth_index`), built without a GGUF.
    /// Test-only: production always uses `Gguf`, so it is compiled out of release builds.
    #[cfg(test)]
    Ram(Vec<u8>),
    /// Chunks read on demand from the GGUF via `pread` — NO host copy (saves ~1x model size of
    /// RAM, ~42 GB for the 70B). `table[j] = (canonical chunk index of tensor j's first chunk,
    /// absolute file byte offset of that chunk)`, ascending by chunk index; `read_chunk`
    /// binary-searches it. The GGUF's on-disk quantized bytes are byte-identical to candle's
    /// `qt.data()` used to build the leaves (`tensor` seeks to the same `tensor_data_offset + offset`).
    Gguf { file: File, table: Vec<(u64, u64)> },
}

/// One checkpoint level stored on disk in the sparse Merkle tree file.
struct StoredLevel {
    level: u32,  // level index in the full tree (0 = leaves, root = total_levels - 1)
    offset: u64, // byte offset within the checkpoint file
    count: u64,  // node count at this level
}

/// Canonical weight index built once at startup from the resident model: the per-chunk
/// blake3 leaves (for Merkle paths), the recomputed tier root `R_T` (sanity-checked against
/// the consensus-pinned value), and a chunk reader. Canonical layout = name-sorted GGUF
/// tensors, `floor(len/32)` 32 B chunks — identical to `pom-rt-builder` and the node.
///
/// The sparse checkpoint Merkle tree lives on disk: only every K-th level is stored
/// (multiples of `CHECKPOINT_INTERVAL`, plus the root). Unstored intermediate levels are
/// recomputed from the GGUF on demand via `merkle_path`. This cuts tree storage from ~2N
/// nodes to ~N/(2^K - 1) nodes (~63× reduction for K=6).
pub struct WeightIndex {
    pub n_chunks: u64,
    pub r_t: [u8; 32],
    /// Raw 32 B chunk reader: GGUF-backed in production, RAM-backed in synthetic tests.
    chunks: ChunkSource,
    /// Sparse checkpoint file: only stored levels are persisted (pread).
    tree_file: File,
    #[allow(dead_code)]
    tree_path: PathBuf,
    /// Stored checkpoint levels (multiples of CHECKPOINT_INTERVAL + root).
    checkpoints: Vec<StoredLevel>,
    /// Full tree depth: levels 0..total_levels-1 where total_levels-1 is the root.
    total_levels: u32,
}

impl Drop for WeightIndex {
    fn drop(&mut self) {
        // Tree is intentionally persistent across restarts (GGUF is immutable).
    }
}

/// Compute checkpoint levels from leaf count alone — purely arithmetic, no I/O.
/// Returns (checkpoints, total_levels). Only stores multiples of CHECKPOINT_INTERVAL
/// plus the root; level 0 is never stored.
fn compute_checkpoint_offsets(n_chunks: u64) -> (Vec<StoredLevel>, u32) {
    let mut checkpoints = Vec::new();
    let mut count = n_chunks;
    let mut off: u64 = 0;
    let mut level: u32 = 0;

    loop {
        // Root (count=1) is always stored; other checkpoints at multiples of K, level > 0.
        let is_checkpoint = (level > 0 && level.is_multiple_of(CHECKPOINT_INTERVAL)) || count == 1;
        if is_checkpoint {
            checkpoints.push(StoredLevel { level, offset: off, count });
        }
        if count == 1 {
            break;
        }
        if is_checkpoint {
            off += count * 32;
        }
        count = count.div_ceil(2);
        level += 1;
    }
    // level is 0-indexed root index; total_levels = root index + 1
    (checkpoints, level + 1)
}

/// Open an existing checkpoint tree file and reconstruct the WeightIndex.
/// Detects legacy full-tree files (size mismatch) and returns an error so the caller can rebuild.
fn open_existing_tree(tree_path: &std::path::Path, gguf_path: &str) -> candle_core::Result<WeightIndex> {
    let mut file = File::open(gguf_path).map_err(candle_core::Error::wrap)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    names.sort();

    // Compute n_chunks (fast — only reads headers, no full tensor data).
    let device = Device::Cpu;
    let mut n_chunks: u64 = 0;
    let mut table: Vec<(u64, u64)> = Vec::with_capacity(names.len());
    for name in &names {
        let file_off = content.tensor_data_offset + content.tensor_infos[name].offset;
        let qt = content.tensor(&mut file, name, &device)?;
        let bytes = qt.data()?;
        let full = bytes.len() / 32;
        if full > 0 {
            table.push((n_chunks, file_off));
        }
        n_chunks += full as u64;
    }
    if n_chunks == 0 {
        return Err(candle_core::Error::Msg("PoM: model produced 0 chunks".into()));
    }
    drop(file);

    let (checkpoints, total_levels) = compute_checkpoint_offsets(n_chunks);
    let expected_size = checkpoints.last().map(|cp| cp.offset + 32).unwrap_or(0);
    let actual_size = std::fs::metadata(tree_path).map_err(candle_core::Error::wrap)?.len();

    // Detect legacy full-tree file: it's ~2× the checkpoint size.
    if actual_size > expected_size + expected_size {
        log::info!(
            "PoM: legacy full-tree pom-tree.bin detected ({} bytes → {} MB); will rebuild as sparse checkpoint (~{} MB for ~{}× savings)",
            actual_size,
            actual_size / 1_048_576,
            expected_size / 1_048_576,
            actual_size / expected_size.max(1),
        );
        return Err(candle_core::Error::Msg(format!(
            "PoM: legacy full-tree detected ({} bytes) — rebuild with sparse checkpoints (expect ~{} bytes)",
            actual_size, expected_size
        )));
    }
    if actual_size != expected_size {
        return Err(candle_core::Error::Msg(format!(
            "PoM: cached tree size mismatch (expected {}, got {}) — delete pom-tree.bin to rebuild",
            expected_size, actual_size
        )));
    }

    let tree_file = OpenOptions::new().read(true).open(tree_path).map_err(candle_core::Error::wrap)?;

    let root_cp = checkpoints.last().unwrap();
    let mut r_t = [0u8; 32];
    read_exact_at(&tree_file, &mut r_t, root_cp.offset).map_err(candle_core::Error::wrap)?;

    let gguf = File::open(gguf_path).map_err(candle_core::Error::wrap)?;
    Ok(WeightIndex {
        n_chunks,
        r_t,
        chunks: ChunkSource::Gguf { file: gguf, table },
        tree_file,
        tree_path: tree_path.to_path_buf(),
        checkpoints,
        total_levels,
    })
}

impl WeightIndex {
    /// Build from a GGUF on disk (CPU dtoh of each tensor). The bytes are candle's exact quantized
    /// bytes — the same the miner serves in VRAM and the builder pinned in `R_T`. The sparse
    /// checkpoint Merkle tree is persisted to `pom-tree.bin` next to the GGUF: only every
    /// K-th level is stored (~N/(2^K-1) nodes vs ~2N for a full tree). On subsequent restarts
    /// the existing tree is reused (GGUF is immutable), avoiding a rebuild.
    pub fn build_from_gguf(path: &str) -> candle_core::Result<Self> {
        let dir = std::path::Path::new(path).parent().unwrap_or_else(|| std::path::Path::new("."));
        let tree_path = dir.join("pom-tree.bin");

        // Clean up old PID-named files left by previous versions.
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("pom-tree-") && name_str.ends_with(".bin") && name_str != "pom-tree.bin" {
                    log::info!("PoM: removing legacy tree file {}", entry.path().display());
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }

        // Reuse existing checkpoint tree if valid.
        if tree_path.exists() {
            match open_existing_tree(&tree_path, path) {
                Ok(idx) => {
                    log::info!("PoM: reusing cached weight index — {} chunks", idx.n_chunks);
                    return Ok(idx);
                }
                Err(e) => {
                    log::warn!("PoM: cached tree invalid ({}), rebuilding…", e);
                    let _ = std::fs::remove_file(&tree_path);
                }
            }
        }

        let device = Device::Cpu;
        let mut file = File::open(path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort(); // canonical order

        // Phase 0: hash leaves from GGUF chunks → write first checkpoint level (level K) to disk.
        // Process in batches of 2^K leaves, building a mini-tree per batch and writing only
        // its root (the level-K node). Uses duplicate-last for the final partial batch.
        let k = CHECKPOINT_INTERVAL;
        let batch_size = 1u64 << k; // 64 for K=6

        let mut writer = BufWriter::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tree_path)
                .map_err(candle_core::Error::wrap)?,
        );

        let mut table: Vec<(u64, u64)> = Vec::with_capacity(names.len());
        let mut n_chunks: u64 = 0;
        let mut batch_buf: Vec<[u8; 32]> = Vec::with_capacity(batch_size as usize);

        for name in &names {
            let file_off = content.tensor_data_offset + content.tensor_infos[name].offset;
            let qt = content.tensor(&mut file, name, &device)?;
            let bytes = qt.data()?;
            let full = bytes.len() / 32;
            if full > 0 {
                table.push((n_chunks, file_off));
            }
            for c in 0..full {
                let chunk = &bytes[c * 32..c * 32 + 32];
                batch_buf.push(blake(chunk));
                n_chunks += 1;
                if batch_buf.len() == batch_size as usize {
                    let level_k_node = fold_levels(&batch_buf, k);
                    writer.write_all(&level_k_node).map_err(candle_core::Error::wrap)?;
                    batch_buf.clear();
                }
            }
        }
        // Final partial batch: fold_levels carries the partial tail the full K levels (duplicate-last).
        // Do NOT pad to batch_size — padding at level 0 changes intermediate hashes.
        if !batch_buf.is_empty() {
            let level_k_node = fold_levels(&batch_buf, k);
            writer.write_all(&level_k_node).map_err(candle_core::Error::wrap)?;
        }

        if n_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM: model produced 0 chunks".into()));
        }

        // Build higher checkpoint levels (2K, 3K, ..., root) from level-K nodes.
        writer.flush().map_err(candle_core::Error::wrap)?;
        drop(writer);
        let (checkpoints, total_levels, r_t) = finalize_checkpoint_upper(&tree_path, n_chunks)?;

        let gguf = File::open(path).map_err(candle_core::Error::wrap)?;
        let tree_file = File::open(&tree_path).map_err(candle_core::Error::wrap)?;
        Ok(WeightIndex {
            n_chunks,
            r_t,
            chunks: ChunkSource::Gguf { file: gguf, table },
            tree_file,
            tree_path,
            checkpoints,
            total_levels,
        })
    }

    /// 32 B chunk at canonical index `off` (panics if out of range — `off < n_chunks`).
    pub fn read_chunk(&self, off: u64) -> [u64; CHUNK_WORDS] {
        chunk_to_words(&self.read_chunk_bytes(off))
    }

    /// Raw 32 B chunk bytes — used for leaf hashing in merkle_path.
    fn read_chunk_bytes(&self, off: u64) -> [u8; 32] {
        let mut arr = [0u8; 32];
        match &self.chunks {
            #[cfg(test)]
            ChunkSource::Ram(data) => {
                let base = (off as usize) * 32;
                arr.copy_from_slice(&data[base..base + 32]);
            }
            ChunkSource::Gguf { file, table } => {
                let j = table.partition_point(|&(start, _)| start <= off) - 1;
                let (start, file_off) = table[j];
                read_exact_at(file, &mut arr, file_off + (off - start) * 32).expect("PoM gguf chunk read");
            }
        }
        arr
    }

    /// Bulk read of canonical chunks `[first, first + out.len()/32)` into `out` (`out.len()` must
    /// be a multiple of 32, the range in bounds). This is the zero-dup GPU upload source: the
    /// Vulkan staging window is filled straight from the GGUF via positional reads, spanning
    /// tensors through the same `(chunk_start, file_offset)` table `read_chunk` uses — so no
    /// packed host copy of the blob is ever built. Positional reads keep it safe to call from
    /// several GPU workers concurrently (multi-GPU installs share one `WeightIndex`).
    pub fn read_chunk_range(&self, first: u64, out: &mut [u8]) -> std::io::Result<()> {
        assert!(out.len() % 32 == 0, "read_chunk_range: length not chunk-aligned");
        let n = out.len() as u64 / 32;
        assert!(first + n <= self.n_chunks, "read_chunk_range: range past n_chunks");
        match &self.chunks {
            #[cfg(test)]
            ChunkSource::Ram(data) => {
                let base = (first as usize) * 32;
                out.copy_from_slice(&data[base..base + out.len()]);
                Ok(())
            }
            ChunkSource::Gguf { file, table } => {
                let mut cur = first; // next canonical chunk to read
                let mut filled = 0usize; // bytes of `out` already filled
                let mut j = table.partition_point(|&(start, _)| start <= cur) - 1;
                while filled < out.len() {
                    let (t_start, t_file_off) = table[j];
                    // This tensor's span ends where the next table entry starts (or at n_chunks).
                    let t_end = table.get(j + 1).map(|&(s, _)| s).unwrap_or(self.n_chunks);
                    let take = ((t_end - cur) * 32).min((out.len() - filled) as u64) as usize;
                    read_exact_at(
                        file,
                        &mut out[filled..filled + take],
                        t_file_off + (cur - t_start) * 32,
                    )?;
                    filled += take;
                    cur += take as u64 / 32;
                    j += 1;
                }
                Ok(())
            }
        }
    }

    /// Find the stored checkpoint at `level`, panics if not found.
    fn find_checkpoint(&self, level: u32) -> &StoredLevel {
        self.checkpoints.iter().find(|cp| cp.level == level).expect("PoM: checkpoint not found")
    }

    /// Number of nodes at `level` in the full tree (0-indexed, level 0 = leaves).
    fn count_at_level(&self, level: u32) -> u64 {
        let mut count = self.n_chunks;
        for _ in 0..level {
            count = count.div_ceil(2);
        }
        count
    }

    /// Compute the hash of the subtree whose root sits `log2(span)` levels above `src_level`, rooted
    /// at source-level index `start` and covering `span` source nodes. `src_level`: 0 = GGUF chunks,
    /// >0 = stored checkpoint level. `span` is always a power of two (= 2^(target_level - src_level)).
    ///
    /// Reads ONLY the in-range source nodes (a partial subtree exists only at the right edge) and
    /// folds them EXACTLY `log2(span)` levels with per-level duplicate-last (`fold_levels`). Padding
    /// the source by clamping the last valid index — the old approach — was WRONG: it injects extra
    /// duplicated leaves that fold into a different node than the dense tree's `hash(x, x)` carry of a
    /// lone INNER node, so reconstructed siblings (and thus proofs) mismatched at right-edge offsets.
    fn compute_subtree_hash(&self, start: u64, span: u64, src_level: u32) -> [u8; 32] {
        debug_assert!(span.is_power_of_two());
        let rounds = span.trailing_zeros();
        let source_count = if src_level == 0 { self.n_chunks } else { self.find_checkpoint(src_level).count };
        if start >= source_count {
            return [0u8; 32]; // guard: a real sibling subtree always starts in range
        }
        let end = (start + span).min(source_count);
        let nodes: Vec<[u8; 32]> = if src_level == 0 {
            // Source is GGUF: read the in-range chunks via pread and hash each into a leaf.
            (start..end).map(|i| blake(&self.read_chunk_bytes(i))).collect()
        } else {
            // Source is a stored checkpoint: read the in-range nodes from file.
            let cp = self.find_checkpoint(src_level);
            (start..end)
                .map(|i| {
                    let mut buf = [0u8; 32];
                    read_exact_at(&self.tree_file, &mut buf, cp.offset + i * 32).expect("PoM checkpoint read subtree");
                    buf
                })
                .collect()
        };
        fold_levels(&nodes, rounds)
    }

    /// Inclusion path for chunk index `off`, reading stored siblings from the checkpoint file
    /// and computing unstored intermediate levels on-the-fly from the GGUF.
    /// Byte-identical to the full-tree `merkle_path`: an out-of-range sibling is the node itself.
    pub fn merkle_path(&self, off: u64) -> Vec<[u8; 32]> {
        let total_levels = self.total_levels;
        let mut path = Vec::with_capacity(total_levels as usize);
        let mut idx: u64 = off;

        for level in 0..total_levels {
            if level == total_levels - 1 {
                break; // root has no sibling
            }

            let sib_idx = idx ^ 1;
            let is_stored = level > 0 && (level.is_multiple_of(CHECKPOINT_INTERVAL) || level == total_levels - 1);

            let node = if is_stored {
                // Read sibling directly from checkpoint file.
                let cp = self.find_checkpoint(level);
                let real_idx = if sib_idx < cp.count { sib_idx } else { idx };
                let mut buf = [0u8; 32];
                read_exact_at(&self.tree_file, &mut buf, cp.offset + real_idx * 32).expect("PoM checkpoint read");
                buf
            } else {
                // Compute sibling from nearest source below.
                // If sibling index is out of range, duplicate-last: use the node itself as sibling.
                let node_count = self.count_at_level(level);
                let real_sib_idx = if sib_idx < node_count { sib_idx } else { idx };
                let src_level = (level / CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL;
                let span = 1u64 << (level - src_level);
                self.compute_subtree_hash(real_sib_idx * span, span, src_level)
            };

            path.push(node);
            idx >>= 1;
        }
        path
    }
}

/// Reduce a slice of leaves straight to the single canonical root (duplicate-last each level).
/// Applied to ALL leaves at once this is the dense reference root; it is NOT safe for batched
/// sub-folds (it stops at one node, dropping the remaining `hash(x,x)` carries — the e1811a0 bug),
/// so the build/path use `fold_levels` instead. Retained as the independent dense oracle in tests.
#[cfg(test)]
#[inline]
fn merkle_root_mini(leaves: &[[u8; 32]]) -> [u8; 32] {
    debug_assert!(!leaves.is_empty());
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            next.push(hash_pair(&level[i], &r));
            i += 2;
        }
        level = next;
    }
    level[0]
}

/// Reduce `batch` by EXACTLY `rounds` canonical levels — duplicate-last each round, AND keep
/// carrying a lone node via `hash(x, x)` once the batch collapses to one node before `rounds` is
/// reached. For a full `2^rounds` batch this equals `merkle_root_mini`; for a short tail it carries
/// the remaining levels, matching the dense `merkle_root` the node pins in `POM_TIERS`.
///
/// This is the fix for the sparse-build `R_T` bug: `merkle_root_mini` stops at `len == 1`, so a
/// partial batch of `m ≤ 2^(rounds-1)` nodes lands fewer than `rounds` levels up and drops the
/// remaining `hash(x, x)` carries — yielding a wrong checkpoint node (hence wrong `R_T`) for every
/// non-power-of-two `N`. A batch fold must always land exactly `rounds` levels up.
#[inline]
fn fold_levels(batch: &[[u8; 32]], rounds: u32) -> [u8; 32] {
    debug_assert!(!batch.is_empty());
    let mut level = batch.to_vec();
    for _ in 0..rounds {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            next.push(hash_pair(&level[i], &r));
            i += 2;
        }
        level = next;
    }
    level[0]
}

/// Build higher checkpoint levels from the already-written level-K nodes in the tree file.
/// Reads level-K from the file, writes each higher checkpoint level (2K, 3K, ..., root),
/// and returns the checkpoint layout + R_T.
fn finalize_checkpoint_upper(
    tree_path: &std::path::Path,
    n_chunks: u64,
) -> candle_core::Result<(Vec<StoredLevel>, u32, [u8; 32])> {
    let (checkpoints, total_levels) = compute_checkpoint_offsets(n_chunks);
    let mut file_for_read = File::open(tree_path).map_err(candle_core::Error::wrap)?;
    let mut prev_offset: u64 = checkpoints[0].offset;
    let mut prev_count = checkpoints[0].count;
    let mut prev_level = checkpoints[0].level;

    // Open for appending higher levels
    let mut writer = OpenOptions::new().read(true).write(true).open(tree_path).map_err(candle_core::Error::wrap)?;
    writer.seek(SeekFrom::End(0)).map_err(candle_core::Error::wrap)?;
    let mut buf_writer = BufWriter::new(writer);

    for cp in &checkpoints[1..] {
        // Fold the previous stored level up to this checkpoint's level. A regular checkpoint sits
        // CHECKPOINT_INTERVAL levels above the previous; the final (root) fold may span fewer. Batch
        // the previous level by exactly 2^rounds and fold each batch EXACTLY `rounds` levels, so a
        // partial tail carries via hash(x,x) like the dense tree. Node count per level is
        // ceil(prev_count / 2^rounds) == cp.count (ceil(ceil(n/2)/2)…=ceil(n/2^rounds)), so offsets line up.
        let rounds = cp.level - prev_level;
        let batch_size = 1u64 << rounds;
        let mut batch: Vec<[u8; 32]> = Vec::with_capacity(batch_size as usize);
        let mut read_idx: u64 = 0;

        while read_idx < prev_count {
            let take = batch_size.min(prev_count - read_idx);
            batch.clear();
            for i in 0..take {
                let index = read_idx + i;
                let mut node = [0u8; 32];
                read_exact_at(&file_for_read, &mut node, prev_offset + index * 32).map_err(candle_core::Error::wrap)?;
                batch.push(node);
            }
            let parent_node = fold_levels(&batch, rounds);
            buf_writer.write_all(&parent_node).map_err(candle_core::Error::wrap)?;
            read_idx += take;
        }

        buf_writer.flush().map_err(candle_core::Error::wrap)?;
        file_for_read = File::open(tree_path).map_err(candle_core::Error::wrap)?;
        prev_offset = cp.offset;
        prev_count = cp.count;
        prev_level = cp.level;
    }

    // Read R_T from the last checkpoint (root)
    let root_cp = checkpoints.last().unwrap();
    let mut r_t = [0u8; 32];
    read_exact_at(&file_for_read, &mut r_t, root_cp.offset).map_err(candle_core::Error::wrap)?;

    Ok((checkpoints, total_levels, r_t))
}

/// PoM possession activation DAA score — MUST match the node's `pom_activation`.
/// `u64::MAX` = never (dormant): mining stays on legacy kHeavyHash, no proof produced.
///
/// Testnet: `5_000` = mid-chain activation, to observe the kHeavyHash→PoM transition (incl.
/// the difficulty drift: PoM ~30x slower → blocks slow at the cutover, then the DAA window
/// recovers). Mainnet will need a difficulty reset at H.
/// Mainnet: 37_780_000 (2026-06-26 18:00 UTC) — MUST equal the node's
/// MAINNET_PARAMS.pom_activation = new(37_780_000).
pub const POM_ACTIVATION_DAA: u64 = 37_780_000;

/// H3 (PoM block-level hardfork) activation DAA score. At/after this score the block header
/// commits to the winning walk's `final_state` (`pomFinalState`): the node hashes it into the
/// block hash, re-checks `pom_pow_value(final_state, pre_pow_hash) <= target` header-only, and
/// derives the block level from it again (bounded pruning proof, from-scratch IBD). The miner
/// MUST fill it on submit exactly like the nonce — a post-H3 block without it is rejected
/// (`InvalidPoW` / `PomFinalStateMismatch`). The pre-PoW hash is NOT affected (the walk seed
/// derives from it, so the field lives only in the final block hash). For stratum shares no
/// wire change is needed: the borsh proof already carries `final_state`.
///
/// H3 also salts the pph words feeding both PoM folds from this score (POM_H3_PPH_SALT).
///
/// Mainnet: 43_450_000 — picked 2026-07-05 08:49 UTC (tip 43,117,871) targeting activation
/// ≈ 2026-07-05 18:00 UTC. MUST equal the node's MAINNET_PARAMS.pom_level_activation
/// = new(43_450_000).
pub const POM_LEVEL_ACTIVATION_DAA: u64 = 43_450_000;

/// The resident tier weight index + tier id, installed once at startup when PoM is enabled.
static POM_INDEX: OnceLock<(WeightIndex, u8)> = OnceLock::new();

/// Install the possession index (built from the resident model) and its tier. Call once.
pub fn set_index(index: WeightIndex, tier: u8) {
    let _ = POM_INDEX.set((index, tier));
}

/// The active possession index + tier, if installed.
pub fn active_index() -> Option<&'static (WeightIndex, u8)> {
    POM_INDEX.get()
}

/// Guards the one-time possession-index build. Every PoM GPU worker races into activation at the
/// same DAA, but `WeightIndex::build_from_gguf` writes a single Merkle tree (`pom-tree.bin`) next to
/// the GGUF, so two threads building concurrently clobber the same file — the "failed to fill whole
/// buffer" startup failure seen on multi-GPU rigs. Harmless on a single worker; only bites once >1
/// PoM worker exists (multi-GPU, or a GPU + a future CPU worker).
static INDEX_BUILD_LOCK: Mutex<()> = Mutex::new(());

/// Build + install the possession index exactly once across racing workers. Returns true when the
/// index is ready (already installed, or built here). `build` runs only on the single thread that
/// wins the lock with the index still absent; the others wait, then observe it installed.
///
/// Double-checked: the cheap `active_index()` read short-circuits the common case (index already
/// built) without taking the lock; the second check under the lock closes the race window.
pub fn get_or_build_index<F>(tier: u8, build: F) -> bool
where
    F: FnOnce() -> candle_core::Result<WeightIndex>,
{
    if active_index().is_some() {
        return true;
    }
    // A poisoned lock just means a prior builder panicked; the guard's data is `()`, so recover it
    // and proceed — the double-check below still keeps the build single.
    let _guard = INDEX_BUILD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    if active_index().is_some() {
        return true;
    }
    log::info!("PoM: building possession index (first PoM activation) — this can take a while…");
    match build() {
        Ok(idx) => {
            log::info!("PoM: weight index ready — N={} chunks", idx.n_chunks);
            set_index(idx, tier);
            true
        }
        Err(e) => {
            log::error!("PoM: index build failed: {e}");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_chunk(off: u64) -> [u64; CHUNK_WORDS] {
        let mut c = [0u64; CHUNK_WORDS];
        for (j, w) in c.iter_mut().enumerate() {
            *w = mix64(off.wrapping_mul(CHUNK_WORDS as u64) + j as u64 + 1);
        }
        c
    }

    // Synthetic WeightIndex (no GGUF) — exercises the real read_chunk + O(log N) merkle_path
    // with the sparse checkpoint tree (same structure as production).
    fn synth_index(n: u64) -> WeightIndex {
        use std::sync::atomic::{AtomicU64, Ordering as O};
        static UNIQ: AtomicU64 = AtomicU64::new(0);
        let uid = UNIQ.fetch_add(1, O::Relaxed);
        let tree_path = std::env::temp_dir().join(format!("keryx-pom-synth-{}-{}.bin", std::process::id(), uid));
        let _ = std::fs::remove_file(&tree_path);

        let k = CHECKPOINT_INTERVAL;
        let batch_size = 1u64 << k; // 64 for K=6

        // Write level-K nodes from batches of synth chunks.
        let mut writer = BufWriter::new(
            OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&tree_path).unwrap(),
        );
        let mut data = Vec::new();
        let mut batch: Vec<[u8; 32]> = Vec::with_capacity(batch_size as usize);

        for o in 0..n {
            let b = words_to_bytes(&synth_chunk(o));
            data.extend_from_slice(&b);
            batch.push(blake(&b));
            if batch.len() == batch_size as usize {
                let level_k_node = fold_levels(&batch, k);
                writer.write_all(&level_k_node).unwrap();
                batch.clear();
            }
        }
        // Final partial batch: fold_levels carries the partial tail the full K levels (duplicate-last).
        if !batch.is_empty() {
            writer.write_all(&fold_levels(&batch, k)).unwrap();
        }

        writer.flush().unwrap();
        drop(writer);

        // Build higher checkpoints
        let (checkpoints, total_levels, r_t) = finalize_checkpoint_upper(&tree_path, n).unwrap();

        let tree_file = File::open(&tree_path).unwrap();
        WeightIndex {
            n_chunks: n,
            r_t,
            chunks: ChunkSource::Ram(data),
            tree_file,
            tree_path,
            checkpoints,
            total_levels,
        }
    }

    /// `read_chunk_range` (the zero-dup GPU upload source) must return byte-identical data to the
    /// per-chunk `read_chunk_bytes` path across tensor-span boundaries. Uses a synthetic GGUF-like
    /// file: three "tensors" at padded (non-contiguous) file offsets, one with a ragged sub-chunk
    /// tail that the canonical layout drops — the exact shapes the table walk must handle.
    #[test]
    fn read_chunk_range_matches_per_chunk_reads() {
        let path = std::env::temp_dir().join(format!("keryx-pom-range-{}.bin", std::process::id()));
        let mut blob = Vec::new();
        blob.extend_from_slice(&[0xAAu8; 100]); // fake header
        let mut table: Vec<(u64, u64)> = Vec::new();
        let mut n_chunks = 0u64;
        // (full chunks, ragged tail bytes, pad bytes before the tensor)
        for (full, ragged, pad) in [(5u64, 0usize, 3usize), (1, 20, 7), (3, 5, 1)] {
            blob.extend_from_slice(&vec![0xEEu8; pad]);
            table.push((n_chunks, blob.len() as u64));
            for c in 0..full {
                blob.extend_from_slice(&words_to_bytes(&synth_chunk(n_chunks + c)));
            }
            blob.extend_from_slice(&vec![0x55u8; ragged]); // dropped from the canonical layout
            n_chunks += full;
        }
        std::fs::write(&path, &blob).unwrap();

        let (checkpoints, total_levels) = compute_checkpoint_offsets(n_chunks);
        let idx = WeightIndex {
            n_chunks,
            r_t: [0u8; 32],
            chunks: ChunkSource::Gguf { file: File::open(&path).unwrap(), table },
            tree_file: File::open(&path).unwrap(), // unused by chunk reads
            tree_path: path.clone(),
            checkpoints,
            total_levels,
        };

        for first in 0..n_chunks {
            for n in 1..=(n_chunks - first) {
                let mut got = vec![0u8; (n * 32) as usize];
                idx.read_chunk_range(first, &mut got).unwrap();
                for c in 0..n {
                    assert_eq!(
                        &got[(c * 32) as usize..(c * 32 + 32) as usize],
                        &idx.read_chunk_bytes(first + c),
                        "range [{first}, +{n}) mismatch at chunk {}",
                        first + c
                    );
                }
            }
        }
        drop(idx);
        let _ = std::fs::remove_file(&path);
    }

    /// Regression for the sparse-checkpoint R_T bug (commit e1811a0): the checkpoint-built root MUST
    /// equal the dense canonical root for every N — including non-power-of-two sizes whose short leaf
    /// tail OR intermediate-fold tail used to drop the `hash(x, x)` carries (`merkle_root_mini` stopped
    /// at one node). The dense reference is `merkle_root_mini` over ALL leaves at once (it reduces
    /// straight to the true root, un-batched), which is exactly what `pom-rt-builder` pins in
    /// `POM_TIERS`. Includes the report's known-broken sizes (2000, 4968, 12345, 100000).
    #[test]
    fn sparse_build_root_matches_dense_root() {
        for n in [64u64, 65, 100, 1000, 2000, 4096, 4968, 12345, 65536, 100000, 131072] {
            let leaves: Vec<[u8; 32]> = (0..n).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();
            let dense = merkle_root_mini(&leaves);
            let idx = synth_index(n);
            assert_eq!(idx.r_t, dense, "sparse-built R_T != dense root for N={n}");
            let _ = std::fs::remove_file(&idx.tree_path);
        }
    }

    /// End-to-end check against a node-pinned root: build the sparse index from a real GGUF and
    /// assert its R_T equals the value `pom-rt-builder` pinned in the node's `POM_TIERS`. This closes
    /// the loop the synthetic test can't (real chunking: name-sorted tensors, floor(len/32), the exact
    /// candle quantized bytes). `#[ignore]`d — needs the GGUF on disk; run with:
    ///   KERYX_POM_TEST_GGUF=/path/model.gguf KERYX_POM_TEST_ROOT=<hex> \
    ///     cargo test --release weight_index_matches_pinned_root -- --ignored --nocapture
    #[test]
    #[ignore]
    fn weight_index_matches_pinned_root() {
        let path = std::env::var("KERYX_POM_TEST_GGUF").expect("set KERYX_POM_TEST_GGUF=/path/model.gguf");
        let expected = std::env::var("KERYX_POM_TEST_ROOT").expect("set KERYX_POM_TEST_ROOT=<hex>").to_lowercase();
        // Force a fresh build (don't reuse a possibly-stale cached tree from an older binary).
        let dir = std::path::Path::new(&path).parent().unwrap();
        let _ = std::fs::remove_file(dir.join("pom-tree.bin"));
        let idx = WeightIndex::build_from_gguf(&path).unwrap();
        let got: String = idx.r_t.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(got, expected, "R_T mismatch vs pinned root for {path}");
    }

    /// GGUF-backed `read_chunk`: lay the canonical chunks across 3 "tensors" with header + inter-
    /// tensor padding (so file offset != off*32), build the per-tensor offset table, and assert
    /// `read_chunk` (pread) returns the exact canonical chunks AND that a proof verifies — same as
    /// the RAM path, with no host copy of the weights.
    #[test]
    fn gguf_chunk_source_reads_match_and_proof_verifies() {
        let n = 1000u64;
        let uid = std::process::id();
        let gguf_path = std::env::temp_dir().join(format!("keryx-pom-fakegguf-{uid}.bin"));
        let _ = std::fs::remove_file(&gguf_path);
        let mut f = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&gguf_path).unwrap();

        // 3 tensors at chunk-start boundaries, with padding so file_off is not simply off*32.
        let splits = [0u64, 400, 750, n];
        let mut table: Vec<(u64, u64)> = Vec::new();
        let mut pos: u64 = 17; // header padding
        f.seek(SeekFrom::Start(pos)).unwrap();
        for w in splits.windows(2) {
            table.push((w[0], pos));
            for o in w[0]..w[1] {
                f.write_all(&words_to_bytes(&synth_chunk(o))).unwrap();
                pos += 32;
            }
            pos += 13; // inter-tensor padding gap
            f.seek(SeekFrom::Start(pos)).unwrap();
        }
        f.flush().unwrap();
        let file = File::open(&gguf_path).unwrap();

        // Build the sparse checkpoint tree over the canonical synth chunks, with the GGUF chunk source.
        let tree_path = std::env::temp_dir().join(format!("keryx-pom-fakegguf-tree-{uid}.bin"));
        let _ = std::fs::remove_file(&tree_path);

        let k = CHECKPOINT_INTERVAL;
        let batch_size = 1u64 << k;
        let mut writer = BufWriter::new(
            OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&tree_path).unwrap(),
        );
        let mut batch: Vec<[u8; 32]> = Vec::with_capacity(batch_size as usize);
        for o in 0..n {
            batch.push(blake(&words_to_bytes(&synth_chunk(o))));
            if batch.len() == batch_size as usize {
                writer.write_all(&fold_levels(&batch, k)).unwrap();
                batch.clear();
            }
        }
        if !batch.is_empty() {
            writer.write_all(&fold_levels(&batch, k)).unwrap();
        }
        writer.flush().unwrap();
        drop(writer);

        let (checkpoints, total_levels, r_t) = finalize_checkpoint_upper(&tree_path, n).unwrap();
        let tree_file = File::open(&tree_path).unwrap();
        let idx = WeightIndex {
            n_chunks: n,
            r_t,
            chunks: ChunkSource::Gguf { file, table },
            tree_file,
            tree_path,
            checkpoints,
            total_levels,
        };

        // Every chunk read by pread matches the canonical chunk, across all segments + padding.
        for o in 0..n {
            assert_eq!(idx.read_chunk(o), synth_chunk(o), "chunk {o}");
        }
        // A proof built from the GGUF source verifies against R_T (target 0xff..ff = first nonce wins).
        let (k, t) = (POM_WALK_STEPS, POM_OPENINGS);
        let pph = [7u8; 32];
        let target = [0xffu8; 32];
        let (nonce, proof) = mine_pom(&idx, 2, &pph, 123, &target, k, t, 0, 1, false).expect("max target → win");
        let seed = pom_block_seed(&pph, 123, nonce, false);
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &target, false));

        let _ = std::fs::remove_file(&gguf_path);
    }

    /// Real-GGUF byte-identity: build the index from a downloaded model and prove that chunks
    /// read by `pread` (GGUF) verify against the model's own `R_T` (whose leaves were hashed from
    /// candle's `qt.data()`). Confirms `pread(tensor_data_offset + offset)` == `qt.data()` for real
    /// quant types. Ignored (needs the GGUF); run: `cargo test -p keryx-miner -- --ignored gguf_real`.
    #[test]
    #[ignore]
    fn gguf_real_model_read_chunk_byte_identical() {
        let path = "/home/slash/KERYX-KRX/claude/Outils PoM/keryx-miner-test CPU-Llama3-70B/target/release/models/Gemma-3-4B/model.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("skip: GGUF not found at {path}");
            return;
        }
        let idx = WeightIndex::build_from_gguf(path).expect("build index from real GGUF");
        eprintln!("real model index: N={} chunks", idx.n_chunks);
        let (k, t) = (POM_WALK_STEPS, POM_OPENINGS);
        let pph = [3u8; 32];
        let target = [0xffu8; 32]; // max → the first nonce wins, so 1 nonce suffices
        let (nonce, proof) = mine_pom(&idx, 0, &pph, 99, &target, k, t, 0, 1, false).expect("max target → win");
        let seed = pom_block_seed(&pph, 99, nonce, false);
        assert!(
            verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &target, false),
            "GGUF-pread chunks must verify against the model's R_T (byte-identity broken otherwise)"
        );
    }

    #[test]
    fn weight_index_root_matches_standalone() {
        // The prebuilt-tree root equals the standalone merkle_root over the same leaves.
        let n = 1000u64;
        let idx = synth_index(n);
        let leaves: Vec<[u8; 32]> = (0..n).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();
        assert_eq!(idx.r_t, merkle_root(&leaves));
    }

    #[test]
    fn merkle_path_matches_in_memory_proof() {
        // The checkpoint merkle_path must be byte-identical to the in-memory merkle_proof.
        let n = 4096;
        let idx = synth_index(n);
        let leaves: Vec<[u8; 32]> = (0..n).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();

        for off in [0, 1, n / 2, n - 2, n - 1] {
            let checkpoint_path = idx.merkle_path(off);
            let memory_path = merkle_proof(&leaves, off as usize);
            assert_eq!(checkpoint_path.len(), memory_path.len(), "path length mismatch at off={off}");
            for (i, (cp, mp)) in checkpoint_path.iter().zip(memory_path.iter()).enumerate() {
                assert_eq!(cp, mp, "path mismatch at off={off}, level={i}");
            }
        }
    }

    /// Regression for the sparse-checkpoint PATH bug: every offset's reconstructed `merkle_path`
    /// must be byte-identical to the dense `merkle_proof` for non-power-of-two N. The old
    /// `compute_subtree_hash` clamped the source to fill the span and mismatched the dense
    /// duplicate-last carry at right-edge offsets. Exhaustive over the report's broken sizes; the
    /// pre-existing test only used n=4096 (pow2) and missed it entirely.
    #[test]
    fn merkle_path_matches_dense_proof_nonpow2() {
        for n in [65u64, 100, 1000, 2000, 4968, 12345] {
            let idx = synth_index(n);
            let leaves: Vec<[u8; 32]> = (0..n).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();
            for off in 0..n {
                assert_eq!(idx.merkle_path(off), merkle_proof(&leaves, off as usize), "path mismatch N={n} off={off}");
            }
            let _ = std::fs::remove_file(&idx.tree_path);
        }
        // Larger N: strided sweep + dense right edge (where the duplicate-last carry bites hardest).
        let n = 100_000u64;
        let idx = synth_index(n);
        let leaves: Vec<[u8; 32]> = (0..n).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();
        for off in (0..n).step_by(257).chain(n - 300..n) {
            assert_eq!(idx.merkle_path(off), merkle_proof(&leaves, off as usize), "path mismatch N={n} off={off}");
        }
        let _ = std::fs::remove_file(&idx.tree_path);
    }

    #[test]
    fn build_then_self_verify() {
        let (k, t) = (256u32, 32usize);
        let idx = synth_index(4096);
        let pph = blake(b"pph");
        let nonce = 0xabc;
        let seed = pom_block_seed(&pph, 111, nonce, false);

        let proof =
            build_proof(2, &pph, nonce, seed, idx.n_chunks, k, t, |o| idx.read_chunk(o), |o| idx.merkle_path(o), false);
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &[0xff; 32], false));
        // borsh wire-format round-trips (same encoding the node decodes).
        let bytes = borsh::to_vec(&proof).unwrap();
        let back: PomProof = borsh::from_slice(&bytes).unwrap();
        assert!(verify_proof(&pph, nonce, seed, &back, idx.n_chunks, k, t, &idx.r_t, &[0xff; 32], false));
        assert_eq!(back.tier, 2);
    }

    #[test]
    fn h3_salt_changes_seed_and_pow_and_roundtrips() {
        let (k, t) = (256u32, 32usize);
        let idx = synth_index(4096);
        let pph = blake(b"h3-pph");
        let nonce = 0xdef;
        // The salted era must diverge from the raw era on both folds.
        let seed_pre = pom_block_seed(&pph, 42, nonce, false);
        let seed_h3 = pom_block_seed(&pph, 42, nonce, true);
        assert_ne!(seed_pre, seed_h3, "H3 salt must change the walk seed");
        assert_ne!(pom_pow_value(7, &pph, false), pom_pow_value(7, &pph, true), "H3 salt must change the pow value");
        // A proof built in the H3 era verifies in the H3 era and fails in the pre-H3 era.
        let proof =
            build_proof(1, &pph, nonce, seed_h3, idx.n_chunks, k, t, |o| idx.read_chunk(o), |o| idx.merkle_path(o), true);
        assert!(verify_proof(&pph, nonce, seed_h3, &proof, idx.n_chunks, k, t, &idx.r_t, &[0xff; 32], true));
        assert!(
            !verify_proof(&pph, nonce, seed_h3, &proof, idx.n_chunks, k, t, &idx.r_t, &[0xff; 32], false),
            "an H3-era proof must not verify under the unsalted fold"
        );
    }

    #[test]
    fn wrong_target_or_root_fails() {
        let (k, t) = (256u32, 32usize);
        let idx = synth_index(4096);
        let pph = blake(b"pph2");
        let nonce = 7;
        let seed = pom_block_seed(&pph, 1, nonce, false);
        let proof =
            build_proof(0, &pph, nonce, seed, idx.n_chunks, k, t, |o| idx.read_chunk(o), |o| idx.merkle_path(o), false);
        assert!(
            !verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &[0u8; 32], false),
            "zero target must fail"
        );
        assert!(
            !verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &blake(b"wrong"), &[0xff; 32], false),
            "wrong R_T must fail"
        );
    }

    #[test]
    fn cpu_mine_finds_nonce_and_proof_verifies() {
        let (k, t) = (128u32, 32usize);
        let idx = synth_index(4096);
        let pph = blake(b"mine-pph");
        let ts = 555;
        // Target requiring pow_value MSB <= 0x10 (~6.6% of nonces) — found within a few tries.
        let mut target = [0xffu8; 32];
        target[31] = 0x10;
        let (nonce, proof) = mine_pom(&idx, 1, &pph, ts, &target, k, t, 0, 100_000, false).expect("mine a nonce");
        let seed = pom_block_seed(&pph, ts, nonce, false);
        // The proof verifies against the same target the node would use.
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, k, t, &idx.r_t, &target, false));
        assert_eq!(proof.tier, 1);
    }

    // Validates the canonical layout against the consensus-pinned R_T. Needs the Gemma GGUF.
    // Run: cargo test --lib pom -- --ignored --nocapture
    #[test]
    #[ignore = "needs Gemma-3-4B GGUF on disk"]
    fn weight_index_matches_pinned_gemma() {
        let path = "/home/slash/KERYX-KRX/claude/keryx-miner/target/release/models/Gemma-3-4B/model.gguf";
        let idx = WeightIndex::build_from_gguf(path).expect("build index");
        assert_eq!(idx.n_chunks, 77_604_776, "chunk count must match pinned GEMMA_3_4B_POM_CHUNKS");
        let pinned: [u8; 32] = [
            0x84, 0x6c, 0xaa, 0x40, 0x0c, 0xf0, 0x14, 0x13, 0x21, 0x18, 0x49, 0x5d, 0x22, 0xe4, 0xbf, 0xa2, 0x42, 0x45,
            0x4e, 0xac, 0x0d, 0x83, 0x5c, 0x3f, 0x8e, 0x63, 0x47, 0xd0, 0x13, 0x9d, 0x1b, 0x7e,
        ];
        assert_eq!(idx.r_t, pinned, "miner R_T must equal node-pinned GEMMA_3_4B_POM_ROOT");

        // A real proof over the real model self-verifies against the pinned R_T.
        let pph = blake(b"gemma-pph");
        let nonce = 1234;
        let seed = pom_block_seed(&pph, 99, nonce, false);
        let proof =
            build_proof(0, &pph, nonce, seed, idx.n_chunks, 256, 32, |o| idx.read_chunk(o), |o| idx.merkle_path(o), false);
        assert!(verify_proof(&pph, nonce, seed, &proof, idx.n_chunks, 256, 32, &idx.r_t, &[0xff; 32], false));
    }
}
