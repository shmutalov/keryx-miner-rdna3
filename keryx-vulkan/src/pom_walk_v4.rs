//! PoM v4 (D=32 re-walk) GPU grind — host side of `shaders/pom_walk_v4.comp`. Finds the lowest
//! nonce in a batch whose v4 walk + H3 pow fold meets the target. The walk is byte-identical to
//! `src/pom_v4.rs` / the node; the miner still re-walks every winner on the host before submit.
//!
//! Chunk addressing is a segment table — `prefix` (cumulative canonical-chunk starts, T+1
//! entries) + `addrs` (one device address per segment) — so the same kernel serves the miner's
//! streamed blob (segments = its power-of-two shards) and the zero-dup walk over the inference
//! engine's tensors (segments = tensors). A host-built bucket LUT turns the per-chunk segment
//! search into a short forward scan.

use crate::{GpuBuffer, Kernel, Vk};
use std::io::Cursor;

/// SPIR-V for the v4 walk, compiled from `shaders/pom_walk_v4.comp` by build.rs.
const POM_WALK_V4_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_walk_v4.spv"));

/// Nonces per workgroup — MUST match `NPW` in pom_walk_v4.comp (local_size = 32 x NPW).
pub const V4_NPW: u32 = 4;

/// Canonical 32 B chunks per 1 KB tile (`pom_v4::POM_V4_TILE_CHUNKS`).
pub const V4_TILE_CHUNKS: u64 = 32;

/// Max nonces per dispatch. A v4 nonce is ~256 dependent 1 KB reads plus a 32x32 int8 matmul per
/// step, so this keeps one dispatch to tens of ms — far inside the Windows TDR watchdog (~2 s) —
/// while staying well above per-submit overhead.
const V4_MAX_DISPATCH_NONCES: u32 = 1 << 15;

/// Bucket LUT ceiling (u32 entries): 4 MiB covers the largest tier (~0.93 G chunks) with
/// 1024-chunk buckets.
const V4_LUT_MAX_ENTRIES: u64 = 1 << 20;

const NO_WINNER: u32 = 0xFFFF_FFFF;

/// Push-constant block — layout MUST match `Push` in pom_walk_v4.comp (std430: twelve u64 at
/// 0..96, four u32 at 96..112; 112 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct V4Push {
    p: [u64; 4],
    s: [u64; 4],
    timestamp: u64,
    n_tiles: u64,
    inv_n: u64,
    start_nonce: u64,
    batch: u32,
    t_count: u32,
    lut_sh: u32,
    seed_h10: u32,
}

/// Binding-0 dispatch I/O block — MUST match `Io` in pom_walk_v4.comp (240 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct V4Io {
    offset: u32,
    _pad: u32,
    t: [u64; 4],
    h10: [u64; 25],
}

/// Era inputs of one v4 grind, prepared by the miner (`pom_gpu`) from the template.
#[derive(Clone, Copy)]
pub struct V4Job {
    /// pre_pow_hash words for the POW fold, H3-salted (`pom::pph_words_for_era(.., true)`).
    pub pow_words: [u64; 4],
    /// pre_pow_hash words for the pre-H10 v4 SEED fold (`pom::pph_words_v4`); ignored under H10.
    pub seed_words: [u64; 4],
    /// Header timestamp (pre-H10 seed fold only — the H10 sponge already absorbed it).
    pub timestamp: u64,
    /// H10 sponge state (`pom::pom_seed_h10_state`); `Some` selects the H10 keccak seed.
    pub h10_state: Option<[u64; 25]>,
    /// 256-bit little-endian target.
    pub target_le: [u8; 32],
}

/// The v4 kernel and its VRAM-resident segment tables. Owned by a resident walker
/// ([`PomWalkGpu`](crate::pom_walk::PomWalkGpu) / [`PomWalkShared`](crate::pom_walk::PomWalkShared))
/// that also owns the `Vk`; released through [`destroy`](Self::destroy) before that `Vk` drops.
pub struct PomWalkV4 {
    kernel: Kernel,
    io: GpuBuffer,
    prefix: GpuBuffer,
    addrs: GpuBuffer,
    lut: GpuBuffer,
    lut_sh: u32,
    t_count: u32,
    n_tiles: u64,
    inv_n: u64,
}

impl PomWalkV4 {
    /// Build over a segment table: `prefix` = T+1 cumulative canonical-chunk starts
    /// (`prefix[0] == 0`, `prefix[T] == n_chunks`), `addrs` = the T segment device addresses.
    pub fn new(vk: &Vk, prefix: &[u64], addrs: &[u64]) -> Result<Self, String> {
        let t = addrs.len();
        if t == 0 || prefix.len() != t + 1 || prefix[0] != 0 {
            return Err(format!("v4 walk: malformed segment table ({} starts, {} addresses)", prefix.len(), t));
        }
        if prefix.windows(2).any(|w| w[1] < w[0]) {
            return Err("v4 walk: segment starts are not ascending".into());
        }
        let n_chunks = prefix[t];
        let n_tiles = n_chunks / V4_TILE_CHUNKS;
        if n_tiles == 0 {
            return Err(format!("v4 walk: blob too small ({n_chunks} chunks, need >= {V4_TILE_CHUNKS})"));
        }
        let t_count = u32::try_from(t).map_err(|_| "v4 walk: too many segments")?;
        let (lut, lut_sh) = build_lut(prefix);

        let spirv = ash::util::read_spv(&mut Cursor::new(POM_WALK_V4_SPV)).map_err(|e| e.to_string())?;
        let kernel = vk.make_kernel(&spirv, 4, std::mem::size_of::<V4Push>() as u32)?;
        let io = vk.create_buffer_vram_mapped(std::mem::size_of::<V4Io>() as u64)?;
        let prefix_buf = vk.create_buffer_vram_mapped((prefix.len() * 8) as u64)?;
        vk.write_buffer(&prefix_buf, as_bytes(prefix));
        let addrs_buf = vk.create_buffer_vram_mapped((addrs.len() * 8) as u64)?;
        vk.write_buffer(&addrs_buf, as_bytes(addrs));
        let lut_buf = vk.create_buffer_vram_mapped((lut.len() * 4) as u64)?;
        vk.write_buffer(&lut_buf, as_bytes(&lut));

        Ok(Self {
            kernel,
            io,
            prefix: prefix_buf,
            addrs: addrs_buf,
            lut: lut_buf,
            lut_sh,
            t_count,
            n_tiles,
            inv_n: u64::MAX / n_tiles,
        })
    }

    /// Tiles the walk addresses (`floor(n_chunks / 32)`).
    pub fn n_tiles(&self) -> u64 {
        self.n_tiles
    }

    /// Search nonces `[start, start + batch)`; returns the lowest winning nonce. Ground in
    /// ascending TDR-bounded sub-dispatches, so the first sub-batch with a winner holds the global
    /// lowest one.
    pub fn mine(&self, vk: &Vk, job: &V4Job, start: u64, batch: u32) -> Option<u64> {
        let io = V4Io { offset: NO_WINNER, _pad: 0, t: words4(&job.target_le), h10: job.h10_state.unwrap_or([0; 25]) };
        let mut done: u32 = 0;
        while done < batch {
            let sub = (batch - done).min(V4_MAX_DISPATCH_NONCES);
            vk.write_buffer(&self.io, struct_bytes(&io));
            let push = V4Push {
                p: job.pow_words,
                s: job.seed_words,
                timestamp: job.timestamp,
                n_tiles: self.n_tiles,
                inv_n: self.inv_n,
                start_nonce: start.wrapping_add(done as u64),
                batch: sub,
                t_count: self.t_count,
                lut_sh: self.lut_sh,
                seed_h10: job.h10_state.is_some() as u32,
            };
            vk.dispatch(&self.kernel, &[&self.io, &self.prefix, &self.addrs, &self.lut], struct_bytes(&push), sub.div_ceil(V4_NPW));

            let mut out = [0u8; 4];
            vk.read_buffer(&self.io, &mut out);
            if let offset @ 0..=0xFFFF_FFFE = u32::from_le_bytes(out) {
                return Some(start.wrapping_add(done as u64 + offset as u64));
            }
            done += sub;
        }
        None
    }

    /// Release the kernel and tables. Must run before the owning `Vk` drops.
    pub fn destroy(&self, vk: &Vk) {
        vk.destroy_buffer(&self.lut);
        vk.destroy_buffer(&self.addrs);
        vk.destroy_buffer(&self.prefix);
        vk.destroy_buffer(&self.io);
        vk.destroy_kernel(&self.kernel);
    }
}

/// Bucket LUT over the canonical chunk range: `lut[b]` = the segment holding chunk `b << sh`,
/// with `sh` the smallest shift that keeps the table within [`V4_LUT_MAX_ENTRIES`].
fn build_lut(prefix: &[u64]) -> (Vec<u32>, u32) {
    let t = prefix.len() - 1;
    let n_chunks = prefix[t];
    let mut sh = 0u32;
    while ((n_chunks - 1) >> sh) + 1 > V4_LUT_MAX_ENTRIES {
        sh += 1;
    }
    let buckets = ((n_chunks - 1) >> sh) + 1;
    let mut lut = Vec::with_capacity(buckets as usize);
    let mut s = 0usize;
    for b in 0..buckets {
        let first = b << sh;
        while s + 1 < t && prefix[s + 1] <= first {
            s += 1;
        }
        lut.push(s as u32);
    }
    (lut, sh)
}

/// 32 LE bytes -> 4 u64 words.
fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn struct_bytes<T: Copy>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The LUT + forward scan resolves every chunk to the same segment as a binary search,
    /// including zero-length segments and buckets that straddle many segment starts.
    #[test]
    fn lut_scan_matches_binary_search() {
        let prefix: Vec<u64> = vec![0, 3, 3, 40, 41, 1000, 1001, 1002, 5000];
        let (lut, sh) = build_lut(&prefix);
        let t = prefix.len() - 1;
        for idx in 0..prefix[t] {
            let mut lo = lut[(idx >> sh) as usize] as usize;
            while lo + 1 < t && prefix[lo + 1] <= idx {
                lo += 1;
            }
            let want = prefix.partition_point(|&p| p <= idx) - 1;
            // `want` may name the empty segment's successor; both must start at or before idx and
            // the next start must be past it.
            assert!(prefix[lo] <= idx && idx < prefix[lo + 1], "chunk {idx}: segment {lo}");
            assert_eq!(prefix[want], prefix[lo], "chunk {idx}");
        }
    }

    #[test]
    fn lut_stays_within_its_ceiling() {
        let prefix: Vec<u64> = vec![0, 927_994_064];
        let (lut, sh) = build_lut(&prefix);
        assert!(lut.len() as u64 <= V4_LUT_MAX_ENTRIES);
        assert_eq!(((927_994_064u64 - 1) >> sh) + 1, lut.len() as u64);
    }

    #[test]
    fn push_and_io_layouts_match_the_shader() {
        assert_eq!(std::mem::size_of::<V4Push>(), 112);
        assert_eq!(std::mem::size_of::<V4Io>(), 240);
    }
}
