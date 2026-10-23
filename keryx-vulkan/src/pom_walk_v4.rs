//! PoM v4 (D=32 re-walk) GPU grind — host side of `shaders/pom_walk_v4.comp`. Finds the lowest
//! nonce in a batch whose v4 walk + H3 pow fold meets the target. The walk is byte-identical to
//! `src/pom_v4.rs` / the node; the miner still re-walks every winner on the host before submit.
//!
//! Chunk addressing is a segment table — `prefix` (cumulative canonical-chunk starts, T+1
//! entries) + `addrs` (one device address per segment) — so the same kernel serves the miner's
//! streamed blob (segments = its power-of-two shards) and the zero-dup walk over the inference
//! engine's tensors (segments = tensors). A host-built bucket LUT turns the per-chunk segment
//! search into a short forward scan, and its [`V4_LUT_SCAN`] bit tells the matrix-core walk when
//! a whole tile lies in the bucket's segment (nearly always), so it skips the scan.

use crate::{GpuBuffer, Kernel, Vk};
use std::io::Cursor;

/// SPIR-V for the v4 walk (v_dot4 and matrix-core builds of one source) and its seed pre-pass,
/// compiled from `shaders/` by build.rs.
const POM_WALK_V4_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_walk_v4.spv"));
const POM_WALK_V4_CM_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_walk_v4_cm.spv"));
const POM_SEED_V4_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_seed_v4.spv"));
/// SPIR-V for the PoM v5 context pad fill (`shaders/pom_v5_pad_fill.comp`).
const POM_V5_PAD_FILL_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_v5_pad_fill.spv"));

/// How the walk computes its 32x32x32 int8 state transition. Both are byte-identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum V4Kernel {
    /// Packed int8 dot products (v_dot4).
    Dot4,
    /// int8 cooperative matrices on the matrix cores (RDNA3+: v_wmma), at wave32.
    MatrixCores,
}

impl V4Kernel {
    pub fn name(self) -> &'static str {
        match self {
            V4Kernel::Dot4 => "v_dot4",
            V4Kernel::MatrixCores => "matrix cores",
        }
    }
}

/// The transition to try first: the matrix cores (~+35% on a 7900 XT), unless
/// `KERYX_POM_V4_KERNEL=dot` asks for v_dot4.
fn requested_kernel() -> V4Kernel {
    match std::env::var("KERYX_POM_V4_KERNEL").ok().as_deref().map(str::trim) {
        Some("dot") => V4Kernel::Dot4,
        _ => V4Kernel::MatrixCores,
    }
}

/// Invocations per workgroup of the seed pre-pass (`local_size_x` in pom_seed_v4.comp).
const SEED_WG: u32 = 64;

/// Nonces per workgroup — MUST match `NPW` in pom_walk_v4.comp (local_size = 32 x NPW).
pub const V4_NPW: u32 = 1;

/// Canonical 32 B chunks per 1 KB tile (`pom_v4::POM_V4_TILE_CHUNKS`).
pub const V4_TILE_CHUNKS: u64 = 32;

/// Max nonces per dispatch. A v4 nonce is ~256 dependent 1 KB reads plus a 32x32 int8 matmul per
/// step, so this keeps one dispatch to tens of ms — far inside the Windows TDR watchdog (~2 s) —
/// while staying well above per-submit overhead.
const V4_MAX_DISPATCH_NONCES: u32 = 1 << 15;

/// Bucket LUT ceiling (u32 entries): 4 MiB covers the largest tier (~0.93 G chunks) with
/// 1024-chunk buckets.
const V4_LUT_MAX_ENTRIES: u64 = 1 << 20;

/// LUT entry bit 31 (`V4_LUT_SCAN` in pom_walk_v4.comp): the bucket's segment ends before every
/// tile starting in the bucket does, so a lookup must scan. Bits 0..30 are the segment.
const V4_LUT_SCAN: u32 = 1 << 31;

const NO_WINNER: u32 = 0xFFFF_FFFF;

/// Walk push-constant block — layout MUST match `Push` in pom_walk_v4.comp (std430: six u64 at
/// 0..48, four u32 at 48..64, two u64 at 64..80, two u32 at 80..88; 88 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct V4Push {
    p: [u64; 4],
    n_tiles: u64,
    inv_n: u64,
    batch: u32,
    t_count: u32,
    lut_sh: u32,
    /// Always 0: the shader's rho8 uses it as a bit-field offset the compiler can't fold.
    zero: u32,
    /// v5 pad tiles (0 walks v4), its Barrett reciprocal and the pad steps per walk.
    n_pad: u64,
    inv_p: u64,
    k_pad: u32,
    _pad1: u32,
}

/// Walk steps per nonce (`pom_v4::POM_V4_K`).
const V4_K: u64 = 256;

/// ARX rounds per pad entry — MUST equal `pom_v5::POM_V5_P_ROUNDS`.
pub const V5_PAD_ROUNDS: u32 = 1_920;

/// Pad entries per fill dispatch: a couple of ms on a 7900 XT, far inside the TDR watchdog.
const V5_FILL_SLICE: u64 = 1 << 20;

/// Pad entries (one 32 B chunk each) per pad buffer: 1 GiB, like the weight shards.
const V5_PAD_SHARD_ENTRIES: u64 = 1 << 25;

/// Fill push-constant block — layout MUST match `Push` in pom_v5_pad_fill.comp (56 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct FillPush {
    base: u64,
    first: u64,
    w: [u64; 4],
    count: u32,
    rounds: u32,
}

/// Pad steps of a v5 walk over `weight_tiles` weight and `pad_tiles` pad tiles — MUST equal
/// `pom_v5::v5_pad_steps`.
pub fn v5_pad_steps(weight_tiles: u64, pad_tiles: u64) -> u32 {
    let n = weight_tiles + pad_tiles;
    ((V4_K * pad_tiles + n / 2) / n).clamp(1, V4_K - 1) as u32
}

/// The PoM v5 context pad of a tier, resident on the device: entry `e` (32 B) is canonical chunk
/// `weight_tiles * 32 + e` of the v5 walk. Filled by the device from its seed words, so none of it
/// crosses the bus. Released through [`destroy`](Self::destroy) before the owning `Vk` drops.
pub struct PadV5 {
    shards: Vec<GpuBuffer>,
    /// (device address, entries) of each shard, in pad order.
    segs: Vec<(u64, u64)>,
    tiles: u64,
}

impl PadV5 {
    /// Allocates a pad of `tiles` tiles and fills it from `seed_words`
    /// (`pom_v5::v5_pad_seed_words`, IV already folded in).
    pub fn fill(vk: &Vk, seed_words: &[u64; 4], tiles: u64) -> Result<Self, String> {
        if tiles == 0 {
            return Err("v5 pad: zero tiles".into());
        }
        let spirv = ash::util::read_spv(&mut Cursor::new(POM_V5_PAD_FILL_SPV)).map_err(|e| e.to_string())?;
        let kernel = vk.make_kernel(&spirv, 0, std::mem::size_of::<FillPush>() as u32)?;
        let mut pad = Self { shards: Vec::new(), segs: Vec::new(), tiles };
        let entries = tiles * V4_TILE_CHUNKS;
        let mut fill_all = || -> Result<(), String> {
            let mut first = 0u64;
            while first < entries {
                let n = (entries - first).min(V5_PAD_SHARD_ENTRIES);
                let (buf, addr) = vk.create_device_local_address_buffer_uninit(n * 32)?;
                pad.shards.push(buf);
                pad.segs.push((addr, n));
                let mut off = 0u64;
                while off < n {
                    let count = (n - off).min(V5_FILL_SLICE);
                    let push = FillPush {
                        base: addr + off * 32,
                        first: first + off,
                        w: *seed_words,
                        count: count as u32,
                        rounds: V5_PAD_ROUNDS,
                    };
                    vk.dispatch(&kernel, &[], struct_bytes(&push), count.div_ceil(64) as u32);
                    off += count;
                }
                first += n;
            }
            Ok(())
        };
        let res = fill_all();
        vk.destroy_kernel(&kernel);
        if let Err(e) = res {
            pad.destroy(vk);
            return Err(e);
        }
        Ok(pad)
    }

    pub fn tiles(&self) -> u64 {
        self.tiles
    }

    /// VRAM the pad holds.
    pub fn bytes(&self) -> u64 {
        self.tiles * V4_TILE_CHUNKS * 32
    }

    /// The v5 segment table: the weights of (`prefix`, `addrs`) up to their last whole tile, then
    /// this pad's shards. Returns the table and the weight tile count.
    fn tables(&self, prefix: &[u64], addrs: &[u64]) -> (Vec<u64>, Vec<u64>, u64) {
        let t = addrs.len();
        let weight_tiles = prefix[t] / V4_TILE_CHUNKS;
        let cap = weight_tiles * V4_TILE_CHUNKS;
        let mut p = vec![0u64];
        let mut a = Vec::with_capacity(t + self.segs.len());
        for i in 0..t {
            if prefix[i] >= cap {
                break;
            }
            a.push(addrs[i]);
            p.push(prefix[i + 1].min(cap));
        }
        let mut end = cap;
        for &(addr, n) in &self.segs {
            a.push(addr);
            end += n;
            p.push(end);
        }
        (p, a, weight_tiles)
    }

    /// Release the pad buffers. Must run before the owning `Vk` drops.
    pub fn destroy(&self, vk: &Vk) {
        for b in &self.shards {
            vk.destroy_buffer(b);
        }
    }
}

/// Seed pre-pass push-constant block — layout MUST match `Push` in pom_seed_v4.comp (std430: six
/// u64 at 0..48, two u32 at 48..56; 56 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct SeedPush {
    s: [u64; 4],
    timestamp: u64,
    start_nonce: u64,
    batch: u32,
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
    kind: V4Kernel,
    /// Why the requested matrix-core kernel is not the one running, if it was refused.
    fallback: Option<String>,
    seed_kernel: Kernel,
    /// Per-nonce seeds of one dispatch, written by the pre-pass and read by the walk (VRAM only).
    seeds: GpuBuffer,
    io: GpuBuffer,
    prefix: GpuBuffer,
    addrs: GpuBuffer,
    lut: GpuBuffer,
    lut_sh: u32,
    t_count: u32,
    n_tiles: u64,
    inv_n: u64,
    /// v5: pad tiles after the `n_tiles` weight tiles (0 = a v4 walk), and the pad steps per walk.
    n_pad: u64,
    k_pad: u32,
}

impl PomWalkV4 {
    /// Build over a segment table: `prefix` = T+1 cumulative canonical-chunk starts
    /// (`prefix[0] == 0`, `prefix[T] == n_chunks`), `addrs` = the T segment device addresses.
    /// The transition runs on the matrix cores where they pass their checks, else (or with
    /// `KERYX_POM_V4_KERNEL=dot`) on v_dot4; a refusal is explained by [`describe`](Self::describe).
    pub fn new(vk: &Vk, prefix: &[u64], addrs: &[u64]) -> Result<Self, String> {
        Self::pick_kernel(|kind| Self::with_kernel(vk, prefix, addrs, kind))
    }

    /// The PoM v5 walk over the weights of (`prefix`, `addrs`) and `pad`. Like [`new`](Self::new),
    /// the matrix-core kernel must agree with v_dot4 over this very table, pad included.
    pub fn new_v5(vk: &Vk, prefix: &[u64], addrs: &[u64], pad: &PadV5) -> Result<Self, String> {
        Self::pick_kernel(|kind| Self::with_kernel_v5(vk, prefix, addrs, pad, kind))
    }

    /// The requested transition, falling back from refused matrix cores to v_dot4.
    fn pick_kernel(build: impl Fn(V4Kernel) -> Result<Self, String>) -> Result<Self, String> {
        match requested_kernel() {
            V4Kernel::Dot4 => build(V4Kernel::Dot4),
            V4Kernel::MatrixCores => build(V4Kernel::MatrixCores).or_else(|e| {
                let mut walk = build(V4Kernel::Dot4)?;
                walk.fallback = Some(e);
                Ok(walk)
            }),
        }
    }

    /// [`new`](Self::new) with an explicit transition; an unusable matrix-core kernel is an error.
    pub fn with_kernel(vk: &Vk, prefix: &[u64], addrs: &[u64], kind: V4Kernel) -> Result<Self, String> {
        let n_tiles = prefix.last().copied().unwrap_or(0) / V4_TILE_CHUNKS;
        Self::with_tables(vk, prefix, addrs, kind, n_tiles, 0)
    }

    /// [`new_v5`](Self::new_v5) with an explicit transition.
    pub fn with_kernel_v5(vk: &Vk, prefix: &[u64], addrs: &[u64], pad: &PadV5, kind: V4Kernel) -> Result<Self, String> {
        if addrs.is_empty() || prefix.len() != addrs.len() + 1 {
            return Err(format!("v5 walk: malformed segment table ({} starts, {} addresses)", prefix.len(), addrs.len()));
        }
        let (p, a, weight_tiles) = pad.tables(prefix, addrs);
        Self::with_tables(vk, &p, &a, kind, weight_tiles, pad.tiles)
    }

    /// The walk over a segment table holding `n_tiles` weight tiles, then `n_pad` pad tiles.
    fn with_tables(vk: &Vk, prefix: &[u64], addrs: &[u64], kind: V4Kernel, n_tiles: u64, n_pad: u64) -> Result<Self, String> {
        let t = addrs.len();
        if t == 0 || prefix.len() != t + 1 || prefix[0] != 0 {
            return Err(format!("v4 walk: malformed segment table ({} starts, {} addresses)", prefix.len(), t));
        }
        if prefix.windows(2).any(|w| w[1] < w[0]) {
            return Err("v4 walk: segment starts are not ascending".into());
        }
        let n_chunks = prefix[t];
        if n_tiles == 0 {
            return Err(format!("v4 walk: blob too small ({n_chunks} chunks, need >= {V4_TILE_CHUNKS})"));
        }
        if (n_tiles + n_pad) * V4_TILE_CHUNKS > n_chunks {
            return Err(format!("v4 walk: {n_tiles} + {n_pad} tiles do not fit {n_chunks} chunks"));
        }
        // Segment indices share their LUT word with the V4_LUT_SCAN bit.
        let t_count = u32::try_from(t).ok().filter(|&n| n <= V4_LUT_SCAN).ok_or("v4 walk: too many segments")?;
        let (lut, lut_sh) = build_lut(prefix);

        let push_size = std::mem::size_of::<V4Push>() as u32;
        let kernel = match kind {
            V4Kernel::Dot4 => {
                let spirv = ash::util::read_spv(&mut Cursor::new(POM_WALK_V4_SPV)).map_err(|e| e.to_string())?;
                // wave32: each 32-invocation nonce slot is exactly one full wave on RDNA (driver
                // default where subgroup size control is unavailable, e.g. on ggml's borrowed device).
                vk.make_kernel_with_subgroup(&spirv, 5, push_size, Some(32))?
            }
            V4Kernel::MatrixCores => {
                if !vk.coopmat_wave32() {
                    return Err("no int8 cooperative matrices with a wave32 pin on this device".into());
                }
                let spirv = ash::util::read_spv(&mut Cursor::new(POM_WALK_V4_CM_SPV)).map_err(|e| e.to_string())?;
                vk.make_kernel_wave32(&spirv, 5, push_size)?
            }
        };
        let seed_spirv = ash::util::read_spv(&mut Cursor::new(POM_SEED_V4_SPV)).map_err(|e| e.to_string())?;
        let seed_kernel = vk.make_kernel(&seed_spirv, 2, std::mem::size_of::<SeedPush>() as u32)?;
        let seeds = vk.create_device_local_buffer(V4_MAX_DISPATCH_NONCES as u64 * 8)?;
        let io = vk.create_buffer_vram_mapped(std::mem::size_of::<V4Io>() as u64)?;
        let prefix_buf = vk.create_buffer_vram_mapped((prefix.len() * 8) as u64)?;
        vk.write_buffer(&prefix_buf, as_bytes(prefix));
        let addrs_buf = vk.create_buffer_vram_mapped((addrs.len() * 8) as u64)?;
        vk.write_buffer(&addrs_buf, as_bytes(addrs));
        let lut_buf = vk.create_buffer_vram_mapped((lut.len() * 4) as u64)?;
        vk.write_buffer(&lut_buf, as_bytes(&lut));

        let walk = Self {
            kernel,
            kind,
            fallback: None,
            seed_kernel,
            seeds,
            io,
            prefix: prefix_buf,
            addrs: addrs_buf,
            lut: lut_buf,
            lut_sh,
            t_count,
            n_tiles,
            inv_n: u64::MAX / n_tiles,
            n_pad,
            k_pad: if n_pad == 0 { 0 } else { v5_pad_steps(n_tiles, n_pad) },
        };
        if kind == V4Kernel::MatrixCores {
            if let Err(e) = walk.check_wave32(vk).and_then(|()| walk.check_against_dot4(vk)) {
                walk.destroy(vk);
                return Err(e);
            }
        }
        Ok(walk)
    }

    /// The transition this walk runs.
    pub fn kernel(&self) -> V4Kernel {
        self.kind
    }

    /// The transition for the log: its name, and why matrix cores were refused if they were asked for.
    pub fn describe(&self) -> String {
        match &self.fallback {
            Some(why) => format!("{} (matrix cores refused: {why})", self.kind.name()),
            None => self.kind.name().to_string(),
        }
    }

    /// Weight tiles the walk addresses (`floor(n_chunks / 32)`).
    pub fn n_tiles(&self) -> u64 {
        self.n_tiles
    }

    /// Pad tiles after the weight tiles: 0 for a v4 walk.
    pub fn n_pad(&self) -> u64 {
        self.n_pad
    }

    /// The matrix-core kernel is only correct at wave32: walk one nonce and read back the subgroup
    /// size it ran at (the kernel reports it in `Io._pad`).
    fn check_wave32(&self, vk: &Vk) -> Result<(), String> {
        let job = V4Job { pow_words: [0; 4], seed_words: [0; 4], timestamp: 0, h10_state: None, target_le: [0; 32] };
        self.dispatch(vk, &self.kernel, &job, 0, 1);
        let mut out = [0u8; 8];
        vk.read_buffer(&self.io, &mut out);
        match u32::from_le_bytes(out[4..8].try_into().unwrap()) {
            32 => Ok(()),
            n => Err(format!("the driver ran it at subgroup size {n}, not 32")),
        }
    }

    /// Before the matrix-core kernel mines, it must name the same lowest winner as v_dot4 over
    /// this walk's own table: 2048 nonces at two target hardnesses (first winners ~2 and ~128
    /// nonces in, so every walk up to there is compared). A driver that runs the matrix ops but
    /// gets them wrong would otherwise mine silently without ever finding a valid share.
    fn check_against_dot4(&self, vk: &Vk) -> Result<(), String> {
        let spirv = ash::util::read_spv(&mut Cursor::new(POM_WALK_V4_SPV)).map_err(|e| e.to_string())?;
        let dot = vk.make_kernel_with_subgroup(&spirv, 5, std::mem::size_of::<V4Push>() as u32, Some(32))?;
        let mut verdict = Ok(());
        for msb in [0x7Fu8, 0x01] {
            let mut target_le = [0xFFu8; 32];
            target_le[31] = msb;
            let job = V4Job {
                pow_words: [1, 2, 3, 4],
                seed_words: [5, 6, 7, 8],
                timestamp: 1_788_000_000_000,
                h10_state: Some([0x0123_4567_89AB_CDEF; 25]),
                target_le,
            };
            let (cm, reference) = (self.winner(vk, &self.kernel, &job, 2048), self.winner(vk, &dot, &job, 2048));
            if cm != reference {
                verdict = Err(format!("it disagrees with v_dot4 (lowest winner {cm:?} vs {reference:?})"));
                break;
            }
        }
        vk.destroy_kernel(&dot);
        verdict
    }

    /// Lowest winning batch index among nonces `[0, n)` under `kernel`.
    fn winner(&self, vk: &Vk, kernel: &Kernel, job: &V4Job, n: u32) -> Option<u32> {
        self.dispatch(vk, kernel, job, 0, n);
        let mut out = [0u8; 4];
        vk.read_buffer(&self.io, &mut out);
        Some(u32::from_le_bytes(out)).filter(|&offset| offset != NO_WINNER)
    }

    /// One seed pre-pass + `kernel` walk over nonces `[start, start + n)` (n <= the dispatch cap),
    /// with the winner slot reset; blocks until the GPU is done.
    fn dispatch(&self, vk: &Vk, kernel: &Kernel, job: &V4Job, start: u64, n: u32) {
        let io = V4Io { offset: NO_WINNER, _pad: 0, t: words4(&job.target_le), h10: job.h10_state.unwrap_or([0; 25]) };
        vk.write_buffer(&self.io, struct_bytes(&io));
        let seed_push = SeedPush {
            s: job.seed_words,
            timestamp: job.timestamp,
            start_nonce: start,
            batch: n,
            seed_h10: job.h10_state.is_some() as u32,
        };
        let push = V4Push {
            p: job.pow_words,
            n_tiles: self.n_tiles,
            inv_n: self.inv_n,
            batch: n,
            t_count: self.t_count,
            lut_sh: self.lut_sh,
            zero: 0,
            n_pad: self.n_pad,
            inv_p: if self.n_pad == 0 { 0 } else { u64::MAX / self.n_pad },
            k_pad: self.k_pad,
            _pad1: 0,
        };
        // Seeds first, then the walk that reads them — one submit, barrier in between.
        let seed_bufs: [&GpuBuffer; 2] = [&self.io, &self.seeds];
        let walk_bufs: [&GpuBuffer; 5] = [&self.io, &self.prefix, &self.addrs, &self.lut, &self.seeds];
        vk.dispatch_seq(&[
            (&self.seed_kernel, &seed_bufs[..], struct_bytes(&seed_push), n.div_ceil(SEED_WG)),
            (kernel, &walk_bufs[..], struct_bytes(&push), n.div_ceil(V4_NPW)),
        ]);
    }

    /// Search nonces `[start, start + batch)`; returns the lowest winning nonce. Ground in
    /// ascending TDR-bounded sub-dispatches, so the first sub-batch with a winner holds the global
    /// lowest one.
    pub fn mine(&self, vk: &Vk, job: &V4Job, start: u64, batch: u32) -> Option<u64> {
        let mut done: u32 = 0;
        while done < batch {
            let sub = (batch - done).min(V4_MAX_DISPATCH_NONCES);
            self.dispatch(vk, &self.kernel, job, start.wrapping_add(done as u64), sub);
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
        vk.destroy_buffer(&self.seeds);
        vk.destroy_kernel(&self.seed_kernel);
        vk.destroy_kernel(&self.kernel);
    }
}

/// Bucket LUT over the canonical chunk range: `lut[b]` = the segment holding chunk `b << sh`,
/// with `sh` the smallest shift that keeps the table within [`V4_LUT_MAX_ENTRIES`], plus
/// [`V4_LUT_SCAN`] where that segment ends inside the bucket — or, when buckets are smaller than
/// a tile, inside the tile starting at the bucket (the only tile a walk looks up from it).
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
        let end = (first + (1 << sh)).max(first + V4_TILE_CHUNKS).min(n_chunks);
        lut.push(s as u32 | if prefix[s + 1] < end { V4_LUT_SCAN } else { 0 });
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
            let mut lo = (lut[(idx >> sh) as usize] & !V4_LUT_SCAN) as usize;
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

    /// An entry without V4_LUT_SCAN names a segment that holds every tile starting in its bucket —
    /// with 1-chunk buckets (sh = 0) as well as multi-tile ones — and flags stay rare.
    #[test]
    fn lut_scan_bit_marks_every_bucket_a_tile_can_leave() {
        // 40 K chunks with tiny and empty segments up front (sh = 0, 1-chunk buckets), and a 4 GiB
        // table with an empty segment and a start on a bucket boundary (sh = 7, 4 tiles a bucket).
        let small = vec![0u64, 1, 4, 41, 42, 42, 1042, 1047, 5143, 5174, 15_174, 40_000];
        let wide = vec![0u64, 3_000_000, 3_000_000, 3_000_017, 70_000_000, 134_217_728];
        for prefix in [small, wide] {
            let (lut, sh) = build_lut(&prefix);
            let t = prefix.len() - 1;
            let n_tiles = prefix[t] / V4_TILE_CHUNKS;
            let mut flagged = 0;
            for tile in 0..n_tiles {
                let t0 = tile * V4_TILE_CHUNKS;
                let e = lut[(t0 >> sh) as usize];
                flagged += (e & V4_LUT_SCAN != 0) as u64;
                if e & V4_LUT_SCAN == 0 {
                    let lo = e as usize;
                    assert!(prefix[lo] <= t0 && t0 + V4_TILE_CHUNKS <= prefix[lo + 1], "sh {sh}: tile {tile} leaves segment {lo}");
                }
            }
            assert!(flagged * 100 < n_tiles, "sh {sh}: {flagged} of {n_tiles} tiles flagged");
        }
    }

    /// The v5 table caps the weights at their last whole tile and appends the pad shards.
    #[test]
    fn v5_tables_cap_the_weights_and_append_the_pad() {
        let pad = PadV5 { shards: Vec::new(), segs: vec![(0xA000, 64), (0xB000, 32)], tiles: 3 };
        // 2 tiles + 5 tail chunks over three segments; the last one starts past the cap.
        let (p, a, w) = pad.tables(&[0, 40, 64, 69], &[0x10, 0x20, 0x30]);
        assert_eq!(w, 2);
        assert_eq!(p, vec![0, 40, 64, 128, 160]);
        assert_eq!(a, vec![0x10, 0x20, 0xA000, 0xB000]);
        // pom_v5::v5_pad_steps: round(256 * P / (W + P)), clamped to [1, 255].
        assert_eq!(v5_pad_steps(16_384, 638), ((256 * 638 + 17_022 / 2) / 17_022) as u32);
        assert_eq!((v5_pad_steps(10, 1), v5_pad_steps(1, 1_000)), (23, 255));
        assert_eq!(v5_pad_steps(1_000_000, 1), 1);
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
        assert_eq!(std::mem::size_of::<V4Push>(), 88);
        assert_eq!(std::mem::size_of::<FillPush>(), 56);
        assert_eq!(std::mem::size_of::<SeedPush>(), 56);
        assert_eq!(std::mem::size_of::<V4Io>(), 240);
    }
}
