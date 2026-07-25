//! PoM-walk kernel micro-benchmark (feature `bench`). Measures the walk in isolation — no miner,
//! no model download, no IPFS — over a large, cache-defeating synthetic weight blob, so the number
//! reflects real VRAM random-access performance rather than an L2/Infinity-Cache-resident toy.
//!
//! It builds the blob's device-local shards ONCE, then times byte-identical kernel variants against
//! it and prints MH/s + speedup vs the shipping baseline:
//!   * `baseline` — the production `pom_walk.comp` (local_size 64, hardware 64-bit `%`)
//!   * `wave32`   — identical math, `local_size_x = 32` (native RDNA3 wave width)
//!   * `magic`    — hardware `%` replaced by a precomputed libdivide multiply-shift
//!   * `ilp4`     — 4 independent nonces per thread (4× memory-level parallelism)
//!
//! Every variant is first checked BIT-EXACT against a host reference (an exact copy of the
//! `src/pom.rs` folds) on a small blob; a variant that fails verification is reported and excluded
//! from the timed comparison, so a wrong magic can never masquerade as a speedup.
//!
//! Run:  `cargo run -p keryx-vulkan --example bench_pom_walk --features bench --release`
//! Env:  POM_BENCH_BLOB_MB (default 1024)  POM_BENCH_NONCES (default 16777216)  POM_BENCH_ITERS (default 5)
//!       POM_BENCH_WALK_V2 (default 1 — the live H5 era; set 0 to measure the frozen pre-H5 fold)
//!
//! ─────────────────────────────────────────────────────────────────────────────────────────────
//! H5 FINDING (2026-07-25, RX 7900 XT, 1 GiB blob) — the H5 `transition_v2` is FREE on RDNA3:
//!
//!   era                          baseline median MH/s
//!   v1 (pre-H5 XOR fold)                19.16
//!   v2 (H5 mix64-chained)               19.45
//!
//! v2 does 4 `mix64` per chunk instead of 1 and costs nothing measurable — consistent with the
//! findings below that the walk is memory-latency-bound, not ALU-bound: the extra mixing hides
//! entirely under the 256 dependent VRAM reads. Also measured: 8 GiB blob → 18.16 MH/s (v2), and
//! the zero-dup prefix kernel → 18.26 MH/s (v2) at a 4.6 GiB / 300-tensor layout (see
//! `tests/prefix_vs_blob.rs`). So NO kernel-level H5 hashrate regression exists on this GPU.
//! If a rig reports a large post-H5 hashrate drop, look OUTSIDE the walk (OPoI inference pauses,
//! model load/index build stalls, tier reassignment, clocks) — not at these kernels.
//!
//! FINDINGS (RX 7900 XT, 20 GiB, driver 32.0.31019.2002, 1 GiB blob) — READ THIS BEFORE OPTIMIZING:
//!
//!   variant    median MH/s   vs baseline
//!   baseline      ~20.0          —
//!   wave32        ~19.7        -1.3%
//!   magic         ~18.6        -7.1%
//!   ilp4          ~18.2        -9.0%
//!
//! Every perturbation REGRESSED. What each one rules out:
//!   * magic slower  → the walk is NOT ALU/modulo-bound; the hardware 64-bit `%` is already fully
//!                     hidden under memory-stall latency, so removing it only adds register pressure.
//!   * wave32 slower → occupancy/wave-width is not the lever; the default wave64 packs better here.
//!   * ilp4 slower   → the memory system is ALREADY saturated at 1 nonce/thread; more per-thread
//!                     outstanding reads don't add throughput, they just cost registers → occupancy.
//!
//! Conclusion: the baseline kernel sits at a local optimum on RDNA3. ~20 MH/s is the practical
//! ceiling for this GPU on the consensus-fixed walk (256 dependent random 32-byte reads/nonce over
//! a multi-GiB resident blob). The kernel is memory-bound; the real lever is VRAM clock/timings,
//! not the shader. The remaining gap to NVIDIA at equal nominal bandwidth is best explained by
//! memory-access granularity — NVIDIA services a random 32-byte read from a 32-byte sector (no
//! waste), while RDNA3 over-fetches a larger cache line per read. The 32-byte chunk size is
//! consensus-locked, so that cannot be worked around in the shader. If you think you have a real
//! speedup, prove it HERE (bit-exact + faster than baseline) before touching the production kernel.

use crate::{GpuBuffer, Kernel, Vk};
use std::io::Cursor;
use std::time::Instant;

// SPIR-V for each variant, compiled from shaders/*.comp by build.rs.
const BASELINE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_walk.spv"));
const WAVE32_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_walk_wave32.spv"));
const MAGIC_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_walk_magic.spv"));
const ILP4_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_walk_ilp4.spv"));

/// Nonces per invocation for the ILP variant — MUST match `const uint G` in pom_walk_ilp4.comp.
const ILP4_G: u32 = 4;

const POM_WALK_STEPS: u32 = 256;
const SHARD_CHUNKS: u64 = 1 << 25; // 1 GiB / 32 B — same as production pom_walk.rs
const MAX_DISPATCH_NONCES: u32 = 1 << 16; // same TDR-safe sub-dispatch as production
const NO_WINNER: u32 = 0xFFFF_FFFF;
const IMPOSSIBLE: [u8; 32] = [0u8; 32]; // pow_value <= 0 essentially never → full grind, no early exit

// ── Push-constant blocks (must match the shaders' `Push` layouts) ──────────────────────────────

/// Baseline / wave32 push (116 bytes).
///
/// NOTE: this is deliberately NOT the production `PomPush` any more. Production carries the H5.1
/// seed word set (`s0..s3`) and reads the target out of the binding-0 buffer to stay inside the
/// 128 B guaranteed `maxPushConstantsSize`; the bench variants keep the target inline (they only
/// ever mine the impossible target) and skip the seed split entirely, because they measure the WALK
/// and the seed fold runs once per nonce. They do carry `walk_v2` — that is the hot loop.
#[repr(C)]
#[derive(Clone, Copy)]
struct PomPush {
    p: [u64; 4],
    t: [u64; 4],
    timestamp: u64,
    n_chunks: u64,
    start_nonce: u64,
    shard_mask: u64,
    k: u32,
    batch: u32,
    shard_shift: u32,
    walk_v2: u32,
}

/// PRODUCTION push layout — must stay byte-identical to `pom_walk::PomPush` / the `Push` block in
/// `pom_walk.comp`, because the `baseline` variant benches the real production kernel. Production
/// carries the H5.1 seed word set and reads the target from the binding-0 I/O buffer instead of
/// inline (see [`IoBlock`]); the bench-only variants keep the older inline-target layout.
#[repr(C)]
#[derive(Clone, Copy)]
struct ProdPush {
    p: [u64; 4],
    s: [u64; 4],
    timestamp: u64,
    n_chunks: u64,
    start_nonce: u64,
    shard_mask: u64,
    k: u32,
    batch: u32,
    shard_shift: u32,
    walk_v2: u32,
}

/// Binding-0 I/O block: winner slot + target, matching production's `Io` block. The bench-only
/// variants declare just `{ uint offset; }` over the same buffer (a legal prefix view) and take
/// their target from push constants, so writing the full block is safe for every variant.
#[repr(C)]
#[derive(Clone, Copy)]
struct IoBlock {
    offset: u32,
    _pad: u32,
    t: [u64; 4],
}

/// Magic push — `PomPush` + magic/shift/is_pow2 (128 bytes = exactly the guaranteed
/// `maxPushConstantsSize`; nothing more fits here).
#[repr(C)]
#[derive(Clone, Copy)]
struct MagicPush {
    p: [u64; 4],
    t: [u64; 4],
    timestamp: u64,
    n_chunks: u64,
    start_nonce: u64,
    shard_mask: u64,
    magic: u64,
    k: u32,
    batch: u32,
    shard_shift: u32,
    magic_shift: u32,
    is_pow2: u32,
    walk_v2: u32,
}

fn as_bytes<T: Copy>(p: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(p as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn words_as_bytes(words: &[u64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words)) }
}

fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

// ── libdivide u64 branchfree magic (host side) ─────────────────────────────────────────────────

/// Precompute (magic, shift, is_pow2) so that `n % d == n - (magic_div(n) * d)` for all u64 n.
/// Power-of-two d uses the mask fast-path (magic unused). d must be >= 1.
fn magic_gen(d: u64) -> (u64, u32, bool) {
    assert!(d >= 1, "n_chunks must be >= 1");
    if d.is_power_of_two() {
        return (0, d.trailing_zeros(), true);
    }
    let l = 63 - d.leading_zeros(); // floor(log2(d))
    let num: u128 = 1u128 << (64 + l); // 2^(64+l)
    let proposed_m = (num / d as u128) as u64; // floor(2^(64+l) / d)
    let rem = (num % d as u128) as u64;
    let mut m = proposed_m.wrapping_add(proposed_m); // 2*proposed_m
    let twice_rem = rem.wrapping_add(rem);
    if twice_rem >= d || twice_rem < rem {
        m = m.wrapping_add(1);
    }
    (m.wrapping_add(1), l, false)
}

fn mulhi64(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

/// Host mirror of the shader's `mod_magic`, for the pre-flight self-test.
fn mod_magic_host(n: u64, d: u64, magic: u64, shift: u32, is_pow2: bool) -> u64 {
    if is_pow2 {
        return n & (d - 1);
    }
    let q = mulhi64(magic, n);
    let t = ((n - q) >> 1) + q;
    let q = t >> shift;
    n - q * d
}

/// Panic if the magic formula disagrees with hardware `%` for any tested n — catches a bad magic on
/// the HOST before a single GPU dispatch, independent of the (identical) u64 arithmetic on the GPU.
fn selftest_magic(d: u64) {
    let (magic, shift, is_pow2) = magic_gen(d);
    let mut probes: Vec<u64> = vec![0, 1, d - 1, d, d + 1, u64::MAX, u64::MAX - 1, 1 << 63];
    // deterministic pseudo-random spread (no rng dep in the lib)
    let mut x = 0x1234_5678_9abc_def0u64;
    for _ in 0..20000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        probes.push(x);
    }
    for n in probes {
        let got = mod_magic_host(n, d, magic, shift, is_pow2);
        assert_eq!(got, n % d, "magic modulo wrong for n={n}, d={d} (got {got}, want {})", n % d);
    }
}

// ── Host reference (exact copy of src/pom.rs folds) for GPU bit-exactness ───────────────────────

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

fn pom_block_seed(p: &[u64; 4], timestamp: u64, nonce: u64) -> u64 {
    let mut s = mix64(nonce ^ 0x4B65727978531);
    s = mix64(s ^ timestamp);
    s = mix64(s ^ p[0]);
    s = mix64(s ^ p[1]);
    s = mix64(s ^ p[2]);
    s = mix64(s ^ p[3]);
    s
}

fn walk_final(seed: u64, n_chunks: u64, words: &[u64], walk_v2: bool) -> u64 {
    let mut state = seed;
    let mut off = state % n_chunks;
    for _ in 0..POM_WALK_STEPS {
        let base = (off * 4) as usize;
        let mut h = state;
        if walk_v2 {
            // H5 `transition_v2` — mix64 chained through every word.
            h = mix64(h ^ words[base]);
            h = mix64(h ^ words[base + 1]);
            h = mix64(h ^ words[base + 2]);
            h = mix64(h ^ words[base + 3]);
            state = h;
        } else {
            h ^= words[base];
            h ^= words[base + 1];
            h ^= words[base + 2];
            h ^= words[base + 3];
            state = mix64(h);
        }
        off = state % n_chunks;
    }
    state
}

fn pom_pow_value(final_state: u64, p: &[u64; 4]) -> [u8; 32] {
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

fn le_leq(a: &[u8; 32], b: &[u8; 32]) -> bool {
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

#[allow(clippy::too_many_arguments)]
fn host_lowest_winner(words: &[u64], n_chunks: u64, p: &[u64; 4], ts: u64, target: &[u8; 32], start: u64, batch: u32, walk_v2: bool) -> Option<u64> {
    (0..batch as u64).map(|i| start + i).find(|&nonce| {
        let fs = walk_final(pom_block_seed(p, ts, nonce), n_chunks, words, walk_v2);
        le_leq(&pom_pow_value(fs, p), target)
    })
}

// ── Resident blob + variant dispatch ───────────────────────────────────────────────────────────

/// One resident weight blob (device-local shards + address table) reused across all variant kernels.
struct Blob<'a> {
    vk: &'a Vk,
    shards: Vec<GpuBuffer>,
    addr_table: GpuBuffer,
    winner: GpuBuffer,
    shard_chunks: u64,
}

impl<'a> Blob<'a> {
    fn new(vk: &'a Vk, words: &[u64], n_chunks: u64, shard_chunks: u64) -> Result<Self, String> {
        let n_shards = n_chunks.div_ceil(shard_chunks);
        let mut shards = Vec::with_capacity(n_shards as usize);
        let mut addrs = Vec::with_capacity(n_shards as usize);
        for s in 0..n_shards {
            let first = (s * shard_chunks * 4) as usize;
            let last = (((s + 1) * shard_chunks).min(n_chunks) * 4) as usize;
            let (buf, addr) = vk.create_device_local_address_buffer(words_as_bytes(&words[first..last]))?;
            shards.push(buf);
            addrs.push(addr);
        }
        let addr_table = vk.create_buffer((addrs.len() * 8) as u64)?;
        vk.write_buffer(&addr_table, words_as_bytes(&addrs));
        let winner = vk.create_buffer(std::mem::size_of::<IoBlock>() as u64)?;
        Ok(Self { vk, shards, addr_table, winner, shard_chunks })
    }

    /// Grind nonces `[start, start+batch)` with `kernel`; returns the lowest winning nonce (or None).
    /// `mk_push` builds the per-sub-dispatch push bytes given (sub_batch, sub_start_nonce). Sub-
    /// dispatches are ascending so the first with a winner holds the global lowest — same as prod.
    #[allow(clippy::too_many_arguments)]
    fn grind(&self, kernel: &Kernel, nonces_per_group: u32, mk_push: &dyn Fn(u32, u64) -> Vec<u8>, start: u64, batch: u32, target: [u64; 4]) -> Option<u64> {
        let mut done = 0u32;
        while done < batch {
            let sub = (batch - done).min(MAX_DISPATCH_NONCES);
            // Winner reset + target in one write: production reads the target from here.
            self.vk.write_buffer(&self.winner, as_bytes(&IoBlock { offset: NO_WINNER, _pad: 0, t: target }));
            let push = mk_push(sub, start + done as u64);
            let groups = sub.div_ceil(nonces_per_group);
            self.vk.dispatch(kernel, &[&self.winner, &self.addr_table], &push, groups);
            let mut out = [0u8; 4];
            self.vk.read_buffer(&self.winner, &mut out);
            let off = u32::from_le_bytes(out);
            if off != NO_WINNER {
                return Some(start + done as u64 + off as u64);
            }
            done += sub;
        }
        None
    }
}

impl Drop for Blob<'_> {
    fn drop(&mut self) {
        self.vk.destroy_buffer(&self.winner);
        self.vk.destroy_buffer(&self.addr_table);
        for s in &self.shards {
            self.vk.destroy_buffer(s);
        }
    }
}

/// A benchable kernel + how to build its push bytes for this blob.
struct Variant {
    name: &'static str,
    kernel: Kernel,
    local_size: u32,
    nonces_per_invocation: u32, // ILP factor (G); 1 for the scalar variants
    magic: Option<(u64, u32, bool)>, // Some => magic push layout
    prod: bool,                      // true => the real production kernel (ProdPush layout)
}

impl Variant {
    /// Nonces ground per workgroup = local_size × ILP factor. Used to size the dispatch.
    fn nonces_per_group(&self) -> u32 {
        self.local_size * self.nonces_per_invocation
    }
}

impl Variant {
    /// Curried push builder for a given header/target — captures everything except (sub, start_nonce).
    #[allow(clippy::too_many_arguments)]
    fn push_fn<'b>(&'b self, p: [u64; 4], t: [u64; 4], ts: u64, n_chunks: u64, shard_mask: u64, shard_shift: u32, walk_v2: bool) -> Box<dyn Fn(u32, u64) -> Vec<u8> + 'b> {
        let walk_v2 = walk_v2 as u32;
        if self.prod {
            // Production kernel: seed words == pow words here (the bench never crosses H5.1, and the
            // seed fold is one-per-nonce so it cannot move the walk timing), target rides the buffer.
            return Box::new(move |sub, start_nonce| {
                as_bytes(&ProdPush {
                    p, s: p, timestamp: ts, n_chunks, start_nonce, shard_mask,
                    k: POM_WALK_STEPS, batch: sub, shard_shift, walk_v2,
                })
                .to_vec()
            });
        }
        match self.magic {
            None => Box::new(move |sub, start_nonce| {
                as_bytes(&PomPush {
                    p, t, timestamp: ts, n_chunks, start_nonce, shard_mask,
                    k: POM_WALK_STEPS, batch: sub, shard_shift, walk_v2,
                })
                .to_vec()
            }),
            Some((magic, magic_shift, is_pow2)) => Box::new(move |sub, start_nonce| {
                as_bytes(&MagicPush {
                    p, t, timestamp: ts, n_chunks, start_nonce, shard_mask, magic,
                    k: POM_WALK_STEPS, batch: sub, shard_shift, magic_shift,
                    is_pow2: is_pow2 as u32, walk_v2,
                })
                .to_vec()
            }),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn make_variant(vk: &Vk, name: &'static str, spv: &[u8], local_size: u32, npi: u32, magic: Option<(u64, u32, bool)>, prod: bool) -> Result<Variant, String> {
    let spirv = ash::util::read_spv(&mut Cursor::new(spv)).map_err(|e| e.to_string())?;
    let push_size = if prod {
        std::mem::size_of::<ProdPush>()
    } else if magic.is_some() {
        std::mem::size_of::<MagicPush>()
    } else {
        std::mem::size_of::<PomPush>()
    } as u32;
    let kernel = vk.make_kernel(&spirv, 2, push_size)?;
    Ok(Variant { name, kernel, local_size, nonces_per_invocation: npi, magic, prod })
}

/// Deterministic well-distributed blob content so the data-dependent walk spreads across the whole
/// blob (a degenerate/constant fill could cycle inside cache and inflate the result).
fn fill_blob(n_words: usize) -> Vec<u64> {
    (0..n_words as u64).map(mix64).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Entry point. Verifies every variant bit-exact, then times them on a large blob and prints MH/s.
pub fn run() {
    let blob_mb: u64 = std::env::var("POM_BENCH_BLOB_MB").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let nonces: u32 = std::env::var("POM_BENCH_NONCES").ok().and_then(|s| s.parse().ok()).unwrap_or(1 << 24);
    let iters: usize = std::env::var("POM_BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    // H5 era by default: v2 is what mainnet runs now, so it is what needs optimizing. The v1 fold
    // is still selectable to reproduce the pre-H5 numbers in the table above.
    let walk_v2: bool = std::env::var("POM_BENCH_WALK_V2").ok().map(|s| s != "0").unwrap_or(true);
    eprintln!("walk era: {}", if walk_v2 { "v2 (H5, mix64-chained — 4 mix64/chunk)" } else { "v1 (pre-H5 XOR fold)" });

    let vk = match Vk::new() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("SKIP: no Vulkan device ({e})");
            return;
        }
    };
    println!("PoM-walk benchmark on: {}", vk.device_name());
    println!("VRAM: {} MiB\n", vk.device_local_vram_mb());

    // ── 1. Bit-exactness: every variant must match the host reference on a small blob. ──
    // 257 chunks (non-power-of-two) exercises the magic branchfree path; 256 exercises the
    // is_pow2 mask fast-path.
    println!("== bit-exactness ==");
    let mut ok_variants: Vec<&'static str> = Vec::new();
    for &vn in &["baseline", "wave32", "magic", "ilp4"] {
        let mut all_ok = true;
        for &nc in &[257u64, 256u64] {
            selftest_magic(nc); // host-only sanity for the magic used at this n_chunks
            let words = fill_blob((nc * 4) as usize);
            // shard_chunks must be a power of two (the shader maps chunk→shard with shift+mask);
            // one power-of-two shard >= nc keeps everything in shard 0 with local == off.
            let blob = match Blob::new(&vk, &words, nc, nc.next_power_of_two()) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("  {vn} (n={nc}): blob build failed: {e}");
                    all_ok = false;
                    break;
                }
            };
            let variant = build_named(&vk, vn, nc);
            let variant = match variant {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("  {vn}: kernel build failed: {e}");
                    all_ok = false;
                    break;
                }
            };
            if !verify_variant(&vk, &blob, &variant, &words, nc) {
                all_ok = false;
            }
            vk.destroy_kernel(&variant.kernel);
        }
        println!("  {:<9} {}", vn, if all_ok { "PASS" } else { "FAIL — excluded from timing" });
        if all_ok {
            ok_variants.push(vn);
        }
    }

    // ── 2. Timing on a large, cache-defeating blob (shared across variants). ──
    let n_chunks = blob_mb * 32768 - 1; // ~blob_mb MiB, forced non-power-of-two (1 MiB = 32768 chunks)
    println!("\n== timing ==");
    println!("blob: {} MiB ({} chunks), {} nonces/iter x {} iters\n", n_chunks * 32 / (1024 * 1024), n_chunks, nonces, iters);
    println!("building blob in VRAM…");
    let words = fill_blob((n_chunks * 4) as usize);
    let blob = match Blob::new(&vk, &words, n_chunks, SHARD_CHUNKS) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("blob build failed: {e}");
            return;
        }
    };
    drop(words); // free host copy; blob is resident in VRAM

    let p = words4(&[0x11u8; 32]);
    let t = words4(&IMPOSSIBLE);
    let ts = 0x0123_4567_89ab_cdefu64;
    let shard_mask = SHARD_CHUNKS - 1;
    let shard_shift = SHARD_CHUNKS.trailing_zeros();

    println!("{:<10} {:>10} {:>10} {:>10}   {:>8}", "variant", "min MH/s", "med MH/s", "max MH/s", "vs base");
    let mut baseline_med = 0.0f64;
    for &vn in &ok_variants {
        let variant = match build_named(&vk, vn, n_chunks) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{vn}: kernel build failed: {e}");
                continue;
            }
        };
        let push = variant.push_fn(p, t, ts, n_chunks, shard_mask, shard_shift, walk_v2);
        // warm-up (untimed): page in, warm the pipeline/caches consistently.
        blob.grind(&variant.kernel, variant.nonces_per_group(), &*push, 0, nonces.min(1 << 20), t);

        let mut rates = Vec::with_capacity(iters);
        for it in 0..iters {
            let start = 0x8000_0000_0000_0000u64 + (it as u64) * nonces as u64;
            let t0 = Instant::now();
            blob.grind(&variant.kernel, variant.nonces_per_group(), &*push, start, nonces, t);
            let dt = t0.elapsed().as_secs_f64();
            rates.push(nonces as f64 / dt / 1e6);
        }
        let (mn, md, mx) = (
            rates.iter().cloned().fold(f64::MAX, f64::min),
            median(rates.clone()),
            rates.iter().cloned().fold(0.0, f64::max),
        );
        if vn == "baseline" {
            baseline_med = md;
        }
        let vs = if baseline_med > 0.0 { format!("{:+.1}%", (md / baseline_med - 1.0) * 100.0) } else { "—".into() };
        println!("{:<10} {:>10.2} {:>10.2} {:>10.2}   {:>8}", vn, mn, md, mx, vs);
        vk.destroy_kernel(&variant.kernel);
    }
    println!("\n(median is the comparison figure; 'vs base' is median vs baseline median)");
}

/// Build the named variant with the correct magic for `n_chunks`.
fn build_named(vk: &Vk, name: &str, n_chunks: u64) -> Result<Variant, String> {
    match name {
        "baseline" => make_variant(vk, "baseline", BASELINE_SPV, 64, 1, None, true),
        "wave32" => make_variant(vk, "wave32", WAVE32_SPV, 32, 1, None, false),
        "magic" => make_variant(vk, "magic", MAGIC_SPV, 64, 1, Some(magic_gen(n_chunks)), false),
        "ilp4" => make_variant(vk, "ilp4", ILP4_SPV, 64, ILP4_G, None, false),
        other => Err(format!("unknown variant {other}")),
    }
}

/// Check a variant's lowest-winner selection against the host across several targets, in BOTH walk
/// eras — a variant that only agrees under the frozen v1 fold is useless for tuning the live H5 walk.
fn verify_variant(_vk: &Vk, blob: &Blob, variant: &Variant, words: &[u64], n_chunks: u64) -> bool {
    let p = words4(&[0x5au8; 32]);
    let ts = 0xDEAD_BEEF_0000_0001u64;
    let start = 0x1122_3344_0000_0000u64;
    let batch: u32 = 8192;
    // shard params MUST match the power-of-two shard_chunks the blob was built with.
    let shard_mask = blob.shard_chunks - 1;
    let shard_shift = blob.shard_chunks.trailing_zeros();
    for walk_v2 in [false, true] {
        // pick a median target so there is a real mid-batch winner to disagree on
        let mut pows: Vec<[u8; 32]> = (0..batch as u64)
            .map(|i| pom_pow_value(walk_final(pom_block_seed(&p, ts, start + i), n_chunks, words, walk_v2), &p))
            .collect();
        pows.sort_by(|a, b| {
            for i in (0..32).rev() {
                if a[i] != b[i] {
                    return a[i].cmp(&b[i]);
                }
            }
            std::cmp::Ordering::Equal
        });
        let targets = [[0u8; 32], pows[pows.len() / 2], [0xFFu8; 32]];
        for target in targets {
            let pt = words4(&target);
            let push = variant.push_fn(p, pt, ts, n_chunks, shard_mask, shard_shift, walk_v2);
            let got = blob.grind(&variant.kernel, variant.nonces_per_group(), &*push, start, batch, pt);
            let host = host_lowest_winner(words, n_chunks, &p, ts, &target, start, batch, walk_v2);
            if got != host {
                eprintln!(
                    "  {} (n={n_chunks}, walk_v2={walk_v2}): MISMATCH got={got:?} host={host:?} (target msb {})",
                    variant.name, target[31]
                );
                return false;
            }
        }
    }
    true
}
