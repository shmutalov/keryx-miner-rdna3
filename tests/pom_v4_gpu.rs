//! Lockstep: the Vulkan PoM v4 grind (`shaders/pom_walk_v4.comp`) against the host v4 walk
//! (`keryx_miner::pom_v4`, byte-exact with the node) — every seed era (pre-H10, H10, H14), both
//! resident layouts.
//!
//! The kernel only reports the LOWEST nonce whose pow value meets the target, so the test sets the
//! target to several order statistics of the host-computed pow values of a nonce batch: for every
//! threshold the GPU must name exactly the first nonce the host says passes, and nothing below the
//! minimum. A single wrong final_state anywhere in the batch shifts at least one of those answers.
//!
//! Layouts: an irregular segment table (segments of 1, 3 and 37 chunks, tiles straddling segment
//! boundaries — the zero-dup tensor case, exercising the bucket LUT + forward scan) and the miner's
//! streamed-blob shards (`PomWalkGpu`).
//!
//! Run: `cargo test --test pom_v4_gpu -- --nocapture` (skips when no Vulkan device is present).

use keryx_miner::pom::{mix64, pom_block_seed_v4_era, pom_pow_value, pom_seed_state_v4_era, pph_words_for_era, pph_words_v4};
use keryx_miner::pom_v4::{
    v4_first_offset, v4_initial_state, v4_next_offset, v4_state_root, v4_transition, POM_V4_K, POM_V4_TILE_BYTES,
    POM_V4_TILE_CHUNKS,
};
use keryx_vulkan::pom_walk::PomWalkGpu;
use keryx_vulkan::pom_walk_v4::{PomWalkV4, V4Job};
use keryx_vulkan::Vk;

/// Same generator as `pom_v3::lockstep_blob` (whole tiles + a sub-tile tail the walk never reads).
fn blob() -> Vec<u8> {
    let n_bytes = 1024 * POM_V4_TILE_BYTES + 5 * 32;
    let mut out = vec![0u8; n_bytes];
    let mut h = 0xDEAD_BEEFu64;
    for b in out.iter_mut() {
        h = mix64(h);
        *b = h as u8;
    }
    out
}

/// Host v4 walk over a contiguous canonical blob — the `pom_v4::walk_final_v4` loop.
fn host_final(seed: u64, blob: &[u8]) -> u64 {
    let n_tiles = (blob.len() / 32) as u64 / POM_V4_TILE_CHUNKS;
    let mut state = v4_initial_state(seed);
    let mut off = v4_first_offset(seed, n_tiles);
    for step in 1..=POM_V4_K as u64 {
        let tile = &blob[off as usize * POM_V4_TILE_BYTES..(off as usize + 1) * POM_V4_TILE_BYTES];
        state = v4_transition(&state, tile, step as u32);
        if step < POM_V4_K as u64 {
            off = v4_next_offset(seed, step, tile[..32].try_into().unwrap(), n_tiles);
        }
    }
    u64::from_le_bytes(v4_state_root(&state)[..8].try_into().unwrap())
}

fn le_leq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    true
}

/// Pow values of nonces `[start, start + n)` for a block at `daa`, from the host walk.
fn host_pows(blob: &[u8], pph: &[u8; 32], ts: u64, start: u64, n: u64, daa: u64) -> Vec<[u8; 32]> {
    (0..n)
        .map(|i| {
            let seed = pom_block_seed_v4_era(pph, ts, start.wrapping_add(i), daa);
            pom_pow_value(host_final(seed, blob), pph, true)
        })
        .collect()
}

/// The job `pom_gpu::mine_v4` hands the kernel for a block at `daa` (same era helper).
fn job(pph: &[u8; 32], ts: u64, daa: u64, target: [u8; 32]) -> V4Job {
    V4Job {
        pow_words: pph_words_for_era(pph, true),
        seed_words: pph_words_v4(pph),
        timestamp: ts,
        h10_state: pom_seed_state_v4_era(pph, ts, daa),
        target_le: target,
    }
}

/// Every order-statistic threshold must yield exactly the host's first passing nonce.
fn check(label: &str, mine: &dyn Fn(&V4Job, u64, u32) -> Option<u64>, pows: &[[u8; 32]], pph: &[u8; 32], ts: u64, start: u64, daa: u64) {
    let mut sorted = pows.to_vec();
    sorted.sort_by(|a, b| {
        for i in (0..32).rev() {
            if a[i] != b[i] {
                return a[i].cmp(&b[i]);
            }
        }
        std::cmp::Ordering::Equal
    });
    let n = pows.len();
    for &k in &[0, 1, n / 4, n / 2, n - 1] {
        let target = sorted[k];
        let want = pows.iter().position(|p| le_leq(p, &target)).map(|i| start.wrapping_add(i as u64));
        let got = mine(&job(pph, ts, daa, target), start, n as u32);
        assert_eq!(got, want, "{label}: threshold = order statistic {k}");
    }
    // Below the minimum nothing wins.
    let mut below = sorted[0];
    let i = below.iter().position(|&b| b != 0).expect("pow value is not zero");
    below[i] -= 1;
    for b in below[..i].iter_mut() {
        *b = 0xff;
    }
    assert_eq!(mine(&job(pph, ts, daa, below), start, n as u32), None, "{label}: below the minimum");
}

#[test]
fn v4_gpu_grind_matches_the_host_walk() {
    let vk = match Vk::new() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("SKIP: no Vulkan device ({e})");
            return;
        }
    };
    eprintln!("device: {}", vk.device_name());

    let blob = blob();
    let n_chunks = (blob.len() / 32) as u64;
    let pph = [0x5au8; 32];
    let ts = 1_788_000_000_000u64;
    let start = 0xFFFF_FFFF_FFFF_FFF0u64; // crosses the u64 wrap inside the batch
    let n: u64 = 24;

    // Irregular segment table: tiny segments near the front so early tiles straddle 2-4 segments,
    // then a few large ones.
    let mut sizes: Vec<u64> = vec![1, 3, 37, 1, 1000, 5, 4096, 31, 10_000];
    let used: u64 = sizes.iter().sum();
    sizes.push(n_chunks - used);
    let mut prefix = vec![0u64];
    let mut addrs = Vec::new();
    let mut bufs = Vec::new();
    for &sz in &sizes {
        let first = *prefix.last().unwrap();
        let (buf, addr) = vk
            .create_device_local_address_buffer(&blob[(first * 32) as usize..((first + sz) * 32) as usize])
            .expect("segment buffer");
        bufs.push(buf);
        addrs.push(addr);
        prefix.push(first + sz);
    }
    let seg = PomWalkV4::new(&vk, &prefix, &addrs).expect("v4 walk over segments");

    // Streamed-blob layout with small shards (1024 chunks) — many shards, never straddled.
    let words: Vec<u64> = blob[..(n_chunks * 32) as usize]
        .chunks(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let shards = PomWalkGpu::new_sharded(&words, n_chunks, 1024).expect("sharded blob");

    // Mainnet gates: v4 79,210,000, H10 87,360,000, H14 121,985,000.
    for (era, daa) in [("H14 seed", u64::MAX), ("H10 seed", 100_000_000u64), ("pre-H10 v4 seed", 80_000_000u64)] {
        let pows = host_pows(&blob, &pph, ts, start, n, daa);
        check(&format!("segments, {era}"), &|j, s, b| seg.mine(&vk, j, s, b), &pows, &pph, ts, start, daa);
        check(&format!("shards, {era}"), &|j, s, b| shards.mine_v4(j, s, b), &pows, &pph, ts, start, daa);
    }

    seg.destroy(&vk);
    for b in &bufs {
        vk.destroy_buffer(b);
    }
}
