//! Lockstep: the Vulkan PoM v5 grind (the v4 kernel over the weights + the device-filled context
//! pad, `shaders/pom_v5_pad_fill.comp`) against the host v5 walk (`keryx_miner::pom_v5`,
//! byte-exact with the node).
//!
//! Same order-statistic scheme as `pom_v4_gpu`: for every threshold the GPU must name exactly the
//! host's first passing nonce. About a third of the steps read the pad, so a wrong pad entry, pad
//! offset or state fold shifts the answers.
//!
//! Run: `cargo test --test pom_v5_gpu -- --nocapture` (skips when no Vulkan device is present).

use keryx_miner::pom::{mix64, pom_block_seed_v4_era, pom_pow_value, pom_seed_state_v4_era, pph_words_for_era, pph_words_v4};
use keryx_miner::pom_v4::{v4_initial_state, v4_state_root, v4_transition, POM_V4_K, POM_V4_TILE_BYTES, POM_V4_TILE_CHUNKS};
use keryx_miner::pom_v5::{v5_first_offset, v5_next_offset, v5_pad_seed_words, v5_pad_steps, v5_pad_tile, v5_state_fold};
use keryx_vulkan::pom_walk::PomWalkGpu;
use keryx_vulkan::pom_walk_v4::{PadV5, PomWalkV4, V4Job, V4Kernel};
use keryx_vulkan::Vk;

/// Whole tiles + a sub-tile tail the walk never reads (the pad starts at the last whole tile).
fn blob() -> Vec<u8> {
    let n_bytes = 1024 * POM_V4_TILE_BYTES + 5 * 32;
    let mut out = vec![0u8; n_bytes];
    let mut h = 0xC0FF_EE11u64;
    for b in out.iter_mut() {
        h = mix64(h);
        *b = h as u8;
    }
    out
}

/// Host v5 walk over a contiguous canonical blob — the `pom_v5::walk` loop.
fn host_final(seed: u64, blob: &[u8], pad_seed: &[u8; 32], pad_tiles: u64) -> u64 {
    let weight_tiles = (blob.len() / 32) as u64 / POM_V4_TILE_CHUNKS;
    let pad_steps = v5_pad_steps(weight_tiles, pad_tiles);
    let mut state = v4_initial_state(seed);
    let mut off = v5_first_offset(seed, weight_tiles);
    for step in 1..=POM_V4_K as u64 {
        let tile = if off < weight_tiles {
            blob[off as usize * POM_V4_TILE_BYTES..(off as usize + 1) * POM_V4_TILE_BYTES].to_vec()
        } else {
            v5_pad_tile(pad_seed, off - weight_tiles)
        };
        if step < POM_V4_K as u64 {
            let snippet = tile[..32].try_into().unwrap();
            off = v5_next_offset(seed, step, snippet, v5_state_fold(&state), weight_tiles, pad_tiles, pad_steps);
        }
        state = v4_transition(&state, &tile, step as u32);
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

fn job(pph: &[u8; 32], ts: u64, target: [u8; 32]) -> V4Job {
    V4Job {
        pow_words: pph_words_for_era(pph, true),
        seed_words: pph_words_v4(pph),
        timestamp: ts,
        h10_state: pom_seed_state_v4_era(pph, ts, u64::MAX),
        target_le: target,
    }
}

/// Every order-statistic threshold must yield exactly the host's first passing nonce.
fn check(label: &str, mine: &dyn Fn(&V4Job, u64, u32) -> Option<u64>, pows: &[[u8; 32]], pph: &[u8; 32], ts: u64, start: u64) {
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
        let got = mine(&job(pph, ts, target), start, n as u32);
        assert_eq!(got, want, "{label}: threshold = order statistic {k}");
    }
    let mut below = sorted[0];
    let i = below.iter().position(|&b| b != 0).expect("pow value is not zero");
    below[i] -= 1;
    for b in below[..i].iter_mut() {
        *b = 0xff;
    }
    assert_eq!(mine(&job(pph, ts, below), start, n as u32), None, "{label}: below the minimum");
}

#[test]
fn v5_gpu_grind_matches_the_host_walk() {
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
    let pph = [0xa5u8; 32];
    let ts = 1_790_000_000_000u64;
    let start = 0x1234_5678_9ABC_DEF0u64;
    let n: u64 = 24;
    let pad_seed = [0x3cu8; 32];
    let seed_words = v5_pad_seed_words(&pad_seed);

    // 1024 weight tiles; pads of 1 tile (one pad step, every read the same tile), 509 (prime, ~85
    // pad steps), 3000 (~190 pad steps) and 40,000 (250 pad steps; 1.28 M entries, so the device
    // fills it in two dispatch slices).
    for pad_tiles in [1u64, 509, 3000, 40_000] {
        let pows: Vec<[u8; 32]> = (0..n)
            .map(|i| {
                let seed = pom_block_seed_v4_era(&pph, ts, start.wrapping_add(i), u64::MAX);
                pom_pow_value(host_final(seed, &blob, &pad_seed, pad_tiles), &pph, true)
            })
            .collect();

        // Irregular weight segments; the last one carries the sub-tile tail the pad must cut off.
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
        let pad = PadV5::fill(&vk, &seed_words, pad_tiles).expect("pad fill");
        let dot = PomWalkV4::with_kernel_v5(&vk, &prefix, &addrs, &pad, V4Kernel::Dot4).expect("v5 walk (dot4)");
        assert_eq!((dot.n_tiles(), dot.n_pad()), (1024, pad_tiles));
        check(&format!("segments, dot4, pad {pad_tiles}"), &|j, s, b| dot.mine(&vk, j, s, b), &pows, &pph, ts, start);
        dot.destroy(&vk);
        match PomWalkV4::with_kernel_v5(&vk, &prefix, &addrs, &pad, V4Kernel::MatrixCores) {
            Ok(cm) => {
                check(&format!("segments, matrix cores, pad {pad_tiles}"), &|j, s, b| cm.mine(&vk, j, s, b), &pows, &pph, ts, start);
                cm.destroy(&vk);
            }
            Err(e) => eprintln!("SKIP matrix cores: {e}"),
        }
        pad.destroy(&vk);
        for b in &bufs {
            vk.destroy_buffer(b);
        }

        // The miner's resident path: streamed-blob shards, pad filled and cached by the walker.
        let words: Vec<u64> = blob[..(n_chunks * 32) as usize]
            .chunks(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let shards = PomWalkGpu::new_sharded(&words, n_chunks, 1024).expect("sharded blob");
        check(
            &format!("shards, pad {pad_tiles}"),
            &|j, s, b| shards.mine_v5(j, s, b, &seed_words, pad_tiles).expect("v5 grind"),
            &pows,
            &pph,
            ts,
            start,
        );
        assert_eq!(shards.release_pad(), pad_tiles * POM_V4_TILE_BYTES as u64);
    }
}
