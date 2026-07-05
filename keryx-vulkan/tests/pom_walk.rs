//! Bit-exactness test: the Vulkan PoM walk MUST agree with the host reference (an exact copy of
//! the `src/pom.rs` folds) on the real GPU. Runs only when a Vulkan device is present.

use keryx_vulkan::pom_walk::{words4, PomWalkGpu, POM_WALK_STEPS};
use rand::{rngs::StdRng, Rng, SeedableRng};

// ── Host reference: byte-identical to src/pom.rs (mix64 / pom_block_seed / walk_final / pow). ──

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

fn pph_words(b: &[u8; 32]) -> [u64; 4] {
    words4(b)
}

fn pom_block_seed(pph: &[u8; 32], timestamp: u64, nonce: u64) -> u64 {
    let p = pph_words(pph);
    let mut s = mix64(nonce ^ 0x4B65727978531);
    s = mix64(s ^ timestamp);
    s = mix64(s ^ p[0]);
    s = mix64(s ^ p[1]);
    s = mix64(s ^ p[2]);
    s = mix64(s ^ p[3]);
    s
}

fn transition(state: u64, chunk: &[u64; 4]) -> u64 {
    let mut h = state;
    for &w in chunk {
        h ^= w;
    }
    mix64(h)
}

fn walk_final(seed: u64, n_chunks: u64, k: u32, words: &[u64]) -> u64 {
    let mut state = seed;
    let mut off = state % n_chunks;
    for _ in 0..k {
        let base = (off * 4) as usize;
        let chunk = [words[base], words[base + 1], words[base + 2], words[base + 3]];
        state = transition(state, &chunk);
        off = state % n_chunks;
    }
    state
}

fn pom_pow_value(final_state: u64, pph: &[u8; 32]) -> [u8; 32] {
    let p = pph_words(pph);
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

/// Host brute-force: lowest nonce in [start, start+batch) whose pow_value <= target.
fn host_lowest_winner(words: &[u64], n_chunks: u64, pph: &[u8; 32], ts: u64, target: &[u8; 32], start: u64, batch: u32) -> Option<u64> {
    for i in 0..batch as u64 {
        let nonce = start + i;
        let seed = pom_block_seed(pph, ts, nonce);
        let fs = walk_final(seed, n_chunks, POM_WALK_STEPS, words);
        if le_leq(&pom_pow_value(fs, pph), target) {
            return Some(nonce);
        }
    }
    None
}

#[test]
fn vulkan_pom_walk_matches_host_reference() {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE_1234_5678);

    // Synthetic weight blob: 257 chunks (avoids power-of-two modulo coincidences).
    let n_chunks: u64 = 257;
    let words: Vec<u64> = (0..n_chunks * 4).map(|_| rng.r#gen::<u64>()).collect();

    let gpu = match PomWalkGpu::new(&words, n_chunks) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no Vulkan device available ({e})");
            return;
        }
    };
    eprintln!("PoM walk running on: {}", gpu.device_name());

    let start: u64 = 0x9988_7766_5544_0000;
    let batch: u32 = 8192;

    // A few random headers; for each, derive the host pow distribution and probe several targets
    // (impossible / median / loose / always) so we exercise the no-winner, mid, and edge cases.
    for trial in 0..6 {
        let mut pph = [0u8; 32];
        rng.fill(&mut pph);
        let ts: u64 = rng.r#gen();

        // Host pow values across the batch → sorted, to choose meaningful targets.
        let mut pows: Vec<[u8; 32]> = (0..batch as u64)
            .map(|i| {
                let seed = pom_block_seed(&pph, ts, start + i);
                pom_pow_value(walk_final(seed, n_chunks, POM_WALK_STEPS, &words), &pph)
            })
            .collect();
        pows.sort_by(|a, b| {
            for i in (0..32).rev() {
                if a[i] != b[i] {
                    return a[i].cmp(&b[i]);
                }
            }
            std::cmp::Ordering::Equal
        });

        let impossible = [0u8; 32]; // essentially no winner
        let median = pows[pows.len() / 2];
        let max = [0xFFu8; 32]; // every nonce wins → lowest is `start`

        for target in [impossible, median, max] {
            let host = host_lowest_winner(&words, n_chunks, &pph, ts, &target, start, batch);
            let got = gpu.mine(&words4(&pph), ts, &target, start, batch);
            assert_eq!(
                got, host,
                "trial {trial}: GPU winner {got:?} != host {host:?} (target msbyte {})",
                target[31]
            );
        }
    }

    // 'max' target must always return the very first nonce.
    assert_eq!(gpu.mine(&words4(&[7u8; 32]), 42, &[0xFFu8; 32], start, batch), Some(start));
}

/// The zero-dup streamed constructor MUST produce a blob byte-identical to the packed one: same
/// winners for the same headers/targets, across a multi-shard layout (per-shard `source` offsets
/// are the part the packed path never exercises).
#[test]
fn vulkan_pom_walk_streamed_matches_packed() {
    let mut rng = StdRng::seed_from_u64(0x57EA_4ED5_7EA4);
    let n_chunks: u64 = 257;
    let words: Vec<u64> = (0..n_chunks * 4).map(|_| rng.r#gen::<u64>()).collect();
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

    // 64 chunks/shard → 5 shards; the source is called per staging window with chunk offsets.
    let mut calls = 0u32;
    let streamed = match PomWalkGpu::new_streamed_sharded(None, n_chunks, 64, &mut |first_chunk, out| {
        calls += 1;
        let base = (first_chunk * 32) as usize;
        out.copy_from_slice(&bytes[base..base + out.len()]);
        Ok(())
    }) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no Vulkan device available ({e})");
            return;
        }
    };
    assert!(calls >= 5, "streamed source called {calls} times, expected at least one per shard");
    let packed = PomWalkGpu::new_sharded(&words, n_chunks, 64).expect("packed constructor");

    let start: u64 = 0x1234_5678_9ABC_0000;
    let batch: u32 = 4096;
    for trial in 0..4 {
        let mut pph = [0u8; 32];
        rng.fill(&mut pph);
        let ts: u64 = rng.r#gen();
        let mut mid = [0u8; 32];
        mid[31] = 0x40; // ~25% of nonces win → exercises real winner selection
        for target in [[0u8; 32], mid, [0xFFu8; 32]] {
            assert_eq!(
                streamed.mine(&words4(&pph), ts, &target, start, batch),
                packed.mine(&words4(&pph), ts, &target, start, batch),
                "trial {trial}: streamed and packed blobs disagree (target msbyte {})",
                target[31]
            );
        }
    }
}

/// Same bit-exactness check, but force a MULTI-SHARD layout (tiny shards over a 257-chunk blob) so
/// the shard-mapping path (shift/mask + per-shard device address) is exercised without multi-GiB
/// allocations. The host reference reads the blob contiguously; the GPU must agree across shards.
#[test]
fn vulkan_pom_walk_multishard_matches_host_reference() {
    let mut rng = StdRng::seed_from_u64(0xABCD_4321);
    let n_chunks: u64 = 257;
    let words: Vec<u64> = (0..n_chunks * 4).map(|_| rng.r#gen::<u64>()).collect();

    // 64 chunks/shard → ceil(257/64) = 5 shards, last one partial.
    let gpu = match PomWalkGpu::new_sharded(&words, n_chunks, 64) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no Vulkan device available ({e})");
            return;
        }
    };
    eprintln!("multi-shard PoM walk running on: {}", gpu.device_name());

    let start: u64 = 0x0102_0304_0500_0000;
    let batch: u32 = 8192;
    for trial in 0..4 {
        let mut pph = [0u8; 32];
        rng.fill(&mut pph);
        let ts: u64 = rng.r#gen();
        let mut pows: Vec<[u8; 32]> = (0..batch as u64)
            .map(|i| {
                let seed = pom_block_seed(&pph, ts, start + i);
                pom_pow_value(walk_final(seed, n_chunks, POM_WALK_STEPS, &words), &pph)
            })
            .collect();
        pows.sort_by(|a, b| {
            for i in (0..32).rev() {
                if a[i] != b[i] {
                    return a[i].cmp(&b[i]);
                }
            }
            std::cmp::Ordering::Equal
        });
        for target in [[0u8; 32], pows[pows.len() / 2], [0xFFu8; 32]] {
            let host = host_lowest_winner(&words, n_chunks, &pph, ts, &target, start, batch);
            let got = gpu.mine(&words4(&pph), ts, &target, start, batch);
            assert_eq!(got, host, "multishard trial {trial}: GPU {got:?} != host {host:?}");
        }
    }
}

/// Exercise the sub-dispatch loop: a batch larger than one dispatch (the miner uses 1<<20; the GPU
/// caps each dispatch internally) must still agree with the host across the dispatch boundary,
/// including the no-winner case that grinds every sub-dispatch to completion.
#[test]
fn vulkan_pom_walk_spans_dispatches() {
    let mut rng = StdRng::seed_from_u64(0x5151_2323);
    let n_chunks: u64 = 257;
    let words: Vec<u64> = (0..n_chunks * 4).map(|_| rng.r#gen::<u64>()).collect();

    let gpu = match PomWalkGpu::new(&words, n_chunks) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no Vulkan device available ({e})");
            return;
        }
    };

    // Spans 3 internal dispatches (MAX_DISPATCH_NONCES = 1<<16): exercises the `start + done` math.
    let start: u64 = 0xAABB_CCDD_0000_0000;
    let batch: u32 = (1 << 17) + 4096;
    let mut pph = [0u8; 32];
    rng.fill(&mut pph);
    let ts: u64 = rng.r#gen();

    // Impossible target → no winner anywhere, so every sub-dispatch runs and the loop returns None.
    assert_eq!(gpu.mine(&words4(&pph), ts, &[0u8; 32], start, batch), None, "impossible target must yield None");

    // Loose target → many winners; GPU must return the same global-lowest nonce the host finds.
    let loose = [0x40u8; 32];
    let host = host_lowest_winner(&words, n_chunks, &pph, ts, &loose, start, batch);
    assert_eq!(gpu.mine(&words4(&pph), ts, &loose, start, batch), host, "spanning batch: GPU != host");
    assert!(host.is_some(), "loose target should have a winner to make this meaningful");
}
