//! Bit-exactness test: the Vulkan PoM walk MUST agree with the host reference (an exact copy of
//! the `src/pom.rs` folds) on the real GPU. Runs only when a Vulkan device is present.
//!
//! Every case is run in BOTH walk eras — the frozen pre-H5 XOR fold (`walk_v2 = false`) and the H5
//! non-foldable mix64-chained transition (`walk_v2 = true`) — because the kernel now branches on it
//! and a drift in either branch means every block the miner finds in that era is rejected.

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

/// `pom::pom_block_seed`, taking the SEED word set directly (H5.1 salts these independently of the
/// pow words — passing them separately is exactly what this test needs to cover).
fn pom_block_seed(seed_words: &[u64; 4], timestamp: u64, nonce: u64) -> u64 {
    let mut s = mix64(nonce ^ 0x4B65727978531);
    s = mix64(s ^ timestamp);
    s = mix64(s ^ seed_words[0]);
    s = mix64(s ^ seed_words[1]);
    s = mix64(s ^ seed_words[2]);
    s = mix64(s ^ seed_words[3]);
    s
}

/// `pom::transition_v1` — frozen pre-H5 XOR fold (only the 4 words' XOR is load-bearing).
fn transition_v1(state: u64, chunk: &[u64; 4]) -> u64 {
    let mut h = state;
    for &w in chunk {
        h ^= w;
    }
    mix64(h)
}

/// `pom::transition_v2` — H5 mix64 chained through every word (all 32 bytes load-bearing).
fn transition_v2(state: u64, chunk: &[u64; 4]) -> u64 {
    let mut h = state;
    for &w in chunk {
        h = mix64(h ^ w);
    }
    h
}

fn transition(state: u64, chunk: &[u64; 4], walk_v2: bool) -> u64 {
    if walk_v2 {
        transition_v2(state, chunk)
    } else {
        transition_v1(state, chunk)
    }
}

fn walk_final(seed: u64, n_chunks: u64, k: u32, words: &[u64], walk_v2: bool) -> u64 {
    let mut state = seed;
    let mut off = state % n_chunks;
    for _ in 0..k {
        let base = (off * 4) as usize;
        let chunk = [words[base], words[base + 1], words[base + 2], words[base + 3]];
        state = transition(state, &chunk, walk_v2);
        off = state % n_chunks;
    }
    state
}

/// `pom::pom_pow_value` — always folds the POW word set, in every era (H5.1 does not touch it).
fn pom_pow_value(final_state: u64, pow_words: &[u64; 4]) -> [u8; 32] {
    let p = pow_words;
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
#[allow(clippy::too_many_arguments)]
fn host_lowest_winner(
    words: &[u64],
    n_chunks: u64,
    pow_words: &[u64; 4],
    seed_words: &[u64; 4],
    ts: u64,
    target: &[u8; 32],
    start: u64,
    batch: u32,
    walk_v2: bool,
) -> Option<u64> {
    for i in 0..batch as u64 {
        let nonce = start + i;
        let seed = pom_block_seed(seed_words, ts, nonce);
        let fs = walk_final(seed, n_chunks, POM_WALK_STEPS, words, walk_v2);
        if le_leq(&pom_pow_value(fs, pow_words), target) {
            return Some(nonce);
        }
    }
    None
}

/// Host pow values across a batch, sorted ascending — used to pick meaningful targets.
fn sorted_pows(
    words: &[u64],
    n_chunks: u64,
    pph: &[u8; 32],
    ts: u64,
    start: u64,
    batch: u32,
    walk_v2: bool,
) -> Vec<[u8; 32]> {
    let w = words4(pph);
    let mut pows: Vec<[u8; 32]> = (0..batch as u64)
        .map(|i| {
            let seed = pom_block_seed(&w, ts, start + i);
            pom_pow_value(walk_final(seed, n_chunks, POM_WALK_STEPS, words, walk_v2), &w)
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
    pows
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
    // (impossible / median / always) so we exercise the no-winner, mid, and edge cases — in both eras.
    for walk_v2 in [false, true] {
        for trial in 0..6 {
            let mut pph = [0u8; 32];
            rng.fill(&mut pph);
            let ts: u64 = rng.r#gen();
            let w = words4(&pph);

            let pows = sorted_pows(&words, n_chunks, &pph, ts, start, batch, walk_v2);
            let impossible = [0u8; 32]; // essentially no winner
            let median = pows[pows.len() / 2];
            let max = [0xFFu8; 32]; // every nonce wins → lowest is `start`

            for target in [impossible, median, max] {
                let host =
                    host_lowest_winner(&words, n_chunks, &w, &w, ts, &target, start, batch, walk_v2);
                let got = gpu.mine(&w, &w, ts, &target, start, batch, walk_v2);
                assert_eq!(
                    got, host,
                    "walk_v2={walk_v2} trial {trial}: GPU winner {got:?} != host {host:?} \
                     (target msbyte {})",
                    target[31]
                );
            }
        }

        // 'max' target must always return the very first nonce.
        let w = words4(&[7u8; 32]);
        assert_eq!(gpu.mine(&w, &w, 42, &[0xFFu8; 32], start, batch, walk_v2), Some(start));
    }
}

/// The H5 era MUST actually change the walk on the GPU: the same weights/header under `walk_v2`
/// produce a different pow distribution, so the exploit's precomputed 8-byte folds are worthless.
/// If the kernel ignored the flag this test is what catches it.
#[test]
fn vulkan_pom_walk_eras_diverge() {
    let mut rng = StdRng::seed_from_u64(0x0501_E4A5_5EED);
    let n_chunks: u64 = 257;
    let words: Vec<u64> = (0..n_chunks * 4).map(|_| rng.r#gen::<u64>()).collect();

    let gpu = match PomWalkGpu::new(&words, n_chunks) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no Vulkan device available ({e})");
            return;
        }
    };

    let start: u64 = 0x4242_0000_0000_0000;
    let batch: u32 = 4096;
    let pph = [0x5Au8; 32];
    let w = words4(&pph);
    let ts: u64 = 0x1122_3344;

    // A target that only a fraction of nonces beat, so the winning nonce is era-sensitive.
    let mut target = [0xFFu8; 32];
    target[31] = 0x02;

    let v1 = gpu.mine(&w, &w, ts, &target, start, batch, false);
    let v2 = gpu.mine(&w, &w, ts, &target, start, batch, true);

    // Each era matches its own host reference…
    assert_eq!(v1, host_lowest_winner(&words, n_chunks, &w, &w, ts, &target, start, batch, false));
    assert_eq!(v2, host_lowest_winner(&words, n_chunks, &w, &w, ts, &target, start, batch, true));
    // …and the eras are genuinely different walks (a shared winner here would be a 1-in-2^64 fluke).
    assert_ne!(v1, v2, "walk_v2 must change the walk — the kernel is ignoring the era flag");
}

/// H5.1 wiring: the seed fold must read the SEED word set while the pow fold keeps the POW set.
/// Passing deliberately different sets (as the H5.1 era does) must reproduce the host exactly — if
/// the shader wired either fold to the wrong push-constant block, this diverges.
#[test]
fn vulkan_pom_walk_seed_and_pow_words_are_separate() {
    let mut rng = StdRng::seed_from_u64(0x5EED_D1FF_0001);
    let n_chunks: u64 = 257;
    let words: Vec<u64> = (0..n_chunks * 4).map(|_| rng.r#gen::<u64>()).collect();

    let gpu = match PomWalkGpu::new(&words, n_chunks) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no Vulkan device available ({e})");
            return;
        }
    };

    let start: u64 = 0x7777_0000_0000_0000;
    let batch: u32 = 4096;
    let ts: u64 = 0xDEAD_BEEF;
    let pph = [0x3Cu8; 32];
    let pow_words = words4(&pph);
    // Stand-in for the real H5.1 salt: any distinct set exercises the same wiring.
    let seed_words = [
        pow_words[0] ^ 0x0F86_D140_0D3F_8664,
        pow_words[1] ^ 0xC296_B67C_7A7A_6A5B,
        pow_words[2] ^ 0x5F89_AD33_D961_FEAA,
        pow_words[3] ^ 0xAC6C_9AFD_FA05_3580,
    ];

    for walk_v2 in [false, true] {
        let mut target = [0xFFu8; 32];
        target[31] = 0x08;
        let host = host_lowest_winner(
            &words, n_chunks, &pow_words, &seed_words, ts, &target, start, batch, walk_v2,
        );
        let got = gpu.mine(&pow_words, &seed_words, ts, &target, start, batch, walk_v2);
        assert_eq!(got, host, "walk_v2={walk_v2}: split seed/pow words disagree with host");

        // And the split must actually matter: same pow words, seed words swapped back → different walk.
        let same = gpu.mine(&pow_words, &pow_words, ts, &target, start, batch, walk_v2);
        assert_ne!(got, same, "walk_v2={walk_v2}: seed words are being ignored by the shader");
    }
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
    for walk_v2 in [false, true] {
        for trial in 0..4 {
            let mut pph = [0u8; 32];
            rng.fill(&mut pph);
            let ts: u64 = rng.r#gen();
            let w = words4(&pph);
            let mut mid = [0u8; 32];
            mid[31] = 0x40; // ~25% of nonces win → exercises real winner selection
            for target in [[0u8; 32], mid, [0xFFu8; 32]] {
                assert_eq!(
                    streamed.mine(&w, &w, ts, &target, start, batch, walk_v2),
                    packed.mine(&w, &w, ts, &target, start, batch, walk_v2),
                    "walk_v2={walk_v2} trial {trial}: streamed and packed blobs disagree \
                     (target msbyte {})",
                    target[31]
                );
            }
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
    for walk_v2 in [false, true] {
        for trial in 0..4 {
            let mut pph = [0u8; 32];
            rng.fill(&mut pph);
            let ts: u64 = rng.r#gen();
            let w = words4(&pph);
            let pows = sorted_pows(&words, n_chunks, &pph, ts, start, batch, walk_v2);
            for target in [[0u8; 32], pows[pows.len() / 2], [0xFFu8; 32]] {
                let host =
                    host_lowest_winner(&words, n_chunks, &w, &w, ts, &target, start, batch, walk_v2);
                let got = gpu.mine(&w, &w, ts, &target, start, batch, walk_v2);
                assert_eq!(
                    got, host,
                    "walk_v2={walk_v2} multishard trial {trial}: GPU {got:?} != host {host:?}"
                );
            }
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
    let w = words4(&pph);

    for walk_v2 in [false, true] {
        // Impossible target → no winner anywhere, so every sub-dispatch runs and the loop returns None.
        assert_eq!(
            gpu.mine(&w, &w, ts, &[0u8; 32], start, batch, walk_v2),
            None,
            "walk_v2={walk_v2}: impossible target must yield None"
        );

        // Loose target → many winners; GPU must return the same global-lowest nonce the host finds.
        let loose = [0x40u8; 32];
        let host = host_lowest_winner(&words, n_chunks, &w, &w, ts, &loose, start, batch, walk_v2);
        assert_eq!(
            gpu.mine(&w, &w, ts, &loose, start, batch, walk_v2),
            host,
            "walk_v2={walk_v2}: spanning batch: GPU != host"
        );
        assert!(host.is_some(), "loose target should have a winner to make this meaningful");
    }
}
