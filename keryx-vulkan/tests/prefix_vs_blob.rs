//! Diagnostic: the ZERO-DUP prefix walk (`PomWalkShared`, one binary search per step over a
//! per-tensor cumulative-chunk table) vs the shard walk (`PomWalkGpu`, shift/mask) at the SAME blob
//! size, in BOTH walk eras.
//!
//! Why this exists: the miner uses the prefix kernel on the inference GPU (zero-dup — it walks the
//! engine's own resident tensors, so no second VRAM copy). If the prefix table's binary search costs
//! real throughput, the miner's live hashrate is well below the `bench_pom_walk` shard-kernel figure,
//! and the gap has nothing to do with any hardfork. This test measures that gap directly, with no
//! llama/model/IPFS dependency, so it can run anywhere a Vulkan device exists.
//!
//! Run: `cargo test -p keryx-vulkan --test prefix_vs_blob --release -- --nocapture`

use keryx_vulkan::pom_walk::{words4, PomWalkGpu, PomWalkShared};
use keryx_vulkan::Vk;

/// Deterministic well-distributed fill so the data-dependent walk spreads over the whole blob.
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

#[test]
fn prefix_walk_vs_blob_walk_hashrate() {
    // Blob size and tensor count are tunable: the real 8B-class tier is ~300 tensors over ~4.6 GB,
    // which is ~9 binary-search iterations per walk step.
    let blob_mb: u64 = std::env::var("POM_BENCH_BLOB_MB").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let n_tensors: u64 = std::env::var("POM_BENCH_TENSORS").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
    let batch: u32 = std::env::var("POM_BENCH_NONCES").ok().and_then(|s| s.parse().ok()).unwrap_or(1 << 18);

    let total_chunks = blob_mb * 1024 * 1024 / 32;
    // Split into n_tensors near-equal pieces, each a multiple of 32 B (whole chunks).
    let per = total_chunks / n_tensors;
    assert!(per > 0, "blob too small for {n_tensors} tensors");
    let n_chunks = per * n_tensors;

    let vk = match Vk::new() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("SKIP: no Vulkan device ({e})");
            return;
        }
    };
    eprintln!("device: {}", vk.device_name());
    eprintln!(
        "blob: {} MiB ({} chunks) split across {} tensors (~{} binary-search steps per walk step)",
        n_chunks * 32 / (1024 * 1024),
        n_chunks,
        n_tensors,
        (n_tensors as f64).log2().ceil() as u32
    );

    // One device-local buffer per "tensor", filled with the canonical chunk bytes for its range, so
    // the prefix table's (tensor, offset) mapping reproduces the same contiguous blob the shard walk
    // sees. Both walks must therefore agree chunk-for-chunk.
    let mut words: Vec<u64> = Vec::with_capacity((n_chunks * 4) as usize);
    for i in 0..n_chunks * 4 {
        words.push(mix64(i));
    }
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(&words[..])) };

    let mut tensors: Vec<(u64, u64)> = Vec::with_capacity(n_tensors as usize);
    let mut owned = Vec::with_capacity(n_tensors as usize);
    for t in 0..n_tensors {
        let first = (t * per * 32) as usize;
        let last = ((t + 1) * per * 32) as usize;
        let (buf, addr) = match vk.create_device_local_address_buffer(&bytes[first..last]) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("SKIP: cannot allocate tensor {t} ({e})");
                return;
            }
        };
        tensors.push((addr, per));
        owned.push(buf);
    }

    let shared = match PomWalkShared::new(vk, &tensors, owned) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: shared walk init failed ({e})");
            return;
        }
    };
    let blob = match PomWalkGpu::new(&words, n_chunks) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: blob walk init failed ({e})");
            return;
        }
    };

    let pph = [0x5au8; 32];
    let p = words4(&pph);
    let ts: u64 = 1_772_000_000;
    let start: u64 = 0xA1B2_C3D4_0000_0000;
    let none = [0u8; 32]; // impossible target → every nonce walks all 256 steps, no early exit

    // Correctness first: the two addressing schemes must pick the same winner in both eras, or the
    // timings below are comparing different computations.
    for walk_v2 in [false, true] {
        let mut mid = [0xFFu8; 32];
        mid[31] = 0x08;
        let a = shared.mine(&p, &p, ts, &mid, start, 4096, walk_v2);
        let b = blob.mine(&p, &p, ts, &mid, start, 4096, walk_v2);
        assert_eq!(a, b, "walk_v2={walk_v2}: prefix and shard walks disagree — timings meaningless");
    }

    eprintln!("\n{:<16}{:>12}{:>12}", "walk", "v1 MH/s", "v2 MH/s");
    let row = |name: &str, f: &dyn Fn(bool)| {
        let mut rates = [0f64; 2];
        for (i, walk_v2) in [false, true].into_iter().enumerate() {
            f(walk_v2); // warm
            let t0 = std::time::Instant::now();
            f(walk_v2);
            let dt = t0.elapsed().as_secs_f64();
            rates[i] = batch as f64 / dt / 1.0e6;
        }
        eprintln!("{name:<16}{:>12.2}{:>12.2}", rates[0], rates[1]);
    };
    row("shard (blob)", &|v2| {
        let _ = blob.mine(&p, &p, ts, &none, start, batch, v2);
    });
    row("prefix (0-dup)", &|v2| {
        let _ = shared.mine(&p, &p, ts, &none, start, batch, v2);
    });
    eprintln!(
        "\nThe miner mines through the PREFIX kernel on the inference GPU (zero-dup). If the prefix\n\
         row is well below the shard row, that gap — not the H5 walk change — is the live hashrate loss."
    );
}
