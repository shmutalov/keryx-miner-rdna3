//! Diagnostic: PoM v4 grind throughput on this GPU, over a VRAM-resident pseudo-random blob in
//! both resident layouts — the streamed-blob shards (`PomWalkGpu`) and a many-segment table like
//! the zero-dup tensor walk (`PomWalkV4` over ~300 segments, exercising the bucket LUT).
//!
//! Correctness lives in the miner crate's `tests/pom_v4_gpu.rs` lockstep test; this only times.
//!
//! Run: `cargo test -p keryx-vulkan --test pom_v4_rate --release -- --ignored --nocapture`
//! Env: POM_BENCH_BLOB_MB (default 4096)  POM_BENCH_NONCES (default 65536)  POM_BENCH_ITERS (default 5)
//!      POM_BENCH_SEGMENTS (default 300)

use keryx_vulkan::pom_walk::PomWalkGpu;
use keryx_vulkan::pom_walk_v4::{PomWalkV4, V4Job};
use keryx_vulkan::Vk;
use std::time::Instant;

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

fn env(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Deterministic chunk bytes for canonical chunk range starting at `first_chunk`.
fn fill(first_chunk: u64, out: &mut [u8]) {
    for (i, w) in out.chunks_exact_mut(8).enumerate() {
        w.copy_from_slice(&mix64(first_chunk * 4 + i as u64).to_le_bytes());
    }
}

fn time(label: &str, iters: u64, batch: u32, mut run: impl FnMut(u64) -> Option<u64>) {
    run(0); // warm-up (pipeline + caches)
    let mut rates = Vec::new();
    for it in 0..iters {
        let t = Instant::now();
        let found = run((it + 1) * batch as u64);
        let dt = t.elapsed().as_secs_f64();
        assert!(found.is_none(), "an all-zero target can never be met");
        rates.push(batch as f64 / dt);
    }
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "{label:<28} min {:>9.0}  med {:>9.0}  max {:>9.0} nonces/s",
        rates[0],
        rates[rates.len() / 2],
        rates[rates.len() - 1]
    );
}

#[test]
#[ignore = "GPU throughput diagnostic; run explicitly with --ignored --release"]
fn pom_v4_grind_rate() {
    let blob_mb = env("POM_BENCH_BLOB_MB", 4096);
    let batch = env("POM_BENCH_NONCES", 65536) as u32;
    let iters = env("POM_BENCH_ITERS", 5);
    let n_segments = env("POM_BENCH_SEGMENTS", 300);
    let n_chunks = blob_mb * 1024 * 1024 / 32;

    let job = V4Job {
        pow_words: [1, 2, 3, 4],
        seed_words: [5, 6, 7, 8],
        timestamp: 1_788_000_000_000,
        h10_state: Some([0x0123_4567_89ab_cdef; 25]),
        target_le: [0u8; 32],
    };

    let blob = match PomWalkGpu::new_streamed(None, n_chunks, &mut |first, out| {
        fill(first, out);
        Ok(())
    }) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: no Vulkan device / blob upload failed ({e})");
            return;
        }
    };
    eprintln!("device: {}  blob: {} MiB ({} tiles)  batch: {}", blob.device_name(), blob_mb, n_chunks / 32, batch);
    time("shards (streamed blob)", iters, batch, |start| blob.mine_v4(&job, start, batch));
    drop(blob);

    // Many-segment table over separately allocated device-local buffers (zero-dup shape).
    let vk = Vk::new().expect("vulkan device");
    let per = n_chunks / n_segments;
    let mut prefix = vec![0u64];
    let mut addrs = Vec::new();
    let mut bufs = Vec::new();
    for s in 0..n_segments {
        let first = s * per;
        let len = if s + 1 == n_segments { n_chunks - first } else { per };
        let (buf, addr) = vk
            .create_device_local_address_buffer_streamed(len * 32, &mut |off, out| {
                fill(first + off / 32, out);
                Ok(())
            })
            .expect("segment buffer");
        bufs.push(buf);
        addrs.push(addr);
        prefix.push(first + len);
    }
    let seg = PomWalkV4::new(&vk, &prefix, &addrs).expect("v4 walk over segments");
    time(&format!("{n_segments} segments (LUT)"), iters, batch, |start| seg.mine(&vk, &job, start, batch));
    seg.destroy(&vk);
    for b in &bufs {
        vk.destroy_buffer(b);
    }
}
