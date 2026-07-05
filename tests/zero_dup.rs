//! Zero-dup end-to-end equivalence: the shared PoM walk (over the in-process inference
//! engine's OWN resident ggml-vulkan weight buffers) must agree bit-for-bit with the
//! miner-owned streamed-blob walk — same chunks, same winners — before it is trusted to mine.
//!
//! Ignored by default (loads a multi-GB model + a full weight blob on the GPU; ~11 GB VRAM):
//!   KERYX_TEST_GGUF=<path to model.gguf> cargo test --release --test zero_dup -- --ignored --nocapture

use keryx_miner::llm_engine::LlamaEngine;
use keryx_miner::pom::WeightIndex;
use keryx_vulkan::pom_walk::{words4, PomWalkGpu};

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[test]
#[ignore = "loads a multi-GB GGUF twice on the GPU; set KERYX_TEST_GGUF to run"]
fn shared_walk_matches_streamed_blob() {
    let Ok(gguf) = std::env::var("KERYX_TEST_GGUF") else {
        eprintln!("SKIP: KERYX_TEST_GGUF not set");
        return;
    };

    // 1. In-process engine: the model becomes VRAM-resident in ggml's buffers.
    let engine = LlamaEngine::launch(&gguf).expect("engine launch");

    // 2. Shared walk over the engine's tensors — VK-resident in place, host-side (token_embd)
    //    supplemented into a small owned buffer, canonical name-sorted table throughout.
    let (shared, supl_bytes) = engine.build_shared_walk().expect("shared walk");
    eprintln!(
        "shared walk: N={} chunks, {} MB supplemented (host-side tensors)",
        shared.n_chunks(),
        supl_bytes / (1024 * 1024)
    );

    // 3. Possession index (GGUF-backed ground truth) — N must agree exactly.
    let idx = WeightIndex::build_from_gguf(&gguf).expect("weight index");
    assert_eq!(shared.n_chunks(), idx.n_chunks, "shared table N != index N");

    // 4. Sample-verify: 512 pseudo-random chunks through the GPU prefix table vs the index.
    let mut seed = 0xC0FFEE ^ idx.n_chunks;
    for _ in 0..512 {
        let off = splitmix(&mut seed) % idx.n_chunks;
        let gpu = shared.read_chunk(off);
        let mut gpu_words = [0u64; 4];
        for (i, w) in gpu_words.iter_mut().enumerate() {
            *w = u64::from_le_bytes(gpu[i * 8..i * 8 + 8].try_into().unwrap());
        }
        assert_eq!(gpu_words, idx.read_chunk(off), "chunk {off} differs (shared vs index)");
    }
    eprintln!("sample-verify: 512 chunks identical");

    // 5. The reference: the live-validated streamed blob (second VRAM copy, same GPU).
    let mut source =
        |first: u64, out: &mut [u8]| idx.read_chunk_range(first, out).map_err(|e| e.to_string());
    let blob = PomWalkGpu::new_streamed(None, idx.n_chunks, &mut source).expect("streamed blob");

    // 6. Winner equality across target hardnesses: all-win, ~50%, ~0.4%, ~0.002%, none.
    let start: u64 = 0xA1B2_C3D4_0000_0000;
    let batch: u32 = 1 << 16;
    let mut pph = [0u8; 32];
    for (i, b) in pph.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(37).wrapping_add(11);
    }
    // Kernel-identity test, era-agnostic: raw (pre-H3) pph words on both paths. H3 salting
    // happens host-side before the words reach mine() and cannot diverge the two walks.
    let ts: u64 = 1_772_000_000;
    let p = words4(&pph);
    for msb in [0xFFu8, 0x80, 0x01, 0x00] {
        let mut target = [0u8; 32];
        target[31] = msb;
        if msb == 0xFF {
            target = [0xFF; 32];
        }
        let a = shared.mine(&p, ts, &target, start, batch);
        let b = blob.mine(&p, ts, &target, start, batch);
        assert_eq!(a, b, "winner mismatch at target msb {msb:#x}");
        let c = shared.mine(&p, ts, &target, start, batch);
        assert_eq!(a, c, "shared walk not deterministic at target msb {msb:#x}");
        eprintln!("target msb {msb:#04x}: shared == blob == {a:?}");
    }

    // 7. Hashrate: prefix-table binary search vs shard shift/mask, no-winner target so every
    //    nonce walks all 256 steps. 8×65536 nonces each, warm.
    let none = [0u8; 32];
    let rounds: u64 = 8;
    let blob_f = |s: u64| blob.mine(&p, ts, &none, s, batch);
    let shared_f = |s: u64| shared.mine(&p, ts, &none, s, batch);
    let timed = |name: &str, f: &dyn Fn(u64) -> Option<u64>| {
        let t0 = std::time::Instant::now();
        for r in 0..rounds {
            let _ = f(start.wrapping_add(r * batch as u64));
        }
        let dt = t0.elapsed().as_secs_f64();
        eprintln!("{name}: {:.2} MH/s ({} nonces in {dt:.3}s)", (rounds * batch as u64) as f64 / dt / 1.0e6, rounds * batch as u64);
    };
    timed("blob  ", &blob_f);
    timed("shared", &shared_f);
}
