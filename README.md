# Keryx Miner — RDNA3 / Vulkan fork

A fork of [keryx-miner](https://github.com/Keryx-Labs/keryx-miner) that runs **entirely on AMD
RDNA3 GPUs (e.g. Radeon RX 7900 XT/XTX) via Vulkan** — no CUDA, no OpenCL, no CPU compute. It
combines GPU PoW (kHeavyHash), GPU Proof-of-Model possession mining (PoM), and on-chain AI
inference (OPoI — Optimistic Proof of Inference).

> Upstream is NVIDIA/CUDA-only (the engine is candle + CUDA, and the build needs `nvcc`). This fork
> replaces every GPU compute path with Vulkan and runs the OPoI models through a prebuilt
> **llama.cpp Vulkan** server, so the miner builds and runs on an AMD-only host.

---

## What runs where

| Workload | Backend | Notes |
|---|---|---|
| **PoW** (kHeavyHash) | Vulkan compute shader | `keccak-f1600 → 64×64 matmul → wave_mix → keccak`. Bit-exact vs the host reference, verified on a 7900 XT. |
| **PoM** (possession walk) | Vulkan compute shader | walk over the model weights resident in **device-local VRAM**, sharded across ≤1 GiB buffers reached by buffer-device-address (an 8B+ model's blob exceeds AMD's 4 GiB SSBO range and 2 GiB single-allocation limits). Bit-exact vs `pom::walk_final`, verified on a 7900 XT. |
| **OPoI inference** (H4 lineup: EXAONE / Mistral / GLM-4 / Qwen3.6 / Kimi-Linear) | **llama.cpp Vulkan, in-process** | linked into the miner (all layers on GPU); the PoM walk shares the engine's resident weight buffers (zero-dup) — no child process, no HTTP. |
| **OPoI fraud-proof commitment** | CPU (integer) | `model_fixed::forward` — a deterministic, bit-exact 32-byte fold required by consensus. Not a GPU/LLM workload; unchanged. |

The custom kernels live in the [`keryx-vulkan`](keryx-vulkan/) crate (uses [`ash`](https://crates.io/crates/ash);
shaders compiled GLSL→SPIR-V with `glslc`). Both kernels have bit-exactness tests that run on the
real GPU (`cargo test -p keryx-vulkan`).

---

## Requirements

**Build:**
- Rust + Cargo ([rustup.rs](https://rustup.rs/))
- `protoc` (protobuf compiler)
- **Vulkan SDK** — provides `glslc`, used at build time to compile the compute shaders to SPIR-V.
  Set `VULKAN_SDK` (the installer does this) or put `glslc` on `PATH`, or point `GLSLC` at it.

**Run:**
- An AMD RDNA3 GPU + recent driver (the Vulkan runtime loader `vulkan-1` ships with the AMD
  Adrenalin / Mesa RADV driver — no Vulkan SDK needed at runtime).

---

## Build from source

```bash
git clone <this-fork> keryx-miner-rdna3
cd keryx-miner-rdna3
cargo build --release
```

Binary: `target/release/keryx-miner-rdna3` (`.exe` on Windows). No `nvcc`, no CUDA toolkit, no
model SDKs are required.

---

## Inference setup

Nothing to install: OPoI inference runs **in-process** (llama.cpp/Vulkan is linked into the
miner, all layers on the GPU), and on the inference GPU the PoM walk reads the engine's own
resident weight buffers (zero-dup) — mining a tier costs ~281 MB of extra VRAM on that card
instead of a second full weight copy. The GGUF model files are downloaded on demand over IPFS
on first run (same as upstream).

---

## Usage

```bash
./keryx-miner-rdna3 --mining-address keryx:YOUR_ADDRESS
```

### Inference tiers (OPoI — H6 lineup `POM_TIERS_H6`, tier 3 swapped at H14 `POM_TIERS_H14`)

| Flag | Model | Tier | Min VRAM | Fits a 7900 XT (20 GB)? |
|------|-------|------|----------|--------------------------|
| `--very-light` | Qwen3.5-9B-abliterated (Q5_K_M) | 0 | 8 GB | ✅ |
| `--light` | GLM-4-9B-0414 (Q6_K) | 1 | 12 GB | ✅ |
| *(default)* | gemma-4-12B-it-abliterated (Q6_K) | 2 | 16 GB | ✅ |
| `--high` | Qwen3.6-27B (Q4_K_M); **Qwen3.8-27B (Q4_K) from H14** | 3 | 24 GB | no |
| `--very-high` | Kimi-Linear-48B (Q4_K_M) | 4 | 30 GB | no |

> Under PoM, **1 GPU = 1 tier**: each tier proves possession of and serves exactly the single
> model above — a PoM GPU is bound to the one model whose weights are resident in VRAM. The
> miner checks the GGUF it indexes against the node-pinned `(R_T, N)` anchor of its tier and
> refuses to mine on a mismatch. `--high` downloads both tier-3 models until the H14 gate and swaps
> the resident model on the first block past it.

### Consensus this build mines

**PoM v4** (the D=32 matrix re-walk, mainnet DAA 79,210,000) with the **H10 keccak walk seed**
(mainnet DAA 87,360,000; both from DAA 1 on testnet) and, from **H14** (private inference, mainnet
DAA 121,985,000 ≈ 2026-10-09 14:00 UTC; testnet 6,000), the H10 seed taken over the H14-tagged
pre-PoW hash — the rules of keryx-node v1.6.4. Each
nonce walks a 32×32 int8 state through 256 chained 1 KB weight tiles on the GPU
(`keryx-vulkan/shaders/pom_walk_v4.comp`); every winner is re-walked on the host, which builds the
proof (all 256 tiles + their Merkle range proofs under the tier's `R_T`) and self-verifies it before
submitting. The header carries `pomFinalState`, `pomTier` and the node's `serviceStateHash` (H6).
Older eras (the v1/v2 hash walk, the H6 v3 matrix walk) are history on every network.

**Private inference (H14).** Every AiRequest is sealed to its tier's provider cohort. A solo miner
opens the ones addressed to its escrow key, answers, and seals the answer back to the requester; the
sealed body rides inline in the signed AiResponse, so IPFS (kubo) is no longer needed past the gate.
Pool miners get the opened prompt from the pool (`mining.ai_request`), and legacy notify-borne
inference tasks are ignored past the gate.

### Solo mining: escrow delegation cert (required since H6)

A solo (`grpc://`) coinbase must carry `/escrow:<key>` **and** `/esig:<cert>` — a schnorr signature
by your **payout address** over the miner's escrow key — or the node rejects the block. On start the
miner prints `Escrow key to authorise in your wallet: <64 hex>`:

1. In your wallet, use **Authorise a miner** and paste that key.
2. Pass the returned 128-hex line once with `--escrow-cert <cert>`; it is saved to `escrow.cert`
   (`--escrow-cert-file`) and loaded on later starts.

When the payout address *is* the miner's own escrow key, the cert is signed locally. Without a valid
cert the miner refuses to start in solo mode. Pool mining needs none (the pool builds the coinbase).

The miner is **GPU-only by default** (no CPU mining threads); pass `--threads N` (`-t`) to add CPU
PoW workers if you want them.

### Multi-GPU

By default the miner spawns **one PoW/PoM worker per discrete Vulkan GPU** (the startup log prints
the enumerated device list); restrict with `--gpu 0,2` (raw Vulkan device indices). Every mining
GPU keeps its own resident copy of the same tier's weight blob, streamed straight from the GGUF on
disk (no full-model host RAM copy — peak host overhead is a 256 MiB staging window). Inference is
pinned to the first discrete GPU via `GGML_VK_VISIBLE_DEVICES` (override with `KERYX_INFER_GPU=N`),
where the walk shares its weights (zero-dup); only the worker sharing that GPU pauses during OPoI
challenges — the other cards keep mining with their own streamed weight blobs.

### Building from source

The miner links llama.cpp (Vulkan) in-process via [`llama-cpp-2`], so building needs a C++
toolchain besides Rust: CMake + Ninja, LLVM (`libclang` for bindgen — set `LIBCLANG_PATH` if
not auto-detected), and the Vulkan SDK (`glslc` + headers; set `VULKAN_SDK`). On a GNU-toolchain
(MinGW) Windows host also set
`BINDGEN_EXTRA_CLANG_ARGS="--target=x86_64-w64-mingw32 -I<mingw>/x86_64-w64-mingw32/include -I<llvm>/lib/clang/<ver>/include"`
and keep MinGW's `bin` on `PATH` at runtime (`libstdc++-6.dll`). Tip: put your machine's values
in a git-ignored `.cargo/config.toml` `[env]` section so plain `cargo build` works without
exporting anything. Prebuilt release binaries need none of this.

[`llama-cpp-2`]: https://github.com/utilityai/llama-cpp-rs

### Solo vs pool

`--keryxd-address` takes either a `grpc://` node (solo) or a `stratum+tcp://` pool URL. For pools:

```bash
./keryx-miner-rdna3 --mining-address keryx:YOUR_ADDRESS \
  --keryxd-address stratum+tcp://krx.suprnova.cc:4401 \
  --worker rig1 \
  --password d=1
```

- `--worker` is sent as `address.worker` so the pool credits shares per rig.
- `--password` is the stratum `mining.authorize` password; on suprnova-style pools it requests a
  fixed difficulty (e.g. `d=1`). Default `x` = the pool's own (vardiff) difficulty.

### All options

```bash
./keryx-miner-rdna3 --help
```

### Useful environment variables

| Var | Meaning |
|-----|---------|
| `KERYX_VULKAN_WORKLOAD` | nonces per PoW dispatch (default `1048576`) |
| `KERYX_POW_ONLY` | `1` = mine kHeavyHash shares only; skip OPoI models + inference (no PoM) |
| `KERYX_POM_KEEP_RESIDENT` | `1` = keep the PoM weight blob resident across inference when VRAM fits (skips reload) |
| `KERYX_INFER_GPU` | raw Vulkan device index inference is pinned to on multi-GPU rigs (default: first discrete GPU) |
| `KERYX_MODELS_DIR` | model storage root (default `<exe_dir>/models`); also settable per-run via `--models-dir` |
| `KERYX_PURGE_LEGACY_MODELS` | (HiveOS) `1` = h-run.sh force-removes a leftover in-package model cache after merging it into the shared one |
| `GLSLC` / `VULKAN_SDK` | (build only) locate `glslc` for shader compilation |

On HiveOS the models are kept in the shared `/hive/miners/custom/models` cache (exported as
`KERYX_MODELS_DIR` by `h-run.sh`), so package upgrades no longer re-download them. **Upgrading a
rig from ≤ v0.3.12:** the old layout keeps models *inside* the package dir, which HiveOS deletes
when the Install URL changes — run this once on the rig first to move them out:

```bash
wget -qO- https://raw.githubusercontent.com/shmutalov/keryx-miner-rdna3/rdna3/integrations/hiveos/pre-hive-upgrade.sh | bash
```

(Skipping it only costs a one-time re-download; v0.3.13+ migrates any surviving in-package cache
automatically at every start.)

---

## Status & limitations

- ✅ **Verified on a 7900 XT:** both compute kernels are bit-exact against the host/`keccak`
  references; the full miner builds clean and the integrated Vulkan probe detects the GPU.
- ✅ **Live PoM mining** has been exercised end-to-end against the post-hardfork mainnet
  (keryx-node **v1.2.8**, OPoI v2 / PoM active at DAA `37,780,000`): the Dolphin-8B tier mines at
  **~17–18 MH/s** on a 7900 XT and the pool **accepts** the submitted PoM proofs.
- ✅ **PoM weight blob is device-local (VRAM).** It is staged into ≤1 GiB device-local shards
  reached by buffer-device-address — required because an 8B model's ~4.6 GiB blob exceeds AMD's
  4 GiB `maxStorageBufferRange` and 2 GiB `maxMemoryAllocationSize`. Random reads hit GDDR6, not
  PCIe. Each GPU dispatch is also bounded to avoid the Windows TDR watchdog.
- ✅ **Pool shares** work: the Vulkan kHeavyHash kernel applies `nonce_mask`/`nonce_fixed` for the
  pool's extranonce sub-range, and the stratum client handles short-notify DAA inheritance so PoM
  stays active on Short-only pools post-fork.
- ✅ **VRAM capability gate runs on AMD:** total VRAM is queried via Vulkan (the largest
  device-local heap), not `nvidia-smi`, so the model-vs-VRAM filter announces only the tiers your
  card can actually serve (a 7900 XT comfortably runs `--light` and the default tier).
- ⚠️ **Pool version gate:** some pools (suprnova) reject PoM shares from miners that don't
  advertise a recent enough `keryx-miner-supr` version in `mining.subscribe` (the floor has moved
  with each hardfork — `0.7.0+` at H4, `0.9.2+` at H5.2, `0.12.0+` at H10); this fork advertises a
  current identity (`0.13.3`) and subscribes as `keryx-stratum-v3`, so notifies carry the block bits
  and a share that solves the block is never dropped under a high pool difficulty. Single string in
  `client/stratum.rs`.
- ⚠️ **Known benign:** an occasional panic in `MinerManager`'s shutdown/reconnect path (a worker
  thread exits before the drop-time join) — harmless under a supervised restart loop; a clean-up
  candidate.

---

## Connect

* **Website:** [keryx-labs.com](https://keryx-labs.com)
* **X (Twitter):** [@Keryx_Labs](https://x.com/Keryx_Labs)
* **Discord:** [Join the Community](https://discord.gg/U9eDmBUKTF)

---

## Dev Fund

**Disabled in this fork** — `devfund_percent` is forced to `0`, so no blocks are diverted and
**100% of mining rewards go to the miner**. (The upstream `--devfund-percent XX.YY` flag still
parses but is overridden.)
