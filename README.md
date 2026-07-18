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
| **OPoI inference** (Dolphin-8B / Qwen3-32B / Gemma / Llama) | **llama.cpp Vulkan, in-process** | linked into the miner (all layers on GPU); the PoM walk shares the engine's resident weight buffers (zero-dup) — no child process, no HTTP. |
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

### Inference tiers (OPoI)

| Flag | Models | Min VRAM | Fits a 7900 XT (20 GB)? |
|------|--------|----------|--------------------------|
| `--light` | Gemma-3-4B | 4 GB | ✅ |
| *(default)* | Dolphin-Llama3-8B | 8 GB | ✅ |
| `--high` | Qwen3-32B (Q4_K_M) | 24 GB | tight / no |
| `--very-high` | Llama-3.3-70B | 48 GB | no |

> Post-hardfork (OPoI v2 / PoM), **1 GPU = 1 tier**: each tier proves possession of and serves
> exactly the single model above — the cumulative "serve everything below my tier" behaviour is
> dropped, because a PoM GPU is bound to the one model whose weights are resident in VRAM.

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
- ⚠️ **Pool version gate:** some pools (suprnova) reject post-fork PoM shares from miners that don't
  advertise `keryx-miner-supr/0.6.3+` in `mining.subscribe`; this fork advertises a compatible
  identity so its (valid) proofs are accepted.
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
