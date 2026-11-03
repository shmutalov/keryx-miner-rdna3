# Keryx Miner — RDNA3 / Vulkan fork

A fork of [keryx-miner](https://github.com/Keryx-Labs/keryx-miner) that runs **entirely on AMD
RDNA3 GPUs (e.g. Radeon RX 7900 XT/XTX) via Vulkan** — no CUDA, no OpenCL, no CPU compute. It
combines GPU PoW (kHeavyHash), GPU Proof-of-Model possession mining (PoM), and on-chain AI
inference (OPoI — Optimistic Proof of Inference).

> Upstream is NVIDIA/CUDA-only (the build needs `nvcc`). This fork replaces every GPU compute path
> with Vulkan and runs the OPoI models on **llama.cpp (Vulkan), linked into the miner**, so it
> builds and runs on an AMD-only host.

Current release: mines the **H15** consensus (keryx-node v1.6.6, live on mainnet since DAA
140,115,000 on 2026-10-30) and serves **traced AI answers through a pool**. It can report to a
**[Telegram bot](#telegram-bot)**: alerts as they happen, stats on `/stats`.

---

## What runs where

| Workload | Backend | Notes |
|---|---|---|
| **PoW** (kHeavyHash) | Vulkan compute shader | `keccak-f1600 → 64×64 matmul → wave_mix → keccak`. Bit-exact vs the host reference, verified on a 7900 XT. |
| **PoM** (possession walk) | Vulkan compute shader | walk over the model weights resident in **device-local VRAM**, sharded across ≤1 GiB buffers reached by buffer-device-address (an 8B+ model's blob exceeds AMD's 4 GiB SSBO range and 2 GiB single-allocation limits). Bit-exact vs `pom::walk_final`, verified on a 7900 XT. |
| **OPoI inference** (Qwen3.5-9B / GLM-4-9B / gemma-4-12B / Qwen3.8-27B / Kimi-Linear-48B) | **llama.cpp Vulkan, in-process** | linked into the miner (all layers on GPU); the PoM walk shares the engine's resident weight buffers (zero-dup) — no child process, no HTTP. |
| **Verifiable inference** (H15 trace) | llama.cpp Vulkan + keryx shaders | commits every weight product, the KV-cache rows, softmax masses and delta-rule states of an answer, replays it for the pool's openings (tiers 0, 1, 3). |
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

| Flag | Model | Tier | Reward | Min VRAM | Fits a 7900 XT (20 GB)? | Traced pool answers |
|------|-------|------|--------|----------|--------------------------|---------------------|
| `--very-light` | Qwen3.5-9B-abliterated (Q5_K_M) | 0 | 60% | 8 GB | ✅ | ✅ |
| `--light` | GLM-4-9B-0414 (Q6_K) | 1 | 70% | 12 GB | ✅ | ✅ |
| *(default)* | gemma-4-12B-it-abliterated (Q6_K) | 2 | 80% | 16 GB | ✅ | no |
| `--high` | Qwen3.8-27B (Q4_K) | 3 | 90% | 24 GB | no (since H15) | ✅ |
| `--very-high` | Kimi-Linear-48B (Q4_K_M) | 4 | 100% | 30 GB | no | no |

On a 7900 XT, **`--light` is the recommended tier** since H15: tier 3 plus its context pad no
longer fits 20 GB, and tier 2 (gemma) cannot answer traced pool requests with this build.

> Under PoM, **1 GPU = 1 tier**: each tier proves possession of and serves exactly the single
> model above — a PoM GPU is bound to the one model whose weights are resident in VRAM. The
> miner checks the GGUF it indexes against the node-pinned `(R_T, N)` anchor of its tier and
> refuses to mine on a mismatch. With H14 live, `--high` stages only Qwen3.8-27B; the retired
> `models/Qwen3.6-27B` folder can be deleted.

**Tier at H15 (`--h15-tier`).** Since the H15 gate the PoM v5 walk also reads the tier's context
pad, which stays in VRAM next to the weights: 0.6 GiB for tier 0, 6.2 GiB for tier 3. Qwen3.8-27B
plus its pad no longer fits a 20 GB card. `--h15-tier <tier>` mines and serves the given tier from
the gate on, whatever the tier flag says. Now that the gate has passed, pass the tier flag itself
(e.g. `--light`): `--high --h15-tier light` still works but also downloads Qwen3.8-27B (16.8 GB)
for the era before the gate. The v5 walk runs at about 1.6 M nonces/s on a 7900 XT (tiers 0 and 1).

### Consensus this build mines

The rules of **keryx-node v1.6.6**. **PoM v5** (H15, below) builds on the v4 matrix walk (the D=32
re-walk, mainnet DAA 79,210,000) with the **H10 keccak walk seed** (mainnet DAA 87,360,000; both
from DAA 1 on testnet), taken since **H14** (private inference, mainnet DAA 121,985,000; testnet
6,000) over the H14-tagged pre-PoW hash. Each nonce walks a 32×32 int8 state through 256 chained
1 KB tiles of the tier's weights and context pad on the GPU (`keryx-vulkan/shaders/`); every winner
is re-walked on the host, which builds the proof (all 256 tiles + their Merkle range proofs under the
tier's `R_T`) and self-verifies it before submitting. The header carries `pomFinalState`, `pomTier`
and the node's `serviceStateHash` (H6). Older eras (the v1/v2 hash walk, the H6 v3 matrix walk) are
history on every network.

**Private inference (H14).** Every AiRequest is sealed to its tier's provider cohort. A solo miner
opens the ones addressed to its escrow key, answers, and seals the answer back to the requester; the
sealed body rides inline in the signed AiResponse, so IPFS (kubo) is no longer needed past the gate.
Pool miners get the opened prompt from the pool (`mining.ai_request`), and legacy notify-borne
inference tasks are ignored past the gate.

**H15 (verifiable inference; mainnet DAA 140,115,000, reached 2026-10-30; testnet 3,500)** — the
rules of keryx-node v1.6.6:
- **PoM v5**: the walk also reads the tier's context pad (`pom_v5`), which the GPU fills from its
  seed. A share of the steps lands on pad tiles, and each next offset folds in the walk state.
- **No block task**: the node dropped it from H15 (`block_task_activation: never`). Headers keep the
  H14 layout, and pool shares and solo blocks need no trace. The block-task code stays dormant.
- **Verifiable inference.** A traced answer commits the trace of its forward pass, with the prompt
  committed in 512-token batches (inputs in CUDA's MMQ layouts past 8 tokens), and under the
  **attention rule** (from the same gate on both networks) its sequence leaves: the q8_0 key and
  value cache rows of every attention layer and the softmax mass of every query head, and for the
  gated delta-rule layers of hybrid models the per-head records of the delta rule (q, k, v, output,
  beta, decay) and the state every 32 positions. This build commits them on Vulkan for **tiers 0, 1
  and 3 (Qwen3.5-9B, GLM-4-9B, Qwen3.8-27B)**: it computes in 8-token batches and commits each input in
  the layout of its 512-token batch (Q4_K/Q5_K products of such a batch take CUDA's MMQ arithmetic,
  f16 tile scales), reads the cache rows back from llama.cpp's KV cache (rotation off), the masses
  from a shader that recomputes them after every flash-attention node, and the delta-rule records
  and states from a shader after every gated-delta-net node, hashed batch by batch. Before it serves
  a traced answer it runs a self-check on the GPU: a traced answer, then its openings (trace,
  attention phase, recurrent point on the hybrid tiers, residual) under two seeds, checked with the
  node's own verifier. Tier 2 (sliding windows, value norms) and tier 4 (latent cache, key-wise
  decays) are not traced.
- **Traced pool answers** (keryx-miner-supr 0.15.3's pool protocol): a pool asks for a traced answer
  with the optional 8th `mining.ai_request` parameter; the miner answers with the trace commitment
  and the answer body for the pool to seal and sign, keeps the trace for 9 h
  (`keryx-pool-traces/` beside the executable, or `KERYX_POOL_TRACE_DIR`) and builds the openings
  when the pool sends `mining.ai_opening`, replaying the answer on the GPU. Past the gate the miner
  runs the self-check once at its first job (mining pauses meanwhile) and reports `"trace": 1` in
  its `mining.hello`/`mining.telemetry` once it passed. As suprnova's miner does, it declares
  (`mining.declare_capabilities`) only the models it can serve: past the gate, only those whose
  self-check passed; the set goes out again on every change, as an empty list when nothing is
  left, and every 90 s.
- **Solo AI answers stay skipped past the gate**: solo audits (attention phases through the node,
  deadlines) are not wired up.
- **Solo mining past the gate is not viable with this build.** Every request is audited against the
  whole cohort of its tier (each identity with a proven block of that tier in the last ~5 minutes),
  and every member that does not answer in time takes a service strike, at most one an hour: 5 escrow
  claims burned, then all pending escrow, then a 24 h suspension of the miner cut (blocks stay valid
  but pay nothing). Mine through a pool: the pool is the cohort identity there, and answers for it.

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

### Telegram bot

The miner can report to a Telegram chat. Solo mining has no pool dashboard, and a pool's dashboard
does not show the service bond or the AI requests. Alerts arrive as things happen; the numbers come
when you ask with `/stats`.

#### Set up

1. In Telegram, open [@BotFather](https://t.me/BotFather), send `/newbot` and pick a name. It
   answers with the bot token (`123456789:AAE…`).
2. Open your new bot and send it any message (e.g. `/start`), so the miner can find your chat.
3. Copy `telegram.conf.example` to `telegram.conf` next to the launcher (the miner's working
   directory) and put the token in it.
4. Start the miner. It logs the chats that wrote to the bot:
   ```
   Telegram: chats that wrote to the bot: 123456789 (@you) — put yours in telegram.conf as `chat = <id>` and restart the miner
   ```
5. Put your id in as `chat = 123456789` and restart. The bot sends **▶️ Miner started**.

A group works too: add the bot to the group, send `/start` there, and use the group's id (negative,
e.g. `-1001234567890`).

```ini
token = 123456789:AAE...           # from @BotFather
chat = 123456789                   # your chat id, or a group's
ai = 0                             # AI messages: 0 = off, 1 = on, all = solo private ones too
# name = rx7900xt                  # rig name in messages (default: the computer name)
# api = https://api.telegram.org   # Bot API base URL (for a local Bot API server)
```

`telegram.conf` is git-ignored, and the token is masked in the miner's log. With no file, or no
token in it, notifications are off. `--telegram-config <FILE>` reads it from elsewhere. On HiveOS,
where an upgrade replaces the package directory, keep it in `/hive-config` and add
`--telegram-config /hive-config/keryx-telegram.conf` to the flight sheet's extra config arguments.

Use one bot per rig: only one program at a time can read a bot's messages, so two rigs on one bot
take turns answering `/stats` (alerts still arrive from both).

#### Commands

| Command | Answer |
|---|---|
| `/stats` (or `/status`) | the miner's numbers, below |
| `/help` (or `/start`, or any other command) | what the bot reports |

The bot answers only the chat in `telegram.conf`; commands from any other chat are ignored (and
logged once). `/stats` on a solo rig:

```
⛏ RIG · solo · up 5h12m
Hashrate   1.61 MH/s
  average  1.58 MH/s
Network    591.01 MH/s
  this rig 0.27%
  reward   5.02 KRX
  DAA      141,032,821

Blocks     3 (+1)
  rejected 0
Balance    1,234.56 KRX (+2.40)
  session  +12.40
  UTXOs    192
Escrow     2.30 KRX pending
  outputs  7
  claimed  4.10 KRX (2)

AI         7 (+4)
  answered 0
  missed   7
Bond       strike 2
  lifetime 3 strikes
Node       up 3h00m
average and (+n): last 1h00m
⚠️ Service bond: strike 2 (last at daa 140200000)
```

- `average` and the `(+n)` changes cover the last hour, or the time since start if that is shorter.
  `session` is the balance change since the miner started.
- A pool rig shows `Shares` (accepted, rejected, stale) instead of `Blocks`, and no `Escrow` or
  `Bond`: the pool holds those.
- Balance, UTXOs and the network numbers come from the explorer API at `--recover-escrow-api`
  (default `https://keryx-labs.com`), which is sent the payout address. The other counters cover
  the current process, so a restart resets them; the lifetime strike count comes from the node.
- Anything that needs a look gets a ⚠️ line under the table: a strike, the node or pool down, a
  failed balance read.

#### Alerts

They arrive on their own, with a sound. There are no periodic messages: a PC that lost power sends
nothing, so ask `/stats` when things are quiet.

| Alert | When |
|---|---|
| ▶️ Miner started, ⏹️ Miner stopping | on start (a launcher restart after a crash shows up as a new start) and on a clean stop |
| ⚠️ Service bond, ✅ Service bond clear | the node reports a new strike, burn or suspension against the payout address, or clears it |
| ⚠️ Hashrate at 0, ✅ Hashrate back | no hashes for 5 minutes outside inference pauses |
| ⚠️ Node lost, ✅ Node back (Pool on a pool rig) | the connection stays down for 2 minutes |
| ⚠️ Block rejected | the node refuses a solo block (at most one alert an hour; `/stats` keeps the count) |

#### AI messages

With `ai = 1` the bot also sends each AI request the miner serves: the prompt and the answer, in
collapsible quotes, as silent messages (the prompt cut at 1,400 characters, the answer at 2,200).
Solo private requests stay out unless `ai = all`. A pool hands over opened prompts without saying
which were private, so every pool request is sent. Solo requests this build skips past the H15
attention gate are sent with "not answered"; node and pool challenges are not sent. **Telegram keeps
what it is sent**: leave this off if the requesters' prompts must stay on the rig.

Telegram being slow or unreachable never holds up mining: messages wait in a small queue and are
dropped when it is full (AI messages first). `HTTPS_PROXY` / `ALL_PROXY` are honoured.

### All options

```bash
./keryx-miner-rdna3 --help
```

### Useful environment variables

| Var | Meaning |
|-----|---------|
| `KERYX_POM_V4_BATCH` | nonces per PoM dispatch (default `32768`) |
| `KERYX_POM_V4_KERNEL` | `dot` = run the PoM walk's int8 matrix product on packed dot products instead of the matrix cores (the default wherever the GPU has int8 cooperative matrices and they pass a startup check against the dot-product kernel — RDNA3+; ~+35% on a 7900 XT) |
| `KERYX_POW_ONLY` | (testing) `1` = skip OPoI models + inference; every block is PoM, which needs the model, so nothing is mined |
| `RUST_LOG` | log filter (`-d` = everything at debug). llama.cpp's own output logs under the `llama.cpp` target at debug: `RUST_LOG=info,llama.cpp=debug` brings back just that |
| `KERYX_LOG_AI` | `1` = append the prompt and answer of every inference, private requests included, to `ai-requests.log` next to the miner; any other value is the file's path (same as `--log-ai[=FILE]`) |
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

- ✅ **Mainnet past H15** (keryx-node v1.6.6): suprnova accepted the miner (tested as
  `keryx-miner-supr/0.15.5`) and its shares; on a 7900 XT tier 1 (GLM-4-9B) mines at about 1.6 M nonces/s with the v5 pad.
- ✅ **Traced answers** pass the node's own verifier on the GPU: the self-check (trace, then its
  openings under two seeds) passes for tiers 0, 1 and 3. End to end, traced pool answers and their
  openings were tested against a stand-in pool speaking suprnova's protocol.
- ✅ **Solo past H15 on a local testnet** (keryx-node v1.6.6): 4,272 blocks accepted across the
  gate, including the automatic switch from tier 3 to tier 1, none rejected.
- ✅ **Both compute kernels are bit-exact** against the host/`keccak` references, and every PoM
  winner is re-walked and self-verified on the host before it is submitted.
- ✅ **PoM weight blob is device-local (VRAM).** It is staged into ≤1 GiB device-local shards
  reached by buffer-device-address — required because an 8B model's ~4.6 GiB blob exceeds AMD's
  4 GiB `maxStorageBufferRange` and 2 GiB `maxMemoryAllocationSize`. Random reads hit GDDR6, not
  PCIe. Each GPU dispatch is also bounded to avoid the Windows TDR watchdog.
- ✅ **Pool shares** work: the Vulkan kHeavyHash kernel applies `nonce_mask`/`nonce_fixed` for the
  pool's extranonce sub-range.
- ✅ **VRAM capability gate runs on AMD:** total VRAM is queried via Vulkan (the largest
  device-local heap), not `nvidia-smi`, so the model-vs-VRAM filter announces only the tiers your
  card can actually serve.
- ⚠️ **Solo mining past H15 is not viable with this build:** solo AI requests go unanswered, and
  each miss is a service strike ([details](#consensus-this-build-mines)). Mine through a pool.
- ⚠️ **Pool version gate:** suprnova refuses miners that don't advertise a recent enough
  `keryx-miner-supr` version (the floor moves with each hardfork — `0.7.0+` at H4, `0.12.0+` at
  H10, `0.14.0+` at H14, above `0.15.1` at H15). This fork advertises `0.15.8` and subscribes as
  `keryx-stratum-v3`. Single string in `client/stratum.rs`.
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
