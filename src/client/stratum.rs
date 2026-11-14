use futures::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;
use tokio_util::codec::Framed;

mod ai;
mod statum_codec;
mod traces;

/// How long an unacknowledged `mining.ai_response` holds off further requests. Without a limit a
/// pool that never acknowledges one (an id-less reply, a lost line) would get every later request
/// on the connection refused as busy.
const AI_ACK_TIMEOUT: Duration = Duration::from_secs(30);

/// A `mining.ai_response` awaiting the pool's ack.
struct PendingAiResponse {
    /// Its stratum id.
    id: u32,
    /// When its request arrived.
    received: Instant,
    /// When it went out.
    sent: Instant,
    task_id: String,
    /// Request hash of a traced answer, whose trace is kept for openings.
    trace_key: Option<String>,
}

impl PendingAiResponse {
    /// How long the answer took and how long the pool took to ack it, for the ack log line.
    fn timing(&self) -> String {
        format!(
            "answered {:.1} s after the request, acked {} ms after the send",
            self.sent.duration_since(self.received).as_secs_f64(),
            self.sent.elapsed().as_millis()
        )
    }
}

use crate::client::stratum::statum_codec::{ErrorCode, MiningNotify, MiningSubmit, NewLineJsonCodecError, StratumLine};
use crate::client::stratum::statum_codec::{
    InferenceRequestParams, InferenceResultParams, MiningSubscribe, SetExtranonce, StratumCommand, StratumError,
    StratumLinePayload, StratumResult,
};
use crate::client::Client;
use crate::pow::BlockSeed;
use crate::pow::BlockSeed::PartialBlock;
use crate::{miner::MinerManager, Error, Uint256};
use async_trait::async_trait;
use futures_util::TryStreamExt;
use log::{debug, error, info, warn};
use num::Float;
use rand::{thread_rng, RngCore};
use statum_codec::NewLineJsonCodec;
use std::sync::OnceLock;
use tokio::sync::mpsc::{self, Sender};
use tokio::sync::Mutex;
use tokio::task;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tokio_stream::wrappers::ReceiverStream;

//const DIFFICULTY_1_TARGET: Uint256 = Uint256([0x00000000ffff0000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000]);
const DIFFICULTY_1_TARGET: (u64, i16) = (0xffffu64, 208); // 0xffff 2^208
// v3: notifies carry the block's compact bits (v2 forms are still decoded for older bridges).
const KERYX_STRATUM_DAA_CAPABILITY: &str = "keryx-stratum-v3";
const LOG_RATE: Duration = Duration::from_secs(30);
const CHALLENGE_MAX_TOKENS: usize = 128;
// Pool chat (`mining.inference_request`) bounds — the same limits keryx-miner-supr enforces.
const MAX_CHAT_REQ_ID_BYTES: usize = 256;
const MAX_CHAT_PROMPT_BYTES: usize = 4 * 1024;
const MAX_CHAT_TOKENS: usize = 2_048;
const MAX_CHAT_DEADLINE_MS: u64 = 120_000;
/// Seconds between `mining.telemetry` reports (suprnova's default; its floor is 30 s).
const TELEMETRY_INTERVAL_SECS: u64 = 120;
/// Longest a pool opening waits in the queue for the inference card.
const OPENING_QUEUE_TIMEOUT: Duration = Duration::from_secs(600);
/// Openings queued at most.
const MAX_QUEUED_OPENINGS: usize = 32;

/// The traced self-check under the attention rule has been started in this process (once).
static TRACE_SELF_TEST_STARTED: AtomicBool = AtomicBool::new(false);
/// An unchanged capability set is declared again after this long. The declaration has no pool ack,
/// so one lost to a startup race or a bridge registry reset would otherwise stand until the set
/// changes (suprnova's `REDECLARE_EVERY`).
const REDECLARE_EVERY: Duration = Duration::from_secs(90);

/// Whether `ids` must be declared: on a new connection (even when empty, which clears what the
/// bridge kept from an earlier session), on any change (nonempty to empty included, or the pool
/// keeps routing requests for a model we no longer serve), and when the last declaration is stale.
fn redeclare_due(declared: Option<&[String]>, ids: &[String], since_last: Option<Duration>) -> bool {
    declared != Some(ids) || since_last.map_or(true, |t| t >= REDECLARE_EVERY)
}

// ── Phase 2 OPoI — inference cache & task types ─────────────────────────────

/// AiRequest task dispatched by the bridge in a `mining.notify` 5th parameter (JSON).
#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
struct AiTask {
    #[serde(default)]
    stable_id: String,
    model_id_hex: String,
    prompt: String,
    max_tokens: usize,
    #[serde(default)]
    inference_reward: u64,
    #[serde(default)]
    request_hash: String,
}

/// Task attached to the current mining job, cleared on each new `mining.notify`.
struct CurrentTask {
    job_id: String,
    task: AiTask,
}

/// Shared inference result cache — persists across block changes so that if the
/// same AiRequest is included in multiple consecutive job templates the miner can
/// immediately submit with a CID once inference completed for the first occurrence.

/// Max cached inference results — evict when full to prevent unbounded growth.
const MAX_INFERENCE_CACHE_SIZE: usize = 1_000;

struct InferenceCacheInner {
    /// stable_id → base58 CIDv0 string returned by IPFS after upload.
    results: HashMap<String, String>,
    /// stable_ids currently being inferred (guards against duplicate spawn_blocking calls).
    in_progress: HashSet<String>,
}

type InferenceCache = Arc<Mutex<InferenceCacheInner>>;

type BlockHandle = JoinHandle<()>;

#[derive(Default)]
pub struct ShareStats {
    pub accepted: AtomicU64,
    pub stale: AtomicU64,
    pub low_diff: AtomicU64,
    pub duplicate: AtomicU64,
    pub invalid_opoi: AtomicU64,
    pub invalid_pom: AtomicU64,
    /// Shares refused without a recognised reason: a `false`/`null` result, or any error for a
    /// base64 proof.
    pub rejected_other: AtomicU64,
    // std::sync::Mutex (NOT tokio): every critical section is a tiny synchronous map op
    // (insert/remove/len) that is never held across an `.await`, and `Display::fmt` (a sync
    // context) also needs it. The old tokio `try_lock().unwrap()` panicked the moment two tasks
    // touched the map at once (submit insert vs. accept/reject remove vs. periodic len), which
    // killed the submit task and produced an endless "channel closed" flood.
    pub shares_pending: std::sync::Mutex<HashMap<u32, String>>,
}

static SHARE_STATS: OnceLock<Arc<ShareStats>> = OnceLock::new();

/// This process's pool share counters; `None` until a pool connection was made.
pub fn share_stats() -> Option<Arc<ShareStats>> {
    SHARE_STATS.get().cloned()
}

/// Subscribe-result entry by which a pool announces that it accepts base64 proofs.
const PROOF_ENCODING_KEY: &str = "keryx.proof_encoding";
const PROOF_ENCODING_B64: &str = "b64";
/// Tag that marks a base64 proof in `mining.submit` params[5]; hex never contains ':'.
const PROOF_B64_PREFIX: &str = "b64:";

/// Per-connection proof encoding for `mining.submit` params[5] (ported from keryx-miner-supr
/// 0.15.1).
///
/// Hex is the default and what every pool understands. A pool that lists
/// `["keryx.proof_encoding","b64"]` in its `mining.subscribe` result accepts `"b64:" + base64`
/// instead (RFC 4648 standard alphabet, padded), a third smaller: ~650 KB instead of ~975 KB per
/// H15 proof. Any error for a base64 submit falls back to hex for that pool for the rest of the
/// process, so a mis-advertising pool can never strand the miner.
#[derive(Default)]
struct ProofEncoding {
    /// Pool address of this connection; a base64 rejection is remembered per address.
    pool: String,
    b64: AtomicBool,
    b64_ids: std::sync::Mutex<HashSet<u32>>,
}

/// Pools that rejected a base64 submit during this process. They are only ever sent hex from then
/// on, even after a reconnect re-advertises base64: a pool that advertises but rejects (with a
/// code that forces a reconnect, say) would otherwise put the miner in a reconnect loop.
static B64_REFUSED_BY: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

impl ProofEncoding {
    fn for_pool(pool: &str) -> Self {
        ProofEncoding { pool: pool.to_string(), ..Default::default() }
    }

    fn refused_here(&self) -> bool {
        B64_REFUSED_BY.lock().map(|v| v.iter().any(|p| p == &self.pool)).unwrap_or(false)
    }

    /// Called with the subscription list of every `mining.subscribe` result.
    fn absorb_subscribe(&self, subscriptions: &[(String, String)]) {
        let offered = subscriptions.iter().any(|(k, v)| k == PROOF_ENCODING_KEY && v == PROOF_ENCODING_B64);
        if offered && self.refused_here() {
            info!("Pool advertises base64 proofs but rejected one earlier in this session — keeping hex proofs.");
            return;
        }
        if offered && !self.b64.swap(true, Ordering::SeqCst) {
            info!("Pool accepts base64 proofs ({}={}) — submitting proofs base64-encoded on this connection.", PROOF_ENCODING_KEY, PROOF_ENCODING_B64);
        }
    }

    /// Encodes a proof for params[5] and remembers base64 submit ids for the fallback.
    fn encode(&self, msg_id: u32, proof: &[u8]) -> String {
        if !self.b64.load(Ordering::SeqCst) {
            return hex::encode(proof);
        }
        if let Ok(mut ids) = self.b64_ids.lock() {
            if ids.len() > 4096 {
                ids.clear(); // unanswered ids from a pool that never replies; bound the set
            }
            ids.insert(msg_id);
        }
        let mut out = String::with_capacity(PROOF_B64_PREFIX.len() + proof.len().div_ceil(3) * 4);
        out.push_str(PROOF_B64_PREFIX);
        base64::Engine::encode_string(&base64::engine::general_purpose::STANDARD, proof, &mut out);
        out
    }

    /// A reply for `msg_id` arrived. Returns true if it was a base64 submit.
    fn answered(&self, msg_id: u32) -> bool {
        self.b64_ids.lock().map(|mut ids| ids.remove(&msg_id)).unwrap_or(false)
    }

    /// An error for a base64 submit: hex for the rest of this connection and for every later
    /// connection to the same pool in this process.
    fn reject_b64(&self, reason: &str) {
        if let Ok(mut v) = B64_REFUSED_BY.lock() {
            if !v.iter().any(|p| p == &self.pool) {
                v.push(self.pool.clone());
            }
        }
        if self.b64.swap(false, Ordering::SeqCst) {
            warn!("Pool refused a base64 proof submit ({reason}) — using hex proofs for this pool from now on.");
        }
    }
}

/// Last real DAA score from a job that carries one (WithTask / ShortV2). The plain `Short` notify
/// carries NO daa_score; pinning it to the SALT-v4 era (21,932,751) leaves it BELOW PoM activation
/// (37,780,000), so a Short-only pool never crosses `daa >= POM_ACTIVATION_DAA` — the miner keeps
/// hashing kHeavyHash post-fork and submits an empty/invalid PoM proof the pool rejects. Remembering
/// the last real daa lets `Short` inherit it so PoM activates. (Ported from keryx-miner-supr v0.6.3.)
static LAST_DAA_SCORE: AtomicU64 = AtomicU64::new(0);

/// Share target for a job: the pool target, or the block's own target when that is easier (a v3
/// notify carries the block bits). With the pool difficulty set above the network's — right after
/// a relaunch difficulty reset — a nonce that solves the block would otherwise be discarded.
fn effective_target(pool_target: Uint256, block_bits: Option<u32>) -> Result<Uint256, Error> {
    let Some(bits) = block_bits else {
        return Ok(pool_target);
    };
    let size = bits >> 24;
    let mantissa = bits & 0x007fffff;
    if bits & 0x00800000 != 0 || size > 34 || (size > 33 && mantissa > 0xff) || (size > 32 && mantissa > 0xffff) {
        return Err("Invalid block target".into());
    }
    let target = crate::target::u256_from_compact_target(bits);
    if target == Uint256::default() {
        return Err("Invalid block target".into());
    }
    Ok(pool_target.max(target))
}

impl Display for ShareStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Shares: {}{}{}{}{}{}{}Pending: {}",
            match self.accepted.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Accepted: {} ", v),
            },
            match self.stale.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Stale: {} ", v),
            },
            match self.low_diff.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Low difficulty: {} ", v),
            },
            match self.duplicate.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Duplicate: {} ", v),
            },
            match self.invalid_opoi.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Invalid OPoI tag: {} ", v),
            },
            match self.invalid_pom.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Invalid PoM proof: {} ", v),
            },
            match self.rejected_other.load(Ordering::SeqCst) {
                0 => "".to_string(),
                v => format!("Rejected: {} ", v),
            },
            self.shares_pending.lock().unwrap().len()
        )
    }
}

#[allow(dead_code)]
pub struct StratumHandler {
    log_handler: JoinHandle<()>,

    //client: Framed<TcpStream, NewLineJsonCodec>,
    send_channel: Sender<StratumLine>,
    stream: Pin<Box<dyn Stream<Item = Result<StratumLine, NewLineJsonCodecError>>>>,
    miner_address: String,
    /// Pool worker name (appended to the authorize username as `address.worker`); empty = none.
    worker: String,
    /// Stratum password sent at mining.authorize (pool difficulty request, e.g. `d=1000`); `x` = default.
    password: String,
    mine_when_not_synced: bool,
    devfund_address: Option<String>,
    devfund_percent: u16,
    mining_dev: Option<bool>,
    block_template_ctr: Arc<AtomicU16>,

    target_pool: Uint256,
    target_real: Uint256,
    nonce_mask: u64,
    nonce_fixed: u64,
    extranonce: Option<String>,
    last_stratum_id: Arc<AtomicU32>,

    shares_stats: Arc<ShareStats>,
    proof_encoding: Arc<ProofEncoding>,
    block_channel: Sender<BlockSeed>,
    block_handle: BlockHandle,

    /// IPFS Kubo API URL for uploading inference results (e.g. "http://127.0.0.1:5001").
    ipfs_url: String,
    /// Task dispatched by the bridge for the current mining job (None = no AiRequest in job).
    current_task_slot: Arc<Mutex<Option<CurrentTask>>>,
    /// Completed inferences: stable_id → base58 CIDv0 string (persists across block changes).
    inference_cache: InferenceCache,
    /// True while a capability challenge inference is in flight — prevents duplicate spawns.
    challenge_in_flight: Arc<AtomicBool>,
    /// The in-flight `mining.ai_response`, until the bridge acknowledges it or [`AI_ACK_TIMEOUT`]
    /// passes.
    ai_response_pending: Arc<Mutex<Option<PendingAiResponse>>>,
    /// Ids of our requests whose answers are no share results: `mining.hello`, `mining.telemetry`
    /// and `mining.ai_opening_result`.
    aux_ids: Arc<std::sync::Mutex<HashSet<u32>>>,
    /// Miner telemetry stays on until the pool refuses `mining.hello`/`mining.telemetry`.
    telemetry_on: bool,
    /// Pool openings waiting for the inference card: the request, its result id, when it came.
    openings: std::collections::VecDeque<(ai::AiOpening, u32, std::time::Instant)>,
    started: std::time::Instant,
    /// The model set last sent in `mining.declare_capabilities` on this connection, and when.
    declared_model_ids: Option<Vec<String>>,
    last_declare_at: Option<std::time::Instant>,
}

#[async_trait(?Send)]
impl Client for StratumHandler {
    fn add_devfund(&mut self, address: String, percent: u16) {
        self.devfund_address = Some(address);
        self.devfund_percent = percent;
    }

    async fn register(&mut self) -> Result<(), Error> {
        let mut id = { Some(self.last_stratum_id.fetch_add(1, Ordering::SeqCst)) };
        self.send_channel
            .send(StratumLine {
                id,
                payload: StratumLinePayload::StratumCommand(StratumCommand::Subscribe(
                    MiningSubscribe::MiningSubscribeOptions((
                        // suprnova's bridge version-gates PoM shares by the reported keryx-miner-supr
                        // version, raising the floor at every hardfork (H4 >= 0.7.0, H5.2 >= 0.9.2,
                        // H10 >= 0.12.0 — the mandatory one-way-seed release, H14 >= 0.14.0). This
                        // build mines the H15 consensus of keryx-node 1.6.6: PoM v5 (context pad) and
                        // no block task, as supr 0.15.1+ does (0.15.0 still hashes a task root, which
                        // makes every post-H15 block invalid), and H14 before the gate — so advertise
                        // the current supr release. Past the gate the pool refused 0.15.1 (2026-10-31,
                        // after supr 0.15.3/0.15.4 shipped traced answers; 0.15.5 makes every Windows
                        // 0.15.4 rig update). supr 0.15.6–0.15.8 changed no wire protocol. supr 0.16.0
                        // is its H16 release (keryx-node 1.6.7: the 200 MiB smaller context pad from
                        // DAA 147,000,000, again no wire change), which this build mines too. supr
                        // 0.16.1–0.16.4 added the busy reply (46), NUL-byte and compressed prompts and
                        // stratum+ssl://, all of which this build has too, so it advertises 0.16.4.
                        // This build serves supr's pool trace
                        // protocol too (traced `mining.ai_request`, `mining.ai_opening`) for tiers 0,
                        // 1 and 3. Bump this string and the `mining.hello` version if the pool raises
                        // the floor again.
                        // (Real build: keryx-miner/CARGO_PKG_VERSION.)
                        "keryx-miner-supr/0.16.4".to_string(),
                        KERYX_STRATUM_DAA_CAPABILITY.into(),
                    )),
                )),
                jsonrpc: None,
                error: None,
            })
            .await?;
        id = Some(self.last_stratum_id.fetch_add(1, Ordering::SeqCst));

        let pay_address = match &self.devfund_address {
            Some(devfund_address) if self.block_template_ctr.load(Ordering::SeqCst) <= self.devfund_percent => {
                self.mining_dev = Some(true);
                info!("Mining to devfund");
                devfund_address.clone()
            }
            _ => {
                self.mining_dev = Some(false);
                self.miner_address.clone()
            }
        };
        self.send_channel
            .send(StratumLine {
                id,
                payload: StratumLinePayload::StratumCommand(StratumCommand::Authorize((
                    // Pool username: `address.worker` (so the pool tags shares per rig), or just
                    // the address when no worker name is set.
                    if self.worker.is_empty() { pay_address.clone() } else { format!("{}.{}", pay_address, self.worker) },
                    self.password.clone(),
                ))),
                jsonrpc: None,
                error: None,
            })
            .await?;

        // Declare the models we can serve so the bridge challenges and routes only those; every
        // job declares again when the set changes (a model finishing its download, the traced
        // self-check passing) or the declaration goes stale.
        self.declare_capabilities_if_changed().await?;
        // Miner telemetry (suprnova's `mining.hello`): the static rig identity and whether this rig
        // can serve traced answers. Best effort: a pool that refuses it turns telemetry off.
        let hello = self.hello();
        self.send_aux(StratumCommand::MiningHello((hello,))).await;
        Ok(())
    }

    async fn listen(&mut self, miner: &mut MinerManager) -> Result<(), Error> {
        info!("Waiting for stuff");
        let mut telemetry = tokio::time::interval(Duration::from_secs(TELEMETRY_INTERVAL_SECS));
        telemetry.set_missed_tick_behavior(MissedTickBehavior::Skip);
        telemetry.tick().await; // the hello went out at register
        loop {
            {
                // Devfund payout rotation: the pool authorizes one payout address per session, so
                // switching between the miner's address and the devfund address requires a reconnect.
                // Drop the session (the supervisor reconnects + re-runs register()) when the currently
                // authorized address no longer matches what block_template_ctr now says it should be.
                // Only relevant when a devfund is configured — without one, devfund_percent == 0 and
                // this would spuriously fire every time the (randomly-seeded, wrapping) counter hits 0.
                if self.devfund_address.is_some()
                    && ((!self.mining_dev.unwrap_or(true)
                        && self.block_template_ctr.load(Ordering::SeqCst) <= self.devfund_percent)
                        || (self.mining_dev.unwrap_or(false)
                            && self.block_template_ctr.load(Ordering::SeqCst) > self.devfund_percent))
                {
                    return Ok(());
                }
            }
            tokio::select! {
                msg = self.stream.try_next() => match msg? {
                    Some(msg) => self.handle_message(msg, miner).await?,
                    None => return Err("stratum message payload is empty".into()),
                },
                // The stream future is dropped once this branch runs; a dropped `try_next` loses
                // nothing (the framed reader keeps its buffer).
                _ = telemetry.tick() => {
                    if self.telemetry_on {
                        let report = self.telemetry_report();
                        self.send_aux(StratumCommand::MiningTelemetry((report,))).await;
                    }
                }
            }
        }
    }

    fn get_block_channel(&self) -> Sender<BlockSeed> {
        self.block_channel.clone()
    }
}

/// Whether `url` names a stratum pool: `stratum+tcp://`, or `stratum+ssl://` / `stratum+tls://`
/// over TLS (upstream keryx-miner dfca69d).
pub fn is_stratum_url(url: &str) -> bool {
    STRATUM_SCHEMES.iter().any(|p| url.starts_with(p))
}

const STRATUM_SCHEMES: [&str; 3] = ["stratum+tcp://", "stratum+ssl://", "stratum+tls://"];

/// TLS options of a stratum connection.
pub struct TlsSettings {
    /// Extra PEM certificate authority trusted besides the public roots (`--pool-ca-file`).
    pub ca_file: Option<String>,
}

trait PoolIo: AsyncRead + AsyncWrite + Send + Unpin {}
impl<T: AsyncRead + AsyncWrite + Send + Unpin> PoolIo for T {}

/// The TLS session over `socket` to the pool at `address` (`host:port`), its certificate checked
/// against the public web roots (webpki-roots) and the extra CA of `settings`.
async fn tls_wrap(socket: TcpStream, address: &str, settings: &TlsSettings) -> Result<Box<dyn PoolIo>, Error> {
    use tokio_rustls::rustls::{self, pki_types::pem::PemObject, pki_types::CertificateDer, pki_types::ServerName, RootCertStore};

    let mut roots = RootCertStore { roots: webpki_roots::TLS_SERVER_ROOTS.to_vec() };
    if let Some(path) = &settings.ca_file {
        let mut added = 0;
        for cert in CertificateDer::pem_file_iter(path).map_err(|e| format!("pool CA file {}: {}", path, e))? {
            roots.add(cert.map_err(|e| format!("pool CA file {}: {}", path, e))?)?;
            added += 1;
        }
        if added == 0 {
            return Err(format!("pool CA file {}: no certificate", path).into());
        }
    }
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions()?
        .with_root_certificates(roots)
        .with_no_client_auth();
    let host = tls_host(address);
    let name = ServerName::try_from(host.to_string()).map_err(|e| format!("pool host {}: {}", host, e))?;
    let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
    let stream = tokio::time::timeout(Duration::from_secs(10), connector.connect(name, socket)).await??;
    Ok(Box::new(stream))
}

/// The host of `host:port` (brackets of an IPv6 literal dropped), the name the certificate must carry.
fn tls_host(address: &str) -> &str {
    address.rsplit_once(':').map_or(address, |(h, _)| h).trim_start_matches('[').trim_end_matches(']')
}

impl StratumHandler {
    #[allow(clippy::too_many_arguments)]
    pub async fn connect(
        address: String,
        tls: Option<TlsSettings>,
        miner_address: String,
        worker: String,
        password: String,
        mine_when_not_synced: bool,
        block_template_ctr: Option<Arc<AtomicU16>>,
        ipfs_url: String,
    ) -> Result<Box<Self>, Error> {
        info!("Connecting to {}{}", address, if tls.is_some() { " (TLS)" } else { "" });
        let proof_encoding = Arc::new(ProofEncoding::for_pool(&address));
        let tcp = TcpStream::connect(&address).await?;
        let socket: Box<dyn PoolIo> = match &tls {
            Some(settings) => {
                let stream = tls_wrap(tcp, &address, settings).await?;
                info!("TLS session established with {}", address);
                stream
            }
            None => Box::new(tcp),
        };

        let client = Framed::new(socket, NewLineJsonCodec::new());
        let (send_channel, recv) = mpsc::channel::<StratumLine>(3);
        let (sink, stream) = client.split();
        tokio::spawn(async move { ReceiverStream::new(recv).map(Ok).forward(sink).await });

        let share_state = SHARE_STATS.get_or_init(|| Arc::new(ShareStats::default())).clone();
        let last_stratum_id = Arc::new(AtomicU32::new(0));
        let current_task_slot: Arc<Mutex<Option<CurrentTask>>> = Arc::new(Mutex::new(None));
        let inference_cache: InferenceCache = Arc::new(Mutex::new(InferenceCacheInner {
            results: HashMap::new(),
            in_progress: HashSet::new(),
        }));
        // Stratum-spec §4.3: mining.submit param[0] must carry the SAME `.<worker>` suffix used at
        // authorize, so the pool credits shares to the right rig. Build it once and hand it to the
        // submit channel (register() does the equivalent for mining.authorize).
        let submit_username =
            if worker.is_empty() { miner_address.clone() } else { format!("{}.{}", miner_address, worker) };
        let (block_channel, block_handle) = Self::create_block_channel(
            send_channel.clone(),
            submit_username,
            last_stratum_id.clone(),
            share_state.clone(),
            proof_encoding.clone(),
            Arc::clone(&current_task_slot),
            Arc::clone(&inference_cache),
        );
        Ok(Box::new(Self {
            log_handler: task::spawn(Self::log_shares(share_state.clone())),
            stream: Box::pin(stream),
            send_channel,
            miner_address,
            worker,
            password,
            mine_when_not_synced,
            devfund_address: None,
            devfund_percent: 0,
            block_template_ctr: block_template_ctr
                .unwrap_or_else(|| Arc::new(AtomicU16::new((thread_rng().next_u64() % 10_000u64) as u16))),
            target_pool: Default::default(),
            target_real: Default::default(),
            nonce_mask: u64::MAX, // full nonce space until set_extranonce assigns a sub-range
            nonce_fixed: 0,
            extranonce: None,
            last_stratum_id,
            shares_stats: share_state,
            proof_encoding,
            mining_dev: None,
            block_channel,
            block_handle,
            ipfs_url,
            current_task_slot,
            inference_cache,
            challenge_in_flight: Arc::new(AtomicBool::new(false)),
            ai_response_pending: Arc::new(Mutex::new(None)),
            aux_ids: Arc::new(std::sync::Mutex::new(HashSet::new())),
            telemetry_on: true,
            openings: std::collections::VecDeque::new(),
            started: std::time::Instant::now(),
            declared_model_ids: None,
            last_declare_at: None,
        }))
    }

    fn create_block_channel(
        send_channel: Sender<StratumLine>,
        submit_username: String,
        last_stratum_id: Arc<AtomicU32>,
        share_stats: Arc<ShareStats>,
        proof_encoding: Arc<ProofEncoding>,
        current_task_slot: Arc<Mutex<Option<CurrentTask>>>,
        inference_cache: InferenceCache,
    ) -> (Sender<BlockSeed>, BlockHandle) {
        let (send, recv) = mpsc::channel::<BlockSeed>(1);

        let handle = tokio::spawn(async move {
            let mut recv_stream = ReceiverStream::new(recv);
            while let Some(seed) = recv_stream.next().await {
                let (nonce, job_id, pom_proof, task_root) = match seed {
                    BlockSeed::PartialBlock { nonce, id, pom_proof, task_root, .. } => (nonce, id, pom_proof, task_root),
                    BlockSeed::FullBlock(_) => unreachable!(),
                };
                let msg_id = last_stratum_id.fetch_add(1, Ordering::SeqCst);
                share_stats.shares_pending.lock().unwrap().insert(msg_id, job_id.clone());
                let nonce_hex = format!("{:016x}", nonce);
                let opoi_tag = keryx_inference::tag_fixed(nonce);

                // Phase 2: check inference cache for the current job's task
                let cid_opt = {
                    let task_guard = current_task_slot.lock().await;
                    if let Some(ref ct) = *task_guard {
                        if ct.job_id == job_id && !ct.task.stable_id.is_empty() {
                            let cache_guard = inference_cache.lock().await;
                            cache_guard.results.get(&ct.task.stable_id).cloned()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };

                let line = if let (false, Some(root)) = (pom_proof.is_empty(), task_root) {
                    // H15: the 6-slot PoM submit plus the block task root the share's header commits.
                    let cid = cid_opt.unwrap_or_default();
                    info!("PoM: submitting share with proof ({} B) and block task for job {}", pom_proof.len(), job_id);
                    StratumLine {
                        id: Some(msg_id),
                        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                            MiningSubmit::MiningSubmitWithPomTask((
                                submit_username.clone(),
                                job_id,
                                nonce_hex,
                                opoi_tag,
                                cid,
                                proof_encoding.encode(msg_id, &pom_proof),
                                hex::encode(root),
                            )),
                        )),
                        jsonrpc: None,
                        error: None,
                    }
                } else if !pom_proof.is_empty() {
                    // PoM (post-fork): fixed 6-slot submit — proof always at params[5], CID-or-empty
                    // at params[4]. The pool relays params[5] -> RpcBlock.pomProof (it does not verify).
                    // Lowercase hex, or "b64:"+base64 when the pool offered it.
                    let proof_hex = proof_encoding.encode(msg_id, &pom_proof);
                    let cid = cid_opt.unwrap_or_default();
                    info!("PoM: submitting share with proof ({} B) for job {}", pom_proof.len(), job_id);
                    StratumLine {
                        id: Some(msg_id),
                        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                            MiningSubmit::MiningSubmitWithPom((
                                submit_username.clone(),
                                job_id,
                                nonce_hex,
                                opoi_tag,
                                cid,
                                proof_hex,
                            )),
                        )),
                        jsonrpc: None,
                        error: None,
                    }
                } else if let Some(cid) = cid_opt {
                    info!("OPoI Phase 2: submitting share with CID for job {}", job_id);
                    StratumLine {
                        id: Some(msg_id),
                        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                            MiningSubmit::MiningSubmitWithCID((
                                submit_username.clone(),
                                job_id,
                                nonce_hex,
                                opoi_tag,
                                cid,
                            )),
                        )),
                        jsonrpc: None,
                        error: None,
                    }
                } else {
                    StratumLine {
                        id: Some(msg_id),
                        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(
                            MiningSubmit::MiningSubmitWithTag((
                                submit_username.clone(),
                                job_id,
                                nonce_hex,
                                opoi_tag,
                            )),
                        )),
                        jsonrpc: None,
                        error: None,
                    }
                };

                if send_channel.send(line).await.is_err() {
                    break;
                }
            }
        });
        (send, handle)
    }

    async fn handle_message(&mut self, msg: StratumLine, miner: &mut MinerManager) -> Result<(), Error> {
        match msg.clone() {
            StratumLine { id, payload, error: None, .. } => {
                match payload {
                    StratumLinePayload::StratumResult { result } if id.is_some() => {
                        match result {
                            StratumResult::Plain(Some(true)) | StratumResult::Eth((true, _)) => {
                                let rid = id.expect("We checked id is not none");
                                if self.take_aux(rid) {
                                    debug!("Pool acknowledged request {rid}");
                                    return Ok(());
                                }
                                self.proof_encoding.answered(rid);
                                let removed = self.shares_stats.shares_pending.lock().unwrap().remove(&rid);
                                if removed.is_some() {
                                    self.shares_stats.accepted.fetch_add(1, Ordering::SeqCst);
                                    info!("Share accepted");
                                } else if let Some(p) = self.take_ai_response_ack(id).await {
                                    info!("AI response {:.16} accepted by the pool ({})", p.task_id, p.timing());
                                    crate::notify::ai_answered();
                                } else {
                                    info!("{:?} (Last: {})", msg.clone(), self.last_stratum_id.load(Ordering::SeqCst));
                                    warn!("Ignoring result for now");
                                }
                                Ok(())
                            }
                            StratumResult::Subscribe((ref subscriptions, ref extranonce, ref nonce_size)) => {
                                self.proof_encoding.absorb_subscribe(subscriptions);
                                self.set_extranonce(extranonce.as_str(), nonce_size)
                                /*for (name, value) in _subscriptions {
                                    match name.as_str() {
                                        "mining.set_difficulty" => {self.set_difficulty(&f32::from_str(value.as_str())?)?;},
                                        _ => {warn!("Ignored {} (={})", name, value);}
                                    }
                                }
                                Ok(())*/
                            }
                            other => {
                                // A non-true result without an error object (`false` / `null`). For a
                                // pending share that is a rejection; anything else is a reply we cannot
                                // attribute. Neither is a reason to drop the connection.
                                let rid = id.expect("We checked id is not none");
                                if self.take_aux(rid) {
                                    debug!("Pool answered request {rid} with {:?}", other);
                                    return Ok(());
                                }
                                if let Some(p) = self.take_ai_response_ack(id).await {
                                    warn!("AI response {:.16} not accepted by the pool ({}): result {:?}", p.task_id, p.timing(), other);
                                    crate::notify::ai_missed();
                                    return Ok(());
                                }
                                let was_b64 = self.proof_encoding.answered(rid);
                                match self.shares_stats.shares_pending.lock().unwrap().remove(&rid) {
                                    Some(jobid) => {
                                        if was_b64 {
                                            self.proof_encoding.reject_b64(&format!("result {:?}", other));
                                        }
                                        self.shares_stats.rejected_other.fetch_add(1, Ordering::SeqCst);
                                        warn!("Share not accepted by pool (Job id: {}): result {:?}", jobid, other);
                                    }
                                    None => warn!("Ignoring unmatched pool result for id {}: {:?}", rid, other),
                                }
                                Ok(())
                            }
                        }
                    }
                    StratumLinePayload::StratumCommand(command) => match command {
                        StratumCommand::SetExtranonce(SetExtranonce::SetExtranoncePlain((
                            ref extranonce,
                            ref nonce_size,
                        ))) => self.set_extranonce(extranonce.as_str(), nonce_size),
                        StratumCommand::MiningSetDifficulty((ref difficulty,)) => self.set_difficulty(difficulty),
                        // H15 (keryx-stratum-v4): the full template header and the window seed of its
                        // block task. The worker commits its own task root in the header it hashes.
                        StratumCommand::MiningNotify(MiningNotify::MiningNotifyV4((id, header_hex, timestamp, daa_score, bits, seed_hex))) => {
                            self.handle_notify_v4(id, header_hex, timestamp, daa_score, bits, seed_hex, miner).await
                        }
                        // Every notify form that carries the real DAA score: v3 (with the block's compact
                        // bits) and v2, each with or without an AiRequest task (Phase 2 OPoI).
                        StratumCommand::MiningNotify(
                            notify @ (MiningNotify::MiningNotifyWithTaskV3(_)
                            | MiningNotify::MiningNotifyShortV3(_)
                            | MiningNotify::MiningNotifyWithTask(_)
                            | MiningNotify::MiningNotifyShortV2(_)),
                        ) => {
                            let (id, header_hash, timestamp, daa_score, block_bits, task_json) = match notify {
                                MiningNotify::MiningNotifyWithTaskV3((id, hash, time, daa, bits, task)) => {
                                    (id, hash, time, daa, Some(bits), Some(task))
                                }
                                MiningNotify::MiningNotifyShortV3((id, hash, time, daa, bits)) => {
                                    (id, hash, time, daa, Some(bits), None)
                                }
                                MiningNotify::MiningNotifyWithTask((id, hash, time, daa, task)) => {
                                    (id, hash, time, daa, None, Some(task))
                                }
                                MiningNotify::MiningNotifyShortV2((id, hash, time, daa)) => (id, hash, time, daa, None, None),
                                _ => unreachable!("matched above"),
                            };
                            self.block_template_ctr
                                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000))
                                .unwrap();
                            // Remember the real daa so plain `Short` notifies can inherit it (PoM gate).
                            LAST_DAA_SCORE.store(daa_score, Ordering::Relaxed);
                            keryx_miner::slm::note_chain_daa(daa_score);
                            self.declare_capabilities_if_changed().await?;
                            // OPoI hard gate (mirrors solo grpc.rs): no models ready = no mining.
                            if !keryx_miner::pow_only() && keryx_miner::slm::loaded_model_ids().is_empty() {
                                if self.block_template_ctr.load(Ordering::SeqCst) % 200 == 0 {
                                    warn!("OPoI: no models ready — mining suspended (no inference = no mining)");
                                }
                                return miner.process_block(None).await;
                            }
                            // As on v4 jobs: past the attention gate only a self-checked model is
                            // declared, so pools on these notify forms (cryo-pool's bridge sends
                            // short v3) need the check started here too.
                            self.drive_openings(miner).await;
                            self.maybe_start_trace_self_test(daa_score, miner).await;
                            // Past the block task gate a v3/v2 job cannot become a valid block.
                            if keryx_miner::block_task::active(daa_score) {
                                keryx_miner::pom_gpu::log_throttled(u32::MAX, "pool-no-task", || {
                                    warn!("Block task: the pool sends jobs without a block task (header and window seed) — no block can be valid, job skipped")
                                });
                                return miner.process_block(None).await;
                            }
                            let target = effective_target(self.target_pool, block_bits)?;
                            let inference_started = match task_json {
                                // PoW-only test mode: ignore the AiRequest task entirely and just mine.
                                Some(_) if keryx_miner::pow_only() => false,
                                // The task path publishes plaintext answers to IPFS; from the
                                // private-inference gate on only the pool can build an answer
                                // (it opens the sealed request and dispatches `mining.ai_request`).
                                Some(_) if daa_score >= keryx_miner::pom::private_inference_activation_daa() => {
                                    warn!("OPoI: pool still dispatches inference tasks; private inference needs a v3 pool (mining.ai_request) — task ignored");
                                    *self.current_task_slot.lock().await = None;
                                    false
                                }
                                Some(task_json) => self.handle_ai_task(id.clone(), task_json, miner).await,
                                None => {
                                    // No AiRequest in this job — clear the task slot.
                                    *self.current_task_slot.lock().await = None;
                                    false
                                }
                            };
                            if inference_started {
                                // PoW already paused inside handle_ai_task — do NOT feed a new
                                // block template or the GPU immediately resumes hashing.
                                Ok(())
                            } else {
                                miner
                                    .process_block(Some(PartialBlock {
                                        id,
                                        header_hash,
                                        timestamp,
                                        daa_score,
                                        nonce: 0,
                                        target,
                                        nonce_mask: self.nonce_mask,
                                        nonce_fixed: self.nonce_fixed,
                                        hash: None,
                                        pom_proof: Vec::new(),
                                        header: None,
                                        task_root: None,
                                    }))
                                    .await
                            }
                        }
                        StratumCommand::MiningNotify(MiningNotify::MiningNotifyShort((id, header_hash, timestamp))) => {
                            self.block_template_ctr
                                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000))
                                .unwrap();
                            self.declare_capabilities_if_changed().await?;
                            // OPoI hard gate (mirrors solo grpc.rs): no models ready = no mining.
                            if !keryx_miner::pow_only() && keryx_miner::slm::loaded_model_ids().is_empty() {
                                if self.block_template_ctr.load(Ordering::SeqCst) % 200 == 0 {
                                    warn!("OPoI: no models ready — mining suspended (no inference = no mining)");
                                }
                                return miner.process_block(None).await;
                            }
                            *self.current_task_slot.lock().await = None;
                            self.drive_openings(miner).await;
                            self.maybe_start_trace_self_test(LAST_DAA_SCORE.load(Ordering::Relaxed), miner).await;
                            miner
                                .process_block(Some(PartialBlock {
                                    id,
                                    header_hash,
                                    timestamp,
                                    // Short stratum notify carries no daa_score. Inherit the last
                                    // real daa (from WithTask/ShortV2) so post-fork PoM still
                                    // activates on Short-only pools; floor at the SALT-v4 era so the
                                    // host still builds the right kHeavyHash matrix before the fork.
                                    daa_score: LAST_DAA_SCORE
                                        .load(Ordering::Relaxed)
                                        .max(crate::pow::heavy_hash::pow_salt_v4_activation_daa()),
                                    nonce: 0,
                                    target: self.target_pool,
                                    nonce_mask: self.nonce_mask,
                                    nonce_fixed: self.nonce_fixed,
                                    hash: None,
                                    pom_proof: Vec::new(),
                                    header: None,
                                    task_root: None,
                                }))
                                .await
                        }
                        StratumCommand::MiningChallenge((model_id_hex, nonce_hex)) => {
                            self.handle_challenge(model_id_hex, nonce_hex, miner).await;
                            Ok(())
                        }
                        StratumCommand::MiningAiRequest(params) => {
                            self.handle_ai_request(id, params.fields, params.ext, miner).await;
                            Ok(())
                        }
                        StratumCommand::MiningAiOpening(params) => {
                            self.handle_ai_opening(params, miner).await;
                            Ok(())
                        }
                        StratumCommand::MiningInferenceRequest(req) => {
                            self.handle_inference_request(req, miner).await;
                            Ok(())
                        }
                        _ => Err(format!("Unexpected stratum message: {:?}", msg).into()),
                    },
                    // A result without an id: some pools answer our id-less notifications (such as
                    // `mining.declare_capabilities`) with `{"id":null,"result":true}`. It answers
                    // nothing we wait for; ignoring it keeps the connection, and mining, alive.
                    StratumLinePayload::StratumResult { result } => {
                        debug!("Ignoring pool result without id: {:?}", result);
                        Ok(())
                    }
                }
            }
            StratumLine {
                id: Some(id),
                payload: StratumLinePayload::StratumResult { .. },
                error: Some(StratumError(code, error, data)),
                ..
            } => {
                if self.take_aux(id) {
                    // A pool without the telemetry extension refuses `mining.hello` (error 20 or
                    // JSON-RPC's method-not-found); stop sending it for this session.
                    if matches!(code, ErrorCode::Unknown | ErrorCode::Other(-32601)) && self.telemetry_on {
                        info!("Pool does not take miner telemetry ({}: {}) — not sent this session", code, error);
                        self.telemetry_on = false;
                    } else {
                        warn!("Pool refused request {}: {}: {}", id, code, error);
                    }
                    return Ok(());
                }
                if let Some(p) = self.take_ai_response_ack(Some(id)).await {
                    warn!(
                        "AI response {:.16} rejected by the pool ({}): {}: {}{}",
                        p.task_id,
                        p.timing(),
                        code,
                        error,
                        data.as_ref().map(|d| format!(" (data {d})")).unwrap_or_default()
                    );
                    crate::notify::ai_missed();
                    // A refused answer is never submitted, so the node never audits it (upstream
                    // drops its trace too).
                    if let Some(key) = p.trace_key.as_deref() {
                        if traces::forget(key) {
                            info!("Trace: dropped the kept trace of the refused answer {:.16}", p.task_id);
                        }
                    }
                    return Ok(());
                }
                // Option, not unwrap(): a pool error for an id we never tracked (or already
                // removed) must not panic — jobid is only used for the warn! below.
                let jobid = self.shares_stats.shares_pending.lock().unwrap().remove(&id);
                if self.proof_encoding.answered(id) {
                    // Whatever code the pool chose (some use 25 = not subscribed, which would
                    // otherwise force a reconnect), a refused base64 proof is just a rejected share:
                    // fall back to hex and keep the connection.
                    self.proof_encoding.reject_b64(&format!("{}: {}", code, error));
                    self.shares_stats.rejected_other.fetch_add(1, Ordering::SeqCst);
                    warn!("Share with a base64 proof rejected by pool (Job id: {:?}): {}", jobid, error);
                    return Ok(());
                }
                match code {
                    ErrorCode::Unknown => {
                        // Match solo-mining behaviour (grpc.rs SubmitBlockResponse): a rejected
                        // share/block is logged but never fatal. Returning Err here tore down the
                        // whole connection and caused an infinite reconnect loop on every share.
                        self.shares_stats.low_diff.fetch_add(1, Ordering::SeqCst);
                        warn!("Share rejected by pool (Job id: {:?}): {}", jobid, error);
                        Ok(())
                    }
                    ErrorCode::JobNotFound => {
                        self.shares_stats.stale.fetch_add(1, Ordering::SeqCst);
                        warn!("Stale share (Job id: {:?})", jobid);
                        Ok(())
                    }
                    ErrorCode::DuplicateShare => {
                        self.shares_stats.duplicate.fetch_add(1, Ordering::SeqCst);
                        warn!("Duplicate share (Job id: {:?})", jobid);
                        Ok(())
                    }
                    ErrorCode::LowDifficultyShare => {
                        self.shares_stats.low_diff.fetch_add(1, Ordering::SeqCst);
                        warn!("Low difficulty share (Job id: {:?})", jobid);
                        Ok(())
                    }
                    // Keryx OPoI/PoM share rejections — non-fatal, same as any rejected share.
                    // The connection must survive; only the share is lost.
                    ErrorCode::InvalidOpoiTag => {
                        self.shares_stats.invalid_opoi.fetch_add(1, Ordering::SeqCst);
                        warn!("Invalid OPoI tag, share rejected (Job id: {:?}): {}", jobid, error);
                        Ok(())
                    }
                    ErrorCode::InvalidPomProof => {
                        self.shares_stats.invalid_pom.fetch_add(1, Ordering::SeqCst);
                        warn!("Invalid/missing PoM proof, share rejected (Job id: {:?}): {}", jobid, error);
                        Ok(())
                    }
                    ErrorCode::Unauthorized => {
                        error!("Got error code {}: {}", code, error);
                        Err(error.into())
                    }
                    ErrorCode::NotSubscribed => {
                        error!("Got error code {}: {}", code, error);
                        Err(error.into())
                    }
                    // Pool-specific/unrecognized code — treat as a rejected share, never fatal.
                    ErrorCode::Busy | ErrorCode::Other(_) => {
                        warn!("Share rejected with unrecognized error code {} (Job id: {:?}): {}", code, jobid, error);
                        Ok(())
                    }
                }
            }
            _ => Err(format!("Unhandled stratum response: {:?}", msg).into()),
        }
    }

    /// True while a `mining.ai_response` awaits the pool's ack, for at most [`AI_ACK_TIMEOUT`].
    async fn ai_ack_outstanding(&self) -> bool {
        let mut pending = self.ai_response_pending.lock().await;
        match pending.as_ref() {
            Some(p) if p.sent.elapsed() >= AI_ACK_TIMEOUT => {
                warn!("AI response {:.16} (stratum id {}) never acknowledged by the pool — taking requests again", p.task_id, p.id);
                *pending = None;
                false
            }
            other => other.is_some(),
        }
    }

    /// Tells the pool this rig will not answer its `mining.ai_request` `pool_id` (code 46), so it
    /// re-dispatches at once instead of waiting out its deadline (keryx-miner-supr 0.16.1,
    /// keryx-miner v0.6.0). Nothing when the request came without an id.
    async fn refuse_ai(&self, pool_id: Option<u32>, reason: &str) {
        if let Some(id) = pool_id {
            self.send_channel.send(StratumLine::error_reply(id, ErrorCode::Busy, reason)).await.ok();
        }
    }

    /// The in-flight `mining.ai_response` (cleared) when `id` acknowledges it.
    async fn take_ai_response_ack(&self, id: Option<u32>) -> Option<PendingAiResponse> {
        let mut pending = self.ai_response_pending.lock().await;
        if id.is_some() && pending.as_ref().map(|p| p.id) == id {
            pending.take()
        } else {
            None
        }
    }

    /// Pool-dispatched inference (`mining.ai_request`): pause mining, run the request on the
    /// served model, and answer with `mining.ai_response`. One request at a time, shared with
    /// capability challenges; a busy or unready miner, or a failed inference, declines with a busy
    /// reply (46) to the request's `pool_id`, or silently when it had none.
    async fn handle_ai_request(
        &mut self,
        pool_id: Option<u32>,
        fields: (String, String, String, String, String, u32, String),
        ext: Option<serde_json::Value>,
        miner: &mut MinerManager,
    ) {
        let received = Instant::now();
        let reward = fields.6.clone();
        let request = match ai::AiRequest::parse_ext(fields, ext.as_ref()) {
            Ok(request) => request,
            Err(error) => {
                warn!("Invalid AI request: {}", error);
                return;
            }
        };
        // What the pool sent besides the prompt, to tell later why it refused an answer.
        info!(
            "AI request {:.16}: from the pool (stratum id {}, reward {} sompi, extension {})",
            request.task_id,
            pool_id.map_or("none".to_string(), |id| id.to_string()),
            reward,
            ext.as_ref().map_or("none".to_string(), |e| ai::clip(&e.to_string(), 400))
        );
        crate::notify::ai_request();
        if keryx_miner::pow_only() || !keryx_miner::slm::is_model_ready(&request.model_id) {
            warn!("AI request {:.16}: model is not ready", request.task_id);
            crate::notify::ai_missed();
            self.refuse_ai(pool_id, "busy: model not ready").await;
            return;
        }
        // H15 traced answer: the tier is the model's own (its manifests define the trace), and this
        // engine commits stage-2 traces with the sequence leaves of grouped-query attention and
        // gated delta-rule layers (tiers 0, 1 and 3).
        if let Some(t) = request.trace {
            let own = keryx_miner::models::pom_tier_index(&request.model_id, u64::MAX);
            let supported = keryx_miner::trace::TierManifest::for_attn(t.tier).map(|m| m.seq_spec());
            if own != Some(t.tier) || !t.chain || matches!(supported, Some(Err(_)) | None) {
                warn!(
                    "AI request {:.16}: a traced answer for tier {} cannot be served (model tier {:?}, chain {}, attention {}: {})",
                    request.task_id,
                    t.tier,
                    own,
                    t.chain,
                    t.attention,
                    match supported {
                        Some(Err(e)) => e,
                        None => "no manifest".to_string(),
                        Some(Ok(_)) => "supported".to_string(),
                    }
                );
                crate::notify::ai_missed();
                self.refuse_ai(pool_id, "busy: this trace cannot be served").await;
                return;
            }
        }
        if self.ai_ack_outstanding().await || self.challenge_in_flight.swap(true, Ordering::SeqCst) {
            warn!("AI request {:.16}: inference is busy", request.task_id);
            crate::notify::ai_missed();
            self.refuse_ai(pool_id, "busy: inference running").await;
            return;
        }
        let guard = ai::InferenceGuard::new(miner.opoi_challenge_flag(), self.challenge_in_flight.clone());
        if let Err(error) = miner.process_block(None).await {
            warn!("Could not pause mining for AI request: {}", error);
            crate::notify::ai_missed();
            self.refuse_ai(pool_id, "busy: mining could not pause").await;
            return;
        }
        info!(
            "AI request {:.16}: running inference (max_tokens={}{})",
            request.task_id,
            request.max_tokens,
            if request.trace.is_some() { ", traced" } else { "" }
        );
        let sender = self.send_channel.clone();
        // The username this connection authorized with.
        let worker = self.username();
        let pending = self.ai_response_pending.clone();
        let last_id = self.last_stratum_id.clone();
        let refuse = move |sender: &Sender<StratumLine>, reason: &str| {
            if let Some(id) = pool_id {
                sender.blocking_send(StratumLine::error_reply(id, ErrorCode::Busy, reason)).ok();
            }
        };
        task::spawn_blocking(move || {
            let label = format!("pool ai_request {:.16}", request.task_id);
            let (result, traced) = match request.trace {
                Some(t) => {
                    let g = keryx_miner::slm::load_and_run_traced_inference(
                        &request.model_id,
                        &request.prompt,
                        request.max_tokens,
                        t.tier,
                        t.prompt_len,
                        &request.request_hash_bytes(),
                        t.attention,
                        &label,
                    );
                    match g {
                        // A traced request without its trace is no valid answer: refuse it, the pool
                        // re-dispatches.
                        Some(g) if g.trace.is_some() => {
                            let trace = g.trace.unwrap();
                            let commitment = trace.commitment().to_bytes();
                            let body = keryx_miner::slm::answer_body(&g.text, Some(&trace));
                            info!(
                                "AI request {:.16}: traced answer (root {}, {} prompt / {} positions{}) kept for openings",
                                request.task_id,
                                hex::encode(&trace.record.trace_root[..8]),
                                trace.record.n_prompt,
                                trace.record.n_pos,
                                if trace.seq.is_some() { ", sequence leaves" } else { "" }
                            );
                            traces::keep(
                                &request.request_hash,
                                traces::Kept {
                                    model_id: request.model_id,
                                    tier: t.tier,
                                    request_hash: request.request_hash_bytes(),
                                    max_tokens: request.max_tokens.min(u32::MAX as usize) as u32,
                                    at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()),
                                    trace: Arc::new(trace),
                                },
                            );
                            (g.text, Some((commitment, body)))
                        }
                        _ => {
                            drop(guard);
                            warn!("AI request {:.16}: no traced answer — left to the pool", request.task_id);
                            crate::notify::ai_missed();
                            refuse(&sender, "busy: inference failed");
                            return;
                        }
                    }
                }
                None => (
                    keryx_miner::slm::load_and_run_inference(&request.model_id, &request.prompt, request.max_tokens, &label).unwrap_or_default(),
                    None,
                ),
            };
            let id = last_id.fetch_add(1, Ordering::SeqCst);
            let trace_key = traced.is_some().then(|| request.request_hash.to_ascii_lowercase());
            let traced = traced.as_ref().map(|(c, b)| (c.as_slice(), b.as_slice()));
            match request.response(id, worker, &result, traced) {
                Ok(line) => {
                    *pending.blocking_lock() =
                        Some(PendingAiResponse { id, received, sent: Instant::now(), task_id: request.task_id.clone(), trace_key });
                    // The GPU is done: resume mining before the send, which blocks while the pool
                    // connection is backed up. `pending` keeps further requests out until the ack.
                    drop(guard);
                    if sender.blocking_send(line).is_err() {
                        *pending.blocking_lock() = None;
                        warn!("AI response {:.16}: connection closed", request.task_id);
                        crate::notify::ai_missed();
                    }
                }
                Err(error) => {
                    drop(guard);
                    warn!("AI request {:.16}: {}", request.task_id, error);
                    crate::notify::ai_missed();
                    refuse(&sender, "busy: inference failed");
                }
            }
        });
    }

    /// A pool's `mining.ai_opening` (keryx-miner-supr 0.15.3): queues the replay of the kept trace
    /// of one of our traced answers. [`drive_openings`](Self::drive_openings) runs it once the
    /// inference card is free and answers with the unsigned opening (`mining.ai_opening_result`):
    /// the bundle of every piece for the first opening (trace, first attention phase, residual),
    /// one attention phase otherwise. The pool wraps every piece in the seeded envelope and signs it.
    async fn handle_ai_opening(&mut self, params: Vec<serde_json::Value>, miner: &mut MinerManager) {
        let op = match ai::AiOpening::parse(&params) {
            Ok(op) => op,
            Err(e) => {
                warn!("Invalid AI opening request from pool: {}", e);
                return;
            }
        };
        let id = self.last_stratum_id.fetch_add(1, Ordering::SeqCst);
        self.aux_ids.lock().unwrap_or_else(|p| p.into_inner()).insert(id);
        if traces::get(&op.request_hash).is_none() {
            warn!("AI opening {:.16}: no kept trace for this request", op.request_hash);
            let worker = self.username();
            self.send_channel.send(op.result(id, worker, Err("no kept trace for this request"))).await.ok();
            return;
        }
        if self.openings.len() >= MAX_QUEUED_OPENINGS {
            warn!("AI opening {:.16}: {} openings already queued", op.request_hash, self.openings.len());
            let worker = self.username();
            self.send_channel.send(op.result(id, worker, Err("too many openings queued"))).await.ok();
            return;
        }
        self.openings.push_back((op, id, std::time::Instant::now()));
        self.drive_openings(miner).await;
    }

    /// Starts the oldest queued opening when the inference card is free (called again at every job,
    /// so a queued one starts soon after the card frees). An opening that waits too long is
    /// answered with an error.
    async fn drive_openings(&mut self, miner: &mut MinerManager) {
        while let Some((op, id, _)) = self.openings.front().filter(|(_, _, at)| at.elapsed() > OPENING_QUEUE_TIMEOUT) {
            warn!("AI opening {:.16}: the inference card stayed busy", op.request_hash);
            let line = op.result(*id, self.username(), Err("inference card busy"));
            self.openings.pop_front();
            self.send_channel.send(line).await.ok();
        }
        if self.openings.is_empty() || self.challenge_in_flight.swap(true, Ordering::SeqCst) {
            return;
        }
        let Some((op, id, _)) = self.openings.pop_front() else { return };
        let guard = ai::InferenceGuard::new(miner.opoi_challenge_flag(), self.challenge_in_flight.clone());
        let worker = self.username();
        let Some(k) = traces::get(&op.request_hash) else {
            drop(guard);
            self.send_channel.send(op.result(id, worker, Err("no kept trace for this request"))).await.ok();
            return;
        };
        if let Err(error) = miner.process_block(None).await {
            warn!("Could not pause mining for an AI opening: {}", error);
        }
        info!("AI opening {:.16}: replaying for response {:.16} (phase {})", op.request_hash, op.response_txid_hex, op.path.len());
        let sender = self.send_channel.clone();
        task::spawn_blocking(move || {
            let t0 = std::time::Instant::now();
            let body = if op.path.is_empty() {
                keryx_miner::slm::open_trace(&k.model_id, k.tier, &k.trace, &op.seed, op.response_txid, &k.request_hash, k.max_tokens)
            } else {
                keryx_miner::slm::open_attention_phase(
                    &k.model_id,
                    k.tier,
                    &k.trace,
                    &op.seed,
                    op.response_txid,
                    &k.request_hash,
                    &op.path,
                    &op.prev,
                    k.max_tokens,
                )
            };
            drop(guard);
            let line = match &body {
                Some(b) => {
                    info!("AI opening {:.16}: built ({} B in {:.1} s)", op.request_hash, b.len(), t0.elapsed().as_secs_f64());
                    crate::notify::opening(true);
                    op.result(id, worker, Ok(b))
                }
                None => {
                    warn!("AI opening {:.16}: the replay or its check failed", op.request_hash);
                    crate::notify::opening(false);
                    op.result(id, worker, Err("replay failed"))
                }
            };
            if sender.blocking_send(line).is_err() {
                warn!("AI opening {:.16}: connection closed", op.request_hash);
            }
        });
    }

    /// The username this connection authorized with.
    fn username(&self) -> String {
        if self.worker.is_empty() {
            self.miner_address.clone()
        } else {
            format!("{}.{}", self.miner_address, self.worker)
        }
    }

    /// True (and cleared) when `id` answers a hello, a telemetry report or an opening result.
    fn take_aux(&self, id: u32) -> bool {
        self.aux_ids.lock().unwrap_or_else(|p| p.into_inner()).remove(&id)
    }

    /// Sends a request whose answer is no share result.
    async fn send_aux(&mut self, command: StratumCommand) {
        let id = self.last_stratum_id.fetch_add(1, Ordering::SeqCst);
        {
            let mut ids = self.aux_ids.lock().unwrap_or_else(|p| p.into_inner());
            if ids.len() > 256 {
                ids.clear();
            }
            ids.insert(id);
        }
        let line = StratumLine { id: Some(id), payload: StratumLinePayload::StratumCommand(command), jsonrpc: None, error: None };
        if self.send_channel.send(line).await.is_err() {
            self.telemetry_on = false;
        }
    }

    /// The models this rig can serve now, as declared to the pool.
    fn serveable_model_hex() -> Vec<String> {
        keryx_miner::slm::serveable_model_ids(LAST_DAA_SCORE.load(Ordering::Relaxed)).into_iter().map(hex::encode).collect()
    }

    /// Sends `mining.declare_capabilities` when [`redeclare_due`]: called at authorize and on every
    /// job, so the pool learns a change on the next job and an unchanged set every 90 s.
    async fn declare_capabilities_if_changed(&mut self) -> Result<(), Error> {
        let ids = Self::serveable_model_hex();
        if !redeclare_due(self.declared_model_ids.as_deref(), &ids, self.last_declare_at.map(|t| t.elapsed())) {
            return Ok(());
        }
        if self.declared_model_ids.as_ref() != Some(&ids) {
            info!(
                "OPoI: declaring {} model(s) to pool bridge ({})",
                ids.len(),
                ids.iter().map(|m| &m[..8.min(m.len())]).collect::<Vec<_>>().join(",")
            );
        }
        self.send_channel
            .send(StratumLine {
                id: None,
                payload: StratumLinePayload::StratumCommand(StratumCommand::MiningDeclareCapabilities(ids.clone())),
                jsonrpc: None,
                error: None,
            })
            .await?;
        self.declared_model_ids = Some(ids);
        self.last_declare_at = Some(std::time::Instant::now());
        Ok(())
    }

    /// The model this rig serves, with its tier.
    fn served_model() -> Option<([u8; 32], u8)> {
        if keryx_miner::pow_only() {
            return None;
        }
        keryx_miner::slm::loaded_model_ids().into_iter().find_map(|m| keryx_miner::models::pom_tier_index(&m, u64::MAX).map(|t| (m, t)))
    }

    /// 1 once the traced self-check under the attention rule passed on this rig.
    fn trace_flag() -> u8 {
        Self::served_model().is_some_and(|(m, _)| keryx_miner::slm::attention_trace_ready(&m)) as u8
    }

    /// `mining.hello` (schema v1): the miner the subscription names, the trace capability, the host
    /// and the served model. Display only for the pool.
    fn hello(&self) -> serde_json::Value {
        let model = match Self::served_model() {
            Some((m, tier)) => serde_json::json!({ "loaded": true, "tier": tier, "model_id": hex::encode(m) }),
            None => serde_json::json!({ "loaded": false }),
        };
        serde_json::json!({
            "v": 1,
            "miner": { "name": "keryx-miner-supr", "version": "0.16.4", "build": "vulkan", "features": ["pom-v2", "opoi", "pom-vulkan"] },
            "trace": Self::trace_flag(),
            "host": { "os": std::env::consts::OS, "cpu_cores": std::thread::available_parallelism().map_or(1, |n| n.get()) },
            "model": model,
        })
    }

    /// `mining.telemetry` (schema v1): uptime, hash rate, trace capability, local share counts and
    /// the served models.
    fn telemetry_report(&self) -> serde_json::Value {
        let s = &self.shares_stats;
        let rejected = s.low_diff.load(Ordering::Relaxed)
            + s.duplicate.load(Ordering::Relaxed)
            + s.invalid_opoi.load(Ordering::Relaxed)
            + s.invalid_pom.load(Ordering::Relaxed)
            + s.rejected_other.load(Ordering::Relaxed);
        let hashrate = f64::from_bits(crate::miner::LAST_HASHRATE.load(Ordering::Relaxed));
        let models = Self::serveable_model_hex();
        serde_json::json!({
            "v": 1,
            "uptime_s": self.started.elapsed().as_secs(),
            "hashrate_total": if hashrate.is_finite() { hashrate.round() as u64 } else { 0 },
            "trace": Self::trace_flag(),
            "local_shares": { "acc": s.accepted.load(Ordering::Relaxed), "rej": rejected, "stale": s.stale.load(Ordering::Relaxed) },
            "models": models,
        })
    }

    /// Past the attention gate, once per process: the traced self-check of the served model under
    /// the attention rule, mining paused while it runs. A rig whose check passes reports
    /// `"trace": 1`, declares the model and serves traced pool requests; one that fails declares
    /// no model and serves none of them.
    /// An unknown DAA (0: no notify carrying one yet) counts as past the gate, as in
    /// `slm::serveable_model_ids`, or a pool sending only DAA-less jobs would never get a model.
    async fn maybe_start_trace_self_test(&mut self, daa_score: u64, miner: &mut MinerManager) {
        let past_gate = daa_score == 0 || keryx_miner::trace::attention_active(daa_score);
        if !past_gate || TRACE_SELF_TEST_STARTED.load(Ordering::SeqCst) {
            return;
        }
        let Some((model_id, tier)) = Self::served_model() else { return };
        if keryx_miner::trace::TierManifest::for_attn(tier).map(|m| m.seq_spec().is_ok()) != Some(true) {
            TRACE_SELF_TEST_STARTED.store(true, Ordering::SeqCst);
            info!("Trace: tier {tier} cannot be traced by this engine — traced pool requests are not served");
            return;
        }
        if self.ai_ack_outstanding().await || self.challenge_in_flight.swap(true, Ordering::SeqCst) {
            return;
        }
        TRACE_SELF_TEST_STARTED.store(true, Ordering::SeqCst);
        let guard = ai::InferenceGuard::new(miner.opoi_challenge_flag(), self.challenge_in_flight.clone());
        if let Err(error) = miner.process_block(None).await {
            warn!("Could not pause mining for the trace self-check: {}", error);
        }
        info!("Trace: checking traced answers under the attention rule on this GPU (tier {tier}); mining pauses meanwhile");
        task::spawn_blocking(move || {
            let ok = keryx_miner::slm::check_attention_trace(&model_id, tier);
            drop(guard);
            if ok {
                info!("Trace: traced pool answers enabled (the model is declared to the pool with the next job)");
            }
        });
    }

    /// Pool chat (`mining.inference_request` → `mining.inference_result`; suprnova's H6 extension,
    /// also its "ping" serve-ability probe): off-chain inference answered inline — no tx, no
    /// escrow. Pauses mining like `mining.ai_request`. Unlike that consensus path, every outcome
    /// is answered: a busy, unready or failed run replies `{ reqId, ok: false, error }` so the
    /// pool can route the chat elsewhere instead of waiting out its deadline.
    async fn handle_inference_request(&mut self, req: InferenceRequestParams, miner: &mut MinerManager) {
        let sender = self.send_channel.clone();
        if req.req_id.is_empty() || req.req_id.len() > MAX_CHAT_REQ_ID_BYTES {
            warn!("chat: invalid reqId");
            sender.send(inference_result(String::new(), Err("invalid reqId"))).await.ok();
            return;
        }
        let rejection = if req.prompt.is_empty() || req.prompt.len() > MAX_CHAT_PROMPT_BYTES {
            Some("prompt must be 1..=4096 bytes")
        } else if req.max_tokens == 0 || req.max_tokens > MAX_CHAT_TOKENS {
            Some("max_tokens must be 1..=2048")
        } else if req.deadline_ms > MAX_CHAT_DEADLINE_MS {
            Some("deadline_ms exceeds 120000")
        } else {
            None
        };
        if let Some(reason) = rejection {
            warn!(
                "chat[{}]: rejected: {} (prompt {} B, max_tokens {}, deadline_ms {})",
                req.req_id,
                reason,
                req.prompt.len(),
                req.max_tokens,
                req.deadline_ms
            );
            sender.send(inference_result(req.req_id, Err(reason))).await.ok();
            return;
        }
        let model_id: [u8; 32] = match hex::decode(&req.model_id).ok().and_then(|b| b.try_into().ok()) {
            Some(id) => id,
            None => {
                warn!("chat[{}]: invalid model_id '{}'", req.req_id, req.model_id);
                sender.send(inference_result(req.req_id, Err("invalid model_id"))).await.ok();
                return;
            }
        };
        if keryx_miner::pow_only() || !keryx_miner::slm::is_model_ready(&model_id) {
            warn!("chat[{}]: model {:.8} not ready", req.req_id, req.model_id);
            sender.send(inference_result(req.req_id, Err("model not ready"))).await.ok();
            return;
        }
        if self.ai_ack_outstanding().await || self.challenge_in_flight.swap(true, Ordering::SeqCst) {
            info!("chat[{}]: inference busy — replying busy", req.req_id);
            sender.send(inference_result(req.req_id, Err("busy"))).await.ok();
            return;
        }
        let guard = ai::InferenceGuard::new(miner.opoi_challenge_flag(), self.challenge_in_flight.clone());
        if let Err(error) = miner.process_block(None).await {
            warn!("chat[{}]: could not pause mining: {}", req.req_id, error);
            sender.send(inference_result(req.req_id, Err("inference failed"))).await.ok();
            return;
        }
        info!("chat[{}]: inference (max_tokens={}) — mining paused", req.req_id, req.max_tokens);
        task::spawn_blocking(move || {
            let started = std::time::Instant::now();
            let label = format!("pool chat {}", req.req_id);
            let text = keryx_miner::slm::load_and_run_inference(&model_id, &req.prompt, req.max_tokens, &label)
                .unwrap_or_default();
            let ms = started.elapsed().as_millis() as u32;
            // The GPU is done: resume mining before the send, which blocks while the pool
            // connection is backed up (a held pause once kept mining stopped ~1 min past a 0.65 s
            // probe).
            drop(guard);
            let line = if text.is_empty() {
                warn!("chat[{}]: inference produced no output", req.req_id);
                inference_result(req.req_id.clone(), Err("inference failed"))
            } else {
                info!("chat[{}]: done in {} ms — mining resumes", req.req_id, ms);
                inference_result(req.req_id.clone(), Ok((text, ms)))
            };
            if sender.blocking_send(line).is_err() {
                warn!("chat[{}]: connection closed before the result was sent", req.req_id);
            }
        });
    }

    fn set_difficulty(&mut self, difficulty: &f32) -> Result<(), Error> {
        let mut buf = [0u64, 0u64, 0u64, 0u64];
        let (mantissa, exponent, _) = difficulty.recip().integer_decode();
        let new_mantissa = mantissa * DIFFICULTY_1_TARGET.0;
        let new_exponent = (DIFFICULTY_1_TARGET.1 + exponent) as u64;
        let start = (new_exponent / 64) as usize;
        let remainder = new_exponent % 64;

        buf[start] = new_mantissa << remainder; // bottom
        if start < 3 {
            buf[start + 1] = new_mantissa >> (64 - remainder); // top
        } else if new_mantissa.leading_zeros() < remainder as u32 {
            return Err("Target is too big".into());
        }

        self.target_pool = Uint256::new(buf);
        info!("Difficulty: {:?}, Target: 0x{}", difficulty, hex::encode(self.target_pool.to_be_bytes()));
        Ok(())
    }

    fn set_extranonce(&mut self, extranonce: &str, nonce_size: &u32) -> Result<(), Error> {
        self.extranonce = Some(extranonce.to_string());
        info!("Extra! {:?}", extranonce);
        self.nonce_fixed = u64::from_str_radix(extranonce, 16)? << (nonce_size * 8);
        info!("Extra Done!");
        self.nonce_mask = (1 << (nonce_size * 8)) - 1;
        Ok(())
    }

    async fn log_shares(shares_info: Arc<ShareStats>) {
        let mut ticker = tokio::time::interval(LOG_RATE);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut _last_instant = ticker.tick().await;
        loop {
            let _now = ticker.tick().await;
            info!("{}", shares_info)
        }
    }

    /// Parse the task JSON from a `MiningNotifyWithTask`, store it in `current_task_slot`,
    /// and spawn a background inference+IPFS upload if the result is not already cached.
    /// Handle a `mining.challenge` from the bridge.
    ///
    /// The bridge relays the node's periodic capability challenge: the miner must prove
    /// it has the requested model loaded and can produce inference output. The result is
    /// sent back as `mining.challenge_response` so the bridge can forward it to the node.
    async fn handle_challenge(&mut self, model_id_hex: String, nonce_hex: String, miner: &mut MinerManager) {
        // PoW-only mode runs no inference. We declare zero models, so a well-behaved bridge never
        // challenges us — but a stale declaration from a prior full-mode session might. Ignore it
        // silently rather than replying empty (an empty response makes the pool drop the socket).
        if keryx_miner::pow_only() {
            debug!("OPoI challenge for model {:.8} ignored — PoW-only mode (no inference)", model_id_hex);
            return;
        }
        // Only one challenge in flight at a time — bridge will re-challenge if needed.
        if self.challenge_in_flight.swap(true, Ordering::SeqCst) {
            warn!("OPoI challenge: already in flight, dropping new challenge for model {:.8}", model_id_hex);
            return;
        }

        let model_id_bytes = match hex::decode(&model_id_hex) {
            Ok(b) if b.len() == 32 => b,
            _ => {
                warn!("OPoI challenge: invalid model_id_hex '{}'", model_id_hex);
                self.challenge_in_flight.store(false, Ordering::SeqCst);
                return;
            }
        };
        let mut model_id = [0u8; 32];
        model_id.copy_from_slice(&model_id_bytes);

        if !keryx_miner::slm::is_model_ready(&model_id) {
            warn!("OPoI challenge: model {:.8} not ready — sending empty response", model_id_hex);
            self.challenge_in_flight.store(false, Ordering::SeqCst);
            self.send_channel.send(make_challenge_response_line(&model_id_hex, &nonce_hex, "")).await.ok();
            return;
        }

        // Pause PoW so the GPU is fully available for the challenge inference.
        let miner_flag = miner.opoi_challenge_flag();
        miner_flag.store(true, Ordering::SeqCst);
        miner.process_block(None).await.ok();
        info!("OPoI challenge: PoW suspended — model={:.8} nonce={:.8}", model_id_hex, nonce_hex);

        let prompt = format!("Keryx inference challenge {}: briefly describe what you are.", nonce_hex);
        let send_channel = self.send_channel.clone();
        let challenge_flag = Arc::clone(&self.challenge_in_flight);

        tokio::task::spawn_blocking(move || {
            let label = format!("pool challenge {:.8}", nonce_hex);
            let result = keryx_miner::slm::load_and_run_inference(&model_id, &prompt, CHALLENGE_MAX_TOKENS, &label);
            let text = result.unwrap_or_default();
            // Clear both flags — PoW resumes on the next mining.notify from the bridge.
            miner_flag.store(false, Ordering::SeqCst);
            challenge_flag.store(false, Ordering::SeqCst);
            if text.is_empty() {
                warn!("OPoI challenge: inference returned empty text for model {:.8}", model_id_hex);
            } else {
                info!("OPoI challenge: done for model {:.8} ({} chars) — PoW resumes on next notify", model_id_hex, text.len());
            }
            let line = make_challenge_response_line(&model_id_hex, &nonce_hex, &text);
            if send_channel.blocking_send(line).is_err() {
                warn!("OPoI challenge: send_channel closed, could not deliver response");
            }
        });
    }

    /// A v4 job: decode its header, record the window seed of its block task, and mine it.
    #[allow(clippy::too_many_arguments)]
    async fn handle_notify_v4(
        &mut self,
        id: String,
        header_hex: String,
        timestamp: u64,
        daa_score: u64,
        bits: u32,
        seed_hex: String,
        miner: &mut MinerManager,
    ) -> Result<(), Error> {
        let bytes = hex::decode(&header_hex).map_err(|e| format!("mining.notify header is not hex: {e}"))?;
        let header = <crate::proto::RpcBlockHeader as prost::Message>::decode(bytes.as_slice())
            .map_err(|e| format!("mining.notify header does not decode: {e}"))?;
        let window_seed = match seed_hex.as_str() {
            "" => None,
            s => {
                let mut seed = [0u8; 32];
                hex::decode_to_slice(s, &mut seed).map_err(|e| format!("mining.notify window seed is not 32-byte hex: {e}"))?;
                Some(seed)
            }
        };
        self.block_template_ctr.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000)).unwrap();
        LAST_DAA_SCORE.store(daa_score, Ordering::Relaxed);
        keryx_miner::slm::note_chain_daa(daa_score);
        self.declare_capabilities_if_changed().await?;
        // OPoI hard gate (mirrors solo grpc.rs): no models ready = no mining.
        if !keryx_miner::pow_only() && keryx_miner::slm::loaded_model_ids().is_empty() {
            if self.block_template_ctr.load(Ordering::SeqCst) % 200 == 0 {
                warn!("OPoI: no models ready — mining suspended (no inference = no mining)");
            }
            return miner.process_block(None).await;
        }
        *self.current_task_slot.lock().await = None;
        self.drive_openings(miner).await;
        self.maybe_start_trace_self_test(daa_score, miner).await;
        if keryx_miner::block_task::active(daa_score) {
            let window = keryx_verifiable_inference::block_task::window(daa_score);
            let parents = header.parents.first().map(|p| keryx_miner::block_task::parents_key(&p.parent_hashes));
            match (window, parents, window_seed) {
                (Some(w), Some(p), Some(seed)) => keryx_miner::block_task::set_seed(w, &p, seed),
                _ => {
                    keryx_miner::pom_gpu::log_throttled(u32::MAX, "pool-no-task", || {
                        warn!("Block task: the pool sends jobs without a block task (header and window seed) — no block can be valid, job skipped")
                    });
                    return miner.process_block(None).await;
                }
            }
        }
        let target = effective_target(self.target_pool, Some(bits))?;
        miner
            .process_block(Some(PartialBlock {
                id,
                header_hash: [0; 4],
                timestamp,
                daa_score,
                nonce: 0,
                target,
                nonce_mask: self.nonce_mask,
                nonce_fixed: self.nonce_fixed,
                hash: None,
                pom_proof: Vec::new(),
                header: Some(Box::new(header)),
                task_root: None,
            }))
            .await
    }

    /// Parse the task JSON from a `MiningNotifyWithTask`, store it in `current_task_slot`,
    /// Handles an AiTask dispatched by the bridge. Returns `true` if inference was launched
    /// and PoW has been paused — the caller must NOT call `process_block(Some(...))` in that
    /// case; PoW resumes automatically on the next `mining.notify` after inference completes.
    async fn handle_ai_task(&mut self, job_id: String, task_json: String, miner: &mut MinerManager) -> bool {
        let task: AiTask = match serde_json::from_str(&task_json) {
            Ok(t) => t,
            Err(e) => {
                warn!("OPoI: failed to parse task JSON from bridge: {}", e);
                *self.current_task_slot.lock().await = None;
                return false;
            }
        };

        // Store task for this job so create_block_channel can look up the CID.
        *self.current_task_slot.lock().await = Some(CurrentTask { job_id, task: task.clone() });

        // Skip inference if stable_id is missing (malformed task) or already done/running.
        if task.stable_id.is_empty() {
            return false;
        }
        let already_handled = {
            let cache = self.inference_cache.lock().await;
            cache.results.contains_key(&task.stable_id) || cache.in_progress.contains(&task.stable_id)
        };
        if already_handled {
            return false;
        }

        // Decode model_id hex and check it is ready on disk.
        let model_id_bytes = match hex::decode(&task.model_id_hex) {
            Ok(b) if b.len() == 32 => b,
            _ => {
                warn!("OPoI [{}]: invalid model_id_hex '{}'", task.stable_id, task.model_id_hex);
                return false;
            }
        };
        let mut model_id = [0u8; 32];
        model_id.copy_from_slice(&model_id_bytes);

        if !keryx_miner::slm::is_model_ready(&model_id) {
            warn!("OPoI [{}]: model not ready — inference skipped", task.stable_id);
            return false;
        }

        // Guard against two concurrent inferences (challenge may already hold the GPU).
        if self.challenge_in_flight.swap(true, Ordering::SeqCst) {
            warn!("OPoI AiTask [{}]: inference already in flight, skipping", task.stable_id);
            return false;
        }

        // Pause PoW — running kHeavyHash and SLM inference simultaneously crashes the GPU.
        let miner_flag = miner.opoi_challenge_flag();
        miner_flag.store(true, Ordering::SeqCst);
        miner.process_block(None).await.ok();
        info!("OPoI AiTask [{}]: PoW suspended for GPU inference", task.stable_id);

        // Mark in-progress and spawn the blocking inference + IPFS upload.
        {
            let mut cache = self.inference_cache.lock().await;
            cache.in_progress.insert(task.stable_id.clone());
        }
        let stable_id = task.stable_id.clone();
        let prompt = task.prompt.clone();
        let max_tokens = task.max_tokens;
        let ipfs_url = self.ipfs_url.clone();
        let cache_ref = Arc::clone(&self.inference_cache);
        let challenge_flag = Arc::clone(&self.challenge_in_flight);

        tokio::task::spawn_blocking(move || {
            run_inference_and_upload(model_id, prompt, max_tokens, ipfs_url, stable_id, cache_ref);
            // Clear both flags — PoW resumes on the next mining.notify from the bridge.
            miner_flag.store(false, Ordering::SeqCst);
            challenge_flag.store(false, Ordering::SeqCst);
        });

        // PoW was paused for GPU inference, so the caller must not feed a new block.
        true
    }
}

impl Drop for StratumHandler {
    fn drop(&mut self) {
        self.log_handler.abort();
        self.block_handle.abort()
    }
}

// ── Phase 2 OPoI — blocking inference helpers ────────────────────────────────

/// Runs SLM inference, uploads the result to IPFS, then stores the CID in the cache.
/// Called from `spawn_blocking` — must not call async functions.
fn run_inference_and_upload(
    model_id: [u8; 32],
    prompt: String,
    max_tokens: usize,
    ipfs_url: String,
    stable_id: String,
    cache: InferenceCache,
) {
    let cid_opt = do_inference_and_upload(&model_id, &prompt, max_tokens, &ipfs_url, &stable_id);
    let mut guard = cache.blocking_lock();
    guard.in_progress.remove(&stable_id);
    if let Some(cid) = cid_opt {
        if guard.results.len() >= MAX_INFERENCE_CACHE_SIZE {
            guard.results.clear();
            guard.results.shrink_to_fit();
        }
        guard.results.insert(stable_id, cid);
    }
}

/// A `mining.inference_result` notification: `Ok((text, ms))` or `Err(reason)`. `tokens` is a
/// whitespace word count, as keryx-miner-supr reports it.
fn inference_result(req_id: String, outcome: std::result::Result<(String, u32), &str>) -> StratumLine {
    let params = match outcome {
        Ok((text, ms)) => InferenceResultParams {
            req_id,
            ok: true,
            tokens: Some(text.split_whitespace().count() as u32),
            text: Some(text),
            ms: Some(ms),
            error: None,
        },
        Err(reason) => InferenceResultParams {
            req_id,
            ok: false,
            text: None,
            tokens: None,
            ms: None,
            error: Some(reason.to_string()),
        },
    };
    StratumLine {
        id: None,
        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningInferenceResult(params)),
        jsonrpc: None,
        error: None,
    }
}

fn make_challenge_response_line(model_id_hex: &str, nonce_hex: &str, result: &str) -> StratumLine {
    StratumLine {
        id: None,
        payload: StratumLinePayload::StratumCommand(StratumCommand::MiningChallengeResponse((
            model_id_hex.to_string(),
            nonce_hex.to_string(),
            result.to_string(),
        ))),
        jsonrpc: None,
        error: None,
    }
}

fn do_inference_and_upload(
    model_id: &[u8; 32],
    prompt: &str,
    max_tokens: usize,
    ipfs_url: &str,
    stable_id: &str,
) -> Option<String> {
    info!("OPoI [{}]: starting SLM inference (max_tokens={})", stable_id, max_tokens);
    let text = keryx_miner::slm::load_and_run_inference(model_id, prompt, max_tokens, &format!("pool task {stable_id}"))?;
    if text.is_empty() {
        warn!("OPoI [{}]: inference returned empty text — skipping IPFS upload", stable_id);
        return None;
    }
    match crate::ipfs::upload(&text, ipfs_url) {
        Ok(cid_bytes) => {
            // Convert raw 34-byte multihash to base58 CIDv0 string via AiResponsePayload helper.
            let cid = keryx_inference::AiResponsePayload::new([0u8; 32], 0, cid_bytes, 0).cid_v0();
            info!("OPoI [{}]: inference complete, IPFS CID={}", stable_id, cid);
            Some(cid)
        }
        Err(e) => {
            warn!("OPoI [{}]: IPFS upload failed: {}", stable_id, e);
            None
        }
    }
}

#[cfg(test)]
mod proof_encoding_tests {
    use super::*;
    use base64::Engine;

    fn subs(entries: &[(&str, &str)]) -> Vec<(String, String)> {
        entries.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn hex_until_the_pool_offers_base64() {
        let enc = ProofEncoding::for_pool("test-pool-hex:1");
        let proof: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
        assert_eq!(enc.encode(1, &proof), hex::encode(&proof));
        enc.absorb_subscribe(&subs(&[("mining.notify", "abc"), (PROOF_ENCODING_KEY, "hex")]));
        assert_eq!(enc.encode(2, &proof), hex::encode(&proof));
        assert!(!enc.answered(1) && !enc.answered(2));
    }

    #[test]
    fn base64_round_trip_and_tag() {
        let enc = ProofEncoding::for_pool("test-pool-b64:1");
        enc.absorb_subscribe(&subs(&[("mining.notify", "abc"), (PROOF_ENCODING_KEY, PROOF_ENCODING_B64)]));
        for len in [0usize, 1, 2, 3, 452_705, 487_281] {
            let proof: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
            let s = enc.encode(len as u32, &proof);
            let body = s.strip_prefix(PROOF_B64_PREFIX).expect("tagged");
            assert_eq!(body.len(), len.div_ceil(3) * 4);
            assert!(!s.contains('\n') && !s.contains('\\'));
            assert_eq!(base64::engine::general_purpose::STANDARD.decode(body).unwrap(), proof);
            assert!(enc.answered(len as u32));
        }
        // A hex proof never contains the tag separator, so the pool can always tell them apart.
        assert!(!hex::encode([0xffu8; 64]).contains(':'));
    }

    #[test]
    fn a_refused_base64_submit_sticks_to_hex_for_that_pool() {
        let enc = ProofEncoding::for_pool("test-pool-fallback:1");
        enc.absorb_subscribe(&subs(&[(PROOF_ENCODING_KEY, PROOF_ENCODING_B64)]));
        let proof = vec![1u8, 2, 3];
        assert!(enc.encode(7, &proof).starts_with(PROOF_B64_PREFIX));
        assert!(enc.answered(7));
        enc.reject_b64("pom proof not hex");
        assert_eq!(enc.encode(8, &proof), "010203");
        assert!(!enc.answered(8));
        // A reconnect to the same pool that re-advertises base64 stays on hex.
        let again = ProofEncoding::for_pool("test-pool-fallback:1");
        again.absorb_subscribe(&subs(&[(PROOF_ENCODING_KEY, PROOF_ENCODING_B64)]));
        assert_eq!(again.encode(9, &proof), "010203");
        // A different pool is unaffected.
        let other = ProofEncoding::for_pool("test-pool-other:1");
        other.absorb_subscribe(&subs(&[(PROOF_ENCODING_KEY, PROOF_ENCODING_B64)]));
        assert!(other.encode(10, &proof).starts_with(PROOF_B64_PREFIX));
    }

    #[test]
    fn subscribe_result_with_the_encoding_entry_parses() {
        let raw = r#"{"id":0,"result":[[["mining.notify","1"],["keryx.proof_encoding","b64"]],"a1b2",6],"error":null}"#;
        let line: StratumLine = serde_json::from_str(raw).unwrap();
        match line.payload {
            StratumLinePayload::StratumResult { result: StratumResult::Subscribe((s, ex, n)) } => {
                assert_eq!(ex, "a1b2");
                assert_eq!(n, 6);
                let enc = ProofEncoding::for_pool("test-pool-subscribe:1");
                enc.absorb_subscribe(&s);
                assert!(enc.encode(1, &[0u8]).starts_with(PROOF_B64_PREFIX));
            }
            other => panic!("unexpected {:?}", other),
        }
    }

    #[test]
    fn id_less_and_non_true_results_parse_as_results() {
        // `{"id":null,"result":true}` must decode as a result (it is then ignored, not fatal).
        let line: StratumLine = serde_json::from_str(r#"{"id":null,"result":true,"error":null}"#).unwrap();
        assert!(line.id.is_none());
        assert!(matches!(line.payload, StratumLinePayload::StratumResult { .. }));
        // A `false` result for a submit decodes too (counted as a rejected share).
        let line: StratumLine = serde_json::from_str(r#"{"id":12,"result":false,"error":null}"#).unwrap();
        assert_eq!(line.id, Some(12));
        assert!(matches!(line.payload, StratumLinePayload::StratumResult { result: StratumResult::Plain(Some(false)) }));
    }

    #[test]
    fn capabilities_are_declared_on_change_and_when_stale() {
        let m = vec!["ab".repeat(32)];
        let none: Vec<String> = Vec::new();
        let fresh = Some(Duration::from_secs(5));
        let stale = Some(REDECLARE_EVERY);
        // A new connection declares, even nothing: the bridge may keep an earlier session's set.
        assert!(redeclare_due(None, &none, None));
        assert!(redeclare_due(None, &m, None));
        // An unchanged, fresh declaration is not repeated every job...
        assert!(!redeclare_due(Some(&m[..]), &m, fresh));
        assert!(!redeclare_due(Some(&none[..]), &none, fresh));
        // ...but a stale one is, and any change goes out at once, losing the last model included.
        assert!(redeclare_due(Some(&m[..]), &m, stale));
        assert!(redeclare_due(Some(&none[..]), &m, fresh));
        assert!(redeclare_due(Some(&m[..]), &none, fresh));
    }

    #[test]
    fn an_empty_declaration_is_sent_as_empty_params() {
        let line = StratumLine {
            id: None,
            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningDeclareCapabilities(Vec::new())),
            jsonrpc: None,
            error: None,
        };
        let v = serde_json::to_value(&line).unwrap();
        assert_eq!(v["method"], "mining.declare_capabilities");
        assert_eq!(v["params"], serde_json::json!([]));
    }
}

#[cfg(test)]
mod notify_v4_tests {
    use super::*;
    use crate::proto::{RpcBlockHeader, RpcBlockLevelParents};

    /// A post-H15 template header, as the pool's node hands it over.
    fn header() -> RpcBlockHeader {
        RpcBlockHeader {
            version: 1,
            parents: vec![
                RpcBlockLevelParents { parent_hashes: vec!["11".repeat(32), "12".repeat(32)] },
                RpcBlockLevelParents { parent_hashes: vec!["13".repeat(32)] },
            ],
            hash_merkle_root: "22".repeat(32),
            accepted_id_merkle_root: "23".repeat(32),
            utxo_commitment: "24".repeat(32),
            timestamp: 1_790_000_000_000,
            bits: 0x1d2f_ffff,
            daa_score: 140_115_001,
            blue_work: "123456789abcdef".into(),
            blue_score: 128_000_000,
            pruning_point: "33".repeat(32),
            service_state_hash: "55".repeat(32),
            ..Default::default()
        }
    }

    fn pre_pow(h: &RpcBlockHeader) -> [u8; 32] {
        let mut hasher = crate::pow::HeaderHasher::new();
        crate::pow::serialize_header(&mut hasher, h, true);
        hasher.finalize().to_le_bytes()
    }

    /// The header of a v4 job survives the pool's encoding (prost bytes as hex in `mining.notify`
    /// params[1]) and our decoding in `handle_notify_v4` unchanged, so the pre-PoW hash this miner
    /// grinds is the one of the pool's template.
    #[test]
    fn v4_job_header_round_trips_to_the_same_pre_pow_hash() {
        let sent = header();
        let header_hex = hex::encode(prost::Message::encode_to_vec(&sent));
        let raw = format!(
            r#"{{"id":null,"method":"mining.notify","params":["job7","{header_hex}",{},{},{},""]}}"#,
            sent.timestamp, sent.daa_score, sent.bits
        );
        let line: StratumLine = serde_json::from_str(&raw).unwrap();
        let StratumLinePayload::StratumCommand(StratumCommand::MiningNotify(MiningNotify::MiningNotifyV4((_, h, ts, daa, bits, _)))) = line.payload else {
            panic!("expected MiningNotifyV4");
        };
        assert_eq!((ts, daa, bits), (sent.timestamp as u64, sent.daa_score, sent.bits));
        // Exactly the decode `handle_notify_v4` runs.
        let bytes = hex::decode(&h).unwrap();
        let got = <RpcBlockHeader as prost::Message>::decode(bytes.as_slice()).unwrap();
        assert_eq!(got, sent);
        assert_eq!(pre_pow(&got), pre_pow(&sent));
        // No block task past the gate: the header carries no task root, and none is hashed.
        assert!(got.task_root.is_empty());
        let mut with_root = got.clone();
        with_root.task_root = "44".repeat(32);
        assert_eq!(pre_pow(&with_root), pre_pow(&got));
    }
}

#[cfg(test)]
mod tls_url_tests {
    use super::{is_stratum_url, tls_host};

    #[test]
    fn stratum_schemes_and_tls_host() {
        for url in ["stratum+tcp://krx.suprnova.cc:4404", "stratum+ssl://krx.cryo-pool.com:5555", "stratum+tls://pool:443"] {
            assert!(is_stratum_url(url), "{url}");
        }
        for url in ["grpc://127.0.0.1:22110", "stratum://pool:1", "krx.cryo-pool.com:5555"] {
            assert!(!is_stratum_url(url), "{url}");
        }
        assert_eq!(tls_host("krx.cryo-pool.com:5555"), "krx.cryo-pool.com");
        assert_eq!(tls_host("[2001:db8::1]:443"), "2001:db8::1");
        assert_eq!(tls_host("pool.example"), "pool.example");
    }
}

/// Network check of the TLS path against a live pool: handshake with the public roots, then one
/// `mining.subscribe` round trip. No mining, no shares. Run with:
///   KERYX_TEST_TLS_POOL=krx.cryo-pool.com:5555 cargo test --release --bin keryx-miner-rdna3 -- --ignored --nocapture tls_pool_subscribe
#[cfg(test)]
mod tls_live_tests {
    use super::{tls_wrap, TlsSettings};
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    #[tokio::test]
    #[ignore = "connects to a live pool; set KERYX_TEST_TLS_POOL=host:port"]
    async fn tls_pool_subscribe() {
        let Ok(address) = std::env::var("KERYX_TEST_TLS_POOL") else {
            eprintln!("SKIP: KERYX_TEST_TLS_POOL not set");
            return;
        };
        let tcp = tokio::net::TcpStream::connect(&address).await.expect("tcp connect");
        let mut tls = tls_wrap(tcp, &address, &TlsSettings { ca_file: None }).await.expect("TLS handshake against the public roots");
        tls.write_all(b"{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"keryx-miner-supr/0.16.4\",\"keryx-stratum-v3\"],\"jsonrpc\":\"2.0\"}\n").await.unwrap();
        let mut line = String::new();
        tokio::time::timeout(std::time::Duration::from_secs(15), BufReader::new(tls).read_line(&mut line)).await.expect("reply in time").unwrap();
        eprintln!("pool replied over TLS: {}", line.trim());
        let reply: serde_json::Value = serde_json::from_str(&line).expect("a JSON line");
        assert_eq!(reply["id"], 1);
    }
}
