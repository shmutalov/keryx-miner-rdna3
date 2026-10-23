use crate::client::Client;
use crate::pow::BlockSeed;
use crate::pow::BlockSeed::{FullBlock, PartialBlock};
use crate::proto::kaspad_message::Payload;
use crate::proto::rpc_client::RpcClient;
use crate::proto::{
    GetBlockRequestMessage, GetBlockTemplateRequestMessage, GetInfoRequestMessage,
    GetServiceStrikesRequestMessage, KaspadMessage, NotifyBlockAddedRequestMessage,
    NotifyNewBlockTemplateRequestMessage, NotifyVirtualSelectedParentChainChangedRequestMessage,
};
use crate::{miner::MinerManager, Error};

/// Selected-parent chain entries seen while resolving block task seeds: `(daa_score, selected
/// parent, blue_work)` by block hash.
type ChainCache = std::collections::HashMap<String, (u64, String, String)>;

async fn chain_entry(
    tx: &mpsc::Sender<KaspadMessage>,
    stream: &mut Streaming<KaspadMessage>,
    cache: &mut ChainCache,
    hash: &str,
) -> Option<(u64, String, String)> {
    if let Some(e) = cache.get(hash) {
        return Some(e.clone());
    }
    tx.send(GetBlockRequestMessage { hash: hash.to_string(), include_transactions: false }.into()).await.ok()?;
    while let Ok(Some(msg)) = stream.message().await {
        if let Some(Payload::GetBlockResponse(resp)) = msg.payload {
            if resp.error.is_some() {
                return None;
            }
            let block = resp.block?;
            let (header, verbose) = (block.header?, block.verbose_data?);
            let e = (header.daa_score, verbose.selected_parent_hash, header.blue_work);
            cache.insert(hash.to_string(), e.clone());
            return Some(e);
        }
    }
    None
}

fn blue_work_key(w: &str) -> (usize, String) {
    let w = w.trim_start_matches('0').to_ascii_lowercase();
    (w.len(), w)
}

/// Seed block of the task window starting at `start` for a block over `parents`: its selected
/// parent's chain walked back to the first block whose DAA score is at most `start`.
async fn task_window_seed(client: &RpcClient<TonicChannel>, parents: &[String], start: u64, cache: &mut ChainCache) -> Option<[u8; 32]> {
    let mut client = client.clone();
    let (tx, rx) = mpsc::channel(4);
    let mut stream = client.message_stream(ReceiverStream::new(rx)).await.ok()?.into_inner();
    let mut best: Option<((usize, String), String)> = None;
    for p in parents {
        let (_, _, work) = chain_entry(&tx, &mut stream, cache, p).await?;
        let cand = (blue_work_key(&work), p.to_ascii_lowercase());
        if best.as_ref().is_none_or(|b| cand > *b) {
            best = Some(cand);
        }
    }
    let mut cur = best?.1;
    for _ in 0..4096 {
        let (daa, selected_parent, _) = chain_entry(&tx, &mut stream, cache, &cur).await?;
        if daa <= start {
            let mut seed = [0u8; 32];
            hex::decode_to_slice(&cur, &mut seed).ok()?;
            return Some(seed);
        }
        if selected_parent.is_empty() {
            return None;
        }
        cur = selected_parent;
    }
    None
}

/// Max AiRequest queue size — drop oldest when full to prevent unbounded memory growth.
const MAX_AI_QUEUE_SIZE: usize = 64;
/// Max boot-time escrow-validation GetBlock requests in flight at once — each answer
/// sends the next queued one, so thousands of state entries never overwhelm the
/// HTTP/2 flow-control window or delay the mining stream.
const VALIDATION_WINDOW: usize = 64;
/// Max unique stable-ids tracked for deduplication — evict when full.
const MAX_AI_SEEN_IDS: usize = 10_000;

/// DAA lookback of the startup backfill scan: deep enough to cover any service window still
/// open when the miner process comes up.
const AI_BACKFILL_WINDOW_DAA: u64 = 6_000;

/// Backfill GetBlock requests kept in flight.
const AI_BACKFILL_INFLIGHT: usize = 32;

/// Seconds between AiResponse resubmissions while the mempool has not accepted it.
const AI_RESPONSE_RETRY_SECS: u64 = 3;

/// Chain blocks fetched at once while rebuilding the chain after a restart.
const CHAIN_RESTORE_INFLIGHT: usize = 8;

/// Submission attempts before giving up on an AiResponse.
const AI_RESPONSE_MAX_ATTEMPTS: u32 = 100;

/// DAA an unaudited trace is kept past its seed block.
const TRACE_UNAUDITED_GRACE_DAA: u64 = 600;

/// Chain-log entries kept beyond the ones the pending traces still need.
const CHAIN_LOG_TAIL: usize = 512;

/// Rejections of an opening the node does not await before it is dropped.
const OPENING_NOT_AWAITED_MAX_ATTEMPTS: u32 = 5;

/// Built opening bodies kept per response, one per seed block.
const BUILT_OPENINGS_KEPT: usize = 4;

/// Node silence after which a template request probes the connection.
const NODE_SILENCE_PROBE: std::time::Duration = std::time::Duration::from_secs(30);
/// Node silence after which the connection is dropped and reopened.
const NODE_SILENCE_LIMIT: std::time::Duration = std::time::Duration::from_secs(90);

/// Failed chain rebuilds from one accepting block before its response is dropped.
const CHAIN_RESTORE_MAX_FAILURES: u32 = 3;

/// Failed replays of one draw before it is given up.
const REPLAY_MAX_FAILURES: u32 = 2;

use async_trait::async_trait;
use futures_util::StreamExt;
use log::{error, info, warn};
use rand::{thread_rng, RngCore};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc::{self, error::SendError, Sender}, oneshot};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::{PollSendError, PollSender};
use tonic::{transport::Channel as TonicChannel, Streaming};

static EXTRA_DATA: &str = concat!(env!("CARGO_PKG_VERSION"), "/", env!("PACKAGE_COMPILE_TIME"));

/// A finished inference waiting for its CID to be confirmed readable.
struct PendingPublish {
    request_hash: [u8; 32],
    result: String,
    rx: oneshot::Receiver<Option<[u8; 34]>>,
}

/// An inference running for a queued request.
struct InferenceJob {
    stable_id: String,
    request_hash: [u8; 32],
    private: Option<PrivateRequest>,
    model_id: [u8; 32],
    tier: Option<u8>,
    max_tokens: u32,
    /// `None`: the inference failed, no AiResponse is submitted.
    rx: oneshot::Receiver<Option<keryx_miner::slm::Generated>>,
}

/// The trace of a finished inference, on its way into the AiResponse.
struct TraceContext {
    model_id: [u8; 32],
    tier: u8,
    max_tokens: u32,
    trace: keryx_miner::trace::ResponseTrace,
}

/// A traced response, kept until its draw is settled and its opening window closes.
struct PendingTrace {
    model_id: [u8; 32],
    tier: u8,
    request_hash: [u8; 32],
    max_tokens: u32,
    trace: Arc<keryx_miner::trace::ResponseTrace>,
    submitted_daa: u64,
    /// Chain block that accepted the response.
    accepted_hash: Option<String>,
    draw: Option<TraceDraw>,
    /// Opening already built and submitted for this response.
    opening: Option<keryx_miner::trace_store::StoredOpening>,
    /// Unsigned opening bodies already built, by seed block.
    built: Vec<(String, Vec<u8>)>,
    /// Chain reorgs that reset its acceptance or draw.
    reorgs: u32,
}

/// The draw of an accepted response once its seed block is on the chain.
struct TraceDraw {
    seed_hash: String,
    seed_daa: u64,
    seed: [u8; 32],
    audited: bool,
    deadline_daa: u64,
    /// A replay was started for this seed.
    opened: bool,
    /// Replays of this seed that built no opening.
    replay_failures: u32,
}

/// A chain block in chain order; its DAA score arrives with the GetBlock answer.
struct ChainEntry {
    hash: String,
    daa: Option<u64>,
}

/// The replay of an audited response, running on the model's GPU.
struct ReplayJob {
    response_txid: String,
    seed_hash: String,
    rx: oneshot::Receiver<Option<Vec<u8>>>,
}

/// An opening the mempool has not accepted yet.
struct OpeningInflight {
    tx: crate::proto::RpcTransaction,
    response_txid: String,
    deadline_daa: u64,
    attempts: u32,
    last: std::time::Instant,
}

fn hash32(hex_str: &str) -> Option<[u8; 32]> {
    hex::decode(hex_str).ok()?.try_into().ok()
}

/// The seed block of a response accepted by chain block `accepted`: the first later chain block
/// at `TRACE_SEED_DELAY_DAA` or more, with its DAA score and the accepting block's. `None` while
/// a block on the way has no DAA score yet.
fn seed_block_after<'a>(log: &'a VecDeque<ChainEntry>, accepted: &str) -> Option<(&'a str, u64, u64)> {
    let start = log.iter().position(|e| e.hash == accepted)?;
    let accepted_daa = log[start].daa?;
    for e in log.iter().skip(start + 1) {
        let daa = e.daa?;
        if daa >= accepted_daa + keryx_miner::trace::TRACE_SEED_DELAY_DAA {
            return Some((e.hash.as_str(), daa, accepted_daa));
        }
    }
    None
}

/// Puts `added`, the chain after `start` in chain order, right behind `start` in `log` (at its end
/// when `start` is not logged), moving the blocks already logged elsewhere and keeping their DAA
/// scores. Returns the placed blocks whose DAA score is unknown.
fn place_chain_after(log: &mut VecDeque<ChainEntry>, start: &str, added: Vec<String>) -> Vec<String> {
    let pos = log.iter().position(|e| e.hash == start).unwrap_or(usize::MAX);
    let added_set: std::collections::HashSet<&str> = added.iter().map(String::as_str).collect();
    let mut known: std::collections::HashMap<String, Option<u64>> = std::collections::HashMap::new();
    let mut kept: VecDeque<ChainEntry> = VecDeque::with_capacity(log.len() + added.len());
    let mut at = 0;
    for (i, e) in std::mem::take(log).into_iter().enumerate() {
        if added_set.contains(e.hash.as_str()) {
            known.insert(e.hash, e.daa);
            continue;
        }
        kept.push_back(e);
        if i == pos {
            at = kept.len();
        }
    }
    if pos == usize::MAX {
        at = kept.len();
    }
    let mut unknown = Vec::new();
    for (k, hash) in added.into_iter().enumerate() {
        let daa = known.remove(&hash).flatten();
        if daa.is_none() {
            unknown.push(hash.clone());
        }
        kept.insert(at + k, ChainEntry { hash, daa });
    }
    *log = kept;
    unknown
}

/// Private-inference state of a queued request; `None` for a plaintext request.
#[derive(Clone)]
enum PrivateRequest {
    /// Opened with the escrow key; the answer is sealed under this key.
    Opened(keryx_inference::ResponseKey),
    /// Names this miner but cannot be opened; answered with a plaintext error body.
    Unreadable,
}

/// Body of the answer to a private request this miner could not open.
const UNREADABLE_PRIVATE_REQUEST_ANSWER: &str = "keryx: private request could not be opened by this responder";
type BlockHandle = JoinHandle<Result<(), PollSendError<KaspadMessage>>>;

#[allow(dead_code)]
pub struct KeryxdHandler {
    client: RpcClient<TonicChannel>,
    task_chain: ChainCache,
    pub send_channel: Sender<KaspadMessage>,
    stream: Streaming<KaspadMessage>,
    miner_address: String,
    mine_when_not_synced: bool,

    block_channel: Sender<BlockSeed>,
    block_handle: BlockHandle,

    /// Queue of AiRequests waiting for inference.
    /// Each entry: (stable_id_hex16, request_hash, model_id, prompt, max_tokens, private state).
    /// Fed by both BlockAdded scans and block template scans.
    ai_request_queue: VecDeque<(String, [u8; 32], [u8; 32], String, u32, usize, Option<PrivateRequest>)>,

    /// Block hashes queued for boot-time escrow-state validation, drained in slices of
    /// VALIDATION_WINDOW so thousands of GetBlock requests never saturate the HTTP/2
    /// flow-control window (each consumed answer sends the next queued request).
    validation_queue: VecDeque<String>,

    /// Stable IDs already queued or in-flight — used for deduplication.
    ai_seen_prefixes: std::collections::HashSet<String>,

    /// Stable IDs of requests this escrow key has already answered on-chain; such a request is
    /// never served again (the node rejects the duplicate, the inference would be wasted).
    ai_answered_by_me: std::collections::HashSet<String>,

    /// Maps stable_id → (txid, inference_reward_sompi) for confirmed AiRequest TXs.
    /// Used by poll_inference to register the escrow outpoint after a successful AiResponse.
    ai_request_txids: std::collections::HashMap<String, (String, u64)>,

    /// In-flight AiResponse submissions not yet accepted by the mempool, keyed by txid.
    /// Value: (tx, submit attempts, last submit time). Resubmitted until accepted or expired —
    /// a transiently rejected response must keep trying while the service window is open.
    ai_response_inflight: std::collections::HashMap<String, (crate::proto::RpcTransaction, u32, std::time::Instant)>,

    /// Startup backfill: recent ancestors are walked parent-by-parent and scanned, so an
    /// AiRequest accepted before this process started is still served while its window is
    /// open. `None` cutoff = not seeded yet.
    backfill_cutoff_daa: Option<u64>,
    backfill_queue: VecDeque<String>,
    backfill_pending: std::collections::HashSet<String>,
    backfill_visited: std::collections::HashSet<String>,

    /// In-flight LLM inference for a queued request.
    inference_rx: Option<InferenceJob>,

    /// Traced responses by response txid, from submission to the end of their opening window.
    pending_traces: std::collections::HashMap<String, PendingTrace>,

    /// Recent chain blocks in chain order, back to the oldest one a pending trace needs.
    chain_log: VecDeque<ChainEntry>,

    /// On-disk copy of the pending traces.
    trace_store: keryx_miner::trace_store::TraceStore,
    /// Chain blocks to fetch after a restart, from the accepting block of each restored trace.
    chain_restore_queue: VecDeque<String>,
    chain_restore_pending: std::collections::HashSet<String>,
    /// Accepting blocks whose chain is asked for after a restart, one request at a time; their
    /// traces are not drawn before the answer.
    chain_restore_starts: VecDeque<String>,
    /// Failed chain rebuilds, by accepting block.
    chain_restore_failures: std::collections::HashMap<String, u32>,

    /// In-flight replay building an opening.
    replay_rx: Option<ReplayJob>,

    /// In-flight opening submissions not yet accepted by the mempool, keyed by txid.
    opening_inflight: std::collections::HashMap<String, OpeningInflight>,

    /// Finished inference being uploaded and confirmed on the public gateways; its AiResponse
    /// is built once the task hands back the CID (None: dropped).
    publish_rx: Option<PendingPublish>,

    /// In-flight inference for a node-issued challenge.
    /// Tuple: (challenge_string, result_receiver) where challenge_string = "model_id_hex:nonce_hex".
    /// When the result arrives, it is sent back via inference_result in the next GetBlockTemplateRequest.
    challenge_inference_rx: Option<(String, oneshot::Receiver<Option<String>>)>,

    /// Shared flag with MinerManager — suppresses GPU stall warnings during inference.
    opoi_challenge_active: Option<Arc<AtomicBool>>,

    /// Last DAA score seen in a block template — used to compute challenge_window_end.
    last_known_daa: u64,

    /// IPFS Kubo API URL for uploading inference results.
    ipfs_url: String,

    /// 64-char hex Schnorr pubkey embedded in coinbase extra_data as `/escrow:<pubkey>`.
    /// The node routes 20% of the block reward to the corresponding CSV-locked escrow output.
    escrow_pubkey: Option<String>,

    /// Auto-claim module: present when an escrow private key is available.
    escrow_watcher: Option<crate::escrow::EscrowWatcher>,

    /// 128-char hex delegation cert embedded as `/esig:<cert>`, binding the escrow key above to
    /// the payout address. Mandatory from H6 — a block without it is invalid.
    escrow_cert: Option<String>,

    /// Service-ledger identity of the payout address (the node's `miner_key`), the key strikes,
    /// burns and suspensions are reported against.
    service_identity: Option<String>,

    /// Last service-bond strike poll instant.
    last_strike_poll: std::time::Instant,
    /// Last message received from the node, and last probe sent while it is silent.
    last_message: std::time::Instant,
    last_probe: std::time::Instant,

    /// Last rendered service-bond status — logged only on change.
    strike_status: Option<String>,

    /// DAA of every miss seen since this miner started, for the lifetime tally. Counting the
    /// active strike counter would miss the worst ones: the third strike resets it to zero, and a
    /// served response clears it too. Each miss carries its own daa, so a set of those is exact
    /// for as long as the process runs.
    misses_seen: std::collections::HashSet<u64>,
}

#[async_trait(?Send)]
impl Client for KeryxdHandler {
    fn add_devfund(&mut self, _address: String, _percent: u16) {
        // Solo templates always pay the miner: from H6 the coinbase's `/esig:` cert is bound to
        // the payout key, so a template paying any other address would be an invalid block.
    }

    async fn register(&mut self) -> Result<(), Error> {
        // We actually register in connect
        Ok(())
    }

    async fn listen(&mut self, miner: &mut MinerManager) -> Result<(), Error> {
        self.opoi_challenge_active = Some(miner.opoi_challenge_flag());
        // Harvest in-flight inference on a timer, independently of node notifications.
        // On a sole-producer node, pausing mining for inference stops block production,
        // so the node stops sending NewBlockTemplate notifications — without this timer
        // the finished inference would never be collected and mining would deadlock.
        let mut tick = tokio::time::interval(tokio::time::Duration::from_millis(200));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            let maybe_msg = tokio::select! {
                msg = self.stream.message() => Some(msg?),
                _ = tick.tick() => None,
            };
            if maybe_msg.is_some() {
                self.last_message = std::time::Instant::now();
            }
            match maybe_msg {
                Some(Some(m)) => match m.payload {
                    Some(payload) => self.handle_message(payload, miner).await?,
                    None => warn!("keryxd message payload is empty"),
                },
                Some(None) => break, // stream closed by node
                None => {
                    let silent = self.last_message.elapsed();
                    if silent >= NODE_SILENCE_LIMIT {
                        return Err(format!("no message from keryxd for {} s — reconnecting", silent.as_secs()).into());
                    }
                    if silent >= NODE_SILENCE_PROBE && self.last_probe.elapsed() >= NODE_SILENCE_PROBE {
                        self.last_probe = std::time::Instant::now();
                        warn!("No message from keryxd for {} s — probing", silent.as_secs());
                        self.client_get_block_template().await?;
                    }
                    // Timer tick: if a regular inference just finished, get a fresh template.
                    if (self.inference_rx.is_some() || self.publish_rx.is_some()) && self.poll_inference().await {
                        self.client_get_block_template().await?;
                    // If a challenge is in flight, keep pinging the node so the result is
                    // delivered as soon as the inference task completes. This is critical on
                    // sole-producer nodes where mining suspension stops NewBlockTemplate
                    // notifications and the response would otherwise never be sent.
                    } else if self.challenge_inference_rx.is_some() {
                        self.client_get_block_template().await?;
                    } else if self.replay_rx.is_some() && self.poll_replay().await {
                        self.client_get_block_template().await?;
                    }
                    if self.escrow_pubkey.is_some() && self.last_strike_poll.elapsed().as_secs() >= 60 {
                        self.last_strike_poll = std::time::Instant::now();
                        self.client_send(GetServiceStrikesRequestMessage {}).await?;
                    }
                    self.retry_pending_ai_responses().await?;
                    self.prune_traces();
                    self.retry_pending_openings().await?;
                    self.try_start_replay(miner).await?;
                }
            }
        }
        Ok(())
    }

    fn get_block_channel(&self) -> Sender<BlockSeed> {
        self.block_channel.clone()
    }

    fn flush_escrow_state(&mut self) -> Result<(), Error> {
        self.escrow_watcher.as_mut().map_or(Ok(()), |watcher| watcher.flush_state().map_err(Into::into))
    }
}

impl KeryxdHandler {
    pub async fn connect<D>(
        address: D,
        miner_address: String,
        mine_when_not_synced: bool,
        escrow_privkey: Option<String>,
        escrow_state_file: String,
        escrow_cert: Option<String>,
        ipfs_url: String,
    ) -> Result<Box<Self>, Error>
    where
        D: std::convert::TryInto<tonic::transport::Endpoint>,
        D::Error: Into<Error>,
    {
        let trace_store = keryx_miner::trace_store::TraceStore::new(&escrow_state_file);
        let (restored_traces, restored_openings) = Self::restore_traces(&trace_store);
        // Build EscrowWatcher from the resolved escrow privkey (derived or loaded from file).
        // The watcher also provides the pubkey to embed in coinbase extra_data.
        let (escrow_pubkey, escrow_watcher) = match escrow_privkey {
            Some(ref privkey) => {
                match crate::escrow::EscrowWatcher::new(privkey, &miner_address, escrow_state_file.into()) {
                    Ok(watcher) => {
                        let pk = watcher.pubkey_hex();
                        info!("Escrow active: pubkey={}", pk);
                        (Some(pk), Some(watcher))
                    }
                    Err(e) => {
                        log::error!("Failed to initialise EscrowWatcher: {} — escrow disabled", e);
                        (None, None)
                    }
                }
            }
            None => (None, None),
        };

        let service_identity = match crate::escrow::service_identity_hex(&miner_address) {
            Ok(id) => Some(id),
            Err(e) => {
                log::warn!("Cannot derive the service identity of the payout address: {}", e);
                None
            }
        };

        let endpoint: tonic::transport::Endpoint = address.try_into().map_err(Into::into)?;
        let channel = endpoint
            .connect_timeout(std::time::Duration::from_secs(10))
            .http2_keep_alive_interval(std::time::Duration::from_secs(15))
            .keep_alive_timeout(std::time::Duration::from_secs(20))
            .keep_alive_while_idle(true)
            .connect()
            .await?;
        let mut client = RpcClient::new(channel);
        // Outbound message channel to the node. ALL client->node messages share this:
        // mining (submit_block, GetBlockTemplate) AND inference traffic (per-block GetBlock,
        // escrow submit_transaction). With a capacity of 2 the inference traffic could fill the
        // buffer and block GetBlockTemplate, stalling template delivery → the GPU sits idle
        // between blocks. A large buffer keeps the mining requests from queuing behind inference traffic.
        let (send_channel, recv) = mpsc::channel(1024);
        send_channel.send(GetInfoRequestMessage {}.into()).await?;
        let stream = client.message_stream(ReceiverStream::new(recv)).await?.into_inner();
        let (block_channel, block_handle) = Self::create_block_channel(send_channel.clone());
        let mut handler = Box::new(Self {
            client,
            task_chain: ChainCache::new(),
            stream,
            send_channel,
            miner_address,
            mine_when_not_synced,
            block_channel,
            block_handle,
            ai_request_queue: VecDeque::new(),
            validation_queue: VecDeque::new(),
            ai_seen_prefixes: std::collections::HashSet::new(),
            ai_answered_by_me: std::collections::HashSet::new(),
            ai_request_txids: std::collections::HashMap::new(),
            ai_response_inflight: std::collections::HashMap::new(),
            backfill_cutoff_daa: None,
            backfill_queue: VecDeque::new(),
            backfill_pending: std::collections::HashSet::new(),
            backfill_visited: std::collections::HashSet::new(),
            inference_rx: None,
            pending_traces: restored_traces,
            chain_log: VecDeque::new(),
            trace_store,
            chain_restore_queue: VecDeque::new(),
            chain_restore_pending: std::collections::HashSet::new(),
            chain_restore_starts: VecDeque::new(),
            chain_restore_failures: std::collections::HashMap::new(),
            replay_rx: None,
            opening_inflight: restored_openings,
            publish_rx: None,
            challenge_inference_rx: None,
            opoi_challenge_active: None,
            last_known_daa: 0,
            ipfs_url,
            escrow_pubkey,
            escrow_watcher,
            escrow_cert,
            service_identity,
            last_strike_poll: std::time::Instant::now() - std::time::Duration::from_secs(55),
            last_message: std::time::Instant::now(),
            last_probe: std::time::Instant::now(),
            strike_status: None,
            misses_seen: std::collections::HashSet::new(),
        });
        handler.restore_parked();
        Ok(handler)
    }

    /// Takes over the request queue and in-flight inference of the previous connection.
    fn restore_parked(&mut self) {
        let Some(parked) = PARKED.lock().unwrap_or_else(|p| p.into_inner()).take() else { return };
        for entry in parked.queue {
            self.ai_seen_prefixes.insert(entry.0.clone());
            self.ai_request_queue.push_back(entry);
        }
        if let Some(job) = parked.inference {
            info!("Inference: resuming request id={} across the reconnect", job.stable_id);
            self.ai_seen_prefixes.insert(job.stable_id.clone());
            self.inference_rx = Some(job);
        }
    }

    fn create_block_channel(send_channel: Sender<KaspadMessage>) -> (Sender<BlockSeed>, BlockHandle) {
        // KaspadMessage::submit_block(block)
        let (send, recv) = mpsc::channel::<BlockSeed>(1);
        (
            send,
            tokio::spawn(async move {
                ReceiverStream::new(recv)
                    .map(|block_seed| match block_seed {
                        FullBlock(block) => KaspadMessage::submit_block(*block),
                        PartialBlock { .. } => unreachable!("All blocks sent here should have arrived from here"),
                    })
                    .map(Ok)
                    .forward(PollSender::new(send_channel))
                    .await
            }),
        )
    }

    async fn client_send(&self, msg: impl Into<KaspadMessage>) -> Result<(), SendError<KaspadMessage>> {
        self.send_channel.send(msg.into()).await
    }

    async fn client_get_block_template(&mut self) -> Result<(), SendError<KaspadMessage>> {
        let pay_address = self.miner_address.clone();
        // Append a per-request random nonce so that parallel blocks at the same blue_score
        // get distinct coinbase payloads → distinct tx_ids (avoids DAG coinbase collisions).
        let nonce_hex = format!("{:016x}", thread_rng().next_u64());
        // Inference: run the deterministic fixed-point MLP (matches node validation).
        let opoi_tag = keryx_miner::inference::compute_opoi_tag(&nonce_hex);
        // Embed escrow pubkey so the node routes 20% to the CSV-locked escrow output.
        let escrow_part = self.escrow_pubkey
            .as_deref()
            .map(|pk| format!("/escrow:{}", pk))
            .unwrap_or_default();
        // Delegation cert binding that escrow key to the payout address. From H6 the node rejects
        // a block whose coinbase carries no valid pair.
        let esig_part = self.escrow_cert
            .as_deref()
            .map(|cert| format!("/esig:{}", cert))
            .unwrap_or_default();
        // Announce loaded model capabilities so the node can enforce model_id matching.
        let cap_part = {
            let ids = keryx_miner::slm::loaded_model_ids();
            if ids.is_empty() {
                String::new()
            } else {
                let hex_ids: Vec<String> = ids.iter().map(|id| hex::encode(id)).collect();
                format!("/ai:cap:{}", hex_ids.join(","))
            }
        };
        let extra_data =
            format!("{}{}{}/{}/ai:v1:{}{}", EXTRA_DATA, escrow_part, esig_part, nonce_hex, opoi_tag, cap_part);
        // Harvest a pending challenge response if the inference task just finished.
        let inference_result = match self.challenge_inference_rx.take() {
            Some((challenge_str, mut rx)) => match rx.try_recv() {
                Ok(Some(text)) => {
                    // challenge_str = "model_id_hex:nonce_hex"
                    let mut parts = challenge_str.splitn(2, ':');
                    let model_id_hex = parts.next().unwrap_or("");
                    let nonce_hex_c  = parts.next().unwrap_or("");
                    info!("Inference: sending challenge response model={:.8}", model_id_hex);
                    if let Some(flag) = &self.opoi_challenge_active {
                        flag.store(false, Ordering::Relaxed);
                    }
                    // Response format: "model_id_hex:nonce_hex:result_text"
                    format!("{}:{}:{}", model_id_hex, nonce_hex_c, text)
                }
                Ok(None) => {
                    warn!("Inference: challenge inference failed — sending empty result, node will re-challenge");
                    if let Some(flag) = &self.opoi_challenge_active {
                        flag.store(false, Ordering::Relaxed);
                    }
                    String::new()
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    self.challenge_inference_rx = Some((challenge_str, rx));
                    String::new()
                }
                Err(_) => {
                    warn!("Inference: challenge inference task dropped — sending empty result");
                    if let Some(flag) = &self.opoi_challenge_active {
                        flag.store(false, Ordering::Relaxed);
                    }
                    String::new()
                }
            },
            None => String::new(),
        };
        self.client_send(GetBlockTemplateRequestMessage { pay_address, extra_data, inference_result }).await
    }

    /// Scans a slice of transactions for AiRequest payloads and pushes new
    /// entries into `ai_request_queue` (deduplication by payload hash prefix).
    ///
    /// Handles two formats:
    ///   - Subnetwork 0x03 + binary `AiRequestPayload` (future on-chain format)
    ///   - Any non-coinbase TX + `KRX:AI:1:` JSON prefix (web wallet format)
    fn scan_txs_for_ai_requests(&mut self, txs: &[crate::proto::RpcTransaction], block_daa: u64) {
        // Identity of a request follows the same gate as the node: transaction id past it, payload
        // digest before. Decided from the daa of the block the request is observed in, so both
        // sides classify a request the same way across the activation.
        let txid_identity = block_daa >= keryx_miner::pom::reward_routing_activation_daa();
        if let Some(mine) = self.escrow_watcher.as_ref().map(|w| w.pubkey_bytes()) {
            for sid in Self::own_answered_request_ids(txs, &mine) {
                if self.ai_answered_by_me.insert(sid.clone()) {
                    log::debug!("Inference: request id={} already answered by this miner on-chain", sid);
                }
                self.ai_request_queue.retain(|(queued, ..)| *queued != sid);
            }
            if self.ai_answered_by_me.len() > MAX_AI_SEEN_IDS {
                self.ai_answered_by_me.clear();
                self.ai_answered_by_me.shrink_to_fit();
            }
        }
        // Hard gate: if no models are ready, refuse to accept any AiRequest.
        // Prevents miners with missing/truncated model files from ever queuing inference work.
        let ready_ids = keryx_miner::slm::loaded_model_ids();
        if ready_ids.is_empty() {
            log::warn!("Inference: no models ready — skipping AiRequest scan (run miner with valid model files)");
            return;
        }
        log::debug!(
            "scan_ai: {} txs, subnetwork_ids: {:?}",
            txs.len(),
            txs.iter().map(|t| t.subnetwork_id.as_str()).collect::<Vec<_>>()
        );
        let escrow_secret = self.escrow_watcher.as_ref().map(|w| w.secret_bytes());
        for tx in txs {
            // (raw, model_id, prompt, plaintext prompt length, max_tokens, inference_reward, private state)
            let extracted: Option<(Vec<u8>, [u8; 32], String, u32, usize, u64, Option<PrivateRequest>)> =
                if tx.subnetwork_id == keryx_inference::SUBNETWORK_ID_AI_REQUEST_HEX {
                    // Binary AiRequestPayload (dedicated AI subnetwork).
                    hex::decode(&tx.payload).ok().and_then(|raw| {
                        let req = keryx_inference::AiRequestPayload::deserialize(&raw)?;
                        let (prompt, prompt_len, private) = if req.is_private() {
                            Self::open_private_request(&req, escrow_secret.as_ref())?
                        } else {
                            (String::from_utf8_lossy(&req.prompt).into_owned(), req.prompt.len() as u32, None)
                        };
                        Some((raw, req.model_id, prompt, prompt_len, req.max_tokens as usize, req.inference_reward, private))
                    })
                } else if !tx.inputs.is_empty() {
                    // KRX:AI:1: JSON format — model routed by "m" field, skipped if not loaded.
                    hex::decode(&tx.payload).ok().and_then(|raw| {
                        Self::parse_krx_ai_payload(&raw).and_then(|(model_name, prompt, max_tokens)| {
                            let model_id = keryx_miner::models::find(&model_name)?.model_id;
                            let prompt_len = prompt.len() as u32;
                            Some((raw, model_id, prompt, prompt_len, max_tokens, 0u64, None))
                        })
                    })
                } else {
                    None // coinbase — skip
                };

            if let Some((raw, model_id, prompt, prompt_len, max_tokens, inference_reward, private)) = extracted {
                if !ready_ids.contains(&model_id) {
                    log::debug!("Inference: skipping AiRequest — model not supported or files not ready");
                    continue;
                }
                let txid_hex = tx
                    .verbose_data
                    .as_ref()
                    .map(|v| v.transaction_id.clone())
                    .filter(|id| !id.is_empty())
                    .or_else(|| Self::compute_rpc_txid(tx));
                let request_hash: [u8; 32] = if txid_identity {
                    match txid_hex.as_deref().and_then(|h| hex::decode(h).ok()).and_then(|b| <[u8; 32]>::try_from(b).ok()) {
                        Some(id) => id,
                        None => {
                            log::warn!("Inference: cannot resolve the AiRequest transaction id — request skipped");
                            continue;
                        }
                    }
                } else {
                    blake2b_simd::blake2b(&raw).as_bytes()[..32].try_into().unwrap()
                };
                let stable_id = hex::encode(&request_hash[..8]);
                if self.ai_answered_by_me.contains(&stable_id) {
                    log::debug!("Inference: skipping AiRequest id={} — already answered by this miner", stable_id);
                    continue;
                }
                if !self.ai_seen_prefixes.contains(&stable_id) {
                    info!("Inference: queued AiRequest id={}", stable_id);
                    self.ai_seen_prefixes.insert(stable_id.clone());
                    self.ai_request_queue.push_back((stable_id.clone(), request_hash, model_id, prompt, prompt_len, max_tokens, private));
                    while self.ai_request_queue.len() > MAX_AI_QUEUE_SIZE {
                        self.ai_request_queue.pop_front();
                    }
                    while self.ai_seen_prefixes.len() > MAX_AI_SEEN_IDS {
                        self.ai_seen_prefixes.clear();
                        self.ai_seen_prefixes.shrink_to_fit();
                        for (sid, ..) in &self.ai_request_queue {
                            self.ai_seen_prefixes.insert(sid.clone());
                        }
                    }
                }
                // Track txid for escrow claims. Prefer verbose_data.transaction_id when
                // present, fall back to computing the txid from the transaction fields —
                // verbose_data is not populated for non-coinbase transactions in block
                // template or block notifications, so without this fallback the escrow
                // outpoint is never tracked and the inference_reward is never claimed.
                if inference_reward > 0 {
                    if let Some(txid) = txid_hex {
                        self.ai_request_txids.insert(stable_id, (txid, inference_reward));
                    }
                }
            }
        }
    }

    /// Stable IDs of the requests answered in `txs` by the responder `mine`.
    fn own_answered_request_ids(txs: &[crate::proto::RpcTransaction], mine: &[u8; 32]) -> Vec<String> {
        txs.iter()
            .filter(|tx| tx.subnetwork_id == keryx_inference::SUBNETWORK_ID_AI_RESPONSE_HEX)
            .filter_map(|tx| hex::decode(&tx.payload).ok())
            .filter_map(|raw| keryx_inference::AiResponsePayload::deserialize(&raw))
            .filter(|resp| resp.responder.as_ref().is_some_and(|r| r.escrow_pubkey == *mine))
            .map(|resp| hex::encode(&resp.request_hash[..8]))
            .collect()
    }

    /// Opens a private request with the escrow key. `None` when the request is not for this
    /// miner (no escrow key, or not among the recipients); an addressed request that cannot be
    /// opened is still queued, to be answered with a plaintext error body.
    fn open_private_request(
        req: &keryx_inference::AiRequestPayload,
        escrow_secret: Option<&[u8; 32]>,
    ) -> Option<(String, u32, Option<PrivateRequest>)> {
        let Some(secret) = escrow_secret else {
            log::debug!("Inference: skipping private AiRequest — no escrow key configured");
            return None;
        };
        match keryx_inference::open_request(req, secret) {
            Ok(opened) => match decode_private_prompt(&opened.prompt) {
                Some(prompt) => Some((prompt, opened.prompt.len() as u32, Some(PrivateRequest::Opened(opened.response_key.clone())))),
                None => {
                    warn!("Inference: private AiRequest opened but its compressed prompt is invalid — answering with an error body");
                    Some((String::new(), 0, Some(PrivateRequest::Unreadable)))
                }
            },
            Err(keryx_inference::PrivateError::NotARecipient) => {
                log::debug!("Inference: skipping private AiRequest — this escrow key is not a recipient");
                None
            }
            Err(e) => {
                warn!("Inference: private AiRequest names this miner but cannot be opened ({}) — answering with an error body", e);
                Some((String::new(), 0, Some(PrivateRequest::Unreadable)))
            }
        }
    }

    /// Compute the Kaspa transaction ID for a non-coinbase RpcTransaction.
    ///
    /// Mirrors keryx-node consensus/core/src/hashing/tx.rs `id()` with
    /// EXCLUDE_SIGNATURE_SCRIPT | EXCLUDE_MASS_COMMIT flags (standard for non-coinbase txs).
    ///
    /// Serialization: blake2b-256 keyed "TransactionID" over:
    ///   version(u16 LE) | inputs_count(u64 LE) | inputs... | outputs_count(u64 LE) | outputs...
    ///   | lock_time(u64 LE) | subnetwork_id(20B) | gas(u64 LE) | payload_len(u64 LE) | payload
    ///
    /// For each input (sig script excluded): txid(32B) | index(u32 LE) | 0u64(empty var_bytes) | seq(u64 LE)
    /// For each output: amount(u64 LE) | spk_version(u16 LE) | script_len(u64 LE) | script
    fn compute_rpc_txid(tx: &crate::proto::RpcTransaction) -> Option<String> {
        const KEY: &[u8] = b"TransactionID";
        let mut h = blake2b_simd::Params::new().hash_length(32).key(KEY).to_state();

        h.update(&(tx.version as u16).to_le_bytes());
        h.update(&(tx.inputs.len() as u64).to_le_bytes());
        for input in &tx.inputs {
            let prev = input.previous_outpoint.as_ref()?;
            let txid_bytes = hex::decode(&prev.transaction_id).ok()?;
            if txid_bytes.len() != 32 {
                return None;
            }
            h.update(&txid_bytes);
            h.update(&prev.index.to_le_bytes());
            h.update(&0u64.to_le_bytes()); // write_var_bytes(&[]) — empty sig script
            h.update(&input.sequence.to_le_bytes());
        }

        h.update(&(tx.outputs.len() as u64).to_le_bytes());
        for output in &tx.outputs {
            h.update(&output.amount.to_le_bytes());
            let spk = output.script_public_key.as_ref()?;
            h.update(&(spk.version as u16).to_le_bytes());
            let script = hex::decode(&spk.script_public_key).ok()?;
            h.update(&(script.len() as u64).to_le_bytes());
            h.update(&script);
        }

        h.update(&tx.lock_time.to_le_bytes());
        let subnet = hex::decode(&tx.subnetwork_id).ok()?;
        if subnet.len() != 20 {
            return None;
        }
        h.update(&subnet);
        h.update(&tx.gas.to_le_bytes());
        let payload = hex::decode(&tx.payload).ok()?;
        h.update(&(payload.len() as u64).to_le_bytes());
        h.update(&payload);

        Some(hex::encode(h.finalize().as_bytes()))
    }

    /// Parses a `KRX:AI:1:` JSON payload, returning `(model_name, prompt, max_tokens)`.
    fn parse_krx_ai_payload(raw: &[u8]) -> Option<(String, String, usize)> {
        const PREFIX: &[u8] = b"KRX:AI:1:";
        if raw.len() <= PREFIX.len() || !raw.starts_with(PREFIX) {
            return None;
        }
        let v: serde_json::Value = serde_json::from_slice(&raw[PREFIX.len()..]).ok()?;
        let model = v["m"].as_str().unwrap_or("glm-4-9b-0414").to_string();
        let prompt = v["p"].as_str()?.to_string();
        let max_tokens = v["n"].as_u64().unwrap_or(128) as usize;
        Some((model, prompt, max_tokens))
    }

    /// Starts LLM inference for the next queued AiRequest, if no inference is
    /// already in flight and a response slot is free.
    /// Marks whether mining is paused by inference work (inference in flight, or model files not
    /// ready) rather than by the node. Set at every pause decision, so no exit path can leave it
    /// stale. Also suppresses the GPU stall warnings, which a deliberate pause would trip.
    fn set_opoi_pause(&self, paused: bool) {
        if let Some(flag) = &self.opoi_challenge_active {
            flag.store(paused, Ordering::Relaxed);
        }
    }

    /// Whether the GPU is taken by inference work: a request, a challenge or a replay.
    fn inference_busy(&self) -> bool {
        self.inference_rx.is_some() || self.challenge_inference_rx.is_some() || self.replay_rx.is_some()
    }

    fn try_start_inference(&mut self) {
        if self.inference_busy() {
            return;
        }
        while self.ai_request_queue.front().is_some_and(|(sid, ..)| self.ai_answered_by_me.contains(sid)) {
            if let Some((sid, ..)) = self.ai_request_queue.pop_front() {
                info!("Inference: request id={} already answered by this miner — skipped", sid);
            }
        }
        if let Some((stable_id, request_hash, model_id, prompt, prompt_len, max_tokens, private)) = self.ai_request_queue.pop_front() {
            let (tx_done, rx_done) = oneshot::channel::<Option<keryx_miner::slm::Generated>>();
            // From H15 every answer commits its stage-2 trace (the verifiable-inference and
            // stage-2 gates are the same block).
            let traced = self.last_known_daa >= keryx_miner::pom::verifiable_inference_activation_daa()
                && keryx_miner::trace::chain_active(self.last_known_daa);
            let tier = traced.then(|| keryx_miner::models::pom_tier_index(&model_id, self.last_known_daa)).flatten();
            let unreadable = matches!(private, Some(PrivateRequest::Unreadable));
            let label = match &private {
                Some(PrivateRequest::Opened(_)) => format!("solo request {stable_id}, private"),
                _ => format!("solo request {stable_id}"),
            };
            let job = InferenceJob { stable_id: stable_id.clone(), request_hash, private, model_id, tier, max_tokens: max_tokens as u32, rx: rx_done };
            if unreadable {
                // No inference to run: the answer is the fixed error text.
                let _ = tx_done.send(Some(keryx_miner::slm::Generated { text: UNREADABLE_PRIVATE_REQUEST_ANSWER.to_string(), trace: None }));
                self.inference_rx = Some(job);
                return;
            }
            // Second guard: re-check readiness at execution time (files could have been deleted).
            if !keryx_miner::slm::is_model_ready(&model_id) {
                log::error!("Inference: model became unavailable after queuing id={} — discarding request", stable_id);
                return;
            }
            info!("Inference: spawning LLM inference (max_tokens={}{})", max_tokens, if tier.is_some() { ", traced" } else { "" });
            tokio::task::spawn_blocking(move || {
                let result = match tier {
                    Some(tier) => keryx_miner::slm::load_and_run_traced_inference(&model_id, &prompt, max_tokens, tier, prompt_len, &request_hash, &label),
                    None => keryx_miner::slm::load_and_run_inference(&model_id, &prompt, max_tokens, &label)
                        .map(|text| keryx_miner::slm::Generated { text, trace: None }),
                };
                if result.is_none() {
                    log::warn!("Inference: inference returned no result for id={} — AiResponse will be skipped", stable_id);
                }
                let _ = tx_done.send(result);
            });
            self.inference_rx = Some(job);
        }
    }

    /// Records the seed block of the block task window of a template, once per window and parent set.
    async fn resolve_task_seed(&mut self, header: &crate::proto::RpcBlockHeader) {
        use keryx_verifiable_inference::block_task::{window, window_start};
        if !keryx_miner::block_task::active(header.daa_score) {
            return;
        }
        let Some(w) = window(header.daa_score) else { return };
        let Some(parents) = header.parents.first().map(|p| p.parent_hashes.clone()) else { return };
        let key = keryx_miner::block_task::parents_key(&parents);
        if keryx_miner::block_task::seed(w, &key).is_some() {
            return;
        }
        if self.task_chain.len() > 4096 {
            self.task_chain.clear();
        }
        let lookup = task_window_seed(&self.client, &parents, window_start(w), &mut self.task_chain);
        match tokio::time::timeout(std::time::Duration::from_secs(5), lookup).await {
            Ok(Some(seed)) => {
                keryx_miner::block_task::set_seed(w, &key, seed);
                keryx_miner::pom_gpu::log_throttled(u32::MAX, "seed", || info!("Block task: seed of window {} resolved", w));
            }
            Ok(None) => warn!("Block task: no seed block found for window {}", w),
            Err(_) => warn!("Block task: seed lookup for window {} timed out", w),
        }
    }

    /// Starts the replay of the audited response closest to its deadline, when the GPU is free.
    /// A response whose opening was already built for its current seed block is submitted
    /// without a replay.
    async fn try_start_replay(&mut self, miner: &mut MinerManager) -> Result<(), Error> {
        let ready: Vec<(String, String, Vec<u8>)> = self
            .pending_traces
            .iter_mut()
            .filter_map(|(txid, p)| {
                let d = p.draw.as_mut().filter(|d| d.audited && !d.opened)?;
                let body = p.built.iter().find(|(h, _)| *h == d.seed_hash)?.1.clone();
                d.opened = true;
                Some((txid.clone(), d.seed_hash.clone(), body))
            })
            .collect();
        for (txid, seed_hash, body) in ready {
            info!("Trace: response {} audited at chain block {} — opening already built", txid, seed_hash);
            self.submit_opening(&txid, &seed_hash, body).await;
        }
        if self.inference_busy() || self.publish_rx.is_some() || !self.ai_request_queue.is_empty() {
            return Ok(());
        }
        let Some(txid) = self
            .pending_traces
            .iter()
            .filter(|(_, p)| p.draw.as_ref().is_some_and(|d| d.audited && !d.opened))
            .min_by_key(|(_, p)| p.draw.as_ref().map_or(u64::MAX, |d| d.deadline_daa))
            .map(|(txid, _)| txid.clone())
        else {
            return Ok(());
        };
        let Some(txid_bytes) = hash32(&txid) else {
            self.pending_traces.remove(&txid);
            self.trace_store.remove(&txid);
            return Ok(());
        };
        let p = self.pending_traces.get_mut(&txid).unwrap();
        let d = p.draw.as_mut().unwrap();
        d.opened = true;
        let (seed, seed_hash) = (d.seed, d.seed_hash.clone());
        let (model_id, tier, trace, request_hash, max_tokens) = (p.model_id, p.tier, Arc::clone(&p.trace), p.request_hash, p.max_tokens);
        info!("Trace: response {} audited at chain block {} — replaying {} points", txid, seed_hash, keryx_miner::trace::POINTS);
        self.set_opoi_pause(true);
        miner.process_block(None).await?;
        let (tx_done, rx_done) = oneshot::channel::<Option<Vec<u8>>>();
        tokio::task::spawn_blocking(move || {
            let body = keryx_miner::slm::open_trace(&model_id, tier, &trace, &seed, txid_bytes, &request_hash, max_tokens);
            let _ = tx_done.send(body);
        });
        self.replay_rx = Some(ReplayJob { response_txid: txid, seed_hash, rx: rx_done });
        Ok(())
    }

    /// Polls the in-flight replay; a finished body is signed and submitted as the opening.
    /// Returns `true` when the replay just ended.
    async fn poll_replay(&mut self) -> bool {
        let Some(mut job) = self.replay_rx.take() else { return false };
        let body = match job.rx.try_recv() {
            Ok(body) => body,
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                self.replay_rx = Some(job);
                return false;
            }
            Err(_) => None,
        };
        let Some(body) = body else {
            let retry = self
                .pending_traces
                .get_mut(&job.response_txid)
                .and_then(|p| p.draw.as_mut())
                .filter(|d| d.seed_hash == job.seed_hash)
                .map(|d| {
                    d.replay_failures += 1;
                    d.opened = d.replay_failures > REPLAY_MAX_FAILURES;
                    !d.opened
                });
            if retry == Some(true) {
                warn!("Trace: no opening built for response {} — retrying", job.response_txid);
            } else {
                warn!("Trace: no opening built for response {}", job.response_txid);
            }
            return true;
        };
        let Some(p) = self.pending_traces.get_mut(&job.response_txid) else { return true };
        p.built.retain(|(h, _)| *h != job.seed_hash);
        if p.built.len() >= BUILT_OPENINGS_KEPT {
            p.built.remove(0);
        }
        p.built.push((job.seed_hash.clone(), body.clone()));
        if !p.draw.as_ref().is_some_and(|d| d.seed_hash == job.seed_hash) {
            info!("Trace: seed block of response {} left the chain during the replay — opening kept for its return", job.response_txid);
            return true;
        }
        self.submit_opening(&job.response_txid, &job.seed_hash, body).await;
        true
    }

    /// Signs and submits the opening `body` of response `response_txid`, built for `seed_hash`,
    /// the seed block of its current draw.
    async fn submit_opening(&mut self, response_txid: &str, seed_hash: &str, body: Vec<u8>) {
        let Some(d) = self.pending_traces.get(response_txid).and_then(|p| p.draw.as_ref()) else { return };
        let deadline_daa = d.deadline_daa;
        let body = if self.last_known_daa.max(d.seed_daa) >= keryx_miner::pom::trace_seed_bind_activation_daa() {
            let Some(b) = hash32(seed_hash).and_then(|h| keryx_verifiable_inference::opening::seeded(&body, &h)) else {
                warn!("Trace: cannot name seed block {} in the opening of response {}", seed_hash, response_txid);
                return;
            };
            b
        } else {
            body
        };
        let Some(payload) = self.escrow_watcher.as_ref().map(|w| w.sign_opening(&body)) else { return };
        let rpc_tx = crate::proto::RpcTransaction {
            version: 0,
            inputs: vec![],
            outputs: vec![],
            lock_time: 0,
            subnetwork_id: keryx_inference::SUBNETWORK_ID_AI_OPENING_HEX.to_string(),
            gas: 0,
            payload: hex::encode(payload),
            mass: 0,
            verbose_data: None,
        };
        if let Some(txid) = Self::compute_rpc_txid(&rpc_tx) {
            info!("Trace: submitting the opening of response {} ({} B)", response_txid, body.len());
            self.opening_inflight.insert(
                txid,
                OpeningInflight {
                    tx: rpc_tx.clone(),
                    response_txid: response_txid.to_string(),
                    deadline_daa,
                    attempts: 1,
                    last: std::time::Instant::now(),
                },
            );
        }
        if let Some(p) = self.pending_traces.get_mut(response_txid) {
            p.opening = Some(keryx_miner::trace_store::StoredOpening {
                seed_hash: seed_hash.to_string(),
                deadline_daa,
                payload: rpc_tx.payload.clone(),
            });
        }
        self.persist_trace(response_txid);
        if let Err(e) = self.client_send(KaspadMessage::submit_transaction(rpc_tx)).await {
            warn!("Trace: failed to send the opening tx: {}", e);
        }
    }

    /// Resubmits in-flight openings the mempool has not accepted yet.
    async fn retry_pending_openings(&mut self) -> Result<(), Error> {
        if self.opening_inflight.is_empty() {
            return Ok(());
        }
        let now = std::time::Instant::now();
        let daa = self.last_known_daa;
        let mut resend: Vec<crate::proto::RpcTransaction> = Vec::new();
        self.opening_inflight.retain(|txid, o| {
            if daa > o.deadline_daa {
                warn!("Trace: opening {} of response {} missed its window", txid, o.response_txid);
                return false;
            }
            if now.duration_since(o.last).as_secs() < AI_RESPONSE_RETRY_SECS {
                return true;
            }
            if o.attempts >= AI_RESPONSE_MAX_ATTEMPTS {
                warn!("Trace: giving up on opening {} after {} submit attempts", txid, o.attempts);
                return false;
            }
            o.attempts += 1;
            o.last = now;
            resend.push(o.tx.clone());
            true
        });
        for tx in resend {
            self.client_send(KaspadMessage::submit_transaction(tx)).await?;
        }
        Ok(())
    }

    /// Records a chain block's DAA score once its GetBlock answer arrives.
    /// Writes the on-disk copy of a pending trace.
    fn persist_trace(&self, txid: &str) {
        let Some(p) = self.pending_traces.get(txid) else { return };
        let record = keryx_miner::trace_store::StoredTraceRef {
            version: keryx_miner::trace_store::StoredTraceRef::version(),
            response_txid: txid,
            model_id: &p.model_id,
            tier: p.tier,
            request_hash: &p.request_hash,
            max_tokens: p.max_tokens,
            submitted_daa: p.submitted_daa,
            accepted_hash: &p.accepted_hash,
            opening: &p.opening,
            trace: &p.trace,
        };
        if let Err(e) = self.trace_store.save(&record) {
            warn!("Trace: cannot persist response {}: {}", txid, e);
        }
    }

    /// Loads the stored traces of a previous run and the openings they had already submitted.
    fn restore_traces(
        store: &keryx_miner::trace_store::TraceStore,
    ) -> (std::collections::HashMap<String, PendingTrace>, std::collections::HashMap<String, OpeningInflight>) {
        let mut traces = std::collections::HashMap::new();
        let mut openings = std::collections::HashMap::new();
        for t in store.load_all() {
            if let Some(o) = t.opening.as_ref() {
                let rpc_tx = crate::proto::RpcTransaction {
                    version: 0,
                    inputs: vec![],
                    outputs: vec![],
                    lock_time: 0,
                    subnetwork_id: keryx_inference::SUBNETWORK_ID_AI_OPENING_HEX.to_string(),
                    gas: 0,
                    payload: o.payload.clone(),
                    mass: 0,
                    verbose_data: None,
                };
                if let Some(txid) = Self::compute_rpc_txid(&rpc_tx) {
                    openings.insert(
                        txid,
                        OpeningInflight {
                            tx: rpc_tx,
                            response_txid: t.response_txid.clone(),
                            deadline_daa: o.deadline_daa,
                            attempts: 1,
                            last: std::time::Instant::now() - std::time::Duration::from_secs(60),
                        },
                    );
                }
            }
            traces.insert(
                t.response_txid.clone(),
                PendingTrace {
                    model_id: t.model_id,
                    tier: t.tier,
                    request_hash: t.request_hash,
                    max_tokens: t.max_tokens,
                    trace: Arc::new(t.trace),
                    submitted_daa: t.submitted_daa,
                    accepted_hash: t.accepted_hash,
                    draw: None,
                    opening: t.opening,
                    built: Vec::new(),
                    reorgs: 0,
                },
            );
        }
        if !traces.is_empty() {
            info!("Trace: restored {} pending response(s) from {} ({} opening(s) to resend)", traces.len(), store.dir().display(), openings.len());
        }
        (traces, openings)
    }

    /// After a restart, asks the node for the chain from the accepting block of every restored
    /// trace so its draw can be resolved again.
    async fn start_chain_restore(&mut self) -> Result<(), Error> {
        let accepted: Vec<String> = self
            .pending_traces
            .values()
            .filter(|p| p.draw.is_none())
            .filter_map(|p| p.accepted_hash.clone())
            .collect();
        let idle = self.chain_restore_starts.is_empty();
        for hash in accepted {
            if self.chain_restore_starts.contains(&hash) {
                continue;
            }
            if !self.chain_log.iter().any(|e| e.hash == hash) {
                self.chain_log.push_back(ChainEntry { hash: hash.clone(), daa: None });
                self.chain_restore_queue.push_back(hash.clone());
            }
            self.chain_restore_starts.push_back(hash);
        }
        if idle {
            self.request_next_chain_restore().await?;
        }
        self.chain_restore_pump().await
    }

    /// Asks the node for the chain after the next restored accepting block.
    async fn request_next_chain_restore(&mut self) -> Result<(), Error> {
        let Some(hash) = self.chain_restore_starts.front().cloned() else { return Ok(()) };
        info!("Trace: rebuilding the chain from block {} for a restored response", hash);
        self.client_send(crate::proto::GetVirtualSelectedParentChainFromBlockRequestMessage {
            start_hash: hash,
            include_accepted_transaction_ids: true,
        })
        .await?;
        Ok(())
    }

    /// Places the chain the node returned after restored accepting block `start` right behind it
    /// in the chain log.
    fn place_restored_chain(&mut self, start: &str, removed: &[String], added: Vec<String>) {
        if !removed.is_empty() {
            self.on_chain_blocks_removed(removed);
        }
        let unknown = place_chain_after(&mut self.chain_log, start, added);
        self.chain_restore_queue.extend(unknown);
    }

    /// Retries a failed chain rebuild from `start`; past the limit, drops the responses it
    /// accepted.
    fn on_chain_restore_failed(&mut self, start: String, error: &str) {
        let failures = self.chain_restore_failures.entry(start.clone()).or_insert(0);
        *failures += 1;
        if *failures < CHAIN_RESTORE_MAX_FAILURES {
            warn!("Trace: chain rebuild from block {} failed: {} — retrying", start, error);
            self.chain_restore_starts.push_back(start);
            return;
        }
        warn!("Trace: chain rebuild from block {} failed: {} — its responses are dropped", start, error);
        self.chain_restore_failures.remove(&start);
        let dropped: Vec<String> =
            self.pending_traces.iter().filter(|(_, p)| p.accepted_hash.as_deref() == Some(start.as_str())).map(|(t, _)| t.clone()).collect();
        for txid in dropped {
            self.pending_traces.remove(&txid);
            self.trace_store.remove(&txid);
            self.opening_inflight.retain(|_, o| o.response_txid != txid);
        }
        self.chain_log.retain(|e| e.hash != start);
    }

    /// Notes the acceptances of restored responses that lost their accepting block.
    fn note_restored_acceptances(&mut self, accepted: &[crate::proto::RpcAcceptedTransactionIds]) {
        let mut noted: Vec<String> = Vec::new();
        for a in accepted {
            for txid in &a.accepted_transaction_ids {
                if let Some(p) = self.pending_traces.get_mut(txid).filter(|p| p.accepted_hash.is_none()) {
                    p.accepted_hash = Some(a.accepting_block_hash.clone());
                    info!("Trace: response {} accepted by chain block {}", txid, a.accepting_block_hash);
                    noted.push(txid.clone());
                }
            }
        }
        for txid in noted {
            self.persist_trace(&txid);
        }
    }

    /// Fetches the queued chain blocks a few at a time until every restored trace is drawn.
    async fn chain_restore_pump(&mut self) -> Result<(), Error> {
        let needed = self.pending_traces.values().any(|p| p.accepted_hash.is_some() && p.draw.is_none());
        if !needed {
            self.chain_restore_queue.clear();
            return Ok(());
        }
        while self.chain_restore_pending.len() < CHAIN_RESTORE_INFLIGHT {
            let Some(hash) = self.chain_restore_queue.pop_front() else { break };
            if !self.chain_restore_pending.insert(hash.clone()) {
                continue;
            }
            self.client_send(GetBlockRequestMessage { hash, include_transactions: false }).await?;
        }
        Ok(())
    }

    fn note_chain_block(&mut self, hash: &str, daa: u64) {
        for e in self.chain_log.iter_mut().filter(|e| e.hash == hash) {
            e.daa = Some(daa);
        }
    }

    /// Forgets removed chain blocks and the acceptance or draw that depended on them.
    fn on_chain_blocks_removed(&mut self, removed: &[String]) {
        if removed.is_empty() {
            return;
        }
        let removed: std::collections::HashSet<&str> = removed.iter().map(String::as_str).collect();
        self.chain_log.retain(|e| !removed.contains(e.hash.as_str()));
        let mut reset: Vec<String> = Vec::new();
        for (txid, p) in self.pending_traces.iter_mut() {
            if p.accepted_hash.as_deref().is_some_and(|h| removed.contains(h)) {
                p.accepted_hash = None;
                p.draw = None;
                reset.push(txid.clone());
            } else if p.draw.as_ref().is_some_and(|d| removed.contains(d.seed_hash.as_str())) {
                p.draw = None;
                reset.push(txid.clone());
            }
        }
        for txid in reset {
            log::debug!("Trace: chain reorg — the draw of response {} is reset", txid);
            self.opening_inflight.retain(|_, o| o.response_txid != txid);
            if let Some(p) = self.pending_traces.get_mut(&txid) {
                p.opening = None;
                p.reorgs += 1;
            }
            self.persist_trace(&txid);
        }
    }

    /// Draws every accepted trace whose seed block is known: the first chain block after the
    /// accepting one at `TRACE_SEED_DELAY_DAA` or more.
    fn resolve_traces(&mut self) {
        let mut drawn: Vec<(String, TraceDraw)> = Vec::new();
        for (txid, p) in self.pending_traces.iter() {
            if p.draw.is_some() {
                continue;
            }
            let Some(accepted) = p.accepted_hash.as_deref() else { continue };
            if self.chain_restore_starts.iter().any(|h| h == accepted) {
                continue;
            }
            let Some((seed_hash, daa, accepted_daa)) = seed_block_after(&self.chain_log, accepted) else { continue };
            let (Some(txid_bytes), Some(seed_block)) = (hash32(txid), hash32(seed_hash)) else { continue };
            let seed = keryx_miner::trace::trace_seed(&seed_block, &txid_bytes);
            let audited = keryx_miner::trace::audit_all_active(accepted_daa) || keryx_miner::trace::audited(&seed);
            let deadline_daa = daa + keryx_miner::trace::opening_window_daa(p.tier, p.max_tokens);
            drawn.push((
                txid.clone(),
                TraceDraw {
                    seed_hash: seed_hash.to_string(),
                    seed_daa: daa,
                    seed,
                    audited,
                    deadline_daa,
                    opened: p.opening.as_ref().is_some_and(|o| o.seed_hash == seed_hash),
                    replay_failures: 0,
                },
            ));
        }
        for (txid, d) in drawn {
            if d.audited {
                let reorgs = self.pending_traces.get(&txid).map_or(0, |p| p.reorgs);
                if reorgs == 0 {
                    info!("Trace: response {} drawn for audit, opening due by DAA {}", txid, d.deadline_daa);
                } else {
                    info!("Trace: response {} drawn for audit after {} chain reorgs, opening due by DAA {}", txid, reorgs, d.deadline_daa);
                }
            }
            if let Some(p) = self.pending_traces.get_mut(&txid) {
                p.draw = Some(d);
            }
        }
    }

    /// Drops the traces past their window and the chain log they no longer need.
    fn prune_traces(&mut self) {
        let daa = self.last_known_daa;
        let before: Vec<String> = self.pending_traces.keys().cloned().collect();
        self.pending_traces.retain(|txid, p| match p.draw.as_ref() {
            None => p.accepted_hash.is_some() || p.submitted_daa + AI_BACKFILL_WINDOW_DAA > daa,
            Some(d) if !d.audited => d.seed_daa + TRACE_UNAUDITED_GRACE_DAA > daa,
            Some(d) => {
                if daa > d.deadline_daa && !d.opened {
                    warn!("Trace: response {} was never opened — window closed", txid);
                }
                daa <= d.deadline_daa
            }
        });
        for txid in before.iter().filter(|t| !self.pending_traces.contains_key(*t)) {
            self.trace_store.remove(txid);
        }
        let needed = self
            .pending_traces
            .values()
            .filter_map(|p| p.accepted_hash.as_deref())
            .filter_map(|h| self.chain_log.iter().position(|e| e.hash == h))
            .min()
            .unwrap_or(self.chain_log.len());
        let keep_from = needed.min(self.chain_log.len().saturating_sub(CHAIN_LOG_TAIL));
        self.chain_log.drain(..keep_from);
    }

    /// Seeds the startup backfill from the first block notification.
    fn backfill_seed(&mut self, header: Option<&crate::proto::RpcBlockHeader>) {
        if self.backfill_cutoff_daa.is_some() {
            return;
        }
        let Some(header) = header else { return };
        let cutoff = header.daa_score.saturating_sub(AI_BACKFILL_WINDOW_DAA);
        self.backfill_cutoff_daa = Some(cutoff);
        if let Some(level0) = header.parents.first() {
            for hash in &level0.parent_hashes {
                if self.backfill_visited.insert(hash.clone()) {
                    self.backfill_queue.push_back(hash.clone());
                }
            }
        }
        info!("Inference: backfilling recent blocks for in-window AiRequests (cutoff daa {})", cutoff);
    }

    /// Keeps up to AI_BACKFILL_INFLIGHT backfill block requests in flight.
    async fn backfill_pump(&mut self) -> Result<(), Error> {
        if self.backfill_cutoff_daa.is_none() {
            return Ok(());
        }
        while self.backfill_pending.len() < AI_BACKFILL_INFLIGHT {
            let Some(hash) = self.backfill_queue.pop_front() else { break };
            self.backfill_pending.insert(hash.clone());
            self.client_send(GetBlockRequestMessage { hash, include_transactions: true }).await?;
        }
        Ok(())
    }

    /// Notes a fetched backfill block: walks its parents while above the cutoff.
    fn backfill_note_block(&mut self, hash: &str, header: Option<&crate::proto::RpcBlockHeader>) {
        if !self.backfill_pending.remove(hash) {
            return;
        }
        let Some(cutoff) = self.backfill_cutoff_daa else { return };
        if let Some(header) = header {
            if header.daa_score > cutoff {
                if let Some(level0) = header.parents.first() {
                    for parent in &level0.parent_hashes {
                        if self.backfill_visited.insert(parent.clone()) {
                            self.backfill_queue.push_back(parent.clone());
                        }
                    }
                }
            }
        }
        if self.backfill_queue.is_empty() && self.backfill_pending.is_empty() {
            info!("Inference: startup backfill complete ({} blocks scanned)", self.backfill_visited.len());
            self.backfill_visited.clear();
            self.backfill_visited.shrink_to_fit();
        }
    }

    /// Clears a backfill entry whose GetBlock failed (pruned or unknown block).
    fn backfill_note_error(&mut self, message: &str) {
        if let Some(hash) = self.backfill_pending.iter().find(|h| message.contains(h.as_str())).cloned() {
            self.backfill_pending.remove(&hash);
        }
    }

    /// Resubmits in-flight AiResponses the mempool has not accepted yet.
    async fn retry_pending_ai_responses(&mut self) -> Result<(), Error> {
        if self.ai_response_inflight.is_empty() {
            return Ok(());
        }
        let now = std::time::Instant::now();
        let mut resend: Vec<crate::proto::RpcTransaction> = Vec::new();
        self.ai_response_inflight.retain(|txid, (tx, attempts, last)| {
            if now.duration_since(*last).as_secs() < AI_RESPONSE_RETRY_SECS {
                return true;
            }
            if *attempts >= AI_RESPONSE_MAX_ATTEMPTS {
                warn!("Inference: giving up on AiResponse {} after {} submit attempts", txid, attempts);
                return false;
            }
            *attempts += 1;
            *last = now;
            resend.push(tx.clone());
            true
        });
        for tx in resend {
            self.client_send(KaspadMessage::submit_transaction(tx)).await?;
        }
        Ok(())
    }

    /// Polls the in-flight inference task, then the publish task. A finished inference hands
    /// its text to a publish task (IPFS upload, then a gateway fetch of the CID); the
    /// AiResponse is submitted only once a public gateway has served the content.
    /// Returns `true` when a request just reached its end (tx submitted or dropped).
    async fn poll_inference(&mut self) -> bool {
        if let Some(mut pending) = self.publish_rx.take() {
            return match pending.rx.try_recv() {
                Ok(Some(cid)) => {
                    self.submit_ai_response(pending.request_hash, &pending.result, cid, None, None).await;
                    true
                }
                Ok(None) => true,
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    self.publish_rx = Some(pending);
                    false
                }
                Err(_) => {
                    warn!("Inference: publish task dropped — AiResponse skipped");
                    true
                }
            };
        }
        let Some(mut job) = self.inference_rx.take() else {
            return false;
        };
        let Ok(result_opt) = job.rx.try_recv() else {
            self.inference_rx = Some(job);
            return false;
        };
        let Some(generated) = result_opt else {
            // Inference returned None: model not ready or think block exhausted max_tokens.
            // Do NOT upload anything to IPFS — skip this AiResponse entirely.
            info!("Inference: inference produced no result — AiResponse skipped");
            return true;
        };
        let InferenceJob { request_hash, private, model_id, tier, max_tokens, .. } = job;
        let result = generated.text;
        let trace = match (generated.trace, tier) {
            (Some(trace), Some(tier)) => Some(TraceContext { model_id, tier, max_tokens, trace }),
            _ => None,
        };

        info!("Inference: inference complete, request_hash={}", hex::encode(&request_hash[..8]));

        // A private answer is sealed to the requester; the plaintext is never published. From
        // the gate on it travels inline and the CID field carries its multihash; before the
        // gate the sealed body is pinned on IPFS like a plaintext answer.
        let publish: Vec<u8> = match private {
            None => result.clone().into_bytes(),
            Some(p) => {
                let body = match p {
                    PrivateRequest::Opened(key) => {
                        let plain = match trace.as_ref().filter(|t| t.trace.record.stage == keryx_miner::trace::STAGE_CHAIN) {
                            Some(t) => {
                                let ids: Vec<u32> = t.trace.tokens[t.trace.record.n_prompt as usize..].iter().map(|&v| v as u32).collect();
                                keryx_verifiable_inference::answer::encode(result.as_bytes(), &ids)
                            }
                            None => result.clone().into_bytes(),
                        };
                        keryx_inference::seal_response(&key, &request_hash, &plain)
                    }
                    PrivateRequest::Unreadable => result.clone().into_bytes(),
                };
                let cap = if keryx_miner::trace::chain_active(self.last_known_daa) {
                    keryx_inference::MAX_AI_RESPONSE_PRIVATE_BODY_LEN
                } else {
                    keryx_inference::MAX_AI_RESPONSE_PRIVATE_BODY_LEN_H14
                };
                if body.len() > cap {
                    warn!("Inference: sealed answer of {} B exceeds the {} B body limit — AiResponse skipped", body.len(), cap);
                    return true;
                }
                if self.last_known_daa >= keryx_miner::pom::private_inference_activation_daa() {
                    let cid = crate::ipfs::sha256_multihash(&body);
                    self.submit_ai_response(request_hash, &result, cid, Some(body), trace).await;
                    return true;
                }
                body
            }
        };

        let (tx_cid, rx_cid) = oneshot::channel::<Option<[u8; 34]>>();
        let ipfs_url = self.ipfs_url.clone();
        tokio::task::spawn_blocking(move || {
            let cid = match crate::ipfs::upload_bytes(&publish, &ipfs_url) {
                Ok(cid) => cid,
                Err(e) => {
                    warn!("Inference: IPFS upload failed: {} — AiResponse tx skipped", e);
                    let _ = tx_cid.send(None);
                    return;
                }
            };
            info!("Inference: response published to IPFS");
            let _ = tx_cid.send(Some(cid));
        });
        self.publish_rx = Some(PendingPublish { request_hash, result, rx: rx_cid });
        false
    }

    /// Builds and submits the zero-input/zero-output AiResponse for a confirmed CID. A
    /// `private_body` rides inline under the responder signature (V2 only).
    async fn submit_ai_response(
        &mut self,
        request_hash: [u8; 32],
        result: &str,
        cid: [u8; 34],
        private_body: Option<Vec<u8>>,
        trace: Option<TraceContext>,
    ) {
        let challenge_window_end = self.last_known_daa + 1000;
        let response_length = result.split_whitespace().count() as u32;
        // The trace commitment rides after the private body only.
        let trace = trace.filter(|_| private_body.is_some());
        let commitment = trace.as_ref().map(|t| t.trace.record.commitment());
        // H6 service-bond era: sign the response with the escrow key (payload V2) so it counts
        // as served for the tier cohort — an unsigned response no longer cancels a strike. The
        // era rule mirrors the node's: V2 is rejected before the gate, so v1 is kept below it.
        let v2 = self.last_known_daa >= keryx_miner::pom::pom_v3_activation_daa();
        let resp = match (&self.escrow_watcher, v2) {
            (Some(w), true) => {
                let mut unsigned = keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length);
                if let Some(body) = &private_body {
                    unsigned = unsigned.with_private_body(body.clone());
                }
                if let Some(c) = commitment {
                    unsigned = unsigned.with_trace(c);
                }
                let responder = w.sign_responder(&unsigned.signed_bytes());
                let resp = keryx_inference::AiResponsePayload::new_v2(request_hash, challenge_window_end, cid, response_length, responder);
                let resp = match private_body {
                    Some(body) => resp.with_private_body(body),
                    None => resp,
                };
                match commitment {
                    Some(c) => resp.with_trace(c),
                    None => resp,
                }
            }
            (None, true) => {
                warn!("Inference: no escrow key configured — submitting an unsigned (v1) response; it will NOT count for the service bond");
                keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length)
            }
            (_, false) => keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length),
        };
        info!("Inference: uploading response CID={}, challenge_window_end={}{}{}{}", resp.cid_v0(), challenge_window_end,
            if resp.responder.is_some() { " (signed, V2)" } else { "" },
            if resp.private_body.is_some() { " (private body inline)" } else { "" },
            if resp.trace.is_some() { " (traced)" } else { "" });

        let rpc_tx = crate::proto::RpcTransaction {
            version: 0,
            inputs: vec![],
            outputs: vec![],
            lock_time: 0,
            subnetwork_id: keryx_inference::SUBNETWORK_ID_AI_RESPONSE_HEX.to_string(),
            gas: 0,
            payload: hex::encode(resp.serialize()),
            mass: 0,
            verbose_data: None,
        };
        if let Some(txid) = Self::compute_rpc_txid(&rpc_tx) {
            if let Some(t) = trace.filter(|_| resp.trace.is_some()) {
                let pending = PendingTrace {
                    model_id: t.model_id,
                    tier: t.tier,
                    request_hash,
                    max_tokens: t.max_tokens,
                    trace: Arc::new(t.trace),
                    submitted_daa: self.last_known_daa,
                    accepted_hash: None,
                    draw: None,
                    opening: None,
                    built: Vec::new(),
                    reorgs: 0,
                };
                self.pending_traces.insert(txid.clone(), pending);
                self.persist_trace(&txid);
            }
            self.ai_response_inflight.insert(txid, (rpc_tx.clone(), 1, std::time::Instant::now()));
        }
        if let Err(e) = self.client_send(KaspadMessage::submit_transaction(rpc_tx)).await {
            warn!("Inference: failed to send AiResponse tx: {}", e);
        }

        // Register inference escrow outpoint for auto-claim after the challenge window.
        let stable_id = hex::encode(&request_hash[..8]);
        if let Some((txid, inference_reward)) = self.ai_request_txids.remove(&stable_id) {
            if let Some(w) = self.escrow_watcher.as_mut() {
                w.track_inference_escrow(txid, self.last_known_daa, inference_reward);
            }
        }
    }

    /// Logs this miner's service-bond standing when it changes: strike count, burns awaiting
    /// finality and production suspensions, matched by payout-address identity.
    fn report_service_strikes(&mut self, resp: &crate::proto::GetServiceStrikesResponseMessage) {
        let Some(me) = self.service_identity.as_deref() else { return };
        let strike = resp.strikes.iter().find(|s| s.miner.eq_ignore_ascii_case(me));
        let suspension = resp.suspended.iter().find(|s| s.miner.eq_ignore_ascii_case(me));
        let burns: Vec<_> = resp.pending_burns.iter().filter(|b| b.miner.eq_ignore_ascii_case(me)).collect();
        let status = if strike.is_none() && suspension.is_none() && burns.is_empty() {
            "clear".to_string()
        } else {
            let mut parts = Vec::new();
            if let Some(s) = strike {
                parts.push(format!("strike {} (last at daa {})", s.consecutive_misses, s.last_strike_daa_score));
            }
            if !burns.is_empty() {
                let claims: u32 = burns.iter().map(|b| b.burned_claims).sum();
                let sompi: u64 = burns.iter().map(|b| b.burned_sompi).sum();
                parts.push(format!("{} escrow claims / {:.2} KRX burning at finality", claims, sompi as f64 / 100_000_000.0));
            }
            if let Some(s) = suspension {
                parts.push(format!("production suspended until daa {}", s.until_daa_score));
            }
            parts.join("; ")
        };
        let active = strike.map(|s| s.consecutive_misses).unwrap_or(0);
        // The node keeps the lifetime tally: it survives restarts and covers the time this miner
        // was down. Fall back to the locally observed misses only when talking to a node that
        // predates the field (it answers with an empty list).
        for burn in &burns {
            self.misses_seen.insert(burn.miss_daa_score);
        }
        let total = resp
            .lifetime_strikes
            .iter()
            .find(|t| t.miner.eq_ignore_ascii_case(me))
            .map(|t| t.strikes as usize)
            .unwrap_or(self.misses_seen.len());

        if self.strike_status.as_deref() != Some(status.as_str()) {
            match status.as_str() {
                "clear" => info!("service-bond: no strikes against this miner"),
                s => warn!("service-bond: {} ({} active, {} lifetime)", s, active, total),
            }
            self.strike_status = Some(status);
        }
    }

    async fn handle_message(&mut self, msg: Payload, miner: &mut MinerManager) -> Result<(), Error> {
        match msg {
            // BlockAdded: scan confirmed block for AiRequests and escrow UTXOs.
            // Do NOT trigger a new block template here — NewBlockTemplate handles that.
            Payload::BlockAddedNotification(notif) => {
                if let Some(block) = notif.block {
                    self.backfill_seed(block.header.as_ref());
                    self.backfill_pump().await?;
                    if !block.transactions.is_empty() {
                        // Full block — scan directly.
                        self.scan_txs_for_ai_requests(&block.transactions, block.header.as_ref().map_or(0, |h| h.daa_score));
                        // Pause and drain mining BEFORE spawning inference, so no PoM op can
                        // race the model swap: PAUSE -> DRAIN -> SWAP -> GENERATE -> RESUME.
                        if !self.inference_busy() && !self.ai_request_queue.is_empty()
                        {
                            self.set_opoi_pause(true);
                            keryx_miner::pom_gpu::log_throttled(u32::MAX, "tpl-queue-a", || info!("Template held: request queue of {} waiting to start", self.ai_request_queue.len()));
                            miner.process_block(None).await?;
                            self.try_start_inference();
                        }
                        // Escrow: check for new escrow UTXOs and mature claims.
                        let claim_tx = self.escrow_watcher.as_mut().and_then(|w| w.handle_block(&block));
                        if let Some(w) = self.escrow_watcher.as_ref() {
                            let (outputs, sompi) = w.pending_escrow();
                            miner.record_escrow_pending(outputs, sompi);
                        }
                        if let Some(tx) = claim_tx {
                            self.client_send(KaspadMessage::submit_transaction(tx)).await?;
                        }
                    } else {
                        // Transactions absent — fetch the full block from the node.
                        let hash = block
                            .verbose_data
                            .as_ref()
                            .map(|v| v.hash.clone())
                            .unwrap_or_default();
                        if !hash.is_empty() {
                            self.client_send(GetBlockRequestMessage {
                                hash,
                                include_transactions: true,
                            })
                            .await?;
                        }
                    }
                }
            }
            Payload::NewBlockTemplateNotification(_) => self.client_get_block_template().await?,
            Payload::GetServiceStrikesResponse(resp) => match resp.error.as_ref() {
                Some(e) => warn!("service-bond status unavailable: {}", e.message),
                None => self.report_service_strikes(&resp),
            },
            Payload::GetBlockTemplateResponse(template) => {
                static TEMPLATES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let n = TEMPLATES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                let daa = template.block.as_ref().and_then(|b| b.header.as_ref()).map_or(0, |h| h.daa_score);
                let (synced, err) = (template.is_synced, template.error.is_some());
                keryx_miner::pom_gpu::log_throttled(u32::MAX, "tpl-count", || {
                    info!("Templates: {} received, last at daa {} (synced {}, error {}, queue {})", n, daa, synced, err, self.ai_request_queue.len())
                });
                // Track DAA score for challenge_window_end computation.
                if let Some(daa) = template.block.as_ref()
                    .and_then(|b| b.header.as_ref())
                    .map(|h| h.daa_score)
                {
                    if daa > self.last_known_daa {
                        self.last_known_daa = daa;
                        keryx_miner::slm::note_chain_daa(daa);
                    }
                }
                // Handle node-issued inference challenge: spawn an inference task if a new
                // challenge arrived and no challenge is already in flight. Ignored under PoM — the
                // per-block possession proof is the capability gate, so no synthetic challenge
                // (defensive: holds even against a node that still issues them post-hardfork).
                if !template.inference_challenge.is_empty()
                    && !self.inference_busy()
                    && self.last_known_daa < keryx_miner::pom::pom_activation_daa()
                {
                    let challenge = template.inference_challenge.clone();
                    let mut parts = challenge.splitn(2, ':');
                    let model_id_hex = parts.next().unwrap_or("").to_string();
                    let nonce_hex = parts.next().unwrap_or("").to_string();
                    if let Ok(model_id_bytes) = hex::decode(&model_id_hex) {
                        if model_id_bytes.len() == 32 {
                            let mut model_id = [0u8; 32];
                            model_id.copy_from_slice(&model_id_bytes);
                            if keryx_miner::slm::is_model_ready(&model_id) {
                                self.set_opoi_pause(true);
                                miner.process_block(None).await?;
                                info!("Inference: challenge received model={:.8} nonce={:.8} — spawning inference", model_id_hex, nonce_hex);
                                let prompt = format!("Keryx inference challenge {}: briefly describe what you are.", nonce_hex);
                                let (tx_done, rx_done) = oneshot::channel::<Option<String>>();
                                let label = format!("solo challenge {:.8}", nonce_hex);
                                tokio::task::spawn_blocking(move || {
                                    let result = keryx_miner::slm::load_and_run_inference(&model_id, &prompt, 64, &label);
                                    let _ = tx_done.send(result);
                                });
                                self.challenge_inference_rx = Some((challenge, rx_done));
                            } else {
                                warn!("Inference: challenge for unready model={:.8} — cannot respond", model_id_hex);
                            }
                        }
                    }
                }
                // Poll in-flight inference; if done, submit AiResponse tx then get fresh template.
                if self.poll_inference().await {
                    self.client_get_block_template().await?;
                    return Ok(());
                }
                // Inference is mandatory: refuse to mine if no models are ready.
                // Covers miners with missing/truncated model files that somehow passed prefetch.
                if keryx_miner::slm::loaded_model_ids().is_empty() {
                    keryx_miner::pom_gpu::log_throttled(u32::MAX, "tpl-no-models", || warn!("Template held: no loaded model"));
                    // Throttle to one log per ~200 templates (~every 20s at 10 BPS) to avoid spam.
                    if self.last_known_daa % 200 == 0 {
                        log::warn!("Inference: no models ready — mining suspended until model files are available");
                    }
                    self.set_opoi_pause(true);
                    miner.process_block(None).await?;
                    return Ok(());
                }
                if let Some(ref block) = template.block {
                    self.scan_txs_for_ai_requests(&block.transactions, block.header.as_ref().map_or(0, |h| h.daa_score));
                }
                if !self.inference_busy() && !self.ai_request_queue.is_empty()
                {
                    self.set_opoi_pause(true);
                    keryx_miner::pom_gpu::log_throttled(u32::MAX, "tpl-queue-b", || info!("Template held: request queue of {} waiting to start", self.ai_request_queue.len()));
                    miner.process_block(None).await?;
                    self.try_start_inference();
                }
                // Pause GPU mining while any inference is in flight (GPU is occupied by the model).
                // This covers both regular AiRequest inference and node-issued challenge inference.
                if self.inference_busy() {
                    let (inf, chal, rep) = (self.inference_rx.is_some(), self.challenge_inference_rx.is_some(), self.replay_rx.is_some());
                    keryx_miner::pom_gpu::log_throttled(u32::MAX, "tpl-busy", || {
                        info!("Template held: inference {} challenge {} replay {}", inf, chal, rep)
                    });
                    self.set_opoi_pause(true);
                    miner.process_block(None).await?;
                    return Ok(());
                }
                // Past this point any pause comes from the node, not from us.
                self.set_opoi_pause(false);
                if let Some(header) = template.block.as_ref().and_then(|b| b.header.as_ref()) {
                    self.resolve_task_seed(header).await;
                }
                match (template.block, template.is_synced, template.error) {
                    (Some(b), true, None) => miner.process_block(Some(FullBlock(Box::new(b)))).await?,
                    (Some(b), false, None) if self.mine_when_not_synced => {
                        miner.process_block(Some(FullBlock(Box::new(b)))).await?
                    }
                    (_, false, None) => {
                        keryx_miner::pom_gpu::log_throttled(u32::MAX, "tpl-unsynced", || warn!("Template held: node reports not synced"));
                        miner.process_block(None).await?
                    }
                    (_, _, Some(e)) => {
                        return Err(format!("GetTemplate returned with an error: {:?}", e).into());
                    }
                    (None, true, None) => error!("No block and No Error!"),
                }
            }
            // GetBlock response: either a boot-time validation answer, or a full block we
            // requested from BlockAdded (scanned for AiRequests and escrow UTXOs).
            Payload::GetBlockResponse(msg) => {
                let mut was_validation = false;
                if let Some(e) = msg.error {
                    if let Some(pos) = self.chain_log.iter().position(|c| e.message.contains(c.hash.as_str())) {
                        warn!("Trace: chain block {} unavailable — left out of the chain log", self.chain_log[pos].hash);
                        self.chain_log.remove(pos);
                        self.resolve_traces();
                    }
                    // Validation answer: "cannot find header <hash>" — unknown to this
                    // node (pruned or not yet synced), the entries are kept.
                    was_validation = self
                        .escrow_watcher
                        .as_mut()
                        .map_or(false, |w| w.on_block_validation_error(&e.message));
                    if let Some(h) = self.chain_restore_pending.iter().find(|h| e.message.contains(h.as_str())).cloned() {
                        self.chain_restore_pending.remove(&h);
                        self.chain_restore_pump().await?;
                    }
                    if !was_validation {
                        if self.backfill_pending.iter().any(|h| e.message.contains(h.as_str())) {
                            self.backfill_note_error(&e.message);
                            self.backfill_pump().await?;
                        } else {
                            warn!("GetBlockResponse error: {}", e.message);
                        }
                    }
                } else if let Some(block) = msg.block {
                    let hash = block.verbose_data.as_ref().map(|v| v.hash.clone()).unwrap_or_default();
                    // Chain membership from the node's live verdict: a stored-but-reorged
                    // block must purge its entries just like a missing one.
                    let is_chain = block.verbose_data.as_ref().map_or(false, |v| v.is_chain_block);
                    if let Some(daa) = block.header.as_ref().map(|h| h.daa_score) {
                        self.note_chain_block(&hash, daa);
                        self.resolve_traces();
                    }
                    if self.chain_restore_pending.remove(&hash) {
                        self.chain_restore_pump().await?;
                    }
                    was_validation = block.transactions.is_empty()
                        && self.escrow_watcher.as_mut().map_or(false, |w| w.consume_validation_ok(&hash, is_chain));
                    if !was_validation {
                        self.scan_txs_for_ai_requests(&block.transactions, block.header.as_ref().map_or(0, |h| h.daa_score));
                        self.backfill_note_block(&hash, block.header.as_ref());
                        self.backfill_pump().await?;
                        if !self.inference_busy() && !self.ai_request_queue.is_empty()
                        {
                            self.set_opoi_pause(true);
                            keryx_miner::pom_gpu::log_throttled(u32::MAX, "tpl-queue-a", || info!("Template held: request queue of {} waiting to start", self.ai_request_queue.len()));
                            miner.process_block(None).await?;
                            self.try_start_inference();
                        }
                        let claim_tx = self.escrow_watcher.as_mut().and_then(|w| w.handle_block(&block));
                        if let Some(w) = self.escrow_watcher.as_ref() {
                            let (outputs, sompi) = w.pending_escrow();
                            miner.record_escrow_pending(outputs, sompi);
                        }
                        if let Some(tx) = claim_tx {
                            self.client_send(KaspadMessage::submit_transaction(tx)).await?;
                        }
                    }
                }
                // Self-paced validation flow: every consumed answer pulls the next
                // queued request, keeping at most VALIDATION_WINDOW in flight.
                if was_validation {
                    if let Some(hash) = self.validation_queue.pop_front() {
                        self.client_send(GetBlockRequestMessage { hash, include_transactions: false }).await?;
                    }
                }
            }
            Payload::SubmitBlockResponse(res) => match res.error {
                None => info!("Block submitted successfully!"),
                Some(e) => warn!("Failed submitting block: {:?}", e),
            },
            Payload::SubmitTransactionResponse(res) => {
                // Escrow claims and inference submissions share this stream. Match responses to
                // in-flight claims by identity (txid, or the txid embedded in the rejection
                // text) — attributing by position slashed valid escrow entries before.
                use crate::escrow::SubmitResponseOutcome;
                let err = res.error.as_ref().map(|e| e.message.clone());
                let outcome = self
                    .escrow_watcher
                    .as_mut()
                    .map_or(SubmitResponseOutcome::NotOurs, |w| {
                        w.on_submit_response(&res.transaction_id, err.as_deref())
                    });
                match outcome {
                    SubmitResponseOutcome::Accepted { outputs, amount_sompi } => {
                        miner.record_claim_accepted(outputs, amount_sompi);
                    }
                    SubmitResponseOutcome::Handled => {}
                    SubmitResponseOutcome::NotOurs => {
                        let opening_txid = if self.opening_inflight.contains_key(&res.transaction_id) {
                            Some(res.transaction_id.clone())
                        } else {
                            err.as_ref().and_then(|e| self.opening_inflight.keys().find(|k| e.contains(k.as_str())).cloned())
                        };
                        if let Some(txid) = opening_txid {
                            match err.as_deref() {
                                None => {
                                    self.opening_inflight.remove(&txid);
                                    info!("Trace: opening accepted by the mempool");
                                }
                                Some(e) if e.contains("an opening for response") => {
                                    info!("Trace: another opening of this response is in the mempool — retrying");
                                }
                                Some(e) if e.contains("already") => {
                                    self.opening_inflight.remove(&txid);
                                    info!("Trace: opening already known to the node — done");
                                }
                                Some(e) if e.contains("not awaited") => {
                                    let attempts = self.opening_inflight.get(&txid).map_or(1, |o| o.attempts);
                                    if attempts >= OPENING_NOT_AWAITED_MAX_ATTEMPTS {
                                        self.opening_inflight.remove(&txid);
                                        warn!("Trace: opening not awaited by the node — dropped: {}", e);
                                    }
                                }
                                Some(e) => {
                                    let attempts = self.opening_inflight.get(&txid).map_or(1, |o| o.attempts);
                                    if attempts <= 1 {
                                        warn!("Trace: opening rejected: {} — retrying until accepted or expired", e);
                                    } else {
                                        log::debug!("Trace: opening rejected (attempt {}): {}", attempts, e);
                                    }
                                }
                            }
                            return Ok(());
                        }
                        let inflight_txid = if self.ai_response_inflight.contains_key(&res.transaction_id) {
                            Some(res.transaction_id.clone())
                        } else {
                            // Rejections may carry the txid in the message text instead.
                            err.as_ref().and_then(|e| self.ai_response_inflight.keys().find(|k| e.contains(k.as_str())).cloned())
                        };
                        match (inflight_txid, err) {
                            (Some(txid), None) => {
                                self.ai_response_inflight.remove(&txid);
                                info!("Inference: AiResponse accepted by the mempool");
                            }
                            (Some(txid), Some(e)) if (e.contains(&txid) && e.contains("already")) || e.contains("same responder") => {
                                self.ai_response_inflight.remove(&txid);
                                info!("Inference: AiResponse already known to the node — done");
                            }
                            (Some(txid), Some(e)) => {
                                let attempts = self.ai_response_inflight.get(&txid).map(|(_, a, _)| *a).unwrap_or(1);
                                if attempts <= 1 {
                                    warn!("Inference: AiResponse rejected: {} — retrying until accepted or expired", e);
                                } else {
                                    log::debug!("Inference: AiResponse rejected (attempt {}): {}", attempts, e);
                                }
                            }
                            (None, Some(e)) => log::debug!("Inference: submit_transaction error: {}", e),
                            (None, None) => {}
                        }
                    }
                }
            }
            Payload::GetInfoResponse(info) => {
                info!("Keryxd version: {}", info.server_version);
                // Register for all notification types:
                // - NewBlockTemplate drives the mining loop
                // - BlockAdded lets us scan confirmed blocks for AiRequests
                //   that were confirmed before the miner saw them in mempool
                // - VirtualChainChanged drives escrow tracking: only chain-block coinbases
                //   materialize UTXOs, so escrow outputs are tracked from chain blocks only
                self.client_send(NotifyNewBlockTemplateRequestMessage {}).await?;
                self.client_send(NotifyBlockAddedRequestMessage {}).await?;
                self.client_send(NotifyVirtualSelectedParentChainChangedRequestMessage { include_accepted_transaction_ids: true }).await?;
                // Boot-time escrow-state validation: check every referenced block against
                // the node so ghost entries (orphaned-chain coinbases) are purged before
                // any claim ships. Send an initial slice; each answer sends the next.
                if let Some(hashes) = self.escrow_watcher.as_mut().map(|w| w.start_state_validation()) {
                    self.validation_queue = hashes.into();
                    for _ in 0..VALIDATION_WINDOW {
                        if let Some(hash) = self.validation_queue.pop_front() {
                            self.client_send(GetBlockRequestMessage { hash, include_transactions: false }).await?;
                        }
                    }
                }
                self.client_get_block_template().await?;
            }
            Payload::NotifyNewBlockTemplateResponse(res) => match res.error {
                None => info!("Registered for new template notifications"),
                Some(e) => error!("Failed registering for new template notifications: {:?}", e),
            },
            Payload::NotifyBlockAddedResponse(res) => match res.error {
                None => info!("Registered for block added notifications (AI request scanning)"),
                Some(e) => error!("Failed registering for block added notifications: {:?}", e),
            },
            Payload::NotifyVirtualSelectedParentChainChangedResponse(res) => match res.error {
                None => {
                    info!("Registered for virtual chain notifications (escrow tracking)");
                    self.start_chain_restore().await?;
                }
                Some(e) => error!("Failed registering for virtual chain notifications: {:?}", e),
            },
            // Virtual chain advanced: fetch every added chain block in full. Their coinbases
            // are the only ones that materialize UTXOs, so escrow tracking feeds off this
            // stream (handle_block gates tracking on is_chain_block). Entries of removed
            // chain blocks are purged right away: their coinbase no longer exists.
            Payload::VirtualSelectedParentChainChangedNotification(notif) => {
                if let Some(watcher) = self.escrow_watcher.as_mut() {
                    watcher.on_chain_blocks_removed(&notif.removed_chain_block_hashes);
                }
                self.on_chain_blocks_removed(&notif.removed_chain_block_hashes);
                for hash in &notif.added_chain_block_hashes {
                    if !self.chain_log.iter().any(|e| &e.hash == hash) {
                        self.chain_log.push_back(ChainEntry { hash: hash.clone(), daa: None });
                    }
                }
                let mut accepted_now: Vec<String> = Vec::new();
                for accepted in &notif.accepted_transaction_ids {
                    for txid in &accepted.accepted_transaction_ids {
                        if let Some(p) = self.pending_traces.get_mut(txid).filter(|p| p.accepted_hash.is_none()) {
                            p.accepted_hash = Some(accepted.accepting_block_hash.clone());
                            if p.reorgs == 0 {
                                info!("Trace: response {} accepted by chain block {}", txid, accepted.accepting_block_hash);
                            } else {
                                log::debug!("Trace: response {} accepted by chain block {}", txid, accepted.accepting_block_hash);
                            }
                            accepted_now.push(txid.clone());
                        }
                    }
                }
                for txid in accepted_now {
                    self.persist_trace(&txid);
                }
                for hash in notif.added_chain_block_hashes {
                    self.client_send(GetBlockRequestMessage { hash, include_transactions: true }).await?;
                }
            }
            Payload::GetVirtualSelectedParentChainFromBlockResponse(msg) => {
                let Some(start) = self.chain_restore_starts.pop_front() else { return Ok(()) };
                match msg.error {
                    Some(e) => self.on_chain_restore_failed(start, &e.message),
                    None => {
                        self.chain_restore_failures.remove(&start);
                        self.place_restored_chain(&start, &msg.removed_chain_block_hashes, msg.added_chain_block_hashes);
                        self.note_restored_acceptances(&msg.accepted_transaction_ids);
                    }
                }
                self.resolve_traces();
                self.request_next_chain_restore().await?;
                self.chain_restore_pump().await?;
            }
            msg => info!("got unknown msg: {:?}", msg),
        }
        Ok(())
    }
}

impl Drop for KeryxdHandler {
    fn drop(&mut self) {
        self.block_handle.abort();
        let queue = std::mem::take(&mut self.ai_request_queue);
        let inference = self.inference_rx.take();
        if !queue.is_empty() || inference.is_some() {
            *PARKED.lock().unwrap_or_else(|p| p.into_inner()) = Some(Parked { queue, inference });
        }
    }
}

type QueuedRequest = (String, [u8; 32], [u8; 32], String, u32, usize, Option<PrivateRequest>);

/// Request queue and in-flight inference left by a closed connection.
struct Parked {
    queue: VecDeque<QueuedRequest>,
    inference: Option<InferenceJob>,
}

static PARKED: Mutex<Option<Parked>> = Mutex::new(None);

/// Leading bytes of a private prompt compressed with raw DEFLATE.
const COMPRESSED_PROMPT_MAGIC: [u8; 4] = [0x00, b'K', b'Z', 0x01];
const MAX_DECOMPRESSED_PROMPT_LEN: u64 = 1 << 20;

/// Plaintext of an opened private prompt: inflated when it carries the compression magic.
/// `None` for a compressed prompt that is corrupt or inflates past the cap.
fn decode_private_prompt(bytes: &[u8]) -> Option<String> {
    use std::io::Read;
    let Some(deflated) = bytes.strip_prefix(&COMPRESSED_PROMPT_MAGIC[..]) else {
        return Some(String::from_utf8_lossy(bytes).into_owned());
    };
    let mut inflated = Vec::new();
    flate2::read::DeflateDecoder::new(deflated).take(MAX_DECOMPRESSED_PROMPT_LEN + 1).read_to_end(&mut inflated).ok()?;
    if inflated.len() as u64 > MAX_DECOMPRESSED_PROMPT_LEN {
        return None;
    }
    Some(String::from_utf8_lossy(&inflated).into_owned())
}

#[cfg(test)]
mod tests {
    use super::PrivateRequest;
    use keryx_inference::{escrow_pubkey_of, open_response, seal_request, AiRequestPayload};

    /// Only the responses signed by this miner's escrow key mark a request as already answered.
    #[test]
    fn own_answered_request_ids_keeps_only_this_responder() {
        use keryx_inference::{AiResponder, AiResponsePayload};
        let mine = [0xAAu8; 32];
        let other = [0xBBu8; 32];
        let response = |request_hash: [u8; 32], key: [u8; 32]| crate::proto::RpcTransaction {
            subnetwork_id: keryx_inference::SUBNETWORK_ID_AI_RESPONSE_HEX.to_string(),
            payload: hex::encode(
                AiResponsePayload::new_v2(request_hash, 10, [0x12u8; 34], 3, AiResponder { escrow_pubkey: key, signature: [0u8; 64] })
                    .serialize(),
            ),
            ..Default::default()
        };
        let txs = vec![response([1u8; 32], mine), response([2u8; 32], other), response([3u8; 32], mine)];
        let ids = super::KeryxdHandler::own_answered_request_ids(&txs, &mine);
        assert_eq!(ids, vec![hex::encode(&[1u8; 8]), hex::encode(&[3u8; 8])]);
        assert!(super::KeryxdHandler::own_answered_request_ids(&txs, &[0xCCu8; 32]).is_empty());
    }

    /// A private request is opened for a named recipient, skipped for anyone else, and queued as
    /// unreadable when addressed to this key but tampered with.
    #[test]
    fn open_private_request_classifies_recipients() {
        let mine = [0x11u8; 32];
        let other = [0x22u8; 32];
        let recipients = [escrow_pubkey_of(&mine).unwrap()];
        let (req, _secret) = seal_request([5u8; 32], 64, 1_000, 2_000, b"sealed prompt", &recipients).unwrap();

        assert!(super::KeryxdHandler::open_private_request(&req, None).is_none());
        assert!(super::KeryxdHandler::open_private_request(&req, Some(&other)).is_none());
        let (prompt, prompt_len, private) = super::KeryxdHandler::open_private_request(&req, Some(&mine)).unwrap();
        assert_eq!(prompt, "sealed prompt");
        assert_eq!(prompt_len, 13);
        assert!(matches!(private, Some(PrivateRequest::Opened(_))));

        // A flipped ciphertext byte still names this key: answered, with the error body.
        let mut tampered = req.serialize();
        let last = tampered.len() - 1;
        tampered[last] ^= 1;
        let tampered = AiRequestPayload::deserialize(&tampered).unwrap();
        let (prompt, _, private) = super::KeryxdHandler::open_private_request(&tampered, Some(&mine)).unwrap();
        assert!(prompt.is_empty());
        assert!(matches!(private, Some(PrivateRequest::Unreadable)));
    }

    /// The sealed answer opens with the requester's root key and nobody else's.
    #[test]
    fn sealed_answer_round_trips_to_the_requester() {
        let mine = [0x11u8; 32];
        let pubkey = escrow_pubkey_of(&mine).unwrap();
        let (req, secret) = seal_request([5u8; 32], 64, 1_000, 2_000, b"q", &[pubkey]).unwrap();
        let (_, _, private) = super::KeryxdHandler::open_private_request(&req, Some(&mine)).unwrap();
        let Some(PrivateRequest::Opened(key)) = private else { panic!("not opened") };
        let request_hash = [9u8; 32];
        let body = keryx_inference::seal_response(&key, &request_hash, b"answer");
        assert_eq!(open_response(&secret, &request_hash, &pubkey, &body).unwrap(), b"answer");
        let stranger = keryx_inference::PrivateRequestSecret::from_hex(&"00".repeat(32)).unwrap();
        assert!(open_response(&stranger, &request_hash, &pubkey, &body).is_err());
        let cid = crate::ipfs::sha256_multihash(&body);
        assert_eq!(&cid[..2], &[0x12, 0x20]);
    }

    #[test]
    fn private_prompt_inflates_when_marked() {
        use super::{decode_private_prompt, COMPRESSED_PROMPT_MAGIC, MAX_DECOMPRESSED_PROMPT_LEN};
        use std::io::Write;
        let deflate = |data: &[u8]| {
            let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::best());
            enc.write_all(data).unwrap();
            let mut out = COMPRESSED_PROMPT_MAGIC.to_vec();
            out.extend(enc.finish().unwrap());
            out
        };

        assert_eq!(decode_private_prompt("Bonjour, ça va ?".as_bytes()).as_deref(), Some("Bonjour, ça va ?"));
        let text = "Entrée 41 — le code est ORCHIDÉE-7319. ".repeat(500);
        let packed = deflate(text.as_bytes());
        assert!(packed.len() < text.len() / 10);
        assert_eq!(decode_private_prompt(&packed).as_deref(), Some(text.as_str()));

        let mut corrupt = COMPRESSED_PROMPT_MAGIC.to_vec();
        corrupt.extend([0xFF; 32]);
        assert_eq!(decode_private_prompt(&corrupt), None);
        let bomb = deflate(&vec![b'a'; MAX_DECOMPRESSED_PROMPT_LEN as usize + 1]);
        assert_eq!(decode_private_prompt(&bomb), None);
    }

    #[test]
    fn seed_block_is_the_first_chain_block_past_the_delay() {
        use super::{seed_block_after, ChainEntry};
        let entry = |hash: &str, daa: Option<u64>| ChainEntry { hash: hash.to_string(), daa };
        let delay = keryx_miner::trace::TRACE_SEED_DELAY_DAA;
        let mut log: std::collections::VecDeque<ChainEntry> =
            [entry("a", Some(1_000)), entry("b", Some(1_000 + delay - 1)), entry("c", None), entry("d", Some(1_000 + delay + 5))].into();
        assert_eq!(seed_block_after(&log, "a"), None, "an unknown DAA on the way holds the draw");
        log[2].daa = Some(1_000 + delay);
        assert_eq!(seed_block_after(&log, "a"), Some(("c", 1_000 + delay, 1_000)));
        assert_eq!(seed_block_after(&log, "b"), None, "no block past the delay yet");
        assert_eq!(seed_block_after(&log, "zz"), None);
        log[0].daa = None;
        assert_eq!(seed_block_after(&log, "a"), None, "the accepting block needs its own DAA");
    }

    #[test]
    fn restored_chain_is_placed_behind_its_start() {
        use super::{place_chain_after, seed_block_after, ChainEntry};
        let entry = |hash: &str, daa: Option<u64>| ChainEntry { hash: hash.to_string(), daa };
        let delay = keryx_miner::trace::TRACE_SEED_DELAY_DAA;
        // Accepting blocks of two restored responses, logged in no chain order.
        let mut log: std::collections::VecDeque<ChainEntry> = [entry("b", Some(1_000)), entry("z", Some(1_000 + delay + 30))].into();
        assert_eq!(seed_block_after(&log, "b"), Some(("z", 1_000 + delay + 30, 1_000)), "an unordered log draws the wrong seed block");

        let unknown = place_chain_after(&mut log, "b", vec!["c".into(), "d".into(), "z".into()]);
        assert_eq!(unknown, vec!["c".to_string(), "d".to_string()]);
        let order: Vec<&str> = log.iter().map(|e| e.hash.as_str()).collect();
        assert_eq!(order, ["b", "c", "d", "z"]);
        assert_eq!(log[3].daa, Some(1_000 + delay + 30), "a moved block keeps its DAA");
        log[1].daa = Some(1_000 + delay - 1);
        log[2].daa = Some(1_000 + delay + 1);
        assert_eq!(seed_block_after(&log, "b"), Some(("d", 1_000 + delay + 1, 1_000)));

        assert_eq!(place_chain_after(&mut log, "gone", vec!["d".into(), "x".into()]), vec!["x".to_string()]);
        let order: Vec<&str> = log.iter().map(|e| e.hash.as_str()).collect();
        assert_eq!(order, ["b", "c", "z", "d", "x"], "a chain whose start left the log goes to its end");
    }
}
