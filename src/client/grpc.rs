use crate::client::Client;
use crate::pow::BlockSeed;
use crate::pow::BlockSeed::{FullBlock, PartialBlock};
use crate::proto::kaspad_message::Payload;
use crate::proto::rpc_client::RpcClient;
use crate::proto::{
    GetBlockRequestMessage, GetBlockTemplateRequestMessage, GetInfoRequestMessage, KaspadMessage,
    NotifyBlockAddedRequestMessage, NotifyNewBlockTemplateRequestMessage,
    NotifyVirtualSelectedParentChainChangedRequestMessage,
};
use crate::{miner::MinerManager, Error};

/// Max AiRequest queue size — drop oldest when full to prevent unbounded memory growth.
const MAX_AI_QUEUE_SIZE: usize = 64;
/// Max boot-time escrow-validation GetBlock requests in flight at once — each answer
/// sends the next queued one, so thousands of state entries never overwhelm the
/// HTTP/2 flow-control window or delay the mining stream.
const VALIDATION_WINDOW: usize = 64;
/// Max unique stable-ids tracked for deduplication — evict when full.
const MAX_AI_SEEN_IDS: usize = 10_000;

use async_trait::async_trait;
use futures_util::StreamExt;
use log::{error, info, warn};
use rand::{thread_rng, RngCore};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc::{self, error::SendError, Sender}, oneshot};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::{PollSendError, PollSender};
use tonic::{transport::Channel as TonicChannel, Streaming};

static EXTRA_DATA: &str = concat!(env!("CARGO_PKG_VERSION"), "/", env!("PACKAGE_COMPILE_TIME"));
type BlockHandle = JoinHandle<Result<(), PollSendError<KaspadMessage>>>;

/// Private-inference state of a queued request; `None` for a plaintext request.
#[derive(Clone, Copy)]
enum PrivateRequest {
    /// Opened with the escrow key; the answer is sealed under this root key.
    Opened([u8; 32]),
    /// Names this miner but cannot be opened; answered with a plaintext error body.
    Unreadable,
}

/// Body of the answer to a private request this miner could not open (upstream's text).
const UNREADABLE_PRIVATE_REQUEST_ANSWER: &str = "keryx: private request could not be opened by this responder";

/// A queued AiRequest: (stable_id_hex16, request_hash, model_id, prompt, max_tokens, private state).
type QueuedAiRequest = (String, [u8; 32], [u8; 32], String, usize, Option<PrivateRequest>);

#[allow(dead_code)]
pub struct KeryxdHandler {
    client: RpcClient<TonicChannel>,
    pub send_channel: Sender<KaspadMessage>,
    stream: Streaming<KaspadMessage>,
    miner_address: String,
    mine_when_not_synced: bool,
    block_channel: Sender<BlockSeed>,
    block_handle: BlockHandle,

    /// Queue of AiRequests waiting for inference.
    /// Each entry: (stable_id_hex16, request_hash, model_id, prompt, max_tokens, private state).
    /// `request_hash` is the identity the AiResponse names: the AiRequest transaction id from H8,
    /// the payload digest before it. A private request's prompt is the opened plaintext. Fed by
    /// both BlockAdded scans and block template scans.
    ai_request_queue: VecDeque<QueuedAiRequest>,

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

    /// In-flight SLM inference task: (request_hash, private state, result_receiver).
    /// None result means inference failed (model not ready or empty output) — no AiResponse.
    inference_rx: Option<([u8; 32], Option<PrivateRequest>, oneshot::Receiver<Option<String>>)>,

    /// In-flight inference for a node-issued challenge.
    /// Tuple: (challenge_string, result_receiver) where challenge_string = "model_id_hex:nonce_hex".
    /// When the result arrives, it is sent back via inference_result in the next GetBlockTemplateRequest.
    challenge_inference_rx: Option<(String, oneshot::Receiver<Option<String>>)>,

    /// Shared flag with MinerManager — suppresses GPU stall warnings during OPoI inference.
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
}

#[async_trait(?Send)]
impl Client for KeryxdHandler {
    fn flush_escrow_state(&mut self) -> Result<(), Error> {
        self.escrow_watcher.as_mut().map_or(Ok(()), |watcher| watcher.flush_state().map_err(Into::into))
    }

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
            match maybe_msg {
                Some(Some(m)) => match m.payload {
                    Some(payload) => self.handle_message(payload, miner).await?,
                    None => warn!("keryxd message payload is empty"),
                },
                Some(None) => break, // stream closed by node
                None => {
                    // Timer tick: if a regular inference just finished, get a fresh template.
                    if self.inference_rx.is_some() && self.poll_inference().await {
                        self.client_get_block_template().await?;
                    // If a challenge is in flight, keep pinging the node so the result is
                    // delivered as soon as the inference task completes. This is critical on
                    // sole-producer nodes where mining suspension stops NewBlockTemplate
                    // notifications and the response would otherwise never be sent.
                    } else if self.challenge_inference_rx.is_some() {
                        self.client_get_block_template().await?;
                    }
                }
            }
        }
        Ok(())
    }

    fn get_block_channel(&self) -> Sender<BlockSeed> {
        self.block_channel.clone()
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
        // Build EscrowWatcher from the resolved escrow privkey (derived or loaded from file).
        // The watcher also provides the pubkey to embed in coinbase extra_data.
        let (escrow_pubkey, escrow_watcher) = match escrow_privkey {
            Some(ref privkey) => {
                match crate::escrow::EscrowWatcher::new(privkey, &miner_address, escrow_state_file.into()) {
                    Ok(watcher) => {
                        let pk = watcher.pubkey_hex();
                        info!("OPoI escrow active: pubkey={}", pk);
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

        let mut client = RpcClient::connect(address).await?;
        // Outbound message channel to the node. ALL client->node messages share this:
        // mining (submit_block, GetBlockTemplate) AND OPoI traffic (per-block GetBlock,
        // escrow submit_transaction). With a capacity of 2 the OPoI traffic could fill the
        // buffer and block GetBlockTemplate, stalling template delivery → the GPU sits idle
        // between blocks. A large buffer keeps the mining requests from queuing behind OPoI.
        let (send_channel, recv) = mpsc::channel(1024);
        send_channel.send(GetInfoRequestMessage {}.into()).await?;
        let stream = client.message_stream(ReceiverStream::new(recv)).await?.into_inner();
        let (block_channel, block_handle) = Self::create_block_channel(send_channel.clone());
        Ok(Box::new(Self {
            client,
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
            inference_rx: None,
            challenge_inference_rx: None,
            opoi_challenge_active: None,
            last_known_daa: 0,
            ipfs_url,
            escrow_pubkey,
            escrow_watcher,
            escrow_cert,
        }))
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
        // OPoI Phase 2: run the deterministic fixed-point MLP (matches node validation).
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
                    info!("OPoI: sending challenge response model={:.8}", model_id_hex);
                    if let Some(flag) = &self.opoi_challenge_active {
                        flag.store(false, Ordering::Relaxed);
                    }
                    // Response format: "model_id_hex:nonce_hex:result_text"
                    format!("{}:{}:{}", model_id_hex, nonce_hex_c, text)
                }
                Ok(None) => {
                    warn!("OPoI: challenge inference failed — sending empty result, node will re-challenge");
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
                    warn!("OPoI: challenge inference task dropped — sending empty result");
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
        // Requests this escrow key already answered on-chain are never served again.
        if let Some(mine) = self.escrow_watcher.as_ref().map(|w| w.pubkey_bytes()) {
            for sid in Self::own_answered_request_ids(txs, &mine) {
                if self.ai_answered_by_me.insert(sid.clone()) {
                    log::debug!("OPoI: request id={} already answered by this miner on-chain", sid);
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
            log::warn!("OPoI: no models ready — skipping AiRequest scan (run miner with valid model files)");
            return;
        }
        log::debug!(
            "scan_ai: {} txs, subnetwork_ids: {:?}",
            txs.len(),
            txs.iter().map(|t| t.subnetwork_id.as_str()).collect::<Vec<_>>()
        );
        let escrow_secret = self.escrow_watcher.as_ref().map(|w| w.secret_bytes());
        for tx in txs {
            // (raw, model_id, prompt, max_tokens, inference_reward, private state)
            let extracted: Option<(Vec<u8>, [u8; 32], String, usize, u64, Option<PrivateRequest>)> =
                if tx.subnetwork_id == keryx_inference::SUBNETWORK_ID_AI_REQUEST_HEX {
                    // Binary AiRequestPayload (dedicated AI subnetwork). A private one (every request
                    // from H14) is sealed to its tier cohort: opened here with the escrow key.
                    hex::decode(&tx.payload).ok().and_then(|raw| {
                        let req = keryx_inference::AiRequestPayload::deserialize(&raw)?;
                        let (prompt, private) = if req.is_private() {
                            Self::open_private_request(&req, escrow_secret.as_ref())?
                        } else {
                            (String::from_utf8_lossy(&req.prompt).into_owned(), None)
                        };
                        Some((raw, req.model_id, prompt, req.max_tokens as usize, req.inference_reward, private))
                    })
                } else if !tx.inputs.is_empty() {
                    // KRX:AI:1: JSON format — model routed by "m" field, skipped if not loaded.
                    hex::decode(&tx.payload).ok().and_then(|raw| {
                        Self::parse_krx_ai_payload(&raw).and_then(|(model_name, prompt, max_tokens)| {
                            let model_id = keryx_miner::models::find(&model_name)?.model_id;
                            Some((raw, model_id, prompt, max_tokens, 0u64, None))
                        })
                    })
                } else {
                    None // coinbase — skip
                };

            if let Some((raw, model_id, prompt, max_tokens, inference_reward, private)) = extracted {
                if !ready_ids.contains(&model_id) {
                    log::debug!("OPoI: skipping AiRequest — model not supported or files not ready");
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
                            log::warn!("OPoI: cannot resolve the AiRequest transaction id — request skipped");
                            continue;
                        }
                    }
                } else {
                    blake2b_simd::blake2b(&raw).as_bytes()[..32].try_into().unwrap()
                };
                let stable_id = hex::encode(&request_hash[..8]);
                if self.ai_answered_by_me.contains(&stable_id) {
                    log::debug!("OPoI: skipping AiRequest id={} — already answered by this miner", stable_id);
                    continue;
                }
                if !self.ai_seen_prefixes.contains(&stable_id) {
                    info!("OPoI: queued AiRequest id={}{}", stable_id, if private.is_some() { " (private)" } else { "" });
                    self.ai_seen_prefixes.insert(stable_id.clone());
                    self.ai_request_queue.push_back((stable_id.clone(), request_hash, model_id, prompt, max_tokens, private));
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
    ) -> Option<(String, Option<PrivateRequest>)> {
        let Some(secret) = escrow_secret else {
            log::debug!("OPoI: skipping private AiRequest — no escrow key configured");
            return None;
        };
        match keryx_inference::open_request(req, secret) {
            Ok(opened) => match decode_private_prompt(&opened.prompt) {
                Some(prompt) => Some((prompt, Some(PrivateRequest::Opened(opened.root_key)))),
                None => {
                    warn!("OPoI: private AiRequest opened but its compressed prompt is invalid — answering with an error body");
                    Some((String::new(), Some(PrivateRequest::Unreadable)))
                }
            },
            Err(keryx_inference::PrivateError::NotARecipient) => {
                log::debug!("OPoI: skipping private AiRequest — this escrow key is not a recipient");
                None
            }
            Err(e) => {
                warn!("OPoI: private AiRequest names this miner but cannot be opened ({}) — answering with an error body", e);
                Some((String::new(), Some(PrivateRequest::Unreadable)))
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

    /// Starts SLM inference for the next queued AiRequest, if no inference is
    /// already in flight and a response slot is free.
    fn try_start_inference(&mut self) {
        if self.inference_rx.is_some() {
            return;
        }
        while self.ai_request_queue.front().is_some_and(|(sid, ..)| self.ai_answered_by_me.contains(sid)) {
            if let Some((sid, ..)) = self.ai_request_queue.pop_front() {
                info!("OPoI: request id={} already answered by this miner — skipped", sid);
            }
        }
        if let Some((stable_id, request_hash, model_id, prompt, max_tokens, private)) = self.ai_request_queue.pop_front() {
            let (tx_done, rx_done) = oneshot::channel::<Option<String>>();
            if matches!(private, Some(PrivateRequest::Unreadable)) {
                // No inference to run: the answer is the fixed error text.
                let _ = tx_done.send(Some(UNREADABLE_PRIVATE_REQUEST_ANSWER.to_string()));
                self.inference_rx = Some((request_hash, private, rx_done));
                return;
            }
            // Second guard: re-check readiness at execution time (files could have been deleted).
            if !keryx_miner::slm::is_model_ready(&model_id) {
                log::error!("OPoI: model became unavailable after queuing id={} — discarding request", stable_id);
                return;
            }
            info!("OPoI: spawning SLM inference (max_tokens={})", max_tokens);
            tokio::task::spawn_blocking(move || {
                let result = keryx_miner::slm::load_and_run_inference(&model_id, &prompt, max_tokens);
                if result.is_none() {
                    log::warn!("OPoI: inference returned no result for id={} — AiResponse will be skipped", stable_id);
                }
                let _ = tx_done.send(result);
            });
            self.inference_rx = Some((request_hash, private, rx_done));
        }
    }

    /// Polls the in-flight inference task. When complete, submits a zero-input/zero-output
    /// AiResponse: a private answer is sealed to the requester and, from the private-inference
    /// gate on, travels inline; anything else is uploaded to IPFS first and named by its CID.
    /// Returns `true` if inference just finished (regardless of tx success).
    async fn poll_inference(&mut self) -> bool {
        let Some((request_hash, private, mut rx)) = self.inference_rx.take() else {
            return false;
        };
        let Ok(result_opt) = rx.try_recv() else {
            self.inference_rx = Some((request_hash, private, rx));
            return false;
        };
        let Some(result) = result_opt else {
            // Inference returned None: model not ready or think block exhausted max_tokens.
            // Do NOT upload anything to IPFS — skip this AiResponse entirely.
            info!("OPoI: inference produced no result — AiResponse skipped");
            return true;
        };

        info!("OPoI: inference complete, request_hash={}", hex::encode(&request_hash[..8]));

        // A private answer is sealed to the requester; the plaintext is never published. From
        // the gate on it travels inline and the CID field carries its multihash; before the
        // gate the sealed body is pinned on IPFS like a plaintext answer.
        let publish: Vec<u8> = match private {
            None => result.clone().into_bytes(),
            Some(p) => {
                let body = match p {
                    PrivateRequest::Opened(root_key) => {
                        let Some(pubkey) = self.escrow_watcher.as_ref().map(|w| w.pubkey_bytes()) else {
                            warn!("OPoI: escrow key gone — private AiResponse skipped");
                            return true;
                        };
                        keryx_inference::seal_response(&root_key, &request_hash, &pubkey, result.as_bytes())
                    }
                    PrivateRequest::Unreadable => result.clone().into_bytes(),
                };
                if self.last_known_daa >= keryx_miner::pom::private_inference_activation_daa() {
                    let cid = crate::ipfs::sha256_multihash(&body);
                    self.submit_ai_response(request_hash, &result, cid, Some(body)).await;
                    return true;
                }
                body
            }
        };

        let ipfs_url = self.ipfs_url.clone();
        let cid = match tokio::task::spawn_blocking(move || crate::ipfs::upload_bytes(&publish, &ipfs_url)).await {
            Ok(Ok(cid)) => cid,
            Ok(Err(e)) => { warn!("OPoI: IPFS upload failed: {} — AiResponse tx skipped", e); return true; }
            Err(e) => { warn!("OPoI: IPFS spawn_blocking failed: {} — AiResponse tx skipped", e); return true; }
        };
        self.submit_ai_response(request_hash, &result, cid, None).await;
        true
    }

    /// Builds and submits the zero-input/zero-output AiResponse for `cid`, then registers the
    /// request's inference escrow for auto-claim. A `private_body` rides inline under the
    /// responder signature (V2 only).
    async fn submit_ai_response(&mut self, request_hash: [u8; 32], result: &str, cid: [u8; 34], private_body: Option<Vec<u8>>) {
        let challenge_window_end = self.last_known_daa + 1000;
        let response_length = result.split_whitespace().count() as u32;
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
                let responder = w.sign_responder(&unsigned.signed_bytes());
                let resp = keryx_inference::AiResponsePayload::new_v2(request_hash, challenge_window_end, cid, response_length, responder);
                match private_body {
                    Some(body) => resp.with_private_body(body),
                    None => resp,
                }
            }
            (None, true) => {
                warn!("OPoI: no escrow key configured — submitting an unsigned (v1) response; it will NOT count for the service bond");
                keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length)
            }
            (_, false) => keryx_inference::AiResponsePayload::new(request_hash, challenge_window_end, cid, response_length),
        };
        info!(
            "OPoI: uploading response CID={}, challenge_window_end={}{}{}",
            resp.cid_v0(),
            challenge_window_end,
            if resp.responder.is_some() { " (signed, V2)" } else { "" },
            if resp.private_body.is_some() { " (private body inline)" } else { "" }
        );

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
        if let Err(e) = self.client_send(KaspadMessage::submit_transaction(rpc_tx)).await {
            warn!("OPoI: failed to send AiResponse tx: {}", e);
        }

        // Register inference escrow outpoint for auto-claim after the challenge window.
        let stable_id = hex::encode(&request_hash[..8]);
        if let Some((txid, inference_reward)) = self.ai_request_txids.remove(&stable_id) {
            if let Some(w) = self.escrow_watcher.as_mut() {
                w.track_inference_escrow(txid, self.last_known_daa, inference_reward);
            }
        }
    }

    async fn handle_message(&mut self, msg: Payload, miner: &mut MinerManager) -> Result<(), Error> {
        match msg {
            // BlockAdded: scan confirmed block for AiRequests and escrow UTXOs.
            // Do NOT trigger a new block template here — NewBlockTemplate handles that.
            Payload::BlockAddedNotification(notif) => {
                if let Some(block) = notif.block {
                    if !block.transactions.is_empty() {
                        // Full block — scan directly.
                        self.scan_txs_for_ai_requests(&block.transactions, block.header.as_ref().map_or(0, |h| h.daa_score));
                        self.try_start_inference();
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
            Payload::GetBlockTemplateResponse(template) => {
                // Track DAA score for challenge_window_end computation.
                if let Some(daa) = template.block.as_ref()
                    .and_then(|b| b.header.as_ref())
                    .map(|h| h.daa_score)
                {
                    if daa > self.last_known_daa {
                        self.last_known_daa = daa;
                        keryx_miner::slm::note_chain_daa(daa);
                    }
                    // H4-only binary: the served lineup is static (set once at startup) —
                    // no era crossing left to hot-swap.
                }
                // Handle node-issued inference challenge: spawn an inference task if a new
                // challenge arrived and no challenge is already in flight. Ignored under PoM — the
                // per-block possession proof is the capability gate, so no synthetic challenge
                // (defensive: holds even against a node that still issues them post-hardfork).
                if !template.inference_challenge.is_empty()
                    && self.challenge_inference_rx.is_none()
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
                                info!("OPoI: challenge received model={:.8} nonce={:.8} — spawning inference", model_id_hex, nonce_hex);
                                if let Some(flag) = &self.opoi_challenge_active {
                                    flag.store(true, Ordering::Relaxed);
                                }
                                let prompt = format!("Keryx inference challenge {}: briefly describe what you are.", nonce_hex);
                                let (tx_done, rx_done) = oneshot::channel::<Option<String>>();
                                tokio::task::spawn_blocking(move || {
                                    let result = keryx_miner::slm::load_and_run_inference(&model_id, &prompt, 64);
                                    let _ = tx_done.send(result);
                                });
                                self.challenge_inference_rx = Some((challenge, rx_done));
                            } else {
                                warn!("OPoI: challenge for unready model={:.8} — cannot respond", model_id_hex);
                            }
                        }
                    }
                }
                // Poll in-flight inference; if done, submit AiResponse tx then get fresh template.
                if self.poll_inference().await {
                    self.client_get_block_template().await?;
                    return Ok(());
                }
                // OPoI is mandatory: refuse to mine if no models are ready.
                // Covers miners with missing/truncated model files that somehow passed prefetch.
                if keryx_miner::slm::loaded_model_ids().is_empty() {
                    // Throttle to one log per ~200 templates (~every 20s at 10 BPS) to avoid spam.
                    if self.last_known_daa % 200 == 0 {
                        log::warn!("OPoI: no models ready — mining suspended until model files are available");
                    }
                    miner.process_block(None).await?;
                    return Ok(());
                }
                if let Some(ref block) = template.block {
                    self.scan_txs_for_ai_requests(&block.transactions, block.header.as_ref().map_or(0, |h| h.daa_score));
                }
                self.try_start_inference();
                // Pause GPU mining while any inference is in flight (GPU is occupied by the model).
                // This covers both regular AiRequest inference and node-issued challenge inference.
                if self.inference_rx.is_some() || self.challenge_inference_rx.is_some() {
                    miner.process_block(None).await?;
                    return Ok(());
                }
                match (template.block, template.is_synced, template.error) {
                    (Some(b), true, None) => miner.process_block(Some(FullBlock(Box::new(b)))).await?,
                    (Some(b), false, None) if self.mine_when_not_synced => {
                        miner.process_block(Some(FullBlock(Box::new(b)))).await?
                    }
                    (_, false, None) => miner.process_block(None).await?,
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
                    // Validation answer: "cannot find header <hash>" — the block never
                    // existed on this chain, its escrow entries are ghosts.
                    was_validation = self
                        .escrow_watcher
                        .as_mut()
                        .map_or(false, |w| w.on_block_validation_error(&e.message));
                    if !was_validation {
                        warn!("GetBlockResponse error: {}", e.message);
                    }
                } else if let Some(block) = msg.block {
                    let hash = block.verbose_data.as_ref().map(|v| v.hash.clone()).unwrap_or_default();
                    // Chain membership from the node's live verdict: a stored-but-reorged
                    // block must purge its entries just like a missing one.
                    let is_chain = block.verbose_data.as_ref().map_or(false, |v| v.is_chain_block);
                    was_validation = self
                        .escrow_watcher
                        .as_mut()
                        .map_or(false, |w| w.consume_validation_ok(&hash, is_chain));
                    if !was_validation {
                        self.scan_txs_for_ai_requests(&block.transactions, block.header.as_ref().map_or(0, |h| h.daa_score));
                        self.try_start_inference();
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
                None => info!("block submitted successfully!"),
                Some(e) => warn!("Failed submitting block: {:?}", e),
            },
            Payload::SubmitTransactionResponse(res) => {
                // Escrow claims and OPoI submissions share this stream. Match responses to
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
                        if let Some(e) = err {
                            log::debug!("OPoI: submit_transaction error: {}", e);
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
                self.client_send(NotifyVirtualSelectedParentChainChangedRequestMessage {}).await?;
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
                None => info!("Registered for virtual chain notifications (escrow tracking)"),
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
                for hash in notif.added_chain_block_hashes {
                    self.client_send(GetBlockRequestMessage { hash, include_transactions: true }).await?;
                }
            }
            msg => info!("got unknown msg: {:?}", msg),
        }
        Ok(())
    }
}

impl Drop for KeryxdHandler {
    fn drop(&mut self) {
        self.block_handle.abort();
    }
}

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
        assert_eq!(ids, vec![hex::encode([1u8; 8]), hex::encode([3u8; 8])]);
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
        let (prompt, private) = super::KeryxdHandler::open_private_request(&req, Some(&mine)).unwrap();
        assert_eq!(prompt, "sealed prompt");
        assert!(matches!(private, Some(PrivateRequest::Opened(_))));

        // A flipped ciphertext byte still names this key: answered, with the error body.
        let mut tampered = req.serialize();
        let last = tampered.len() - 1;
        tampered[last] ^= 1;
        let tampered = AiRequestPayload::deserialize(&tampered).unwrap();
        let (prompt, private) = super::KeryxdHandler::open_private_request(&tampered, Some(&mine)).unwrap();
        assert!(prompt.is_empty());
        assert!(matches!(private, Some(PrivateRequest::Unreadable)));
    }

    /// The sealed answer opens with the requester's root key and nobody else's, and its inline CID
    /// is the body's sha2-256 multihash.
    #[test]
    fn sealed_answer_round_trips_to_the_requester() {
        let mine = [0x11u8; 32];
        let pubkey = escrow_pubkey_of(&mine).unwrap();
        let (req, secret) = seal_request([5u8; 32], 64, 1_000, 2_000, b"q", &[pubkey]).unwrap();
        let (_, private) = super::KeryxdHandler::open_private_request(&req, Some(&mine)).unwrap();
        let Some(PrivateRequest::Opened(root_key)) = private else { panic!("not opened") };
        let request_hash = [9u8; 32];
        let body = keryx_inference::seal_response(&root_key, &request_hash, &pubkey, b"answer");
        assert_eq!(open_response(&secret.root_key, &request_hash, &pubkey, &body).unwrap(), b"answer");
        assert!(open_response(&[0u8; 32], &request_hash, &pubkey, &body).is_err());
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
}
