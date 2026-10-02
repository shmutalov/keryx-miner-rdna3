use bytes::BytesMut;
use log::error;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::{Display, Formatter};
use std::{fmt, io};
use tokio_util::codec::{Decoder, Encoder, LinesCodec};

// Every code a pool can send MUST deserialize: a failure here rejects the whole line, tearing
// down the connection and triggering a reconnect loop on every rejection. Known codes are from
// stratum-spec v1.1 §error-codes (26/27 are the Keryx OPoI/PoM extension); anything else —
// including negative JSON-RPC codes like -32601 — lands in Other.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(from = "i64", into = "i64")]
pub enum ErrorCode {
    Unknown,
    JobNotFound,
    DuplicateShare,
    LowDifficultyShare,
    Unauthorized,
    NotSubscribed,
    InvalidOpoiTag,
    InvalidPomProof,
    Other(i64),
}

impl From<i64> for ErrorCode {
    fn from(code: i64) -> Self {
        match code {
            20 => ErrorCode::Unknown,
            21 => ErrorCode::JobNotFound,
            22 => ErrorCode::DuplicateShare,
            23 => ErrorCode::LowDifficultyShare,
            24 => ErrorCode::Unauthorized,
            25 => ErrorCode::NotSubscribed,
            26 => ErrorCode::InvalidOpoiTag,
            27 => ErrorCode::InvalidPomProof,
            other => ErrorCode::Other(other),
        }
    }
}

impl From<ErrorCode> for i64 {
    fn from(code: ErrorCode) -> Self {
        match code {
            ErrorCode::Unknown => 20,
            ErrorCode::JobNotFound => 21,
            ErrorCode::DuplicateShare => 22,
            ErrorCode::LowDifficultyShare => 23,
            ErrorCode::Unauthorized => 24,
            ErrorCode::NotSubscribed => 25,
            ErrorCode::InvalidOpoiTag => 26,
            ErrorCode::InvalidPomProof => 27,
            ErrorCode::Other(other) => other,
        }
    }
}

impl Display for ErrorCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match &self {
            ErrorCode::Unknown => write!(f, "Unknown"),
            ErrorCode::JobNotFound => write!(f, "JobNotFound"),
            ErrorCode::DuplicateShare => write!(f, "DuplicateShare"),
            ErrorCode::LowDifficultyShare => write!(f, "LowDifficultyShare"),
            ErrorCode::Unauthorized => write!(f, "Unauthorized"),
            ErrorCode::NotSubscribed => write!(f, "NotSubscribed"),
            ErrorCode::InvalidOpoiTag => write!(f, "InvalidOpoiTag"),
            ErrorCode::InvalidPomProof => write!(f, "InvalidPomProof"),
            ErrorCode::Other(code) => write!(f, "Other({})", code),
        }
    }
}

// Pools send errors in two wire shapes: classic stratum arrays `[code, message, data?]` and
// JSON-RPC 2.0 objects `{"code":..,"message":..,"data":..}`. Both must parse — see ErrorCode
// above for why a parse failure here is fatal to the connection. Serialization stays array-form.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(from = "StratumErrorRepr")]
pub(crate) struct StratumError(pub(crate) ErrorCode, pub(crate) String, pub(crate) Option<Value>);

#[derive(Deserialize)]
struct StratumErrorTuple(ErrorCode, String, #[serde(default)] Option<Value>);

#[derive(Deserialize)]
#[serde(untagged)]
enum StratumErrorRepr {
    Tuple(StratumErrorTuple),
    Object {
        code: ErrorCode,
        message: String,
        #[serde(default)]
        data: Option<Value>,
    },
}

impl From<StratumErrorRepr> for StratumError {
    fn from(repr: StratumErrorRepr) -> Self {
        match repr {
            StratumErrorRepr::Tuple(StratumErrorTuple(code, message, data)) => StratumError(code, message, data),
            StratumErrorRepr::Object { code, message, data } => StratumError(code, message, data),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub(crate) enum MiningNotify {
    // keryx-stratum-v3: the block's compact `bits` ride along (6-element with an AiRequest task,
    // 5-element without) so a share that meets the block target is never discarded when the pool
    // difficulty is set above the network's. Listed first: untagged decoding tries in order.
    MiningNotifyWithTaskV3((String, [u64; 4], u64, u64, u32, String)),
    MiningNotifyShortV3((String, [u64; 4], u64, u64, u32)),
    // 5-element: job_id, header_hash, timestamp, daa_score, task_json (AiRequest payload)
    MiningNotifyWithTask((String, [u64; 4], u64, u64, String)),
    MiningNotifyShortV2((String, [u64; 4], u64, u64)),
    MiningNotifyShort((String, [u64; 4], u64)),
    MiningNotifyLong((String, String, String, String, Vec<String>, String, String, String, bool)),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum MiningSubmit {
    // 6-element (PoM, post-fork): address, job_id, nonce, opoi_tag, ipfs_cid (or ""), pom_proof_hex.
    // Fixed slot layout — CID stays at params[4] even when empty so the proof is always params[5].
    // Pool relays params[5] -> RpcBlock.pomProof (it does NOT verify; the node does).
    MiningSubmitWithPom((String, String, String, String, String, String)),
    // 5-element: address, job_id, nonce, opoi_tag, ipfs_cid (Phase 2 full inference submit)
    MiningSubmitWithCID((String, String, String, String, String)),
    MiningSubmitWithTag((String, String, String, String)), // address, job_id, nonce, opoi_tag
    MiningSubmitShort((String, String, String)),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum MiningSubscribe {
    MiningSubscribeDefault((String,)),
    MiningSubscribeOptions((String, String)),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum SetExtranonce {
    SetExtranoncePlain((String, u32)),
    SetExtranoncePlainEth((String,)),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "method", content = "params")]
pub(crate) enum StratumCommand {
    #[serde(rename = "mining.set_extranonce", alias = "set_extranonce")]
    SetExtranonce(SetExtranonce),
    #[serde(rename = "mining.set_difficulty")]
    MiningSetDifficulty((f32,)),
    #[serde(rename = "mining.notify")]
    MiningNotify(MiningNotify),
    #[serde(rename = "mining.subscribe")]
    Subscribe(MiningSubscribe),
    #[serde(rename = "mining.authorize")]
    Authorize((String, String)),
    #[serde(rename = "mining.submit")]
    MiningSubmit(MiningSubmit),
    // Phase 2 OPoI: miner → bridge — declare loaded SLM model IDs (sent after authorize)
    #[serde(rename = "mining.declare_capabilities")]
    MiningDeclareCapabilities(Vec<String>),
    // Pool-dispatched inference: bridge → miner
    // (task_id == request_hash, txid, request_hash, model_hex, prompt_b64, max_tokens, reward).
    #[serde(rename = "mining.ai_request")]
    MiningAiRequest((String, String, String, String, String, u32, String)),
    // miner → bridge: (worker, task_id, request_hash, model_hex, result_b64).
    #[serde(rename = "mining.ai_response")]
    MiningAiResponse((String, String, String, String, String)),
    // Phase 2 OPoI: bridge → miner — "model_id_hex:nonce_hex" capability challenge
    #[serde(rename = "mining.challenge")]
    MiningChallenge((String, String)),
    // Phase 2 OPoI: miner → bridge — [model_id_hex, nonce_hex, result_text] challenge
    // response. The nonce is echoed back so the bridge can reject replayed/stale responses.
    #[serde(rename = "mining.challenge_response")]
    MiningChallengeResponse((String, String, String)),
    /*#[serde(rename = "mining.submit_hashrate")]
    MiningSubmitHashrate {
        params: (String, String),
        worker: String,
    },*/ //{"id":9,"method":"mining.submit_hashrate","jsonrpc":"2.0","worker":"rig","params":["0x00000000000000000000000000000000","0x85198cd10b915d560722cdfdf490d4d93892d2cc3fa5f2ff2195d499d04ee54c"]}
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub(crate) enum StratumResult {
    Plain(Option<bool>),
    Eth((bool, String)),
    Subscribe((Vec<(String, String)>, String, u32)),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub(crate) enum StratumLinePayload {
    StratumCommand(StratumCommand),
    StratumResult { result: StratumResult },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct StratumLine {
    pub(crate) id: Option<u32>,
    #[serde(flatten)]
    pub(crate) payload: StratumLinePayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) jsonrpc: Option<String>,
    pub(crate) error: Option<StratumError>,
}

/// An error occurred while encoding or decoding a line.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum NewLineJsonCodecError {
    JsonParseError(String),
    JsonEncodeError,
    LineSplitError,
    LineEncodeError,
    Io(io::Error),
}

impl fmt::Display for NewLineJsonCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Some error occured")
    }
}
impl From<io::Error> for NewLineJsonCodecError {
    fn from(e: io::Error) -> NewLineJsonCodecError {
        NewLineJsonCodecError::Io(e)
    }
}
impl std::error::Error for NewLineJsonCodecError {}

impl From<(String, String)> for NewLineJsonCodecError {
    fn from(e: (String, String)) -> Self {
        NewLineJsonCodecError::JsonParseError(format!("{}: {}", e.0, e.1))
    }
}

pub(crate) struct NewLineJsonCodec {
    lines_codec: LinesCodec,
}

impl NewLineJsonCodec {
    pub fn new() -> Self {
        Self { lines_codec: LinesCodec::new() }
    }
}

impl Decoder for NewLineJsonCodec {
    type Item = StratumLine;
    type Error = NewLineJsonCodecError;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        match self.lines_codec.decode(src) {
            Ok(Some(s)) => {
                serde_json::from_str::<StratumLine>(s.as_str()).map_err(|e| (e.to_string(), s).into()).map(Some)
            }
            Err(_) => Err(NewLineJsonCodecError::LineSplitError),
            _ => Ok(None),
        }
    }

    fn decode_eof(&mut self, buf: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        match self.lines_codec.decode_eof(buf) {
            Ok(Some(s)) => serde_json::from_str(s.as_str()).map_err(|e| (e.to_string(), s).into()),
            Err(_) => Err(NewLineJsonCodecError::LineSplitError),
            _ => Ok(None),
        }
    }
}

impl Encoder<StratumLine> for NewLineJsonCodec {
    type Error = NewLineJsonCodecError;

    fn encode(&mut self, item: StratumLine, dst: &mut BytesMut) -> Result<(), Self::Error> {
        match serde_json::to_string(&item) {
            Ok(json) => self.lines_codec.encode(json, dst).map_err(|_| NewLineJsonCodecError::LineEncodeError),
            Err(e) => {
                error!("Error! {:?}", e);
                Err(NewLineJsonCodecError::JsonEncodeError)
            }
        }
    }
}

impl Default for NewLineJsonCodec {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_error(line: &str) -> StratumError {
        let parsed: StratumLine = serde_json::from_str(line).unwrap_or_else(|e| panic!("failed to parse {line}: {e}"));
        parsed.error.expect("line should carry an error")
    }

    #[test]
    fn error_as_array() {
        let err = parse_error(r#"{"id":1,"result":null,"error":[23,"Low difficulty share",null]}"#);
        assert_eq!(err.0, ErrorCode::LowDifficultyShare);
        assert_eq!(err.1, "Low difficulty share");
    }

    #[test]
    fn error_as_two_element_array() {
        let err = parse_error(r#"{"id":1,"result":null,"error":[21,"Job not found"]}"#);
        assert_eq!(err.0, ErrorCode::JobNotFound);
        assert!(err.2.is_none());
    }

    #[test]
    fn error_as_object() {
        let err = parse_error(r#"{"id":422,"result":null,"error":{"code":25,"message":"Not subscribed"}}"#);
        assert_eq!(err.0, ErrorCode::NotSubscribed);
        assert_eq!(err.1, "Not subscribed");
    }

    #[test]
    fn unknown_and_negative_codes_do_not_fail() {
        let err = parse_error(r#"{"id":1,"result":null,"error":[30,"Banned"]}"#);
        assert_eq!(err.0, ErrorCode::Other(30));
        let err = parse_error(r#"{"id":1,"result":null,"error":{"code":-32601,"message":"Method not found"}}"#);
        assert_eq!(err.0, ErrorCode::Other(-32601));
    }

    #[test]
    fn error_serializes_as_array() {
        let json = serde_json::to_string(&StratumError(ErrorCode::NotSubscribed, "Not subscribed".into(), None)).unwrap();
        assert_eq!(json, r#"[25,"Not subscribed",null]"#);
    }
}
