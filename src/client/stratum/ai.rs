use super::statum_codec::{StratumCommand, StratumLine, StratumLinePayload};
use crate::Error;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

const MAX_PAYLOAD_BYTES: usize = 1_048_576;

/// The H15 trace a pool asks for (keryx-miner-supr 0.15.3's optional 8th `mining.ai_request`
/// param `{"trace":{"tier","prompt_len","chain","attention"}}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TraceAsk {
    pub tier: u8,
    /// Length of the request's plaintext prompt (the shape rule pads the traced prompt to it); the
    /// length of the prompt bytes sent when the pool leaves it out.
    pub prompt_len: u32,
    /// Stage-2 (chain) trace.
    pub chain: bool,
    /// Sequence leaves and the attention rule's manifest.
    pub attention: bool,
}

impl TraceAsk {
    /// `sent_len`: byte length of the prompt as the pool sent it, the `prompt_len` when the pool
    /// leaves it out (upstream keryx-miner v0.6.0 makes it optional). Capped like a private prompt.
    fn parse(ext: &serde_json::Value, sent_len: usize) -> Result<Option<Self>, Error> {
        let Some(t) = ext.get("trace").filter(|t| !t.is_null()) else { return Ok(None) };
        let tier = t.get("tier").and_then(|v| v.as_u64()).filter(|v| *v < 16).ok_or("trace.tier missing")? as u8;
        let cap = keryx_inference::MAX_AI_REQUEST_PRIVATE_PAYLOAD_LEN as u32;
        let prompt_len = match t.get("prompt_len").filter(|v| !v.is_null()) {
            None => (sent_len as u32).min(cap),
            Some(v) => v.as_u64().and_then(|v| u32::try_from(v).ok()).ok_or("trace.prompt_len is not a u32")?.min(cap),
        };
        let chain = t.get("chain").and_then(|v| v.as_bool()).unwrap_or(true);
        let attention = t.get("attention").and_then(|v| v.as_bool()).unwrap_or(true);
        // `trace.upload` (pool custody of the compact record) is optional and not used: the trace
        // stays on this rig, which builds the openings itself.
        Ok(Some(Self { tier, prompt_len, chain, attention }))
    }
}

pub(super) struct AiRequest {
    pub task_id: String,
    pub request_hash: String,
    pub model_hex: String,
    pub model_id: [u8; 32],
    pub prompt: String,
    pub max_tokens: usize,
    /// H15: answer traced (the pool seals `body` and attaches the commitment before signing).
    pub trace: Option<TraceAsk>,
}

impl AiRequest {
    #[cfg(test)]
    pub fn parse(fields: (String, String, String, String, String, u32, String)) -> Result<Self, Error> {
        Self::parse_ext(fields, None)
    }

    /// `[task_id, txid, request_hash, model_id_hex, prompt_base64, max_tokens, reward]` and the
    /// optional trace object; task_id must equal request_hash (case-insensitive), ids are 64 hex.
    pub fn parse_ext(fields: (String, String, String, String, String, u32, String), ext: Option<&serde_json::Value>) -> Result<Self, Error> {
        let (task_id, txid, request_hash, model_hex, prompt_b64, max_tokens, reward) = fields;
        if !task_id.eq_ignore_ascii_case(&request_hash) || max_tokens == 0 || max_tokens > i32::MAX as u32 {
            return Err("Invalid AI task metadata".into());
        }
        for value in [&task_id, &txid, &request_hash, &model_hex] {
            if value.len() != 64 || !value.bytes().all(|c| c.is_ascii_hexdigit()) {
                return Err("Invalid AI task identifier".into());
            }
        }
        reward.parse::<u64>()?;
        if prompt_b64.len() > MAX_PAYLOAD_BYTES.div_ceil(3) * 4 {
            return Err("AI prompt is too large".into());
        }
        let raw = BASE64.decode(prompt_b64)?;
        if raw.len() > MAX_PAYLOAD_BYTES {
            return Err("Invalid AI prompt".into());
        }
        // The node takes any bytes as a prompt, so a request is answered whatever they are
        // (upstream 77ff767, supr 0.16.1): a compressed one (`00 'K' 'Z' 01` + DEFLATE) is
        // inflated, invalid UTF-8 is replaced, and NUL bytes are kept: the engine tokenizes the
        // prompt's bytes with their length and formats it without a C string.
        let prompt = crate::client::grpc::decode_private_prompt(&raw);
        let trace = match ext {
            Some(e) => TraceAsk::parse(e, raw.len())?,
            None => None,
        };
        let mut model_id = [0; 32];
        hex::decode_to_slice(&model_hex, &mut model_id)?;
        Ok(Self { task_id, request_hash, model_hex, model_id, prompt, max_tokens: max_tokens as usize, trace })
    }

    /// The request hash as bytes (hex validated by [`parse_ext`](Self::parse_ext)).
    pub fn request_hash_bytes(&self) -> [u8; 32] {
        let mut h = [0u8; 32];
        let _ = hex::decode_to_slice(&self.request_hash, &mut h);
        h
    }

    /// `mining.ai_response` `[worker, task_id, request_hash, model_id_hex, answer_base64]`, plus for
    /// a traced answer `{"trace": hex commitment, "body": hex answer body}`.
    pub fn response(&self, id: u32, worker: String, result: &str, traced: Option<(&[u8], &[u8])>) -> Result<StratumLine, Error> {
        if result.is_empty() || result.len() > MAX_PAYLOAD_BYTES {
            return Err("Invalid AI result size".into());
        }
        let mut params: Vec<serde_json::Value> = vec![
            worker.into(),
            self.task_id.clone().into(),
            self.request_hash.clone().into(),
            self.model_hex.clone().into(),
            BASE64.encode(result.as_bytes()).into(),
        ];
        if let Some((commitment, body)) = traced {
            params.push(serde_json::json!({ "trace": hex::encode(commitment), "body": hex::encode(body) }));
        }
        Ok(StratumLine {
            id: Some(id),
            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningAiResponse(params)),
            jsonrpc: Some("2.0".into()),
            error: None,
        })
    }
}

/// A pool's `mining.ai_opening`: build the opening (or one attention phase) of a traced answer.
#[derive(Debug)]
pub(super) struct AiOpening {
    pub request_hash: String,
    pub response_txid: [u8; 32],
    pub response_txid_hex: String,
    pub seed: [u8; 32],
    pub path: Vec<u64>,
    pub prev: Vec<u8>,
}

impl AiOpening {
    /// `[request_hash, response_txid, seed, attention_path, prev_partials_hex(, record_url)]`.
    pub fn parse(v: &[serde_json::Value]) -> Result<Self, Error> {
        if v.len() < 3 {
            return Err("mining.ai_opening takes [request_hash, response_txid, seed, path, prev]".into());
        }
        let hex32 = |i: usize| -> Result<([u8; 32], String), Error> {
            let s = v[i].as_str().ok_or("mining.ai_opening ids must be strings")?;
            let mut b = [0u8; 32];
            hex::decode_to_slice(s, &mut b)?;
            Ok((b, s.to_ascii_lowercase()))
        };
        let (_, request_hash) = hex32(0)?;
        let (response_txid, response_txid_hex) = hex32(1)?;
        let (seed, _) = hex32(2)?;
        let path = match v.get(3) {
            Some(serde_json::Value::Array(a)) => a.iter().map(|x| x.as_u64().ok_or("attention path must be u64s")).collect::<Result<Vec<_>, _>>()?,
            None | Some(serde_json::Value::Null) => Vec::new(),
            _ => return Err("attention path must be an array".into()),
        };
        // The previous phase's partials only save a replay: upstream keryx-miner v0.6.0 ignores
        // them, so a bridge may leave them out, and the rig rebuilds them from its kept trace.
        let prev = match v.get(4).and_then(|x| x.as_str()) {
            Some(s) if !s.is_empty() && !path.is_empty() => hex::decode(s)?,
            _ => Vec::new(),
        };
        Ok(Self { request_hash, response_txid, response_txid_hex, seed, path, prev })
    }

    /// `mining.ai_opening_result` `[worker, request_hash, response_txid, path, body_hex, error]`.
    pub fn result(&self, id: u32, worker: String, body: Result<&[u8], &str>) -> StratumLine {
        let (body_hex, err) = match body {
            Ok(b) => (hex::encode(b), String::new()),
            Err(e) => (String::new(), e.to_string()),
        };
        StratumLine {
            id: Some(id),
            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningAiOpeningResult(vec![
                worker.into(),
                self.request_hash.clone().into(),
                self.response_txid_hex.clone().into(),
                serde_json::json!(self.path),
                body_hex.into(),
                err.into(),
            ])),
            jsonrpc: Some("2.0".into()),
            error: None,
        }
    }
}

pub(super) struct InferenceGuard {
    miner: Arc<AtomicBool>,
    busy: Arc<AtomicBool>,
}

impl InferenceGuard {
    pub fn new(miner: Arc<AtomicBool>, busy: Arc<AtomicBool>) -> Self {
        miner.store(true, Ordering::SeqCst);
        Self { miner, busy }
    }
}

impl Drop for InferenceGuard {
    fn drop(&mut self) {
        self.miner.store(false, Ordering::SeqCst);
        self.busy.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(prompt: &str) -> (String, String, String, String, String, u32, String) {
        let h = "ab".repeat(32);
        (h.clone(), "cd".repeat(32), h.to_uppercase(), "73740b443bdc00afda5fa34eb9999d3fea77dcc3f6de238fab701394cdc96fb3".into(), BASE64.encode(prompt), 64, "200000000".into())
    }

    #[test]
    fn parses_a_pool_request_and_builds_the_response() {
        let req = AiRequest::parse(fields("Hello")).unwrap();
        assert_eq!((req.prompt.as_str(), req.max_tokens, req.trace), ("Hello", 64, None));
        let json = serde_json::to_value(req.response(7, "worker".into(), "Hi there", None).unwrap()).unwrap();
        assert_eq!(json["method"], "mining.ai_response");
        assert_eq!(json["id"], 7);
        assert_eq!(json["params"].as_array().unwrap().len(), 5);
        assert_eq!(json["params"][4], BASE64.encode("Hi there"));
        assert!(req.response(8, "w".into(), "", None).is_err(), "never send an empty answer");
    }

    /// Any prompt bytes are served: NUL bytes kept, invalid UTF-8 replaced, a compressed prompt
    /// inflated. Only the size cap and broken base64 refuse a request.
    #[test]
    fn odd_prompts_are_served() {
        use std::io::Write;
        let with = |raw: &[u8]| {
            let mut f = fields("");
            f.4 = BASE64.encode(raw);
            f
        };
        assert_eq!(AiRequest::parse(with(b"Say hel\0lo")).unwrap().prompt, "Say hel\0lo");
        assert_eq!(AiRequest::parse(with(&[b'h', b'i', 0xFF])).unwrap().prompt, "hi\u{FFFD}");
        let text = "abc ".repeat(2_000);
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::best());
        enc.write_all(text.as_bytes()).unwrap();
        let mut packed = vec![0x00, b'K', b'Z', 0x01];
        packed.extend(enc.finish().unwrap());
        assert_eq!(AiRequest::parse(with(&packed)).unwrap().prompt, text);
        assert!(AiRequest::parse(with(&vec![b'a'; MAX_PAYLOAD_BYTES + 1])).is_err());
        let mut f = fields("x");
        f.4 = "not base64!".into();
        assert!(AiRequest::parse(f).is_err());
    }

    /// `trace.prompt_len` is optional (upstream keryx-miner v0.6.0): without it the length of the
    /// prompt bytes sent stands in. Either way it is capped like a private prompt.
    #[test]
    fn trace_prompt_len_falls_back_to_the_sent_length() {
        let cap = keryx_inference::MAX_AI_REQUEST_PRIVATE_PAYLOAD_LEN as u32;
        let ask = |trace: serde_json::Value, prompt: &str| AiRequest::parse_ext(fields(prompt), Some(&serde_json::json!({ "trace": trace }))).map(|r| r.trace.unwrap().prompt_len);
        assert_eq!(ask(serde_json::json!({"tier": 1, "prompt_len": 17}), "Hello").unwrap(), 17);
        assert_eq!(ask(serde_json::json!({"tier": 1}), "Hello").unwrap(), 5);
        assert_eq!(ask(serde_json::json!({"tier": 1, "prompt_len": null}), "Hello").unwrap(), 5);
        assert_eq!(ask(serde_json::json!({"tier": 1}), &"a".repeat(cap as usize + 9)).unwrap(), cap);
        assert_eq!(ask(serde_json::json!({"tier": 1, "prompt_len": u32::MAX}), "Hello").unwrap(), cap);
        assert!(ask(serde_json::json!({"tier": 1, "prompt_len": "17"}), "Hello").is_err());
        assert!(ask(serde_json::json!({"prompt_len": 17}), "Hello").is_err(), "the tier stays required");
    }

    #[test]
    fn traced_requests_and_openings_round_trip() {
        use super::super::statum_codec::{StratumCommand, StratumLine, StratumLinePayload};
        let raw = format!(
            r#"{{"id":null,"method":"mining.ai_request","params":["{h}","{t}","{h}","{m}","{p}",128,"5",{{"trace":{{"tier":1,"prompt_len":17,"chain":true,"attention":true}}}}]}}"#,
            h = "11".repeat(32),
            t = "22".repeat(32),
            m = "33".repeat(32),
            p = BASE64.encode("prompt")
        );
        let line: StratumLine = serde_json::from_str(&raw).unwrap();
        let StratumLinePayload::StratumCommand(StratumCommand::MiningAiRequest(f)) = line.payload else { panic!("not an ai_request") };
        let req = AiRequest::parse_ext(f.fields, f.ext.as_ref()).unwrap();
        assert_eq!(req.trace, Some(TraceAsk { tier: 1, prompt_len: 17, chain: true, attention: true }));
        assert_eq!(req.request_hash_bytes(), [0x11; 32]);
        let json = serde_json::to_value(req.response(9, "w".into(), "answer", Some((&[1u8, 2], &[3u8]))).unwrap()).unwrap();
        assert_eq!(json["params"].as_array().unwrap().len(), 6);
        assert_eq!((json["params"][5]["trace"].as_str(), json["params"][5]["body"].as_str()), (Some("0102"), Some("03")));

        // A plain 7-param request still decodes, with no trace asked.
        let plain = format!(r#"{{"id":null,"method":"mining.ai_request","params":["{h}","{t}","{h}","{m}","{p}",128,"5"]}}"#, h = "11".repeat(32), t = "22".repeat(32), m = "33".repeat(32), p = BASE64.encode("x"));
        let line: StratumLine = serde_json::from_str(&plain).unwrap();
        let StratumLinePayload::StratumCommand(StratumCommand::MiningAiRequest(f)) = line.payload else { panic!("not an ai_request") };
        assert!(f.ext.is_none());

        let o = serde_json::json!(["11".repeat(32), "aa".repeat(32), "bb".repeat(32), [2, 0], "beef"]);
        let op = AiOpening::parse(o.as_array().unwrap()).unwrap();
        assert_eq!((op.path.clone(), op.prev.clone(), op.seed), (vec![2, 0], vec![0xbe, 0xef], [0xbb; 32]));
        let first = serde_json::json!(["11".repeat(32), "aa".repeat(32), "bb".repeat(32), [], ""]);
        assert!(AiOpening::parse(first.as_array().unwrap()).unwrap().path.is_empty());
        // A phase may come without the previous partials (upstream keryx-miner v0.6.0 sends none),
        // and partials without a phase are ignored.
        for no_prev in [serde_json::json!(["11".repeat(32), "aa".repeat(32), "bb".repeat(32), [1], ""]), serde_json::json!(["11".repeat(32), "aa".repeat(32), "bb".repeat(32), [1]])] {
            let op = AiOpening::parse(no_prev.as_array().unwrap()).unwrap();
            assert_eq!((op.path, op.prev), (vec![1], Vec::new()));
        }
        let stray = serde_json::json!(["11".repeat(32), "aa".repeat(32), "bb".repeat(32), [], "beef"]);
        assert!(AiOpening::parse(stray.as_array().unwrap()).unwrap().prev.is_empty());
        let r = serde_json::to_value(op.result(3, "w".into(), Ok(&[0xb0, 1]))).unwrap();
        assert_eq!(r["method"], "mining.ai_opening_result");
        assert_eq!(r["params"][4], "b001");
        assert_eq!(r["params"][3], serde_json::json!([2, 0]));
    }
}
