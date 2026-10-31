//! Traces of pool-dispatched H15 answers, kept so the pool can have them opened when the node
//! audits the response it submitted (`mining.ai_opening`). Keyed by the request hash the pool sent
//! (the pool's response txid is unknown until it asks for an opening). In memory, plus one file
//! per trace in `keryx-pool-traces/` beside the executable (or `$KERYX_POOL_TRACE_DIR`), so a
//! restarted miner can still open; dropped after [`KEEP`] (the node's audit span is ~7.5 h).

use borsh::{BorshDeserialize, BorshSerialize};
use keryx_miner::trace::ResponseTrace;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// How long a pool trace is kept: the node's 270,000-DAA audit span (~7.5 h) plus slack.
const KEEP: Duration = Duration::from_secs(9 * 3600);
const VERSION: u32 = 1;

/// A kept trace and what an opening of it needs besides the pool's request.
#[derive(Clone)]
pub(super) struct Kept {
    pub model_id: [u8; 32],
    pub tier: u8,
    pub request_hash: [u8; 32],
    /// The request's token budget: it sized the context the answer ran in, and so its replays.
    pub max_tokens: u32,
    /// Unix seconds of the answer.
    pub at: u64,
    pub trace: Arc<ResponseTrace>,
}

/// The file of a kept trace: this header, then the borsh trace.
#[derive(BorshSerialize, BorshDeserialize)]
struct Header {
    version: u32,
    model_id: [u8; 32],
    tier: u8,
    request_hash: [u8; 32],
    max_tokens: u32,
    at: u64,
}

fn encode(k: &Kept) -> std::io::Result<Vec<u8>> {
    let h = Header { version: VERSION, model_id: k.model_id, tier: k.tier, request_hash: k.request_hash, max_tokens: k.max_tokens, at: k.at };
    let mut out = borsh::to_vec(&h)?;
    out.extend(borsh::to_vec(&*k.trace)?);
    Ok(out)
}

fn decode(mut b: &[u8]) -> Option<Kept> {
    let h = Header::deserialize(&mut b).ok()?;
    let trace = ResponseTrace::deserialize(&mut b).ok()?;
    (h.version == VERSION && b.is_empty()).then(|| Kept { model_id: h.model_id, tier: h.tier, request_hash: h.request_hash, max_tokens: h.max_tokens, at: h.at, trace: Arc::new(trace) })
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn dir() -> PathBuf {
    if let Some(d) = std::env::var_os("KERYX_POOL_TRACE_DIR") {
        return PathBuf::from(d);
    }
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())).unwrap_or_default();
    exe_dir.join("keryx-pool-traces")
}

fn file_of(key: &str) -> PathBuf {
    dir().join(format!("{key}.trace"))
}

fn live(at: u64, now: u64) -> bool {
    at + KEEP.as_secs() >= now
}

/// Every kept trace, loaded from disk on first use (expired files are removed).
fn kept() -> &'static Mutex<HashMap<String, Kept>> {
    static M: OnceLock<Mutex<HashMap<String, Kept>>> = OnceLock::new();
    M.get_or_init(|| {
        let mut m = HashMap::new();
        let t = now();
        if let Ok(entries) = std::fs::read_dir(dir()) {
            for e in entries.flatten() {
                let path = e.path();
                let Some(key) = path.file_stem().and_then(|s| s.to_str()).map(str::to_string) else { continue };
                if path.extension().and_then(|x| x.to_str()) != Some("trace") {
                    continue;
                }
                match std::fs::read(&path).ok().and_then(|b| decode(&b)) {
                    Some(k) if live(k.at, t) => {
                        m.insert(key, k);
                    }
                    _ => {
                        let _ = std::fs::remove_file(&path);
                    }
                }
            }
        }
        if !m.is_empty() {
            log::info!("Trace: restored {} pool answer trace(s) from {}", m.len(), dir().display());
        }
        Mutex::new(m)
    })
}

/// Keeps the trace of a pool-dispatched answer (memory and disk) for later openings.
pub(super) fn keep(request_hash_hex: &str, kept_trace: Kept) {
    let key = request_hash_hex.to_ascii_lowercase();
    let t = now();
    // Loaded before this trace's file is written, so the load reports only earlier traces.
    let mut m = kept().lock().unwrap_or_else(|p| p.into_inner());
    match encode(&kept_trace) {
        Ok(bytes) => {
            let write = std::fs::create_dir_all(dir()).and_then(|_| {
                let tmp = file_of(&format!("{key}.tmp"));
                std::fs::write(&tmp, &bytes)?;
                std::fs::rename(&tmp, file_of(&key))
            });
            if let Err(e) = write {
                log::warn!("Trace: pool trace {key:.16} kept in memory only ({e})");
            }
        }
        Err(e) => log::warn!("Trace: pool trace {key:.16} not serializable ({e})"),
    }
    m.retain(|k, v| {
        let keep = live(v.at, t);
        if !keep {
            let _ = std::fs::remove_file(file_of(k));
        }
        keep
    });
    m.insert(key, kept_trace);
}

/// The kept trace of the answer to `request_hash_hex`.
pub(super) fn get(request_hash_hex: &str) -> Option<Kept> {
    let m = kept().lock().unwrap_or_else(|p| p.into_inner());
    m.get(&request_hash_hex.to_ascii_lowercase()).filter(|k| live(k.at, now())).cloned()
}
