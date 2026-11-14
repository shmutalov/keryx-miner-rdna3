//! Traces of pool-dispatched H15 answers, kept so the pool can have them opened when the node
//! audits the response it submitted (`mining.ai_opening`). Keyed by the request hash the pool sent
//! (the pool's response txid is unknown until it asks for an opening). In memory, plus one file
//! per trace in `keryx-pool-traces/` beside the executable (or `$KERYX_POOL_TRACE_DIR`), so a
//! restarted miner can still open; dropped after [`KEEP`] (the node's audit span is ~7.5 h), or
//! as soon as the pool refuses the answer ([`forget`]).

use borsh::{BorshDeserialize, BorshSerialize};
use keryx_miner::trace::ResponseTrace;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// How long a pool trace is kept: the node's 270,000-DAA audit span (~7.5 h) plus slack.
const KEEP: Duration = Duration::from_secs(9 * 3600);
const VERSION: u32 = 1;
/// Stem suffix of a file still being written: `<key>.tmp.trace`, renamed to `<key>.trace`.
const PART: &str = ".tmp";

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

fn file_in(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("{key}.trace"))
}

fn live(at: u64, now: u64) -> bool {
    at + KEEP.as_secs() >= now
}

/// Writes `bytes` as the file of `key` in `dir`, synced to disk before the rename, so a crash or
/// power cut leaves no file, the old one, or a complete one (a synced part file [`load`] renames
/// into place).
fn write_durably(dir: &Path, key: &str, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(dir)?;
    let part = file_in(dir, &format!("{key}{PART}"));
    let mut f = std::fs::File::create(&part)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    drop(f);
    std::fs::rename(&part, file_in(dir, key))?;
    // On Unix the rename is durable once the directory is synced; NTFS journals it itself.
    #[cfg(unix)]
    std::fs::File::open(dir)?.sync_all()?;
    Ok(())
}

/// The live traces in `dir` at unix time `t`. Expired and unreadable files are removed, and a part
/// file whose rename never ran is renamed into place: it was synced first, so one that decodes is
/// complete.
fn load(dir: &Path, t: u64) -> HashMap<String, Kept> {
    let mut m = HashMap::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return m };
    for e in entries.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("trace") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()).map(str::to_string) else { continue };
        let (key, part) = match stem.strip_suffix(PART) {
            Some(k) => (k.to_string(), true),
            None => (stem, false),
        };
        match std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            Some(k) if live(k.at, t) => {
                if m.contains_key(&key) {
                    // Both the part and the final file of one key: the one loaded first stays (a
                    // part already renamed onto this path, or the final file).
                    if part {
                        let _ = std::fs::remove_file(&path);
                    }
                    continue;
                }
                if part && std::fs::rename(&path, file_in(dir, &key)).is_err() {
                    let _ = std::fs::remove_file(&path);
                }
                m.insert(key, k);
            }
            _ => {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
    m
}

/// Every kept trace, loaded from disk on first use.
fn kept() -> &'static Mutex<HashMap<String, Kept>> {
    static M: OnceLock<Mutex<HashMap<String, Kept>>> = OnceLock::new();
    M.get_or_init(|| {
        let m = load(&dir(), now());
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
    let dir = dir();
    // Loaded before this trace's file is written, so the load reports only earlier traces.
    let mut m = kept().lock().unwrap_or_else(|p| p.into_inner());
    match encode(&kept_trace) {
        Ok(bytes) => {
            if let Err(e) = write_durably(&dir, &key, &bytes) {
                log::warn!("Trace: pool trace {key:.16} kept in memory only ({e})");
            }
        }
        Err(e) => log::warn!("Trace: pool trace {key:.16} not serializable ({e})"),
    }
    m.retain(|k, v| {
        let keep = live(v.at, t);
        if !keep {
            let _ = std::fs::remove_file(file_in(&dir, k));
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

/// Drops the kept trace of the answer to `request_hash_hex` (memory and disk), as for an answer the
/// pool refused: it is never submitted, so never audited. Whether one was kept.
pub(super) fn forget(request_hash_hex: &str) -> bool {
    let key = request_hash_hex.to_ascii_lowercase();
    let removed = kept().lock().unwrap_or_else(|p| p.into_inner()).remove(&key).is_some();
    let _ = std::fs::remove_file(file_in(&dir(), &key));
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(at: u64) -> Kept {
        Kept { model_id: [1; 32], tier: 1, request_hash: [2; 32], max_tokens: 2048, at, trace: Arc::new(ResponseTrace::test_sample()) }
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("keryx-pool-traces-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn names(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        v.sort();
        v
    }

    #[test]
    fn a_durable_write_round_trips() {
        let d = scratch("roundtrip");
        write_durably(&d, "aa", &encode(&sample(100)).unwrap()).unwrap();
        assert_eq!(names(&d), ["aa.trace"]);
        let m = load(&d, 100);
        assert_eq!(m["aa"].trace.tokens, ResponseTrace::test_sample().tokens);
        assert_eq!(m["aa"].max_tokens, 2048);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_complete_part_file_is_renamed_into_place() {
        // A power cut between the synced write and the rename.
        let d = scratch("part");
        std::fs::write(file_in(&d, "bb.tmp"), encode(&sample(100)).unwrap()).unwrap();
        let m = load(&d, 100);
        assert!(m.contains_key("bb"));
        assert_eq!(names(&d), ["bb.trace"]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_part_file_beside_its_final_file_leaves_one_file() {
        let d = scratch("both");
        std::fs::write(file_in(&d, "cc"), encode(&sample(100)).unwrap()).unwrap();
        std::fs::write(file_in(&d, "cc.tmp"), encode(&sample(100)).unwrap()).unwrap();
        let m = load(&d, 100);
        assert_eq!(m.len(), 1);
        assert_eq!(names(&d), ["cc.trace"]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn torn_and_expired_files_are_removed() {
        let d = scratch("torn");
        let bytes = encode(&sample(100)).unwrap();
        std::fs::write(file_in(&d, "dd.tmp"), &bytes[..bytes.len() / 2]).unwrap();
        std::fs::write(file_in(&d, "ee"), encode(&sample(1)).unwrap()).unwrap();
        let m = load(&d, 1 + KEEP.as_secs() + 1);
        assert!(m.is_empty());
        assert!(names(&d).is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }
}
