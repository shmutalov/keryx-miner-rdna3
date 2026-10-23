//! On-disk copy of each traced response, kept until its opening window closes so a restart or a
//! reconnection does not lose an audit in flight.

use borsh::{BorshDeserialize, BorshSerialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const STORE_VERSION: u32 = 2;
const FILE_EXT: &str = "trace";

/// The opening already built for a response: resent as is after a restart.
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct StoredOpening {
    pub seed_hash: String,
    pub deadline_daa: u64,
    /// Hex payload of the signed opening transaction.
    pub payload: String,
}

#[derive(BorshDeserialize)]
pub struct StoredTrace {
    pub version: u32,
    pub response_txid: String,
    pub model_id: [u8; 32],
    pub tier: u8,
    pub request_hash: [u8; 32],
    pub max_tokens: u32,
    pub submitted_daa: u64,
    pub accepted_hash: Option<String>,
    pub opening: Option<StoredOpening>,
    pub trace: crate::trace::ResponseTrace,
}

/// [`StoredTrace`] by reference, for writing.
#[derive(BorshSerialize)]
pub struct StoredTraceRef<'a> {
    pub version: u32,
    pub response_txid: &'a str,
    pub model_id: &'a [u8; 32],
    pub tier: u8,
    pub request_hash: &'a [u8; 32],
    pub max_tokens: u32,
    pub submitted_daa: u64,
    pub accepted_hash: &'a Option<String>,
    pub opening: &'a Option<StoredOpening>,
    pub trace: &'a crate::trace::ResponseTrace,
}

impl<'a> StoredTraceRef<'a> {
    pub fn version() -> u32 {
        STORE_VERSION
    }
}

pub struct TraceStore {
    dir: PathBuf,
}

impl TraceStore {
    /// Store next to the escrow state file: `<state>.traces/`.
    pub fn new(escrow_state_file: &str) -> Self {
        Self { dir: Path::new(escrow_state_file).with_extension("traces") }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path(&self, txid: &str) -> PathBuf {
        self.dir.join(format!("{}.{}", txid, FILE_EXT))
    }

    /// Writes the record atomically: temporary file, sync, rename.
    pub fn save(&self, record: &StoredTraceRef) -> io::Result<()> {
        fs::create_dir_all(&self.dir)?;
        let bytes = borsh::to_vec(record).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let path = self.path(record.response_txid);
        let tmp = path.with_extension("tmp");
        {
            let mut f = fs::File::create(&tmp)?;
            io::Write::write_all(&mut f, &bytes)?;
            f.sync_all()?;
        }
        fs::rename(&tmp, &path)
    }

    pub fn remove(&self, txid: &str) {
        let _ = fs::remove_file(self.path(txid));
    }

    /// Every readable record; unreadable files are dropped with a warning.
    pub fn load_all(&self) -> Vec<StoredTrace> {
        let mut out = Vec::new();
        let Ok(entries) = fs::read_dir(&self.dir) else { return out };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some(FILE_EXT) {
                continue;
            }
            match fs::read(&path).map_err(|e| e.to_string()).and_then(|b| StoredTrace::try_from_slice(&b).map_err(|e| e.to_string())) {
                Ok(t) if t.version == STORE_VERSION => out.push(t),
                Ok(t) => {
                    log::warn!("Trace: stored record {} has version {} — dropped", path.display(), t.version);
                    let _ = fs::remove_file(&path);
                }
                Err(e) => {
                    log::warn!("Trace: stored record {} unreadable ({}) — dropped", path.display(), e);
                    let _ = fs::remove_file(&path);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_round_trips_through_the_store() {
        let dir = std::env::temp_dir().join(format!("keryx-trace-store-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = TraceStore { dir: dir.clone() };
        let trace = crate::trace::ResponseTrace::test_sample();
        let opening = Some(StoredOpening { seed_hash: "ab".into(), deadline_daa: 42, payload: "00ff".into() });
        let rec = StoredTraceRef {
            version: StoredTraceRef::version(),
            response_txid: "deadbeef",
            model_id: &[1u8; 32],
            tier: 3,
            request_hash: &[2u8; 32],
            max_tokens: 2048,
            submitted_daa: 7,
            accepted_hash: &Some("cafe".into()),
            opening: &opening,
            trace: &trace,
        };
        store.save(&rec).unwrap();
        let loaded = store.load_all();
        assert_eq!(loaded.len(), 1);
        let t = &loaded[0];
        assert_eq!(t.response_txid, "deadbeef");
        assert_eq!(t.tier, 3);
        assert_eq!(t.accepted_hash.as_deref(), Some("cafe"));
        assert_eq!(t.opening.as_ref().unwrap().deadline_daa, 42);
        assert_eq!(t.trace.tokens, trace.tokens);
        assert_eq!(t.trace.record.trace_root, trace.record.trace_root);
        store.remove("deadbeef");
        assert!(store.load_all().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }
}
