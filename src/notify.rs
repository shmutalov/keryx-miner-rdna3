//! Telegram notifications: instant alerts for the events that need a look — a service-bond
//! change, a hashrate stall, a lost node or pool link, a rejected block, a start or a stop — and
//! the miner's stats on request: `/stats` to the bot, from the configured chat only.
//!
//! With `ai = 1` it also forwards each served AI request: the prompt and the answer.
//!
//! Settings come from a file (`telegram.conf`, or `--telegram-config`; `telegram.conf.example`
//! shows them): off while it is missing or has no token. The token is masked in every log line.
//! Messages go out from their own thread behind a bounded queue: a slow or unreachable Telegram
//! never holds up mining, and a full queue drops messages instead of growing.
//!
//! The counters are fed from the existing event points (block submits, share results, AI
//! requests, escrow claims, the hashrate logger) whether notifications are on or not: they are a
//! few relaxed atomics. They live for the process, so a reconnect keeps them and a restart by
//! the launcher resets them. The wallet balance and the network's numbers come from the explorer
//! API (`--recover-escrow-api`, keryx-labs.com), the lifetime strike tally from the node.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use log::{error, info, warn};
use serde_json::{json, Value};

const DEFAULT_API: &str = "https://api.telegram.org";
/// getUpdates long-poll wait, seconds: a `/stats` is answered as soon as it arrives.
const POLL_WAIT: u64 = 50;
/// `/stats` compares against a snapshot up to an hour old: one every 5 minutes, 13 kept.
const HISTORY_STEP: Duration = Duration::from_secs(5 * 60);
const HISTORY_LEN: usize = 13;
/// Foreign chats whose commands are logged (once each) before the rest are ignored silently.
const STRANGERS_MAX: usize = 100;
/// How long the hashrate may stay at zero outside inference pauses before an alert.
const STALL_ALERT: Duration = Duration::from_secs(5 * 60);
/// How long the node or pool link may stay down before an alert: reconnect blips stay quiet.
const LINK_ALERT: Duration = Duration::from_secs(2 * 60);
/// At most one rejected-block alert per this period; `/stats` keeps the count.
const REJECT_ALERT_GAP: Duration = Duration::from_secs(60 * 60);
const QUEUE_LEN: usize = 32;
const TICK: Duration = Duration::from_secs(5);
const SEND_ATTEMPTS: u32 = 3;
/// Longest error text quoted in an alert.
const REASON_MAX: usize = 200;
/// AI message budget, in characters: Telegram takes 4096 per message (UTF-16 units, so leave
/// room for emoji and other astral characters).
const AI_PROMPT_MAX: usize = 1400;
const AI_ANSWER_MAX: usize = 2200;

/// What this process mines and where, for the start message and `/stats`.
pub struct Session {
    /// `grpc://` node (solo) rather than a stratum pool.
    pub solo: bool,
    pub endpoint: String,
    pub address: String,
    pub tier: String,
    /// Explorer API base URL (`--recover-escrow-api`): balance and UTXO count of `address`, and
    /// the network's numbers.
    pub explorer: String,
}

/// The network's numbers from the explorer's `/info`.
#[derive(Clone, Debug, PartialEq)]
struct Network {
    hashrate: f64,
    reward_krx: f64,
    daa: u64,
}

/// What the `ai` setting forwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AiMode {
    Off,
    /// Every request but solo private ones. A pool sends opened prompts without saying whether
    /// the request was private, so pool requests all go.
    Public,
    /// Solo private requests too.
    All,
}

#[derive(Debug, Clone, PartialEq)]
struct Config {
    token: String,
    /// `None`: the token is set without a chat id; the miner looks up the chats that wrote to
    /// the bot and logs them.
    chat: Option<String>,
    name: String,
    api: String,
    ai: AiMode,
}

/// The settings file's keys.
const KEYS: [&str; 5] = ["token", "chat", "ai", "name", "api"];

/// Parses the settings file: `key = value` lines, keys case-insensitive, a value optionally in
/// double quotes. `#` or `;` starts a comment at the start of a line, `#` also after whitespace.
fn parse_settings(text: &str) -> Result<HashMap<String, String>, String> {
    let mut out = HashMap::new();
    for (n, line) in text.trim_start_matches('\u{feff}').lines().enumerate() {
        let line = strip_comment(line).trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("line {}: expected `key = value`", n + 1));
        };
        let value = value.trim();
        let value = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')).unwrap_or(value);
        out.insert(key.trim().to_ascii_lowercase(), value.to_string());
    }
    Ok(out)
}

fn strip_comment(line: &str) -> &str {
    let t = line.trim_start();
    if t.starts_with('#') || t.starts_with(';') {
        return "";
    }
    match line.find(" #").or_else(|| line.find("\t#")) {
        Some(i) => &line[..i],
        None => line,
    }
}

/// The configuration in `settings`; `None` without a token. `host` names the rig by default.
fn read_config(settings: &HashMap<String, String>, host: Option<String>) -> Option<Config> {
    for key in settings.keys().filter(|k| !KEYS.contains(&k.as_str())) {
        warn!("Telegram: unknown setting `{}` in {} — ignored", key, conf_path());
    }
    let get = |k: &str| settings.get(k).map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    let token = get("token")?;
    Some(Config {
        token,
        chat: get("chat"),
        name: get("name").or(host).unwrap_or_else(|| "miner".into()),
        api: get("api").unwrap_or_else(|| DEFAULT_API.into()).trim_end_matches('/').to_string(),
        ai: match get("ai").map(|v| v.to_ascii_lowercase()).as_deref() {
            None | Some("0") => AiMode::Off,
            Some("1") => AiMode::Public,
            Some("all") => AiMode::All,
            Some(v) => {
                warn!("Telegram: ai = {} in {} is not 0, 1 or all — AI messages off", v, conf_path());
                AiMode::Off
            }
        },
    })
}

/// Reads the settings file at `path`. Missing: notifications off, quietly.
fn load_config(path: &str) -> Option<Config> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            log::debug!("Telegram: no {} — notifications off", path);
            return None;
        }
        Err(e) => {
            warn!("Telegram: cannot read {}: {} — notifications off", path, e);
            return None;
        }
    };
    let settings = match parse_settings(&text) {
        Ok(s) => s,
        Err(e) => {
            warn!("Telegram: {}: {} — notifications off", path, e);
            return None;
        }
    };
    let host = std::env::var("COMPUTERNAME").or_else(|_| std::env::var("HOSTNAME")).ok().filter(|h| !h.is_empty());
    let config = read_config(&settings, host);
    if config.is_none() {
        info!("Telegram: {} has no token — notifications off", path);
    }
    config
}

/// The settings file in use, for messages that point at it.
fn conf_path() -> &'static str {
    CONFIG_PATH.get().map_or("telegram.conf", |s| s.as_str())
}

struct Counters {
    hashes: AtomicU64,
    blocks: AtomicU64,
    blocks_rejected: AtomicU64,
    ai_requests: AtomicU64,
    ai_answered: AtomicU64,
    ai_missed: AtomicU64,
    openings: AtomicU64,
    openings_failed: AtomicU64,
    claimed_outputs: AtomicU64,
    claimed_sompi: AtomicU64,
    pending_outputs: AtomicU64,
    pending_sompi: AtomicU64,
}

static C: Counters = Counters {
    hashes: AtomicU64::new(0),
    blocks: AtomicU64::new(0),
    blocks_rejected: AtomicU64::new(0),
    ai_requests: AtomicU64::new(0),
    ai_answered: AtomicU64::new(0),
    ai_missed: AtomicU64::new(0),
    openings: AtomicU64::new(0),
    openings_failed: AtomicU64::new(0),
    claimed_outputs: AtomicU64::new(0),
    claimed_sompi: AtomicU64::new(0),
    pending_outputs: AtomicU64::new(0),
    pending_sompi: AtomicU64::new(0),
};

#[derive(Clone, Debug, PartialEq)]
struct Bond {
    status: String,
    active: u32,
    lifetime: usize,
}

struct State {
    balance: Option<u64>,
    /// The first balance this process read: `/stats` shows the change since.
    balance_first: Option<u64>,
    balance_error: Option<String>,
    utxos: Option<u64>,
    network: Option<Network>,
    bond: Option<Bond>,
    link_up_since: Option<Instant>,
    link_down_since: Option<Instant>,
    link_down_reason: String,
    link_alerted: bool,
    /// Last hashrate tick with hashes, or inside an inference / model-load pause.
    last_activity: Option<Instant>,
    stall_alerted: bool,
    last_reject_alert: Option<Instant>,
}

static STATE: Mutex<State> = Mutex::new(State {
    balance: None,
    balance_first: None,
    balance_error: None,
    utxos: None,
    network: None,
    bond: None,
    link_up_since: None,
    link_down_since: None,
    link_down_reason: String::new(),
    link_alerted: false,
    last_activity: None,
    stall_alerted: false,
    last_reject_alert: None,
});

fn state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// A queued message. Silent ones (AI messages) arrive without a sound.
struct Msg {
    text: String,
    silent: bool,
}

struct Bot {
    tx: SyncSender<Msg>,
    name: String,
    session: Session,
    ai: AiMode,
}

static BOT: OnceLock<Bot> = OnceLock::new();
/// Messages queued or being sent; `stopping` waits for it to drain.
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
static STOPPING: AtomicBool = AtomicBool::new(false);
static STARTED: OnceLock<Instant> = OnceLock::new();
static CONFIG_PATH: OnceLock<String> = OnceLock::new();
/// Snapshots every `HISTORY_STEP` over the last hour; `/stats` compares against the oldest.
static HISTORY: Mutex<VecDeque<(Instant, Snapshot)>> = Mutex::new(VecDeque::new());

fn history() -> MutexGuard<'static, VecDeque<(Instant, Snapshot)>> {
    HISTORY.lock().unwrap_or_else(|e| e.into_inner())
}

/// Starts notifications when the settings file at `path` has a token and a chat id, and sends
/// the start message. Call once, after logging is up.
pub fn init(session: Session, path: &str) {
    CONFIG_PATH.get_or_init(|| path.to_string());
    let Some(cfg) = load_config(path) else { return };
    let Some(chat) = cfg.chat.clone() else {
        warn!("Telegram: {} has a token but no chat — notifications off. Looking up the chats that wrote to the bot...", path);
        std::thread::Builder::new().name("tg-chats".into()).spawn(move || discover_chats(&cfg)).ok();
        return;
    };
    let (tx, rx) = sync_channel(QUEUE_LEN);
    STARTED.get_or_init(Instant::now);
    let start = format!(
        "▶ {}: miner {} started · {} via {} · tier {} · pays {} · /stats for numbers",
        cfg.name,
        env!("CARGO_PKG_VERSION"),
        if session.solo { "solo" } else { "pool" },
        session.endpoint,
        session.tier,
        short_address(&session.address),
    );
    let ai = cfg.ai;
    if BOT.set(Bot { tx, name: cfg.name.clone(), session, ai }).is_err() {
        return;
    }
    info!("Telegram notifications on: alerts to chat {}, stats on /stats", chat);
    if ai != AiMode::Off {
        keryx_miner::slm::set_ai_observer(observe_inference);
        warn!(
            "Telegram AI messages on (ai in {}): the prompt and answer of every request this miner serves go to Telegram, which stores them{}",
            path,
            match ai {
                AiMode::All => ", private requests included",
                _ => " (solo private requests excluded; a pool does not mark private requests, so every pool request goes)",
            }
        );
    }
    let (api, token, poll_chat) = (cfg.api.clone(), cfg.token.clone(), chat.clone());
    let sender = std::thread::Builder::new().name("tg-send".into()).spawn(move || run_sender(cfg, chat, rx));
    let watch = std::thread::Builder::new().name("tg-watch".into()).spawn(run_watch);
    let poller = std::thread::Builder::new().name("tg-poll".into()).spawn(move || run_poller(api, token, poll_chat));
    if let Err(e) = sender.and(watch).and(poller) {
        error!("Telegram: cannot start the notification threads: {}", e);
        return;
    }
    post(Msg { text: start, silent: false });
}

pub fn enabled() -> bool {
    BOT.get().is_some()
}

/// One hashrate-logger tick: `n` hashes at `rate` H/s, `paused` while inference or a model load
/// holds PoW on purpose.
pub fn hashes(n: u64, rate: f64, paused: bool) {
    C.hashes.fetch_add(n, Ordering::Relaxed);
    let mut s = state();
    if n > 0 || paused || s.last_activity.is_none() {
        s.last_activity = Some(Instant::now());
    }
    if n > 0 && s.stall_alerted {
        s.stall_alerted = false;
        drop(s);
        alert("✅", format!("hashrate back: {}", fmt_rate(rate)));
    }
}

pub fn block_found() {
    C.blocks.fetch_add(1, Ordering::Relaxed);
}

pub fn block_rejected(reason: &str) {
    C.blocks_rejected.fetch_add(1, Ordering::Relaxed);
    let mut s = state();
    if s.last_reject_alert.is_some_and(|t| t.elapsed() < REJECT_ALERT_GAP) {
        return;
    }
    s.last_reject_alert = Some(Instant::now());
    drop(s);
    alert(
        "⚠",
        format!("block rejected by the node: {} (more rejects within the hour only show in /stats)", clip(reason)),
    );
}

/// An AI request this miner is due to answer (solo: queued from a block; pool: dispatched).
pub fn ai_request() {
    C.ai_requests.fetch_add(1, Ordering::Relaxed);
}

/// An answer the node's mempool or the pool accepted.
pub fn ai_answered() {
    C.ai_answered.fetch_add(1, Ordering::Relaxed);
}

/// A request left unanswered: skipped, failed, or the answer refused.
pub fn ai_missed() {
    C.ai_missed.fetch_add(1, Ordering::Relaxed);
}

/// A pool trace opening: built and sent, or its replay failed.
pub fn opening(ok: bool) {
    if ok { &C.openings } else { &C.openings_failed }.fetch_add(1, Ordering::Relaxed);
}

pub fn escrow_claimed(outputs: u64, sompi: u64) {
    C.claimed_outputs.fetch_add(outputs, Ordering::Relaxed);
    C.claimed_sompi.fetch_add(sompi, Ordering::Relaxed);
}

pub fn escrow_pending(outputs: u64, sompi: u64) {
    C.pending_outputs.store(outputs, Ordering::Relaxed);
    C.pending_sompi.store(sompi, Ordering::Relaxed);
}

/// The payout address balance from the explorer, or why it could not be read. A failure keeps
/// the last balance and is logged once per outage.
fn record_balance(result: Result<u64, String>) {
    let mut s = state();
    match result {
        Ok(sompi) => {
            s.balance = Some(sompi);
            s.balance_first.get_or_insert(sompi);
            s.balance_error = None;
        }
        Err(e) => {
            if s.balance_error.is_none() {
                warn!("Telegram: cannot read the wallet balance from the explorer API: {}", e);
            }
            s.balance_error = Some(e);
        }
    }
}

/// The explorer API (keryx-labs.com by default): the payout address's balance and UTXO count, and
/// the network's numbers.
struct Explorer {
    agent: ureq::Agent,
    base: String,
    address: String,
}

impl Explorer {
    fn new(base: &str, address: &str) -> Self {
        Explorer {
            agent: ureq::AgentBuilder::new().timeout(Duration::from_secs(10)).try_proxy_from_env(true).build(),
            base: format!("{}/api/v1", base.trim_end_matches('/')),
            address: url_component(address),
        }
    }

    fn get(&self, path: &str) -> Result<Value, String> {
        match self.agent.get(&format!("{}/{}", self.base, path)).call() {
            Ok(resp) => resp.into_json().map_err(|e| format!("{}: bad answer: {}", path, e)),
            Err(ureq::Error::Status(code, _)) => Err(format!("HTTP {} from {}", code, path)),
            Err(e) => Err(e.to_string()),
        }
    }

    fn balance(&self) -> Result<u64, String> {
        let v = self.get(&format!("addresses/{}/balance", self.address))?;
        v["balance_sompi"].as_u64().ok_or_else(|| "the balance answer has no balance_sompi".to_string())
    }

    fn utxos(&self) -> Option<u64> {
        self.get(&format!("addresses/{}/utxos/count", self.address)).ok()?["count"].as_u64()
    }

    fn network(&self) -> Option<Network> {
        parse_network(&self.get("info").ok()?)
    }

    /// Reads the balance, and with `full` the UTXO count and the network too, into the state.
    fn refresh(&self, full: bool) {
        record_balance(self.balance());
        if full {
            let (utxos, network) = (self.utxos(), self.network());
            let mut s = state();
            s.utxos = utxos.or(s.utxos);
            s.network = network.or(s.network.take());
        }
    }
}

fn parse_network(v: &Value) -> Option<Network> {
    Some(Network {
        hashrate: v["hashrate_hps"].as_f64()?,
        reward_krx: v["block_reward_krx"].as_f64()?,
        daa: v["last_daa_score"].as_u64()?,
    })
}

/// Percent-encodes everything but unreserved characters (`keryx:` → `keryx%3A`).
fn url_component(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{:02X}", b),
        })
        .collect()
}

/// The service-bond standing the node reports for this miner (`status` is "clear" or the
/// strike / burn / suspension summary). Alerts when it changes, and on a first report that
/// is not clear.
pub fn service_bond(status: &str, active: u32, lifetime: usize) {
    let new = Bond { status: status.to_string(), active, lifetime };
    let old = state().bond.replace(new);
    let changed = match &old {
        Some(o) => o.status != status,
        None => status != "clear",
    };
    if !changed {
        return;
    }
    if status == "clear" {
        alert("✅", format!("service bond clear again ({} lifetime strikes)", lifetime));
    } else {
        alert("⚠", format!("service bond: {} — {} active, {} lifetime strikes", status, active, lifetime));
    }
}

/// The node or pool connection is up (registered).
pub fn link_up() {
    let mut s = state();
    let down = s.link_down_since.take();
    let alerted = std::mem::take(&mut s.link_alerted);
    s.link_up_since = Some(Instant::now());
    // The hashrate logger restarts with the connection: give it a fresh stall window.
    s.last_activity = Some(Instant::now());
    s.stall_alerted = false;
    drop(s);
    if let (true, Some(t), Some(bot)) = (alerted, down, BOT.get()) {
        alert("✅", format!("back on {} after {} down", bot.session.endpoint, fmt_dur(t.elapsed())));
    }
}

/// The connection ended or could not be made; the miner keeps reconnecting.
pub fn link_down(reason: &str) {
    let mut s = state();
    s.link_up_since = None;
    s.link_down_since.get_or_insert_with(Instant::now);
    s.link_down_reason = clip(reason);
}

/// The slm observer: one finished inference. Challenges are the node's or pool's own probes of
/// this miner, not requests, and are not forwarded.
fn observe_inference(label: &str, model: &str, prompt: &str, answer: Option<&str>, secs: f64) {
    if label.contains("challenge") {
        return;
    }
    ai_exchange(label, model, prompt, answer.ok_or("no answer"), Some(secs));
}

/// One AI request and what came of it: the answer, or why there is none. Sent only when
/// the `ai` setting allows this request, silently, and only while the queue has room to spare.
pub fn ai_exchange(label: &str, model: &str, prompt: &str, answer: Result<&str, &str>, secs: Option<f64>) {
    let Some(bot) = BOT.get() else { return };
    if !ai_allowed(bot.ai, label) {
        return;
    }
    // Leave half the queue to alerts.
    if IN_FLIGHT.load(Ordering::Acquire) >= QUEUE_LEN / 2 {
        log::debug!("Telegram: queue busy — AI message for {} dropped", label);
        return;
    }
    post(Msg { text: render_ai(&bot.name, label, model, prompt, answer, secs), silent: true });
}

/// Whether `mode` forwards the request `label` names (solo private requests end in ", private").
fn ai_allowed(mode: AiMode, label: &str) -> bool {
    match mode {
        AiMode::Off => false,
        AiMode::Public => !label.ends_with(", private"),
        AiMode::All => true,
    }
}

/// Sends a stop message (once per process) and waits up to 2.5 s for the queue to drain.
/// Blocking: from async code call it where a short block before exit is fine.
pub fn stopping(reason: &str) {
    if !enabled() {
        return;
    }
    if !STOPPING.swap(true, Ordering::AcqRel) {
        alert("⏹", format!("miner stopping: {}", reason));
    }
    let t0 = Instant::now();
    while IN_FLIGHT.load(Ordering::Acquire) > 0 && t0.elapsed() < Duration::from_millis(2500) {
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn post(msg: Msg) {
    let Some(bot) = BOT.get() else { return };
    IN_FLIGHT.fetch_add(1, Ordering::AcqRel);
    if bot.tx.try_send(msg).is_err() {
        IN_FLIGHT.fetch_sub(1, Ordering::AcqRel);
        log::debug!("Telegram: queue full — message dropped");
    }
}

fn alert(icon: &str, text: String) {
    if let Some(bot) = BOT.get() {
        post(Msg { text: format!("{} {}: {}", icon, bot.name, text), silent: false });
    }
}

/// Watches for stalls and lost links, and keeps the hour of snapshots `/stats` compares against.
fn run_watch() {
    let explorer = BOT.get().map(|b| Explorer::new(&b.session.explorer, &b.session.address));
    let mut next_snapshot = Instant::now();
    loop {
        if Instant::now() >= next_snapshot {
            // The balance goes into each snapshot: `/stats` shows what the last hour earned.
            if let Some(e) = &explorer {
                e.refresh(false);
            }
            // Take the snapshot before locking the history: `Snapshot::take` locks the state.
            let snap = Snapshot::take();
            let mut h = history();
            h.push_back((Instant::now(), snap));
            while h.len() > HISTORY_LEN {
                h.pop_front();
            }
            next_snapshot = Instant::now() + HISTORY_STEP;
        }
        watch_link();
        watch_stall();
        std::thread::sleep(TICK);
    }
}

/// The `/stats` answer: now, against the oldest snapshot of the last hour.
fn stats_text() -> String {
    let Some(bot) = BOT.get() else { return String::new() };
    let now = Snapshot::take();
    let (at, prev) = history().front().cloned().unwrap_or_else(|| (Instant::now(), now.clone()));
    let uptime = STARTED.get().map_or(Duration::ZERO, |t| t.elapsed());
    render_summary(&bot.name, bot.session.solo, &now, &prev, at.elapsed(), uptime)
}

fn help_text(name: &str) -> String {
    format!(
        "This bot reports for the Keryx miner {}.\n\
         /stats — hashrate, network share, blocks or shares, balance, escrow, AI requests, service bond, link\n\
         Warnings arrive on their own: service-bond strikes, a hashrate stall, a lost node or pool, a rejected block, a restart.",
        name
    )
}

#[derive(Debug, PartialEq)]
enum Command {
    Stats,
    Help,
}

#[derive(Debug, PartialEq)]
enum Incoming {
    Command(Command),
    /// A command from a chat other than the configured one: anyone can find a bot and write to it.
    Stranger(String),
    Other,
}

/// What one getUpdates entry asks of the miner.
fn incoming(update: &Value, chat: &str) -> Incoming {
    let msg = &update["message"];
    let Some(word) = msg["text"].as_str().and_then(|t| t.split_whitespace().next()).filter(|w| w.starts_with('/')) else {
        return Incoming::Other;
    };
    let from = msg["chat"]["id"].as_i64().map(|id| id.to_string()).unwrap_or_default();
    if from != chat {
        return Incoming::Stranger(from);
    }
    // `/stats@SomeBot` in a group.
    let name = word[1..].split('@').next().unwrap_or("").to_ascii_lowercase();
    Incoming::Command(match name.as_str() {
        "stats" | "status" => Command::Stats,
        _ => Command::Help,
    })
}

/// The offset that confirms every update in `updates`.
fn next_offset(updates: &[Value], current: i64) -> i64 {
    updates.iter().filter_map(|u| u["update_id"].as_i64()).map(|id| id + 1).fold(current, i64::max)
}

/// Answers `/stats` and `/help` from `chat`, long-polling getUpdates.
fn run_poller(api: String, token: String, chat: String) {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(POLL_WAIT + 20))
        .try_proxy_from_env(true)
        .build();
    set_commands(&agent, &api, &token);
    let explorer = BOT.get().map(|b| Explorer::new(&b.session.explorer, &b.session.address));
    let url = format!("{}/bot{}/getUpdates", api, token);
    // Skip what was sent while the miner was down: offset -1 returns only the last update and
    // confirms the ones before it.
    let mut offset = match get_updates(&agent, &url, -1, 0, &token) {
        Ok(updates) => next_offset(&updates, 0),
        Err(_) => 0,
    };
    let mut strangers: HashSet<String> = HashSet::new();
    let mut backoff = Duration::from_secs(5);
    let mut failing = false;
    loop {
        match get_updates(&agent, &url, offset, POLL_WAIT, &token) {
            Ok(updates) => {
                if std::mem::take(&mut failing) {
                    info!("Telegram: reading the bot's messages again");
                }
                backoff = Duration::from_secs(5);
                offset = next_offset(&updates, offset);
                for update in &updates {
                    match incoming(update, &chat) {
                        Incoming::Command(Command::Stats) => {
                            if let Some(e) = &explorer {
                                e.refresh(true);
                            }
                            post(Msg { text: stats_text(), silent: false })
                        }
                        Incoming::Command(Command::Help) => {
                            let name = BOT.get().map_or("", |b| b.name.as_str());
                            post(Msg { text: help_text(name), silent: false })
                        }
                        Incoming::Stranger(id) => {
                            if strangers.len() < STRANGERS_MAX && strangers.insert(id.clone()) {
                                warn!("Telegram: ignoring commands from chat {} — only the configured chat ({}) is answered", id, chat);
                            }
                        }
                        Incoming::Other => {}
                    }
                }
            }
            Err(Failure::Fatal(why)) => {
                error!("Telegram: {} — /stats off until the miner restarts", why);
                return;
            }
            Err(Failure::Dropped(why)) => {
                if !std::mem::replace(&mut failing, true) {
                    warn!("Telegram: cannot read the bot's messages: {} — retrying", why);
                }
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(Duration::from_secs(120));
            }
        }
    }
}

/// One getUpdates call. `Dropped` here means: retry later.
fn get_updates(agent: &ureq::Agent, url: &str, offset: i64, wait: u64, token: &str) -> Result<Vec<Value>, Failure> {
    let request = agent
        .get(url)
        .query("offset", &offset.to_string())
        .query("timeout", &wait.to_string())
        .query("allowed_updates", r#"["message"]"#);
    match request.call() {
        Ok(resp) => {
            let v: Value = resp.into_json().map_err(|e| Failure::Dropped(e.to_string()))?;
            Ok(v["result"].as_array().cloned().unwrap_or_default())
        }
        Err(ureq::Error::Status(code, resp)) => {
            let v: Value = resp.into_json().unwrap_or(Value::Null);
            let desc = mask(&format!("HTTP {} {}", code, v["description"].as_str().unwrap_or("")), token);
            Err(match code {
                401 | 404 => Failure::Fatal(format!("the bot token was rejected ({})", desc)),
                409 => Failure::Dropped(format!(
                    "{} — another program or rig reads this bot's messages, or a webhook is set: give each rig its own bot",
                    desc
                )),
                _ => Failure::Dropped(desc),
            })
        }
        Err(e) => Err(Failure::Dropped(mask(&e.to_string(), token))),
    }
}

/// Puts `/stats` and `/help` in the bot's command menu.
fn set_commands(agent: &ureq::Agent, api: &str, token: &str) {
    let url = format!("{}/bot{}/setMyCommands", api, token);
    let body = json!({ "commands": [
        { "command": "stats", "description": "Miner stats: hashrate, blocks, balance, AI, service bond" },
        { "command": "help", "description": "What this bot reports" },
    ]});
    if let Err(e) = agent.post(&url).send_json(&body) {
        log::debug!("Telegram: setMyCommands failed: {}", mask(&e.to_string(), token));
    }
}

fn watch_link() {
    let mut s = state();
    let Some(since) = s.link_down_since else { return };
    if s.link_alerted || since.elapsed() < LINK_ALERT {
        return;
    }
    s.link_alerted = true;
    let reason = s.link_down_reason.clone();
    drop(s);
    if let Some(bot) = BOT.get() {
        let what = if bot.session.solo { "node" } else { "pool" };
        alert(
            "⚠",
            format!("lost the {} {} {} ago: {} — reconnecting", what, bot.session.endpoint, fmt_dur(since.elapsed()), reason),
        );
    }
}

fn watch_stall() {
    let mut s = state();
    // A link alert covers a stall while the connection is down.
    if s.link_down_since.is_some() || s.stall_alerted {
        return;
    }
    let Some(t) = s.last_activity.filter(|t| t.elapsed() >= STALL_ALERT) else { return };
    s.stall_alerted = true;
    drop(s);
    let what = if BOT.get().is_some_and(|b| b.session.solo) { "node (not synced?)" } else { "pool" };
    alert(
        "⚠",
        format!("hashrate at 0 for {} outside inference pauses — workers stalled, or no work from the {}", fmt_dur(t.elapsed()), what),
    );
}

#[derive(Clone, Debug, Default, PartialEq)]
enum Link {
    #[default]
    Unknown,
    Up(Duration),
    Down(Duration, String),
}

#[derive(Clone, Debug, Default)]
struct Snapshot {
    rate_now: f64,
    hashes: u64,
    blocks: u64,
    blocks_rejected: u64,
    /// Pool shares: accepted, rejected (every reason), of which stale.
    shares: Option<(u64, u64, u64)>,
    ai_requests: u64,
    ai_answered: u64,
    ai_missed: u64,
    openings: u64,
    openings_failed: u64,
    claimed_outputs: u64,
    claimed_sompi: u64,
    pending_outputs: u64,
    pending_sompi: u64,
    balance: Option<u64>,
    balance_first: Option<u64>,
    balance_error: Option<String>,
    utxos: Option<u64>,
    network: Option<Network>,
    bond: Option<Bond>,
    link: Link,
}

impl Snapshot {
    fn take() -> Self {
        let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let shares = crate::client::stratum::share_stats().map(|s| {
            let stale = get(&s.stale);
            let rejected = stale
                + get(&s.low_diff)
                + get(&s.duplicate)
                + get(&s.invalid_opoi)
                + get(&s.invalid_pom)
                + get(&s.rejected_other);
            (get(&s.accepted), rejected, stale)
        });
        let s = state();
        let link = match (s.link_up_since, s.link_down_since) {
            (_, Some(t)) => Link::Down(t.elapsed(), s.link_down_reason.clone()),
            (Some(t), None) => Link::Up(t.elapsed()),
            (None, None) => Link::Unknown,
        };
        let rate = f64::from_bits(crate::miner::LAST_HASHRATE.load(Ordering::Relaxed));
        Snapshot {
            rate_now: if rate.is_finite() { rate } else { 0.0 },
            hashes: get(&C.hashes),
            blocks: get(&C.blocks),
            blocks_rejected: get(&C.blocks_rejected),
            shares,
            ai_requests: get(&C.ai_requests),
            ai_answered: get(&C.ai_answered),
            ai_missed: get(&C.ai_missed),
            openings: get(&C.openings),
            openings_failed: get(&C.openings_failed),
            claimed_outputs: get(&C.claimed_outputs),
            claimed_sompi: get(&C.claimed_sompi),
            pending_outputs: get(&C.pending_outputs),
            pending_sompi: get(&C.pending_sompi),
            balance: s.balance,
            balance_first: s.balance_first,
            balance_error: s.balance_error.clone(),
            utxos: s.utxos,
            network: s.network.clone(),
            bond: s.bond.clone(),
            link,
        }
    }
}

/// The `/stats` text: `now` against `prev`, taken `window` earlier.
fn render_summary(name: &str, solo: bool, now: &Snapshot, prev: &Snapshot, window: Duration, uptime: Duration) -> String {
    let w = fmt_dur(window);
    let mut out = vec![format!("⛏ {} · {} · up {}", name, if solo { "solo" } else { "pool" }, fmt_dur(uptime))];
    let avg = now.hashes.saturating_sub(prev.hashes) as f64 / window.as_secs_f64().max(1.0);
    out.push(format!("Hashrate: {} now · {} avg over {}", fmt_rate(now.rate_now), fmt_rate(avg), w));
    if let Some(net) = &now.network {
        let mine = if avg > 0.0 { avg } else { now.rate_now };
        out.push(format!(
            "Network: {} · this rig {} · reward {:.2} KRX/block · DAA {}",
            fmt_rate(net.hashrate),
            fmt_pct(if net.hashrate > 0.0 { 100.0 * mine / net.hashrate } else { 0.0 }),
            net.reward_krx,
            fmt_count(net.daa)
        ));
    }
    if solo {
        out.push(format!(
            "Blocks: {} found (+{} in {}) · {} rejected",
            fmt_count(now.blocks),
            fmt_count(now.blocks.saturating_sub(prev.blocks)),
            w,
            fmt_count(now.blocks_rejected)
        ));
    }
    if let Some((acc, rej, stale)) = now.shares {
        let prev_acc = prev.shares.map_or(0, |s| s.0);
        out.push(format!(
            "Shares: {} accepted (+{} in {}) · {} rejected ({} stale)",
            fmt_count(acc),
            fmt_count(acc.saturating_sub(prev_acc)),
            w,
            fmt_count(rej),
            fmt_count(stale)
        ));
    }
    out.push(match (now.balance, &now.balance_error) {
        (Some(b), err) => {
            let mut line = format!("Balance: {} KRX", fmt_krx(b as i128));
            let mut deltas = Vec::new();
            if let Some(p) = prev.balance.filter(|_| window >= Duration::from_secs(60)) {
                deltas.push(format!("{} in {}", fmt_signed_krx(b as i128 - p as i128), w));
            }
            if let Some(f) = now.balance_first {
                deltas.push(format!("{} since start", fmt_signed_krx(b as i128 - f as i128)));
            }
            if !deltas.is_empty() {
                line.push_str(&format!(" ({})", deltas.join(", ")));
            }
            if let Some(n) = now.utxos {
                line.push_str(&format!(" · {} UTXOs", fmt_count(n)));
            }
            if let Some(e) = err {
                line.push_str(&format!(" · last read failed: {}", e));
            }
            line
        }
        (None, Some(e)) => format!("Balance: n/a — {}", e),
        (None, None) => "Balance: not read yet".to_string(),
    });
    if solo || now.claimed_outputs > 0 || now.pending_outputs > 0 {
        out.push(format!(
            "Escrow: {} KRX claimed ({} outputs) · {} KRX pending ({} outputs)",
            fmt_krx(now.claimed_sompi as i128),
            fmt_count(now.claimed_outputs),
            fmt_krx(now.pending_sompi as i128),
            fmt_count(now.pending_outputs)
        ));
    }
    let mut ai = format!(
        "AI requests: {} (+{} in {}) · {} answered · {} missed",
        fmt_count(now.ai_requests),
        fmt_count(now.ai_requests.saturating_sub(prev.ai_requests)),
        w,
        fmt_count(now.ai_answered),
        fmt_count(now.ai_missed)
    );
    if now.openings + now.openings_failed > 0 {
        ai.push_str(&format!(" · {} openings ({} failed)", fmt_count(now.openings), fmt_count(now.openings_failed)));
    }
    out.push(ai);
    if solo {
        out.push(match &now.bond {
            Some(b) if b.status == "clear" => format!("Service bond: clear · {} lifetime strikes", b.lifetime),
            Some(b) => format!("⚠ Service bond: {} · {} active, {} lifetime strikes", b.status, b.active, b.lifetime),
            None => "Service bond: not read yet".to_string(),
        });
    }
    let what = if solo { "Node" } else { "Pool" };
    out.push(match &now.link {
        Link::Up(d) => format!("{}: connected for {}", what, fmt_dur(*d)),
        Link::Down(d, why) => format!("⚠ {}: DOWN for {} — {}", what, fmt_dur(*d), why),
        Link::Unknown => format!("{}: not connected yet", what),
    });
    out.join("\n")
}

/// The AI message: what was asked, and what the miner answered.
fn render_ai(name: &str, label: &str, model: &str, prompt: &str, answer: Result<&str, &str>, secs: Option<f64>) -> String {
    let mut head = format!("💬 {} · {}", name, label);
    if !model.is_empty() {
        head.push_str(&format!(" · {}", model));
    }
    if let Some(s) = secs {
        head.push_str(&format!(" · {:.1} s", s));
    }
    let answer = match answer {
        Ok(text) if text.trim().is_empty() => "(empty answer)".to_string(),
        Ok(text) => clip_to(text.trim(), AI_ANSWER_MAX),
        Err(why) => format!("({})", why),
    };
    format!("{}\n\nQ: {}\n\nA: {}", head, clip_to(prompt.trim(), AI_PROMPT_MAX), answer)
}

enum Failure {
    /// Nothing will get through until the setup is fixed: notifications stop.
    Fatal(String),
    /// This message is lost; later ones may still go through.
    Dropped(String),
}

fn run_sender(cfg: Config, chat: String, rx: Receiver<Msg>) {
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(20)).try_proxy_from_env(true).build();
    let url = format!("{}/bot{}/sendMessage", cfg.api, cfg.token);
    let mut off = false;
    for msg in rx {
        if !off {
            match deliver(&agent, &url, &chat, &msg.text, msg.silent, &cfg.token) {
                Ok(()) => {}
                Err(Failure::Fatal(why)) => {
                    error!("Telegram: {} — notifications off until the miner restarts", why);
                    off = true;
                }
                Err(Failure::Dropped(why)) => warn!("Telegram: message dropped: {}", why),
            }
        }
        IN_FLIGHT.fetch_sub(1, Ordering::AcqRel);
    }
}

fn deliver(agent: &ureq::Agent, url: &str, chat: &str, text: &str, silent: bool, token: &str) -> Result<(), Failure> {
    let body = json!({ "chat_id": chat, "text": text, "disable_web_page_preview": true, "disable_notification": silent });
    let mut last = String::new();
    for attempt in 1..=SEND_ATTEMPTS {
        let wait = match agent.post(url).send_json(&body) {
            Ok(_) => return Ok(()),
            Err(ureq::Error::Status(code, resp)) => {
                let v: Value = resp.into_json().unwrap_or(Value::Null);
                let desc = v["description"].as_str().unwrap_or("").to_string();
                last = mask(&format!("HTTP {} {}", code, desc), token);
                match code {
                    401 | 404 => return Err(Failure::Fatal(format!("the bot token was rejected ({})", last))),
                    400 if desc.contains("chat not found") => {
                        return Err(Failure::Fatal(format!("chat {} not found — check chat in {} ({})", chat, conf_path(), last)))
                    }
                    403 => {
                        return Err(Failure::Fatal(format!("the bot may not write to chat {} — send it /start first ({})", chat, last)))
                    }
                    // A message the API refuses: retrying it will not help.
                    400 => return Err(Failure::Dropped(last)),
                    429 => Duration::from_secs(v["parameters"]["retry_after"].as_u64().unwrap_or(5).clamp(1, 60)),
                    _ => retry_delay(attempt),
                }
            }
            // ureq quotes the request URL in transport errors, and the URL holds the token.
            Err(e) => {
                last = mask(&e.to_string(), token);
                retry_delay(attempt)
            }
        };
        if attempt < SEND_ATTEMPTS {
            std::thread::sleep(wait);
        }
    }
    Err(Failure::Dropped(format!("{} after {} attempts", last, SEND_ATTEMPTS)))
}

fn retry_delay(attempt: u32) -> Duration {
    Duration::from_secs(5 * 3u64.pow(attempt.saturating_sub(1)))
}

fn mask(s: &str, token: &str) -> String {
    if token.is_empty() {
        s.to_string()
    } else {
        s.replace(token, "<token>")
    }
}

/// Logs the chats that wrote to the bot, so the user can pick theirs for the `chat` setting.
fn discover_chats(cfg: &Config) {
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(20)).try_proxy_from_env(true).build();
    let url = format!("{}/bot{}/getUpdates", cfg.api, cfg.token);
    let v: Value = match agent.get(&url).call() {
        Ok(resp) => resp.into_json().unwrap_or(Value::Null),
        Err(ureq::Error::Status(code, resp)) => {
            let v: Value = resp.into_json().unwrap_or(Value::Null);
            warn!("Telegram: cannot list the bot's chats: HTTP {} {}", code, v["description"].as_str().unwrap_or(""));
            return;
        }
        Err(e) => {
            warn!("Telegram: cannot list the bot's chats: {}", mask(&e.to_string(), &cfg.token));
            return;
        }
    };
    let chats = chats_in_updates(&v);
    if chats.is_empty() {
        warn!("Telegram: no chat has written to the bot yet — send it any message (e.g. /start), then restart the miner");
    } else {
        let list = chats.iter().map(|(id, label)| format!("{} ({})", id, label)).collect::<Vec<_>>().join(", ");
        warn!("Telegram: chats that wrote to the bot: {} — put yours in {} as `chat = <id>` and restart the miner", list, conf_path());
    }
}

/// `(chat id, label)` of every chat in a getUpdates answer, first seen first.
fn chats_in_updates(v: &Value) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for update in v["result"].as_array().into_iter().flatten() {
        for key in ["message", "edited_message", "channel_post", "my_chat_member"] {
            let chat = &update[key]["chat"];
            let Some(id) = chat["id"].as_i64() else { continue };
            let id = id.to_string();
            if out.iter().any(|(seen, _)| *seen == id) {
                continue;
            }
            let label = chat["title"]
                .as_str()
                .map(str::to_string)
                .or_else(|| chat["username"].as_str().map(|u| format!("@{}", u)))
                .or_else(|| chat["first_name"].as_str().map(str::to_string))
                .unwrap_or_else(|| chat["type"].as_str().unwrap_or("chat").to_string());
            out.push((id, label));
        }
    }
    out
}

fn clip(s: &str) -> String {
    match s.char_indices().nth(REASON_MAX) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s.to_string(),
    }
}

/// `s` cut to `max` UTF-16 units (what Telegram counts: an emoji takes two), saying how many
/// characters were left out.
fn clip_to(s: &str, max: usize) -> String {
    let mut units = 0;
    for (i, c) in s.char_indices() {
        units += c.len_utf16();
        if units > max {
            return format!("{}… (+{} chars)", &s[..i], s[i..].chars().count());
        }
    }
    s.to_string()
}

fn short_address(a: &str) -> String {
    let (prefix, body) = a.split_once(':').unwrap_or(("", a));
    let chars: Vec<char> = body.chars().collect();
    if chars.len() <= 12 {
        return a.to_string();
    }
    let head: String = chars[..5].iter().collect();
    let tail: String = chars[chars.len() - 5..].iter().collect();
    if prefix.is_empty() {
        format!("{}…{}", head, tail)
    } else {
        format!("{}:{}…{}", prefix, head, tail)
    }
}

fn fmt_dur(d: Duration) -> String {
    let s = d.as_secs();
    match s {
        0..=59 => format!("{}s", s),
        60..=3599 => format!("{}m", s / 60),
        3600..=86_399 => format!("{}h{:02}m", s / 3600, s % 3600 / 60),
        _ => format!("{}d{}h", s / 86_400, s % 86_400 / 3600),
    }
}

fn fmt_rate(h: f64) -> String {
    const UNITS: [&str; 5] = ["H/s", "kH/s", "MH/s", "GH/s", "TH/s"];
    let mut v = h.max(0.0);
    let mut i = 0;
    while v >= 1000.0 && i < UNITS.len() - 1 {
        v /= 1000.0;
        i += 1;
    }
    format!("{:.2} {}", v, UNITS[i])
}

fn group(n: u128) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn fmt_count(n: u64) -> String {
    group(n as u128)
}

/// KRX from sompi: 2 decimals from 1 KRX up, 4 below, truncated.
fn fmt_krx(sompi: i128) -> String {
    let sign = if sompi < 0 { "-" } else { "" };
    let a = sompi.unsigned_abs();
    let whole = a / 100_000_000;
    let frac = a % 100_000_000;
    if whole >= 1 {
        format!("{}{}.{:02}", sign, group(whole), frac / 1_000_000)
    } else {
        format!("{}0.{:04}", sign, frac / 10_000)
    }
}

/// A small share stays readable: 0.27%, 0.0042%.
fn fmt_pct(p: f64) -> String {
    match p {
        p if p >= 1.0 => format!("{:.1}%", p),
        p if p >= 0.01 => format!("{:.2}%", p),
        p => format!("{:.4}%", p),
    }
}

fn fmt_signed_krx(sompi: i128) -> String {
    if sompi >= 0 {
        format!("+{}", fmt_krx(sompi))
    } else {
        fmt_krx(sompi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;

    #[test]
    fn ai_message() {
        let text = render_ai("RIG", "pool chat 7", "GLM-4-9B", "  What is 2+2?\n", Ok(" 4 "), Some(3.27));
        assert_eq!(text, "💬 RIG · pool chat 7 · GLM-4-9B · 3.3 s\n\nQ: What is 2+2?\n\nA: 4");
        let skipped = render_ai("RIG", "solo request ab12", "", "hi", Err("not answered: solo audits past the attention gate"), None);
        assert_eq!(skipped, "💬 RIG · solo request ab12\n\nQ: hi\n\nA: (not answered: solo audits past the attention gate)");
        assert!(render_ai("RIG", "x", "m", "q", Ok("  "), None).ends_with("A: (empty answer)"));
        let long = render_ai("RIG", "pool ai_request 0123456789abcdef", "Qwen3.8-27B", &"q".repeat(10_000), Ok(&"a".repeat(10_000)), Some(1.0));
        assert!(long.contains(&format!("{}… (+{} chars)", "q".repeat(AI_PROMPT_MAX), 10_000 - AI_PROMPT_MAX)));
        assert!(long.contains(&format!("{}… (+{} chars)", "a".repeat(AI_ANSWER_MAX), 10_000 - AI_ANSWER_MAX)));
        assert!(long.encode_utf16().count() < 4096, "{} UTF-16 units", long.encode_utf16().count());
        // Emoji take two UTF-16 units each: the budget holds for them too.
        let emoji = render_ai("RIG", "pool chat 7", "GLM-4-9B", &"😀".repeat(5_000), Ok(&"🚀".repeat(5_000)), Some(1.0));
        assert!(emoji.encode_utf16().count() < 4096, "{} UTF-16 units", emoji.encode_utf16().count());
        assert!(emoji.contains(&format!("{}… (+{} chars)", "😀".repeat(AI_PROMPT_MAX / 2), 5_000 - AI_PROMPT_MAX / 2)));
    }

    fn cfg(text: &str) -> Option<Config> {
        read_config(&parse_settings(text).unwrap(), Some("PC".into()))
    }

    #[test]
    fn settings_file() {
        let c = cfg("\u{feff}# Telegram\ntoken = 123:abc\nchat = \" 42 \"\nNAME = rig   # inline comment\n  ; another comment\napi = http://127.0.0.1:9/\n\n").unwrap();
        assert_eq!(c.token, "123:abc");
        assert_eq!(c.chat.as_deref(), Some("42"));
        assert_eq!(c.name, "rig");
        assert_eq!(c.api, "http://127.0.0.1:9");
        assert_eq!(c.ai, AiMode::Off);
        let c = cfg("token = t\nchat =\n").unwrap();
        assert_eq!(c.chat, None);
        assert_eq!(c.name, "PC");
        assert_eq!(c.api, DEFAULT_API);
        assert_eq!(read_config(&parse_settings("token = t").unwrap(), None).unwrap().name, "miner");
        assert_eq!(cfg("chat = 1"), None);
        assert_eq!(cfg("token =   \nchat = 1"), None);
        assert_eq!(cfg("# token = t"), None);
        assert_eq!(parse_settings("token 123:abc").unwrap_err(), "line 1: expected `key = value`");
    }

    #[test]
    fn settings_file_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("telegram.conf");
        assert_eq!(load_config(path.to_str().unwrap()), None, "a missing file turns notifications off");
        std::fs::write(&path, "token = 1:x\nchat = -1001\nai = all\n").unwrap();
        let c = load_config(path.to_str().unwrap()).unwrap();
        assert_eq!((c.token.as_str(), c.chat.as_deref(), c.ai), ("1:x", Some("-1001"), AiMode::All));
        std::fs::write(&path, "token: 1:x\n").unwrap();
        assert_eq!(load_config(path.to_str().unwrap()), None, "a malformed file turns notifications off");
    }

    #[test]
    fn ai_setting() {
        let ai = |v: &str| cfg(&format!("token = t\nai = {}", v)).unwrap().ai;
        assert_eq!(ai("0"), AiMode::Off);
        assert_eq!(ai("1"), AiMode::Public);
        assert_eq!(ai("ALL"), AiMode::All);
        assert_eq!(ai("yes"), AiMode::Off);
        assert!(ai_allowed(AiMode::Public, "pool ai_request 0123456789abcdef"));
        assert!(ai_allowed(AiMode::Public, "pool chat 7"));
        assert!(ai_allowed(AiMode::Public, "solo request ab12"));
        assert!(!ai_allowed(AiMode::Public, "solo request ab12, private"));
        assert!(ai_allowed(AiMode::All, "solo request ab12, private"));
        assert!(!ai_allowed(AiMode::Off, "pool chat 7"));
    }

    #[test]
    fn commands() {
        let msg = |chat: i64, text: &str| json!({"update_id": 5, "message": {"chat": {"id": chat}, "text": text}});
        assert_eq!(incoming(&msg(42, "/stats"), "42"), Incoming::Command(Command::Stats));
        assert_eq!(incoming(&msg(42, "/STATS@KeryxRigBot now"), "42"), Incoming::Command(Command::Stats));
        assert_eq!(incoming(&msg(42, "/status"), "42"), Incoming::Command(Command::Stats));
        assert_eq!(incoming(&msg(42, "/start"), "42"), Incoming::Command(Command::Help));
        assert_eq!(incoming(&msg(42, "/whatever"), "42"), Incoming::Command(Command::Help));
        assert_eq!(incoming(&msg(42, "hello"), "42"), Incoming::Other);
        assert_eq!(incoming(&msg(7, "/stats"), "42"), Incoming::Stranger("7".into()));
        assert_eq!(incoming(&msg(-1001, "/stats"), "-1001"), Incoming::Command(Command::Stats));
        assert_eq!(incoming(&json!({"update_id": 6, "my_chat_member": {}}), "42"), Incoming::Other);
        let updates = vec![json!({"update_id": 10}), json!({"update_id": 12}), json!({"update_id": 11})];
        assert_eq!(next_offset(&updates, 0), 13);
        assert_eq!(next_offset(&[], 9), 9);
        assert!(help_text("RIG").contains("/stats"));
    }

    #[test]
    fn polls_for_updates() {
        let (api, server) = fake_api(vec![
            (200, r#"{"ok":true,"result":[{"update_id":7,"message":{"chat":{"id":42},"text":"/stats"}}]}"#),
            (409, r#"{"ok":false,"description":"Conflict: terminated by other getUpdates request"}"#),
            (401, r#"{"ok":false,"description":"Unauthorized"}"#),
        ]);
        let url = format!("{}/botT/getUpdates", api);
        let updates = get_updates(&agent(), &url, 3, 0, "T").ok().unwrap();
        assert_eq!(next_offset(&updates, 3), 8);
        assert_eq!(incoming(&updates[0], "42"), Incoming::Command(Command::Stats));
        assert!(matches!(get_updates(&agent(), &url, 8, 0, "T"), Err(Failure::Dropped(m)) if m.contains("its own bot")));
        assert!(matches!(get_updates(&agent(), &url, 8, 0, "T"), Err(Failure::Fatal(m)) if m.contains("token was rejected")));
        let seen = server.join().unwrap();
        assert!(seen[0].0.starts_with("GET /botT/getUpdates?offset=3&timeout=0&allowed_updates="), "{}", seen[0].0);
    }

    #[test]
    fn formats() {
        assert_eq!(fmt_dur(Duration::from_secs(42)), "42s");
        assert_eq!(fmt_dur(Duration::from_secs(60 * 60)), "1h00m");
        assert_eq!(fmt_dur(Duration::from_secs(5 * 3600 + 12 * 60 + 9)), "5h12m");
        assert_eq!(fmt_dur(Duration::from_secs(2 * 86_400 + 3 * 3600)), "2d3h");
        assert_eq!(fmt_rate(1_612_345.0), "1.61 MH/s");
        assert_eq!(fmt_rate(0.0), "0.00 H/s");
        assert_eq!(group(1_234_567), "1,234,567");
        assert_eq!(group(999), "999");
        assert_eq!(fmt_krx(123_456_789_012), "1,234.56");
        assert_eq!(fmt_krx(1_234_567), "0.0123");
        assert_eq!(fmt_signed_krx(1_240_000_000), "+12.40");
        assert_eq!(fmt_signed_krx(-50_000_000), "-0.5000");
        assert_eq!(fmt_signed_krx(0), "+0.0000");
        assert_eq!(
            short_address("keryx:qz4zajm2kecfy5c6qctepdnt4666yz0qhff5alnhyy6jjzfm94svxylck57rh"),
            "keryx:qz4za…k57rh"
        );
        assert_eq!(short_address("keryx:short"), "keryx:short");
        assert_eq!(clip(&"x".repeat(300)).chars().count(), REASON_MAX + 1);
        assert_eq!(mask("https://h/bot123:abc/sendMessage: Dns Failed", "123:abc"), "https://h/bot<token>/sendMessage: Dns Failed");
    }

    #[test]
    fn summary_solo() {
        let prev = Snapshot { hashes: 0, blocks: 2, ai_requests: 3, balance: Some(123_216_000_000), ..Default::default() };
        let now = Snapshot {
            rate_now: 1_610_000.0,
            hashes: 1_580_000 * 3600,
            blocks: 3,
            ai_requests: 7,
            ai_missed: 7,
            claimed_sompi: 410_000_000,
            claimed_outputs: 2,
            pending_sompi: 230_000_000,
            pending_outputs: 7,
            balance: Some(123_456_000_000),
            balance_first: Some(122_216_000_000),
            utxos: Some(192),
            network: Some(Network { hashrate: 591_008_278.41, reward_krx: 5.023_847_74, daa: 141_032_821 }),
            bond: Some(Bond { status: "strike 2 (last at daa 140200000)".into(), active: 2, lifetime: 3 }),
            link: Link::Up(Duration::from_secs(3 * 3600)),
            ..Default::default()
        };
        let text = render_summary("RIG", true, &now, &prev, Duration::from_secs(3600), Duration::from_secs(5 * 3600 + 12 * 60));
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "⛏ RIG · solo · up 5h12m");
        assert_eq!(lines[1], "Hashrate: 1.61 MH/s now · 1.58 MH/s avg over 1h00m");
        assert_eq!(lines[2], "Network: 591.01 MH/s · this rig 0.27% · reward 5.02 KRX/block · DAA 141,032,821");
        assert_eq!(lines[3], "Blocks: 3 found (+1 in 1h00m) · 0 rejected");
        assert_eq!(lines[4], "Balance: 1,234.56 KRX (+2.40 in 1h00m, +12.40 since start) · 192 UTXOs");
        assert_eq!(lines[5], "Escrow: 4.10 KRX claimed (2 outputs) · 2.30 KRX pending (7 outputs)");
        assert_eq!(lines[6], "AI requests: 7 (+4 in 1h00m) · 0 answered · 7 missed");
        assert_eq!(lines[7], "⚠ Service bond: strike 2 (last at daa 140200000) · 2 active, 3 lifetime strikes");
        assert_eq!(lines[8], "Node: connected for 3h00m");
        assert_eq!(lines.len(), 9);
    }

    #[test]
    fn summary_pool() {
        let prev = Snapshot { shares: Some((100, 1, 0)), ..Default::default() };
        let now = Snapshot {
            shares: Some((156, 3, 1)),
            ai_requests: 5,
            ai_answered: 4,
            ai_missed: 1,
            openings: 3,
            link: Link::Down(Duration::from_secs(180), "connection reset".into()),
            ..Default::default()
        };
        let text = render_summary("RIG", false, &now, &prev, Duration::from_secs(600), Duration::from_secs(600));
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "⛏ RIG · pool · up 10m");
        assert_eq!(lines[2], "Shares: 156 accepted (+56 in 10m) · 3 rejected (1 stale)");
        assert_eq!(lines[3], "Balance: not read yet");
        assert_eq!(lines[4], "AI requests: 5 (+5 in 10m) · 4 answered · 1 missed · 3 openings (0 failed)");
        assert_eq!(lines[5], "⚠ Pool: DOWN for 3m — connection reset");
        assert_eq!(lines.len(), 6, "no network, blocks, escrow or bond lines here: {text}");
        let none = Snapshot { balance_error: Some("HTTP 502 from info".into()), ..Default::default() };
        let solo = render_summary("RIG", true, &none, &Snapshot::default(), Duration::from_secs(600), Duration::from_secs(600));
        assert!(solo.contains("Balance: n/a — HTTP 502 from info"), "{solo}");
        let stale = Snapshot { balance: Some(100_000_000), balance_error: Some("Dns Failed".into()), ..Default::default() };
        let solo = render_summary("RIG", true, &stale, &Snapshot::default(), Duration::from_secs(30), Duration::from_secs(30));
        assert!(solo.contains("Balance: 1.00 KRX · last read failed: Dns Failed"), "{solo}");
        assert!(solo.contains("Service bond: not read yet"), "{solo}");
        assert!(solo.contains("Node: not connected yet"), "{solo}");
    }

    #[test]
    fn explorer_answers() {
        assert_eq!(url_component("keryx:qz4z-a_b.c~"), "keryx%3Aqz4z-a_b.c~");
        let info = json!({"block_reward_krx":5.02384774,"hashrate_hps":591008278.410111,"last_daa_score":141032821,"network":"keryx-mainnet"});
        assert_eq!(parse_network(&info), Some(Network { hashrate: 591008278.410111, reward_krx: 5.02384774, daa: 141032821 }));
        assert_eq!(parse_network(&json!({"hashrate_hps": 1.0})), None);
        assert_eq!(fmt_pct(0.2673), "0.27%");
        assert_eq!(fmt_pct(12.34), "12.3%");
        assert_eq!(fmt_pct(0.004_21), "0.0042%");

        let (api, server) = fake_api(vec![
            (200, r#"{"address":"keryx:abc","balance_sompi":1916900777939}"#),
            (200, r#"{"count":192}"#),
            (200, r#"{"block_reward_krx":5.02384774,"hashrate_hps":591008278.41,"last_daa_score":141032821}"#),
            (503, r#"{"error":"busy"}"#),
        ]);
        let e = Explorer::new(&format!("{}/", api), "keryx:abc");
        assert_eq!(e.balance(), Ok(1_916_900_777_939));
        assert_eq!(e.utxos(), Some(192));
        assert_eq!(e.network().map(|n| n.daa), Some(141_032_821));
        assert_eq!(e.balance(), Err("HTTP 503 from addresses/keryx%3Aabc/balance".to_string()));
        let seen: Vec<String> = server.join().unwrap().into_iter().map(|(line, _)| line).collect();
        assert_eq!(
            seen,
            [
                "GET /api/v1/addresses/keryx%3Aabc/balance HTTP/1.1",
                "GET /api/v1/addresses/keryx%3Aabc/utxos/count HTTP/1.1",
                "GET /api/v1/info HTTP/1.1",
                "GET /api/v1/addresses/keryx%3Aabc/balance HTTP/1.1",
            ]
        );
    }

    #[test]
    fn chats_from_updates() {
        let v = json!({"ok": true, "result": [
            {"update_id": 1, "message": {"chat": {"id": 42, "type": "private", "first_name": "Sam", "username": "sam"}}},
            {"update_id": 2, "message": {"chat": {"id": 42, "type": "private", "username": "sam"}}},
            {"update_id": 3, "channel_post": {"chat": {"id": -1001, "type": "channel", "title": "Rigs"}}},
            {"update_id": 4, "my_chat_member": {"chat": {"id": -7, "type": "group"}}}
        ]});
        assert_eq!(
            chats_in_updates(&v),
            vec![("42".into(), "@sam".into()), ("-1001".into(), "Rigs".into()), ("-7".into(), "group".into())]
        );
        assert!(chats_in_updates(&json!({"ok": false})).is_empty());
    }

    /// A one-shot HTTP server that answers each connection with the next canned response and
    /// returns the request bodies it saw.
    fn fake_api(responses: Vec<(u16, &'static str)>) -> (String, std::thread::JoinHandle<Vec<(String, String)>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for (code, body) in responses {
                let (stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream);
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                let mut len = 0usize;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap();
                    }
                }
                let mut buf = vec![0u8; len];
                reader.read_exact(&mut buf).unwrap();
                seen.push((request_line.trim().to_string(), String::from_utf8(buf).unwrap()));
                let reply = format!(
                    "HTTP/1.1 {} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    code,
                    body.len(),
                    body
                );
                reader.get_mut().write_all(reply.as_bytes()).unwrap();
            }
            seen
        });
        (url, handle)
    }

    fn agent() -> ureq::Agent {
        ureq::AgentBuilder::new().timeout(Duration::from_secs(10)).build()
    }

    #[test]
    fn delivers_a_message() {
        let (api, server) = fake_api(vec![(200, r#"{"ok":true,"result":{}}"#)]);
        let url = format!("{}/bot123:abc/sendMessage", api);
        assert!(deliver(&agent(), &url, "42", "hello ⛏", true, "123:abc").is_ok());
        let seen = server.join().unwrap();
        assert_eq!(seen[0].0, "POST /bot123:abc/sendMessage HTTP/1.1");
        let body: Value = serde_json::from_str(&seen[0].1).unwrap();
        assert_eq!(body["chat_id"], "42");
        assert_eq!(body["text"], "hello ⛏");
        assert_eq!(body["disable_notification"], true);
    }

    #[test]
    fn retries_after_rate_limit() {
        let (api, server) = fake_api(vec![
            (429, r#"{"ok":false,"description":"Too Many Requests: retry after 1","parameters":{"retry_after":1}}"#),
            (200, r#"{"ok":true}"#),
        ]);
        let url = format!("{}/botT/sendMessage", api);
        assert!(deliver(&agent(), &url, "42", "x", false, "T").is_ok());
        assert_eq!(server.join().unwrap().len(), 2);
    }

    #[test]
    fn bad_token_and_chat_are_fatal() {
        let (api, server) = fake_api(vec![
            (401, r#"{"ok":false,"description":"Unauthorized"}"#),
            (400, r#"{"ok":false,"description":"Bad Request: chat not found"}"#),
            (403, r#"{"ok":false,"description":"Forbidden: bot was blocked by the user"}"#),
            (400, r#"{"ok":false,"description":"Bad Request: message is too long"}"#),
        ]);
        let url = format!("{}/botT/sendMessage", api);
        assert!(matches!(deliver(&agent(), &url, "42", "x", false, "T"), Err(Failure::Fatal(m)) if m.contains("token was rejected")));
        assert!(matches!(deliver(&agent(), &url, "42", "x", false, "T"), Err(Failure::Fatal(m)) if m.contains("chat 42 not found")));
        assert!(matches!(deliver(&agent(), &url, "42", "x", false, "T"), Err(Failure::Fatal(m)) if m.contains("send it /start")));
        assert!(matches!(deliver(&agent(), &url, "42", "x", false, "T"), Err(Failure::Dropped(m)) if m.contains("too long")));
        server.join().unwrap();
    }

    #[test]
    fn transport_errors_never_show_the_token() {
        // Nothing listens on this port: the connect fails at once, every attempt.
        let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let url = format!("http://127.0.0.1:{}/bot999:SECRET/sendMessage", port);
        let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(2)).build();
        // Only the first attempt is timed here: a dropped message waits 5 + 15 s between tries.
        let Err(e) = agent.post(&url).send_json(json!({})) else { panic!("connected to a closed port") };
        let raw = e.to_string();
        assert!(raw.contains("999:SECRET"), "ureq should quote the URL: {raw}");
        assert!(!mask(&raw, "999:SECRET").contains("SECRET"));
    }
}
