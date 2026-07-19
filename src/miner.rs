use std::collections::HashMap;
use std::num::Wrapping;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::{Duration, Instant};

use crate::{pow, watch, Error};
use log::{error, info, warn};
use rand::{thread_rng, RngCore};
use tokio::sync::mpsc::Sender;

use crate::pow::BlockSeed;
use keryx_miner::{PluginManager, WorkerSpec};

type MinerHandler = std::thread::JoinHandle<Result<(), Error>>;

/// Grace period a worker gets to observe `Close` and return on its own before the freeze
/// handler force-kills it. It MUST exceed the worst-case *uninterruptible* op — a one-time GPU
/// VRAM blob load, which is a synchronous Vulkan upload that can take tens of seconds for a large
/// model and cannot poll `Close` mid-call. Too short and the watchdog kills a worker mid-load: on
/// Windows `kernel32::TerminateThread` destroys the thread without letting std write its result, so
/// the drop's `join()` then panics ("threads should not terminate unexpectedly") and crashes the
/// whole process on every reconnect. In steady state a worker checks `Close` between mining batches
/// and exits in well under a second, so this ceiling only ever applies mid-load — it adds no delay
/// to a normal reconnect. 60s covers any RDNA3-servable model's load with margin.
const FREEZE_GRACE: Duration = Duration::from_secs(60);

#[cfg(any(target_os = "linux", target_os = "macos"))]
extern "C-unwind" fn signal_panic(_signal: nix::libc::c_int) {
    // MUST be `extern "C-unwind"`: a plain `extern "C"` handler turns this panic into a
    // process-wide abort ("panic in a function that cannot unwind") — the OPoI shutdown
    // crash-loop. Unwinding lets a genuinely stuck worker's join() return instead. This
    // is a last resort; the cooperative Close checks below normally let workers exit
    // before this handler ever fires.
    panic!("Forced shutdown");
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn register_freeze_handler() {
    // nix's typed SigHandler only accepts `extern "C" fn`, which would reintroduce the
    // abort. Register through libc with a transmute instead: the C and C-unwind ABIs
    // share an identical calling convention (this is ABI-sound), and unwind behavior
    // follows the handler's own `extern "C-unwind"` definition.
    unsafe {
        let handler: nix::libc::sighandler_t =
            std::mem::transmute(signal_panic as extern "C-unwind" fn(nix::libc::c_int));
        let _ = nix::libc::signal(nix::libc::SIGUSR1, handler);
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn trigger_freeze_handler(kill_switch: Arc<AtomicBool>, handle: &MinerHandler) -> std::thread::JoinHandle<()> {
    use std::os::unix::thread::JoinHandleExt;
    let pthread_handle = handle.as_pthread_t();
    std::thread::spawn(move || {
        sleep(FREEZE_GRACE);
        if kill_switch.load(Ordering::SeqCst) {
            match nix::sys::pthread::pthread_kill(pthread_handle, nix::sys::signal::Signal::SIGUSR1) {
                Ok(()) => {
                    info!("Thread killed successfully")
                }
                Err(e) => {
                    info!("Error: {:?}", e)
                }
            }
        }
    })
}

#[cfg(any(target_os = "windows"))]
struct RawHandle(*mut std::ffi::c_void);

#[cfg(any(target_os = "windows"))]
unsafe impl Send for RawHandle {}

#[cfg(any(target_os = "windows"))]
fn register_freeze_handler() {}

#[cfg(target_os = "windows")]
fn trigger_freeze_handler(kill_switch: Arc<AtomicBool>, handle: &MinerHandler) -> std::thread::JoinHandle<()> {
    use std::os::windows::io::AsRawHandle;
    let raw_handle = RawHandle(handle.as_raw_handle());

    std::thread::spawn(move || unsafe {
        let ensure_full_move = raw_handle;
        sleep(FREEZE_GRACE);
        if kill_switch.load(Ordering::SeqCst) {
            kernel32::TerminateThread(ensure_full_move.0, 0);
        }
    })
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn trigger_freeze_handler(kill_switch: Arc<AtomicBool>, handle: &MinerHandler) {
    warn!("Freeze handler is not implemented. Frozen threads are ignored");
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn register_freeze_handler() {
    warn!("Freeze handler is not implemented. Frozen threads are ignored");
}

#[derive(Clone)]
enum WorkerCommand {
    Job(Box<pow::State>),
    Close,
}

#[allow(dead_code)]
pub struct MinerManager {
    handles: Vec<MinerHandler>,
    block_channel: watch::Sender<Option<WorkerCommand>>,
    send_channel: Sender<BlockSeed>,
    logger_stop: Arc<AtomicBool>,
    is_synced: bool,
    hashes_tried: Arc<AtomicU64>,
    hashes_by_worker: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
    current_state_id: AtomicUsize,
    opoi_challenge_active: Arc<AtomicBool>,
}

impl Drop for MinerManager {
    fn drop(&mut self) {
        info!("Closing miner");
        // Signal the detached hashrate logger to exit on its next wake (it polls this flag). We
        // don't join it — that would block shutdown up to LOG_RATE.
        self.logger_stop.store(true, Ordering::Release);
        match self.block_channel.send(Some(WorkerCommand::Close)) {
            Ok(_) => {}
            Err(_) => warn!("All workers are already dead"),
        }
        while !self.handles.is_empty() {
            let handle = self.handles.pop().expect("There should be at least one");
            let kill_switch = Arc::new(AtomicBool::new(true));
            trigger_freeze_handler(kill_switch.clone(), &handle);
            match handle.join() {
                Ok(res) => match res {
                    Ok(()) => {}
                    Err(e) => error!("Error when closing Worker: {}", e),
                },
                Err(_) => error!("Worker failed to close gracefully"),
            };
            kill_switch.fetch_and(false, Ordering::SeqCst);
        }
    }
}

pub fn get_num_cpus(n_cpus: Option<u16>) -> u16 {
    n_cpus.unwrap_or_else(|| {
        num_cpus::get_physical().try_into().expect("Doesn't make sense to have more than 65,536 CPU cores")
    })
}

const LOG_RATE: Duration = Duration::from_secs(10);
// Number of consecutive all-zero hashrate ticks (outside an OPoI inference pause)
// tolerated before reporting a real stall. A brief run of zeros is normal — model
// load/eviction or a gap between block templates — so we wait past this grace window
// to avoid scary "stalled or crashed" warnings during routine operation.
const STALL_GRACE_TICKS: u32 = 3;

impl MinerManager {
    pub fn new(send_channel: Sender<BlockSeed>, n_cpus: Option<u16>, manager: &PluginManager) -> Self {
        register_freeze_handler();
        let hashes_tried = Arc::new(AtomicU64::new(0));
        let hashes_by_worker = Arc::new(Mutex::new(HashMap::<String, Arc<AtomicU64>>::new()));
        let opoi_challenge_active = Arc::new(AtomicBool::new(false));
        let (send, recv) = watch::channel(None);
        let mut handles =
            Self::launch_cpu_threads(send_channel.clone(), Arc::clone(&hashes_tried), recv.clone(), n_cpus)
                .collect::<Vec<MinerHandler>>();
        if manager.has_specs() {
            handles.append(&mut Self::launch_gpu_threads(
                send_channel.clone(),
                Arc::clone(&hashes_tried),
                recv,
                manager,
                hashes_by_worker.clone(),
                Arc::clone(&opoi_challenge_active),
            ));
        }
        let logger_stop = Arc::new(AtomicBool::new(false));
        let logger_stop_spawn = Arc::clone(&logger_stop);
        // Clone the counters the logger reads BEFORE the move-closure, so the originals stay
        // available for the struct fields below. The hashrate logger runs on a dedicated std::thread
        // (not a tokio task) so it never occupies one of the few async workers; it is detached and
        // exits on `logger_stop` (set in Drop) — no join (that would block shutdown up to LOG_RATE).
        let logger_hashes = Arc::clone(&hashes_tried);
        let logger_by_worker = hashes_by_worker.clone();
        let logger_challenge = Arc::clone(&opoi_challenge_active);
        std::thread::spawn(move || {
            Self::log_hashrate(logger_hashes, logger_by_worker, logger_challenge, logger_stop_spawn)
        });
        Self {
            handles,
            block_channel: send,
            send_channel,
            logger_stop,
            is_synced: true,
            hashes_tried,
            current_state_id: AtomicUsize::new(0),
            hashes_by_worker,
            opoi_challenge_active,
        }
    }

    fn launch_cpu_threads(
        send_channel: Sender<BlockSeed>,
        hashes_tried: Arc<AtomicU64>,
        work_channel: watch::Receiver<Option<WorkerCommand>>,
        n_cpus: Option<u16>,
    ) -> impl Iterator<Item = MinerHandler> {
        let n_cpus = get_num_cpus(n_cpus);
        info!("launching: {} cpu miners", n_cpus);
        (0..n_cpus)
            .map(move |_| Self::launch_cpu_miner(send_channel.clone(), work_channel.clone(), Arc::clone(&hashes_tried)))
    }

    fn launch_gpu_threads(
        send_channel: Sender<BlockSeed>,
        hashes_tried: Arc<AtomicU64>,
        work_channel: watch::Receiver<Option<WorkerCommand>>,
        manager: &PluginManager,
        hashes_by_worker: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
        opoi_challenge_active: Arc<AtomicBool>,
    ) -> Vec<MinerHandler> {
        let mut vec = Vec::<MinerHandler>::new();
        let specs = manager.build().unwrap();
        for spec in specs {
            let worker_hashes_tried = Arc::new(AtomicU64::new(0));
            hashes_by_worker.lock().unwrap().insert(spec.id(), worker_hashes_tried.clone());
            vec.push(Self::launch_gpu_miner(
                send_channel.clone(),
                work_channel.clone(),
                Arc::clone(&hashes_tried),
                spec,
                worker_hashes_tried,
                Arc::clone(&opoi_challenge_active),
            ));
        }
        vec
    }

    pub fn opoi_challenge_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.opoi_challenge_active)
    }

    pub async fn process_block(&mut self, block: Option<BlockSeed>) -> Result<(), Error> {
        let state = match block {
            Some(b) => {
                self.is_synced = true;
                let id = self.current_state_id.fetch_add(1, Ordering::SeqCst);
                Some(WorkerCommand::Job(Box::new(pow::State::new(id, b)?)))
            }
            None => {
                if !self.is_synced {
                    return Ok(());
                }
                self.is_synced = false;
                if self.opoi_challenge_active.load(Ordering::Relaxed) {
                    info!("OPoI challenge in progress — PoW template suspended, stand by");
                } else {
                    warn!("Keryxd is not synced, skipping current template");
                }
                None
            }
        };

        self.block_channel.send(state).map_err(|_e| "Failed sending block to threads")?;
        Ok(())
    }

    #[allow(unreachable_code)]
    fn launch_gpu_miner(
        send_channel: Sender<BlockSeed>,
        mut block_channel: watch::Receiver<Option<WorkerCommand>>,
        hashes_tried: Arc<AtomicU64>,
        spec: Box<dyn WorkerSpec>,
        worker_hashes_tried: Arc<AtomicU64>,
        opoi_challenge_active: Arc<AtomicBool>,
    ) -> MinerHandler {
        std::thread::spawn(move || {
            let mut box_ = spec.build();
            let gpu_work = box_.as_mut();
            (|| {
                info!("Spawned Thread for GPU {}", gpu_work.id());
                let mut nonces = vec![0u64; 1];

                let mut state = None;
                // PoM mining: nonce cursor + per-launch batch. The kernel grinds the whole batch
                // before returning, so BPS_max = hashrate / POM_BATCH. At 1<<22 this capped a
                // ~24 MH/s GPU at ~5.8 BPS. 1<<20 lifts the ceiling to ~23 BPS while staying well
                // above kernel-launch overhead (batch ≈ 43 ms at 24 MH/s).
                let mut pom_nonce: u64 = thread_rng().next_u64();
                const POM_BATCH: u64 = 1 << 20;
                // This worker's Vulkan device: keys its resident PoM blob, and decides whether it
                // shares the GPU with inference (only that worker pauses during OPoI challenges).
                let pom_device = gpu_work.device_index().unwrap_or(0);
                let on_inference_gpu = pom_device == keryx_vulkan::inference_device_index() as u32;

                loop {
                    nonces[0] = 0;
                    if state.is_none() {
                        state = match block_channel.wait_for_change() {
                            Ok(cmd) => match cmd {
                                Some(WorkerCommand::Job(s)) => Some(s),
                                Some(WorkerCommand::Close) => {return Ok(());}
                                None => None,
                            },
                            Err(e) => {
                                info!("{}: GPU thread crashed: {}", gpu_work.id(), e.to_string());
                                return Ok(());
                            }
                        };
                    }
                    // PoM possession mining (design A): when active, the walk runs on the GPU
                    // over the resident weights instead of kHeavyHash. On a winning nonce we build
                    // the proof (host) and submit; the legacy plugin path below is skipped.
                    if matches!(state.as_ref(), Some(s) if s.daa_score >= keryx_miner::pom::POM_ACTIVATION_DAA) {
                        // Inference owns ITS GPU while a challenge runs (it has priority). Pause the
                        // walk instead of dispatching: two Vulkan contexts sharing one compute queue
                        // would slow inference past its OPoI deadline, and contention can stretch a
                        // walk dispatch past the Windows TDR watchdog (~2 s) → DEVICE_LOST. When VRAM
                        // allows, the blob stays resident across the pause (see slm::ensure_loaded) so
                        // mining resumes with no reload; otherwise it was unloaded and the
                        // is_installed() check below rebuilds it once the challenge clears.
                        // Workers on OTHER devices keep mining — llama-server is pinned to the
                        // inference GPU, so they never contend with it.
                        if on_inference_gpu && opoi_challenge_active.load(Ordering::Relaxed) {
                            std::thread::sleep(Duration::from_millis(100));
                            if let Some(cmd) = block_channel.get_changed()? {
                                state = match cmd {
                                    Some(WorkerCommand::Job(ns)) => Some(ns),
                                    Some(WorkerCommand::Close) => return Ok(()),
                                    None => state,
                                };
                            }
                            continue;
                        }
                        let (pph, time, target_le, daa) = {
                            let s = state.as_ref().unwrap();
                            let mut pph = [0u8; 32];
                            pph.copy_from_slice(&s.pow_hash_header[0..32]);
                            let time = u64::from_le_bytes(s.pow_hash_header[32..40].try_into().unwrap());
                            (pph, time, s.target.to_le_bytes(), s.daa_score)
                        };
                        // An inference may have evicted the mining model (inference has priority).
                        // Rebuild the walk (reloads the model resident) before mining resumes.
                        if !keryx_miner::pom_gpu::is_installed(pom_device) {
                            // A resident-model reload is a multi-second blocking GPU op with no
                            // cooperative Close check inside it. If a shutdown/new job is already
                            // pending, act on it now instead of starting a reload that would
                            // outlive the shutdown grace window and get force-killed.
                            if let Some(cmd) = block_channel.get_changed()? {
                                match cmd {
                                    Some(WorkerCommand::Close) => return Ok(()),
                                    Some(WorkerCommand::Job(ns)) => { state = Some(ns); continue; }
                                    None => { state = None; continue; }
                                }
                            }
                            keryx_miner::pom_gpu::ensure_installed(daa, pom_device);
                        }
                        let h3 = daa >= keryx_miner::pom::POM_LEVEL_ACTIVATION_DAA;
                        let found =
                            keryx_miner::pom_gpu::mine(pom_device, &pph, time, &target_le, pom_nonce, POM_BATCH, h3);
                        pom_nonce = pom_nonce.wrapping_add(POM_BATCH);
                        hashes_tried.fetch_add(POM_BATCH, Ordering::AcqRel);
                        worker_hashes_tried.fetch_add(POM_BATCH, Ordering::AcqRel);
                        if let Some(nonce) = found {
                            // Emit the tier for THIS block's DAA (not the frozen build-time tier) so the
                            // index reindexing at H2 is applied at the exact boundary — else the node
                            // rejects the proof (BadWeightPath).
                            let built = state.as_ref().and_then(|s| {
                                let (idx, _) = keryx_miner::pom::active_index()?;
                                let tier = keryx_miner::pom_gpu::current_tier(s.daa_score)?;
                                s.generate_block_if_pom(nonce, idx, tier)
                            });
                            if let Some(block_seed) = built {
                                match send_channel.blocking_send(block_seed.clone()) {
                                    Ok(()) => block_seed.report_block(),
                                    Err(e) => error!("Failed submitting PoM block: ({})", e.to_string()),
                                }
                                if let BlockSeed::FullBlock(_) = block_seed {
                                    state = None;
                                }
                            }
                        } else if let Some(cmd) = block_channel.get_changed()? {
                            state = match cmd {
                                Some(WorkerCommand::Job(ns)) => Some(ns),
                                Some(WorkerCommand::Close) => return Ok(()),
                                None => state,
                            };
                        }
                        continue;
                    }

                    let state_ref = match &state {
                        Some(s) => {
                            s.load_to_gpu(gpu_work);
                            s
                        },
                        None => continue,
                    };
                    state_ref.pow_gpu(gpu_work);
                    if let Err(e) = gpu_work.sync() {
                        warn!("CUDA run ignored: {}", e);
                        continue
                    }

                    gpu_work.copy_output_to(&mut nonces)?;
                    // When PoM is active the GPU still runs kHeavyHash (3a is CPU-only); its
                    // solutions are NOT valid PoM blocks, so don't submit them. GPU PoM = 3b.
                    if nonces[0] != 0 && state_ref.daa_score < keryx_miner::pom::POM_ACTIVATION_DAA {
                        if let Some(block_seed) = state_ref.generate_block_if_pow(nonces[0]) {
                            match send_channel.blocking_send(block_seed.clone()) {
                                Ok(()) => block_seed.report_block(),
                                Err(e) => error!("Failed submitting block: ({})", e.to_string()),
                            };
                            if let BlockSeed::FullBlock(_) = block_seed {
                                state = None;
                            }
                            nonces[0] = 0;
                            hashes_tried.fetch_add(gpu_work.get_workload().try_into().unwrap(), Ordering::AcqRel);
                            worker_hashes_tried.fetch_add(gpu_work.get_workload().try_into().unwrap(), Ordering::AcqRel);
                            continue;
                        } else {
                            let hash = state_ref.calculate_pow(nonces[0]);
                            warn!("Something is wrong in GPU results! Got nonce {}, with hash real {:?}  (target: {}*2^196)", nonces[0], hash.0, state_ref.target.0[3]);
                            break;
                        }
                    }

                        /*
                        info!("Output should be: {:02X?}", state_ref.calculate_pow(nonces[0]).to_le_bytes());
                        info!("We got: {:02X?} (Nonces: {:02X?})", hashes[0], nonces[0].to_le_bytes());
                        assert!(state_ref.calculate_pow(nonces[0]).to_le_bytes() == hashes[0]);
                        */
                        /*
                        info!("Output should be: {}", state_ref.calculate_pow(nonces[nonces.len()-1]).0[3]);
                        info!("We got: {} (Nonces: {})", Uint256::from_le_bytes(hashes[nonces.len()-1]).0[3], nonces[nonces.len()-1]);
                        assert!(state_ref.calculate_pow(nonces[nonces.len()-1]).0[0] == Uint256::from_le_bytes(hashes[nonces.len()-1]).0[0]);
                         */
                        /*
                        if state_ref.calculate_pow(nonces[0]).0[0] != Uint256::from_le_bytes(hashes[0]).0[0] {
                            gpu_work.sync()?;
                            let mut nonce_vec = vec![nonces[0]; 1];
                            nonce_vec.append(&mut vec![0u64; gpu_work.workload-1]);
                            gpu_work.calculate_pow_hash(&state_ref.pow_hash_header, Some(&nonce_vec));
                            gpu_work.sync()?;
                            gpu_work.calculate_matrix_mul(&mut state_ref.matrix.clone().0.as_slice().as_dbuf().unwrap());
                            gpu_work.sync()?;
                            gpu_work.calculate_heavy_hash();
                            gpu_work.sync()?;
                            let mut hashes2  = vec![[0u8; 32]; out_size];
                            let mut nonces2= vec![0u64; out_size];
                            gpu_work.copy_output_to(&mut hashes2, &mut nonces2);
                            assert!(state_ref.calculate_pow(nonces[0]).to_le_bytes() == hashes2[0]);
                            assert!(nonces2[0] == nonces[0]);
                            assert!(hashes2 == hashes);
                            assert!(false);
                        }*/

                    hashes_tried.fetch_add(gpu_work.get_workload().try_into().unwrap(), Ordering::AcqRel);
                    worker_hashes_tried.fetch_add(gpu_work.get_workload().try_into().unwrap(), Ordering::AcqRel);

                    {
                        if let Some(new_cmd) = block_channel.get_changed()? {
                            state = match new_cmd {
                                Some(WorkerCommand::Job(s)) => Some(s),
                                Some(WorkerCommand::Close) => {return Ok(());}
                                None => None,
                            };
                        }
                    }
                }
                Ok(())
            })()
            .map_err(|e: Error| {
                error!("{}: GPU thread crashed: {}", gpu_work.id(), e.to_string());
                e
            })
        })
    }

    #[allow(unreachable_code)]
    fn launch_cpu_miner(
        send_channel: Sender<BlockSeed>,
        mut block_channel: watch::Receiver<Option<WorkerCommand>>,
        hashes_tried: Arc<AtomicU64>,
    ) -> MinerHandler {
        let mut nonce = Wrapping(thread_rng().next_u64());
        let mut mask = Wrapping(0);
        let mut fixed = Wrapping(0);
        std::thread::Builder::new()
            .name("cpu-miner".into())
            .stack_size(256 * 1024)
            .spawn(move || {
            (|| {
                let mut state = None;

                loop {
                    if state.is_none() {
                        state = match block_channel.wait_for_change() {
                            Ok(cmd) => match cmd {
                                Some(WorkerCommand::Job(s)) => Some(s),
                                Some(WorkerCommand::Close) => {
                                    return Ok(());
                                }
                                None => None,
                            },
                            Err(e) => {
                                info!("CPU thread crashed: {}", e.to_string());
                                return Ok(());
                            }
                        };
                        if let Some(s) = &state {
                            mask = Wrapping(s.nonce_mask);
                            fixed = Wrapping(s.nonce_fixed);
                        }
                    }
                    let state_ref = match state.as_mut() {
                        Some(s) => s,
                        None => continue,
                    };
                    nonce = (nonce & mask) | fixed;

                    // PoM possession path (CPU) once active; else legacy kHeavyHash. The proof tier is
                    // recomputed from this block's DAA (per-block, not the frozen build-time tier) so
                    // the H2 reindex is applied at the boundary — else the node rejects (BadWeightPath).
                    let found = if state_ref.daa_score >= keryx_miner::pom::POM_ACTIVATION_DAA {
                        keryx_miner::pom::active_index().and_then(|(idx, _)| {
                            let tier = keryx_miner::pom_gpu::current_tier(state_ref.daa_score)?;
                            state_ref.generate_block_if_pom(nonce.0, idx, tier)
                        })
                    } else {
                        state_ref.generate_block_if_pow(nonce.0)
                    };
                    if let Some(block_seed) = found {
                        match send_channel.blocking_send(block_seed.clone()) {
                            Ok(()) => block_seed.report_block(),
                            Err(e) => error!("Failed submitting block: ({})", e.to_string()),
                        };
                        if let BlockSeed::FullBlock(_) = block_seed {
                            state = None;
                        }
                    }
                    nonce += Wrapping(1);
                    // TODO: Is this really necessary? can we just use Relaxed?
                    hashes_tried.fetch_add(1, Ordering::AcqRel);

                    if nonce.0 % 128 == 0 {
                        if let Some(new_cmd) = block_channel.get_changed()? {
                            state = match new_cmd {
                                Some(WorkerCommand::Job(s)) => Some(s),
                                Some(WorkerCommand::Close) => {
                                    return Ok(());
                                }
                                None => None,
                            };
                        }
                    }
                }
                Ok(())
            })()
            .map_err(|e: Error| {
                error!("CPU thread crashed: {}", e.to_string());
                e
            })
        }).expect("failed to spawn cpu-miner thread")
    }

    fn log_hashrate(
        hashes_tried: Arc<AtomicU64>,
        hashes_by_worker: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
        opoi_challenge_active: Arc<AtomicBool>,
        stop: Arc<AtomicBool>,
    ) {
        let mut last_instant = Instant::now();
        // Consecutive all-zero ticks while NOT in an OPoI inference pause.
        let mut zero_streak: u32 = 0;
        while !stop.load(Ordering::Acquire) {
            sleep(LOG_RATE);
            if stop.load(Ordering::Acquire) {
                break;
            }
            let duration = last_instant.elapsed().as_secs_f64();
            last_instant = Instant::now();
            // PoM model (re)load also intentionally pauses PoW — treat it like an inference pause.
            let challenge_active = opoi_challenge_active.load(Ordering::Relaxed)
                || keryx_miner::pom_gpu::is_loading();
            let total = hashes_tried.swap(0, Ordering::AcqRel);

            if total > 0 {
                // Mining normally: report aggregate + per-device rates.
                zero_streak = 0;
                let (rate, suffix) = Self::hash_suffix(total as f64 / duration);
                info!("Current hashrate is {:.2} {}", rate, suffix);
                for (device, counter) in &*hashes_by_worker.lock().unwrap() {
                    let h = counter.swap(0, Ordering::AcqRel);
                    let (r, s) = Self::hash_suffix(h as f64 / duration);
                    info!("Device {}: {:.2} {}", device, r, s);
                }
                continue;
            }

            // No hashes this tick — keep the per-device counters drained for the next window.
            for (_device, counter) in &*hashes_by_worker.lock().unwrap() {
                counter.store(0, Ordering::Release);
            }

            if challenge_active {
                // PoW is intentionally paused while the GPU runs inference / loads a model.
                zero_streak = 0;
                info!("OPoI inference in progress — PoW paused, stand by");
            } else {
                zero_streak = zero_streak.saturating_add(1);
                if zero_streak >= STALL_GRACE_TICKS {
                    // Sustained zeros outside an inference pause — this is a real problem.
                    warn!("Workers stalled or crashed. Consider reducing workload and check that your node is synced");
                    for (device, _) in &*hashes_by_worker.lock().unwrap() {
                        warn!("Device {}: 0 hash/s", device);
                    }
                } else {
                    // Transient pause (model load/eviction or template gap) — not a crash yet.
                    info!("PoW paused (OPoI inference / model load) — stand by");
                }
            }
        }
    }

    #[inline]
    fn hash_suffix(n: f64) -> (f64, &'static str) {
        match n {
            n if n < 1_000.0 => (n, "hash/s"),
            n if n < 1_000_000.0 => (n / 1_000.0, "Khash/s"),
            n if n < 1_000_000_000.0 => (n / 1_000_000.0, "Mhash/s"),
            n if n < 1_000_000_000_000.0 => (n / 1_000_000_000.0, "Ghash/s"),
            n if n < 1_000_000_000_000_000.0 => (n / 1_000_000_000_000.0, "Thash/s"),
            _ => (n, "hash/s"),
        }
    }
}

#[cfg(all(test, feature = "bench"))]
mod benches {
    extern crate test;

    use self::test::{black_box, Bencher};
    use crate::pow::State;
    use crate::proto::{RpcBlock, RpcBlockHeader};
    use rand::{thread_rng, RngCore};

    #[bench]
    pub fn bench_mining(bh: &mut Bencher) {
        let mut state = State::new(
            0,
            RpcBlock {
                header: Some(RpcBlockHeader {
                    version: 1,
                    parents: vec![],
                    hash_merkle_root: "23618af45051560529440541e7dc56be27676d278b1e00324b048d410a19d764".to_string(),
                    accepted_id_merkle_root: "947d1a10378d6478b6957a0ed71866812dee33684968031b1cace4908c149d94"
                        .to_string(),
                    utxo_commitment: "ec5e8fc0bc0c637004cee262cef12e7cf6d9cd7772513dbd466176a07ab7c4f4".to_string(),
                    timestamp: 654654353,
                    bits: 0x1e7fffff,
                    nonce: 0,
                    daa_score: 654456,
                    blue_work: "d8e28a03234786".to_string(),
                    pruning_point: "be4c415d378f9113fabd3c09fcc84ddb6a00f900c87cb6a1186993ddc3014e2d".to_string(),
                    blue_score: 1164419,
                    pom_final_state: 0,
                }),
                transactions: vec![],
                verbose_data: None,
            },
        )
        .unwrap();
        nonce = thread_rng().next_u64();
        bh.iter(|| {
            for _ in 0..100 {
                black_box(state.check_pow(nonce));
                nonce += 1;
            }
        });
    }
}
