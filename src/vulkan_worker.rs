//! In-process Vulkan PoW workers (RDNA3). Replaces the CUDA/OpenCL cdylib plugins: a [`Plugin`]
//! that yields one [`Worker`] per mining GPU, each driving the verified `keryx-vulkan` kHeavyHash
//! kernel on its own device. Registered directly into the [`PluginManager`](crate::PluginManager)
//! by `main` — no shared library is loaded. The miner's existing GPU loop drives each worker
//! through the `Worker` trait unchanged.
//!
//! Device selection: `--gpu 0,2` restricts mining to those raw Vulkan device indices (see the
//! startup log for the enumerated list); the default is every discrete GPU, falling back to the
//! historical single auto-picked device when none is discrete.

use clap::ArgMatches;
use keryx_vulkan::khh::{KhhGpu, MATRIX_LEN};
use log::info;
use rand::{thread_rng, RngCore};

use crate::{Error, Plugin, Worker, WorkerSpec};

/// Nonces per `calculate_hash` dispatch. The kernel grinds the whole batch before returning, so
/// this trades latency for launch-overhead amortisation. Overridable via `KERYX_VULKAN_WORKLOAD`.
const DEFAULT_WORKLOAD: usize = 1 << 20;

fn configured_workload() -> usize {
    std::env::var("KERYX_VULKAN_WORKLOAD").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(DEFAULT_WORKLOAD)
}

/// In-process plugin exposing each mining GPU as one PoW worker.
pub struct VulkanPlugin {
    workload: usize,
    /// Raw Vulkan device indices to mine on, resolved in `process_option`.
    devices: Vec<usize>,
}

impl Default for VulkanPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl VulkanPlugin {
    pub fn new() -> Self {
        Self { workload: configured_workload(), devices: Vec::new() }
    }
}

impl Plugin for VulkanPlugin {
    fn name(&self) -> &'static str {
        "vulkan"
    }
    fn enabled(&self) -> bool {
        true
    }
    fn get_worker_specs(&self) -> Vec<Box<dyn WorkerSpec>> {
        // Device table is logged HERE, not in process_option: options are processed before
        // main initialises the logger, so anything logged there is silently dropped. This
        // runs at worker launch, and multi-GPU users need the index table to use --gpu.
        for d in keryx_vulkan::enumerate_devices() {
            info!(
                "Vulkan device {}: {} ({} MiB VRAM{})",
                d.index,
                d.name,
                d.vram_mb,
                if d.discrete { ", discrete" } else { "" }
            );
        }
        // Log the PoM batch, not `self.workload`: every block on every network is PoM, so the
        // kHeavyHash workload is never dispatched.
        info!(
            "Vulkan mining: {} worker(s) on device(s) {:?} (PoM batch {} nonces/dispatch)",
            self.devices.len(),
            self.devices,
            crate::pom_gpu::v4_batch()
        );
        self.devices
            .iter()
            .map(|&device_index| {
                Box::new(VulkanWorkerSpec { workload: self.workload, device_index }) as Box<dyn WorkerSpec>
            })
            .collect()
    }
    fn process_option(&mut self, matches: &ArgMatches) -> Result<usize, Error> {
        let all = keryx_vulkan::enumerate_devices();

        self.devices = match matches.value_of("gpu") {
            // Explicit selection: comma-separated raw device indices.
            Some(list) => {
                let mut devices = Vec::new();
                for part in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    let idx: usize = part.parse().map_err(|_| format!("--gpu: '{part}' is not a device index"))?;
                    if idx >= all.len() {
                        return Err(format!("--gpu: device index {idx} out of range ({} device(s) present)", all.len()).into());
                    }
                    if !devices.contains(&idx) {
                        devices.push(idx);
                    }
                }
                if devices.is_empty() {
                    return Err("--gpu: no device indices given".into());
                }
                devices
            }
            // Default: every discrete GPU; else the historical single auto pick (device 0-equivalent).
            None => {
                let discrete: Vec<usize> = all.iter().filter(|d| d.discrete).map(|d| d.index).collect();
                if !discrete.is_empty() {
                    discrete
                } else if !all.is_empty() {
                    vec![all[0].index]
                } else {
                    Vec::new() // no Vulkan device; the startup probe reports this separately
                }
            }
        };

        Ok(self.devices.len())
    }
}

pub struct VulkanWorkerSpec {
    workload: usize,
    device_index: usize,
}

impl WorkerSpec for VulkanWorkerSpec {
    fn id(&self) -> String {
        format!("Vulkan #{}", self.device_index)
    }
    fn build(&self) -> Box<dyn Worker> {
        Box::new(
            VulkanKhhWorker::new(self.workload, self.device_index).expect("Vulkan kHeavyHash worker init failed"),
        )
    }
}

/// kHeavyHash PoW worker backed by the `keryx-vulkan` GPU kernel, bound to one device.
pub struct VulkanKhhWorker {
    gpu: KhhGpu,
    workload: usize,
    nonce_cursor: u64,
    last_winner: u64,
    id: String,
}

impl VulkanKhhWorker {
    fn new(workload: usize, device_index: usize) -> Result<Self, Error> {
        let gpu = KhhGpu::new_for_device(Some(device_index)).map_err(|e| -> Error { e.into() })?;
        let id = format!("#{} {}", device_index, gpu.device_name());
        info!("Vulkan kHeavyHash worker on: {}", id);
        Ok(Self { gpu, workload, nonce_cursor: thread_rng().next_u64(), last_winner: 0, id })
    }
}

impl Worker for VulkanKhhWorker {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn load_block_constants(&mut self, hash_header: &[u8; 72], matrix: &[[u16; 64]; 64], target: &[u64; 4]) {
        // 72-byte pow header → 9 little-endian u64 words (pre_pow_hash || time || 0^32).
        let mut header = [0u64; 9];
        for (i, w) in header.iter_mut().enumerate() {
            *w = u64::from_le_bytes(hash_header[i * 8..i * 8 + 8].try_into().unwrap());
        }
        // [[u16;64];64] (4-bit entries) → flat row-major [u32; 4096] for the SSBO.
        let mut mat = [0u32; MATRIX_LEN];
        for (r, row) in matrix.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                mat[r * 64 + c] = v as u32;
            }
        }
        self.gpu.upload_block(&mat, header, *target);
    }

    fn calculate_hash(&mut self, _nonces: Option<&Vec<u64>>, nonce_mask: u64, nonce_fixed: u64) {
        // The kernel forms each effective nonce as ((cursor + idx) & nonce_mask) | nonce_fixed, so
        // this honours both solo mining (mask all-ones, fixed 0) and a pool's extranonce sub-range.
        let start = self.nonce_cursor;
        self.last_winner = self.gpu.mine(start, self.workload as u32, nonce_mask, nonce_fixed).unwrap_or(0);
        self.nonce_cursor = self.nonce_cursor.wrapping_add(self.workload as u64);
    }

    fn sync(&self) -> Result<(), Error> {
        Ok(()) // mine() already blocks on the GPU fence
    }

    fn get_workload(&self) -> usize {
        self.workload
    }

    fn copy_output_to(&mut self, nonces: &mut Vec<u64>) -> Result<(), Error> {
        nonces[0] = self.last_winner;
        Ok(())
    }

    fn device_index(&self) -> Option<u32> {
        Some(self.gpu.device_index() as u32)
    }
}
