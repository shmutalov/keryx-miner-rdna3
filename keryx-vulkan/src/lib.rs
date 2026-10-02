//! Minimal headless Vulkan **compute** foundation for the Keryx RDNA3 miner.
//!
//! `ash` loads `vulkan-1` at runtime (no Vulkan SDK needed to build or run — only the loader that
//! ships with the AMD driver), so this is a zero-toolchain GPU compute backend. It exposes just
//! what the miner needs: pick the RDNA3 device, allocate storage buffers, and dispatch a compiled
//! SPIR-V compute shader with push constants. The PoM walk and (later) kHeavyHash PoW kernels are
//! built on top of this in their own modules.

use ash::vk;
use ash::vk::Handle;
use std::ffi::{CStr, CString};

pub mod khh;
pub mod pom_walk;
pub mod pom_walk_v4;

// Opt-in PoM-walk micro-benchmark (variant kernels + timing). Excluded from the shipped miner.
#[cfg(feature = "bench")]
pub mod bench;

/// Quick probe: the name of the Vulkan compute device, or None if no loader/device is usable.
/// Used by the miner's startup inference check (the RDNA3 equivalent of the old cuBLAS probe).
pub fn probe_device() -> Option<String> {
    Vk::new().ok().map(|vk| vk.device_name().to_string())
}

/// Quick probe: total VRAM (MiB) of the Vulkan compute device the miner mines/serves on, or
/// None if no usable device. RDNA3 replacement for the upstream `nvidia-smi` VRAM query — the
/// figure comes from the same device pick the compute backend uses, so it matches what the
/// miner can actually allocate. Used by the model capability gate.
pub fn probe_vram_mb() -> Option<u64> {
    Vk::new().ok().map(|vk| vk.device_local_vram_mb())
}

/// Total VRAM (MiB) of a specific Vulkan device (by raw enumeration index), or None if unusable.
/// Multi-GPU: the model capability / PoM-residency gates size the INFERENCE device specifically,
/// which need not be the device a given mining worker runs on.
pub fn probe_vram_mb_for(index: usize) -> Option<u64> {
    Vk::new_for_device(Some(index)).ok().map(|vk| vk.device_local_vram_mb())
}

/// One enumerated Vulkan physical device. `index` is the raw `vkEnumeratePhysicalDevices` position
/// — the same order the loader reports to every Vulkan client in this process tree (including
/// llama.cpp's `GGML_VK_VISIBLE_DEVICES`), so it is the stable cross-component device id.
#[derive(Clone, Debug)]
pub struct VkDeviceInfo {
    pub index: usize,
    pub name: String,
    pub vram_mb: u64,
    pub discrete: bool,
}

/// Enumerate all Vulkan physical devices (raw loader order). Empty if no loader/instance.
pub fn enumerate_devices() -> Vec<VkDeviceInfo> {
    unsafe {
        let Ok(entry) = ash::Entry::load() else { return Vec::new() };
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"keryx-miner-rdna3")
            .api_version(vk::make_api_version(0, 1, 3, 0));
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let Ok(instance) = entry.create_instance(&create_info, None) else { return Vec::new() };
        let devices = instance
            .enumerate_physical_devices()
            .unwrap_or_default()
            .iter()
            .enumerate()
            .map(|(index, &pd)| {
                let props = instance.get_physical_device_properties(pd);
                let mem = instance.get_physical_device_memory_properties(pd);
                let heaps = &mem.memory_heaps[..mem.memory_heap_count as usize];
                let vram_mb = heaps
                    .iter()
                    .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                    .map(|h| h.size)
                    .max()
                    .unwrap_or(0)
                    / (1024 * 1024);
                VkDeviceInfo {
                    index,
                    name: cstr_array_to_string(&props.device_name),
                    vram_mb,
                    discrete: props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
                }
            })
            .collect();
        instance.destroy_instance(None);
        devices
    }
}

/// The device inference (llama-server) runs on: `KERYX_INFER_GPU` override, else the first
/// discrete GPU, else device 0. PoM blob eviction and the VRAM fit gate key off this index, and
/// llama-server is pinned to it via `GGML_VK_VISIBLE_DEVICES` on multi-GPU rigs so inference
/// never silently layer-splits across cards that are busy mining.
pub fn inference_device_index() -> usize {
    if let Some(idx) = std::env::var("KERYX_INFER_GPU").ok().and_then(|s| s.parse::<usize>().ok()) {
        return idx;
    }
    let devices = enumerate_devices();
    devices.iter().find(|d| d.discrete).map(|d| d.index).unwrap_or(0)
}

/// Zero-dup: routes a queue submission through an external owner's guarded submit hook
/// (ggml-vulkan's mutex-protected compute queue) instead of a queue we own. Arguments are
/// FFI-shaped so the miner can wrap the ggml C export without depending on ash:
/// `(submit_info: *const VkSubmitInfo, fence: VkFence-as-u64)`.
pub type ExternalSubmit = Box<dyn Fn(*const std::ffi::c_void, u64) + Send + Sync>;

/// A ready-to-use compute device: instance, the chosen physical device, a logical device with a
/// compute queue, and a command pool. One per (worker, kernel family) — on multi-GPU rigs each
/// worker opens its own `Vk` bound to its device index. Alternatively BORROWED from another
/// Vulkan owner in-process (ggml) via [`Vk::from_raw_handles`]: then only the command pool is
/// ours, and submissions route through the owner's guarded hook.
pub struct Vk {
    _entry: ash::Entry,
    instance: ash::Instance,
    pub device: ash::Device,
    // Retained for device-local staging + limit queries added during miner integration.
    #[allow(dead_code)]
    pdevice: vk::PhysicalDevice,
    queue: vk::Queue,
    #[allow(dead_code)]
    queue_family: u32,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    cmd_pool: vk::CommandPool,
    device_name: String,
    device_index: usize,
    /// False when the instance/device are borrowed (ggml owns them; we only own cmd_pool).
    owned: bool,
    /// Present iff borrowed: the owner's serialized queue-submit hook.
    external_submit: Option<ExternalSubmit>,
}

impl Vk {
    /// Open a compute-capable Vulkan device, preferring a discrete GPU (the RDNA3 card). Enables
    /// `shaderInt64` — required by the PoM/PoW kernels' 64-bit folds.
    pub fn new() -> Result<Self, String> {
        Self::new_for_device(None)
    }

    /// Open a specific Vulkan device by raw enumeration index (`None` = first DISCRETE_GPU, else
    /// the first available — the historical single-GPU pick).
    pub fn new_for_device(index: Option<usize>) -> Result<Self, String> {
        unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("Vulkan loader (vulkan-1) not found: {e}"))?;
            let app_info = vk::ApplicationInfo::default()
                .application_name(c"keryx-miner-rdna3")
                .api_version(vk::make_api_version(0, 1, 3, 0));
            let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
            let instance = entry
                .create_instance(&create_info, None)
                .map_err(|e| format!("create_instance failed: {e}"))?;

            let pdevices = instance
                .enumerate_physical_devices()
                .map_err(|e| format!("enumerate_physical_devices: {e}"))?;
            if pdevices.is_empty() {
                instance.destroy_instance(None);
                return Err("no Vulkan physical devices found".into());
            }
            let pick = match index {
                // Explicit device: raw enumeration index (matches `enumerate_devices`).
                Some(i) => {
                    let Some(&pd) = pdevices.get(i) else {
                        instance.destroy_instance(None);
                        return Err(format!(
                            "Vulkan device index {i} out of range ({} device(s) present)",
                            pdevices.len()
                        ));
                    };
                    let props = instance.get_physical_device_properties(pd);
                    (pd, cstr_array_to_string(&props.device_name), i)
                }
                // Auto: first DISCRETE_GPU, else the first available.
                None => {
                    let mut pick: Option<(vk::PhysicalDevice, String, bool, usize)> = None;
                    for (i, &pd) in pdevices.iter().enumerate() {
                        let props = instance.get_physical_device_properties(pd);
                        let name = cstr_array_to_string(&props.device_name);
                        let discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
                        match &pick {
                            None => pick = Some((pd, name, discrete, i)),
                            Some((_, _, picked_discrete, _)) if !picked_discrete && discrete => {
                                pick = Some((pd, name, discrete, i))
                            }
                            _ => {}
                        }
                    }
                    let (pd, name, _, i) = pick.unwrap();
                    (pd, name, i)
                }
            };
            let (pdevice, device_name, device_index) = pick;

            // Find a queue family that supports COMPUTE.
            let qfams = instance.get_physical_device_queue_family_properties(pdevice);
            let queue_family = qfams
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or("no compute-capable queue family")? as u32;

            let priorities = [1.0f32];
            let qcis = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities)];
            // shader_int64 for the 64-bit folds; shader_integer_dot_product for the kHeavyHash
            // matmul (the hardware 4x8-bit packed dot — RDNA3's v_dot4_u32_u8).
            let features = vk::PhysicalDeviceFeatures::default().shader_int64(true);
            // bufferDeviceAddress: lets the PoM weight blob be read through a 64-bit pointer
            // (GL_EXT_buffer_reference) instead of a bound storage descriptor. An 8B+ model's blob
            // is >4 GiB, which exceeds `maxStorageBufferRange` (0xFFFFFFFF on AMD) — accessing it by
            // device address sidesteps that per-descriptor cap. Core in Vulkan 1.2; RDNA3 supports it.
            let mut features12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true);
            let mut features13 = vk::PhysicalDeviceVulkan13Features::default().shader_integer_dot_product(true);
            let dci = vk::DeviceCreateInfo::default()
                .queue_create_infos(&qcis)
                .enabled_features(&features)
                .push_next(&mut features12)
                .push_next(&mut features13);
            let device = instance
                .create_device(pdevice, &dci, None)
                .map_err(|e| format!("create_device failed: {e}"))?;
            let queue = device.get_device_queue(queue_family, 0);
            let mem_props = instance.get_physical_device_memory_properties(pdevice);
            let cmd_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(queue_family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|e| format!("create_command_pool: {e}"))?;

            Ok(Self {
                _entry: entry,
                instance,
                device,
                pdevice,
                queue,
                queue_family,
                mem_props,
                cmd_pool,
                device_name,
                device_index,
                owned: true,
                external_submit: None,
            })
        }
    }

    /// Zero-dup: wrap ANOTHER in-process Vulkan owner's live handles (ggml's instance /
    /// physical device / device) so kernels built here read buffers created there — buffer
    /// device addresses are only valid on the device that owns them. We create only a command
    /// pool; every submission routes through `external_submit` (the owner's mutex-guarded
    /// queue hook), and Drop releases only what we created.
    ///
    /// # Safety
    /// The handles must remain valid for this `Vk`'s lifetime (ggml keeps its device alive for
    /// the process lifetime; the shared PoM walk is torn down before engine eviction anyway).
    pub unsafe fn from_raw_handles(
        instance_ptr: *mut std::ffi::c_void,
        physical_device_ptr: *mut std::ffi::c_void,
        device_ptr: *mut std::ffi::c_void,
        queue_family: u32,
        device_index: usize,
        external_submit: ExternalSubmit,
    ) -> Result<Self, String> {
        let entry = ash::Entry::load().map_err(|e| format!("Vulkan loader (vulkan-1) not found: {e}"))?;
        let instance = ash::Instance::load(entry.static_fn(), vk::Instance::from_raw(instance_ptr as u64));
        let pdevice = vk::PhysicalDevice::from_raw(physical_device_ptr as u64);
        let device = ash::Device::load(instance.fp_v1_0(), vk::Device::from_raw(device_ptr as u64));
        let props = instance.get_physical_device_properties(pdevice);
        let device_name = cstr_array_to_string(&props.device_name);
        let mem_props = instance.get_physical_device_memory_properties(pdevice);
        let cmd_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .map_err(|e| format!("create_command_pool (borrowed device): {e}"))?;
        Ok(Self {
            _entry: entry,
            instance,
            device,
            pdevice,
            queue: vk::Queue::null(), // never used: submissions go through external_submit
            queue_family,
            mem_props,
            cmd_pool,
            device_name,
            device_index,
            owned: false,
            external_submit: Some(external_submit),
        })
    }

    /// Submit one batch on the compute queue: ours directly, or the borrowed owner's through
    /// its guarded hook (VkSubmitInfo is ABI-stable, so the raw pointer cast is sound).
    unsafe fn queue_submit_routed(&self, submit: &vk::SubmitInfo, fence: vk::Fence) -> Result<(), String> {
        match &self.external_submit {
            Some(hook) => {
                hook(submit as *const vk::SubmitInfo as *const std::ffi::c_void, fence.as_raw());
                Ok(())
            }
            None => self
                .device
                .queue_submit(self.queue, std::slice::from_ref(submit), fence)
                .map_err(|e| e.to_string()),
        }
    }

    /// Human-readable name of the selected GPU (e.g. "AMD Radeon RX 7900 XT").
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Raw enumeration index of the selected GPU (stable id across workers/llama-server pinning).
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Total VRAM (MiB) = the largest `DEVICE_LOCAL` memory heap on the selected GPU. Taking the
    /// max (not the sum) avoids double-counting the small host-visible BAR heap AMD reports as a
    /// separate device-local window into the same VRAM, so this matches the figure the driver and
    /// llama-server print (e.g. 20464 MiB on a 7900 XT).
    pub fn device_local_vram_mb(&self) -> u64 {
        let heaps = &self.mem_props.memory_heaps[..self.mem_props.memory_heap_count as usize];
        let bytes = heaps
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .max()
            .unwrap_or(0);
        bytes / (1024 * 1024)
    }

    fn find_memory_type(&self, type_bits: u32, flags: vk::MemoryPropertyFlags) -> Result<u32, String> {
        for i in 0..self.mem_props.memory_type_count {
            let suitable = (type_bits & (1 << i)) != 0;
            if suitable && self.mem_props.memory_types[i as usize].property_flags.contains(flags) {
                return Ok(i);
            }
        }
        Err(format!("no memory type for flags {flags:?}"))
    }

    /// Allocate a host-visible, coherent STORAGE buffer (mapped reads/writes; GPU reads over the
    /// bus). Simple and correct; the hot PoM weight blob moves to device-local in integration.
    pub fn create_buffer(&self, size: u64) -> Result<GpuBuffer, String> {
        assert!(size > 0, "zero-size buffer");
        unsafe {
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self.device.create_buffer(&info, None).map_err(|e| e.to_string())?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let mt = self.find_memory_type(
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let memory = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt),
                    None,
                )
                .map_err(|e| e.to_string())?;
            self.device.bind_buffer_memory(buffer, memory, 0).map_err(|e| e.to_string())?;
            Ok(GpuBuffer { buffer, memory, size })
        }
    }

    /// Allocate a host-writable STORAGE buffer that the GPU reads at VRAM speed: a
    /// `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` (BAR / ReBAR) type when the device exposes one,
    /// else the plain host-visible type of [`create_buffer`](Self::create_buffer). For small tables
    /// the kernel reads on every step (the v4 walk's segment LUT/prefix/addresses and its dispatch
    /// I/O block), where a system-memory buffer would put a PCIe round trip in the hot loop.
    pub fn create_buffer_vram_mapped(&self, size: u64) -> Result<GpuBuffer, String> {
        assert!(size > 0, "zero-size buffer");
        unsafe {
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self.device.create_buffer(&info, None).map_err(|e| e.to_string())?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let bar = vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT;
            let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let mut allocated = None;
            for flags in [bar, host] {
                let Ok(mt) = self.find_memory_type(req.memory_type_bits, flags) else { continue };
                let alloc = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt);
                // The BAR heap can be small (256 MiB without ReBAR) — fall back rather than fail.
                if let Ok(memory) = self.device.allocate_memory(&alloc, None) {
                    allocated = Some(memory);
                    break;
                }
            }
            let Some(memory) = allocated else {
                self.device.destroy_buffer(buffer, None);
                return Err(format!("no host-visible memory for a {size}-byte buffer"));
            };
            self.device.bind_buffer_memory(buffer, memory, 0).map_err(|e| e.to_string())?;
            Ok(GpuBuffer { buffer, memory, size })
        }
    }

    /// Allocate a host-visible buffer that also exposes a GPU **device address**
    /// (`VK_KHR_buffer_device_address`), returning the buffer and its 64-bit address. The shader
    /// reads it via a `buffer_reference` pointer rather than a bound SSBO, so the buffer can exceed
    /// the 4 GiB `maxStorageBufferRange` descriptor cap — required for the PoM weight blob of an 8B+
    /// model (~4.6 GiB). Usage is `SHADER_DEVICE_ADDRESS` only (no `STORAGE_BUFFER`): it is never
    /// bound as a descriptor, and dropping that usage also avoids the driver's >4 GiB storage-buffer
    /// rejection at creation time.
    pub fn create_device_address_buffer(&self, size: u64) -> Result<(GpuBuffer, u64), String> {
        assert!(size > 0, "zero-size buffer");
        unsafe {
            // A single allocation can't exceed maxMemoryAllocationSize. Surface a clear error
            // instead of the driver's opaque VK_ERROR_UNKNOWN if a future tier's blob is too large.
            let max_alloc = self.max_memory_allocation_size();
            if max_alloc != 0 && size > max_alloc {
                return Err(format!(
                    "weight blob {size} bytes exceeds maxMemoryAllocationSize {max_alloc} bytes"
                ));
            }

            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self.device.create_buffer(&info, None).map_err(|e| e.to_string())?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let mt = self.find_memory_type(
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            // DEVICE_ADDRESS allocate flag is mandatory when the buffer carries SHADER_DEVICE_ADDRESS.
            let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let memory = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(mt)
                        .push_next(&mut flags),
                    None,
                )
                .map_err(|e| e.to_string())?;
            self.device.bind_buffer_memory(buffer, memory, 0).map_err(|e| e.to_string())?;
            let address = self
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            Ok((GpuBuffer { buffer, memory, size }, address))
        }
    }

    /// Allocate a **DEVICE_LOCAL** buffer that exposes a GPU device address, and fill it from `data`
    /// via a host-visible staging copy. Device-local is essential for the PoM walk: its 256
    /// data-dependent reads per nonce are ~100x faster from VRAM than from host-visible memory over
    /// PCIe — a host-visible blob made a single batch overrun the Windows TDR watchdog (DEVICE_LOST).
    pub fn create_device_local_address_buffer(&self, data: &[u8]) -> Result<(GpuBuffer, u64), String> {
        self.create_device_local_address_buffer_streamed(data.len() as u64, &mut |offset, out| {
            out.copy_from_slice(&data[offset as usize..offset as usize + out.len()]);
            Ok(())
        })
    }

    /// Staging window for [`create_device_local_address_buffer_streamed`]: 256 MiB bounds the
    /// transient host-visible allocation regardless of blob size (a 1 GiB shard streams in 4 fills).
    const STAGING_BYTES: u64 = 256 * 1024 * 1024;

    /// Like [`create_device_local_address_buffer`], but the contents are produced incrementally by
    /// `fill(offset, window)` into a bounded staging window instead of being passed as one slice.
    /// This is the zero-dup upload path: the caller streams bytes straight from the GGUF on disk,
    /// so no full-blob host copy ever exists (peak host overhead = one 256 MiB staging window,
    /// vs. the ~4.6 GiB packed `Vec` for the 8B tier — and ~25-40 GiB for the 32B/70B tiers).
    pub fn create_device_local_address_buffer_streamed(
        &self,
        size: u64,
        fill: &mut dyn FnMut(u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<(GpuBuffer, u64), String> {
        assert!(size > 0, "zero-size buffer");
        unsafe {
            let max_alloc = self.max_memory_allocation_size();
            if max_alloc != 0 && size > max_alloc {
                return Err(format!(
                    "weight shard {size} bytes exceeds maxMemoryAllocationSize {max_alloc} bytes"
                ));
            }

            // Device-local destination: read by the shader via its device address, written by copy.
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self.device.create_buffer(&info, None).map_err(|e| e.to_string())?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let mt = self.find_memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let memory = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(mt)
                        .push_next(&mut flags),
                    None,
                )
                .map_err(|e| e.to_string())?;
            self.device.bind_buffer_memory(buffer, memory, 0).map_err(|e| e.to_string())?;

            // Bounded host-visible staging window, refilled and re-copied until the blob is up.
            let stage_size = size.min(Self::STAGING_BYTES);
            let staging_info = vk::BufferCreateInfo::default()
                .size(stage_size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let staging = self.device.create_buffer(&staging_info, None).map_err(|e| e.to_string())?;
            let sreq = self.device.get_buffer_memory_requirements(staging);
            let smt = self.find_memory_type(
                sreq.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let smem = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default().allocation_size(sreq.size).memory_type_index(smt),
                    None,
                )
                .map_err(|e| e.to_string())?;
            let mut upload = || -> Result<(), String> {
                self.device.bind_buffer_memory(staging, smem, 0).map_err(|e| e.to_string())?;
                let ptr = self
                    .device
                    .map_memory(smem, 0, stage_size, vk::MemoryMapFlags::empty())
                    .map_err(|e| e.to_string())? as *mut u8;
                let mut done: u64 = 0;
                let res = loop {
                    if done == size {
                        break Ok(());
                    }
                    let len = (size - done).min(stage_size);
                    let window = std::slice::from_raw_parts_mut(ptr, len as usize);
                    if let Err(e) = fill(done, window) {
                        break Err(e);
                    }
                    if let Err(e) = self.immediate_copy_region(staging, buffer, len, done) {
                        break Err(e);
                    }
                    done += len;
                };
                self.device.unmap_memory(smem);
                res
            };
            let copy_res = upload();
            self.device.destroy_buffer(staging, None);
            self.device.free_memory(smem, None);
            if let Err(e) = copy_res {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
                return Err(e);
            }

            let address = self
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            Ok((GpuBuffer { buffer, memory, size }, address))
        }
    }

    /// The device's single-allocation ceiling (`maxMemoryAllocationSize`), 0 if unreported.
    unsafe fn max_memory_allocation_size(&self) -> u64 {
        let mut limits11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut limits11);
        self.instance.get_physical_device_properties2(self.pdevice, &mut props2);
        limits11.max_memory_allocation_size
    }

    /// Submit a one-shot buffer→buffer copy (`src[0..size]` → `dst[dst_offset..]`) and block until
    /// it completes.
    unsafe fn immediate_copy_region(
        &self,
        src: vk::Buffer,
        dst: vk::Buffer,
        size: u64,
        dst_offset: u64,
    ) -> Result<(), String> {
        let cmd = self
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(|e| e.to_string())?[0];
        let cmds = [cmd];
        let run = || -> Result<(), String> {
            self.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| e.to_string())?;
            self.device
                .cmd_copy_buffer(cmd, src, dst, &[vk::BufferCopy::default().size(size).dst_offset(dst_offset)]);
            self.device.end_command_buffer(cmd).map_err(|e| e.to_string())?;
            let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| e.to_string())?;
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let res = self
                .queue_submit_routed(&submit, fence)
                .and_then(|_| self.device.wait_for_fences(&[fence], true, u64::MAX).map_err(|e| e.to_string()));
            self.device.destroy_fence(fence, None);
            res
        };
        let res = run();
        self.device.free_command_buffers(self.cmd_pool, &cmds);
        res
    }

    /// Copy `data` into a host-visible buffer (`data.len()` must be ≤ the buffer size).
    pub fn write_buffer(&self, b: &GpuBuffer, data: &[u8]) {
        assert!(data.len() as u64 <= b.size, "write past buffer end");
        unsafe {
            let ptr = self
                .device
                .map_memory(b.memory, 0, b.size, vk::MemoryMapFlags::empty())
                .expect("map_memory") as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            self.device.unmap_memory(b.memory);
        }
    }

    /// Read `out.len()` bytes back from a host-visible buffer.
    pub fn read_buffer(&self, b: &GpuBuffer, out: &mut [u8]) {
        assert!(out.len() as u64 <= b.size, "read past buffer end");
        unsafe {
            let ptr = self
                .device
                .map_memory(b.memory, 0, b.size, vk::MemoryMapFlags::empty())
                .expect("map_memory") as *const u8;
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), out.len());
            self.device.unmap_memory(b.memory);
        }
    }

    /// Build a compute pipeline from SPIR-V with `n_bindings` storage buffers and `push_size`
    /// bytes of push constants. The returned `Kernel` is reusable across dispatches.
    pub fn make_kernel(&self, spirv: &[u32], n_bindings: u32, push_size: u32) -> Result<Kernel, String> {
        unsafe {
            let bindings: Vec<_> = (0..n_bindings)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = self
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| e.to_string())?;
            let pc_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push_size)];
            let set_layouts = [set_layout];
            let pipeline_layout = self
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&pc_ranges),
                    None,
                )
                .map_err(|e| e.to_string())?;
            let module = self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)
                .map_err(|e| e.to_string())?;
            let entry = CString::new("main").unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&entry);
            let pipeline = self
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout)],
                    None,
                )
                .map_err(|(_, e)| e.to_string())?[0];
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(n_bindings.max(1))];
            let desc_pool = self
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes)
                        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET),
                    None,
                )
                .map_err(|e| e.to_string())?;
            let cmd = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(|e| e.to_string())?[0];
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| e.to_string())?;
            Ok(Kernel { set_layout, pipeline_layout, pipeline, module, desc_pool, cmd, fence })
        }
    }

    /// Bind `buffers` (in binding order 0..n) + `push` constants and dispatch `groups` workgroups
    /// on x. Blocks until the GPU finishes (fence wait).
    pub fn dispatch(&self, k: &Kernel, buffers: &[&GpuBuffer], push: &[u8], groups: u32) {
        unsafe {
            let dev = &self.device;
            dev.reset_descriptor_pool(k.desc_pool, vk::DescriptorPoolResetFlags::empty()).unwrap();
            let layouts = [k.set_layout];
            let set = dev
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default().descriptor_pool(k.desc_pool).set_layouts(&layouts),
                )
                .unwrap()[0];
            let infos: Vec<_> = buffers
                .iter()
                .map(|b| vk::DescriptorBufferInfo::default().buffer(b.buffer).offset(0).range(vk::WHOLE_SIZE))
                .collect();
            let writes: Vec<_> = (0..buffers.len())
                .map(|i| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&infos[i]))
                })
                .collect();
            dev.update_descriptor_sets(&writes, &[]);

            dev.reset_command_buffer(k.cmd, vk::CommandBufferResetFlags::empty()).unwrap();
            dev.begin_command_buffer(
                k.cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
            dev.cmd_bind_pipeline(k.cmd, vk::PipelineBindPoint::COMPUTE, k.pipeline);
            dev.cmd_bind_descriptor_sets(k.cmd, vk::PipelineBindPoint::COMPUTE, k.pipeline_layout, 0, &[set], &[]);
            if !push.is_empty() {
                dev.cmd_push_constants(k.cmd, k.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            }
            dev.cmd_dispatch(k.cmd, groups, 1, 1);
            dev.end_command_buffer(k.cmd).unwrap();

            let cmds = [k.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            dev.reset_fences(&[k.fence]).unwrap();
            self.queue_submit_routed(&submit, k.fence).expect("queue submit");
            dev.wait_for_fences(&[k.fence], true, u64::MAX).unwrap();
        }
    }

    /// Release a kernel's Vulkan objects.
    pub fn destroy_kernel(&self, k: &Kernel) {
        unsafe {
            self.device.destroy_fence(k.fence, None);
            self.device.destroy_descriptor_pool(k.desc_pool, None);
            self.device.destroy_pipeline(k.pipeline, None);
            self.device.destroy_shader_module(k.module, None);
            self.device.destroy_pipeline_layout(k.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(k.set_layout, None);
        }
    }

    /// Release a buffer's Vulkan objects.
    pub fn destroy_buffer(&self, b: &GpuBuffer) {
        unsafe {
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
    }
}

impl Drop for Vk {
    fn drop(&mut self) {
        unsafe {
            if self.owned {
                let _ = self.device.device_wait_idle();
                self.device.destroy_command_pool(self.cmd_pool, None);
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            } else {
                // Borrowed (ggml) device: we own only the command pool. Every submission is
                // fence-waited before its caller returns, so the pool is idle; skipping
                // device_wait_idle avoids stalling the owner's in-flight inference work.
                self.device.destroy_command_pool(self.cmd_pool, None);
            }
        }
    }
}

/// A storage buffer + its backing device memory. Destroy via [`Vk::destroy_buffer`].
pub struct GpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

/// A compiled compute pipeline (+ descriptor pool, command buffer, fence), reusable across
/// dispatches. Destroy via [`Vk::destroy_kernel`].
pub struct Kernel {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    module: vk::ShaderModule,
    desc_pool: vk::DescriptorPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
}

fn cstr_array_to_string(arr: &[std::os::raw::c_char]) -> String {
    unsafe { CStr::from_ptr(arr.as_ptr()).to_string_lossy().into_owned() }
}
