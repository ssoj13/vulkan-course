//! Step 16 - Sub-allocation.
//!
//! Steps 13-15 gave every buffer its own vkAllocateMemory. That does not
//! scale: each allocation is a kernel-mode call with its own bookkeeping
//! (page tables, residency), the number of live allocations is limited by
//! maxMemoryAllocationCount (the spec guarantees only 4096), and small
//! allocations waste memory to the driver's own granularity. Real programs
//! allocate a few BIG blocks and place many resources inside them at aligned
//! offsets (step 13 bound two buffers into one allocation by hand) - like a
//! texture atlas, but for memory.
//!
//!   raw:       N buffers -> N vkAllocateMemory
//!   allocator: N buffers -> a few blocks, each buffer bound at (block, offset)
//!
//! This step creates the same N buffers both ways and compares the counts and
//! the time. The C++ version uses VMA (Vulkan Memory Allocator). The Rust
//! crate `gpu-allocator` 0.28 (the latest release) is built against crates.io
//! ash 0.38, whose types are not the types of the ash git revision this book
//! pins for Vulkan 1.4, so this file contains its own minimal allocator: a
//! LINEAR (bump) allocator, which is how every allocator starts. Lines whose
//! content depends on the allocator library start with `[lib]`, timings with
//! `[timing]`; all other lines are identical in both languages. No window:
//! steps 01-05 are repeated, reduced to what this step needs.

use std::error::Error;
use std::ffi::{CStr, c_void};
use std::time::{Duration, Instant};

use ash::{
    Entry,
    ext::debug_utils,
    vk::{self, TaggedStructure as _},
};

/// Validation messages are printed here (explained in step 02).
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let level = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        "ERROR"
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        "WARNING"
    } else {
        "INFO"
    };
    let kind = if msg_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION) {
        "VALIDATION"
    } else if msg_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE) {
        "PERFORMANCE"
    } else {
        "GENERAL"
    };
    // SAFETY: the layer passes a valid pointer for the duration of the call
    // (we still guard against null); `p_message` is a NUL-terminated string.
    let message = unsafe { data.as_ref() }
        .filter(|d| !d.p_message.is_null())
        .map(|d| unsafe { CStr::from_ptr(d.p_message) }.to_string_lossy())
        .unwrap_or(std::borrow::Cow::Borrowed("<no message>"));
    eprintln!("[{level}][{kind}] {message}");
    vk::FALSE
}

// #region cli
/// Command line: `--count N`, the number of buffers (default 3000, below the
/// 4096 the spec guarantees, so the raw path works on every device). The
/// buffers must also fit their memory heap; that is checked once the memory
/// type is known.
fn parse_count() -> Result<u32, Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let count = match args.as_slice() {
        [] => 3000,
        [flag, value] if flag == "--count" => value.parse()?,
        _ => return Err("usage: 16_allocator [--count N]".into()),
    };
    if count == 0 || count > 1_000_000 {
        return Err("--count must be 1-1000000".into());
    }
    Ok(count)
}
// #endregion

/// The sizes of the test buffers, used in turn: small uniform-like, a 4 KiB
/// page, a 64 KiB mesh chunk.
const SIZES: [vk::DeviceSize; 3] = [256, 4096, 65536];
const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::VERTEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
);

/// Step 13's find-memory-type rule.
fn find_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count)
        .zip(props.memory_types_as_slice())
        .find_map(|(i, t)| (bits & (1 << i) != 0 && t.property_flags.contains(flags)).then_some(i))
}

/// Rounds `offset` up to a multiple of `alignment` (a power of two in Vulkan).
fn align_up(offset: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize {
    offset.div_ceil(alignment) * alignment
}

/// Milliseconds and microseconds per buffer, for the `[timing]` lines.
fn per_buffer(time: Duration, count: u32) -> String {
    format!(
        "{:.2} ms ({:.2} us per buffer)",
        time.as_secs_f64() * 1e3,
        time.as_secs_f64() * 1e6 / f64::from(count)
    )
}

// #region allocator
/// One big VkDeviceMemory block, filled from the front.
struct Block {
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    /// Bytes handed out so far, including alignment padding.
    used: vk::DeviceSize,
}

/// Where a buffer lives: the block's memory and the offset inside it.
#[derive(Clone, Copy)]
struct Allocation {
    memory: vk::DeviceMemory,
    offset: vk::DeviceSize,
}

/// A minimal sub-allocator for ONE memory type: big blocks, carved linearly.
/// `alloc` rounds the current end of the last block up to the resource's
/// alignment and bumps it by the size; when the block is full it allocates
/// the next one. A request bigger than a block gets a block of its own
/// (a "dedicated" allocation). Individual frees are not possible: the whole
/// allocator is reset or destroyed at once (fine for level data or per-frame
/// scratch). General-purpose allocators (VMA, gpu-allocator) add free lists
/// (VMA uses the TLSF algorithm) so any allocation can be freed alone.
///
/// Only buffers are placed here. Mixing buffers with optimal-tiling images
/// in one block also needs bufferImageGranularity between the two kinds.
struct LinearAllocator {
    memory_type: u32,
    block_size: vk::DeviceSize,
    blocks: Vec<Block>,
    /// vkAllocateMemory calls made, to compare with the raw path.
    allocate_calls: u32,
}

impl LinearAllocator {
    fn new(memory_type: u32, block_size: vk::DeviceSize) -> Self {
        Self {
            memory_type,
            block_size,
            blocks: Vec::new(),
            allocate_calls: 0,
        }
    }

    fn alloc(
        &mut self,
        device: &ash::Device,
        req: &vk::MemoryRequirements,
    ) -> Result<Allocation, Box<dyn Error>> {
        if req.memory_type_bits & (1 << self.memory_type) == 0 {
            return Err(format!(
                "the resource does not allow memory type {}",
                self.memory_type
            )
            .into());
        }
        // Room in the last block?
        if let Some(block) = self.blocks.last_mut() {
            let offset = align_up(block.used, req.alignment);
            if offset + req.size <= block.size {
                block.used = offset + req.size;
                return Ok(Allocation {
                    memory: block.memory,
                    offset,
                });
            }
        }
        // A new block: the usual size, or exactly the request if it is bigger.
        let size = self.block_size.max(req.size);
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(self.memory_type);
        // SAFETY: the type index comes from this device's memory properties.
        let memory = unsafe { device.allocate_memory(&info, None)? };
        self.allocate_calls += 1;
        self.blocks.push(Block {
            memory,
            size,
            used: req.size,
        });
        Ok(Allocation { memory, offset: 0 })
    }

    /// Bytes handed out, including alignment padding.
    fn used(&self) -> vk::DeviceSize {
        self.blocks.iter().map(|b| b.used).sum()
    }

    /// Frees every block. Nothing may still be bound to them in use.
    fn destroy(self, device: &ash::Device) {
        for block in self.blocks {
            // SAFETY: the caller destroyed every buffer bound to the block.
            unsafe { device.free_memory(block.memory, None) };
        }
    }
}
// #endregion

fn main() -> Result<(), Box<dyn Error>> {
    let count = parse_count()?;

    // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
    let entry = unsafe { Entry::load()? };
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"Vulkan from Zero - 16 allocator")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"No Engine")
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_4);

    // Validation setup from step 02.
    let validation_layer = c"VK_LAYER_KHRONOS_validation";
    // SAFETY: plain query, no pointers involved.
    let layers = unsafe { entry.enumerate_instance_layer_properties()? };
    if !layers.iter().any(|l| {
        l.layer_name_as_c_str()
            .is_ok_and(|name| name == validation_layer)
    }) {
        return Err("VK_LAYER_KHRONOS_validation not found. Install the Vulkan SDK.".into());
    }
    let mut messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback));
    let layer_ptrs = [validation_layer.as_ptr()];
    let ext_ptrs = [debug_utils::NAME.as_ptr()];
    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layer_ptrs)
        .enabled_extension_names(&ext_ptrs)
        .push(&mut messenger_info);
    // SAFETY: `create_info` and everything it points to outlive the call.
    let instance = unsafe { entry.create_instance(&create_info, None)? };
    let debug_utils = debug_utils::Instance::load(&entry, &instance);
    // SAFETY: `messenger_info` is valid and the callback is a valid `extern "system"` fn.
    let messenger = unsafe { debug_utils.create_debug_utils_messenger(&messenger_info, None)? };

    // SAFETY: `instance` is alive.
    let devices = unsafe { instance.enumerate_physical_devices()? };
    if devices.is_empty() {
        return Err("No Vulkan-capable GPU found (is the driver installed?)".into());
    }
    println!("Found {} GPU(s)", devices.len());
    for (i, &pd) in devices.iter().enumerate() {
        // SAFETY: `pd` was enumerated from this instance.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        println!(
            "GPU {i}: {}",
            props.device_name_as_c_str()?.to_string_lossy()
        );
    }

    // Step 05's choice: a Vulkan 1.4 GPU with dynamic rendering +
    // synchronization2, discrete first.
    let mut best: Option<(u32, vk::PhysicalDevice)> = None;
    for &pd in &devices {
        // SAFETY: `pd` was enumerated from this instance.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        if props.api_version < vk::API_VERSION_1_4 {
            continue;
        }
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push(&mut f13);
        // SAFETY: `pd` came from this instance, and the chain structs outlive the call.
        unsafe { instance.get_physical_device_features2(pd, &mut f2) };
        if f13.dynamic_rendering == 0 || f13.synchronization2 == 0 {
            continue;
        }
        let score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 10,
            _ => 1,
        };
        if best.is_none_or(|(s, _)| score > s) {
            best = Some((score, pd));
        }
    }
    let (_, physical_device) =
        best.ok_or("No GPU supports Vulkan 1.4 with dynamic rendering + synchronization2")?;
    // SAFETY: `physical_device` was enumerated from this instance.
    let chosen = unsafe { instance.get_physical_device_properties(physical_device) };
    println!(
        "\nPicked: {}",
        chosen.device_name_as_c_str()?.to_string_lossy()
    );
    // SAFETY: `physical_device` was enumerated from this instance.
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let family = u32::try_from(
        families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or("No graphics queue family")?,
    )?;
    println!("Graphics family: {family}");
    let priorities = [1.0_f32];
    let queue_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&priorities)];
    let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);
    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .push(&mut features13);
    // SAFETY: `physical_device` belongs to `instance`, and all create-info data outlives the call.
    let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
    println!("VkDevice created");

    // #region allocation-limit
    // The limit that makes sub-allocation mandatory on some drivers. It counts
    // live vkAllocateMemory allocations, whatever their size; the spec
    // guarantees 4096. Drivers may report much more (a desktop driver may
    // report 2^32 - 1), but every allocation still costs a kernel call.
    let limit = chosen.limits.max_memory_allocation_count;
    println!("\nmaxMemoryAllocationCount: {limit} (the spec guarantees at least 4096)");
    // #endregion

    // #region test-buffers
    // N buffers of the three sizes in turn, all DEVICE_LOCAL. The buffers are
    // created once per path; their requirements are the same every time.
    let create = |i: u32| -> Result<vk::Buffer, vk::Result> {
        let size = SIZES[i as usize % SIZES.len()];
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(USAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: plain create call on a live device.
        unsafe { device.create_buffer(&info, None) }
    };
    // SAFETY: `physical_device` was enumerated from this instance.
    let mem = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let mut reqs = Vec::with_capacity(SIZES.len());
    for i in 0..u32::try_from(SIZES.len())? {
        let probe = create(i)?;
        // SAFETY: `probe` belongs to this device; it is destroyed right after.
        reqs.push(unsafe { device.get_buffer_memory_requirements(probe) });
        // SAFETY: never bound or used.
        unsafe { device.destroy_buffer(probe, None) };
    }
    let memory_type = find_type(
        &mem,
        reqs.first().ok_or("no probe buffer")?.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or("no DEVICE_LOCAL memory type for the buffers")?;
    // Each path holds all N buffers at once, so their sizes must fit the heap
    // behind the type: check before allocating anything (the default fits
    // easily; --count 1000000 asks for about 23 GB). The heap size is only an
    // upper bound, not the budget: the driver and other programs use the heap
    // too, and VK_EXT_memory_budget reports how much of it this process may use.
    let need: vk::DeviceSize = reqs
        .iter()
        .cycle()
        .take(usize::try_from(count)?)
        .map(|r| r.size)
        .sum();
    let heap = mem
        .memory_types_as_slice()
        .get(usize::try_from(memory_type)?)
        .ok_or("memory type out of range")?
        .heap_index;
    let heap_size = mem
        .memory_heaps_as_slice()
        .get(usize::try_from(heap)?)
        .ok_or("memory heap out of range")?
        .size;
    if need > heap_size {
        return Err(format!(
            "--count {count}: the buffers need {need} bytes, but heap {heap} has only {heap_size} bytes (an upper bound, not the budget: see VK_EXT_memory_budget)"
        )
        .into());
    }
    println!(
        "Buffers: {count}, sizes {} / {} / {} bytes in turn, usage VERTEX_BUFFER | INDEX_BUFFER | TRANSFER_DST, DEVICE_LOCAL (memory type {memory_type})",
        SIZES[0], SIZES[1], SIZES[2]
    );
    // #endregion

    // #region raw
    // One vkAllocateMemory per buffer, exactly the size it asks for.
    println!("\nRaw: one vkAllocateMemory per buffer");
    if count > limit {
        println!(
            "Raw: skipped, {count} allocations would exceed maxMemoryAllocationCount ({limit})"
        );
    } else {
        let start = Instant::now();
        // Each handle is recorded as soon as it exists, so the cleanup below
        // also runs after an error midway (the C++ version gets this from RAII).
        let mut buffers = Vec::with_capacity(usize::try_from(count)?);
        let mut memories = Vec::with_capacity(usize::try_from(count)?);
        let mut fill = || -> Result<vk::DeviceSize, Box<dyn Error>> {
            let mut requested: vk::DeviceSize = 0;
            for i in 0..count {
                let buffer = create(i)?;
                buffers.push(buffer);
                // SAFETY: `buffer` belongs to this device.
                let req = unsafe { device.get_buffer_memory_requirements(buffer) };
                requested += req.size;
                let info = vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(memory_type);
                // SAFETY: the type is allowed for the buffer (same create info as the probe).
                let memory = unsafe { device.allocate_memory(&info, None)? };
                memories.push(memory);
                // SAFETY: the allocation fits and offset 0 is aligned.
                unsafe { device.bind_buffer_memory(buffer, memory, 0)? };
            }
            Ok(requested)
        };
        let filled = fill();
        let created = start.elapsed();
        if let Ok(requested) = &filled {
            println!(
                "Raw: {count} buffers, {count} vkAllocateMemory calls = {count} VkDeviceMemory objects, {requested} bytes requested"
            );
        }
        let start = Instant::now();
        // SAFETY: never used by the GPU; each buffer is destroyed and each
        // allocation freed once.
        unsafe {
            for &buffer in &buffers {
                device.destroy_buffer(buffer, None);
            }
            for &memory in &memories {
                device.free_memory(memory, None);
            }
        }
        let destroyed = start.elapsed();
        filled?;
        println!(
            "[timing] raw: create + allocate + bind {}, destroy + free {}",
            per_buffer(created, count),
            per_buffer(destroyed, count)
        );
    }
    // #endregion

    // #region sub-allocate
    // The same buffers through the allocator: each one is bound to a block
    // at the offset the allocator returns.
    println!("\nAllocator: buffers bound at offsets inside big blocks");
    let block_size: vk::DeviceSize = 64 * 1024 * 1024;
    let mut allocator = LinearAllocator::new(memory_type, block_size);
    let start = Instant::now();
    // As in the raw path, every buffer is recorded as soon as it exists, so the
    // cleanup below also runs after an error midway.
    let mut placed = Vec::with_capacity(usize::try_from(count)?);
    let mut offsets = Vec::new();
    let mut fill = || -> Result<vk::DeviceSize, Box<dyn Error>> {
        let mut requested: vk::DeviceSize = 0;
        for i in 0..count {
            let buffer = create(i)?;
            placed.push(buffer);
            // SAFETY: `buffer` belongs to this device.
            let req = unsafe { device.get_buffer_memory_requirements(buffer) };
            requested += req.size;
            let a = allocator.alloc(&device, &req)?;
            // SAFETY: the allocator returned an aligned range inside a block of an allowed type.
            unsafe { device.bind_buffer_memory(buffer, a.memory, a.offset)? };
            if offsets.len() < 4 {
                offsets.push(a.offset.to_string());
            }
        }
        Ok(requested)
    };
    let filled = fill();
    let created = start.elapsed();
    if let Ok(requested) = &filled {
        println!(
            "Allocator: {count} buffers in memory type {memory_type}, {requested} bytes requested"
        );
        println!(
            "[lib] own linear allocator, block size {} MiB: {} vkAllocateMemory call(s), {} block(s) totalling {} bytes, {} bytes used (incl. alignment padding)",
            block_size / (1024 * 1024),
            allocator.allocate_calls,
            allocator.blocks.len(),
            allocator
                .blocks
                .iter()
                .map(|b| b.size)
                .sum::<vk::DeviceSize>(),
            allocator.used()
        );
        println!("[lib] first buffers at offsets {}", offsets.join(", "));
    }
    let start = Instant::now();
    for &buffer in &placed {
        // SAFETY: never used by the GPU; destroyed once.
        unsafe { device.destroy_buffer(buffer, None) };
    }
    allocator.destroy(&device);
    let destroyed = start.elapsed();
    filled?;
    println!(
        "[timing] allocator: create + allocate + bind {}, destroy + free {}",
        per_buffer(created, count),
        per_buffer(destroyed, count)
    );
    // #endregion

    // #region cleanup
    // SAFETY: nothing uses these objects any more; each is destroyed once.
    unsafe {
        device.destroy_device(None);
        debug_utils.destroy_debug_utils_messenger(messenger, None);
        instance.destroy_instance(None);
    }
    println!("\nAll buffers destroyed, all memory freed");
    // #endregion
    Ok(())
}
