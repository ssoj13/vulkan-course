//! Step 13 - Memory heaps and memory types.
//!
//! A buffer is only a description (size, usage); the bytes live in a
//! `VkDeviceMemory` allocation that we bind to it. Where those bytes can live is
//! what the device reports in `VkPhysicalDeviceMemoryProperties`:
//!
//!   heaps = physical pools of memory (VRAM, system RAM, the CPU-visible window of VRAM)
//!   types = ways to use a heap (DEVICE_LOCAL, HOST_VISIBLE, HOST_COHERENT, HOST_CACHED)
//!
//! This step prints both tables, asks the driver what three kinds of buffers
//! need (vkGetBufferMemoryRequirements: size, alignment, allowed types),
//! chooses memory types with the spec's find-memory-type rule, and finally
//! allocates, binds and maps a HOST_VISIBLE buffer and places two buffers in
//! one allocation. No window: steps 01-05 are repeated, reduced to what this
//! step needs.

use std::ffi::{CStr, c_void};

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

/// Flag names as the spec spells them without the `VK_..._BIT` decoration, in
/// bit order; both languages print them from tables like this one, so the
/// output is identical. Unknown bits are printed in hex.
const MEMORY_PROPERTY_NAMES: [(vk::MemoryPropertyFlags, &str); 9] = [
    (vk::MemoryPropertyFlags::DEVICE_LOCAL, "DEVICE_LOCAL"),
    (vk::MemoryPropertyFlags::HOST_VISIBLE, "HOST_VISIBLE"),
    (vk::MemoryPropertyFlags::HOST_COHERENT, "HOST_COHERENT"),
    (vk::MemoryPropertyFlags::HOST_CACHED, "HOST_CACHED"),
    (
        vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
        "LAZILY_ALLOCATED",
    ),
    (vk::MemoryPropertyFlags::PROTECTED, "PROTECTED"),
    (
        vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD,
        "DEVICE_COHERENT_AMD",
    ),
    (
        vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD,
        "DEVICE_UNCACHED_AMD",
    ),
    (vk::MemoryPropertyFlags::RDMA_CAPABLE_NV, "RDMA_CAPABLE_NV"),
];
const HEAP_NAMES: [(vk::MemoryHeapFlags, &str); 2] = [
    (vk::MemoryHeapFlags::DEVICE_LOCAL, "DEVICE_LOCAL"),
    (vk::MemoryHeapFlags::MULTI_INSTANCE, "MULTI_INSTANCE"),
];
const USAGE_NAMES: [(vk::BufferUsageFlags, &str); 8] = [
    (vk::BufferUsageFlags::TRANSFER_SRC, "TRANSFER_SRC"),
    (vk::BufferUsageFlags::TRANSFER_DST, "TRANSFER_DST"),
    (
        vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER,
        "UNIFORM_TEXEL_BUFFER",
    ),
    (
        vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER,
        "STORAGE_TEXEL_BUFFER",
    ),
    (vk::BufferUsageFlags::UNIFORM_BUFFER, "UNIFORM_BUFFER"),
    (vk::BufferUsageFlags::STORAGE_BUFFER, "STORAGE_BUFFER"),
    (vk::BufferUsageFlags::INDEX_BUFFER, "INDEX_BUFFER"),
    (vk::BufferUsageFlags::VERTEX_BUFFER, "VERTEX_BUFFER"),
];

/// "A | B | C" for the set bits of `raw`, "(none)" for zero.
fn names(raw: u32, table: &[(u32, &str)]) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut rest = raw;
    for &(bit, name) in table {
        if raw & bit != 0 {
            parts.push(name.to_owned());
            rest &= !bit;
        }
    }
    if rest != 0 {
        parts.push(format!("{rest:#x}"));
    }
    if parts.is_empty() {
        "(none)".to_owned()
    } else {
        parts.join(" | ")
    }
}
fn memory_names(f: vk::MemoryPropertyFlags) -> String {
    names(
        f.as_raw(),
        &MEMORY_PROPERTY_NAMES.map(|(b, n)| (b.as_raw(), n)),
    )
}
fn heap_names(f: vk::MemoryHeapFlags) -> String {
    names(f.as_raw(), &HEAP_NAMES.map(|(b, n)| (b.as_raw(), n)))
}
fn usage_names(f: vk::BufferUsageFlags) -> String {
    names(f.as_raw(), &USAGE_NAMES.map(|(b, n)| (b.as_raw(), n)))
}

/// "0, 1, 3" for the set bits of a memoryTypeBits mask.
fn type_list(bits: u32) -> String {
    (0..32)
        .filter(|i| bits & (1 << i) != 0)
        .map(|i: u32| i.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Bytes as MiB with one decimal (exact: sizes are integers, 2^20 is a power of two).
fn mib(bytes: vk::DeviceSize) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

// #region find-memory-type
/// The spec's find-memory-type rule (chapter "Device Memory", sample code
/// `findProperties`): the FIRST memory type, in index order, that
/// 1. the resource allows: bit `i` of its memoryTypeBits is set, and
/// 2. has every wanted property flag.
///
/// Drivers must order the types so that this simple loop finds the plainest
/// fitting type: a type whose flags are a strict subset of another's comes
/// first. The interactive labs of this course use the same rule.
fn find_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count)
        .zip(props.memory_types_as_slice())
        .find_map(|(i, t)| (bits & (1 << i) != 0 && t.property_flags.contains(flags)).then_some(i))
}

/// Required flags must be present; preferred ones are nice to have. Search
/// with both first, then with the required ones alone (the same two-pass
/// search as the spec sample). Returns the type and whether the preferred
/// flags were met.
fn choose_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    required: vk::MemoryPropertyFlags,
    preferred: vk::MemoryPropertyFlags,
) -> Option<(u32, bool)> {
    find_type(props, bits, required | preferred)
        .map(|i| (i, true))
        .or_else(|| find_type(props, bits, required).map(|i| (i, false)))
}
// #endregion

/// Rounds `offset` up to a multiple of `alignment` (a power of two in Vulkan).
fn align_up(offset: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize {
    offset.div_ceil(alignment) * alignment
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
    let entry = unsafe { Entry::load()? };

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"Vulkan from Zero - 13 memory types")
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

    // Step 05's device. Buffers and memory are created from it; this step
    // submits no work, but every device needs at least one queue.
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

    // #region memory-properties
    // Heaps are the physical pools; types say how a heap can be used. Each
    // type points at one heap, and several types may share a heap (system RAM
    // appears as plain, HOST_VISIBLE|HOST_COHERENT and +HOST_CACHED here).
    // DEVICE_LOCAL = fastest for the GPU (VRAM on a discrete card);
    // HOST_VISIBLE = the CPU can map it; HOST_COHERENT = CPU writes need no
    // explicit flush; HOST_CACHED = CPU reads go through the CPU cache.
    // SAFETY: `physical_device` was enumerated from this instance.
    let mem = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    println!("\nMemory heaps: {}", mem.memory_heap_count);
    for (i, heap) in mem.memory_heaps_as_slice().iter().enumerate() {
        println!(
            "  heap {i}: {} bytes = {}, {}",
            heap.size,
            mib(heap.size),
            heap_names(heap.flags)
        );
    }
    println!("Memory types: {}", mem.memory_type_count);
    for (i, t) in mem.memory_types_as_slice().iter().enumerate() {
        println!(
            "  type {i}: heap {}, {}",
            t.heap_index,
            memory_names(t.property_flags)
        );
    }
    // #endregion

    // #region memory-limits
    // Limits that matter for memory: how many vkAllocateMemory allocations may
    // exist at once (the spec guarantees at least 4096, which is why real
    // programs sub-allocate, step 16), the alignment of a mapped pointer, the
    // granularity of flushes for non-coherent memory, and how far apart
    // linear (buffers) and optimal-tiling (images) resources must be when they
    // share an allocation.
    let limits = &chosen.limits;
    println!(
        "\nLimits: maxMemoryAllocationCount {}, minMemoryMapAlignment {}, nonCoherentAtomSize {}, bufferImageGranularity {}",
        limits.max_memory_allocation_count,
        limits.min_memory_map_alignment,
        limits.non_coherent_atom_size,
        limits.buffer_image_granularity
    );
    // #endregion

    // #region requirements
    // A buffer is created WITHOUT memory. Its usage flags tell the driver what
    // it will be used for; the driver answers with what the memory must look
    // like: the size to allocate (may be rounded up), the alignment of the
    // offset it is bound at, and memoryTypeBits - bit i set = type i allowed.
    let samples = [
        ("vertex", vk::BufferUsageFlags::VERTEX_BUFFER, 1000),
        ("index", vk::BufferUsageFlags::INDEX_BUFFER, 100),
        ("uniform", vk::BufferUsageFlags::UNIFORM_BUFFER, 100),
        ("storage", vk::BufferUsageFlags::STORAGE_BUFFER, 1000),
        ("staging", vk::BufferUsageFlags::TRANSFER_SRC, 65536),
        (
            "vertex (copy target)",
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            65536,
        ),
    ];
    println!("\nBuffer requirements (vkGetBufferMemoryRequirements):");
    let mut requirements = Vec::with_capacity(samples.len());
    for (label, usage, size) in samples {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: plain create call on a live device; the buffer is destroyed right below.
        let buffer = unsafe { device.create_buffer(&info, None)? };
        // SAFETY: `buffer` belongs to this device.
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        // SAFETY: the buffer was never bound or used.
        unsafe { device.destroy_buffer(buffer, None) };
        println!(
            "  {label:<21} {}, {size} bytes -> size {}, alignment {}, memoryTypeBits {:#b} (types {})",
            usage_names(usage),
            req.size,
            req.alignment,
            req.memory_type_bits,
            type_list(req.memory_type_bits)
        );
        requirements.push(req);
    }
    // #endregion

    // #region choose-types
    // What each kind of data wants, as (required, preferred) flags, matched
    // against the memoryTypeBits of the vertex buffer above:
    // - GPU-only data (meshes uploaded once, step 15): DEVICE_LOCAL;
    // - a staging buffer the CPU fills for a copy: HOST_VISIBLE + HOST_COHERENT;
    // - results the CPU reads back: HOST_VISIBLE, preferably HOST_CACHED
    //   (uncached reads from the CPU are very slow);
    // - small data the CPU rewrites every frame: HOST_VISIBLE + HOST_COHERENT,
    //   preferably also DEVICE_LOCAL (the CPU-visible window of VRAM).
    // LAZILY_ALLOCATED is not in this table: the spec never lets a buffer use a
    // lazily allocated type (memoryTypeBits of a buffer never includes one).
    // Such memory exists only for images created with TRANSIENT_ATTACHMENT
    // usage, which module 10 covers.
    let bits = requirements
        .first()
        .ok_or("no requirements")?
        .memory_type_bits;
    let intents = [
        (
            "GPU-only mesh",
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        ),
        (
            "staging upload",
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::empty(),
        ),
        (
            "readback",
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::HOST_CACHED,
        ),
        (
            "per-frame data",
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ),
    ];
    println!(
        "\nChoosing memory types for memoryTypeBits {bits:#b} (first type in index order with the flags; preferred flags tried first):"
    );
    for (label, required, preferred) in intents {
        let result = match choose_type(&mem, bits, required, preferred) {
            Some((i, met)) => {
                let heap = mem
                    .memory_types_as_slice()
                    .get(usize::try_from(i)?)
                    .ok_or("memory type out of range")?
                    .heap_index;
                let note = if met || preferred.is_empty() {
                    ""
                } else {
                    ", preferred flags not available"
                };
                format!("type {i} (heap {heap}){note}")
            }
            None => "no such type".to_owned(),
        };
        println!(
            "  {label:<21} required {}, preferred {} -> {result}",
            memory_names(required),
            memory_names(preferred)
        );
    }
    println!(
        "  (LAZILY_ALLOCATED: never for buffers, only for images with TRANSIENT_ATTACHMENT usage, module 10)"
    );
    // #endregion

    // #region allocate-and-bind
    // The full manual path for one buffer the CPU writes: create the buffer,
    // ask for its requirements, choose a type, allocate that much memory of
    // that type, bind the buffer at offset 0.
    let upload_size: vk::DeviceSize = 4096;
    let upload_info = vk::BufferCreateInfo::default()
        .size(upload_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: plain create call on a live device.
    let upload = unsafe { device.create_buffer(&upload_info, None)? };
    // SAFETY: `upload` belongs to this device.
    let upload_req = unsafe { device.get_buffer_memory_requirements(upload) };
    let host_flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let upload_type = find_type(&mem, upload_req.memory_type_bits, host_flags)
        .ok_or("no HOST_VISIBLE | HOST_COHERENT memory type for the buffer")?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(upload_req.size)
        .memory_type_index(upload_type);
    // SAFETY: the type index comes from this device's memory properties.
    let upload_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    // SAFETY: the memory is large enough, of an allowed type, and offset 0 is aligned.
    unsafe { device.bind_buffer_memory(upload, upload_memory, 0)? };
    println!(
        "\nUpload buffer: {upload_size} bytes (TRANSFER_SRC), memory type {upload_type}, allocation {} bytes, bound at offset 0",
        upload_req.size
    );
    // #endregion

    // #region map
    // Mapping gives the CPU a pointer into the allocation. The memory is
    // HOST_COHERENT, so plain writes are enough: no vkFlushMappedMemoryRanges
    // (a non-coherent type needs a flush of ranges aligned to
    // nonCoherentAtomSize after writing). Reading back from this type works
    // but is slow: it is uncached (typically write-combined) memory.
    // SAFETY: the memory is HOST_VISIBLE and not mapped yet; the range is inside it.
    let ptr = unsafe {
        device.map_memory(
            upload_memory,
            0,
            vk::WHOLE_SIZE,
            vk::MemoryMapFlags::empty(),
        )?
    };
    let aligned = (ptr as usize).is_multiple_of(limits.min_memory_map_alignment);
    println!(
        "Mapped: pointer aligned to minMemoryMapAlignment ({}): {}",
        limits.min_memory_map_alignment,
        if aligned { "yes" } else { "no" }
    );
    let count = usize::try_from(upload_size)? / size_of::<f32>();
    // SAFETY: the mapping covers `upload_size` bytes, is aligned for f32 (64),
    // and nothing else accesses the memory while this slice lives.
    let floats = unsafe { std::slice::from_raw_parts_mut(ptr.cast::<f32>(), count) };
    for (i, f) in floats.iter_mut().enumerate() {
        *f = i as f32;
    }
    let sum: f64 = floats.iter().map(|&f| f64::from(f)).sum();
    let expected = (count * (count - 1) / 2) as f64;
    println!(
        "Wrote {count} floats through the pointer, read back: sum {sum} (expected {expected})"
    );
    // SAFETY: mapped above; `floats` is not used after this.
    unsafe { device.unmap_memory(upload_memory) };
    println!("Unmapped (HOST_COHERENT: no flush was needed)");
    // #endregion

    // #region two-buffers-one-allocation
    // Nothing forces one allocation per buffer: a buffer is bound at an
    // OFFSET. Two buffers can share an allocation of a type both allow, each
    // at an offset that is a multiple of its own alignment. This is what an
    // allocator does for thousands of buffers (step 16).
    // SAFETY: plain create calls on a live device.
    let vertex = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(1000)
                .usage(vk::BufferUsageFlags::VERTEX_BUFFER),
            None,
        )?
    };
    // SAFETY: as above.
    let index = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(100)
                .usage(vk::BufferUsageFlags::INDEX_BUFFER),
            None,
        )?
    };
    // SAFETY: both buffers belong to this device.
    let (vertex_req, index_req) = unsafe {
        (
            device.get_buffer_memory_requirements(vertex),
            device.get_buffer_memory_requirements(index),
        )
    };
    // One type for both: it must be allowed by both masks (AND them).
    let shared_bits = vertex_req.memory_type_bits & index_req.memory_type_bits;
    let shared_type = find_type(&mem, shared_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .ok_or("no DEVICE_LOCAL type for both buffers")?;
    let index_offset = align_up(vertex_req.size, index_req.alignment);
    let shared_size = index_offset + index_req.size;
    let shared_info = vk::MemoryAllocateInfo::default()
        .allocation_size(shared_size)
        .memory_type_index(shared_type);
    // SAFETY: the type index comes from this device's memory properties.
    let shared_memory = unsafe { device.allocate_memory(&shared_info, None)? };
    // SAFETY: offsets are aligned and both ranges lie inside the allocation.
    unsafe {
        device.bind_buffer_memory(vertex, shared_memory, 0)?;
        device.bind_buffer_memory(index, shared_memory, index_offset)?;
    }
    println!(
        "\nOne allocation, two buffers: memory type {shared_type}, {shared_size} bytes; vertex buffer ({} bytes) at offset 0, index buffer ({} bytes) at offset {index_offset} (alignment {})",
        vertex_req.size, index_req.size, index_req.alignment
    );
    // #endregion

    // #region cleanup
    // Buffers first, then the memory they are bound to (freeing memory that a
    // live buffer is bound to is allowed, but such a buffer may not be used
    // any more; this order keeps it simple). vkFreeMemory also unmaps.
    // SAFETY: nothing uses these objects any more; each is destroyed once.
    unsafe {
        device.destroy_buffer(vertex, None);
        device.destroy_buffer(index, None);
        device.free_memory(shared_memory, None);
        device.destroy_buffer(upload, None);
        device.free_memory(upload_memory, None);
        device.destroy_device(None);
        debug_utils.destroy_debug_utils_messenger(messenger, None);
        instance.destroy_instance(None);
    }
    println!("\nBuffers destroyed, memory freed");
    // #endregion
    Ok(())
}
