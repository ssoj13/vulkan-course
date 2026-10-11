//! Step 15 - Staging uploads.
//!
//! Step 14 left the mesh in system RAM, read by the GPU over PCIe on every
//! draw. The fastest memory for the GPU is DEVICE_LOCAL VRAM, which the CPU
//! usually cannot map. So the CPU writes into a STAGING buffer it can map, and
//! the GPU copies from there into the device-local buffers - like writing a
//! cache file to local scratch before the farm renders from it:
//!
//!   CPU memcpy -> staging (HOST_VISIBLE) -> vkCmdCopyBuffer -> vertex/index buffers (DEVICE_LOCAL)
//!
//! Three upload paths, chosen on the command line:
//! - default: the copy runs on the graphics queue, followed by a barrier;
//! - `--transfer-queue`: the copy runs on a queue of the dedicated transfer
//!   family (the GPU's copy engines); the buffers then change owner from the
//!   transfer family to the graphics family (release + acquire barriers,
//!   ordered by a semaphore);
//! - `--bar`: no staging at all: the buffers live in the memory type that is
//!   DEVICE_LOCAL and HOST_VISIBLE (the CPU-visible window of VRAM, 214 MiB on
//!   the course GPU with Resizable BAR off) and the CPU writes them directly.
//!
//! `--mib N` pads the vertex buffer to N MiB (zeros after the mesh) so the
//! timing shows bandwidth, not just latency. The upload is timed (`[timing]`
//! lines); then step 14's frame loop draws the mesh, `--seconds S` closes the
//! window. Steps 01-14 are repeated, reduced to what this step needs.

use std::error::Error;
use std::ffi::{CStr, c_void};
use std::time::{Duration, Instant};

use ash::{
    Entry,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk::{self, TaggedStructure as _},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _},
    window::{Window, WindowId},
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

/// Step 10's measurements showed that 2 frames in flight keep both the CPU
/// and the GPU busy; from here on it is fixed.
const FRAMES_IN_FLIGHT: usize = 2;

// #region vertex
/// One vertex as it lies in the vertex buffer: 2 + 3 floats = 20 bytes, no
/// padding. `repr(C)` fixes the field order and offsets, which the vertex
/// input state of the pipeline repeats (location 0 at offset 0, location 1
/// at offset 8) and shaders/mesh.slang declares.
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    /// Clip-space x, y (Vulkan: +y points down).
    position: [f32; 2],
    /// Linear RGB.
    color: [f32; 3],
}
const _: () = assert!(size_of::<Vertex>() == 20);
// #endregion

// #region mesh
/// A hexagon: the center plus 6 corners = 7 vertices, 6 triangles = 18 indices.
/// Corner i sits at 60 degrees * i on an ellipse 0.45 wide and 0.6 tall: the
/// window is 800x600, so x is scaled by 600/800 to keep the hexagon regular on
/// screen. Angles grow from +x towards +y, which is DOWN on screen, so each
/// triangle (center, corner i, corner i+1) is clockwise - the front face of
/// the pipeline (back faces are culled).
#[rustfmt::skip] // one vertex per line reads like a table
const VERTICES: [Vertex; 7] = [
    Vertex { position: [0.0, 0.0],             color: [1.0, 1.0, 1.0] }, // center, white
    Vertex { position: [0.45, 0.0],            color: [1.0, 0.0, 0.0] }, // 0 deg, red
    Vertex { position: [0.225, 0.519_615_2],   color: [1.0, 1.0, 0.0] }, // 60 deg, yellow
    Vertex { position: [-0.225, 0.519_615_2],  color: [0.0, 1.0, 0.0] }, // 120 deg, green
    Vertex { position: [-0.45, 0.0],           color: [0.0, 1.0, 1.0] }, // 180 deg, cyan
    Vertex { position: [-0.225, -0.519_615_2], color: [0.0, 0.0, 1.0] }, // 240 deg, blue
    Vertex { position: [0.225, -0.519_615_2],  color: [1.0, 0.0, 1.0] }, // 300 deg, magenta
];
/// Three indices per triangle; vertex 0 (the center) is used six times but
/// stored once. 16-bit indices are enough for up to 65536 vertices.
const INDICES: [u16; 18] = [
    0, 1, 2, //
    0, 2, 3, //
    0, 3, 4, //
    0, 4, 5, //
    0, 5, 6, //
    0, 6, 1, //
];
// #endregion

/// Memory property names for printing, in bit order (step 13).
fn memory_names(flags: vk::MemoryPropertyFlags) -> String {
    let table = [
        (vk::MemoryPropertyFlags::DEVICE_LOCAL, "DEVICE_LOCAL"),
        (vk::MemoryPropertyFlags::HOST_VISIBLE, "HOST_VISIBLE"),
        (vk::MemoryPropertyFlags::HOST_COHERENT, "HOST_COHERENT"),
        (vk::MemoryPropertyFlags::HOST_CACHED, "HOST_CACHED"),
    ];
    let parts: Vec<&str> = table
        .iter()
        .filter(|(bit, _)| flags.contains(*bit))
        .map(|&(_, name)| name)
        .collect();
    if parts.is_empty() {
        "(none)".to_owned()
    } else {
        parts.join(" | ")
    }
}

// #region find-memory-type
/// Step 13's find-memory-type rule: the first memory type, in index order,
/// that `bits` (the resource's memoryTypeBits) allows and that has every flag.
fn find_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count)
        .zip(props.memory_types_as_slice())
        .find_map(|(i, t)| (bits & (1 << i) != 0 && t.property_flags.contains(flags)).then_some(i))
}
// #endregion

// #region create-buffer
/// A buffer and the memory bound to it. One allocation per buffer is fine
/// for a few buffers; step 16 shows why and how real programs sub-allocate.
struct GpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    memory_type: u32,
}

/// Step 13's manual path: create the buffer, query its requirements, take the
/// first memory type with `flags`, allocate, bind at offset 0.
fn create_buffer(
    device: &ash::Device,
    props: &vk::PhysicalDeviceMemoryProperties,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    flags: vk::MemoryPropertyFlags,
) -> Result<GpuBuffer, Box<dyn Error>> {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: plain create call on a live device.
    let buffer = unsafe { device.create_buffer(&info, None)? };
    // SAFETY: `buffer` belongs to this device.
    let req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type = find_type(props, req.memory_type_bits, flags).ok_or_else(|| {
        format!(
            "no memory type with {} for this buffer",
            memory_names(flags)
        )
    })?;
    // An allocation larger than its heap is invalid
    // (VUID-vkAllocateMemory-pAllocateInfo-01713), not just "out of memory":
    // check it here, which matters for `--bar --mib N` (the BAR heap is small).
    let heap = props
        .memory_types_as_slice()
        .get(usize::try_from(memory_type)?)
        .and_then(|t| {
            props
                .memory_heaps_as_slice()
                .get(usize::try_from(t.heap_index).ok()?)
        })
        .ok_or("memory type without a heap")?;
    if req.size > heap.size {
        return Err(format!(
            "{} bytes do not fit memory type {memory_type}: its heap has {} bytes",
            req.size, heap.size
        )
        .into());
    }
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(memory_type);
    // SAFETY: the type index comes from this device's memory properties.
    let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    // SAFETY: the allocation has the required size and type; offset 0 is aligned.
    unsafe { device.bind_buffer_memory(buffer, memory, 0)? };
    Ok(GpuBuffer {
        buffer,
        memory,
        memory_type,
    })
}

/// Copies byte blocks into a HOST_VISIBLE | HOST_COHERENT allocation, each at
/// its offset: map, memcpy, unmap. Coherent memory needs no flush: host writes
/// to it are available in the host domain at once, and vkQueueSubmit makes
/// them visible to all device accesses of every command submitted after this.
fn write_mapped(
    device: &ash::Device,
    memory: vk::DeviceMemory,
    blocks: &[(usize, &[u8])],
) -> Result<(), vk::Result> {
    // SAFETY: the memory is HOST_VISIBLE, not mapped yet, and covers every
    // block (the callers size it for them); nothing else touches it meanwhile.
    unsafe {
        let ptr = device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
            .cast::<u8>();
        for &(offset, bytes) in blocks {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(offset), bytes.len());
        }
        device.unmap_memory(memory);
    }
    Ok(())
}
// #endregion

/// The bytes of plain data (`Vertex`, `u16`): repr(C) types without padding.
fn as_bytes<T: Copy>(data: &[T]) -> &[u8] {
    // SAFETY: every byte of a padding-free Copy type is initialized; the slice
    // covers exactly the memory of `data` and borrows it.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) }
}

// #region cli
/// How the mesh gets into GPU memory (see the file header).
#[derive(Clone, Copy, PartialEq)]
enum Upload {
    GraphicsQueue,
    TransferQueue,
    Bar,
}

/// Command line: `--transfer-queue` or `--bar` (default: the graphics queue),
/// `--mib N` (pad the vertex buffer to N MiB, 0-1024), `--seconds S`.
#[derive(Clone, Copy)]
struct Args {
    upload: Upload,
    mib: u32,
    seconds: Option<f64>,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut args = Self {
            upload: Upload::GraphicsQueue,
            mib: 0,
            seconds: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            let mut value = || it.next().ok_or(format!("{arg} needs a value"));
            let upload = match arg.as_str() {
                "--transfer-queue" => Upload::TransferQueue,
                "--bar" => Upload::Bar,
                "--mib" => {
                    args.mib = value()?.parse()?;
                    continue;
                }
                "--seconds" => {
                    args.seconds = Some(value()?.parse()?);
                    continue;
                }
                _ => {
                    return Err(format!(
                        "unknown argument {arg} (use --transfer-queue | --bar, --mib N, --seconds S)"
                    )
                    .into());
                }
            };
            if args.upload != Upload::GraphicsQueue {
                return Err("--transfer-queue and --bar exclude each other".into());
            }
            args.upload = upload;
        }
        if args.mib > 1024 {
            return Err("--mib must be 0-1024".into());
        }
        Ok(args)
    }
}
// #endregion

// #region frame-resources
/// What the CPU touches while recording one frame: one set per frame in
/// flight (step 10).
struct Frame {
    command_buffer: vk::CommandBuffer,
    /// Signaled by vkAcquireNextImageKHR, waited on by this slot's submit.
    image_available: vk::Semaphore,
    /// Signaled when this slot's submission completes.
    in_flight: vk::Fence,
}
// #endregion

const MIB: usize = 1024 * 1024;

/// Megabytes per second (10^6 bytes, like disk and PCIe figures) for printing.
fn mb_per_s(bytes: usize, time: Duration) -> f64 {
    bytes as f64 / 1e6 / time.as_secs_f64().max(1e-9)
}

/// `--bar`: no staging, no copy. The buffers live in a DEVICE_LOCAL memory
/// type the CPU can also map, and the CPU writes VRAM directly through the
/// PCIe BAR window. With Resizable BAR off that window is a 256 MiB BAR, of
/// which the driver exposes 214 MiB as heap 2 on the course GPU; the writes are uncached
/// (typically write-combined), so they are fast to write and very slow to read back.
fn upload_bar(
    device: &ash::Device,
    mem: &vk::PhysicalDeviceMemoryProperties,
    vertex_data: &[u8],
    index_data: &[u8],
) -> Result<(GpuBuffer, GpuBuffer), Box<dyn Error>> {
    // #region bar-write
    println!(
        "Upload path: direct CPU writes into DEVICE_LOCAL | HOST_VISIBLE memory (no staging buffer, no copy)"
    );
    let flags = vk::MemoryPropertyFlags::DEVICE_LOCAL
        | vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT;
    // Both buffers come from the same small heap, and create_buffer checks one
    // allocation at a time: check their sum before allocating anything.
    // vkGetDeviceBufferMemoryRequirements (Vulkan 1.3) gives a buffer's
    // requirements without creating it. The heap size is only an upper bound,
    // not the budget: the driver and other programs use the heap too, and
    // VK_EXT_memory_budget reports how much of it this process may use.
    let need = |len: usize, usage| -> Result<(u32, vk::DeviceSize), Box<dyn Error>> {
        let info = vk::BufferCreateInfo::default()
            .size(vk::DeviceSize::try_from(len)?)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let mut req = vk::MemoryRequirements2::default();
        // SAFETY: plain query on a live device; `info` outlives the call.
        unsafe {
            device.get_device_buffer_memory_requirements(
                &vk::DeviceBufferMemoryRequirements::default().create_info(&info),
                &mut req,
            );
        }
        let req = req.memory_requirements;
        let memory_type = find_type(mem, req.memory_type_bits, flags).ok_or_else(|| {
            format!(
                "no memory type with {} for this buffer",
                memory_names(flags)
            )
        })?;
        let heap = mem
            .memory_types_as_slice()
            .get(usize::try_from(memory_type)?)
            .ok_or("memory type out of range")?
            .heap_index;
        Ok((heap, req.size))
    };
    let (vertex_heap, vertex_need) = need(vertex_data.len(), vk::BufferUsageFlags::VERTEX_BUFFER)?;
    let (index_heap, index_need) = need(index_data.len(), vk::BufferUsageFlags::INDEX_BUFFER)?;
    if vertex_heap == index_heap {
        let heap_size = mem
            .memory_heaps_as_slice()
            .get(usize::try_from(vertex_heap)?)
            .ok_or("memory heap out of range")?
            .size;
        let total = vertex_need + index_need;
        if total > heap_size {
            return Err(format!(
                "vertex + index buffers need {total} bytes, but heap {vertex_heap} has only {heap_size} bytes (an upper bound, not the budget: see VK_EXT_memory_budget)"
            )
            .into());
        }
    }
    let vertex_buffer = create_buffer(
        device,
        mem,
        vk::DeviceSize::try_from(vertex_data.len())?,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        flags,
    )?;
    let index_buffer = create_buffer(
        device,
        mem,
        vk::DeviceSize::try_from(index_data.len())?,
        vk::BufferUsageFlags::INDEX_BUFFER,
        flags,
    )?;
    let heap = mem
        .memory_types_as_slice()
        .get(usize::try_from(vertex_buffer.memory_type)?)
        .ok_or("memory type out of range")?
        .heap_index;
    let start = Instant::now();
    write_mapped(device, vertex_buffer.memory, &[(0, vertex_data)])?;
    write_mapped(device, index_buffer.memory, &[(0, index_data)])?;
    let cpu = start.elapsed();
    println!(
        "Vertex + index buffers: memory type {} ({}), heap {heap}",
        vertex_buffer.memory_type,
        memory_names(flags)
    );
    let total = vertex_data.len() + index_data.len();
    println!(
        "[timing] upload of {total} bytes: CPU writes {:.3} ms ({:.0} MB/s) straight into VRAM, no copy",
        cpu.as_secs_f64() * 1e3,
        mb_per_s(total, cpu)
    );
    // #endregion
    Ok((vertex_buffer, index_buffer))
}

/// The objects `upload_staged` creates. Dropping it waits for the device to go
/// idle, then destroys them: if a `?` returns early after a submit, the GPU may
/// still use them (step 10's rule; the C++ version gets the same from its
/// IdleGuard and RAII). On success the fences were already waited, so the wait
/// returns at once, and `targets` is emptied first: those buffers are the
/// result.
struct UploadScratch<'a> {
    device: &'a ash::Device,
    /// The staging buffer and its memory.
    staging: Option<(vk::Buffer, vk::DeviceMemory)>,
    /// The vertex and index buffers, destroyed only on an error.
    targets: Vec<(vk::Buffer, vk::DeviceMemory)>,
    /// Command pools; destroying one frees its command buffers.
    pools: Vec<vk::CommandPool>,
    fences: Vec<vk::Fence>,
    semaphores: Vec<vk::Semaphore>,
}

impl Drop for UploadScratch<'_> {
    fn drop(&mut self) {
        // SAFETY: after the idle wait no submitted work uses these objects;
        // each was created on `device` and is destroyed exactly once.
        unsafe {
            if let Err(e) = self.device.device_wait_idle() {
                eprintln!("device_wait_idle failed: {e}");
            }
            for &fence in &self.fences {
                self.device.destroy_fence(fence, None);
            }
            for &semaphore in &self.semaphores {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &pool in &self.pools {
                self.device.destroy_command_pool(pool, None);
            }
            for &(buffer, memory) in self.staging.iter().chain(&self.targets) {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
        }
    }
}

/// Default and `--transfer-queue`: CPU -> staging buffer -> vkCmdCopyBuffer
/// -> DEVICE_LOCAL buffers. `graphics` is the family and queue that will draw;
/// `transfer` the dedicated transfer family and its queue, if used.
fn upload_staged(
    device: &ash::Device,
    mem: &vk::PhysicalDeviceMemoryProperties,
    vertex_data: &[u8],
    index_data: &[u8],
    graphics: (u32, vk::Queue),
    transfer: Option<(u32, vk::Queue)>,
) -> Result<(GpuBuffer, GpuBuffer), Box<dyn Error>> {
    let (graphics_family, graphics_queue) = graphics;
    let (copy_family, copy_queue) = transfer.unwrap_or(graphics);
    match transfer {
        None => println!(
            "Upload path: staging buffer + vkCmdCopyBuffer on the graphics queue (family {graphics_family}), then a pipeline barrier"
        ),
        Some((tf, _)) => println!(
            "Upload path: staging buffer + vkCmdCopyBuffer on the transfer queue (family {tf}), then an ownership transfer {tf} -> {graphics_family} (release, semaphore, acquire)"
        ),
    }
    let mut scratch = UploadScratch {
        device,
        staging: None,
        targets: Vec::new(),
        pools: Vec::new(),
        fences: Vec::new(),
        semaphores: Vec::new(),
    };

    // #region staging-buffer
    // The staging buffer: memory the CPU can map, used only as a copy source.
    // Vertices at offset 0, indices right after them (copy offsets need no
    // particular alignment).
    let vertex_size = vk::DeviceSize::try_from(vertex_data.len())?;
    let index_size = vk::DeviceSize::try_from(index_data.len())?;
    let host_flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let staging = create_buffer(
        device,
        mem,
        vertex_size + index_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        host_flags,
    )?;
    scratch.staging = Some((staging.buffer, staging.memory));
    let cpu_start = Instant::now();
    write_mapped(
        device,
        staging.memory,
        &[(0, vertex_data), (vertex_data.len(), index_data)],
    )?;
    let cpu = cpu_start.elapsed();
    println!(
        "Staging buffer: {} bytes, TRANSFER_SRC, memory type {} ({}); vertices at offset 0, indices at offset {vertex_size}",
        vertex_size + index_size,
        staging.memory_type,
        memory_names(host_flags)
    );
    // #endregion

    // #region device-local-buffers
    // The real buffers: DEVICE_LOCAL (VRAM), which the CPU cannot map here,
    // with TRANSFER_DST added to their usage so a copy may write them.
    let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    let vertex_buffer = create_buffer(
        device,
        mem,
        vertex_size,
        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        local,
    )?;
    scratch
        .targets
        .push((vertex_buffer.buffer, vertex_buffer.memory));
    let index_buffer = create_buffer(
        device,
        mem,
        index_size,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        local,
    )?;
    scratch
        .targets
        .push((index_buffer.buffer, index_buffer.memory));
    println!(
        "Vertex buffer: {vertex_size} bytes, VERTEX_BUFFER | TRANSFER_DST, memory type {} ({})",
        vertex_buffer.memory_type,
        memory_names(local)
    );
    println!(
        "Index buffer: {index_size} bytes, INDEX_BUFFER | TRANSFER_DST, memory type {} ({})",
        index_buffer.memory_type,
        memory_names(local)
    );
    // #endregion

    // #region record-copy
    // A one-time command buffer from a TRANSIENT pool (a hint: its buffers
    // live briefly) of the family that runs the copy. Two copy regions: the
    // staging bytes [0, vertex_size) to the vertex buffer, the rest to the
    // index buffer. The host writes above need no barrier: the submit makes
    // them visible to the device.
    let copy_start = Instant::now();
    let new_pool = |family: u32| {
        let info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(family);
        // SAFETY: plain create call on a live device.
        unsafe { device.create_command_pool(&info, None) }
    };
    let new_cmd = |pool: vk::CommandPool| -> Result<vk::CommandBuffer, vk::Result> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `pool` belongs to this device; one buffer was requested.
        let cmd = unsafe { device.allocate_command_buffers(&info)? }
            .first()
            .copied()
            .ok_or(vk::Result::ERROR_UNKNOWN)?;
        // SAFETY: a fresh command buffer is in the initial state.
        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        Ok(cmd)
    };
    let copy_pool = new_pool(copy_family)?;
    scratch.pools.push(copy_pool);
    let cmd = new_cmd(copy_pool)?;
    // SAFETY: `cmd` is recording; the regions lie inside both buffers.
    unsafe {
        device.cmd_copy_buffer(
            cmd,
            staging.buffer,
            vertex_buffer.buffer,
            &[vk::BufferCopy::default().size(vertex_size)],
        );
        device.cmd_copy_buffer(
            cmd,
            staging.buffer,
            index_buffer.buffer,
            &[vk::BufferCopy::default()
                .src_offset(vertex_size)
                .size(index_size)],
        );
    }
    // #endregion

    let fence_info = vk::FenceCreateInfo::default();
    match transfer {
        None => {
            // #region copy-barrier
            // Same queue: one global memory barrier orders the copy's writes
            // (COPY stage, TRANSFER_WRITE) before every later read of vertex
            // attributes and indices on this queue, including the draws of
            // command buffers submitted later.
            // The fence waited below does not replace it: a fence signal makes
            // the copy's writes AVAILABLE, not VISIBLE; only a barrier like
            // this (or a semaphore wait, whose second access scope covers all
            // device accesses) makes them visible to vertex input. Without it
            // nothing complains: sync validation is off by default, and a host
            // fence wait closes the batch's hazards for it. The memory model
            // still requires the barrier: without it the writes are never made
            // visible to vertex input, and the draws may read stale data.
            let barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
                        | vk::PipelineStageFlags2::INDEX_INPUT,
                )
                .dst_access_mask(
                    vk::AccessFlags2::VERTEX_ATTRIBUTE_READ | vk::AccessFlags2::INDEX_READ,
                );
            // SAFETY: plain create call on a live device.
            let fence = unsafe { device.create_fence(&fence_info, None)? };
            scratch.fences.push(fence);
            let cmd_info = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let submit = vk::SubmitInfo2::default().command_buffer_infos(&cmd_info);
            // SAFETY: `cmd` is recording; the fence is new and unsignaled.
            unsafe {
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier)),
                );
                device.end_command_buffer(cmd)?;
                device.queue_submit2(graphics_queue, &[submit], fence)?;
            }
            // #endregion
        }
        Some((tf, _)) => {
            // #region release-ownership
            // The buffers are EXCLUSIVE: in a well-defined state for one queue
            // family at a time. The transfer family RELEASES them to the
            // graphics family: a buffer barrier with src/dst family indices,
            // after the copy's writes (its dst stage and access are ignored).
            // The semaphore orders release before acquire; its signal and wait
            // must use ALL_COMMANDS unless the barriers set
            // VK_DEPENDENCY_QUEUE_FAMILY_OWNERSHIP_TRANSFER_USE_ALL_STAGES_BIT_KHR
            // (allowed only with the maintenance8 feature) (spec, "Queue Family
            // Ownership Transfer").
            // The semaphore wait alone already makes the copy's writes visible
            // (its second access scope covers all device accesses), yet the
            // release/acquire pair is still required: "If memory dependencies
            // are correctly expressed between uses of such a resource between
            // two queues in different families, but no ownership transfer is
            // defined, the contents of that resource are undefined for any
            // read accesses performed by the second queue family" (spec,
            // "Queue Family Ownership Transfer").
            let ownership = |buffer: vk::Buffer| {
                vk::BufferMemoryBarrier2::default()
                    .src_queue_family_index(tf)
                    .dst_queue_family_index(graphics_family)
                    .buffer(buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
            };
            let release = [vertex_buffer.buffer, index_buffer.buffer].map(|b| {
                ownership(b)
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            });
            // SAFETY: plain create call on a live device.
            let sem =
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };
            scratch.semaphores.push(sem);
            // SAFETY: as above.
            let fence = unsafe { device.create_fence(&fence_info, None)? };
            scratch.fences.push(fence);
            let signal = [vk::SemaphoreSubmitInfo::default()
                .semaphore(sem)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let cmd_info = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let submit = vk::SubmitInfo2::default()
                .command_buffer_infos(&cmd_info)
                .signal_semaphore_infos(&signal);
            // SAFETY: `cmd` is recording; the queue belongs to `tf`, the
            // family of the pool `cmd` came from.
            unsafe {
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&release),
                );
                device.end_command_buffer(cmd)?;
                device.queue_submit2(copy_queue, &[submit], fence)?;
            }
            // #endregion

            // #region acquire-ownership
            // The graphics family ACQUIRES the buffers with matching barriers
            // (same buffers, ranges and family indices) in a command buffer of
            // its own, submitted after waiting on the semaphore. Its src stage
            // and access are ignored; dst = the vertex input reads that follow.
            let acquire = [
                ownership(vertex_buffer.buffer)
                    .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT)
                    .dst_access_mask(vk::AccessFlags2::VERTEX_ATTRIBUTE_READ),
                ownership(index_buffer.buffer)
                    .dst_stage_mask(vk::PipelineStageFlags2::INDEX_INPUT)
                    .dst_access_mask(vk::AccessFlags2::INDEX_READ),
            ];
            let pool = new_pool(graphics_family)?;
            scratch.pools.push(pool);
            let acquire_cmd = new_cmd(pool)?;
            // SAFETY: plain create call on a live device.
            let fence = unsafe { device.create_fence(&fence_info, None)? };
            scratch.fences.push(fence);
            let wait = [vk::SemaphoreSubmitInfo::default()
                .semaphore(sem)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let cmd_info = [vk::CommandBufferSubmitInfo::default().command_buffer(acquire_cmd)];
            let submit = vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait)
                .command_buffer_infos(&cmd_info);
            // SAFETY: `acquire_cmd` is recording and comes from a graphics-family pool.
            unsafe {
                device.cmd_pipeline_barrier2(
                    acquire_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&acquire),
                );
                device.end_command_buffer(acquire_cmd)?;
                device.queue_submit2(graphics_queue, &[submit], fence)?;
            }
            // #endregion
        }
    }

    // #region finish-upload
    // Wait for the copy (and the acquire) to complete; only then may the
    // staging buffer, the semaphore and the pools go. A real engine would not
    // block here: it keeps a staging ring and checks a fence or a timeline
    // value later (step 12), and the frame that first draws the mesh waits on
    // the upload's semaphore instead.
    // The fences say the work is complete, so the objects may be destroyed;
    // they make nothing visible to the draws: that is the job of the barrier
    // (or the acquire) above.
    // SAFETY: the fences belong to this device and were submitted above.
    unsafe { device.wait_for_fences(&scratch.fences, true, u64::MAX)? };
    let copy = copy_start.elapsed();
    // The vertex and index buffers are the result; `scratch` destroys the rest
    // (fences, semaphore, pools with their command buffers, staging buffer).
    scratch.targets.clear();
    drop(scratch);
    println!("Upload complete (fences waited); staging buffer destroyed");
    let total = vertex_data.len() + index_data.len();
    println!(
        "[timing] upload of {total} bytes: CPU writes {:.3} ms ({:.0} MB/s), GPU copy {:.3} ms ({:.0} MB/s; record + submit + wait)",
        cpu.as_secs_f64() * 1e3,
        mb_per_s(total, cpu),
        copy.as_secs_f64() * 1e3,
        mb_per_s(total, copy)
    );
    // #endregion
    Ok((vertex_buffer, index_buffer))
}
/// Everything this step creates. `window` is declared last so it is dropped
/// after `destroy` has destroyed the surface that refers to it.
struct Vk {
    /// The loaded Vulkan loader DLL; must outlive every Vulkan object.
    _entry: Entry,
    instance: ash::Instance,
    debug_utils: debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,
    device: ash::Device,
    swapchain_loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    image_views: Vec<vk::ImageView>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    vertex_buffer: GpuBuffer,
    index_buffer: GpuBuffer,
    queue: vk::Queue,
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    command_pool: vk::CommandPool,
    frames: Vec<Frame>,
    frame_index: usize,
    /// One per swapchain image, indexed by the acquired image index.
    render_finished: Vec<vk::Semaphore>,
    start: Instant,
    window: Window,
}

impl Vk {
    /// Steps 01-14, then the mesh upload chosen by `args.upload`.
    fn new(event_loop: &ActiveEventLoop, args: Args) -> Result<Self, Box<dyn Error>> {
        // Window from step 06: 800x600 physical pixels, fixed size.
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("Vulkan from Zero - 15 staging")
                .with_inner_size(PhysicalSize::new(800, 600))
                .with_resizable(false),
        )?;
        let size = window.inner_size();
        println!("Window: {}x{} pixels", size.width, size.height);

        // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vulkan from Zero - 15 staging")
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

        // Surface extensions from step 06.
        let display = event_loop.display_handle()?.as_raw();
        let mut ext_ptrs = ash_window::enumerate_required_extensions(display)?.to_vec();
        ext_ptrs.push(debug_utils::NAME.as_ptr());

        let layer_ptrs = [validation_layer.as_ptr()];
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

        // Surface from step 06.
        let surface_factory = ash_window::SurfaceFactory::new(&entry, &instance, display)?;
        // SAFETY: the window handle is valid, and `Vk` destroys the surface
        // before the window and the instance (see `destroy`).
        let surface =
            unsafe { surface_factory.create_surface(window.window_handle()?.as_raw(), None)? };
        let surface_loader = surface::Instance::load(&entry, &instance);

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

        // Steps 03 + 06: Vulkan 1.4, dynamic rendering + synchronization2, a
        // graphics + present family, VK_KHR_swapchain, discrete first.
        let mut best: Option<(u32, vk::PhysicalDevice, u32)> = None;
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
            // SAFETY: `pd` was enumerated from this instance.
            let exts = unsafe { instance.enumerate_device_extension_properties(pd)? };
            if !exts
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(swapchain::NAME))
            {
                continue;
            }
            // SAFETY: `pd` was enumerated from this instance.
            let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            let mut family = None;
            for (i, f) in families.iter().enumerate() {
                let index = u32::try_from(i)?;
                // SAFETY: `pd` and `surface` belong to this instance; `index` is a valid family.
                let present = unsafe {
                    surface_loader.get_physical_device_surface_support(pd, index, surface)?
                };
                if f.queue_flags.contains(vk::QueueFlags::GRAPHICS) && present {
                    family = Some(index);
                    break;
                }
            }
            let Some(family) = family else {
                continue;
            };
            let score = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 10,
                _ => 1,
            };
            if best.is_none_or(|(s, _, _)| score > s) {
                best = Some((score, pd, family));
            }
        }
        let (_, physical_device, family) = best.ok_or(
            "No Vulkan 1.4 GPU with dynamic rendering + synchronization2 + VK_KHR_swapchain can present to this window",
        )?;
        // SAFETY: `physical_device` was enumerated from this instance.
        let chosen = unsafe { instance.get_physical_device_properties(physical_device) };
        println!(
            "\nPicked: {}",
            chosen.device_name_as_c_str()?.to_string_lossy()
        );
        println!("Graphics + present family: {family}");

        // #region transfer-family
        // Step 04's dedicated transfer family: TRANSFER without GRAPHICS and
        // COMPUTE, the front end of the GPU's copy (DMA) engines. A copy on its
        // queue can run while the graphics queue renders.
        // SAFETY: `physical_device` was enumerated from this instance.
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let dedicated = families
            .iter()
            .position(|f| {
                f.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !f
                        .queue_flags
                        .intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            })
            .map(u32::try_from)
            .transpose()?;
        match dedicated {
            Some(i) => println!("Dedicated transfer family: {i}"),
            None => println!("No dedicated transfer family"),
        }
        let transfer_family = match args.upload {
            Upload::TransferQueue => Some(
                dedicated.ok_or("--transfer-queue: this GPU has no dedicated transfer family")?,
            ),
            _ => None,
        };
        // #endregion

        // Step 07's device, plus one queue of the transfer family when the
        // upload uses it: queues exist only if requested at device creation.
        let priorities = [1.0_f32];
        let mut queue_infos = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities),
        ];
        if let Some(tf) = transfer_family {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(tf)
                    .queue_priorities(&priorities),
            );
        }
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let device_exts = [swapchain::NAME.as_ptr()];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_exts)
            .push(&mut features13);
        // SAFETY: `physical_device` belongs to `instance`, and all create-info data outlives the call.
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        let swapchain_loader = swapchain::Device::load(&instance, &device);
        println!(
            "VkDevice created with {} and {} queue(s)",
            swapchain::NAME.to_string_lossy(),
            queue_infos.len()
        );

        // Step 07's swapchain: B8G8R8A8_SRGB + SRGB_NONLINEAR if offered,
        // MAILBOX if offered (else FIFO), the window's extent, min + 1 images.
        // SAFETY: `physical_device` and `surface` belong to this instance.
        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?
        };
        let format = formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or(formats.first())
            .copied()
            .ok_or("The surface reports no formats")?;
        // SAFETY: as above.
        let modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
        };
        let present_mode = if modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };
        // SAFETY: as above.
        let caps = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
        };
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: size
                    .width
                    .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: size
                    .height
                    .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            image_count = image_count.min(caps.max_image_count);
        }
        if !caps
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
        {
            return Err("The surface does not support OPAQUE composite alpha".into());
        }
        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);
        // SAFETY: `swapchain_info` refers to a live surface of this instance.
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_info, None)? };
        // SAFETY: `swapchain` is alive.
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let mut image_views = Vec::with_capacity(images.len());
        for &image in &images {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            // SAFETY: `image` belongs to this device's swapchain.
            image_views.push(unsafe { device.create_image_view(&view_info, None)? });
        }
        println!(
            "Swapchain: {}x{} {:?}, {present_mode:?}, {} images",
            extent.width,
            extent.height,
            format.format,
            images.len()
        );

        // The SPIR-V is compiled from shaders/mesh.slang by `bun run sync:shaders`
        // and embedded at build time (step 08 explains `read_spv`).
        const MESH_SPV: &[u8] = include_bytes!("../../../../shaders/spv/mesh.spv");
        let spirv = ash::util::read_spv(&mut std::io::Cursor::new(MESH_SPV))?;
        let module_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
        // SAFETY: `spirv` is valid SPIR-V (checked by spirv-val in sync:shaders).
        let shader_module = unsafe { device.create_shader_module(&module_info, None)? };
        println!(
            "\nSPIR-V: {} bytes (mesh.spv), entry points vertexMain, fragmentMain",
            MESH_SPV.len()
        );
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shader_module)
                .name(c"vertexMain"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shader_module)
                .name(c"fragmentMain"),
        ];

        // #region vertex-input
        // How the input assembler turns buffer bytes into shader inputs.
        // A BINDING is a vertex buffer slot: here slot 0, advancing `stride`
        // bytes per vertex (VERTEX rate; INSTANCE rate would advance per
        // instance). An ATTRIBUTE is one shader input: its location (the
        // `vk::location` in mesh.slang), the binding it reads from, the format
        // of the data and its offset inside one vertex.
        let stride = u32::try_from(size_of::<Vertex>())?;
        let bindings = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(stride)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let attributes = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(u32::try_from(std::mem::offset_of!(Vertex, position))?),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(u32::try_from(std::mem::offset_of!(Vertex, color))?),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        println!("Vertex input: binding 0, stride {stride} bytes, per vertex");
        for a in &attributes {
            println!(
                "  location {}: {:?} at offset {}",
                a.location, a.format, a.offset
            );
        }
        // #endregion

        // The rest of step 08's pipeline, unchanged.
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let layout_info = vk::PipelineLayoutCreateInfo::default();
        // SAFETY: plain create call on a live device.
        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };
        let color_formats = [format.format];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .push(&mut rendering_info);
        // SAFETY: all create-info data outlives the call; no pipeline cache yet.
        let pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| e)?
        };
        let pipeline = pipelines
            .first()
            .copied()
            .ok_or("vkCreateGraphicsPipelines returned no pipeline")?;
        // SAFETY: the module is used by no other pending create call.
        unsafe { device.destroy_shader_module(shader_module, None) };
        println!("Graphics pipeline created (vertex input from binding 0)");

        // Steps 09-10: the queue, the pool, per-frame buffers and sync objects.
        // SAFETY: one queue of `family` was requested in `queue_infos`.
        let queue = unsafe { device.get_device_queue(family, 0) };
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(family);
        // SAFETY: plain create call on a live device.
        let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(u32::try_from(FRAMES_IN_FLIGHT)?);
        // SAFETY: `command_pool` belongs to this device.
        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let mut frames = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for &command_buffer in &command_buffers {
            // SAFETY: plain create calls on a live device.
            let image_available = unsafe { device.create_semaphore(&semaphore_info, None)? };
            // SAFETY: as above.
            let in_flight = unsafe { device.create_fence(&fence_info, None)? };
            frames.push(Frame {
                command_buffer,
                image_available,
                in_flight,
            });
        }
        let mut render_finished = Vec::with_capacity(images.len());
        for _ in &images {
            // SAFETY: as above.
            render_finished.push(unsafe { device.create_semaphore(&semaphore_info, None)? });
        }
        println!("{FRAMES_IN_FLIGHT} frames in flight");

        // #region mesh-bytes
        // The bytes to upload: the vertices, padded with zeros to --mib MiB
        // (only the 7 real vertices are drawn; the padding makes the timing
        // measure bandwidth), and the indices.
        let mut vertex_data = as_bytes(&VERTICES).to_vec();
        vertex_data.resize(vertex_data.len().max(usize::try_from(args.mib)? * MIB), 0);
        let index_data = as_bytes(&INDICES);
        println!(
            "\nMesh: hexagon, {} vertices + {} indices; vertex buffer {} bytes{}, index buffer {} bytes",
            VERTICES.len(),
            INDICES.len(),
            vertex_data.len(),
            if args.mib > 0 {
                " (padded with --mib)"
            } else {
                ""
            },
            index_data.len()
        );
        // SAFETY: `physical_device` was enumerated from this instance.
        let mem = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        // #endregion

        let (vertex_buffer, index_buffer) = match args.upload {
            Upload::Bar => upload_bar(&device, &mem, &vertex_data, index_data)?,
            Upload::GraphicsQueue | Upload::TransferQueue => {
                let transfer = transfer_family.map(|tf| {
                    // SAFETY: one queue of `tf` was requested in `queue_infos`.
                    (tf, unsafe { device.get_device_queue(tf, 0) })
                });
                upload_staged(
                    &device,
                    &mem,
                    &vertex_data,
                    index_data,
                    (family, queue),
                    transfer,
                )?
            }
        };
        window.request_redraw();
        println!("Rendering. Close the window to exit.");
        Ok(Self {
            _entry: entry,
            instance,
            debug_utils,
            messenger,
            surface_loader,
            surface,
            device,
            swapchain_loader,
            swapchain,
            image_views,
            pipeline_layout,
            pipeline,
            vertex_buffer,
            index_buffer,
            queue,
            images,
            extent,
            command_pool,
            frames,
            frame_index: 0,
            render_finished,
            start: Instant::now(),
            window,
        })
    }

    /// Renders and presents one frame (step 10's loop). Returns `Ok(false)`
    /// when the swapchain is out of date (recreation is step 11).
    fn draw_frame(&mut self) -> Result<bool, Box<dyn Error>> {
        let device = &self.device;
        let slot = self.frame_index % self.frames.len();
        let frame = self.frames.get(slot).ok_or("frame slot out of range")?;
        let cmd = frame.command_buffer;

        // #region wait-fence
        // The last submission that used this slot must be finished before its
        // command buffer and semaphore are reused (step 10).
        // SAFETY: the fence belongs to this device.
        unsafe { device.wait_for_fences(&[frame.in_flight], true, u64::MAX)? };
        // #endregion

        // #region acquire
        // SAFETY: the swapchain and the semaphore belong to this device; the
        // semaphore is unsignaled (its last wait completed, see wait-fence).
        let acquired = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                frame.image_available,
                vk::Fence::null(),
            )
        };
        let (image_index, _suboptimal) = match acquired {
            Ok(r) => r,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(false),
            Err(e) => return Err(e.into()),
        };
        let index = usize::try_from(image_index)?;
        let image = *self.images.get(index).ok_or("image index out of range")?;
        let view = *self
            .image_views
            .get(index)
            .ok_or("image index out of range")?;
        let render_finished = *self
            .render_finished
            .get(index)
            .ok_or("image index out of range")?;
        // SAFETY: the fence is not in use by a pending submission (we waited on it).
        unsafe { device.reset_fences(&[frame.in_flight])? };
        // #endregion

        // SAFETY: the buffer is not pending (wait-fence) and its pool allows reset.
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        let color_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        // Step 09's layout transition to COLOR_ATTACHMENT_OPTIMAL, chained to
        // the acquire semaphore's wait stage.
        let to_attachment = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(color_range);
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.02, 0.02, 0.03, 1.0],
                },
            });
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment));
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.extent.width as f32,
            height: self.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent,
        };
        // SAFETY: `cmd` is recording; all handles are alive.
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&to_attachment)),
            );
            device.cmd_begin_rendering(cmd, &rendering_info);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
        }

        // #region bind-and-draw
        // Vertex buffer slot 0 = the buffer from offset 0 (the binding of the
        // vertex input state); the index buffer with its index type. Then
        // vkCmdDrawIndexed: 18 indices, 1 instance, first index 0, vertex
        // offset 0 (added to every index), first instance 0. The input
        // assembler reads index k, fetches that vertex, and the vertex shader
        // runs for it (a post-transform cache may reuse the result of a vertex
        // that repeats, like the center).
        let index_count = u32::try_from(INDICES.len())?;
        // SAFETY: `cmd` is recording inside a rendering compatible with the
        // pipeline; both buffers are bound to memory and stay alive while pending.
        unsafe {
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, self.index_buffer.buffer, 0, vk::IndexType::UINT16);
            device.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);
            device.cmd_end_rendering(cmd);
        }
        // #endregion

        // Step 09's transition to PRESENT_SRC_KHR, chained to render_finished.
        let to_present = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(color_range);
        // SAFETY: `cmd` is recording.
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&to_present)),
            );
            device.end_command_buffer(cmd)?;
        }

        // #region submit
        // Step 10's submit: wait image_available at COLOR_ATTACHMENT_OUTPUT
        // (vertex fetch and vertex shading may start before the image is ours),
        // signal render_finished, then the slot's fence.
        let wait = vk::SemaphoreSubmitInfo::default()
            .semaphore(frame.image_available)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        let signal = vk::SemaphoreSubmitInfo::default()
            .semaphore(render_finished)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(std::slice::from_ref(&wait))
            .command_buffer_infos(std::slice::from_ref(&cmd_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal));
        // SAFETY: all handles belong to this device; the fence is unsignaled.
        unsafe { device.queue_submit2(self.queue, &[submit], frame.in_flight)? };
        // #endregion

        // #region present
        let swapchains = [self.swapchain];
        let indices = [image_index];
        let waits = [render_finished];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&swapchains)
            .image_indices(&indices);
        // SAFETY: the image was acquired above and is in PRESENT_SRC_KHR after the submit.
        let presented = unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present_info)
        };
        match presented {
            Ok(_suboptimal) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(false),
            Err(e) => return Err(e.into()),
        }
        // #endregion

        self.frame_index += 1;
        Ok(true)
    }

    /// Destroys the Vulkan objects in reverse order of creation.
    fn destroy(self) -> Result<(), vk::Result> {
        // #region cleanup
        // Wait until the GPU is idle (step 10): the last frames may still read
        // the vertex and index buffers. Then the per-frame objects, the mesh
        // buffers (buffer first, then its memory), and steps 07-08 in reverse.
        // SAFETY: plain call on a live device.
        let waited = unsafe { self.device.device_wait_idle() };
        if let Err(e) = &waited {
            eprintln!("device_wait_idle failed: {e}");
        }
        // SAFETY: nothing else uses these objects, and each is destroyed exactly once.
        unsafe {
            for &semaphore in &self.render_finished {
                self.device.destroy_semaphore(semaphore, None);
            }
            for frame in &self.frames {
                self.device.destroy_fence(frame.in_flight, None);
                self.device.destroy_semaphore(frame.image_available, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            for b in [&self.index_buffer, &self.vertex_buffer] {
                self.device.destroy_buffer(b.buffer, None);
                self.device.free_memory(b.memory, None);
            }
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            for &view in &self.image_views {
                self.device.destroy_image_view(view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils
                .destroy_debug_utils_messenger(self.messenger, None);
            self.instance.destroy_instance(None);
        }
        // #endregion
        waited
    }
}

// #region event-loop
/// winit drives the program (step 06): `vk` is built in the first `resumed`,
/// and every `RedrawRequested` renders one frame.
struct App {
    args: Args,
    vk: Option<Vk>,
    error: Option<Box<dyn Error>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.vk.is_some() {
            return;
        }
        match Vk::new(event_loop, self.args) {
            Ok(vk) => self.vk = Some(vk),
            Err(e) => {
                self.error = Some(e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let Some(vk) = self.vk.as_mut() else {
                    return;
                };
                match vk.draw_frame() {
                    Ok(true) => {
                        let done = self
                            .args
                            .seconds
                            .is_some_and(|s| vk.start.elapsed().as_secs_f64() >= s);
                        if done {
                            event_loop.exit();
                        } else {
                            vk.window.request_redraw();
                        }
                    }
                    Ok(false) => {
                        println!(
                            "Swapchain out of date (the window changed); recreation is step 11, stopping"
                        );
                        event_loop.exit();
                    }
                    Err(e) => {
                        self.error = Some(e);
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let event_loop = EventLoop::new()?;
    let mut app = App {
        args,
        vk: None,
        error: None,
    };
    event_loop.run_app(&mut app)?;
    if let Some(vk) = app.vk.take() {
        println!("Window closed");
        let elapsed = vk.start.elapsed().as_secs_f64();
        println!(
            "[timing] {} frames in {elapsed:.1} s, {:.0} fps",
            vk.frame_index,
            vk.frame_index as f64 / elapsed
        );
        vk.destroy()?;
    }
    app.error.map_or(Ok(()), Err)
}
// #endregion
