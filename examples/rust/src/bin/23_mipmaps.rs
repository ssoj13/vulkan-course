//! Step 23 - Mipmaps.
//!
//! A checkerboard floor recedes to the horizon. Far away, one pixel covers
//! dozens of texels; a sampler that reads only the base image picks a few of
//! them almost at random, and the floor dissolves into shimmering moire
//! patterns (aliasing). MIPMAPS fix that: a chain of prefiltered copies, each
//! half the size of the previous one, down to 1x1; the sampler reads the level
//! whose texels are about the size of a pixel.
//!
//!   upload mip 0 (step 21) -> for each level i = 1..N-1:
//!     barrier level i-1: TRANSFER_DST -> TRANSFER_SRC
//!     vkCmdBlitImage2 level i-1 -> level i (half size, LINEAR filter)
//!     barrier level i-1: TRANSFER_SRC -> SHADER_READ_ONLY
//!   barrier level N-1: TRANSFER_DST -> SHADER_READ_ONLY
//!   sampler: mipmapMode LINEAR, minLod 0, maxLod N (trilinear filtering)
//!
//! `--no-mips` clamps the sampler to level 0 (maxLod 0) to show the aliasing;
//! `--anisotropy` adds anisotropic filtering (16x, clamped to the limit);
//! `--seconds S` closes the window after S seconds. Lines that depend on
//! timing start with `[timing]`.
//! Steps 01-21 are repeated, reduced to what this step needs.

use std::error::Error;
use std::ffi::{CStr, c_void};
use std::time::Instant;

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

// #region texture-data
/// Width and height of the texture in texels.
const TEX_SIZE: u32 = 512;
/// Side of one checker cell in texels (32 x 32 cells): small cells are what
/// makes the far floor alias.
const CELL: u32 = 16;

/// A high-contrast checkerboard, RGBA8, row by row (generated, not loaded:
/// see step 21). Light cells get a faint gradient so the levels are not all
/// the same gray.
fn texture_pixels() -> Vec<u8> {
    let mut pixels = Vec::with_capacity((TEX_SIZE * TEX_SIZE * 4) as usize);
    for y in 0..TEX_SIZE {
        for x in 0..TEX_SIZE {
            let rgb: [u32; 3] = if (x / CELL + y / CELL).is_multiple_of(2) {
                [200 + x / 16, 200 + y / 16, 230]
            } else {
                [25, 25, 30]
            };
            // Every value is <= 255 by construction (x, y < 512).
            pixels.extend(rgb.map(|c| c as u8));
            pixels.push(255);
        }
    }
    pixels
}

/// FNV-1a, 32 bit: a fingerprint of the texture bytes, to compare runs.
fn fnv1a(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811c_9dc5_u32, |h, &b| {
        (h ^ u32::from(b)).wrapping_mul(0x0100_0193)
    })
}
// #endregion

// #region mip-count
/// The number of levels down to 1x1: floor(log2(max(w, h))) + 1. For a u32,
/// floor(log2(n)) + 1 is the number of significant bits, 32 - leading zeros.
fn mip_count(width: u32, height: u32) -> u32 {
    32 - width.max(height).leading_zeros()
}
// #endregion

/// Step 21's vertex: position + UV, 20 bytes (shaders/textured.slang).
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
}
const _: () = assert!(size_of::<Vertex>() == 20);

// #region floor
/// The floor: y = 0, x from -20 to 20, z from 5 (behind the camera) to -200.
/// UV = world position / 2, so the texture repeats every 2 units (REPEAT
/// address mode) - 20 x 102.5 copies. Corners counter-clockwise seen from
/// above, the front face (step 22).
#[rustfmt::skip] // one vertex per line reads like a table
const VERTICES: [Vertex; 4] = [
    Vertex { position: [-20.0, 0.0, -200.0], uv: [0.0, 102.5] },
    Vertex { position: [-20.0, 0.0, 5.0],    uv: [0.0, 0.0] },
    Vertex { position: [20.0, 0.0, 5.0],     uv: [20.0, 0.0] },
    Vertex { position: [20.0, 0.0, -200.0],  uv: [20.0, 102.5] },
];
const INDICES: [u16; 6] = [0, 1, 2, 0, 2, 3];
// #endregion

/// The host side of `Camera` in shaders/textured.slang.
#[repr(C)]
#[derive(Clone, Copy)]
struct Camera {
    mvp: Mat4,
}
const _: () = assert!(size_of::<Camera>() == 64);

/// A 4x4 matrix, column-major (step 22).
type Mat4 = [[f32; 4]; 4];

/// a * b for column-major matrices (step 22).
fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for (c, col) in m.iter_mut().enumerate() {
        for (r, out) in col.iter_mut().enumerate() {
            *out = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    m
}

/// Moves points by (x, y, z) (step 22).
fn translate(x: f32, y: f32, z: f32) -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [x, y, z, 1.0],
    ]
}

/// Rotation by `a` radians about +x (step 22).
fn rotate_x(a: f32) -> Mat4 {
    let (s, c) = a.sin_cos();
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, s, 0.0],
        [0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Step 22's perspective projection into Vulkan's clip space.
fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fovy / 2.0).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ]
}

// #region camera
/// The camera floats 1 unit above the floor, tilted down by 0.2 rad, and
/// glides forward at 1 unit/s. Moving the floor toward the camera by
/// (time mod 2) looks the same as moving the camera, and since the texture
/// repeats every 2 units the floor never runs out.
fn camera(time: f32, proj: &Mat4) -> Camera {
    let view = mul(&rotate_x(0.2), &translate(0.0, -1.0, 0.0));
    let model = translate(0.0, 0.0, time % 2.0);
    Camera {
        mvp: mul(proj, &mul(&view, &model)),
    }
}
// #endregion

// #region cli
/// Command line: `--no-mips`, `--anisotropy`, `--seconds S`.
struct Args {
    mips: bool,
    anisotropy: bool,
    seconds: Option<f64>,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut args = Self {
            mips: true,
            anisotropy: false,
            seconds: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--no-mips" => args.mips = false,
                "--anisotropy" => args.anisotropy = true,
                "--seconds" => {
                    let value = it.next().ok_or("--seconds needs a value")?;
                    args.seconds = Some(value.parse()?);
                }
                _ => {
                    return Err(format!(
                        "unknown argument {arg} (use --no-mips, --anisotropy, --seconds S)"
                    )
                    .into());
                }
            }
        }
        Ok(args)
    }
}
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

/// A buffer and the memory bound to it (step 14).
struct GpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

/// Step 14's manual path: create the buffer, query its requirements, take the
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
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(memory_type);
    // SAFETY: the type index comes from this device's memory properties.
    let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    // SAFETY: the allocation has the required size and type; offset 0 is aligned.
    unsafe { device.bind_buffer_memory(buffer, memory, 0)? };
    Ok(GpuBuffer { buffer, memory })
}

/// Step 14's upload: map, memcpy, unmap a HOST_VISIBLE | HOST_COHERENT
/// allocation (no flush needed; the submit makes the writes visible).
fn write_mapped<T: Copy>(
    device: &ash::Device,
    memory: vk::DeviceMemory,
    data: &[T],
) -> Result<(), vk::Result> {
    // SAFETY: the memory is HOST_VISIBLE, not mapped yet, and at least
    // `size_of_val(data)` bytes long; nothing else touches it meanwhile.
    unsafe {
        let ptr = device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(
            data.as_ptr().cast::<u8>(),
            ptr.cast::<u8>(),
            size_of_val(data),
        );
        device.unmap_memory(memory);
    }
    Ok(())
}

// #region format-support
/// What generating mips on the GPU needs from the format with OPTIMAL tiling:
/// step 21's three features, plus BLIT_SRC and BLIT_DST (vkCmdBlitImage2 may
/// read and write it). The blit's LINEAR filter additionally needs
/// SAMPLED_IMAGE_FILTER_LINEAR on the source format, which is already in the
/// list. Formats without blit support (many compressed and integer formats)
/// get their mips from the CPU or an offline tool instead.
const NEEDED_FEATURES: [(vk::FormatFeatureFlags, &str); 5] = [
    (vk::FormatFeatureFlags::SAMPLED_IMAGE, "SAMPLED_IMAGE"),
    (
        vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
        "SAMPLED_IMAGE_FILTER_LINEAR",
    ),
    (vk::FormatFeatureFlags::TRANSFER_DST, "TRANSFER_DST"),
    (vk::FormatFeatureFlags::BLIT_SRC, "BLIT_SRC"),
    (vk::FormatFeatureFlags::BLIT_DST, "BLIT_DST"),
];

/// Prints which needed features `format` has with OPTIMAL tiling and returns
/// whether it has all of them.
fn check_format(instance: &ash::Instance, pd: vk::PhysicalDevice, format: vk::Format) -> bool {
    let mut props = vk::FormatProperties2::default();
    // SAFETY: `pd` was enumerated from this instance; `props` outlives the call.
    unsafe { instance.get_physical_device_format_properties2(pd, format, &mut props) };
    let optimal = props.format_properties.optimal_tiling_features;
    let list: Vec<String> = NEEDED_FEATURES
        .iter()
        .map(|&(bit, name)| {
            let yes = if optimal.contains(bit) { "yes" } else { "NO" };
            format!("{name} {yes}")
        })
        .collect();
    println!("  {format:?} (optimal tiling): {}", list.join(", "));
    NEEDED_FEATURES
        .iter()
        .all(|&(bit, _)| optimal.contains(bit))
}
// #endregion

/// The objects `upload_with_mips` creates only for the upload (step 21's
/// UploadScratch): dropping it waits for the device to go idle, then destroys
/// them, also when a `?` returns early after the submit.
struct UploadScratch<'a> {
    device: &'a ash::Device,
    staging: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    pool: vk::CommandPool,
    fence: vk::Fence,
}

impl Drop for UploadScratch<'_> {
    fn drop(&mut self) {
        // SAFETY: after the idle wait no submitted work uses these objects;
        // each was created on `device` (or is null) and is destroyed once.
        unsafe {
            if let Err(e) = self.device.device_wait_idle() {
                eprintln!("device_wait_idle failed: {e}");
            }
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_buffer(self.staging, None);
            self.device.free_memory(self.staging_memory, None);
        }
    }
}

/// A barrier on ONE mip level of `image` (all other fields as in step 21).
fn level_barrier(
    image: vk::Image,
    level: u32,
    (src_stage, src_access): (vk::PipelineStageFlags2, vk::AccessFlags2),
    (dst_stage, dst_access): (vk::PipelineStageFlags2, vk::AccessFlags2),
    (old_layout, new_layout): (vk::ImageLayout, vk::ImageLayout),
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(level)
                .level_count(1)
                .layer_count(1),
        )
}

/// Step 21's staging upload of level 0, then the mip chain generated on the
/// GPU, all in one command buffer. Blocks until it is complete.
fn upload_with_mips(
    device: &ash::Device,
    mem: &vk::PhysicalDeviceMemoryProperties,
    (family, queue): (u32, vk::Queue),
    image: vk::Image,
    levels: u32,
    pixels: &[u8],
) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let mut scratch = UploadScratch {
        device,
        staging: vk::Buffer::null(),
        staging_memory: vk::DeviceMemory::null(),
        pool: vk::CommandPool::null(),
        fence: vk::Fence::null(),
    };
    let staging = create_buffer(
        device,
        mem,
        vk::DeviceSize::try_from(pixels.len())?,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    scratch.staging = staging.buffer;
    scratch.staging_memory = staging.memory;
    write_mapped(device, staging.memory, pixels)?;

    let pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(family);
    // SAFETY: plain create call on a live device.
    scratch.pool = unsafe { device.create_command_pool(&pool_info, None)? };
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(scratch.pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: the pool belongs to this device; one buffer was requested.
    let cmd = unsafe { device.allocate_command_buffers(&alloc_info)? }
        .first()
        .copied()
        .ok_or("vkAllocateCommandBuffers returned no command buffer")?;
    // SAFETY: a fresh command buffer is in the initial state.
    unsafe {
        device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
    }
    let none = (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE);
    let copy_write = (
        vk::PipelineStageFlags2::COPY,
        vk::AccessFlags2::TRANSFER_WRITE,
    );
    let blit_write = (
        vk::PipelineStageFlags2::BLIT,
        vk::AccessFlags2::TRANSFER_WRITE,
    );
    let blit_read = (
        vk::PipelineStageFlags2::BLIT,
        vk::AccessFlags2::TRANSFER_READ,
    );
    let shader_read = (
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::AccessFlags2::SHADER_SAMPLED_READ,
    );
    let undefined_to_dst = (
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );
    let dst_to_src = (
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    );
    let src_to_read = (
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );
    let dst_to_read = (
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );

    // #region upload-level-0
    // Every level starts UNDEFINED -> TRANSFER_DST_OPTIMAL: level 0 for the
    // copy, the others for the blits that will write them. One barrier per
    // level keeps the code uniform; one barrier over all levels
    // (levelCount = levels) would do the same.
    let to_dst: Vec<_> = (0..levels)
        .map(|level| {
            let writer = if level == 0 { copy_write } else { blit_write };
            level_barrier(image, level, none, writer, undefined_to_dst)
        })
        .collect();
    let region = vk::BufferImageCopy2::default()
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .layer_count(1),
        )
        .image_extent(vk::Extent3D {
            width: TEX_SIZE,
            height: TEX_SIZE,
            depth: 1,
        });
    let copy_info = vk::CopyBufferToImageInfo2::default()
        .src_buffer(staging.buffer)
        .dst_image(image)
        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .regions(std::slice::from_ref(&region));
    // SAFETY: `cmd` is recording; the image, the buffer and the region are valid.
    unsafe {
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&to_dst),
        );
        device.cmd_copy_buffer_to_image2(cmd, &copy_info);
    }
    println!("  mip  0: {TEX_SIZE}x{TEX_SIZE} (vkCmdCopyBufferToImage2 from the staging buffer)");
    // #endregion

    // #region blit-chain
    // Level i is made from level i-1, which therefore must be finished and
    // readable first: its writer (the copy for level 0, the previous blit
    // otherwise) -> TRANSFER_SRC_OPTIMAL for the blit's read. The blit
    // averages each 2x2 block with the LINEAR filter (sRGB texels are
    // converted to linear before filtering and back after, spec "Image Copies
    // With Scaling"). Once read, level i-1 is done: -> SHADER_READ_ONLY_OPTIMAL.
    // Its read needs no memory to be made available, only the execution
    // dependency (a write-after-read is ordered by execution alone): src
    // access NONE. Each extent halves, rounded down, but never below 1.
    let mut size = TEX_SIZE;
    for level in 1..levels {
        let src_size = size;
        size = (size / 2).max(1);
        let writer = if level == 1 { copy_write } else { blit_write };
        let to_src = level_barrier(image, level - 1, writer, blit_read, dst_to_src);
        let edge = |s: u32| -> Result<vk::Offset3D, Box<dyn Error>> {
            let s = i32::try_from(s)?;
            Ok(vk::Offset3D { x: s, y: s, z: 1 })
        };
        let layers = |mip_level| {
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(mip_level)
                .layer_count(1)
        };
        let blit = vk::ImageBlit2::default()
            .src_subresource(layers(level - 1))
            .src_offsets([vk::Offset3D::default(), edge(src_size)?])
            .dst_subresource(layers(level))
            .dst_offsets([vk::Offset3D::default(), edge(size)?]);
        let blit_info = vk::BlitImageInfo2::default()
            .src_image(image)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(image)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .regions(std::slice::from_ref(&blit))
            .filter(vk::Filter::LINEAR);
        let done = level_barrier(
            image,
            level - 1,
            (vk::PipelineStageFlags2::BLIT, vk::AccessFlags2::NONE),
            shader_read,
            src_to_read,
        );
        // SAFETY: `cmd` is recording; both levels exist and are in the
        // layouts named above when the blit executes.
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_src)),
            );
            device.cmd_blit_image2(cmd, &blit_info);
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&done)),
            );
        }
        println!(
            "  mip {level:2}: {size}x{size} (vkCmdBlitImage2 from mip {}, LINEAR)",
            level - 1
        );
    }
    // The last level was only written, never read by a blit.
    let writer = if levels == 1 { copy_write } else { blit_write };
    let last = level_barrier(image, levels - 1, writer, shader_read, dst_to_read);
    // SAFETY: `cmd` is recording.
    unsafe {
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&last)),
        );
        device.end_command_buffer(cmd)?;
    }
    // #endregion

    // SAFETY: plain create call on a live device.
    scratch.fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
    let cmd_info = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
    let submit = vk::SubmitInfo2::default().command_buffer_infos(&cmd_info);
    // SAFETY: `cmd` is executable and came from a pool of `family`, the
    // queue's family; the fence is new and unsignaled.
    unsafe {
        device.queue_submit2(queue, &[submit], scratch.fence)?;
        device.wait_for_fences(&[scratch.fence], true, u64::MAX)?;
    }
    drop(scratch);
    println!(
        "Upload + mip generation complete (fence waited); every level is SHADER_READ_ONLY_OPTIMAL"
    );
    println!(
        "[timing] upload + {} blits {:.3} ms",
        levels - 1,
        start.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}

/// What the CPU touches while recording one frame (step 21).
struct Frame {
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
    uniform: GpuBuffer,
    mapped: *mut Camera,
    descriptor_set: vk::DescriptorSet,
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
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    vertex_buffer: GpuBuffer,
    index_buffer: GpuBuffer,
    texture: vk::Image,
    texture_memory: vk::DeviceMemory,
    texture_view: vk::ImageView,
    sampler: vk::Sampler,
    descriptor_pool: vk::DescriptorPool,
    queue: vk::Queue,
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    proj: Mat4,
    command_pool: vk::CommandPool,
    frames: Vec<Frame>,
    frame_index: usize,
    /// One per swapchain image, indexed by the acquired image index.
    render_finished: Vec<vk::Semaphore>,
    start: Instant,
    window: Window,
}

impl Vk {
    /// Step 21 with a mipmapped texture and a perspective camera.
    fn new(event_loop: &ActiveEventLoop, args: &Args) -> Result<Self, Box<dyn Error>> {
        // Window from step 06: 800x600 physical pixels, fixed size.
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("Vulkan from Zero - 23 mipmaps")
                .with_inner_size(PhysicalSize::new(800, 600))
                .with_resizable(false),
        )?;
        let size = window.inner_size();
        println!("Window: {}x{} pixels", size.width, size.height);

        // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vulkan from Zero - 23 mipmaps")
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

        // Step 21's device selection: Vulkan 1.4, dynamic rendering +
        // synchronization2 + samplerAnisotropy, graphics + present, swapchain.
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
            let anisotropy = f2.features.sampler_anisotropy;
            if f13.dynamic_rendering == 0 || f13.synchronization2 == 0 || anisotropy == 0 {
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
            "No Vulkan 1.4 GPU with dynamic rendering + synchronization2 + samplerAnisotropy + VK_KHR_swapchain can present to this window",
        )?;
        // SAFETY: `physical_device` was enumerated from this instance.
        let chosen = unsafe { instance.get_physical_device_properties(physical_device) };
        println!(
            "\nPicked: {}",
            chosen.device_name_as_c_str()?.to_string_lossy()
        );
        println!("Graphics + present family: {family}");

        // Step 21's device: samplerAnisotropy enabled (used with --anisotropy).
        let priorities = [1.0_f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let features = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let device_exts = [swapchain::NAME.as_ptr()];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_exts)
            .enabled_features(&features)
            .push(&mut features13);
        // SAFETY: `physical_device` belongs to `instance`, and all create-info data outlives the call.
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        let swapchain_loader = swapchain::Device::load(&instance, &device);
        println!(
            "VkDevice created with {} and samplerAnisotropy",
            swapchain::NAME.to_string_lossy()
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
        // SAFETY: one queue of `family` was requested in `queue_infos`.
        let queue = unsafe { device.get_device_queue(family, 0) };
        // SAFETY: `physical_device` was enumerated from this instance.
        let mem = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let pixels = texture_pixels();
        println!(
            "\nTexture: {TEX_SIZE}x{TEX_SIZE} RGBA8 checkerboard, {CELL}-texel cells, {} bytes (FNV-1a {:#010x})",
            pixels.len(),
            fnv1a(&pixels)
        );
        let tex_format = vk::Format::R8G8B8A8_SRGB;
        println!("Format support (vkGetPhysicalDeviceFormatProperties2):");
        if !check_format(&instance, physical_device, tex_format) {
            return Err(
                format!("{tex_format:?} cannot be blitted with a LINEAR filter here").into(),
            );
        }

        // #region create-image
        // Step 21's image with a full mip chain: mipLevels = floor(log2(512))
        // + 1 = 10, and TRANSFER_SRC added to the usage because every level but
        // the last is read by a blit.
        let levels = mip_count(TEX_SIZE, TEX_SIZE);
        println!("Mip levels: floor(log2(max({TEX_SIZE}, {TEX_SIZE}))) + 1 = {levels}");
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(tex_format)
            .extent(vk::Extent3D {
                width: TEX_SIZE,
                height: TEX_SIZE,
                depth: 1,
            })
            .mip_levels(levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: plain create call on a live device; the format supports this use.
        let texture = unsafe { device.create_image(&image_info, None)? };
        // SAFETY: `texture` belongs to this device.
        let req = unsafe { device.get_image_memory_requirements(texture) };
        let tex_type = find_type(
            &mem,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or("no DEVICE_LOCAL memory type for the texture")?;
        let level0 = u64::from(TEX_SIZE) * u64::from(TEX_SIZE) * 4;
        println!(
            "VkImage: {levels} mip levels, TRANSFER_SRC | TRANSFER_DST | SAMPLED; memory {} bytes (level 0 alone: {level0}; the chain adds about a third), memory type {tex_type}",
            req.size
        );
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(tex_type);
        // SAFETY: the type index comes from this device's memory properties.
        let texture_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        // SAFETY: the allocation has the required size and type; offset 0 is aligned.
        unsafe { device.bind_image_memory(texture, texture_memory, 0)? };
        // #endregion

        println!("Recording the upload and the mip chain:");
        upload_with_mips(&device, &mem, (family, queue), texture, levels, &pixels)?;

        // #region image-view
        // The view covers ALL levels (levelCount = levels): the sampler can
        // only choose among the levels the view exposes.
        let view_info = vk::ImageViewCreateInfo::default()
            .image(texture)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(tex_format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(levels)
                    .layer_count(1),
            );
        // SAFETY: `texture` belongs to this device and has memory bound.
        let texture_view = unsafe { device.create_image_view(&view_info, None)? };
        // #endregion

        // #region sampler
        // Trilinear filtering: LINEAR within a level (min/mag filters) and
        // LINEAR between the two nearest levels (mipmapMode). The GPU picks
        // the level from the UV derivatives across a 2x2 pixel quad: lambda =
        // log2(texels per pixel) + mipLodBias, clamped to [minLod, maxLod].
        // maxLod = the level count lets it reach the 1x1 level
        // (VK_LOD_CLAMP_NONE would too); --no-mips sets maxLod = 0, so only
        // level 0 is ever read although the image still has all levels.
        // Anisotropic filtering (--anisotropy) takes several samples along
        // the long axis of the pixel's footprint: on a floor seen at a grazing
        // angle the footprint is long and thin, and plain trilinear blurs it.
        let max_lod = if args.mips { levels as f32 } else { 0.0 };
        let max_aniso = chosen.limits.max_sampler_anisotropy;
        let anisotropy = args.anisotropy.then_some(max_aniso.min(16.0));
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .mip_lod_bias(0.0)
            .anisotropy_enable(anisotropy.is_some())
            .max_anisotropy(anisotropy.unwrap_or(1.0))
            .compare_enable(false)
            .min_lod(0.0)
            .max_lod(max_lod)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
            .unnormalized_coordinates(false);
        // SAFETY: samplerAnisotropy is enabled; maxAnisotropy is within the limit.
        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };
        let aniso_text = anisotropy.map_or("off".to_owned(), |a| a.to_string());
        println!(
            "Sampler: LINEAR filters, mipmapMode LINEAR, minLod 0, maxLod {max_lod}{}, anisotropy {aniso_text} (maxSamplerAnisotropy {max_aniso})",
            if args.mips {
                ""
            } else {
                " (--no-mips: level 0 only)"
            }
        );
        // #endregion

        // The SPIR-V is compiled from shaders/textured.slang by `bun run
        // sync:shaders` and embedded at build time (step 08 explains `read_spv`).
        const TEXTURED_SPV: &[u8] = include_bytes!("../../../../shaders/spv/textured.spv");
        let spirv = ash::util::read_spv(&mut std::io::Cursor::new(TEXTURED_SPV))?;
        println!(
            "\nSPIR-V: {} bytes (textured.spv), entry points vertexMain, fragmentMain",
            TEXTURED_SPV.len()
        );
        let module_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
        // SAFETY: `spirv` is valid SPIR-V (checked by spirv-val in sync:shaders).
        let shader_module = unsafe { device.create_shader_module(&module_info, None)? };
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

        // Step 21's vertex input: binding 0, stride 20, position + uv.
        let bindings = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(u32::try_from(size_of::<Vertex>())?)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let attributes = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(u32::try_from(std::mem::offset_of!(Vertex, position))?),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(u32::try_from(std::mem::offset_of!(Vertex, uv))?),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);

        // Step 21's set layout: binding 0 = uniform buffer, binding 1 =
        // combined image sampler.
        let set_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&set_bindings);
        // SAFETY: plain create call on a live device.
        let set_layout = unsafe { device.create_descriptor_set_layout(&set_layout_info, None)? };
        let set_layouts = [set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        // SAFETY: `set_layout` belongs to this device.
        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        // Step 22's projection keeps counter-clockwise faces counter-clockwise.
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
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
        println!("Graphics pipeline created");

        // Step 14's mesh buffers: HOST_VISIBLE | HOST_COHERENT, written once.
        let host_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let vertex_buffer = create_buffer(
            &device,
            &mem,
            vk::DeviceSize::try_from(size_of_val(&VERTICES))?,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            host_flags,
        )?;
        write_mapped(&device, vertex_buffer.memory, &VERTICES)?;
        let index_buffer = create_buffer(
            &device,
            &mem,
            vk::DeviceSize::try_from(size_of_val(&INDICES))?,
            vk::BufferUsageFlags::INDEX_BUFFER,
            host_flags,
        )?;
        write_mapped(&device, index_buffer.memory, &INDICES)?;
        println!("Mesh: floor 40 x 205 units, UVs 0..20 x 0..102.5 (REPEAT)");

        // Step 21's descriptors: per frame a uniform buffer + the texture.
        let uniform_size = vk::DeviceSize::try_from(size_of::<Camera>())?;
        let frames_u32 = u32::try_from(FRAMES_IN_FLIGHT)?;
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(frames_u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(frames_u32),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(frames_u32)
            .pool_sizes(&pool_sizes);
        // SAFETY: plain create call on a live device.
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };
        let layouts = [set_layout; FRAMES_IN_FLIGHT];
        let alloc_sets = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        // SAFETY: the pool and the layout belong to this device; the pool has room.
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_sets)? };
        let image_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(texture_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let mut uniforms = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for &set in &descriptor_sets {
            let buffer = create_buffer(
                &device,
                &mem,
                uniform_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                host_flags,
            )?;
            // SAFETY: the memory is HOST_VISIBLE and not mapped; it stays mapped
            // until vkFreeMemory, which unmaps it implicitly.
            let ptr = unsafe {
                device.map_memory(
                    buffer.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )?
            };
            let buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(buffer.buffer)
                .offset(0)
                .range(uniform_size)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_info),
            ];
            // SAFETY: the set, the buffer, the view and the sampler outlive the
            // call; no command buffer uses the set yet.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
            uniforms.push((buffer, ptr.cast::<Camera>()));
        }

        // Steps 09-10: the pool, per-frame buffers and sync objects.
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(family);
        // SAFETY: plain create call on a live device.
        let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(frames_u32);
        // SAFETY: `command_pool` belongs to this device.
        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let mut frames = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for ((&command_buffer, (uniform, mapped)), &descriptor_set) in
            command_buffers.iter().zip(uniforms).zip(&descriptor_sets)
        {
            // SAFETY: plain create calls on a live device.
            let image_available = unsafe { device.create_semaphore(&semaphore_info, None)? };
            // SAFETY: as above.
            let in_flight = unsafe { device.create_fence(&fence_info, None)? };
            frames.push(Frame {
                command_buffer,
                image_available,
                in_flight,
                uniform,
                mapped,
                descriptor_set,
            });
        }
        let mut render_finished = Vec::with_capacity(images.len());
        for _ in &images {
            // SAFETY: as above.
            render_finished.push(unsafe { device.create_semaphore(&semaphore_info, None)? });
        }
        let aspect = extent.width as f32 / extent.height as f32;
        let proj = perspective(std::f32::consts::FRAC_PI_3, aspect, 0.1, 300.0);
        println!(
            "\n{FRAMES_IN_FLIGHT} frames in flight; camera 1 unit above the floor, gliding forward at 1 unit/s"
        );

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
            set_layout,
            pipeline_layout,
            pipeline,
            vertex_buffer,
            index_buffer,
            texture,
            texture_memory,
            texture_view,
            sampler,
            descriptor_pool,
            queue,
            images,
            extent,
            proj,
            command_pool,
            frames,
            frame_index: 0,
            render_finished,
            start: Instant::now(),
            window,
        })
    }

    /// Renders and presents one frame (step 21's loop). Returns `Ok(false)`
    /// when the swapchain is out of date (recreation is step 11).
    fn draw_frame(&mut self) -> Result<bool, Box<dyn Error>> {
        let device = &self.device;
        let slot = self.frame_index % self.frames.len();
        let frame = self.frames.get(slot).ok_or("frame slot out of range")?;
        let cmd = frame.command_buffer;

        // SAFETY: the fence belongs to this device.
        unsafe { device.wait_for_fences(&[frame.in_flight], true, u64::MAX)? };

        let time = self.start.elapsed().as_secs_f32();
        // SAFETY: `mapped` points to this slot's mapped allocation of at least
        // size_of::<Camera>() bytes; no pending GPU work reads it (fence above).
        unsafe { frame.mapped.write(camera(time, &self.proj)) };

        // SAFETY: the swapchain and the semaphore belong to this device; the
        // semaphore is unsignaled (its last wait completed, see the fence).
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

        // SAFETY: the buffer is not pending (fence) and its pool allows reset.
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
        // A sky-like clear color behind the horizon.
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.35, 0.45, 0.6, 1.0],
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
        let index_count = u32::try_from(INDICES.len())?;
        // SAFETY: `cmd` is recording; all handles are alive and stay alive
        // while the command buffer is pending.
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
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[frame.descriptor_set],
                &[],
            );
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, self.index_buffer.buffer, 0, vk::IndexType::UINT16);
            device.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);
            device.cmd_end_rendering(cmd);
        }

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

        // Step 10's submit: wait image_available at COLOR_ATTACHMENT_OUTPUT,
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

        self.frame_index += 1;
        Ok(true)
    }

    /// Destroys the Vulkan objects in reverse order of creation.
    fn destroy(self) -> Result<(), vk::Result> {
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
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            for b in self
                .frames
                .iter()
                .map(|f| &f.uniform)
                .chain([&self.index_buffer, &self.vertex_buffer])
            {
                self.device.destroy_buffer(b.buffer, None);
                self.device.free_memory(b.memory, None);
            }
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_image_view(self.texture_view, None);
            self.device.destroy_image(self.texture, None);
            self.device.free_memory(self.texture_memory, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
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
        waited
    }
}

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
        match Vk::new(event_loop, &self.args) {
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
