//! Step 21 - Textures and samplers.
//!
//! A shader reads an image through two objects: an IMAGE VIEW (which part of
//! which image, in which format) and a SAMPLER (how to filter between texels
//! and what to do outside [0, 1]). Getting pixels into the image is step 15's
//! staging upload, plus two things buffers never needed: a copy that knows
//! rows and texel formats (vkCmdCopyBufferToImage2), and LAYOUT TRANSITIONS,
//! because an OPTIMAL-tiling image stores its texels in a hidden,
//! GPU-friendly arrangement that differs per use.
//!
//!   CPU pixels -> staging buffer -> [UNDEFINED -> TRANSFER_DST_OPTIMAL]
//!     -> vkCmdCopyBufferToImage2 -> [TRANSFER_DST -> SHADER_READ_ONLY_OPTIMAL]
//!   image view + sampler -> COMBINED_IMAGE_SAMPLER descriptor (set 0, binding 1)
//!   fragment shader: albedo.Sample(uv)
//!
//! `--nearest` switches the filters from LINEAR to NEAREST; `--address MODE`
//! (repeat | mirror | clamp | border) picks what UVs outside [0, 1] show (the
//! quad's UVs run 0..2); `--seconds S` closes the window after S seconds.
//! Lines that depend on timing start with `[timing]`.
//! Steps 01-17 are repeated, reduced to what this step needs.

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
const TEX_SIZE: u32 = 256;
/// Side of one checker cell in texels (8 x 8 cells).
const CELL: u32 = 32;

/// The texture's RGBA8 bytes, row by row from the top, generated here rather
/// than loaded from a PNG: no image-decoding library in either language, the
/// same bytes in Rust and C++ (integer math only; the hash printed below
/// proves it), and the lesson stays about Vulkan, not file formats. A real
/// program decodes a file (or, better, loads a GPU-ready KTX2/DDS) into
/// exactly such a byte array. Cells alternate between a color gradient and
/// dark gray; a 2-texel white frame marks the edges, so the address modes
/// show where one copy of the texture ends.
fn texture_pixels() -> Vec<u8> {
    let mut pixels = Vec::with_capacity((TEX_SIZE * TEX_SIZE * 4) as usize);
    for y in 0..TEX_SIZE {
        for x in 0..TEX_SIZE {
            let edge = x < 2 || y < 2 || x >= TEX_SIZE - 2 || y >= TEX_SIZE - 2;
            let rgb: [u32; 3] = if edge {
                [255, 255, 255]
            } else if (x / CELL + y / CELL).is_multiple_of(2) {
                [x, y, 128 + (x + y) / 4]
            } else {
                [40, 40, 40]
            };
            // Every value is <= 255 by construction (x, y < 256).
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

/// Step 14's vertex, with a third coordinate and a UV instead of a color:
/// 3 + 2 floats = 20 bytes (shaders/textured.slang).
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    /// Texture coordinates: (0, 0) = the texture's first texel row, left.
    uv: [f32; 2],
}
const _: () = assert!(size_of::<Vertex>() == 20);

// #region quad
/// A square of side 1.5 with UVs 0..2: the texture appears 2 x 2 times with
/// REPEAT, and the address mode decides what the other three copies show.
/// Top-left, top-right, bottom-right, bottom-left (+y points down): clockwise
/// on screen, the pipeline's front face (step 17).
#[rustfmt::skip] // one vertex per line reads like a table
const VERTICES: [Vertex; 4] = [
    Vertex { position: [-0.75, -0.75, 0.0], uv: [0.0, 0.0] },
    Vertex { position: [0.75, -0.75, 0.0],  uv: [2.0, 0.0] },
    Vertex { position: [0.75, 0.75, 0.0],   uv: [2.0, 2.0] },
    Vertex { position: [-0.75, 0.75, 0.0],  uv: [0.0, 2.0] },
];
const INDICES: [u16; 6] = [0, 1, 2, 0, 2, 3];
// #endregion

/// The host side of `Camera` in shaders/textured.slang: one column-major
/// matrix (step 17 checked this layout against the SPIR-V).
#[repr(C)]
#[derive(Clone, Copy)]
struct Camera {
    mvp: [[f32; 4]; 4],
}
const _: () = assert!(size_of::<Camera>() == 64);

/// Rotation speed of the quad, radians per second: slow, so the filtering
/// stays visible.
const SPIN: f32 = 0.3;

/// Step 17's matrix: a rotation by `angle` about z, then x scaled by `aspect`
/// (height / width). Column-major.
fn transform(angle: f32, aspect: f32) -> [[f32; 4]; 4] {
    let (s, c) = angle.sin_cos();
    [
        [aspect * c, s, 0.0, 0.0],
        [-aspect * s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

// #region cli
/// Command line: `--nearest`, `--address repeat|mirror|clamp|border`,
/// `--seconds S`.
struct Args {
    filter: vk::Filter,
    address: vk::SamplerAddressMode,
    seconds: Option<f64>,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut args = Self {
            filter: vk::Filter::LINEAR,
            address: vk::SamplerAddressMode::REPEAT,
            seconds: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            let mut value = || it.next().ok_or(format!("{arg} needs a value"));
            match arg.as_str() {
                "--nearest" => args.filter = vk::Filter::NEAREST,
                "--address" => {
                    args.address = match value()?.as_str() {
                        "repeat" => vk::SamplerAddressMode::REPEAT,
                        "mirror" => vk::SamplerAddressMode::MIRRORED_REPEAT,
                        "clamp" => vk::SamplerAddressMode::CLAMP_TO_EDGE,
                        "border" => vk::SamplerAddressMode::CLAMP_TO_BORDER,
                        other => {
                            return Err(format!(
                                "unknown address mode {other} (use repeat | mirror | clamp | border)"
                            )
                            .into());
                        }
                    }
                }
                "--seconds" => args.seconds = Some(value()?.parse()?),
                _ => {
                    return Err(format!(
                        "unknown argument {arg} (use --nearest, --address MODE, --seconds S)"
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
    memory_type: u32,
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
    Ok(GpuBuffer {
        buffer,
        memory,
        memory_type,
    })
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
/// The format features this step needs from an OPTIMAL-tiling image, with
/// their names for printing.
const NEEDED_FEATURES: [(vk::FormatFeatureFlags, &str); 3] = [
    (vk::FormatFeatureFlags::SAMPLED_IMAGE, "SAMPLED_IMAGE"),
    (
        vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
        "SAMPLED_IMAGE_FILTER_LINEAR",
    ),
    (vk::FormatFeatureFlags::TRANSFER_DST, "TRANSFER_DST"),
];

/// What `format` supports with OPTIMAL tiling: one line per format, and
/// whether every needed feature is there. Linear tiling (texels row by row,
/// mappable) supports far less and is not used for textures.
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

/// The objects `upload_texture` creates only for the upload. Dropping it waits
/// for the device to go idle, then destroys them: if a `?` returns early after
/// the submit, the GPU may still use them (step 15's UploadScratch; the C++
/// version gets the same from its IdleGuard and RAII). Destroying a null
/// handle is a no-op, so fields not created yet need no special case.
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

/// Step 15's staging upload, for an image: CPU -> staging buffer ->
/// vkCmdCopyBufferToImage2 on the graphics queue, with the two layout
/// transitions around the copy. Blocks until the copy is complete.
fn upload_texture(
    device: &ash::Device,
    mem: &vk::PhysicalDeviceMemoryProperties,
    (family, queue): (u32, vk::Queue),
    image: vk::Image,
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
    let size = vk::DeviceSize::try_from(pixels.len())?;
    let staging = create_buffer(
        device,
        mem,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    scratch.staging = staging.buffer;
    scratch.staging_memory = staging.memory;
    write_mapped(device, staging.memory, pixels)?;
    println!(
        "Staging buffer: {size} bytes, TRANSFER_SRC, memory type {}",
        staging.memory_type
    );

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
    let color_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);

    // #region to-transfer-dst
    // Barrier 1: UNDEFINED -> TRANSFER_DST_OPTIMAL, the only layout a copy may
    // write. UNDEFINED = "the contents may be thrown away", which is right for
    // a fresh image. Nothing earlier touches it, so the first scope is empty
    // (stage NONE, access NONE); the second scope is the copy's write
    // (COPY, TRANSFER_WRITE), which must wait for the transition.
    let to_transfer = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::NONE)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(color_range);
    // #endregion

    // #region copy
    // One region: the whole buffer into mip level 0, layer 0. Row length and
    // image height 0 = "tightly packed": the rows follow each other with no
    // padding, exactly as `texture_pixels` wrote them. The copy converts the
    // linear rows of the buffer into the image's hidden optimal tiling.
    let region = vk::BufferImageCopy2::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
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
    // #endregion

    // #region to-shader-read
    // Barrier 2: TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL, the layout
    // a sampled image is read in. First scope: the copy's write (made
    // available); second scope: the fragment shader's sampled reads (made
    // visible), in this and every later submission on the queue.
    let to_shader = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COPY)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(color_range);
    // #endregion

    // SAFETY: `cmd` is recording; the image, the buffer and the region are valid.
    unsafe {
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&to_transfer)),
        );
        device.cmd_copy_buffer_to_image2(cmd, &copy_info);
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_shader)),
        );
        device.end_command_buffer(cmd)?;
    }
    println!(
        "Recorded: barrier UNDEFINED -> TRANSFER_DST_OPTIMAL, vkCmdCopyBufferToImage2 ({TEX_SIZE}x{TEX_SIZE}, mip 0), barrier TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL"
    );

    // The fence says the copy is complete, so the staging buffer may go; the
    // barrier above, not the fence, makes the texels visible to the shader.
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
    println!("Upload complete (fence waited); staging buffer destroyed");
    println!(
        "[timing] texture upload {:.3} ms (staging write + record + submit + wait)",
        start.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}

// #region frame-resources
/// What the CPU touches while recording one frame (step 17): command buffer,
/// sync objects, the slot's uniform buffer and its descriptor set, which now
/// also points to the texture.
struct Frame {
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
    uniform: GpuBuffer,
    mapped: *mut Camera,
    descriptor_set: vk::DescriptorSet,
}
// #endregion

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
    command_pool: vk::CommandPool,
    frames: Vec<Frame>,
    frame_index: usize,
    /// One per swapchain image, indexed by the acquired image index.
    render_finished: Vec<vk::Semaphore>,
    start: Instant,
    window: Window,
}

impl Vk {
    /// Steps 01-17, plus the texture, its view, the sampler and a second
    /// binding in the descriptor set.
    fn new(event_loop: &ActiveEventLoop, args: &Args) -> Result<Self, Box<dyn Error>> {
        // Window from step 06: 800x600 physical pixels, fixed size.
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("Vulkan from Zero - 21 texture")
                .with_inner_size(PhysicalSize::new(800, 600))
                .with_resizable(false),
        )?;
        let size = window.inner_size();
        println!("Window: {}x{} pixels", size.width, size.height);

        // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vulkan from Zero - 21 texture")
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
        // graphics + present family, VK_KHR_swapchain, discrete first. New:
        // samplerAnisotropy, which every Vulkan 1.4 device must support (spec,
        // "Feature Requirements") but which still has to be ENABLED.
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

        // #region device-features
        // Step 07's device, plus samplerAnisotropy from VkPhysicalDeviceFeatures
        // (pEnabledFeatures; the 1.3 features stay in the pNext chain).
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
        // #endregion

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

        // #region texture-format
        // The texels are COLOR, authored on a monitor: their values are sRGB
        // encoded (non-linear). With R8G8B8A8_SRGB the sampler decodes them to
        // linear before filtering, so blending between texels and the shader
        // math happen in linear light, and the _SRGB swapchain re-encodes on
        // write. R8G8B8A8_UNORM would hand the shader the raw encoded numbers:
        // right for data (normal maps, roughness, masks), wrong for color - the
        // already-encoded values are taken as linear, the _SRGB swapchain
        // encodes them a second time (too bright, washed out), and filtering
        // blends non-linear values.
        let pixels = texture_pixels();
        println!(
            "\nTexture: {TEX_SIZE}x{TEX_SIZE} RGBA8, {} bytes, generated on the CPU (FNV-1a {:#010x})",
            pixels.len(),
            fnv1a(&pixels)
        );
        let tex_format = vk::Format::R8G8B8A8_SRGB;
        println!("Format support (vkGetPhysicalDeviceFormatProperties2):");
        let srgb_ok = check_format(&instance, physical_device, tex_format);
        check_format(&instance, physical_device, vk::Format::R8G8B8A8_UNORM);
        if !srgb_ok {
            return Err(format!("{tex_format:?} lacks a needed optimal-tiling feature").into());
        }
        println!("Texture format: {tex_format:?} (color data is sRGB-encoded)");
        // #endregion

        // #region create-image
        // OPTIMAL tiling: the driver arranges the texels (tiles, swizzles,
        // maybe compression) as the GPU's texture units like them; the CPU can
        // neither map nor address them, hence the copy from a buffer. Usage:
        // TRANSFER_DST (the copy writes it) | SAMPLED (shaders read it).
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(tex_format)
            .extent(vk::Extent3D {
                width: TEX_SIZE,
                height: TEX_SIZE,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: plain create call on a live device; the format supports this use.
        let texture = unsafe { device.create_image(&image_info, None)? };
        println!("VkImage: 2D, 1 mip level, OPTIMAL tiling, TRANSFER_DST | SAMPLED");
        // #endregion

        // #region image-memory
        // Images ask for memory like buffers (step 13), but the size is the
        // driver's: tiling and alignment padding make it differ from
        // width * height * 4, and the allowed types may differ from buffers'.
        // SAFETY: `texture` belongs to this device.
        let req = unsafe { device.get_image_memory_requirements(texture) };
        let tex_type = find_type(
            &mem,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or("no DEVICE_LOCAL memory type for the texture")?;
        println!(
            "Memory requirements: size {}, alignment {}, memoryTypeBits {:#b} -> memory type {tex_type} (DEVICE_LOCAL)",
            req.size, req.alignment, req.memory_type_bits
        );
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(tex_type);
        // SAFETY: the type index comes from this device's memory properties.
        let texture_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        // SAFETY: the allocation has the required size and type; offset 0 is aligned.
        unsafe { device.bind_image_memory(texture, texture_memory, 0)? };
        // #endregion

        upload_texture(&device, &mem, (family, queue), texture, &pixels)?;

        // #region image-view
        // The view is what descriptors point to: the image seen as a 2D
        // texture of this format, color aspect, mip 0, layer 0.
        let view_info = vk::ImageViewCreateInfo::default()
            .image(texture)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(tex_format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );
        // SAFETY: `texture` belongs to this device and has memory bound.
        let texture_view = unsafe { device.create_image_view(&view_info, None)? };
        // #endregion

        // #region sampler
        // The sampler is independent of any image: filters (magnification:
        // a texel covers many pixels; minification: many texels per pixel),
        // address modes for UVs outside [0, 1] (u, v and w), anisotropy, and
        // the mip settings (one level here: maxLod 0). maxLod 0 clamps the LOD
        // lambda to <= 0, so the texture always counts as magnified and
        // magFilter is the one used; harmless here because both filters are
        // equal (spec Note "Mapping of OpenGL to Vulkan Filter Modes": GL's
        // non-mipmapped minification is maxLod 0.25). Anisotropic filtering
        // takes up to maxAnisotropy samples along the direction in which the
        // texture is squeezed on screen; 16 is the usual maximum, clamped to
        // the device limit (VUID-VkSamplerCreateInfo-anisotropyEnable-01071).
        // With --nearest it is off, so NEAREST shows its raw blockiness.
        let max_aniso = chosen.limits.max_sampler_anisotropy;
        let anisotropy = (args.filter == vk::Filter::LINEAR).then_some(max_aniso.min(16.0));
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(args.filter)
            .min_filter(args.filter)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(args.address)
            .address_mode_v(args.address)
            .address_mode_w(args.address)
            .mip_lod_bias(0.0)
            .anisotropy_enable(anisotropy.is_some())
            .max_anisotropy(anisotropy.unwrap_or(1.0))
            .compare_enable(false)
            .min_lod(0.0)
            .max_lod(0.0)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
            .unnormalized_coordinates(false);
        // SAFETY: samplerAnisotropy is enabled; maxAnisotropy is within the limit.
        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };
        println!("maxSamplerAnisotropy = {max_aniso}");
        match anisotropy {
            Some(a) => println!(
                "Sampler: {:?} filters, address mode {:?}, anisotropy {a}",
                args.filter, args.address
            ),
            None => println!(
                "Sampler: {:?} filters, address mode {:?}, anisotropy off",
                args.filter, args.address
            ),
        }
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

        // Vertex input: binding 0, stride 20, position (3 floats) + uv (2).
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

        // #region set-layout
        // Set 0 now has two bindings: 0 = step 17's uniform buffer (vertex
        // stage), 1 = the texture as ONE combined image sampler (fragment
        // stage). Combined = view + sampler in one descriptor, the simplest
        // form and what Sampler2D in Slang/GLSL's sampler2D map to. The other
        // option is SAMPLED_IMAGE + SAMPLER as two bindings (Texture2D and
        // SamplerState in the shader): worth it when one sampler serves many
        // images, as in bindless renderers (step 20).
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
        println!(
            "Descriptor set layout: binding 0 = UNIFORM_BUFFER (VERTEX), binding 1 = COMBINED_IMAGE_SAMPLER (FRAGMENT)"
        );
        // #endregion

        let set_layouts = [set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        // SAFETY: `set_layout` belongs to this device.
        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

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
        println!(
            "Mesh: quad, {} vertices, {} indices, UVs 0..2",
            VERTICES.len(),
            INDICES.len()
        );

        // Step 17's per-frame uniform buffers, persistently mapped.
        let uniform_size = vk::DeviceSize::try_from(size_of::<Camera>())?;
        let mut uniforms = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
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
            uniforms.push((buffer, ptr.cast::<Camera>()));
        }

        // #region descriptor-pool
        // Room for exactly what we allocate: 2 sets, each with one uniform
        // buffer and one combined image sampler.
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
        println!(
            "Descriptor pool: maxSets {frames_u32}, UNIFORM_BUFFER x {frames_u32}, COMBINED_IMAGE_SAMPLER x {frames_u32}; {} sets allocated",
            descriptor_sets.len()
        );
        // #endregion

        // #region write-sets
        // Each set gets its own uniform buffer (binding 0) and the same
        // texture (binding 1). An image descriptor carries the LAYOUT the
        // image will be in when the shader reads it: SHADER_READ_ONLY_OPTIMAL,
        // the layout the upload's last barrier left it in.
        let image_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(texture_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = uniforms
            .iter()
            .map(|(b, _)| {
                vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(uniform_size)
            })
            .collect();
        let mut writes = Vec::with_capacity(2 * FRAMES_IN_FLIGHT);
        for (&set, info) in descriptor_sets.iter().zip(&buffer_infos) {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(info)),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_info),
            );
        }
        // SAFETY: every set, buffer, view and sampler outlives the call; no
        // command buffer uses the sets yet.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        println!(
            "vkUpdateDescriptorSets: {} writes (per set: binding 0 -> its uniform buffer, binding 1 -> texture view + sampler, SHADER_READ_ONLY_OPTIMAL)",
            writes.len()
        );
        // #endregion

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
        println!("\n{FRAMES_IN_FLIGHT} frames in flight; the quad turns at {SPIN} rad/s");

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
            command_pool,
            frames,
            frame_index: 0,
            render_finished,
            start: Instant::now(),
            window,
        })
    }

    /// Renders and presents one frame (step 17's loop). Returns `Ok(false)`
    /// when the swapchain is out of date (recreation is step 11).
    fn draw_frame(&mut self) -> Result<bool, Box<dyn Error>> {
        let device = &self.device;
        let slot = self.frame_index % self.frames.len();
        let frame = self.frames.get(slot).ok_or("frame slot out of range")?;
        let cmd = frame.command_buffer;

        // SAFETY: the fence belongs to this device.
        unsafe { device.wait_for_fences(&[frame.in_flight], true, u64::MAX)? };

        let time = self.start.elapsed().as_secs_f32();
        let aspect = self.extent.height as f32 / self.extent.width as f32;
        // SAFETY: `mapped` points to this slot's mapped allocation of at least
        // size_of::<Camera>() bytes; no pending GPU work reads it (fence above).
        unsafe {
            frame.mapped.write(Camera {
                mvp: transform(time * SPIN, aspect),
            });
        }

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
        let index_count = u32::try_from(INDICES.len())?;
        // #region bind-and-draw
        // Step 17's draw: binding the set binds both descriptors at once; the
        // texture needs no per-frame work, it never changes.
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
        // #region cleanup
        // Wait until the GPU is idle, then destroy: the descriptor pool (its
        // sets), the sampler and the view before the image, the image before
        // its memory.
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
        // #endregion
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
