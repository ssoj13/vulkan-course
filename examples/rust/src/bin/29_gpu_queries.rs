//! Step 29 - GPU queries: timestamps, pipeline statistics, calibration.
//!
//! Step 22's two cubes, measured from the inside. A QUERY is a slot in a
//! VkQueryPool that the GPU fills while it executes a command buffer; the
//! host reads it back once the work has completed:
//!
//!   TIMESTAMP queries     vkCmdWriteTimestamp2: a device tick count at a point
//!                         of the command stream -> GPU time per pass.
//!   PIPELINE STATISTICS   vkCmdBeginQuery/EndQuery: counters of what the
//!                         pipeline did (vertices, primitives, invocations).
//!   CALIBRATED TIMESTAMPS vkGetCalibratedTimestampsKHR: the device clock and a
//!                         host clock read "at the same time". The device value
//!                         read right before the submit -> how long a submitted
//!                         frame waits before the GPU starts it; the host value
//!                         only measures the host clock's rate (the QPC line).
//!
//! Queries are reset on the HOST (vkResetQueryPool, core in Vulkan 1.2 behind
//! the hostQueryReset feature) right after their results are read, instead of
//! with vkCmdResetQueryPool in the command buffer.
//!
//! `--frames N` (1-3, default 2) frames in flight, `--no-stats` leaves the
//! statistics query out, `--seconds S` closes the window after S seconds.
//! Lines that depend on timing, or on the animation at the moment they are
//! printed, start with `[timing]`. Steps 01-22 are repeated, reduced to what
//! this step needs.

use std::error::Error;
use std::ffi::c_void;
use std::time::Instant;

use ash::{
    Entry,
    ext::debug_utils,
    khr::{calibrated_timestamps, surface, swapchain},
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
        .map(|d| unsafe { std::ffi::CStr::from_ptr(d.p_message) }.to_string_lossy())
        .unwrap_or(std::borrow::Cow::Borrowed("<no message>"));
    eprintln!("[{level}][{kind}] {message}");
    vk::FALSE
}

/// Position + color: 3 + 3 floats = 24 bytes (shaders/cube.slang).
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    /// Linear RGB, one color per face.
    color: [f32; 3],
}
const _: () = assert!(size_of::<Vertex>() == 24);

/// Step 22's unit cube: 4 vertices per face, counter-clockwise from outside.
#[rustfmt::skip] // one vertex per line reads like a table
const VERTICES: [Vertex; 24] = [
    // +x, red
    Vertex { position: [0.5, -0.5, -0.5],  color: [0.9, 0.2, 0.2] },
    Vertex { position: [0.5, 0.5, -0.5],   color: [0.9, 0.2, 0.2] },
    Vertex { position: [0.5, 0.5, 0.5],    color: [0.9, 0.2, 0.2] },
    Vertex { position: [0.5, -0.5, 0.5],   color: [0.9, 0.2, 0.2] },
    // -x, cyan
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.2, 0.8, 0.8] },
    Vertex { position: [-0.5, -0.5, 0.5],  color: [0.2, 0.8, 0.8] },
    Vertex { position: [-0.5, 0.5, 0.5],   color: [0.2, 0.8, 0.8] },
    Vertex { position: [-0.5, 0.5, -0.5],  color: [0.2, 0.8, 0.8] },
    // +y, green
    Vertex { position: [-0.5, 0.5, -0.5],  color: [0.2, 0.85, 0.3] },
    Vertex { position: [-0.5, 0.5, 0.5],   color: [0.2, 0.85, 0.3] },
    Vertex { position: [0.5, 0.5, 0.5],    color: [0.2, 0.85, 0.3] },
    Vertex { position: [0.5, 0.5, -0.5],   color: [0.2, 0.85, 0.3] },
    // -y, magenta
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.8, 0.3, 0.8] },
    Vertex { position: [0.5, -0.5, -0.5],  color: [0.8, 0.3, 0.8] },
    Vertex { position: [0.5, -0.5, 0.5],   color: [0.8, 0.3, 0.8] },
    Vertex { position: [-0.5, -0.5, 0.5],  color: [0.8, 0.3, 0.8] },
    // +z, blue
    Vertex { position: [-0.5, -0.5, 0.5],  color: [0.25, 0.4, 0.95] },
    Vertex { position: [0.5, -0.5, 0.5],   color: [0.25, 0.4, 0.95] },
    Vertex { position: [0.5, 0.5, 0.5],    color: [0.25, 0.4, 0.95] },
    Vertex { position: [-0.5, 0.5, 0.5],   color: [0.25, 0.4, 0.95] },
    // -z, yellow
    Vertex { position: [-0.5, -0.5, -0.5], color: [0.95, 0.85, 0.2] },
    Vertex { position: [-0.5, 0.5, -0.5],  color: [0.95, 0.85, 0.2] },
    Vertex { position: [0.5, 0.5, -0.5],   color: [0.95, 0.85, 0.2] },
    Vertex { position: [0.5, -0.5, -0.5],  color: [0.95, 0.85, 0.2] },
];
/// Two triangles per face: (0, 1, 2) and (0, 2, 3) of its 4 corners.
const INDICES: [u16; 36] = [
    0, 1, 2, 0, 2, 3, //
    4, 5, 6, 4, 6, 7, //
    8, 9, 10, 8, 10, 11, //
    12, 13, 14, 12, 14, 15, //
    16, 17, 18, 16, 18, 19, //
    20, 21, 22, 20, 22, 23, //
];
/// One draw of both cubes (instance i reads mvp[i], step 22).
const INSTANCES: u32 = 2;

/// A 4x4 matrix, column-major: `m[c]` is column c (step 17).
type Mat4 = [[f32; 4]; 4];

/// The host side of `Scene` in shaders/cube.slang: one matrix per cube.
#[repr(C)]
#[derive(Clone, Copy)]
struct Scene {
    mvp: [Mat4; 2],
}
const _: () = assert!(size_of::<Scene>() == 128);

/// a * b for column-major matrices: (a * b) * v = a * (b * v).
fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for (c, col) in m.iter_mut().enumerate() {
        for (r, out) in col.iter_mut().enumerate() {
            *out = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    m
}

/// Moves points by (x, y, z).
fn translate(x: f32, y: f32, z: f32) -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [x, y, z, 1.0],
    ]
}

/// Rotation by `a` radians about +x.
fn rotate_x(a: f32) -> Mat4 {
    let (s, c) = a.sin_cos();
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, s, 0.0],
        [0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Rotation by `a` radians about +y.
fn rotate_y(a: f32) -> Mat4 {
    let (s, c) = a.sin_cos();
    [
        [c, 0.0, -s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Step 22's perspective projection into Vulkan's clip space (y down, depth 0..1).
fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fovy / 2.0).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ]
}

/// Camera: 60 degrees vertical field of view, depth range 0.1..100.
const FOVY: f32 = std::f32::consts::FRAC_PI_3;
const NEAR: f32 = 0.1;
const FAR: f32 = 100.0;

/// Step 22's scene: both cubes' model-view-projection matrices at `time`.
fn scene(time: f32, proj: &Mat4) -> Scene {
    let view = translate(0.0, 0.0, -3.5);
    let model0 = mul(&rotate_y(time * 0.7), &rotate_x(time * 0.4));
    let model1 = mul(
        &translate(0.45, 0.15, 0.1),
        &mul(&rotate_y(-time * 0.5), &rotate_x(0.6)),
    );
    let view_proj = mul(proj, &view);
    Scene {
        mvp: [mul(&view_proj, &model0), mul(&view_proj, &model1)],
    }
}

// #region cli
/// Command line: `--frames N`, `--no-stats`, `--seconds S`.
struct Args {
    frames: usize,
    stats: bool,
    seconds: Option<f64>,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut args = Self {
            frames: 2,
            stats: true,
            seconds: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--frames" => {
                    let value = it.next().ok_or("--frames needs a value")?;
                    args.frames = value.parse()?;
                    if !(1..=3).contains(&args.frames) {
                        return Err("--frames must be 1, 2 or 3".into());
                    }
                }
                "--no-stats" => args.stats = false,
                "--seconds" => {
                    let value = it.next().ok_or("--seconds needs a value")?;
                    args.seconds = Some(value.parse()?);
                }
                _ => {
                    return Err(format!(
                        "unknown argument {arg} (use --frames N, --no-stats, --seconds S)"
                    )
                    .into());
                }
            }
        }
        Ok(args)
    }
}
// #endregion

// #region queries
/// Timestamps per frame: start, attachments ready, main pass done, end.
const TIMESTAMPS: u32 = 4;
/// What each pair of consecutive timestamps encloses.
const PASSES: [&str; 3] = ["attachments", "main pass", "to present"];

/// The six counters this step asks for. Results come back in the order of
/// the flag bits (spec: "in the order of the bits"), one u64 each.
const STATISTICS: vk::QueryPipelineStatisticFlags = vk::QueryPipelineStatisticFlags::from_raw(
    vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_VERTICES.as_raw()
        | vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_PRIMITIVES.as_raw()
        | vk::QueryPipelineStatisticFlags::VERTEX_SHADER_INVOCATIONS.as_raw()
        | vk::QueryPipelineStatisticFlags::CLIPPING_INVOCATIONS.as_raw()
        | vk::QueryPipelineStatisticFlags::CLIPPING_PRIMITIVES.as_raw()
        | vk::QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS.as_raw(),
);
/// One statistics query's result: the six counters in bit order.
type Statistics = [u64; 6];

/// Ticks between two timestamps when only the low `bits` are valid: the
/// difference modulo 2^bits, read as a signed number (the counter wraps at
/// 2^bits; right while the two lie less than 2^(bits-1) ticks apart).
fn ticks_between(from: u64, to: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((to.wrapping_sub(from) << shift) as i64) >> shift
}
// #endregion

// #region calibration
/// A device timestamp and a host timestamp captured together by
/// vkGetCalibratedTimestampsKHR, with the call's maximum deviation in ns.
#[derive(Clone, Copy)]
struct Calibration {
    device: u64,
    host: u64,
    max_deviation_ns: u64,
}

/// The host time domain to pair with VK_TIME_DOMAIN_DEVICE_KHR: the Windows
/// performance counter (QueryPerformanceCounter units) or, on Linux,
/// CLOCK_MONOTONIC_RAW / CLOCK_MONOTONIC (nanoseconds).
fn host_domain(domains: &[vk::TimeDomainKHR]) -> Option<vk::TimeDomainKHR> {
    [
        vk::TimeDomainKHR::QUERY_PERFORMANCE_COUNTER,
        vk::TimeDomainKHR::CLOCK_MONOTONIC_RAW,
        vk::TimeDomainKHR::CLOCK_MONOTONIC,
    ]
    .into_iter()
    .find(|d| domains.contains(d))
}
// #endregion

/// Sums over one second (and over the whole run) of what the queries showed.
#[derive(Default, Clone, Copy)]
struct Sums {
    frames: u64,
    /// GPU microseconds per pass, then the whole frame.
    pass_us: [f64; 4],
    /// Submit -> GPU start, microseconds: sum, min, max, and how many frames.
    latency_us: (f64, f64, f64, u64),
    max_deviation_us: f64,
    fragment_invocations: u64,
}

impl Sums {
    fn print(&self, label: &str) {
        if self.frames == 0 {
            return;
        }
        let n = self.frames as f64;
        let [a, m, p, whole] = self.pass_us.map(|us| us / n);
        let mut line = format!(
            "[timing] {label}: {} frames, GPU us per frame: {} {a:.1}, {} {m:.1}, {} {p:.1}, whole {whole:.1}",
            self.frames, PASSES[0], PASSES[1], PASSES[2]
        );
        let (sum, min, max, count) = self.latency_us;
        if count > 0 {
            let c = count as f64;
            line += &format!(
                "; submit -> GPU start {:.1} us (min {min:.1}, max {max:.1}), maxDeviation {:.1} us",
                sum / c,
                self.max_deviation_us / c
            );
        }
        if self.fragment_invocations > 0 {
            line += &format!(
                "; FS invocations {:.0}",
                self.fragment_invocations as f64 / n
            );
        }
        println!("{line}");
    }
}

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

/// A buffer and the memory bound to it (step 14).
struct GpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

/// Step 14's manual path: create, query requirements, allocate, bind.
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
    let memory_type = find_type(props, req.memory_type_bits, flags)
        .ok_or("no suitable memory type for this buffer")?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(memory_type);
    // SAFETY: the type index comes from this device's memory properties.
    let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    // SAFETY: the allocation has the required size and type; offset 0 is aligned.
    unsafe { device.bind_buffer_memory(buffer, memory, 0)? };
    Ok(GpuBuffer { buffer, memory })
}

/// Step 14's upload into HOST_VISIBLE | HOST_COHERENT memory.
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

/// Step 22's depth formats in order of preference.
const DEPTH_CANDIDATES: [vk::Format; 3] = [
    vk::Format::D32_SFLOAT,
    vk::Format::D24_UNORM_S8_UINT,
    vk::Format::D16_UNORM,
];

/// The first candidate whose OPTIMAL tiling supports DEPTH_STENCIL_ATTACHMENT.
fn pick_depth_format(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
) -> Result<vk::Format, Box<dyn Error>> {
    DEPTH_CANDIDATES
        .into_iter()
        .find(|&format| {
            let mut props = vk::FormatProperties2::default();
            // SAFETY: `pd` was enumerated from this instance; `props` outlives the call.
            unsafe { instance.get_physical_device_format_properties2(pd, format, &mut props) };
            props
                .format_properties
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        })
        .ok_or_else(|| "no depth format supports DEPTH_STENCIL_ATTACHMENT".into())
}

/// Whether `format` also has a stencil component (step 22).
fn has_stencil(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
    )
}

/// One frame slot (step 17) plus what its queries need: its queries are
/// timestamps [slot * 4, slot * 4 + 4) and statistics query `slot`.
struct Frame {
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
    uniform: GpuBuffer,
    mapped: *mut Scene,
    descriptor_set: vk::DescriptorSet,
    /// The frame this slot rendered last and the calibration taken at its
    /// submit; None until the slot's first submit.
    pending: Option<(usize, Option<Calibration>)>,
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
    /// VK_KHR_calibrated_timestamps and the host domain, if supported.
    calibration: Option<(calibrated_timestamps::Device, vk::TimeDomainKHR)>,
    swapchain_loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    image_views: Vec<vk::ImageView>,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    vertex_buffer: GpuBuffer,
    index_buffer: GpuBuffer,
    depth_layout: vk::ImageLayout,
    depth_aspect: vk::ImageAspectFlags,
    has_stencil: bool,
    depth_image: vk::Image,
    depth_memory: vk::DeviceMemory,
    depth_view: vk::ImageView,
    descriptor_pool: vk::DescriptorPool,
    timestamp_pool: vk::QueryPool,
    /// None with `--no-stats`.
    stats_pool: Option<vk::QueryPool>,
    /// Nanoseconds per timestamp tick (timestampPeriod).
    period_ns: f64,
    /// Valid bits of this queue family's timestamps (1-64).
    valid_bits: u32,
    queue: vk::Queue,
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    proj: Mat4,
    command_pool: vk::CommandPool,
    frames: Vec<Frame>,
    frame_index: usize,
    render_finished: Vec<vk::Semaphore>,
    start: Instant,
    /// The animation clock: starts at frame 0, so frame 0 always shows the
    /// scene at time 0 and its statistics are the same in every run.
    anim_start: Option<Instant>,
    second: u64,
    current: Sums,
    total: Sums,
    /// The first and the latest calibration (for the host clock's rate).
    first_calibration: Option<Calibration>,
    last_calibration: Option<Calibration>,
    window: Window,
}

impl Vk {
    /// Steps 01-22, plus the query pools and the calibration extension.
    fn new(event_loop: &ActiveEventLoop, args: &Args) -> Result<Self, Box<dyn Error>> {
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("Vulkan from Zero - 29 gpu queries")
                .with_inner_size(PhysicalSize::new(800, 600))
                .with_resizable(false),
        )?;
        let size = window.inner_size();
        println!("Window: {}x{} pixels", size.width, size.height);

        // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
        let entry = unsafe { Entry::load()? };
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vulkan from Zero - 29 gpu queries")
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
        let surface_factory = ash_window::SurfaceFactory::new(&entry, &instance, display)?;
        // SAFETY: the window handle is valid, and `Vk` destroys the surface
        // before the window and the instance (see `destroy`).
        let surface =
            unsafe { surface_factory.create_surface(window.window_handle()?.as_raw(), None)? };
        let surface_loader = surface::Instance::load(&entry, &instance);

        // SAFETY: `instance` is alive.
        let devices = unsafe { instance.enumerate_physical_devices()? };
        // #region pick-device
        // Steps 03 + 06, plus what this step needs: the hostQueryReset feature
        // (Vulkan 1.2) and pipelineStatisticsQuery (Vulkan 1.0, optional); a
        // graphics family whose timestamps are valid (timestampValidBits > 0).
        // VK_KHR_calibrated_timestamps is used when the device has it.
        let mut best: Option<(u32, vk::PhysicalDevice, u32)> = None;
        for &pd in &devices {
            // SAFETY: `pd` was enumerated from this instance.
            let props = unsafe { instance.get_physical_device_properties(pd) };
            if props.api_version < vk::API_VERSION_1_4 {
                continue;
            }
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default()
                .push(&mut f12)
                .push(&mut f13);
            // SAFETY: `pd` came from this instance, and the chain structs outlive the call.
            unsafe { instance.get_physical_device_features2(pd, &mut f2) };
            let stats_ok = !args.stats || f2.features.pipeline_statistics_query != 0;
            if f13.dynamic_rendering == 0
                || f13.synchronization2 == 0
                || f12.host_query_reset == 0
                || !stats_ok
            {
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
                if f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && f.timestamp_valid_bits > 0
                    && present
                {
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
            "No Vulkan 1.4 GPU with dynamic rendering, synchronization2, hostQueryReset, pipelineStatisticsQuery, timestamps and VK_KHR_swapchain can present to this window",
        )?;
        // SAFETY: `physical_device` was enumerated from this instance.
        let chosen = unsafe { instance.get_physical_device_properties(physical_device) };
        println!(
            "Picked: {}, graphics + present family {family}",
            chosen.device_name_as_c_str()?.to_string_lossy()
        );
        // SAFETY: `physical_device` was enumerated from this instance.
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let valid_bits = families
            .get(usize::try_from(family)?)
            .ok_or("queue family out of range")?
            .timestamp_valid_bits
            .min(64);
        let period_ns = f64::from(chosen.limits.timestamp_period);
        println!(
            "Timestamps: timestampPeriod {} ns per tick, {valid_bits} valid bits on family {family}",
            chosen.limits.timestamp_period
        );
        // SAFETY: `physical_device` was enumerated from this instance.
        let exts = unsafe { instance.enumerate_device_extension_properties(physical_device)? };
        let has_calibration = exts
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(calibrated_timestamps::NAME));
        // #endregion

        // #region device
        // One queue; hostQueryReset (Vulkan12Features), the 1.3 features,
        // pipelineStatisticsQuery (VkPhysicalDeviceFeatures) unless
        // --no-stats, VK_KHR_swapchain and, if present,
        // VK_KHR_calibrated_timestamps.
        let priorities = [1.0_f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let features = vk::PhysicalDeviceFeatures::default().pipeline_statistics_query(args.stats);
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default().host_query_reset(true);
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let mut device_exts = vec![swapchain::NAME.as_ptr()];
        if has_calibration {
            device_exts.push(calibrated_timestamps::NAME.as_ptr());
        }
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_exts)
            .enabled_features(&features)
            .push(&mut features12)
            .push(&mut features13);
        // SAFETY: `physical_device` belongs to `instance`, and all create-info data outlives the call.
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        let swapchain_loader = swapchain::Device::load(&instance, &device);
        // SAFETY: one queue of `family` was requested in `queue_infos`.
        let queue = unsafe { device.get_device_queue(family, 0) };
        // #endregion

        // #region time-domains
        // Which clocks this device can sample together with its own. DEVICE
        // is the clock of vkCmdWriteTimestamp2 ("comparable with device
        // timestamp values"); QUERY_PERFORMANCE_COUNTER is Windows'
        // QueryPerformanceCounter. Reading both in one call is what makes
        // GPU ticks and CPU time comparable.
        let calibration = if has_calibration {
            let loader = calibrated_timestamps::Instance::load(&entry, &instance);
            // SAFETY: `physical_device` was enumerated from this instance.
            let domains =
                unsafe { loader.get_physical_device_calibrateable_time_domains(physical_device)? };
            let names: Vec<String> = domains.iter().map(|d| format!("{d:?}")).collect();
            println!("Calibrateable time domains: {}", names.join(", "));
            match (
                domains.contains(&vk::TimeDomainKHR::DEVICE),
                host_domain(&domains),
            ) {
                (true, Some(host)) => {
                    println!("Calibration pairs DEVICE with {host:?}");
                    Some((
                        calibrated_timestamps::Device::load(&instance, &device),
                        host,
                    ))
                }
                _ => {
                    println!("No DEVICE + host domain pair: no submit -> GPU start latency");
                    None
                }
            }
        } else {
            println!("VK_KHR_calibrated_timestamps: not supported, no submit -> GPU start latency");
            None
        };
        // #endregion

        // Step 07's swapchain.
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
            "Swapchain: {}x{} {:?}, {present_mode:?}, {} images; {} frame(s) in flight",
            extent.width,
            extent.height,
            format.format,
            images.len(),
            args.frames
        );
        // SAFETY: `physical_device` was enumerated from this instance.
        let mem = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // Step 22's depth image, one for all frames in flight.
        let depth_format = pick_depth_format(&instance, physical_device)?;
        let has_stencil = has_stencil(depth_format);
        let (depth_layout, depth_aspect) = if has_stencil {
            (
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            )
        } else {
            (
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::DEPTH,
            )
        };
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(depth_format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: plain create call on a live device; the format supports this use.
        let depth_image = unsafe { device.create_image(&image_info, None)? };
        // SAFETY: `depth_image` belongs to this device.
        let req = unsafe { device.get_image_memory_requirements(depth_image) };
        let depth_type = find_type(
            &mem,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or("no DEVICE_LOCAL memory type for the depth image")?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(depth_type);
        // SAFETY: the type index comes from this device's memory properties.
        let depth_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        // SAFETY: the allocation has the required size and type; offset 0 is aligned.
        unsafe { device.bind_image_memory(depth_image, depth_memory, 0)? };
        let view_info = vk::ImageViewCreateInfo::default()
            .image(depth_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(depth_format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(depth_aspect)
                    .level_count(1)
                    .layer_count(1),
            );
        // SAFETY: `depth_image` belongs to this device and has memory bound.
        let depth_view = unsafe { device.create_image_view(&view_info, None)? };

        let aspect = extent.width as f32 / extent.height as f32;
        let proj = perspective(FOVY, aspect, NEAR, FAR);

        // Step 22's pipeline (cube.spv, depth test LESS).
        const CUBE_SPV: &[u8] = include_bytes!("../../../../shaders/spv/cube.spv");
        let spirv = ash::util::read_spv(&mut std::io::Cursor::new(CUBE_SPV))?;
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
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(u32::try_from(std::mem::offset_of!(Vertex, color))?),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let set_bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];
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
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);
        let stencil_format = if has_stencil {
            depth_format
        } else {
            vk::Format::UNDEFINED
        };
        let color_formats = [format.format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format)
            .stencil_attachment_format(stencil_format);
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .push(&mut rendering_info);
        // SAFETY: all create-info data outlives the call; no pipeline cache.
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

        // Step 14's mesh buffers.
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

        // #region query-pools
        // Two pools, one per query type (a pool holds one type): 4 timestamps
        // per frame slot, and one statistics query per slot with the six
        // counters. A query must be reset before its first use; with
        // hostQueryReset the host does it, here for all queries at once and
        // later per slot after its results are read.
        let frames_u32 = u32::try_from(args.frames)?;
        let pool_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(TIMESTAMPS * frames_u32);
        // SAFETY: plain create call on a live device.
        let timestamp_pool = unsafe { device.create_query_pool(&pool_info, None)? };
        // SAFETY: no command uses the pool yet; hostQueryReset is enabled.
        unsafe { device.reset_query_pool(timestamp_pool, 0, TIMESTAMPS * frames_u32) };
        let stats_pool = if args.stats {
            let pool_info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::PIPELINE_STATISTICS)
                .query_count(frames_u32)
                .pipeline_statistics(STATISTICS);
            // SAFETY: plain create call; pipelineStatisticsQuery is enabled.
            let pool = unsafe { device.create_query_pool(&pool_info, None)? };
            // SAFETY: as above.
            unsafe { device.reset_query_pool(pool, 0, frames_u32) };
            Some(pool)
        } else {
            None
        };
        println!(
            "Query pools: {} timestamps ({TIMESTAMPS} per frame slot), {}; reset on the host",
            TIMESTAMPS * frames_u32,
            if args.stats {
                format!("{} pipeline statistics queries (6 counters)", args.frames)
            } else {
                "no statistics (--no-stats)".to_owned()
            }
        );
        // #endregion

        // Step 17's per-frame uniform buffers and descriptor sets.
        let uniform_size = vk::DeviceSize::try_from(size_of::<Scene>())?;
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(frames_u32)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(frames_u32)
            .pool_sizes(&pool_sizes);
        // SAFETY: plain create call on a live device.
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };
        let layouts = vec![set_layout; args.frames];
        let alloc_sets = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        // SAFETY: the pool and the layout belong to this device; the pool has room.
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_sets)? };
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
        let mut frames = Vec::with_capacity(args.frames);
        for (&command_buffer, &descriptor_set) in command_buffers.iter().zip(&descriptor_sets) {
            let uniform = create_buffer(
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
                    uniform.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )?
            };
            let info = vk::DescriptorBufferInfo::default()
                .buffer(uniform.buffer)
                .offset(0)
                .range(uniform_size);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&info));
            // SAFETY: the set and the buffer outlive the call; no command
            // buffer uses the set yet.
            unsafe { device.update_descriptor_sets(&[write], &[]) };
            // SAFETY: plain create calls on a live device.
            let image_available = unsafe { device.create_semaphore(&semaphore_info, None)? };
            // SAFETY: as above.
            let in_flight = unsafe { device.create_fence(&fence_info, None)? };
            frames.push(Frame {
                command_buffer,
                image_available,
                in_flight,
                uniform,
                mapped: ptr.cast::<Scene>(),
                descriptor_set,
                pending: None,
            });
        }
        let mut render_finished = Vec::with_capacity(images.len());
        for _ in &images {
            // SAFETY: as above.
            render_finished.push(unsafe { device.create_semaphore(&semaphore_info, None)? });
        }

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
            calibration,
            swapchain_loader,
            swapchain,
            image_views,
            set_layout,
            pipeline_layout,
            pipeline,
            vertex_buffer,
            index_buffer,
            depth_layout,
            depth_aspect,
            has_stencil,
            depth_image,
            depth_memory,
            depth_view,
            descriptor_pool,
            timestamp_pool,
            stats_pool,
            period_ns,
            valid_bits,
            queue,
            images,
            extent,
            proj,
            command_pool,
            frames,
            frame_index: 0,
            render_finished,
            start: Instant::now(),
            anim_start: None,
            second: 0,
            current: Sums::default(),
            total: Sums::default(),
            first_calibration: None,
            last_calibration: None,
            window,
        })
    }

    // #region read-results
    /// Reads the queries of `slot`, which rendered frame `frame` and whose
    /// fence has signaled, adds them to the sums, then resets them on the
    /// host for the slot's next frame. Frame 0's statistics are printed with
    /// what can be predicted for them.
    fn read_results(
        &mut self,
        slot: usize,
        frame: usize,
        calibration: Option<Calibration>,
    ) -> Result<(), Box<dyn Error>> {
        let slot_u32 = u32::try_from(slot)?;
        let mut ticks = [0_u64; TIMESTAMPS as usize];
        // SAFETY: the queries were written by a submission whose fence has
        // signaled, so WAIT returns at once; they are reset below, after use.
        unsafe {
            self.device.get_query_pool_results(
                self.timestamp_pool,
                slot_u32 * TIMESTAMPS,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
            self.device
                .reset_query_pool(self.timestamp_pool, slot_u32 * TIMESTAMPS, TIMESTAMPS);
        }
        let us = |from: u64, to: u64| {
            ticks_between(from, to, self.valid_bits) as f64 * self.period_ns / 1000.0
        };
        let [t0, t1, t2, t3] = ticks;
        let pass_us = [us(t0, t1), us(t1, t2), us(t2, t3), us(t0, t3)];

        // #region latency
        // The calibration was taken right before vkQueueSubmit2; t0 is the
        // first timestamp of the frame's command buffer. Both are device
        // ticks, so their difference is how long the frame waited between the
        // submit call and the GPU starting it (queue backlog, scheduling,
        // the driver's own submission work).
        let latency = calibration.map(|c| (us(c.device, t0), c.max_deviation_ns as f64 / 1000.0));
        // #endregion

        let mut fragment_invocations = 0;
        if let Some(pool) = self.stats_pool {
            let mut stats = [Statistics::default()];
            // SAFETY: as above, for the statistics query of this slot.
            unsafe {
                self.device.get_query_pool_results(
                    pool,
                    slot_u32,
                    &mut stats,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )?;
                self.device.reset_query_pool(pool, slot_u32, 1);
            }
            let [ia_vertices, ia_primitives, vs, clip_in, clip_out, fs] = stats[0];
            fragment_invocations = fs;
            if frame == 0 {
                // #region statistics
                let indices = INDICES.len() as u64 * u64::from(INSTANCES);
                println!("Pipeline statistics of frame 0 (scene at time 0), measured / predicted:");
                println!(
                    "  input assembly vertices    {ia_vertices:>6} / {indices} = 36 indices x {INSTANCES} instances"
                );
                println!(
                    "  input assembly primitives  {ia_primitives:>6} / {} = triangles of the index list",
                    indices / 3
                );
                println!(
                    "  vertex shader invocations  {vs:>6} / {}..{indices}: reuse of repeated indices is implementation-dependent",
                    VERTICES.len() as u64 * u64::from(INSTANCES)
                );
                println!(
                    "  clipping invocations       {clip_in:>6} / {}: every triangle reaches clipping (culling comes later)",
                    indices / 3
                );
                println!(
                    "  clipping primitives        {clip_out:>6} / at least 1 per triangle with a vertex inside the clip volume that covers a sample, 0 or more otherwise"
                );
                println!(
                    "  fragment shader invocations{fs:>7} / no formula: covered pixels (helper invocations may count), minus what early tests reject"
                );
                // #endregion
            }
        }

        for s in [&mut self.current, &mut self.total] {
            s.frames += 1;
            for (sum, v) in s.pass_us.iter_mut().zip(pass_us) {
                *sum += v;
            }
            if let Some((l, dev)) = latency {
                let (sum, min, max, n) = &mut s.latency_us;
                *min = if *n == 0 { l } else { min.min(l) };
                *max = if *n == 0 { l } else { max.max(l) };
                *sum += l;
                *n += 1;
                s.max_deviation_us += dev;
            }
            s.fragment_invocations += fragment_invocations;
        }
        Ok(())
    }
    // #endregion

    /// Renders and presents one frame (step 22's loop) with timestamps and a
    /// statistics query. Returns `Ok(false)` when the swapchain is out of date.
    fn draw_frame(&mut self) -> Result<bool, Box<dyn Error>> {
        let slot = self.frame_index % self.frames.len();
        let in_flight = self
            .frames
            .get(slot)
            .ok_or("frame slot out of range")?
            .in_flight;
        // SAFETY: the fence belongs to this device.
        unsafe { self.device.wait_for_fences(&[in_flight], true, u64::MAX)? };
        // The slot's previous frame has completed: its queries are ready.
        if let Some((frame, calibration)) = self
            .frames
            .get_mut(slot)
            .ok_or("frame slot out of range")?
            .pending
            .take()
        {
            self.read_results(slot, frame, calibration)?;
        }
        let second = self.start.elapsed().as_secs();
        if second > self.second {
            self.current.print(&format!("{}s", self.second + 1));
            self.current = Sums::default();
            self.second = second;
        }

        let device = &self.device;
        let frame = self.frames.get(slot).ok_or("frame slot out of range")?;
        let cmd = frame.command_buffer;
        let anim_start = *self.anim_start.get_or_insert_with(Instant::now);
        let time = anim_start.elapsed().as_secs_f32();
        // SAFETY: `mapped` points to this slot's mapped allocation of at least
        // size_of::<Scene>() bytes; no pending GPU work reads it (fence above).
        unsafe { frame.mapped.write(scene(time, &self.proj)) };

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
        let fragment_tests = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        let depth_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(fragment_tests)
            .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_stage_mask(fragment_tests)
            .dst_access_mask(
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(self.depth_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.depth_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(self.depth_aspect)
                    .level_count(1)
                    .layer_count(1),
            );
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
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.depth_view)
            .image_layout(self.depth_layout)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            });
        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment))
            .depth_attachment(&depth_attachment);
        if self.has_stencil {
            rendering_info = rendering_info.stencil_attachment(&depth_attachment);
        }
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
        let index_count = u32::try_from(INDICES.len())?;
        let first = u32::try_from(slot)? * TIMESTAMPS;
        let slot_u32 = u32::try_from(slot)?;
        let pool = self.timestamp_pool;

        // #region record
        // Timestamp i is written once all earlier commands have completed the
        // given stage ("Implementations may write the timestamp at any stage
        // that is logically later than stage"). ALL_COMMANDS therefore marks
        // the end of everything before it, so consecutive timestamps enclose
        // the commands between them; TOP_OF_PIPE marks the start. The
        // statistics query counts what the draw between Begin and End does.
        // SAFETY: `cmd` is recording; all handles are alive and stay alive
        // while the command buffer is pending; the queries were reset.
        unsafe {
            use vk::PipelineStageFlags2 as S;
            device.cmd_write_timestamp2(cmd, S::TOP_OF_PIPE, pool, first);
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(&[to_attachment, depth_barrier]),
            );
            device.cmd_write_timestamp2(cmd, S::ALL_COMMANDS, pool, first + 1);
            if let Some(stats) = self.stats_pool {
                device.cmd_begin_query(cmd, stats, slot_u32, vk::QueryControlFlags::empty());
            }
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
            device.cmd_draw_indexed(cmd, index_count, INSTANCES, 0, 0, 0);
            device.cmd_end_rendering(cmd);
            if let Some(stats) = self.stats_pool {
                device.cmd_end_query(cmd, stats, slot_u32);
            }
            device.cmd_write_timestamp2(cmd, S::ALL_COMMANDS, pool, first + 2);
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&to_present)),
            );
            device.cmd_write_timestamp2(cmd, S::ALL_COMMANDS, pool, first + 3);
            device.end_command_buffer(cmd)?;
        }
        // #endregion

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

        // #region calibrate
        // Device clock and host clock, sampled together right before the
        // submit. maxDeviation bounds how far apart the two samples may lie.
        let calibration = match &self.calibration {
            Some((loader, host)) => {
                let infos = [
                    vk::CalibratedTimestampInfoKHR::default()
                        .time_domain(vk::TimeDomainKHR::DEVICE),
                    vk::CalibratedTimestampInfoKHR::default().time_domain(*host),
                ];
                // SAFETY: both domains were reported calibrateable for this device.
                let (values, max_deviation_ns) =
                    unsafe { loader.get_calibrated_timestamps(&infos)? };
                match values.as_slice() {
                    &[device, host] => Some(Calibration {
                        device,
                        host,
                        max_deviation_ns,
                    }),
                    _ => return Err("vkGetCalibratedTimestampsKHR returned a wrong count".into()),
                }
            }
            None => None,
        };
        // #endregion
        // SAFETY: all handles belong to this device; the fence is unsignaled.
        unsafe { device.queue_submit2(self.queue, &[submit], frame.in_flight)? };
        if let Some(c) = calibration {
            self.first_calibration.get_or_insert(c);
            self.last_calibration = Some(c);
        }
        let frame_index = self.frame_index;
        self.frames
            .get_mut(slot)
            .ok_or("frame slot out of range")?
            .pending = Some((frame_index, calibration));

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

    /// Waits for the device, reads the queries still pending, prints the
    /// totals.
    fn finish(&mut self) -> Result<(), Box<dyn Error>> {
        // SAFETY: plain call on a live device.
        unsafe { self.device.device_wait_idle()? };
        for slot in 0..self.frames.len() {
            if let Some((frame, calibration)) = self
                .frames
                .get_mut(slot)
                .ok_or("frame slot out of range")?
                .pending
                .take()
            {
                self.read_results(slot, frame, calibration)?;
            }
        }
        self.total.print("whole run");
        // #region host-clock
        // The host clock's rate, measured: QPC ticks per device second between
        // the first and the last calibration (QueryPerformanceFrequency would
        // report it directly; this shows the two clocks agree).
        if let (Some(a), Some(b), Some((_, host))) = (
            self.first_calibration,
            self.last_calibration,
            &self.calibration,
        ) {
            let device_s =
                ticks_between(a.device, b.device, self.valid_bits) as f64 * self.period_ns / 1e9;
            if device_s > 0.0 {
                println!(
                    "[timing] {host:?}: {:.0} ticks per second of device time (over {device_s:.2} s)",
                    b.host.wrapping_sub(a.host) as f64 / device_s
                );
            }
        }
        // #endregion
        Ok(())
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
            if let Some(pool) = self.stats_pool {
                self.device.destroy_query_pool(pool, None);
            }
            self.device.destroy_query_pool(self.timestamp_pool, None);
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
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_image_view(self.depth_view, None);
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_memory, None);
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

/// winit drives the program (step 06).
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
    if let Some(mut vk) = app.vk.take() {
        println!("Window closed");
        let elapsed = vk.start.elapsed().as_secs_f64();
        println!(
            "[timing] {} frames in {elapsed:.1} s, {:.0} fps",
            vk.frame_index,
            vk.frame_index as f64 / elapsed
        );
        let finished = vk.finish();
        vk.destroy()?;
        finished?;
    }
    app.error.map_or(Ok(()), Err)
}
