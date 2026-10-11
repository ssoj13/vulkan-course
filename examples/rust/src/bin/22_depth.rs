//! Step 22 - Depth buffer.
//!
//! Two cubes cut through each other. Each cube alone looks right with back
//! faces culled (it is convex), but where they intersect, which surface is in
//! front changes from pixel to pixel, and no draw order can get that right.
//! A DEPTH ATTACHMENT can: an image with one depth value per pixel; every
//! fragment compares its depth with the stored one (the depth TEST) and, if
//! it is nearer, replaces it (the depth WRITE) and its color is kept.
//!
//!   pick a depth format (format properties) -> depth image + view
//!   frame: barrier (UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL, after the previous
//!     frame's depth writes) -> rendering with pDepthAttachment (clear to 1.0)
//!   pipeline: depthTestEnable, depthWriteEnable, compareOp LESS
//!   vertex shader: perspective projection into Vulkan's clip space
//!
//! `--no-depth` turns the test and the write off (the attachment stays) to
//! show what goes wrong; `--seconds S` closes the window after S seconds.
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

/// Position + color: 3 + 3 floats = 24 bytes (shaders/cube.slang).
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    /// Linear RGB, one color per face.
    color: [f32; 3],
}
const _: () = assert!(size_of::<Vertex>() == 24);

// #region cube
/// A unit cube (corners at +-0.5), 4 vertices per face so each face has its
/// own color. Model space is right-handed with +y up. Every face's corners go
/// counter-clockwise seen from OUTSIDE the cube; the projection below keeps
/// that orientation on screen, so the pipeline's front face is
/// COUNTER_CLOCKWISE and back-face culling drops the far side.
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
// #endregion

/// A 4x4 matrix, column-major: `m[c]` is column c (step 17).
type Mat4 = [[f32; 4]; 4];

/// The host side of `Scene` in shaders/cube.slang: one matrix per cube
/// (std140 array stride 64 = the size of a Mat4, so no padding).
#[repr(C)]
#[derive(Clone, Copy)]
struct Scene {
    mvp: [Mat4; 2],
}
const _: () = assert!(size_of::<Scene>() == 128);

// #region math
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

/// Rotation by `a` radians about +x (right-handed: +y turns toward +z).
fn rotate_x(a: f32) -> Mat4 {
    let (s, c) = a.sin_cos();
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, s, 0.0],
        [0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Rotation by `a` radians about +y (right-handed: +z turns toward +x).
fn rotate_y(a: f32) -> Mat4 {
    let (s, c) = a.sin_cos();
    [
        [c, 0.0, -s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
// #endregion

// #region projection
/// Perspective projection from a right-handed view space (camera at the
/// origin looking down -z, +y up) to VULKAN's clip space, which differs from
/// OpenGL's in two ways: +y points DOWN the screen, and depth runs 0..1
/// (OpenGL: -1..1). With f = 1 / tan(fovy / 2):
///
///   x_clip = (f / aspect) * x      the field of view, widened by aspect
///   y_clip = -f * y                minus: view +y up -> screen +y down
///   z_clip = A * z + B             A = far / (near - far),
///                                  B = near * far / (near - far)
///   w_clip = -z                    the distance in front of the camera
///
/// After the divide by w: z = -near gives depth 0, z = -far gives depth 1.
/// Depth is then proportional to 1/distance, so most of its precision sits
/// near the camera. REVERSED-Z (near -> 1, far -> 0, compare GREATER, clear
/// to 0) pairs that with the float format's own precision near 0 and is the
/// usual choice for large scenes with D32_SFLOAT; this step keeps the
/// classic mapping.
fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fovy / 2.0).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ]
}
// #endregion

/// Camera: 60 degrees vertical field of view, depth range 0.1..100.
const FOVY: f32 = std::f32::consts::FRAC_PI_3;
const NEAR: f32 = 0.1;
const FAR: f32 = 100.0;

// #region scene
/// Both cubes' model-view-projection matrices at `time` seconds. The camera
/// sits 3.5 units in front of the origin (view = translate by -3.5 in z);
/// cube 0 tumbles at the origin, cube 1 is shifted by less than a cube's size
/// and turns the other way, so the two always intersect.
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
// #endregion

// #region cli
/// Command line: `--no-depth`, `--seconds S`.
struct Args {
    depth_test: bool,
    seconds: Option<f64>,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut args = Self {
            depth_test: true,
            seconds: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--no-depth" => args.depth_test = false,
                "--seconds" => {
                    let value = it.next().ok_or("--seconds needs a value")?;
                    args.seconds = Some(value.parse()?);
                }
                _ => {
                    return Err(
                        format!("unknown argument {arg} (use --no-depth, --seconds S)").into(),
                    );
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

// #region depth-format
/// Depth formats in order of preference. The spec guarantees depth-attachment
/// support for D16_UNORM, for at least one of X8_D24_UNORM_PACK32 and
/// D32_SFLOAT, and for at least one of D24_UNORM_S8_UINT and
/// D32_SFLOAT_S8_UINT; everything else is queried. D32_SFLOAT first: a float
/// depth buffer, 32 bits, no stencil.
const DEPTH_CANDIDATES: [vk::Format; 3] = [
    vk::Format::D32_SFLOAT,
    vk::Format::D24_UNORM_S8_UINT,
    vk::Format::D16_UNORM,
];

/// The first candidate whose OPTIMAL tiling supports DEPTH_STENCIL_ATTACHMENT,
/// printing the answer for each.
fn pick_depth_format(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
) -> Result<vk::Format, Box<dyn Error>> {
    println!("Depth format candidates (optimal tiling, DEPTH_STENCIL_ATTACHMENT):");
    let mut pick = None;
    for format in DEPTH_CANDIDATES {
        let mut props = vk::FormatProperties2::default();
        // SAFETY: `pd` was enumerated from this instance; `props` outlives the call.
        unsafe { instance.get_physical_device_format_properties2(pd, format, &mut props) };
        let ok = props
            .format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT);
        println!("  {format:?}: {}", if ok { "yes" } else { "no" });
        if ok && pick.is_none() {
            pick = Some(format);
        }
    }
    pick.ok_or_else(|| "no depth format supports DEPTH_STENCIL_ATTACHMENT".into())
}

/// Whether `format` also has a stencil component. Such a format's layout
/// transitions must name both aspects (VUID-VkImageMemoryBarrier2-image-03320
/// without separateDepthStencilLayouts), and this step then attaches it as
/// depth AND stencil in the DEPTH_STENCIL_ATTACHMENT_OPTIMAL layout.
fn has_stencil(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
    )
}
// #endregion

// #region frame-resources
/// What the CPU touches while recording one frame (step 17).
struct Frame {
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
    uniform: GpuBuffer,
    mapped: *mut Scene,
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
    depth_format: vk::Format,
    depth_layout: vk::ImageLayout,
    depth_aspect: vk::ImageAspectFlags,
    depth_image: vk::Image,
    depth_memory: vk::DeviceMemory,
    depth_view: vk::ImageView,
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
    /// Steps 01-17, plus the depth image and the pipeline's depth state.
    fn new(event_loop: &ActiveEventLoop, args: &Args) -> Result<Self, Box<dyn Error>> {
        // Window from step 06: 800x600 physical pixels, fixed size.
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("Vulkan from Zero - 22 depth")
                .with_inner_size(PhysicalSize::new(800, 600))
                .with_resizable(false),
        )?;
        let size = window.inner_size();
        println!("Window: {}x{} pixels", size.width, size.height);

        // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vulkan from Zero - 22 depth")
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
        // Depth testing is Vulkan 1.0: nothing to enable.
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

        // Step 07's device: one queue, the two 1.3 features, VK_KHR_swapchain.
        let priorities = [1.0_f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
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
            "VkDevice created with {}",
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

        println!();
        let depth_format = pick_depth_format(&instance, physical_device)?;
        let (depth_layout, depth_aspect) = if has_stencil(depth_format) {
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
        println!("Depth format: {depth_format:?}, attachment layout {depth_layout:?}");

        // #region depth-image
        // One depth image, the size of the swapchain images, shared by both
        // frames in flight (see the barrier in `draw_frame` for what that
        // costs). Usage DEPTH_STENCIL_ATTACHMENT only: nobody samples or
        // copies it. Memory: DEVICE_LOCAL, like any render target.
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
        // The view sees the depth aspect (and the stencil aspect, if any).
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
        println!(
            "Depth image: {}x{}, DEPTH_STENCIL_ATTACHMENT, {} bytes in memory type {depth_type} (DEVICE_LOCAL), one for both frames in flight",
            extent.width, extent.height, req.size
        );
        // #endregion

        // #region projection-matrix
        let aspect = extent.width as f32 / extent.height as f32;
        let proj = perspective(FOVY, aspect, NEAR, FAR);
        println!(
            "\nProjection: fovy 60 degrees, aspect {aspect}, near {NEAR}, far {FAR} (rows; Vulkan clip space: y down, depth 0..1)"
        );
        for row in (0..4).map(|r| proj.map(|col| col[r])) {
            println!(
                "  {:9.4} {:9.4} {:9.4} {:9.4}",
                row[0], row[1], row[2], row[3]
            );
        }
        // Where the near and far planes land after the divide by w.
        let depth_at = |d: f32| (proj[2][2] * -d + proj[3][2]) / d;
        println!(
            "Depth after the divide: {:.4} at distance {NEAR}, {:.4} at 1, {:.4} at 3.5, {:.4} at {FAR}",
            depth_at(NEAR),
            depth_at(1.0),
            depth_at(3.5),
            depth_at(FAR)
        );
        // #endregion

        // The SPIR-V is compiled from shaders/cube.slang by `bun run
        // sync:shaders` and embedded at build time (step 08 explains `read_spv`).
        const CUBE_SPV: &[u8] = include_bytes!("../../../../shaders/spv/cube.spv");
        let spirv = ash::util::read_spv(&mut std::io::Cursor::new(CUBE_SPV))?;
        println!(
            "\nSPIR-V: {} bytes (cube.spv), entry points vertexMain, fragmentMain",
            CUBE_SPV.len()
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

        // Vertex input: binding 0, stride 24, position + color (3 floats each).
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

        // Step 17's set layout: binding 0 = one uniform buffer, vertex stage.
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
        // Counter-clockwise front faces: see the cube's comment.
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // #region depth-state
        // The depth test: a fragment survives if its depth is LESS than the
        // stored one (cleared to 1.0 = the far plane); a surviving fragment
        // writes its depth. Bounds test and stencil stay off. With --no-depth
        // both are off: the attachment is still cleared, but whatever is drawn
        // last covers what was drawn before.
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(args.depth_test)
            .depth_write_enable(args.depth_test)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);
        // Dynamic rendering: the pipeline declares the attachment formats it
        // will render to (a stencil format only if the image has stencil).
        let stencil_format = if has_stencil(depth_format) {
            depth_format
        } else {
            vk::Format::UNDEFINED
        };
        let color_formats = [format.format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format)
            .stencil_attachment_format(stencil_format);
        if args.depth_test {
            println!("Depth state: test on, write on, compare LESS (clear value 1.0)");
        } else {
            println!("Depth state: test OFF, write OFF (--no-depth): the last cube drawn wins");
        }
        // #endregion

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
        println!(
            "Graphics pipeline created (color {:?} + depth {depth_format:?})",
            format.format
        );

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
            "Mesh: cube, {} vertices, {} indices, drawn as 2 instances",
            VERTICES.len(),
            INDICES.len()
        );

        // Step 17's per-frame uniform buffers, persistently mapped, and one
        // descriptor set per frame in flight.
        let uniform_size = vk::DeviceSize::try_from(size_of::<Scene>())?;
        let frames_u32 = u32::try_from(FRAMES_IN_FLIGHT)?;
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(frames_u32)];
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
            let info = vk::DescriptorBufferInfo::default()
                .buffer(buffer.buffer)
                .offset(0)
                .range(uniform_size);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&info));
            // SAFETY: the set and the buffer outlive the call; no command
            // buffer uses the set yet.
            unsafe { device.update_descriptor_sets(&[write], &[]) };
            uniforms.push((buffer, ptr.cast::<Scene>()));
        }
        println!(
            "Uniform buffers: {FRAMES_IN_FLIGHT} x {uniform_size} bytes (2 matrices), one descriptor set each"
        );

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
        println!("\n{FRAMES_IN_FLIGHT} frames in flight, one shared depth image");

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
            depth_format,
            depth_layout,
            depth_aspect,
            depth_image,
            depth_memory,
            depth_view,
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

        // #region depth-barrier
        // The depth image is SHARED by the frames in flight, so this frame's
        // first depth access (the clear) is a write after the previous frame's
        // depth writes (its tests and its DONT_CARE store): a WRITE-AFTER-WRITE
        // hazard. Nothing else orders them: the fence we waited on belongs to
        // the frame two submissions back, and the GPU may still be running
        // the previous one. So the barrier's first scope is the previous
        // frame's depth writes (EARLY and LATE_FRAGMENT_TESTS, where depth
        // attachments are accessed), the second scope this frame's depth
        // reads and writes in the same stages. oldLayout UNDEFINED: the old
        // contents are not needed (the clear overwrites them), and the
        // transition itself is ordered by the same scopes. Cost: frame N+1's
        // fragment tests wait for frame N's; one depth image per frame in
        // flight would remove that wait at the price of more memory.
        let depth_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
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
        // #endregion

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
        // #region depth-attachment
        // The depth attachment of this rendering: cleared to 1.0 (the far
        // plane) at the start, thrown away at the end (DONT_CARE: only this
        // frame's color image is presented, nobody reads the depth later,
        // and a tiled GPU then never writes it out to memory at all).
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
        // A depth/stencil format is attached as both (same view, same layout).
        if has_stencil(self.depth_format) {
            rendering_info = rendering_info.stencil_attachment(&depth_attachment);
        }
        // #endregion
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
        // #region draw
        // Both barriers in one vkCmdPipelineBarrier2, then one draw of two
        // instances: instance i reads mvp[i].
        // SAFETY: `cmd` is recording; all handles are alive and stay alive
        // while the command buffer is pending.
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(&[to_attachment, depth_barrier]),
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
            device.cmd_draw_indexed(cmd, index_count, 2, 0, 0, 0);
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
