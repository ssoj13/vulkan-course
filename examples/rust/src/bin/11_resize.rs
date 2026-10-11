//! Step 11 - Swapchain recreation.
//!
//! A swapchain is made for one window size. When the window changes, the
//! presentation engine says so: vkAcquireNextImageKHR or vkQueuePresentKHR
//! return VK_ERROR_OUT_OF_DATE_KHR (the swapchain can no longer be used) or
//! VK_SUBOPTIMAL_KHR (it still works, but no longer matches exactly), and the
//! window system reports the new size. Then we build a new swapchain for the
//! new size, handing the old one over (`old_swapchain`), and destroy the old
//! one once nothing uses it any more - like re-rendering a plate at a new
//! resolution while the old version is still on screen.
//!
//! Knowing when "nothing uses it" is the hard part: the presentation engine
//! gives no signal by default. Two ways are shown:
//! - present fences (VK_KHR_swapchain_maintenance1): every present signals a
//!   fence; when the fences of all presents to the old swapchain have signaled,
//!   it is destroyed. The rendering never stops.
//! - `--no-present-fences` (and on GPUs without the extension):
//!   vkDeviceWaitIdle before every recreation, then destroy the old one at once.
//!
//! A minimized window has a 0x0 surface, and no swapchain can have that size:
//! rendering pauses until the window comes back. `--demo-resize` resizes,
//! minimizes and restores the window by itself; `--seconds S` closes it after
//! S seconds. Lines that depend on timing start with `[timing]`. Steps 01-10
//! are repeated, reduced to what this step needs.

use std::error::Error;
use std::ffi::{CStr, c_void};
use std::time::{Duration, Instant};

use ash::{
    Entry,
    ext::debug_utils,
    khr::{
        get_surface_capabilities2, surface, surface_maintenance1, swapchain, swapchain_maintenance1,
    },
    vk::{self, TaggedStructure as _},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
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

// #region cli
/// Command line:
/// - `--no-present-fences`: recreate with vkDeviceWaitIdle even when present
///   fences are available;
/// - `--demo-resize`: resize, minimize and restore the window by script;
/// - `--seconds S`: close the window after S seconds (default: run until closed).
#[derive(Clone, Copy)]
struct Args {
    present_fences: bool,
    demo: bool,
    seconds: Option<f64>,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut args = Self {
            present_fences: true,
            demo: false,
            seconds: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--no-present-fences" => args.present_fences = false,
                "--demo-resize" => args.demo = true,
                "--seconds" => {
                    let value = it.next().ok_or("--seconds needs a value")?;
                    args.seconds = Some(value.parse()?);
                }
                _ => {
                    return Err(format!(
                        "unknown argument {arg} (use --no-present-fences, --demo-resize, --seconds S)"
                    )
                    .into());
                }
            }
        }
        Ok(args)
    }
}
// #endregion

/// Frames in flight, as in step 10.
const FRAMES: usize = 2;

/// How long to wait for a present fence before falling back to
/// vkDeviceWaitIdle (see `wait-present-fence`).
const PRESENT_FENCE_TIMEOUT: Duration = Duration::from_secs(1);

// #region demo
/// One scripted window change of `--demo-resize`: at `at` seconds after start.
#[derive(Clone, Copy)]
enum DemoAction {
    Resize(u32, u32),
    Minimize,
    Restore,
}

const DEMO: [(f64, DemoAction); 4] = [
    (1.0, DemoAction::Resize(1000, 700)),
    (2.0, DemoAction::Resize(640, 480)),
    (3.0, DemoAction::Minimize),
    (4.0, DemoAction::Restore),
];
// #endregion

// #region chain
/// Everything that depends on the swapchain and must be rebuilt with it: the
/// swapchain, its images and their views, and one render_finished semaphore
/// per image (the present waits on it, so it lives as long as the presents).
struct Chain {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    render_finished: Vec<vk::Semaphore>,
    extent: vk::Extent2D,
}

/// An old chain waiting to be destroyed: replaced at frame `retired_at`, so
/// every present that used it belongs to an earlier frame.
struct Retired {
    chain: Chain,
    retired_at: u64,
}
// #endregion

// #region frame-resources
/// Step 10's per-frame objects, plus the fence the present of this slot signals.
struct Frame {
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
    /// Signaled when the presentation engine no longer needs what this
    /// slot's last present referred to. Null on the vkDeviceWaitIdle path.
    present_fence: vk::Fence,
    /// Whether a present was given `present_fence` since the last wait on it.
    present_pending: bool,
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
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    swapchain_loader: swapchain::Device,
    format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    chain: Chain,
    /// Old chains whose presents may still be in progress (present-fence path).
    retired: Vec<Retired>,
    /// Present fences that timed out: never reused, destroyed at exit.
    abandoned: Vec<vk::Fence>,
    /// True when present fences are used, false on the vkDeviceWaitIdle path.
    present_fences: bool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    frames: Vec<Frame>,
    /// Frames rendered so far; `frame_index % FRAMES` is the current slot.
    frame_index: u64,
    /// Set by a window resize or a SUBOPTIMAL result: recreate before the next frame.
    resized: bool,
    /// The surface is 0x0 (minimized): no frames until it has a size again.
    paused: bool,
    recreations: u32,
    start: Instant,
    window: Window,
}

impl Vk {
    /// Steps 01-10 (shortened), with the extensions for present fences.
    fn new(event_loop: &ActiveEventLoop, args: Args) -> Result<Self, Box<dyn Error>> {
        // Step 06's window, now resizable.
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("Vulkan from Zero - 11 resize")
                .with_inner_size(PhysicalSize::new(800, 600))
                .with_resizable(true),
        )?;
        let size = window.inner_size();
        println!("Window: {}x{} pixels, resizable", size.width, size.height);

        // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
        let entry = unsafe { Entry::load()? };
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vulkan from Zero - 11 resize")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"No Engine")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_4);

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

        // #region instance-extensions
        // VK_KHR_swapchain_maintenance1 (device) depends on two instance
        // extensions: VK_KHR_surface_maintenance1, which needs
        // VK_KHR_get_surface_capabilities2. Enable them when the loader has
        // them; without them the device extension is off and step 11 falls
        // back to vkDeviceWaitIdle.
        let display = event_loop.display_handle()?.as_raw();
        let mut ext_ptrs = ash_window::enumerate_required_extensions(display)?.to_vec();
        ext_ptrs.push(debug_utils::NAME.as_ptr());
        // SAFETY: plain query, no layer name.
        let instance_exts = unsafe { entry.enumerate_instance_extension_properties(None)? };
        let has_instance_ext = |name: &CStr| {
            instance_exts
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };
        let surface_maintenance = has_instance_ext(get_surface_capabilities2::NAME)
            && has_instance_ext(surface_maintenance1::NAME);
        if surface_maintenance {
            ext_ptrs.push(get_surface_capabilities2::NAME.as_ptr());
            ext_ptrs.push(surface_maintenance1::NAME.as_ptr());
        }
        // #endregion

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
            "Picked: {}, graphics + present family {family}",
            chosen.device_name_as_c_str()?.to_string_lossy()
        );

        // #region device-extensions
        // Present fences need VK_KHR_swapchain_maintenance1 AND its feature
        // swapchainMaintenance1 enabled on the device. Used only when the
        // instance extensions were enabled, the GPU has both, and
        // --no-present-fences was not given.
        // SAFETY: `physical_device` was enumerated from this instance.
        let exts = unsafe { instance.enumerate_device_extension_properties(physical_device)? };
        let mut maintenance_features =
            vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR::default();
        let has_maintenance = surface_maintenance
            && exts
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(swapchain_maintenance1::NAME))
            && {
                let mut f2 = vk::PhysicalDeviceFeatures2::default().push(&mut maintenance_features);
                // SAFETY: as above; the chain struct outlives the call.
                unsafe { instance.get_physical_device_features2(physical_device, &mut f2) };
                maintenance_features.swapchain_maintenance1 != 0
            };
        let present_fences = has_maintenance && args.present_fences;
        let priorities = [1.0_f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let mut maintenance = vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR::default()
            .swapchain_maintenance1(true);
        let mut device_exts = vec![swapchain::NAME.as_ptr()];
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .push(&mut features13);
        if present_fences {
            device_exts.push(swapchain_maintenance1::NAME.as_ptr());
            device_info = device_info.push(&mut maintenance);
        }
        let device_info = device_info.enabled_extension_names(&device_exts);
        // SAFETY: `physical_device` belongs to `instance`, and all create-info data outlives the call.
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        let swapchain_loader = swapchain::Device::load(&instance, &device);
        println!(
            "Present fences (VK_KHR_swapchain_maintenance1): {}",
            if !has_maintenance {
                "not supported"
            } else if present_fences {
                "supported, used"
            } else {
                "supported, off (--no-present-fences)"
            }
        );
        // #endregion

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
        println!(
            "Surface format {:?} + {:?}, present mode {present_mode:?}",
            format.format, format.color_space
        );

        // Step 08's pipeline. Viewport and scissor are dynamic state, so it
        // works for every swapchain size: recreation needs no new pipeline.
        const TRIANGLE_SPV: &[u8] = include_bytes!("../../../../shaders/spv/triangle.spv");
        let spirv = ash::util::read_spv(&mut std::io::Cursor::new(TRIANGLE_SPV))?;
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
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
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

        // Step 10's queue, pool and per-frame objects, plus a present fence per slot.
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
            .command_buffer_count(u32::try_from(FRAMES)?);
        // SAFETY: `command_pool` belongs to this device.
        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };
        let mut frames = Vec::with_capacity(FRAMES);
        for &command_buffer in &command_buffers {
            // SAFETY: plain create calls on a live device.
            let (image_available, in_flight, present_fence) = unsafe {
                (
                    device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?,
                    device.create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )?,
                    // Unsignaled: a present fence must be unsignaled when it is
                    // handed to vkQueuePresentKHR (VUID-VkSwapchainPresentFenceInfoKHR-pFences-07758).
                    if present_fences {
                        device.create_fence(&vk::FenceCreateInfo::default(), None)?
                    } else {
                        vk::Fence::null()
                    },
                )
            };
            frames.push(Frame {
                command_buffer,
                image_available,
                in_flight,
                present_fence,
                present_pending: false,
            });
        }

        let mut vk = Self {
            _entry: entry,
            instance,
            debug_utils,
            messenger,
            surface_loader,
            surface,
            physical_device,
            device,
            swapchain_loader,
            format,
            present_mode,
            chain: Chain {
                swapchain: vk::SwapchainKHR::null(),
                images: Vec::new(),
                views: Vec::new(),
                render_finished: Vec::new(),
                extent: vk::Extent2D::default(),
            },
            retired: Vec::new(),
            abandoned: Vec::new(),
            present_fences,
            pipeline_layout,
            pipeline,
            queue,
            command_pool,
            frames,
            frame_index: 0,
            resized: false,
            paused: false,
            recreations: 0,
            start: Instant::now(),
            window,
        };
        // The first swapchain is made by the same function as every later one.
        if !vk.recreate()? {
            return Err("The window has no size at start".into());
        }
        println!("\nRendering. Resize or minimize the window; close it to exit.");
        Ok(vk)
    }

    // #region create-chain
    /// Creates a swapchain for the surface's current size, handing over the
    /// old one, plus the views and semaphores of its images.
    fn create_chain(&self, old: vk::SwapchainKHR) -> Result<Option<Chain>, Box<dyn Error>> {
        // Ask the surface again every time: its size changes with the window.
        // SAFETY: `physical_device` and `surface` belong to this instance.
        let caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)?
        };
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            let size = self.window.inner_size();
            vk::Extent2D {
                width: size
                    .width
                    .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: size
                    .height
                    .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        // Minimized: the surface is 0x0, and a swapchain cannot be that small
        // (VUID-VkSwapchainCreateInfoKHR-imageExtent-01689: width and height
        // must be non-zero). Nothing to create until the window has a size.
        if extent.width == 0 || extent.height == 0 {
            return Ok(None);
        }
        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            image_count = image_count.min(caps.max_image_count);
        }
        // old_swapchain: the presentation engine may reuse the old one's
        // resources and switch over without a gap. The old one is "retired":
        // no more acquires from it, but images already acquired from it may
        // still be presented.
        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(self.format.format)
            .image_color_space(self.format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.present_mode)
            .clipped(true)
            .old_swapchain(old);
        // SAFETY: `swapchain_info` refers to a live surface; `old` is null or
        // the current swapchain of this surface.
        let swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(&swapchain_info, None)?
        };
        // SAFETY: `swapchain` is alive.
        let images = unsafe { self.swapchain_loader.get_swapchain_images(swapchain)? };
        let mut views = Vec::with_capacity(images.len());
        let mut render_finished = Vec::with_capacity(images.len());
        for &image in &images {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(self.format.format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            // SAFETY: `image` belongs to the new swapchain.
            views.push(unsafe { self.device.create_image_view(&view_info, None)? });
            // SAFETY: plain create call on a live device.
            render_finished.push(unsafe {
                self.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
            });
        }
        Ok(Some(Chain {
            swapchain,
            images,
            views,
            render_finished,
            extent,
        }))
    }
    // #endregion

    // #region recreate
    /// Replaces the current chain with one for the current surface size.
    /// Returns `Ok(false)` while the window is minimized (nothing created).
    fn recreate(&mut self) -> Result<bool, Box<dyn Error>> {
        // Without present fences we cannot tell when the presentation engine
        // is done with the old swapchain: wait until the GPU is idle (all
        // submitted work and the presents' semaphore waits completed), then
        // destroy it right away. With present fences the old chain is kept
        // in `retired` and the frames go on.
        if !self.present_fences {
            // SAFETY: plain call on a live device.
            unsafe { self.device.device_wait_idle()? };
        }
        let old = self.chain.swapchain;
        // A failed attempt is not retried until the window reports a new size.
        self.resized = false;
        let Some(chain) = self.create_chain(old)? else {
            if !self.paused {
                println!("Window minimized (surface 0x0): rendering paused");
            }
            self.paused = true;
            return Ok(false);
        };
        if self.paused {
            println!("Window restored");
        }
        self.paused = false;
        let old_chain = std::mem::replace(&mut self.chain, chain);
        let first = old == vk::SwapchainKHR::null();
        if !first {
            self.recreations += 1;
            if self.present_fences {
                self.retired.push(Retired {
                    chain: old_chain,
                    retired_at: self.frame_index,
                });
            } else {
                self.destroy_chain(&old_chain);
            }
        }
        println!(
            "Swapchain {}: {}x{}, {} images{}",
            if first { "created" } else { "recreated" },
            self.chain.extent.width,
            self.chain.extent.height,
            self.chain.images.len(),
            match (first, self.present_fences) {
                (true, _) => "",
                (false, true) => "; the old one waits for its present fences",
                (false, false) => "; the old one destroyed after vkDeviceWaitIdle",
            }
        );
        Ok(true)
    }
    // #endregion

    /// Destroys a chain nothing uses any more.
    fn destroy_chain(&self, chain: &Chain) {
        // SAFETY: the caller guarantees that no pending work or present uses
        // the chain (device idle, or all its present fences signaled).
        unsafe {
            for &semaphore in &chain.render_finished {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &view in &chain.views {
                self.device.destroy_image_view(view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(chain.swapchain, None);
        }
    }

    // #region wait-present-fence
    /// Waits for the present fence of `slot` (from the present FRAMES frames
    /// ago) and resets it for this frame's present. The spec does not say
    /// whether the fence of a present that returned VK_ERROR_OUT_OF_DATE_KHR
    /// is signaled, so the wait has a timeout: if it expires, wait for the
    /// whole device instead and retire that fence for good (it is never handed
    /// to a present again, and destroyed only at exit).
    fn wait_present_fence(&mut self, slot: usize) -> Result<(), Box<dyn Error>> {
        let device = &self.device;
        let frame = self.frames.get_mut(slot).ok_or("frame slot out of range")?;
        if !frame.present_pending {
            return Ok(());
        }
        let timeout = u64::try_from(PRESENT_FENCE_TIMEOUT.as_nanos())?;
        // SAFETY: the fence belongs to this device.
        match unsafe { device.wait_for_fences(&[frame.present_fence], true, timeout) } {
            Ok(()) => {
                // SAFETY: the fence is signaled, no present uses it any more.
                unsafe { device.reset_fences(&[frame.present_fence])? };
            }
            Err(vk::Result::TIMEOUT) => {
                println!(
                    "Present fence of slot {slot} not signaled after {} s: falling back to vkDeviceWaitIdle",
                    PRESENT_FENCE_TIMEOUT.as_secs()
                );
                // SAFETY: plain call on a live device.
                unsafe { device.device_wait_idle()? };
                self.abandoned.push(frame.present_fence);
                // SAFETY: plain create call on a live device.
                frame.present_fence =
                    unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
            }
            Err(e) => return Err(e.into()),
        }
        frame.present_pending = false;
        Ok(())
    }
    // #endregion

    // #region destroy-retired
    /// Destroys the retired chains whose presents have all completed. The
    /// presents to a chain retired at frame R belong to frames R-FRAMES..R-1
    /// at the latest (one present per frame, slot = frame % FRAMES), and the
    /// present fence of frame F is waited on at the start of frame F+FRAMES.
    /// So once frame R+FRAMES-1 has waited on its slot's fence, every present
    /// to the old chain has signaled its fence.
    fn destroy_retired(&mut self) {
        let frame = self.frame_index;
        let (done, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.retired)
            .into_iter()
            .partition(|r| frame + 1 >= r.retired_at + FRAMES as u64);
        self.retired = keep;
        for r in &done {
            self.destroy_chain(&r.chain);
            println!(
                "Old {}x{} swapchain destroyed: the fences of all its presents have signaled",
                r.chain.extent.width, r.chain.extent.height
            );
        }
    }
    // #endregion

    /// Renders and presents one frame; recreates the swapchain when needed.
    fn draw_frame(&mut self) -> Result<(), Box<dyn Error>> {
        if self.paused && !self.resized {
            return Ok(());
        }
        let slot = usize::try_from(self.frame_index % FRAMES as u64)?;
        let (cmd, image_available, in_flight) = {
            let frame = self.frames.get(slot).ok_or("frame slot out of range")?;
            (frame.command_buffer, frame.image_available, frame.in_flight)
        };

        // Step 10's fence wait, then the present fence of the same slot.
        // SAFETY: the fence belongs to this device.
        unsafe { self.device.wait_for_fences(&[in_flight], true, u64::MAX)? };
        if self.present_fences {
            self.wait_present_fence(slot)?;
            self.destroy_retired();
        }

        // #region recreate-triggers
        // Three triggers, checked where they can happen:
        // - the window reported a new size (`resized`, set by the event loop)
        //   or the last present was SUBOPTIMAL: recreate before acquiring;
        // - acquire returns OUT_OF_DATE: no image was acquired (and the
        //   semaphore was not signaled), so recreate and acquire again;
        // - acquire returns SUBOPTIMAL: an image WAS acquired and the
        //   semaphore will be signaled, so render and present this frame and
        //   recreate afterwards (an acquired image must be presented, or
        //   released with vkReleaseSwapchainImagesKHR).
        if self.resized && !self.recreate()? {
            return Ok(());
        }
        let acquired = loop {
            // SAFETY: the swapchain and the semaphore belong to this device;
            // the semaphore is unsignaled (its last wait completed: in_flight).
            let result = unsafe {
                self.swapchain_loader.acquire_next_image(
                    self.chain.swapchain,
                    u64::MAX,
                    image_available,
                    vk::Fence::null(),
                )
            };
            match result {
                Ok(r) => break r,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    println!("[timing] acquire: VK_ERROR_OUT_OF_DATE_KHR");
                    if !self.recreate()? {
                        return Ok(());
                    }
                }
                Err(e) => return Err(e.into()),
            }
        };
        let (image_index, suboptimal) = acquired;
        if suboptimal {
            println!("[timing] acquire: VK_SUBOPTIMAL_KHR");
            self.resized = true;
        }
        // #endregion
        let index = usize::try_from(image_index)?;
        let image = *self
            .chain
            .images
            .get(index)
            .ok_or("image index out of range")?;
        let view = *self
            .chain
            .views
            .get(index)
            .ok_or("image index out of range")?;
        let render_finished = *self
            .chain
            .render_finished
            .get(index)
            .ok_or("image index out of range")?;
        let extent = self.chain.extent;
        let device = &self.device;
        // SAFETY: the fence is not in use by a pending submission (we waited on it).
        unsafe { device.reset_fences(&[in_flight])? };

        // Record as in step 10, with the CURRENT chain's extent.
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
        let t = self.start.elapsed().as_secs_f32();
        let wave = |phase: f32| 0.1 + 0.1 * (t * 1.5 + phase).sin();
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [wave(0.0), wave(2.1), wave(4.2), 1.0],
                },
            });
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        let rendering_info = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment));
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        // SAFETY: the buffer is not pending (in_flight waited); every handle belongs to this device.
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_attachment]),
            );
            device.cmd_begin_rendering(cmd, &rendering_info);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[area]);
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_rendering(cmd);
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_present]),
            );
            device.end_command_buffer(cmd)?;
        }
        let wait = vk::SemaphoreSubmitInfo::default()
            .semaphore(image_available)
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
        unsafe { device.queue_submit2(self.queue, &[submit], in_flight)? };

        // #region present
        // With present fences, VkSwapchainPresentFenceInfoKHR in the pNext
        // chain gives the present a fence (one per swapchain in the call).
        let swapchains = [self.chain.swapchain];
        let indices = [image_index];
        let waits = [render_finished];
        let frame = self.frames.get_mut(slot).ok_or("frame slot out of range")?;
        let fences = [frame.present_fence];
        let mut fence_info = vk::SwapchainPresentFenceInfoKHR::default().fences(&fences);
        let mut present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&swapchains)
            .image_indices(&indices);
        if self.present_fences {
            present_info = present_info.push(&mut fence_info);
            frame.present_pending = true;
        }
        // SAFETY: the image was acquired above and is in PRESENT_SRC_KHR after
        // the submit; the present fence is unsignaled (wait-present-fence).
        let presented = unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present_info)
        };
        // OUT_OF_DATE from the present: the image was still "presented" in the
        // sense that the queue operation (the semaphore wait) is enqueued.
        // Either way the swapchain is replaced before the next acquire.
        match presented {
            Ok(false) => {}
            Ok(true) => {
                println!("[timing] present: VK_SUBOPTIMAL_KHR");
                self.resized = true;
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                println!("[timing] present: VK_ERROR_OUT_OF_DATE_KHR");
                self.resized = true;
            }
            Err(e) => return Err(e.into()),
        }
        // #endregion
        self.frame_index += 1;
        Ok(())
    }

    /// Destroys the Vulkan objects in reverse order of creation.
    fn destroy(self) -> Result<(), vk::Result> {
        // #region cleanup
        // vkDeviceWaitIdle covers our submissions; the present fences cover
        // the presents of every chain (current and retired), so wait on all
        // that are pending - with the timeout, as in wait-present-fence.
        // SAFETY: plain calls on a live device.
        let waited = unsafe { self.device.device_wait_idle() };
        if let Err(e) = &waited {
            eprintln!("device_wait_idle failed: {e}");
        }
        let pending: Vec<vk::Fence> = self
            .frames
            .iter()
            .filter(|f| f.present_pending)
            .map(|f| f.present_fence)
            .collect();
        if !pending.is_empty() {
            let timeout = u64::try_from(PRESENT_FENCE_TIMEOUT.as_nanos()).unwrap_or(u64::MAX);
            // SAFETY: the fences belong to this device.
            match unsafe { self.device.wait_for_fences(&pending, true, timeout) } {
                Ok(()) => {}
                Err(e) => eprintln!("waiting for the present fences failed: {e}"),
            }
        }
        println!(
            "[timing] {} frames, {} swapchain recreation(s)",
            self.frame_index, self.recreations
        );
        for r in &self.retired {
            self.destroy_chain(&r.chain);
        }
        self.destroy_chain(&self.chain);
        // SAFETY: nothing else uses these objects, and each is destroyed exactly once.
        unsafe {
            for &fence in &self.abandoned {
                self.device.destroy_fence(fence, None);
            }
            for frame in &self.frames {
                self.device.destroy_fence(frame.in_flight, None);
                if frame.present_fence != vk::Fence::null() {
                    self.device.destroy_fence(frame.present_fence, None);
                }
                self.device.destroy_semaphore(frame.image_available, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
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
/// winit drives the program: `vk` is built in the first `resumed`; a resize
/// sets a flag (and a 0x0 size pauses rendering); `about_to_wait` runs the
/// demo script, checks --seconds and asks for the next frame.
struct App {
    args: Args,
    vk: Option<Vk>,
    error: Option<Box<dyn Error>>,
    /// Index of the next `DEMO` step.
    demo_step: usize,
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
        let Some(vk) = self.vk.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            // A new size (0x0 when minimized): recreate before the next frame.
            // While paused, that recreation is what un-pauses rendering. The
            // event's size can be stale: right after creation winit delivers a
            // Resized with the size the window had before it got its inner
            // size. So compare the window's CURRENT size with the swapchain's.
            WindowEvent::Resized(_) => {
                let size = vk.window.inner_size();
                let current = vk.chain.extent;
                if vk.paused || size.width != current.width || size.height != current.height {
                    vk.resized = true;
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = vk.draw_frame() {
                    self.error = Some(e);
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(vk) = self.vk.as_mut() else {
            return;
        };
        let elapsed = vk.start.elapsed().as_secs_f64();
        if self.args.seconds.is_some_and(|s| elapsed >= s) {
            event_loop.exit();
            return;
        }
        // #region demo-script
        if self.args.demo
            && let Some(&(at, action)) = DEMO.get(self.demo_step)
            && elapsed >= at
        {
            self.demo_step += 1;
            match action {
                DemoAction::Resize(w, h) => {
                    println!("Demo: resize to {w}x{h}");
                    // Some(size): applied at once (Windows); None: the new
                    // size arrives later as WindowEvent::Resized.
                    if vk
                        .window
                        .request_inner_size(PhysicalSize::new(w, h))
                        .is_some()
                    {
                        vk.resized = true;
                    }
                }
                DemoAction::Minimize => {
                    println!("Demo: minimize");
                    vk.window.set_minimized(true);
                }
                DemoAction::Restore => {
                    println!("Demo: restore");
                    vk.window.set_minimized(false);
                }
            }
        }
        // #endregion
        if vk.paused && !vk.resized {
            // Nothing to render: wake up now and then for the demo and --seconds.
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + Duration::from_millis(10),
            ));
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
            vk.window.request_redraw();
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
        demo_step: 0,
    };
    event_loop.run_app(&mut app)?;
    if let Some(vk) = app.vk.take() {
        println!("Window closed");
        vk.destroy()?;
    }
    app.error.map_or(Ok(()), Err)
}
// #endregion
