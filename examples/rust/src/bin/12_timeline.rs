//! Step 12 - Timeline semaphores and exact barriers.
//!
//! Step 10 kept one fence per frame in flight. A timeline semaphore replaces
//! all of them: it holds a 64-bit counter that only grows. Frame n's submit
//! sets the counter to n when the frame completes, and before re-recording a
//! slot the CPU waits until the counter reaches n - frames in flight, the last
//! frame that used the slot. Like a render farm's "frames done" counter:
//! instead of one flag per job, one number says how far the work has got.
//!
//!   frame n: wait counter >= n - frames in flight -> acquire -> record -> submit (signal n) -> present
//!
//! Acquire and present still use BINARY semaphores: the swapchain functions
//! only accept those. The frame also does more than one kind of GPU work, so
//! the barriers between them matter: the triangle is rendered into an
//! offscreen 400x300 image, which is then blitted (copied with scaling) to the
//! swapchain image - render, copy, present, each ordered by its own barrier.
//! `--frames 1|2|3` (default 2) sets the frames in flight, `--seconds S`
//! closes the window after S seconds. Lines that depend on timing start with
//! `[timing]`. Steps 01-10 are repeated, reduced to what this step needs.

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

// #region cli
/// Command line:
/// - `--frames N`: frames in flight, 1-3 (default 2);
/// - `--seconds S`: close the window after S seconds (default: run until closed).
#[derive(Clone, Copy)]
struct Args {
    frames: usize,
    seconds: Option<f64>,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut args = Self {
            frames: 2,
            seconds: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            let value = it.next().ok_or(format!("{arg} needs a value"))?;
            match arg.as_str() {
                "--frames" => args.frames = value.parse()?,
                "--seconds" => args.seconds = Some(value.parse()?),
                _ => {
                    return Err(format!(
                        "unknown argument {arg} (use --frames 1|2|3, --seconds S)"
                    )
                    .into());
                }
            }
        }
        if !(1..=3).contains(&args.frames) {
            return Err("--frames must be 1, 2 or 3".into());
        }
        Ok(args)
    }
}
// #endregion

/// Size of the offscreen image the triangle is rendered into.
const OFFSCREEN: vk::Extent2D = vk::Extent2D {
    width: 400,
    height: 300,
};

/// How many frames print what they wait for and signal.
const TRACED_FRAMES: u64 = 5;

/// GPU timestamps per frame: command buffer start, rendering done, blit done.
const TIMESTAMPS: u32 = 3;

/// The GPU timestamps of the frames read back during one second, reduced to
/// an average and a maximum per interval, printed once per second.
struct GpuTimes {
    /// Nanoseconds per timestamp tick (VkPhysicalDeviceLimits::timestampPeriod).
    period_ns: f64,
    /// The bits of a timestamp that count (timestampValidBits of the family).
    mask: u64,
    /// The last whole second that was printed.
    second: u64,
    frames: u32,
    /// Command buffer start -> color output done: sum and maximum, in us.
    render: (f64, f64),
    /// Color output done -> blit done: sum and maximum, in us.
    to_blit: (f64, f64),
}

impl GpuTimes {
    fn add(&mut self, render: f64, to_blit: f64) {
        self.frames += 1;
        self.render = (self.render.0 + render, self.render.1.max(render));
        self.to_blit = (self.to_blit.0 + to_blit, self.to_blit.1.max(to_blit));
    }

    fn report(&mut self, second: u64) {
        let n = f64::from(self.frames.max(1));
        println!(
            "[timing] {second}s, GPU timestamps of {} frames: render {:.1} us avg, {:.1} us max; render done -> blit done {:.1} us avg, {:.1} us max",
            self.frames,
            self.render.0 / n,
            self.render.1,
            self.to_blit.0 / n,
            self.to_blit.1
        );
        self.second = second;
        self.frames = 0;
        self.render = (0.0, 0.0);
        self.to_blit = (0.0, 0.0);
    }
}

// #region frame-resources
/// What the CPU touches while recording one frame: one set per frame in
/// flight. Compared with step 10 the fence is gone: the timeline semaphore
/// tells when the slot is free again.
struct Frame {
    command_buffer: vk::CommandBuffer,
    /// Signaled by vkAcquireNextImageKHR, waited on by this slot's submit.
    image_available: vk::Semaphore,
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
    /// Function table of VK_KHR_swapchain (a device extension).
    swapchain_loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    /// Owned by the swapchain; the blit writes into them, nothing renders into
    /// them, so this step needs no image views for them.
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    /// The image the triangle is rendered into, its memory and its view.
    offscreen: vk::Image,
    offscreen_memory: vk::DeviceMemory,
    offscreen_view: vk::ImageView,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    /// One per frame in flight.
    frames: Vec<Frame>,
    /// One per swapchain image, indexed by the acquired image index.
    render_finished: Vec<vk::Semaphore>,
    /// The timeline: its counter is the number of the last completed frame.
    timeline: vk::Semaphore,
    /// Frames submitted so far = the value the last submit signals.
    frame_number: u64,
    /// TIMESTAMPS queries per frame in flight.
    query_pool: vk::QueryPool,
    gpu: GpuTimes,
    start: Instant,
    window: Window,
}

impl Vk {
    /// Steps 01-08, the offscreen image, then the queue, the command pool,
    /// per-frame buffers and the sync objects.
    fn new(event_loop: &ActiveEventLoop, args: Args) -> Result<Self, Box<dyn Error>> {
        // Window from step 06: 800x600 physical pixels, fixed size.
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("Vulkan from Zero - 12 timeline semaphores")
                .with_inner_size(PhysicalSize::new(800, 600))
                .with_resizable(false),
        )?;
        let size = window.inner_size();
        println!("Window: {}x{} pixels", size.width, size.height);

        // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vulkan from Zero - 12 timeline semaphores")
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

        // #region pick-device
        // Steps 03 + 06, plus the timelineSemaphore feature (core since 1.2,
        // and every Vulkan 1.4 device supports it - but a feature is only
        // usable when the device is created with it enabled).
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
            if f12.timeline_semaphore == 0
                || f13.dynamic_rendering == 0
                || f13.synchronization2 == 0
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
            "No Vulkan 1.4 GPU with timelineSemaphore + dynamic rendering + synchronization2 + VK_KHR_swapchain can present to this window",
        )?;
        // SAFETY: `physical_device` was enumerated from this instance.
        let chosen = unsafe { instance.get_physical_device_properties(physical_device) };
        println!(
            "\nPicked: {}",
            chosen.device_name_as_c_str()?.to_string_lossy()
        );
        println!("Graphics + present family: {family}");

        let priorities = [1.0_f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let device_exts = [swapchain::NAME.as_ptr()];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_exts)
            .push(&mut features12)
            .push(&mut features13);
        // SAFETY: `physical_device` belongs to `instance`, and all create-info data outlives the call.
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        let swapchain_loader = swapchain::Device::load(&instance, &device);
        println!(
            "VkDevice created with {} and timelineSemaphore",
            swapchain::NAME.to_string_lossy()
        );
        // #endregion

        // Format, present mode, extent and image count as in step 07.
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
        println!(
            "\nSurface format: {:?} + {:?}",
            format.format, format.color_space
        );
        // SAFETY: as above.
        let modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
        };
        let present_mode = if modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };
        println!("Present mode: {present_mode:?}");
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

        // #region swapchain-usage
        // Nothing of ours renders into the swapchain images any more: they are
        // the DESTINATION of a blit, a transfer operation, so they need
        // TRANSFER_DST. Only COLOR_ATTACHMENT is guaranteed for a surface,
        // everything else must be checked. COLOR_ATTACHMENT stays anyway:
        // implicit layers of capture and overlay tools draw into the swapchain
        // images themselves. On the dev machine the Bandicam layer creates
        // views and framebuffers on them, and without COLOR_ATTACHMENT the
        // validation layer reports its calls (VUID-VkImageViewCreateInfo-image-04441,
        // VUID-VkFramebufferCreateInfo-pAttachments-00877).
        if !caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            return Err("The surface does not support TRANSFER_DST swapchain images".into());
        }
        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);
        // SAFETY: `swapchain_info` refers to a live surface of this instance.
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_info, None)? };
        // SAFETY: `swapchain` is alive.
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        println!(
            "Swapchain: {} images of {}x{}, usage TRANSFER_DST (blit target) + COLOR_ATTACHMENT (overlay layers)",
            images.len(),
            extent.width,
            extent.height
        );
        // #endregion

        // #region offscreen-image
        // The image the triangle is rendered into: 400x300, half the window,
        // same format as the swapchain. Usage COLOR_ATTACHMENT (rendered into)
        // and TRANSFER_SRC (blitted from). The format must support both on
        // optimal tiling, and blitting into the swapchain format needs BLIT_DST.
        // SAFETY: `physical_device` belongs to this instance.
        let format_props = unsafe {
            instance.get_physical_device_format_properties(physical_device, format.format)
        };
        let needed = vk::FormatFeatureFlags::COLOR_ATTACHMENT
            | vk::FormatFeatureFlags::BLIT_SRC
            | vk::FormatFeatureFlags::BLIT_DST;
        if !format_props.optimal_tiling_features.contains(needed) {
            return Err(format!(
                "{:?} cannot be rendered into and blitted on this GPU",
                format.format
            )
            .into());
        }
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format.format)
            .extent(vk::Extent3D {
                width: OFFSCREEN.width,
                height: OFFSCREEN.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: plain create call on a live device.
        let offscreen = unsafe { device.create_image(&image_info, None)? };
        // Unlike the swapchain's images, this one has no memory yet. Memory is
        // module 8; here: the first memory type the image accepts that is
        // DEVICE_LOCAL (video memory), one allocation, bound at offset 0.
        // SAFETY: `offscreen` belongs to this device.
        let requirements = unsafe { device.get_image_memory_requirements(offscreen) };
        // SAFETY: `physical_device` belongs to this instance.
        let memory_props =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let memory_type = memory_props
            .memory_types_as_slice()
            .iter()
            .enumerate()
            .position(|(i, t)| {
                requirements.memory_type_bits & (1 << i) != 0
                    && t.property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or("No DEVICE_LOCAL memory type for the offscreen image")?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(u32::try_from(memory_type)?);
        // SAFETY: the size and the type come from the image's requirements.
        let offscreen_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        // SAFETY: the memory fits the image (same requirements), offset 0 is aligned.
        unsafe { device.bind_image_memory(offscreen, offscreen_memory, 0)? };
        let view_info = vk::ImageViewCreateInfo::default()
            .image(offscreen)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format.format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );
        // SAFETY: `offscreen` has memory bound.
        let offscreen_view = unsafe { device.create_image_view(&view_info, None)? };
        println!(
            "Offscreen image: {}x{} {:?}, usage COLOR_ATTACHMENT + TRANSFER_SRC, memory type {memory_type} (DEVICE_LOCAL)",
            OFFSCREEN.width, OFFSCREEN.height, format.format
        );
        // #endregion

        // The pipeline of step 08, rendering into the offscreen image's format.
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
        println!("Graphics pipeline created (triangle.spv, dynamic rendering)");

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
            .command_buffer_count(u32::try_from(args.frames)?);
        // SAFETY: `command_pool` belongs to this device.
        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };

        // #region timestamps
        // A query pool of GPU timestamps, TIMESTAMPS per frame in flight. A
        // timestamp is the value of a GPU tick counter; timestampValidBits of
        // the queue family says how many of its bits count (0 = no timestamps
        // on that family), timestampPeriod how many nanoseconds one tick is.
        // SAFETY: `physical_device` belongs to this instance.
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let valid_bits = families
            .get(usize::try_from(family)?)
            .ok_or("queue family out of range")?
            .timestamp_valid_bits;
        if valid_bits == 0 {
            return Err("The graphics queue family has no timestamps".into());
        }
        let pool_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(TIMESTAMPS * u32::try_from(args.frames)?);
        // SAFETY: plain create call on a live device.
        let query_pool = unsafe { device.create_query_pool(&pool_info, None)? };
        let gpu = GpuTimes {
            period_ns: f64::from(chosen.limits.timestamp_period),
            mask: u64::MAX >> (64 - valid_bits.min(64)),
            second: 0,
            frames: 0,
            render: (0.0, 0.0),
            to_blit: (0.0, 0.0),
        };
        // #endregion

        // #region timeline-semaphore
        // A semaphore is binary unless its create info carries a
        // VkSemaphoreTypeCreateInfo with TIMELINE. The counter starts at
        // initial_value (0: no frame has completed yet) and can only grow.
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let timeline_info = vk::SemaphoreCreateInfo::default().push(&mut type_info);
        // SAFETY: plain create call on a live device with timelineSemaphore enabled.
        let timeline = unsafe { device.create_semaphore(&timeline_info, None)? };
        // #endregion

        // #region binary-semaphores
        // Acquire and present still need BINARY semaphores: a timeline one is
        // invalid there (VUID-vkAcquireNextImageKHR-semaphore-03265,
        // VUID-vkQueuePresentKHR-pWaitSemaphores-03267). So, as in step 10:
        // image_available per frame in flight, render_finished per swapchain image.
        let binary_info = vk::SemaphoreCreateInfo::default();
        let mut frames = Vec::with_capacity(args.frames);
        for &command_buffer in &command_buffers {
            // SAFETY: plain create call on a live device.
            let image_available = unsafe { device.create_semaphore(&binary_info, None)? };
            frames.push(Frame {
                command_buffer,
                image_available,
            });
        }
        let mut render_finished = Vec::with_capacity(images.len());
        for _ in &images {
            // SAFETY: as above.
            render_finished.push(unsafe { device.create_semaphore(&binary_info, None)? });
        }
        println!(
            "\nSync objects: 1 timeline semaphore, {} + {} binary semaphores, no fences ({} frame(s) in flight)",
            frames.len(),
            render_finished.len(),
            frames.len()
        );
        // #endregion

        window.request_redraw();
        println!("\nRendering. Close the window to exit.");
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
            images,
            extent,
            offscreen,
            offscreen_memory,
            offscreen_view,
            pipeline_layout,
            pipeline,
            queue,
            command_pool,
            frames,
            render_finished,
            timeline,
            frame_number: 0,
            query_pool,
            gpu,
            start: Instant::now(),
            window,
        })
    }

    /// Renders and presents one frame. Returns `Ok(false)` when the swapchain
    /// no longer matches the window (VK_ERROR_OUT_OF_DATE_KHR): recreating it is
    /// step 11, here the caller simply stops.
    fn draw_frame(&mut self) -> Result<bool, Box<dyn Error>> {
        let device = &self.device;
        // Frame numbers start at 1: the counter's 0 means "nothing done yet".
        let n = self.frame_number + 1;
        let in_flight = self.frames.len() as u64;
        let slot = usize::try_from((n - 1) % in_flight)?;
        let frame = self.frames.get(slot).ok_or("frame slot out of range")?;
        let cmd = frame.command_buffer;
        let traced = n <= TRACED_FRAMES;
        let first_query = u32::try_from(slot)? * TIMESTAMPS;

        // #region wait-timeline
        // The last frame that used this slot is n - frames in flight. Wait
        // until the counter has reached its number: then its command buffer
        // and its image_available semaphore are free. vkWaitSemaphores waits
        // for "counter >= value", so a counter that has run further ahead is
        // fine. The first frames of each slot have nothing to wait for.
        if n > in_flight {
            let wait_value = n - in_flight;
            let semaphores = [self.timeline];
            let values = [wait_value];
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            // SAFETY: the semaphore is a timeline semaphore of this device.
            unsafe { device.wait_semaphores(&wait_info, u64::MAX)? };
            if traced {
                // SAFETY: as above.
                let counter = unsafe { device.get_semaphore_counter_value(self.timeline)? };
                println!(
                    "Frame {n}: slot {slot}, the CPU waits until the counter reaches {wait_value}, the submit will signal {n}"
                );
                println!("[timing] frame {n}: the counter read after the wait: {counter}");
            }
        } else if traced {
            println!(
                "Frame {n}: slot {slot}, no earlier frame used this slot, the submit will signal {n}"
            );
        }
        // #endregion

        // #region read-timestamps
        // Frame n - F, the last user of this slot, has completed (the wait
        // above), so its three timestamps are written and can be read before
        // they are reset for frame n. Ticks become microseconds through
        // timestampPeriod (nanoseconds per tick); differences are masked to
        // timestampValidBits, the bits the counter really has.
        if n > in_flight {
            let mut ticks = [0_u64; TIMESTAMPS as usize];
            // SAFETY: the queries were written by a completed submission.
            unsafe {
                device.get_query_pool_results(
                    self.query_pool,
                    first_query,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )?;
            }
            let us = |from: u64, to: u64| {
                (to.wrapping_sub(from) & self.gpu.mask) as f64 * self.gpu.period_ns / 1000.0
            };
            let [start, rendered, blitted] = ticks;
            let (render, to_blit) = (us(start, rendered), us(rendered, blitted));
            self.gpu.add(render, to_blit);
            let second = self.start.elapsed().as_secs();
            if second > self.gpu.second {
                self.gpu.report(second);
            }
        }
        // #endregion

        // Acquire as in step 10 (the semaphore is binary).
        // SAFETY: the swapchain and the semaphore belong to this device; the
        // semaphore is unsignaled (its last wait completed, see wait-timeline).
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
        let render_finished = *self
            .render_finished
            .get(index)
            .ok_or("image index out of range")?;

        // SAFETY: the buffer is not pending (wait-timeline) and its pool allows reset.
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
        let barrier = |image: vk::Image,
                       (src_stage, src_access): (vk::PipelineStageFlags2, vk::AccessFlags2),
                       (dst_stage, dst_access): (vk::PipelineStageFlags2, vk::AccessFlags2),
                       old_layout: vk::ImageLayout,
                       new_layout: vk::ImageLayout| {
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
                .subresource_range(color_range)
        };
        use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};

        // #region barriers
        // Five barriers, each "after THIS, before THAT" (stage + access on
        // both sides), recorded where an image changes hands:
        //
        // 1. Offscreen, UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL, before rendering.
        //    After: COLOR_ATTACHMENT_OUTPUT of earlier commands - the previous
        //    frame's color writes (write-after-write; barrier 3 already made
        //    them available) and, through barrier 5, the previous frame's blit.
        //    NOT after BLIT: the submit waits on image_available at BLIT, and a
        //    source stage BLIT here would chain this barrier, and the whole
        //    rendering behind it, to the acquire. Before: the color writes.
        // 2. Swapchain image, UNDEFINED -> TRANSFER_DST_OPTIMAL, before the blit.
        //    After: BLIT - the stage the submit waits on image_available at,
        //    so the transition waits for the acquire. Before: the blit's writes.
        // 3. Offscreen, COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL.
        //    After: the color writes (made available). Before: the blit's reads
        //    (made visible). This is the render -> copy hand-over.
        // 4. Swapchain image, TRANSFER_DST_OPTIMAL -> PRESENT_SRC_KHR.
        //    After: the blit's writes. Before: BLIT, the stage render_finished
        //    is signaled at, so the transition happens before the signal; the
        //    present itself needs no access mask.
        // 5. Offscreen, after the blit, no layout change. After: the blit's
        //    reads. Before: COLOR_ATTACHMENT_OUTPUT of every later command,
        //    which includes the NEXT frame's barrier 1 and rendering (one
        //    offscreen image for all frames in flight). A read followed by a
        //    write (write-after-read) needs only an execution dependency, so
        //    both access masks are NONE.
        let to_render = barrier(
            self.offscreen,
            (S::COLOR_ATTACHMENT_OUTPUT, A::NONE),
            (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
            L::UNDEFINED,
            L::COLOR_ATTACHMENT_OPTIMAL,
        );
        let to_blit_dst = barrier(
            image,
            (S::BLIT, A::NONE),
            (S::BLIT, A::TRANSFER_WRITE),
            L::UNDEFINED,
            L::TRANSFER_DST_OPTIMAL,
        );
        let to_blit_src = barrier(
            self.offscreen,
            (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
            (S::BLIT, A::TRANSFER_READ),
            L::COLOR_ATTACHMENT_OPTIMAL,
            L::TRANSFER_SRC_OPTIMAL,
        );
        let to_present = barrier(
            image,
            (S::BLIT, A::TRANSFER_WRITE),
            (S::BLIT, A::NONE),
            L::TRANSFER_DST_OPTIMAL,
            L::PRESENT_SRC_KHR,
        );
        let after_blit = barrier(
            self.offscreen,
            (S::BLIT, A::NONE),
            (S::COLOR_ATTACHMENT_OUTPUT, A::NONE),
            L::TRANSFER_SRC_OPTIMAL,
            L::TRANSFER_SRC_OPTIMAL,
        );
        // #endregion

        // #region render
        // Barrier 1, then the triangle into the offscreen image (400x300, so
        // the viewport and the scissor are 400x300 too), clearing it to the
        // animated color of step 10. Before it this slot's three queries are
        // reset (a query must be reset before it is written again) and
        // timestamp 0 is written at TOP_OF_PIPE: the moment the GPU starts
        // this command buffer. Timestamp 1, at COLOR_ATTACHMENT_OUTPUT after
        // the rendering, is written once the color writes are done.
        let t = self.start.elapsed().as_secs_f32();
        let wave = |phase: f32| 0.1 + 0.1 * (t * 1.5 + phase).sin();
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.offscreen_view)
            .image_layout(L::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [wave(0.0), wave(2.1), wave(4.2), 1.0],
                },
            });
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: OFFSCREEN,
        };
        let rendering_info = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment));
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: OFFSCREEN.width as f32,
            height: OFFSCREEN.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        // SAFETY: `cmd` is recording; every handle belongs to this device; the
        // slot's queries are not in use by pending work (wait-timeline).
        unsafe {
            device.cmd_reset_query_pool(cmd, self.query_pool, first_query, TIMESTAMPS);
            device.cmd_write_timestamp2(cmd, S::TOP_OF_PIPE, self.query_pool, first_query);
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_render]),
            );
            device.cmd_begin_rendering(cmd, &rendering_info);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[area]);
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_rendering(cmd);
            device.cmd_write_timestamp2(
                cmd,
                S::COLOR_ATTACHMENT_OUTPUT,
                self.query_pool,
                first_query + 1,
            );
        }
        // #endregion

        // #region blit
        // Barriers 2 and 3 in one call (independent images), the blit, then
        // barriers 4 and 5 in one call. The blit copies the whole 400x300 image
        // onto the whole swapchain image, scaling it; NEAREST keeps the pixels
        // square (2x2 screen pixels each in an 800x600 window). Timestamp 2
        // marks the end of the blit.
        let full = |e: vk::Extent2D| -> Result<[vk::Offset3D; 2], Box<dyn Error>> {
            Ok([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: i32::try_from(e.width)?,
                    y: i32::try_from(e.height)?,
                    z: 1,
                },
            ])
        };
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        let region = vk::ImageBlit2::default()
            .src_subresource(layers)
            .src_offsets(full(OFFSCREEN)?)
            .dst_subresource(layers)
            .dst_offsets(full(self.extent)?);
        let blit_info = vk::BlitImageInfo2::default()
            .src_image(self.offscreen)
            .src_image_layout(L::TRANSFER_SRC_OPTIMAL)
            .dst_image(image)
            .dst_image_layout(L::TRANSFER_DST_OPTIMAL)
            .regions(std::slice::from_ref(&region))
            .filter(vk::Filter::NEAREST);
        // SAFETY: `cmd` is recording; the layouts match the barriers before.
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_blit_dst, to_blit_src]),
            );
            device.cmd_blit_image2(cmd, &blit_info);
            device.cmd_write_timestamp2(cmd, S::BLIT, self.query_pool, first_query + 2);
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_present, after_blit]),
            );
            device.end_command_buffer(cmd)?;
        }
        // #endregion

        // #region submit
        // One batch, three semaphore operations:
        // - wait image_available (binary) at BLIT: only barrier 2 (source stage
        //   BLIT, so it chains to the wait) and the blit must wait. Barrier 1
        //   has no BLIT in its source stage, so rendering into the offscreen
        //   image may run before the swapchain image is ours;
        // - signal render_finished (binary) at BLIT, for the present;
        // - signal the timeline with value n at ALL_COMMANDS: the counter
        //   becomes n once everything in the batch has completed. This replaces
        //   step 10's fence argument, which is null now.
        // The value of a binary semaphore operation is ignored.
        let wait = vk::SemaphoreSubmitInfo::default()
            .semaphore(frame.image_available)
            .stage_mask(S::BLIT);
        let signals = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(render_finished)
                .stage_mask(S::BLIT),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline)
                .value(n)
                .stage_mask(S::ALL_COMMANDS),
        ];
        let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(std::slice::from_ref(&wait))
            .command_buffer_infos(std::slice::from_ref(&cmd_info))
            .signal_semaphore_infos(&signals);
        // SAFETY: all handles belong to this device; n is greater than every
        // value signaled before (a timeline must only grow).
        unsafe { device.queue_submit2(self.queue, &[submit], vk::Fence::null())? };
        self.frame_number = n;
        // #endregion

        // Present as in step 10: it waits on the binary render_finished.
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
            Ok(_suboptimal) => Ok(true),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    /// Destroys the Vulkan objects in reverse order of creation.
    fn destroy(self) -> Result<(), vk::Result> {
        // #region cleanup
        // The timeline tells when OUR work is done: wait for the last frame's
        // value. Destroying the swapchain also needs the presents to be done
        // (VUID-vkDestroySwapchainKHR-swapchain-01282), so vkDeviceWaitIdle
        // stays: it waits for queue work, the presents' queue operations
        // included; it does not wait for the presentation engine (present
        // fences, step 11). After it the counter equals the number of
        // submitted frames.
        let semaphores = [self.timeline];
        let values = [self.frame_number];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: plain waits on a live device.
        let waited = unsafe {
            self.device
                .wait_semaphores(&wait_info, u64::MAX)
                .and_then(|()| self.device.device_wait_idle())
        };
        match &waited {
            // SAFETY: the semaphore belongs to this device.
            Ok(()) => match unsafe { self.device.get_semaphore_counter_value(self.timeline) } {
                Ok(counter) => println!(
                    "[timing] timeline counter at exit: {counter} ({} frames submitted)",
                    self.frame_number
                ),
                Err(e) => eprintln!("vkGetSemaphoreCounterValue failed: {e}"),
            },
            Err(e) => eprintln!("waiting for the GPU failed: {e}"),
        }
        // SAFETY: nothing else uses these objects, and each is destroyed exactly once.
        unsafe {
            self.device.destroy_query_pool(self.query_pool, None);
            self.device.destroy_semaphore(self.timeline, None);
            for &semaphore in &self.render_finished {
                self.device.destroy_semaphore(semaphore, None);
            }
            for frame in &self.frames {
                self.device.destroy_semaphore(frame.image_available, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_image_view(self.offscreen_view, None);
            self.device.destroy_image(self.offscreen, None);
            self.device.free_memory(self.offscreen_memory, None);
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
        vk.destroy()?;
    }
    app.error.map_or(Ok(()), Err)
}
// #endregion
