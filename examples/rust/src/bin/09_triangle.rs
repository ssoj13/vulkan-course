//! Step 09 - Command buffers and the first triangle.
//!
//! The GPU does not execute API calls one by one: we *record* commands into a
//! command buffer (like writing a render script), then *submit* the whole
//! buffer to a queue, and the GPU runs it later, on its own schedule. One frame
//! is a small round trip between three parties - the CPU, the GPU queue and the
//! presentation engine - kept in order by a fence (GPU -> CPU) and semaphores
//! (GPU -> GPU):
//!
//!   wait fence -> acquire image -> record -> submit -> present -> (next frame)
//!
//! This step keeps ONE frame in flight: the CPU waits for the GPU before it
//! records the next frame. Overlapping frames, and recreating the swapchain
//! when the window changes, are module 7. Steps 01-08 are repeated, reduced to
//! what this step needs.

use std::error::Error;
use std::ffi::{CStr, c_void};

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
    image_views: Vec<vk::ImageView>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    queue: vk::Queue,
    /// Owned by the swapchain; kept to name them in barriers.
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    /// One per swapchain image, indexed by the acquired image index.
    render_finished: Vec<vk::Semaphore>,
    in_flight: vk::Fence,
    window: Window,
}

impl Vk {
    /// Steps 01-08, then the queue, the command pool and buffer, and the sync objects.
    fn new(event_loop: &ActiveEventLoop) -> Result<Self, Box<dyn Error>> {
        // Window from step 06: 800x600 physical pixels, fixed size.
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("Vulkan from Zero - 09 triangle")
                .with_inner_size(PhysicalSize::new(800, 600))
                .with_resizable(false),
        )?;
        let size = window.inner_size();
        println!("Window: {}x{} pixels", size.width, size.height);

        // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vulkan from Zero - 09 triangle")
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
        // graphics + present family, discrete first - and now VK_KHR_swapchain.
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
            // A GPU that can present to the surface practically always has the
            // swapchain extension, but "practically" is not a check: a device
            // extension we enable must be listed here, or vkCreateDevice fails
            // with VK_ERROR_EXTENSION_NOT_PRESENT.
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

        // Step 05's device, plus the first device extension of the book. The
        // swapchain functions (create, get images, acquire, present) only
        // exist on a device that enabled VK_KHR_swapchain.
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
        // ash loads the extension's device-level functions into their own table.
        let swapchain_loader = swapchain::Device::load(&instance, &device);
        println!(
            "VkDevice created with {}",
            swapchain::NAME.to_string_lossy()
        );

        // We want 8-bit BGRA with the *_SRGB encoding on an sRGB monitor: shaders
        // write linear values, the hardware encodes them to sRGB when storing
        // (and blending happens in linear space). Every desktop driver offers
        // it; if not, the first pair the surface lists is still a valid choice.
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
            "\nSurface formats: {}, chosen {:?} + {:?}",
            formats.len(),
            format.format,
            format.color_space
        );

        // MAILBOX: no tearing and the lowest latency (a newer frame replaces the
        // one waiting for vblank). FIFO is the only mode every driver must
        // support, so it is the fallback.
        // SAFETY: as above.
        let modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
        };
        let present_mode = if modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };
        let names = modes.iter().map(|m| format!("{m:?}")).collect::<Vec<_>>();
        println!(
            "Present modes: {}; chosen {present_mode:?}",
            names.join(", ")
        );

        // Normally the swapchain must match the window: current_extent. The
        // special value u32::MAX means the window takes its size from the
        // swapchain (Wayland); then we use the window size, clamped to the range.
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
        println!("Extent: {}x{}", extent.width, extent.height);

        // The minimum is what the presentation engine needs for itself; one
        // more lets us render into a free image while the others are on screen
        // or queued. max_image_count 0 means "no limit".
        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            image_count = image_count.min(caps.max_image_count);
        }
        println!(
            "Image count: {image_count} (min {} + 1, max {})",
            caps.min_image_count, caps.max_image_count
        );

        // usage COLOR_ATTACHMENT: we will render into the images.
        // EXCLUSIVE: one queue family uses them (graphics = present family).
        // pre_transform = current_transform: no extra rotation by the compositor.
        // OPAQUE: ignore alpha when composing with the desktop.
        // clipped: pixels hidden behind other windows may be skipped.
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
        // The driver may create MORE images than min_image_count; always ask.
        // SAFETY: `swapchain` is alive.
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        println!("Swapchain created: {} images", images.len());

        // An image is memory plus a description; a view says how to read it:
        // as a 2D image, in this format, with channels unswizzled, color aspect,
        // mip level 0 and array layer 0 only. Rendering attaches views, not images.
        let mut image_views = Vec::with_capacity(images.len());
        for &image in &images {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .components(vk::ComponentMapping::default()) // IDENTITY for r, g, b, a
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );
            // SAFETY: `image` belongs to this device's swapchain.
            image_views.push(unsafe { device.create_image_view(&view_info, None)? });
        }
        println!("Image views created: {}", image_views.len());

        // The SPIR-V is compiled from shaders/triangle.slang by `bun run sync:shaders`
        // (one module, two entry points) and embedded in the executable at build
        // time. `read_spv` checks the magic number and copies the bytes into u32
        // words: Vulkan wants 4-byte aligned code, `include_bytes!` guarantees
        // only 1-byte alignment.
        const TRIANGLE_SPV: &[u8] = include_bytes!("../../../../shaders/spv/triangle.spv");
        let spirv = ash::util::read_spv(&mut std::io::Cursor::new(TRIANGLE_SPV))?;
        println!("\nSPIR-V: {} bytes (triangle.spv)", TRIANGLE_SPV.len());

        // A shader module is just the SPIR-V handed to the driver; nothing is
        // compiled for the GPU yet. That happens in vkCreateGraphicsPipelines,
        // when the driver knows the whole pipeline state.
        let module_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
        // SAFETY: `spirv` is valid SPIR-V (checked by spirv-val in sync:shaders).
        let shader_module = unsafe { device.create_shader_module(&module_info, None)? };
        println!("Shader module created: entry points vertexMain, fragmentMain");

        // One module, two stages: each stage names its entry point in the module.
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

        // Vertex input: no vertex buffers yet, the shader makes up its corners.
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        // Input assembly: every 3 vertices form one triangle.
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        // Viewport + scissor: one of each; their values are set while recording
        // (dynamic state below), only the count is baked in.
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        // Rasterizer: filled triangles; back faces culled, and clockwise on
        // screen counts as front (our triangle is clockwise, see triangle.slang).
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);
        // One sample per pixel: no MSAA (module 10).
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        // Color blend, one entry per color attachment: no blending, write RGBA.
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

        // Viewport and scissor are dynamic: one pipeline works for any window
        // size, and recreating the swapchain (module 7) needs no new pipeline.
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // The layout declares what resources the shaders read: descriptor set
        // layouts and push-constant ranges. Our shaders read none, so it is
        // empty - but every pipeline still needs one.
        let layout_info = vk::PipelineLayoutCreateInfo::default();
        // SAFETY: plain create call on a live device.
        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };
        println!("Pipeline layout created: 0 descriptor set layouts, 0 push constant ranges");

        // Dynamic rendering (core in 1.3): instead of a VkRenderPass, the
        // pipeline only states the FORMATS of the attachments it will draw into.
        // vkCmdBeginRendering supplies the actual image views when recording.
        let color_formats = [format.format];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

        // Everything together. render_pass stays null: the rendering info in the
        // pNext chain replaces it. Here the driver compiles SPIR-V into GPU code
        // for exactly this state, which is why pipelines are created up front.
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
        println!(
            "Graphics pipeline created for color format {:?} (dynamic rendering, no render pass)",
            format.format
        );
        // The pipeline keeps its own compiled code: the module is no longer needed.
        // SAFETY: the module is used by no other pending create call.
        unsafe { device.destroy_shader_module(shader_module, None) };
        println!("Shader module destroyed");

        // #region get-queue
        // The one queue of step 05: it records nothing itself, it executes
        // command buffers (vkQueueSubmit2) and presents (vkQueuePresentKHR).
        // SAFETY: one queue of `family` was requested in `queue_infos`.
        let queue = unsafe { device.get_device_queue(family, 0) };
        // #endregion

        // #region command-pool
        // Command buffers are allocated from a pool tied to ONE queue family
        // (the memory they use is managed per family). RESET_COMMAND_BUFFER lets
        // us reset and re-record the same buffer every frame.
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(family);
        // SAFETY: plain create call on a live device.
        let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };
        // #endregion

        // #region command-buffer
        // PRIMARY: can be submitted to a queue (SECONDARY ones are called from
        // primaries). One buffer is enough with one frame in flight.
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `command_pool` belongs to this device.
        let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info)? }
            .first()
            .copied()
            .ok_or("vkAllocateCommandBuffers returned no command buffer")?;
        println!("\nCommand pool + 1 primary command buffer");
        // #endregion

        // #region sync-objects
        // Semaphores order GPU work against GPU work (here: acquire -> render
        // -> present); a fence lets the CPU wait for the GPU.
        // - image_available: signaled when the acquired image may be written.
        // - render_finished: signaled when rendering is done, waited on by the
        //   present. One PER SWAPCHAIN IMAGE: the presentation engine holds it
        //   until that image is shown again, so we may reuse it only after the
        //   same image comes back from vkAcquireNextImageKHR.
        // - in_flight: signaled when the frame's submission completes. Created
        //   signaled so the first frame does not wait forever.
        // One frame in flight: the CPU waits for frame N before recording N+1.
        // Several frames in flight (CPU and GPU overlap) are module 7.
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        // SAFETY: plain create calls on a live device.
        let image_available = unsafe { device.create_semaphore(&semaphore_info, None)? };
        let mut render_finished = Vec::with_capacity(images.len());
        for _ in &images {
            // SAFETY: as above.
            render_finished.push(unsafe { device.create_semaphore(&semaphore_info, None)? });
        }
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        // SAFETY: as above.
        let in_flight = unsafe { device.create_fence(&fence_info, None)? };
        println!(
            "Sync objects: 1 fence, 1 + {} semaphores (one frame in flight)",
            render_finished.len()
        );
        // #endregion

        // winit sends RedrawRequested only when asked: ask for the first frame.
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
            image_views,
            pipeline_layout,
            pipeline,
            queue,
            images,
            extent,
            command_pool,
            command_buffer,
            image_available,
            render_finished,
            in_flight,
            window,
        })
    }

    /// Renders and presents one frame. Returns `Ok(false)` when the swapchain
    /// no longer matches the window (VK_ERROR_OUT_OF_DATE_KHR): recreating it is
    /// module 7, here the caller simply stops.
    fn draw_frame(&mut self) -> Result<bool, Box<dyn Error>> {
        let device = &self.device;
        let cmd = self.command_buffer;

        // #region wait-fence
        // Wait until the previous frame's submission has finished on the GPU:
        // only then may we re-record its command buffer and reuse
        // image_available. The fence is reset only once we are sure to submit
        // (after a successful acquire), or the next wait would never return.
        // SAFETY: `in_flight` belongs to this device.
        unsafe { device.wait_for_fences(&[self.in_flight], true, u64::MAX)? };
        // #endregion

        // #region acquire
        // Ask the presentation engine for a free image. The call returns an
        // index at once, but the image may still be on screen: image_available
        // is signaled when it is really ours. SUBOPTIMAL (`_suboptimal` = true)
        // still works, the image just no longer matches the surface exactly.
        // SAFETY: the swapchain and the semaphore belong to this device; the
        // semaphore is unsignaled (its last wait completed, see wait-fence).
        let acquired = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available,
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
        unsafe { device.reset_fences(&[self.in_flight])? };
        // #endregion

        // #region begin-recording
        // Recording only writes commands into the buffer; the GPU sees nothing
        // until the submit. ONE_TIME_SUBMIT: we re-record it every frame.
        // SAFETY: the buffer is not pending (wait-fence) and its pool allows reset.
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        // #endregion

        let color_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        // #region barrier-to-attachment
        // Image layout = how the image's memory is arranged for one kind of use.
        // UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL: we will clear it anyway, so the
        // old contents may be discarded. The source stage is the same stage the
        // submit waits on image_available at (COLOR_ATTACHMENT_OUTPUT), which
        // chains the two: the transition starts only after the image is ours.
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
        // SAFETY: `cmd` is recording; the barrier refers to a live image.
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&to_attachment)),
            );
        }
        // #endregion

        // #region begin-rendering
        // Dynamic rendering: the attachments are named right here. CLEAR at the
        // start (a dark blue-gray, linear values), STORE at the end so the
        // pixels survive for presentation.
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
        // SAFETY: `cmd` is recording; the view is in COLOR_ATTACHMENT_OPTIMAL now.
        unsafe { device.cmd_begin_rendering(cmd, &rendering_info) };
        // #endregion

        // #region draw
        // Bind the pipeline, fill in its dynamic state (the whole image), and
        // draw 3 vertices of 1 instance: the vertex shader runs for
        // SV_VulkanVertexID 0, 1, 2.
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
        // SAFETY: `cmd` is recording inside a rendering compatible with the pipeline.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_rendering(cmd);
        }
        // #endregion

        // #region barrier-to-present
        // COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR, after the color writes.
        // The destination stage matches the stage render_finished is signaled at
        // (COLOR_ATTACHMENT_OUTPUT), so the transition is ordered before the
        // signal; the present waits on that semaphore, and needs no access mask:
        // presentation makes the writes visible itself.
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
        // #endregion

        // #region submit
        // One batch: wait for image_available before COLOR_ATTACHMENT_OUTPUT
        // (earlier stages - vertex shading - may already run), execute `cmd`,
        // then signal render_finished; in_flight is signaled when it all completes.
        let wait = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.image_available)
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
        unsafe { device.queue_submit2(self.queue, &[submit], self.in_flight)? };
        // #endregion

        // #region present
        // Hand the image back to the presentation engine, to be shown once
        // render_finished is signaled. SUBOPTIMAL (Ok(true)) is still shown.
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
        // #endregion
    }

    /// Destroys the Vulkan objects in reverse order of creation.
    fn destroy(self) -> Result<(), vk::Result> {
        // #region cleanup
        // First wait until the GPU is idle: the last frame may still be rendering,
        // and nothing it uses may be destroyed before. The wait covers all work
        // submitted to the queues, which is what destroying the swapchain needs
        // (VUID-vkDestroySwapchainKHR-swapchain-01282); it does not wait for the
        // presentation engine to finish showing the last image. Knowing when a
        // present released its semaphore takes present fences
        // (VK_KHR_swapchain_maintenance1), module 7. `destroy` runs whenever the
        // event loop ends, also after an error. Then this step's objects (the
        // pool frees its command buffer), then everything from steps 07-08 in
        // reverse order.
        // SAFETY: plain call on a live device.
        let waited = unsafe { self.device.device_wait_idle() };
        if let Err(e) = &waited {
            eprintln!("device_wait_idle failed: {e}");
        }
        // SAFETY: nothing else uses these objects, and each is destroyed exactly once.
        unsafe {
            self.device.destroy_fence(self.in_flight, None);
            for &semaphore in &self.render_finished {
                self.device.destroy_semaphore(semaphore, None);
            }
            self.device.destroy_semaphore(self.image_available, None);
            self.device.destroy_command_pool(self.command_pool, None);
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
#[derive(Default)]
struct App {
    vk: Option<Vk>,
    error: Option<Box<dyn Error>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.vk.is_some() {
            return;
        }
        match Vk::new(event_loop) {
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
            // One frame per redraw; each frame asks for the next. The fence wait
            // and the present mode (MAILBOX/FIFO) pace the loop.
            WindowEvent::RedrawRequested => {
                let Some(vk) = self.vk.as_mut() else {
                    return;
                };
                match vk.draw_frame() {
                    Ok(true) => vk.window.request_redraw(),
                    Ok(false) => {
                        // Minimal handling: recreating the swapchain is module 7.
                        println!(
                            "Swapchain out of date (the window changed); recreation is module 7, stopping"
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
    let event_loop = EventLoop::new()?;
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    if let Some(vk) = app.vk.take() {
        println!("Window closed");
        vk.destroy()?;
    }
    app.error.map_or(Ok(()), Err)
}
// #endregion
