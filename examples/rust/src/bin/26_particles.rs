//! Step 26 - Compute and graphics in one frame: particles.
//!
//! A million particles live in ONE storage buffer on the GPU. Every frame a
//! compute shader moves them (shaders/particles.slang, `simulate`), and the
//! graphics pipeline draws the same buffer as a vertex buffer, one point per
//! particle. The CPU never touches the particles: it only records
//!
//!   barrier (last frame's draw + dispatch -> this dispatch)
//!   dispatch simulate          (compute writes the buffer)
//!   barrier (dispatch -> vertex input)
//!   draw N points              (vertex input reads the buffer)
//!
//! into one command buffer on the graphics queue (its family has COMPUTE
//! too). Step 17's frame loop with 2 frames in flight; `--count N` sets the
//! number of particles, `--seconds S` closes the window after S seconds;
//! lines that depend on timing start with `[timing]`. Steps 01-17 are
//! repeated, reduced to what this step needs.

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

// #region cli
/// Command line: `--count N` particles (default 2^20), `--seconds S`.
struct Args {
    count: u32,
    seconds: Option<f64>,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        const USAGE: &str = "usage: 26_particles [--count N] [--seconds S]";
        let mut args = Args {
            count: 1 << 20,
            seconds: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--count" => args.count = it.next().ok_or(USAGE)?.parse()?,
                "--seconds" => args.seconds = Some(it.next().ok_or(USAGE)?.parse()?),
                _ => return Err(USAGE.into()),
            }
        }
        if args.count == 0 || args.count > 1 << 24 {
            return Err("--count must be 1-16777216".into());
        }
        Ok(args)
    }
}
// #endregion

// #region particle
/// One particle as shaders/particles.slang declares it (std430: pos at 0,
/// vel at 8, stride 16). The host never reads or writes particles; the
/// struct only gives the vertex input its stride and offsets.
#[repr(C)]
#[derive(Clone, Copy)]
struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
}
const _: () = assert!(size_of::<Particle>() == 16);

/// The push constant block of particles.slang: 16 bytes, read by the
/// compute shaders (dt, time, count) and the vertex shader (aspect).
#[derive(Clone, Copy)]
struct Params {
    dt: f32,
    time: f32,
    count: u32,
    aspect: f32,
}

impl Params {
    /// The block's bytes in member order (all members are 4 bytes, no
    /// padding); `.as_flattened()` turns them into the &[u8] that
    /// vkCmdPushConstants takes.
    fn bytes(&self) -> [[u8; 4]; 4] {
        [
            self.dt.to_ne_bytes(),
            self.time.to_ne_bytes(),
            self.count.to_ne_bytes(),
            self.aspect.to_ne_bytes(),
        ]
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

/// Step 25's reflection: the LocalSize x of compute entry point `entry`
/// (the kernels are 1D), from OpEntryPoint + OpExecutionMode LocalSize.
fn local_size_x(words: &[u32], entry: &str) -> Result<u32, Box<dyn Error>> {
    const OP_ENTRY_POINT: u32 = 15;
    const OP_EXECUTION_MODE: u32 = 16;
    const LOCAL_SIZE: u32 = 17;
    let mut function = None;
    let mut sizes = Vec::new();
    let mut pos = 5;
    while let Some(&first) = words.get(pos) {
        let count = usize::try_from(first >> 16)?;
        let ins = words
            .get(pos..pos + count)
            .filter(|_| count > 0)
            .ok_or("malformed SPIR-V instruction")?;
        match (first & 0xffff, ins) {
            (OP_ENTRY_POINT, [_, _model, id, name @ ..]) => {
                let bytes: Vec<u8> = name
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .take_while(|&b| b != 0)
                    .collect();
                if bytes == entry.as_bytes() {
                    function = Some(*id);
                }
            }
            (OP_EXECUTION_MODE, [_, id, LOCAL_SIZE, x, 1, 1]) => sizes.push((*id, *x)),
            _ => {}
        }
        pos += count;
    }
    let function = function.ok_or(format!("no entry point {entry}"))?;
    sizes
        .iter()
        .find(|(id, _)| *id == function)
        .map(|&(_, x)| x)
        .ok_or_else(|| format!("{entry} has no 1D LocalSize").into())
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

/// What the CPU touches while recording one frame (step 10). No per-frame
/// particle data: the particles are GPU-only, see `frame-barriers`.
struct Frame {
    command_buffer: vk::CommandBuffer,
    /// Signaled by vkAcquireNextImageKHR, waited on by this slot's submit.
    image_available: vk::Semaphore,
    /// Signaled when this slot's submission completes.
    in_flight: vk::Fence,
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
    compute_layout: vk::PipelineLayout,
    simulate: vk::Pipeline,
    graphics_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    particles: GpuBuffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    queue: vk::Queue,
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    command_pool: vk::CommandPool,
    frames: Vec<Frame>,
    frame_index: usize,
    /// One per swapchain image, indexed by the acquired image index.
    render_finished: Vec<vk::Semaphore>,
    count: u32,
    groups: u32,
    start: Instant,
    last: Instant,
    window: Window,
}

impl Vk {
    /// Steps 01-17 plus the particle buffer, two compute pipelines and the
    /// point-drawing graphics pipeline.
    fn new(event_loop: &ActiveEventLoop, count: u32) -> Result<Self, Box<dyn Error>> {
        // Window from step 06: 800x600 physical pixels, fixed size.
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("Vulkan from Zero - 26 particles")
                .with_inner_size(PhysicalSize::new(800, 600))
                .with_resizable(false),
        )?;
        let size = window.inner_size();
        println!("Window: {}x{} pixels", size.width, size.height);

        // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vulkan from Zero - 26 particles")
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
        // Steps 03 + 06, and the family must ALSO support COMPUTE: the
        // dispatch and the draw go into the same command buffer, so into the
        // same queue. A family with GRAPHICS usually has COMPUTE too, but the
        // spec does not promise it for every family, so it is checked.
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
                let flags = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
                if f.queue_flags.contains(flags) && present {
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
            "No Vulkan 1.4 GPU with dynamic rendering + synchronization2 + VK_KHR_swapchain and a graphics + compute family can present to this window",
        )?;
        // SAFETY: `physical_device` was enumerated from this instance.
        let chosen = unsafe { instance.get_physical_device_properties(physical_device) };
        println!(
            "\nPicked: {}",
            chosen.device_name_as_c_str()?.to_string_lossy()
        );
        println!("Graphics + compute + present family: {family}");
        // #endregion

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

        // The SPIR-V is compiled from shaders/particles.slang by `bun run
        // sync:shaders` and embedded at build time (step 08 explains `read_spv`).
        const PARTICLES_SPV: &[u8] = include_bytes!("../../../../shaders/spv/particles.spv");
        let spirv = ash::util::read_spv(&mut std::io::Cursor::new(PARTICLES_SPV))?;
        let group_size = local_size_x(&spirv, "simulate")?;
        if local_size_x(&spirv, "init")? != group_size {
            return Err("init and simulate must have the same workgroup size".into());
        }
        let groups = count.div_ceil(group_size);
        if groups > chosen.limits.max_compute_work_group_count[0] {
            return Err(format!("{groups} workgroups exceed maxComputeWorkGroupCount[0]").into());
        }
        println!(
            "\nSPIR-V: {} bytes (particles.spv), entry points init, simulate, vertexMain, fragmentMain; LocalSize {group_size} x 1 x 1",
            PARTICLES_SPV.len()
        );
        let module_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
        // SAFETY: `spirv` is valid SPIR-V (checked by spirv-val in sync:shaders).
        let shader_module = unsafe { device.create_shader_module(&module_info, None)? };

        // #region particle-buffer
        // GPU-only data in DEVICE_LOCAL memory. Two usages: STORAGE_BUFFER
        // lets the compute shader bind it through a descriptor,
        // VERTEX_BUFFER lets vkCmdBindVertexBuffers bind the very same
        // buffer for the draw. No staging and no CPU copy: the `init` shader
        // writes the starting state.
        // SAFETY: `physical_device` was enumerated from this instance.
        let mem = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let particle_bytes = vk::DeviceSize::from(count) * size_of::<Particle>() as u64;
        let particles = create_buffer(
            &device,
            &mem,
            particle_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        println!(
            "Particles: {count} x {} bytes = {particle_bytes} bytes, STORAGE_BUFFER | VERTEX_BUFFER, memory type {} (DEVICE_LOCAL)",
            size_of::<Particle>(),
            particles.memory_type
        );
        // #endregion

        // #region compute-pipelines
        // Set 0, binding 0 = the particle buffer as a STORAGE_BUFFER, for the
        // compute stage. One push constant range of 16 bytes for the compute
        // stage. Two pipelines share this layout: `init` and `simulate`.
        let set_bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&set_bindings);
        // SAFETY: plain create call on a live device.
        let set_layout = unsafe { device.create_descriptor_set_layout(&set_layout_info, None)? };
        let push_size = u32::try_from(size_of::<Params>())?;
        let compute_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(push_size)];
        let set_layouts = [set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&compute_ranges);
        // SAFETY: `set_layout` belongs to this device.
        let compute_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };
        let compute_infos = [c"init", c"simulate"].map(|name| {
            vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(shader_module)
                        .name(name),
                )
                .layout(compute_layout)
        });
        // SAFETY: all create-info data outlives the call; no pipeline cache.
        let compute = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &compute_infos, None)
                .map_err(|(_, e)| e)?
        };
        let [init, simulate] = compute[..] else {
            return Err("vkCreateComputePipelines returned the wrong number of pipelines".into());
        };
        println!(
            "Compute pipelines: init, simulate (set 0: binding 0 = STORAGE_BUFFER; {push_size} push constant bytes)"
        );
        // #endregion

        // #region graphics-pipeline
        // The draw reads the particle buffer through VERTEX INPUT, not a
        // descriptor: binding 0, stride 16, pos at offset 0, vel at offset 8,
        // one vertex per particle, topology POINT_LIST. The vertex shader
        // only needs `aspect` from the push block; its range covers the whole
        // 16-byte block for simplicity.
        let bindings = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(u32::try_from(size_of::<Particle>())?)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let attributes = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(u32::try_from(std::mem::offset_of!(Particle, pos))?),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(u32::try_from(std::mem::offset_of!(Particle, vel))?),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::POINT_LIST);
        let graphics_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .size(push_size)];
        let graphics_layout_info =
            vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&graphics_ranges);
        // SAFETY: plain create call on a live device.
        let graphics_layout =
            unsafe { device.create_pipeline_layout(&graphics_layout_info, None)? };
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
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        // Points have no winding: no culling.
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
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
            .layout(graphics_layout)
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
        println!(
            "Graphics pipeline: POINT_LIST, vertex input binding 0 = the particle buffer (stride {})",
            size_of::<Particle>()
        );
        // #endregion

        // One descriptor set: binding 0 -> the particle buffer (step 17).
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        // SAFETY: plain create call on a live device.
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };
        let alloc_sets = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        // SAFETY: the pool and the layout belong to this device; the pool has room.
        let descriptor_set = *unsafe { device.allocate_descriptor_sets(&alloc_sets)? }
            .first()
            .ok_or("no descriptor set")?;
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(particles.buffer)
            .range(vk::WHOLE_SIZE);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info));
        // SAFETY: the set, the buffer and the info outlive the call; no command buffer uses the set yet.
        unsafe { device.update_descriptor_sets(&[write], &[]) };

        // Steps 09-10: the queue, the pool, per-frame buffers and sync objects.
        // SAFETY: one queue of `family` was requested in `queue_infos`.
        let queue = unsafe { device.get_device_queue(family, 0) };
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(family);
        // SAFETY: plain create call on a live device.
        let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };
        let frames_u32 = u32::try_from(FRAMES_IN_FLIGHT)?;
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(frames_u32);
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

        // #region init
        // The starting state, written by the GPU: one dispatch of `init` in
        // a command buffer of its own, submitted once and waited for with a
        // fence before the first frame. The first frame's barrier
        // (`frame-barriers`) has the compute stage in its source scope, so it
        // also orders the first `simulate` after these writes.
        let init_cmd = *command_buffers.first().ok_or("no command buffer")?;
        let params = Params {
            dt: 0.0,
            time: 0.0,
            count,
            aspect: extent.height as f32 / extent.width as f32,
        };
        // SAFETY: `init_cmd` is a fresh command buffer of this device; every
        // handle stays alive until the fence wait below returns.
        unsafe {
            device.begin_command_buffer(
                init_cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_bind_pipeline(init_cmd, vk::PipelineBindPoint::COMPUTE, init);
            device.cmd_bind_descriptor_sets(
                init_cmd,
                vk::PipelineBindPoint::COMPUTE,
                compute_layout,
                0,
                &[descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                init_cmd,
                compute_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                params.bytes().as_flattened(),
            );
            device.cmd_dispatch(init_cmd, groups, 1, 1);
            device.end_command_buffer(init_cmd)?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(init_cmd);
            let submit =
                vk::SubmitInfo2::default().command_buffer_infos(std::slice::from_ref(&cmd_info));
            let waited = device
                .queue_submit2(queue, &[submit], fence)
                .and_then(|()| device.wait_for_fences(&[fence], true, u64::MAX));
            device.destroy_fence(fence, None);
            waited?;
            // `init` is not needed again; the command buffer is reused by frame slot 0.
            device.destroy_pipeline(init, None);
        }
        println!(
            "\ninit: {groups} workgroups of {group_size} wrote {count} particles (one submit, fence waited)"
        );
        // #endregion

        println!(
            "Per frame: barrier -> dispatch simulate {groups} x 1 x 1 -> barrier COMPUTE_SHADER/SHADER_STORAGE_WRITE -> VERTEX_ATTRIBUTE_INPUT/VERTEX_ATTRIBUTE_READ -> draw {count} points"
        );
        println!(
            "{FRAMES_IN_FLIGHT} frames in flight, ONE particle buffer: the GPU orders the frames with barriers, the CPU never touches it"
        );

        window.request_redraw();
        println!("Rendering. Close the window to exit.");
        let now = Instant::now();
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
            compute_layout,
            simulate,
            graphics_layout,
            pipeline,
            particles,
            descriptor_pool,
            descriptor_set,
            queue,
            images,
            extent,
            command_pool,
            frames,
            frame_index: 0,
            render_finished,
            count,
            groups,
            start: now,
            last: now,
            window,
        })
    }

    /// Simulates, renders and presents one frame (step 10's loop). Returns
    /// `Ok(false)` when the swapchain is out of date (recreation is step 11).
    fn draw_frame(&mut self) -> Result<bool, Box<dyn Error>> {
        let device = &self.device;
        let slot = self.frame_index % self.frames.len();
        let frame = self.frames.get(slot).ok_or("frame slot out of range")?;
        let cmd = frame.command_buffer;

        // The last submission that used this slot must be finished before its
        // command buffer and semaphore are reused (step 10).
        // SAFETY: the fence belongs to this device.
        unsafe { device.wait_for_fences(&[frame.in_flight], true, u64::MAX)? };

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

        // The time step: real time since the last frame, at most 1/60 s so
        // a stalled frame (window dragged, breakpoint) does not make the
        // explicit Euler step explode.
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f32().min(1.0 / 60.0);
        self.last = now;
        let params = Params {
            dt,
            time: now.duration_since(self.start).as_secs_f32(),
            count: self.count,
            aspect: self.extent.height as f32 / self.extent.width as f32,
        };

        // SAFETY: the buffer is not pending (fence) and its pool allows reset.
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        use vk::AccessFlags2 as A;
        use vk::PipelineStageFlags2 as S;
        // #region frame-barriers
        // Two frames are in flight, but there is ONE particle buffer. Frame
        // n's dispatch must not start while frame n-1 still uses the buffer:
        // - WAR: frame n-1's draw READS the particles as vertex input; this
        //   dispatch overwrites them. Execution order is all it needs: the
        //   dispatch waits for earlier VERTEX_ATTRIBUTE_INPUT work (access
        //   NONE on both sides - a read leaves nothing to make available).
        // - RAW: this dispatch reads what frame n-1's dispatch WROTE: the
        //   writes must be made available and visible to compute reads and
        //   writes.
        // A pipeline barrier's first scope is every command submitted
        // earlier to this queue, including other command buffers and
        // submissions - which is why it reaches back into frame n-1 (and,
        // for the first frame, into `init`). Frames in flight still overlap
        // on the CPU and in the parts of the GPU pipeline the barriers leave
        // free; per-frame COPIES (step 17's uniform buffers) are only needed
        // for data the CPU writes while the GPU may still read the old one.
        let before_dispatch = [
            vk::MemoryBarrier2::default()
                .src_stage_mask(S::VERTEX_ATTRIBUTE_INPUT)
                .src_access_mask(A::NONE)
                .dst_stage_mask(S::COMPUTE_SHADER)
                .dst_access_mask(A::NONE),
            vk::MemoryBarrier2::default()
                .src_stage_mask(S::COMPUTE_SHADER)
                .src_access_mask(A::SHADER_STORAGE_WRITE)
                .dst_stage_mask(S::COMPUTE_SHADER)
                .dst_access_mask(A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE),
        ];
        // RAW within this frame: the draw's vertex input reads what the
        // dispatch wrote.
        let before_draw = vk::MemoryBarrier2::default()
            .src_stage_mask(S::COMPUTE_SHADER)
            .src_access_mask(A::SHADER_STORAGE_WRITE)
            .dst_stage_mask(S::VERTEX_ATTRIBUTE_INPUT)
            .dst_access_mask(A::VERTEX_ATTRIBUTE_READ);
        // #endregion

        // #region simulate
        // Dispatches cannot be recorded inside a render pass instance, so the
        // compute work comes before vkCmdBeginRendering.
        // SAFETY: `cmd` is recording; all handles are alive.
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&before_dispatch),
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.simulate);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.compute_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                params.bytes().as_flattened(),
            );
            device.cmd_dispatch(cmd, self.groups, 1, 1);
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&before_draw)),
            );
        }
        // #endregion

        let color_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        // Step 09's layout transition to COLOR_ATTACHMENT_OPTIMAL, chained to
        // the acquire semaphore's wait stage.
        let to_attachment = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(S::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(A::NONE)
            .dst_stage_mask(S::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(A::COLOR_ATTACHMENT_WRITE)
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
                    float32: [0.0, 0.0, 0.0, 1.0],
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

        // #region draw
        // The particle buffer bound as vertex buffer 0: one point per particle.
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
            device.cmd_push_constants(
                cmd,
                self.graphics_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                params.bytes().as_flattened(),
            );
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.particles.buffer], &[0]);
            device.cmd_draw(cmd, self.count, 1, 0, 0);
            device.cmd_end_rendering(cmd);
        }
        // #endregion

        // Step 09's transition to PRESENT_SRC_KHR, chained to render_finished.
        let to_present = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(S::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(A::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(S::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(A::NONE)
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

        // Step 10's submit: wait image_available at COLOR_ATTACHMENT_OUTPUT
        // (the dispatch does not wait for the swapchain image), signal
        // render_finished, then the slot's fence.
        let wait = vk::SemaphoreSubmitInfo::default()
            .semaphore(frame.image_available)
            .stage_mask(S::COLOR_ATTACHMENT_OUTPUT);
        let signal = vk::SemaphoreSubmitInfo::default()
            .semaphore(render_finished)
            .stage_mask(S::COLOR_ATTACHMENT_OUTPUT);
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
        // Wait until the GPU is idle: the last frames may still simulate and
        // draw the particles. Destroying the pool frees its set.
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
            self.device.destroy_buffer(self.particles.buffer, None);
            self.device.free_memory(self.particles.memory, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.graphics_layout, None);
            self.device.destroy_pipeline(self.simulate, None);
            self.device
                .destroy_pipeline_layout(self.compute_layout, None);
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
        match Vk::new(event_loop, self.args.count) {
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
