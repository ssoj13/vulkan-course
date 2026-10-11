//! Step 19 - Buffer device address: GPU pointers instead of descriptors.
//!
//! 64 hexagons, one draw: each instance reads its own record (position,
//! size, color, spin) from a buffer of 64 records. The shader does not find
//! that buffer through a descriptor. The program asks Vulkan for the buffer's
//! DEVICE ADDRESS - a 64-bit GPU virtual address, like a C pointer - and
//! passes the number in a push constant. In the shader it is `Instance*`.
//!
//!   feature bufferDeviceAddress (core 1.2) -> buffer usage SHADER_DEVICE_ADDRESS
//!   memory allocated with VkMemoryAllocateFlagsInfo DEVICE_ADDRESS
//!   vkGetBufferDeviceAddress -> u64 -> push constants -> instances[instance index]
//!
//! No set layout, no pool, no set: the pipeline layout has only a 16-byte
//! push constant range. `--seconds S` closes the window after S seconds;
//! lines that depend on timing start with `[timing]`, and lines with GPU
//! addresses (which differ between runs and programs) start with `[addr]`.
//! Steps 01-18 are repeated, reduced to what this step needs.

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

/// Step 14's vertex: 2 + 3 floats = 20 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    /// x, y before the per-instance transform.
    position: [f32; 2],
    /// Linear RGB.
    color: [f32; 3],
}
const _: () = assert!(size_of::<Vertex>() == 20);

/// Step 17's regular hexagon (radius 0.6).
#[rustfmt::skip] // one vertex per line reads like a table
const VERTICES: [Vertex; 7] = [
    Vertex { position: [0.0, 0.0],           color: [1.0, 1.0, 1.0] }, // center, white
    Vertex { position: [0.6, 0.0],           color: [1.0, 0.0, 0.0] }, // 0 deg, red
    Vertex { position: [0.3, 0.519_615_2],   color: [1.0, 1.0, 0.0] }, // 60 deg, yellow
    Vertex { position: [-0.3, 0.519_615_2],  color: [0.0, 1.0, 0.0] }, // 120 deg, green
    Vertex { position: [-0.6, 0.0],          color: [0.0, 1.0, 1.0] }, // 180 deg, cyan
    Vertex { position: [-0.3, -0.519_615_2], color: [0.0, 0.0, 1.0] }, // 240 deg, blue
    Vertex { position: [0.3, -0.519_615_2],  color: [1.0, 0.0, 1.0] }, // 300 deg, magenta
];
/// Step 14's indices: six triangles around the center.
const INDICES: [u16; 18] = [
    0, 1, 2, //
    0, 2, 3, //
    0, 3, 4, //
    0, 4, 5, //
    0, 5, 6, //
    0, 6, 1, //
];

// #region instance
/// One hexagon instance as the shader's `Instance` declares it: 32 bytes,
/// offsets 0 / 8 / 12 / 16 / 28 (shaders/device_address.slang; std430 and
/// Slang's natural pointer layout agree for this struct, and repr(C) gives
/// the same offsets because every field is f32 and none needs padding).
#[repr(C)]
#[derive(Clone, Copy)]
struct Instance {
    /// Center, in units where the window is 2 tall.
    offset: [f32; 2],
    /// Multiplies the mesh corners.
    scale: f32,
    /// Starting angle, radians.
    phase: f32,
    /// Multiplies the vertex colors.
    color: [f32; 3],
    /// Radians per second; the sign is the direction.
    spin: f32,
}
const _: () = assert!(size_of::<Instance>() == 32);
const _: () = assert!(std::mem::offset_of!(Instance, color) == 16);
const _: () = assert!(std::mem::offset_of!(Instance, spin) == 28);

/// Grid size: GRID x GRID instances.
const GRID: usize = 8;

/// An 8x8 grid of small hexagons, cell 0.25 wide, centered on the window;
/// colors shade from blue-ish (column 0, row 0) to white, neighbors spin in
/// opposite directions.
fn grid() -> Vec<Instance> {
    let last = (GRID - 1) as f32;
    let mut instances = Vec::with_capacity(GRID * GRID);
    for row in 0..GRID {
        for col in 0..GRID {
            let (r, c) = (row as f32, col as f32);
            instances.push(Instance {
                offset: [-0.875 + 0.25 * c, -0.875 + 0.25 * r],
                scale: 0.18,
                phase: 0.25 * (r * GRID as f32 + c),
                color: [0.4 + 0.6 * c / last, 0.4 + 0.6 * r / last, 1.0],
                spin: if (row + col) % 2 == 0 { 1.0 } else { -1.0 },
            });
        }
    }
    instances
}
// #endregion

// #region push-data
/// The push constant block of shaders/device_address.slang: 16 bytes. The
/// u64 comes first, so it is 8-byte aligned without padding (std430: an
/// 8-byte pointer is 8-aligned).
#[repr(C)]
#[derive(Clone, Copy)]
struct PushData {
    /// vkGetBufferDeviceAddress of the instance buffer = `Instance*` in the shader.
    instances: vk::DeviceAddress,
    /// Seconds since start.
    time: f32,
    /// Window height / width.
    aspect: f32,
}
// 8 + 4 + 4: no padding, so every byte is initialized (see `as_bytes`).
const _: () = assert!(size_of::<PushData>() == 16);

impl PushData {
    /// The struct's 16 bytes, for vkCmdPushConstants.
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: repr(C), only u64/f32 fields and no padding (size assert
        // above), so all bytes are initialized; the slice borrows `self`.
        unsafe {
            std::slice::from_raw_parts(std::ptr::from_ref(self).cast::<u8>(), size_of::<Self>())
        }
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

// #region create-buffer
/// Step 14's manual path with one addition: `alloc_flags` goes into
/// VkMemoryAllocateFlagsInfo. DEVICE_ADDRESS is required, not a hint: a
/// buffer with SHADER_DEVICE_ADDRESS usage may only be bound to memory
/// allocated with it (VUID-vkBindBufferMemory-bufferDeviceAddress-03339), and
/// the flag itself needs the bufferDeviceAddress feature
/// (VUID-VkMemoryAllocateInfo-flags-03331). Empty flags are a plain
/// allocation.
fn create_buffer(
    device: &ash::Device,
    props: &vk::PhysicalDeviceMemoryProperties,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    flags: vk::MemoryPropertyFlags,
    alloc_flags: vk::MemoryAllocateFlags,
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
    let mut flags_info = vk::MemoryAllocateFlagsInfo::default().flags(alloc_flags);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(memory_type)
        .push(&mut flags_info);
    // SAFETY: the type index comes from this device's memory properties; the
    // DEVICE_ADDRESS flag is only passed with bufferDeviceAddress enabled.
    let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    // SAFETY: the allocation has the required size and type; offset 0 is aligned.
    unsafe { device.bind_buffer_memory(buffer, memory, 0)? };
    Ok(GpuBuffer {
        buffer,
        memory,
        memory_type,
    })
}
// #endregion

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

/// What the CPU touches while recording one frame (step 10). The instance
/// buffer is written once and only read afterwards, so it needs no copy per
/// frame in flight.
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
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    vertex_buffer: GpuBuffer,
    index_buffer: GpuBuffer,
    instance_buffer: GpuBuffer,
    /// vkGetBufferDeviceAddress(instance_buffer).
    instance_address: vk::DeviceAddress,
    instance_count: u32,
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
    /// Steps 01-18, with the bufferDeviceAddress feature, the instance buffer
    /// and its address.
    fn new(event_loop: &ActiveEventLoop) -> Result<Self, Box<dyn Error>> {
        // Window from step 06: 800x600 physical pixels, fixed size.
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("Vulkan from Zero - 19 buffer device address")
                .with_inner_size(PhysicalSize::new(800, 600))
                .with_resizable(false),
        )?;
        let size = window.inner_size();
        println!("Window: {}x{} pixels", size.width, size.height);

        // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vulkan from Zero - 19 buffer device address")
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
        // graphics + present family, VK_KHR_swapchain, discrete first; plus
        // the 1.2 feature bufferDeviceAddress.
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
            if f13.dynamic_rendering == 0
                || f13.synchronization2 == 0
                || f12.buffer_device_address == 0
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
            "No Vulkan 1.4 GPU with dynamic rendering + synchronization2 + bufferDeviceAddress + VK_KHR_swapchain can present to this window",
        )?;
        // SAFETY: `physical_device` was enumerated from this instance.
        let chosen = unsafe { instance.get_physical_device_properties(physical_device) };
        println!(
            "\nPicked: {}",
            chosen.device_name_as_c_str()?.to_string_lossy()
        );
        println!("Graphics + present family: {family}");

        // #region device-features
        // Step 07's device plus bufferDeviceAddress from the Vulkan 1.2
        // feature struct. Without it, vkGetBufferDeviceAddress is invalid
        // (VUID-vkGetBufferDeviceAddress-bufferDeviceAddress-03324) and so is
        // the DEVICE_ADDRESS allocation flag
        // (VUID-VkMemoryAllocateInfo-flags-03331). SHADER_DEVICE_ADDRESS usage
        // alone is not tied to the feature, but is useless without it.
        let priorities = [1.0_f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let mut features12 =
            vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true);
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
        // #endregion
        let swapchain_loader = swapchain::Device::load(&instance, &device);
        println!(
            "VkDevice created with {} + feature bufferDeviceAddress",
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

        // The SPIR-V is compiled from shaders/device_address.slang by `bun run
        // sync:shaders` and embedded at build time (step 08 explains `read_spv`).
        const SPV: &[u8] = include_bytes!("../../../../shaders/spv/device_address.spv");
        let spirv = ash::util::read_spv(&mut std::io::Cursor::new(SPV))?;
        let module_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
        // SAFETY: `spirv` is valid SPIR-V (checked by spirv-val in sync:shaders).
        let shader_module = unsafe { device.create_shader_module(&module_info, None)? };
        println!(
            "\nSPIR-V: {} bytes (device_address.spv), entry points vertexMain, fragmentMain",
            SPV.len()
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

        // Step 14's vertex input: binding 0, stride 20, position + color.
        let bindings = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(u32::try_from(size_of::<Vertex>())?)
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

        // #region pipeline-layout
        // Step 18's push constant range, 16 bytes: the address + time + aspect.
        // No set layouts: this pipeline reads memory through no descriptor.
        let push_size = u32::try_from(size_of::<PushData>())?;
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(push_size)];
        let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges);
        // SAFETY: plain create call on a live device.
        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };
        println!(
            "Push constant range: stage VERTEX, offset 0, size {push_size} (8-byte address + time + aspect)"
        );
        println!("Pipeline layout: no set layouts, 1 push constant range");
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
        println!("Graphics pipeline created (vertexMain reads instances through the pointer)");

        // Step 14's mesh buffers: HOST_VISIBLE | HOST_COHERENT, written once.
        // SAFETY: `physical_device` was enumerated from this instance.
        let mem = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let host_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let vertex_buffer = create_buffer(
            &device,
            &mem,
            vk::DeviceSize::try_from(size_of_val(&VERTICES))?,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            host_flags,
            vk::MemoryAllocateFlags::empty(),
        )?;
        write_mapped(&device, vertex_buffer.memory, &VERTICES)?;
        let index_buffer = create_buffer(
            &device,
            &mem,
            vk::DeviceSize::try_from(size_of_val(&INDICES))?,
            vk::BufferUsageFlags::INDEX_BUFFER,
            host_flags,
            vk::MemoryAllocateFlags::empty(),
        )?;
        write_mapped(&device, index_buffer.memory, &INDICES)?;
        println!(
            "\nMesh: regular hexagon, {} vertices, {} indices (step 14's buffers, memory type {})",
            VERTICES.len(),
            INDICES.len(),
            vertex_buffer.memory_type
        );

        // #region instance-buffer
        // The 64 records, written once. Usage SHADER_DEVICE_ADDRESS is what
        // allows vkGetBufferDeviceAddress on it; no STORAGE_BUFFER usage is
        // needed, because no descriptor will ever point at it. The memory is
        // step 14's HOST_VISIBLE | HOST_COHERENT kind (the GPU reads it over
        // PCIe; step 15's staging would move it into VRAM, the address works
        // the same).
        let instances = grid();
        let instance_bytes = size_of_val(instances.as_slice());
        let instance_buffer = create_buffer(
            &device,
            &mem,
            vk::DeviceSize::try_from(instance_bytes)?,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            host_flags,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
        )?;
        write_mapped(&device, instance_buffer.memory, &instances)?;
        println!(
            "Instance struct: {} bytes (offset 0, scale {}, phase {}, color {}, spin {})",
            size_of::<Instance>(),
            std::mem::offset_of!(Instance, scale),
            std::mem::offset_of!(Instance, phase),
            std::mem::offset_of!(Instance, color),
            std::mem::offset_of!(Instance, spin)
        );
        println!(
            "Instance buffer: {} x {} = {instance_bytes} bytes, SHADER_DEVICE_ADDRESS, memory type {} ({}), allocated with DEVICE_ADDRESS",
            instances.len(),
            size_of::<Instance>(),
            instance_buffer.memory_type,
            memory_names(host_flags)
        );
        // #endregion

        // #region get-address
        // The buffer's address in the GPU's virtual address space. It stays
        // valid until the buffer is destroyed; buffer + offset = address +
        // offset, so instances[i] lies at address + i * 32, which is exactly
        // what the shader's pointer indexing computes.
        let address_info = vk::BufferDeviceAddressInfo::default().buffer(instance_buffer.buffer);
        // SAFETY: the buffer has SHADER_DEVICE_ADDRESS usage and is bound to
        // DEVICE_ADDRESS memory; bufferDeviceAddress is enabled.
        let instance_address = unsafe { device.get_buffer_device_address(&address_info) };
        if instance_address == 0 {
            return Err("vkGetBufferDeviceAddress returned 0".into());
        }
        let stride = size_of::<Instance>() as u64;
        let last = instances.len() as u64 - 1;
        println!("[addr] instance buffer device address: {instance_address:#x}");
        println!(
            "[addr] instances[1] at {:#x} (+{stride}), instances[{last}] at {:#x} (+{})",
            instance_address + stride,
            instance_address + last * stride,
            last * stride
        );
        // #endregion

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
        let instance_count = u32::try_from(instances.len())?;
        println!(
            "\n{FRAMES_IN_FLIGHT} frames in flight; each frame: {} indices x {instance_count} instances in one vkCmdDrawIndexed",
            INDICES.len()
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
            pipeline_layout,
            pipeline,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            instance_address,
            instance_count,
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

        // #region push-and-draw
        // The pointer travels like any other push constant: 8 bytes of the
        // block. Then ONE draw with 64 instances: SV_VulkanInstanceID runs
        // 0..63, and instance i reads instances[i] through the pointer. The
        // GPU memory model sees an ordinary read of the instance buffer; the
        // host wrote it before the first submit, which makes it visible.
        let push = PushData {
            instances: self.instance_address,
            time: self.start.elapsed().as_secs_f32(),
            aspect: self.extent.height as f32 / self.extent.width as f32,
        };
        let index_count = u32::try_from(INDICES.len())?;
        // SAFETY: `cmd` is recording inside a rendering compatible with the
        // pipeline; the push range matches the layout; the instance buffer
        // (and so its address) stays alive while the command buffer is pending.
        unsafe {
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                push.as_bytes(),
            );
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, self.index_buffer.buffer, 0, vk::IndexType::UINT16);
            device.cmd_draw_indexed(cmd, index_count, self.instance_count, 0, 0, 0);
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
        // Wait until the GPU is idle: pending frames still dereference the
        // address. After vkDestroyBuffer the address is dangling, like a freed
        // pointer in C - nothing stops a shader from using it, so the host
        // must not hand it out any more.
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
            for b in [
                &self.instance_buffer,
                &self.index_buffer,
                &self.vertex_buffer,
            ] {
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

/// Command line: `--seconds S` closes the window after S seconds.
fn parse_seconds() -> Result<Option<f64>, Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [] => Ok(None),
        [flag, value] if flag == "--seconds" => Ok(Some(value.parse()?)),
        _ => Err("usage: 19_buffer_device_address [--seconds S]".into()),
    }
}

/// winit drives the program (step 06): `vk` is built in the first `resumed`,
/// and every `RedrawRequested` renders one frame.
struct App {
    seconds: Option<f64>,
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
            WindowEvent::RedrawRequested => {
                let Some(vk) = self.vk.as_mut() else {
                    return;
                };
                match vk.draw_frame() {
                    Ok(true) => {
                        let done = self
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
    let seconds = parse_seconds()?;
    let event_loop = EventLoop::new()?;
    let mut app = App {
        seconds,
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
