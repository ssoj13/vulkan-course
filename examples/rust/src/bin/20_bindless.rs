//! Step 20 - Bindless: one descriptor array, indexed in the shader.
//!
//! Step 19's 64 hexagons again, but every instance's record lives in its OWN
//! storage buffer (think: one buffer per object or material). Binding a set
//! per object would mean a draw per object. Instead ONE set holds an ARRAY of
//! storage buffer descriptors, bound once, and the shader picks
//! `objects[instance index]` - the "bindless" pattern of modern renderers.
//! Descriptor indexing (core 1.2) makes this practical:
//!
//!   runtimeDescriptorArray            the shader declares `objects[]` without a size
//!   variable descriptor count         the set's real length is chosen at allocation
//!   partially bound                   unwritten elements are allowed if never used
//!   non-uniform indexing              the index may differ within one draw
//!   update after bind / unused while pending
//!                                     the set may be updated while frames that
//!                                     use it are still in flight
//!
//! Streaming demo: 63 buffers are written at start; at frame 120 the 64th is
//! written into the live set while the previous frame may still be in flight,
//! and from then on 64 instances are drawn. Textures and samplers in such
//! arrays come in module 10; here the array holds buffers only.
//! `--seconds S` closes the window after S seconds; lines that depend on
//! timing start with `[timing]`. Steps 01-19 are repeated, reduced to what
//! this step needs.

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

/// Step 19's instance record: 32 bytes, std430 offsets 0 / 8 / 12 / 16 / 28
/// (shaders/bindless.slang reads it from a StructuredBuffer).
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

/// Step 19's 8x8 grid of small hexagons.
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

/// The push constant block of shaders/bindless.slang: 8 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct PushData {
    /// Seconds since start.
    time: f32,
    /// Window height / width.
    aspect: f32,
}
// Two f32, no padding, so every byte is initialized (see `as_bytes`).
const _: () = assert!(size_of::<PushData>() == 8);

impl PushData {
    /// The struct's 8 bytes, for vkCmdPushConstants.
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: repr(C), only f32 fields and no padding (size assert
        // above), so all bytes are initialized; the slice borrows `self`.
        unsafe {
            std::slice::from_raw_parts(std::ptr::from_ref(self).cast::<u8>(), size_of::<Self>())
        }
    }
}

// #region sizes
/// Upper bound of the descriptor array in the set LAYOUT: the largest set any
/// allocation with this layout may have. Must not exceed the device's
/// update-after-bind limits (checked at start).
const CAPACITY: u32 = 4096;
/// Length of the ONE set we allocate (its variable descriptor count): room
/// for 256 buffers, of which only the first 64 are ever written.
const ALLOCATED: u32 = 256;
/// Descriptors written before the first frame; the 64th is streamed in later.
const INITIAL: u32 = 63;
/// The frame (counted from 0) before whose recording the 64th descriptor is written.
const STREAM_FRAME: usize = 120;
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

// #region write-descriptors
/// Writes storage buffer descriptors for `buffers` into consecutive elements
/// of binding 0, starting at array element `first`. One VkWriteDescriptorSet
/// can fill a run of array elements: descriptorCount = buffers.len().
fn write_descriptors(
    device: &ash::Device,
    set: vk::DescriptorSet,
    first: u32,
    buffers: &[GpuBuffer],
) {
    let infos: Vec<vk::DescriptorBufferInfo> = buffers
        .iter()
        .map(|b| {
            vk::DescriptorBufferInfo::default()
                .buffer(b.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)
        })
        .collect();
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .dst_array_element(first)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&infos);
    // SAFETY: the set and the buffers belong to this device; the binding has
    // UPDATE_AFTER_BIND | UPDATE_UNUSED_WHILE_PENDING | PARTIALLY_BOUND, and
    // the caller writes only elements no pending command buffer uses.
    unsafe { device.update_descriptor_sets(&[write], &[]) };
}
// #endregion

/// What the CPU touches while recording one frame (step 10). The bindless
/// set is shared by all frames: nothing per frame but commands and sync.
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
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    vertex_buffer: GpuBuffer,
    index_buffer: GpuBuffer,
    /// One storage buffer per instance: element i of the array = buffer i.
    object_buffers: Vec<GpuBuffer>,
    descriptor_pool: vk::DescriptorPool,
    /// THE set: binding 0 = storage buffer array, ALLOCATED elements.
    descriptor_set: vk::DescriptorSet,
    /// Instances drawn (= descriptors written so far).
    live_count: u32,
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
    /// Steps 01-19, with the descriptor indexing features, the bindless set
    /// layout, pool and set, and one storage buffer per instance.
    fn new(event_loop: &ActiveEventLoop) -> Result<Self, Box<dyn Error>> {
        // Window from step 06: 800x600 physical pixels, fixed size.
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("Vulkan from Zero - 20 bindless")
                .with_inner_size(PhysicalSize::new(800, 600))
                .with_resizable(false),
        )?;
        let size = window.inner_size();
        println!("Window: {}x{} pixels", size.width, size.height);

        // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vulkan from Zero - 20 bindless")
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
        // the six descriptor indexing features this step uses.
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
            let indexing = [
                f12.runtime_descriptor_array,
                f12.shader_storage_buffer_array_non_uniform_indexing,
                f12.descriptor_binding_partially_bound,
                f12.descriptor_binding_variable_descriptor_count,
                f12.descriptor_binding_storage_buffer_update_after_bind,
                f12.descriptor_binding_update_unused_while_pending,
            ];
            if f13.dynamic_rendering == 0
                || f13.synchronization2 == 0
                || indexing.contains(&vk::FALSE)
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
            "No Vulkan 1.4 GPU with dynamic rendering + synchronization2 + the descriptor indexing features + VK_KHR_swapchain can present to this window",
        )?;
        // SAFETY: `physical_device` was enumerated from this instance.
        let chosen = unsafe { instance.get_physical_device_properties(physical_device) };
        println!(
            "\nPicked: {}",
            chosen.device_name_as_c_str()?.to_string_lossy()
        );
        println!("Graphics + present family: {family}");

        // #region limits
        // Two sets of limits apply to descriptor arrays. The classic ones
        // (VkPhysicalDeviceLimits) count descriptors in sets WITHOUT
        // UPDATE_AFTER_BIND; the Vulkan 1.2 properties carry separate, often
        // larger limits for UPDATE_AFTER_BIND layouts (the spec lets the
        // classic per-stage limit be as low as 4). The ...Native flag
        // says whether non-uniform indexing costs nothing extra in hardware.
        let mut props12 = vk::PhysicalDeviceVulkan12Properties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut props12);
        // SAFETY: `physical_device` came from this instance; the chain outlives the call.
        unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
        let limits = chosen.limits;
        println!("Descriptor limits for storage buffers:");
        println!(
            "  maxPerStageDescriptorStorageBuffers                = {} (sets without UPDATE_AFTER_BIND)",
            limits.max_per_stage_descriptor_storage_buffers
        );
        println!(
            "  maxDescriptorSetStorageBuffers                     = {}",
            limits.max_descriptor_set_storage_buffers
        );
        println!(
            "  maxPerStageDescriptorUpdateAfterBindStorageBuffers = {} (UPDATE_AFTER_BIND sets)",
            props12.max_per_stage_descriptor_update_after_bind_storage_buffers
        );
        println!(
            "  maxDescriptorSetUpdateAfterBindStorageBuffers      = {}",
            props12.max_descriptor_set_update_after_bind_storage_buffers
        );
        println!(
            "  maxPerStageUpdateAfterBindResources                = {} (all resource types)",
            props12.max_per_stage_update_after_bind_resources
        );
        println!(
            "  maxUpdateAfterBindDescriptorsInAllPools            = {}",
            props12.max_update_after_bind_descriptors_in_all_pools
        );
        println!(
            "  shaderStorageBufferArrayNonUniformIndexingNative   = {}",
            props12.shader_storage_buffer_array_non_uniform_indexing_native != 0
        );
        // The layout gives the vertex stage CAPACITY storage buffers and
        // nothing else, so CAPACITY counts against the per-stage and per-set
        // storage buffer limits and the per-stage limit for all resources.
        let cap = props12
            .max_per_stage_descriptor_update_after_bind_storage_buffers
            .min(props12.max_descriptor_set_update_after_bind_storage_buffers)
            .min(props12.max_per_stage_update_after_bind_resources);
        if CAPACITY > cap {
            return Err(format!(
                "the layout's {CAPACITY} storage buffers exceed this device's update-after-bind limit {cap}"
            )
            .into());
        }
        // The pool holds ALLOCATED descriptors. All UPDATE_AFTER_BIND pools
        // together share maxUpdateAfterBindDescriptorsInAllPools; beyond it,
        // pool creation may fail.
        if ALLOCATED > props12.max_update_after_bind_descriptors_in_all_pools {
            return Err(format!(
                "the pool's {ALLOCATED} descriptors exceed maxUpdateAfterBindDescriptorsInAllPools {}",
                props12.max_update_after_bind_descriptors_in_all_pools
            )
            .into());
        }
        println!(
            "This program: layout capacity {CAPACITY}, one set of {ALLOCATED}, {} buffers written",
            INITIAL + 1
        );
        // #endregion

        // #region device-features
        // Step 07's device plus the descriptor indexing features of the
        // Vulkan 1.2 feature struct. Each one is checked by validation where
        // it is used, e.g. UPDATE_AFTER_BIND on a storage buffer binding
        // without descriptorBindingStorageBufferUpdateAfterBind is
        // VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingStorageBufferUpdateAfterBind-03008.
        let priorities = [1.0_f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .runtime_descriptor_array(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_update_unused_while_pending(true);
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
            "VkDevice created with {} + runtimeDescriptorArray, shaderStorageBufferArrayNonUniformIndexing,",
            swapchain::NAME.to_string_lossy()
        );
        println!("  descriptorBindingPartiallyBound, descriptorBindingVariableDescriptorCount,");
        println!(
            "  descriptorBindingStorageBufferUpdateAfterBind, descriptorBindingUpdateUnusedWhilePending"
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

        // The SPIR-V is compiled from shaders/bindless.slang by `bun run
        // sync:shaders` and embedded at build time (step 08 explains `read_spv`).
        const SPV: &[u8] = include_bytes!("../../../../shaders/spv/bindless.spv");
        let spirv = ash::util::read_spv(&mut std::io::Cursor::new(SPV))?;
        let module_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
        // SAFETY: `spirv` is valid SPIR-V (checked by spirv-val in sync:shaders).
        let shader_module = unsafe { device.create_shader_module(&module_info, None)? };
        println!(
            "\nSPIR-V: {} bytes (bindless.spv), entry points vertexMain, fragmentMain",
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

        // #region set-layout
        // Binding 0: up to CAPACITY storage buffers, vertex stage. The binding
        // flags (VkDescriptorSetLayoutBindingFlagsCreateInfo, one entry per
        // binding) make it bindless:
        //   VARIABLE_DESCRIPTOR_COUNT  CAPACITY is only an upper bound; each set
        //                              gets its own length at allocation (only
        //                              allowed on the highest-numbered binding)
        //   PARTIALLY_BOUND            elements never used by the shader may
        //                              stay unwritten
        //   UPDATE_AFTER_BIND          a write after the set is bound in a
        //                              recording command buffer does not
        //                              invalidate it (the UPDATE_AFTER_BIND
        //                              limits above apply to such layouts)
        //   UPDATE_UNUSED_WHILE_PENDING elements that no pending command buffer
        //                              uses may be written while it executes
        // A layout with an UPDATE_AFTER_BIND binding must itself carry
        // UPDATE_AFTER_BIND_POOL (VUID-VkDescriptorSetLayoutCreateInfo-flags-03000).
        let set_bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(CAPACITY)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];
        let binding_flags = [vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
            | vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING];
        let mut flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .bindings(&set_bindings)
            .push(&mut flags_info);
        // SAFETY: the features behind every flag are enabled on this device.
        let set_layout = unsafe { device.create_descriptor_set_layout(&set_layout_info, None)? };
        println!(
            "Descriptor set layout (UPDATE_AFTER_BIND_POOL): binding 0 = STORAGE_BUFFER x up to {CAPACITY}, stage VERTEX"
        );
        println!(
            "  binding flags: VARIABLE_DESCRIPTOR_COUNT | PARTIALLY_BOUND | UPDATE_AFTER_BIND | UPDATE_UNUSED_WHILE_PENDING"
        );
        // #endregion

        // Step 18's push constant range (time + aspect) and the one set layout.
        let push_size = u32::try_from(size_of::<PushData>())?;
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(push_size)];
        let set_layouts = [set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&ranges);
        // SAFETY: `set_layout` belongs to this device.
        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };
        println!(
            "Pipeline layout: 1 set layout (set 0), push constants {push_size} bytes (time + aspect)"
        );

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
        println!(
            "Graphics pipeline created (vertexMain reads objects[NonUniformResourceIndex(instance)])"
        );

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
            "\nMesh: regular hexagon, {} vertices, {} indices (step 14's buffers, memory type {})",
            VERTICES.len(),
            INDICES.len(),
            vertex_buffer.memory_type
        );

        // #region object-buffers
        // One small STORAGE_BUFFER per instance, each holding one record (a
        // stand-in for per-object data of different sizes and lifetimes). 64
        // separate allocations keep the example short; step 16 explains why a
        // real renderer sub-allocates them.
        let instances = grid();
        let mut object_buffers = Vec::with_capacity(instances.len());
        for record in &instances {
            let buffer = create_buffer(
                &device,
                &mem,
                vk::DeviceSize::try_from(size_of::<Instance>())?,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                host_flags,
            )?;
            write_mapped(&device, buffer.memory, std::slice::from_ref(record))?;
            object_buffers.push(buffer);
        }
        println!(
            "Object buffers: {} x {} bytes (one Instance each), STORAGE_BUFFER, memory type {} ({})",
            object_buffers.len(),
            size_of::<Instance>(),
            object_buffers.first().map_or(0, |b| b.memory_type),
            memory_names(host_flags)
        );
        // #endregion

        // #region pool-and-set
        // The pool must be created with UPDATE_AFTER_BIND to hold sets of an
        // UPDATE_AFTER_BIND_POOL layout; its size counts the descriptors we
        // actually allocate (ALLOCATED), not the layout's CAPACITY. The
        // variable count of the one set is passed at allocation.
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(ALLOCATED)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        // SAFETY: plain create call on a live device.
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };
        let counts = [ALLOCATED];
        let mut variable_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(&counts);
        let alloc_layouts = [set_layout];
        let alloc_sets = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&alloc_layouts)
            .push(&mut variable_info);
        // SAFETY: the pool and the layout belong to this device; the pool has room.
        let descriptor_set = *unsafe { device.allocate_descriptor_sets(&alloc_sets)? }
            .first()
            .ok_or("vkAllocateDescriptorSets returned no set")?;
        println!("Descriptor pool (UPDATE_AFTER_BIND): maxSets 1, STORAGE_BUFFER x {ALLOCATED}");
        println!("Allocated 1 set with variable descriptor count {ALLOCATED}");
        // #endregion

        // #region initial-writes
        // Elements 0..62 now, in one write. Elements 63..255 stay unwritten:
        // allowed because the binding is PARTIALLY_BOUND and the draws read
        // only indices below the number of written descriptors.
        let initial = usize::try_from(INITIAL)?;
        write_descriptors(
            &device,
            descriptor_set,
            0,
            object_buffers.get(..initial).ok_or("too few buffers")?,
        );
        println!(
            "vkUpdateDescriptorSets: elements 0..{} -> object buffers 0..{} ({INITIAL} descriptors in one write); {} elements unwritten",
            INITIAL - 1,
            INITIAL - 1,
            ALLOCATED - INITIAL
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
        println!(
            "\n{FRAMES_IN_FLIGHT} frames in flight; each frame binds the one set and draws {} indices x {INITIAL} instances (64 from frame {STREAM_FRAME})",
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
            set_layout,
            pipeline_layout,
            pipeline,
            vertex_buffer,
            index_buffer,
            object_buffers,
            descriptor_pool,
            descriptor_set,
            live_count: INITIAL,
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
        let slot = self.frame_index % self.frames.len();
        let frame = self.frames.get(slot).ok_or("frame slot out of range")?;
        let cmd = frame.command_buffer;

        // The last submission that used this slot must be finished before its
        // command buffer and semaphore are reused (step 10).
        // SAFETY: the fence belongs to this device.
        unsafe {
            self.device
                .wait_for_fences(&[frame.in_flight], true, u64::MAX)?;
        };

        // #region stream-in
        // Streaming a resource into the live set. The other slot's command
        // buffer may still be PENDING with this set bound, so only elements
        // it does not use may change: it draws `live_count` instances, so
        // element `live_count` is not dynamically used. UPDATE_UNUSED_WHILE_PENDING
        // is the flag whose definition allows exactly this (UPDATE_AFTER_BIND
        // covers the window between bind and submit, not a pending command
        // buffer). Note: VUID-vkUpdateDescriptorSets-None-03047 accepts EITHER
        // flag, so validation stayed silent when this program was tried without
        // UPDATE_UNUSED_WHILE_PENDING - the flag is here because of the spec's
        // definition, not because of a message. This slot's command buffer is
        // re-recorded below, so it sees the new element; its draw may read it.
        if self.frame_index == STREAM_FRAME {
            let first = usize::try_from(self.live_count)?;
            let streamed = self
                .object_buffers
                .get(first..=first)
                .ok_or("no buffer left to stream in")?;
            write_descriptors(&self.device, self.descriptor_set, self.live_count, streamed);
            println!(
                "Frame {STREAM_FRAME}: element {} written while the set is bound by a command buffer that may be pending; instances {} -> {}",
                self.live_count,
                self.live_count,
                self.live_count + 1
            );
            self.live_count += 1;
        }
        // #endregion

        let device = &self.device;
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

        // #region bind-and-draw
        // ONE set bind for all objects, then ONE draw of `live_count`
        // instances; instance i reads objects[i]. A renderer with thousands of
        // objects keeps exactly this shape: bind the big set once per frame
        // (or once per command buffer), pass indices, never rebind per object.
        let push = PushData {
            time: self.start.elapsed().as_secs_f32(),
            aspect: self.extent.height as f32 / self.extent.width as f32,
        };
        let index_count = u32::try_from(INDICES.len())?;
        // SAFETY: `cmd` is recording inside a rendering compatible with the
        // pipeline; elements 0..live_count are written and their buffers
        // stay alive while the command buffer is pending.
        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                push.as_bytes(),
            );
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, self.index_buffer.buffer, 0, vk::IndexType::UINT16);
            device.cmd_draw_indexed(cmd, index_count, self.live_count, 0, 0, 0);
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
        // Wait until the GPU is idle, then destroy in reverse order: the pool
        // (which frees the set), the 64 object buffers, the rest.
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
                .object_buffers
                .iter()
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
        _ => Err("usage: 20_bindless [--seconds S]".into()),
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
