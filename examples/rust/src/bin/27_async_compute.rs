//! Step 27 - Async compute: the simulation on its own queue.
//!
//! Step 26 put the dispatch and the draw into one command buffer: the GPU
//! runs them one after the other. Here the simulation moves to a queue of a
//! family WITHOUT graphics (family 2 on the RTX 3080 Ti, 8 queues), so the
//! GPU may run "compute step n+1" while "graphics frame n" draws:
//!
//!   compute queue:  [step 1] [step 2]  [step 3]  [step 4] ...
//!   graphics queue:          [frame 1] [frame 2] [frame 3] ...
//!
//! Frame n draws state n while step n+1 writes state n+1, so the states need
//! TWO buffers (ping-pong): step n writes P[n % 2] and reads P[(n-1) % 2].
//! Two timeline semaphores order them: `computed` (step n done = n) and
//! `rendered` (frame n done = n). The buffers are shared by two queue
//! families with VK_SHARING_MODE_CONCURRENT. GPU timestamps on both queues
//! measure whether the work really overlaps; `--same-queue` submits the
//! compute work to the graphics queue instead, for comparison.
//! `--count N` particles, `--seconds S`; lines that depend on timing start
//! with `[timing]`. Steps 01-26 are repeated, reduced to what this step needs.

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
/// and the GPU busy; from here on it is fixed. Here it applies to both
/// queues: up to 2 graphics and 2 compute submissions are pending.
const FRAMES_IN_FLIGHT: u64 = 2;

// #region cli
/// Command line: `--count N` particles (default 2^20), `--seconds S`,
/// `--same-queue`.
struct Args {
    count: u32,
    seconds: Option<f64>,
    same_queue: bool,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        const USAGE: &str = "usage: 27_async_compute [--count N] [--seconds S] [--same-queue]";
        let mut args = Args {
            count: 1 << 20,
            seconds: None,
            same_queue: false,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--count" => args.count = it.next().ok_or(USAGE)?.parse()?,
                "--seconds" => args.seconds = Some(it.next().ok_or(USAGE)?.parse()?),
                "--same-queue" => args.same_queue = true,
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

/// One particle as shaders/particles.slang declares it (step 26).
#[repr(C)]
#[derive(Clone, Copy)]
struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
}
const _: () = assert!(size_of::<Particle>() == 16);

/// The push constant block of particles.slang (step 26).
#[derive(Clone, Copy)]
struct Params {
    dt: f32,
    time: f32,
    count: u32,
    aspect: f32,
}

impl Params {
    /// The block's bytes in member order; `.as_flattened()` gives the &[u8].
    fn bytes(&self) -> [[u8; 4]; 4] {
        [
            self.dt.to_ne_bytes(),
            self.time.to_ne_bytes(),
            self.count.to_ne_bytes(),
            self.aspect.to_ne_bytes(),
        ]
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

/// Step 25's reflection: the LocalSize x of the 1D compute entry point `entry`.
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

// #region timestamps
/// Two timestamps per submission: start and end, on both queues.
const TIMESTAMPS: u32 = 2;

/// GPU time of one graphics frame and one compute step, accumulated over a
/// second and over the whole run.
#[derive(Default)]
struct GpuTimes {
    /// Nanoseconds per tick (timestampPeriod).
    period_ns: f64,
    /// The valid bits common to both families' timestamps (1-64).
    bits: u32,
    second: u64,
    /// (frames, graphics us, compute us, overlap us) of the current second.
    current: (u64, f64, f64, f64),
    /// The same sums over the whole run.
    total: (u64, f64, f64, f64),
}

impl GpuTimes {
    /// Adds graphics frame n [g0, g1] and compute step n+1 [c0, c1]. All
    /// device timestamps are in one time domain (VK_TIME_DOMAIN_DEVICE_KHR),
    /// so values from the two queues can be compared: the overlap is the
    /// length of the intersection of the two intervals. Only the low `bits`
    /// of a timestamp are meaningful and the counter wraps there, so every
    /// value is taken as a signed tick offset from g0, computed modulo
    /// 2^bits (right while the four lie less than 2^(bits-1) ticks apart).
    fn add(&mut self, [g0, g1]: [u64; 2], [c0, c1]: [u64; 2]) {
        let shift = 64 - self.bits;
        let from_g0 = |t: u64| ((t.wrapping_sub(g0) << shift) as i64) >> shift;
        let us = |ticks: i64| ticks as f64 * self.period_ns / 1000.0;
        let (g1, c0, c1) = (from_g0(g1), from_g0(c0), from_g0(c1));
        let graphics = us(g1);
        let compute = us(c1 - c0);
        let overlap = us((g1.min(c1) - c0.max(0)).max(0));
        for t in [&mut self.current, &mut self.total] {
            t.0 += 1;
            t.1 += graphics;
            t.2 += compute;
            t.3 += overlap;
        }
    }

    /// Prints one `[timing]` line for `label` from `sums`.
    fn print(label: &str, (n, graphics, compute, overlap): (u64, f64, f64, f64)) {
        if n == 0 {
            return;
        }
        let n = n as f64;
        println!(
            "[timing] {label}: {n} frames, GPU time per frame: graphics {:.1} us, compute {:.1} us, overlapping {:.1} us ({:.0}% of the compute time)",
            graphics / n,
            compute / n,
            overlap / n,
            100.0 * overlap / compute.max(f64::MIN_POSITIVE)
        );
    }

    /// Prints the current second once it is over.
    fn tick(&mut self, second: u64) {
        if second > self.second {
            Self::print(&format!("{}s", self.second + 1), self.current);
            self.current = (0, 0.0, 0.0, 0.0);
            self.second = second;
        }
    }
}
// #endregion

/// One frame slot (step 10), now with a command buffer on each queue.
struct Frame {
    graphics_cmd: vk::CommandBuffer,
    compute_cmd: vk::CommandBuffer,
    /// Signaled by vkAcquireNextImageKHR, waited on by this slot's graphics submit.
    image_available: vk::Semaphore,
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
    advance: vk::Pipeline,
    graphics_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    /// P[0], P[1]: step n writes P[n % 2].
    particles: [GpuBuffer; 2],
    descriptor_pool: vk::DescriptorPool,
    /// Set i: binding 0 (state) = P[i], binding 1 (previous) = P[1 - i].
    sets: [vk::DescriptorSet; 2],
    graphics_queue: vk::Queue,
    compute_queue: vk::Queue,
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    graphics_pool: vk::CommandPool,
    compute_pool: vk::CommandPool,
    frames: Vec<Frame>,
    /// One per swapchain image, indexed by the acquired image index.
    render_finished: Vec<vk::Semaphore>,
    /// Timeline: n once compute step n has completed (0 = the init state).
    computed: vk::Semaphore,
    /// Timeline: n once graphics frame n has completed.
    rendered: vk::Semaphore,
    query_pool: vk::QueryPool,
    gpu: GpuTimes,
    /// Graphics frame n - F's timestamps, waiting for compute step n - F + 1's.
    pending_graphics: Option<[u64; 2]>,
    /// The last graphics frame submitted (0 = none yet).
    frame_number: u64,
    count: u32,
    groups: u32,
    start: Instant,
    last: Instant,
    window: Window,
}

impl Vk {
    /// Steps 01-26 with a second queue, two particle buffers and two
    /// timeline semaphores.
    fn new(event_loop: &ActiveEventLoop, args: &Args) -> Result<Self, Box<dyn Error>> {
        // Window from step 06: 800x600 physical pixels, fixed size.
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("Vulkan from Zero - 27 async compute")
                .with_inner_size(PhysicalSize::new(800, 600))
                .with_resizable(false),
        )?;
        let size = window.inner_size();
        println!("Window: {}x{} pixels", size.width, size.height);

        // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vulkan from Zero - 27 async compute")
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

        // Steps 03 + 06 + 12: Vulkan 1.4, dynamic rendering, synchronization2,
        // timelineSemaphore, a graphics + present family, VK_KHR_swapchain.
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
        let (_, physical_device, graphics_family) = best.ok_or(
            "No Vulkan 1.4 GPU with timelineSemaphore + dynamic rendering + synchronization2 + VK_KHR_swapchain can present to this window",
        )?;
        // SAFETY: `physical_device` was enumerated from this instance.
        let chosen = unsafe { instance.get_physical_device_properties(physical_device) };
        println!(
            "\nPicked: {}",
            chosen.device_name_as_c_str()?.to_string_lossy()
        );
        println!("Graphics + present family: {graphics_family}");

        // #region compute-family
        // A family with COMPUTE but without GRAPHICS is the "async compute"
        // family: its queues feed the GPU's compute work through a separate
        // hardware queue, which the GPU may schedule next to graphics work.
        // The spec promises nothing about concurrency; whether the two really
        // overlap is up to the hardware and the driver - hence the
        // timestamps. `--same-queue` (or a GPU without such a family) puts
        // the compute submissions on the graphics queue itself; then the
        // graphics family must support COMPUTE.
        // SAFETY: `physical_device` was enumerated from this instance.
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let dedicated = families.iter().position(|f| {
            f.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        });
        let graphics_flags = families
            .get(usize::try_from(graphics_family)?)
            .ok_or("queue family out of range")?
            .queue_flags;
        let compute_family = match (args.same_queue, dedicated) {
            (false, Some(i)) => {
                let i = u32::try_from(i)?;
                println!(
                    "Compute family: {i} (COMPUTE without GRAPHICS), its own queue: async compute"
                );
                i
            }
            (same_queue, _) => {
                if !graphics_flags.contains(vk::QueueFlags::COMPUTE) {
                    return Err("The graphics family has no COMPUTE: cannot run the compute work on its queue".into());
                }
                let why = if same_queue {
                    "--same-queue"
                } else {
                    "no family has COMPUTE without GRAPHICS"
                };
                println!("Compute work on the graphics queue (family {graphics_family}): {why}");
                graphics_family
            }
        };
        let separate = compute_family != graphics_family;
        // Timestamps on both queues are compared: both families need them;
        // the comparison uses the bits valid in both.
        let mut valid_bits = u32::MAX;
        for f in [graphics_family, compute_family] {
            let bits = families
                .get(usize::try_from(f)?)
                .ok_or("queue family out of range")?
                .timestamp_valid_bits;
            if bits == 0 {
                return Err(format!("queue family {f} has no timestamps").into());
            }
            valid_bits = valid_bits.min(bits);
        }
        // #endregion

        // #region device
        // One queue per family used. With --same-queue there is one queue,
        // and both kinds of submissions go to it.
        let priorities = [1.0_f32];
        let mut queue_infos = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(graphics_family)
                .queue_priorities(&priorities),
        ];
        if separate {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(compute_family)
                    .queue_priorities(&priorities),
            );
        }
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
        // SAFETY: one queue of each family in `queue_infos` was requested.
        let (graphics_queue, compute_queue) = unsafe {
            (
                device.get_device_queue(graphics_family, 0),
                device.get_device_queue(compute_family, 0),
            )
        };
        println!(
            "VkDevice created with {}, timelineSemaphore, {} queue(s)",
            swapchain::NAME.to_string_lossy(),
            queue_infos.len()
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

        // The SPIR-V is compiled from shaders/particles.slang (step 26).
        const PARTICLES_SPV: &[u8] = include_bytes!("../../../../shaders/spv/particles.spv");
        let spirv = ash::util::read_spv(&mut std::io::Cursor::new(PARTICLES_SPV))?;
        let group_size = local_size_x(&spirv, "advance")?;
        if local_size_x(&spirv, "init")? != group_size {
            return Err("init and advance must have the same workgroup size".into());
        }
        let count = args.count;
        let groups = count.div_ceil(group_size);
        if groups > chosen.limits.max_compute_work_group_count[0] {
            return Err(format!("{groups} workgroups exceed maxComputeWorkGroupCount[0]").into());
        }
        println!(
            "\nSPIR-V: {} bytes (particles.spv), entry points init, advance, vertexMain, fragmentMain; LocalSize {group_size} x 1 x 1",
            PARTICLES_SPV.len()
        );
        let module_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
        // SAFETY: `spirv` is valid SPIR-V (checked by spirv-val in sync:shaders).
        let shader_module = unsafe { device.create_shader_module(&module_info, None)? };

        // #region sharing
        // Two queue families touch the particle buffers: compute writes them,
        // graphics reads them as vertex input. An EXCLUSIVE buffer belongs to
        // one family at a time; reading it from another family without an
        // ownership transfer leaves "the contents of that resource ...
        // undefined for any read accesses performed by the second queue
        // family". Here every state is read by BOTH families at once (frame
        // n draws P[n % 2] while step n+1 reads it on the compute queue), so
        // EXCLUSIVE would need a release + acquire pair per family switch and
        // would serialize exactly the overlap this step is after.
        // CONCURRENT lists the families that may access the buffer and needs
        // no transfers; the spec warns it "may result in lower performance
        // access" (on some GPUs it disables compression - mostly a concern
        // for images). With one family (--same-queue) EXCLUSIVE is the only
        // choice: CONCURRENT needs at least two distinct families.
        let family_list = [graphics_family, compute_family];
        let (sharing, sharing_text) = if separate {
            (
                vk::SharingMode::CONCURRENT,
                format!(
                    "CONCURRENT (families {graphics_family} and {compute_family}): no ownership transfers"
                ),
            )
        } else {
            (
                vk::SharingMode::EXCLUSIVE,
                format!("EXCLUSIVE (one family, {graphics_family})"),
            )
        };
        // SAFETY: `physical_device` was enumerated from this instance.
        let mem = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let particle_bytes = vk::DeviceSize::from(count) * size_of::<Particle>() as u64;
        // Each descriptor binds a whole buffer, and a storage-buffer
        // descriptor may cover at most maxStorageBufferRange bytes
        // (VUID-VkWriteDescriptorSet-descriptorType-00333). The spec only
        // guarantees 2^27 = 128 MiB, less than 2^24 particles x 16 bytes.
        let max_range = chosen.limits.max_storage_buffer_range;
        if particle_bytes > vk::DeviceSize::from(max_range) {
            return Err(format!(
                "{count} particles = {particle_bytes} bytes exceed maxStorageBufferRange = {max_range}"
            )
            .into());
        }
        let mut particles = Vec::with_capacity(2);
        for _ in 0..2 {
            let mut info = vk::BufferCreateInfo::default()
                .size(particle_bytes)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER)
                .sharing_mode(sharing);
            if separate {
                info = info.queue_family_indices(&family_list);
            }
            // SAFETY: plain create call on a live device.
            let buffer = unsafe { device.create_buffer(&info, None)? };
            // SAFETY: `buffer` belongs to this device.
            let req = unsafe { device.get_buffer_memory_requirements(buffer) };
            let memory_type = find_type(
                &mem,
                req.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .ok_or("no DEVICE_LOCAL memory type for the particle buffers")?;
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(memory_type);
            // SAFETY: the type index comes from this device's memory properties.
            let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
            // SAFETY: the allocation has the required size and type; offset 0 is aligned.
            unsafe { device.bind_buffer_memory(buffer, memory, 0)? };
            particles.push(GpuBuffer {
                buffer,
                memory,
                memory_type,
            });
        }
        let [p0, p1] =
            <[GpuBuffer; 2]>::try_from(particles).map_err(|_| "expected two particle buffers")?;
        println!(
            "Particles: 2 buffers (ping-pong) x {count} x {} bytes, memory type {} (DEVICE_LOCAL), sharing mode {sharing_text}",
            size_of::<Particle>(),
            p0.memory_type
        );
        // #endregion

        // #region pipelines
        // Step 26's pipelines, with `advance` instead of `simulate`: binding
        // 0 = the state it writes, binding 1 = the previous state it reads.
        let set_bindings = [0, 1].map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
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
        let compute_infos = [c"init", c"advance"].map(|name| {
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
        let [init, advance] = compute[..] else {
            return Err("vkCreateComputePipelines returned the wrong number of pipelines".into());
        };

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
            "Pipelines: compute init + advance (set 0: binding 0 = state, binding 1 = previous), graphics POINT_LIST"
        );
        // #endregion

        // #region descriptor-sets
        // Two sets, one per direction of the ping-pong: set i writes P[i]
        // and reads P[1 - i]. Step n binds set n % 2. They never change.
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(4)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2)
            .pool_sizes(&pool_sizes);
        // SAFETY: plain create call on a live device.
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };
        let layouts = [set_layout; 2];
        let alloc_sets = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        // SAFETY: the pool and the layout belong to this device; the pool has room.
        let allocated = unsafe { device.allocate_descriptor_sets(&alloc_sets)? };
        let sets = <[vk::DescriptorSet; 2]>::try_from(allocated)
            .map_err(|_| "expected two descriptor sets")?;
        let buffers = [p0.buffer, p1.buffer];
        let infos = [(0, 0), (0, 1), (1, 1), (1, 0)].map(|(set, b): (usize, usize)| {
            let info = vk::DescriptorBufferInfo::default()
                .buffer(buffers[b])
                .range(vk::WHOLE_SIZE);
            (set, info)
        });
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, (set, info))| {
                vk::WriteDescriptorSet::default()
                    .dst_set(sets[*set])
                    .dst_binding(u32::from(i % 2 == 1))
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        // SAFETY: the sets, buffers and infos outlive the call; no command buffer uses the sets yet.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        println!(
            "Descriptor sets: set 0 writes P[0] and reads P[1], set 1 writes P[1] and reads P[0]"
        );
        // #endregion

        // Command pools: a command buffer is recorded for, and submitted to,
        // queues of its pool's family - so one pool per family.
        let pool_for = |family: u32| {
            let info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(family);
            // SAFETY: plain create call on a live device.
            unsafe { device.create_command_pool(&info, None) }
        };
        let graphics_pool = pool_for(graphics_family)?;
        let compute_pool = pool_for(compute_family)?;
        let frames_u32 = u32::try_from(FRAMES_IN_FLIGHT)?;
        let allocate = |pool: vk::CommandPool| {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(frames_u32);
            // SAFETY: `pool` belongs to this device.
            unsafe { device.allocate_command_buffers(&info) }
        };
        let graphics_cmds = allocate(graphics_pool)?;
        let compute_cmds = allocate(compute_pool)?;
        let binary_info = vk::SemaphoreCreateInfo::default();
        let mut frames = Vec::with_capacity(graphics_cmds.len());
        for (&graphics_cmd, &compute_cmd) in graphics_cmds.iter().zip(&compute_cmds) {
            // SAFETY: plain create call on a live device.
            let image_available = unsafe { device.create_semaphore(&binary_info, None)? };
            frames.push(Frame {
                graphics_cmd,
                compute_cmd,
                image_available,
            });
        }
        let mut render_finished = Vec::with_capacity(images.len());
        for _ in &images {
            // SAFETY: as above.
            render_finished.push(unsafe { device.create_semaphore(&binary_info, None)? });
        }

        // #region timelines
        // Two timeline semaphores (step 12), one per producer: each counts
        // the finished work of one queue. A submission waits on the OTHER
        // queue's counter; the host waits on both to reuse a frame slot.
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let timeline_info = vk::SemaphoreCreateInfo::default().push(&mut type_info);
        // SAFETY: plain create calls on a live device with timelineSemaphore enabled.
        let (computed, rendered) = unsafe {
            (
                device.create_semaphore(&timeline_info, None)?,
                device.create_semaphore(&timeline_info, None)?,
            )
        };
        println!(
            "Timeline semaphores: computed (step n done = n, the init state = 0), rendered (frame n done = n)"
        );
        // #endregion

        let pool_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2 * TIMESTAMPS * frames_u32);
        // SAFETY: plain create call on a live device.
        let query_pool = unsafe { device.create_query_pool(&pool_info, None)? };
        let gpu = GpuTimes {
            period_ns: f64::from(chosen.limits.timestamp_period),
            bits: valid_bits.min(64),
            ..GpuTimes::default()
        };

        // #region init
        // Step 26's init, on the compute queue: writes state 0 into P[0]
        // (set 0). The host waits for it with a fence; compute step 1, on the
        // same queue, is ordered after it by its own leading barrier.
        let aspect = extent.height as f32 / extent.width as f32;
        let init_cmd = *compute_cmds.first().ok_or("no command buffer")?;
        let params = Params {
            dt: 0.0,
            time: 0.0,
            count,
            aspect,
        };
        let set0 = sets[0];
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
                &[set0],
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
                .queue_submit2(compute_queue, &[submit], fence)
                .and_then(|()| device.wait_for_fences(&[fence], true, u64::MAX));
            // A fence must not be destroyed while a submission that signals it
            // is pending (VUID-vkDestroyFence-fence-01120): if the wait failed,
            // wait for the whole device first.
            if waited.is_err()
                && let Err(e) = device.device_wait_idle()
            {
                eprintln!("device_wait_idle failed: {e}");
            }
            device.destroy_fence(fence, None);
            waited?;
            device.destroy_pipeline(init, None);
        }
        println!("\ninit: {groups} workgroups of {group_size} wrote state 0 into P[0]");
        // #endregion

        println!(
            "Per frame n: graphics draws P[n % 2] after computed >= n; compute step n+1 writes P[(n+1) % 2] after rendered >= n-1"
        );
        let now = Instant::now();
        let mut vk = Self {
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
            advance,
            graphics_layout,
            pipeline,
            particles: [p0, p1],
            descriptor_pool,
            sets,
            graphics_queue,
            compute_queue,
            images,
            extent,
            graphics_pool,
            compute_pool,
            frames,
            render_finished,
            computed,
            rendered,
            query_pool,
            gpu,
            pending_graphics: None,
            frame_number: 0,
            count,
            groups,
            start: now,
            last: now,
            window,
        };
        // The pipeline starts one step ahead: step 1 before frame 1.
        vk.submit_compute(1)?;
        vk.window.request_redraw();
        println!("Rendering. Close the window to exit.");
        Ok(vk)
    }

    /// Reads the two timestamps of `first` (their submission has completed).
    fn read_timestamps(&self, first: u32) -> Result<[u64; 2], vk::Result> {
        let mut ticks = [0_u64; 2];
        // SAFETY: the queries were written by a completed submission.
        unsafe {
            self.device.get_query_pool_results(
                self.query_pool,
                first,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        Ok(ticks)
    }

    /// Host wait until `semaphore`'s counter reaches `value` (step 12).
    fn wait_timeline(&self, semaphore: vk::Semaphore, value: u64) -> Result<(), vk::Result> {
        let semaphores = [semaphore];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: the semaphore is a timeline semaphore of this device.
        unsafe { self.device.wait_semaphores(&info, u64::MAX) }
    }

    // #region submit-compute
    /// Records and submits compute step `n`: P[n % 2] = advance(P[(n-1) % 2]).
    fn submit_compute(&mut self, n: u64) -> Result<(), Box<dyn Error>> {
        let slot = usize::try_from(n % FRAMES_IN_FLIGHT)?;
        let cmd = self
            .frames
            .get(slot)
            .ok_or("frame slot out of range")?
            .compute_cmd;
        // Queries 2 * TIMESTAMPS * F.. hold the compute timestamps.
        let first_query =
            TIMESTAMPS * u32::try_from(FRAMES_IN_FLIGHT)? + TIMESTAMPS * u32::try_from(slot)?;
        // The slot's previous step (n - F) must be done before its command
        // buffer and queries are reused.
        if n > FRAMES_IN_FLIGHT {
            self.wait_timeline(self.computed, n - FRAMES_IN_FLIGHT)?;
            let ticks = self.read_timestamps(first_query)?;
            // Paired with graphics frame n - F - 1, read in draw_frame.
            if let Some(graphics) = self.pending_graphics.take() {
                self.gpu.add(graphics, ticks);
                self.gpu.tick(self.start.elapsed().as_secs());
            }
        }
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f32().min(1.0 / 60.0);
        self.last = now;
        let params = Params {
            dt,
            time: now.duration_since(self.start).as_secs_f32(),
            count: self.count,
            aspect: self.extent.height as f32 / self.extent.width as f32,
        };
        let set = *self
            .sets
            .get(usize::try_from(n % 2)?)
            .ok_or("descriptor set out of range")?;
        use vk::AccessFlags2 as A;
        use vk::PipelineStageFlags2 as S;
        // Same queue, previous step: RAW (this step reads what step n-1
        // wrote) and WAR (this step overwrites what step n-1 read).
        let after_previous_step = vk::MemoryBarrier2::default()
            .src_stage_mask(S::COMPUTE_SHADER)
            .src_access_mask(A::SHADER_STORAGE_WRITE)
            .dst_stage_mask(S::COMPUTE_SHADER)
            .dst_access_mask(A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE);
        let device = &self.device;
        // SAFETY: the command buffer is not pending (the wait above); every
        // handle is alive until the device is idle at exit.
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_reset_query_pool(cmd, self.query_pool, first_query, TIMESTAMPS);
            device.cmd_write_timestamp2(cmd, S::TOP_OF_PIPE, self.query_pool, first_query);
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .memory_barriers(std::slice::from_ref(&after_previous_step)),
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.advance);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_layout,
                0,
                &[set],
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
            device.cmd_write_timestamp2(cmd, S::COMPUTE_SHADER, self.query_pool, first_query + 1);
            device.end_command_buffer(cmd)?;
        }
        // Cross-queue: step n overwrites P[n % 2], which graphics frame n-2
        // drew - wait until rendered >= n - 2 (WAR across queues: execution
        // order is enough). Signal computed = n when the step is done; the
        // signal makes the writes available, the graphics queue's wait makes
        // them visible to its vertex input (semaphores carry memory
        // dependencies, no barrier needed on either side). Both at
        // ALL_COMMANDS: the start timestamp (TOP_OF_PIPE) is inside the
        // wait's scope, so the interval measures the work, not the waiting.
        // A wait at a later stage would let TOP_OF_PIPE pass early and the
        // interval would include the wait.
        let waits: Vec<vk::SemaphoreSubmitInfo> = (n > 2)
            .then(|| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.rendered)
                    .value(n - 2)
                    .stage_mask(S::ALL_COMMANDS)
            })
            .into_iter()
            .collect();
        let signal = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.computed)
            .value(n)
            .stage_mask(S::ALL_COMMANDS);
        let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(std::slice::from_ref(&cmd_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal));
        // SAFETY: all handles belong to this device; n grows by one per call.
        unsafe { device.queue_submit2(self.compute_queue, &[submit], vk::Fence::null())? };
        Ok(())
    }
    // #endregion

    /// Graphics frame n, then compute step n + 1. Returns `Ok(false)` when the
    /// swapchain is out of date (recreation is step 11).
    fn draw_frame(&mut self) -> Result<bool, Box<dyn Error>> {
        let n = self.frame_number + 1;
        let slot = usize::try_from(n % FRAMES_IN_FLIGHT)?;
        let frame = self.frames.get(slot).ok_or("frame slot out of range")?;
        let (cmd, image_available) = (frame.graphics_cmd, frame.image_available);
        let first_query = TIMESTAMPS * u32::try_from(slot)?;

        // #region wait-slot
        // Frame n - F used this slot: its command buffer, image_available
        // semaphore and queries are free once rendered >= n - F.
        if n > FRAMES_IN_FLIGHT {
            self.wait_timeline(self.rendered, n - FRAMES_IN_FLIGHT)?;
            self.pending_graphics = Some(self.read_timestamps(first_query)?);
        }
        // #endregion

        let device = &self.device;
        // SAFETY: the swapchain and the semaphore belong to this device; the
        // semaphore is unsignaled (its last wait completed, see wait-slot).
        let acquired = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                image_available,
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
        let particles = self
            .particles
            .get(usize::try_from(n % 2)?)
            .ok_or("particle buffer out of range")?
            .buffer;
        let params = Params {
            dt: 0.0,
            time: self.start.elapsed().as_secs_f32(),
            count: self.count,
            aspect: self.extent.height as f32 / self.extent.width as f32,
        };

        use vk::AccessFlags2 as A;
        use vk::PipelineStageFlags2 as S;
        let color_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
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

        // #region record-graphics
        // Step 26's draw without the dispatch and without a buffer barrier:
        // the particles of state n arrive through the semaphore wait below.
        // SAFETY: the command buffer is not pending (wait-slot); all handles are alive.
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_reset_query_pool(cmd, self.query_pool, first_query, TIMESTAMPS);
            device.cmd_write_timestamp2(cmd, S::TOP_OF_PIPE, self.query_pool, first_query);
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
            device.cmd_bind_vertex_buffers(cmd, 0, &[particles], &[0]);
            device.cmd_draw(cmd, self.count, 1, 0, 0);
            device.cmd_end_rendering(cmd);
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&to_present)),
            );
            device.cmd_write_timestamp2(
                cmd,
                S::COLOR_ATTACHMENT_OUTPUT,
                self.query_pool,
                first_query + 1,
            );
            device.end_command_buffer(cmd)?;
        }
        // #endregion

        // #region submit-graphics
        // Waits: the swapchain image (binary, as in step 10) and state n
        // (computed >= n). Both at ALL_COMMANDS, so the start timestamp comes
        // after both waits and the graphics interval measures the work, not
        // the waiting (see submit-compute). Step 10 waits for the image at
        // COLOR_ATTACHMENT_OUTPUT only, which lets the vertex work start
        // before the image is there - but then this interval would include
        // the wait for the image. The layout transition (src
        // COLOR_ATTACHMENT_OUTPUT) still chains to the wait: ALL_COMMANDS
        // includes that stage. Signals: render_finished for the present and
        // rendered = n, which the CPU (wait-slot) and compute step n + 2
        // wait for.
        let waits = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(image_available)
                .stage_mask(S::ALL_COMMANDS),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.computed)
                .value(n)
                .stage_mask(S::ALL_COMMANDS),
        ];
        let signals = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(render_finished)
                .stage_mask(S::COLOR_ATTACHMENT_OUTPUT),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.rendered)
                .value(n)
                .stage_mask(S::ALL_COMMANDS),
        ];
        let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(std::slice::from_ref(&cmd_info))
            .signal_semaphore_infos(&signals);
        // SAFETY: all handles belong to this device; n grows by one per frame.
        unsafe { device.queue_submit2(self.graphics_queue, &[submit], vk::Fence::null())? };
        self.frame_number = n;

        let swapchains = [self.swapchain];
        let indices = [image_index];
        let present_waits = [render_finished];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&present_waits)
            .swapchains(&swapchains)
            .image_indices(&indices);
        // SAFETY: the image was acquired above and is in PRESENT_SRC_KHR after the submit.
        let presented = unsafe {
            self.swapchain_loader
                .queue_present(self.graphics_queue, &present_info)
        };
        match presented {
            Ok(_suboptimal) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(false),
            Err(e) => return Err(e.into()),
        }
        // #endregion

        // The next state, while this frame draws.
        self.submit_compute(n + 1)?;
        Ok(true)
    }

    /// Destroys the Vulkan objects in reverse order of creation.
    fn destroy(self) -> Result<(), vk::Result> {
        // #region cleanup
        // Both queues must be done; vkDeviceWaitIdle waits for all queues of
        // the device.
        // SAFETY: plain call on a live device.
        let waited = unsafe { self.device.device_wait_idle() };
        if let Err(e) = &waited {
            eprintln!("device_wait_idle failed: {e}");
        }
        // SAFETY: nothing else uses these objects, and each is destroyed exactly once.
        unsafe {
            self.device.destroy_query_pool(self.query_pool, None);
            self.device.destroy_semaphore(self.rendered, None);
            self.device.destroy_semaphore(self.computed, None);
            for &semaphore in &self.render_finished {
                self.device.destroy_semaphore(semaphore, None);
            }
            for frame in &self.frames {
                self.device.destroy_semaphore(frame.image_available, None);
            }
            self.device.destroy_command_pool(self.compute_pool, None);
            self.device.destroy_command_pool(self.graphics_pool, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            for p in &self.particles {
                self.device.destroy_buffer(p.buffer, None);
                self.device.free_memory(p.memory, None);
            }
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.graphics_layout, None);
            self.device.destroy_pipeline(self.advance, None);
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
            vk.frame_number,
            vk.frame_number as f64 / elapsed
        );
        GpuTimes::print("whole run", vk.gpu.total);
        vk.destroy()?;
    }
    app.error.map_or(Ok(()), Err)
}
