//! Step 06 - A window and its Vulkan surface.
//!
//! Vulkan itself knows nothing about windows: it renders into images. Showing
//! an image on screen is the job of the *window system integration* (WSI)
//! extensions. `VkSurfaceKHR` is Vulkan's handle for "this OS window": created
//! from the native window (an HWND on Windows) by a platform extension
//! (VK_KHR_win32_surface), and queried through the platform-neutral
//! VK_KHR_surface. Both are *instance* extensions: the surface belongs to the
//! instance, not to a GPU, so we can ask every GPU whether it can present to it.
//!
//! winit owns the window and the event loop. Since winit 0.30 the program does
//! not loop itself: it hands an `ApplicationHandler` to `run_app`, and winit
//! calls it back. Windows may only be created inside those callbacks
//! (`resumed`), so all Vulkan setup lives in `Vk::new`, called from `resumed`.
//! Steps 01-05 are repeated inside it, adapted where the surface changes them.

use std::error::Error;
use std::ffi::{CStr, c_void};

use ash::{
    Entry,
    ext::debug_utils,
    khr::surface,
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

/// Everything this step creates. Field order matters only for `_window`: it is
/// declared last so it is dropped after `destroy` has destroyed the surface
/// that refers to it.
struct Vk {
    /// The loaded Vulkan loader DLL; must outlive every Vulkan object.
    _entry: Entry,
    instance: ash::Instance,
    debug_utils: debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
    /// Function table of VK_KHR_surface (destroy and query surfaces).
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,
    /// Never read: owning it keeps the OS window (and its HWND) alive.
    _window: Window,
}

impl Vk {
    /// Creates the window, the instance with the surface extensions, the
    /// surface, and reports what the picked GPU can do with that surface.
    fn new(event_loop: &ActiveEventLoop) -> Result<Self, Box<dyn Error>> {
        // #region create-window
        // Size in physical pixels, so the swapchain extent printed later is
        // exactly 800x600 whatever the display scaling. Not resizable: a new
        // size invalidates the swapchain, and recreating it is module 7.
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("Vulkan from Zero - 06 window and surface")
                .with_inner_size(PhysicalSize::new(800, 600))
                .with_resizable(false),
        )?;
        let size = window.inner_size();
        println!("Window: {}x{} pixels", size.width, size.height);
        // #endregion

        // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Vulkan from Zero - 06 window and surface")
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

        // #region surface-extensions
        // Which instance extensions does a surface need on this OS? ash-window
        // answers from the display handle: VK_KHR_surface (the portable part)
        // plus the platform one (VK_KHR_win32_surface on Windows). They must be
        // enabled when the instance is created - a surface cannot be added later.
        let display = event_loop.display_handle()?.as_raw();
        let surface_exts = ash_window::enumerate_required_extensions(display)?;
        let names = surface_exts
            .iter()
            // SAFETY: ash-window returns pointers to static NUL-terminated names.
            .map(|&p| unsafe { CStr::from_ptr(p) }.to_string_lossy())
            .collect::<Vec<_>>();
        println!("Surface extensions: {}", names.join(", "));
        let mut ext_ptrs = surface_exts.to_vec();
        ext_ptrs.push(debug_utils::NAME.as_ptr());
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

        // #region create-surface
        // ash-window picks the platform path from the handles. On Windows it fills
        // VkWin32SurfaceCreateInfoKHR { hinstance, hwnd } with the window's module
        // and HWND and calls vkCreateWin32SurfaceKHR - the same call GLFW makes
        // in the C++ version. The OS window must stay alive as long as the surface.
        let surface_factory = ash_window::SurfaceFactory::new(&entry, &instance, display)?;
        // SAFETY: the window handle is valid, and `Vk` destroys the surface
        // before the window and the instance (see `destroy`).
        let surface =
            unsafe { surface_factory.create_surface(window.window_handle()?.as_raw(), None)? };
        // The portable VK_KHR_surface functions (queries, destroy) live in their own table.
        let surface_loader = surface::Instance::load(&entry, &instance);
        println!("Surface created");
        // #endregion

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

        // Step 03's choice (Vulkan 1.4, dynamic rendering + synchronization2,
        // discrete first), with one more requirement: a queue family that can
        // both draw and present to OUR surface.
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
            // #region present-family
            // Presenting is a queue operation (vkQueuePresentKHR), so "can this
            // GPU show images in this window?" is asked per queue family AND per
            // surface: a GPU without a display connection, or a compute-only
            // family, answers no. We want one family that does both graphics and
            // present, so one queue can render and then present (on desktop GPUs
            // the graphics family always can).
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
            // #endregion
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
        let (_, physical_device, family) =
            best.ok_or("No Vulkan 1.4 GPU with dynamic rendering + synchronization2 can present to this window")?;
        // SAFETY: `physical_device` was enumerated from this instance.
        let chosen = unsafe { instance.get_physical_device_properties(physical_device) };
        println!(
            "\nPicked: {}",
            chosen.device_name_as_c_str()?.to_string_lossy()
        );

        // #region family-table
        // The whole answer for the picked GPU, family by family.
        // SAFETY: `physical_device` was enumerated from this instance.
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        for (i, f) in families.iter().enumerate() {
            let index = u32::try_from(i)?;
            // SAFETY: as above.
            let present = unsafe {
                surface_loader.get_physical_device_surface_support(
                    physical_device,
                    index,
                    surface,
                )?
            };
            let yes_no = |b: bool| if b { "yes" } else { "no" };
            println!(
                "  family {i}: graphics {}, present {}",
                yes_no(f.queue_flags.contains(vk::QueueFlags::GRAPHICS)),
                yes_no(present)
            );
        }
        println!("Graphics + present family: {family}");
        // #endregion

        // #region surface-capabilities
        // What the window system allows for images shown in this surface: how
        // many images a swapchain may have, their size, how they may be rotated
        // and blended with the desktop, and what else they may be used for.
        // SAFETY: `physical_device` and `surface` belong to this instance.
        let caps = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
        };
        println!("\nSurface capabilities:");
        // max_image_count 0 means "no upper limit".
        println!(
            "  image count: min {}, max {}",
            caps.min_image_count, caps.max_image_count
        );
        // 0xFFFFFFFF x 0xFFFFFFFF would mean "the swapchain decides the window size"
        // (Wayland); on Windows it is the window's client area.
        println!(
            "  current extent: {}x{}",
            caps.current_extent.width, caps.current_extent.height
        );
        println!(
            "  extent range: {}x{} .. {}x{}",
            caps.min_image_extent.width,
            caps.min_image_extent.height,
            caps.max_image_extent.width,
            caps.max_image_extent.height
        );
        println!("  max array layers: {}", caps.max_image_array_layers);
        println!("  supported transforms: {:?}", caps.supported_transforms);
        println!("  current transform: {:?}", caps.current_transform);
        println!("  composite alpha: {:?}", caps.supported_composite_alpha);
        println!("  image usage: {:?}", caps.supported_usage_flags);
        // #endregion

        // #region surface-formats
        // Pixel format + color space pairs the window system can display. The
        // color space says how the presentation engine interprets the values
        // (SRGB_NONLINEAR = an ordinary sRGB monitor); the format says how they
        // are stored. *_SRGB formats encode linear shader output to sRGB on write.
        // SAFETY: as above.
        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?
        };
        println!("Surface formats ({}):", formats.len());
        for f in &formats {
            println!("  {:?} + {:?}", f.format, f.color_space);
        }
        // #endregion

        // #region present-modes
        // How finished images replace each other on screen: FIFO = a queue
        // drained once per vertical blank (vsync, always supported), MAILBOX =
        // vsync but a newer image replaces the waiting one, IMMEDIATE = no wait
        // (tearing), FIFO_RELAXED = FIFO that tears when a frame is late.
        // SAFETY: as above.
        let modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
        };
        let names = modes.iter().map(|m| format!("{m:?}")).collect::<Vec<_>>();
        println!("Present modes ({}): {}", modes.len(), names.join(", "));
        // #endregion

        println!("\nClose the window to exit.");
        Ok(Self {
            _entry: entry,
            instance,
            debug_utils,
            messenger,
            surface_loader,
            surface,
            _window: window,
        })
    }

    /// Destroys the Vulkan objects in reverse order of creation.
    fn destroy(self) {
        // #region cleanup
        // The surface is a child of the instance, so it goes first. The window
        // (`self._window`) is dropped at the end of this function, after the
        // surface that pointed at it.
        // SAFETY: nothing else uses these objects, and each is destroyed exactly once.
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils
                .destroy_debug_utils_messenger(self.messenger, None);
            self.instance.destroy_instance(None);
        }
        // #endregion
    }
}

// #region event-loop
/// winit drives the program: it calls these methods when something happens.
/// `vk` is `None` until the first `resumed` (the moment windows may be created).
#[derive(Default)]
struct App {
    vk: Option<Vk>,
    error: Option<Box<dyn Error>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Desktop platforms resume once; mobile ones may resume again after a
        // suspend, and we keep what we already built.
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
        // Nothing is drawn yet: the only event we care about is the close button.
        if matches!(event, WindowEvent::CloseRequested) {
            event_loop.exit();
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // `run_app` returns once a handler calls `exit()`. Then we clean up here,
    // outside winit's callbacks, and report an error from setup if there was one.
    let event_loop = EventLoop::new()?;
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    if let Some(vk) = app.vk.take() {
        println!("Window closed");
        vk.destroy();
    }
    app.error.map_or(Ok(()), Err)
}
// #endregion
