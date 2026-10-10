//! Step 04 - Queue families.
//!
//! The GPU does not run your commands directly: you submit them to a *queue*.
//! Queues come in *families* with different abilities (graphics, compute,
//! transfer...). Like Maya's dependency graph evaluating on separate threads,
//! independent queues can work in parallel. Steps 01-03 are repeated unchanged
//! above the new code.

use std::ffi::{CStr, c_void};

use ash::{
    Entry,
    ext::debug_utils,
    vk::{self, TaggedStructure as _},
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
    let entry = unsafe { Entry::load()? };

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"Vulkan from Zero - 04 queue families")
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
    let layer_ptrs = [validation_layer.as_ptr()];
    let ext_ptrs = [debug_utils::NAME.as_ptr()];
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
    // Classic Vulkan "two-call" query (count, then fill) - ash does both for us.
    // SAFETY: `instance` is alive.
    let devices = unsafe { instance.enumerate_physical_devices()? };
    if devices.is_empty() {
        return Err("No Vulkan-capable GPU found (is the driver installed?)".into());
    }
    println!("Found {} GPU(s)", devices.len());

    // Step 03 printed every property, feature and limit; here we only list names.
    for (i, &pd) in devices.iter().enumerate() {
        // SAFETY: `pd` was enumerated from this instance.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        println!(
            "GPU {i}: {}",
            props.device_name_as_c_str()?.to_string_lossy()
        );
    }

    // Eligible = a Vulkan 1.4 device with dynamic rendering and synchronization2
    // (the two features the book relies on). Among eligible GPUs we prefer a
    // discrete card (fastest, own VRAM) over integrated (shares system RAM),
    // then virtual, then CPU software rasterisers. Highest score wins; on a
    // tie the first one found stays.
    let mut best: Option<(u32, vk::PhysicalDevice)> = None;
    for &pd in &devices {
        // SAFETY: `pd` was enumerated from this instance.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        // Check the version before asking about 1.3 features: querying a core
        // feature struct of a newer version on an older device is invalid
        // (spec: "Extending Physical Device Core Functionality").
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
        let score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 10,
            _ => 1,
        };
        // Strictly greater: on a tie the first GPU found stays.
        if best.is_none_or(|(s, _)| score > s) {
            best = Some((score, pd));
        }
    }
    let (_, physical_device) =
        best.ok_or("No GPU supports Vulkan 1.4 with dynamic rendering + synchronization2")?;
    // SAFETY: `physical_device` was enumerated from this instance.
    let chosen = unsafe { instance.get_physical_device_properties(physical_device) };
    println!(
        "\nPicked: {}",
        chosen.device_name_as_c_str()?.to_string_lossy()
    );

    // #region families
    // Every queue belongs to a family. The GPU reports how many queues each
    // family has and what they can do; we have to look, because the layout
    // differs per vendor (NVIDIA, AMD and Intel expose very different sets).
    // SAFETY: `physical_device` was enumerated from this instance.
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    // #endregion

    // #region print-families
    // Flags are a bitmask; a family may have several capabilities at once.
    let known_flags = [
        (vk::QueueFlags::GRAPHICS, "GRAPHICS"),
        (vk::QueueFlags::COMPUTE, "COMPUTE"),
        (vk::QueueFlags::TRANSFER, "TRANSFER"),
        (vk::QueueFlags::SPARSE_BINDING, "SPARSE_BINDING"),
        (vk::QueueFlags::PROTECTED, "PROTECTED"),
        (vk::QueueFlags::VIDEO_DECODE_KHR, "VIDEO_DECODE"),
        (vk::QueueFlags::VIDEO_ENCODE_KHR, "VIDEO_ENCODE"),
        (vk::QueueFlags::OPTICAL_FLOW_NV, "OPTICAL_FLOW"),
    ];
    println!("\n{} queue families:", families.len());
    for (index, family) in families.iter().enumerate() {
        let flags: Vec<&str> = known_flags
            .iter()
            .filter(|(flag, _)| family.queue_flags.contains(*flag))
            .map(|(_, name)| *name)
            .collect();
        // timestamp_valid_bits == 0 means this family cannot write GPU timestamps.
        println!(
            "  family {index}: {} queue(s), timestamp bits {}, [{}]",
            family.queue_count,
            family.timestamp_valid_bits,
            flags.join(" | ")
        );
    }
    // #endregion

    // #region find-graphics
    // Pick families by capability. Our main family needs GRAPHICS. Only TRANSFER
    // is implied by GRAPHICS/COMPUTE (the spec makes that bit optional for such
    // families); COMPUTE on a graphics family is not implied for every family -
    // the spec only guarantees that if graphics is supported at all, at least
    // one family supports graphics AND compute. On desktop GPUs the first
    // graphics family is that one, which is all we need for rendering.
    let graphics_family = families
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .ok_or("No graphics queue family")?;
    println!("\nGraphics family: {graphics_family}");

    // Dedicated queues map to separate DMA / compute hardware, so uploads and
    // async compute can overlap with rendering. Not every GPU has them.
    let transfer_only = families.iter().position(|f| {
        f.queue_flags.contains(vk::QueueFlags::TRANSFER)
            && !f
                .queue_flags
                .intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
    });
    let compute_only = families.iter().position(|f| {
        f.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
    });
    match transfer_only {
        Some(i) => println!("Dedicated transfer family: {i}"),
        None => println!("No dedicated transfer family"),
    }
    match compute_only {
        Some(i) => println!("Async compute family: {i}"),
        None => println!("No async compute family"),
    }
    // #endregion

    // Physical devices are owned by the instance - nothing to destroy for them.
    // SAFETY: both objects are alive, nothing else uses them, and they are destroyed once.
    unsafe {
        debug_utils.destroy_debug_utils_messenger(messenger, None);
        instance.destroy_instance(None);
    }
    Ok(())
}
