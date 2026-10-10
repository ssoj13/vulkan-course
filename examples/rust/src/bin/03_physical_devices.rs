//! Step 03 - Find the GPUs and pick one.
//!
//! A `VkPhysicalDevice` is a read-only handle to one GPU (or software
//! rasteriser) that the driver exposes. We cannot "create" it - we can only
//! ask what it is and what it can do, then choose. Steps 01-02 (instance and
//! validation) are repeated unchanged at the top of `main`.

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
        .application_name(c"Vulkan from Zero - 03 physical devices")
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

    // #region enumerate
    // Classic Vulkan "two-call" query (count, then fill) - ash does both for us.
    // SAFETY: `instance` is alive.
    let devices = unsafe { instance.enumerate_physical_devices()? };
    if devices.is_empty() {
        return Err("No Vulkan-capable GPU found (is the driver installed?)".into());
    }
    println!("Found {} GPU(s)", devices.len());
    // #endregion

    for (i, &pd) in devices.iter().enumerate() {
        // #region properties
        // SAFETY: `pd` was enumerated from this instance.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        println!(
            "\nGPU {i}: {}",
            props.device_name_as_c_str()?.to_string_lossy()
        );
        println!("  type:       {:?}", props.device_type);
        // apiVersion is a packed integer: variant | major | minor | patch.
        println!(
            "  api:        {}.{}.{}",
            vk::api_version_major(props.api_version),
            vk::api_version_minor(props.api_version),
            vk::api_version_patch(props.api_version),
        );
        // The driver version encoding is vendor specific (NVIDIA, AMD and
        // Intel all pack it differently), so we print it raw.
        println!("  driver:     {:#x}", props.driver_version);
        // PCI vendor id: 0x10DE NVIDIA, 0x1002 AMD, 0x8086 Intel.
        println!("  vendor id:  {:#06x}", props.vendor_id);
        // #endregion

        // #region features
        // Vulkan 1.0 features are queried directly; newer sets are *chained*
        // onto `PhysicalDeviceFeatures2` through `p_next`: the driver walks the
        // chain and fills in every struct it recognises. Each core version
        // since 1.1 has its own struct; we chain the 1.3 and 1.4 ones. A struct
        // of a newer core version may only be queried on a device that supports
        // that version, so the 1.4 struct is chained only when api >= 1.4.
        let has14 = props.api_version >= vk::API_VERSION_1_4;
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut f14 = vk::PhysicalDeviceVulkan14Features::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push(&mut f13);
        if has14 {
            f2 = f2.push(&mut f14);
        }
        // SAFETY: `pd` came from this instance, and the chain structs outlive the call.
        unsafe { instance.get_physical_device_features2(pd, &mut f2) };
        let f10 = f2.features;
        println!("  features:");
        println!(
            "    sampler_anisotropy (1.0): {}",
            f10.sampler_anisotropy != 0
        );
        println!(
            "    dynamic_rendering  (1.3): {}",
            f13.dynamic_rendering != 0
        );
        println!(
            "    synchronization2   (1.3): {}",
            f13.synchronization2 != 0
        );
        if has14 {
            println!("    push_descriptor    (1.4): {}", f14.push_descriptor != 0);
            println!("    maintenance5       (1.4): {}", f14.maintenance5 != 0);
        } else {
            println!("    push_descriptor    (1.4): n/a (device below 1.4)");
            println!("    maintenance5       (1.4): n/a (device below 1.4)");
        }
        // #endregion

        // #region limits
        // Limits are hard numbers your program must respect; exceeding one is
        // an error that validation usually catches.
        let limits = props.limits;
        println!("  limits:");
        println!(
            "    max_image_dimension_2d:               {}",
            limits.max_image_dimension2_d
        );
        println!(
            "    max_push_constants_size:              {} bytes",
            limits.max_push_constants_size
        );
        println!(
            "    min_uniform_buffer_offset_alignment:  {} bytes",
            limits.min_uniform_buffer_offset_alignment
        );
        // Nanoseconds per GPU timestamp tick - needed to convert timer queries to time.
        println!(
            "    timestamp_period:                     {} ns/tick",
            limits.timestamp_period
        );
        // #endregion
    }

    // #region pick
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
    // #endregion

    // Physical devices are owned by the instance - nothing to destroy for them.
    // SAFETY: both objects are alive, nothing else uses them, and they are destroyed once.
    unsafe {
        debug_utils.destroy_debug_utils_messenger(messenger, None);
        instance.destroy_instance(None);
    }
    Ok(())
}
