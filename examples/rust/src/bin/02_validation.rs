//! Step 02 - Validation layers and a debug messenger.
//!
//! Vulkan drivers do almost no error checking (for speed): a wrong call can
//! crash or silently misrender. *Layers* are optional shims that sit between
//! your program and the driver. `VK_LAYER_KHRONOS_validation` checks every call
//! against the spec and reports mistakes through a *debug messenger* callback.
//! Think of it as a strict code reviewer that you only run while developing.

use std::ffi::{CStr, c_void};

use ash::{
    Entry,
    ext::debug_utils,
    vk::{self, TaggedStructure as _},
};

// #region debug-callback
/// Called by the validation layer for every message it wants to report.
///
/// `extern "system"` gives the function the calling convention the C loader
/// expects. Returning `FALSE` means "do not abort the Vulkan call that caused
/// this message" - the only value the spec allows for application callbacks.
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
        .unwrap_or_default();
    eprintln!("[{level}][{kind}] {message}");
    vk::FALSE
}
// #endregion

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
    let entry = unsafe { Entry::load()? };

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"Vulkan Book - 02 validation")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"No Engine")
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_4);

    // #region layers-available
    // Layers are installed separately (the Vulkan SDK ships the validation
    // one), so a user's machine may not have it. Asking for a missing layer
    // makes create_instance fail with a vague error - so check first.
    let validation_layer = c"VK_LAYER_KHRONOS_validation";
    // SAFETY: plain query, no pointers involved.
    let layers = unsafe { entry.enumerate_instance_layer_properties()? };
    let has_validation = layers.iter().any(|l| {
        l.layer_name_as_c_str()
            .is_ok_and(|name| name == validation_layer)
    });
    if !has_validation {
        return Err(format!(
            "{} not found. Install the Vulkan SDK (https://vulkan.lunarg.com).",
            validation_layer.to_string_lossy()
        )
        .into());
    }
    println!("Validation layer found ({} layers installed)", layers.len());
    // #endregion

    // #region messenger-info
    // Which messages do we want? Severity filters how bad, type filters what
    // kind. INFO and VERBOSE are left out: they are a firehose of loader chatter
    // that would hide the messages that matter.
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
    // #endregion

    // #region create-with-validation
    // Enabling a layer = passing its name; the debug messenger itself lives in
    // the VK_EXT_debug_utils *instance extension*, which must be enabled too.
    let layer_ptrs = [validation_layer.as_ptr()];
    let ext_ptrs = [debug_utils::NAME.as_ptr()];

    // A messenger can only exist once the instance exists, so create_instance
    // and destroy_instance would go unchecked. Chaining the same create-info
    // into `p_next` (push_next) makes the layer report on those two calls too.
    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layer_ptrs)
        .enabled_extension_names(&ext_ptrs)
        .push(&mut messenger_info);

    // SAFETY: `create_info` and everything it points to outlive the call.
    let instance = unsafe { entry.create_instance(&create_info, None)? };
    // #endregion

    // #region create-messenger
    // Extension functions are not part of the core loader table, so ash wraps
    // them in a small struct that loads their function pointers.
    let debug_utils = debug_utils::Instance::load(&entry, &instance);
    // SAFETY: `messenger_info` is valid and the callback is a valid `extern "system"` fn.
    let messenger = unsafe { debug_utils.create_debug_utils_messenger(&messenger_info, None)? };
    println!("Instance + debug messenger created");
    // #endregion

    // #region cleanup
    // Destroy in reverse order of creation: the messenger belongs to the instance.
    // SAFETY: both objects are alive, nothing else uses them, and they are destroyed once.
    unsafe {
        debug_utils.destroy_debug_utils_messenger(messenger, None);
        instance.destroy_instance(None);
    }
    println!("Cleaned up");
    // #endregion

    Ok(())
}
