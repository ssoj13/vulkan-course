//! Step 01 - Load Vulkan and create an instance.
//!
//! A `VkInstance` is the connection between your program and the Vulkan
//! *loader* (vulkan-1.dll on Windows). Think of it as the "session" object:
//! every other Vulkan object is created, directly or indirectly, from it.
//! Nothing here touches the GPU yet - we only say who we are and which
//! Vulkan version we want.

use ash::{Entry, vk};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // #region load-loader
    // `Entry` holds the few global functions that work before any instance
    // exists (create_instance, enumerate_*_properties...). `Entry::load()`
    // opens vulkan-1.dll at runtime and looks those functions up by name.
    //
    // SAFETY: loading a dynamic library runs its initialisation code, and the
    // compiler cannot check that the library really is a conforming Vulkan
    // loader. We trust the system's Vulkan runtime, and keep `entry` alive for
    // as long as anything created from it (it unloads the DLL on drop).
    let entry = unsafe { Entry::load()? };

    // SAFETY: plain query, no pointers involved. Vulkan 1.0 loaders do not
    // have this function, in which case ash reports `None`.
    let loader_version =
        unsafe { entry.try_enumerate_instance_version()? }.unwrap_or(vk::API_VERSION_1_0);
    println!(
        "Vulkan loader supports up to {}.{}.{}",
        vk::api_version_major(loader_version),
        vk::api_version_minor(loader_version),
        vk::api_version_patch(loader_version),
    );
    // #endregion

    // #region app-info
    // Optional metadata for drivers (they use it to apply per-game fixes).
    // The important field is `api_version`: the highest Vulkan version this
    // program is written for. We target 1.4. Dynamic rendering and
    // synchronization2 have been core since 1.3 (they are still switched on
    // through VkPhysicalDeviceVulkan13Features); 1.4 additionally promotes push
    // descriptors, dynamic rendering local read, maintenance5/6 and more to core.
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"Vulkan from Zero - 01 instance")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"No Engine")
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_4);
    // #endregion

    // #region list-extensions
    // Instance extensions add optional features to the loader (surfaces,
    // debug utils...). `None` = extensions of the Vulkan implementation and of
    // implicitly enabled layers (not of all installed layers). We only list
    // them here; later steps enable some.
    // SAFETY: plain query, no pointers involved.
    let extensions = unsafe { entry.enumerate_instance_extension_properties(None)? };
    println!("{} instance extensions available:", extensions.len());
    for ext in &extensions {
        // `extension_name` is a fixed-size C char array; ash decodes it.
        println!(
            "  {} (rev {})",
            ext.extension_name_as_c_str()?.to_string_lossy(),
            ext.spec_version
        );
    }
    // #endregion

    // #region create-instance
    // `*CreateInfo` structs are Vulkan's way of passing many parameters: you
    // fill a struct and hand a pointer to the driver. ash's builder methods
    // set the fields (and the hidden `sType`) for us and tie lifetimes to the
    // borrowed data, so `app_info` cannot be freed before the call.
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

    // The second argument is an optional custom host allocator; `None` = default.
    // SAFETY: `create_info` and everything it points to outlive the call.
    let instance = unsafe { entry.create_instance(&create_info, None)? };
    println!("VkInstance created: {:?}", instance.handle());
    // #endregion

    // #region cleanup
    // Vulkan has no garbage collector: whatever we create we must destroy,
    // children before parents. The instance has no children yet.
    // SAFETY: no objects created from `instance` are alive, and it is not used afterwards.
    unsafe { instance.destroy_instance(None) };
    // #endregion

    Ok(())
}
