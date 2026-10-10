// Step 01 - Instance
//
// The very first thing every Vulkan program does: load the Vulkan loader and
// create a VkInstance - the connection between your program and the Vulkan
// runtime. Nothing is drawn here; we only prove that Vulkan is alive.
//
// Style note: we use Vulkan-Hpp's RAII layer (vk::raii::*). Every object
// destroys itself when it goes out of scope, so there are no vkDestroy* calls.
// VULKAN_HPP_NO_STRUCT_CONSTRUCTORS (set in CMakeLists.txt) turns Vulkan
// structs into plain aggregates, so we fill them with designated initializers:
//     vk::ApplicationInfo{ .pApplicationName = "x", .apiVersion = ... }
// Fields not mentioned are zero; sType is filled in for us. The fields must be
// written in the same order as in the Vulkan specification.

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <print>

int main()
try {
    // #region load-loader
    // vk::raii::Context loads the Vulkan loader (vulkan-1.dll on Windows) and
    // fetches the few "global" functions that work without an instance:
    // vkCreateInstance, vkEnumerateInstanceExtensionProperties, ...
    // The loader is the small library that forwards our calls to the GPU driver.
    vk::raii::Context context;
    // #endregion

    // #region app-info
    // ApplicationInfo is optional metadata that drivers may use to enable
    // per-game workarounds. apiVersion is the one field that matters: it is the
    // highest Vulkan version we promise to use. We target 1.4. Dynamic rendering
    // and synchronization2 have been core since 1.3 (still enabled through
    // VkPhysicalDeviceVulkan13Features); 1.4 additionally promotes push descriptors,
    // dynamic rendering local read, maintenance5/6 and more to core.
    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName   = "Vulkan Textbook - 01 Instance",
        .applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
        .pEngineName        = "No Engine",
        .engineVersion      = VK_MAKE_API_VERSION(0, 1, 0, 0),
        .apiVersion         = vk::ApiVersion14,
    };
    // #endregion

    // #region list-extensions
    // Extensions are optional features of the loader/drivers. Asking which
    // instance-level extensions exist is a harmless way to see what the
    // installed Vulkan runtime offers (surface support, debug utils, ...).
    const std::vector<vk::ExtensionProperties> extensions = context.enumerateInstanceExtensionProperties();
    std::println("Available instance extensions ({}):", extensions.size());
    for (const vk::ExtensionProperties& ext : extensions) {
        std::println("  {} (spec revision {})", ext.extensionName.data(), ext.specVersion);
    }
    // #endregion

    // #region create-instance
    // InstanceCreateInfo says which application we are and which layers and
    // extensions we want. We need none yet, so the counts stay zero.
    // The instance is the root object: all other Vulkan objects are created
    // from it, directly or indirectly.
    const vk::InstanceCreateInfo createInfo{
        .pApplicationInfo = &appInfo,
    };
    vk::raii::Instance instance(context, createInfo);

    const uint32_t version = context.enumerateInstanceVersion();
    std::println("Instance created. Loader supports Vulkan {}.{}.{}", VK_API_VERSION_MAJOR(version),
                 VK_API_VERSION_MINOR(version), VK_API_VERSION_PATCH(version));
    // #endregion

    // #region cleanup
    // There is no cleanup code to write. When main() returns, `instance` and
    // then `context` go out of scope and are destroyed in reverse order of
    // declaration (vkDestroyInstance first, then the loader is unloaded). The
    // order is exactly what Vulkan requires: children before parents.
    // #endregion
    return EXIT_SUCCESS;
} catch (const vk::SystemError& e) {
    std::println(stderr, "Vulkan error: {}", e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return EXIT_FAILURE;
}
