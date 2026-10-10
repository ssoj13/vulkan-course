// Step 02 - Validation layers and the debug messenger
//
// Vulkan does almost no error checking by itself: a wrong call is undefined
// behaviour (a crash, garbage, or "works on my GPU"). The Khronos validation
// layer sits between your program and the driver and checks every call against
// the specification. Think of it as a very strict linter for the GPU API.
// It reports problems through a debug messenger: a callback that we provide.
//
// This step repeats step 01 and adds the layer and the messenger.

#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// #region debug-callback
// The driver/layer calls this function whenever it has something to say.
// Return vk::False: "do not abort the Vulkan call that triggered the message".
// The signature (vk:: enums instead of Vk* ones) is what Vulkan-Hpp 1.4
// expects for PFN_DebugUtilsMessengerCallbackEXT.
VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                               vk::DebugUtilsMessageTypeFlagsEXT type,
                                               const vk::DebugUtilsMessengerCallbackDataEXT* data, void* /*userData*/) {
    std::println(stderr, "[{}] [{}] {}", vk::to_string(severity), vk::to_string(type), data->pMessage);
    return vk::False;
}
// #endregion

int main()
try {
    vk::raii::Context context;

    // #region layers-available
    // Layers are optional libraries the loader can inject. The validation layer
    // comes with the Vulkan SDK, so a machine without the SDK may not have it.
    // Fail early with a clear message instead of a cryptic error later.
    constexpr const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    const std::vector<vk::LayerProperties> layers = context.enumerateInstanceLayerProperties();
    const bool hasValidation = std::ranges::any_of(
        layers, [&](const vk::LayerProperties& l) { return std::string_view{l.layerName.data()} == validationLayer; });
    if (!hasValidation) {
        throw std::runtime_error(std::string{validationLayer} + " not found - install the Vulkan SDK");
    }
    std::println("Validation layer found ({} layers installed)", layers.size());
    // #endregion

    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName   = "Vulkan Textbook - 02 Validation",
        .applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
        .pEngineName        = "No Engine",
        .engineVersion      = VK_MAKE_API_VERSION(0, 1, 0, 0),
        .apiVersion         = vk::ApiVersion14,
    };

    // #region messenger-info
    // What do we want to hear about? Severity = how bad; type = what kind.
    // Info and Verbose are left out on purpose: the loader sends 100+ info lines
    // (layer manifests, registry paths) that would hide the messages that matter.
    using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using Type     = vk::DebugUtilsMessageTypeFlagBitsEXT;
    constexpr vk::DebugUtilsMessengerCreateInfoEXT messengerInfo{
        .messageSeverity = Severity::eWarning | Severity::eError,
        .messageType     = Type::eGeneral | Type::eValidation | Type::ePerformance,
        .pfnUserCallback = &debugCallback,
    };
    // #endregion

    // #region create-with-validation
    // Enable the layer by name, and the VK_EXT_debug_utils extension, which is
    // the mechanism that delivers messages to our callback.
    // The messenger create-info is also chained into InstanceCreateInfo via
    // pNext. A normal messenger only exists while the instance does, so it
    // could not report problems from vkCreateInstance / vkDestroyInstance
    // themselves; the chained copy covers exactly those two calls.
    // StructureChain wires the pNext pointers for us.
    constexpr std::array layerNames{validationLayer};
    constexpr std::array extensionNames{vk::EXTDebugUtilsExtensionName};

    const vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> createChain{
        vk::InstanceCreateInfo{
            .pApplicationInfo        = &appInfo,
            .enabledLayerCount       = static_cast<uint32_t>(layerNames.size()),
            .ppEnabledLayerNames     = layerNames.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(extensionNames.size()),
            .ppEnabledExtensionNames = extensionNames.data(),
        },
        messengerInfo,
    };
    vk::raii::Instance instance(context, createChain.get<vk::InstanceCreateInfo>());
    // #endregion

    // #region create-messenger
    // The persistent messenger that watches everything between instance
    // creation and destruction (all our later calls).
    vk::raii::DebugUtilsMessengerEXT messenger(instance, messengerInfo);
    std::println("Instance with validation + debug messenger created");
    // #endregion

    // #region cleanup
    // Destruction is automatic and runs in reverse order of declaration:
    // messenger first, then instance, then the context (loader). The messenger
    // must die before the instance it was created from - declaration order
    // guarantees that.
    // #endregion
    return EXIT_SUCCESS;
} catch (const vk::SystemError& e) {
    std::println(stderr, "Vulkan error: {}", e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return EXIT_FAILURE;
}
