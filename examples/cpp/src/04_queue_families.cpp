// Step 04 - Queue families
//
// The GPU does not run your calls directly. You record commands and SUBMIT them
// to a queue; the GPU consumes the queue asynchronously. Queues come in groups
// called queue families, and each family supports a subset of operations:
//   graphics - draw calls (rasterization)
//   compute  - compute shaders (like Houdini OpenCL / Maya compute nodes)
//   transfer - memory copies (DMA engines that run beside the main GPU cores)
// Before creating a logical device we must know which family index to use.
//
// This step repeats steps 01-03 (instance, validation, device picking) and
// inspects the queue families of the chosen GPU.

#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <optional>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Step 02: validation messages end up here.
VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                               vk::DebugUtilsMessageTypeFlagsEXT type,
                                               const vk::DebugUtilsMessengerCallbackDataEXT* data, void* /*userData*/) {
    // pMessage is optional in the spec, so check it. Nothing may throw out of here:
    // the layer calls us through Vulkan's C ABI (VKAPI_CALL), and exceptions must
    // not unwind through its frames, so std::println (which can throw) is wrapped
    // and any failure ignored.
    try {
        // Same format as the Rust examples: [LEVEL][KIND] message.
        const char* level = severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError     ? "ERROR"
                            : severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning ? "WARNING"
                                                                                             : "INFO";
        const char* kind = (type & vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation)    ? "VALIDATION"
                           : (type & vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance) ? "PERFORMANCE"
                                                                                         : "GENERAL";
        std::println(stderr, "[{}][{}] {}", level, kind, data && data->pMessage ? data->pMessage : "<no message>");
    } catch (...) {
    }
    return vk::False;
}

int main()
try {
    // ---- Steps 01-02: loader, instance with validation, debug messenger ----
    vk::raii::Context context;

    constexpr const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    const std::vector<vk::LayerProperties> layers = context.enumerateInstanceLayerProperties();
    const bool hasValidation = std::ranges::any_of(
        layers, [&](const vk::LayerProperties& l) { return std::string_view{l.layerName.data()} == validationLayer; });
    if (!hasValidation) {
        throw std::runtime_error(std::string{validationLayer} + " not found - install the Vulkan SDK");
    }

    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName   = "Vulkan from Zero - 04 queue families",
        .applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .pEngineName        = "No Engine",
        .engineVersion      = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .apiVersion         = vk::ApiVersion14,
    };

    using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using Type     = vk::DebugUtilsMessageTypeFlagBitsEXT;
    constexpr vk::DebugUtilsMessengerCreateInfoEXT messengerInfo{
        .messageSeverity = Severity::eWarning | Severity::eError,
        .messageType     = Type::eGeneral | Type::eValidation | Type::ePerformance,
        .pfnUserCallback = &debugCallback,
    };

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
    vk::raii::DebugUtilsMessengerEXT messenger(instance, messengerInfo);

    // ---- Step 03: pick a physical device (discrete GPU preferred, Vulkan 1.4 + features required) ----
    const std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }
    std::println("Found {} GPU(s)", devices.size());
    // Step 03 printed every property, feature and limit; here we only list names.
    for (size_t i = 0; i < devices.size(); ++i) {
        std::println("GPU {}: {}", i, devices[i].getProperties().deviceName.data());
    }
    // Eligible = a Vulkan 1.4 device with dynamicRendering and synchronization2
    // (the two features this textbook relies on: dynamic rendering replaces
    // render passes, synchronization2 is the modern barrier API). Among eligible
    // GPUs we prefer a discrete card (fastest, own VRAM) over integrated (shares
    // system RAM), then virtual, then CPU software rasterisers. Highest score
    // wins; on a tie the first one found stays.
    const vk::raii::PhysicalDevice* best = nullptr;
    int bestScore                        = 0;
    for (const vk::raii::PhysicalDevice& device : devices) {
        const vk::PhysicalDeviceProperties props = device.getProperties();
        // Check the version before asking about 1.3 features: querying a core
        // feature struct of a newer version on an older device is invalid
        // (spec: "Extending Physical Device Core Functionality").
        if (props.apiVersion < vk::ApiVersion14) {
            continue;
        }
        // Named chain: reading through a reference avoids copying a struct (with
        // a dangling pNext) out of a temporary.
        const auto featureChain = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>();
        const auto& features13  = featureChain.get<vk::PhysicalDeviceVulkan13Features>();
        if (features13.dynamicRendering != vk::True || features13.synchronization2 != vk::True) {
            continue;
        }
        int score = 1;
        switch (props.deviceType) {
        case vk::PhysicalDeviceType::eDiscreteGpu:
            score = 1000;
            break;
        case vk::PhysicalDeviceType::eIntegratedGpu:
            score = 100;
            break;
        case vk::PhysicalDeviceType::eVirtualGpu:
            score = 10;
            break;
        default:
            break;
        }
        // Strictly greater: on a tie the first GPU found stays.
        if (score > bestScore) {
            bestScore = score;
            best      = &device;
        }
    }
    if (best == nullptr) {
        throw std::runtime_error("No GPU supports Vulkan 1.4 with dynamicRendering and synchronization2");
    }
    const vk::raii::PhysicalDevice& physicalDevice = *best;
    std::println("\nPicked: {}", physicalDevice.getProperties().deviceName.data());

    // ---- Step 04: queue families ----

    // #region families
    // Every GPU exposes its own list of families. The INDEX in this vector is
    // the "family index" that all later calls (device and queue creation) use.
    const std::vector<vk::QueueFamilyProperties> families = physicalDevice.getQueueFamilyProperties();
    // #endregion

    // #region print-families
    // queueCount   - how many queues exist in the family.
    // queueFlags   - which operations the family supports. A bitmask: a family
    //                may have several capabilities at once, so we print every
    //                flag that is set, by its spec name.
    // timestampValidBits - how many bits of GPU timestamps are meaningful
    //                      (0 = the family cannot write timestamps).
    using Flag = vk::QueueFlagBits;
    constexpr std::array<std::pair<Flag, std::string_view>, 8> knownFlags{{
        {Flag::eGraphics, "GRAPHICS"},
        {Flag::eCompute, "COMPUTE"},
        {Flag::eTransfer, "TRANSFER"},
        {Flag::eSparseBinding, "SPARSE_BINDING"},
        {Flag::eProtected, "PROTECTED"},
        {Flag::eVideoDecodeKHR, "VIDEO_DECODE"},
        {Flag::eVideoEncodeKHR, "VIDEO_ENCODE"},
        {Flag::eOpticalFlowNV, "OPTICAL_FLOW"},
    }};
    std::println("\n{} queue families:", families.size());
    for (size_t i = 0; i < families.size(); ++i) {
        const vk::QueueFamilyProperties& f = families[i];
        std::string flags;
        for (const auto& [flag, name] : knownFlags) {
            if (f.queueFlags & flag) {
                flags += flags.empty() ? "" : " | ";
                flags += name;
            }
        }
        std::println("  family {}: {} queue(s), timestamp bits {}, [{}]", i, f.queueCount, f.timestampValidBits,
                     flags);
    }
    // #endregion

    // #region find-graphics
    // Pick families by capability. Our main family needs GRAPHICS. Only TRANSFER
    // is implied by GRAPHICS/COMPUTE (the spec makes that bit optional for such
    // families); COMPUTE on a graphics family is not implied. The spec only
    // promises: "If an implementation exposes any queue family that supports
    // graphics operations, at least one queue family of at least one physical
    // device exposed by the implementation must support both graphics and
    // compute operations" - per implementation, not per GPU. On desktop GPUs
    // the first graphics family also has COMPUTE; rendering needs only GRAPHICS.
    // Beyond that, many GPUs have extra specialised families, useful for async
    // work later:
    //  - a transfer-only family: DMA engine, uploads that do not stall rendering
    //  - a compute family without graphics: "async compute"
    // std::optional means "may not exist on this GPU".
    const auto findFamily = [&](auto predicate) -> std::optional<uint32_t> {
        const auto it = std::ranges::find_if(families, predicate);
        if (it == families.end()) {
            return std::nullopt;
        }
        return static_cast<uint32_t>(it - families.begin());
    };

    const std::optional<uint32_t> graphicsFamily = findFamily([](const vk::QueueFamilyProperties& f) {
        return static_cast<bool>(f.queueFlags & Flag::eGraphics);
    });
    const std::optional<uint32_t> transferFamily = findFamily([](const vk::QueueFamilyProperties& f) {
        return (f.queueFlags & Flag::eTransfer) && !(f.queueFlags & (Flag::eGraphics | Flag::eCompute));
    });
    const std::optional<uint32_t> computeFamily = findFamily([](const vk::QueueFamilyProperties& f) {
        return (f.queueFlags & Flag::eCompute) && !(f.queueFlags & Flag::eGraphics);
    });

    if (!graphicsFamily) {
        throw std::runtime_error("GPU has no graphics queue family");
    }
    std::println("\nGraphics family: {}", *graphicsFamily);
    if (transferFamily) {
        std::println("Dedicated transfer family: {}", *transferFamily);
    } else {
        std::println("No dedicated transfer family");
    }
    if (computeFamily) {
        std::println("Async compute family: {}", *computeFamily);
    } else {
        std::println("No async compute family");
    }
    // #endregion

    return EXIT_SUCCESS;
} catch (const vk::SystemError& e) {
    std::println(stderr, "Vulkan error: {}", e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return EXIT_FAILURE;
}
