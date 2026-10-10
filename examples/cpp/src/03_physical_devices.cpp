// Step 03 - Physical devices (GPUs)
//
// A VkPhysicalDevice is a read-only handle to one GPU (or software renderer)
// that the Vulkan runtime found. We cannot render with it yet - that needs a
// logical device (step 05) - but we can ask what it is and what it can do, and
// pick the one we like. Think of it as listing the render devices in Maya's
// Viewport 2.0 preferences before choosing one.
//
// This step repeats steps 01-02 (instance + validation) and adds device queries.

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

// Step 02: validation messages end up here.
VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                               vk::DebugUtilsMessageTypeFlagsEXT type,
                                               const vk::DebugUtilsMessengerCallbackDataEXT* data, void* /*userData*/) {
    // pMessage is optional in the spec, so check it. Nothing may throw out of here:
    // exceptions must not unwind through the C loader/layer frames that called
    // us, so std::println (which can throw) is wrapped and any failure ignored.
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
        .pApplicationName   = "Vulkan from Zero - 03 physical devices",
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

    // ---- Step 03: physical devices ----

    // #region enumerate
    // Ask the loader for every GPU it can see. A laptop typically reports two
    // (integrated + discrete); a machine whose drivers expose no GPU reports none.
    // The returned objects are lightweight handles owned by the instance.
    const std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }
    std::println("Found {} GPU(s)", devices.size());
    // #endregion

    for (size_t i = 0; i < devices.size(); ++i) {
        const vk::raii::PhysicalDevice& device = devices[i];
        // #region properties
        // Properties are fixed facts about the device. apiVersion is the newest
        // Vulkan the DRIVER supports on this GPU; it is packed into one integer,
        // so we decode it with the VK_API_VERSION_* macros. driverVersion is
        // vendor-specific (NVIDIA, AMD and Intel all pack it differently), so we
        // just print the raw number.
        const vk::PhysicalDeviceProperties props = device.getProperties();
        std::println("\nGPU {}: {}", i, props.deviceName.data());
        std::println("  type:       {}", vk::to_string(props.deviceType));
        std::println("  api:        {}.{}.{}", VK_API_VERSION_MAJOR(props.apiVersion),
                     VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion));
        std::println("  driver:     {:#x}", props.driverVersion);
        std::println("  vendor id:  {:#06x}", props.vendorID);
        // #endregion

        // #region features
        // Features are optional capabilities that are OFF until we ask for them
        // when creating the logical device. Newer features live in extra structs
        // (here the Vulkan 1.3 and 1.4 ones) that we chain to the query with
        // pNext; StructureChain does the chaining and getFeatures2 fills
        // everything in. A struct of a newer core version may only be queried on
        // a device that supports that version, so the 1.4 struct is queried only
        // when apiVersion >= 1.4. The chains stay in named variables: their pNext
        // pointers point into the chain itself, so we read through references
        // instead of copying a struct out of a temporary chain.
        const auto chain13 = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>();
        const vk::PhysicalDeviceFeatures& features10 = chain13.get<vk::PhysicalDeviceFeatures2>().features;
        const auto& features13                       = chain13.get<vk::PhysicalDeviceVulkan13Features>();
        std::println("  features:");
        std::println("    sampler_anisotropy (1.0): {}", features10.samplerAnisotropy == vk::True);
        std::println("    dynamic_rendering  (1.3): {}", features13.dynamicRendering == vk::True);
        std::println("    synchronization2   (1.3): {}", features13.synchronization2 == vk::True);
        if (props.apiVersion >= vk::ApiVersion14) {
            const auto chain14 =
                device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan14Features>();
            const auto& features14 = chain14.get<vk::PhysicalDeviceVulkan14Features>();
            std::println("    push_descriptor    (1.4): {}", features14.pushDescriptor == vk::True);
            std::println("    maintenance5       (1.4): {}", features14.maintenance5 == vk::True);
        } else {
            std::println("    push_descriptor    (1.4): n/a (device below 1.4)");
            std::println("    maintenance5       (1.4): n/a (device below 1.4)");
        }
        // #endregion

        // #region limits
        // Limits are hard numeric bounds. Exceeding one is an error that
        // validation usually catches.
        // timestampPeriod converts GPU timestamp ticks to nanoseconds.
        const vk::PhysicalDeviceLimits& limits = props.limits;
        std::println("  limits:");
        std::println("    max_image_dimension_2d:               {}", limits.maxImageDimension2D);
        std::println("    max_push_constants_size:              {} bytes", limits.maxPushConstantsSize);
        std::println("    min_uniform_buffer_offset_alignment:  {} bytes", limits.minUniformBufferOffsetAlignment);
        std::println("    timestamp_period:                     {} ns/tick", limits.timestampPeriod);
        // #endregion
    }

    // #region pick
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
    // #endregion

    return EXIT_SUCCESS;
} catch (const vk::SystemError& e) {
    std::println(stderr, "Vulkan error: {}", e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return EXIT_FAILURE;
}
