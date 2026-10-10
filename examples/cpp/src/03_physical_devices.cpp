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
    std::println(stderr, "[{}] [{}] {}", vk::to_string(severity), vk::to_string(type), data->pMessage);
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
        .pApplicationName   = "Vulkan Textbook - 03 Physical Devices",
        .applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
        .pEngineName        = "No Engine",
        .engineVersion      = VK_MAKE_API_VERSION(0, 1, 0, 0),
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
    // (integrated + discrete); a machine without a Vulkan driver reports none.
    // The returned objects are lightweight handles owned by the instance.
    const std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }
    std::println("Found {} GPU(s)", devices.size());
    // #endregion

    for (const vk::raii::PhysicalDevice& device : devices) {
        // #region properties
        // Properties are fixed facts about the device. apiVersion is the newest
        // Vulkan the DRIVER supports on this GPU; it is packed into one integer,
        // so we decode it with the VK_API_VERSION_* macros. driverVersion is
        // vendor-specific (NVIDIA, AMD and Intel all pack it differently), so we
        // just print the raw number.
        const vk::PhysicalDeviceProperties props = device.getProperties();
        std::println("\nGPU: {}", props.deviceName.data());
        std::println("  type:           {}", vk::to_string(props.deviceType));
        std::println("  api version:    {}.{}.{}", VK_API_VERSION_MAJOR(props.apiVersion),
                     VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion));
        std::println("  driver version: {:#x} (raw, vendor-specific)", props.driverVersion);
        std::println("  vendor id:      {:#06x}", props.vendorID);
        // #endregion

        // #region features
        // Features are optional capabilities that are OFF until we ask for them
        // when creating the logical device. Newer features live in extra structs
        // (here the Vulkan 1.3 and 1.4 ones) that we chain to the query with
        // pNext; StructureChain does the chaining and getFeatures2 fills
        // everything in.
        const auto featureChain = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
                                                      vk::PhysicalDeviceVulkan14Features>();
        const auto& features13  = featureChain.get<vk::PhysicalDeviceVulkan13Features>();
        const auto& features14  = featureChain.get<vk::PhysicalDeviceVulkan14Features>();
        std::println("  dynamicRendering:  {}", features13.dynamicRendering == vk::True);
        std::println("  synchronization2:  {}", features13.synchronization2 == vk::True);
        std::println("  pushDescriptor:    {}", features14.pushDescriptor == vk::True);
        std::println("  maintenance5:      {}", features14.maintenance5 == vk::True);
        // #endregion

        // #region limits
        // Limits are hard numeric bounds. Exceeding one is a validation error.
        // timestampPeriod converts GPU timestamp ticks to nanoseconds.
        const vk::PhysicalDeviceLimits& limits = props.limits;
        std::println("  maxImageDimension2D:             {}", limits.maxImageDimension2D);
        std::println("  maxPushConstantsSize:            {} bytes", limits.maxPushConstantsSize);
        std::println("  minUniformBufferOffsetAlignment: {} bytes", limits.minUniformBufferOffsetAlignment);
        std::println("  timestampPeriod:                 {} ns/tick", limits.timestampPeriod);
        // #endregion
    }

    // #region pick
    // Scoring: a discrete GPU beats an integrated one, which beats everything
    // else; ties are broken by the largest supported texture size. A device is
    // only a candidate at all if it supports Vulkan 1.4 and the two features
    // this textbook relies on (dynamic rendering replaces render passes,
    // synchronization2 is the modern barrier API). Unsuitable devices are skipped.
    const vk::raii::PhysicalDevice* best = nullptr;
    uint64_t bestScore                   = 0;
    for (const vk::raii::PhysicalDevice& device : devices) {
        const vk::PhysicalDeviceProperties props = device.getProperties();
        const auto features13 = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>()
                                    .get<vk::PhysicalDeviceVulkan13Features>();

        const bool suitable = props.apiVersion >= vk::ApiVersion14 && features13.dynamicRendering == vk::True &&
                              features13.synchronization2 == vk::True;
        if (!suitable) {
            continue;
        }
        uint64_t score = props.limits.maxImageDimension2D;
        if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            score += 100'000;
        } else if (props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
            score += 50'000;
        }
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
