// Step 05 - Logical device and queue
//
// The physical device (step 03) is a read-only description of a GPU. The
// logical device (VkDevice) is our actual working session with it: it owns all
// GPU resources we create later (buffers, images, pipelines) and it is where we
// switch on optional features and ask for queues. You can create several logical
// devices from one physical device; each has its own enabled feature set.
//
// This step repeats steps 01-04 and finally creates the device and fetches a queue.

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
        .pApplicationName   = "Vulkan Textbook - 05 Logical Device",
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

    // ---- Step 03: pick a physical device (discrete GPU preferred, Vulkan 1.4 + features required) ----
    const std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }
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
    std::println("Picked: {}", physicalDevice.getProperties().deviceName.data());

    // ---- Step 04: find the graphics queue family ----
    const std::vector<vk::QueueFamilyProperties> families = physicalDevice.getQueueFamilyProperties();
    const auto graphicsIt                                 = std::ranges::find_if(families, [](const auto& f) {
        return static_cast<bool>(f.queueFlags & vk::QueueFlagBits::eGraphics);
    });
    if (graphicsIt == families.end()) {
        throw std::runtime_error("GPU has no graphics queue family");
    }
    const auto graphicsFamily = static_cast<uint32_t>(graphicsIt - families.begin());
    std::println("Graphics queue family: {}", graphicsFamily);

    // ---- Step 05: logical device ----

    // #region queue-create-info
    // We ask for ONE queue from the graphics family. The priority is a float in
    // [0, 1] that hints how the driver should share GPU time when several queues
    // compete; with a single queue it does not matter, so we use the maximum.
    // Queue priorities are passed as an array (one entry per requested queue).
    constexpr float queuePriority = 1.0f;
    const vk::DeviceQueueCreateInfo queueInfo{
        .queueFamilyIndex = graphicsFamily,
        .queueCount       = 1,
        .pQueuePriorities = &queuePriority,
    };
    // #endregion

    // #region enable-features
    // Features the GPU supports are still disabled until we switch them on here.
    // Enabling a feature the GPU does not have makes device creation fail, which
    // is why step 03 checked them first (and required a Vulkan 1.4 device). Both live
    // in the Vulkan 1.3 feature struct (core since 1.3). We enable no Vulkan 1.4 feature
    // yet: each one is switched on in the step that needs it.
    // dynamicRendering  - draw without VkRenderPass / VkFramebuffer objects.
    // synchronization2  - the cleaner vkCmdPipelineBarrier2 / vkQueueSubmit2 API.
    vk::PhysicalDeviceVulkan13Features features13{
        .synchronization2 = vk::True,
        .dynamicRendering = vk::True,
    };
    // #endregion

    // #region create-device
    // DeviceCreateInfo lists the queues and (via pNext) the features: Vulkan13Features
    // may extend DeviceCreateInfo directly, no PhysicalDeviceFeatures2 wrapper needed.
    // We pass no device extensions yet (a swapchain extension will come with
    // windowing) and no device layers: those are deprecated, instance layers cover
    // devices too.
    const vk::DeviceCreateInfo deviceInfo{
        .pNext                = &features13,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos    = &queueInfo,
    };
    vk::raii::Device device(physicalDevice, deviceInfo);
    std::println("Logical device created");
    // #endregion

    // #region get-queue
    // The queue was created together with the device; here we only fetch its
    // handle by (family index, queue index within the family).
    vk::raii::Queue queue(device, graphicsFamily, 0);
    std::println("Got graphics queue (family {}, index 0)", graphicsFamily);
    // #endregion

    // #region cleanup
    // Before destroying anything, wait until the GPU has finished all submitted
    // work - destroying a device that is still busy is an error. We submitted
    // nothing here, so it returns immediately, but real programs must do this.
    device.waitIdle();
    // RAII then destroys, in reverse order of declaration: queue (just a handle,
    // owned by the device), device, messenger, instance, context. Children always
    // die before their parents, as Vulkan requires.
    // #endregion

    return EXIT_SUCCESS;
} catch (const vk::SystemError& e) {
    std::println(stderr, "Vulkan error: {}", e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return EXIT_FAILURE;
}
