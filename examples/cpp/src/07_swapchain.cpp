// Step 07 - The swapchain and its image views
//
// A surface (step 06) is only "the window". To show pixels in it we need images
// the presentation engine (the OS compositor + display driver) can take from
// us: a swapchain is a small ring of such images, owned by the presentation
// engine and lent to the application one at a time. Think of a render farm
// writing frames into a fixed set of buffers that the playback system displays
// in turn. VK_KHR_swapchain is a DEVICE extension: the images live in the GPU's
// memory.
//
// Choices made here (each printed): pixel format + color space, present mode,
// size (extent) and number of images. Then an image view per image: the "how to
// look at this image" object that rendering needs. Steps 01-06 are repeated,
// reduced to what this step needs.

#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Step 02: validation messages end up here.
VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                               vk::DebugUtilsMessageTypeFlagsEXT type,
                                               const vk::DebugUtilsMessengerCallbackDataEXT* data, void* /*userData*/) {
    try {
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

// Step 06: short spec names ("VK_FORMAT_B8G8R8A8_SRGB" -> "B8G8R8A8_SRGB"), as the Rust examples print them.
std::string shortName(std::string_view names, std::string_view prefix, std::string_view suffix = "") {
    std::string out;
    while (!names.empty()) {
        const size_t bar = names.find('|');
        std::string_view name = names.substr(0, bar);
        names = bar == std::string_view::npos ? std::string_view{} : names.substr(bar + 1);
        if (name.starts_with(prefix)) {
            name.remove_prefix(prefix.size());
        }
        if (!suffix.empty() && name.ends_with(suffix)) {
            name.remove_suffix(suffix.size());
        }
        std::string part{name};
        if (const size_t bit = part.find("_BIT"); bit != std::string::npos && (bit + 4 == part.size() || part[bit + 4] == '_')) {
            part.erase(bit, 4);
        }
        out += out.empty() ? "" : " | ";
        out += part;
    }
    return out;
}

// Step 06: GLFW init/terminate and the window as RAII objects.
struct Glfw {
    Glfw() {
        glfwSetErrorCallback([](int code, const char* text) { std::println(stderr, "GLFW error {}: {}", code, text); });
        if (glfwInit() != GLFW_TRUE) {
            throw std::runtime_error("glfwInit failed");
        }
    }
    ~Glfw() { glfwTerminate(); }
    Glfw(const Glfw&)            = delete;
    Glfw& operator=(const Glfw&) = delete;
};
using WindowPtr = std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)>;

int main()
try {
    // ---- Step 06: window (800x600 physical pixels, fixed size) ----
    Glfw glfw;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 07 swapchain", nullptr, nullptr), &glfwDestroyWindow);
    if (!window) {
        throw std::runtime_error("glfwCreateWindow failed");
    }
    int width  = 0;
    int height = 0;
    glfwGetFramebufferSize(window.get(), &width, &height);
    std::println("Window: {}x{} pixels", width, height);

    // ---- Steps 01-02 + 06: instance with validation and the surface extensions ----
    vk::raii::Context context;

    constexpr const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    const std::vector<vk::LayerProperties> layers = context.enumerateInstanceLayerProperties();
    const bool hasValidation = std::ranges::any_of(
        layers, [&](const vk::LayerProperties& l) { return std::string_view{l.layerName.data()} == validationLayer; });
    if (!hasValidation) {
        throw std::runtime_error(std::string{validationLayer} + " not found - install the Vulkan SDK");
    }

    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName   = "Vulkan from Zero - 07 swapchain",
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

    uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    if (glfwExts == nullptr) {
        throw std::runtime_error("GLFW found no Vulkan loader or no surface extensions");
    }
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);
    extensions.push_back(vk::EXTDebugUtilsExtensionName);

    constexpr std::array layerNames{validationLayer};
    const vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> createChain{
        vk::InstanceCreateInfo{
            .pApplicationInfo        = &appInfo,
            .enabledLayerCount       = static_cast<uint32_t>(layerNames.size()),
            .ppEnabledLayerNames     = layerNames.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        },
        messengerInfo,
    };
    vk::raii::Instance instance(context, createChain.get<vk::InstanceCreateInfo>());
    vk::raii::DebugUtilsMessengerEXT messenger(instance, messengerInfo);

    VkSurfaceKHR rawSurface = VK_NULL_HANDLE;
    if (const VkResult r = glfwCreateWindowSurface(*instance, window.get(), nullptr, &rawSurface); r != VK_SUCCESS) {
        throw std::runtime_error(std::string{"glfwCreateWindowSurface failed: "} + string_VkResult(r));
    }
    vk::raii::SurfaceKHR surface(instance, rawSurface);

    // ---- Steps 03 + 06: Vulkan 1.4, features, graphics + present family - and now VK_KHR_swapchain ----
    const std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }
    std::println("Found {} GPU(s)", devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        std::println("GPU {}: {}", i, devices[i].getProperties().deviceName.data());
    }
    const vk::raii::PhysicalDevice* best = nullptr;
    int bestScore                        = 0;
    uint32_t family                      = 0;
    for (const vk::raii::PhysicalDevice& device : devices) {
        const vk::PhysicalDeviceProperties props = device.getProperties();
        if (props.apiVersion < vk::ApiVersion14) {
            continue;
        }
        const auto featureChain = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>();
        const auto& features13  = featureChain.get<vk::PhysicalDeviceVulkan13Features>();
        if (features13.dynamicRendering != vk::True || features13.synchronization2 != vk::True) {
            continue;
        }
        // #region swapchain-support
        // A GPU that can present to the surface practically always has the
        // swapchain extension, but "practically" is not a check: a device
        // extension we enable must be listed here, or vkCreateDevice fails with
        // VK_ERROR_EXTENSION_NOT_PRESENT.
        const std::vector<vk::ExtensionProperties> exts = device.enumerateDeviceExtensionProperties();
        const bool hasSwapchain = std::ranges::any_of(exts, [](const vk::ExtensionProperties& e) {
            return std::string_view{e.extensionName.data()} == vk::KHRSwapchainExtensionName;
        });
        if (!hasSwapchain) {
            continue;
        }
        // #endregion
        const std::vector<vk::QueueFamilyProperties> families = device.getQueueFamilyProperties();
        bool found                                            = false;
        uint32_t candidate                                    = 0;
        for (uint32_t i = 0; i < families.size(); ++i) {
            const bool graphics = static_cast<bool>(families[i].queueFlags & vk::QueueFlagBits::eGraphics);
            if (graphics && device.getSurfaceSupportKHR(i, *surface) == vk::True) {
                found     = true;
                candidate = i;
                break;
            }
        }
        if (!found) {
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
        if (score > bestScore) {
            bestScore = score;
            best      = &device;
            family    = candidate;
        }
    }
    if (best == nullptr) {
        throw std::runtime_error(
            "No Vulkan 1.4 GPU with dynamicRendering + synchronization2 + VK_KHR_swapchain can present to this window");
    }
    const vk::raii::PhysicalDevice& physicalDevice = *best;
    std::println("\nPicked: {}", physicalDevice.getProperties().deviceName.data());
    std::println("Graphics + present family: {}", family);

    // #region create-device
    // Step 05's device, plus the first device extension of the book. The
    // swapchain functions (create, get images, acquire, present) only exist on a
    // device that enabled VK_KHR_swapchain; vk::raii loads them automatically.
    constexpr float queuePriority = 1.0f;
    const vk::DeviceQueueCreateInfo queueInfo{
        .queueFamilyIndex = family,
        .queueCount       = 1,
        .pQueuePriorities = &queuePriority,
    };
    vk::PhysicalDeviceVulkan13Features features13{
        .synchronization2 = vk::True,
        .dynamicRendering = vk::True,
    };
    constexpr std::array deviceExtensions{vk::KHRSwapchainExtensionName};
    const vk::DeviceCreateInfo deviceInfo{
        .pNext                   = &features13,
        .queueCreateInfoCount    = 1,
        .pQueueCreateInfos       = &queueInfo,
        .enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
    };
    vk::raii::Device device(physicalDevice, deviceInfo);
    std::println("VkDevice created with {}", vk::KHRSwapchainExtensionName);
    // #endregion

    // #region choose-format
    // We want 8-bit BGRA with the *_SRGB encoding on an sRGB monitor: shaders
    // write linear values, the hardware encodes them to sRGB when storing (and
    // blending happens in linear space). Every desktop driver offers it; if not,
    // the first pair the surface lists is still a valid choice.
    const std::vector<vk::SurfaceFormatKHR> formats = physicalDevice.getSurfaceFormatsKHR(*surface);
    if (formats.empty()) {
        throw std::runtime_error("The surface reports no formats");
    }
    const auto preferred = std::ranges::find_if(formats, [](const vk::SurfaceFormatKHR& f) {
        return f.format == vk::Format::eB8G8R8A8Srgb && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    });
    const vk::SurfaceFormatKHR format = preferred != formats.end() ? *preferred : formats.front();
    std::println("\nSurface formats: {}, chosen {} + {}", formats.size(),
                 shortName(string_VkFormat(static_cast<VkFormat>(format.format)), "VK_FORMAT_"),
                 shortName(string_VkColorSpaceKHR(static_cast<VkColorSpaceKHR>(format.colorSpace)), "VK_COLOR_SPACE_", "_KHR"));
    // #endregion

    // #region choose-present-mode
    // MAILBOX: no tearing and the lowest latency (a newer frame replaces the one
    // waiting for vblank). FIFO is the only mode every driver must support, so
    // it is the fallback.
    const std::vector<vk::PresentModeKHR> modes = physicalDevice.getSurfacePresentModesKHR(*surface);
    const vk::PresentModeKHR presentMode =
        std::ranges::contains(modes, vk::PresentModeKHR::eMailbox) ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
    const auto modeName = [](vk::PresentModeKHR m) {
        return shortName(string_VkPresentModeKHR(static_cast<VkPresentModeKHR>(m)), "VK_PRESENT_MODE_", "_KHR");
    };
    std::string modeList;
    for (const vk::PresentModeKHR m : modes) {
        modeList += modeList.empty() ? "" : ", ";
        modeList += modeName(m);
    }
    std::println("Present modes: {}; chosen {}", modeList, modeName(presentMode));
    // #endregion

    // #region choose-extent
    // Normally the swapchain must match the window: currentExtent. The special
    // value 0xFFFFFFFF means the window takes its size from the swapchain
    // (Wayland); then we use the window size, clamped to the allowed range.
    const vk::SurfaceCapabilitiesKHR caps = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    vk::Extent2D extent = caps.currentExtent;
    if (caps.currentExtent.width == std::numeric_limits<uint32_t>::max()) {
        extent = vk::Extent2D{
            .width  = std::clamp(static_cast<uint32_t>(width), caps.minImageExtent.width, caps.maxImageExtent.width),
            .height = std::clamp(static_cast<uint32_t>(height), caps.minImageExtent.height, caps.maxImageExtent.height),
        };
    }
    std::println("Extent: {}x{}", extent.width, extent.height);
    // #endregion

    // #region choose-image-count
    // The minimum is what the presentation engine needs for itself; one more
    // lets us render into a free image while the others are on screen or
    // queued. maxImageCount 0 means "no limit".
    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) {
        imageCount = std::min(imageCount, caps.maxImageCount);
    }
    std::println("Image count: {} (min {} + 1, max {})", imageCount, caps.minImageCount, caps.maxImageCount);
    // #endregion

    // #region create-swapchain
    // imageUsage COLOR_ATTACHMENT: we will render into the images.
    // EXCLUSIVE: one queue family uses them (graphics = present family).
    // preTransform = currentTransform: no extra rotation by the compositor.
    // OPAQUE: ignore alpha when composing with the desktop.
    // clipped: pixels hidden behind other windows may be skipped.
    if (!(caps.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque)) {
        throw std::runtime_error("The surface does not support OPAQUE composite alpha");
    }
    const vk::SwapchainCreateInfoKHR swapchainInfo{
        .surface          = *surface,
        .minImageCount    = imageCount,
        .imageFormat      = format.format,
        .imageColorSpace  = format.colorSpace,
        .imageExtent      = extent,
        .imageArrayLayers = 1,
        .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .preTransform     = caps.currentTransform,
        .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode      = presentMode,
        .clipped          = vk::True,
    };
    vk::raii::SwapchainKHR swapchain(device, swapchainInfo);
    // The driver may create MORE images than minImageCount; always ask.
    const std::vector<vk::Image> images = swapchain.getImages();
    std::println("Swapchain created: {} images", images.size());
    // #endregion

    // #region create-image-views
    // An image is memory plus a description; a view says how to read it: as a 2D
    // image, in this format, with channels unswizzled, color aspect, mip level 0
    // and array layer 0 only. Rendering attaches views, not images.
    std::vector<vk::raii::ImageView> imageViews;
    imageViews.reserve(images.size());
    for (const vk::Image image : images) {
        imageViews.emplace_back(device, vk::ImageViewCreateInfo{
                                            .image            = image,
                                            .viewType         = vk::ImageViewType::e2D,
                                            .format           = format.format,
                                            .components       = {}, // eIdentity for r, g, b, a
                                            .subresourceRange = {
                                                .aspectMask     = vk::ImageAspectFlagBits::eColor,
                                                .baseMipLevel   = 0,
                                                .levelCount     = 1,
                                                .baseArrayLayer = 0,
                                                .layerCount     = 1,
                                            },
                                        });
    }
    std::println("Image views created: {}", imageViews.size());
    // #endregion

    // #region event-loop
    // Still nothing is drawn (that is step 09): pump events until the close button.
    std::println("\nClose the window to exit.");
    while (glfwWindowShouldClose(window.get()) == GLFW_FALSE) {
        glfwWaitEvents();
    }
    std::println("Window closed");
    // #endregion

    // #region cleanup
    // Wait for the GPU, then RAII destroys in reverse order of declaration:
    // image views (they look at the swapchain's images), swapchain, device,
    // surface, messenger, instance, window. The images themselves belong to the
    // swapchain: never destroy them yourself.
    device.waitIdle();
    // #endregion
    return EXIT_SUCCESS;
} catch (const vk::SystemError& e) {
    std::println(stderr, "Vulkan error: {}", e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return EXIT_FAILURE;
}
