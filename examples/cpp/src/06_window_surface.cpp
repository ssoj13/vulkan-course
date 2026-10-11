// Step 06 - A window and its Vulkan surface
//
// Vulkan itself knows nothing about windows: it renders into images. Showing an
// image on screen is the job of the window system integration (WSI) extensions.
// VkSurfaceKHR is Vulkan's handle for "this OS window": created from the native
// window (an HWND on Windows) by a platform extension (VK_KHR_win32_surface) and
// queried through the platform-neutral VK_KHR_surface. Both are INSTANCE
// extensions: the surface belongs to the instance, not to a GPU, so we can ask
// every GPU whether it can present to it.
//
// GLFW owns the window and the event loop. This step repeats steps 01-05 where
// the surface changes them, then prints what the picked GPU can do with it.

#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_raii.hpp>

// GLFW_INCLUDE_NONE: no OpenGL header. Because vulkan.h is already included,
// glfw3.h also declares its Vulkan helpers (glfwCreateWindowSurface, ...).
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
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

// Short spec names, the same ones the Rust examples print (ash's Debug output):
// "VK_FORMAT_B8G8R8A8_SRGB" -> "B8G8R8A8_SRGB", "VK_PRESENT_MODE_FIFO_KHR" -> "FIFO",
// "VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT" -> "SAMPLED | STORAGE".
// The C names come from vk_enum_string_helper.h (Vulkan SDK); Vulkan-Hpp's own
// vk::to_string would print "B8G8R8A8Srgb". `suffix` is the vendor tag of the
// enum type itself ("_KHR" for VkPresentModeKHR), which ash leaves out too.
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
        // Flag bits: "SAMPLED_BIT" -> "SAMPLED", "FEEDBACK_LOOP_BIT_EXT" -> "FEEDBACK_LOOP_EXT".
        if (const size_t bit = part.find("_BIT"); bit != std::string::npos && (bit + 4 == part.size() || part[bit + 4] == '_')) {
            part.erase(bit, 4);
        }
        out += out.empty() ? "" : " | ";
        out += part;
    }
    return out;
}

// GLFW is a C library: init/terminate and the window are wrapped so they are
// released in the right order even when an exception unwinds main.
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
    // #region create-window
    // GLFW_NO_API: GLFW must not create an OpenGL context, Vulkan brings its own
    // way to show images. Not resizable: a new size invalidates the swapchain,
    // and recreating it is module 7. On Windows GLFW sizes are physical pixels
    // (unless GLFW_SCALE_TO_MONITOR is set), so the client area is 800x600.
    Glfw glfw;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 06 window and surface", nullptr, nullptr),
                     &glfwDestroyWindow);
    if (!window) {
        throw std::runtime_error("glfwCreateWindow failed");
    }
    int width  = 0;
    int height = 0;
    glfwGetFramebufferSize(window.get(), &width, &height);
    std::println("Window: {}x{} pixels", width, height);
    // #endregion

    vk::raii::Context context;

    constexpr const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    const std::vector<vk::LayerProperties> layers = context.enumerateInstanceLayerProperties();
    const bool hasValidation = std::ranges::any_of(
        layers, [&](const vk::LayerProperties& l) { return std::string_view{l.layerName.data()} == validationLayer; });
    if (!hasValidation) {
        throw std::runtime_error(std::string{validationLayer} + " not found - install the Vulkan SDK");
    }

    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName   = "Vulkan from Zero - 06 window and surface",
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

    // #region surface-extensions
    // Which instance extensions does a surface need on this OS? GLFW knows:
    // VK_KHR_surface (the portable part) plus the platform one
    // (VK_KHR_win32_surface on Windows). They must be enabled when the instance
    // is created - a surface cannot be added later.
    uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    if (glfwExts == nullptr) {
        throw std::runtime_error("GLFW found no Vulkan loader or no surface extensions");
    }
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);
    std::string extList;
    for (const char* e : extensions) {
        extList += extList.empty() ? "" : ", ";
        extList += e;
    }
    std::println("Surface extensions: {}", extList);
    extensions.push_back(vk::EXTDebugUtilsExtensionName);
    // #endregion

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

    // #region create-surface
    // GLFW fills VkWin32SurfaceCreateInfoKHR { hinstance, hwnd } with the
    // window's module and HWND and calls vkCreateWin32SurfaceKHR - the same call
    // ash-window makes in the Rust version. vk::raii::SurfaceKHR then owns the
    // raw handle and destroys it before the instance. The OS window must stay
    // alive as long as the surface: `window` is declared before it, so it is
    // destroyed after it.
    VkSurfaceKHR rawSurface = VK_NULL_HANDLE;
    if (const VkResult r = glfwCreateWindowSurface(*instance, window.get(), nullptr, &rawSurface); r != VK_SUCCESS) {
        throw std::runtime_error(std::string{"glfwCreateWindowSurface failed: "} + string_VkResult(r));
    }
    vk::raii::SurfaceKHR surface(instance, rawSurface);
    std::println("Surface created");
    // #endregion

    const std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }
    std::println("Found {} GPU(s)", devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        std::println("GPU {}: {}", i, devices[i].getProperties().deviceName.data());
    }

    // Step 03's choice (Vulkan 1.4, dynamicRendering + synchronization2,
    // discrete first), with one more requirement: a queue family that can both
    // draw and present to OUR surface.
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
        // #region present-family
        // Presenting is a queue operation (vkQueuePresentKHR), so "can this GPU
        // show images in this window?" is asked per queue family AND per surface.
        // A GPU without a display connection answers no, and any family may
        // answer either way (on the course GPU even the compute family can
        // present), so we ask. We want one family that does both graphics and
        // present, so one queue can render and then present.
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
        // #endregion
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
            "No Vulkan 1.4 GPU with dynamicRendering + synchronization2 can present to this window");
    }
    const vk::raii::PhysicalDevice& physicalDevice = *best;
    std::println("\nPicked: {}", physicalDevice.getProperties().deviceName.data());

    // #region family-table
    // The whole answer for the picked GPU, family by family.
    const std::vector<vk::QueueFamilyProperties> families = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < families.size(); ++i) {
        const bool graphics = static_cast<bool>(families[i].queueFlags & vk::QueueFlagBits::eGraphics);
        const bool present  = physicalDevice.getSurfaceSupportKHR(i, *surface) == vk::True;
        std::println("  family {}: graphics {}, present {}", i, graphics ? "yes" : "no", present ? "yes" : "no");
    }
    std::println("Graphics + present family: {}", family);
    // #endregion

    // #region surface-capabilities
    // What the window system allows for images shown in this surface: how many
    // images a swapchain may have, their size, how they may be rotated and
    // blended with the desktop, and what else they may be used for.
    const vk::SurfaceCapabilitiesKHR caps = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    std::println("\nSurface capabilities:");
    // maxImageCount 0 means "no upper limit".
    std::println("  image count: min {}, max {}", caps.minImageCount, caps.maxImageCount);
    // 0xFFFFFFFF x 0xFFFFFFFF would mean "the swapchain decides the window size"
    // (Wayland); on Windows it is the window's client area.
    std::println("  current extent: {}x{}", caps.currentExtent.width, caps.currentExtent.height);
    std::println("  extent range: {}x{} .. {}x{}", caps.minImageExtent.width, caps.minImageExtent.height,
                 caps.maxImageExtent.width, caps.maxImageExtent.height);
    std::println("  max array layers: {}", caps.maxImageArrayLayers);
    std::println("  supported transforms: {}",
                 shortName(string_VkSurfaceTransformFlagsKHR(static_cast<VkSurfaceTransformFlagsKHR>(caps.supportedTransforms)),
                           "VK_SURFACE_TRANSFORM_", "_KHR"));
    std::println("  current transform: {}",
                 shortName(string_VkSurfaceTransformFlagBitsKHR(static_cast<VkSurfaceTransformFlagBitsKHR>(caps.currentTransform)),
                           "VK_SURFACE_TRANSFORM_", "_KHR"));
    std::println("  composite alpha: {}",
                 shortName(string_VkCompositeAlphaFlagsKHR(static_cast<VkCompositeAlphaFlagsKHR>(caps.supportedCompositeAlpha)),
                           "VK_COMPOSITE_ALPHA_", "_KHR"));
    std::println("  image usage: {}",
                 shortName(string_VkImageUsageFlags(static_cast<VkImageUsageFlags>(caps.supportedUsageFlags)), "VK_IMAGE_USAGE_"));
    // #endregion

    // #region surface-formats
    // Pixel format + color space pairs the window system can display. The color
    // space says how the presentation engine interprets the values
    // (SRGB_NONLINEAR = an ordinary sRGB monitor); the format says how they are
    // stored. *_SRGB formats encode linear shader output to sRGB on write.
    const std::vector<vk::SurfaceFormatKHR> formats = physicalDevice.getSurfaceFormatsKHR(*surface);
    std::println("Surface formats ({}):", formats.size());
    for (const vk::SurfaceFormatKHR& f : formats) {
        std::println("  {} + {}", shortName(string_VkFormat(static_cast<VkFormat>(f.format)), "VK_FORMAT_"),
                     shortName(string_VkColorSpaceKHR(static_cast<VkColorSpaceKHR>(f.colorSpace)), "VK_COLOR_SPACE_", "_KHR"));
    }
    // #endregion

    // #region present-modes
    // How finished images replace each other on screen: FIFO = a queue drained
    // once per vertical blank (vsync, always supported), MAILBOX = vsync but a
    // newer image replaces the waiting one, IMMEDIATE = no wait (tearing),
    // FIFO_RELAXED = FIFO that tears when a frame is late.
    const std::vector<vk::PresentModeKHR> modes = physicalDevice.getSurfacePresentModesKHR(*surface);
    std::string modeList;
    for (const vk::PresentModeKHR m : modes) {
        modeList += modeList.empty() ? "" : ", ";
        modeList += shortName(string_VkPresentModeKHR(static_cast<VkPresentModeKHR>(m)), "VK_PRESENT_MODE_", "_KHR");
    }
    std::println("Present modes ({}): {}", modes.size(), modeList);
    // #endregion

    // #region event-loop
    // Nothing is drawn yet: we only pump the OS message queue until the close
    // button sets the "should close" flag.
    std::println("\nClose the window to exit.");
    while (glfwWindowShouldClose(window.get()) == GLFW_FALSE) {
        glfwWaitEvents();
    }
    std::println("Window closed");
    // #endregion

    // #region cleanup
    // RAII destroys in reverse order of declaration: surface (a child of the
    // instance), messenger, instance, context, then the window it pointed at,
    // and finally glfwTerminate.
    // #endregion
    return EXIT_SUCCESS;
} catch (const vk::SystemError& e) {
    std::println(stderr, "Vulkan error: {}", e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return EXIT_FAILURE;
}
