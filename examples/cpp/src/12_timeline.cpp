// Step 12 - Timeline semaphores and exact barriers
//
// Step 10 kept one fence per frame in flight. A timeline semaphore replaces all
// of them: it holds a 64-bit counter that only grows. Frame n's submit sets the
// counter to n when the frame completes, and before re-recording a slot the CPU
// waits until the counter reaches n - frames in flight, the last frame that used
// the slot. Like a render farm's "frames done" counter: instead of one flag per
// job, one number says how far the work has got.
//
//   frame n: wait counter >= n - frames in flight -> acquire -> record -> submit (signal n) -> present
//
// Acquire and present still use BINARY semaphores: the swapchain functions only
// accept those. The frame also does more than one kind of GPU work, so the
// barriers between them matter: the triangle is rendered into an offscreen
// 400x300 image, which is then blitted (copied with scaling) to the swapchain
// image - render, copy, present, each ordered by its own barrier.
// --frames 1|2|3 (default 2) sets the frames in flight, --seconds S closes the
// window after S seconds. Lines that depend on timing start with [timing].
// Steps 01-10 are repeated, reduced to what this step needs.

// A window that changed is an expected event, not an error: get
// VK_ERROR_OUT_OF_DATE_KHR back as a vk::Result (step 10).
#define VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS

#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
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
std::string shortName(std::string_view name, std::string_view prefix, std::string_view suffix = "") {
    if (name.starts_with(prefix)) {
        name.remove_prefix(prefix.size());
    }
    if (!suffix.empty() && name.ends_with(suffix)) {
        name.remove_suffix(suffix.size());
    }
    return std::string{name};
}
std::string formatName(vk::Format f) { return shortName(string_VkFormat(static_cast<VkFormat>(f)), "VK_FORMAT_"); }

// Reads a .spv file into 32-bit words (SPIR-V is a stream of uint32; Vulkan
// wants the code 4-byte aligned, which a vector<uint32_t> guarantees).
std::vector<uint32_t> readSpirv(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open " + path.string() + " (run `bun run sync:shaders`)");
    }
    const auto size = static_cast<size_t>(file.tellg());
    if (size == 0 || size % sizeof(uint32_t) != 0) {
        throw std::runtime_error(path.string() + " is not SPIR-V (size is not a multiple of 4)");
    }
    std::vector<uint32_t> words(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(size));
    if (!file || words.front() != 0x07230203u) { // SPIR-V magic number
        throw std::runtime_error(path.string() + " is not SPIR-V (bad magic number)");
    }
    return words;
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

// Step 10: waits for the device to go idle when it leaves scope, so RAII never
// destroys an object the GPU may still be using, even when a frame throws.
struct IdleGuard {
    const vk::raii::Device& device;
    ~IdleGuard() {
        if (const VkResult r = device.getDispatcher()->vkDeviceWaitIdle(*device); r != VK_SUCCESS) {
            std::fprintf(stderr, "vkDeviceWaitIdle failed: %s\n", string_VkResult(r));
        }
    }
};

// #region cli
// Command line:
// - --frames N: frames in flight, 1-3 (default 2);
// - --seconds S: close the window after S seconds (default: run until closed).
struct Args {
    size_t frames = 2;
    std::optional<double> seconds;
};

template <typename T> T parseValue(std::string_view arg, std::string_view text) {
    T value{};
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || end != text.data() + text.size()) {
        throw std::runtime_error(std::string{arg} + ": invalid value " + std::string{text});
    }
    return value;
}

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i += 2) {
        const std::string_view arg = argv[i];
        if (i + 1 >= argc) {
            throw std::runtime_error(std::string{arg} + " needs a value");
        }
        const std::string_view value = argv[i + 1];
        if (arg == "--frames") {
            args.frames = parseValue<size_t>(arg, value);
        } else if (arg == "--seconds") {
            args.seconds = parseValue<double>(arg, value);
        } else {
            throw std::runtime_error("unknown argument " + std::string{arg} + " (use --frames 1|2|3, --seconds S)");
        }
    }
    if (args.frames < 1 || args.frames > 3) {
        throw std::runtime_error("--frames must be 1, 2 or 3");
    }
    return args;
}
// #endregion

// Size of the offscreen image the triangle is rendered into.
constexpr vk::Extent2D offscreenExtent{.width = 400, .height = 300};

// How many frames print what they wait for and signal.
constexpr uint64_t tracedFrames = 5;

// GPU timestamps per frame: command buffer start, rendering done, blit done.
constexpr uint32_t timestamps = 3;

// The GPU timestamps of the frames read back during one second, reduced to an
// average and a maximum per interval, printed once per second.
struct GpuTimes {
    double periodNs = 1.0; // nanoseconds per timestamp tick (VkPhysicalDeviceLimits::timestampPeriod)
    uint64_t mask   = 0;   // the bits of a timestamp that count (timestampValidBits of the family)
    uint64_t second = 0;   // the last whole second that was printed
    uint32_t frames = 0;
    double renderSum = 0.0, renderMax = 0.0; // command buffer start -> color output done, us
    double blitSum = 0.0, blitMax = 0.0;     // color output done -> blit done, us

    void add(double render, double toBlit) {
        ++frames;
        renderSum += render;
        renderMax = std::max(renderMax, render);
        blitSum += toBlit;
        blitMax = std::max(blitMax, toBlit);
    }

    void report(uint64_t now) {
        const double n = std::max(frames, 1u);
        std::println("[timing] {}s, GPU timestamps of {} frames: render {:.1f} us avg, {:.1f} us max; render done -> "
                     "blit done {:.1f} us avg, {:.1f} us max",
                     now, frames, renderSum / n, renderMax, blitSum / n, blitMax);
        second = now;
        frames = 0;
        renderSum = renderMax = blitSum = blitMax = 0.0;
    }
};

// #region frame-resources
// What the CPU touches while recording one frame: one set per frame in flight.
// Compared with step 10 the fence is gone: the timeline semaphore tells when
// the slot is free again.
struct Frame {
    vk::raii::CommandBuffer commandBuffer;
    // Signaled by vkAcquireNextImageKHR, waited on by this slot's submit.
    vk::raii::Semaphore imageAvailable;
};
// #endregion

int main(int argc, char** argv)
try {
    const Args args = parseArgs(argc, argv);

    // ---- Step 06: window (800x600 physical pixels, fixed size) ----
    Glfw glfw;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 12 timeline semaphores", nullptr, nullptr),
                     &glfwDestroyWindow);
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
        .pApplicationName   = "Vulkan from Zero - 12 timeline semaphores",
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

    const std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }
    std::println("Found {} GPU(s)", devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        std::println("GPU {}: {}", i, devices[i].getProperties().deviceName.data());
    }

    // #region pick-device
    // Steps 03 + 06, plus the timelineSemaphore feature (core since 1.2, and
    // every Vulkan 1.4 device supports it - but a feature is only usable when
    // the device is created with it enabled).
    const vk::raii::PhysicalDevice* best = nullptr;
    int bestScore                        = 0;
    uint32_t family                      = 0;
    for (const vk::raii::PhysicalDevice& device : devices) {
        const vk::PhysicalDeviceProperties props = device.getProperties();
        if (props.apiVersion < vk::ApiVersion14) {
            continue;
        }
        const auto featureChain = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,
                                                      vk::PhysicalDeviceVulkan13Features>();
        const auto& f12 = featureChain.get<vk::PhysicalDeviceVulkan12Features>();
        const auto& f13 = featureChain.get<vk::PhysicalDeviceVulkan13Features>();
        if (f12.timelineSemaphore != vk::True || f13.dynamicRendering != vk::True || f13.synchronization2 != vk::True) {
            continue;
        }
        const std::vector<vk::ExtensionProperties> exts = device.enumerateDeviceExtensionProperties();
        const bool hasSwapchain = std::ranges::any_of(exts, [](const vk::ExtensionProperties& e) {
            return std::string_view{e.extensionName.data()} == vk::KHRSwapchainExtensionName;
        });
        if (!hasSwapchain) {
            continue;
        }
        const std::vector<vk::QueueFamilyProperties> families = device.getQueueFamilyProperties();
        std::optional<uint32_t> candidate;
        for (uint32_t i = 0; i < families.size(); ++i) {
            const bool graphics = static_cast<bool>(families[i].queueFlags & vk::QueueFlagBits::eGraphics);
            if (graphics && device.getSurfaceSupportKHR(i, *surface) == vk::True) {
                candidate = i;
                break;
            }
        }
        if (!candidate) {
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
            family    = *candidate;
        }
    }
    if (best == nullptr) {
        throw std::runtime_error("No Vulkan 1.4 GPU with timelineSemaphore + dynamicRendering + synchronization2 + "
                                 "VK_KHR_swapchain can present to this window");
    }
    const vk::raii::PhysicalDevice& physicalDevice = *best;
    std::println("\nPicked: {}", physicalDevice.getProperties().deviceName.data());
    std::println("Graphics + present family: {}", family);

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
    vk::PhysicalDeviceVulkan12Features features12{
        .pNext             = &features13,
        .timelineSemaphore = vk::True,
    };
    constexpr std::array deviceExtensions{vk::KHRSwapchainExtensionName};
    const vk::DeviceCreateInfo deviceInfo{
        .pNext                   = &features12,
        .queueCreateInfoCount    = 1,
        .pQueueCreateInfos       = &queueInfo,
        .enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
    };
    vk::raii::Device device(physicalDevice, deviceInfo);
    std::println("VkDevice created with {} and timelineSemaphore", vk::KHRSwapchainExtensionName);
    // #endregion

    // Format, present mode, extent and image count as in step 07.
    const std::vector<vk::SurfaceFormatKHR> formats = physicalDevice.getSurfaceFormatsKHR(*surface);
    if (formats.empty()) {
        throw std::runtime_error("The surface reports no formats");
    }
    const auto preferred = std::ranges::find_if(formats, [](const vk::SurfaceFormatKHR& f) {
        return f.format == vk::Format::eB8G8R8A8Srgb && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    });
    const vk::SurfaceFormatKHR format = preferred != formats.end() ? *preferred : formats.front();
    std::println("\nSurface format: {} + {}", formatName(format.format),
                 shortName(string_VkColorSpaceKHR(static_cast<VkColorSpaceKHR>(format.colorSpace)), "VK_COLOR_SPACE_", "_KHR"));
    const std::vector<vk::PresentModeKHR> modes = physicalDevice.getSurfacePresentModesKHR(*surface);
    const vk::PresentModeKHR presentMode =
        std::ranges::contains(modes, vk::PresentModeKHR::eMailbox) ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
    std::println("Present mode: {}",
                 shortName(string_VkPresentModeKHR(static_cast<VkPresentModeKHR>(presentMode)), "VK_PRESENT_MODE_", "_KHR"));
    const vk::SurfaceCapabilitiesKHR caps = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    vk::Extent2D extent = caps.currentExtent;
    if (caps.currentExtent.width == std::numeric_limits<uint32_t>::max()) {
        extent = vk::Extent2D{
            .width  = std::clamp(static_cast<uint32_t>(width), caps.minImageExtent.width, caps.maxImageExtent.width),
            .height = std::clamp(static_cast<uint32_t>(height), caps.minImageExtent.height, caps.maxImageExtent.height),
        };
    }
    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) {
        imageCount = std::min(imageCount, caps.maxImageCount);
    }
    if (!(caps.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque)) {
        throw std::runtime_error("The surface does not support OPAQUE composite alpha");
    }

    // #region swapchain-usage
    // Nothing of ours renders into the swapchain images any more: they are the
    // DESTINATION of a blit, a transfer operation, so they need eTransferDst.
    // Only eColorAttachment is guaranteed for a surface, everything else must be
    // checked. eColorAttachment stays anyway: implicit layers of capture and
    // overlay tools draw into the swapchain images themselves. On the dev
    // machine the Bandicam layer creates views and framebuffers on them, and
    // without eColorAttachment the validation layer reports its calls
    // (VUID-VkImageViewCreateInfo-image-04441, VUID-VkFramebufferCreateInfo-pAttachments-00877).
    if (!(caps.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferDst)) {
        throw std::runtime_error("The surface does not support TRANSFER_DST swapchain images");
    }
    vk::raii::SwapchainKHR swapchain(device, vk::SwapchainCreateInfoKHR{
                                                 .surface          = *surface,
                                                 .minImageCount    = imageCount,
                                                 .imageFormat      = format.format,
                                                 .imageColorSpace  = format.colorSpace,
                                                 .imageExtent      = extent,
                                                 .imageArrayLayers = 1,
                                                 .imageUsage       = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment,
                                                 .imageSharingMode = vk::SharingMode::eExclusive,
                                                 .preTransform     = caps.currentTransform,
                                                 .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                                 .presentMode      = presentMode,
                                                 .clipped          = vk::True,
                                             });
    const std::vector<vk::Image> images = swapchain.getImages();
    std::println("Swapchain: {} images of {}x{}, usage TRANSFER_DST (blit target) + COLOR_ATTACHMENT (overlay layers)", images.size(), extent.width,
                 extent.height);
    // #endregion

    // #region offscreen-image
    // The image the triangle is rendered into: 400x300, half the window, same
    // format as the swapchain. Usage eColorAttachment (rendered into) and
    // eTransferSrc (blitted from). The format must support both on optimal
    // tiling, and blitting into the swapchain format needs eBlitDst.
    const vk::FormatProperties formatProps = physicalDevice.getFormatProperties(format.format);
    constexpr vk::FormatFeatureFlags needed = vk::FormatFeatureFlagBits::eColorAttachment |
                                              vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eBlitDst;
    if ((formatProps.optimalTilingFeatures & needed) != needed) {
        throw std::runtime_error(formatName(format.format) + " cannot be rendered into and blitted on this GPU");
    }
    vk::raii::Image offscreen(device, vk::ImageCreateInfo{
                                          .imageType     = vk::ImageType::e2D,
                                          .format        = format.format,
                                          .extent        = {offscreenExtent.width, offscreenExtent.height, 1},
                                          .mipLevels     = 1,
                                          .arrayLayers   = 1,
                                          .samples       = vk::SampleCountFlagBits::e1,
                                          .tiling        = vk::ImageTiling::eOptimal,
                                          .usage         = vk::ImageUsageFlagBits::eColorAttachment |
                                                   vk::ImageUsageFlagBits::eTransferSrc,
                                          .sharingMode   = vk::SharingMode::eExclusive,
                                          .initialLayout = vk::ImageLayout::eUndefined,
                                      });
    // Unlike the swapchain's images, this one has no memory yet. Memory is
    // module 8; here: the first memory type the image accepts that is
    // DEVICE_LOCAL (video memory), one allocation, bound at offset 0. RAII frees
    // the memory before it destroys the image (reverse declaration order): the
    // spec allows freeing memory that is still bound, as long as the image is
    // not used afterwards.
    const vk::MemoryRequirements requirements          = offscreen.getMemoryRequirements();
    const vk::PhysicalDeviceMemoryProperties memoryProps = physicalDevice.getMemoryProperties();
    std::optional<uint32_t> memoryType;
    for (uint32_t i = 0; i < memoryProps.memoryTypeCount; ++i) {
        if ((requirements.memoryTypeBits & (1u << i)) != 0 &&
            (memoryProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            memoryType = i;
            break;
        }
    }
    if (!memoryType) {
        throw std::runtime_error("No DEVICE_LOCAL memory type for the offscreen image");
    }
    vk::raii::DeviceMemory offscreenMemory(device, vk::MemoryAllocateInfo{
                                                       .allocationSize  = requirements.size,
                                                       .memoryTypeIndex = *memoryType,
                                                   });
    offscreen.bindMemory(*offscreenMemory, 0);
    vk::raii::ImageView offscreenView(device, vk::ImageViewCreateInfo{
                                                  .image            = *offscreen,
                                                  .viewType         = vk::ImageViewType::e2D,
                                                  .format           = format.format,
                                                  .subresourceRange = {
                                                      .aspectMask = vk::ImageAspectFlagBits::eColor,
                                                      .levelCount = 1,
                                                      .layerCount = 1,
                                                  },
                                              });
    std::println("Offscreen image: {}x{} {}, usage COLOR_ATTACHMENT + TRANSFER_SRC, memory type {} (DEVICE_LOCAL)",
                 offscreenExtent.width, offscreenExtent.height, formatName(format.format), *memoryType);
    // #endregion

    // The pipeline of step 08, rendering into the offscreen image's format.
    const std::vector<uint32_t> spirv = readSpirv(std::filesystem::path{SHADER_DIR} / "triangle.spv");
    std::optional<vk::raii::ShaderModule> shaderModule;
    shaderModule.emplace(device, vk::ShaderModuleCreateInfo{
                                     .codeSize = spirv.size() * sizeof(uint32_t),
                                     .pCode    = spirv.data(),
                                 });
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex, .module = **shaderModule, .pName = "vertexMain"},
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment, .module = **shaderModule, .pName = "fragmentMain"},
    };
    const vk::PipelineVertexInputStateCreateInfo vertexInput{};
    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
    const vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
    const vk::PipelineRasterizationStateCreateInfo rasterization{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode    = vk::CullModeFlagBits::eBack,
        .frontFace   = vk::FrontFace::eClockwise,
        .lineWidth   = 1.0f,
    };
    const vk::PipelineMultisampleStateCreateInfo multisample{.rasterizationSamples = vk::SampleCountFlagBits::e1};
    const vk::PipelineColorBlendAttachmentState blendAttachment{
        .blendEnable    = vk::False,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
    const vk::PipelineColorBlendStateCreateInfo colorBlend{.attachmentCount = 1, .pAttachments = &blendAttachment};
    constexpr std::array dynamicStates{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates    = dynamicStates.data(),
    };
    vk::raii::PipelineLayout pipelineLayout(device, vk::PipelineLayoutCreateInfo{});
    const vk::PipelineRenderingCreateInfo renderingInfo{
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &format.format,
    };
    vk::raii::Pipeline pipeline(device, nullptr,
                                vk::GraphicsPipelineCreateInfo{
                                    .pNext               = &renderingInfo,
                                    .stageCount          = static_cast<uint32_t>(stages.size()),
                                    .pStages             = stages.data(),
                                    .pVertexInputState   = &vertexInput,
                                    .pInputAssemblyState = &inputAssembly,
                                    .pViewportState      = &viewportState,
                                    .pRasterizationState = &rasterization,
                                    .pMultisampleState   = &multisample,
                                    .pColorBlendState    = &colorBlend,
                                    .pDynamicState       = &dynamicState,
                                    .layout              = *pipelineLayout,
                                });
    shaderModule.reset();
    std::println("Graphics pipeline created (triangle.spv, dynamic rendering)");

    vk::raii::Queue queue(device, family, 0);
    vk::raii::CommandPool commandPool(device, vk::CommandPoolCreateInfo{
                                                  .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                  .queueFamilyIndex = family,
                                              });
    vk::raii::CommandBuffers commandBuffers(device, vk::CommandBufferAllocateInfo{
                                                        .commandPool        = *commandPool,
                                                        .level              = vk::CommandBufferLevel::ePrimary,
                                                        .commandBufferCount = static_cast<uint32_t>(args.frames),
                                                    });

    // #region timestamps
    // A query pool of GPU timestamps, `timestamps` per frame in flight. A
    // timestamp is the value of a GPU tick counter; timestampValidBits of the
    // queue family says how many of its bits count (0 = no timestamps on that
    // family), timestampPeriod how many nanoseconds one tick is.
    const uint32_t validBits = physicalDevice.getQueueFamilyProperties().at(family).timestampValidBits;
    if (validBits == 0) {
        throw std::runtime_error("The graphics queue family has no timestamps");
    }
    vk::raii::QueryPool queryPool(device, vk::QueryPoolCreateInfo{
                                              .queryType  = vk::QueryType::eTimestamp,
                                              .queryCount = timestamps * static_cast<uint32_t>(args.frames),
                                          });
    GpuTimes gpu{
        .periodNs = physicalDevice.getProperties().limits.timestampPeriod,
        .mask     = std::numeric_limits<uint64_t>::max() >> (64 - std::min(validBits, 64u)),
    };
    // #endregion

    // #region timeline-semaphore
    // A semaphore is binary unless its create info carries a
    // VkSemaphoreTypeCreateInfo with eTimeline. The counter starts at
    // initialValue (0: no frame has completed yet) and can only grow.
    const vk::SemaphoreTypeCreateInfo typeInfo{
        .semaphoreType = vk::SemaphoreType::eTimeline,
        .initialValue  = 0,
    };
    vk::raii::Semaphore timeline(device, vk::SemaphoreCreateInfo{.pNext = &typeInfo});
    // #endregion

    // #region binary-semaphores
    // Acquire and present still need BINARY semaphores: a timeline one is
    // invalid there (VUID-vkAcquireNextImageKHR-semaphore-03265,
    // VUID-vkQueuePresentKHR-pWaitSemaphores-03267). So, as in step 10:
    // imageAvailable per frame in flight, renderFinished per swapchain image.
    std::vector<Frame> frames;
    frames.reserve(args.frames);
    for (vk::raii::CommandBuffer& commandBuffer : commandBuffers) {
        frames.push_back(Frame{
            .commandBuffer  = std::move(commandBuffer),
            .imageAvailable = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{}),
        });
    }
    std::vector<vk::raii::Semaphore> renderFinished;
    renderFinished.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        renderFinished.emplace_back(device, vk::SemaphoreCreateInfo{});
    }
    std::println("\nSync objects: 1 timeline semaphore, {} + {} binary semaphores, no fences ({} frame(s) in flight)",
                 frames.size(), renderFinished.size(), frames.size());
    // #endregion

    using Clock      = std::chrono::steady_clock;
    const auto start = Clock::now();
    // Frames submitted so far = the value the last submit signals.
    uint64_t frameNumber = 0;

    // Renders and presents one frame. Returns false when the swapchain no longer
    // matches the window (VK_ERROR_OUT_OF_DATE_KHR): recreating it is step 11,
    // here the caller simply stops.
    const auto drawFrame = [&]() -> bool {
        // Frame numbers start at 1: the counter's 0 means "nothing done yet".
        const uint64_t n                   = frameNumber + 1;
        const uint64_t inFlight            = frames.size();
        const size_t slot                  = static_cast<size_t>((n - 1) % inFlight);
        Frame& frame                       = frames.at(slot);
        const vk::raii::CommandBuffer& cmd = frame.commandBuffer;
        const bool traced                  = n <= tracedFrames;
        const uint32_t firstQuery          = static_cast<uint32_t>(slot) * timestamps;

        // #region wait-timeline
        // The last frame that used this slot is n - frames in flight. Wait until
        // the counter has reached its number: then its command buffer and its
        // imageAvailable semaphore are free. vkWaitSemaphores waits for
        // "counter >= value", so a counter that has run further ahead is fine.
        // The first frames of each slot have nothing to wait for.
        if (n > inFlight) {
            const uint64_t waitValue  = n - inFlight;
            const vk::Semaphore waitOn = *timeline;
            const vk::Result waited   = device.waitSemaphores(
                vk::SemaphoreWaitInfo{.semaphoreCount = 1, .pSemaphores = &waitOn, .pValues = &waitValue},
                std::numeric_limits<uint64_t>::max());
            if (waited != vk::Result::eSuccess) {
                throw std::runtime_error("waitSemaphores failed");
            }
            if (traced) {
                const uint64_t counter = timeline.getCounterValue();
                std::println("Frame {}: slot {}, the CPU waits until the counter reaches {}, the submit will signal {}", n,
                             slot, waitValue, n);
                std::println("[timing] frame {}: the counter read after the wait: {}", n, counter);
            }
        } else if (traced) {
            std::println("Frame {}: slot {}, no earlier frame used this slot, the submit will signal {}", n, slot, n);
        }
        // #endregion

        // #region read-timestamps
        // Frame n - F, the last user of this slot, has completed (the wait
        // above), so its three timestamps are written and can be read before
        // they are reset for frame n. Ticks become microseconds through
        // timestampPeriod (nanoseconds per tick); differences are masked to
        // timestampValidBits, the bits the counter really has.
        if (n > inFlight) {
            const auto [result, ticks] = queryPool.getResults<uint64_t>(
                firstQuery, timestamps, timestamps * sizeof(uint64_t), sizeof(uint64_t),
                vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
            if (result != vk::Result::eSuccess) {
                throw std::runtime_error("getQueryPoolResults failed");
            }
            const auto us = [&](uint64_t from, uint64_t to) {
                return static_cast<double>((to - from) & gpu.mask) * gpu.periodNs / 1000.0;
            };
            gpu.add(us(ticks.at(0), ticks.at(1)), us(ticks.at(1), ticks.at(2)));
            const auto second = static_cast<uint64_t>(std::chrono::duration<double>(Clock::now() - start).count());
            if (second > gpu.second) {
                gpu.report(second);
            }
        }
        // #endregion

        // Acquire as in step 10 (the semaphore is binary).
        const vk::ResultValue<uint32_t> acquired =
            swapchain.acquireNextImage(std::numeric_limits<uint64_t>::max(), *frame.imageAvailable, nullptr);
        if (acquired.result == vk::Result::eErrorOutOfDateKHR) {
            return false;
        }
        const uint32_t imageIndex = acquired.value;
        const vk::Image image     = images.at(imageIndex);

        cmd.reset();
        cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        using S = vk::PipelineStageFlagBits2;
        using A = vk::AccessFlagBits2;
        using L = vk::ImageLayout;
        struct Scope {
            vk::PipelineStageFlags2 stage;
            vk::AccessFlags2 access;
        };
        const auto barrier = [](vk::Image img, Scope src, Scope dst, L oldLayout, L newLayout) {
            return vk::ImageMemoryBarrier2{
                .srcStageMask        = src.stage,
                .srcAccessMask       = src.access,
                .dstStageMask        = dst.stage,
                .dstAccessMask       = dst.access,
                .oldLayout           = oldLayout,
                .newLayout           = newLayout,
                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                .image               = img,
                .subresourceRange    = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1},
            };
        };

        // #region barriers
        // Five barriers, each "after THIS, before THAT" (stage + access on both
        // sides), recorded where an image changes hands:
        //
        // 1. Offscreen, eUndefined -> eColorAttachmentOptimal, before rendering.
        //    After: COLOR_ATTACHMENT_OUTPUT of earlier commands - the previous
        //    frame's color writes (write-after-write; barrier 3 already made
        //    them available) and, through barrier 5, the previous frame's blit.
        //    NOT after BLIT: the submit waits on imageAvailable at BLIT, and a
        //    source stage BLIT here would chain this barrier, and the whole
        //    rendering behind it, to the acquire. Before: the color writes.
        // 2. Swapchain image, eUndefined -> eTransferDstOptimal, before the blit.
        //    After: BLIT - the stage the submit waits on imageAvailable at, so
        //    the transition waits for the acquire. Before: the blit's writes.
        // 3. Offscreen, eColorAttachmentOptimal -> eTransferSrcOptimal.
        //    After: the color writes (made available). Before: the blit's reads
        //    (made visible). This is the render -> copy hand-over.
        // 4. Swapchain image, eTransferDstOptimal -> ePresentSrcKHR.
        //    After: the blit's writes. Before: BLIT, the stage renderFinished is
        //    signaled at, so the transition happens before the signal; the
        //    present itself needs no access mask.
        // 5. Offscreen, after the blit, no layout change. After: the blit's
        //    reads. Before: COLOR_ATTACHMENT_OUTPUT of every later command,
        //    which includes the NEXT frame's barrier 1 and rendering (one
        //    offscreen image for all frames in flight). A read followed by a
        //    write (write-after-read) needs only an execution dependency, so
        //    both access masks are eNone.
        const vk::ImageMemoryBarrier2 toRender =
            barrier(*offscreen, {S::eColorAttachmentOutput, A::eNone},
                    {S::eColorAttachmentOutput, A::eColorAttachmentWrite}, L::eUndefined, L::eColorAttachmentOptimal);
        const vk::ImageMemoryBarrier2 toBlitDst =
            barrier(image, {S::eBlit, A::eNone}, {S::eBlit, A::eTransferWrite}, L::eUndefined, L::eTransferDstOptimal);
        const vk::ImageMemoryBarrier2 toBlitSrc =
            barrier(*offscreen, {S::eColorAttachmentOutput, A::eColorAttachmentWrite}, {S::eBlit, A::eTransferRead},
                    L::eColorAttachmentOptimal, L::eTransferSrcOptimal);
        const vk::ImageMemoryBarrier2 toPresent =
            barrier(image, {S::eBlit, A::eTransferWrite}, {S::eBlit, A::eNone}, L::eTransferDstOptimal, L::ePresentSrcKHR);
        const vk::ImageMemoryBarrier2 afterBlit =
            barrier(*offscreen, {S::eBlit, A::eNone}, {S::eColorAttachmentOutput, A::eNone}, L::eTransferSrcOptimal,
                    L::eTransferSrcOptimal);
        // #endregion

        // #region render
        // Barrier 1, then the triangle into the offscreen image (400x300, so the
        // viewport and the scissor are 400x300 too), clearing it to the animated
        // color of step 10. Before it this slot's three queries are reset (a
        // query must be reset before it is written again) and timestamp 0 is
        // written at TOP_OF_PIPE: the moment the GPU starts this command buffer.
        // Timestamp 1, at COLOR_ATTACHMENT_OUTPUT after the rendering, is
        // written once the color writes are done.
        const float t   = static_cast<float>(std::chrono::duration<double>(Clock::now() - start).count());
        const auto wave = [t](float phase) { return 0.1f + 0.1f * std::sin(t * 1.5f + phase); };
        const vk::RenderingAttachmentInfo colorAttachment{
            .imageView   = *offscreenView,
            .imageLayout = L::eColorAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eStore,
            .clearValue  = vk::ClearColorValue{std::array{wave(0.0f), wave(2.1f), wave(4.2f), 1.0f}},
        };
        const vk::Rect2D area{.offset = {0, 0}, .extent = offscreenExtent};
        cmd.resetQueryPool(*queryPool, firstQuery, timestamps);
        cmd.writeTimestamp2(S::eTopOfPipe, *queryPool, firstQuery);
        cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toRender});
        cmd.beginRendering(vk::RenderingInfo{
            .renderArea           = area,
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &colorAttachment,
        });
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
        cmd.setViewport(0, vk::Viewport{
                               .x        = 0.0f,
                               .y        = 0.0f,
                               .width    = static_cast<float>(offscreenExtent.width),
                               .height   = static_cast<float>(offscreenExtent.height),
                               .minDepth = 0.0f,
                               .maxDepth = 1.0f,
                           });
        cmd.setScissor(0, area);
        cmd.draw(3, 1, 0, 0);
        cmd.endRendering();
        cmd.writeTimestamp2(S::eColorAttachmentOutput, *queryPool, firstQuery + 1);
        // #endregion

        // #region blit
        // Barriers 2 and 3 in one call (independent images), the blit, then
        // barriers 4 and 5 in one call. The blit copies the whole 400x300 image
        // onto the whole swapchain image, scaling it; eNearest keeps the pixels
        // square (2x2 screen pixels each in an 800x600 window). Timestamp 2
        // marks the end of the blit.
        const auto full = [](vk::Extent2D e) {
            return std::array{vk::Offset3D{0, 0, 0},
                              vk::Offset3D{static_cast<int32_t>(e.width), static_cast<int32_t>(e.height), 1}};
        };
        constexpr vk::ImageSubresourceLayers layers{.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1};
        const vk::ImageBlit2 region{
            .srcSubresource = layers,
            .srcOffsets     = full(offscreenExtent),
            .dstSubresource = layers,
            .dstOffsets     = full(extent),
        };
        const std::array toBlit{toBlitDst, toBlitSrc};
        cmd.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = static_cast<uint32_t>(toBlit.size()),
            .pImageMemoryBarriers    = toBlit.data(),
        });
        cmd.blitImage2(vk::BlitImageInfo2{
            .srcImage       = *offscreen,
            .srcImageLayout = L::eTransferSrcOptimal,
            .dstImage       = image,
            .dstImageLayout = L::eTransferDstOptimal,
            .regionCount    = 1,
            .pRegions       = &region,
            .filter         = vk::Filter::eNearest,
        });
        cmd.writeTimestamp2(S::eBlit, *queryPool, firstQuery + 2);
        const std::array afterBlitBarriers{toPresent, afterBlit};
        cmd.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = static_cast<uint32_t>(afterBlitBarriers.size()),
            .pImageMemoryBarriers    = afterBlitBarriers.data(),
        });
        cmd.end();
        // #endregion

        // #region submit
        // One batch, three semaphore operations:
        // - wait imageAvailable (binary) at BLIT: only barrier 2 (source stage
        //   BLIT, so it chains to the wait) and the blit must wait. Barrier 1
        //   has no BLIT in its source stage, so rendering into the offscreen
        //   image may run before the swapchain image is ours;
        // - signal renderFinished (binary) at BLIT, for the present;
        // - signal the timeline with value n at ALL_COMMANDS: the counter becomes
        //   n once everything in the batch has completed. This replaces step 10's
        //   fence argument, which is null now.
        // The value of a binary semaphore operation is ignored.
        const vk::Semaphore signalSemaphore = *renderFinished.at(imageIndex);
        const vk::SemaphoreSubmitInfo waitInfo{.semaphore = *frame.imageAvailable, .stageMask = S::eBlit};
        const std::array signalInfos{
            vk::SemaphoreSubmitInfo{.semaphore = signalSemaphore, .stageMask = S::eBlit},
            vk::SemaphoreSubmitInfo{.semaphore = *timeline, .value = n, .stageMask = S::eAllCommands},
        };
        const vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *cmd};
        queue.submit2(vk::SubmitInfo2{
            .waitSemaphoreInfoCount   = 1,
            .pWaitSemaphoreInfos      = &waitInfo,
            .commandBufferInfoCount   = 1,
            .pCommandBufferInfos      = &cmdInfo,
            .signalSemaphoreInfoCount = static_cast<uint32_t>(signalInfos.size()),
            .pSignalSemaphoreInfos    = signalInfos.data(),
        });
        frameNumber = n;
        // #endregion

        // Present as in step 10: it waits on the binary renderFinished.
        const vk::SwapchainKHR swapchainHandle = *swapchain;
        const vk::Result presented             = queue.presentKHR(vk::PresentInfoKHR{
                        .waitSemaphoreCount = 1,
                        .pWaitSemaphores    = &signalSemaphore,
                        .swapchainCount     = 1,
                        .pSwapchains        = &swapchainHandle,
                        .pImageIndices      = &imageIndex,
        });
        return presented != vk::Result::eErrorOutOfDateKHR;
    };

    // #region event-loop
    // From here on the GPU runs our frames: wait for it on every way out.
    const IdleGuard idleGuard{device};
    std::println("\nRendering. Close the window to exit.");
    const auto timeUp = [&] {
        return args.seconds && std::chrono::duration<double>(Clock::now() - start).count() >= *args.seconds;
    };
    while (glfwWindowShouldClose(window.get()) == GLFW_FALSE && !timeUp()) {
        glfwPollEvents();
        if (!drawFrame()) {
            std::println("Swapchain out of date (the window changed); recreation is step 11, stopping");
            break;
        }
    }
    std::println("Window closed");
    // #endregion

    // #region cleanup
    // The timeline tells when OUR work is done: wait for the last frame's value.
    // Destroying the swapchain also needs the presents to be done
    // (VUID-vkDestroySwapchainKHR-swapchain-01282), so vkDeviceWaitIdle stays:
    // it waits for queue work, the presents' queue operations included; it does
    // not wait for the presentation engine (present fences, step 11). After it
    // the counter equals the number of submitted frames. Then RAII
    // destroys everything in reverse order of declaration.
    const vk::Semaphore last = *timeline;
    if (device.waitSemaphores(vk::SemaphoreWaitInfo{.semaphoreCount = 1, .pSemaphores = &last, .pValues = &frameNumber},
                              std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
        throw std::runtime_error("waitSemaphores failed");
    }
    device.waitIdle();
    std::println("[timing] timeline counter at exit: {} ({} frames submitted)", timeline.getCounterValue(), frameNumber);
    // #endregion
    return EXIT_SUCCESS;
} catch (const vk::SystemError& e) {
    std::println(stderr, "Vulkan error: {}", e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return EXIT_FAILURE;
}
