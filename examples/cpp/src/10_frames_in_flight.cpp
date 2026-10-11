// Step 10 - Frames in flight
//
// Step 09 kept ONE frame in flight: after submitting frame N the CPU waited for
// the GPU to finish it before recording frame N+1, so the CPU and the GPU took
// turns. Here every resource the CPU touches while recording - the command
// buffer, the fence, the acquire semaphore - exists once PER FRAME IN FLIGHT, so
// the CPU records frame N+1 while the GPU still renders frame N. Like double
// buffering in a VFX pipeline: while one batch renders, the next one is being
// prepared.
//
//   frame slot = frame number % frames in flight
//   wait the slot's fence -> acquire -> record the slot's buffer -> submit -> present
//
// The clear color is animated on the CPU (a value written into the command
// buffer every frame), and once a second the program prints how long the CPU
// waited on fences. --frames 1|2|3 (default 2) compares the counts; --cpu-ms and
// --draws set the simulated CPU and GPU work per frame; --seconds S closes the
// window after S seconds. Lines that depend on timing start with [timing].
// Steps 01-09 are repeated, reduced to what this step needs.

// By default Vulkan-Hpp throws vk::OutOfDateKHRError from acquireNextImage and
// presentKHR. A window that changed is an expected event, not an error, so we
// get it back as a vk::Result instead (the Rust version matches on it too).
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

// Waits for the device to go idle when it leaves scope. Declared after the
// last object the frames use, it is destroyed first, so even when a frame
// throws, RAII never destroys an object the GPU may still be using (the Rust
// version waits in Vk::destroy, which runs after an error too).
struct IdleGuard {
    const vk::raii::Device& device;
    ~IdleGuard() {
        // A destructor must not throw: call the C function, which returns the
        // error instead, and report it.
        if (const VkResult r = device.getDispatcher()->vkDeviceWaitIdle(*device); r != VK_SUCCESS) {
            std::fprintf(stderr, "vkDeviceWaitIdle failed: %s\n", string_VkResult(r));
        }
    }
};

// #region cli
// Command line:
// - --frames N: frames in flight, 1-3 (default 2);
// - --cpu-ms X: simulated CPU work per frame in milliseconds (default 2);
// - --draws N: instances of the triangle per frame, the GPU work (default 5000);
// - --seconds S: close the window after S seconds (default: run until closed).
//
// A triangle alone costs the CPU and the GPU almost nothing, and frames in
// flight only matter when both sides have real work: the CPU builds the frame
// (game logic, culling, recording), the GPU renders it. The two knobs stand in
// for that work, so the effect can be measured.
struct Args {
    size_t frames = 2;
    double cpuMs  = 2.0;
    uint32_t draws = 5000;
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
        } else if (arg == "--cpu-ms") {
            args.cpuMs = parseValue<double>(arg, value);
        } else if (arg == "--draws") {
            args.draws = parseValue<uint32_t>(arg, value);
        } else if (arg == "--seconds") {
            args.seconds = parseValue<double>(arg, value);
        } else {
            throw std::runtime_error("unknown argument " + std::string{arg} +
                                     " (use --frames 1|2|3, --cpu-ms X, --draws N, --seconds S)");
        }
    }
    if (args.frames < 1 || args.frames > 3) {
        throw std::runtime_error("--frames must be 1, 2 or 3");
    }
    if (args.cpuMs < 0.0 || args.cpuMs > 100.0 || args.draws == 0) {
        throw std::runtime_error("--cpu-ms must be 0-100 and --draws at least 1");
    }
    return args;
}
// #endregion

// #region frame-resources
// What the CPU touches while recording one frame: one set per frame in flight.
// Frame N uses slot N % frames-in-flight, so frame N+1 records into another
// command buffer while the GPU may still execute the one of frame N.
struct Frame {
    vk::raii::CommandBuffer commandBuffer;
    // Signaled by vkAcquireNextImageKHR, waited on by this slot's submit.
    vk::raii::Semaphore imageAvailable;
    // Signaled when this slot's submission completes: the CPU waits on it
    // before reusing the command buffer and the semaphore.
    vk::raii::Fence inFlight;
};
// #endregion

// #region stats
// CPU-side frame statistics: how long the CPU blocked in vkWaitForFences
// (waiting for the GPU) and in vkAcquireNextImageKHR (waiting for the
// presentation engine), printed once a second and as a total at the end.
struct Stats {
    using Clock    = std::chrono::steady_clock;
    using Seconds  = std::chrono::duration<double>;
    Clock::time_point start       = Clock::now();
    Clock::time_point windowStart = start;
    uint32_t windowFrames         = 0;
    Seconds windowFence{};
    Seconds windowAcquire{};
    uint32_t seconds = 0;
    uint64_t frames  = 0;
    Seconds fence{};
    Seconds acquire{};

    // Adds one frame; prints a line when a second has passed.
    void add(Seconds fenceWait, Seconds acquireWait) {
        ++windowFrames;
        windowFence += fenceWait;
        windowAcquire += acquireWait;
        ++frames;
        fence += fenceWait;
        acquire += acquireWait;
        const double elapsed = Seconds(Clock::now() - windowStart).count();
        if (elapsed < 1.0) {
            return;
        }
        ++seconds;
        const double n = windowFrames;
        std::println("[timing] {}s: {} frames, {:.0f} fps; fence wait {:.3f} ms/frame ({:.1f}% of the time), acquire {:.3f} ms/frame",
                     seconds, windowFrames, n / elapsed, windowFence.count() * 1e3 / n, windowFence.count() * 100.0 / elapsed,
                     windowAcquire.count() * 1e3 / n);
        windowStart   = Clock::now();
        windowFrames  = 0;
        windowFence   = {};
        windowAcquire = {};
    }

    void printTotal(size_t framesInFlight) const {
        const double elapsed = Seconds(Clock::now() - start).count();
        const double n       = static_cast<double>(std::max<uint64_t>(frames, 1));
        std::println("[timing] total with {} frame(s) in flight: {} frames in {:.1f} s, {:.0f} fps; fence wait {:.3f} ms/frame "
                     "({:.1f}% of the time), acquire {:.3f} ms/frame",
                     framesInFlight, frames, elapsed, static_cast<double>(frames) / elapsed, fence.count() * 1e3 / n,
                     fence.count() * 100.0 / elapsed, acquire.count() * 1e3 / n);
    }
};
// #endregion

int main(int argc, char** argv)
try {
    const Args args = parseArgs(argc, argv);

    // ---- Step 06: window (800x600 physical pixels, fixed size) ----
    Glfw glfw;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 10 frames in flight", nullptr, nullptr), &glfwDestroyWindow);
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
        .pApplicationName   = "Vulkan from Zero - 10 frames in flight",
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

    // The minimum is what the presentation engine needs for itself; one more
    // lets us render into a free image while the others are on screen or
    // queued. maxImageCount 0 means "no limit".
    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) {
        imageCount = std::min(imageCount, caps.maxImageCount);
    }
    std::println("Image count: {} (min {} + 1, max {})", imageCount, caps.minImageCount, caps.maxImageCount);

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

    // The SPIR-V is compiled from shaders/triangle.slang by `bun run sync:shaders`
    // (one module, two entry points). CMake bakes the absolute path of
    // shaders/spv into SHADER_DIR; we read the same file the Rust version embeds.
    const std::vector<uint32_t> spirv = readSpirv(std::filesystem::path{SHADER_DIR} / "triangle.spv");
    std::println("\nSPIR-V: {} bytes (triangle.spv)", spirv.size() * sizeof(uint32_t));

    // A shader module is just the SPIR-V handed to the driver; nothing is
    // compiled for the GPU yet. That happens in vkCreateGraphicsPipelines, when
    // the driver knows the whole pipeline state. The module only has to live
    // until then, hence the optional (reset right after the pipeline exists).
    std::optional<vk::raii::ShaderModule> shaderModule;
    shaderModule.emplace(device, vk::ShaderModuleCreateInfo{
                                     .codeSize = spirv.size() * sizeof(uint32_t), // in BYTES
                                     .pCode    = spirv.data(),
                                 });
    std::println("Shader module created: entry points vertexMain, fragmentMain");

    // One module, two stages: each stage names its entry point in the module.
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{
            .stage  = vk::ShaderStageFlagBits::eVertex,
            .module = **shaderModule,
            .pName  = "vertexMain",
        },
        vk::PipelineShaderStageCreateInfo{
            .stage  = vk::ShaderStageFlagBits::eFragment,
            .module = **shaderModule,
            .pName  = "fragmentMain",
        },
    };

    // Vertex input: no vertex buffers yet, the shader makes up its corners.
    const vk::PipelineVertexInputStateCreateInfo vertexInput{};
    // Input assembly: every 3 vertices form one triangle.
    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
    // Viewport + scissor: one of each; their values are set while recording
    // (dynamic state below), only the count is baked in.
    const vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
    // Rasterizer: filled triangles; back faces culled, and clockwise on screen
    // counts as front (our triangle is clockwise, see triangle.slang).
    const vk::PipelineRasterizationStateCreateInfo rasterization{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode    = vk::CullModeFlagBits::eBack,
        .frontFace   = vk::FrontFace::eClockwise,
        .lineWidth   = 1.0f,
    };
    // One sample per pixel: no MSAA (module 10).
    const vk::PipelineMultisampleStateCreateInfo multisample{.rasterizationSamples = vk::SampleCountFlagBits::e1};
    // Color blend, one entry per color attachment: no blending, write RGBA.
    const vk::PipelineColorBlendAttachmentState blendAttachment{
        .blendEnable    = vk::False,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
    const vk::PipelineColorBlendStateCreateInfo colorBlend{.attachmentCount = 1, .pAttachments = &blendAttachment};

    // Viewport and scissor are dynamic: one pipeline works for any window size,
    // and recreating the swapchain (module 7) needs no new pipeline.
    constexpr std::array dynamicStates{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates    = dynamicStates.data(),
    };

    // The layout declares what resources the shaders read: descriptor set
    // layouts and push-constant ranges. Our shaders read none, so it is empty -
    // but every pipeline still needs one.
    vk::raii::PipelineLayout pipelineLayout(device, vk::PipelineLayoutCreateInfo{});
    std::println("Pipeline layout created: 0 descriptor set layouts, 0 push constant ranges");

    // Dynamic rendering (core in 1.3): instead of a VkRenderPass, the pipeline
    // only states the FORMATS of the attachments it will draw into.
    // vkCmdBeginRendering supplies the actual image views when recording.
    const vk::PipelineRenderingCreateInfo renderingInfo{
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &format.format,
    };

    // Everything together. renderPass stays null: the rendering info in the
    // pNext chain replaces it. Here the driver compiles SPIR-V into GPU code for
    // exactly this state, which is why pipelines are created up front.
    const vk::GraphicsPipelineCreateInfo pipelineInfo{
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
    };
    // nullptr = no pipeline cache yet.
    vk::raii::Pipeline pipeline(device, nullptr, pipelineInfo);
    std::println("Graphics pipeline created for color format {} (dynamic rendering, no render pass)",
                 shortName(string_VkFormat(static_cast<VkFormat>(format.format)), "VK_FORMAT_"));
    // The pipeline keeps its own compiled code: the module is no longer needed.
    shaderModule.reset();
    std::println("Shader module destroyed");

    // #region get-queue
    // The one queue of step 05: it records nothing itself, it executes command
    // buffers (vkQueueSubmit2) and presents (vkQueuePresentKHR).
    vk::raii::Queue queue(device, family, 0);
    // #endregion

    // #region command-pool
    // Command buffers are allocated from a pool tied to ONE queue family (the
    // memory they use is managed per family). eResetCommandBuffer lets us reset
    // and re-record the same buffer every frame.
    vk::raii::CommandPool commandPool(device, vk::CommandPoolCreateInfo{
                                                  .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                  .queueFamilyIndex = family,
                                              });
    // #endregion

    // #region command-buffers
    // One ePrimary command buffer per frame in flight: while the GPU executes
    // the buffer of frame N, the CPU records frame N+1 into another one (a
    // pending buffer must not be reset or re-recorded).
    vk::raii::CommandBuffers commandBuffers(device, vk::CommandBufferAllocateInfo{
                                                        .commandPool        = *commandPool,
                                                        .level              = vk::CommandBufferLevel::ePrimary,
                                                        .commandBufferCount = static_cast<uint32_t>(args.frames),
                                                    });
    std::println("\nCommand pool + {} primary command buffer(s)", commandBuffers.size());
    // #endregion

    // #region sync-objects
    // Semaphores order GPU work against GPU work (acquire -> render -> present);
    // a fence lets the CPU wait for the GPU.
    // - imageAvailable, one per FRAME IN FLIGHT: signaled by the acquire, waited
    //   on by the submit of the same frame. The slot's fence tells when that
    //   wait has happened, so the semaphore may be reused.
    // - renderFinished, one per SWAPCHAIN IMAGE: waited on by the present. The
    //   presentation engine gives no signal when it is done with it; the same
    //   image coming back from vkAcquireNextImageKHR is the sign that its last
    //   present has finished (step 11 shows present fences).
    // - inFlight, one per frame in flight: signaled when the slot's submission
    //   completes. Created signaled so the first wait returns.
    std::vector<Frame> frames;
    frames.reserve(args.frames);
    for (vk::raii::CommandBuffer& commandBuffer : commandBuffers) {
        frames.push_back(Frame{
            .commandBuffer  = std::move(commandBuffer),
            .imageAvailable = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{}),
            .inFlight       = vk::raii::Fence(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled}),
        });
    }
    std::vector<vk::raii::Semaphore> renderFinished;
    renderFinished.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        renderFinished.emplace_back(device, vk::SemaphoreCreateInfo{});
    }
    std::println("Sync objects: {} fence(s), {} + {} semaphores ({} frame(s) in flight)", frames.size(), frames.size(),
                 renderFinished.size(), frames.size());
    // #endregion

    Stats stats;
    // Counts all frames; frameIndex % frames.size() is the current slot.
    size_t frameIndex = 0;

    // Renders and presents one frame. Returns false when the swapchain no longer
    // matches the window (VK_ERROR_OUT_OF_DATE_KHR): recreating it is step 11,
    // here the caller simply stops.
    const auto drawFrame = [&]() -> bool {
        Frame& frame                 = frames.at(frameIndex % frames.size());
        const vk::raii::CommandBuffer& cmd = frame.commandBuffer;

        // #region wait-fence
        // Wait until the LAST submission that used this slot has finished on the
        // GPU (with 2 frames in flight: frame N-2, not N-1): only then may we
        // re-record its command buffer and reuse its imageAvailable. Frame N-1
        // may still be rendering - that overlap is the point. The fence is reset
        // only once we are sure to submit (after a successful acquire), or the
        // next wait on it would never return.
        const auto waitStart = Stats::Clock::now();
        if (device.waitForFences(*frame.inFlight, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
            throw std::runtime_error("waitForFences failed");
        }
        const Stats::Seconds fenceWait = Stats::Clock::now() - waitStart;
        // #endregion

        // #region acquire
        // Ask the presentation engine for a free image. The call returns an index
        // at once, but the image may still be on screen: imageAvailable is
        // signaled when it is really ours. eSuboptimalKHR still works, the image
        // just no longer matches the surface exactly. VK_ERROR_OUT_OF_DATE_KHR
        // comes back as a result, not an exception (see the #define at the top).
        const auto acquireStart = Stats::Clock::now();
        const vk::ResultValue<uint32_t> acquired =
            swapchain.acquireNextImage(std::numeric_limits<uint64_t>::max(), *frame.imageAvailable, nullptr);
        const Stats::Seconds acquireWait = Stats::Clock::now() - acquireStart;
        if (acquired.result == vk::Result::eErrorOutOfDateKHR) {
            return false;
        }
        const uint32_t imageIndex = acquired.value;
        const vk::Image image     = images.at(imageIndex);
        device.resetFences(*frame.inFlight);
        // #endregion

        // #region cpu-work
        // Stand-in for the CPU half of a real frame (game logic, culling, filling
        // this frame's buffers): a busy loop of --cpu-ms. It runs after the fence
        // wait because real per-frame work writes per-frame resources. With one
        // frame in flight the GPU idles meanwhile; with two it renders the
        // previous frame.
        const auto cpuWork   = Stats::Seconds(args.cpuMs / 1e3);
        const auto workStart = Stats::Clock::now();
        while (Stats::Clock::now() - workStart < cpuWork) {
        }
        // #endregion

        // #region begin-recording
        // Recording only writes commands into the buffer; the GPU sees nothing
        // until the submit. eOneTimeSubmit: we re-record it every frame.
        cmd.reset();
        cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        // #endregion

        constexpr vk::ImageSubresourceRange colorRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .layerCount = 1,
        };

        // #region barrier-to-attachment
        // Image layout = how the image's memory is arranged for one kind of use.
        // eUndefined -> eColorAttachmentOptimal: we will clear it anyway, so the
        // old contents may be discarded. The source stage is the same stage the
        // submit waits on imageAvailable at (COLOR_ATTACHMENT_OUTPUT), which
        // chains the two: the transition starts only after the image is ours.
        const vk::ImageMemoryBarrier2 toAttachment{
            .srcStageMask        = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask       = vk::AccessFlagBits2::eNone,
            .dstStageMask        = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask       = vk::AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout           = vk::ImageLayout::eUndefined,
            .newLayout           = vk::ImageLayout::eColorAttachmentOptimal,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image               = image,
            .subresourceRange    = colorRange,
        };
        cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toAttachment});
        // #endregion

        // #region animate
        // The animation lives in the command buffer: the clear color is a value
        // recorded into vkCmdBeginRendering, so every frame records its own. This
        // is why a buffer that is still pending on the GPU must not be
        // re-recorded - and why each frame in flight has its own buffer. Three
        // phase-shifted sines of the time since start, dark linear values.
        const float t    = static_cast<float>(Stats::Seconds(Stats::Clock::now() - stats.start).count());
        const auto wave  = [t](float phase) { return 0.1f + 0.1f * std::sin(t * 1.5f + phase); };
        const std::array clear{wave(0.0f), wave(2.1f), wave(4.2f), 1.0f};
        // #endregion

        // #region begin-rendering
        // Dynamic rendering: the attachments are named right here. eClear at the
        // start (the animated color), eStore at the end so the pixels survive for
        // presentation.
        const vk::RenderingAttachmentInfo colorAttachment{
            .imageView   = *imageViews.at(imageIndex),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eStore,
            .clearValue  = vk::ClearColorValue{clear},
        };
        cmd.beginRendering(vk::RenderingInfo{
            .renderArea           = {.offset = {0, 0}, .extent = extent},
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &colorAttachment,
        });
        // #endregion

        // #region draw
        // Bind the pipeline, fill in its dynamic state (the whole image), and
        // draw 3 vertices of --draws instances. The shader ignores the instance
        // index, so all of them land on the same pixels: the GPU half of the
        // frame, pure fill work.
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
        cmd.setViewport(0, vk::Viewport{
                               .x        = 0.0f,
                               .y        = 0.0f,
                               .width    = static_cast<float>(extent.width),
                               .height   = static_cast<float>(extent.height),
                               .minDepth = 0.0f,
                               .maxDepth = 1.0f,
                           });
        cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = extent});
        cmd.draw(3, args.draws, 0, 0);
        cmd.endRendering();
        // #endregion

        // #region barrier-to-present
        // eColorAttachmentOptimal -> ePresentSrcKHR, after the color writes. The
        // destination stage matches the stage renderFinished is signaled at
        // (COLOR_ATTACHMENT_OUTPUT), so the transition is ordered before the
        // signal; the present waits on that semaphore, and needs no access mask:
        // presentation makes the writes visible itself.
        const vk::ImageMemoryBarrier2 toPresent{
            .srcStageMask        = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask       = vk::AccessFlagBits2::eColorAttachmentWrite,
            .dstStageMask        = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask       = vk::AccessFlagBits2::eNone,
            .oldLayout           = vk::ImageLayout::eColorAttachmentOptimal,
            .newLayout           = vk::ImageLayout::ePresentSrcKHR,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image               = image,
            .subresourceRange    = colorRange,
        };
        cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toPresent});
        cmd.end();
        // #endregion

        // #region submit
        // One batch: wait for imageAvailable before COLOR_ATTACHMENT_OUTPUT
        // (earlier stages - vertex shading - may already run), execute `cmd`,
        // then signal renderFinished; the slot's inFlight fence is signaled when
        // it all completes.
        const vk::Semaphore signalSemaphore = *renderFinished.at(imageIndex);
        const vk::SemaphoreSubmitInfo waitInfo{
            .semaphore = *frame.imageAvailable,
            .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        };
        const vk::SemaphoreSubmitInfo signalInfo{
            .semaphore = signalSemaphore,
            .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        };
        const vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *cmd};
        queue.submit2(vk::SubmitInfo2{
                          .waitSemaphoreInfoCount   = 1,
                          .pWaitSemaphoreInfos      = &waitInfo,
                          .commandBufferInfoCount   = 1,
                          .pCommandBufferInfos      = &cmdInfo,
                          .signalSemaphoreInfoCount = 1,
                          .pSignalSemaphoreInfos    = &signalInfo,
                      },
                      *frame.inFlight);
        // #endregion

        // #region present
        // Hand the image back to the presentation engine, to be shown once
        // renderFinished is signaled. eSuboptimalKHR is still shown.
        const vk::SwapchainKHR swapchainHandle = *swapchain;
        const vk::Result presented             = queue.presentKHR(vk::PresentInfoKHR{
                        .waitSemaphoreCount = 1,
                        .pWaitSemaphores    = &signalSemaphore,
                        .swapchainCount     = 1,
                        .pSwapchains        = &swapchainHandle,
                        .pImageIndices      = &imageIndex,
        });
        if (presented == vk::Result::eErrorOutOfDateKHR) {
            return false;
        }
        // #endregion

        // #region advance-frame
        // Next frame, next slot: the CPU goes on recording while the GPU still
        // works on this one. Only the wait-fence of the NEXT use of a slot brings
        // them back in step.
        ++frameIndex;
        stats.add(fenceWait, acquireWait);
        return true;
        // #endregion
    };

    // #region event-loop
    // glfwPollEvents handles the window's messages without blocking (step 06
    // waited for them), then one frame is rendered. The fence waits and the
    // present mode (MAILBOX/FIFO) pace the loop.
    // From here on the GPU runs our frames: wait for it on every way out.
    const IdleGuard idleGuard{device};
    std::println("Simulated work per frame: {} ms on the CPU, {} triangle instances on the GPU", args.cpuMs, args.draws);
    std::println("\nRendering. Close the window to exit.");
    const auto timeUp = [&] {
        return args.seconds && Stats::Seconds(Stats::Clock::now() - stats.start).count() >= *args.seconds;
    };
    while (glfwWindowShouldClose(window.get()) == GLFW_FALSE && !timeUp()) {
        glfwPollEvents();
        if (!drawFrame()) {
            // Recreating the swapchain is step 11.
            std::println("Swapchain out of date (the window changed); recreation is step 11, stopping");
            break;
        }
    }
    std::println("Window closed");
    stats.printTotal(args.frames);
    // #endregion

    // #region cleanup
    // First wait until the GPU is idle: with several frames in flight more than
    // one submission may still be running, and nothing they use may be
    // destroyed before. The wait covers all work submitted to the queues, which
    // is what destroying the swapchain needs
    // (VUID-vkDestroySwapchainKHR-swapchain-01282); it does not wait for the
    // presentation engine (step 11). This explicit wait reports an error as an
    // exception; idleGuard (declared before the loop) also waits on its way out,
    // which covers the paths where a frame threw. Then RAII destroys in reverse
    // order of declaration: the per-frame objects, the pool, then everything
    // from steps 07-08.
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