// Step 27 - Async compute: the simulation on its own queue
//
// Step 26 put the dispatch and the draw into one command buffer: the GPU runs
// them one after the other. Here the simulation moves to a queue of a family
// WITHOUT graphics (family 2 on the RTX 3080 Ti, 8 queues), so the GPU may run
// "compute step n+1" while "graphics frame n" draws:
//
//   compute queue:  [step 1] [step 2]  [step 3]  [step 4] ...
//   graphics queue:          [frame 1] [frame 2] [frame 3] ...
//
// Frame n draws state n while step n+1 writes state n+1, so the states need
// TWO buffers (ping-pong): step n writes P[n % 2] and reads P[(n-1) % 2]. Two
// timeline semaphores order them: computed (step n done = n) and rendered
// (frame n done = n). The buffers are shared by two queue families with
// VK_SHARING_MODE_CONCURRENT. GPU timestamps on both queues measure whether the
// work really overlaps; --same-queue submits the compute work to the graphics
// queue instead, for comparison. --count N particles, --seconds S; lines that
// depend on timing start with [timing]. Steps 01-26 are repeated, reduced to
// what this step needs.

// Step 10: VK_ERROR_OUT_OF_DATE_KHR comes back as a vk::Result, not an exception.
#define VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS

#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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

// Step 08: reads a .spv file into 32-bit words.
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

// Step 10's measurements showed that 2 frames in flight keep both the CPU and
// the GPU busy; from here on it is fixed. Here it applies to both queues: up
// to 2 graphics and 2 compute submissions are pending.
constexpr uint64_t framesInFlight = 2;

// #region cli
// Command line: --count N particles (default 2^20), --seconds S, --same-queue.
struct Args {
    uint32_t count = 1u << 20;
    std::optional<double> seconds;
    bool sameQueue = false;
};

Args parseArgs(int argc, char** argv) {
    constexpr std::string_view usage = "usage: 27_async_compute [--count N] [--seconds S] [--same-queue]";
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--same-queue") {
            args.sameQueue = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::runtime_error(std::string{usage});
        }
        const std::string_view text = argv[++i];
        std::from_chars_result r{};
        if (flag == "--count") {
            r = std::from_chars(text.data(), text.data() + text.size(), args.count);
        } else if (flag == "--seconds") {
            double value = 0.0;
            r            = std::from_chars(text.data(), text.data() + text.size(), value);
            args.seconds = value;
        } else {
            throw std::runtime_error(std::string{usage});
        }
        if (r.ec != std::errc{} || r.ptr != text.data() + text.size()) {
            throw std::runtime_error(std::string{usage});
        }
    }
    if (args.count == 0 || args.count > (1u << 24)) {
        throw std::runtime_error("--count must be 1-16777216");
    }
    return args;
}
// #endregion

// One particle as shaders/particles.slang declares it (step 26).
struct Particle {
    std::array<float, 2> pos;
    std::array<float, 2> vel;
};
static_assert(sizeof(Particle) == 16);

// The push constant block of particles.slang (step 26).
struct Params {
    float dt;
    float time;
    uint32_t count;
    float aspect;
};
static_assert(sizeof(Params) == 16);

// Step 13's find-memory-type rule.
std::optional<uint32_t> findType(const vk::PhysicalDeviceMemoryProperties& props, uint32_t bits,
                                 vk::MemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) != 0 && (props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return std::nullopt;
}

// Decodes a SPIR-V literal string: UTF-8 bytes packed little-endian into
// words, NUL-terminated (step 17).
std::string spvString(std::span<const uint32_t> words) {
    std::string out;
    for (const uint32_t w : words) {
        for (int i = 0; i < 4; ++i) {
            const auto c = static_cast<char>((w >> (8 * i)) & 0xffu);
            if (c == '\0') {
                return out;
            }
            out += c;
        }
    }
    return out;
}

// Step 25's reflection: the LocalSize x of the 1D compute entry point entry.
uint32_t localSizeX(std::span<const uint32_t> words, std::string_view entry) {
    constexpr uint32_t opEntryPoint    = 15;
    constexpr uint32_t opExecutionMode = 16;
    constexpr uint32_t modeLocalSize   = 17;
    std::optional<uint32_t> function;
    std::vector<std::pair<uint32_t, uint32_t>> sizes;
    size_t pos = 5;
    while (pos < words.size()) {
        const uint32_t first = words[pos];
        const size_t count   = first >> 16;
        if (count == 0 || pos + count > words.size()) {
            throw std::runtime_error("malformed SPIR-V instruction");
        }
        const std::span<const uint32_t> ins = words.subspan(pos, count);
        const uint32_t op                   = first & 0xffffu;
        if (op == opEntryPoint && count >= 4 && spvString(ins.subspan(3)) == entry) {
            function = ins[2];
        } else if (op == opExecutionMode && count == 6 && ins[2] == modeLocalSize && ins[4] == 1 && ins[5] == 1) {
            sizes.emplace_back(ins[1], ins[3]);
        }
        pos += count;
    }
    if (!function) {
        throw std::runtime_error(std::format("no entry point {}", entry));
    }
    const auto found = std::ranges::find_if(sizes, [&](const auto& s) { return s.first == *function; });
    if (found == sizes.end()) {
        throw std::runtime_error(std::format("{} has no 1D LocalSize", entry));
    }
    return found->second;
}

// A buffer and the memory bound to it (step 14). Members are destroyed in
// reverse order: the buffer first, then its memory.
struct GpuBuffer {
    vk::raii::DeviceMemory memory;
    vk::raii::Buffer buffer;
    uint32_t memoryType;
};

// #region timestamps
// Two timestamps per submission: start and end, on both queues.
constexpr uint32_t timestamps = 2;
using Ticks                   = std::array<uint64_t, 2>;

// GPU time of one graphics frame and one compute step, accumulated over a
// second and over the whole run.
struct GpuTimes {
    double periodNs = 1.0; // nanoseconds per tick (timestampPeriod)
    uint32_t bits   = 64;  // the valid bits common to both families' timestamps (1-64)
    uint64_t second = 0;
    struct Sums {
        uint64_t frames = 0;
        double graphics = 0.0, compute = 0.0, overlap = 0.0; // microseconds
    };
    Sums current{}; // the current second
    Sums total{};   // the whole run

    // Adds graphics frame n [g0, g1] and compute step n+1 [c0, c1]. All device
    // timestamps are in one time domain (VK_TIME_DOMAIN_DEVICE_KHR), so values
    // from the two queues can be compared: the overlap is the length of the
    // intersection of the two intervals. Only the low bits of a timestamp are
    // meaningful and the counter wraps there, so every value is taken as a
    // signed tick offset from g0, computed modulo 2^bits (right while the four
    // lie less than 2^(bits-1) ticks apart).
    void add(const Ticks& g, const Ticks& c) {
        const uint32_t shift = 64 - bits;
        const auto fromG0    = [&](uint64_t t) { return static_cast<int64_t>((t - g[0]) << shift) >> shift; };
        const auto us        = [&](int64_t ticks) { return static_cast<double>(ticks) * periodNs / 1000.0; };
        const int64_t g1 = fromG0(g[1]), c0 = fromG0(c[0]), c1 = fromG0(c[1]);
        const double graphic = us(g1);
        const double comp    = us(c1 - c0);
        const double overlap = us(std::max<int64_t>(std::min(g1, c1) - std::max<int64_t>(c0, 0), 0));
        for (Sums* s : {&current, &total}) {
            s->frames += 1;
            s->graphics += graphic;
            s->compute += comp;
            s->overlap += overlap;
        }
    }

    // Prints one [timing] line for label from sums.
    static void print(std::string_view label, const Sums& s) {
        if (s.frames == 0) {
            return;
        }
        const auto n = static_cast<double>(s.frames);
        std::println("[timing] {}: {} frames, GPU time per frame: graphics {:.1f} us, compute {:.1f} us, overlapping {:.1f} us ({:.0f}% "
                     "of the compute time)",
                     label, s.frames, s.graphics / n, s.compute / n, s.overlap / n,
                     100.0 * s.overlap / std::max(s.compute, std::numeric_limits<double>::min()));
    }

    // Prints the current second once it is over.
    void tick(uint64_t now) {
        if (now > second) {
            print(std::format("{}s", second + 1), current);
            current = {};
            second  = now;
        }
    }
};
// #endregion

// One frame slot (step 10), now with a command buffer on each queue.
struct Frame {
    vk::raii::CommandBuffer graphicsCmd;
    vk::raii::CommandBuffer computeCmd;
    // Signaled by vkAcquireNextImageKHR, waited on by this slot's graphics submit.
    vk::raii::Semaphore imageAvailable;
};

int main(int argc, char** argv)
try {
    const Args args = parseArgs(argc, argv);

    // ---- Step 06: window (800x600 physical pixels, fixed size) ----
    Glfw glfw;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 27 async compute", nullptr, nullptr), &glfwDestroyWindow);
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
        .pApplicationName   = "Vulkan from Zero - 27 async compute",
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

    // Steps 03 + 06 + 12: Vulkan 1.4, dynamic rendering, synchronization2,
    // timelineSemaphore, a graphics + present family, VK_KHR_swapchain.
    const vk::raii::PhysicalDevice* best = nullptr;
    int bestScore                        = 0;
    uint32_t graphicsFamily              = 0;
    for (const vk::raii::PhysicalDevice& device : devices) {
        const vk::PhysicalDeviceProperties props = device.getProperties();
        if (props.apiVersion < vk::ApiVersion14) {
            continue;
        }
        const auto featureChain = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,
                                                      vk::PhysicalDeviceVulkan13Features>();
        const auto& f12         = featureChain.get<vk::PhysicalDeviceVulkan12Features>();
        const auto& f13         = featureChain.get<vk::PhysicalDeviceVulkan13Features>();
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
            bestScore      = score;
            best           = &device;
            graphicsFamily = *candidate;
        }
    }
    if (best == nullptr) {
        throw std::runtime_error("No Vulkan 1.4 GPU with timelineSemaphore + dynamic rendering + synchronization2 + VK_KHR_swapchain "
                                 "can present to this window");
    }
    const vk::raii::PhysicalDevice& physicalDevice = *best;
    const vk::PhysicalDeviceProperties chosen      = physicalDevice.getProperties();
    std::println("\nPicked: {}", chosen.deviceName.data());
    std::println("Graphics + present family: {}", graphicsFamily);

    // #region compute-family
    // A family with COMPUTE but without GRAPHICS is the "async compute" family:
    // its queues feed the GPU's compute work through a separate hardware queue,
    // which the GPU may schedule next to graphics work. The spec promises
    // nothing about concurrency; whether the two really overlap is up to the
    // hardware and the driver - hence the timestamps. --same-queue (or a GPU
    // without such a family) puts the compute submissions on the graphics queue
    // itself; then the graphics family must support COMPUTE.
    const std::vector<vk::QueueFamilyProperties> families = physicalDevice.getQueueFamilyProperties();
    const auto dedicated = std::ranges::find_if(families, [](const vk::QueueFamilyProperties& f) {
        return (f.queueFlags & vk::QueueFlagBits::eCompute) && !(f.queueFlags & vk::QueueFlagBits::eGraphics);
    });
    uint32_t computeFamily = graphicsFamily;
    if (!args.sameQueue && dedicated != families.end()) {
        computeFamily = static_cast<uint32_t>(dedicated - families.begin());
        std::println("Compute family: {} (COMPUTE without GRAPHICS), its own queue: async compute", computeFamily);
    } else {
        if (!(families.at(graphicsFamily).queueFlags & vk::QueueFlagBits::eCompute)) {
            throw std::runtime_error("The graphics family has no COMPUTE: cannot run the compute work on its queue");
        }
        std::println("Compute work on the graphics queue (family {}): {}", graphicsFamily,
                     args.sameQueue ? "--same-queue" : "no family has COMPUTE without GRAPHICS");
    }
    const bool separate = computeFamily != graphicsFamily;
    // Timestamps on both queues are compared: both families need them; the
    // comparison uses the bits valid in both.
    uint32_t validBits = std::numeric_limits<uint32_t>::max();
    for (const uint32_t f : {graphicsFamily, computeFamily}) {
        const uint32_t bits = families.at(f).timestampValidBits;
        if (bits == 0) {
            throw std::runtime_error(std::format("queue family {} has no timestamps", f));
        }
        validBits = std::min(validBits, bits);
    }
    // #endregion

    // #region device
    // One queue per family used. With --same-queue there is one queue, and both
    // kinds of submissions go to it.
    constexpr float queuePriority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queueInfos{
        {.queueFamilyIndex = graphicsFamily, .queueCount = 1, .pQueuePriorities = &queuePriority},
    };
    if (separate) {
        queueInfos.push_back({.queueFamilyIndex = computeFamily, .queueCount = 1, .pQueuePriorities = &queuePriority});
    }
    vk::PhysicalDeviceVulkan13Features features13{
        .synchronization2 = vk::True,
        .dynamicRendering = vk::True,
    };
    vk::PhysicalDeviceVulkan12Features features12{.pNext = &features13, .timelineSemaphore = vk::True};
    constexpr std::array deviceExtensions{vk::KHRSwapchainExtensionName};
    vk::raii::Device device(physicalDevice, vk::DeviceCreateInfo{
                                                .pNext                   = &features12,
                                                .queueCreateInfoCount    = static_cast<uint32_t>(queueInfos.size()),
                                                .pQueueCreateInfos       = queueInfos.data(),
                                                .enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size()),
                                                .ppEnabledExtensionNames = deviceExtensions.data(),
                                            });
    const vk::raii::Queue graphicsQueue(device, graphicsFamily, 0);
    const vk::raii::Queue computeQueue(device, computeFamily, 0);
    std::println("VkDevice created with {}, timelineSemaphore, {} queue(s)", vk::KHRSwapchainExtensionName, queueInfos.size());
    // #endregion

    // Step 07's swapchain: B8G8R8A8_SRGB + SRGB_NONLINEAR if offered, MAILBOX if
    // offered (else FIFO), the window's extent, min + 1 images.
    const std::vector<vk::SurfaceFormatKHR> formats = physicalDevice.getSurfaceFormatsKHR(*surface);
    if (formats.empty()) {
        throw std::runtime_error("The surface reports no formats");
    }
    const auto preferred = std::ranges::find_if(formats, [](const vk::SurfaceFormatKHR& f) {
        return f.format == vk::Format::eB8G8R8A8Srgb && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    });
    const vk::SurfaceFormatKHR format = preferred != formats.end() ? *preferred : formats.front();
    const std::vector<vk::PresentModeKHR> modes = physicalDevice.getSurfacePresentModesKHR(*surface);
    const vk::PresentModeKHR presentMode =
        std::ranges::contains(modes, vk::PresentModeKHR::eMailbox) ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
    const vk::SurfaceCapabilitiesKHR caps = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    vk::Extent2D extent                   = caps.currentExtent;
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
    vk::raii::SwapchainKHR swapchain(device, vk::SwapchainCreateInfoKHR{
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
                                             });
    const std::vector<vk::Image> images = swapchain.getImages();
    std::vector<vk::raii::ImageView> imageViews;
    imageViews.reserve(images.size());
    for (const vk::Image image : images) {
        imageViews.emplace_back(device, vk::ImageViewCreateInfo{
                                            .image            = image,
                                            .viewType         = vk::ImageViewType::e2D,
                                            .format           = format.format,
                                            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                                 .levelCount = 1,
                                                                 .layerCount = 1},
                                        });
    }
    std::println("Swapchain: {}x{} {}, {}, {} images", extent.width, extent.height, formatName(format.format),
                 shortName(string_VkPresentModeKHR(static_cast<VkPresentModeKHR>(presentMode)), "VK_PRESENT_MODE_", "_KHR"),
                 images.size());

    // The SPIR-V is compiled from shaders/particles.slang (step 26).
    const std::vector<uint32_t> spirv = readSpirv(std::filesystem::path{SHADER_DIR} / "particles.spv");
    const uint32_t groupSize          = localSizeX(spirv, "advance");
    if (localSizeX(spirv, "init") != groupSize) {
        throw std::runtime_error("init and advance must have the same workgroup size");
    }
    const uint32_t count  = args.count;
    const uint32_t groups = (count + groupSize - 1) / groupSize;
    if (groups > chosen.limits.maxComputeWorkGroupCount[0]) {
        throw std::runtime_error(std::format("{} workgroups exceed maxComputeWorkGroupCount[0]", groups));
    }
    std::println("\nSPIR-V: {} bytes (particles.spv), entry points init, advance, vertexMain, fragmentMain; LocalSize {} x 1 x 1",
                 spirv.size() * sizeof(uint32_t), groupSize);
    std::optional<vk::raii::ShaderModule> shaderModule;
    shaderModule.emplace(device, vk::ShaderModuleCreateInfo{
                                     .codeSize = spirv.size() * sizeof(uint32_t),
                                     .pCode    = spirv.data(),
                                 });

    // #region sharing
    // Two queue families touch the particle buffers: compute writes them,
    // graphics reads them as vertex input. An EXCLUSIVE buffer belongs to one
    // family at a time; reading it from another family without an ownership
    // transfer leaves "the contents of that resource ... undefined for any read
    // accesses performed by the second queue family". Here every state is read
    // by BOTH families at once (frame n draws P[n % 2] while step n+1 reads it
    // on the compute queue), so EXCLUSIVE would need a release + acquire pair
    // per family switch and would serialize exactly the overlap this step is
    // after. CONCURRENT lists the families that may access the buffer and needs
    // no transfers; the spec warns it "may result in lower performance access"
    // (on some GPUs it disables compression - mostly a concern for images).
    // With one family (--same-queue) EXCLUSIVE is the only choice: CONCURRENT
    // needs at least two distinct families.
    const std::array familyList{graphicsFamily, computeFamily};
    const vk::SharingMode sharing  = separate ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive;
    const std::string sharingText  = separate ? std::format("CONCURRENT (families {} and {}): no ownership transfers", graphicsFamily,
                                                            computeFamily)
                                              : std::format("EXCLUSIVE (one family, {})", graphicsFamily);
    const vk::PhysicalDeviceMemoryProperties mem = physicalDevice.getMemoryProperties();
    const vk::DeviceSize particleBytes           = vk::DeviceSize{count} * sizeof(Particle);
    // Each descriptor binds a whole buffer, and a storage-buffer descriptor may
    // cover at most maxStorageBufferRange bytes
    // (VUID-VkWriteDescriptorSet-descriptorType-00333). The spec only
    // guarantees 2^27 = 128 MiB, less than 2^24 particles x 16 bytes.
    const uint32_t maxRange = chosen.limits.maxStorageBufferRange;
    if (particleBytes > maxRange) {
        throw std::runtime_error(
            std::format("{} particles = {} bytes exceed maxStorageBufferRange = {}", count, particleBytes, maxRange));
    }
    std::vector<GpuBuffer> particles;
    for (int i = 0; i < 2; ++i) {
        vk::raii::Buffer buffer(device, vk::BufferCreateInfo{
                                            .size                  = particleBytes,
                                            .usage                 = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer,
                                            .sharingMode           = sharing,
                                            .queueFamilyIndexCount = separate ? 2u : 0u,
                                            .pQueueFamilyIndices   = separate ? familyList.data() : nullptr,
                                        });
        const vk::MemoryRequirements req = buffer.getMemoryRequirements();
        const auto memoryType            = findType(mem, req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        if (!memoryType) {
            throw std::runtime_error("no DEVICE_LOCAL memory type for the particle buffers");
        }
        vk::raii::DeviceMemory memory(device, vk::MemoryAllocateInfo{.allocationSize = req.size, .memoryTypeIndex = *memoryType});
        buffer.bindMemory(*memory, 0);
        particles.push_back(GpuBuffer{.memory = std::move(memory), .buffer = std::move(buffer), .memoryType = *memoryType});
    }
    std::println("Particles: 2 buffers (ping-pong) x {} x {} bytes, memory type {} (DEVICE_LOCAL), sharing mode {}", count,
                 sizeof(Particle), particles.front().memoryType, sharingText);
    // #endregion

    // #region pipelines
    // Step 26's pipelines, with advance instead of simulate: binding 0 = the
    // state it writes, binding 1 = the previous state it reads.
    std::array<vk::DescriptorSetLayoutBinding, 2> setBindings{};
    for (uint32_t b = 0; b < 2; ++b) {
        setBindings[b] = vk::DescriptorSetLayoutBinding{
            .binding         = b,
            .descriptorType  = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags      = vk::ShaderStageFlagBits::eCompute,
        };
    }
    vk::raii::DescriptorSetLayout setLayout(device, vk::DescriptorSetLayoutCreateInfo{.bindingCount = 2, .pBindings = setBindings.data()});
    const vk::DescriptorSetLayout setLayoutHandle = *setLayout;
    constexpr auto pushSize                       = static_cast<uint32_t>(sizeof(Params));
    const vk::PushConstantRange computeRange{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = pushSize};
    vk::raii::PipelineLayout computeLayout(device, vk::PipelineLayoutCreateInfo{
                                                       .setLayoutCount         = 1,
                                                       .pSetLayouts            = &setLayoutHandle,
                                                       .pushConstantRangeCount = 1,
                                                       .pPushConstantRanges    = &computeRange,
                                                   });
    const auto computePipeline = [&](const char* entry) {
        return vk::raii::Pipeline(
            device, nullptr,
            vk::ComputePipelineCreateInfo{
                .stage  = {.stage = vk::ShaderStageFlagBits::eCompute, .module = **shaderModule, .pName = entry},
                .layout = *computeLayout,
            });
    };
    std::optional<vk::raii::Pipeline> init{computePipeline("init")};
    vk::raii::Pipeline advance = computePipeline("advance");

    const vk::VertexInputBindingDescription vertexBinding{
        .binding   = 0,
        .stride    = sizeof(Particle),
        .inputRate = vk::VertexInputRate::eVertex,
    };
    const std::array attributes{
        vk::VertexInputAttributeDescription{.location = 0, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Particle, pos)},
        vk::VertexInputAttributeDescription{.location = 1, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Particle, vel)},
    };
    const vk::PipelineVertexInputStateCreateInfo vertexInput{
        .vertexBindingDescriptionCount   = 1,
        .pVertexBindingDescriptions      = &vertexBinding,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size()),
        .pVertexAttributeDescriptions    = attributes.data(),
    };
    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::ePointList};
    const vk::PushConstantRange graphicsRange{.stageFlags = vk::ShaderStageFlagBits::eVertex, .offset = 0, .size = pushSize};
    vk::raii::PipelineLayout graphicsLayout(device,
                                            vk::PipelineLayoutCreateInfo{.pushConstantRangeCount = 1, .pPushConstantRanges = &graphicsRange});
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = **shaderModule, .pName = "vertexMain"},
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = **shaderModule, .pName = "fragmentMain"},
    };
    const vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
    const vk::PipelineRasterizationStateCreateInfo rasterization{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode    = vk::CullModeFlagBits::eNone,
        .lineWidth   = 1.0f,
    };
    const vk::PipelineMultisampleStateCreateInfo multisample{.rasterizationSamples = vk::SampleCountFlagBits::e1};
    const vk::PipelineColorBlendAttachmentState blendAttachment{
        .blendEnable    = vk::False,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
                          vk::ColorComponentFlagBits::eA,
    };
    const vk::PipelineColorBlendStateCreateInfo colorBlend{.attachmentCount = 1, .pAttachments = &blendAttachment};
    constexpr std::array dynamicStates{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates    = dynamicStates.data(),
    };
    const vk::PipelineRenderingCreateInfo renderingInfo{.colorAttachmentCount = 1, .pColorAttachmentFormats = &format.format};
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
                                    .layout              = *graphicsLayout,
                                });
    shaderModule.reset();
    std::println("Pipelines: compute init + advance (set 0: binding 0 = state, binding 1 = previous), graphics POINT_LIST");
    // #endregion

    // #region descriptor-sets
    // Two sets, one per direction of the ping-pong: set i writes P[i] and
    // reads P[1 - i]. Step n binds set n % 2. They never change. The plain
    // handles are kept (release()): the sets go with the pool.
    const vk::DescriptorPoolSize poolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 4};
    vk::raii::DescriptorPool descriptorPool(device, vk::DescriptorPoolCreateInfo{.maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &poolSize});
    const std::array setLayouts{setLayoutHandle, setLayoutHandle};
    std::array<vk::DescriptorSet, 2> sets{};
    {
        vk::raii::DescriptorSets allocated(device, vk::DescriptorSetAllocateInfo{
                                                       .descriptorPool     = *descriptorPool,
                                                       .descriptorSetCount = 2,
                                                       .pSetLayouts        = setLayouts.data(),
                                                   });
        for (size_t i = 0; i < 2; ++i) {
            sets[i] = allocated[i].release();
        }
    }
    std::array<vk::DescriptorBufferInfo, 4> infos{};
    std::array<vk::WriteDescriptorSet, 4> writes{};
    for (size_t i = 0; i < 4; ++i) {
        const size_t set     = i / 2;                            // set 0, 0, 1, 1
        const auto binding   = static_cast<uint32_t>(i % 2);     // binding 0 = state, 1 = previous
        const size_t buffer  = binding == 0 ? set : 1 - set;     // state P[set], previous P[1 - set]
        infos[i]             = vk::DescriptorBufferInfo{.buffer = *particles[buffer].buffer, .offset = 0, .range = vk::WholeSize};
        writes[i]            = vk::WriteDescriptorSet{
                       .dstSet          = sets[set],
                       .dstBinding      = binding,
                       .descriptorCount = 1,
                       .descriptorType  = vk::DescriptorType::eStorageBuffer,
                       .pBufferInfo     = &infos[i],
        };
    }
    device.updateDescriptorSets(writes, nullptr);
    std::println("Descriptor sets: set 0 writes P[0] and reads P[1], set 1 writes P[1] and reads P[0]");
    // #endregion

    // Command pools: a command buffer is recorded for, and submitted to, queues
    // of its pool's family - so one pool per family.
    const auto poolFor = [&](uint32_t family) {
        return vk::raii::CommandPool(device, vk::CommandPoolCreateInfo{
                                                 .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                 .queueFamilyIndex = family,
                                             });
    };
    vk::raii::CommandPool graphicsPool = poolFor(graphicsFamily);
    vk::raii::CommandPool computePool  = poolFor(computeFamily);
    constexpr auto frameCount          = static_cast<uint32_t>(framesInFlight);
    const auto allocate                = [&](const vk::raii::CommandPool& pool) {
        return vk::raii::CommandBuffers(device, vk::CommandBufferAllocateInfo{
                                                    .commandPool        = *pool,
                                                    .level              = vk::CommandBufferLevel::ePrimary,
                                                    .commandBufferCount = frameCount,
                                                });
    };
    vk::raii::CommandBuffers graphicsCmds = allocate(graphicsPool);
    vk::raii::CommandBuffers computeCmds  = allocate(computePool);
    std::vector<Frame> frames;
    for (size_t i = 0; i < framesInFlight; ++i) {
        frames.push_back(Frame{
            .graphicsCmd    = std::move(graphicsCmds[i]),
            .computeCmd     = std::move(computeCmds[i]),
            .imageAvailable = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{}),
        });
    }
    std::vector<vk::raii::Semaphore> renderFinished;
    renderFinished.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        renderFinished.emplace_back(device, vk::SemaphoreCreateInfo{});
    }

    // #region timelines
    // Two timeline semaphores (step 12), one per producer: each counts the
    // finished work of one queue. A submission waits on the OTHER queue's
    // counter; the host waits on both to reuse a frame slot.
    const vk::StructureChain<vk::SemaphoreCreateInfo, vk::SemaphoreTypeCreateInfo> timelineInfo{
        {},
        {.semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0},
    };
    vk::raii::Semaphore computed(device, timelineInfo.get<vk::SemaphoreCreateInfo>());
    vk::raii::Semaphore rendered(device, timelineInfo.get<vk::SemaphoreCreateInfo>());
    std::println("Timeline semaphores: computed (step n done = n, the init state = 0), rendered (frame n done = n)");
    // #endregion

    vk::raii::QueryPool queryPool(device, vk::QueryPoolCreateInfo{.queryType = vk::QueryType::eTimestamp, .queryCount = 2 * timestamps * frameCount});
    GpuTimes gpu{
        .periodNs = static_cast<double>(chosen.limits.timestampPeriod),
        .bits     = std::min(validBits, 64u),
    };

    // #region init
    // Step 26's init, on the compute queue: writes state 0 into P[0] (set 0).
    // The host waits for it with a fence; compute step 1, on the same queue, is
    // ordered after it by its own leading barrier. The guard is declared after
    // the init fence and everything else the GPU uses, so if anything below
    // throws, it waits for both queues before RAII destroys them (a fence of a
    // pending submission must not be destroyed: VUID-vkDestroyFence-fence-01120).
    const float aspect = static_cast<float>(extent.height) / static_cast<float>(extent.width);
    const vk::raii::Fence initFence(device, vk::FenceCreateInfo{});
    const IdleGuard idleGuard{device};
    {
        const vk::raii::CommandBuffer& initCmd = frames.front().computeCmd;
        const Params params{.dt = 0.0f, .time = 0.0f, .count = count, .aspect = aspect};
        initCmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        initCmd.bindPipeline(vk::PipelineBindPoint::eCompute, **init);
        initCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computeLayout, 0, sets[0], nullptr);
        initCmd.pushConstants<Params>(*computeLayout, vk::ShaderStageFlagBits::eCompute, 0, params);
        initCmd.dispatch(groups, 1, 1);
        initCmd.end();
        const vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *initCmd};
        computeQueue.submit2(vk::SubmitInfo2{.commandBufferInfoCount = 1, .pCommandBufferInfos = &cmdInfo}, *initFence);
        if (device.waitForFences(*initFence, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
            throw std::runtime_error("waitForFences failed");
        }
        init.reset();
    }
    std::println("\ninit: {} workgroups of {} wrote state 0 into P[0]", groups, groupSize);
    // #endregion

    std::println("Per frame n: graphics draws P[n % 2] after computed >= n; compute step n+1 writes P[(n+1) % 2] after rendered >= n-1");

    using Clock      = std::chrono::steady_clock;
    using Seconds    = std::chrono::duration<double>;
    const auto start = Clock::now();
    auto last        = start;
    uint64_t frameNumber = 0; // the last graphics frame submitted (0 = none yet)
    std::optional<Ticks> pendingGraphics; // graphics frame n - F's timestamps, waiting for compute step n - F + 1's

    // Reads the two timestamps at first (their submission has completed).
    const auto readTimestamps = [&](uint32_t first) {
        const auto [result, ticks] = queryPool.getResults<uint64_t>(first, timestamps, timestamps * sizeof(uint64_t), sizeof(uint64_t),
                                                                    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("getQueryPoolResults failed");
        }
        return Ticks{ticks.at(0), ticks.at(1)};
    };
    // Host wait until semaphore's counter reaches value (step 12).
    const auto waitTimeline = [&](const vk::raii::Semaphore& semaphore, uint64_t value) {
        const vk::Semaphore handle = *semaphore;
        if (device.waitSemaphores(vk::SemaphoreWaitInfo{.semaphoreCount = 1, .pSemaphores = &handle, .pValues = &value},
                                  std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
            throw std::runtime_error("waitSemaphores failed");
        }
    };
    const auto elapsedSeconds = [&] { return static_cast<uint64_t>(Seconds(Clock::now() - start).count()); };

    using S = vk::PipelineStageFlagBits2;
    using A = vk::AccessFlagBits2;

    // #region submit-compute
    // Records and submits compute step n: P[n % 2] = advance(P[(n-1) % 2]).
    const auto submitCompute = [&](uint64_t n) {
        const size_t slot                  = n % framesInFlight;
        const vk::raii::CommandBuffer& cmd = frames.at(slot).computeCmd;
        // Queries 2 * timestamps * F.. hold the compute timestamps.
        const uint32_t firstQuery = timestamps * frameCount + timestamps * static_cast<uint32_t>(slot);
        // The slot's previous step (n - F) must be done before its command
        // buffer and queries are reused.
        if (n > framesInFlight) {
            waitTimeline(computed, n - framesInFlight);
            const Ticks ticks = readTimestamps(firstQuery);
            // Paired with graphics frame n - F - 1, read in drawFrame.
            if (pendingGraphics) {
                gpu.add(*pendingGraphics, ticks);
                pendingGraphics.reset();
                gpu.tick(elapsedSeconds());
            }
        }
        const auto now = Clock::now();
        const float dt = std::min(static_cast<float>(Seconds(now - last).count()), 1.0f / 60.0f);
        last           = now;
        const Params params{.dt = dt, .time = static_cast<float>(Seconds(now - start).count()), .count = count, .aspect = aspect};
        // Same queue, previous step: RAW (this step reads what step n-1 wrote)
        // and WAR (this step overwrites what step n-1 read).
        const vk::MemoryBarrier2 afterPreviousStep{
            .srcStageMask  = S::eComputeShader,
            .srcAccessMask = A::eShaderStorageWrite,
            .dstStageMask  = S::eComputeShader,
            .dstAccessMask = A::eShaderStorageRead | A::eShaderStorageWrite,
        };
        cmd.reset();
        cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cmd.resetQueryPool(*queryPool, firstQuery, timestamps);
        cmd.writeTimestamp2(S::eTopOfPipe, *queryPool, firstQuery);
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &afterPreviousStep});
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *advance);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computeLayout, 0, sets.at(n % 2), nullptr);
        cmd.pushConstants<Params>(*computeLayout, vk::ShaderStageFlagBits::eCompute, 0, params);
        cmd.dispatch(groups, 1, 1);
        cmd.writeTimestamp2(S::eComputeShader, *queryPool, firstQuery + 1);
        cmd.end();
        // Cross-queue: step n overwrites P[n % 2], which graphics frame n-2 drew
        // - wait until rendered >= n - 2 (WAR across queues: execution order is
        // enough). Signal computed = n when the step is done; the signal makes
        // the writes available, the graphics queue's wait makes them visible to
        // its vertex input (semaphores carry memory dependencies, no barrier
        // needed on either side). Both at ALL_COMMANDS: the start timestamp
        // (TOP_OF_PIPE) is inside the wait's scope, so the interval measures the
        // work, not the waiting. A wait at a later stage would let TOP_OF_PIPE
        // pass early and the interval would include the wait.
        const vk::SemaphoreSubmitInfo wait{.semaphore = *rendered, .value = n > 2 ? n - 2 : 0, .stageMask = S::eAllCommands};
        const vk::SemaphoreSubmitInfo signal{.semaphore = *computed, .value = n, .stageMask = S::eAllCommands};
        const vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *cmd};
        computeQueue.submit2(vk::SubmitInfo2{
                                 .waitSemaphoreInfoCount   = n > 2 ? 1u : 0u,
                                 .pWaitSemaphoreInfos      = &wait,
                                 .commandBufferInfoCount   = 1,
                                 .pCommandBufferInfos      = &cmdInfo,
                                 .signalSemaphoreInfoCount = 1,
                                 .pSignalSemaphoreInfos    = &signal,
                             },
                             nullptr);
    };
    // #endregion

    // Graphics frame n, then compute step n + 1. Returns false when the
    // swapchain is out of date (recreation is step 11).
    const auto drawFrame = [&]() -> bool {
        const uint64_t n                   = frameNumber + 1;
        const size_t slot                  = n % framesInFlight;
        Frame& frame                       = frames.at(slot);
        const vk::raii::CommandBuffer& cmd = frame.graphicsCmd;
        const uint32_t firstQuery          = timestamps * static_cast<uint32_t>(slot);

        // #region wait-slot
        // Frame n - F used this slot: its command buffer, imageAvailable
        // semaphore and queries are free once rendered >= n - F.
        if (n > framesInFlight) {
            waitTimeline(rendered, n - framesInFlight);
            pendingGraphics = readTimestamps(firstQuery);
        }
        // #endregion

        const vk::ResultValue<uint32_t> acquired =
            swapchain.acquireNextImage(std::numeric_limits<uint64_t>::max(), *frame.imageAvailable, nullptr);
        if (acquired.result == vk::Result::eErrorOutOfDateKHR) {
            return false;
        }
        const uint32_t imageIndex = acquired.value;
        const vk::Image image     = images.at(imageIndex);
        const Params params{.dt = 0.0f, .time = static_cast<float>(Seconds(Clock::now() - start).count()), .count = count, .aspect = aspect};

        constexpr vk::ImageSubresourceRange colorRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .layerCount = 1,
        };
        const vk::ImageMemoryBarrier2 toAttachment{
            .srcStageMask        = S::eColorAttachmentOutput,
            .srcAccessMask       = A::eNone,
            .dstStageMask        = S::eColorAttachmentOutput,
            .dstAccessMask       = A::eColorAttachmentWrite,
            .oldLayout           = vk::ImageLayout::eUndefined,
            .newLayout           = vk::ImageLayout::eColorAttachmentOptimal,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image               = image,
            .subresourceRange    = colorRange,
        };
        const vk::ImageMemoryBarrier2 toPresent{
            .srcStageMask        = S::eColorAttachmentOutput,
            .srcAccessMask       = A::eColorAttachmentWrite,
            .dstStageMask        = S::eColorAttachmentOutput,
            .dstAccessMask       = A::eNone,
            .oldLayout           = vk::ImageLayout::eColorAttachmentOptimal,
            .newLayout           = vk::ImageLayout::ePresentSrcKHR,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image               = image,
            .subresourceRange    = colorRange,
        };
        const vk::RenderingAttachmentInfo colorAttachment{
            .imageView   = *imageViews.at(imageIndex),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eStore,
            .clearValue  = vk::ClearColorValue{std::array{0.0f, 0.0f, 0.0f, 1.0f}},
        };

        // #region record-graphics
        // Step 26's draw without the dispatch and without a buffer barrier: the
        // particles of state n arrive through the semaphore wait below.
        cmd.reset();
        cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cmd.resetQueryPool(*queryPool, firstQuery, timestamps);
        cmd.writeTimestamp2(S::eTopOfPipe, *queryPool, firstQuery);
        cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toAttachment});
        cmd.beginRendering(vk::RenderingInfo{
            .renderArea           = {.offset = {0, 0}, .extent = extent},
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &colorAttachment,
        });
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
        cmd.pushConstants<Params>(*graphicsLayout, vk::ShaderStageFlagBits::eVertex, 0, params);
        cmd.bindVertexBuffers(0, *particles.at(n % 2).buffer, vk::DeviceSize{0});
        cmd.draw(count, 1, 0, 0);
        cmd.endRendering();
        cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toPresent});
        cmd.writeTimestamp2(S::eColorAttachmentOutput, *queryPool, firstQuery + 1);
        cmd.end();
        // #endregion

        // #region submit-graphics
        // Waits: the swapchain image (binary, as in step 10) and state n
        // (computed >= n). Both at ALL_COMMANDS, so the start timestamp comes
        // after both waits and the graphics interval measures the work, not the
        // waiting (see submit-compute). Step 10 waits for the image at
        // COLOR_ATTACHMENT_OUTPUT only, which lets the vertex work start before
        // the image is there - but then this interval would include the wait for
        // the image. The layout transition (src COLOR_ATTACHMENT_OUTPUT) still
        // chains to the wait: ALL_COMMANDS includes that stage. Signals:
        // renderFinished for the present and rendered = n, which the CPU
        // (wait-slot) and compute step n + 2 wait for.
        const vk::Semaphore finished = *renderFinished.at(imageIndex);
        const std::array waits{
            vk::SemaphoreSubmitInfo{.semaphore = *frame.imageAvailable, .stageMask = S::eAllCommands},
            vk::SemaphoreSubmitInfo{.semaphore = *computed, .value = n, .stageMask = S::eAllCommands},
        };
        const std::array signals{
            vk::SemaphoreSubmitInfo{.semaphore = finished, .stageMask = S::eColorAttachmentOutput},
            vk::SemaphoreSubmitInfo{.semaphore = *rendered, .value = n, .stageMask = S::eAllCommands},
        };
        const vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *cmd};
        graphicsQueue.submit2(vk::SubmitInfo2{
                                  .waitSemaphoreInfoCount   = static_cast<uint32_t>(waits.size()),
                                  .pWaitSemaphoreInfos      = waits.data(),
                                  .commandBufferInfoCount   = 1,
                                  .pCommandBufferInfos      = &cmdInfo,
                                  .signalSemaphoreInfoCount = static_cast<uint32_t>(signals.size()),
                                  .pSignalSemaphoreInfos    = signals.data(),
                              },
                              nullptr);
        frameNumber = n;

        const vk::SwapchainKHR swapchainHandle = *swapchain;
        const vk::Result presented             = graphicsQueue.presentKHR(vk::PresentInfoKHR{
                        .waitSemaphoreCount = 1,
                        .pWaitSemaphores    = &finished,
                        .swapchainCount     = 1,
                        .pSwapchains        = &swapchainHandle,
                        .pImageIndices      = &imageIndex,
        });
        if (presented == vk::Result::eErrorOutOfDateKHR) {
            return false;
        }
        // #endregion

        // The next state, while this frame draws.
        submitCompute(n + 1);
        return true;
    };

    // The pipeline starts one step ahead: step 1 before frame 1.
    submitCompute(1);
    std::println("Rendering. Close the window to exit.");
    const auto timeUp = [&] { return args.seconds && Seconds(Clock::now() - start).count() >= *args.seconds; };
    while (glfwWindowShouldClose(window.get()) == GLFW_FALSE && !timeUp()) {
        glfwPollEvents();
        if (!drawFrame()) {
            std::println("Swapchain out of date (the window changed); recreation is step 11, stopping");
            break;
        }
    }
    std::println("Window closed");
    const double elapsed = Seconds(Clock::now() - start).count();
    std::println("[timing] {} frames in {:.1f} s, {:.0f} fps", frameNumber, elapsed, static_cast<double>(frameNumber) / elapsed);
    GpuTimes::print("whole run", gpu.total);

    // #region cleanup
    // Both queues must be done; vkDeviceWaitIdle waits for all queues of the
    // device. Then RAII destroys in reverse order of declaration.
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
