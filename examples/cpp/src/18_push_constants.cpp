// Step 18 - Push constants and push descriptors
//
// Step 17 needed a set layout, a pool, one set per frame, a buffer per frame
// and vkUpdateDescriptorSets - for 80 bytes. Two lighter roads exist:
//
//   push constants (default): the 80 bytes go INTO the command buffer with
//     vkCmdPushConstants. No buffer, no descriptor; the pipeline layout only
//     declares a push constant range. Small: maxPushConstantsSize is at least
//     128 bytes in Vulkan 1.0 and at least 256 in Vulkan 1.4.
//   push descriptors (--push-descriptor, core in Vulkan 1.4, feature
//     pushDescriptor): the data stays in step 17's uniform buffers, but the
//     DESCRIPTOR is recorded into the command buffer with
//     vkCmdPushDescriptorSet. No pool, no sets, no vkUpdateDescriptorSets.
//
// The same FrameUniforms struct and the same shader file
// (shaders/transform.slang: vertexPush or vertexUniform). --seconds S closes
// the window after S seconds; lines that depend on timing start with
// [timing]. Steps 01-17 are repeated, reduced to what this step needs.

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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// the GPU busy; from here on it is fixed.
constexpr size_t framesInFlight = 2;

// Step 14's vertex: 2 + 3 floats = 20 bytes.
struct Vertex {
    std::array<float, 2> position; // x, y before the transform
    std::array<float, 3> color;    // linear RGB
};
static_assert(sizeof(Vertex) == 20);

// Step 17's regular hexagon (radius 0.6).
constexpr std::array<Vertex, 7> vertices{{
    {{0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},         // center, white
    {{0.6f, 0.0f}, {1.0f, 0.0f, 0.0f}},         // 0 deg, red
    {{0.3f, 0.5196152f}, {1.0f, 1.0f, 0.0f}},   // 60 deg, yellow
    {{-0.3f, 0.5196152f}, {0.0f, 1.0f, 0.0f}},  // 120 deg, green
    {{-0.6f, 0.0f}, {0.0f, 1.0f, 1.0f}},        // 180 deg, cyan
    {{-0.3f, -0.5196152f}, {0.0f, 0.0f, 1.0f}}, // 240 deg, blue
    {{0.3f, -0.5196152f}, {1.0f, 0.0f, 1.0f}},  // 300 deg, magenta
}};
// Step 14's indices: six triangles around the center.
constexpr std::array<uint16_t, 18> indices{
    0, 1, 2, //
    0, 2, 3, //
    0, 3, 4, //
    0, 4, 5, //
    0, 5, 6, //
    0, 6, 1, //
};

// Step 17's FrameUniforms: 80 bytes, offsets 0 / 64 / 76. As push constants
// the shader reads it with std430 rules, which give the same offsets for this
// struct (step 17 checked std140 against the SPIR-V).
struct FrameUniforms {
    std::array<std::array<float, 4>, 4> transform; // column-major: transform[c] is column c (Slang's default)
    std::array<float, 3> tint;                     // linear RGB multiplier
    float time;                                    // seconds since start
};
static_assert(sizeof(FrameUniforms) == 80);

// Rotation speed of the hexagon, radians per second (step 17).
constexpr float spin = 0.8f;
// A warm tint, multiplied into the vertex colors by the shader (step 17).
constexpr std::array<float, 3> tint{1.0f, 0.85f, 0.7f};

// Step 17's column-major S * R: rotation about z, then x scaled by aspect.
std::array<std::array<float, 4>, 4> transform(float angle, float aspect) {
    const float s = std::sin(angle);
    const float c = std::cos(angle);
    return {{
        {aspect * c, s, 0.0f, 0.0f},  // column 0
        {-aspect * s, c, 0.0f, 0.0f}, // column 1
        {0.0f, 0.0f, 1.0f, 0.0f},     // column 2
        {0.0f, 0.0f, 0.0f, 1.0f},     // column 3
    }};
}

// #region cli
// How the per-frame FrameUniforms reach the vertex shader.
enum class Path {
    PushConstants,  // vkCmdPushConstants: the bytes themselves go into the command buffer
    PushDescriptor, // vkCmdPushDescriptorSet: a descriptor for this slot's uniform buffer goes into the command buffer
};

struct Args {
    Path path = Path::PushConstants;
    std::optional<double> seconds;
};

// Command line: [--push-descriptor] [--seconds S].
Args parseArgs(int argc, char** argv) {
    constexpr const char* usage = "usage: 18_push_constants [--push-descriptor] [--seconds S]";
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--push-descriptor") {
            args.path = Path::PushDescriptor;
        } else if (arg == "--seconds" && i + 1 < argc) {
            const std::string_view text = argv[++i];
            double value                = 0.0;
            const auto [end, ec]        = std::from_chars(text.data(), text.data() + text.size(), value);
            if (ec != std::errc{} || end != text.data() + text.size()) {
                throw std::runtime_error(usage);
            }
            args.seconds = value;
        } else {
            throw std::runtime_error(usage);
        }
    }
    return args;
}
// #endregion

// Memory property names for printing, in bit order (step 13).
std::string memoryNames(vk::MemoryPropertyFlags flags) {
    using Mem = vk::MemoryPropertyFlagBits;
    constexpr std::array<std::pair<Mem, std::string_view>, 4> table{{
        {Mem::eDeviceLocal, "DEVICE_LOCAL"},
        {Mem::eHostVisible, "HOST_VISIBLE"},
        {Mem::eHostCoherent, "HOST_COHERENT"},
        {Mem::eHostCached, "HOST_CACHED"},
    }};
    std::string out;
    for (const auto& [bit, name] : table) {
        if (flags & bit) {
            out += out.empty() ? "" : " | ";
            out += name;
        }
    }
    return out.empty() ? "(none)" : out;
}

// Step 13's find-memory-type rule: the first memory type, in index order, that
// bits (the resource's memoryTypeBits) allows and that has every flag.
std::optional<uint32_t> findType(const vk::PhysicalDeviceMemoryProperties& props, uint32_t bits,
                                 vk::MemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) != 0 && (props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return std::nullopt;
}

// A buffer and the memory bound to it (step 14). Members are destroyed in
// reverse order: the buffer first, then its memory.
struct GpuBuffer {
    vk::raii::DeviceMemory memory;
    vk::raii::Buffer buffer;
    uint32_t memoryType;
};

// Step 14's manual path: create the buffer, query its requirements, take the
// first memory type with flags, allocate, bind at offset 0.
GpuBuffer createBuffer(const vk::raii::Device& device, const vk::PhysicalDeviceMemoryProperties& props, vk::DeviceSize size,
                       vk::BufferUsageFlags usage, vk::MemoryPropertyFlags flags) {
    vk::raii::Buffer buffer(device, vk::BufferCreateInfo{
                                        .size        = size,
                                        .usage       = usage,
                                        .sharingMode = vk::SharingMode::eExclusive,
                                    });
    const vk::MemoryRequirements req = buffer.getMemoryRequirements();
    const auto memoryType            = findType(props, req.memoryTypeBits, flags);
    if (!memoryType) {
        throw std::runtime_error("no memory type with " + memoryNames(flags) + " for this buffer");
    }
    vk::raii::DeviceMemory memory(device, vk::MemoryAllocateInfo{
                                              .allocationSize  = req.size,
                                              .memoryTypeIndex = *memoryType,
                                          });
    buffer.bindMemory(*memory, 0);
    return GpuBuffer{.memory = std::move(memory), .buffer = std::move(buffer), .memoryType = *memoryType};
}

// Step 14's upload: map, memcpy, unmap a HOST_VISIBLE | HOST_COHERENT
// allocation (no flush needed; the submit makes the writes visible).
template <typename T, size_t N> void writeMapped(const vk::raii::DeviceMemory& memory, std::span<T, N> data) {
    void* ptr = memory.mapMemory(0, vk::WholeSize);
    std::memcpy(ptr, data.data(), data.size_bytes());
    memory.unmapMemory();
}

// #region frame-resources
// What the CPU touches while recording one frame (step 10). Push constants need
// nothing per frame: each command buffer carries its own copy of the bytes. The
// push-descriptor path keeps step 17's buffer per slot.
struct Frame {
    vk::raii::CommandBuffer commandBuffer;
    // Signaled by vkAcquireNextImageKHR, waited on by this slot's submit.
    vk::raii::Semaphore imageAvailable;
    // Signaled when this slot's submission completes.
    vk::raii::Fence inFlight;
    // Push-descriptor path only: this slot's uniform buffer and its persistently mapped pointer.
    std::optional<GpuBuffer> uniform = std::nullopt;
    FrameUniforms* mapped = nullptr;
};
// #endregion

int main(int argc, char** argv)
try {
    const Args args = parseArgs(argc, argv);
    const Path path = args.path;

    // ---- Step 06: window (800x600 physical pixels, fixed size) ----
    Glfw glfw;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 18 push constants", nullptr, nullptr), &glfwDestroyWindow);
    if (!window) {
        throw std::runtime_error("glfwCreateWindow failed");
    }
    int width  = 0;
    int height = 0;
    glfwGetFramebufferSize(window.get(), &width, &height);
    std::println("Window: {}x{} pixels", width, height);
    std::println("{}", path == Path::PushConstants ? "Path: push constants (--push-descriptor selects push descriptors)"
                                                   : "Path: push descriptors (--push-descriptor)");

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
        .pApplicationName   = "Vulkan from Zero - 18 push constants",
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

    // ---- Steps 03 + 06: Vulkan 1.4, features, graphics + present family, VK_KHR_swapchain ----
    // The push-descriptor path also needs the 1.4 feature pushDescriptor.
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
        const auto featureChain = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
                                                      vk::PhysicalDeviceVulkan14Features>();
        const auto& f13         = featureChain.get<vk::PhysicalDeviceVulkan13Features>();
        const auto& f14         = featureChain.get<vk::PhysicalDeviceVulkan14Features>();
        if (f13.dynamicRendering != vk::True || f13.synchronization2 != vk::True) {
            continue;
        }
        if (path == Path::PushDescriptor && f14.pushDescriptor != vk::True) {
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
        throw std::runtime_error("No Vulkan 1.4 GPU with dynamicRendering + synchronization2 (+ pushDescriptor for "
                                 "--push-descriptor) + VK_KHR_swapchain can present to this window");
    }
    const vk::raii::PhysicalDevice& physicalDevice = *best;
    const vk::PhysicalDeviceProperties chosen      = physicalDevice.getProperties();
    std::println("\nPicked: {}", chosen.deviceName.data());
    std::println("Graphics + present family: {}", family);

    // #region limits
    // How much each road can carry. maxPushConstantsSize is a core limit (the
    // spec guarantees at least 128 bytes in Vulkan 1.0 and at least 256 in
    // Vulkan 1.4; 128 is the budget that fits every version);
    // maxPushDescriptors (Vulkan 1.4 properties, at least 32) is how many
    // descriptors one push-descriptor set layout may hold.
    const auto propsChain = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceVulkan14Properties>();
    const auto& props14   = propsChain.get<vk::PhysicalDeviceVulkan14Properties>();
    std::println("Limits: maxPushConstantsSize = {} bytes (at least 128 everywhere), maxPushDescriptors = {}",
                 chosen.limits.maxPushConstantsSize, props14.maxPushDescriptors);
    // #endregion

    // #region device-features
    // Step 07's device; for push descriptors also pushDescriptor, the 1.4
    // feature that replaced the VK_KHR_push_descriptor extension (a 1.4 device
    // may not call vkCmdPushDescriptorSet without it:
    // VUID-vkCmdPushDescriptorSet-None-10356). Push constants are Vulkan 1.0:
    // nothing to enable.
    constexpr float queuePriority = 1.0f;
    const vk::DeviceQueueCreateInfo queueInfo{
        .queueFamilyIndex = family,
        .queueCount       = 1,
        .pQueuePriorities = &queuePriority,
    };
    vk::PhysicalDeviceVulkan14Features features14{
        .pushDescriptor = path == Path::PushDescriptor ? vk::True : vk::False,
    };
    vk::PhysicalDeviceVulkan13Features features13{
        .pNext            = &features14,
        .synchronization2 = vk::True,
        .dynamicRendering = vk::True,
    };
    constexpr std::array deviceExtensions{vk::KHRSwapchainExtensionName};
    vk::raii::Device device(physicalDevice, vk::DeviceCreateInfo{
                                                .pNext                   = &features13,
                                                .queueCreateInfoCount    = 1,
                                                .pQueueCreateInfos       = &queueInfo,
                                                .enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size()),
                                                .ppEnabledExtensionNames = deviceExtensions.data(),
                                            });
    // #endregion
    std::println("VkDevice created with {}{}", vk::KHRSwapchainExtensionName,
                 path == Path::PushDescriptor ? " + feature pushDescriptor" : "");

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

    // Step 17's shader file; the entry point depends on the path.
    const std::vector<uint32_t> spirv = readSpirv(std::filesystem::path{SHADER_DIR} / "transform.spv");
    std::optional<vk::raii::ShaderModule> shaderModule;
    shaderModule.emplace(device, vk::ShaderModuleCreateInfo{
                                     .codeSize = spirv.size() * sizeof(uint32_t),
                                     .pCode    = spirv.data(),
                                 });
    const char* vertexEntry = path == Path::PushConstants ? "vertexPush" : "vertexUniform";
    std::println("\nSPIR-V: {} bytes (transform.spv), entry points {}, fragmentMain", spirv.size() * sizeof(uint32_t), vertexEntry);
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = **shaderModule, .pName = vertexEntry},
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = **shaderModule, .pName = "fragmentMain"},
    };

    // Step 14's vertex input: binding 0, stride 20, position + color.
    const vk::VertexInputBindingDescription vertexBinding{
        .binding   = 0,
        .stride    = sizeof(Vertex),
        .inputRate = vk::VertexInputRate::eVertex,
    };
    const std::array attributes{
        vk::VertexInputAttributeDescription{
            .location = 0, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Vertex, position)},
        vk::VertexInputAttributeDescription{
            .location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, color)},
    };
    const vk::PipelineVertexInputStateCreateInfo vertexInput{
        .vertexBindingDescriptionCount   = 1,
        .pVertexBindingDescriptions      = &vertexBinding,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size()),
        .pVertexAttributeDescriptions    = attributes.data(),
    };

    constexpr uint32_t uniformSize = sizeof(FrameUniforms);
    std::optional<vk::raii::DescriptorSetLayout> setLayout; // push-descriptor path only
    std::optional<vk::raii::PipelineLayout> pipelineLayout;
    if (path == Path::PushConstants) {
        // #region push-constant-range
        // The pipeline layout declares which byte range of the push constant
        // block each stage reads: here bytes 0..80, vertex stage, matching
        // [[vk::push_constant]] in the shader. Ranges must fit in
        // maxPushConstantsSize (VUID-VkPushConstantRange-size-00298).
        if (uniformSize > chosen.limits.maxPushConstantsSize) {
            throw std::runtime_error(std::format("FrameUniforms ({} bytes) exceeds maxPushConstantsSize", uniformSize));
        }
        const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eVertex, .offset = 0, .size = uniformSize};
        pipelineLayout.emplace(device, vk::PipelineLayoutCreateInfo{.pushConstantRangeCount = 1, .pPushConstantRanges = &range});
        std::println("Push constant range: stage VERTEX, offset 0, size {}", uniformSize);
        std::println("Pipeline layout: no set layouts, 1 push constant range");
        // #endregion
    } else {
        // #region push-descriptor-layout
        // Step 17's set layout plus one flag: ePushDescriptor. Sets of this
        // layout are never allocated; the descriptors are pushed into the
        // command buffer. At most one set of a pipeline layout may be a
        // push-descriptor set (VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00293),
        // and it holds at most maxPushDescriptors descriptors.
        const vk::DescriptorSetLayoutBinding setBinding{
            .binding         = 0,
            .descriptorType  = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags      = vk::ShaderStageFlagBits::eVertex,
        };
        setLayout.emplace(device, vk::DescriptorSetLayoutCreateInfo{
                                      .flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
                                      .bindingCount = 1,
                                      .pBindings    = &setBinding,
                                  });
        const vk::DescriptorSetLayout setLayoutHandle = **setLayout;
        pipelineLayout.emplace(device, vk::PipelineLayoutCreateInfo{.setLayoutCount = 1, .pSetLayouts = &setLayoutHandle});
        std::println("Descriptor set layout: binding 0 = UNIFORM_BUFFER x 1, stage VERTEX, flags PUSH_DESCRIPTOR");
        std::println("Pipeline layout: 1 set layout (set 0), no push constants");
        // #endregion
    }

    // The rest of step 08's pipeline, unchanged.
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
                                    .layout              = **pipelineLayout,
                                });
    shaderModule.reset();
    std::println("Graphics pipeline created ({} reads {})", vertexEntry,
                 path == Path::PushConstants ? "the push constants" : "set 0, binding 0");

    // Step 14's mesh buffers: HOST_VISIBLE | HOST_COHERENT, written once.
    const vk::PhysicalDeviceMemoryProperties mem = physicalDevice.getMemoryProperties();
    const vk::MemoryPropertyFlags hostFlags      = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    const std::span vertexData{vertices};
    const std::span indexData{indices};
    GpuBuffer vertexBuffer = createBuffer(device, mem, vertexData.size_bytes(), vk::BufferUsageFlagBits::eVertexBuffer, hostFlags);
    writeMapped(vertexBuffer.memory, vertexData);
    GpuBuffer indexBuffer = createBuffer(device, mem, indexData.size_bytes(), vk::BufferUsageFlagBits::eIndexBuffer, hostFlags);
    writeMapped(indexBuffer.memory, indexData);
    std::println("\nMesh: regular hexagon, {} vertices, {} indices (step 14's buffers, memory type {})", vertices.size(),
                 indices.size(), vertexBuffer.memoryType);

    // Steps 09-10: the queue, the pool, per-frame buffers and sync objects.
    vk::raii::Queue queue(device, family, 0);
    vk::raii::CommandPool commandPool(device, vk::CommandPoolCreateInfo{
                                                  .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                  .queueFamilyIndex = family,
                                              });
    vk::raii::CommandBuffers commandBuffers(device, vk::CommandBufferAllocateInfo{
                                                        .commandPool        = *commandPool,
                                                        .level              = vk::CommandBufferLevel::ePrimary,
                                                        .commandBufferCount = static_cast<uint32_t>(framesInFlight),
                                                    });
    std::vector<Frame> frames;
    frames.reserve(framesInFlight);
    for (vk::raii::CommandBuffer& commandBuffer : commandBuffers) {
        Frame frame{
            .commandBuffer  = std::move(commandBuffer),
            .imageAvailable = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{}),
            .inFlight       = vk::raii::Fence(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled}),
        };
        if (path == Path::PushDescriptor) {
            // Step 17's per-slot uniform buffer; stays mapped until the memory is freed.
            frame.uniform = createBuffer(device, mem, uniformSize, vk::BufferUsageFlagBits::eUniformBuffer, hostFlags);
            frame.mapped  = static_cast<FrameUniforms*>(frame.uniform->memory.mapMemory(0, vk::WholeSize));
        }
        frames.push_back(std::move(frame));
    }
    std::vector<vk::raii::Semaphore> renderFinished;
    renderFinished.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        renderFinished.emplace_back(device, vk::SemaphoreCreateInfo{});
    }
    if (frames.front().uniform) {
        std::println("Uniform buffers: {} x {} bytes, UNIFORM_BUFFER, memory type {} ({}), persistently mapped", framesInFlight,
                     uniformSize, frames.front().uniform->memoryType, memoryNames(hostFlags));
    }

    // #region compare
    // What each road costs to set up, for the same 80 bytes per frame.
    std::println("\nSetup for the per-frame FrameUniforms:");
    std::println("  descriptor sets (step 17): set layout + pool + {} sets + {} uniform buffers + vkUpdateDescriptorSets; per "
                 "frame: write buffer, bind set",
                 framesInFlight, framesInFlight);
    std::println("  push descriptors:          set layout (PUSH_DESCRIPTOR) + {} uniform buffers; per frame: write buffer, "
                 "vkCmdPushDescriptorSet",
                 framesInFlight);
    std::println("  push constants:            a push constant range in the pipeline layout; per frame: vkCmdPushConstants ({} "
                 "bytes)",
                 uniformSize);
    // #endregion
    const float aspect = static_cast<float>(extent.height) / static_cast<float>(extent.width);
    std::println("\n{} frames in flight", framesInFlight);
    std::println("Animation: rotation {} rad/s about z, x scaled by height/width = {}", spin, aspect);

    using Clock       = std::chrono::steady_clock;
    using Seconds     = std::chrono::duration<double>;
    const auto start  = Clock::now();
    size_t frameIndex = 0;

    // Renders and presents one frame (step 10's loop). Returns false when the
    // swapchain is out of date (recreation is step 11).
    const auto drawFrame = [&]() -> bool {
        Frame& frame                       = frames.at(frameIndex % frames.size());
        const vk::raii::CommandBuffer& cmd = frame.commandBuffer;

        // The last submission that used this slot must be finished before its
        // command buffer, semaphore and uniform buffer are reused (step 10).
        if (device.waitForFences(*frame.inFlight, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
            throw std::runtime_error("waitForFences failed");
        }

        const auto time = static_cast<float>(Seconds(Clock::now() - start).count());
        const FrameUniforms uniformData{
            .transform = transform(time * spin, aspect),
            .tint      = tint,
            .time      = time,
        };
        if (frame.mapped != nullptr) {
            // Step 17's write into this slot's mapped buffer (no pending GPU work reads it: fence above).
            std::memcpy(frame.mapped, &uniformData, sizeof uniformData);
        }

        const vk::ResultValue<uint32_t> acquired =
            swapchain.acquireNextImage(std::numeric_limits<uint64_t>::max(), *frame.imageAvailable, nullptr);
        if (acquired.result == vk::Result::eErrorOutOfDateKHR) {
            return false;
        }
        const uint32_t imageIndex = acquired.value;
        const vk::Image image     = images.at(imageIndex);
        device.resetFences(*frame.inFlight);

        cmd.reset();
        cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        constexpr vk::ImageSubresourceRange colorRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .layerCount = 1,
        };
        // Step 09's layout transition to eColorAttachmentOptimal, chained to the
        // acquire semaphore's wait stage.
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
        const vk::RenderingAttachmentInfo colorAttachment{
            .imageView   = *imageViews.at(imageIndex),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eStore,
            .clearValue  = vk::ClearColorValue{std::array{0.02f, 0.02f, 0.03f, 1.0f}},
        };
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

        if (path == Path::PushConstants) {
            // #region push-constants
            // The 80 bytes are copied into the command buffer NOW, at record
            // time: each recorded draw sees the values pushed before it, and
            // nothing on the host has to stay alive or unchanged afterwards. That
            // is why push constants need no per-frame buffer and no fence-guarded
            // write.
            cmd.pushConstants<FrameUniforms>(**pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, uniformData);
            // #endregion
        } else {
            // #region push-descriptor
            // Step 17's write, but recorded instead of stored in a set: the
            // descriptor (buffer, offset, range) is copied into the command buffer
            // and becomes set 0 for the draws after it. dstSet is ignored. The
            // BUFFER is still referenced, so it must stay alive and unchanged
            // until the GPU is done (the per-slot buffer + fence of step 17).
            const vk::DescriptorBufferInfo bufferInfo{.buffer = *frame.uniform->buffer, .offset = 0, .range = uniformSize};
            const vk::WriteDescriptorSet write{
                .dstBinding      = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo     = &bufferInfo,
            };
            cmd.pushDescriptorSet(vk::PipelineBindPoint::eGraphics, **pipelineLayout, 0, write);
            // #endregion
        }

        cmd.bindVertexBuffers(0, *vertexBuffer.buffer, vk::DeviceSize{0});
        cmd.bindIndexBuffer(*indexBuffer.buffer, 0, vk::IndexType::eUint16);
        cmd.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        cmd.endRendering();

        // Step 09's transition to ePresentSrcKHR, chained to renderFinished.
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

        // Step 10's submit: wait imageAvailable at COLOR_ATTACHMENT_OUTPUT, signal
        // renderFinished, then the slot's fence.
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

        ++frameIndex;
        return true;
    };

    // Step 10's loop: poll window messages, render one frame.
    const IdleGuard idleGuard{device};
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
    std::println("[timing] {} frames in {:.1f} s, {:.0f} fps", frameIndex, elapsed, static_cast<double>(frameIndex) / elapsed);

    // #region cleanup
    // Wait until the GPU is idle, then RAII destroys in reverse order of
    // declaration. There is no descriptor pool to destroy on either path.
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
