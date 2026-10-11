// Step 26 - Compute and graphics in one frame: particles
//
// A million particles live in ONE storage buffer on the GPU. Every frame a
// compute shader moves them (shaders/particles.slang, simulate), and the
// graphics pipeline draws the same buffer as a vertex buffer, one point per
// particle. The CPU never touches the particles: it only records
//
//   barrier (last frame's draw + dispatch -> this dispatch)
//   dispatch simulate          (compute writes the buffer)
//   barrier (dispatch -> vertex input)
//   draw N points              (vertex input reads the buffer)
//
// into one command buffer on the graphics queue (its family has COMPUTE too).
// Step 17's frame loop with 2 frames in flight; --count N sets the number of
// particles, --seconds S closes the window after S seconds; lines that depend
// on timing start with [timing]. Steps 01-17 are repeated, reduced to what
// this step needs.

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
// the GPU busy; from here on it is fixed.
constexpr size_t framesInFlight = 2;

// #region cli
// Command line: --count N particles (default 2^20), --seconds S.
struct Args {
    uint32_t count = 1u << 20;
    std::optional<double> seconds;
};

Args parseArgs(int argc, char** argv) {
    constexpr std::string_view usage = "usage: 26_particles [--count N] [--seconds S]";
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
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

// #region particle
// One particle as shaders/particles.slang declares it (std430: pos at 0, vel
// at 8, stride 16). The host never reads or writes particles; the struct only
// gives the vertex input its stride and offsets.
struct Particle {
    std::array<float, 2> pos;
    std::array<float, 2> vel;
};
static_assert(sizeof(Particle) == 16);

// The push constant block of particles.slang: 16 bytes, read by the compute
// shaders (dt, time, count) and the vertex shader (aspect).
struct Params {
    float dt;
    float time;
    uint32_t count;
    float aspect;
};
static_assert(sizeof(Params) == 16);
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

// Step 25's reflection: the LocalSize x of compute entry point entry (the
// kernels are 1D), from OpEntryPoint + OpExecutionMode LocalSize.
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

// What the CPU touches while recording one frame (step 10). No per-frame
// particle data: the particles are GPU-only, see frame-barriers.
struct Frame {
    vk::raii::CommandBuffer commandBuffer;
    // Signaled by vkAcquireNextImageKHR, waited on by this slot's submit.
    vk::raii::Semaphore imageAvailable;
    // Signaled when this slot's submission completes.
    vk::raii::Fence inFlight;
};

int main(int argc, char** argv)
try {
    const Args args = parseArgs(argc, argv);

    // ---- Step 06: window (800x600 physical pixels, fixed size) ----
    Glfw glfw;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 26 particles", nullptr, nullptr), &glfwDestroyWindow);
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
        .pApplicationName   = "Vulkan from Zero - 26 particles",
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
    // Steps 03 + 06, and the family must ALSO support COMPUTE: the dispatch and
    // the draw go into the same command buffer, so into the same queue. A family
    // with GRAPHICS usually has COMPUTE too, but the spec does not promise it for
    // every family, so it is checked.
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
            const vk::QueueFlags flags = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;
            if ((families[i].queueFlags & flags) == flags && device.getSurfaceSupportKHR(i, *surface) == vk::True) {
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
        throw std::runtime_error("No Vulkan 1.4 GPU with dynamicRendering + synchronization2 + VK_KHR_swapchain and a graphics + "
                                 "compute family can present to this window");
    }
    const vk::raii::PhysicalDevice& physicalDevice = *best;
    const vk::PhysicalDeviceProperties chosen      = physicalDevice.getProperties();
    std::println("\nPicked: {}", chosen.deviceName.data());
    std::println("Graphics + compute + present family: {}", family);
    // #endregion

    // Step 07's device: one queue, the two 1.3 features, VK_KHR_swapchain.
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
    vk::raii::Device device(physicalDevice, vk::DeviceCreateInfo{
                                                .pNext                   = &features13,
                                                .queueCreateInfoCount    = 1,
                                                .pQueueCreateInfos       = &queueInfo,
                                                .enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size()),
                                                .ppEnabledExtensionNames = deviceExtensions.data(),
                                            });
    std::println("VkDevice created with {}", vk::KHRSwapchainExtensionName);

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

    // The SPIR-V is compiled from shaders/particles.slang by `bun run
    // sync:shaders`; CMake bakes the absolute path of shaders/spv into SHADER_DIR.
    const std::vector<uint32_t> spirv = readSpirv(std::filesystem::path{SHADER_DIR} / "particles.spv");
    const uint32_t groupSize          = localSizeX(spirv, "simulate");
    if (localSizeX(spirv, "init") != groupSize) {
        throw std::runtime_error("init and simulate must have the same workgroup size");
    }
    const uint32_t count  = args.count;
    const uint32_t groups = (count + groupSize - 1) / groupSize;
    if (groups > chosen.limits.maxComputeWorkGroupCount[0]) {
        throw std::runtime_error(std::format("{} workgroups exceed maxComputeWorkGroupCount[0]", groups));
    }
    std::println("\nSPIR-V: {} bytes (particles.spv), entry points init, simulate, vertexMain, fragmentMain; LocalSize {} x 1 x 1",
                 spirv.size() * sizeof(uint32_t), groupSize);
    std::optional<vk::raii::ShaderModule> shaderModule;
    shaderModule.emplace(device, vk::ShaderModuleCreateInfo{
                                     .codeSize = spirv.size() * sizeof(uint32_t),
                                     .pCode    = spirv.data(),
                                 });

    // #region particle-buffer
    // GPU-only data in DEVICE_LOCAL memory. Two usages: STORAGE_BUFFER lets the
    // compute shader bind it through a descriptor, VERTEX_BUFFER lets
    // vkCmdBindVertexBuffers bind the very same buffer for the draw. No staging
    // and no CPU copy: the init shader writes the starting state.
    const vk::PhysicalDeviceMemoryProperties mem = physicalDevice.getMemoryProperties();
    const vk::DeviceSize particleBytes           = vk::DeviceSize{count} * sizeof(Particle);
    // The descriptor below binds the whole buffer, and a storage-buffer
    // descriptor may cover at most maxStorageBufferRange bytes
    // (VUID-VkWriteDescriptorSet-descriptorType-00333). The spec only
    // guarantees 2^27 = 128 MiB, less than 2^24 particles x 16 bytes.
    const uint32_t maxRange = chosen.limits.maxStorageBufferRange;
    if (particleBytes > maxRange) {
        throw std::runtime_error(
            std::format("{} particles = {} bytes exceed maxStorageBufferRange = {}", count, particleBytes, maxRange));
    }
    GpuBuffer particles = createBuffer(device, mem, particleBytes,
                                       vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer,
                                       vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::println("Particles: {} x {} bytes = {} bytes, STORAGE_BUFFER | VERTEX_BUFFER, memory type {} (DEVICE_LOCAL)", count,
                 sizeof(Particle), particleBytes, particles.memoryType);
    // #endregion

    // #region compute-pipelines
    // Set 0, binding 0 = the particle buffer as a STORAGE_BUFFER, for the
    // compute stage. One push constant range of 16 bytes for the compute stage.
    // Two pipelines share this layout: init and simulate.
    const vk::DescriptorSetLayoutBinding setBinding{
        .binding         = 0,
        .descriptorType  = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags      = vk::ShaderStageFlagBits::eCompute,
    };
    vk::raii::DescriptorSetLayout setLayout(device, vk::DescriptorSetLayoutCreateInfo{.bindingCount = 1, .pBindings = &setBinding});
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
    vk::raii::Pipeline simulate = computePipeline("simulate");
    std::println("Compute pipelines: init, simulate (set 0: binding 0 = STORAGE_BUFFER; {} push constant bytes)", pushSize);
    // #endregion

    // #region graphics-pipeline
    // The draw reads the particle buffer through VERTEX INPUT, not a
    // descriptor: binding 0, stride 16, pos at offset 0, vel at offset 8, one
    // vertex per particle, topology POINT_LIST. The vertex shader only needs
    // aspect from the push block; its range covers the whole 16-byte block for
    // simplicity.
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
    // Points have no winding: no culling.
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
    std::println("Graphics pipeline: POINT_LIST, vertex input binding 0 = the particle buffer (stride {})", sizeof(Particle));
    // #endregion

    // One descriptor set: binding 0 -> the particle buffer (step 17). The
    // plain handle is kept (release()): the set goes with the pool.
    const vk::DescriptorPoolSize poolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1};
    vk::raii::DescriptorPool descriptorPool(device, vk::DescriptorPoolCreateInfo{.maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &poolSize});
    vk::raii::DescriptorSets allocated(device, vk::DescriptorSetAllocateInfo{
                                                   .descriptorPool     = *descriptorPool,
                                                   .descriptorSetCount = 1,
                                                   .pSetLayouts        = &setLayoutHandle,
                                               });
    const vk::DescriptorSet descriptorSet = allocated.front().release();
    const vk::DescriptorBufferInfo bufferInfo{.buffer = *particles.buffer, .offset = 0, .range = vk::WholeSize};
    device.updateDescriptorSets(vk::WriteDescriptorSet{
                                    .dstSet          = descriptorSet,
                                    .dstBinding      = 0,
                                    .descriptorCount = 1,
                                    .descriptorType  = vk::DescriptorType::eStorageBuffer,
                                    .pBufferInfo     = &bufferInfo,
                                },
                                nullptr);

    // Steps 09-10: the queue, the pool, per-frame buffers and sync objects.
    vk::raii::Queue queue(device, family, 0);
    vk::raii::CommandPool commandPool(device, vk::CommandPoolCreateInfo{
                                                  .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                  .queueFamilyIndex = family,
                                              });
    constexpr auto frameCount = static_cast<uint32_t>(framesInFlight);
    vk::raii::CommandBuffers commandBuffers(device, vk::CommandBufferAllocateInfo{
                                                        .commandPool        = *commandPool,
                                                        .level              = vk::CommandBufferLevel::ePrimary,
                                                        .commandBufferCount = frameCount,
                                                    });
    std::vector<Frame> frames;
    frames.reserve(framesInFlight);
    for (size_t i = 0; i < framesInFlight; ++i) {
        frames.push_back(Frame{
            .commandBuffer  = std::move(commandBuffers[i]),
            .imageAvailable = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{}),
            .inFlight       = vk::raii::Fence(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled}),
        });
    }
    std::vector<vk::raii::Semaphore> renderFinished;
    renderFinished.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        renderFinished.emplace_back(device, vk::SemaphoreCreateInfo{});
    }
    const float aspect = static_cast<float>(extent.height) / static_cast<float>(extent.width);

    // #region init
    // The starting state, written by the GPU: one dispatch of init in a command
    // buffer of its own, submitted once and waited for with a fence before the
    // first frame. The first frame's barrier (frame-barriers) has the compute
    // stage in its source scope, so it also orders the first simulate after
    // these writes. The guard is declared after the init fence and everything
    // else the GPU uses, so if anything below throws, it waits for the device
    // before RAII destroys them (a fence of a pending submission must not be
    // destroyed: VUID-vkDestroyFence-fence-01120).
    const vk::raii::Fence initFence(device, vk::FenceCreateInfo{});
    const IdleGuard idleGuard{device};
    {
        const vk::raii::CommandBuffer& initCmd = frames.front().commandBuffer;
        const Params params{.dt = 0.0f, .time = 0.0f, .count = count, .aspect = aspect};
        initCmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        initCmd.bindPipeline(vk::PipelineBindPoint::eCompute, **init);
        initCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computeLayout, 0, descriptorSet, nullptr);
        initCmd.pushConstants<Params>(*computeLayout, vk::ShaderStageFlagBits::eCompute, 0, params);
        initCmd.dispatch(groups, 1, 1);
        initCmd.end();
        const vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *initCmd};
        queue.submit2(vk::SubmitInfo2{.commandBufferInfoCount = 1, .pCommandBufferInfos = &cmdInfo}, *initFence);
        if (device.waitForFences(*initFence, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
            throw std::runtime_error("waitForFences failed");
        }
        // init is not needed again; the command buffer is reused by frame slot 0.
        init.reset();
    }
    std::println("\ninit: {} workgroups of {} wrote {} particles (one submit, fence waited)", groups, groupSize, count);
    // #endregion

    std::println("Per frame: barrier -> dispatch simulate {} x 1 x 1 -> barrier "
                 "COMPUTE_SHADER/SHADER_STORAGE_WRITE -> VERTEX_ATTRIBUTE_INPUT/VERTEX_ATTRIBUTE_READ -> draw {} points",
                 groups, count);
    std::println("{} frames in flight, ONE particle buffer: the GPU orders the frames with barriers, the CPU never touches it",
                 framesInFlight);

    using Clock       = std::chrono::steady_clock;
    using Seconds     = std::chrono::duration<double>;
    const auto start  = Clock::now();
    auto last         = start;
    size_t frameIndex = 0;

    // Simulates, renders and presents one frame (step 10's loop). Returns false
    // when the swapchain is out of date (recreation is step 11).
    const auto drawFrame = [&]() -> bool {
        Frame& frame                       = frames.at(frameIndex % frames.size());
        const vk::raii::CommandBuffer& cmd = frame.commandBuffer;

        // The last submission that used this slot must be finished before its
        // command buffer and semaphore are reused (step 10).
        if (device.waitForFences(*frame.inFlight, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
            throw std::runtime_error("waitForFences failed");
        }
        const vk::ResultValue<uint32_t> acquired =
            swapchain.acquireNextImage(std::numeric_limits<uint64_t>::max(), *frame.imageAvailable, nullptr);
        if (acquired.result == vk::Result::eErrorOutOfDateKHR) {
            return false;
        }
        const uint32_t imageIndex = acquired.value;
        const vk::Image image     = images.at(imageIndex);
        device.resetFences(*frame.inFlight);

        // The time step: real time since the last frame, at most 1/60 s so a
        // stalled frame (window dragged, breakpoint) does not make the
        // semi-implicit Euler step explode.
        const auto now = Clock::now();
        const float dt = std::min(static_cast<float>(Seconds(now - last).count()), 1.0f / 60.0f);
        last           = now;
        const Params params{.dt = dt, .time = static_cast<float>(Seconds(now - start).count()), .count = count, .aspect = aspect};

        cmd.reset();
        cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        using S = vk::PipelineStageFlagBits2;
        using A = vk::AccessFlagBits2;
        // #region frame-barriers
        // Two frames are in flight, but there is ONE particle buffer. Frame n's
        // dispatch must not start while frame n-1 still uses the buffer:
        // - WAR: frame n-1's draw READS the particles as vertex input; this
        //   dispatch overwrites them. Execution order is all it needs: the
        //   dispatch waits for earlier VERTEX_ATTRIBUTE_INPUT work (access NONE
        //   on both sides - a read leaves nothing to make available).
        // - RAW: this dispatch reads what frame n-1's dispatch WROTE: the writes
        //   must be made available and visible to compute reads and writes.
        // A pipeline barrier's first scope is every command submitted earlier to
        // this queue, including other command buffers and submissions - which
        // is why it reaches back into frame n-1 (and, for the first frame, into
        // init). Frames in flight still overlap on the CPU and in the parts of
        // the GPU pipeline the barriers leave free; per-frame COPIES (step 17's
        // uniform buffers) are only needed for data the CPU writes while the GPU
        // may still read the old one.
        const std::array beforeDispatch{
            vk::MemoryBarrier2{
                .srcStageMask  = S::eVertexAttributeInput,
                .srcAccessMask = A::eNone,
                .dstStageMask  = S::eComputeShader,
                .dstAccessMask = A::eNone,
            },
            vk::MemoryBarrier2{
                .srcStageMask  = S::eComputeShader,
                .srcAccessMask = A::eShaderStorageWrite,
                .dstStageMask  = S::eComputeShader,
                .dstAccessMask = A::eShaderStorageRead | A::eShaderStorageWrite,
            },
        };
        // RAW within this frame: the draw's vertex input reads what the dispatch
        // wrote.
        const vk::MemoryBarrier2 beforeDraw{
            .srcStageMask  = S::eComputeShader,
            .srcAccessMask = A::eShaderStorageWrite,
            .dstStageMask  = S::eVertexAttributeInput,
            .dstAccessMask = A::eVertexAttributeRead,
        };
        // #endregion

        // #region simulate
        // Dispatches cannot be recorded inside a render pass instance, so the
        // compute work comes before vkCmdBeginRendering.
        cmd.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = static_cast<uint32_t>(beforeDispatch.size()),
            .pMemoryBarriers    = beforeDispatch.data(),
        });
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *simulate);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computeLayout, 0, descriptorSet, nullptr);
        cmd.pushConstants<Params>(*computeLayout, vk::ShaderStageFlagBits::eCompute, 0, params);
        cmd.dispatch(groups, 1, 1);
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &beforeDraw});
        // #endregion

        constexpr vk::ImageSubresourceRange colorRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .layerCount = 1,
        };
        // Step 09's layout transition to eColorAttachmentOptimal, chained to the
        // acquire semaphore's wait stage.
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
        const vk::RenderingAttachmentInfo colorAttachment{
            .imageView   = *imageViews.at(imageIndex),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eStore,
            .clearValue  = vk::ClearColorValue{std::array{0.0f, 0.0f, 0.0f, 1.0f}},
        };

        // #region draw
        // The particle buffer bound as vertex buffer 0: one point per particle.
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
        cmd.bindVertexBuffers(0, *particles.buffer, vk::DeviceSize{0});
        cmd.draw(count, 1, 0, 0);
        cmd.endRendering();
        // #endregion

        // Step 09's transition to ePresentSrcKHR, chained to renderFinished.
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
        cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toPresent});
        cmd.end();

        // Step 10's submit: wait imageAvailable at COLOR_ATTACHMENT_OUTPUT (the
        // dispatch does not wait for the swapchain image), signal renderFinished,
        // then the slot's fence.
        const vk::Semaphore signalSemaphore = *renderFinished.at(imageIndex);
        const vk::SemaphoreSubmitInfo waitInfo{.semaphore = *frame.imageAvailable, .stageMask = S::eColorAttachmentOutput};
        const vk::SemaphoreSubmitInfo signalInfo{.semaphore = signalSemaphore, .stageMask = S::eColorAttachmentOutput};
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
    // Wait until the GPU is idle: the last frames may still simulate and draw
    // the particles. Then RAII destroys in reverse order of declaration (the
    // pool frees its set).
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
