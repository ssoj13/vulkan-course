// Step 20 - Bindless: one descriptor array, indexed in the shader
//
// Step 19's 64 hexagons again, but every instance's record lives in its OWN
// storage buffer (think: one buffer per object or material). Binding a set per
// object would mean a draw per object. Instead ONE set holds an ARRAY of
// storage buffer descriptors, bound once, and the shader picks
// objects[instance index] - the "bindless" pattern of modern renderers.
// Descriptor indexing (core 1.2) makes this practical:
//
//   runtimeDescriptorArray            the shader declares objects[] without a size
//   variable descriptor count         the set's real length is chosen at allocation
//   partially bound                   unwritten elements are allowed if never used
//   non-uniform indexing              the index may differ within one draw
//   update after bind / unused while pending
//                                     the set may be updated while frames that
//                                     use it are still in flight
//
// Streaming demo: 63 buffers are written at start; at frame 120 the 64th is
// written into the live set while the previous frame may still be in flight,
// and from then on 64 instances are drawn. Textures and samplers in such arrays
// come in module 10; here the array holds buffers only.
// --seconds S closes the window after S seconds; lines that depend on timing
// start with [timing]. Steps 01-19 are repeated, reduced to what this step needs.

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
    std::array<float, 2> position; // x, y before the per-instance transform
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

// Step 19's instance record: 32 bytes, std430 offsets 0 / 8 / 12 / 16 / 28
// (shaders/bindless.slang reads it from a StructuredBuffer).
struct Instance {
    std::array<float, 2> offset; // center, in units where the window is 2 tall
    float scale;                 // multiplies the mesh corners
    float phase;                 // starting angle, radians
    std::array<float, 3> color;  // multiplies the vertex colors
    float spin;                  // radians per second; the sign is the direction
};
static_assert(sizeof(Instance) == 32);
static_assert(offsetof(Instance, color) == 16);
static_assert(offsetof(Instance, spin) == 28);

// Grid size: grid x grid instances.
constexpr size_t gridSize = 8;

// Step 19's 8x8 grid of small hexagons.
std::vector<Instance> grid() {
    constexpr auto last = static_cast<float>(gridSize - 1);
    std::vector<Instance> instances;
    instances.reserve(gridSize * gridSize);
    for (size_t row = 0; row < gridSize; ++row) {
        for (size_t col = 0; col < gridSize; ++col) {
            const auto r = static_cast<float>(row);
            const auto c = static_cast<float>(col);
            instances.push_back(Instance{
                .offset = {-0.875f + 0.25f * c, -0.875f + 0.25f * r},
                .scale  = 0.18f,
                .phase  = 0.25f * (r * static_cast<float>(gridSize) + c),
                .color  = {0.4f + 0.6f * c / last, 0.4f + 0.6f * r / last, 1.0f},
                .spin   = (row + col) % 2 == 0 ? 1.0f : -1.0f,
            });
        }
    }
    return instances;
}

// The push constant block of shaders/bindless.slang: 8 bytes.
struct PushData {
    float time;   // seconds since start
    float aspect; // window height / width
};
static_assert(sizeof(PushData) == 8);

// #region sizes
// Upper bound of the descriptor array in the set LAYOUT: the largest set any
// allocation with this layout may have. Must not exceed the device's
// update-after-bind limits (checked at start).
constexpr uint32_t capacity = 4096;
// Length of the ONE set we allocate (its variable descriptor count): room for
// 256 buffers, of which only the first 64 are ever written.
constexpr uint32_t allocated = 256;
// Descriptors written before the first frame; the 64th is streamed in later.
constexpr uint32_t initial = 63;
// The frame (counted from 0) before whose recording the 64th descriptor is written.
constexpr size_t streamFrame = 120;
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

// #region write-descriptors
// Writes storage buffer descriptors for buffers into consecutive elements of
// binding 0, starting at array element first. One VkWriteDescriptorSet can fill
// a run of array elements: descriptorCount = buffers.size(). The binding has
// UPDATE_AFTER_BIND | UPDATE_UNUSED_WHILE_PENDING, and the caller writes only
// elements no pending command buffer uses.
void writeDescriptors(const vk::raii::Device& device, vk::DescriptorSet set, uint32_t first, std::span<const GpuBuffer> buffers) {
    std::vector<vk::DescriptorBufferInfo> infos;
    infos.reserve(buffers.size());
    for (const GpuBuffer& b : buffers) {
        infos.push_back(vk::DescriptorBufferInfo{.buffer = *b.buffer, .offset = 0, .range = vk::WholeSize});
    }
    device.updateDescriptorSets(vk::WriteDescriptorSet{
                                    .dstSet          = set,
                                    .dstBinding      = 0,
                                    .dstArrayElement = first,
                                    .descriptorCount = static_cast<uint32_t>(infos.size()),
                                    .descriptorType  = vk::DescriptorType::eStorageBuffer,
                                    .pBufferInfo     = infos.data(),
                                },
                                nullptr);
}
// #endregion

// What the CPU touches while recording one frame (step 10). The bindless set is
// shared by all frames: nothing per frame but commands and sync.
struct Frame {
    vk::raii::CommandBuffer commandBuffer;
    // Signaled by vkAcquireNextImageKHR, waited on by this slot's submit.
    vk::raii::Semaphore imageAvailable;
    // Signaled when this slot's submission completes.
    vk::raii::Fence inFlight;
};

// Command line: --seconds S closes the window after S seconds.
std::optional<double> parseSeconds(int argc, char** argv) {
    if (argc == 1) {
        return std::nullopt;
    }
    if (argc == 3 && std::string_view{argv[1]} == "--seconds") {
        const std::string_view text = argv[2];
        double value                = 0.0;
        const auto [end, ec]        = std::from_chars(text.data(), text.data() + text.size(), value);
        if (ec == std::errc{} && end == text.data() + text.size()) {
            return value;
        }
    }
    throw std::runtime_error("usage: 20_bindless [--seconds S]");
}

int main(int argc, char** argv)
try {
    const std::optional<double> seconds = parseSeconds(argc, argv);

    // ---- Step 06: window (800x600 physical pixels, fixed size) ----
    Glfw glfw;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 20 bindless", nullptr, nullptr), &glfwDestroyWindow);
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
        .pApplicationName   = "Vulkan from Zero - 20 bindless",
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
    // Plus the six descriptor indexing features this step uses.
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
        const auto featureChain = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,
                                                      vk::PhysicalDeviceVulkan13Features>();
        const auto& f12         = featureChain.get<vk::PhysicalDeviceVulkan12Features>();
        const auto& f13         = featureChain.get<vk::PhysicalDeviceVulkan13Features>();
        const std::array indexing{
            f12.runtimeDescriptorArray,
            f12.shaderStorageBufferArrayNonUniformIndexing,
            f12.descriptorBindingPartiallyBound,
            f12.descriptorBindingVariableDescriptorCount,
            f12.descriptorBindingStorageBufferUpdateAfterBind,
            f12.descriptorBindingUpdateUnusedWhilePending,
        };
        if (f13.dynamicRendering != vk::True || f13.synchronization2 != vk::True || std::ranges::contains(indexing, vk::False)) {
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
        throw std::runtime_error("No Vulkan 1.4 GPU with dynamicRendering + synchronization2 + the descriptor indexing features + "
                                 "VK_KHR_swapchain can present to this window");
    }
    const vk::raii::PhysicalDevice& physicalDevice = *best;
    const vk::PhysicalDeviceProperties chosen      = physicalDevice.getProperties();
    std::println("\nPicked: {}", chosen.deviceName.data());
    std::println("Graphics + present family: {}", family);

    // #region limits
    // Two sets of limits apply to descriptor arrays. The classic ones
    // (VkPhysicalDeviceLimits) count descriptors in sets WITHOUT
    // UPDATE_AFTER_BIND; the Vulkan 1.2 properties carry separate, often larger
    // limits for UPDATE_AFTER_BIND layouts (the spec lets the classic per-stage
    // limit be as low as 4). The ...Native flag says whether non-uniform
    // indexing costs nothing extra in hardware.
    const auto propsChain = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceVulkan12Properties>();
    const auto& props12   = propsChain.get<vk::PhysicalDeviceVulkan12Properties>();
    const vk::PhysicalDeviceLimits& limits = chosen.limits;
    std::println("Descriptor limits for storage buffers:");
    std::println("  maxPerStageDescriptorStorageBuffers                = {} (sets without UPDATE_AFTER_BIND)",
                 limits.maxPerStageDescriptorStorageBuffers);
    std::println("  maxDescriptorSetStorageBuffers                     = {}", limits.maxDescriptorSetStorageBuffers);
    std::println("  maxPerStageDescriptorUpdateAfterBindStorageBuffers = {} (UPDATE_AFTER_BIND sets)",
                 props12.maxPerStageDescriptorUpdateAfterBindStorageBuffers);
    std::println("  maxDescriptorSetUpdateAfterBindStorageBuffers      = {}", props12.maxDescriptorSetUpdateAfterBindStorageBuffers);
    std::println("  maxPerStageUpdateAfterBindResources                = {} (all resource types)",
                 props12.maxPerStageUpdateAfterBindResources);
    std::println("  maxUpdateAfterBindDescriptorsInAllPools            = {}", props12.maxUpdateAfterBindDescriptorsInAllPools);
    std::println("  shaderStorageBufferArrayNonUniformIndexingNative   = {}",
                 props12.shaderStorageBufferArrayNonUniformIndexingNative == vk::True);
    // The layout gives the vertex stage capacity storage buffers and nothing
    // else, so capacity counts against the per-stage and per-set storage buffer
    // limits and the per-stage limit for all resources.
    const uint32_t cap = std::min({props12.maxPerStageDescriptorUpdateAfterBindStorageBuffers,
                                   props12.maxDescriptorSetUpdateAfterBindStorageBuffers, props12.maxPerStageUpdateAfterBindResources});
    if (capacity > cap) {
        throw std::runtime_error(
            std::format("the layout's {} storage buffers exceed this device's update-after-bind limit {}", capacity, cap));
    }
    // The pool holds allocated descriptors. All UPDATE_AFTER_BIND pools
    // together share maxUpdateAfterBindDescriptorsInAllPools; beyond it, pool
    // creation may fail.
    if (allocated > props12.maxUpdateAfterBindDescriptorsInAllPools) {
        throw std::runtime_error(std::format("the pool's {} descriptors exceed maxUpdateAfterBindDescriptorsInAllPools {}", allocated,
                                             props12.maxUpdateAfterBindDescriptorsInAllPools));
    }
    std::println("This program: layout capacity {}, one set of {}, {} buffers written", capacity, allocated, initial + 1);
    // #endregion

    // #region device-features
    // Step 07's device plus the descriptor indexing features of the Vulkan 1.2
    // feature struct. Each one is checked by validation where it is used, e.g.
    // UPDATE_AFTER_BIND on a storage buffer binding without
    // descriptorBindingStorageBufferUpdateAfterBind is
    // VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingStorageBufferUpdateAfterBind-03008.
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
        .pNext                                         = &features13,
        .shaderStorageBufferArrayNonUniformIndexing    = vk::True,
        .descriptorBindingStorageBufferUpdateAfterBind = vk::True,
        .descriptorBindingUpdateUnusedWhilePending     = vk::True,
        .descriptorBindingPartiallyBound               = vk::True,
        .descriptorBindingVariableDescriptorCount      = vk::True,
        .runtimeDescriptorArray                        = vk::True,
    };
    constexpr std::array deviceExtensions{vk::KHRSwapchainExtensionName};
    vk::raii::Device device(physicalDevice, vk::DeviceCreateInfo{
                                                .pNext                   = &features12,
                                                .queueCreateInfoCount    = 1,
                                                .pQueueCreateInfos       = &queueInfo,
                                                .enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size()),
                                                .ppEnabledExtensionNames = deviceExtensions.data(),
                                            });
    // #endregion
    std::println("VkDevice created with {} + runtimeDescriptorArray, shaderStorageBufferArrayNonUniformIndexing,",
                 vk::KHRSwapchainExtensionName);
    std::println("  descriptorBindingPartiallyBound, descriptorBindingVariableDescriptorCount,");
    std::println("  descriptorBindingStorageBufferUpdateAfterBind, descriptorBindingUpdateUnusedWhilePending");

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

    // The SPIR-V is compiled from shaders/bindless.slang by `bun run
    // sync:shaders`; CMake bakes the absolute path of shaders/spv into SHADER_DIR.
    const std::vector<uint32_t> spirv = readSpirv(std::filesystem::path{SHADER_DIR} / "bindless.spv");
    std::optional<vk::raii::ShaderModule> shaderModule;
    shaderModule.emplace(device, vk::ShaderModuleCreateInfo{
                                     .codeSize = spirv.size() * sizeof(uint32_t),
                                     .pCode    = spirv.data(),
                                 });
    std::println("\nSPIR-V: {} bytes (bindless.spv), entry points vertexMain, fragmentMain", spirv.size() * sizeof(uint32_t));
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = **shaderModule, .pName = "vertexMain"},
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

    // #region set-layout
    // Binding 0: up to capacity storage buffers, vertex stage. The binding flags
    // (VkDescriptorSetLayoutBindingFlagsCreateInfo, one entry per binding) make
    // it bindless:
    //   eVariableDescriptorCount   capacity is only an upper bound; each set gets
    //                              its own length at allocation (only allowed on
    //                              the highest-numbered binding)
    //   ePartiallyBound            elements never used by the shader may stay
    //                              unwritten
    //   eUpdateAfterBind           a write after the set is bound in a recording
    //                              command buffer does not invalidate it (the
    //                              UPDATE_AFTER_BIND limits above apply to such
    //                              layouts)
    //   eUpdateUnusedWhilePending  elements that no pending command buffer uses
    //                              may be written while it executes
    // A layout with an UPDATE_AFTER_BIND binding must itself carry
    // eUpdateAfterBindPool (VUID-VkDescriptorSetLayoutCreateInfo-flags-03000).
    const vk::DescriptorSetLayoutBinding setBinding{
        .binding         = 0,
        .descriptorType  = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = capacity,
        .stageFlags      = vk::ShaderStageFlagBits::eVertex,
    };
    const vk::DescriptorBindingFlags bindingFlags =
        vk::DescriptorBindingFlagBits::eVariableDescriptorCount | vk::DescriptorBindingFlagBits::ePartiallyBound |
        vk::DescriptorBindingFlagBits::eUpdateAfterBind | vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
    const vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{.bindingCount = 1, .pBindingFlags = &bindingFlags};
    vk::raii::DescriptorSetLayout setLayout(device, vk::DescriptorSetLayoutCreateInfo{
                                                        .pNext        = &flagsInfo,
                                                        .flags        = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
                                                        .bindingCount = 1,
                                                        .pBindings    = &setBinding,
                                                    });
    std::println("Descriptor set layout (UPDATE_AFTER_BIND_POOL): binding 0 = STORAGE_BUFFER x up to {}, stage VERTEX", capacity);
    std::println("  binding flags: VARIABLE_DESCRIPTOR_COUNT | PARTIALLY_BOUND | UPDATE_AFTER_BIND | UPDATE_UNUSED_WHILE_PENDING");
    // #endregion

    // Step 18's push constant range (time + aspect) and the one set layout.
    constexpr uint32_t pushSize = sizeof(PushData);
    const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eVertex, .offset = 0, .size = pushSize};
    const vk::DescriptorSetLayout setLayoutHandle = *setLayout;
    vk::raii::PipelineLayout pipelineLayout(device, vk::PipelineLayoutCreateInfo{
                                                        .setLayoutCount         = 1,
                                                        .pSetLayouts            = &setLayoutHandle,
                                                        .pushConstantRangeCount = 1,
                                                        .pPushConstantRanges    = &range,
                                                    });
    std::println("Pipeline layout: 1 set layout (set 0), push constants {} bytes (time + aspect)", pushSize);

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
                                    .layout              = *pipelineLayout,
                                });
    shaderModule.reset();
    std::println("Graphics pipeline created (vertexMain reads objects[NonUniformResourceIndex(instance)])");

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

    // #region object-buffers
    // One small STORAGE_BUFFER per instance, each holding one record (a stand-in
    // for per-object data of different sizes and lifetimes). 64 separate
    // allocations keep the example short; step 16 explains why a real renderer
    // sub-allocates them.
    const std::vector<Instance> instances = grid();
    std::vector<GpuBuffer> objectBuffers;
    objectBuffers.reserve(instances.size());
    for (const Instance& record : instances) {
        objectBuffers.push_back(createBuffer(device, mem, sizeof(Instance), vk::BufferUsageFlagBits::eStorageBuffer, hostFlags));
        writeMapped(objectBuffers.back().memory, std::span{&record, 1});
    }
    std::println("Object buffers: {} x {} bytes (one Instance each), STORAGE_BUFFER, memory type {} ({})", objectBuffers.size(),
                 sizeof(Instance), objectBuffers.front().memoryType, memoryNames(hostFlags));
    // #endregion

    // #region pool-and-set
    // The pool must be created with eUpdateAfterBind to hold sets of an
    // UPDATE_AFTER_BIND_POOL layout; its size counts the descriptors we
    // actually allocate (allocated), not the layout's capacity. The variable
    // count of the one set is passed at allocation. vk::raii::DescriptorSet
    // would free the set in its destructor (needs eFreeDescriptorSet); the set
    // goes with the pool, so we keep the plain handle (release()).
    const vk::DescriptorPoolSize poolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = allocated};
    vk::raii::DescriptorPool descriptorPool(device, vk::DescriptorPoolCreateInfo{
                                                        .flags         = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
                                                        .maxSets       = 1,
                                                        .poolSizeCount = 1,
                                                        .pPoolSizes    = &poolSize,
                                                    });
    const vk::DescriptorSetVariableDescriptorCountAllocateInfo variableInfo{.descriptorSetCount = 1, .pDescriptorCounts = &allocated};
    vk::raii::DescriptorSets sets(device, vk::DescriptorSetAllocateInfo{
                                              .pNext              = &variableInfo,
                                              .descriptorPool     = *descriptorPool,
                                              .descriptorSetCount = 1,
                                              .pSetLayouts        = &setLayoutHandle,
                                          });
    const vk::DescriptorSet descriptorSet = sets.front().release();
    std::println("Descriptor pool (UPDATE_AFTER_BIND): maxSets 1, STORAGE_BUFFER x {}", allocated);
    std::println("Allocated 1 set with variable descriptor count {}", allocated);
    // #endregion

    // #region initial-writes
    // Elements 0..62 now, in one write. Elements 63..255 stay unwritten: allowed
    // because the binding is PARTIALLY_BOUND and the draws read only indices
    // below the number of written descriptors.
    writeDescriptors(device, descriptorSet, 0, std::span{objectBuffers}.first(initial));
    std::println("vkUpdateDescriptorSets: elements 0..{} -> object buffers 0..{} ({} descriptors in one write); {} elements unwritten",
                 initial - 1, initial - 1, initial, allocated - initial);
    // #endregion

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
    std::println("\n{} frames in flight; each frame binds the one set and draws {} indices x {} instances (64 from frame {})",
                 framesInFlight, indices.size(), initial, streamFrame);

    using Clock        = std::chrono::steady_clock;
    using Seconds      = std::chrono::duration<double>;
    const auto start   = Clock::now();
    size_t frameIndex  = 0;
    uint32_t liveCount = initial; // instances drawn = descriptors written so far

    // Renders and presents one frame (step 10's loop). Returns false when the
    // swapchain is out of date (recreation is step 11).
    const auto drawFrame = [&]() -> bool {
        Frame& frame                       = frames.at(frameIndex % frames.size());
        const vk::raii::CommandBuffer& cmd = frame.commandBuffer;

        // The last submission that used this slot must be finished before its
        // command buffer and semaphore are reused (step 10).
        if (device.waitForFences(*frame.inFlight, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
            throw std::runtime_error("waitForFences failed");
        }

        // #region stream-in
        // Streaming a resource into the live set. The other slot's command buffer
        // may still be PENDING with this set bound, so only elements it does not
        // use may change: it draws liveCount instances, so element liveCount is not
        // dynamically used. UPDATE_UNUSED_WHILE_PENDING is the flag whose
        // definition allows exactly this (UPDATE_AFTER_BIND covers the window
        // between bind and submit, not a pending command buffer). Note:
        // VUID-vkUpdateDescriptorSets-None-03047 accepts EITHER flag, so
        // validation stayed silent when this program was tried without
        // UPDATE_UNUSED_WHILE_PENDING - the flag is here because of the spec's
        // definition, not because of a message. This slot's command buffer is
        // re-recorded below, so it sees the new element; its draw may read it.
        if (frameIndex == streamFrame) {
            writeDescriptors(device, descriptorSet, liveCount, std::span{objectBuffers}.subspan(liveCount, 1));
            std::println(
                "Frame {}: element {} written while the set is bound by a command buffer that may be pending; instances {} -> {}",
                streamFrame, liveCount, liveCount, liveCount + 1);
            ++liveCount;
        }
        // #endregion

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

        // #region bind-and-draw
        // ONE set bind for all objects, then ONE draw of liveCount instances;
        // instance i reads objects[i]. A renderer with thousands of objects keeps
        // exactly this shape: bind the big set once per frame (or once per
        // command buffer), pass indices, never rebind per object.
        const PushData push{
            .time   = static_cast<float>(Seconds(Clock::now() - start).count()),
            .aspect = static_cast<float>(extent.height) / static_cast<float>(extent.width),
        };
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, descriptorSet, nullptr);
        cmd.pushConstants<PushData>(*pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, push);
        cmd.bindVertexBuffers(0, *vertexBuffer.buffer, vk::DeviceSize{0});
        cmd.bindIndexBuffer(*indexBuffer.buffer, 0, vk::IndexType::eUint16);
        cmd.drawIndexed(static_cast<uint32_t>(indices.size()), liveCount, 0, 0, 0);
        cmd.endRendering();
        // #endregion

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
    const auto timeUp = [&] { return seconds && Seconds(Clock::now() - start).count() >= *seconds; };
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
    // declaration: the descriptor pool (which frees the set), the 64 object
    // buffers, the rest.
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
