// Step 28 - Debug names and labels
//
// Step 22's two cubes, now told to the tools. Validation messages, RenderDoc and
// Nsight see raw handles (0x2a5f...) and anonymous commands unless the program
// describes them through VK_EXT_debug_utils, the same instance extension whose
// messenger prints validation messages since step 02:
//
//   OBJECT NAMES   vkSetDebugUtilsObjectNameEXT: a string per object; the layer
//                  quotes it in messages, capture tools show it.
//   COMMAND BUFFER LABELS  vkCmdBegin/EndDebugUtilsLabelEXT: nested regions
//                  ("frame" > "main pass" > "cube 0"), vkCmdInsert... a point.
//   QUEUE LABELS   vkQueueBegin/EndDebugUtilsLabelEXT around a frame's submit
//                  and present.
//
// None of it changes what is rendered: names and labels are information for
// whoever inspects the program (the validation layer, RenderDoc, Nsight).
//
// --printf renders frame 0 with a fragment shader that calls printf for one
// pixel (shaders/debug_printf.slang); the validation layer prints its output to
// stdout with VK_KHRONOS_VALIDATION_PRINTF_ENABLE=true and
// VK_KHRONOS_VALIDATION_PRINTF_TO_STDOUT=true (without the second one it goes to
// the messenger at INFO severity, which this messenger filters out; step 02).
// --seconds S closes the window after S seconds. Lines that depend on timing
// start with [timing]. Steps 01-22 are repeated, reduced to what this step needs.

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
#include <numbers>
#include <optional>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

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

// #region debug-callback
// Step 02's callback, extended: besides the message it prints what the layer
// passes along with it. pObjects lists the objects the message is about ("the
// 0th element is always guaranteed to be the most important object"), each with
// its type, handle and the name this program gave it; pCmdBufLabels /
// pQueueLabels are the labels active in the command buffer / queue when the
// message was triggered (filled only for messages raised while recording or
// submitting).
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
        if (data == nullptr) {
            std::println(stderr, "[{}][{}] <no message>", level, kind);
            return vk::False;
        }
        std::println(stderr, "[{}][{}] {}", level, kind, data->pMessage ? data->pMessage : "");
        // Arrays from the layer: a null pointer means "none".
        const auto span = [](const auto* ptr, uint32_t count) {
            return ptr ? std::span{ptr, count} : std::span<std::remove_pointer_t<decltype(ptr)>>{};
        };
        for (size_t i = 0; const vk::DebugUtilsObjectNameInfoEXT& o : span(data->pObjects, data->objectCount)) {
            const std::string name = o.pObjectName && *o.pObjectName ? std::format("\"{}\"", o.pObjectName) : "(no name)";
            std::println(stderr, "    object {}: {} {:#x} {}", i++,
                         shortName(string_VkObjectType(static_cast<VkObjectType>(o.objectType)), "VK_OBJECT_TYPE_"),
                         o.objectHandle, name);
        }
        const auto labels = [&](const vk::DebugUtilsLabelEXT* ptr, uint32_t count) {
            std::string out;
            for (const vk::DebugUtilsLabelEXT& l : span(ptr, count)) {
                out += out.empty() ? "" : " | ";
                out += l.pLabelName ? l.pLabelName : "";
            }
            return out;
        };
        if (data->cmdBufLabelCount > 0) {
            std::println(stderr, "    command buffer labels: {}", labels(data->pCmdBufLabels, data->cmdBufLabelCount));
        }
        if (data->queueLabelCount > 0) {
            std::println(stderr, "    queue labels: {}", labels(data->pQueueLabels, data->queueLabelCount));
        }
    } catch (...) {
    }
    return vk::False;
}
// #endregion

// #region object-names
// Names Vulkan objects through vkSetDebugUtilsObjectNameEXT and counts them. The
// object type comes from the handle's C++ type (Vulkan-Hpp's Handle::objectType),
// so a name can never be attached under the wrong VkObjectType
// (VUID-VkDebugUtilsObjectNameInfoEXT-objectType-02590). The layer copies the
// string: it need not outlive the call.
struct Names {
    const vk::raii::Device& device;
    uint32_t count = 0;

    template <typename Handle> void set(Handle handle, const std::string& name) {
        device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
            .objectType   = Handle::objectType,
            .objectHandle = reinterpret_cast<uint64_t>(static_cast<typename Handle::CType>(handle)),
            .pObjectName  = name.c_str(),
        });
        ++count;
    }
};
// #endregion

// #region labels
// A label for vkCmd*/vkQueue*DebugUtilsLabelEXT: a name and an RGBA color (tools
// may use the color; all zeros means "no color").
vk::DebugUtilsLabelEXT label(const char* name, std::array<float, 4> color) {
    return vk::DebugUtilsLabelEXT{.pLabelName = name, .color = color};
}

constexpr std::array grey{0.6f, 0.6f, 0.6f, 1.0f};
constexpr std::array blue{0.3f, 0.5f, 0.9f, 1.0f};
constexpr std::array green{0.3f, 0.8f, 0.4f, 1.0f};
constexpr std::array orange{0.95f, 0.6f, 0.2f, 1.0f};
// #endregion

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

// Position + color: 3 + 3 floats = 24 bytes (shaders/cube.slang).
struct Vertex {
    std::array<float, 3> position;
    std::array<float, 3> color; // linear RGB, one color per face
};
static_assert(sizeof(Vertex) == 24);

// Step 22's unit cube: 4 vertices per face, counter-clockwise from outside.
constexpr std::array<Vertex, 24> vertices{{
    // +x, red
    {{0.5f, -0.5f, -0.5f}, {0.9f, 0.2f, 0.2f}},
    {{0.5f, 0.5f, -0.5f}, {0.9f, 0.2f, 0.2f}},
    {{0.5f, 0.5f, 0.5f}, {0.9f, 0.2f, 0.2f}},
    {{0.5f, -0.5f, 0.5f}, {0.9f, 0.2f, 0.2f}},
    // -x, cyan
    {{-0.5f, -0.5f, -0.5f}, {0.2f, 0.8f, 0.8f}},
    {{-0.5f, -0.5f, 0.5f}, {0.2f, 0.8f, 0.8f}},
    {{-0.5f, 0.5f, 0.5f}, {0.2f, 0.8f, 0.8f}},
    {{-0.5f, 0.5f, -0.5f}, {0.2f, 0.8f, 0.8f}},
    // +y, green
    {{-0.5f, 0.5f, -0.5f}, {0.2f, 0.85f, 0.3f}},
    {{-0.5f, 0.5f, 0.5f}, {0.2f, 0.85f, 0.3f}},
    {{0.5f, 0.5f, 0.5f}, {0.2f, 0.85f, 0.3f}},
    {{0.5f, 0.5f, -0.5f}, {0.2f, 0.85f, 0.3f}},
    // -y, magenta
    {{-0.5f, -0.5f, -0.5f}, {0.8f, 0.3f, 0.8f}},
    {{0.5f, -0.5f, -0.5f}, {0.8f, 0.3f, 0.8f}},
    {{0.5f, -0.5f, 0.5f}, {0.8f, 0.3f, 0.8f}},
    {{-0.5f, -0.5f, 0.5f}, {0.8f, 0.3f, 0.8f}},
    // +z, blue
    {{-0.5f, -0.5f, 0.5f}, {0.25f, 0.4f, 0.95f}},
    {{0.5f, -0.5f, 0.5f}, {0.25f, 0.4f, 0.95f}},
    {{0.5f, 0.5f, 0.5f}, {0.25f, 0.4f, 0.95f}},
    {{-0.5f, 0.5f, 0.5f}, {0.25f, 0.4f, 0.95f}},
    // -z, yellow
    {{-0.5f, -0.5f, -0.5f}, {0.95f, 0.85f, 0.2f}},
    {{-0.5f, 0.5f, -0.5f}, {0.95f, 0.85f, 0.2f}},
    {{0.5f, 0.5f, -0.5f}, {0.95f, 0.85f, 0.2f}},
    {{0.5f, -0.5f, -0.5f}, {0.95f, 0.85f, 0.2f}},
}};
// Two triangles per face: (0, 1, 2) and (0, 2, 3) of its 4 corners.
constexpr std::array<uint16_t, 36> indices{
    0,  1,  2,  0,  2,  3,  //
    4,  5,  6,  4,  6,  7,  //
    8,  9,  10, 8,  10, 11, //
    12, 13, 14, 12, 14, 15, //
    16, 17, 18, 16, 18, 19, //
    20, 21, 22, 20, 22, 23, //
};

// A 4x4 matrix, column-major: m[c] is column c (step 17).
using Mat4 = std::array<std::array<float, 4>, 4>;

// The host side of Scene in shaders/cube.slang: one matrix per cube.
struct Scene {
    std::array<Mat4, 2> mvp;
};
static_assert(sizeof(Scene) == 128);

// a * b for column-major matrices: (a * b) * v = a * (b * v).
Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 m{};
    for (size_t c = 0; c < 4; ++c) {
        for (size_t r = 0; r < 4; ++r) {
            float sum = 0.0f;
            for (size_t k = 0; k < 4; ++k) {
                sum += a[k][r] * b[c][k];
            }
            m[c][r] = sum;
        }
    }
    return m;
}

// Moves points by (x, y, z).
Mat4 translate(float x, float y, float z) {
    return {{
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {x, y, z, 1.0f},
    }};
}

// Rotation by a radians about +x.
Mat4 rotateX(float a) {
    const float s = std::sin(a);
    const float c = std::cos(a);
    return {{
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, c, s, 0.0f},
        {0.0f, -s, c, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
    }};
}

// Rotation by a radians about +y.
Mat4 rotateY(float a) {
    const float s = std::sin(a);
    const float c = std::cos(a);
    return {{
        {c, 0.0f, -s, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {s, 0.0f, c, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
    }};
}

// Step 22's perspective projection into Vulkan's clip space (y down, depth 0..1).
Mat4 perspective(float fovy, float aspect, float nearZ, float farZ) {
    const float f = 1.0f / std::tan(fovy / 2.0f);
    return {{
        {f / aspect, 0.0f, 0.0f, 0.0f},
        {0.0f, -f, 0.0f, 0.0f},
        {0.0f, 0.0f, farZ / (nearZ - farZ), -1.0f},
        {0.0f, 0.0f, nearZ * farZ / (nearZ - farZ), 0.0f},
    }};
}

// Camera: 60 degrees vertical field of view, depth range 0.1..100.
constexpr float fovy  = std::numbers::pi_v<float> / 3.0f;
constexpr float nearZ = 0.1f;
constexpr float farZ  = 100.0f;

// Step 22's scene: both cubes' model-view-projection matrices at time seconds.
Scene scene(float time, const Mat4& proj) {
    const Mat4 view     = translate(0.0f, 0.0f, -3.5f);
    const Mat4 model0   = mul(rotateY(time * 0.7f), rotateX(time * 0.4f));
    const Mat4 model1   = mul(translate(0.45f, 0.15f, 0.1f), mul(rotateY(-time * 0.5f), rotateX(0.6f)));
    const Mat4 viewProj = mul(proj, view);
    return Scene{.mvp = {mul(viewProj, model0), mul(viewProj, model1)}};
}

// #region cli
// Command line: --printf, --seconds S.
struct Args {
    bool printf = false;
    std::optional<double> seconds;
};

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--printf") {
            args.printf = true;
        } else if (arg == "--seconds") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--seconds needs a value");
            }
            const std::string_view text = argv[++i];
            double seconds              = 0.0;
            const auto [end, ec]        = std::from_chars(text.data(), text.data() + text.size(), seconds);
            if (ec != std::errc{} || end != text.data() + text.size()) {
                throw std::runtime_error(std::format("--seconds: not a number: {}", text));
            }
            args.seconds = seconds;
        } else {
            throw std::runtime_error(std::format("unknown argument {} (use --printf, --seconds S)", arg));
        }
    }
    return args;
}
// #endregion

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
        throw std::runtime_error("no suitable memory type for this buffer");
    }
    vk::raii::DeviceMemory memory(device, vk::MemoryAllocateInfo{
                                              .allocationSize  = req.size,
                                              .memoryTypeIndex = *memoryType,
                                          });
    buffer.bindMemory(*memory, 0);
    return GpuBuffer{.memory = std::move(memory), .buffer = std::move(buffer)};
}

// Step 14's upload: map, memcpy, unmap a HOST_VISIBLE | HOST_COHERENT
// allocation (no flush needed; the submit makes the writes visible).
template <typename T, size_t N> void writeMapped(const vk::raii::DeviceMemory& memory, std::span<T, N> data) {
    void* ptr = memory.mapMemory(0, vk::WholeSize);
    std::memcpy(ptr, data.data(), data.size_bytes());
    memory.unmapMemory();
}

// Step 22's depth formats in order of preference.
constexpr std::array depthCandidates{
    vk::Format::eD32Sfloat,
    vk::Format::eD24UnormS8Uint,
    vk::Format::eD16Unorm,
};

// The first candidate whose OPTIMAL tiling supports DEPTH_STENCIL_ATTACHMENT.
vk::Format pickDepthFormat(const vk::raii::PhysicalDevice& physicalDevice) {
    for (const vk::Format format : depthCandidates) {
        if (physicalDevice.getFormatProperties2(format).formatProperties.optimalTilingFeatures &
            vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
            return format;
        }
    }
    throw std::runtime_error("no depth format supports DEPTH_STENCIL_ATTACHMENT");
}

// Whether format also has a stencil component (step 22).
bool hasStencil(vk::Format format) {
    return format == vk::Format::eD16UnormS8Uint || format == vk::Format::eD24UnormS8Uint || format == vk::Format::eD32SfloatS8Uint;
}

// What the CPU touches while recording one frame (step 17).
struct Frame {
    vk::raii::CommandBuffer commandBuffer;
    vk::raii::Semaphore imageAvailable;
    vk::raii::Fence inFlight;
    GpuBuffer uniform;
    Scene* mapped;
    vk::DescriptorSet descriptorSet;
};

int main(int argc, char** argv)
try {
    const Args args = parseArgs(argc, argv);

    // ---- Step 06: window (800x600 physical pixels, fixed size) ----
    Glfw glfw;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 28 debug labels", nullptr, nullptr), &glfwDestroyWindow);
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
        .pApplicationName   = "Vulkan from Zero - 28 debug labels",
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
    // Surface extensions from step 06, plus VK_EXT_debug_utils (step 02): the
    // messenger, names and labels are all this one extension.
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
    const std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }
    std::println("Found {} GPU(s)", devices.size());
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
        throw std::runtime_error(
            "No Vulkan 1.4 GPU with dynamic rendering + synchronization2 + VK_KHR_swapchain can present to this window");
    }
    const vk::raii::PhysicalDevice& physicalDevice = *best;
    const std::string gpuName{physicalDevice.getProperties().deviceName.data()};
    std::println("Picked: {}", gpuName);
    std::println("Graphics + present family: {}", family);

    // Step 07's device: one queue, the two 1.3 features, VK_KHR_swapchain. Names
    // and labels need no device extension: VK_EXT_debug_utils is an instance
    // extension whose device-level commands work on any device of that instance.
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
    vk::raii::Queue queue(device, family, 0);

    // #region name-objects
    // Names can be given as soon as an object exists, from the device's
    // VK_EXT_debug_utils commands; instance-level objects (the instance, the
    // surface) and the physical device too, as long as the device descends from
    // the same instance and physical device
    // (VUID-vkSetDebugUtilsObjectNameEXT-pNameInfo-07872/07873). Only the layer
    // and the tools keep the strings. The messenger stays unnamed: naming it is
    // valid usage, but the Vulkan-Loader cannot pass it on. The handle the
    // application gets is the loader's own pointer
    // (terminator_CreateDebugUtilsMessengerEXT in loader/debug_utils.c returns
    // its pNextIndex allocation), and terminator_SetDebugUtilsObjectNameEXT
    // (loader/generated/vk_loader_extensions.c) translates only
    // PHYSICAL_DEVICE, SURFACE_KHR and INSTANCE handles, so the driver receives
    // a handle it never created. On the development machine (SDK 1.4.341
    // loader, NVIDIA 616.64) the call crashed with an access violation.
    Names names{.device = device};
    names.set(*instance, "instance");
    names.set(*surface, "window surface");
    names.set(*physicalDevice, gpuName);
    names.set(*device, "device");
    names.set(*queue, "graphics + present queue");
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
    names.set(*swapchain, "swapchain");
    const std::vector<vk::Image> images = swapchain.getImages();
    std::vector<vk::raii::ImageView> imageViews;
    imageViews.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        // The swapchain owns its images, but they are VkImages like any other and
        // take names the same way.
        names.set(images[i], std::format("swapchain image {}", i));
        imageViews.emplace_back(device, vk::ImageViewCreateInfo{
                                            .image            = images[i],
                                            .viewType         = vk::ImageViewType::e2D,
                                            .format           = format.format,
                                            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                                 .levelCount = 1,
                                                                 .layerCount = 1},
                                        });
        names.set(*imageViews.back(), std::format("swapchain view {}", i));
    }
    std::println("Swapchain: {}x{} {}, {}, {} images", extent.width, extent.height, formatName(format.format),
                 shortName(string_VkPresentModeKHR(static_cast<VkPresentModeKHR>(presentMode)), "VK_PRESENT_MODE_", "_KHR"),
                 images.size());
    const vk::PhysicalDeviceMemoryProperties mem = physicalDevice.getMemoryProperties();

    // Step 22's depth image, one for both frames in flight.
    const vk::Format depthFormat = pickDepthFormat(physicalDevice);
    const bool stencil           = hasStencil(depthFormat);
    const vk::ImageLayout depthLayout =
        stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal : vk::ImageLayout::eDepthAttachmentOptimal;
    const vk::ImageAspectFlags depthAspect =
        stencil ? vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil : vk::ImageAspectFlagBits::eDepth;
    vk::raii::Image depthImage(device, vk::ImageCreateInfo{
                                           .imageType     = vk::ImageType::e2D,
                                           .format        = depthFormat,
                                           .extent        = {extent.width, extent.height, 1},
                                           .mipLevels     = 1,
                                           .arrayLayers   = 1,
                                           .samples       = vk::SampleCountFlagBits::e1,
                                           .tiling        = vk::ImageTiling::eOptimal,
                                           .usage         = vk::ImageUsageFlagBits::eDepthStencilAttachment,
                                           .sharingMode   = vk::SharingMode::eExclusive,
                                           .initialLayout = vk::ImageLayout::eUndefined,
                                       });
    names.set(*depthImage, "depth image");
    const vk::MemoryRequirements req = depthImage.getMemoryRequirements();
    const auto depthType             = findType(mem, req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    if (!depthType) {
        throw std::runtime_error("no DEVICE_LOCAL memory type for the depth image");
    }
    vk::raii::DeviceMemory depthMemory(device, vk::MemoryAllocateInfo{.allocationSize = req.size, .memoryTypeIndex = *depthType});
    names.set(*depthMemory, "depth memory");
    depthImage.bindMemory(*depthMemory, 0);
    vk::raii::ImageView depthView(device, vk::ImageViewCreateInfo{
                                              .image            = *depthImage,
                                              .viewType         = vk::ImageViewType::e2D,
                                              .format           = depthFormat,
                                              .subresourceRange = {.aspectMask = depthAspect, .levelCount = 1, .layerCount = 1},
                                          });
    names.set(*depthView, "depth view");
    std::println("Depth: {}, one image for both frames in flight", formatName(depthFormat));

    const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    const Mat4 proj    = perspective(fovy, aspect, nearZ, farZ);

    // The SPIR-V is compiled from shaders/*.slang by `bun run sync:shaders`;
    // CMake bakes the absolute path of shaders/spv into SHADER_DIR.
    const std::vector<uint32_t> spirv = readSpirv(std::filesystem::path{SHADER_DIR} / "cube.spv");
    std::optional<vk::raii::ShaderModule> shaderModule;
    shaderModule.emplace(device, vk::ShaderModuleCreateInfo{
                                     .codeSize = spirv.size() * sizeof(uint32_t),
                                     .pCode    = spirv.data(),
                                 });
    names.set(**shaderModule, "cube.spv");
    std::optional<vk::raii::ShaderModule> printfModule;
    if (args.printf) {
        const std::vector<uint32_t> printfSpirv = readSpirv(std::filesystem::path{SHADER_DIR} / "debug_printf.spv");
        printfModule.emplace(device, vk::ShaderModuleCreateInfo{
                                         .codeSize = printfSpirv.size() * sizeof(uint32_t),
                                         .pCode    = printfSpirv.data(),
                                     });
        names.set(**printfModule, "debug_printf.spv");
    }

    // Vertex input: binding 0, stride 24, position + color (3 floats each).
    const vk::VertexInputBindingDescription vertexBinding{
        .binding   = 0,
        .stride    = sizeof(Vertex),
        .inputRate = vk::VertexInputRate::eVertex,
    };
    const std::array attributes{
        vk::VertexInputAttributeDescription{
            .location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, position)},
        vk::VertexInputAttributeDescription{
            .location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, color)},
    };
    const vk::PipelineVertexInputStateCreateInfo vertexInput{
        .vertexBindingDescriptionCount   = 1,
        .pVertexBindingDescriptions      = &vertexBinding,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size()),
        .pVertexAttributeDescriptions    = attributes.data(),
    };

    // Step 17's set layout: binding 0 = one uniform buffer, vertex stage.
    const vk::DescriptorSetLayoutBinding setBinding{
        .binding         = 0,
        .descriptorType  = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags      = vk::ShaderStageFlagBits::eVertex,
    };
    vk::raii::DescriptorSetLayout setLayout(device, vk::DescriptorSetLayoutCreateInfo{.bindingCount = 1, .pBindings = &setBinding});
    names.set(*setLayout, "scene set layout");
    const vk::DescriptorSetLayout setLayoutHandle = *setLayout;
    vk::raii::PipelineLayout pipelineLayout(device, vk::PipelineLayoutCreateInfo{.setLayoutCount = 1, .pSetLayouts = &setLayoutHandle});
    names.set(*pipelineLayout, "cube pipeline layout");

    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
    const vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
    const vk::PipelineRasterizationStateCreateInfo rasterization{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode    = vk::CullModeFlagBits::eBack,
        .frontFace   = vk::FrontFace::eCounterClockwise,
        .lineWidth   = 1.0f,
    };
    const vk::PipelineMultisampleStateCreateInfo multisample{.rasterizationSamples = vk::SampleCountFlagBits::e1};
    const vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable  = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp   = vk::CompareOp::eLess,
    };
    const vk::PipelineRenderingCreateInfo renderingInfo{
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &format.format,
        .depthAttachmentFormat   = depthFormat,
        .stencilAttachmentFormat = stencil ? depthFormat : vk::Format::eUndefined,
    };
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

    // #region printf-pipeline
    // One pipeline per fragment shader: the regular one, and with --printf a
    // second one whose fragment stage comes from another module
    // (debug_printf.spv) while the vertex stage stays cube.spv's. Both are
    // created in one vkCreateGraphicsPipelines call.
    std::vector<std::array<vk::PipelineShaderStageCreateInfo, 2>> stageSets;
    const auto addVariant = [&](vk::ShaderModule fragmentModule, const char* entry) {
        stageSets.push_back({
            vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = **shaderModule, .pName = "vertexMain"},
            vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = fragmentModule, .pName = entry},
        });
    };
    addVariant(**shaderModule, "fragmentMain");
    if (printfModule) {
        addVariant(**printfModule, "fragmentPrintf");
    }
    std::vector<vk::GraphicsPipelineCreateInfo> pipelineInfos;
    for (const auto& stages : stageSets) {
        pipelineInfos.push_back(vk::GraphicsPipelineCreateInfo{
            .pNext               = &renderingInfo,
            .stageCount          = static_cast<uint32_t>(stages.size()),
            .pStages             = stages.data(),
            .pVertexInputState   = &vertexInput,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState      = &viewportState,
            .pRasterizationState = &rasterization,
            .pMultisampleState   = &multisample,
            .pDepthStencilState  = &depthStencil,
            .pColorBlendState    = &colorBlend,
            .pDynamicState       = &dynamicState,
            .layout              = *pipelineLayout,
        });
    }
    vk::raii::Pipelines pipelines(device, nullptr, pipelineInfos);
    shaderModule.reset();
    printfModule.reset();
    const vk::raii::Pipeline& pipeline = pipelines.at(0);
    names.set(*pipeline, "cube pipeline");
    const vk::raii::Pipeline* printfPipeline = pipelines.size() > 1 ? &pipelines.at(1) : nullptr;
    if (printfPipeline != nullptr) {
        names.set(**printfPipeline, "cube pipeline (printf)");
        std::println("--printf: frame 0 uses fragmentPrintf (debug_printf.spv); output needs "
                     "VK_KHRONOS_VALIDATION_PRINTF_ENABLE=true + VK_KHRONOS_VALIDATION_PRINTF_TO_STDOUT=true");
    }
    // #endregion

    // Step 14's mesh buffers: HOST_VISIBLE | HOST_COHERENT, written once.
    const vk::MemoryPropertyFlags hostFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    const std::span vertexData{vertices};
    const std::span indexData{indices};
    GpuBuffer vertexBuffer = createBuffer(device, mem, vertexData.size_bytes(), vk::BufferUsageFlagBits::eVertexBuffer, hostFlags);
    names.set(*vertexBuffer.buffer, "cube vertices");
    names.set(*vertexBuffer.memory, "cube vertex memory");
    writeMapped(vertexBuffer.memory, vertexData);
    GpuBuffer indexBuffer = createBuffer(device, mem, indexData.size_bytes(), vk::BufferUsageFlagBits::eIndexBuffer, hostFlags);
    names.set(*indexBuffer.buffer, "cube indices");
    names.set(*indexBuffer.memory, "cube index memory");
    writeMapped(indexBuffer.memory, indexData);

    // Step 17's per-frame uniform buffers, persistently mapped, and one
    // descriptor set per frame in flight (plain handles: they go with the pool).
    constexpr vk::DeviceSize uniformSize = sizeof(Scene);
    constexpr auto frameCount            = static_cast<uint32_t>(framesInFlight);
    const vk::DescriptorPoolSize poolSize{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = frameCount};
    vk::raii::DescriptorPool descriptorPool(device, vk::DescriptorPoolCreateInfo{
                                                        .maxSets       = frameCount,
                                                        .poolSizeCount = 1,
                                                        .pPoolSizes    = &poolSize,
                                                    });
    names.set(*descriptorPool, "descriptor pool");
    const std::vector<vk::DescriptorSetLayout> setLayouts(framesInFlight, *setLayout);
    std::vector<vk::DescriptorSet> descriptorSets;
    for (vk::raii::DescriptorSet& set : vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
                                                                             .descriptorPool     = *descriptorPool,
                                                                             .descriptorSetCount = frameCount,
                                                                             .pSetLayouts        = setLayouts.data(),
                                                                         })) {
        descriptorSets.push_back(set.release());
    }
    std::vector<GpuBuffer> uniforms;
    std::vector<Scene*> mapped;
    for (size_t slot = 0; slot < descriptorSets.size(); ++slot) {
        const vk::DescriptorSet set = descriptorSets[slot];
        uniforms.push_back(createBuffer(device, mem, uniformSize, vk::BufferUsageFlagBits::eUniformBuffer, hostFlags));
        names.set(*uniforms.back().buffer, std::format("frame {} scene uniforms", slot));
        names.set(*uniforms.back().memory, std::format("frame {} scene memory", slot));
        names.set(set, std::format("frame {} descriptor set", slot));
        mapped.push_back(static_cast<Scene*>(uniforms.back().memory.mapMemory(0, vk::WholeSize)));
        const vk::DescriptorBufferInfo info{.buffer = *uniforms.back().buffer, .offset = 0, .range = uniformSize};
        device.updateDescriptorSets(vk::WriteDescriptorSet{
                                        .dstSet          = set,
                                        .dstBinding      = 0,
                                        .descriptorCount = 1,
                                        .descriptorType  = vk::DescriptorType::eUniformBuffer,
                                        .pBufferInfo     = &info,
                                    },
                                    nullptr);
    }

    // Steps 09-10: the pool, per-frame buffers and sync objects.
    vk::raii::CommandPool commandPool(device, vk::CommandPoolCreateInfo{
                                                  .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                  .queueFamilyIndex = family,
                                              });
    names.set(*commandPool, "frame command pool");
    vk::raii::CommandBuffers commandBuffers(device, vk::CommandBufferAllocateInfo{
                                                        .commandPool        = *commandPool,
                                                        .level              = vk::CommandBufferLevel::ePrimary,
                                                        .commandBufferCount = frameCount,
                                                    });
    std::vector<Frame> frames;
    frames.reserve(framesInFlight);
    for (size_t slot = 0; slot < framesInFlight; ++slot) {
        frames.push_back(Frame{
            .commandBuffer  = std::move(commandBuffers[slot]),
            .imageAvailable = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{}),
            .inFlight       = vk::raii::Fence(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled}),
            .uniform        = std::move(uniforms[slot]),
            .mapped         = mapped[slot],
            .descriptorSet  = descriptorSets[slot],
        });
        names.set(*frames.back().commandBuffer, std::format("frame {} command buffer", slot));
        names.set(*frames.back().imageAvailable, std::format("frame {} image available", slot));
        names.set(*frames.back().inFlight, std::format("frame {} in flight", slot));
    }
    std::vector<vk::raii::Semaphore> renderFinished;
    renderFinished.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        renderFinished.emplace_back(device, vk::SemaphoreCreateInfo{});
        names.set(*renderFinished.back(), std::format("render finished {}", i));
    }
    std::println("Named {} objects with vkSetDebugUtilsObjectNameEXT", names.count);
    std::println("Labels per frame: queue \"frame N\" around submit + present; command buffer \"frame\" > \"attachments\", "
                 "\"main pass\" > \"cube 0\", \"cube 1\"; inserted \"to present\"");

    using Clock       = std::chrono::steady_clock;
    using Seconds     = std::chrono::duration<double>;
    const auto start  = Clock::now();
    size_t frameIndex = 0;
    // The animation clock: starts at frame 0, so frame 0 always shows the scene
    // at time 0 (and the printf output of --printf is reproducible).
    std::optional<Clock::time_point> animStart;

    // Renders and presents one frame (step 22's loop) with labels around its
    // parts. Returns false when the swapchain is out of date.
    const auto drawFrame = [&]() -> bool {
        Frame& frame                       = frames.at(frameIndex % frames.size());
        const vk::raii::CommandBuffer& cmd = frame.commandBuffer;
        if (device.waitForFences(*frame.inFlight, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
            throw std::runtime_error("waitForFences failed");
        }
        if (!animStart) {
            animStart = Clock::now();
        }
        const auto time       = static_cast<float>(Seconds(Clock::now() - *animStart).count());
        const Scene sceneData = scene(time, proj);
        std::memcpy(frame.mapped, &sceneData, sizeof sceneData);

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
        // Step 22's depth barrier: WAW against the previous frame's depth writes.
        constexpr vk::PipelineStageFlags2 fragmentTests =
            vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
        const vk::ImageMemoryBarrier2 depthBarrier{
            .srcStageMask        = fragmentTests,
            .srcAccessMask       = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            .dstStageMask        = fragmentTests,
            .dstAccessMask       = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            .oldLayout           = vk::ImageLayout::eUndefined,
            .newLayout           = depthLayout,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image               = *depthImage,
            .subresourceRange    = {.aspectMask = depthAspect, .levelCount = 1, .layerCount = 1},
        };
        const vk::RenderingAttachmentInfo colorAttachment{
            .imageView   = *imageViews.at(imageIndex),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eStore,
            .clearValue  = vk::ClearColorValue{std::array{0.02f, 0.02f, 0.03f, 1.0f}},
        };
        const vk::RenderingAttachmentInfo depthAttachment{
            .imageView   = *depthView,
            .imageLayout = depthLayout,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eDontCare,
            .clearValue  = vk::ClearDepthStencilValue{.depth = 1.0f, .stencil = 0},
        };
        const vk::RenderingInfo renderingAttachments{
            .renderArea           = {.offset = {0, 0}, .extent = extent},
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &colorAttachment,
            .pDepthAttachment     = &depthAttachment,
            .pStencilAttachment   = stencil ? &depthAttachment : nullptr,
        };
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
        // --printf: frame 0 (the scene at time 0) uses the printf pipeline.
        const vk::Pipeline framePipeline = printfPipeline != nullptr && frameIndex == 0 ? **printfPipeline : *pipeline;

        // #region cmd-labels
        // Label regions nest like brackets and only mark the command stream: they
        // cost a few bytes per command buffer and change nothing the GPU does.
        // "cube 0" and "cube 1" are separate draws here (one instance each,
        // firstInstance 0 and 1: cube.slang reads mvp[InstanceIndex]), so a
        // capture tool lists them under their own labels. A region may even start
        // in one command buffer and end in another: the rule is that, along the
        // series of submissions to one queue, begins and ends "must be matched
        // and balanced" (spec text); an End without an outstanding Begin is
        // VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-01912. Only a SECONDARY command buffer must close every region it opens
        // (VUID-vkEndCommandBuffer-commandBuffer-01815). This frame closes all of
        // its own.
        cmd.beginDebugUtilsLabelEXT(label("frame", grey));

        cmd.beginDebugUtilsLabelEXT(label("attachments", orange));
        const std::array barriers{toAttachment, depthBarrier};
        cmd.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
            .pImageMemoryBarriers    = barriers.data(),
        });
        cmd.endDebugUtilsLabelEXT();

        cmd.beginDebugUtilsLabelEXT(label("main pass", blue));
        cmd.beginRendering(renderingAttachments);
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, framePipeline);
        cmd.setViewport(0, vk::Viewport{
                               .x        = 0.0f,
                               .y        = 0.0f,
                               .width    = static_cast<float>(extent.width),
                               .height   = static_cast<float>(extent.height),
                               .minDepth = 0.0f,
                               .maxDepth = 1.0f,
                           });
        cmd.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = extent});
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, frame.descriptorSet, nullptr);
        cmd.bindVertexBuffers(0, *vertexBuffer.buffer, vk::DeviceSize{0});
        cmd.bindIndexBuffer(*indexBuffer.buffer, 0, vk::IndexType::eUint16);
        for (const auto& [instance, name] : {std::pair{0u, "cube 0"}, std::pair{1u, "cube 1"}}) {
            cmd.beginDebugUtilsLabelEXT(label(name, green));
            cmd.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, instance);
            cmd.endDebugUtilsLabelEXT();
        }
        cmd.endRendering();
        cmd.endDebugUtilsLabelEXT();

        // A single point in the stream rather than a region.
        cmd.insertDebugUtilsLabelEXT(label("to present", orange));
        cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toPresent});

        cmd.endDebugUtilsLabelEXT();
        cmd.end();
        // #endregion

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
        const vk::SwapchainKHR swapchainHandle = *swapchain;

        // #region queue-labels
        // A queue label region brackets queue operations instead of commands: here
        // the frame's submit and its present, named with the frame number. Like
        // command buffer labels, queue label begins and ends "must be matched and
        // balanced" (spec text); an End without an outstanding Begin is
        // VUID-vkQueueEndDebugUtilsLabelEXT-None-01911. So the region is closed on
        // every path, also when submit or present throw (the guard's destructor).
        const std::string frameName = std::format("frame {}", frameIndex);
        queue.beginDebugUtilsLabelEXT(label(frameName.c_str(), grey));
        struct QueueLabelEnd {
            const vk::raii::Queue& queue;
            ~QueueLabelEnd() { queue.endDebugUtilsLabelEXT(); }
        } const queueLabelEnd{queue};
        queue.submit2(vk::SubmitInfo2{
                          .waitSemaphoreInfoCount   = 1,
                          .pWaitSemaphoreInfos      = &waitInfo,
                          .commandBufferInfoCount   = 1,
                          .pCommandBufferInfos      = &cmdInfo,
                          .signalSemaphoreInfoCount = 1,
                          .pSignalSemaphoreInfos    = &signalInfo,
                      },
                      *frame.inFlight);
        queue.insertDebugUtilsLabelEXT(label("present", orange));
        const vk::Result presented = queue.presentKHR(vk::PresentInfoKHR{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &signalSemaphore,
            .swapchainCount     = 1,
            .pSwapchains        = &swapchainHandle,
            .pImageIndices      = &imageIndex,
        });
        // #endregion
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
    device.waitIdle();
    return EXIT_SUCCESS;
} catch (const vk::SystemError& e) {
    std::println(stderr, "Vulkan error: {}", e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return EXIT_FAILURE;
}
