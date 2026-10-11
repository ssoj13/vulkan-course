// Step 22 - Depth buffer
//
// Two cubes cut through each other. Each cube alone looks right with back faces
// culled (it is convex), but where they intersect, which surface is in front
// changes from pixel to pixel, and no draw order can get that right. A DEPTH
// ATTACHMENT can: an image with one depth value per pixel; every fragment
// compares its depth with the stored one (the depth TEST) and, if it is nearer,
// replaces it (the depth WRITE) and its color is kept.
//
//   pick a depth format (format properties) -> depth image + view
//   frame: barrier (UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL, after the previous
//     frame's depth writes) -> rendering with pDepthAttachment (clear to 1.0)
//   pipeline: depthTestEnable, depthWriteEnable, compareOp LESS
//   vertex shader: perspective projection into Vulkan's clip space
//
// --no-depth turns the test and the write off (the attachment stays) to show
// what goes wrong; --seconds S closes the window after S seconds. Lines that
// depend on timing start with [timing].
// Steps 01-17 are repeated, reduced to what this step needs.

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
std::string layoutName(vk::ImageLayout l) {
    return shortName(string_VkImageLayout(static_cast<VkImageLayout>(l)), "VK_IMAGE_LAYOUT_");
}

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

// #region cube
// A unit cube (corners at +-0.5), 4 vertices per face so each face has its own
// color. Model space is right-handed with +y up. Every face's corners go
// counter-clockwise seen from OUTSIDE the cube; the projection below keeps that
// orientation on screen, so the pipeline's front face is COUNTER_CLOCKWISE and
// back-face culling drops the far side.
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
// #endregion

// A 4x4 matrix, column-major: m[c] is column c (step 17).
using Mat4 = std::array<std::array<float, 4>, 4>;

// The host side of Scene in shaders/cube.slang: one matrix per cube (std140
// array stride 64 = the size of a Mat4, so no padding).
struct Scene {
    std::array<Mat4, 2> mvp;
};
static_assert(sizeof(Scene) == 128);

// #region math
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

// Rotation by a radians about +x (right-handed: +y turns toward +z).
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

// Rotation by a radians about +y (right-handed: +z turns toward +x).
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
// #endregion

// #region projection
// Perspective projection from a right-handed view space (camera at the origin
// looking down -z, +y up) to VULKAN's clip space, which differs from OpenGL's in
// two ways: +y points DOWN the screen, and depth runs 0..1 (OpenGL: -1..1).
// With f = 1 / tan(fovy / 2):
//
//   x_clip = (f / aspect) * x      the field of view, widened by aspect
//   y_clip = -f * y                minus: view +y up -> screen +y down
//   z_clip = A * z + B             A = far / (near - far),
//                                  B = near * far / (near - far)
//   w_clip = -z                    the distance in front of the camera
//
// After the divide by w: z = -near gives depth 0, z = -far gives depth 1.
// Depth is then proportional to 1/distance, so most of its precision sits near
// the camera. REVERSED-Z (near -> 1, far -> 0, compare GREATER, clear to 0)
// pairs that with the float format's own precision near 0 and is the usual
// choice for large scenes with D32_SFLOAT; this step keeps the classic mapping.
Mat4 perspective(float fovy, float aspect, float nearZ, float farZ) {
    const float f = 1.0f / std::tan(fovy / 2.0f);
    return {{
        {f / aspect, 0.0f, 0.0f, 0.0f},
        {0.0f, -f, 0.0f, 0.0f},
        {0.0f, 0.0f, farZ / (nearZ - farZ), -1.0f},
        {0.0f, 0.0f, nearZ * farZ / (nearZ - farZ), 0.0f},
    }};
}
// #endregion

// Camera: 60 degrees vertical field of view, depth range 0.1..100.
constexpr float fovy  = std::numbers::pi_v<float> / 3.0f;
constexpr float nearZ = 0.1f;
constexpr float farZ  = 100.0f;

// #region scene
// Both cubes' model-view-projection matrices at time seconds. The camera sits
// 3.5 units in front of the origin (view = translate by -3.5 in z); cube 0
// tumbles at the origin, cube 1 is shifted by less than a cube's size and turns
// the other way, so the two always intersect.
Scene scene(float time, const Mat4& proj) {
    const Mat4 view     = translate(0.0f, 0.0f, -3.5f);
    const Mat4 model0   = mul(rotateY(time * 0.7f), rotateX(time * 0.4f));
    const Mat4 model1   = mul(translate(0.45f, 0.15f, 0.1f), mul(rotateY(-time * 0.5f), rotateX(0.6f)));
    const Mat4 viewProj = mul(proj, view);
    return Scene{.mvp = {mul(viewProj, model0), mul(viewProj, model1)}};
}
// #endregion

// #region cli
// Command line: --no-depth, --seconds S.
struct Args {
    bool depthTest = true;
    std::optional<double> seconds;
};

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--no-depth") {
            args.depthTest = false;
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
            throw std::runtime_error(std::format("unknown argument {} (use --no-depth, --seconds S)", arg));
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
    return GpuBuffer{.memory = std::move(memory), .buffer = std::move(buffer)};
}

// Step 14's upload: map, memcpy, unmap a HOST_VISIBLE | HOST_COHERENT
// allocation (no flush needed; the submit makes the writes visible).
template <typename T, size_t N> void writeMapped(const vk::raii::DeviceMemory& memory, std::span<T, N> data) {
    void* ptr = memory.mapMemory(0, vk::WholeSize);
    std::memcpy(ptr, data.data(), data.size_bytes());
    memory.unmapMemory();
}

// #region depth-format
// Depth formats in order of preference. The spec guarantees depth-attachment
// support for D16_UNORM, for at least one of X8_D24_UNORM_PACK32 and
// D32_SFLOAT, and for at least one of D24_UNORM_S8_UINT and D32_SFLOAT_S8_UINT;
// everything else is queried. D32_SFLOAT first: a float depth buffer, 32 bits,
// no stencil.
constexpr std::array depthCandidates{
    vk::Format::eD32Sfloat,
    vk::Format::eD24UnormS8Uint,
    vk::Format::eD16Unorm,
};

// The first candidate whose OPTIMAL tiling supports DEPTH_STENCIL_ATTACHMENT,
// printing the answer for each.
vk::Format pickDepthFormat(const vk::raii::PhysicalDevice& physicalDevice) {
    std::println("Depth format candidates (optimal tiling, DEPTH_STENCIL_ATTACHMENT):");
    std::optional<vk::Format> pick;
    for (const vk::Format format : depthCandidates) {
        const bool ok = static_cast<bool>(physicalDevice.getFormatProperties2(format).formatProperties.optimalTilingFeatures &
                                          vk::FormatFeatureFlagBits::eDepthStencilAttachment);
        std::println("  {}: {}", formatName(format), ok ? "yes" : "no");
        if (ok && !pick) {
            pick = format;
        }
    }
    if (!pick) {
        throw std::runtime_error("no depth format supports DEPTH_STENCIL_ATTACHMENT");
    }
    return *pick;
}

// Whether format also has a stencil component. Such a format's layout
// transitions must name both aspects (VUID-VkImageMemoryBarrier2-image-03320
// without separateDepthStencilLayouts), and this step then attaches it as depth
// AND stencil in the DEPTH_STENCIL_ATTACHMENT_OPTIMAL layout.
bool hasStencil(vk::Format format) {
    return format == vk::Format::eD16UnormS8Uint || format == vk::Format::eD24UnormS8Uint || format == vk::Format::eD32SfloatS8Uint;
}
// #endregion

// #region frame-resources
// What the CPU touches while recording one frame (step 17).
struct Frame {
    vk::raii::CommandBuffer commandBuffer;
    vk::raii::Semaphore imageAvailable;
    vk::raii::Fence inFlight;
    GpuBuffer uniform;
    Scene* mapped;
    vk::DescriptorSet descriptorSet;
};
// #endregion

int main(int argc, char** argv)
try {
    const Args args = parseArgs(argc, argv);

    // ---- Step 06: window (800x600 physical pixels, fixed size) ----
    Glfw glfw;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 22 depth", nullptr, nullptr), &glfwDestroyWindow);
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
        .pApplicationName   = "Vulkan from Zero - 22 depth",
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
    // Depth testing is Vulkan 1.0: nothing to enable.
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
    const vk::PhysicalDeviceProperties chosen      = physicalDevice.getProperties();
    std::println("\nPicked: {}", chosen.deviceName.data());
    std::println("Graphics + present family: {}", family);

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
    vk::raii::Queue queue(device, family, 0);
    const vk::PhysicalDeviceMemoryProperties mem = physicalDevice.getMemoryProperties();

    std::println("");
    const vk::Format depthFormat = pickDepthFormat(physicalDevice);
    const bool stencil           = hasStencil(depthFormat);
    const vk::ImageLayout depthLayout =
        stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal : vk::ImageLayout::eDepthAttachmentOptimal;
    const vk::ImageAspectFlags depthAspect =
        stencil ? vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil : vk::ImageAspectFlagBits::eDepth;
    std::println("Depth format: {}, attachment layout {}", formatName(depthFormat), layoutName(depthLayout));

    // #region depth-image
    // One depth image, the size of the swapchain images, shared by both frames
    // in flight (see the barrier in drawFrame for what that costs). Usage
    // DEPTH_STENCIL_ATTACHMENT only: nobody samples or copies it. Memory:
    // DEVICE_LOCAL, like any render target.
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
    const vk::MemoryRequirements req = depthImage.getMemoryRequirements();
    const auto depthType             = findType(mem, req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    if (!depthType) {
        throw std::runtime_error("no DEVICE_LOCAL memory type for the depth image");
    }
    vk::raii::DeviceMemory depthMemory(device, vk::MemoryAllocateInfo{.allocationSize = req.size, .memoryTypeIndex = *depthType});
    depthImage.bindMemory(*depthMemory, 0);
    // The view sees the depth aspect (and the stencil aspect, if any).
    vk::raii::ImageView depthView(device, vk::ImageViewCreateInfo{
                                              .image            = *depthImage,
                                              .viewType         = vk::ImageViewType::e2D,
                                              .format           = depthFormat,
                                              .subresourceRange = {.aspectMask = depthAspect, .levelCount = 1, .layerCount = 1},
                                          });
    std::println("Depth image: {}x{}, DEPTH_STENCIL_ATTACHMENT, {} bytes in memory type {} (DEVICE_LOCAL), one for both frames in flight",
                 extent.width, extent.height, req.size, *depthType);
    // #endregion

    // #region projection-matrix
    const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    const Mat4 proj    = perspective(fovy, aspect, nearZ, farZ);
    std::println("\nProjection: fovy 60 degrees, aspect {}, near {}, far {} (rows; Vulkan clip space: y down, depth 0..1)", aspect,
                 nearZ, farZ);
    for (size_t r = 0; r < 4; ++r) {
        std::println("  {:9.4f} {:9.4f} {:9.4f} {:9.4f}", proj[0][r], proj[1][r], proj[2][r], proj[3][r]);
    }
    // Where the near and far planes land after the divide by w.
    const auto depthAt = [&](float d) { return (proj[2][2] * -d + proj[3][2]) / d; };
    std::println("Depth after the divide: {:.4f} at distance {}, {:.4f} at 1, {:.4f} at 3.5, {:.4f} at {}", depthAt(nearZ), nearZ,
                 depthAt(1.0f), depthAt(3.5f), depthAt(farZ), farZ);
    // #endregion

    // The SPIR-V is compiled from shaders/cube.slang by `bun run sync:shaders`;
    // CMake bakes the absolute path of shaders/spv into SHADER_DIR.
    const std::vector<uint32_t> spirv = readSpirv(std::filesystem::path{SHADER_DIR} / "cube.spv");
    std::println("\nSPIR-V: {} bytes (cube.spv), entry points vertexMain, fragmentMain", spirv.size() * sizeof(uint32_t));
    std::optional<vk::raii::ShaderModule> shaderModule;
    shaderModule.emplace(device, vk::ShaderModuleCreateInfo{
                                     .codeSize = spirv.size() * sizeof(uint32_t),
                                     .pCode    = spirv.data(),
                                 });
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = **shaderModule, .pName = "vertexMain"},
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = **shaderModule, .pName = "fragmentMain"},
    };

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
    const vk::DescriptorSetLayout setLayoutHandle = *setLayout;
    vk::raii::PipelineLayout pipelineLayout(device, vk::PipelineLayoutCreateInfo{.setLayoutCount = 1, .pSetLayouts = &setLayoutHandle});

    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
    const vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
    // Counter-clockwise front faces: see the cube's comment.
    const vk::PipelineRasterizationStateCreateInfo rasterization{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode    = vk::CullModeFlagBits::eBack,
        .frontFace   = vk::FrontFace::eCounterClockwise,
        .lineWidth   = 1.0f,
    };
    const vk::PipelineMultisampleStateCreateInfo multisample{.rasterizationSamples = vk::SampleCountFlagBits::e1};

    // #region depth-state
    // The depth test: a fragment survives if its depth is LESS than the stored
    // one (cleared to 1.0 = the far plane); a surviving fragment writes its
    // depth. Bounds test and stencil stay off. With --no-depth both are off: the
    // attachment is still cleared, but whatever is drawn last covers what was
    // drawn before.
    const vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable       = args.depthTest ? vk::True : vk::False,
        .depthWriteEnable      = args.depthTest ? vk::True : vk::False,
        .depthCompareOp        = vk::CompareOp::eLess,
        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable     = vk::False,
    };
    // Dynamic rendering: the pipeline declares the attachment formats it will
    // render to (a stencil format only if the image has stencil).
    const vk::PipelineRenderingCreateInfo renderingInfo{
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &format.format,
        .depthAttachmentFormat   = depthFormat,
        .stencilAttachmentFormat = stencil ? depthFormat : vk::Format::eUndefined,
    };
    if (args.depthTest) {
        std::println("Depth state: test on, write on, compare LESS (clear value 1.0)");
    } else {
        std::println("Depth state: test OFF, write OFF (--no-depth): the last cube drawn wins");
    }
    // #endregion

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
                                    .pDepthStencilState  = &depthStencil,
                                    .pColorBlendState    = &colorBlend,
                                    .pDynamicState       = &dynamicState,
                                    .layout              = *pipelineLayout,
                                });
    shaderModule.reset();
    std::println("Graphics pipeline created (color {} + depth {})", formatName(format.format), formatName(depthFormat));

    // Step 14's mesh buffers: HOST_VISIBLE | HOST_COHERENT, written once.
    const vk::MemoryPropertyFlags hostFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    const std::span vertexData{vertices};
    const std::span indexData{indices};
    GpuBuffer vertexBuffer = createBuffer(device, mem, vertexData.size_bytes(), vk::BufferUsageFlagBits::eVertexBuffer, hostFlags);
    writeMapped(vertexBuffer.memory, vertexData);
    GpuBuffer indexBuffer = createBuffer(device, mem, indexData.size_bytes(), vk::BufferUsageFlagBits::eIndexBuffer, hostFlags);
    writeMapped(indexBuffer.memory, indexData);
    std::println("Mesh: cube, {} vertices, {} indices, drawn as 2 instances", vertices.size(), indices.size());

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
    for (const vk::DescriptorSet set : descriptorSets) {
        uniforms.push_back(createBuffer(device, mem, uniformSize, vk::BufferUsageFlagBits::eUniformBuffer, hostFlags));
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
    std::println("Uniform buffers: {} x {} bytes (2 matrices), one descriptor set each", framesInFlight, uniformSize);

    // Steps 09-10: the pool, per-frame buffers and sync objects.
    vk::raii::CommandPool commandPool(device, vk::CommandPoolCreateInfo{
                                                  .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                  .queueFamilyIndex = family,
                                              });
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
            .uniform        = std::move(uniforms[i]),
            .mapped         = mapped[i],
            .descriptorSet  = descriptorSets[i],
        });
    }
    std::vector<vk::raii::Semaphore> renderFinished;
    renderFinished.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        renderFinished.emplace_back(device, vk::SemaphoreCreateInfo{});
    }
    std::println("\n{} frames in flight, one shared depth image", framesInFlight);

    using Clock       = std::chrono::steady_clock;
    using Seconds     = std::chrono::duration<double>;
    const auto start  = Clock::now();
    size_t frameIndex = 0;

    // Renders and presents one frame (step 17's loop). Returns false when the
    // swapchain is out of date (recreation is step 11).
    const auto drawFrame = [&]() -> bool {
        Frame& frame                       = frames.at(frameIndex % frames.size());
        const vk::raii::CommandBuffer& cmd = frame.commandBuffer;
        if (device.waitForFences(*frame.inFlight, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
            throw std::runtime_error("waitForFences failed");
        }
        const auto time         = static_cast<float>(Seconds(Clock::now() - start).count());
        const Scene sceneData   = scene(time, proj);
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

        // #region depth-barrier
        // The depth image is SHARED by the frames in flight, so this frame's
        // first depth access (the clear) is a write after the previous frame's
        // depth writes (its tests and its DONT_CARE store): a WRITE-AFTER-WRITE
        // hazard. Nothing else orders them: the fence we waited on belongs to the
        // frame two submissions back, and the GPU may still be running the
        // previous one. So the barrier's first scope is the previous frame's
        // depth writes (EARLY and LATE_FRAGMENT_TESTS, where depth attachments
        // are accessed), the second scope this frame's depth reads and writes in
        // the same stages. oldLayout UNDEFINED: the old contents are not needed
        // (the clear overwrites them), and the transition itself is ordered by
        // the same scopes. Cost: frame N+1's fragment tests wait for frame N's;
        // one depth image per frame in flight would remove that wait at the
        // price of more memory.
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
        // #endregion

        const vk::RenderingAttachmentInfo colorAttachment{
            .imageView   = *imageViews.at(imageIndex),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eStore,
            .clearValue  = vk::ClearColorValue{std::array{0.02f, 0.02f, 0.03f, 1.0f}},
        };
        // #region depth-attachment
        // The depth attachment of this rendering: cleared to 1.0 (the far plane)
        // at the start, thrown away at the end (DONT_CARE: only this frame's
        // color image is presented, nobody reads the depth later, and a tiled GPU
        // then never writes it out to memory at all).
        const vk::RenderingAttachmentInfo depthAttachment{
            .imageView   = *depthView,
            .imageLayout = depthLayout,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eDontCare,
            .clearValue  = vk::ClearDepthStencilValue{.depth = 1.0f, .stencil = 0},
        };
        // A depth/stencil format is attached as both (same view, same layout).
        const vk::RenderingInfo renderingInfo{
            .renderArea           = {.offset = {0, 0}, .extent = extent},
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &colorAttachment,
            .pDepthAttachment     = &depthAttachment,
            .pStencilAttachment   = stencil ? &depthAttachment : nullptr,
        };
        // #endregion

        // #region draw
        // Both barriers in one vkCmdPipelineBarrier2, then one draw of two
        // instances: instance i reads mvp[i].
        const std::array barriers{toAttachment, depthBarrier};
        cmd.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
            .pImageMemoryBarriers    = barriers.data(),
        });
        cmd.beginRendering(renderingInfo);
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
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, frame.descriptorSet, nullptr);
        cmd.bindVertexBuffers(0, *vertexBuffer.buffer, vk::DeviceSize{0});
        cmd.bindIndexBuffer(*indexBuffer.buffer, 0, vk::IndexType::eUint16);
        cmd.drawIndexed(static_cast<uint32_t>(indices.size()), 2, 0, 0, 0);
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
