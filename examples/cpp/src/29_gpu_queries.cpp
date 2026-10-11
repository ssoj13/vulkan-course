// Step 29 - GPU queries: timestamps, pipeline statistics, calibration
//
// Step 22's two cubes, measured from the inside. A QUERY is a slot in a
// VkQueryPool that the GPU fills while it executes a command buffer; the host
// reads it back once the work has completed:
//
//   TIMESTAMP queries     vkCmdWriteTimestamp2: a device tick count at a point of
//                         the command stream -> GPU time per pass.
//   PIPELINE STATISTICS   vkCmdBeginQuery/EndQuery: counters of what the
//                         pipeline did (vertices, primitives, invocations).
//   CALIBRATED TIMESTAMPS vkGetCalibratedTimestampsKHR: the device clock and a
//                         host clock read "at the same time". The device value
//                         read right before the submit -> how long a submitted
//                         frame waits before the GPU starts it; the host value
//                         only measures the host clock's rate (the QPC line).
//
// Queries are reset on the HOST (vkResetQueryPool, core in Vulkan 1.2 behind the
// hostQueryReset feature) right after their results are read, instead of with
// vkCmdResetQueryPool in the command buffer.
//
// --frames N (1-3, default 2) frames in flight, --no-stats leaves the statistics
// query out, --seconds S closes the window after S seconds. Lines that depend on
// timing, or on the animation at the moment they are printed, start with
// [timing]. Steps 01-22 are repeated, reduced to what this step needs.

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
std::string domainName(vk::TimeDomainKHR d) {
    return shortName(string_VkTimeDomainKHR(static_cast<VkTimeDomainKHR>(d)), "VK_TIME_DOMAIN_", "_KHR");
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

// One draw of both cubes (instance i reads mvp[i], step 22).
constexpr uint32_t instanceCount = 2;

// #region cli
// Command line: --frames N, --no-stats, --seconds S.
struct Args {
    size_t frames = 2;
    bool stats    = true;
    std::optional<double> seconds;
};

Args parseArgs(int argc, char** argv) {
    Args args;
    const auto number = [&](int& i, std::string_view flag, auto& out) {
        if (i + 1 >= argc) {
            throw std::runtime_error(std::format("{} needs a value", flag));
        }
        const std::string_view text = argv[++i];
        const auto [end, ec]        = std::from_chars(text.data(), text.data() + text.size(), out);
        if (ec != std::errc{} || end != text.data() + text.size()) {
            throw std::runtime_error(std::format("{}: not a number: {}", flag, text));
        }
    };
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--frames") {
            number(i, arg, args.frames);
            if (args.frames < 1 || args.frames > 3) {
                throw std::runtime_error("--frames must be 1, 2 or 3");
            }
        } else if (arg == "--no-stats") {
            args.stats = false;
        } else if (arg == "--seconds") {
            double seconds = 0.0;
            number(i, arg, seconds);
            args.seconds = seconds;
        } else {
            throw std::runtime_error(std::format("unknown argument {} (use --frames N, --no-stats, --seconds S)", arg));
        }
    }
    return args;
}
// #endregion

// #region queries
// Timestamps per frame: start, attachments ready, main pass done, end.
constexpr uint32_t timestampCount = 4;
// What each pair of consecutive timestamps encloses.
constexpr std::array<std::string_view, 3> passes{"attachments", "main pass", "to present"};

// The six counters this step asks for. Results come back in the order of the
// flag bits (spec: "in the order of the bits"), one uint64_t each.
constexpr vk::QueryPipelineStatisticFlags statistics =
    vk::QueryPipelineStatisticFlagBits::eInputAssemblyVertices | vk::QueryPipelineStatisticFlagBits::eInputAssemblyPrimitives |
    vk::QueryPipelineStatisticFlagBits::eVertexShaderInvocations | vk::QueryPipelineStatisticFlagBits::eClippingInvocations |
    vk::QueryPipelineStatisticFlagBits::eClippingPrimitives | vk::QueryPipelineStatisticFlagBits::eFragmentShaderInvocations;
// One statistics query's result: the six counters in bit order.
using Statistics = std::array<uint64_t, 6>;

// Ticks between two timestamps when only the low bits are valid: the difference
// modulo 2^bits, read as a signed number (the counter wraps at 2^bits; right
// while the two lie less than 2^(bits-1) ticks apart).
int64_t ticksBetween(uint64_t from, uint64_t to, uint32_t bits) {
    const uint32_t shift = 64 - bits;
    return static_cast<int64_t>((to - from) << shift) >> shift;
}
// #endregion

// #region calibration
// A device timestamp and a host timestamp captured together by
// vkGetCalibratedTimestampsKHR, with the call's maximum deviation in ns.
struct Calibration {
    uint64_t device;
    uint64_t host;
    uint64_t maxDeviationNs;
};

// The host time domain to pair with VK_TIME_DOMAIN_DEVICE_KHR: the Windows
// performance counter (QueryPerformanceCounter units) or, on Linux,
// CLOCK_MONOTONIC_RAW / CLOCK_MONOTONIC (nanoseconds).
std::optional<vk::TimeDomainKHR> hostDomain(const std::vector<vk::TimeDomainKHR>& domains) {
    for (const vk::TimeDomainKHR d : {vk::TimeDomainKHR::eQueryPerformanceCounter, vk::TimeDomainKHR::eClockMonotonicRaw,
                                      vk::TimeDomainKHR::eClockMonotonic}) {
        if (std::ranges::contains(domains, d)) {
            return d;
        }
    }
    return std::nullopt;
}
// #endregion

// Sums over one second (and over the whole run) of what the queries showed.
struct Sums {
    uint64_t frames = 0;
    // GPU microseconds per pass, then the whole frame.
    std::array<double, 4> passUs{};
    // Submit -> GPU start, microseconds: sum, min, max, and how many frames.
    double latencySum = 0.0, latencyMin = 0.0, latencyMax = 0.0;
    uint64_t latencyCount = 0;
    double maxDeviationUs         = 0.0;
    uint64_t fragmentInvocations  = 0;

    void print(std::string_view label) const {
        if (frames == 0) {
            return;
        }
        const auto n = static_cast<double>(frames);
        std::string line =
            std::format("[timing] {}: {} frames, GPU us per frame: {} {:.1f}, {} {:.1f}, {} {:.1f}, whole {:.1f}", label, frames,
                        passes[0], passUs[0] / n, passes[1], passUs[1] / n, passes[2], passUs[2] / n, passUs[3] / n);
        if (latencyCount > 0) {
            const auto c = static_cast<double>(latencyCount);
            line += std::format("; submit -> GPU start {:.1f} us (min {:.1f}, max {:.1f}), maxDeviation {:.1f} us", latencySum / c,
                                latencyMin, latencyMax, maxDeviationUs / c);
        }
        if (fragmentInvocations > 0) {
            line += std::format("; FS invocations {:.0f}", static_cast<double>(fragmentInvocations) / n);
        }
        std::println("{}", line);
    }
};

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

// One frame slot (step 17) plus what its queries need: its queries are
// timestamps [slot * 4, slot * 4 + 4) and statistics query slot.
struct Frame {
    vk::raii::CommandBuffer commandBuffer;
    vk::raii::Semaphore imageAvailable;
    vk::raii::Fence inFlight;
    GpuBuffer uniform;
    Scene* mapped;
    vk::DescriptorSet descriptorSet;
    // The frame this slot rendered last and the calibration taken at its
    // submit; empty until the slot's first submit.
    std::optional<std::pair<size_t, std::optional<Calibration>>> pending;
};

int main(int argc, char** argv)
try {
    const Args args = parseArgs(argc, argv);

    // ---- Step 06: window (800x600 physical pixels, fixed size) ----
    Glfw glfw;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 29 gpu queries", nullptr, nullptr), &glfwDestroyWindow);
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
        .pApplicationName   = "Vulkan from Zero - 29 gpu queries",
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
    // #region pick-device
    // Steps 03 + 06, plus what this step needs: the hostQueryReset feature
    // (Vulkan 1.2) and pipelineStatisticsQuery (Vulkan 1.0, optional); a
    // graphics family whose timestamps are valid (timestampValidBits > 0).
    // VK_KHR_calibrated_timestamps is used when the device has it.
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
        const auto& features    = featureChain.get<vk::PhysicalDeviceFeatures2>().features;
        const auto& features12  = featureChain.get<vk::PhysicalDeviceVulkan12Features>();
        const auto& features13  = featureChain.get<vk::PhysicalDeviceVulkan13Features>();
        const bool statsOk      = !args.stats || features.pipelineStatisticsQuery == vk::True;
        if (features13.dynamicRendering != vk::True || features13.synchronization2 != vk::True ||
            features12.hostQueryReset != vk::True || !statsOk) {
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
            if (graphics && families[i].timestampValidBits > 0 && device.getSurfaceSupportKHR(i, *surface) == vk::True) {
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
        throw std::runtime_error("No Vulkan 1.4 GPU with dynamic rendering, synchronization2, hostQueryReset, pipelineStatisticsQuery, "
                                 "timestamps and VK_KHR_swapchain can present to this window");
    }
    const vk::raii::PhysicalDevice& physicalDevice = *best;
    const vk::PhysicalDeviceProperties chosen      = physicalDevice.getProperties();
    std::println("Picked: {}, graphics + present family {}", chosen.deviceName.data(), family);
    const uint32_t validBits = std::min(physicalDevice.getQueueFamilyProperties().at(family).timestampValidBits, 64u);
    const double periodNs    = chosen.limits.timestampPeriod;
    std::println("Timestamps: timestampPeriod {} ns per tick, {} valid bits on family {}", chosen.limits.timestampPeriod, validBits,
                 family);
    const std::vector<vk::ExtensionProperties> deviceExts = physicalDevice.enumerateDeviceExtensionProperties();
    const bool hasCalibration = std::ranges::any_of(deviceExts, [](const vk::ExtensionProperties& e) {
        return std::string_view{e.extensionName.data()} == vk::KHRCalibratedTimestampsExtensionName;
    });
    // #endregion

    // #region device
    // One queue; hostQueryReset (Vulkan12Features), the 1.3 features,
    // pipelineStatisticsQuery (VkPhysicalDeviceFeatures) unless --no-stats,
    // VK_KHR_swapchain and, if present, VK_KHR_calibrated_timestamps.
    constexpr float queuePriority = 1.0f;
    const vk::DeviceQueueCreateInfo queueInfo{
        .queueFamilyIndex = family,
        .queueCount       = 1,
        .pQueuePriorities = &queuePriority,
    };
    const vk::PhysicalDeviceFeatures features{.pipelineStatisticsQuery = args.stats ? vk::True : vk::False};
    vk::PhysicalDeviceVulkan13Features features13{
        .synchronization2 = vk::True,
        .dynamicRendering = vk::True,
    };
    vk::PhysicalDeviceVulkan12Features features12{.pNext = &features13, .hostQueryReset = vk::True};
    std::vector<const char*> deviceExtensions{vk::KHRSwapchainExtensionName};
    if (hasCalibration) {
        deviceExtensions.push_back(vk::KHRCalibratedTimestampsExtensionName);
    }
    vk::raii::Device device(physicalDevice, vk::DeviceCreateInfo{
                                                .pNext                   = &features12,
                                                .queueCreateInfoCount    = 1,
                                                .pQueueCreateInfos       = &queueInfo,
                                                .enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size()),
                                                .ppEnabledExtensionNames = deviceExtensions.data(),
                                                .pEnabledFeatures        = &features,
                                            });
    vk::raii::Queue queue(device, family, 0);
    // #endregion

    // #region time-domains
    // Which clocks this device can sample together with its own. DEVICE is the
    // clock of vkCmdWriteTimestamp2 ("comparable with device timestamp values");
    // QUERY_PERFORMANCE_COUNTER is Windows' QueryPerformanceCounter. Reading both
    // in one call is what makes GPU ticks and CPU time comparable.
    std::optional<vk::TimeDomainKHR> calibrationHost;
    if (hasCalibration) {
        const std::vector<vk::TimeDomainKHR> domains = physicalDevice.getCalibrateableTimeDomainsKHR();
        std::string names;
        for (const vk::TimeDomainKHR d : domains) {
            names += names.empty() ? "" : ", ";
            names += domainName(d);
        }
        std::println("Calibrateable time domains: {}", names);
        const auto host = hostDomain(domains);
        if (std::ranges::contains(domains, vk::TimeDomainKHR::eDevice) && host) {
            calibrationHost = host;
            std::println("Calibration pairs DEVICE with {}", domainName(*host));
        } else {
            std::println("No DEVICE + host domain pair: no submit -> GPU start latency");
        }
    } else {
        std::println("VK_KHR_calibrated_timestamps: not supported, no submit -> GPU start latency");
    }
    // #endregion

    // Step 07's swapchain.
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
    std::println("Swapchain: {}x{} {}, {}, {} images; {} frame(s) in flight", extent.width, extent.height, formatName(format.format),
                 shortName(string_VkPresentModeKHR(static_cast<VkPresentModeKHR>(presentMode)), "VK_PRESENT_MODE_", "_KHR"),
                 images.size(), args.frames);
    const vk::PhysicalDeviceMemoryProperties mem = physicalDevice.getMemoryProperties();

    // Step 22's depth image, one for all frames in flight.
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
    const vk::MemoryRequirements req = depthImage.getMemoryRequirements();
    const auto depthType             = findType(mem, req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    if (!depthType) {
        throw std::runtime_error("no DEVICE_LOCAL memory type for the depth image");
    }
    vk::raii::DeviceMemory depthMemory(device, vk::MemoryAllocateInfo{.allocationSize = req.size, .memoryTypeIndex = *depthType});
    depthImage.bindMemory(*depthMemory, 0);
    vk::raii::ImageView depthView(device, vk::ImageViewCreateInfo{
                                              .image            = *depthImage,
                                              .viewType         = vk::ImageViewType::e2D,
                                              .format           = depthFormat,
                                              .subresourceRange = {.aspectMask = depthAspect, .levelCount = 1, .layerCount = 1},
                                          });

    const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    const Mat4 proj    = perspective(fovy, aspect, nearZ, farZ);

    // Step 22's pipeline (cube.spv, depth test LESS).
    const std::vector<uint32_t> spirv = readSpirv(std::filesystem::path{SHADER_DIR} / "cube.spv");
    std::optional<vk::raii::ShaderModule> shaderModule;
    shaderModule.emplace(device, vk::ShaderModuleCreateInfo{
                                     .codeSize = spirv.size() * sizeof(uint32_t),
                                     .pCode    = spirv.data(),
                                 });
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = **shaderModule, .pName = "vertexMain"},
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = **shaderModule, .pName = "fragmentMain"},
    };
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

    // Step 14's mesh buffers.
    const vk::MemoryPropertyFlags hostFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    const std::span vertexData{vertices};
    const std::span indexData{indices};
    GpuBuffer vertexBuffer = createBuffer(device, mem, vertexData.size_bytes(), vk::BufferUsageFlagBits::eVertexBuffer, hostFlags);
    writeMapped(vertexBuffer.memory, vertexData);
    GpuBuffer indexBuffer = createBuffer(device, mem, indexData.size_bytes(), vk::BufferUsageFlagBits::eIndexBuffer, hostFlags);
    writeMapped(indexBuffer.memory, indexData);

    // #region query-pools
    // Two pools, one per query type (a pool holds one type): 4 timestamps per
    // frame slot, and one statistics query per slot with the six counters. A
    // query must be reset before its first use; with hostQueryReset the host does
    // it, here for all queries at once and later per slot after its results are
    // read.
    const auto frameCount = static_cast<uint32_t>(args.frames);
    vk::raii::QueryPool timestampPool(device, vk::QueryPoolCreateInfo{
                                                  .queryType  = vk::QueryType::eTimestamp,
                                                  .queryCount = timestampCount * frameCount,
                                              });
    timestampPool.reset(0, timestampCount * frameCount);
    std::optional<vk::raii::QueryPool> statsPool;
    if (args.stats) {
        statsPool.emplace(device, vk::QueryPoolCreateInfo{
                                      .queryType          = vk::QueryType::ePipelineStatistics,
                                      .queryCount         = frameCount,
                                      .pipelineStatistics = statistics,
                                  });
        statsPool->reset(0, frameCount);
    }
    std::println("Query pools: {} timestamps ({} per frame slot), {}; reset on the host", timestampCount * frameCount, timestampCount,
                 args.stats ? std::format("{} pipeline statistics queries (6 counters)", args.frames)
                            : std::string{"no statistics (--no-stats)"});
    // #endregion

    // Step 17's per-frame uniform buffers and descriptor sets.
    constexpr vk::DeviceSize uniformSize = sizeof(Scene);
    const vk::DescriptorPoolSize poolSize{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = frameCount};
    vk::raii::DescriptorPool descriptorPool(device, vk::DescriptorPoolCreateInfo{
                                                        .maxSets       = frameCount,
                                                        .poolSizeCount = 1,
                                                        .pPoolSizes    = &poolSize,
                                                    });
    const std::vector<vk::DescriptorSetLayout> setLayouts(args.frames, *setLayout);
    std::vector<vk::DescriptorSet> descriptorSets;
    for (vk::raii::DescriptorSet& set : vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
                                                                             .descriptorPool     = *descriptorPool,
                                                                             .descriptorSetCount = frameCount,
                                                                             .pSetLayouts        = setLayouts.data(),
                                                                         })) {
        descriptorSets.push_back(set.release());
    }
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
    frames.reserve(args.frames);
    for (size_t slot = 0; slot < args.frames; ++slot) {
        GpuBuffer uniform = createBuffer(device, mem, uniformSize, vk::BufferUsageFlagBits::eUniformBuffer, hostFlags);
        auto* mapped      = static_cast<Scene*>(uniform.memory.mapMemory(0, vk::WholeSize));
        const vk::DescriptorBufferInfo info{.buffer = *uniform.buffer, .offset = 0, .range = uniformSize};
        device.updateDescriptorSets(vk::WriteDescriptorSet{
                                        .dstSet          = descriptorSets[slot],
                                        .dstBinding      = 0,
                                        .descriptorCount = 1,
                                        .descriptorType  = vk::DescriptorType::eUniformBuffer,
                                        .pBufferInfo     = &info,
                                    },
                                    nullptr);
        frames.push_back(Frame{
            .commandBuffer  = std::move(commandBuffers[slot]),
            .imageAvailable = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{}),
            .inFlight       = vk::raii::Fence(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled}),
            .uniform        = std::move(uniform),
            .mapped         = mapped,
            .descriptorSet  = descriptorSets[slot],
            .pending        = std::nullopt,
        });
    }
    std::vector<vk::raii::Semaphore> renderFinished;
    renderFinished.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        renderFinished.emplace_back(device, vk::SemaphoreCreateInfo{});
    }

    using Clock       = std::chrono::steady_clock;
    using Seconds     = std::chrono::duration<double>;
    const auto start  = Clock::now();
    size_t frameIndex = 0;
    // The animation clock: starts at frame 0, so frame 0 always shows the scene
    // at time 0 and its statistics are the same in every run.
    std::optional<Clock::time_point> animStart;
    uint64_t currentSecond = 0;
    Sums current;
    Sums total;
    // The first and the latest calibration (for the host clock's rate).
    std::optional<Calibration> firstCalibration;
    std::optional<Calibration> lastCalibration;

    // #region read-results
    // Reads the queries of slot, which rendered frame `frame` and whose fence has
    // signaled, adds them to the sums, then resets them on the host for the
    // slot's next frame. Frame 0's statistics are printed with what can be
    // predicted for them.
    const auto readResults = [&](size_t slot, size_t frame, std::optional<Calibration> calibration) {
        const auto first = static_cast<uint32_t>(slot) * timestampCount;
        // The submission's fence has signaled, so WAIT returns at once.
        const auto [result, ticks] = timestampPool.getResults<uint64_t>(first, timestampCount, timestampCount * sizeof(uint64_t),
                                                                        sizeof(uint64_t),
                                                                        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(std::string{"vkGetQueryPoolResults: "} + string_VkResult(static_cast<VkResult>(result)));
        }
        timestampPool.reset(first, timestampCount);
        const auto us = [&](uint64_t from, uint64_t to) {
            return static_cast<double>(ticksBetween(from, to, validBits)) * periodNs / 1000.0;
        };
        const std::array passUs{us(ticks[0], ticks[1]), us(ticks[1], ticks[2]), us(ticks[2], ticks[3]), us(ticks[0], ticks[3])};

        // #region latency
        // The calibration was taken right before vkQueueSubmit2; ticks[0] is the
        // first timestamp of the frame's command buffer. Both are device ticks,
        // so their difference is how long the frame waited between the submit
        // call and the GPU starting it (queue backlog, scheduling, the driver's
        // own submission work).
        std::optional<std::pair<double, double>> latency;
        if (calibration) {
            latency = std::pair{us(calibration->device, ticks[0]), static_cast<double>(calibration->maxDeviationNs) / 1000.0};
        }
        // #endregion

        uint64_t fragmentInvocations = 0;
        if (statsPool) {
            const auto [statsResult, stats] = statsPool->getResult<Statistics>(
                static_cast<uint32_t>(slot), 1, sizeof(Statistics), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
            if (statsResult != vk::Result::eSuccess) {
                throw std::runtime_error(std::string{"vkGetQueryPoolResults: "} +
                                         string_VkResult(static_cast<VkResult>(statsResult)));
            }
            statsPool->reset(static_cast<uint32_t>(slot), 1);
            const auto [iaVertices, iaPrimitives, vs, clipIn, clipOut, fs] = stats;
            fragmentInvocations                                           = fs;
            if (frame == 0) {
                // #region statistics
                const uint64_t indexTotal = indices.size() * uint64_t{instanceCount};
                std::println("Pipeline statistics of frame 0 (scene at time 0), measured / predicted:");
                std::println("  input assembly vertices    {:>6} / {} = 36 indices x {} instances", iaVertices, indexTotal,
                             instanceCount);
                std::println("  input assembly primitives  {:>6} / {} = triangles of the index list", iaPrimitives, indexTotal / 3);
                std::println("  vertex shader invocations  {:>6} / {}..{}: reuse of repeated indices is implementation-dependent", vs,
                             vertices.size() * uint64_t{instanceCount}, indexTotal);
                std::println("  clipping invocations       {:>6} / {}: every triangle reaches clipping (culling comes later)", clipIn,
                             indexTotal / 3);
                std::println("  clipping primitives        {:>6} / at least 1 per triangle with a vertex inside the clip volume that covers a sample, 0 or more otherwise",
                             clipOut);
                std::println("  fragment shader invocations{:>7} / no formula: covered pixels (helper invocations may count), minus what early tests reject", fs);
                // #endregion
            }
        }

        for (Sums* s : {&current, &total}) {
            ++s->frames;
            for (size_t i = 0; i < passUs.size(); ++i) {
                s->passUs[i] += passUs[i];
            }
            if (latency) {
                const auto [l, dev] = *latency;
                s->latencyMin       = s->latencyCount == 0 ? l : std::min(s->latencyMin, l);
                s->latencyMax       = s->latencyCount == 0 ? l : std::max(s->latencyMax, l);
                s->latencySum += l;
                ++s->latencyCount;
                s->maxDeviationUs += dev;
            }
            s->fragmentInvocations += fragmentInvocations;
        }
    };
    // #endregion

    // Renders and presents one frame (step 22's loop) with timestamps and a
    // statistics query. Returns false when the swapchain is out of date.
    const auto drawFrame = [&]() -> bool {
        const size_t slot = frameIndex % frames.size();
        Frame& frame      = frames.at(slot);
        if (device.waitForFences(*frame.inFlight, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
            throw std::runtime_error("waitForFences failed");
        }
        // The slot's previous frame has completed: its queries are ready.
        if (frame.pending) {
            const auto [done, calibration] = *frame.pending;
            frame.pending.reset();
            readResults(slot, done, calibration);
        }
        const auto second = static_cast<uint64_t>(Seconds(Clock::now() - start).count());
        if (second > currentSecond) {
            current.print(std::format("{}s", currentSecond + 1));
            current       = Sums{};
            currentSecond = second;
        }

        const vk::raii::CommandBuffer& cmd = frame.commandBuffer;
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
        const auto first = static_cast<uint32_t>(slot) * timestampCount;

        // #region record
        // Timestamp i is written once all earlier commands have completed the
        // given stage ("Implementations may write the timestamp at any stage that
        // is logically later than stage"). ALL_COMMANDS therefore marks the end of
        // everything before it, so consecutive timestamps enclose the commands
        // between them; TOP_OF_PIPE marks the start. The statistics query counts
        // what the draw between Begin and End does.
        using Stage = vk::PipelineStageFlagBits2;
        cmd.writeTimestamp2(Stage::eTopOfPipe, *timestampPool, first);
        const std::array barriers{toAttachment, depthBarrier};
        cmd.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
            .pImageMemoryBarriers    = barriers.data(),
        });
        cmd.writeTimestamp2(Stage::eAllCommands, *timestampPool, first + 1);
        if (statsPool) {
            cmd.beginQuery(**statsPool, static_cast<uint32_t>(slot), {});
        }
        cmd.beginRendering(renderingAttachments);
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
        cmd.drawIndexed(static_cast<uint32_t>(indices.size()), instanceCount, 0, 0, 0);
        cmd.endRendering();
        if (statsPool) {
            cmd.endQuery(**statsPool, static_cast<uint32_t>(slot));
        }
        cmd.writeTimestamp2(Stage::eAllCommands, *timestampPool, first + 2);
        cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toPresent});
        cmd.writeTimestamp2(Stage::eAllCommands, *timestampPool, first + 3);
        cmd.end();
        // #endregion

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

        // #region calibrate
        // Device clock and host clock, sampled together right before the submit.
        // maxDeviation bounds how far apart the two samples may lie.
        std::optional<Calibration> calibration;
        if (calibrationHost) {
            const std::array infos{
                vk::CalibratedTimestampInfoKHR{.timeDomain = vk::TimeDomainKHR::eDevice},
                vk::CalibratedTimestampInfoKHR{.timeDomain = *calibrationHost},
            };
            const auto [values, maxDeviationNs] = device.getCalibratedTimestampsKHR(infos);
            if (values.size() != 2) {
                throw std::runtime_error("vkGetCalibratedTimestampsKHR returned a wrong count");
            }
            calibration = Calibration{.device = values[0], .host = values[1], .maxDeviationNs = maxDeviationNs};
        }
        // #endregion
        queue.submit2(vk::SubmitInfo2{
                          .waitSemaphoreInfoCount   = 1,
                          .pWaitSemaphoreInfos      = &waitInfo,
                          .commandBufferInfoCount   = 1,
                          .pCommandBufferInfos      = &cmdInfo,
                          .signalSemaphoreInfoCount = 1,
                          .pSignalSemaphoreInfos    = &signalInfo,
                      },
                      *frame.inFlight);
        if (calibration) {
            if (!firstCalibration) {
                firstCalibration = calibration;
            }
            lastCalibration = calibration;
        }
        frame.pending = std::pair{frameIndex, calibration};

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

    // Wait for the device, read the queries still pending, print the totals.
    device.waitIdle();
    for (size_t slot = 0; slot < frames.size(); ++slot) {
        if (frames[slot].pending) {
            const auto [done, calibration] = *frames[slot].pending;
            frames[slot].pending.reset();
            readResults(slot, done, calibration);
        }
    }
    total.print("whole run");
    // #region host-clock
    // The host clock's rate, measured: QPC ticks per device second between the
    // first and the last calibration (QueryPerformanceFrequency would report it
    // directly; this shows the two clocks agree).
    if (firstCalibration && lastCalibration && calibrationHost) {
        const double deviceS =
            static_cast<double>(ticksBetween(firstCalibration->device, lastCalibration->device, validBits)) * periodNs / 1e9;
        if (deviceS > 0.0) {
            std::println("[timing] {}: {:.0f} ticks per second of device time (over {:.2f} s)", domainName(*calibrationHost),
                         static_cast<double>(lastCalibration->host - firstCalibration->host) / deviceS, deviceS);
        }
    }
    // #endregion
    return EXIT_SUCCESS;
} catch (const vk::SystemError& e) {
    std::println(stderr, "Vulkan error: {}", e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return EXIT_FAILURE;
}
