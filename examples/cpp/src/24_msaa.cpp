// Step 24 - Multisample anti-aliasing (MSAA)
//
// Step 22's cubes have jagged edges: the rasterizer tests ONE point per pixel
// (its center), so a pixel is either fully inside a triangle or fully outside.
// MSAA tests N points per pixel (the samples) and stores color and depth PER
// SAMPLE, but runs the fragment shader only once per pixel and triangle, writing
// its color to the covered samples. At the end the samples are averaged into
// one color per pixel: the RESOLVE. Edge pixels get a mix of both sides;
// interiors cost no extra shading (but still store N color and depth samples).
//
//   sample count: framebufferColorSampleCounts & framebufferDepthSampleCounts
//   multisampled color + depth images (TRANSIENT_ATTACHMENT, lazily allocated
//     memory where it exists) -> rendering: color resolveMode AVERAGE into the
//     swapchain image (resolveImageView)
//   pipeline: rasterizationSamples = N
//
// --samples N (1 | 2 | 4 | 8, default 4; 1 = no MSAA, draw straight into the
// swapchain image); --seconds S closes the window after S seconds. Lines that
// depend on timing start with [timing].
// Steps 01-22 are repeated, reduced to what this step needs.

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

// A 4x4 matrix, column-major: m[c] is column c (step 17).
using Mat4 = std::array<std::array<float, 4>, 4>;

// The host side of Scene in shaders/cube.slang: one matrix per cube (std140
// array stride 64 = the size of a Mat4, so no padding).
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

// Step 22's perspective projection into Vulkan's clip space.
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

// Step 22's two intersecting cubes at time seconds. Their silhouettes and the
// line where they cut through each other are the edges MSAA smooths.
Scene scene(float time, const Mat4& proj) {
    const Mat4 view     = translate(0.0f, 0.0f, -3.5f);
    const Mat4 model0   = mul(rotateY(time * 0.7f), rotateX(time * 0.4f));
    const Mat4 model1   = mul(translate(0.45f, 0.15f, 0.1f), mul(rotateY(-time * 0.5f), rotateX(0.6f)));
    const Mat4 viewProj = mul(proj, view);
    return Scene{.mvp = {mul(viewProj, model0), mul(viewProj, model1)}};
}

// #region cli
// Command line: --samples N (1, 2, 4 or 8; default 4), --seconds S.
struct Args {
    uint32_t samples = 4;
    std::optional<double> seconds;
};

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value           = [&]() -> std::string_view {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::format("{} needs a value", arg));
            }
            return argv[++i];
        };
        const auto number = [](std::string_view text, auto& out) {
            const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
            if (ec != std::errc{} || end != text.data() + text.size()) {
                throw std::runtime_error(std::format("not a number: {}", text));
            }
        };
        if (arg == "--samples") {
            number(value(), args.samples);
        } else if (arg == "--seconds") {
            double seconds = 0.0;
            number(value(), seconds);
            args.seconds = seconds;
        } else {
            throw std::runtime_error(std::format("unknown argument {} (use --samples N, --seconds S)", arg));
        }
    }
    if (args.samples != 1 && args.samples != 2 && args.samples != 4 && args.samples != 8) {
        throw std::runtime_error("--samples must be 1, 2, 4 or 8");
    }
    return args;
}
// #endregion

// #region sample-counts
// The sample counts a flag set allows, as numbers ("1, 2, 4, 8").
std::string sampleList(vk::SampleCountFlags flags) {
    std::string out;
    for (uint32_t bit = 0; bit < 7; ++bit) {
        const uint32_t n = 1u << bit;
        if (flags & static_cast<vk::SampleCountFlagBits>(n)) {
            out += std::format("{}{}", out.empty() ? "" : ", ", n);
        }
    }
    return out;
}

// Which sample counts can render color AND depth together: a rendering's
// attachments must all have the same count (for one pipeline), so the usable
// counts are the intersection of the two framebuffer limits. (Images have their
// own per-format limit, vkGetPhysicalDeviceImageFormatProperties2 sampleCounts,
// which createAttachment checks as well.)
vk::SampleCountFlagBits pickSamples(const vk::PhysicalDeviceLimits& limits, uint32_t wanted) {
    const vk::SampleCountFlags color = limits.framebufferColorSampleCounts;
    const vk::SampleCountFlags depth = limits.framebufferDepthSampleCounts;
    const vk::SampleCountFlags both  = color & depth;
    std::println("framebufferColorSampleCounts: {}", sampleList(color));
    std::println("framebufferDepthSampleCounts: {}", sampleList(depth));
    std::println("Usable for color + depth: {}", sampleList(both));
    const auto pick = static_cast<vk::SampleCountFlagBits>(wanted);
    if (!(both & pick)) {
        throw std::runtime_error(std::format("{} samples are not supported for color + depth here", wanted));
    }
    return pick;
}
// #endregion

// Memory property names for printing, in bit order (step 13).
std::string memoryNames(vk::MemoryPropertyFlags flags) {
    using Mem = vk::MemoryPropertyFlagBits;
    constexpr std::array<std::pair<Mem, std::string_view>, 5> table{{
        {Mem::eDeviceLocal, "DEVICE_LOCAL"},
        {Mem::eHostVisible, "HOST_VISIBLE"},
        {Mem::eHostCoherent, "HOST_COHERENT"},
        {Mem::eHostCached, "HOST_CACHED"},
        {Mem::eLazilyAllocated, "LAZILY_ALLOCATED"},
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

// An image used only as an attachment of this frame's rendering, with its
// memory and a view. Members are destroyed in reverse order: view, image,
// memory.
struct Attachment {
    vk::raii::DeviceMemory memory;
    vk::raii::Image image;
    vk::raii::ImageView view;
};

// What an attachment is: format, the aspect its view sees, attachment usage.
struct AttachmentKind {
    vk::Format format;
    vk::ImageAspectFlags aspect;
    vk::ImageUsageFlags usage;
};

// #region transient-attachment
// Creates a render target that lives only inside one rendering: its contents
// are cleared at the start and never stored (DONT_CARE), so it is
// TRANSIENT_ATTACHMENT. That usage lets it take LAZILY_ALLOCATED memory, where
// it exists: on tile-based GPUs (most mobile ones) the samples then stay in
// on-chip tile memory and the allocation may never get physical pages - a 4x
// MSAA target costs almost nothing. Desktop GPUs render to VRAM and offer no
// such memory type, so the image falls back to DEVICE_LOCAL. Also checks that
// this format, usage and tiling allow samples samples. Prints one line
// starting with label.
Attachment createAttachment(const vk::raii::PhysicalDevice& physicalDevice, const vk::raii::Device& device,
                            const vk::PhysicalDeviceMemoryProperties& mem, const AttachmentKind& kind, vk::Extent2D extent,
                            vk::SampleCountFlagBits samples, std::string_view label) {
    const vk::ImageUsageFlags usage = kind.usage | vk::ImageUsageFlagBits::eTransientAttachment;
    const vk::SampleCountFlags allowed =
        physicalDevice
            .getImageFormatProperties2(vk::PhysicalDeviceImageFormatInfo2{
                .format = kind.format,
                .type   = vk::ImageType::e2D,
                .tiling = vk::ImageTiling::eOptimal,
                .usage  = usage,
            })
            .imageFormatProperties.sampleCounts;
    if (!(allowed & samples)) {
        throw std::runtime_error(std::format("{} (optimal tiling, attachment usage) allows sample counts {}, not {}", formatName(kind.format),
                                             sampleList(allowed), static_cast<uint32_t>(samples)));
    }
    vk::raii::Image image(device, vk::ImageCreateInfo{
                                      .imageType     = vk::ImageType::e2D,
                                      .format        = kind.format,
                                      .extent        = {extent.width, extent.height, 1},
                                      .mipLevels     = 1,
                                      .arrayLayers   = 1,
                                      .samples       = samples,
                                      .tiling        = vk::ImageTiling::eOptimal,
                                      .usage         = usage,
                                      .sharingMode   = vk::SharingMode::eExclusive,
                                      .initialLayout = vk::ImageLayout::eUndefined,
                                  });
    const vk::MemoryRequirements req = image.getMemoryRequirements();
    std::optional<uint32_t> memoryType =
        findType(mem, req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eLazilyAllocated);
    std::string_view note = "lazily allocated";
    if (!memoryType) {
        memoryType = findType(mem, req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        note       = "no LAZILY_ALLOCATED type allowed, fallback";
    }
    if (!memoryType) {
        throw std::runtime_error("no DEVICE_LOCAL memory type for an attachment");
    }
    vk::raii::DeviceMemory memory(device, vk::MemoryAllocateInfo{.allocationSize = req.size, .memoryTypeIndex = *memoryType});
    image.bindMemory(*memory, 0);
    vk::raii::ImageView view(device, vk::ImageViewCreateInfo{
                                         .image            = *image,
                                         .viewType         = vk::ImageViewType::e2D,
                                         .format           = kind.format,
                                         .subresourceRange = {.aspectMask = kind.aspect, .levelCount = 1, .layerCount = 1},
                                     });
    std::println("{} attachment: {} x{}, {} bytes, memory type {} ({}; {})", label, formatName(kind.format),
                 static_cast<uint32_t>(samples), req.size, *memoryType, memoryNames(mem.memoryTypes[*memoryType].propertyFlags), note);
    return Attachment{.memory = std::move(memory), .image = std::move(image), .view = std::move(view)};
}
// #endregion

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
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 24 msaa", nullptr, nullptr), &glfwDestroyWindow);
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
        .pApplicationName   = "Vulkan from Zero - 24 msaa",
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

    // #region choose-samples
    std::println("");
    const vk::SampleCountFlagBits samples = pickSamples(chosen.limits, args.samples);
    if (args.samples == 1) {
        std::println("Sample count: 1 (--samples 1): no MSAA, rendering straight into the swapchain image");
    } else {
        std::println("Sample count: {} (--samples): MSAA color + depth, resolved into the swapchain image", args.samples);
    }
    // #endregion

    const vk::Format depthFormat = pickDepthFormat(physicalDevice);
    const bool stencil           = hasStencil(depthFormat);
    const vk::ImageLayout depthLayout =
        stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal : vk::ImageLayout::eDepthAttachmentOptimal;
    const vk::ImageAspectFlags depthAspect =
        stencil ? vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil : vk::ImageAspectFlagBits::eDepth;
    std::println("Depth format: {}, attachment layout {}", formatName(depthFormat), layoutName(depthLayout));

    // #region msaa-images
    // Every attachment of one rendering has the SAME sample count: the
    // multisampled color image (swapchain format) and step 22's depth image,
    // now with samples samples per pixel. Both are transient: cleared at the
    // start, never stored. The swapchain image stays single-sampled and only
    // receives the resolve. Like step 22's depth image, both are shared by the
    // frames in flight (see drawFrame). Memory grows with the sample count: 4
    // samples = 4 colors and 4 depths per pixel.
    const Attachment depth = createAttachment(physicalDevice, device, mem,
                                              AttachmentKind{depthFormat, depthAspect, vk::ImageUsageFlagBits::eDepthStencilAttachment},
                                              extent, samples, "Depth");
    std::optional<Attachment> msaaColor;
    if (args.samples > 1) {
        msaaColor.emplace(createAttachment(physicalDevice, device, mem,
                                           AttachmentKind{format.format, vk::ImageAspectFlagBits::eColor, vk::ImageUsageFlagBits::eColorAttachment},
                                           extent, samples, "Color"));
    }
    const auto lazyTypes = std::ranges::count_if(std::span{mem.memoryTypes.data(), mem.memoryTypeCount}, [](const vk::MemoryType& t) {
        return static_cast<bool>(t.propertyFlags & vk::MemoryPropertyFlagBits::eLazilyAllocated);
    });
    std::println("Memory types with LAZILY_ALLOCATED on this device: {} (tile-based GPUs have them; desktop GPUs render to VRAM)",
                 lazyTypes);
    // #endregion

    const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    const Mat4 proj    = perspective(fovy, aspect, nearZ, farZ);

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
    // #region multisample-state
    // The pipeline rasterizes with samples coverage samples per pixel; it must
    // match the attachments' sample count. The fragment shader still runs once
    // per pixel and triangle, and its color goes to every covered sample: edges
    // are smoothed, but aliasing INSIDE a triangle (a high-frequency texture, a
    // sharp specular highlight) is not. sampleShadingEnable + minSampleShading
    // (the sampleRateShading feature, required in Vulkan 1.4) would run the
    // shader per sample to treat that too, at up to N times the shading cost.
    const vk::PipelineMultisampleStateCreateInfo multisample{
        .rasterizationSamples = samples,
        .sampleShadingEnable  = vk::False,
    };
    // #endregion

    // Step 22's depth test.
    const vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable       = vk::True,
        .depthWriteEnable      = vk::True,
        .depthCompareOp        = vk::CompareOp::eLess,
        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable     = vk::False,
    };
    // The formats are those of the multisampled images, which have the
    // swapchain's color format; the sample count lives in the multisample state
    // above.
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
    std::println("Graphics pipeline created (rasterizationSamples {}, color {} + depth {})", args.samples, formatName(format.format), formatName(depthFormat));

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
    std::println("\n{} frames in flight, multisampled images shared by both", framesInFlight);

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
        // acquire semaphore's wait stage. Second scope READ | WRITE: with MSAA
        // the swapchain image is the resolve target, and "access to the single
        // sample attachment can be synchronized with COLOR_ATTACHMENT_READ and
        // COLOR_ATTACHMENT_WRITE" (spec, "Render Pass Multisample Resolve
        // Operations").
        const vk::ImageMemoryBarrier2 toAttachment{
            .srcStageMask        = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask       = vk::AccessFlagBits2::eNone,
            .dstStageMask        = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask       = vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout           = vk::ImageLayout::eUndefined,
            .newLayout           = vk::ImageLayout::eColorAttachmentOptimal,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image               = image,
            .subresourceRange    = colorRange,
        };

        // #region msaa-barriers
        // The multisampled images are shared by the frames in flight, like step
        // 22's depth image, so each gets step 22's write-after-write barrier
        // against the previous frame. Depth: as in step 22. Color: the previous
        // frame's last accesses were its DONT_CARE store (a COLOR_ATTACHMENT_WRITE)
        // and the resolve's read of the samples (a COLOR_ATTACHMENT_READ), both in
        // COLOR_ATTACHMENT_OUTPUT. The write must be made available (src access
        // WRITE); the read only needs the execution dependency (src stage).
        // Second scope: this frame's clear and draws (WRITE) and its resolve read
        // (READ), same stage.
        constexpr vk::PipelineStageFlags2 fragmentTests =
            vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
        std::vector<vk::ImageMemoryBarrier2> barriers{
            toAttachment,
            vk::ImageMemoryBarrier2{
                .srcStageMask        = fragmentTests,
                .srcAccessMask       = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                .dstStageMask        = fragmentTests,
                .dstAccessMask       = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                .oldLayout           = vk::ImageLayout::eUndefined,
                .newLayout           = depthLayout,
                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                .image               = *depth.image,
                .subresourceRange    = {.aspectMask = depthAspect, .levelCount = 1, .layerCount = 1},
            },
        };
        if (msaaColor) {
            barriers.push_back(vk::ImageMemoryBarrier2{
                .srcStageMask        = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .srcAccessMask       = vk::AccessFlagBits2::eColorAttachmentWrite,
                .dstStageMask        = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .dstAccessMask       = vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
                .oldLayout           = vk::ImageLayout::eUndefined,
                .newLayout           = vk::ImageLayout::eColorAttachmentOptimal,
                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                .image               = *msaaColor->image,
                .subresourceRange    = colorRange,
            });
        }
        // #endregion

        // #region resolve-attachment
        // With MSAA the color attachment is the multisampled image: cleared,
        // drawn, then RESOLVED - each pixel's samples averaged (resolveMode
        // AVERAGE: for a non-integer color format the only fixed-function mode;
        // NONE and CUSTOM_BIT_EXT are the other allowed values,
        // VUID-VkRenderingAttachmentInfo-imageView-06129) - into
        // resolveImageView, the swapchain image, at the end of the rendering. The
        // samples themselves are not needed after that: DONT_CARE. The resolve
        // runs in COLOR_ATTACHMENT_OUTPUT and accesses the swapchain image as
        // COLOR_ATTACHMENT_READ | WRITE, which the barrier into the attachment
        // layout (dst READ | WRITE above) and step 09's barrier to PRESENT_SRC
        // (src WRITE) cover. Without MSAA (--samples 1) the swapchain image is the
        // color attachment, as in step 22.
        const vk::ClearValue clear = vk::ClearColorValue{std::array{0.02f, 0.02f, 0.03f, 1.0f}};
        const vk::RenderingAttachmentInfo colorAttachment =
            msaaColor ? vk::RenderingAttachmentInfo{
                            .imageView          = *msaaColor->view,
                            .imageLayout        = vk::ImageLayout::eColorAttachmentOptimal,
                            .resolveMode        = vk::ResolveModeFlagBits::eAverage,
                            .resolveImageView   = *imageViews.at(imageIndex),
                            .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                            .loadOp             = vk::AttachmentLoadOp::eClear,
                            .storeOp            = vk::AttachmentStoreOp::eDontCare,
                            .clearValue         = clear,
                        }
                      : vk::RenderingAttachmentInfo{
                            .imageView   = *imageViews.at(imageIndex),
                            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                            .loadOp      = vk::AttachmentLoadOp::eClear,
                            .storeOp     = vk::AttachmentStoreOp::eStore,
                            .clearValue  = clear,
                        };
        // Depth: multisampled too, and never resolved (nobody reads it).
        const vk::RenderingAttachmentInfo depthAttachment{
            .imageView   = *depth.view,
            .imageLayout = depthLayout,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eDontCare,
            .clearValue  = vk::ClearDepthStencilValue{.depth = 1.0f, .stencil = 0},
        };
        // #endregion
        // A depth/stencil format is attached as both (same view, same layout).
        const vk::RenderingInfo renderingInfo{
            .renderArea           = {.offset = {0, 0}, .extent = extent},
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &colorAttachment,
            .pDepthAttachment     = &depthAttachment,
            .pStencilAttachment   = stencil ? &depthAttachment : nullptr,
        };
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
