// Step 23 - Mipmaps
//
// A checkerboard floor recedes to the horizon. Far away, one pixel covers dozens
// of texels; a sampler that reads only the base image picks a few of them
// almost at random, and the floor dissolves into shimmering moire patterns
// (aliasing). MIPMAPS fix that: a chain of prefiltered copies, each half the
// size of the previous one, down to 1x1; the sampler reads the level whose
// texels are about the size of a pixel.
//
//   upload mip 0 (step 21) -> for each level i = 1..N-1:
//     barrier level i-1: TRANSFER_DST -> TRANSFER_SRC
//     vkCmdBlitImage2 level i-1 -> level i (half size, LINEAR filter)
//     barrier level i-1: TRANSFER_SRC -> SHADER_READ_ONLY
//   barrier level N-1: TRANSFER_DST -> SHADER_READ_ONLY
//   sampler: mipmapMode LINEAR, minLod 0, maxLod N (trilinear filtering)
//
// --no-mips clamps the sampler to level 0 (maxLod 0) to show the aliasing;
// --anisotropy adds anisotropic filtering (16x, clamped to the limit);
// --seconds S closes the window after S seconds. Lines that depend on timing
// start with [timing].
// Steps 01-21 are repeated, reduced to what this step needs.

// Step 10: VK_ERROR_OUT_OF_DATE_KHR comes back as a vk::Result, not an exception.
#define VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS

#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <bit>
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

// #region texture-data
// Width and height of the texture in texels.
constexpr uint32_t texSize = 512;
// Side of one checker cell in texels (32 x 32 cells): small cells are what
// makes the far floor alias.
constexpr uint32_t cell = 16;

// A high-contrast checkerboard, RGBA8, row by row (generated, not loaded: see
// step 21). Light cells get a faint gradient so the levels are not all the same
// gray.
std::vector<uint8_t> texturePixels() {
    std::vector<uint8_t> pixels;
    pixels.reserve(size_t{texSize} * texSize * 4);
    for (uint32_t y = 0; y < texSize; ++y) {
        for (uint32_t x = 0; x < texSize; ++x) {
            std::array<uint32_t, 3> rgb{25, 25, 30};
            if ((x / cell + y / cell) % 2 == 0) {
                rgb = {200 + x / 16, 200 + y / 16, 230};
            }
            // Every value is <= 255 by construction (x, y < 512).
            for (const uint32_t c : rgb) {
                pixels.push_back(static_cast<uint8_t>(c));
            }
            pixels.push_back(255);
        }
    }
    return pixels;
}

// FNV-1a, 32 bit: a fingerprint of the texture bytes, to compare runs.
uint32_t fnv1a(std::span<const uint8_t> bytes) {
    uint32_t h = 0x811c9dc5u;
    for (const uint8_t b : bytes) {
        h = (h ^ b) * 0x01000193u;
    }
    return h;
}
// #endregion

// #region mip-count
// The number of levels down to 1x1: floor(log2(max(w, h))) + 1. For an
// unsigned integer, floor(log2(n)) + 1 is the number of significant bits,
// std::bit_width(n).
uint32_t mipCount(uint32_t width, uint32_t height) { return static_cast<uint32_t>(std::bit_width(std::max(width, height))); }
// #endregion

// Step 21's vertex: position + UV, 20 bytes (shaders/textured.slang).
struct Vertex {
    std::array<float, 3> position;
    std::array<float, 2> uv;
};
static_assert(sizeof(Vertex) == 20);

// #region floor
// The floor: y = 0, x from -20 to 20, z from 5 (behind the camera) to -200.
// UV = offset world position / 2 (u = (x + 20) / 2, v = (5 - z) / 2), so the
// texture repeats every 2 units (REPEAT address mode) - 20 x 102.5 copies. Corners counter-clockwise seen from above, the
// front face (step 22).
constexpr std::array<Vertex, 4> vertices{{
    {{-20.0f, 0.0f, -200.0f}, {0.0f, 102.5f}},
    {{-20.0f, 0.0f, 5.0f}, {0.0f, 0.0f}},
    {{20.0f, 0.0f, 5.0f}, {20.0f, 0.0f}},
    {{20.0f, 0.0f, -200.0f}, {20.0f, 102.5f}},
}};
constexpr std::array<uint16_t, 6> indices{0, 1, 2, 0, 2, 3};
// #endregion

// A 4x4 matrix, column-major (step 22).
using Mat4 = std::array<std::array<float, 4>, 4>;

// The host side of Camera in shaders/textured.slang.
struct Camera {
    Mat4 mvp;
};
static_assert(sizeof(Camera) == 64);

// a * b for column-major matrices (step 22).
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

// Moves points by (x, y, z) (step 22).
Mat4 translate(float x, float y, float z) {
    return {{
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {x, y, z, 1.0f},
    }};
}

// Rotation by a radians about +x (step 22).
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

// #region camera
// The camera floats 1 unit above the floor, tilted down by 0.2 rad, and glides
// forward at 1 unit/s. Moving the floor toward the camera by (time mod 2) looks
// the same as moving the camera, and since the texture repeats every 2 units
// the floor never runs out.
Camera camera(float time, const Mat4& proj) {
    const Mat4 view  = mul(rotateX(0.2f), translate(0.0f, -1.0f, 0.0f));
    const Mat4 model = translate(0.0f, 0.0f, std::fmod(time, 2.0f));
    return Camera{.mvp = mul(proj, mul(view, model))};
}
// #endregion

// #region cli
// Command line: --no-mips, --anisotropy, --seconds S.
struct Args {
    bool mips       = true;
    bool anisotropy = false;
    std::optional<double> seconds;
};

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--no-mips") {
            args.mips = false;
        } else if (arg == "--anisotropy") {
            args.anisotropy = true;
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
            throw std::runtime_error(std::format("unknown argument {} (use --no-mips, --anisotropy, --seconds S)", arg));
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

// #region format-support
// What generating mips on the GPU needs from the format with OPTIMAL tiling:
// step 21's three features, plus BLIT_SRC and BLIT_DST (vkCmdBlitImage2 may
// read and write it). The blit's LINEAR filter additionally needs
// SAMPLED_IMAGE_FILTER_LINEAR on the source format, which is already in the
// list. Formats without blit or linear-filter support (block-compressed formats;
// integer formats, which lack SAMPLED_IMAGE_FILTER_LINEAR - none has it on the
// RTX 3080 Ti - and so cannot be blitted with LINEAR,
// VUID-VkBlitImageInfo2-filter-02001) get their mips from the CPU or an offline
// tool instead.
constexpr std::array<std::pair<vk::FormatFeatureFlagBits, std::string_view>, 5> neededFeatures{{
    {vk::FormatFeatureFlagBits::eSampledImage, "SAMPLED_IMAGE"},
    {vk::FormatFeatureFlagBits::eSampledImageFilterLinear, "SAMPLED_IMAGE_FILTER_LINEAR"},
    {vk::FormatFeatureFlagBits::eTransferDst, "TRANSFER_DST"},
    {vk::FormatFeatureFlagBits::eBlitSrc, "BLIT_SRC"},
    {vk::FormatFeatureFlagBits::eBlitDst, "BLIT_DST"},
}};

// Prints which needed features format has with OPTIMAL tiling and returns
// whether it has all of them.
bool checkFormat(const vk::raii::PhysicalDevice& physicalDevice, vk::Format format) {
    const vk::FormatFeatureFlags optimal = physicalDevice.getFormatProperties2(format).formatProperties.optimalTilingFeatures;
    std::string list;
    bool all = true;
    for (const auto& [bit, name] : neededFeatures) {
        const bool yes = static_cast<bool>(optimal & bit);
        all            = all && yes;
        list += std::format("{}{} {}", list.empty() ? "" : ", ", name, yes ? "yes" : "NO");
    }
    std::println("  {} (optimal tiling): {}", formatName(format), list);
    return all;
}
// #endregion

// A stage + access pair: one side of a barrier.
struct Scope {
    vk::PipelineStageFlags2 stage;
    vk::AccessFlags2 access;
};

// A barrier on ONE mip level of image (all other fields as in step 21).
vk::ImageMemoryBarrier2 levelBarrier(vk::Image image, uint32_t level, Scope src, Scope dst, vk::ImageLayout oldLayout,
                                     vk::ImageLayout newLayout) {
    return vk::ImageMemoryBarrier2{
        .srcStageMask        = src.stage,
        .srcAccessMask       = src.access,
        .dstStageMask        = dst.stage,
        .dstAccessMask       = dst.access,
        .oldLayout           = oldLayout,
        .newLayout           = newLayout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image               = image,
        .subresourceRange    = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = level, .levelCount = 1, .layerCount = 1},
    };
}

// Step 21's staging upload of level 0, then the mip chain generated on the GPU,
// all in one command buffer. Blocks until it is complete.
void uploadWithMips(const vk::raii::Device& device, const vk::PhysicalDeviceMemoryProperties& mem, uint32_t family,
                    const vk::raii::Queue& queue, vk::Image image, uint32_t levels, std::span<const uint8_t> pixels) {
    using Clock      = std::chrono::steady_clock;
    const auto start = Clock::now();
    GpuBuffer staging = createBuffer(device, mem, pixels.size_bytes(), vk::BufferUsageFlagBits::eTransferSrc,
                                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    writeMapped(staging.memory, pixels);
    vk::raii::CommandPool pool(device, vk::CommandPoolCreateInfo{
                                           .flags            = vk::CommandPoolCreateFlagBits::eTransient,
                                           .queueFamilyIndex = family,
                                       });
    vk::raii::CommandBuffer cmd = std::move(vk::raii::CommandBuffers(device, vk::CommandBufferAllocateInfo{
                                                                                 .commandPool        = *pool,
                                                                                 .level              = vk::CommandBufferLevel::ePrimary,
                                                                                 .commandBufferCount = 1,
                                                                             })
                                                .front());
    cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    using Stage  = vk::PipelineStageFlagBits2;
    using Access = vk::AccessFlagBits2;
    using Layout = vk::ImageLayout;
    constexpr Scope none{Stage::eNone, Access::eNone};
    constexpr Scope copyWrite{Stage::eCopy, Access::eTransferWrite};
    constexpr Scope blitWrite{Stage::eBlit, Access::eTransferWrite};
    constexpr Scope blitRead{Stage::eBlit, Access::eTransferRead};
    constexpr Scope shaderRead{Stage::eFragmentShader, Access::eShaderSampledRead};

    // #region upload-level-0
    // Every level starts UNDEFINED -> TRANSFER_DST_OPTIMAL: level 0 for the copy,
    // the others for the blits that will write them. One barrier per level
    // keeps the code uniform; one barrier over all levels (levelCount = levels)
    // would do the same.
    std::vector<vk::ImageMemoryBarrier2> toDst;
    for (uint32_t level = 0; level < levels; ++level) {
        toDst.push_back(levelBarrier(image, level, none, level == 0 ? copyWrite : blitWrite, Layout::eUndefined, Layout::eTransferDstOptimal));
    }
    const vk::BufferImageCopy2 region{
        .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .layerCount = 1},
        .imageExtent      = {texSize, texSize, 1},
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<uint32_t>(toDst.size()),
        .pImageMemoryBarriers    = toDst.data(),
    });
    cmd.copyBufferToImage2(vk::CopyBufferToImageInfo2{
        .srcBuffer      = *staging.buffer,
        .dstImage       = image,
        .dstImageLayout = Layout::eTransferDstOptimal,
        .regionCount    = 1,
        .pRegions       = &region,
    });
    std::println("  mip  0: {}x{} (vkCmdCopyBufferToImage2 from the staging buffer)", texSize, texSize);
    // #endregion

    // #region blit-chain
    // Level i is made from level i-1, which therefore must be finished and
    // readable first: its writer (the copy for level 0, the previous blit
    // otherwise) -> TRANSFER_SRC_OPTIMAL for the blit's read. The blit averages
    // each 2x2 block with the LINEAR filter (sRGB texels are converted to linear
    // before filtering and back after, spec "Image Copies With Scaling"). Once
    // read, level i-1 is done: -> SHADER_READ_ONLY_OPTIMAL. Its read needs no
    // memory to be made available, only the execution dependency (a
    // write-after-read is ordered by execution alone): src access NONE. Each
    // extent halves, rounded down, but never below 1.
    uint32_t size = texSize;
    for (uint32_t level = 1; level < levels; ++level) {
        const uint32_t srcSize = size;
        size                   = std::max(size / 2, 1u);
        const vk::ImageMemoryBarrier2 toSrc =
            levelBarrier(image, level - 1, level == 1 ? copyWrite : blitWrite, blitRead, Layout::eTransferDstOptimal, Layout::eTransferSrcOptimal);
        const auto edge = [](uint32_t s) { return vk::Offset3D{static_cast<int32_t>(s), static_cast<int32_t>(s), 1}; };
        const vk::ImageBlit2 blit{
            .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = level - 1, .layerCount = 1},
            .srcOffsets     = std::array{vk::Offset3D{0, 0, 0}, edge(srcSize)},
            .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = level, .layerCount = 1},
            .dstOffsets     = std::array{vk::Offset3D{0, 0, 0}, edge(size)},
        };
        const vk::ImageMemoryBarrier2 done =
            levelBarrier(image, level - 1, Scope{Stage::eBlit, Access::eNone}, shaderRead, Layout::eTransferSrcOptimal, Layout::eShaderReadOnlyOptimal);
        cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toSrc});
        cmd.blitImage2(vk::BlitImageInfo2{
            .srcImage       = image,
            .srcImageLayout = Layout::eTransferSrcOptimal,
            .dstImage       = image,
            .dstImageLayout = Layout::eTransferDstOptimal,
            .regionCount    = 1,
            .pRegions       = &blit,
            .filter         = vk::Filter::eLinear,
        });
        cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &done});
        std::println("  mip {:2}: {}x{} (vkCmdBlitImage2 from mip {}, LINEAR)", level, size, size, level - 1);
    }
    // The last level was only written, never read by a blit.
    const vk::ImageMemoryBarrier2 last =
        levelBarrier(image, levels - 1, levels == 1 ? copyWrite : blitWrite, shaderRead, Layout::eTransferDstOptimal, Layout::eShaderReadOnlyOptimal);
    cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &last});
    cmd.end();
    // #endregion

    // The guard is declared after everything the submit uses, so it is destroyed
    // first: an exception after the submit waits for the GPU before RAII
    // destroys the staging buffer, the pool and the fence (step 21).
    vk::raii::Fence fence(device, vk::FenceCreateInfo{});
    const IdleGuard idleGuard{device};
    const vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *cmd};
    queue.submit2(vk::SubmitInfo2{.commandBufferInfoCount = 1, .pCommandBufferInfos = &cmdInfo}, *fence);
    if (device.waitForFences(*fence, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
        throw std::runtime_error("waitForFences failed");
    }
    std::println("Upload + mip generation complete (fence waited); every level is SHADER_READ_ONLY_OPTIMAL");
    std::println("[timing] upload + {} blits {:.3f} ms", levels - 1, std::chrono::duration<double, std::milli>(Clock::now() - start).count());
}

// What the CPU touches while recording one frame (step 21).
struct Frame {
    vk::raii::CommandBuffer commandBuffer;
    vk::raii::Semaphore imageAvailable;
    vk::raii::Fence inFlight;
    GpuBuffer uniform;
    Camera* mapped;
    vk::DescriptorSet descriptorSet;
};

int main(int argc, char** argv)
try {
    const Args args = parseArgs(argc, argv);

    // ---- Step 06: window (800x600 physical pixels, fixed size) ----
    Glfw glfw;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 23 mipmaps", nullptr, nullptr), &glfwDestroyWindow);
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
        .pApplicationName   = "Vulkan from Zero - 23 mipmaps",
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

    // ---- Step 21's device selection: Vulkan 1.4, dynamic rendering +
    // synchronization2 + samplerAnisotropy, graphics + present, swapchain ----
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
        const bool anisotropy   = featureChain.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy == vk::True;
        if (features13.dynamicRendering != vk::True || features13.synchronization2 != vk::True || !anisotropy) {
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
        throw std::runtime_error("No Vulkan 1.4 GPU with dynamic rendering + synchronization2 + samplerAnisotropy + VK_KHR_swapchain "
                                 "can present to this window");
    }
    const vk::raii::PhysicalDevice& physicalDevice = *best;
    const vk::PhysicalDeviceProperties chosen      = physicalDevice.getProperties();
    std::println("\nPicked: {}", chosen.deviceName.data());
    std::println("Graphics + present family: {}", family);

    // Step 21's device: samplerAnisotropy enabled (used with --anisotropy).
    constexpr float queuePriority = 1.0f;
    const vk::DeviceQueueCreateInfo queueInfo{
        .queueFamilyIndex = family,
        .queueCount       = 1,
        .pQueuePriorities = &queuePriority,
    };
    const vk::PhysicalDeviceFeatures features{.samplerAnisotropy = vk::True};
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
                                                .pEnabledFeatures        = &features,
                                            });
    std::println("VkDevice created with {} and samplerAnisotropy", vk::KHRSwapchainExtensionName);

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

    const std::vector<uint8_t> pixels = texturePixels();
    std::println("\nTexture: {}x{} RGBA8 checkerboard, {}-texel cells, {} bytes (FNV-1a {:#010x})", texSize, texSize, cell,
                 pixels.size(), fnv1a(pixels));
    constexpr vk::Format texFormat = vk::Format::eR8G8B8A8Srgb;
    std::println("Format support (vkGetPhysicalDeviceFormatProperties2):");
    if (!checkFormat(physicalDevice, texFormat)) {
        throw std::runtime_error(formatName(texFormat) + " cannot be blitted with a LINEAR filter here");
    }

    // #region create-image
    // Step 21's image with a full mip chain: mipLevels = floor(log2(512)) + 1 =
    // 10, and TRANSFER_SRC added to the usage because every level but the last
    // is read by a blit.
    const uint32_t levels = mipCount(texSize, texSize);
    std::println("Mip levels: floor(log2(max({}, {}))) + 1 = {}", texSize, texSize, levels);
    vk::raii::Image texture(device, vk::ImageCreateInfo{
                                        .imageType   = vk::ImageType::e2D,
                                        .format      = texFormat,
                                        .extent      = {texSize, texSize, 1},
                                        .mipLevels   = levels,
                                        .arrayLayers = 1,
                                        .samples     = vk::SampleCountFlagBits::e1,
                                        .tiling      = vk::ImageTiling::eOptimal,
                                        .usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
                                                 vk::ImageUsageFlagBits::eSampled,
                                        .sharingMode   = vk::SharingMode::eExclusive,
                                        .initialLayout = vk::ImageLayout::eUndefined,
                                    });
    const vk::MemoryRequirements req = texture.getMemoryRequirements();
    const auto texType               = findType(mem, req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    if (!texType) {
        throw std::runtime_error("no DEVICE_LOCAL memory type for the texture");
    }
    const uint64_t level0 = uint64_t{texSize} * texSize * 4;
    std::println("VkImage: {} mip levels, TRANSFER_SRC | TRANSFER_DST | SAMPLED; memory {} bytes (level 0 alone: {}; the chain adds "
                 "about a third), memory type {}",
                 levels, req.size, level0, *texType);
    vk::raii::DeviceMemory textureMemory(device, vk::MemoryAllocateInfo{.allocationSize = req.size, .memoryTypeIndex = *texType});
    texture.bindMemory(*textureMemory, 0);
    // #endregion

    std::println("Recording the upload and the mip chain:");
    uploadWithMips(device, mem, family, queue, *texture, levels, pixels);

    // #region image-view
    // The view covers ALL levels (levelCount = levels): the sampler can only
    // choose among the levels the view exposes.
    vk::raii::ImageView textureView(device, vk::ImageViewCreateInfo{
                                                .image            = *texture,
                                                .viewType         = vk::ImageViewType::e2D,
                                                .format           = texFormat,
                                                .subresourceRange = {.aspectMask   = vk::ImageAspectFlagBits::eColor,
                                                                     .baseMipLevel = 0,
                                                                     .levelCount   = levels,
                                                                     .layerCount   = 1},
                                            });
    // #endregion

    // #region sampler
    // Trilinear filtering: LINEAR within a level (min/mag filters) and LINEAR
    // between the two nearest levels (mipmapMode). The GPU picks the level from
    // the UV derivatives across a 2x2 pixel quad: lambda = log2(texels per
    // pixel) + mipLodBias, clamped to [minLod, maxLod]. maxLod = the level count
    // lets it reach the 1x1 level (VK_LOD_CLAMP_NONE would too); --no-mips sets
    // maxLod = 0, so only level 0 is ever read although the image still has all
    // levels, and with lambda <= 0 the texture always counts as magnified
    // (magFilter; the same LINEAR as minFilter here, spec Note "Mapping of
    // OpenGL to Vulkan Filter Modes"). Anisotropic filtering (--anisotropy) takes several samples along
    // the long axis of the pixel's footprint: on a floor seen at a grazing angle
    // the footprint is long and thin, and plain trilinear blurs it.
    const float maxLod   = args.mips ? static_cast<float>(levels) : 0.0f;
    const float maxAniso = chosen.limits.maxSamplerAnisotropy;
    const std::optional<float> anisotropy = args.anisotropy ? std::optional{std::min(maxAniso, 16.0f)} : std::nullopt;
    vk::raii::Sampler sampler(device, vk::SamplerCreateInfo{
                                          .magFilter               = vk::Filter::eLinear,
                                          .minFilter               = vk::Filter::eLinear,
                                          .mipmapMode              = vk::SamplerMipmapMode::eLinear,
                                          .addressModeU            = vk::SamplerAddressMode::eRepeat,
                                          .addressModeV            = vk::SamplerAddressMode::eRepeat,
                                          .addressModeW            = vk::SamplerAddressMode::eRepeat,
                                          .mipLodBias              = 0.0f,
                                          .anisotropyEnable        = anisotropy ? vk::True : vk::False,
                                          .maxAnisotropy           = anisotropy.value_or(1.0f),
                                          .compareEnable           = vk::False,
                                          .minLod                  = 0.0f,
                                          .maxLod                  = maxLod,
                                          .borderColor             = vk::BorderColor::eFloatOpaqueBlack,
                                          .unnormalizedCoordinates = vk::False,
                                      });
    const std::string anisoText = anisotropy ? std::format("{}", *anisotropy) : std::string{"off"};
    std::println("Sampler: LINEAR filters, mipmapMode LINEAR, minLod 0, maxLod {}{}, anisotropy {} (maxSamplerAnisotropy {})", maxLod,
                 args.mips ? "" : " (--no-mips: level 0 only)", anisoText, maxAniso);
    // #endregion

    // The SPIR-V is compiled from shaders/textured.slang by `bun run
    // sync:shaders`; CMake bakes the absolute path of shaders/spv into SHADER_DIR.
    const std::vector<uint32_t> spirv = readSpirv(std::filesystem::path{SHADER_DIR} / "textured.spv");
    std::println("\nSPIR-V: {} bytes (textured.spv), entry points vertexMain, fragmentMain", spirv.size() * sizeof(uint32_t));
    std::optional<vk::raii::ShaderModule> shaderModule;
    shaderModule.emplace(device, vk::ShaderModuleCreateInfo{
                                     .codeSize = spirv.size() * sizeof(uint32_t),
                                     .pCode    = spirv.data(),
                                 });
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = **shaderModule, .pName = "vertexMain"},
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = **shaderModule, .pName = "fragmentMain"},
    };

    // Step 21's vertex input: binding 0, stride 20, position + uv.
    const vk::VertexInputBindingDescription vertexBinding{
        .binding   = 0,
        .stride    = sizeof(Vertex),
        .inputRate = vk::VertexInputRate::eVertex,
    };
    const std::array attributes{
        vk::VertexInputAttributeDescription{
            .location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, position)},
        vk::VertexInputAttributeDescription{.location = 1, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Vertex, uv)},
    };
    const vk::PipelineVertexInputStateCreateInfo vertexInput{
        .vertexBindingDescriptionCount   = 1,
        .pVertexBindingDescriptions      = &vertexBinding,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size()),
        .pVertexAttributeDescriptions    = attributes.data(),
    };

    // Step 21's set layout: binding 0 = uniform buffer, binding 1 = combined
    // image sampler.
    const std::array setBindings{
        vk::DescriptorSetLayoutBinding{
            .binding         = 0,
            .descriptorType  = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags      = vk::ShaderStageFlagBits::eVertex,
        },
        vk::DescriptorSetLayoutBinding{
            .binding         = 1,
            .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags      = vk::ShaderStageFlagBits::eFragment,
        },
    };
    vk::raii::DescriptorSetLayout setLayout(device, vk::DescriptorSetLayoutCreateInfo{
                                                        .bindingCount = static_cast<uint32_t>(setBindings.size()),
                                                        .pBindings    = setBindings.data(),
                                                    });
    const vk::DescriptorSetLayout setLayoutHandle = *setLayout;
    vk::raii::PipelineLayout pipelineLayout(device, vk::PipelineLayoutCreateInfo{.setLayoutCount = 1, .pSetLayouts = &setLayoutHandle});

    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
    const vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
    // Step 22's projection keeps counter-clockwise faces counter-clockwise.
    const vk::PipelineRasterizationStateCreateInfo rasterization{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode    = vk::CullModeFlagBits::eBack,
        .frontFace   = vk::FrontFace::eCounterClockwise,
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
    std::println("Graphics pipeline created");

    // Step 14's mesh buffers: HOST_VISIBLE | HOST_COHERENT, written once.
    const vk::MemoryPropertyFlags hostFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    const std::span vertexData{vertices};
    const std::span indexData{indices};
    GpuBuffer vertexBuffer = createBuffer(device, mem, vertexData.size_bytes(), vk::BufferUsageFlagBits::eVertexBuffer, hostFlags);
    writeMapped(vertexBuffer.memory, vertexData);
    GpuBuffer indexBuffer = createBuffer(device, mem, indexData.size_bytes(), vk::BufferUsageFlagBits::eIndexBuffer, hostFlags);
    writeMapped(indexBuffer.memory, indexData);
    std::println("Mesh: floor 40 x 205 units, UVs 0..20 x 0..102.5 (REPEAT)");

    // Step 21's descriptors: per frame a uniform buffer + the texture.
    constexpr vk::DeviceSize uniformSize = sizeof(Camera);
    constexpr auto frameCount            = static_cast<uint32_t>(framesInFlight);
    const std::array poolSizes{
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = frameCount},
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = frameCount},
    };
    vk::raii::DescriptorPool descriptorPool(device, vk::DescriptorPoolCreateInfo{
                                                        .maxSets       = frameCount,
                                                        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                                                        .pPoolSizes    = poolSizes.data(),
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
    const vk::DescriptorImageInfo imageInfo{
        .sampler     = *sampler,
        .imageView   = *textureView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    std::vector<GpuBuffer> uniforms;
    std::vector<Camera*> mapped;
    for (const vk::DescriptorSet set : descriptorSets) {
        uniforms.push_back(createBuffer(device, mem, uniformSize, vk::BufferUsageFlagBits::eUniformBuffer, hostFlags));
        mapped.push_back(static_cast<Camera*>(uniforms.back().memory.mapMemory(0, vk::WholeSize)));
        const vk::DescriptorBufferInfo bufferInfo{.buffer = *uniforms.back().buffer, .offset = 0, .range = uniformSize};
        const std::array writes{
            vk::WriteDescriptorSet{
                .dstSet          = set,
                .dstBinding      = 0,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo     = &bufferInfo,
            },
            vk::WriteDescriptorSet{
                .dstSet          = set,
                .dstBinding      = 1,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo      = &imageInfo,
            },
        };
        device.updateDescriptorSets(writes, nullptr);
    }

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
    const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    const Mat4 proj    = perspective(std::numbers::pi_v<float> / 3.0f, aspect, 0.1f, 300.0f);
    std::println("\n{} frames in flight; camera 1 unit above the floor, gliding forward at 1 unit/s", framesInFlight);

    using Clock       = std::chrono::steady_clock;
    using Seconds     = std::chrono::duration<double>;
    const auto start  = Clock::now();
    size_t frameIndex = 0;

    // Renders and presents one frame (step 21's loop). Returns false when the
    // swapchain is out of date (recreation is step 11).
    const auto drawFrame = [&]() -> bool {
        Frame& frame                       = frames.at(frameIndex % frames.size());
        const vk::raii::CommandBuffer& cmd = frame.commandBuffer;
        if (device.waitForFences(*frame.inFlight, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
            throw std::runtime_error("waitForFences failed");
        }
        const auto time         = static_cast<float>(Seconds(Clock::now() - start).count());
        const Camera cameraData = camera(time, proj);
        std::memcpy(frame.mapped, &cameraData, sizeof cameraData);

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
        // A sky-like clear color behind the horizon.
        const vk::RenderingAttachmentInfo colorAttachment{
            .imageView   = *imageViews.at(imageIndex),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eStore,
            .clearValue  = vk::ClearColorValue{std::array{0.35f, 0.45f, 0.6f, 1.0f}},
        };
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
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, frame.descriptorSet, nullptr);
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
    device.waitIdle();
    return EXIT_SUCCESS;
} catch (const vk::SystemError& e) {
    std::println(stderr, "Vulkan error: {}", e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return EXIT_FAILURE;
}
