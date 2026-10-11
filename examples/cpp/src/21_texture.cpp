// Step 21 - Textures and samplers
//
// A shader reads an image through two objects: an IMAGE VIEW (which part of
// which image, in which format) and a SAMPLER (how to filter between texels
// and what to do outside [0, 1]). Getting pixels into the image is step 15's
// staging upload, plus two things buffers never needed: a copy that knows rows
// and texel formats (vkCmdCopyBufferToImage2), and LAYOUT TRANSITIONS, because
// an OPTIMAL-tiling image stores its texels in a hidden, GPU-friendly
// arrangement that differs per use.
//
//   CPU pixels -> staging buffer -> [UNDEFINED -> TRANSFER_DST_OPTIMAL]
//     -> vkCmdCopyBufferToImage2 -> [TRANSFER_DST -> SHADER_READ_ONLY_OPTIMAL]
//   image view + sampler -> COMBINED_IMAGE_SAMPLER descriptor (set 0, binding 1)
//   fragment shader: albedo.Sample(uv)
//
// --nearest switches the filters from LINEAR to NEAREST; --address MODE
// (repeat | mirror | clamp | border) picks what UVs outside [0, 1] show (the
// quad's UVs run 0..2); --seconds S closes the window after S seconds. Lines
// that depend on timing start with [timing].
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
std::string filterName(vk::Filter f) { return shortName(string_VkFilter(static_cast<VkFilter>(f)), "VK_FILTER_"); }
std::string addressName(vk::SamplerAddressMode m) {
    return shortName(string_VkSamplerAddressMode(static_cast<VkSamplerAddressMode>(m)), "VK_SAMPLER_ADDRESS_MODE_");
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

// #region texture-data
// Width and height of the texture in texels.
constexpr uint32_t texSize = 256;
// Side of one checker cell in texels (8 x 8 cells).
constexpr uint32_t cell = 32;

// The texture's RGBA8 bytes, row by row from the top, generated here rather
// than loaded from a PNG: no image-decoding library in either language, the
// same bytes in Rust and C++ (integer math only; the hash printed below proves
// it), and the lesson stays about Vulkan, not file formats. A real program
// decodes a file (or, better, loads a GPU-ready KTX2/DDS) into exactly such a
// byte array. Cells alternate between a color gradient and dark gray; a
// 2-texel white frame marks the edges, so the address modes show where one
// copy of the texture ends.
std::vector<uint8_t> texturePixels() {
    std::vector<uint8_t> pixels;
    pixels.reserve(size_t{texSize} * texSize * 4);
    for (uint32_t y = 0; y < texSize; ++y) {
        for (uint32_t x = 0; x < texSize; ++x) {
            const bool edge = x < 2 || y < 2 || x >= texSize - 2 || y >= texSize - 2;
            std::array<uint32_t, 3> rgb{40, 40, 40};
            if (edge) {
                rgb = {255, 255, 255};
            } else if ((x / cell + y / cell) % 2 == 0) {
                rgb = {x, y, 128 + (x + y) / 4};
            }
            // Every value is <= 255 by construction (x, y < 256).
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

// Step 14's vertex, with a third coordinate and a UV instead of a color:
// 3 + 2 floats = 20 bytes (shaders/textured.slang).
struct Vertex {
    std::array<float, 3> position;
    std::array<float, 2> uv; // texture coordinates: (0, 0) = the texture's first texel row, left
};
static_assert(sizeof(Vertex) == 20);

// #region quad
// A square of side 1.5 with UVs 0..2: the texture appears 2 x 2 times with
// REPEAT, and the address mode decides what the other three copies show.
// Top-left, top-right, bottom-right, bottom-left (+y points down): clockwise on
// screen, the pipeline's front face (step 17).
constexpr std::array<Vertex, 4> vertices{{
    {{-0.75f, -0.75f, 0.0f}, {0.0f, 0.0f}},
    {{0.75f, -0.75f, 0.0f}, {2.0f, 0.0f}},
    {{0.75f, 0.75f, 0.0f}, {2.0f, 2.0f}},
    {{-0.75f, 0.75f, 0.0f}, {0.0f, 2.0f}},
}};
constexpr std::array<uint16_t, 6> indices{0, 1, 2, 0, 2, 3};
// #endregion

// The host side of Camera in shaders/textured.slang: one column-major matrix
// (step 17 checked this layout against the SPIR-V).
struct Camera {
    std::array<std::array<float, 4>, 4> mvp;
};
static_assert(sizeof(Camera) == 64);

// Rotation speed of the quad, radians per second: slow, so the filtering stays
// visible.
constexpr float spin = 0.3f;

// Step 17's matrix: a rotation by angle about z, then x scaled by aspect
// (height / width). Column-major.
std::array<std::array<float, 4>, 4> transform(float angle, float aspect) {
    const float s = std::sin(angle);
    const float c = std::cos(angle);
    return {{
        {aspect * c, s, 0.0f, 0.0f},
        {-aspect * s, c, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
    }};
}

// #region cli
// Command line: --nearest, --address repeat|mirror|clamp|border, --seconds S.
struct Args {
    vk::Filter filter               = vk::Filter::eLinear;
    vk::SamplerAddressMode address  = vk::SamplerAddressMode::eRepeat;
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
        if (arg == "--nearest") {
            args.filter = vk::Filter::eNearest;
        } else if (arg == "--address") {
            const std::string_view mode = value();
            if (mode == "repeat") {
                args.address = vk::SamplerAddressMode::eRepeat;
            } else if (mode == "mirror") {
                args.address = vk::SamplerAddressMode::eMirroredRepeat;
            } else if (mode == "clamp") {
                args.address = vk::SamplerAddressMode::eClampToEdge;
            } else if (mode == "border") {
                args.address = vk::SamplerAddressMode::eClampToBorder;
            } else {
                throw std::runtime_error(std::format("unknown address mode {} (use repeat | mirror | clamp | border)", mode));
            }
        } else if (arg == "--seconds") {
            const std::string_view text = value();
            double seconds              = 0.0;
            const auto [end, ec]        = std::from_chars(text.data(), text.data() + text.size(), seconds);
            if (ec != std::errc{} || end != text.data() + text.size()) {
                throw std::runtime_error(std::format("--seconds: not a number: {}", text));
            }
            args.seconds = seconds;
        } else {
            throw std::runtime_error(std::format("unknown argument {} (use --nearest, --address MODE, --seconds S)", arg));
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

// #region format-support
// The format features this step needs from an OPTIMAL-tiling image, with their
// names for printing.
constexpr std::array<std::pair<vk::FormatFeatureFlagBits, std::string_view>, 3> neededFeatures{{
    {vk::FormatFeatureFlagBits::eSampledImage, "SAMPLED_IMAGE"},
    {vk::FormatFeatureFlagBits::eSampledImageFilterLinear, "SAMPLED_IMAGE_FILTER_LINEAR"},
    {vk::FormatFeatureFlagBits::eTransferDst, "TRANSFER_DST"},
}};

// What format supports with OPTIMAL tiling: one line per format, and whether
// every needed feature is there. Linear tiling (texels row by row, mappable)
// supports far less and is not used for textures.
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

// Step 15's staging upload, for an image: CPU -> staging buffer ->
// vkCmdCopyBufferToImage2 on the graphics queue, with the two layout
// transitions around the copy. Blocks until the copy is complete.
void uploadTexture(const vk::raii::Device& device, const vk::PhysicalDeviceMemoryProperties& mem, uint32_t family,
                   const vk::raii::Queue& queue, vk::Image image, std::span<const uint8_t> pixels) {
    using Clock      = std::chrono::steady_clock;
    const auto start = Clock::now();
    const vk::DeviceSize size = pixels.size_bytes();
    GpuBuffer staging         = createBuffer(device, mem, size, vk::BufferUsageFlagBits::eTransferSrc,
                                             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    writeMapped(staging.memory, pixels);
    std::println("Staging buffer: {} bytes, TRANSFER_SRC, memory type {}", size, staging.memoryType);

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
    constexpr vk::ImageSubresourceRange colorRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };

    // #region to-transfer-dst
    // Barrier 1: UNDEFINED -> TRANSFER_DST_OPTIMAL, the only layout a copy may
    // write. UNDEFINED = "the contents may be thrown away", which is right for a
    // fresh image. Nothing earlier touches it, so the first scope is empty
    // (stage NONE, access NONE); the second scope is the copy's write (COPY,
    // TRANSFER_WRITE), which must wait for the transition.
    const vk::ImageMemoryBarrier2 toTransfer{
        .srcStageMask        = vk::PipelineStageFlagBits2::eNone,
        .srcAccessMask       = vk::AccessFlagBits2::eNone,
        .dstStageMask        = vk::PipelineStageFlagBits2::eCopy,
        .dstAccessMask       = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout           = vk::ImageLayout::eUndefined,
        .newLayout           = vk::ImageLayout::eTransferDstOptimal,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image               = image,
        .subresourceRange    = colorRange,
    };
    // #endregion

    // #region copy
    // One region: the whole buffer into mip level 0, layer 0. Row length and
    // image height 0 = "tightly packed": the rows follow each other with no
    // padding, exactly as texturePixels() wrote them. The copy converts the
    // linear rows of the buffer into the image's hidden optimal tiling.
    const vk::BufferImageCopy2 region{
        .bufferOffset      = 0,
        .bufferRowLength   = 0,
        .bufferImageHeight = 0,
        .imageSubresource  = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset       = {0, 0, 0},
        .imageExtent       = {texSize, texSize, 1},
    };
    const vk::CopyBufferToImageInfo2 copyInfo{
        .srcBuffer      = *staging.buffer,
        .dstImage       = image,
        .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
        .regionCount    = 1,
        .pRegions       = &region,
    };
    // #endregion

    // #region to-shader-read
    // Barrier 2: TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL, the layout a
    // sampled image is read in. First scope: the copy's write (made available);
    // second scope: the fragment shader's sampled reads (made visible), in this
    // and every later submission on the queue.
    const vk::ImageMemoryBarrier2 toShader{
        .srcStageMask        = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask       = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask        = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask       = vk::AccessFlagBits2::eShaderSampledRead,
        .oldLayout           = vk::ImageLayout::eTransferDstOptimal,
        .newLayout           = vk::ImageLayout::eShaderReadOnlyOptimal,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image               = image,
        .subresourceRange    = colorRange,
    };
    // #endregion

    cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toTransfer});
    cmd.copyBufferToImage2(copyInfo);
    cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toShader});
    cmd.end();
    std::println("Recorded: barrier UNDEFINED -> TRANSFER_DST_OPTIMAL, vkCmdCopyBufferToImage2 ({}x{}, mip 0), barrier "
                 "TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL",
                 texSize, texSize);

    // The fence says the copy is complete, so the staging buffer may go; the
    // barrier above, not the fence, makes the texels visible to the shader.
    // The guard is declared after everything the submit uses, so it is
    // destroyed first: an exception after the submit waits for the GPU before
    // RAII destroys the staging buffer, the pool and the fence.
    vk::raii::Fence fence(device, vk::FenceCreateInfo{});
    const IdleGuard idleGuard{device};
    const vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *cmd};
    queue.submit2(vk::SubmitInfo2{.commandBufferInfoCount = 1, .pCommandBufferInfos = &cmdInfo}, *fence);
    if (device.waitForFences(*fence, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
        throw std::runtime_error("waitForFences failed");
    }
    // Destroy the upload's objects now, as Rust's UploadScratch drop does, so
    // the line below is true: the fence wait proved the GPU is done with them.
    // The command buffer is freed before its pool, the buffer before its memory.
    fence.clear();
    cmd.clear();
    pool.clear();
    staging.buffer.clear();
    staging.memory.clear();
    std::println("Upload complete (fence waited); staging buffer destroyed");
    std::println("[timing] texture upload {:.3f} ms (staging write + record + submit + wait)",
                 std::chrono::duration<double, std::milli>(Clock::now() - start).count());
}

// #region frame-resources
// What the CPU touches while recording one frame (step 17): command buffer,
// sync objects, the slot's uniform buffer and its descriptor set, which now
// also points to the texture.
struct Frame {
    vk::raii::CommandBuffer commandBuffer;
    vk::raii::Semaphore imageAvailable;
    vk::raii::Fence inFlight;
    GpuBuffer uniform;
    Camera* mapped;
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
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 21 texture", nullptr, nullptr), &glfwDestroyWindow);
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
        .pApplicationName   = "Vulkan from Zero - 21 texture",
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
    // New: samplerAnisotropy, which every Vulkan 1.4 device must support (spec,
    // "Feature Requirements") but which still has to be ENABLED.
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

    // #region device-features
    // Step 07's device, plus samplerAnisotropy from VkPhysicalDeviceFeatures
    // (pEnabledFeatures; the 1.3 features stay in the pNext chain).
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
    vk::raii::Queue queue(device, family, 0);
    const vk::PhysicalDeviceMemoryProperties mem = physicalDevice.getMemoryProperties();

    // #region texture-format
    // The texels are COLOR, authored on a monitor: their values are sRGB encoded
    // (non-linear). With R8G8B8A8_SRGB the sampler decodes them to linear before
    // filtering, so blending between texels and the shader math happen in
    // linear light, and the _SRGB swapchain re-encodes on write. R8G8B8A8_UNORM
    // would hand the shader the raw encoded numbers: right for data (normal
    // maps, roughness, masks), wrong for color - the already-encoded values are
    // taken as linear, the _SRGB swapchain encodes them a second time (too
    // bright, washed out), and filtering blends non-linear values.
    const std::vector<uint8_t> pixels = texturePixels();
    std::println("\nTexture: {}x{} RGBA8, {} bytes, generated on the CPU (FNV-1a {:#010x})", texSize, texSize, pixels.size(),
                 fnv1a(pixels));
    constexpr vk::Format texFormat = vk::Format::eR8G8B8A8Srgb;
    std::println("Format support (vkGetPhysicalDeviceFormatProperties2):");
    const bool srgbOk = checkFormat(physicalDevice, texFormat);
    checkFormat(physicalDevice, vk::Format::eR8G8B8A8Unorm);
    if (!srgbOk) {
        throw std::runtime_error(formatName(texFormat) + " lacks a needed optimal-tiling feature");
    }
    std::println("Texture format: {} (color data is sRGB-encoded)", formatName(texFormat));
    // #endregion

    // #region create-image
    // OPTIMAL tiling: the driver arranges the texels (tiles, swizzles, maybe
    // compression) as the GPU's texture units like them; the CPU can neither
    // map nor address them, hence the copy from a buffer. Usage: TRANSFER_DST
    // (the copy writes it) | SAMPLED (shaders read it).
    vk::raii::Image texture(device, vk::ImageCreateInfo{
                                        .imageType     = vk::ImageType::e2D,
                                        .format        = texFormat,
                                        .extent        = {texSize, texSize, 1},
                                        .mipLevels     = 1,
                                        .arrayLayers   = 1,
                                        .samples       = vk::SampleCountFlagBits::e1,
                                        .tiling        = vk::ImageTiling::eOptimal,
                                        .usage         = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                                        .sharingMode   = vk::SharingMode::eExclusive,
                                        .initialLayout = vk::ImageLayout::eUndefined,
                                    });
    std::println("VkImage: 2D, 1 mip level, OPTIMAL tiling, TRANSFER_DST | SAMPLED");
    // #endregion

    // #region image-memory
    // Images ask for memory like buffers (step 13), but the size is the
    // driver's: tiling and alignment padding make it differ from
    // width * height * 4, and the allowed types may differ from buffers'.
    const vk::MemoryRequirements req = texture.getMemoryRequirements();
    const auto texType               = findType(mem, req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    if (!texType) {
        throw std::runtime_error("no DEVICE_LOCAL memory type for the texture");
    }
    std::println("Memory requirements: size {}, alignment {}, memoryTypeBits {:#b} -> memory type {} (DEVICE_LOCAL)", req.size,
                 req.alignment, req.memoryTypeBits, *texType);
    vk::raii::DeviceMemory textureMemory(device, vk::MemoryAllocateInfo{.allocationSize = req.size, .memoryTypeIndex = *texType});
    texture.bindMemory(*textureMemory, 0);
    // #endregion

    uploadTexture(device, mem, family, queue, *texture, pixels);

    // #region image-view
    // The view is what descriptors point to: the image seen as a 2D texture of
    // this format, color aspect, mip 0, layer 0.
    vk::raii::ImageView textureView(device, vk::ImageViewCreateInfo{
                                                .image            = *texture,
                                                .viewType         = vk::ImageViewType::e2D,
                                                .format           = texFormat,
                                                .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                                     .levelCount = 1,
                                                                     .layerCount = 1},
                                            });
    // #endregion

    // #region sampler
    // The sampler is independent of any image: filters (magnification: a texel
    // covers many pixels; minification: many texels per pixel), address modes
    // for UVs outside [0, 1] (u, v and w), anisotropy, and the mip settings
    // (one level here: maxLod 0). maxLod 0 clamps the LOD lambda to <= 0, so
    // the texture always counts as magnified and magFilter is the one used;
    // harmless here because both filters are equal (spec Note "Mapping of
    // OpenGL to Vulkan Filter Modes": GL's non-mipmapped minification is
    // maxLod 0.25). Anisotropic filtering takes up to
    // maxAnisotropy samples along the direction in which the texture is
    // squeezed on screen; 16 is the usual maximum, clamped to the device limit
    // (VUID-VkSamplerCreateInfo-anisotropyEnable-01071). With --nearest it is
    // off, so NEAREST shows its raw blockiness.
    const float maxAniso = chosen.limits.maxSamplerAnisotropy;
    const std::optional<float> anisotropy =
        args.filter == vk::Filter::eLinear ? std::optional{std::min(maxAniso, 16.0f)} : std::nullopt;
    vk::raii::Sampler sampler(device, vk::SamplerCreateInfo{
                                          .magFilter               = args.filter,
                                          .minFilter               = args.filter,
                                          .mipmapMode              = vk::SamplerMipmapMode::eNearest,
                                          .addressModeU            = args.address,
                                          .addressModeV            = args.address,
                                          .addressModeW            = args.address,
                                          .mipLodBias              = 0.0f,
                                          .anisotropyEnable        = anisotropy ? vk::True : vk::False,
                                          .maxAnisotropy           = anisotropy.value_or(1.0f),
                                          .compareEnable           = vk::False,
                                          .minLod                  = 0.0f,
                                          .maxLod                  = 0.0f,
                                          .borderColor             = vk::BorderColor::eFloatOpaqueBlack,
                                          .unnormalizedCoordinates = vk::False,
                                      });
    std::println("maxSamplerAnisotropy = {}", maxAniso);
    if (anisotropy) {
        std::println("Sampler: {} filters, address mode {}, anisotropy {}", filterName(args.filter), addressName(args.address),
                     *anisotropy);
    } else {
        std::println("Sampler: {} filters, address mode {}, anisotropy off", filterName(args.filter), addressName(args.address));
    }
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

    // Vertex input: binding 0, stride 20, position (3 floats) + uv (2).
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

    // #region set-layout
    // Set 0 now has two bindings: 0 = step 17's uniform buffer (vertex stage),
    // 1 = the texture as ONE combined image sampler (fragment stage). Combined =
    // view + sampler in one descriptor, the simplest form and what Sampler2D in
    // Slang/GLSL's sampler2D map to. The other option is SAMPLED_IMAGE + SAMPLER
    // as two bindings (Texture2D and SamplerState in the shader): worth it when
    // one sampler serves many images, as in bindless renderers (step 20).
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
    std::println("Descriptor set layout: binding 0 = UNIFORM_BUFFER (VERTEX), binding 1 = COMBINED_IMAGE_SAMPLER (FRAGMENT)");
    // #endregion

    const vk::DescriptorSetLayout setLayoutHandle = *setLayout;
    vk::raii::PipelineLayout pipelineLayout(device, vk::PipelineLayoutCreateInfo{.setLayoutCount = 1, .pSetLayouts = &setLayoutHandle});

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
    std::println("Graphics pipeline created");

    // Step 14's mesh buffers: HOST_VISIBLE | HOST_COHERENT, written once.
    const vk::MemoryPropertyFlags hostFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    const std::span vertexData{vertices};
    const std::span indexData{indices};
    GpuBuffer vertexBuffer = createBuffer(device, mem, vertexData.size_bytes(), vk::BufferUsageFlagBits::eVertexBuffer, hostFlags);
    writeMapped(vertexBuffer.memory, vertexData);
    GpuBuffer indexBuffer = createBuffer(device, mem, indexData.size_bytes(), vk::BufferUsageFlagBits::eIndexBuffer, hostFlags);
    writeMapped(indexBuffer.memory, indexData);
    std::println("Mesh: quad, {} vertices, {} indices, UVs 0..2", vertices.size(), indices.size());

    // Step 17's per-frame uniform buffers, persistently mapped.
    constexpr vk::DeviceSize uniformSize = sizeof(Camera);
    std::vector<GpuBuffer> uniforms;
    std::vector<Camera*> mapped;
    for (size_t i = 0; i < framesInFlight; ++i) {
        uniforms.push_back(createBuffer(device, mem, uniformSize, vk::BufferUsageFlagBits::eUniformBuffer, hostFlags));
        mapped.push_back(static_cast<Camera*>(uniforms.back().memory.mapMemory(0, vk::WholeSize)));
    }

    // #region descriptor-pool
    // Room for exactly what we allocate: 2 sets, each with one uniform buffer
    // and one combined image sampler.
    constexpr auto frameCount = static_cast<uint32_t>(framesInFlight);
    const std::array poolSizes{
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = frameCount},
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = frameCount},
    };
    vk::raii::DescriptorPool descriptorPool(device, vk::DescriptorPoolCreateInfo{
                                                        .maxSets       = frameCount,
                                                        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                                                        .pPoolSizes    = poolSizes.data(),
                                                    });
    // Plain handles via release(): the sets go with the pool (step 17).
    const std::vector<vk::DescriptorSetLayout> setLayouts(framesInFlight, *setLayout);
    std::vector<vk::DescriptorSet> descriptorSets;
    for (vk::raii::DescriptorSet& set : vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
                                                                             .descriptorPool     = *descriptorPool,
                                                                             .descriptorSetCount = frameCount,
                                                                             .pSetLayouts        = setLayouts.data(),
                                                                         })) {
        descriptorSets.push_back(set.release());
    }
    std::println("Descriptor pool: maxSets {}, UNIFORM_BUFFER x {}, COMBINED_IMAGE_SAMPLER x {}; {} sets allocated", frameCount,
                 frameCount, frameCount, descriptorSets.size());
    // #endregion

    // #region write-sets
    // Each set gets its own uniform buffer (binding 0) and the same texture
    // (binding 1). An image descriptor carries the LAYOUT the image will be in
    // when the shader reads it: SHADER_READ_ONLY_OPTIMAL, the layout the
    // upload's last barrier left it in.
    const vk::DescriptorImageInfo imageInfo{
        .sampler     = *sampler,
        .imageView   = *textureView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    std::vector<vk::DescriptorBufferInfo> bufferInfos;
    for (const GpuBuffer& u : uniforms) {
        bufferInfos.push_back(vk::DescriptorBufferInfo{.buffer = *u.buffer, .offset = 0, .range = uniformSize});
    }
    std::vector<vk::WriteDescriptorSet> writes;
    for (size_t i = 0; i < descriptorSets.size(); ++i) {
        writes.push_back(vk::WriteDescriptorSet{
            .dstSet          = descriptorSets[i],
            .dstBinding      = 0,
            .descriptorCount = 1,
            .descriptorType  = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo     = &bufferInfos[i],
        });
        writes.push_back(vk::WriteDescriptorSet{
            .dstSet          = descriptorSets[i],
            .dstBinding      = 1,
            .descriptorCount = 1,
            .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo      = &imageInfo,
        });
    }
    device.updateDescriptorSets(writes, nullptr);
    std::println("vkUpdateDescriptorSets: {} writes (per set: binding 0 -> its uniform buffer, binding 1 -> texture view + sampler, "
                 "SHADER_READ_ONLY_OPTIMAL)",
                 writes.size());
    // #endregion

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
    const float aspect = static_cast<float>(extent.height) / static_cast<float>(extent.width);
    std::println("\n{} frames in flight; the quad turns at {} rad/s", framesInFlight, spin);

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
        const auto time = static_cast<float>(Seconds(Clock::now() - start).count());
        const Camera camera{.mvp = transform(time * spin, aspect)};
        std::memcpy(frame.mapped, &camera, sizeof camera);

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
        const vk::RenderingAttachmentInfo colorAttachment{
            .imageView   = *imageViews.at(imageIndex),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eStore,
            .clearValue  = vk::ClearColorValue{std::array{0.02f, 0.02f, 0.03f, 1.0f}},
        };
        // #region bind-and-draw
        // Step 17's draw: binding the set binds both descriptors at once; the
        // texture needs no per-frame work, it never changes.
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

    // #region cleanup
    // Wait until the GPU is idle, then RAII destroys in reverse order of
    // declaration: frames, descriptor pool (its sets), mesh buffers, pipeline,
    // layouts, the sampler, the view, the texture's memory, the image. Freeing
    // memory that is still bound is allowed once nothing uses the image again.
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
