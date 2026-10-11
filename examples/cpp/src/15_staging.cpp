// Step 15 - Staging uploads
//
// Step 14 left the mesh in system RAM, read by the GPU over PCIe on every draw.
// The fastest memory for the GPU is DEVICE_LOCAL VRAM, which the CPU usually
// cannot map. So the CPU writes into a STAGING buffer it can map, and the GPU
// copies from there into the device-local buffers - like writing a cache file
// to local scratch before the farm renders from it:
//
//   CPU memcpy -> staging (HOST_VISIBLE) -> vkCmdCopyBuffer -> vertex/index buffers (DEVICE_LOCAL)
//
// Three upload paths, chosen on the command line:
// - default: the copy runs on the graphics queue, followed by a barrier;
// - --transfer-queue: the copy runs on a queue of the dedicated transfer family
//   (the GPU's copy engines); the buffers then change owner from the transfer
//   family to the graphics family (release + acquire barriers, ordered by a
//   semaphore);
// - --bar: no staging at all: the buffers live in the memory type that is
//   DEVICE_LOCAL and HOST_VISIBLE (the CPU-visible window of VRAM, 214 MiB on
//   the course GPU with Resizable BAR off) and the CPU writes them directly.
//
// --mib N pads the vertex buffer to N MiB (zeros after the mesh) so the timing
// shows bandwidth, not just latency. The upload is timed ([timing] lines); then
// step 14's frame loop draws the mesh, --seconds S closes the window. Steps
// 01-14 are repeated, reduced to what this step needs.

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
#include <initializer_list>
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

// #region vertex
// One vertex as it lies in the vertex buffer: 2 + 3 floats = 20 bytes, no
// padding. A standard-layout struct fixes the field order and offsets, which
// the vertex input state of the pipeline repeats (location 0 at offset 0,
// location 1 at offset 8) and shaders/mesh.slang declares.
struct Vertex {
    std::array<float, 2> position; // clip-space x, y (Vulkan: +y points down)
    std::array<float, 3> color;    // linear RGB
};
static_assert(sizeof(Vertex) == 20);
// #endregion

// #region mesh
// A hexagon: the center plus 6 corners = 7 vertices, 6 triangles = 18 indices.
// Corner i sits at 60 degrees * i on an ellipse 0.45 wide and 0.6 tall: the
// window is 800x600, so x is scaled by 600/800 to keep the hexagon regular on
// screen. Angles grow from +x towards +y, which is DOWN on screen, so each
// triangle (center, corner i, corner i+1) is clockwise - the front face of the
// pipeline (back faces are culled).
constexpr std::array<Vertex, 7> vertices{{
    {{0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},           // center, white
    {{0.45f, 0.0f}, {1.0f, 0.0f, 0.0f}},          // 0 deg, red
    {{0.225f, 0.5196152f}, {1.0f, 1.0f, 0.0f}},   // 60 deg, yellow
    {{-0.225f, 0.5196152f}, {0.0f, 1.0f, 0.0f}},  // 120 deg, green
    {{-0.45f, 0.0f}, {0.0f, 1.0f, 1.0f}},         // 180 deg, cyan
    {{-0.225f, -0.5196152f}, {0.0f, 0.0f, 1.0f}}, // 240 deg, blue
    {{0.225f, -0.5196152f}, {1.0f, 0.0f, 1.0f}},  // 300 deg, magenta
}};
// Three indices per triangle; vertex 0 (the center) is used six times but
// stored once. 16-bit indices are enough for up to 65536 vertices.
constexpr std::array<uint16_t, 18> indices{
    0, 1, 2, //
    0, 2, 3, //
    0, 3, 4, //
    0, 4, 5, //
    0, 5, 6, //
    0, 6, 1, //
};
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

// #region find-memory-type
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
// #endregion

// #region create-buffer
// A buffer and the memory bound to it. One allocation per buffer is fine for a
// few buffers; step 16 shows why and how real programs sub-allocate. Members
// are destroyed in reverse order: the buffer first, then its memory.
struct GpuBuffer {
    vk::raii::DeviceMemory memory;
    vk::raii::Buffer buffer;
    uint32_t memoryType;
};

// Step 13's manual path: create the buffer, query its requirements, take the
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
    // An allocation larger than its heap is invalid
    // (VUID-vkAllocateMemory-pAllocateInfo-01713), not just "out of memory":
    // check it here, which matters for --bar --mib N (the BAR heap is small).
    const vk::DeviceSize heapSize = props.memoryHeaps[props.memoryTypes[*memoryType].heapIndex].size;
    if (req.size > heapSize) {
        throw std::runtime_error(
            std::format("{} bytes do not fit memory type {}: its heap has {} bytes", req.size, *memoryType, heapSize));
    }
    vk::raii::DeviceMemory memory(device, vk::MemoryAllocateInfo{
                                              .allocationSize  = req.size,
                                              .memoryTypeIndex = *memoryType,
                                          });
    buffer.bindMemory(*memory, 0);
    return GpuBuffer{.memory = std::move(memory), .buffer = std::move(buffer), .memoryType = *memoryType};
}

// A block of bytes to write at an offset.
struct Block {
    size_t offset;
    std::span<const std::byte> bytes;
};

// Copies byte blocks into a HOST_VISIBLE | HOST_COHERENT allocation, each at its
// offset: map, memcpy, unmap. Coherent memory needs no flush: host writes to it
// are available in the host domain at once, and vkQueueSubmit makes them visible
// to all device accesses of every command submitted after this.
void writeMapped(const vk::raii::DeviceMemory& memory, std::initializer_list<Block> blocks) {
    auto* ptr = static_cast<std::byte*>(memory.mapMemory(0, vk::WholeSize));
    for (const Block& b : blocks) {
        std::memcpy(ptr + b.offset, b.bytes.data(), b.bytes.size());
    }
    memory.unmapMemory();
}// #endregion

// #region frame-resources
// What the CPU touches while recording one frame: one set per frame in flight (step 10).
struct Frame {
    vk::raii::CommandBuffer commandBuffer;
    // Signaled by vkAcquireNextImageKHR, waited on by this slot's submit.
    vk::raii::Semaphore imageAvailable;
    // Signaled when this slot's submission completes.
    vk::raii::Fence inFlight;
};
// #endregion

// #region cli
// How the mesh gets into GPU memory (see the file header).
enum class Upload { GraphicsQueue, TransferQueue, Bar };

// Command line: --transfer-queue or --bar (default: the graphics queue),
// --mib N (pad the vertex buffer to N MiB, 0-1024), --seconds S.
struct Args {
    Upload upload = Upload::GraphicsQueue;
    uint32_t mib  = 0;
    std::optional<double> seconds;
};

template <typename T> T parseValue(std::string_view arg, std::string_view text) {
    T value{};
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || end != text.data() + text.size()) {
        throw std::runtime_error(std::string{arg} + ": invalid value " + std::string{text});
    }
    return value;
}

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value           = [&] {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string{arg} + " needs a value");
            }
            return std::string_view{argv[++i]};
        };
        Upload upload = Upload::GraphicsQueue;
        if (arg == "--transfer-queue") {
            upload = Upload::TransferQueue;
        } else if (arg == "--bar") {
            upload = Upload::Bar;
        } else if (arg == "--mib") {
            args.mib = parseValue<uint32_t>(arg, value());
            continue;
        } else if (arg == "--seconds") {
            args.seconds = parseValue<double>(arg, value());
            continue;
        } else {
            throw std::runtime_error("unknown argument " + std::string{arg} + " (use --transfer-queue | --bar, --mib N, --seconds S)");
        }
        if (args.upload != Upload::GraphicsQueue) {
            throw std::runtime_error("--transfer-queue and --bar exclude each other");
        }
        args.upload = upload;
    }
    if (args.mib > 1024) {
        throw std::runtime_error("--mib must be 0-1024");
    }
    return args;
}
// #endregion

constexpr size_t mib = 1024 * 1024;
using Clock          = std::chrono::steady_clock;
using Seconds        = std::chrono::duration<double>;

// Megabytes per second (10^6 bytes, like disk and PCIe figures) for printing.
double mbPerS(size_t bytes, Seconds time) { return static_cast<double>(bytes) / 1e6 / std::max(time.count(), 1e-9); }

// --bar: no staging, no copy. The buffers live in a DEVICE_LOCAL memory type
// the CPU can also map, and the CPU writes VRAM directly through the PCIe BAR
// window. With Resizable BAR off that is a 256 MiB BAR, of which the driver
// exposes 214 MiB as heap 2; the writes are uncached (typically
// write-combined), so they are fast to write and very slow to read back.
std::pair<GpuBuffer, GpuBuffer> uploadBar(const vk::raii::Device& device, const vk::PhysicalDeviceMemoryProperties& mem,
                                          std::span<const std::byte> vertexData, std::span<const std::byte> indexData) {
    // #region bar-write
    std::println("Upload path: direct CPU writes into DEVICE_LOCAL | HOST_VISIBLE memory (no staging buffer, no copy)");
    using Mem                           = vk::MemoryPropertyFlagBits;
    const vk::MemoryPropertyFlags flags = Mem::eDeviceLocal | Mem::eHostVisible | Mem::eHostCoherent;
    // Both buffers come from the same small heap, and createBuffer checks one
    // allocation at a time: check their sum before allocating anything.
    // vkGetDeviceBufferMemoryRequirements (Vulkan 1.3) gives a buffer's
    // requirements without creating it. The heap size is only an upper bound,
    // not the budget: the driver and other programs use the heap too, and
    // VK_EXT_memory_budget reports how much of it this process may use.
    const auto need = [&](size_t size, vk::BufferUsageFlags usage) {
        const vk::BufferCreateInfo info{.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};
        const vk::MemoryRequirements req = device.getBufferMemoryRequirements(vk::DeviceBufferMemoryRequirements{.pCreateInfo = &info})
                                               .memoryRequirements;
        const auto memoryType = findType(mem, req.memoryTypeBits, flags);
        if (!memoryType) {
            throw std::runtime_error("no memory type with " + memoryNames(flags) + " for this buffer");
        }
        return std::pair{mem.memoryTypes[*memoryType].heapIndex, req.size};
    };
    const auto [vertexHeap, vertexNeed] = need(vertexData.size(), vk::BufferUsageFlagBits::eVertexBuffer);
    const auto [indexHeap, indexNeed]   = need(indexData.size(), vk::BufferUsageFlagBits::eIndexBuffer);
    if (vertexHeap == indexHeap) {
        const vk::DeviceSize heapSize = mem.memoryHeaps[vertexHeap].size;
        const vk::DeviceSize total    = vertexNeed + indexNeed;
        if (total > heapSize) {
            throw std::runtime_error(std::format("vertex + index buffers need {} bytes, but heap {} has only {} bytes (an upper bound, "
                                                 "not the budget: see VK_EXT_memory_budget)",
                                                 total, vertexHeap, heapSize));
        }
    }
    GpuBuffer vertexBuffer = createBuffer(device, mem, vertexData.size(), vk::BufferUsageFlagBits::eVertexBuffer, flags);
    GpuBuffer indexBuffer  = createBuffer(device, mem, indexData.size(), vk::BufferUsageFlagBits::eIndexBuffer, flags);
    const uint32_t heap    = mem.memoryTypes[vertexBuffer.memoryType].heapIndex;
    const auto start       = Clock::now();
    writeMapped(vertexBuffer.memory, {{0, vertexData}});
    writeMapped(indexBuffer.memory, {{0, indexData}});
    const Seconds cpu = Clock::now() - start;
    std::println("Vertex + index buffers: memory type {} ({}), heap {}", vertexBuffer.memoryType, memoryNames(flags), heap);
    const size_t total = vertexData.size() + indexData.size();
    std::println("[timing] upload of {} bytes: CPU writes {:.3f} ms ({:.0f} MB/s) straight into VRAM, no copy", total,
                 cpu.count() * 1e3, mbPerS(total, cpu));
    // #endregion
    return {std::move(vertexBuffer), std::move(indexBuffer)};
}

// A queue family index and one of its queues.
struct FamilyQueue {
    uint32_t family;
    const vk::raii::Queue* queue;
};

// Default and --transfer-queue: CPU -> staging buffer -> vkCmdCopyBuffer ->
// DEVICE_LOCAL buffers. graphics is the family and queue that will draw;
// transfer the dedicated transfer family and its queue, if used.
std::pair<GpuBuffer, GpuBuffer> uploadStaged(const vk::raii::Device& device, const vk::PhysicalDeviceMemoryProperties& mem,
                                             std::span<const std::byte> vertexData, std::span<const std::byte> indexData,
                                             FamilyQueue graphics, std::optional<FamilyQueue> transfer) {
    const FamilyQueue copy = transfer.value_or(graphics);
    if (!transfer) {
        std::println("Upload path: staging buffer + vkCmdCopyBuffer on the graphics queue (family {}), then a pipeline barrier",
                     graphics.family);
    } else {
        std::println("Upload path: staging buffer + vkCmdCopyBuffer on the transfer queue (family {}), then an ownership transfer {} "
                     "-> {} (release, semaphore, acquire)",
                     transfer->family, transfer->family, graphics.family);
    }

    // #region staging-buffer
    // The staging buffer: memory the CPU can map, used only as a copy source.
    // Vertices at offset 0, indices right after them (copy offsets need no
    // particular alignment).
    const vk::DeviceSize vertexSize         = vertexData.size();
    const vk::DeviceSize indexSize          = indexData.size();
    const vk::MemoryPropertyFlags hostFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    GpuBuffer staging = createBuffer(device, mem, vertexSize + indexSize, vk::BufferUsageFlagBits::eTransferSrc, hostFlags);
    const auto cpuStart = Clock::now();
    writeMapped(staging.memory, {{0, vertexData}, {vertexData.size(), indexData}});
    const Seconds cpu = Clock::now() - cpuStart;
    std::println("Staging buffer: {} bytes, TRANSFER_SRC, memory type {} ({}); vertices at offset 0, indices at offset {}",
                 vertexSize + indexSize, staging.memoryType, memoryNames(hostFlags), vertexSize);
    // #endregion

    // #region device-local-buffers
    // The real buffers: DEVICE_LOCAL (VRAM), which the CPU cannot map here, with
    // TRANSFER_DST added to their usage so a copy may write them.
    using Usage                         = vk::BufferUsageFlagBits;
    const vk::MemoryPropertyFlags local = vk::MemoryPropertyFlagBits::eDeviceLocal;
    GpuBuffer vertexBuffer = createBuffer(device, mem, vertexSize, Usage::eVertexBuffer | Usage::eTransferDst, local);
    GpuBuffer indexBuffer  = createBuffer(device, mem, indexSize, Usage::eIndexBuffer | Usage::eTransferDst, local);
    std::println("Vertex buffer: {} bytes, VERTEX_BUFFER | TRANSFER_DST, memory type {} ({})", vertexSize, vertexBuffer.memoryType,
                 memoryNames(local));
    std::println("Index buffer: {} bytes, INDEX_BUFFER | TRANSFER_DST, memory type {} ({})", indexSize, indexBuffer.memoryType,
                 memoryNames(local));
    // #endregion

    // #region record-copy
    // A one-time command buffer from a TRANSIENT pool (a hint: its buffers live
    // briefly) of the family that runs the copy. Two copy regions: the staging
    // bytes [0, vertexSize) to the vertex buffer, the rest to the index buffer.
    // The host writes above need no barrier: the submit makes them visible to
    // the device.
    const auto copyStart = Clock::now();
    const auto newPool   = [&](uint32_t family) {
        return vk::raii::CommandPool(device, vk::CommandPoolCreateInfo{
                                                 .flags            = vk::CommandPoolCreateFlagBits::eTransient,
                                                 .queueFamilyIndex = family,
                                             });
    };
    const auto newCmd = [&](const vk::raii::CommandPool& pool) {
        vk::raii::CommandBuffers buffers(device, vk::CommandBufferAllocateInfo{
                                                     .commandPool        = *pool,
                                                     .level              = vk::CommandBufferLevel::ePrimary,
                                                     .commandBufferCount = 1,
                                                 });
        vk::raii::CommandBuffer cmd = std::move(buffers.front());
        cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        return cmd;
    };
    const vk::raii::CommandPool copyPool = newPool(copy.family);
    const vk::raii::CommandBuffer cmd    = newCmd(copyPool);
    cmd.copyBuffer(*staging.buffer, *vertexBuffer.buffer, vk::BufferCopy{.size = vertexSize});
    cmd.copyBuffer(*staging.buffer, *indexBuffer.buffer, vk::BufferCopy{.srcOffset = vertexSize, .size = indexSize});
    // #endregion

    std::vector<vk::raii::Fence> fences;
    std::optional<vk::raii::CommandPool> acquirePool;
    std::optional<vk::raii::CommandBuffer> acquireCmd;
    std::optional<vk::raii::Semaphore> semaphore;
    // If a submit below throws after an earlier one succeeded, the GPU may still use the
    // objects above: wait for it before RAII destroys them (step 10's guard).
    const IdleGuard uploadIdle{device};
    if (!transfer) {
        // #region copy-barrier
        // Same queue: one global memory barrier orders the copy's writes (COPY
        // stage, TRANSFER_WRITE) before every later read of vertex attributes and
        // indices on this queue, including the draws of command buffers
        // submitted later.
        // The fence waited below does not replace it: a fence signal makes the
        // copy's writes AVAILABLE, not VISIBLE; only a barrier like this (or a
        // semaphore wait, whose second access scope covers all device accesses)
        // makes them visible to vertex input. Without it nothing complains: sync
        // validation is off by default, and a host fence wait closes the batch's
        // hazards for it. The memory model still requires the barrier: without
        // it the writes are never made visible to vertex input, and the draws
        // may read stale data.
        const vk::MemoryBarrier2 barrier{
            .srcStageMask  = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask  = vk::PipelineStageFlagBits2::eVertexAttributeInput | vk::PipelineStageFlagBits2::eIndexInput,
            .dstAccessMask = vk::AccessFlagBits2::eVertexAttributeRead | vk::AccessFlagBits2::eIndexRead,
        };
        const vk::raii::Fence& fence = fences.emplace_back(device, vk::FenceCreateInfo{});
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
        cmd.end();
        const vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *cmd};
        graphics.queue->submit2(vk::SubmitInfo2{.commandBufferInfoCount = 1, .pCommandBufferInfos = &cmdInfo}, *fence);
        // #endregion
    } else {
        // #region release-ownership
        // The buffers are EXCLUSIVE: in a well-defined state for one queue family
        // at a time. The transfer family RELEASES them to the graphics family: a
        // buffer barrier with src/dst family indices, after the copy's writes
        // (its dst stage and access are ignored). The semaphore orders release
        // before acquire; its signal and wait must use ALL_COMMANDS unless the
        // barriers set VK_DEPENDENCY_QUEUE_FAMILY_OWNERSHIP_TRANSFER_USE_ALL_STAGES_BIT_KHR
        // (allowed only with the maintenance8 feature) (spec, "Queue Family
        // Ownership Transfer").
        // The semaphore wait alone already makes the copy's writes visible (its
        // second access scope covers all device accesses), yet the
        // release/acquire pair is still required: "If memory dependencies are
        // correctly expressed between uses of such a resource between two queues
        // in different families, but no ownership transfer is defined, the
        // contents of that resource are undefined for any read accesses
        // performed by the second queue family" (spec, "Queue Family Ownership
        // Transfer").
        const auto ownership = [&](const vk::raii::Buffer& buffer) {
            return vk::BufferMemoryBarrier2{
                .srcQueueFamilyIndex = transfer->family,
                .dstQueueFamilyIndex = graphics.family,
                .buffer              = *buffer,
                .offset              = 0,
                .size                = vk::WholeSize,
            };
        };
        std::array release{ownership(vertexBuffer.buffer), ownership(indexBuffer.buffer)};
        for (vk::BufferMemoryBarrier2& b : release) {
            b.srcStageMask  = vk::PipelineStageFlagBits2::eCopy;
            b.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        }
        semaphore.emplace(device, vk::SemaphoreCreateInfo{});
        const vk::raii::Fence& fence = fences.emplace_back(device, vk::FenceCreateInfo{});
        cmd.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = static_cast<uint32_t>(release.size()),
            .pBufferMemoryBarriers    = release.data(),
        });
        cmd.end();
        const vk::SemaphoreSubmitInfo signal{.semaphore = **semaphore, .stageMask = vk::PipelineStageFlagBits2::eAllCommands};
        const vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *cmd};
        transfer->queue->submit2(vk::SubmitInfo2{
                                     .commandBufferInfoCount   = 1,
                                     .pCommandBufferInfos      = &cmdInfo,
                                     .signalSemaphoreInfoCount = 1,
                                     .pSignalSemaphoreInfos    = &signal,
                                 },
                                 *fence);
        // #endregion

        // #region acquire-ownership
        // The graphics family ACQUIRES the buffers with matching barriers (same
        // buffers, ranges and family indices) in a command buffer of its own,
        // submitted after waiting on the semaphore. Its src stage and access are
        // ignored; dst = the vertex input reads that follow.
        std::array acquire{ownership(vertexBuffer.buffer), ownership(indexBuffer.buffer)};
        acquire[0].dstStageMask  = vk::PipelineStageFlagBits2::eVertexAttributeInput;
        acquire[0].dstAccessMask = vk::AccessFlagBits2::eVertexAttributeRead;
        acquire[1].dstStageMask  = vk::PipelineStageFlagBits2::eIndexInput;
        acquire[1].dstAccessMask = vk::AccessFlagBits2::eIndexRead;
        acquirePool.emplace(newPool(graphics.family));
        acquireCmd.emplace(newCmd(*acquirePool));
        const vk::raii::Fence& acquireFence = fences.emplace_back(device, vk::FenceCreateInfo{});
        acquireCmd->pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = static_cast<uint32_t>(acquire.size()),
            .pBufferMemoryBarriers    = acquire.data(),
        });
        acquireCmd->end();
        const vk::SemaphoreSubmitInfo wait{.semaphore = **semaphore, .stageMask = vk::PipelineStageFlagBits2::eAllCommands};
        const vk::CommandBufferSubmitInfo acquireInfo{.commandBuffer = **acquireCmd};
        graphics.queue->submit2(vk::SubmitInfo2{
                                    .waitSemaphoreInfoCount = 1,
                                    .pWaitSemaphoreInfos    = &wait,
                                    .commandBufferInfoCount = 1,
                                    .pCommandBufferInfos    = &acquireInfo,
                                },
                                *acquireFence);
        // #endregion
    }

    // #region finish-upload
    // Wait for the copy (and the acquire) to complete; only then may the staging
    // buffer, the semaphore and the pools go. A real engine would not block
    // here: it keeps a staging ring and checks a fence or a timeline value later
    // (step 12), and the frame that first draws the mesh waits on the upload's
    // semaphore instead.
    // The fences say the work is complete, so the objects may be destroyed;
    // they make nothing visible to the draws: that is the job of the barrier
    // (or the acquire) above.
    std::vector<vk::Fence> handles;
    for (const vk::raii::Fence& fence : fences) {
        handles.push_back(*fence);
    }
    if (device.waitForFences(handles, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
        throw std::runtime_error("waitForFences failed");
    }
    const Seconds copyTime = Clock::now() - copyStart;
    // RAII frees the rest at the end of this function (fences, semaphore, pools
    // with their command buffers); the staging buffer goes now.
    staging.buffer.clear();
    staging.memory.clear();
    std::println("Upload complete (fences waited); staging buffer destroyed");
    const size_t total = vertexData.size() + indexData.size();
    std::println("[timing] upload of {} bytes: CPU writes {:.3f} ms ({:.0f} MB/s), GPU copy {:.3f} ms ({:.0f} MB/s; record + submit + wait)",
                 total, cpu.count() * 1e3, mbPerS(total, cpu), copyTime.count() * 1e3, mbPerS(total, copyTime));
    // #endregion
    return {std::move(vertexBuffer), std::move(indexBuffer)};
}
int main(int argc, char** argv)
try {
    const Args args = parseArgs(argc, argv);

    // ---- Step 06: window (800x600 physical pixels, fixed size) ----
    Glfw glfw;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 15 staging", nullptr, nullptr), &glfwDestroyWindow);
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
        .pApplicationName   = "Vulkan from Zero - 15 staging",
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
            "No Vulkan 1.4 GPU with dynamicRendering + synchronization2 + VK_KHR_swapchain can present to this window");
    }
    const vk::raii::PhysicalDevice& physicalDevice = *best;
    std::println("\nPicked: {}", physicalDevice.getProperties().deviceName.data());
    std::println("Graphics + present family: {}", family);

    // #region transfer-family
    // Step 04's dedicated transfer family: TRANSFER without GRAPHICS and COMPUTE,
    // the front end of the GPU's copy (DMA) engines. A copy on its queue can run
    // while the graphics queue renders.
    const std::vector<vk::QueueFamilyProperties> families = physicalDevice.getQueueFamilyProperties();
    std::optional<uint32_t> dedicated;
    for (uint32_t i = 0; i < families.size(); ++i) {
        const vk::QueueFlags flags = families[i].queueFlags;
        if ((flags & vk::QueueFlagBits::eTransfer) && !(flags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute))) {
            dedicated = i;
            break;
        }
    }
    if (dedicated) {
        std::println("Dedicated transfer family: {}", *dedicated);
    } else {
        std::println("No dedicated transfer family");
    }
    std::optional<uint32_t> transferFamily;
    if (args.upload == Upload::TransferQueue) {
        if (!dedicated) {
            throw std::runtime_error("--transfer-queue: this GPU has no dedicated transfer family");
        }
        transferFamily = dedicated;
    }
    // #endregion

    // Step 07's device, plus one queue of the transfer family when the upload
    // uses it: queues exist only if requested at device creation.
    constexpr float queuePriority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queueInfos{{
        .queueFamilyIndex = family,
        .queueCount       = 1,
        .pQueuePriorities = &queuePriority,
    }};
    if (transferFamily) {
        queueInfos.push_back({.queueFamilyIndex = *transferFamily, .queueCount = 1, .pQueuePriorities = &queuePriority});
    }
    vk::PhysicalDeviceVulkan13Features features13{
        .synchronization2 = vk::True,
        .dynamicRendering = vk::True,
    };
    constexpr std::array deviceExtensions{vk::KHRSwapchainExtensionName};
    vk::raii::Device device(physicalDevice, vk::DeviceCreateInfo{
                                                .pNext                   = &features13,
                                                .queueCreateInfoCount    = static_cast<uint32_t>(queueInfos.size()),
                                                .pQueueCreateInfos       = queueInfos.data(),
                                                .enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size()),
                                                .ppEnabledExtensionNames = deviceExtensions.data(),
                                            });
    std::println("VkDevice created with {} and {} queue(s)", vk::KHRSwapchainExtensionName, queueInfos.size());

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

    // The SPIR-V is compiled from shaders/mesh.slang by `bun run sync:shaders`;
    // CMake bakes the absolute path of shaders/spv into SHADER_DIR.
    const std::vector<uint32_t> spirv = readSpirv(std::filesystem::path{SHADER_DIR} / "mesh.spv");
    std::optional<vk::raii::ShaderModule> shaderModule;
    shaderModule.emplace(device, vk::ShaderModuleCreateInfo{
                                     .codeSize = spirv.size() * sizeof(uint32_t),
                                     .pCode    = spirv.data(),
                                 });
    std::println("\nSPIR-V: {} bytes (mesh.spv), entry points vertexMain, fragmentMain", spirv.size() * sizeof(uint32_t));
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = **shaderModule, .pName = "vertexMain"},
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = **shaderModule, .pName = "fragmentMain"},
    };

    // #region vertex-input
    // How the input assembler turns buffer bytes into shader inputs. A BINDING
    // is a vertex buffer slot: here slot 0, advancing stride bytes per vertex
    // (eVertex rate; eInstance would advance per instance). An ATTRIBUTE is one
    // shader input: its location (the vk::location in mesh.slang), the binding
    // it reads from, the format of the data and its offset inside one vertex.
    constexpr uint32_t stride = sizeof(Vertex);
    const vk::VertexInputBindingDescription binding{
        .binding   = 0,
        .stride    = stride,
        .inputRate = vk::VertexInputRate::eVertex,
    };
    const std::array attributes{
        vk::VertexInputAttributeDescription{
            .location = 0,
            .binding  = 0,
            .format   = vk::Format::eR32G32Sfloat,
            .offset   = offsetof(Vertex, position),
        },
        vk::VertexInputAttributeDescription{
            .location = 1,
            .binding  = 0,
            .format   = vk::Format::eR32G32B32Sfloat,
            .offset   = offsetof(Vertex, color),
        },
    };
    const vk::PipelineVertexInputStateCreateInfo vertexInput{
        .vertexBindingDescriptionCount   = 1,
        .pVertexBindingDescriptions      = &binding,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size()),
        .pVertexAttributeDescriptions    = attributes.data(),
    };
    std::println("Vertex input: binding 0, stride {} bytes, per vertex", stride);
    for (const vk::VertexInputAttributeDescription& a : attributes) {
        std::println("  location {}: {} at offset {}", a.location, formatName(a.format), a.offset);
    }
    // #endregion

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
    vk::raii::PipelineLayout pipelineLayout(device, vk::PipelineLayoutCreateInfo{});
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
    std::println("Graphics pipeline created (vertex input from binding 0)");

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
    std::println("{} frames in flight", framesInFlight);

    // #region mesh-bytes
    // The bytes to upload: the vertices, padded with zeros to --mib MiB (only the
    // 7 real vertices are drawn; the padding makes the timing measure
    // bandwidth), and the indices.
    const std::span<const std::byte> vertexBytes = std::as_bytes(std::span{vertices});
    std::vector<std::byte> vertexData(vertexBytes.begin(), vertexBytes.end());
    vertexData.resize(std::max(vertexData.size(), size_t{args.mib} * mib));
    const std::span<const std::byte> indexData = std::as_bytes(std::span{indices});
    std::println("\nMesh: hexagon, {} vertices + {} indices; vertex buffer {} bytes{}, index buffer {} bytes", vertices.size(),
                 indices.size(), vertexData.size(), args.mib > 0 ? " (padded with --mib)" : "", indexData.size());
    const vk::PhysicalDeviceMemoryProperties mem = physicalDevice.getMemoryProperties();
    // #endregion

    std::optional<vk::raii::Queue> transferQueue;
    if (transferFamily) {
        transferQueue.emplace(device, *transferFamily, 0);
    }
    auto [vertexBuffer, indexBuffer] =
        args.upload == Upload::Bar
            ? uploadBar(device, mem, vertexData, indexData)
            : uploadStaged(device, mem, vertexData, indexData, FamilyQueue{family, &queue},
                           transferQueue ? std::optional{FamilyQueue{*transferFamily, &*transferQueue}} : std::nullopt);
    const auto start  = Clock::now();
    size_t frameIndex = 0;

    // Renders and presents one frame (step 10's loop). Returns false when the
    // swapchain is out of date (recreation is step 11).
    const auto drawFrame = [&]() -> bool {
        Frame& frame                       = frames.at(frameIndex % frames.size());
        const vk::raii::CommandBuffer& cmd = frame.commandBuffer;

        // #region wait-fence
        // The last submission that used this slot must be finished before its
        // command buffer and semaphore are reused (step 10).
        if (device.waitForFences(*frame.inFlight, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
            throw std::runtime_error("waitForFences failed");
        }
        // #endregion

        // #region acquire
        const vk::ResultValue<uint32_t> acquired =
            swapchain.acquireNextImage(std::numeric_limits<uint64_t>::max(), *frame.imageAvailable, nullptr);
        if (acquired.result == vk::Result::eErrorOutOfDateKHR) {
            return false;
        }
        const uint32_t imageIndex = acquired.value;
        const vk::Image image     = images.at(imageIndex);
        device.resetFences(*frame.inFlight);
        // #endregion

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
        // Vertex buffer slot 0 = the buffer from offset 0 (the binding of the
        // vertex input state); the index buffer with its index type. Then
        // vkCmdDrawIndexed: 18 indices, 1 instance, first index 0, vertex offset 0
        // (added to every index), first instance 0. The input assembler reads
        // index k, fetches that vertex, and the vertex shader runs for it (a
        // post-transform cache may reuse the result of a vertex that repeats,
        // like the center).
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

        // #region submit
        // Step 10's submit: wait imageAvailable at COLOR_ATTACHMENT_OUTPUT (vertex
        // fetch and vertex shading may start before the image is ours), signal
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
        // #endregion

        // #region present
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
        // #endregion

        ++frameIndex;
        return true;
    };

    // #region event-loop
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
    // #endregion

    // #region cleanup
    // Wait until the GPU is idle (step 10): the last frames may still read the
    // vertex and index buffers. Then RAII destroys in reverse order of
    // declaration: the per-frame objects, the pool, the mesh buffers (buffer
    // first, then its memory), and steps 07-08.
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
