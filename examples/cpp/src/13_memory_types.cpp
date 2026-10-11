// Step 13 - Memory heaps and memory types
//
// A buffer is only a description (size, usage); the bytes live in a
// VkDeviceMemory allocation that we bind to it. Where those bytes can live is
// what the device reports in VkPhysicalDeviceMemoryProperties:
//
//   heaps = physical pools of memory (VRAM, system RAM, the CPU-visible window of VRAM)
//   types = ways to use a heap (DEVICE_LOCAL, HOST_VISIBLE, HOST_COHERENT, HOST_CACHED)
//
// This step prints both tables, asks the driver what a few kinds of buffers
// need (vkGetBufferMemoryRequirements: size, alignment, allowed types), chooses
// memory types with the spec's find-memory-type rule, and finally allocates,
// binds and maps a HOST_VISIBLE buffer and places two buffers in one
// allocation. No window: steps 01-05 are repeated, reduced to what this step needs.

#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
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

// Flag names as the spec spells them without the VK_..._BIT decoration, in bit
// order; both languages print them from tables like this one, so the output is
// identical. Unknown bits are printed in hex.
using NameTable = std::span<const std::pair<uint32_t, std::string_view>>;
constexpr std::array<std::pair<uint32_t, std::string_view>, 9> memoryPropertyNames{{
    {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "DEVICE_LOCAL"},
    {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, "HOST_VISIBLE"},
    {VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "HOST_COHERENT"},
    {VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "HOST_CACHED"},
    {VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, "LAZILY_ALLOCATED"},
    {VK_MEMORY_PROPERTY_PROTECTED_BIT, "PROTECTED"},
    {VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD, "DEVICE_COHERENT_AMD"},
    {VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD, "DEVICE_UNCACHED_AMD"},
    {VK_MEMORY_PROPERTY_RDMA_CAPABLE_BIT_NV, "RDMA_CAPABLE_NV"},
}};
constexpr std::array<std::pair<uint32_t, std::string_view>, 2> heapNames{{
    {VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, "DEVICE_LOCAL"},
    {VK_MEMORY_HEAP_MULTI_INSTANCE_BIT, "MULTI_INSTANCE"},
}};
constexpr std::array<std::pair<uint32_t, std::string_view>, 8> usageNames{{
    {VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "TRANSFER_SRC"},
    {VK_BUFFER_USAGE_TRANSFER_DST_BIT, "TRANSFER_DST"},
    {VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT, "UNIFORM_TEXEL_BUFFER"},
    {VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT, "STORAGE_TEXEL_BUFFER"},
    {VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, "UNIFORM_BUFFER"},
    {VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "STORAGE_BUFFER"},
    {VK_BUFFER_USAGE_INDEX_BUFFER_BIT, "INDEX_BUFFER"},
    {VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, "VERTEX_BUFFER"},
}};

// "A | B | C" for the set bits of raw, "(none)" for zero.
std::string names(uint32_t raw, NameTable table) {
    std::string out;
    uint32_t rest = raw;
    for (const auto& [bit, name] : table) {
        if ((raw & bit) != 0) {
            out += out.empty() ? "" : " | ";
            out += name;
            rest &= ~bit;
        }
    }
    if (rest != 0) {
        out += out.empty() ? "" : " | ";
        out += std::format("{:#x}", rest);
    }
    return out.empty() ? "(none)" : out;
}
std::string memoryNames(vk::MemoryPropertyFlags f) { return names(static_cast<uint32_t>(f), memoryPropertyNames); }
std::string heapFlagNames(vk::MemoryHeapFlags f) { return names(static_cast<uint32_t>(f), heapNames); }
std::string usageFlagNames(vk::BufferUsageFlags f) { return names(static_cast<uint32_t>(f), usageNames); }

// "0, 1, 3" for the set bits of a memoryTypeBits mask.
std::string typeList(uint32_t bits) {
    std::string out;
    for (uint32_t i = 0; i < 32; ++i) {
        if ((bits & (1u << i)) != 0) {
            out += out.empty() ? "" : ", ";
            out += std::to_string(i);
        }
    }
    return out;
}

// Bytes as MiB with one decimal (exact: sizes are integers, 2^20 is a power of two).
std::string mib(vk::DeviceSize bytes) { return std::format("{:.1f} MiB", static_cast<double>(bytes) / (1024.0 * 1024.0)); }

// #region find-memory-type
// The spec's find-memory-type rule (chapter "Device Memory", sample code
// findProperties): the FIRST memory type, in index order, that
// 1. the resource allows: bit i of its memoryTypeBits is set, and
// 2. has every wanted property flag.
// Drivers must order the types so that this simple loop finds the plainest
// fitting type: a type whose flags are a strict subset of another's comes
// first. The interactive labs of this course use the same rule.
std::optional<uint32_t> findType(const vk::PhysicalDeviceMemoryProperties& props, uint32_t bits,
                                 vk::MemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) != 0 && (props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return std::nullopt;
}

// Required flags must be present; preferred ones are nice to have. Search with
// both first, then with the required ones alone (the same two-pass search as
// the spec sample). Returns the type and whether the preferred flags were met.
std::optional<std::pair<uint32_t, bool>> chooseType(const vk::PhysicalDeviceMemoryProperties& props, uint32_t bits,
                                                    vk::MemoryPropertyFlags required, vk::MemoryPropertyFlags preferred) {
    if (const auto i = findType(props, bits, required | preferred)) {
        return std::pair{*i, true};
    }
    if (const auto i = findType(props, bits, required)) {
        return std::pair{*i, false};
    }
    return std::nullopt;
}
// #endregion

// Rounds offset up to a multiple of alignment (a power of two in Vulkan).
vk::DeviceSize alignUp(vk::DeviceSize offset, vk::DeviceSize alignment) {
    return (offset + alignment - 1) / alignment * alignment;
}

int main()
try {
    // ---- Steps 01-02: loader, instance with validation, debug messenger ----
    vk::raii::Context context;

    constexpr const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    const std::vector<vk::LayerProperties> layers = context.enumerateInstanceLayerProperties();
    const bool hasValidation = std::ranges::any_of(
        layers, [&](const vk::LayerProperties& l) { return std::string_view{l.layerName.data()} == validationLayer; });
    if (!hasValidation) {
        throw std::runtime_error(std::string{validationLayer} + " not found - install the Vulkan SDK");
    }

    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName   = "Vulkan from Zero - 13 memory types",
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
    constexpr std::array layerNames{validationLayer};
    constexpr std::array extensionNames{vk::EXTDebugUtilsExtensionName};
    const vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> createChain{
        vk::InstanceCreateInfo{
            .pApplicationInfo        = &appInfo,
            .enabledLayerCount       = static_cast<uint32_t>(layerNames.size()),
            .ppEnabledLayerNames     = layerNames.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(extensionNames.size()),
            .ppEnabledExtensionNames = extensionNames.data(),
        },
        messengerInfo,
    };
    vk::raii::Instance instance(context, createChain.get<vk::InstanceCreateInfo>());
    vk::raii::DebugUtilsMessengerEXT messenger(instance, messengerInfo);

    const std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }
    std::println("Found {} GPU(s)", devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        std::println("GPU {}: {}", i, devices[i].getProperties().deviceName.data());
    }

    // Step 05's choice: a Vulkan 1.4 GPU with dynamicRendering + synchronization2, discrete first.
    const vk::raii::PhysicalDevice* best = nullptr;
    int bestScore                        = 0;
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
        }
    }
    if (best == nullptr) {
        throw std::runtime_error("No GPU supports Vulkan 1.4 with dynamicRendering + synchronization2");
    }
    const vk::raii::PhysicalDevice& physicalDevice = *best;
    const vk::PhysicalDeviceProperties chosen      = physicalDevice.getProperties();
    std::println("\nPicked: {}", chosen.deviceName.data());
    const std::vector<vk::QueueFamilyProperties> families = physicalDevice.getQueueFamilyProperties();
    const auto graphics = std::ranges::find_if(
        families, [](const vk::QueueFamilyProperties& f) { return static_cast<bool>(f.queueFlags & vk::QueueFlagBits::eGraphics); });
    if (graphics == families.end()) {
        throw std::runtime_error("No graphics queue family");
    }
    const auto family = static_cast<uint32_t>(graphics - families.begin());
    std::println("Graphics family: {}", family);

    // Step 05's device. Buffers and memory are created from it; this step
    // submits no work, but every device needs at least one queue.
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
    vk::raii::Device device(physicalDevice, vk::DeviceCreateInfo{
                                                .pNext                = &features13,
                                                .queueCreateInfoCount = 1,
                                                .pQueueCreateInfos    = &queueInfo,
                                            });
    std::println("VkDevice created");

    // #region memory-properties
    // Heaps are the physical pools; types say how a heap can be used. Each type
    // points at one heap, and several types may share a heap (system RAM
    // appears as plain, HOST_VISIBLE|HOST_COHERENT and +HOST_CACHED here).
    // DEVICE_LOCAL = fastest for the GPU (VRAM on a discrete card);
    // HOST_VISIBLE = the CPU can map it; HOST_COHERENT = CPU writes need no
    // explicit flush; HOST_CACHED = CPU reads go through the CPU cache.
    const vk::PhysicalDeviceMemoryProperties mem = physicalDevice.getMemoryProperties();
    std::println("\nMemory heaps: {}", mem.memoryHeapCount);
    for (uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
        const vk::MemoryHeap& heap = mem.memoryHeaps[i];
        std::println("  heap {}: {} bytes = {}, {}", i, heap.size, mib(heap.size), heapFlagNames(heap.flags));
    }
    std::println("Memory types: {}", mem.memoryTypeCount);
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        const vk::MemoryType& t = mem.memoryTypes[i];
        std::println("  type {}: heap {}, {}", i, t.heapIndex, memoryNames(t.propertyFlags));
    }
    // #endregion

    // #region memory-limits
    // Limits that matter for memory: how many vkAllocateMemory allocations may
    // exist at once (the spec guarantees at least 4096, which is why real
    // programs sub-allocate, step 16), the alignment of a mapped pointer, the
    // granularity of flushes for non-coherent memory, and how far apart linear
    // (buffers) and optimal-tiling (images) resources must be when they share
    // an allocation.
    const vk::PhysicalDeviceLimits& limits = chosen.limits;
    std::println("\nLimits: maxMemoryAllocationCount {}, minMemoryMapAlignment {}, nonCoherentAtomSize {}, bufferImageGranularity {}",
                 limits.maxMemoryAllocationCount, limits.minMemoryMapAlignment, limits.nonCoherentAtomSize,
                 limits.bufferImageGranularity);
    // #endregion

    // #region requirements
    // A buffer is created WITHOUT memory. Its usage flags tell the driver what
    // it will be used for; the driver answers with what the memory must look
    // like: the size to allocate (may be rounded up), the alignment of the
    // offset it is bound at, and memoryTypeBits - bit i set = type i allowed.
    using Usage = vk::BufferUsageFlagBits;
    struct Sample {
        std::string_view label;
        vk::BufferUsageFlags usage;
        vk::DeviceSize size;
    };
    const std::array samples{
        Sample{"vertex", Usage::eVertexBuffer, 1000},
        Sample{"index", Usage::eIndexBuffer, 100},
        Sample{"uniform", Usage::eUniformBuffer, 100},
        Sample{"storage", Usage::eStorageBuffer, 1000},
        Sample{"staging", Usage::eTransferSrc, 65536},
        Sample{"vertex (copy target)", Usage::eVertexBuffer | Usage::eTransferDst, 65536},
    };
    std::println("\nBuffer requirements (vkGetBufferMemoryRequirements):");
    std::vector<vk::MemoryRequirements> requirements;
    for (const Sample& s : samples) {
        // RAII: the buffer is destroyed at the end of each iteration.
        const vk::raii::Buffer buffer(device, vk::BufferCreateInfo{
                                                  .size        = s.size,
                                                  .usage       = s.usage,
                                                  .sharingMode = vk::SharingMode::eExclusive,
                                              });
        const vk::MemoryRequirements req = buffer.getMemoryRequirements();
        std::println("  {:<21} {}, {} bytes -> size {}, alignment {}, memoryTypeBits {:#b} (types {})", s.label,
                     usageFlagNames(s.usage), s.size, req.size, req.alignment, req.memoryTypeBits, typeList(req.memoryTypeBits));
        requirements.push_back(req);
    }
    // #endregion

    // #region choose-types
    // What each kind of data wants, as (required, preferred) flags, matched
    // against the memoryTypeBits of the vertex buffer above:
    // - GPU-only data (meshes uploaded once, step 15): DEVICE_LOCAL;
    // - a staging buffer the CPU fills for a copy: HOST_VISIBLE + HOST_COHERENT;
    // - results the CPU reads back: HOST_VISIBLE, preferably HOST_CACHED
    //   (uncached reads from the CPU are very slow);
    // - small data the CPU rewrites every frame: HOST_VISIBLE + HOST_COHERENT,
    //   preferably also DEVICE_LOCAL (the CPU-visible window of VRAM);
    // - transient attachments: DEVICE_LOCAL, preferably LAZILY_ALLOCATED
    //   (tile-based mobile GPUs; desktop GPUs have no such type, so the required
    //   flags alone decide).
    using Mem               = vk::MemoryPropertyFlagBits;
    const uint32_t bits     = requirements.at(0).memoryTypeBits;
    struct Intent {
        std::string_view label;
        vk::MemoryPropertyFlags required;
        vk::MemoryPropertyFlags preferred;
    };
    const std::array intents{
        Intent{"GPU-only mesh", Mem::eDeviceLocal, {}},
        Intent{"staging upload", Mem::eHostVisible | Mem::eHostCoherent, {}},
        Intent{"readback", Mem::eHostVisible, Mem::eHostCached},
        Intent{"per-frame data", Mem::eHostVisible | Mem::eHostCoherent, Mem::eDeviceLocal},
        Intent{"transient attachment", Mem::eDeviceLocal, Mem::eLazilyAllocated},
    };
    std::println("\nChoosing memory types for memoryTypeBits {:#b} (first type in index order with the flags; preferred flags tried first):",
                 bits);
    for (const Intent& intent : intents) {
        std::string result = "no such type";
        if (const auto chosenType = chooseType(mem, bits, intent.required, intent.preferred)) {
            const auto [i, met]   = *chosenType;
            const std::string_view note = met || !intent.preferred ? "" : ", preferred flags not available";
            result = std::format("type {} (heap {}){}", i, mem.memoryTypes[i].heapIndex, note);
        }
        std::println("  {:<21} required {}, preferred {} -> {}", intent.label, memoryNames(intent.required),
                     memoryNames(intent.preferred), result);
    }
    // #endregion

    // #region allocate-and-bind
    // The full manual path for one buffer the CPU writes: create the buffer, ask
    // for its requirements, choose a type, allocate that much memory of that
    // type, bind the buffer at offset 0.
    constexpr vk::DeviceSize uploadSize = 4096;
    vk::raii::Buffer upload(device, vk::BufferCreateInfo{
                                        .size        = uploadSize,
                                        .usage       = Usage::eTransferSrc,
                                        .sharingMode = vk::SharingMode::eExclusive,
                                    });
    const vk::MemoryRequirements uploadReq = upload.getMemoryRequirements();
    const auto uploadType                  = findType(mem, uploadReq.memoryTypeBits, Mem::eHostVisible | Mem::eHostCoherent);
    if (!uploadType) {
        throw std::runtime_error("no HOST_VISIBLE | HOST_COHERENT memory type for the buffer");
    }
    vk::raii::DeviceMemory uploadMemory(device, vk::MemoryAllocateInfo{
                                                    .allocationSize  = uploadReq.size,
                                                    .memoryTypeIndex = *uploadType,
                                                });
    upload.bindMemory(*uploadMemory, 0);
    std::println("\nUpload buffer: {} bytes (TRANSFER_SRC), memory type {}, allocation {} bytes, bound at offset 0", uploadSize,
                 *uploadType, uploadReq.size);
    // #endregion

    // #region map
    // Mapping gives the CPU a pointer into the allocation. The memory is
    // HOST_COHERENT, so plain writes are enough: no vkFlushMappedMemoryRanges (a
    // non-coherent type needs a flush of ranges aligned to nonCoherentAtomSize
    // after writing). Reading back from this type works but is slow: it is
    // uncached, write-combined memory.
    void* ptr          = uploadMemory.mapMemory(0, vk::WholeSize);
    const bool aligned = reinterpret_cast<uintptr_t>(ptr) % limits.minMemoryMapAlignment == 0;
    std::println("Mapped: pointer aligned to minMemoryMapAlignment ({}): {}", limits.minMemoryMapAlignment, aligned ? "yes" : "no");
    const size_t count = uploadSize / sizeof(float);
    const std::span<float> floats(static_cast<float*>(ptr), count);
    for (size_t i = 0; i < count; ++i) {
        floats[i] = static_cast<float>(i);
    }
    double sum = 0.0;
    for (const float f : floats) {
        sum += f;
    }
    const auto expected = static_cast<double>(count * (count - 1) / 2);
    std::println("Wrote {} floats through the pointer, read back: sum {} (expected {})", count, sum, expected);
    uploadMemory.unmapMemory();
    std::println("Unmapped (HOST_COHERENT: no flush was needed)");
    // #endregion

    // #region two-buffers-one-allocation
    // Nothing forces one allocation per buffer: a buffer is bound at an OFFSET.
    // Two buffers can share an allocation of a type both allow, each at an
    // offset that is a multiple of its own alignment. This is what an allocator
    // does for thousands of buffers (step 16).
    vk::raii::Buffer vertex(device, vk::BufferCreateInfo{.size = 1000, .usage = Usage::eVertexBuffer});
    vk::raii::Buffer index(device, vk::BufferCreateInfo{.size = 100, .usage = Usage::eIndexBuffer});
    const vk::MemoryRequirements vertexReq = vertex.getMemoryRequirements();
    const vk::MemoryRequirements indexReq  = index.getMemoryRequirements();
    // One type for both: it must be allowed by both masks (AND them).
    const uint32_t sharedBits = vertexReq.memoryTypeBits & indexReq.memoryTypeBits;
    const auto sharedType     = findType(mem, sharedBits, Mem::eDeviceLocal);
    if (!sharedType) {
        throw std::runtime_error("no DEVICE_LOCAL type for both buffers");
    }
    const vk::DeviceSize indexOffset = alignUp(vertexReq.size, indexReq.alignment);
    const vk::DeviceSize sharedSize  = indexOffset + indexReq.size;
    vk::raii::DeviceMemory sharedMemory(device, vk::MemoryAllocateInfo{
                                                    .allocationSize  = sharedSize,
                                                    .memoryTypeIndex = *sharedType,
                                                });
    vertex.bindMemory(*sharedMemory, 0);
    index.bindMemory(*sharedMemory, indexOffset);
    std::println("\nOne allocation, two buffers: memory type {}, {} bytes; vertex buffer ({} bytes) at offset 0, index buffer ({} "
                 "bytes) at offset {} (alignment {})",
                 *sharedType, sharedSize, vertexReq.size, indexReq.size, indexOffset, indexReq.alignment);
    // #endregion

    // #region cleanup
    // Buffers first, then the memory they are bound to (freeing memory that a
    // live buffer is bound to is allowed, but such a buffer may not be used any
    // more; this order keeps it simple). vkFreeMemory also unmaps. RAII does it
    // in reverse order of declaration; we spell it out to match the Rust version.
    vertex.clear();
    index.clear();
    sharedMemory.clear();
    upload.clear();
    uploadMemory.clear();
    std::println("\nBuffers destroyed, memory freed");
    // #endregion
    return EXIT_SUCCESS;
} catch (const vk::SystemError& e) {
    std::println(stderr, "Vulkan error: {}", e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return EXIT_FAILURE;
}
