// Step 16 - Sub-allocation
//
// Steps 13-15 gave every buffer its own vkAllocateMemory. That does not scale:
// each allocation is a kernel-mode call with its own bookkeeping (page tables,
// residency), the number of live allocations is limited by
// maxMemoryAllocationCount (the spec guarantees only 4096), and small
// allocations waste memory to the driver's own granularity. Real programs
// allocate a few BIG blocks and place many resources inside them at aligned
// offsets (step 13 bound two buffers into one allocation by hand) - like a
// texture atlas, but for memory.
//
//   raw:       N buffers -> N vkAllocateMemory
//   allocator: N buffers -> a few blocks, each buffer bound at (block, offset)
//
// This step creates the same N buffers both ways and compares the counts and
// the time. The allocator here is VMA (Vulkan Memory Allocator, AMD GPUOpen,
// vcpkg port vulkan-memory-allocator 3.4.0), the de facto standard in C++. The
// Rust version contains its own minimal linear allocator instead (see its
// header), so lines whose content depends on the allocator library start with
// [lib], timings with [timing]; all other lines are identical in both
// languages. No window: steps 01-05 are repeated, reduced to what this step needs.

#include <vulkan/vulkan_raii.hpp>

// VMA is a single header; exactly one .cpp of a program defines
// VMA_IMPLEMENTATION before including it. VMA gets its Vulkan functions through
// the two vkGet*ProcAddr pointers we pass below (dynamic), not from the
// loader's import library (static).
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h> // vcpkg's include dir is .../include/vma (the SDK's older copy is reached as <vma/vk_mem_alloc.h>)

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
#include <optional>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
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

// #region cli
// Command line: --count N, the number of buffers (default 3000, below the 4096
// the spec guarantees, so the raw path works on every device). The buffers must
// also fit their memory heap; that is checked once the memory type is known.
uint32_t parseCount(int argc, char** argv) {
    uint32_t count = 3000;
    if (argc == 3 && std::string_view{argv[1]} == "--count") {
        const std::string_view text = argv[2];
        const auto [end, ec]        = std::from_chars(text.data(), text.data() + text.size(), count);
        if (ec != std::errc{} || end != text.data() + text.size()) {
            throw std::runtime_error("--count: invalid value " + std::string{text});
        }
    } else if (argc != 1) {
        throw std::runtime_error("usage: 16_allocator [--count N]");
    }
    if (count == 0 || count > 1'000'000) {
        throw std::runtime_error("--count must be 1-1000000");
    }
    return count;
}
// #endregion

// The sizes of the test buffers, used in turn: small uniform-like, a 4 KiB
// page, a 64 KiB mesh chunk.
constexpr std::array<vk::DeviceSize, 3> sizes{256, 4096, 65536};
constexpr vk::BufferUsageFlags usage =
    vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst;

// Step 13's find-memory-type rule.
std::optional<uint32_t> findType(const vk::PhysicalDeviceMemoryProperties& props, uint32_t bits,
                                 vk::MemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) != 0 && (props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return std::nullopt;
}

using Clock   = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

// Milliseconds and microseconds per buffer, for the [timing] lines.
std::string perBuffer(Seconds time, uint32_t count) {
    return std::format("{:.2f} ms ({:.2f} us per buffer)", time.count() * 1e3, time.count() * 1e6 / count);
}

// Throws for a failed VMA call (VMA returns plain VkResult codes).
void check(VkResult r, std::string_view what) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::format("{} failed: {}", what, vk::to_string(static_cast<vk::Result>(r))));
    }
}

// A buffer and the memory bound to it; members are destroyed in reverse
// order: the buffer first, then its memory.
struct RawBuffer {
    vk::raii::DeviceMemory memory;
    vk::raii::Buffer buffer;
};

int main(int argc, char** argv)
try {
    const uint32_t count = parseCount(argc, argv);

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
        .pApplicationName   = "Vulkan from Zero - 16 allocator",
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

    // #region allocation-limit
    // The limit that makes sub-allocation mandatory on some drivers. It counts
    // live vkAllocateMemory allocations, whatever their size; the spec
    // guarantees 4096. Drivers may report much more (a desktop driver may
    // report 2^32 - 1), but every allocation still costs a kernel call.
    const uint32_t limit = chosen.limits.maxMemoryAllocationCount;
    std::println("\nmaxMemoryAllocationCount: {} (the spec guarantees at least 4096)", limit);
    // #endregion

    // #region test-buffers
    // N buffers of the three sizes in turn, all DEVICE_LOCAL. The buffers are
    // created once per path; their requirements are the same every time.
    const auto bufferInfo = [](uint32_t i) {
        return vk::BufferCreateInfo{
            .size        = sizes[i % sizes.size()],
            .usage       = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        };
    };
    const vk::PhysicalDeviceMemoryProperties mem = physicalDevice.getMemoryProperties();
    std::array<vk::MemoryRequirements, sizes.size()> reqs{};
    for (uint32_t i = 0; i < reqs.size(); ++i) {
        reqs[i] = vk::raii::Buffer(device, bufferInfo(i)).getMemoryRequirements();
    }
    const auto memoryType = findType(mem, reqs[0].memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    if (!memoryType) {
        throw std::runtime_error("no DEVICE_LOCAL memory type for the buffers");
    }
    // Each path holds all N buffers at once, so their sizes must fit the heap
    // behind the type: check before allocating anything (the default fits
    // easily; --count 1000000 asks for about 23 GB). The heap size is only an
    // upper bound, not the budget: the driver and other programs use the heap
    // too, and VK_EXT_memory_budget reports how much of it this process may use.
    vk::DeviceSize need = 0;
    for (uint32_t i = 0; i < count; ++i) {
        need += reqs[i % reqs.size()].size;
    }
    const uint32_t heap           = mem.memoryTypes[*memoryType].heapIndex;
    const vk::DeviceSize heapSize = mem.memoryHeaps[heap].size;
    if (need > heapSize) {
        throw std::runtime_error(std::format("--count {}: the buffers need {} bytes, but heap {} has only {} bytes (an upper bound, not "
                                             "the budget: see VK_EXT_memory_budget)",
                                             count, need, heap, heapSize));
    }
    std::println("Buffers: {}, sizes {} / {} / {} bytes in turn, usage VERTEX_BUFFER | INDEX_BUFFER | TRANSFER_DST, DEVICE_LOCAL "
                 "(memory type {})",
                 count, sizes[0], sizes[1], sizes[2], *memoryType);
    // #endregion

    // #region raw
    // One vkAllocateMemory per buffer, exactly the size it asks for.
    std::println("\nRaw: one vkAllocateMemory per buffer");
    if (count > limit) {
        std::println("Raw: skipped, {} allocations would exceed maxMemoryAllocationCount ({})", count, limit);
    } else {
        const auto start = Clock::now();
        std::vector<RawBuffer> raw;
        raw.reserve(count);
        vk::DeviceSize requested = 0;
        for (uint32_t i = 0; i < count; ++i) {
            vk::raii::Buffer buffer(device, bufferInfo(i));
            const vk::MemoryRequirements req = buffer.getMemoryRequirements();
            requested += req.size;
            vk::raii::DeviceMemory memory(device, vk::MemoryAllocateInfo{.allocationSize = req.size, .memoryTypeIndex = *memoryType});
            buffer.bindMemory(*memory, 0);
            raw.push_back(RawBuffer{.memory = std::move(memory), .buffer = std::move(buffer)});
        }
        const Seconds created = Clock::now() - start;
        std::println("Raw: {} buffers, {} vkAllocateMemory calls = {} VkDeviceMemory objects, {} bytes requested", count, count, count,
                     requested);
        const auto destroyStart = Clock::now();
        raw.clear();
        const Seconds destroyed = Clock::now() - destroyStart;
        std::println("[timing] raw: create + allocate + bind {}, destroy + free {}", perBuffer(created, count),
                     perBuffer(destroyed, count));
    }
    // #endregion

    // #region allocator
    // A VmaAllocator per device. VMA reads the memory heaps and types itself,
    // keeps blocks per memory type, and places each resource with the TLSF
    // algorithm, so any allocation can be freed alone. preferredLargeHeapBlockSize
    // is set to the 64 MiB blocks of the Rust version (VMA's default is 256 MiB);
    // the device-memory callbacks count VMA's own vkAllocateMemory calls.
    uint32_t vmaAllocateCalls = 0;
    const VmaDeviceMemoryCallbacks memoryCallbacks{
        .pfnAllocate = [](VmaAllocator, uint32_t, VkDeviceMemory, VkDeviceSize, void* user) { ++*static_cast<uint32_t*>(user); },
        .pfnFree     = nullptr,
        .pUserData   = &vmaAllocateCalls,
    };
    // Only the two loader entry points; VMA fills the null members itself.
    // Value-initialized, then assigned: which members the struct has depends on VMA's
    // configuration macros (e.g. the *2KHR and maintenance4 functions), so a designated
    // initializer that names all of them would not be portable.
    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = context.getDispatcher()->vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr   = instance.getDispatcher()->vkGetDeviceProcAddr;
    constexpr vk::DeviceSize blockSize = 64ull * 1024 * 1024;
    const VmaAllocatorCreateInfo allocatorInfo{
        .flags                          = 0,
        .physicalDevice                 = *physicalDevice,
        .device                         = *device,
        .preferredLargeHeapBlockSize    = blockSize,
        .pAllocationCallbacks           = nullptr, // host allocations: VMA's default
        .pDeviceMemoryCallbacks         = &memoryCallbacks,
        .pHeapSizeLimit                 = nullptr, // no per-heap limit
        .pVulkanFunctions               = &functions,
        .instance                       = *instance,
        .vulkanApiVersion               = vk::ApiVersion14,
        .pTypeExternalMemoryHandleTypes = nullptr, // no exportable memory
    };
    VmaAllocator allocator = VK_NULL_HANDLE;
    check(vmaCreateAllocator(&allocatorInfo, &allocator), "vmaCreateAllocator");
    // #endregion

    // #region sub-allocate
    // The same buffers through VMA: vmaCreateBuffer creates the buffer, finds a
    // memory type (required DEVICE_LOCAL), sub-allocates and binds - the four
    // manual steps of step 13 in one call.
    std::println("\nAllocator: buffers bound at offsets inside big blocks");
    const VmaAllocationCreateInfo allocationInfo{
        .flags          = 0,
        .usage          = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags  = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        .preferredFlags = 0,
        .memoryTypeBits = 0, // 0 = any type allowed by the buffer's requirements
        .pool           = VK_NULL_HANDLE, // VMA's default pools
        .pUserData      = nullptr,
        .priority       = 0.0f, // only used with VK_EXT_memory_priority
        .minAlignment   = 0,
    };
    struct Placed {
        VkBuffer buffer;
        VmaAllocation allocation;
    };
    std::vector<Placed> placed;
    placed.reserve(count);
    // Destroys whatever was placed so far; runs on success and on errors.
    const auto release = [&] {
        for (const Placed& p : placed) {
            vmaDestroyBuffer(allocator, p.buffer, p.allocation);
        }
        placed.clear();
        vmaDestroyAllocator(allocator);
        allocator = VK_NULL_HANDLE;
    };
    try {
        const auto start         = Clock::now();
        vk::DeviceSize requested = 0;
        bool sameType            = true;
        std::vector<vk::DeviceSize> offsets;
        for (uint32_t i = 0; i < count; ++i) {
            const VkBufferCreateInfo info = bufferInfo(i);
            Placed p{};
            check(vmaCreateBuffer(allocator, &info, &allocationInfo, &p.buffer, &p.allocation, nullptr), "vmaCreateBuffer");
            placed.push_back(p);
            VmaAllocationInfo a{};
            vmaGetAllocationInfo(allocator, p.allocation, &a);
            requested += a.size;
            sameType = sameType && a.memoryType == *memoryType;
            if (offsets.size() < 4) {
                offsets.push_back(a.offset);
            }
        }
        const Seconds created = Clock::now() - start;
        if (!sameType) {
            throw std::runtime_error("VMA placed some buffers in another memory type");
        }
        std::println("Allocator: {} buffers in memory type {}, {} bytes requested", count, *memoryType, requested);
        VmaTotalStatistics stats{};
        vmaCalculateStatistics(allocator, &stats);
        const VmaStatistics& total = stats.total.statistics;
        // VMA makes its first blocks 1/8, 1/4 and 1/2 of the preferred size while
        // they suffice (VmaBlockVector::AllocatePage), so 64 MiB blocks start at 8 MiB.
        std::println("[lib] VMA 3.4.0, preferred block size {} MiB: {} vkAllocateMemory call(s), {} block(s) totalling {} bytes, {} allocations of {} bytes",
                     blockSize / (1024 * 1024), vmaAllocateCalls, total.blockCount, total.blockBytes, total.allocationCount,
                     total.allocationBytes);
        std::string offsetList;
        for (const vk::DeviceSize o : offsets) {
            offsetList += offsetList.empty() ? "" : ", ";
            offsetList += std::to_string(o);
        }
        std::println("[lib] first buffers at offsets {}", offsetList);
        const auto destroyStart = Clock::now();
        release();
        const Seconds destroyed = Clock::now() - destroyStart;
        std::println("[timing] allocator: create + allocate + bind {}, destroy + free {}", perBuffer(created, count),
                     perBuffer(destroyed, count));
    } catch (...) {
        release();
        throw;
    }
    // #endregion

    // #region cleanup
    // VMA is gone (vmaDestroyAllocator freed its blocks); RAII destroys the
    // device, the messenger and the instance in reverse order of declaration.
    std::println("\nAll buffers destroyed, all memory freed");
    // #endregion
    return EXIT_SUCCESS;
} catch (const vk::SystemError& e) {
    std::println(stderr, "Vulkan error: {}", e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return EXIT_FAILURE;
}
