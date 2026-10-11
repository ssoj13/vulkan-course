// Step 25 - Compute shaders
//
// No window, no rasterizer: a COMPUTE PIPELINE is one shader stage, and
// vkCmdDispatch launches a grid of workgroups that run it. Two kernels:
//
//   saxpy  (shaders/saxpy.slang):  result[i] = a * x[i] + y[i], one invocation per element
//   reduce (shaders/reduce.slang): every workgroup sums its 256 values in groupshared memory
//
//   buffers (STORAGE_BUFFER) -> descriptor set per kernel -> compute pipeline
//   record: push constants, dispatch ceil(N / 256) groups, barrier COMPUTE -> HOST
//   submit, wait for the fence, read the results through mapped HOST_CACHED memory
//
// The CPU computes the same results and compares them exactly (integers in
// float and uint). --count N sets the number of elements (default 1000000,
// deliberately not a multiple of 256), --uncached reads the results from
// uncached memory to compare the CPU read time. Lines that depend on timing
// start with [timing]. Steps 01-05 are repeated, reduced to what this step needs.

#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
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
// Command line: --count N elements (default 1000000) and --uncached.
struct Args {
    uint32_t count = 1'000'000;
    bool uncached  = false;
};

// The largest --count: saxpy's results go up to 2 * count - 1, and every
// integer up to 2^24 is exact in a float, so count <= 2^23 keeps the GPU's
// float results exactly comparable with the CPU's.
constexpr uint32_t maxCount = 1u << 23;

Args parseArgs(int argc, char** argv) {
    constexpr std::string_view usage = "usage: 25_compute [--count N] [--uncached]";
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--count" && i + 1 < argc) {
            const std::string_view text = argv[++i];
            const auto [end, ec]        = std::from_chars(text.data(), text.data() + text.size(), args.count);
            if (ec != std::errc{} || end != text.data() + text.size()) {
                throw std::runtime_error(std::string{usage});
            }
        } else if (flag == "--uncached") {
            args.uncached = true;
        } else {
            throw std::runtime_error(std::string{usage});
        }
    }
    if (args.count == 0 || args.count > maxCount) {
        throw std::runtime_error(std::format("--count must be 1-{}", maxCount));
    }
    return args;
}
// #endregion

// The scale of saxpy.
constexpr float a = 2.0f;

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

// Step 13's two passes: required | preferred first, then required alone.
std::optional<uint32_t> chooseType(const vk::PhysicalDeviceMemoryProperties& props, uint32_t bits,
                                   vk::MemoryPropertyFlags required, vk::MemoryPropertyFlags preferred) {
    if (const auto both = findType(props, bits, required | preferred)) {
        return both;
    }
    return findType(props, bits, required);
}

// The memory types that have any of flags, as a memoryTypeBits-style mask:
// bits & ~typesWith(...) rules them out of a choice.
uint32_t typesWith(const vk::PhysicalDeviceMemoryProperties& props, vk::MemoryPropertyFlags flags) {
    uint32_t mask = 0;
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if (props.memoryTypes[i].propertyFlags & flags) {
            mask |= 1u << i;
        }
    }
    return mask;
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

// Decodes a SPIR-V literal string: UTF-8 bytes packed little-endian into
// words, NUL-terminated (step 17).
std::string spvString(std::span<const uint32_t> words) {
    std::string out;
    for (const uint32_t w : words) {
        for (int i = 0; i < 4; ++i) {
            const auto c = static_cast<char>((w >> (8 * i)) & 0xffu);
            if (c == '\0') {
                return out;
            }
            out += c;
        }
    }
    return out;
}

// #region local-size
// The workgroup size of entry, read from the SPIR-V: the module's OpEntryPoint
// names the function, OpExecutionMode LocalSize gives x, y, z. The host needs
// it to turn "N elements" into "groups to dispatch"; reading it from the module
// keeps one source of truth (the shader's [numthreads]).
std::array<uint32_t, 3> localSize(std::span<const uint32_t> words, std::string_view entry) {
    constexpr uint32_t opEntryPoint    = 15;
    constexpr uint32_t opExecutionMode = 16;
    constexpr uint32_t modeLocalSize   = 17;
    std::optional<uint32_t> function;
    std::vector<std::pair<uint32_t, std::array<uint32_t, 3>>> sizes;
    // Instructions start after the 5-word header; each one's first word is
    // (word count << 16) | opcode.
    size_t pos = 5;
    while (pos < words.size()) {
        const uint32_t first = words[pos];
        const size_t count   = first >> 16;
        if (count == 0 || pos + count > words.size()) {
            throw std::runtime_error("malformed SPIR-V instruction");
        }
        const std::span<const uint32_t> ins = words.subspan(pos, count);
        const uint32_t op                   = first & 0xffffu;
        // Operands: execution model, function id, name (a literal string), interface ids.
        if (op == opEntryPoint && count >= 4 && spvString(ins.subspan(3)) == entry) {
            function = ins[2];
        } else if (op == opExecutionMode && count == 6 && ins[2] == modeLocalSize) {
            sizes.push_back({ins[1], {ins[3], ins[4], ins[5]}});
        }
        pos += count;
    }
    if (!function) {
        throw std::runtime_error(std::format("no entry point {}", entry));
    }
    const auto found = std::ranges::find_if(sizes, [&](const auto& s) { return s.first == *function; });
    if (found == sizes.end()) {
        throw std::runtime_error(std::format("{} has no LocalSize execution mode", entry));
    }
    return found->second;
}
// #endregion

// Step 10: waits for the device to go idle when it leaves scope, so RAII never
// destroys an object the GPU may still be using, even when the program throws.
struct IdleGuard {
    const vk::raii::Device& device;
    ~IdleGuard() {
        if (const VkResult r = device.getDispatcher()->vkDeviceWaitIdle(*device); r != VK_SUCCESS) {
            std::fprintf(stderr, "vkDeviceWaitIdle failed: %s\n", string_VkResult(r));
        }
    }
};

// A buffer, its memory and the memory's persistent mapping. Members are
// destroyed in reverse order: the buffer first, then its memory (freeing a
// mapped allocation unmaps it).
struct GpuBuffer {
    vk::raii::DeviceMemory memory;
    vk::raii::Buffer buffer;
    uint32_t memoryType;
    void* ptr;
};

// Creates a STORAGE_BUFFER of size bytes in the type chooseType picks among
// those without any excluded flag, and maps it for the program's lifetime.
GpuBuffer createBuffer(const vk::raii::Device& device, const vk::PhysicalDeviceMemoryProperties& props, vk::DeviceSize size,
                       vk::MemoryPropertyFlags required, vk::MemoryPropertyFlags preferred, vk::MemoryPropertyFlags excluded) {
    vk::raii::Buffer buffer(device, vk::BufferCreateInfo{
                                        .size        = size,
                                        .usage       = vk::BufferUsageFlagBits::eStorageBuffer,
                                        .sharingMode = vk::SharingMode::eExclusive,
                                    });
    const vk::MemoryRequirements req = buffer.getMemoryRequirements();
    const auto memoryType            = chooseType(props, req.memoryTypeBits & ~typesWith(props, excluded), required, preferred);
    if (!memoryType) {
        throw std::runtime_error("no memory type with " + memoryNames(required) + " and without " + memoryNames(excluded));
    }
    vk::raii::DeviceMemory memory(device, vk::MemoryAllocateInfo{.allocationSize = req.size, .memoryTypeIndex = *memoryType});
    buffer.bindMemory(*memory, 0);
    void* ptr = memory.mapMemory(0, vk::WholeSize);
    return GpuBuffer{.memory = std::move(memory), .buffer = std::move(buffer), .memoryType = *memoryType, .ptr = ptr};
}

// #region kernel
// One compute kernel: a set layout with `bindings` storage buffers (binding 0,
// 1, ...), a pipeline layout with that set and a push constant range of
// pushSize bytes, and a compute pipeline running entry of spirv. Compared with
// a graphics pipeline there is nothing else to describe: no vertex input, no
// rasterizer, no attachments - one stage.
struct Kernel {
    vk::raii::DescriptorSetLayout setLayout;
    vk::raii::PipelineLayout layout;
    vk::raii::Pipeline pipeline;
};

Kernel createKernel(const vk::raii::Device& device, std::span<const uint32_t> spirv, const char* entry, uint32_t bindings,
                    uint32_t pushSize) {
    std::vector<vk::DescriptorSetLayoutBinding> setBindings;
    for (uint32_t b = 0; b < bindings; ++b) {
        setBindings.push_back(vk::DescriptorSetLayoutBinding{
            .binding         = b,
            .descriptorType  = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags      = vk::ShaderStageFlagBits::eCompute,
        });
    }
    vk::raii::DescriptorSetLayout setLayout(device, vk::DescriptorSetLayoutCreateInfo{
                                                        .bindingCount = static_cast<uint32_t>(setBindings.size()),
                                                        .pBindings    = setBindings.data(),
                                                    });
    const vk::DescriptorSetLayout setLayoutHandle = *setLayout;
    const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = pushSize};
    vk::raii::PipelineLayout layout(device, vk::PipelineLayoutCreateInfo{
                                                .setLayoutCount         = 1,
                                                .pSetLayouts            = &setLayoutHandle,
                                                .pushConstantRangeCount = 1,
                                                .pPushConstantRanges    = &range,
                                            });
    // The module is only needed while the pipeline is created; a pipeline
    // keeps working after its module is destroyed (end of this scope).
    const vk::raii::ShaderModule module(device, vk::ShaderModuleCreateInfo{
                                                    .codeSize = spirv.size() * sizeof(uint32_t),
                                                    .pCode    = spirv.data(),
                                                });
    vk::raii::Pipeline pipeline(device, nullptr,
                                vk::ComputePipelineCreateInfo{
                                    .stage  = {.stage = vk::ShaderStageFlagBits::eCompute, .module = *module, .pName = entry},
                                    .layout = *layout,
                                });
    return Kernel{.setLayout = std::move(setLayout), .layout = std::move(layout), .pipeline = std::move(pipeline)};
}
// #endregion

int main(int argc, char** argv)
try {
    const Args args = parseArgs(argc, argv);

    // ---- Steps 01-02: instance with validation ----
    vk::raii::Context context;
    constexpr const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    const std::vector<vk::LayerProperties> layers = context.enumerateInstanceLayerProperties();
    const bool hasValidation = std::ranges::any_of(
        layers, [&](const vk::LayerProperties& l) { return std::string_view{l.layerName.data()} == validationLayer; });
    if (!hasValidation) {
        throw std::runtime_error(std::string{validationLayer} + " not found - install the Vulkan SDK");
    }
    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName   = "Vulkan from Zero - 25 compute",
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
    constexpr std::array extensions{vk::EXTDebugUtilsExtensionName};
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

    // ---- Step 05's choice: Vulkan 1.4 with synchronization2, discrete first ----
    const std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }
    std::println("Found {} GPU(s)", devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        std::println("GPU {}: {}", i, devices[i].getProperties().deviceName.data());
    }
    // synchronization2 is needed for pipelineBarrier2, submit2, writeTimestamp2.
    const vk::raii::PhysicalDevice* best = nullptr;
    int bestScore                        = 0;
    for (const vk::raii::PhysicalDevice& device : devices) {
        const vk::PhysicalDeviceProperties props = device.getProperties();
        if (props.apiVersion < vk::ApiVersion14) {
            continue;
        }
        const auto featureChain = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>();
        if (featureChain.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 != vk::True) {
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
        throw std::runtime_error("No GPU supports Vulkan 1.4 with synchronization2");
    }
    const vk::raii::PhysicalDevice& physicalDevice = *best;
    std::println("\nPicked: {}", physicalDevice.getProperties().deviceName.data());

    // #region queue-family
    // Any family with COMPUTE can run dispatches; on desktop GPUs the graphics
    // family has COMPUTE too (family 0 on the RTX 3080 Ti). The first one is
    // enough here; step 27 looks for a family WITHOUT graphics.
    const std::vector<vk::QueueFamilyProperties> families = physicalDevice.getQueueFamilyProperties();
    const auto computeFamily                              = std::ranges::find_if(
        families, [](const vk::QueueFamilyProperties& f) { return static_cast<bool>(f.queueFlags & vk::QueueFlagBits::eCompute); });
    if (computeFamily == families.end()) {
        throw std::runtime_error("No compute queue family");
    }
    const auto family = static_cast<uint32_t>(computeFamily - families.begin());
    std::println("Compute queue family: {} (the first with COMPUTE)", family);
    // #endregion
    constexpr float queuePriority = 1.0f;
    const vk::DeviceQueueCreateInfo queueInfo{.queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &queuePriority};
    vk::PhysicalDeviceVulkan13Features features13{.synchronization2 = vk::True};
    vk::raii::Device device(physicalDevice, vk::DeviceCreateInfo{
                                                .pNext                = &features13,
                                                .queueCreateInfoCount = 1,
                                                .pQueueCreateInfos    = &queueInfo,
                                            });
    std::println("VkDevice created with synchronization2");

    // #region limits
    // The limits a dispatch must respect, and the subgroup size: how many
    // invocations the hardware packs into one SIMD group (a warp on NVIDIA; the
    // spec promises no lockstep execution). A workgroup is split into
    // subgroups, so its size is best a multiple of this. The subgroup size is a
    // Vulkan 1.1 property, read through the properties2 chain.
    const auto propsChain       = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceVulkan11Properties>();
    const vk::PhysicalDeviceLimits limits = propsChain.get<vk::PhysicalDeviceProperties2>().properties.limits;
    const uint32_t subgroup     = propsChain.get<vk::PhysicalDeviceVulkan11Properties>().subgroupSize;
    const auto& groupCount      = limits.maxComputeWorkGroupCount;
    const auto& groupSize       = limits.maxComputeWorkGroupSize;
    std::println("\nCompute limits:");
    std::println("  maxComputeWorkGroupInvocations = {} (Vulkan 1.4 guarantees 256)", limits.maxComputeWorkGroupInvocations);
    std::println("  maxComputeWorkGroupSize = {} x {} x {}", groupSize[0], groupSize[1], groupSize[2]);
    std::println("  maxComputeWorkGroupCount = {} x {} x {}", groupCount[0], groupCount[1], groupCount[2]);
    std::println("  maxComputeSharedMemorySize = {} bytes", limits.maxComputeSharedMemorySize);
    std::println("  subgroupSize = {}", subgroup);
    // #endregion

    // The SPIR-V is compiled from shaders/{saxpy,reduce}.slang by `bun run
    // sync:shaders`; CMake bakes the absolute path of shaders/spv into SHADER_DIR.
    const std::vector<uint32_t> saxpySpv  = readSpirv(std::filesystem::path{SHADER_DIR} / "saxpy.spv");
    const std::vector<uint32_t> reduceSpv = readSpirv(std::filesystem::path{SHADER_DIR} / "reduce.spv");

    // #region group-count
    // The workgroup size comes from the shader; the host only decides how many
    // groups: ceil(count / group size), so the last group may be partly past
    // the end (the shaders check i < count). Two kinds of limits apply. The
    // workgroup size is a rule on the shader the pipeline is created with:
    // LocalSize x at most maxComputeWorkGroupSize[0] (VUID-RuntimeSpirv-x-06429),
    // x * y * z at most maxComputeWorkGroupInvocations
    // (VUID-RuntimeSpirv-x-06432). The group count is a rule on vkCmdDispatch:
    // at most maxComputeWorkGroupCount (VUID-vkCmdDispatch-groupCountX-00386).
    std::println("\nKernels (workgroup size read from the SPIR-V):");
    std::vector<uint32_t> sizes;
    for (const auto& [file, words, entry] : {std::tuple{"saxpy.spv", &saxpySpv, "saxpy"}, std::tuple{"reduce.spv", &reduceSpv, "reduce"}}) {
        const auto [x, y, z] = localSize(*words, entry);
        // Both kernels are 1D (y = z = 1), so x alone is the invocation count.
        if (y != 1 || z != 1) {
            throw std::runtime_error(std::format("{}: expected a 1D workgroup, got {}x{}x{}", entry, x, y, z));
        }
        if (x > groupSize[0] || x > limits.maxComputeWorkGroupInvocations) {
            throw std::runtime_error(std::format("{}: a workgroup of {} exceeds this device's limits", entry, x));
        }
        std::println("  {}: {} bytes, entry point {}, LocalSize {} x {} x {} = {} subgroups of {}", file,
                     words->size() * sizeof(uint32_t), entry, x, y, z, (x + subgroup - 1) / subgroup, subgroup);
        sizes.push_back(x);
    }
    const uint32_t saxpySize  = sizes.at(0);
    const uint32_t reduceSize = sizes.at(1);
    // reduce.slang's groupshared array has one uint per invocation.
    const uint32_t sharedBytes = reduceSize * 4;
    std::println("  reduce: groupshared {} bytes per workgroup (limit {})", sharedBytes, limits.maxComputeSharedMemorySize);
    const uint32_t saxpyGroups  = (args.count + saxpySize - 1) / saxpySize;
    const uint32_t reduceGroups = (args.count + reduceSize - 1) / reduceSize;
    if (std::max(saxpyGroups, reduceGroups) > groupCount[0]) {
        throw std::runtime_error(std::format("{} groups exceed maxComputeWorkGroupCount[0] = {}", saxpyGroups, groupCount[0]));
    }
    const uint32_t tail = args.count - (saxpyGroups - 1) * saxpySize;
    std::println("\n{} elements -> {} workgroups of {} (the last one has {} elements in range)", args.count, saxpyGroups, saxpySize,
                 tail);
    // #endregion

    // #region memory
    // Inputs: the CPU writes them once, the GPU reads them: HOST_VISIBLE |
    // HOST_COHERENT, the upload type of step 14 (write-combined: fast to write,
    // slow to read). Outputs: the GPU writes them, the CPU READS them: the spec
    // expects HOST_CACHED for that ("when the host performs any read or
    // read-modify-write operations on the memory"), because "host memory read
    // accesses to uncached memory are slower than to cached memory". Both live
    // in system RAM on this GPU, so the shaders read and write them across PCIe
    // - fine for one pass over the data; data the GPU reuses belongs in
    // DEVICE_LOCAL memory (step 15's staging). --uncached rules HOST_CACHED
    // types out instead, for the comparison; the spec adds that "uncached
    // memory is always host coherent".
    using Mem                                    = vk::MemoryPropertyFlagBits;
    const vk::PhysicalDeviceMemoryProperties mem = physicalDevice.getMemoryProperties();
    const vk::MemoryPropertyFlags none{};
    const vk::MemoryPropertyFlags upload            = Mem::eHostVisible | Mem::eHostCoherent;
    const vk::MemoryPropertyFlags readbackRequired  = Mem::eHostVisible;
    const vk::MemoryPropertyFlags readbackPreferred = args.uncached ? none : Mem::eHostCached;
    const vk::MemoryPropertyFlags readbackExcluded  = args.uncached ? Mem::eHostCached : none;
    const size_t n                                  = args.count;
    const vk::DeviceSize floatBytes                 = n * sizeof(float);
    const vk::DeviceSize uintBytes                  = n * sizeof(uint32_t);
    const vk::DeviceSize sumsBytes                  = vk::DeviceSize{reduceGroups} * 4;
    GpuBuffer x      = createBuffer(device, mem, floatBytes, upload, none, none);
    GpuBuffer y      = createBuffer(device, mem, floatBytes, upload, none, none);
    GpuBuffer values = createBuffer(device, mem, uintBytes, upload, none, none);
    GpuBuffer result = createBuffer(device, mem, floatBytes, readbackRequired, readbackPreferred, readbackExcluded);
    GpuBuffer sums   = createBuffer(device, mem, sumsBytes, readbackRequired, readbackPreferred, readbackExcluded);
    // "The memoryTypeBits member is identical for all VkBuffer objects created
    // with the same value for the flags and usage members": the two outputs get
    // the same type from the same choice, and everything below (the printout,
    // the invalidate) treats them as one.
    if (sums.memoryType != result.memoryType) {
        throw std::runtime_error("result and groupSums got different memory types");
    }
    const vk::MemoryPropertyFlags outFlags = mem.memoryTypes.at(result.memoryType).propertyFlags;
    std::println("\nMemory:");
    std::println("  inputs x, y, values ({} bytes each): memory type {} ({}), written by the CPU", floatBytes, x.memoryType,
                 memoryNames(mem.memoryTypes.at(x.memoryType).propertyFlags));
    std::println("  outputs result ({} bytes), groupSums ({} bytes): memory type {} ({}), read by the CPU", floatBytes, sumsBytes,
                 result.memoryType, memoryNames(outFlags));
    // #endregion

    // #region fill-inputs
    // Integer inputs make every result exact: x[i] = i, y[i] = 1 gives
    // result[i] = 2i + 1 < 2^24; Vulkan requires OpFMul and OpFAdd to be
    // correctly rounded, and an exact result needs no rounding, so the GPU must
    // produce exactly what the CPU does (fused or not).
    std::vector<float> xs(n);
    std::vector<uint32_t> vs(n);
    for (uint32_t i = 0; i < args.count; ++i) {
        xs[i] = static_cast<float>(i);
        vs[i] = i % 1000;
    }
    const std::vector<float> ys(n, 1.0f);
    // The mappings cover whole allocations of at least these sizes; the GPU
    // does not use them yet. HOST_COHERENT: the submit makes the writes
    // visible to the device (no flush).
    std::memcpy(x.ptr, xs.data(), floatBytes);
    std::memcpy(y.ptr, ys.data(), floatBytes);
    std::memcpy(values.ptr, vs.data(), uintBytes);
    // #endregion

    // #region descriptors
    // One set per kernel; each binding points at a whole buffer.
    const Kernel saxpy  = createKernel(device, saxpySpv, "saxpy", 3, 8);
    const Kernel reduce = createKernel(device, reduceSpv, "reduce", 2, 4);
    std::println("\nCompute pipelines: saxpy (3 storage buffers, 8 push constant bytes), reduce (2 storage buffers, 4 bytes)");
    const vk::DescriptorPoolSize poolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 5};
    vk::raii::DescriptorPool descriptorPool(device, vk::DescriptorPoolCreateInfo{.maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &poolSize});
    // vk::raii::DescriptorSet would call vkFreeDescriptorSets in its destructor,
    // which needs a pool created with eFreeDescriptorSet; our sets go with the
    // pool, so we keep the plain handles (release()).
    const std::array setLayouts{*saxpy.setLayout, *reduce.setLayout};
    std::vector<vk::DescriptorSet> sets;
    for (vk::raii::DescriptorSet& set : vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
                                                                             .descriptorPool     = *descriptorPool,
                                                                             .descriptorSetCount = 2,
                                                                             .pSetLayouts        = setLayouts.data(),
                                                                         })) {
        sets.push_back(set.release());
    }
    const vk::DescriptorSet saxpySet  = sets.at(0);
    const vk::DescriptorSet reduceSet = sets.at(1);
    struct Binding {
        vk::DescriptorSet set;
        uint32_t binding;
        const GpuBuffer& buffer;
    };
    const std::array<Binding, 5> bindings{{
        {saxpySet, 0, x},
        {saxpySet, 1, y},
        {saxpySet, 2, result},
        {reduceSet, 0, values},
        {reduceSet, 1, sums},
    }};
    std::array<vk::DescriptorBufferInfo, 5> infos{};
    std::vector<vk::WriteDescriptorSet> writes;
    for (size_t i = 0; i < bindings.size(); ++i) {
        infos[i] = vk::DescriptorBufferInfo{.buffer = *bindings[i].buffer.buffer, .offset = 0, .range = vk::WholeSize};
        writes.push_back(vk::WriteDescriptorSet{
            .dstSet          = bindings[i].set,
            .dstBinding      = bindings[i].binding,
            .descriptorCount = 1,
            .descriptorType  = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo     = &infos[i],
        });
    }
    device.updateDescriptorSets(writes, nullptr);
    std::println("Descriptor sets: saxpy x/y/result at bindings 0/1/2, reduce values/groupSums at 0/1");
    // #endregion

    // #region timestamps
    // Three GPU timestamps around the dispatches (step 12 explains the query
    // pool, timestampValidBits and timestampPeriod).
    const uint32_t validBits = families.at(family).timestampValidBits;
    if (validBits == 0) {
        throw std::runtime_error("The compute queue family has no timestamps");
    }
    vk::raii::QueryPool queryPool(device, vk::QueryPoolCreateInfo{.queryType = vk::QueryType::eTimestamp, .queryCount = 3});
    // #endregion

    // One command buffer, submitted once (step 09).
    vk::raii::CommandPool commandPool(device, vk::CommandPoolCreateInfo{.queueFamilyIndex = family});
    vk::raii::Fence fence(device, vk::FenceCreateInfo{});
    vk::raii::CommandBuffers commandBuffers(device, vk::CommandBufferAllocateInfo{
                                                        .commandPool        = *commandPool,
                                                        .level              = vk::CommandBufferLevel::ePrimary,
                                                        .commandBufferCount = 1,
                                                    });
    const vk::raii::CommandBuffer& cmd = commandBuffers.front();

    // #region record
    // The compute bind point has its own pipeline and descriptor sets,
    // separate from the graphics ones; push constants are kept per shader
    // stage, not per bind point. The two dispatches touch different
    // buffers, so nothing orders them: the GPU may run them concurrently.
    // Timestamp 0 at TOP_OF_PIPE, 1 after saxpy's COMPUTE_SHADER work, 2
    // after both.
    struct SaxpyParams {
        float a;
        uint32_t count;
    };
    const SaxpyParams pushSaxpy{.a = a, .count = args.count};
    using S = vk::PipelineStageFlagBits2;
    cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cmd.resetQueryPool(*queryPool, 0, 3);
    cmd.writeTimestamp2(S::eTopOfPipe, *queryPool, 0);

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *saxpy.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *saxpy.layout, 0, saxpySet, nullptr);
    cmd.pushConstants<SaxpyParams>(*saxpy.layout, vk::ShaderStageFlagBits::eCompute, 0, pushSaxpy);
    cmd.dispatch(saxpyGroups, 1, 1);
    cmd.writeTimestamp2(S::eComputeShader, *queryPool, 1);

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *reduce.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *reduce.layout, 0, reduceSet, nullptr);
    cmd.pushConstants<uint32_t>(*reduce.layout, vk::ShaderStageFlagBits::eCompute, 0, args.count);
    cmd.dispatch(reduceGroups, 1, 1);
    cmd.writeTimestamp2(S::eComputeShader, *queryPool, 2);
    std::println("\nRecorded: dispatch saxpy {} x 1 x 1, dispatch reduce {} x 1 x 1", saxpyGroups, reduceGroups);
    // #endregion

    // #region readback-barrier
    // The CPU will read what the shaders wrote. The fence alone is not enough:
    // "Signaling a fence and waiting on the host does not guarantee that the
    // results of memory accesses will be visible to the host, as the access
    // scope of a memory dependency defined by a fence only includes device
    // access." This barrier makes the shader writes available in the HOST
    // domain: source = the compute stage's storage writes, destination = host
    // reads. Visible to the CPU they are through HOST_COHERENT, or else after
    // vkInvalidateMappedMemoryRanges.
    const vk::MemoryBarrier2 toHost{
        .srcStageMask  = S::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask  = S::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toHost});
    cmd.end();
    std::println("Barrier: COMPUTE_SHADER / SHADER_STORAGE_WRITE -> HOST / HOST_READ");
    // #endregion

    // #region submit-and-wait
    // If anything below throws while the GPU still runs, the guard waits for
    // the device before RAII destroys what the submission uses.
    const IdleGuard idleGuard{device};
    const vk::raii::Queue queue(device, family, 0);
    const vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *cmd};
    queue.submit2(vk::SubmitInfo2{.commandBufferInfoCount = 1, .pCommandBufferInfos = &cmdInfo}, *fence);
    if (device.waitForFences(*fence, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
        throw std::runtime_error("waitForFences failed");
    }
    const auto [queried, ticks] = queryPool.getResults<uint64_t>(0, 3, 3 * sizeof(uint64_t), sizeof(uint64_t),
                                                                 vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (queried != vk::Result::eSuccess) {
        throw std::runtime_error("getQueryPoolResults failed");
    }
    const uint64_t mask = validBits >= 64 ? ~uint64_t{0} : (uint64_t{1} << validBits) - 1;
    const auto us       = [&](uint64_t from, uint64_t to) {
        return static_cast<double>((to - from) & mask) * static_cast<double>(limits.timestampPeriod) / 1000.0;
    };
    // saxpy moves 12 bytes per element: reads x and y, writes result.
    const double saxpyUs = us(ticks.at(0), ticks.at(1));
    std::println("[timing] GPU: saxpy {:.1f} us ({:.1f} GB/s over PCIe), saxpy + reduce {:.1f} us", saxpyUs,
                 static_cast<double>(n * 12) / (saxpyUs * 1e3), us(ticks.at(0), ticks.at(2)));
    // #endregion

    // #region read-results
    // A non-coherent type would need vkInvalidateMappedMemoryRanges here, so the
    // CPU does not read stale cache lines; HOST_COHERENT types do not.
    if (!(outFlags & Mem::eHostCoherent)) {
        const std::array ranges{
            vk::MappedMemoryRange{.memory = *result.memory, .offset = 0, .size = vk::WholeSize},
            vk::MappedMemoryRange{.memory = *sums.memory, .offset = 0, .size = vk::WholeSize},
        };
        device.invalidateMappedMemoryRanges(ranges);
        std::println("Output memory is not HOST_COHERENT: vkInvalidateMappedMemoryRanges before reading");
    } else {
        std::println("Output memory is HOST_COHERENT: no vkInvalidateMappedMemoryRanges needed");
    }
    using Clock          = std::chrono::steady_clock;
    const auto readStart = Clock::now();
    const size_t groups  = reduceGroups;
    // Both mappings hold at least this many elements, written by the completed
    // submission, made available to the host by the barrier and visible by
    // coherence (or the invalidate above).
    std::vector<float> gpuResult(n);
    std::vector<uint32_t> gpuSums(groups);
    std::memcpy(gpuResult.data(), result.ptr, n * sizeof(float));
    std::memcpy(gpuSums.data(), sums.ptr, groups * sizeof(uint32_t));
    const double readSeconds = std::chrono::duration<double>(Clock::now() - readStart).count();
    std::println("[timing] CPU copy of the results out of memory type {}: {:.2f} ms ({:.2f} GB/s)", result.memoryType, readSeconds * 1e3,
                 static_cast<double>(n * 4 + groups * 4) / readSeconds / 1e9);
    // #endregion

    // #region verify-saxpy
    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        mismatches += a * xs[i] + ys[i] != gpuResult[i] ? 1 : 0;
    }
    const size_t last = n - 1;
    std::println("\nsaxpy: result[i] = {} * x[i] + y[i], x[i] = i, y[i] = 1", a);
    std::string first;
    const size_t shown = std::min<size_t>(4, n);
    for (size_t i = 0; i < shown; ++i) {
        first += std::format("{}{}", i == 0 ? "" : ", ", gpuResult[i]);
    }
    std::println("  result[0..{}] = {}, result[{}] = {}", shown, first, last, gpuResult[last]);
    if (mismatches != 0) {
        throw std::runtime_error(std::format("saxpy: {} of {} results differ from the CPU", mismatches, n));
    }
    std::println("  all {} results equal the CPU's exactly", n);
    // #endregion

    // #region verify-reduce
    std::vector<uint32_t> cpuSums;
    for (size_t g = 0; g < n; g += reduceSize) {
        const auto end = vs.begin() + static_cast<std::ptrdiff_t>(std::min(n, g + reduceSize));
        cpuSums.push_back(std::accumulate(vs.begin() + static_cast<std::ptrdiff_t>(g), end, uint32_t{0}));
    }
    const uint64_t cpuTotal = std::accumulate(vs.begin(), vs.end(), uint64_t{0});
    const uint64_t gpuTotal = std::accumulate(gpuSums.begin(), gpuSums.end(), uint64_t{0});
    std::println("\nreduce: values[i] = i % 1000, one sum per workgroup of {}", reduceSize);
    std::println("  groupSums[0] = {}, groupSums[{}] = {} (the last group has {} values)", gpuSums.front(), groups - 1, gpuSums.back(),
                 args.count - (reduceGroups - 1) * reduceSize);
    if (cpuSums != gpuSums) {
        throw std::runtime_error("reduce: the group sums differ from the CPU");
    }
    if (cpuTotal != gpuTotal) {
        throw std::runtime_error(std::format("reduce: total {}, CPU {}", gpuTotal, cpuTotal));
    }
    std::println("  all {} group sums equal the CPU's; their total {} = the CPU's sum of all {} values", groups, gpuTotal, n);
    // #endregion

    // #region cleanup
    // The fence wait above means the GPU is done; RAII destroys everything in
    // reverse order of declaration (freeing a mapped allocation unmaps it).
    // #endregion
    return EXIT_SUCCESS;
} catch (const vk::SystemError& e) {
    std::println(stderr, "Vulkan error: {}", e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return EXIT_FAILURE;
}
