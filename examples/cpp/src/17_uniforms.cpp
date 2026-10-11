// Step 17 - Uniform buffers and descriptor sets
//
// Step 14's hexagon gets a transform the CPU changes every frame: a rotation
// and the window's aspect ratio in one 4x4 matrix, plus a tint and the time.
// Shaders read such per-frame values from a UNIFORM BUFFER, and they find the
// buffer through a DESCRIPTOR: a small record "binding 0 of set 0 = this
// buffer, this offset, this range". Descriptors live in DESCRIPTOR SETS,
// allocated from a DESCRIPTOR POOL with a DESCRIPTOR SET LAYOUT that the
// pipeline layout also names - like a shader's parameter list (layout) and the
// arguments for one call (set).
//
//   FrameUniforms (host struct, std140 offsets) -> one uniform buffer per frame in flight
//   set layout: binding 0 = UNIFORM_BUFFER, VERTEX -> pipeline layout
//   pool -> 2 sets -> vkUpdateDescriptorSets: set i -> uniform buffer i
//   frame: write the slot's buffer -> bind its set -> draw
//
// The layout the shader expects is read from the SPIR-V itself and compared
// with the host struct's offsets. --seconds S closes the window after S
// seconds; lines that depend on timing start with [timing].
// Steps 01-14 are repeated, reduced to what this step needs.

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
#include <map>
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

// Step 14's vertex: 2 + 3 floats = 20 bytes (shaders/transform.slang reads it
// with the same vertex input state).
struct Vertex {
    std::array<float, 2> position; // x, y before the transform
    std::array<float, 3> color;    // linear RGB
};
static_assert(sizeof(Vertex) == 20);

// #region mesh
// Step 14's hexagon, now REGULAR (radius 0.6): the transform's aspect-ratio
// column squeezes x for the 800x600 window, so the mesh no longer has to.
// Corner i sits at 60 degrees * i; each triangle (center, i, i+1) is clockwise
// on screen (+y points down), the pipeline's front face.
constexpr std::array<Vertex, 7> vertices{{
    {{0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},         // center, white
    {{0.6f, 0.0f}, {1.0f, 0.0f, 0.0f}},         // 0 deg, red
    {{0.3f, 0.5196152f}, {1.0f, 1.0f, 0.0f}},   // 60 deg, yellow
    {{-0.3f, 0.5196152f}, {0.0f, 1.0f, 0.0f}},  // 120 deg, green
    {{-0.6f, 0.0f}, {0.0f, 1.0f, 1.0f}},        // 180 deg, cyan
    {{-0.3f, -0.5196152f}, {0.0f, 0.0f, 1.0f}}, // 240 deg, blue
    {{0.3f, -0.5196152f}, {1.0f, 0.0f, 1.0f}},  // 300 deg, magenta
}};
// Step 14's indices: six triangles around the center.
constexpr std::array<uint16_t, 18> indices{
    0, 1, 2, //
    0, 2, 3, //
    0, 3, 4, //
    0, 4, 5, //
    0, 5, 6, //
    0, 6, 1, //
};
// #endregion

// #region uniform-struct
// The host side of FrameUniforms in shaders/transform.slang. The shader reads
// the block with std140 rules: a float3 is aligned to 16 bytes but only 12
// long, so time packs into the 4 bytes after tint. A standard-layout struct
// gives the same offsets here (0, 64, 76) because every field is float and the
// struct is ordered to need no padding; a float3 followed by another float3
// would NOT match (std140 puts the second at +16, C++ at +12). reflect() below
// checks the offsets and the matrix layout against the SPIR-V.
struct FrameUniforms {
    std::array<std::array<float, 4>, 4> transform; // column-major: transform[c] is column c (Slang's default)
    std::array<float, 3> tint;                     // linear RGB multiplier
    float time;                                    // seconds since start (the shader makes the colors pulse)
};
static_assert(sizeof(FrameUniforms) == 80);

// Field names, host offsets and whether the field is a matrix (stored as four
// 16-byte columns), in declaration order, for the comparison.
struct HostField {
    std::string_view name;
    size_t offset;
    bool matrix;
};
constexpr std::array<HostField, 3> hostLayout{{
    {"transform", offsetof(FrameUniforms, transform), true},
    {"tint", offsetof(FrameUniforms, tint), false},
    {"time", offsetof(FrameUniforms, time), false},
}};
// #endregion

// #region transform
// Rotation speed of the hexagon, radians per second.
constexpr float spin = 0.8f;
// A warm tint, multiplied into the vertex colors by the shader.
constexpr std::array<float, 3> tint{1.0f, 0.85f, 0.7f};

// Column-major 4x4 = S * R: first a rotation by angle about z, then x scaled
// by aspect (height / width) so the hexagon stays regular in a non-square
// window - the simplest projection matrix.
std::array<std::array<float, 4>, 4> transform(float angle, float aspect) {
    const float s = std::sin(angle);
    const float c = std::cos(angle);
    return {{
        {aspect * c, s, 0.0f, 0.0f},  // column 0
        {-aspect * s, c, 0.0f, 0.0f}, // column 1
        {0.0f, 0.0f, 1.0f, 0.0f},     // column 2
        {0.0f, 0.0f, 0.0f, 1.0f},     // column 3
    }};
}
// #endregion

// #region reflect
// How a matrix member lies in memory. RowMajor means consecutive floats walk
// along a SPIR-V row; stride (MatrixStride) is the byte distance between two
// of the vectors stored that way.
struct MatrixLayout {
    bool rowMajor;
    uint32_t stride;
    bool operator==(const MatrixLayout&) const = default;
};

// One block member: its name, its Offset decoration and, for a matrix, its
// MatrixLayout.
struct Member {
    std::string name;
    uint32_t offset;
    std::optional<MatrixLayout> matrix;
};

// A block as the SPIR-V module declares it: the struct's name and its
// members, in member order.
struct Block {
    std::string name;
    std::vector<Member> members;
};

// Decodes a SPIR-V literal string: UTF-8 bytes packed little-endian into
// words, NUL-terminated.
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

// Reads the layout of the block bound at (set, binding) straight from the
// SPIR-V the pipeline uses - what reflection libraries (SPIRV-Reflect, slangc
// -reflection-json) do, reduced to the four instructions it needs: the
// variable with DescriptorSet/Binding decorations, its pointer type, the struct
// it points to, and that struct's OpMemberDecorate Offset (plus RowMajor/
// ColMajor and MatrixStride on matrix members). Names come from
// OpName/OpMemberName debug info, which slangc emits by default.
Block reflect(std::span<const uint32_t> words, uint32_t set, uint32_t binding) {
    constexpr uint32_t opName           = 5;
    constexpr uint32_t opMemberName     = 6;
    constexpr uint32_t opTypePointer    = 32;
    constexpr uint32_t opVariable       = 59;
    constexpr uint32_t opDecorate       = 71;
    constexpr uint32_t opMemberDecorate = 72;
    constexpr uint32_t decRowMajor      = 4;
    constexpr uint32_t decColMajor      = 5;
    constexpr uint32_t decMatrixStride  = 7;
    constexpr uint32_t decBinding       = 33;
    constexpr uint32_t decDescriptorSet = 34;
    constexpr uint32_t decOffset        = 35;

    using MemberKey = std::pair<uint32_t, uint32_t>; // (struct id, member index)
    std::map<uint32_t, std::string> names;
    std::map<MemberKey, std::string> memberNames;
    std::map<MemberKey, uint32_t> offsets;
    std::map<MemberKey, bool> rowMajor;
    std::map<MemberKey, uint32_t> strides;
    std::map<uint32_t, uint32_t> pointees;
    std::vector<std::pair<uint32_t, uint32_t>> variables; // (id, type)
    std::map<uint32_t, uint32_t> sets;
    std::map<uint32_t, uint32_t> bindings;
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
        if (op == opName && count >= 2) {
            names[ins[1]] = spvString(ins.subspan(2));
        } else if (op == opMemberName && count >= 3) {
            memberNames[{ins[1], ins[2]}] = spvString(ins.subspan(3));
        } else if (op == opTypePointer && count == 4) {
            pointees[ins[1]] = ins[3];
        } else if (op == opVariable && count >= 3) {
            variables.emplace_back(ins[2], ins[1]);
        } else if (op == opDecorate && count == 4 && ins[2] == decDescriptorSet) {
            sets[ins[1]] = ins[3];
        } else if (op == opDecorate && count == 4 && ins[2] == decBinding) {
            bindings[ins[1]] = ins[3];
        } else if (op == opMemberDecorate && count == 5 && ins[3] == decOffset) {
            offsets[{ins[1], ins[2]}] = ins[4];
        } else if (op == opMemberDecorate && count == 4 && (ins[3] == decRowMajor || ins[3] == decColMajor)) {
            rowMajor[{ins[1], ins[2]}] = ins[3] == decRowMajor;
        } else if (op == opMemberDecorate && count == 5 && ins[3] == decMatrixStride) {
            strides[{ins[1], ins[2]}] = ins[4];
        }
        pos += count;
    }

    const auto var = std::ranges::find_if(variables, [&](const auto& v) {
        const auto s = sets.find(v.first);
        const auto b = bindings.find(v.first);
        return s != sets.end() && s->second == set && b != bindings.end() && b->second == binding;
    });
    if (var == variables.end()) {
        throw std::runtime_error(std::format("no variable at set {}, binding {}", set, binding));
    }
    const auto pointee = pointees.find(var->second);
    if (pointee == pointees.end()) {
        throw std::runtime_error("the variable's type is not a pointer");
    }
    const uint32_t block = pointee->second;
    Block out;
    out.name = names.contains(block) ? names.at(block) : std::string{};
    for (uint32_t i = 0;; ++i) {
        const auto offset = offsets.find({block, i});
        if (offset == offsets.end()) {
            break;
        }
        const auto name = memberNames.find({block, i});
        // A matrix member carries MatrixStride and RowMajor or ColMajor; one
        // without RowMajor counts as ColMajor here.
        std::optional<MatrixLayout> matrix;
        if (const auto stride = strides.find({block, i}); stride != strides.end()) {
            const auto major = rowMajor.find({block, i});
            matrix           = MatrixLayout{.rowMajor = major != rowMajor.end() && major->second, .stride = stride->second};
        }
        out.members.push_back(Member{
            .name   = name != memberNames.end() ? name->second : std::string{},
            .offset = offset->second,
            .matrix = matrix,
        });
    }
    return out;
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

// #region frame-resources
// What the CPU touches while recording one frame: one set per frame in flight
// (step 10), now with the frame's uniform buffer and descriptor set. The CPU
// writes slot N's buffer while the GPU may still read slot N-1's: one buffer
// per slot is what makes that safe.
struct Frame {
    vk::raii::CommandBuffer commandBuffer;
    // Signaled by vkAcquireNextImageKHR, waited on by this slot's submit.
    vk::raii::Semaphore imageAvailable;
    // Signaled when this slot's submission completes.
    vk::raii::Fence inFlight;
    // This slot's FrameUniforms, HOST_VISIBLE | HOST_COHERENT.
    GpuBuffer uniform;
    // uniform's memory, mapped once for the program's lifetime.
    FrameUniforms* mapped;
    // Binding 0 = uniform; freed together with the pool.
    vk::DescriptorSet descriptorSet;
};
// #endregion

// Command line: --seconds S closes the window after S seconds.
std::optional<double> parseSeconds(int argc, char** argv) {
    if (argc == 1) {
        return std::nullopt;
    }
    if (argc == 3 && std::string_view{argv[1]} == "--seconds") {
        const std::string_view text = argv[2];
        double value                = 0.0;
        const auto [end, ec]        = std::from_chars(text.data(), text.data() + text.size(), value);
        if (ec == std::errc{} && end == text.data() + text.size()) {
            return value;
        }
    }
    throw std::runtime_error("usage: 17_uniforms [--seconds S]");
}

int main(int argc, char** argv)
try {
    const std::optional<double> seconds = parseSeconds(argc, argv);

    // ---- Step 06: window (800x600 physical pixels, fixed size) ----
    Glfw glfw;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 17 uniforms", nullptr, nullptr), &glfwDestroyWindow);
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
        .pApplicationName   = "Vulkan from Zero - 17 uniforms",
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
    const vk::PhysicalDeviceProperties chosen      = physicalDevice.getProperties();
    std::println("\nPicked: {}", chosen.deviceName.data());
    std::println("Graphics + present family: {}", family);

    // Step 07's device: one queue, the two 1.3 features, VK_KHR_swapchain.
    // Uniform buffers and descriptor sets are Vulkan 1.0: nothing to enable.
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

    // The SPIR-V is compiled from shaders/transform.slang by `bun run
    // sync:shaders`; CMake bakes the absolute path of shaders/spv into SHADER_DIR.
    const std::vector<uint32_t> spirv = readSpirv(std::filesystem::path{SHADER_DIR} / "transform.spv");
    std::println("\nSPIR-V: {} bytes (transform.spv), entry points vertexUniform, fragmentMain", spirv.size() * sizeof(uint32_t));

    // #region check-layout
    // The shader's view of the block, from its SPIR-V, against the host struct.
    // A mismatch would not fail any Vulkan call: the shader would silently read
    // the wrong bytes. So the program refuses to start.
    const Block block = reflect(spirv, 0, 0);
    std::println("Uniform block at set 0, binding 0: {} (from the SPIR-V)", block.name);
    if (block.members.size() != hostLayout.size()) {
        throw std::runtime_error(
            std::format("the shader's block has {} members, the host struct {}", block.members.size(), hostLayout.size()));
    }
    // A host matrix is 4 columns of 4 floats, 16 bytes apart. Slang's
    // column-major matrix shows up in SPIR-V as RowMajor (an HLSL row is a
    // SPIR-V column, see transform.slang), so the shader must say RowMajor +
    // MatrixStride 16; ColMajor would read the transpose.
    constexpr MatrixLayout hostMatrix{.rowMajor = true, .stride = 16};
    const auto describe = [](const std::optional<MatrixLayout>& m) {
        return m ? std::format("{} MatrixStride {}", m->rowMajor ? "RowMajor" : "ColMajor", m->stride) : std::string{"not a matrix"};
    };
    for (size_t i = 0; i < hostLayout.size(); ++i) {
        const Member& m          = block.members[i];
        const HostField& field   = hostLayout[i];
        std::print("  {:<10} SPIR-V offset {:>2}, host struct offset {:>2}", m.name, m.offset, field.offset);
        if (m.matrix) {
            std::print(", {} (host: 16-byte columns)", describe(m.matrix));
        }
        std::println("");
        if (m.name != field.name || m.offset != field.offset) {
            throw std::runtime_error(
                std::format("layout mismatch: shader {} at {}, host {} at {}", m.name, m.offset, field.name, field.offset));
        }
        const std::optional<MatrixLayout> expected = field.matrix ? std::optional{hostMatrix} : std::nullopt;
        if (m.matrix != expected) {
            throw std::runtime_error(
                std::format("layout mismatch: shader {} is {}, host expects {}", m.name, describe(m.matrix), describe(expected)));
        }
    }
    std::println("Host struct: {} bytes, every offset matches: the CPU can copy it byte for byte", sizeof(FrameUniforms));
    // #endregion

    std::optional<vk::raii::ShaderModule> shaderModule;
    shaderModule.emplace(device, vk::ShaderModuleCreateInfo{
                                     .codeSize = spirv.size() * sizeof(uint32_t),
                                     .pCode    = spirv.data(),
                                 });
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = **shaderModule, .pName = "vertexUniform"},
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = **shaderModule, .pName = "fragmentMain"},
    };

    // Step 14's vertex input: binding 0, stride 20, position + color.
    const vk::VertexInputBindingDescription vertexBinding{
        .binding   = 0,
        .stride    = sizeof(Vertex),
        .inputRate = vk::VertexInputRate::eVertex,
    };
    const std::array attributes{
        vk::VertexInputAttributeDescription{
            .location = 0, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Vertex, position)},
        vk::VertexInputAttributeDescription{
            .location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, color)},
    };
    const vk::PipelineVertexInputStateCreateInfo vertexInput{
        .vertexBindingDescriptionCount   = 1,
        .pVertexBindingDescriptions      = &vertexBinding,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size()),
        .pVertexAttributeDescriptions    = attributes.data(),
    };

    // #region set-layout
    // The "parameter list" of set 0: binding 0 holds ONE uniform buffer
    // descriptor, visible to the vertex stage only (a stage that does not use a
    // binding should not list it: drivers may set up descriptors per stage).
    // Every set allocated with this layout has this shape.
    const vk::DescriptorSetLayoutBinding setBinding{
        .binding         = 0,
        .descriptorType  = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags      = vk::ShaderStageFlagBits::eVertex,
    };
    vk::raii::DescriptorSetLayout setLayout(device, vk::DescriptorSetLayoutCreateInfo{.bindingCount = 1, .pBindings = &setBinding});
    std::println("\nDescriptor set layout: binding 0 = UNIFORM_BUFFER x 1, stage VERTEX");
    // #endregion

    // #region pipeline-layout
    // The pipeline layout lists the set layouts by set number: element 0 of
    // pSetLayouts is set 0, matching [[vk::binding(0, 0)]] in the shader.
    const vk::DescriptorSetLayout setLayoutHandle = *setLayout;
    vk::raii::PipelineLayout pipelineLayout(device, vk::PipelineLayoutCreateInfo{.setLayoutCount = 1, .pSetLayouts = &setLayoutHandle});
    std::println("Pipeline layout: 1 set layout (set 0), no push constants");
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
    std::println("Graphics pipeline created (vertexUniform reads set 0, binding 0)");

    // Step 14's mesh buffers: HOST_VISIBLE | HOST_COHERENT, written once.
    const vk::PhysicalDeviceMemoryProperties mem = physicalDevice.getMemoryProperties();
    const vk::MemoryPropertyFlags hostFlags      = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    const std::span vertexData{vertices};
    const std::span indexData{indices};
    GpuBuffer vertexBuffer = createBuffer(device, mem, vertexData.size_bytes(), vk::BufferUsageFlagBits::eVertexBuffer, hostFlags);
    writeMapped(vertexBuffer.memory, vertexData);
    GpuBuffer indexBuffer = createBuffer(device, mem, indexData.size_bytes(), vk::BufferUsageFlagBits::eIndexBuffer, hostFlags);
    writeMapped(indexBuffer.memory, indexData);
    std::println("\nMesh: regular hexagon, {} vertices, {} indices (step 14's buffers, memory type {})", vertices.size(),
                 indices.size(), vertexBuffer.memoryType);

    // #region uniform-buffers
    // One small buffer per frame in flight, in memory the CPU writes directly
    // (HOST_VISIBLE | HOST_COHERENT), mapped ONCE and kept mapped: mapping is
    // not free, and a persistent pointer is the usual way to feed per-frame
    // data. Usage UNIFORM_BUFFER is what lets a UNIFORM_BUFFER descriptor point
    // at it. Limits that apply: the range a descriptor covers is at most
    // maxUniformBufferRange, and an offset into a shared buffer must be a
    // multiple of minUniformBufferOffsetAlignment (one buffer per frame needs
    // no offset).
    std::println("Limits: maxUniformBufferRange = {}, minUniformBufferOffsetAlignment = {}", chosen.limits.maxUniformBufferRange,
                 chosen.limits.minUniformBufferOffsetAlignment);
    constexpr vk::DeviceSize uniformSize = sizeof(FrameUniforms);
    std::vector<GpuBuffer> uniforms;
    std::vector<FrameUniforms*> mapped;
    for (size_t i = 0; i < framesInFlight; ++i) {
        uniforms.push_back(createBuffer(device, mem, uniformSize, vk::BufferUsageFlagBits::eUniformBuffer, hostFlags));
        // Stays mapped until the memory is freed, which unmaps it implicitly.
        mapped.push_back(static_cast<FrameUniforms*>(uniforms.back().memory.mapMemory(0, vk::WholeSize)));
    }
    std::println("Uniform buffers: {} x {} bytes, UNIFORM_BUFFER, memory type {} ({}), persistently mapped", framesInFlight, uniformSize,
                 uniforms.front().memoryType, memoryNames(hostFlags));
    // #endregion

    // #region descriptor-pool
    // A pool is an arena for descriptor sets: it reserves room for at most
    // maxSets sets holding, in total, the descriptors of pPoolSizes. Here
    // exactly what we allocate: 2 sets with 1 uniform buffer each. Without
    // eFreeDescriptorSet, sets cannot be freed one by one; they all go when the
    // pool is reset or destroyed.
    constexpr auto frameCount = static_cast<uint32_t>(framesInFlight);
    const vk::DescriptorPoolSize poolSize{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = frameCount};
    vk::raii::DescriptorPool descriptorPool(device, vk::DescriptorPoolCreateInfo{
                                                        .maxSets       = frameCount,
                                                        .poolSizeCount = 1,
                                                        .pPoolSizes    = &poolSize,
                                                    });
    std::println("Descriptor pool: maxSets {}, UNIFORM_BUFFER x {}", frameCount, frameCount);
    // #endregion

    // #region allocate-sets
    // One layout entry per set to allocate: both sets have set 0's shape.
    // vk::raii::DescriptorSet would call vkFreeDescriptorSets in its destructor,
    // which needs a pool created with eFreeDescriptorSet; our sets go with the
    // pool, so we keep the plain handles (release()).
    const std::vector<vk::DescriptorSetLayout> setLayouts(framesInFlight, *setLayout);
    std::vector<vk::DescriptorSet> descriptorSets;
    for (vk::raii::DescriptorSet& set : vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo{
                                                                             .descriptorPool     = *descriptorPool,
                                                                             .descriptorSetCount = frameCount,
                                                                             .pSetLayouts        = setLayouts.data(),
                                                                         })) {
        descriptorSets.push_back(set.release());
    }
    std::println("Allocated {} descriptor sets", descriptorSets.size());
    // #endregion

    // #region write-sets
    // A freshly allocated set's descriptors are undefined: write them. Each
    // write fills binding 0, element 0 of one set with a buffer range. One
    // vkUpdateDescriptorSets call takes all the writes; it is a host-side
    // operation, done before any command buffer uses the sets.
    std::vector<vk::DescriptorBufferInfo> bufferInfos;
    for (const GpuBuffer& u : uniforms) {
        bufferInfos.push_back(vk::DescriptorBufferInfo{.buffer = *u.buffer, .offset = 0, .range = uniformSize});
    }
    std::vector<vk::WriteDescriptorSet> writes;
    for (size_t i = 0; i < descriptorSets.size(); ++i) {
        writes.push_back(vk::WriteDescriptorSet{
            .dstSet          = descriptorSets[i],
            .dstBinding      = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo     = &bufferInfos[i],
        });
    }
    device.updateDescriptorSets(writes, nullptr);
    for (size_t i = 0; i < writes.size(); ++i) {
        std::println("vkUpdateDescriptorSets: slot {}'s set, binding 0 -> uniform buffer {} (offset 0, range {})", i, i, uniformSize);
    }
    // #endregion

    // Steps 09-10: the queue, the pool, per-frame buffers and sync objects.
    vk::raii::Queue queue(device, family, 0);
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
    std::println("\n{} frames in flight, each with its own uniform buffer and descriptor set", framesInFlight);
    std::println("Animation: rotation {} rad/s about z, x scaled by height/width = {}", spin, aspect);

    using Clock       = std::chrono::steady_clock;
    using Seconds     = std::chrono::duration<double>;
    const auto start  = Clock::now();
    size_t frameIndex = 0;

    // Renders and presents one frame (step 10's loop). Returns false when the
    // swapchain is out of date (recreation is step 11).
    const auto drawFrame = [&]() -> bool {
        Frame& frame                       = frames.at(frameIndex % frames.size());
        const vk::raii::CommandBuffer& cmd = frame.commandBuffer;

        // The last submission that used this slot must be finished before its
        // command buffer, semaphore AND uniform buffer are reused (step 10).
        if (device.waitForFences(*frame.inFlight, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
            throw std::runtime_error("waitForFences failed");
        }

        // #region update-uniforms
        // The GPU is done with this slot's buffer (the fence above), so the CPU
        // may overwrite it while the other slot's frame may still be read by the
        // GPU. HOST_COHERENT: no flush; the submit below makes the write visible
        // to the vertex shader.
        const auto time = static_cast<float>(Seconds(Clock::now() - start).count());
        const FrameUniforms uniformData{
            .transform = transform(time * spin, aspect),
            .tint      = tint,
            .time      = time,
        };
        std::memcpy(frame.mapped, &uniformData, sizeof uniformData);
        // #endregion

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
        // Bind this slot's set as set 0 of the graphics bind point, through the
        // pipeline layout it was made for (first set 0, no dynamic offsets). The
        // binding stays until another set replaces it; draws recorded after it
        // see the descriptors as they are at submit time.
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
    const auto timeUp = [&] { return seconds && Seconds(Clock::now() - start).count() >= *seconds; };
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
    // Wait until the GPU is idle: the last frames may still read the uniform
    // buffers through their sets. Then RAII destroys in reverse order of
    // declaration: the frames (freeing a mapped allocation unmaps it), the
    // descriptor pool (which frees its sets), the mesh buffers, the pipeline,
    // the pipeline layout and the set layout.
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
