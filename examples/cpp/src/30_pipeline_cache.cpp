// Step 30 - Pipeline cache and pipeline binaries
//
// No window, no drawing: this step only CREATES pipelines and measures how long
// that takes. vkCreateGraphicsPipelines is where the driver compiles SPIR-V
// into GPU machine code, and a real application creates hundreds of pipelines
// (materials x passes x vertex formats). Three ways to avoid paying for the
// compile again:
//
//   VkPipelineCache        an object the driver fills with compiled code;
//                          vkGetPipelineCacheData turns it into bytes the
//                          application stores, and the next run hands them back
//   creation feedback      VkPipelineCreationFeedbackCreateInfo: the driver
//                          reports whether the cache was hit, and how long it took
//   pipeline binaries      VK_KHR_pipeline_binary: the compiled code per pipeline,
//                          as objects the application keys, stores and reloads
//
// The workload: --count N (default 64) variants of one graphics pipeline, made
// different by five specialization constants of a PBR-like fragment shader
// (shaders/pbr_variants.slang). Phases:
//
//   1. no VkPipelineCache
//   2. a VkPipelineCache, empty or loaded from --load FILE (header checked)
//   3. the same VkPipelineCache again; --save FILE writes its data
//
// --probe asks for creation WITHOUT compiling first
// (FAIL_ON_PIPELINE_COMPILE_REQUIRED), --no-driver-cache turns off the
// driver's own internal cache (VK_KHR_pipeline_binary), --binary captures and
// reloads pipeline binaries, --verbose prints every pipeline, --no-validation
// leaves the validation layer out (for timing: the layer adds its own work to
// every create call). Lines that depend on timing start with [timing]. Steps
// 01-05 are repeated, reduced to what this step needs.

#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
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

// #region cli
// Command line, see the comment at the top.
struct Args {
    uint32_t count = 64;
    bool probe     = false;
    std::optional<std::filesystem::path> save;
    std::optional<std::filesystem::path> load;
    bool noDriverCache = false;
    bool binary        = false;
    bool validation    = true;
    bool verbose       = false;
};

// The largest --count: VARIANT is a 32-bit constant, so this is only a bound
// on the run time (a cold compile takes milliseconds per pipeline).
constexpr uint32_t maxCount = 4096;

Args parseArgs(int argc, char** argv) {
    constexpr std::string_view usage = "usage: 30_pipeline_cache [--count N] [--probe] [--save FILE] [--load FILE] "
                                       "[--no-driver-cache] [--binary] [--no-validation] [--verbose]";
    Args args;
    const auto value = [&](int& i) -> std::string_view {
        if (i + 1 >= argc) {
            throw std::runtime_error(std::string{usage});
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--count") {
            const std::string text{value(i)};
            size_t end = 0;
            const unsigned long n = std::stoul(text, &end);
            if (end != text.size() || n > maxCount) {
                throw std::runtime_error(std::format("--count must be 1-{}", maxCount));
            }
            args.count = static_cast<uint32_t>(n);
        } else if (flag == "--probe") {
            args.probe = true;
        } else if (flag == "--save") {
            args.save = std::filesystem::path{value(i)};
        } else if (flag == "--load") {
            args.load = std::filesystem::path{value(i)};
        } else if (flag == "--no-driver-cache") {
            args.noDriverCache = true;
        } else if (flag == "--binary") {
            args.binary = true;
        } else if (flag == "--no-validation") {
            args.validation = false;
        } else if (flag == "--verbose") {
            args.verbose = true;
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

// #region spec-constants
// The five specialization constants of pbr_variants.slang, in SpecId order:
// LIGHTS, OCTAVES, BRDF, TONEMAP, VARIANT (all 32-bit ints in the shader).
constexpr std::array<std::string_view, 5> specNames{"LIGHTS", "OCTAVES", "BRDF", "TONEMAP", "VARIANT"};

// The constant values of pipeline variant. The first 64 variants are every
// combination of 4 light counts x 4 octave counts x 2 BRDFs x 2 tone maps;
// VARIANT = the index makes every pipeline different from every other, also
// beyond 64, so no two of them can share a compiled result.
std::array<int32_t, 5> specValues(uint32_t variant) {
    constexpr std::array<int32_t, 4> pick{1, 2, 4, 8};
    return {pick[variant % 4], pick[(variant / 4) % 4], static_cast<int32_t>(variant / 16 % 2),
            static_cast<int32_t>(variant / 32 % 2), static_cast<int32_t>(variant)};
}

// One map entry per constant: SpecId i reads 4 bytes at offset 4 * i of the
// data block (specValues).
std::array<vk::SpecializationMapEntry, 5> specEntries() {
    std::array<vk::SpecializationMapEntry, 5> entries{};
    for (uint32_t i = 0; i < entries.size(); ++i) {
        entries[i] = vk::SpecializationMapEntry{.constantID = i, .offset = i * 4, .size = sizeof(int32_t)};
    }
    return entries;
}
// #endregion

// How a pipeline is created.
enum class Mode {
    eNormal,        // compile if needed
    eFailIfCompile, // FAIL_ON_PIPELINE_COMPILE_REQUIRED: VK_PIPELINE_COMPILE_REQUIRED instead of compiling
    eCapture,       // VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR: keep what binaries need
    eFromBinaries,  // from pipeline binaries (VkPipelineBinaryInfoKHR)
};

// What the driver reports for one pipeline and its two stages.
struct Feedback {
    vk::PipelineCreationFeedback pipeline{};
    std::array<vk::PipelineCreationFeedback, 2> stages{};
};

// The state every variant shares: the pipeline layout and the two modules.
struct Shared {
    vk::PipelineLayout layout{};
    vk::ShaderModule vertex{};
    vk::ShaderModule fragment{};
};

// The format of the (never created) color attachment the pipelines render to.
constexpr vk::Format colorFormat = vk::Format::eR8G8B8A8Unorm;

// #region create-info
// Builds the VkGraphicsPipelineCreateInfo of variant for mode, hands it to f,
// and returns what f returned plus the creation feedback the driver wrote into
// the chain. Everything the create info points to lives on this function's
// stack, so it stays valid for exactly the duration of f.
template <typename F>
auto withCreateInfo(const Shared& shared, uint32_t variant, Mode mode, std::span<const vk::PipelineBinaryKHR> binaries, F&& f) {
    const std::array<int32_t, 5> values                       = specValues(variant);
    const std::array<vk::SpecializationMapEntry, 5> entries = specEntries();
    const vk::SpecializationInfo spec{
        .mapEntryCount = static_cast<uint32_t>(entries.size()),
        .pMapEntries   = entries.data(),
        .dataSize      = sizeof(values),
        .pData         = values.data(),
    };
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = shared.vertex, .pName = "vertexMain"},
        vk::PipelineShaderStageCreateInfo{
            .stage               = vk::ShaderStageFlagBits::eFragment,
            .module              = shared.fragment,
            .pName               = "fragmentMain",
            .pSpecializationInfo = &spec,
        },
    };
    // A full-screen triangle from SV_VulkanVertexID: no vertex buffers.
    const vk::PipelineVertexInputStateCreateInfo vertexInput{};
    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
    const vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
    const vk::PipelineRasterizationStateCreateInfo rasterization{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode    = vk::CullModeFlagBits::eNone,
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

    // The chain: rendering -> feedback -> (flags2 or binary info).
    // With VkPipelineCreateFlags2CreateInfo in the chain, its 64-bit flags
    // replace VkGraphicsPipelineCreateInfo::flags (Vulkan 1.4 / maintenance5);
    // CAPTURE_DATA exists only there.
    const vk::PipelineCreateFlags2CreateInfo flags2{.flags = vk::PipelineCreateFlagBits2::eCaptureDataKHR};
    const vk::PipelineBinaryInfoKHR binaryInfo{
        .binaryCount       = static_cast<uint32_t>(binaries.size()),
        .pPipelineBinaries = binaries.data(),
    };
    const void* tail = mode == Mode::eCapture ? static_cast<const void*>(&flags2)
                       : mode == Mode::eFromBinaries ? static_cast<const void*>(&binaryInfo)
                                                     : nullptr;
    // Creation feedback (core 1.3): one record for the whole pipeline, one per
    // stage (the count must equal stageCount,
    // VUID-VkGraphicsPipelineCreateInfo-pipelineStageCreationFeedbackCount-06594).
    Feedback feedback;
    const vk::PipelineCreationFeedbackCreateInfo feedbackInfo{
        .pNext                              = tail,
        .pPipelineCreationFeedback          = &feedback.pipeline,
        .pipelineStageCreationFeedbackCount = static_cast<uint32_t>(feedback.stages.size()),
        .pPipelineStageCreationFeedbacks    = feedback.stages.data(),
    };
    const vk::PipelineRenderingCreateInfo rendering{
        .pNext                   = &feedbackInfo,
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &colorFormat,
    };
    const vk::GraphicsPipelineCreateInfo info{
        .pNext               = &rendering,
        .flags               = mode == Mode::eFailIfCompile ? vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired
                                                            : vk::PipelineCreateFlags{},
        .stageCount          = static_cast<uint32_t>(stages.size()),
        .pStages             = stages.data(),
        .pVertexInputState   = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState      = &viewportState,
        .pRasterizationState = &rasterization,
        .pMultisampleState   = &multisample,
        .pColorBlendState    = &colorBlend,
        .pDynamicState       = &dynamicState,
        .layout              = shared.layout,
    };
    auto result = std::invoke(std::forward<F>(f), info);
    return std::pair{std::move(result), feedback};
}
// #endregion

// One created pipeline (or VK_NULL_HANDLE) with its result code, the host time
// of the create call and the driver's feedback.
struct Created {
    vk::raii::Pipeline pipeline{nullptr};
    vk::Result result = vk::Result::eSuccess;
    double hostMs     = 0.0;
    Feedback feedback{};
};

// #region create-pipeline
// Creates pipeline variant with one vkCreateGraphicsPipelines call, timed on the
// host. VK_PIPELINE_COMPILE_REQUIRED is a SUCCESS code (positive), returned
// only with FAIL_ON_PIPELINE_COMPILE_REQUIRED, and then the pipeline is
// VK_NULL_HANDLE; the raii wrapper throws for every other non-success code.
Created createPipeline(const vk::raii::Device& device, const Shared& shared, uint32_t variant, const vk::raii::PipelineCache* cache,
                       Mode mode, std::span<const vk::PipelineBinaryKHR> binaries = {}) {
    auto [timed, feedback] = withCreateInfo(shared, variant, mode, binaries, [&](const vk::GraphicsPipelineCreateInfo& info) {
        using Clock      = std::chrono::steady_clock;
        const auto start = Clock::now();
        vk::raii::Pipeline pipeline = device.createGraphicsPipeline(cache, info);
        return std::pair{std::move(pipeline), std::chrono::duration<double, std::milli>(Clock::now() - start).count()};
    });
    const vk::Result result = timed.first.getConstructorSuccessCode();
    return Created{.pipeline = std::move(timed.first), .result = result, .hostMs = timed.second, .feedback = feedback};
}
// #endregion

// The result's name, as the C headers spell it.
std::string resultName(vk::Result r) {
    return string_VkResult(static_cast<VkResult>(r));
}

// The feedback flags of interest as text.
std::string flagNames(vk::PipelineCreationFeedbackFlags flags) {
    using Bit = vk::PipelineCreationFeedbackFlagBits;
    constexpr std::array<std::pair<Bit, std::string_view>, 3> table{{
        {Bit::eValid, "VALID"},
        {Bit::eApplicationPipelineCacheHit, "APPLICATION_PIPELINE_CACHE_HIT"},
        {Bit::eBasePipelineAcceleration, "BASE_PIPELINE_ACCELERATION"},
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

// Median of a non-empty list of times, in milliseconds.
double medianMs(std::vector<double> times) {
    std::ranges::sort(times);
    const size_t mid = times.size() / 2;
    return times.size() % 2 == 0 ? (times[mid - 1] + times[mid]) / 2.0 : times[mid];
}

// #region phase
// Creates all count variants into cache (nullptr = no cache) and prints what the
// driver reported and how long it took. The pipelines are returned to the
// caller (binary capture uses them).
std::vector<vk::raii::Pipeline> runPhase(const vk::raii::Device& device, const Shared& shared, uint32_t count,
                                         const vk::raii::PipelineCache* cache, Mode mode, bool verbose,
                                         const std::vector<std::vector<vk::PipelineBinaryKHR>>& binaries = {}) {
    std::vector<Created> created;
    for (uint32_t variant = 0; variant < count; ++variant) {
        const std::span<const vk::PipelineBinaryKHR> own =
            mode == Mode::eFromBinaries ? std::span<const vk::PipelineBinaryKHR>{binaries.at(variant)} : std::span<const vk::PipelineBinaryKHR>{};
        Created c = createPipeline(device, shared, variant, cache, mode, own);
        if (c.result != vk::Result::eSuccess) {
            throw std::runtime_error(std::format("variant {}: {}", variant, resultName(c.result)));
        }
        created.push_back(std::move(c));
    }
    using Bit       = vk::PipelineCreationFeedbackFlagBits;
    const auto hits = [&](auto flagsOf) {
        return std::ranges::count_if(created, [&](const Created& c) { return static_cast<bool>(flagsOf(c.feedback) & Bit::eApplicationPipelineCacheHit); });
    };
    const auto valid = std::ranges::count_if(created, [](const Created& c) { return static_cast<bool>(c.feedback.pipeline.flags & Bit::eValid); });
    std::println("  feedback: VALID {}/{}, APPLICATION_PIPELINE_CACHE_HIT {}/{} (stages: vertex {}/{}, fragment {}/{})", valid, count,
                 hits([](const Feedback& f) { return f.pipeline.flags; }), count, hits([](const Feedback& f) { return f.stages[0].flags; }),
                 count, hits([](const Feedback& f) { return f.stages[1].flags; }), count);
    std::vector<double> host;
    std::vector<double> driver;
    for (const Created& c : created) {
        host.push_back(c.hostMs);
        driver.push_back(static_cast<double>(c.feedback.pipeline.duration) / 1e6);
    }
    double total = 0.0;
    for (const double t : host) {
        total += t;
    }
    const auto [minIt, maxIt] = std::ranges::minmax_element(host);
    std::println("[timing]   host: {:.1f} ms total, per pipeline median {:.3f} ms (min {:.3f}, max {:.3f}); driver-reported median {:.3f} ms",
                 total, medianMs(host), *minIt, *maxIt, medianMs(driver));
    if (verbose) {
        for (uint32_t variant = 0; variant < count; ++variant) {
            const std::array<int32_t, 5> v = specValues(variant);
            std::string constants;
            for (size_t i = 0; i < 4; ++i) {
                constants += std::format("{}{} {}", i == 0 ? "" : ", ", specNames[i], v[i]);
            }
            const Created& c = created[variant];
            std::println("[timing]     variant {} ({}): host {:.3f} ms, driver {:.3f} ms, {}", variant, constants, c.hostMs,
                         static_cast<double>(c.feedback.pipeline.duration) / 1e6, flagNames(c.feedback.pipeline.flags));
        }
    }
    std::vector<vk::raii::Pipeline> pipelines;
    for (Created& c : created) {
        pipelines.push_back(std::move(c.pipeline));
    }
    return pipelines;
}
// #endregion

// #region header-check
// VkPipelineCacheHeaderVersionOne: the first 32 bytes of pipeline cache data,
// every field little-endian regardless of the host ("written with the least
// significant byte first").
struct CacheHeader {
    uint32_t headerSize    = 0;
    uint32_t headerVersion = 0;
    uint32_t vendorID      = 0;
    uint32_t deviceID      = 0;
    std::array<uint8_t, VK_UUID_SIZE> uuid{};
};

// The header of data, or nullopt if it is shorter than 32 bytes.
std::optional<CacheHeader> parseHeader(std::span<const uint8_t> data) {
    if (data.size() < 32) {
        return std::nullopt;
    }
    const auto word = [&](size_t at) {
        return uint32_t{data[at]} | uint32_t{data[at + 1]} << 8 | uint32_t{data[at + 2]} << 16 | uint32_t{data[at + 3]} << 24;
    };
    CacheHeader h;
    h.headerSize    = word(0);
    h.headerVersion = word(4);
    h.vendorID      = word(8);
    h.deviceID      = word(12);
    std::copy_n(data.begin() + 16, h.uuid.size(), h.uuid.begin());
    return h;
}

// The UUID as text, its 16 bytes in stored order, grouped 4-2-2-2-6.
std::string uuidText(std::span<const uint8_t, VK_UUID_SIZE> uuid) {
    std::string out;
    for (size_t i = 0; i < uuid.size(); ++i) {
        out += (i == 4 || i == 6 || i == 8 || i == 10) ? "-" : "";
        out += std::format("{:02x}", uuid[i]);
    }
    return out;
}

void printHeader(const CacheHeader& h) {
    std::println("  header: headerSize {}, headerVersion {}, vendorID 0x{:04X}, deviceID 0x{:04X}, pipelineCacheUUID {}", h.headerSize,
                 h.headerVersion, h.vendorID, h.deviceID, uuidText(h.uuid));
}

// Accepts cache data only if its header describes this device: the driver
// would treat foreign data as an empty cache anyway ("If the pipeline cache
// data is incompatible ... the pipeline cache will be initially empty"), but
// checking here lets the program say WHY it starts empty, and keeps bytes that
// did not come from vkGetPipelineCacheData away from the driver
// (VUID-VkPipelineCacheCreateInfo-initialDataSize-00769). Returns the reason
// for a rejection, or nullopt.
std::optional<std::string> checkHeader(std::span<const uint8_t> data, const vk::PhysicalDeviceProperties& props) {
    const std::optional<CacheHeader> parsed = parseHeader(data);
    if (!parsed) {
        return std::format("{} bytes, shorter than the 32-byte header", data.size());
    }
    const CacheHeader& h = *parsed;
    printHeader(h);
    if (h.headerSize != 32) {
        // VUID-VkPipelineCacheHeaderVersionOne-headerSize-04967: must be 32.
        return std::format("headerSize {} (must be 32)", h.headerSize);
    }
    if (h.headerVersion != static_cast<uint32_t>(vk::PipelineCacheHeaderVersion::eOne)) {
        // VUID-VkPipelineCacheHeaderVersionOne-headerVersion-04968.
        return std::format("headerVersion {} (this program reads version ONE = 1)", h.headerVersion);
    }
    if (h.vendorID != props.vendorID || h.deviceID != props.deviceID) {
        return std::format("written for vendorID 0x{:04X} deviceID 0x{:04X}, this GPU is 0x{:04X} 0x{:04X}", h.vendorID, h.deviceID,
                           props.vendorID, props.deviceID);
    }
    // Compare as two fixed-size std::arrays (ArrayWrapper1D derives from one): the size is a compile-time 16, whereas
    // std::ranges::equal over the wrapper made GCC 14 -O3 derive an unbounded memcmp length (-Wstringop-overread).
    if (h.uuid != static_cast<const std::array<uint8_t, VK_UUID_SIZE>&>(props.pipelineCacheUUID)) {
        return "pipelineCacheUUID differs (another driver build wrote it)";
    }
    return std::nullopt;
}
// #endregion

// #region load-cache
// The initial data of the phase-2 cache: --load FILE if the file exists and its
// header matches this device, else nothing (an empty cache). A missing or
// rejected file is the normal first-run case for an application, so it is
// reported, not an error; a file that exists but cannot be read is.
std::vector<uint8_t> loadCache(const std::filesystem::path& path, const vk::PhysicalDeviceProperties& props) {
    if (!std::filesystem::exists(path)) {
        std::println("Loading {}: not found; starting with an empty cache", path.string());
        return {};
    }
    std::ifstream file(path, std::ios::binary);
    std::vector<uint8_t> data{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    if (!file && !file.eof()) {
        throw std::runtime_error(path.string() + ": cannot read the file");
    }
    std::println("Loading {}: {} bytes", path.string(), data.size());
    if (const auto why = checkHeader(data, props)) {
        std::println("  rejected: {}; starting with an empty cache", *why);
        return {};
    }
    std::println("  accepted: the header matches this device");
    return data;
}
// #endregion

// #region save-cache
// Writes data to path through a temporary file and a rename, so a crash
// halfway never leaves a truncated cache behind for the next run.
void saveCache(const std::filesystem::path& path, std::span<const uint8_t> data) {
    std::filesystem::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        file.close();
        if (!file) {
            throw std::runtime_error("cannot write " + tmp.string());
        }
    }
    std::filesystem::rename(tmp, path);
    std::println("Saved {} bytes to {}", data.size(), path.string());
}
// #endregion

// #region binary-api
// The pipeline key (vkGetPipelineKeyKHR) of a create info, or with nullptr the
// GLOBAL key: if that changes between saving and loading, every saved binary is
// invalid.
vk::PipelineBinaryKeyKHR pipelineKey(const vk::raii::Device& device, const vk::GraphicsPipelineCreateInfo* info) {
    if (info == nullptr) {
        return device.getPipelineKeyKHR();
    }
    // VkPipelineCreateInfoKHR only points at the real create info (pNext is
    // declared non-const in the C struct; the driver only reads it).
    const vk::PipelineCreateInfoKHR wrapper{.pNext = const_cast<vk::GraphicsPipelineCreateInfo*>(info)};
    return device.getPipelineKeyKHR(wrapper);
}

// A key as a comparable value: its keySize plus the 32 bytes with everything
// past keySize zeroed. Fixed-size storage keeps every comparison bounded at
// compile time (vectors of bytes made GCC 14 -O3 report an unbounded memcmp
// in std::ranges::sort, -Wstringop-overread).
using KeyValue = std::pair<uint32_t, std::array<uint8_t, VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR>>;
KeyValue keyValue(const vk::PipelineBinaryKeyKHR& key) {
    KeyValue v{std::min<uint32_t>(key.keySize, VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR), {}};
    std::copy_n(key.key.begin(), v.first, v.second.begin());
    return v;
}
// #endregion

// One captured binary: what an application would write to disk.
struct SavedBinary {
    vk::PipelineBinaryKeyKHR key{};
    std::vector<uint8_t> data;
};

// The --binary part: capture binaries from freshly created pipelines, drop
// everything, recreate the pipelines from the saved bytes.
void binaries(const vk::raii::Device& device, const Shared& shared, const Args& args) {
    using Clock          = std::chrono::steady_clock;
    const auto msSince   = [](Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); };
    const uint32_t count = args.count;
    std::println("\nPipeline binaries (VK_KHR_pipeline_binary):");

    // #region binary-keys
    // The global key identifies the driver's binary format: an application
    // stores it next to the binaries and discards them if it changed. The
    // pipeline key identifies one create info (here: one variant), the name
    // under which its binaries are stored.
    const vk::PipelineBinaryKeyKHR global = pipelineKey(device, nullptr);
    std::println("  global key: {} bytes (stored with the binaries; a different key on load = discard them)", global.keySize);
    std::vector<KeyValue> keys;
    for (uint32_t variant = 0; variant < count; ++variant) {
        const vk::PipelineBinaryKeyKHR key =
            withCreateInfo(shared, variant, Mode::eNormal, {}, [&](const vk::GraphicsPipelineCreateInfo& info) { return pipelineKey(device, &info); })
                .first;
        keys.push_back(keyValue(key));
    }
    std::vector<KeyValue> distinct = keys;
    std::ranges::sort(distinct);
    distinct.erase(std::ranges::unique(distinct).begin(), distinct.end());
    std::println("  pipeline keys: {} create infos -> {} distinct keys", count, distinct.size());
    // #endregion

    // #region binary-capture
    // CAPTURE_DATA makes each pipeline keep what vkCreatePipelineBinariesKHR
    // needs; such pipelines must be created without a VkPipelineCache
    // (VUID-vkCreateGraphicsPipelines-pNext-09617).
    std::println("\n  Capture: {} pipelines with CAPTURE_DATA, no VkPipelineCache", count);
    std::vector<std::vector<SavedBinary>> saved;
    size_t total = 0;
    size_t bytes = 0;
    {
        const std::vector<vk::raii::Pipeline> pipelines = runPhase(device, shared, count, nullptr, Mode::eCapture, args.verbose);
        const auto start                                = Clock::now();
        for (const vk::raii::Pipeline& pipeline : pipelines) {
            // The raii wrapper runs the two-call idiom (count, then handles) and
            // returns an empty vector for VK_INCOMPLETE / VK_PIPELINE_BINARY_MISSING_KHR.
            const std::vector<vk::raii::PipelineBinaryKHR> handles =
                device.createPipelineBinariesKHR(vk::PipelineBinaryCreateInfoKHR{.pipeline = *pipeline});
            if (handles.empty()) {
                throw std::runtime_error("vkCreatePipelineBinariesKHR returned no binaries");
            }
            std::vector<SavedBinary> perPipeline;
            for (const vk::raii::PipelineBinaryKHR& b : handles) {
                auto [key, data] = device.getPipelineBinaryDataKHR(vk::PipelineBinaryDataInfoKHR{.pipelineBinary = *b});
                bytes += data.size();
                perPipeline.push_back(SavedBinary{.key = key, .data = std::move(data)});
            }
            device.releaseCapturedPipelineDataKHR(vk::ReleaseCapturedPipelineDataInfoKHR{.pipeline = *pipeline});
            total += perPipeline.size();
            saved.push_back(std::move(perPipeline));
        }
        const double extractMs = msSince(start);
        const bool same        = std::ranges::all_of(saved, [&](const auto& s) { return s.size() == saved.front().size(); });
        std::println("  captured: {} binaries ({}), {} bytes of binary data", total,
                     same ? std::format("{} per pipeline", saved.front().size()) : std::string{"a different number per pipeline"}, bytes);
        std::println("[timing]   vkCreatePipelineBinariesKHR + vkGetPipelineBinaryDataKHR for all: {:.1f} ms", extractMs);
    } // the capture pipelines and binaries are destroyed here
    // #endregion

    // #region binary-reload
    // What a later run does with the stored bytes: binaries from keys + data,
    // then pipelines from the binaries. The driver must use the binaries
    // "instead of recalculating" (spec of VkPipelineBinaryInfoKHR): no compile,
    // and no VkPipelineCache (VUID-vkCreateGraphicsPipelines-pNext-09616).
    const auto start = Clock::now();
    std::vector<std::vector<vk::raii::PipelineBinaryKHR>> owned;
    std::vector<std::vector<vk::PipelineBinaryKHR>> perVariant;
    for (const std::vector<SavedBinary>& stored : saved) {
        std::vector<vk::PipelineBinaryKeyKHR> storedKeys;
        std::vector<vk::PipelineBinaryDataKHR> data;
        for (const SavedBinary& b : stored) {
            storedKeys.push_back(b.key);
            // Declared void* in C; the driver only reads it.
            data.push_back(vk::PipelineBinaryDataKHR{.dataSize = b.data.size(), .pData = const_cast<uint8_t*>(b.data.data())});
        }
        const vk::PipelineBinaryKeysAndDataKHR keysAndData{
            .binaryCount         = static_cast<uint32_t>(stored.size()),
            .pPipelineBinaryKeys = storedKeys.data(),
            .pPipelineBinaryData = data.data(),
        };
        std::vector<vk::raii::PipelineBinaryKHR> handles =
            device.createPipelineBinariesKHR(vk::PipelineBinaryCreateInfoKHR{.pKeysAndDataInfo = &keysAndData});
        if (handles.size() != stored.size()) {
            throw std::runtime_error("vkCreatePipelineBinariesKHR did not create every binary");
        }
        std::vector<vk::PipelineBinaryKHR> plain;
        for (const vk::raii::PipelineBinaryKHR& h : handles) {
            plain.push_back(*h);
        }
        perVariant.push_back(std::move(plain));
        owned.push_back(std::move(handles));
    }
    std::println("\n  Reload: {} binaries from the saved keys + data, then {} pipelines from them", total, count);
    std::println("[timing]   vkCreatePipelineBinariesKHR from data for all: {:.1f} ms", msSince(start));
    const std::vector<vk::raii::Pipeline> reloaded =
        runPhase(device, shared, count, nullptr, Mode::eFromBinaries, args.verbose, perVariant);
    // #endregion
}

// Whether physicalDevice offers the device extension name.
bool hasExtension(const vk::raii::PhysicalDevice& physicalDevice, std::string_view name) {
    return std::ranges::any_of(physicalDevice.enumerateDeviceExtensionProperties(),
                               [&](const vk::ExtensionProperties& e) { return std::string_view{e.extensionName.data()} == name; });
}

int main(int argc, char** argv)
try {
    const Args args = parseArgs(argc, argv);

    // ---- Steps 01-02: instance, with validation unless --no-validation ----
    vk::raii::Context context;
    constexpr const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    if (args.validation) {
        const std::vector<vk::LayerProperties> layers = context.enumerateInstanceLayerProperties();
        const bool hasValidation                      = std::ranges::any_of(
            layers, [&](const vk::LayerProperties& l) { return std::string_view{l.layerName.data()} == validationLayer; });
        if (!hasValidation) {
            throw std::runtime_error(std::string{validationLayer} + " not found - install the Vulkan SDK");
        }
    }
    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName   = "Vulkan from Zero - 30 pipeline cache",
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
    const vk::InstanceCreateInfo instanceInfo{
        .pNext                   = args.validation ? &messengerInfo : nullptr,
        .pApplicationInfo        = &appInfo,
        .enabledLayerCount       = args.validation ? static_cast<uint32_t>(layerNames.size()) : 0,
        .ppEnabledLayerNames     = layerNames.data(),
        .enabledExtensionCount   = args.validation ? static_cast<uint32_t>(extensions.size()) : 0,
        .ppEnabledExtensionNames = extensions.data(),
    };
    vk::raii::Instance instance(context, instanceInfo);
    std::optional<vk::raii::DebugUtilsMessengerEXT> messenger;
    if (args.validation) {
        messenger.emplace(instance, messengerInfo);
    }

    const std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }
    std::println("Found {} GPU(s)", devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        std::println("GPU {}: {}", i, devices[i].getProperties().deviceName.data());
    }
    // ---- Step 05's choice: Vulkan 1.4 with dynamicRendering and
    // pipelineCreationCacheControl (both core 1.3 features, required by 1.3+;
    // checked anyway), discrete first ----
    const vk::raii::PhysicalDevice* best = nullptr;
    int bestScore                        = 0;
    for (const vk::raii::PhysicalDevice& device : devices) {
        const vk::PhysicalDeviceProperties props = device.getProperties();
        if (props.apiVersion < vk::ApiVersion14) {
            continue;
        }
        const auto featureChain = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>();
        const auto& f13         = featureChain.get<vk::PhysicalDeviceVulkan13Features>();
        if (f13.dynamicRendering != vk::True || f13.pipelineCreationCacheControl != vk::True) {
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
        throw std::runtime_error("No GPU supports Vulkan 1.4 with dynamicRendering and pipelineCreationCacheControl");
    }
    const vk::raii::PhysicalDevice& physicalDevice = *best;
    const vk::PhysicalDeviceProperties props       = physicalDevice.getProperties();
    std::println("\nPicked: {}", props.deviceName.data());
    std::println("Validation: {}", args.validation ? "on (VK_LAYER_KHRONOS_validation)"
                                                   : "off (--no-validation: the timings below are the driver's alone)");

    // #region cache-identity
    // What a pipeline cache header must match: compiled code depends on the
    // vendor, the device and the driver build; pipelineCacheUUID is the
    // driver's name for "my cache format and compiler" (it changes with driver
    // updates on NVIDIA, so an updated driver starts cold).
    std::println("\nPipeline cache identity (VkPhysicalDeviceProperties):");
    std::println("  vendorID 0x{:04X}, deviceID 0x{:04X}, pipelineCacheUUID {}", props.vendorID, props.deviceID,
                 uuidText(props.pipelineCacheUUID));
    // #endregion

    // #region device
    // VK_KHR_pipeline_binary is needed for --binary, and for --no-driver-cache:
    // VkDevicePipelineBinaryInternalCacheControlKHR is the only standard way to
    // switch off a driver's internal cache, which otherwise makes a "cold" run
    // warm (an implementation may keep compiled pipelines on its own, in memory
    // and on disk).
    const bool binarySupported = hasExtension(physicalDevice, vk::KHRPipelineBinaryExtensionName);
    vk::PhysicalDevicePipelineBinaryPropertiesKHR binProps{};
    if (binarySupported) {
        const auto chain =
            physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDevicePipelineBinaryPropertiesKHR>();
        binProps        = chain.get<vk::PhysicalDevicePipelineBinaryPropertiesKHR>();
        const auto yes  = [](vk::Bool32 b) { return b ? "true" : "false"; };
        std::println("VK_KHR_pipeline_binary: internalCache {}, internalCacheControl {}, prefersInternalCache {}, "
                     "precompiledInternalCache {}, compressedData {}",
                     yes(binProps.pipelineBinaryInternalCache), yes(binProps.pipelineBinaryInternalCacheControl),
                     yes(binProps.pipelineBinaryPrefersInternalCache), yes(binProps.pipelineBinaryPrecompiledInternalCache),
                     yes(binProps.pipelineBinaryCompressedData));
    } else {
        std::println("VK_KHR_pipeline_binary: not supported");
    }
    const bool needBinaryExt = args.binary || args.noDriverCache;
    if (needBinaryExt && !binarySupported) {
        throw std::runtime_error("--binary and --no-driver-cache need VK_KHR_pipeline_binary");
    }
    if (args.noDriverCache && !binProps.pipelineBinaryInternalCacheControl) {
        // VUID-VkDevicePipelineBinaryInternalCacheControlKHR-disableInternalCache-09602.
        throw std::runtime_error("this driver does not allow disabling its internal cache");
    }

    // Pipelines belong to the device, not to a queue, and nothing is
    // submitted; a device still needs one queue to be created with.
    const std::vector<vk::QueueFamilyProperties> families = physicalDevice.getQueueFamilyProperties();
    const auto graphics                                   = std::ranges::find_if(
        families, [](const vk::QueueFamilyProperties& f) { return static_cast<bool>(f.queueFlags & vk::QueueFlagBits::eGraphics); });
    if (graphics == families.end()) {
        throw std::runtime_error("No graphics queue family");
    }
    constexpr float queuePriority = 1.0f;
    const vk::DeviceQueueCreateInfo queueInfo{
        .queueFamilyIndex = static_cast<uint32_t>(graphics - families.begin()),
        .queueCount       = 1,
        .pQueuePriorities = &queuePriority,
    };
    vk::DevicePipelineBinaryInternalCacheControlKHR cacheControl{.disableInternalCache = vk::True};
    vk::PhysicalDevicePipelineBinaryFeaturesKHR binaryFeatures{.pipelineBinaries = vk::True};
    vk::PhysicalDeviceVulkan13Features features13{.pipelineCreationCacheControl = vk::True, .dynamicRendering = vk::True};
    void* chain = nullptr;
    if (args.noDriverCache) {
        cacheControl.pNext = chain;
        chain              = &cacheControl;
    }
    if (args.binary) {
        binaryFeatures.pNext = chain;
        chain                = &binaryFeatures;
    }
    features13.pNext = chain;
    const std::array deviceExtensions{vk::KHRPipelineBinaryExtensionName};
    vk::raii::Device device(physicalDevice, vk::DeviceCreateInfo{
                                                .pNext                   = &features13,
                                                .queueCreateInfoCount    = 1,
                                                .pQueueCreateInfos       = &queueInfo,
                                                .enabledExtensionCount   = needBinaryExt ? 1u : 0u,
                                                .ppEnabledExtensionNames = deviceExtensions.data(),
                                            });
    std::println("VkDevice created with dynamicRendering, pipelineCreationCacheControl{}", args.binary ? ", pipelineBinaries" : "");
    std::println("Driver internal cache: {}", args.noDriverCache ? "DISABLED (--no-driver-cache: VkDevicePipelineBinaryInternalCacheControlKHR)"
                                                                 : "default (whatever the driver does)");
    // #endregion

    // The SPIR-V is compiled from shaders/pbr_variants.slang by `bun run
    // sync:shaders`; CMake bakes the absolute path of shaders/spv into SHADER_DIR.
    const std::filesystem::path spvPath = std::filesystem::path{SHADER_DIR} / "pbr_variants.spv";
    std::ifstream spvFile(spvPath, std::ios::binary);
    const std::vector<char> spvBytes{std::istreambuf_iterator<char>(spvFile), std::istreambuf_iterator<char>()};
    if (spvBytes.empty() || spvBytes.size() % 4 != 0) {
        throw std::runtime_error("cannot read " + spvPath.string() + " (run `bun run sync:shaders`)");
    }
    std::vector<uint32_t> words(spvBytes.size() / 4);
    std::memcpy(words.data(), spvBytes.data(), spvBytes.size());
    // One module holds both entry points; the pipelines reference it twice.
    const vk::raii::ShaderModule module(device, vk::ShaderModuleCreateInfo{.codeSize = spvBytes.size(), .pCode = words.data()});
    // No descriptors, no push constants: the fragment shader needs only its
    // specialization constants and the interpolated uv.
    const vk::raii::PipelineLayout layout(device, vk::PipelineLayoutCreateInfo{});
    const Shared shared{.layout = *layout, .vertex = *module, .fragment = *module};
    std::string specList;
    for (const std::string_view name : specNames) {
        specList += std::format("{}{}", specList.empty() ? "" : ", ", name);
    }
    std::println("\nShader pbr_variants.spv: {} bytes, entry points vertexMain + fragmentMain, specialization constants {}", spvBytes.size(),
                 specList);
    std::println("{} pipeline variants (dynamic rendering, color {}); the first 64 are LIGHTS 1/2/4/8 x OCTAVES 1/2/4/8 x BRDF 0/1 x "
                 "TONEMAP 0/1",
                 args.count, std::string_view{string_VkFormat(static_cast<VkFormat>(colorFormat))}.substr(std::string_view{"VK_FORMAT_"}.size()));

    // #region probe
    // Can the pipeline be created WITHOUT compiling? With
    // FAIL_ON_PIPELINE_COMPILE_REQUIRED the driver answers instead of
    // compiling: VK_PIPELINE_COMPILE_REQUIRED and VK_NULL_HANDLE if it would
    // have to compile (an engine would then compile on a worker thread and draw
    // with a fallback meanwhile). The probe uses its own cache and the variant
    // right after the phases' range, so it warms nothing they create.
    if (args.probe) {
        const uint32_t variant = args.count;
        std::println("\nProbe (variant {}, its own empty VkPipelineCache, FAIL_ON_PIPELINE_COMPILE_REQUIRED):", variant);
        const vk::raii::PipelineCache cache(device, vk::PipelineCacheCreateInfo{});
        const auto handle = [](const Created& c) { return *c.pipeline ? "a pipeline" : "VK_NULL_HANDLE"; };
        const Created first = createPipeline(device, shared, variant, &cache, Mode::eFailIfCompile);
        std::println("  1. empty cache: {}, pipeline = {}", resultName(first.result), handle(first));
        if (first.result == vk::Result::eSuccess) {
            std::println("     (nothing to compile although the application's cache is empty: the driver's internal cache had it)");
        }
        const Created normal = createPipeline(device, shared, variant, &cache, Mode::eNormal);
        std::println("  2. without the flag, same cache: {} (created, and now in the cache)", resultName(normal.result));
        const Created again = createPipeline(device, shared, variant, &cache, Mode::eFailIfCompile);
        std::println("  3. with the flag again: {}, pipeline = {}", resultName(again.result), handle(again));
    }
    // #endregion

    // #region phases
    std::println("\nPhase 1: no VkPipelineCache (pipelineCache = VK_NULL_HANDLE)");
    runPhase(device, shared, args.count, nullptr, Mode::eNormal, args.verbose);

    std::vector<uint8_t> initial;
    if (args.load) {
        std::println();
        initial = loadCache(*args.load, props);
    }
    const vk::raii::PipelineCache cache(device, vk::PipelineCacheCreateInfo{.initialDataSize = initial.size(), .pInitialData = initial.data()});
    if (initial.empty()) {
        std::println("\nPhase 2: a new, empty VkPipelineCache");
    } else {
        std::println("\nPhase 2: VkPipelineCache created from the file ({} bytes)", initial.size());
    }
    runPhase(device, shared, args.count, &cache, Mode::eNormal, args.verbose);

    std::println("\nPhase 3: the same VkPipelineCache again");
    runPhase(device, shared, args.count, &cache, Mode::eNormal, args.verbose);
    // #endregion

    // #region cache-data
    // vkGetPipelineCacheData: everything the cache holds, as bytes that start
    // with the 32-byte header. The raii wrapper runs the two-call idiom (size,
    // then data).
    const std::vector<uint8_t> data = cache.getData();
    std::println("\nVkPipelineCache data: {} bytes", data.size());
    if (const auto why = checkHeader(data, props)) {
        throw std::runtime_error("our own cache data: " + *why);
    }
    if (args.save) {
        saveCache(*args.save, data);
    }
    // #endregion

    if (args.binary) {
        binaries(device, shared, args);
    }

    // #region cleanup
    // Nothing was submitted, so nothing has to be waited for: RAII destroys
    // every object in reverse order of declaration.
    // #endregion
    return EXIT_SUCCESS;
} catch (const vk::SystemError& e) {
    std::println(stderr, "Vulkan error: {}", e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return EXIT_FAILURE;
}
