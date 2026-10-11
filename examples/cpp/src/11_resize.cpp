// Step 11 - Swapchain recreation
//
// A swapchain is made for one window size. When the window changes, the
// presentation engine says so: vkAcquireNextImageKHR or vkQueuePresentKHR return
// VK_ERROR_OUT_OF_DATE_KHR (the swapchain can no longer be used) or
// VK_SUBOPTIMAL_KHR (it still works, but no longer matches exactly), and the
// window system reports the new size. Then we build a new swapchain for the new
// size, handing the old one over (oldSwapchain), and destroy the old one once
// nothing uses it any more - like re-rendering a plate at a new resolution while
// the old version is still on screen.
//
// Knowing when "nothing uses it" is the hard part: the presentation engine
// gives no signal by default. Two ways are shown:
// - present fences (VK_KHR_swapchain_maintenance1): every present signals a
//   fence; when the fences of all presents to the old swapchain have signaled,
//   it is destroyed. The rendering never stops.
// - --no-present-fences (and on GPUs without the extension): vkDeviceWaitIdle
//   before every recreation, then destroy the old one at once.
//
// A minimized window has a 0x0 surface, and no swapchain can have that size:
// rendering pauses until the window comes back. --demo-resize resizes,
// minimizes and restores the window by itself; --seconds S closes it after S
// seconds. Lines that depend on timing start with [timing]. Steps 01-10 are
// repeated, reduced to what this step needs.

// OUT_OF_DATE is an expected event here: get it back as a vk::Result (step 10).
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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <print>
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

// Reads a .spv file into 32-bit words (step 08).
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

// #region cli
// Command line:
// - --no-present-fences: recreate with vkDeviceWaitIdle even when present
//   fences are available;
// - --demo-resize: resize, minimize and restore the window by script;
// - --seconds S: close the window after S seconds (default: run until closed).
struct Args {
    bool presentFences = true;
    bool demo          = false;
    std::optional<double> seconds;
};

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--no-present-fences") {
            args.presentFences = false;
        } else if (arg == "--demo-resize") {
            args.demo = true;
        } else if (arg == "--seconds") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--seconds needs a value");
            }
            const std::string_view text = argv[++i];
            double value{};
            const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (ec != std::errc{} || end != text.data() + text.size()) {
                throw std::runtime_error("--seconds: invalid value " + std::string{text});
            }
            args.seconds = value;
        } else {
            throw std::runtime_error("unknown argument " + std::string{arg} +
                                     " (use --no-present-fences, --demo-resize, --seconds S)");
        }
    }
    return args;
}
// #endregion

// Frames in flight, as in step 10.
constexpr size_t frameCount = 2;

// How long to wait for a present fence before falling back to vkDeviceWaitIdle
// (see wait-present-fence).
constexpr std::chrono::seconds presentFenceTimeout{1};
constexpr uint64_t presentFenceTimeoutNs = std::chrono::nanoseconds(presentFenceTimeout).count();

// #region demo
// One scripted window change of --demo-resize: at `at` seconds after start.
enum class DemoKind { Resize, Minimize, Restore };
struct DemoAction {
    double at;
    DemoKind kind;
    int width  = 0;
    int height = 0;
};
constexpr std::array demo{
    DemoAction{1.0, DemoKind::Resize, 1000, 700},
    DemoAction{2.0, DemoKind::Resize, 640, 480},
    DemoAction{3.0, DemoKind::Minimize},
    DemoAction{4.0, DemoKind::Restore},
};
// #endregion

// #region chain
// Everything that depends on the swapchain and must be rebuilt with it: the
// swapchain, its images and their views, and one renderFinished semaphore per
// image (the present waits on it, so it lives as long as the presents).
// Members are destroyed in reverse order: semaphores, views, then the swapchain.
struct Chain {
    vk::raii::SwapchainKHR swapchain = nullptr;
    std::vector<vk::Image> images;
    std::vector<vk::raii::ImageView> views;
    std::vector<vk::raii::Semaphore> renderFinished;
    vk::Extent2D extent;
};

// An old chain waiting to be destroyed: replaced at frame retiredAt, so every
// present that used it belongs to an earlier frame.
struct Retired {
    Chain chain;
    uint64_t retiredAt = 0;
};
// #endregion

// #region frame-resources
// Step 10's per-frame objects, plus the fence the present of this slot signals.
struct Frame {
    vk::raii::CommandBuffer commandBuffer;
    vk::raii::Semaphore imageAvailable;
    vk::raii::Fence inFlight;
    // Signaled when the presentation engine no longer needs what this slot's
    // last present referred to. Null on the vkDeviceWaitIdle path.
    vk::raii::Fence presentFence = nullptr;
    // Whether a present was given presentFence since the last wait on it.
    bool presentPending = false;
};
// #endregion

// Set by GLFW when the framebuffer changes size (0x0 when minimized).
struct WindowState {
    bool resized = false;
};

int main(int argc, char** argv)
try {
    const Args args = parseArgs(argc, argv);

    // Step 06's window, now resizable.
    Glfw glfw;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    WindowPtr window(glfwCreateWindow(800, 600, "Vulkan from Zero - 11 resize", nullptr, nullptr), &glfwDestroyWindow);
    if (!window) {
        throw std::runtime_error("glfwCreateWindow failed");
    }
    WindowState windowState;
    glfwSetWindowUserPointer(window.get(), &windowState);
    glfwSetFramebufferSizeCallback(window.get(), [](GLFWwindow* w, int, int) {
        static_cast<WindowState*>(glfwGetWindowUserPointer(w))->resized = true;
    });
    int width  = 0;
    int height = 0;
    glfwGetFramebufferSize(window.get(), &width, &height);
    std::println("Window: {}x{} pixels, resizable", width, height);

    vk::raii::Context context;
    constexpr const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    const std::vector<vk::LayerProperties> layers = context.enumerateInstanceLayerProperties();
    if (!std::ranges::any_of(layers, [&](const vk::LayerProperties& l) {
            return std::string_view{l.layerName.data()} == validationLayer;
        })) {
        throw std::runtime_error(std::string{validationLayer} + " not found - install the Vulkan SDK");
    }
    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName   = "Vulkan from Zero - 11 resize",
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

    // #region instance-extensions
    // VK_KHR_swapchain_maintenance1 (device) depends on two instance extensions:
    // VK_KHR_surface_maintenance1, which needs VK_KHR_get_surface_capabilities2.
    // Enable them when the loader has them; without them the device extension is
    // off and step 11 falls back to vkDeviceWaitIdle.
    uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    if (glfwExts == nullptr) {
        throw std::runtime_error("GLFW found no Vulkan loader or no surface extensions");
    }
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);
    extensions.push_back(vk::EXTDebugUtilsExtensionName);
    const std::vector<vk::ExtensionProperties> instanceExts = context.enumerateInstanceExtensionProperties();
    const auto hasInstanceExt = [&](std::string_view name) {
        return std::ranges::any_of(instanceExts,
                                   [&](const vk::ExtensionProperties& e) { return std::string_view{e.extensionName.data()} == name; });
    };
    const bool surfaceMaintenance =
        hasInstanceExt(vk::KHRGetSurfaceCapabilities2ExtensionName) && hasInstanceExt(vk::KHRSurfaceMaintenance1ExtensionName);
    if (surfaceMaintenance) {
        extensions.push_back(vk::KHRGetSurfaceCapabilities2ExtensionName);
        extensions.push_back(vk::KHRSurfaceMaintenance1ExtensionName);
    }
    // #endregion

    constexpr std::array layerNames{validationLayer};
    const vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> instanceChain{
        vk::InstanceCreateInfo{
            .pApplicationInfo        = &appInfo,
            .enabledLayerCount       = static_cast<uint32_t>(layerNames.size()),
            .ppEnabledLayerNames     = layerNames.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        },
        messengerInfo,
    };
    vk::raii::Instance instance(context, instanceChain.get<vk::InstanceCreateInfo>());
    vk::raii::DebugUtilsMessengerEXT messenger(instance, messengerInfo);
    VkSurfaceKHR rawSurface = VK_NULL_HANDLE;
    if (const VkResult r = glfwCreateWindowSurface(*instance, window.get(), nullptr, &rawSurface); r != VK_SUCCESS) {
        throw std::runtime_error(std::string{"glfwCreateWindowSurface failed: "} + string_VkResult(r));
    }
    vk::raii::SurfaceKHR surface(instance, rawSurface);

    const std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    const vk::raii::PhysicalDevice* best = nullptr;
    int bestScore                        = 0;
    uint32_t family                      = 0;
    for (const vk::raii::PhysicalDevice& device : devices) {
        const vk::PhysicalDeviceProperties props = device.getProperties();
        if (props.apiVersion < vk::ApiVersion14) {
            continue;
        }
        const auto featureChain = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>();
        const auto& f13         = featureChain.get<vk::PhysicalDeviceVulkan13Features>();
        if (f13.dynamicRendering != vk::True || f13.synchronization2 != vk::True) {
            continue;
        }
        const std::vector<vk::ExtensionProperties> exts = device.enumerateDeviceExtensionProperties();
        if (!std::ranges::any_of(exts, [](const vk::ExtensionProperties& e) {
                return std::string_view{e.extensionName.data()} == vk::KHRSwapchainExtensionName;
            })) {
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
    std::println("Picked: {}, graphics + present family {}", physicalDevice.getProperties().deviceName.data(), family);

    // #region device-extensions
    // Present fences need VK_KHR_swapchain_maintenance1 AND its feature
    // swapchainMaintenance1 enabled on the device. Used only when the instance
    // extensions were enabled, the GPU has both, and --no-present-fences was not
    // given.
    const std::vector<vk::ExtensionProperties> deviceExts = physicalDevice.enumerateDeviceExtensionProperties();
    const bool hasMaintenance =
        surfaceMaintenance &&
        std::ranges::any_of(deviceExts,
                            [](const vk::ExtensionProperties& e) {
                                return std::string_view{e.extensionName.data()} == vk::KHRSwapchainMaintenance1ExtensionName;
                            }) &&
        physicalDevice
                .getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR>()
                .get<vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR>()
                .swapchainMaintenance1 == vk::True;
    const bool presentFences = hasMaintenance && args.presentFences;
    constexpr float queuePriority = 1.0f;
    const vk::DeviceQueueCreateInfo queueInfo{
        .queueFamilyIndex = family,
        .queueCount       = 1,
        .pQueuePriorities = &queuePriority,
    };
    vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR maintenance{.swapchainMaintenance1 = vk::True};
    vk::PhysicalDeviceVulkan13Features features13{
        .pNext            = presentFences ? &maintenance : nullptr,
        .synchronization2 = vk::True,
        .dynamicRendering = vk::True,
    };
    std::vector<const char*> deviceExtensions{vk::KHRSwapchainExtensionName};
    if (presentFences) {
        deviceExtensions.push_back(vk::KHRSwapchainMaintenance1ExtensionName);
    }
    vk::raii::Device device(physicalDevice, vk::DeviceCreateInfo{
                                                .pNext                   = &features13,
                                                .queueCreateInfoCount    = 1,
                                                .pQueueCreateInfos       = &queueInfo,
                                                .enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size()),
                                                .ppEnabledExtensionNames = deviceExtensions.data(),
                                            });
    std::println("Present fences (VK_KHR_swapchain_maintenance1): {}",
                 !hasMaintenance ? "not supported"
                 : presentFences ? "supported, used"
                                 : "supported, off (--no-present-fences)");
    // #endregion

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
    std::println("Surface format {} + {}, present mode {}",
                 shortName(string_VkFormat(static_cast<VkFormat>(format.format)), "VK_FORMAT_"),
                 shortName(string_VkColorSpaceKHR(static_cast<VkColorSpaceKHR>(format.colorSpace)), "VK_COLOR_SPACE_", "_KHR"),
                 shortName(string_VkPresentModeKHR(static_cast<VkPresentModeKHR>(presentMode)), "VK_PRESENT_MODE_", "_KHR"));

    // Step 08's pipeline. Viewport and scissor are dynamic state, so it works
    // for every swapchain size: recreation needs no new pipeline.
    const std::vector<uint32_t> spirv = readSpirv(std::filesystem::path{SHADER_DIR} / "triangle.spv");
    std::optional<vk::raii::ShaderModule> shaderModule;
    shaderModule.emplace(device, vk::ShaderModuleCreateInfo{.codeSize = spirv.size() * sizeof(uint32_t), .pCode = spirv.data()});
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex, .module = **shaderModule, .pName = "vertexMain"},
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment, .module = **shaderModule, .pName = "fragmentMain"},
    };
    const vk::PipelineVertexInputStateCreateInfo vertexInput{};
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
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
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

    // Step 10's queue, pool and per-frame objects, plus a present fence per slot.
    vk::raii::Queue queue(device, family, 0);
    vk::raii::CommandPool commandPool(device, vk::CommandPoolCreateInfo{
                                                  .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                  .queueFamilyIndex = family,
                                              });
    vk::raii::CommandBuffers commandBuffers(device, vk::CommandBufferAllocateInfo{
                                                        .commandPool        = *commandPool,
                                                        .level              = vk::CommandBufferLevel::ePrimary,
                                                        .commandBufferCount = static_cast<uint32_t>(frameCount),
                                                    });
    std::vector<Frame> frames;
    frames.reserve(frameCount);
    for (vk::raii::CommandBuffer& commandBuffer : commandBuffers) {
        frames.push_back(Frame{
            .commandBuffer  = std::move(commandBuffer),
            .imageAvailable = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{}),
            .inFlight       = vk::raii::Fence(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled}),
            // Unsignaled: a present fence must be unsignaled when it is handed
            // to vkQueuePresentKHR (VUID-VkSwapchainPresentFenceInfoKHR-pFences-07758).
            .presentFence = presentFences ? vk::raii::Fence(device, vk::FenceCreateInfo{}) : vk::raii::Fence(nullptr),
        });
    }

    Chain chain;
    // Old chains whose presents may still be in progress (present-fence path).
    std::vector<Retired> retired;
    // Present fences that timed out: never reused, destroyed at exit.
    std::vector<vk::raii::Fence> abandoned;
    // Frames rendered so far; frameIndex % frameCount is the current slot.
    uint64_t frameIndex = 0;
    // The surface is 0x0 (minimized): no frames until it has a size again.
    bool paused          = false;
    uint32_t recreations = 0;

    // #region create-chain
    // Creates a swapchain for the surface's current size, handing over the old
    // one, plus the views and semaphores of its images.
    const auto createChain = [&](vk::SwapchainKHR old) -> std::optional<Chain> {
        // Ask the surface again every time: its size changes with the window.
        const vk::SurfaceCapabilitiesKHR caps = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        vk::Extent2D extent                   = caps.currentExtent;
        if (caps.currentExtent.width == std::numeric_limits<uint32_t>::max()) {
            int w = 0;
            int h = 0;
            glfwGetFramebufferSize(window.get(), &w, &h);
            extent = vk::Extent2D{
                .width  = std::clamp(static_cast<uint32_t>(w), caps.minImageExtent.width, caps.maxImageExtent.width),
                .height = std::clamp(static_cast<uint32_t>(h), caps.minImageExtent.height, caps.maxImageExtent.height),
            };
        }
        // Minimized: the surface is 0x0, and a swapchain cannot be that small
        // (VUID-VkSwapchainCreateInfoKHR-imageExtent-01689: width and height
        // must be non-zero). Nothing to create until the window has a size.
        if (extent.width == 0 || extent.height == 0) {
            return std::nullopt;
        }
        uint32_t imageCount = caps.minImageCount + 1;
        if (caps.maxImageCount > 0) {
            imageCount = std::min(imageCount, caps.maxImageCount);
        }
        // oldSwapchain: the presentation engine may reuse the old one's
        // resources and switch over without a gap. The old one is "retired": no
        // more acquires from it, but images already acquired from it may still
        // be presented.
        Chain next;
        next.swapchain = vk::raii::SwapchainKHR(device, vk::SwapchainCreateInfoKHR{
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
                                                            .oldSwapchain     = old,
                                                        });
        next.images = next.swapchain.getImages();
        next.extent = extent;
        for (const vk::Image image : next.images) {
            next.views.emplace_back(device, vk::ImageViewCreateInfo{
                                                .image            = image,
                                                .viewType         = vk::ImageViewType::e2D,
                                                .format           = format.format,
                                                .subresourceRange = {
                                                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                                                    .levelCount = 1,
                                                    .layerCount = 1,
                                                },
                                            });
            next.renderFinished.emplace_back(device, vk::SemaphoreCreateInfo{});
        }
        return next;
    };
    // #endregion

    // #region recreate
    // Replaces the current chain with one for the current surface size. Returns
    // false while the window is minimized (nothing created).
    const auto recreate = [&]() -> bool {
        // Without present fences we cannot tell when the presentation engine is
        // done with the old swapchain: wait until the GPU is idle (all submitted
        // work and the presents' semaphore waits completed), then destroy it
        // right away. With present fences the old chain is kept in `retired` and
        // the frames go on.
        if (!presentFences) {
            device.waitIdle();
        }
        // A failed attempt is not retried until the window reports a new size.
        windowState.resized   = false;
        const vk::SwapchainKHR old = *chain.swapchain;
        std::optional<Chain> next  = createChain(old);
        if (!next) {
            if (!paused) {
                std::println("Window minimized (surface 0x0): rendering paused");
            }
            paused = true;
            return false;
        }
        if (paused) {
            std::println("Window restored");
        }
        paused           = false;
        Chain oldChain   = std::exchange(chain, std::move(*next));
        const bool first = old == vk::SwapchainKHR{};
        if (!first) {
            ++recreations;
            if (presentFences) {
                retired.push_back(Retired{.chain = std::move(oldChain), .retiredAt = frameIndex});
            }
            // Otherwise oldChain is destroyed at the end of this scope (the
            // device is idle).
        }
        std::println("Swapchain {}: {}x{}, {} images{}", first ? "created" : "recreated", chain.extent.width,
                     chain.extent.height, chain.images.size(),
                     first           ? ""
                     : presentFences ? "; the old one waits for its present fences"
                                     : "; the old one destroyed after vkDeviceWaitIdle");
        return true;
    };
    // #endregion

    // #region wait-present-fence
    // Waits for the present fence of `slot` (from the present frameCount frames
    // ago) and resets it for this frame's present. The spec does not say whether
    // the fence of a present that returned VK_ERROR_OUT_OF_DATE_KHR is signaled,
    // so the wait has a timeout: if it expires, wait for the whole device instead
    // and retire that fence for good (it is never handed to a present again, and
    // destroyed only at exit).
    const auto waitPresentFence = [&](size_t slot) {
        Frame& frame = frames.at(slot);
        if (!frame.presentPending) {
            return;
        }
        const vk::Result r = device.waitForFences(*frame.presentFence, vk::True, presentFenceTimeoutNs);
        if (r == vk::Result::eSuccess) {
            device.resetFences(*frame.presentFence);
        } else if (r == vk::Result::eTimeout) {
            std::println("Present fence of slot {} not signaled after {} s: falling back to vkDeviceWaitIdle", slot,
                         presentFenceTimeout.count());
            device.waitIdle();
            abandoned.push_back(std::exchange(frame.presentFence, vk::raii::Fence(device, vk::FenceCreateInfo{})));
        } else {
            throw std::runtime_error(std::string{"waitForFences failed: "} + string_VkResult(static_cast<VkResult>(r)));
        }
        frame.presentPending = false;
    };
    // #endregion

    // #region destroy-retired
    // Destroys the retired chains whose presents have all completed. The presents
    // to a chain retired at frame R belong to frames R-frameCount..R-1 at the
    // latest (one present per frame, slot = frame % frameCount), and the present
    // fence of frame F is waited on at the start of frame F+frameCount. So once
    // frame R+frameCount-1 has waited on its slot's fence, every present to the
    // old chain has signaled its fence.
    const auto destroyRetired = [&] {
        std::erase_if(retired, [&](Retired& r) {
            if (frameIndex + 1 < r.retiredAt + frameCount) {
                return false;
            }
            std::println("Old {}x{} swapchain destroyed: the fences of all its presents have signaled",
                         r.chain.extent.width, r.chain.extent.height);
            return true;
        });
    };
    // #endregion

    // The first swapchain is made by the same function as every later one.
    if (!recreate()) {
        throw std::runtime_error("The window has no size at start");
    }
    std::println("\nRendering. Resize or minimize the window; close it to exit.");
    const auto start = std::chrono::steady_clock::now();

    // Renders and presents one frame; recreates the swapchain when needed.
    const auto drawFrame = [&] {
        if (paused && !windowState.resized) {
            return;
        }
        const size_t slot = static_cast<size_t>(frameIndex % frameCount);
        Frame& frame      = frames.at(slot);

        // Step 10's fence wait, then the present fence of the same slot.
        if (device.waitForFences(*frame.inFlight, vk::True, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess) {
            throw std::runtime_error("waitForFences failed");
        }
        if (presentFences) {
            waitPresentFence(slot);
            destroyRetired();
        }

        // #region recreate-triggers
        // Three triggers, checked where they can happen:
        // - the window reported a new size (windowState.resized, set by GLFW) or
        //   the last present was SUBOPTIMAL: recreate before acquiring;
        // - acquire returns OUT_OF_DATE: no image was acquired (and the semaphore
        //   was not signaled), so recreate and acquire again;
        // - acquire returns SUBOPTIMAL: an image WAS acquired and the semaphore
        //   will be signaled, so render and present this frame and recreate
        //   afterwards (an acquired image must be presented, or released with
        //   vkReleaseSwapchainImagesKHR).
        if (windowState.resized && !recreate()) {
            return;
        }
        vk::ResultValue<uint32_t> acquired{vk::Result::eErrorOutOfDateKHR, 0};
        for (;;) {
            acquired = chain.swapchain.acquireNextImage(std::numeric_limits<uint64_t>::max(), *frame.imageAvailable, nullptr);
            if (acquired.result != vk::Result::eErrorOutOfDateKHR) {
                break;
            }
            std::println("[timing] acquire: VK_ERROR_OUT_OF_DATE_KHR");
            if (!recreate()) {
                return;
            }
        }
        if (acquired.result == vk::Result::eSuboptimalKHR) {
            std::println("[timing] acquire: VK_SUBOPTIMAL_KHR");
            windowState.resized = true;
        }
        // #endregion
        const uint32_t imageIndex = acquired.value;
        const vk::Image image     = chain.images.at(imageIndex);
        const vk::Extent2D extent = chain.extent;
        device.resetFences(*frame.inFlight);

        // Record as in step 10, with the CURRENT chain's extent.
        constexpr vk::ImageSubresourceRange colorRange{.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1};
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
        const float t   = static_cast<float>(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        const auto wave = [t](float phase) { return 0.1f + 0.1f * std::sin(t * 1.5f + phase); };
        const vk::RenderingAttachmentInfo colorAttachment{
            .imageView   = *chain.views.at(imageIndex),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eStore,
            .clearValue  = vk::ClearColorValue{std::array{wave(0.0f), wave(2.1f), wave(4.2f), 1.0f}},
        };
        const vk::Rect2D area{.offset = {0, 0}, .extent = extent};
        const vk::raii::CommandBuffer& cmd = frame.commandBuffer;
        cmd.reset();
        cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toAttachment});
        cmd.beginRendering(vk::RenderingInfo{
            .renderArea = area, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &colorAttachment});
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
        cmd.setViewport(0, vk::Viewport{
                               .x        = 0.0f,
                               .y        = 0.0f,
                               .width    = static_cast<float>(extent.width),
                               .height   = static_cast<float>(extent.height),
                               .minDepth = 0.0f,
                               .maxDepth = 1.0f,
                           });
        cmd.setScissor(0, area);
        cmd.draw(3, 1, 0, 0);
        cmd.endRendering();
        cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toPresent});
        cmd.end();
        const vk::Semaphore signalSemaphore = *chain.renderFinished.at(imageIndex);
        const vk::SemaphoreSubmitInfo waitInfo{
            .semaphore = *frame.imageAvailable, .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput};
        const vk::SemaphoreSubmitInfo signalInfo{
            .semaphore = signalSemaphore, .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput};
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

        // #region present
        // With present fences, VkSwapchainPresentFenceInfoKHR in the pNext chain
        // gives the present a fence (one per swapchain in the call).
        const vk::SwapchainKHR swapchainHandle = *chain.swapchain;
        const vk::Fence presentFence           = presentFences ? *frame.presentFence : vk::Fence{};
        const vk::SwapchainPresentFenceInfoKHR fenceInfo{.swapchainCount = 1, .pFences = &presentFence};
        if (presentFences) {
            frame.presentPending = true;
        }
        const vk::Result presented = queue.presentKHR(vk::PresentInfoKHR{
            .pNext              = presentFences ? &fenceInfo : nullptr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &signalSemaphore,
            .swapchainCount     = 1,
            .pSwapchains        = &swapchainHandle,
            .pImageIndices      = &imageIndex,
        });
        // OUT_OF_DATE from the present: the queue operation (the semaphore wait)
        // is still enqueued. Either way the swapchain is replaced before the next
        // acquire.
        if (presented == vk::Result::eSuboptimalKHR) {
            std::println("[timing] present: VK_SUBOPTIMAL_KHR");
            windowState.resized = true;
        } else if (presented == vk::Result::eErrorOutOfDateKHR) {
            std::println("[timing] present: VK_ERROR_OUT_OF_DATE_KHR");
            windowState.resized = true;
        }
        // #endregion
        ++frameIndex;
    };

    // #region event-loop
    // GLFW's framebuffer-size callback sets windowState.resized (0x0 when
    // minimized). While paused there is nothing to render: wait for events with
    // a timeout instead of spinning, so the demo and --seconds still run.
    const IdleGuard idleGuard{device};
    size_t demoStep    = 0;
    const auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
    while (glfwWindowShouldClose(window.get()) == GLFW_FALSE && !(args.seconds && elapsed() >= *args.seconds)) {
        if (paused && !windowState.resized) {
            glfwWaitEventsTimeout(0.01);
        } else {
            glfwPollEvents();
        }
        // #region demo-script
        if (args.demo && demoStep < demo.size() && elapsed() >= demo.at(demoStep).at) {
            const DemoAction& action = demo.at(demoStep++);
            switch (action.kind) {
            case DemoKind::Resize:
                std::println("Demo: resize to {}x{}", action.width, action.height);
                glfwSetWindowSize(window.get(), action.width, action.height);
                break;
            case DemoKind::Minimize:
                std::println("Demo: minimize");
                glfwIconifyWindow(window.get());
                break;
            case DemoKind::Restore:
                std::println("Demo: restore");
                glfwRestoreWindow(window.get());
                break;
            }
        }
        // #endregion
        drawFrame();
    }
    std::println("Window closed");
    // #endregion

    // #region cleanup
    // vkDeviceWaitIdle covers our submissions; the present fences cover the
    // presents of every chain (current and retired), so wait on all that are
    // pending - with the timeout, as in wait-present-fence. Then RAII destroys
    // everything in reverse order of declaration.
    device.waitIdle();
    std::vector<vk::Fence> pending;
    for (const Frame& frame : frames) {
        if (frame.presentPending) {
            pending.push_back(*frame.presentFence);
        }
    }
    if (!pending.empty() && device.waitForFences(pending, vk::True, presentFenceTimeoutNs) != vk::Result::eSuccess) {
        std::println(stderr, "waiting for the present fences failed");
    }
    std::println("[timing] {} frames, {} swapchain recreation(s)", frameIndex, recreations);
    // #endregion
    return EXIT_SUCCESS;
} catch (const vk::SystemError& e) {
    std::println(stderr, "Vulkan error: {}", e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return EXIT_FAILURE;
}
