# Vulkan from Zero

**Live site: <https://ssoj13.github.io/course_vulkan/>** — English; Russian: <https://ssoj13.github.io/course_vulkan/ru/>.

An interactive, visual Vulkan 1.4 course for graphics people: mind maps of every concept, an explorable map of all
Vulkan objects, a create/destroy exercise that answers with the real validation errors, and companion programs in
**Rust (ash)** and **C++ (Vulkan-Hpp)** from which every code snippet in the lessons is cut.

## Build the examples

Every lesson step is one self-contained program in both languages: `examples/rust/src/bin/<step>.rs` and
`examples/cpp/src/<step>.cpp`. Under each code block on the site there is a link to its lines on GitHub and the
command that runs the step.

### Requirements

- A GPU and driver with **Vulkan 1.4** (the programs refuse older devices).
- **Vulkan SDK 1.4.341** or newer from [LunarG](https://vulkan.lunarg.com), with `VULKAN_SDK` set (the installer sets
  it). Every step from 02 on enables `VK_LAYER_KHRONOS_validation`, which comes with the SDK. The C++ build also takes
  the headers and `vulkan-1.lib` from it.
- Rust: stable **1.85+** (edition 2024; no `rust-toolchain` file is pinned). `ash` comes from git at a fixed commit
  (Vulkan 1.4 bindings), so the first build needs network access.
- C++: **CMake 3.25+**, **Ninja**, **MSVC** (Visual Studio 2022 17.7+), run from a Developer PowerShell (`cl.exe` on
  `PATH`); **[vcpkg](https://vcpkg.io)** with `VCPKG_ROOT` set: the presets use
  `$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake`, and manifest mode (`examples/cpp/vcpkg.json`) builds GLFW and VMA on
  the first configure.
- Shaders are committed as SPIR-V (`shaders/spv/`), so no shader compiler is needed unless you edit a `.slang` file
  (then `bun run sync:shaders`).

### Run a step

```powershell
git clone https://github.com/ssoj13/course_vulkan
cd course_vulkan

# Rust
cd examples/rust
cargo run --release --bin 22_depth

# C++ (configure once, then build and run any step)
cd ../cpp
cmake --preset msvc-release
cmake --build --preset msvc-release --target 22_depth
.\build\msvc-release\22_depth.exe
```

`msvc-debug` builds into `build/msvc-debug`. Window steps accept options such as `--seconds N`; see the step's lesson
and the README in each example folder.

### Validation tips

The validation layer is always on; its messages go to the console through the debug messenger. Two deeper checks are
switched on with environment variables before running a step:

```powershell
$env:VK_KHRONOS_VALIDATION_VALIDATE_SYNC = 'true'   # synchronization validation (hazards, missing barriers)
$env:VK_KHRONOS_VALIDATION_GPUAV_ENABLE = 'true'    # GPU-assisted validation (out-of-bounds shader access)
```

Vulkan Configurator (`vkconfig`, in the SDK) can set the same options without environment variables.

### Steps and their lessons

| Step | Lessons that show its code |
| --- | --- |
| `01_instance` | [The loader and VkInstance](https://ssoj13.github.io/course_vulkan/instance/loader-and-instance/) |
| `02_validation` | [Validation layers and the debug messenger](https://ssoj13.github.io/course_vulkan/instance/validation/) |
| `03_physical_devices` | [Physical devices: properties, features, limits](https://ssoj13.github.io/course_vulkan/devices/physical-devices/) |
| `04_queue_families` | [Queue families: the GPU's engines](https://ssoj13.github.io/course_vulkan/devices/queue-families/) |
| `05_logical_device` | [Logical device and queues](https://ssoj13.github.io/course_vulkan/devices/logical-device/) |
| `06_window_surface` | [A window and its surface](https://ssoj13.github.io/course_vulkan/presentation/window-and-surface/) |
| `07_swapchain` | [The swapchain and present modes](https://ssoj13.github.io/course_vulkan/presentation/swapchain/), [Images and image views](https://ssoj13.github.io/course_vulkan/presentation/image-views/) |
| `08_pipeline` | [Shaders, Slang and SPIR-V](https://ssoj13.github.io/course_vulkan/pipeline/shaders-and-spirv/), [The graphics pipeline object](https://ssoj13.github.io/course_vulkan/pipeline/graphics-pipeline/), [The pipeline layout](https://ssoj13.github.io/course_vulkan/pipeline/pipeline-layout/), [Dynamic rendering](https://ssoj13.github.io/course_vulkan/pipeline/dynamic-rendering/) |
| `09_triangle` | [Dynamic rendering](https://ssoj13.github.io/course_vulkan/pipeline/dynamic-rendering/), [Command pools and command buffers](https://ssoj13.github.io/course_vulkan/commands/command-buffers/), [The first triangle](https://ssoj13.github.io/course_vulkan/commands/first-triangle/) |
| `10_frames_in_flight` | [Fences and semaphores](https://ssoj13.github.io/course_vulkan/sync/fences-and-semaphores/), [Frames in flight](https://ssoj13.github.io/course_vulkan/sync/frames-in-flight/) |
| `11_resize` | [Swapchain recreation](https://ssoj13.github.io/course_vulkan/sync/swapchain-recreation/) |
| `12_timeline` | [Pipeline barriers and image layouts](https://ssoj13.github.io/course_vulkan/sync/barriers-and-layouts/), [Timeline semaphores](https://ssoj13.github.io/course_vulkan/sync/timeline-semaphores/), [Swapchain recreation](https://ssoj13.github.io/course_vulkan/sync/swapchain-recreation/) |
| `13_memory_types` | [Memory heaps and memory types](https://ssoj13.github.io/course_vulkan/memory/memory-types/) |
| `14_vertex_buffer` | [Vertex and index buffers](https://ssoj13.github.io/course_vulkan/memory/vertex-and-index-buffers/) |
| `15_staging` | [Staging uploads](https://ssoj13.github.io/course_vulkan/memory/staging-uploads/) |
| `16_allocator` | [Allocators and sub-allocation](https://ssoj13.github.io/course_vulkan/memory/allocators/) |
| `17_uniforms` | [Uniform buffers](https://ssoj13.github.io/course_vulkan/descriptors/uniform-buffers/), [Descriptor sets](https://ssoj13.github.io/course_vulkan/descriptors/descriptor-sets/) |
| `18_push_constants` | [Push constants and push descriptors](https://ssoj13.github.io/course_vulkan/descriptors/push-constants/) |
| `19_buffer_device_address` | [Buffer device address](https://ssoj13.github.io/course_vulkan/descriptors/buffer-device-address/) |
| `20_bindless` | [Bindless descriptors](https://ssoj13.github.io/course_vulkan/descriptors/bindless/) |
| `21_texture` | [Textures and samplers](https://ssoj13.github.io/course_vulkan/images/textures-and-samplers/) |
| `22_depth` | [Depth buffers](https://ssoj13.github.io/course_vulkan/images/depth/) |
| `23_mipmaps` | [Mipmaps](https://ssoj13.github.io/course_vulkan/images/mipmaps/) |
| `24_msaa` | [Multisample anti-aliasing (MSAA)](https://ssoj13.github.io/course_vulkan/images/msaa/) |
| `25_compute` | [Compute pipelines](https://ssoj13.github.io/course_vulkan/compute/compute-pipelines/) |
| `26_particles` | [Particles](https://ssoj13.github.io/course_vulkan/compute/particles/) |
| `27_async_compute` | [Async compute](https://ssoj13.github.io/course_vulkan/compute/async-compute/) |
| `28_debug_labels` | [Debugging and profiling](https://ssoj13.github.io/course_vulkan/advanced/debugging-and-profiling/) |
| `29_gpu_queries` | [Debugging and profiling](https://ssoj13.github.io/course_vulkan/advanced/debugging-and-profiling/) |
| `30_pipeline_cache` | Pipeline cache (lesson in progress) |

## Repository

| Path | What |
| --- | --- |
| `src/content/docs/` | Lessons (MDX), one folder per module |
| `src/data/course.json` | Modules and lessons: order, titles, URLs |
| `src/data/terms.json` | Term graph behind every mind map and the glossary |
| `src/data/objects.json` | Vulkan object model behind the ObjectGraph widget |
| `src/data/vuids.json` | Valid-usage texts quoted by the widgets (generated by `scripts/sync-vuids.ts` from the SDK) |
| `examples/rust/` | Companion programs, Rust + ash (`cargo run --bin 01_instance`) |
| `examples/cpp/` | Companion programs, C++ + Vulkan-Hpp (`cmake --preset msvc-debug`) |

The site engine (sidebar, links, progress, quizzes, mind maps) is the shared package
[course-kit](https://github.com/ssoj13/course-kit).

## Develop

```sh
bun install
bun run dev      # local site
bun test tests   # object model and VUID data tests
bun run build    # static site in dist/
```

Requires [bun](https://bun.sh) 1.4.3. The examples need the [Vulkan SDK](https://vulkan.lunarg.com) for the
validation layers; see the README in each example folder.
