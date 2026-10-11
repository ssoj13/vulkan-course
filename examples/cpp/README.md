# Vulkan textbook - C++ examples

Companion code for the lessons. One self-contained program per step; each step
repeats the setup of the previous ones, so a single file shows the whole program.
Style: Vulkan-Hpp RAII (`vk::raii::*`) with designated initializers
(`VULKAN_HPP_NO_STRUCT_CONSTRUCTORS`), like the Khronos Vulkan Tutorial.

## Requirements

- Vulkan SDK 1.4.x (`VULKAN_SDK` set; provides headers, loader, validation layer)
- A Vulkan 1.4 GPU driver with `dynamicRendering` and `synchronization2` (the examples refuse older devices)
- CMake 3.25+, Ninja, MSVC (Visual Studio 2022 17.7+ / toolset 14.51). Run from a
  developer prompt (or any shell where `cl.exe` is on `PATH`).
- C++23 mode is used only for `std::print`; the code is otherwise plain C++20.
- [vcpkg](https://vcpkg.io) with `VCPKG_ROOT` set: the presets use its toolchain file, and
  `vcpkg.json` (manifest mode) builds GLFW 3 (steps 06+) and VMA 3.4.0, the header-only
  Vulkan Memory Allocator (`vulkan-memory-allocator`, step 16), on the first configure.
  Vulkan itself still comes from the SDK. Step 16 includes `<vk_mem_alloc.h>` from vcpkg;
  the SDK's own older copy (`<vma/vk_mem_alloc.h>`, 3.3.0) is not used.

## Build and run

```powershell
cmake --preset msvc-debug            # build dir: build/msvc-debug
cmake --build --preset msvc-debug
.\build\msvc-debug\01_instance.exe
```

`msvc-release` works the same way (`build/msvc-release`). No LTO is used.
A build directory configured before vcpkg was added needs `cmake --preset <name> --fresh`.

Steps 08+ read `../../shaders/spv/triangle.spv` (steps 14-15: `mesh.spv`) at run time (the absolute path is baked in
as `SHADER_DIR` by CMake), the same file the Rust examples embed. It is compiled from
`../../shaders/triangle.slang` by `bun run sync:shaders` (repository root, needs the Vulkan SDK).

## Steps

| Step | File | New in this step |
|------|------|------------------|
| 01 | `src/01_instance.cpp` | loader, `ApplicationInfo`, instance extensions, `Instance` |
| 02 | `src/02_validation.cpp` | validation layer, debug callback, messenger (also chained into instance creation) |
| 03 | `src/03_physical_devices.cpp` | enumerate GPUs, properties, features, limits, pick one |
| 04 | `src/04_queue_families.cpp` | queue families: graphics, dedicated transfer, async compute |
| 05 | `src/05_logical_device.cpp` | queue create info, enabling Vulkan 1.3 features on a 1.4 device, `Device`, `Queue` |
| 06 | `src/06_window_surface.cpp` | GLFW window, surface extensions, `SurfaceKHR`, graphics + present family, surface capabilities, formats, present modes |
| 07 | `src/07_swapchain.cpp` | `VK_KHR_swapchain` device; format, present mode, extent, image count; `SwapchainKHR`, `ImageView`s |
| 08 | `src/08_pipeline.cpp` | SPIR-V from Slang, `ShaderModule`, empty `PipelineLayout`, graphics `Pipeline` with dynamic rendering |
| 09 | `src/09_triangle.cpp` | `CommandPool`/`CommandBuffer`, fence + semaphores, the frame loop (acquire, record, submit, present): the first triangle |
| 10 | `src/10_frames_in_flight.cpp` | per-frame command buffer, fence and acquire semaphore (`--frames 1\|2\|3`); simulated CPU/GPU work (`--cpu-ms`, `--draws`), fence-wait statistics |
| 11 | `src/11_resize.cpp` | resizable window; swapchain recreation (`oldSwapchain`) on resize / OUT_OF_DATE / SUBOPTIMAL; present fences (`VK_KHR_swapchain_maintenance1`) + retire list, or `--no-present-fences`; minimized = paused; `--demo-resize` |
| 12 | `src/12_timeline.cpp` | timeline semaphore instead of fences; binary acquire/present semaphores; offscreen 400x300 image blitted to the swapchain with exact sync2 barriers |
| 13 | `src/13_memory_types.cpp` | memory heaps and types, memory limits, buffer memory requirements, the find-memory-type rule (required + preferred), allocate/bind/map, two buffers in one allocation; no window |
| 14 | `src/14_vertex_buffer.cpp` | indexed hexagon from vertex + index buffers in HOST_VISIBLE \| HOST_COHERENT memory; vertex input state; `drawIndexed`; 2 frames in flight |
| 15 | `src/15_staging.cpp` | DEVICE_LOCAL mesh via staging buffer + `copyBuffer` on the graphics queue, or the transfer queue with a queue family ownership transfer (`--transfer-queue`), or direct BAR writes (`--bar`); `--mib N`; timed upload |
| 16 | `src/16_allocator.cpp` | N buffers with one `vkAllocateMemory` each vs. VMA (`vmaCreateBuffer`, 64 MiB preferred blocks); counts, VMA statistics, timings; no window |
Lessons embed code by `// #region <name>` ... `// #endregion` markers; the names
match the Rust examples.

Steps 06-12, 14 and 15 keep their window open until you close it; steps 10-12, 14 and 15
also take `--seconds S`. Lines that depend on timing start with `[timing]` (drop them to
compare with the Rust output). Steps 06-10, 12, 14 and 15 have a fixed-size window: resizing
(and minimizing, which makes the swapchain out of date) needs swapchain recreation, which is
step 11 - the others stop cleanly with a message. In step 16 the lines about the allocator's
internals start with `[lib]`: VMA here, a minimal linear allocator in the Rust version (whose
`gpu-allocator` crate is built against an older ash); all other lines are identical.
Short enum names (`B8G8R8A8_SRGB`, `MAILBOX`) come from the SDK's `vk_enum_string_helper.h`,
so both languages print identical output.

![Step 09](../screenshots/09_triangle.png)

## Notes

- Debug messenger severities are Warning | Error only: Info makes the loader print
  100+ lines. Some machines have third-party implicit layers that warn once (for
  example `VK_LAYER_bandicam_helper` reporting API 1.2 < 1.4); that is not a bug here.
