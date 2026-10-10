# Vulkan ecosystem research (2026-10-10)

Legend: **V** = verified from fetched source or local disk. **S** = search-snippet only. **UNVERIFIED** = not confirmed.

## 0. Local environment (V, disk)

| Item | Finding |
|---|---|
| SDK | `C:/Programs/VulkanSDK/1.4.341.1`; `Include/vulkan/vulkan_core.h`: `VK_HEADER_VERSION 341` (1.4.341) |
| Shader tools (`Bin`) | `slangc` 2026.1-52; `glslc`/`glslangValidator` shaderc v2026.1; `dxc`; `spirv-val/opt/cross/dis/reflect`; `vkconfig`; `vulkaninfoSDK` |
| C++ headers | `vulkan.cppm`, `vulkan.hpp`, `vulkan_raii.hpp` present |
| Extension names in header | `VK_KHR_unified_image_layouts`, `VK_EXT_descriptor_heap`, `VK_EXT_shader_object` |
| cargo | 1.99.0 |
| vcpkg | `C:/vcpkg/installed`: x64-windows, x64-windows-static, x64-windows-static-md-release, x64-windows-static-release |

## 1. Khronos Vulkan Tutorial

- **V** Target: Vulkan 1.4 baseline, C++20, Vulkan-Hpp with `vk::raii`, dynamic rendering, timeline semaphores, GLFW, GLM, Slang as primary shader language. Source: https://docs.vulkan.org/tutorial/latest/Building_a_Simple_Engine/introduction.html
- **V** Repo: "Khronos version of the Vulkan Tutorial as parts of the Vulkan documentation project", Antora-built; Windows script uses vcpkg; Linux installs cmake, ninja. Source: https://github.com/KhronosGroup/Vulkan-Tutorial
- **V (partial) main tutorial chapters** (https://docs.vulkan.org/tutorial/latest/01_Overview.html): Overview; Origin of Vulkan (Coding conventions); What it takes to draw a triangle: Step 1 Instance and physical device selection, Step 2 Logical device and queue families, Step 3 Window surface and swap chain, Step 4 Swap chain image views, Step 5 Dynamic Rendering Overview, Step 6 Graphics pipeline, Step 7 Command pools and command buffers, Step 8 Main loop, Summary; API concepts: Validation layers.
- **UNVERIFIED:** full chapter list beyond the above; whether main tutorial text mentions synchronization2, CMake or vcpkg.

### Building a Simple Engine (V)

- Source: https://docs.vulkan.org/tutorial/latest/Building_a_Simple_Engine/introduction.html
- Prerequisite: "This series targets readers who have completed the Vulkan Tutorial and feel comfortable with the fundamentals."
- Uses vk::raii, dynamic rendering ("eliminating the need for explicit render passes and framebuffers"), C++20 modules.
- Chapters: Engine Architecture; Camera Transformations; Lighting & Materials; GUI; Loading Models; Subsystems; Tooling; Mobile Development; Advanced Topics.
- Launch: Khronos blog https://www.khronos.org/blog/new-vulkan-game-engine-tutorial-build-your-own-production-ready-rendering-engine (S). Exact date **UNVERIFIED** (secondary sources conflict).

## 2. Other references

| Resource | URL | Facts | Best for |
|---|---|---|---|
| vkguide.dev | https://vkguide.dev/ | Victor Blanco, "Copyright 2020"; Vulkan 1.3, C++20, dynamic rendering; ch. 0 setup, 1 init/clear, 2 compute, 3 graphics pipeline, 4 textures and descriptors, 5 glTF, extras (GPU-driven); Vulkan 1.1 legacy marked outdated. VMA/bindless/BDA: **UNVERIFIED** | Engine-oriented walkthrough, 1.3 era |
| Sascha Willems Vulkan | https://github.com/SaschaWillems/Vulkan | C++, about 96 examples in 15 categories; dynamic rendering, shader objects, descriptor buffer and heap samples | Feature reference samples |
| Khronos Vulkan-Samples | https://github.com/KhronosGroup/Vulkan-Samples | "One stop solution for all Vulkan samples"; API + performance samples; Vulkan 1.1+, C++20 | Official samples, performance |
| Vulkan-Guide | https://github.com/KhronosGroup/Vulkan-Guide | "One stop shop for getting started with the Vulkan API"; chapters on synchronization2 porting, descriptors (buffers, heaps), dynamic rendering | Concept guide |
| Synchronization examples | https://docs.vulkan.org/guide/latest/synchronization_examples.html | "This page hosts examples for the VK_KHR_synchronization2 APIs"; compute, graphics, transfer, semaphore, swapchain patterns | Barrier patterns |
| TBR best practices | https://docs.vulkan.org/guide/latest/tile_based_rendering_best_practices.html | Bandwidth often dominant cost on tile GPUs | Mobile / tile-based |
| Arm subpasses article | https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/vulkan-subpasses-the-good-the-bad-and-the-ugly | Subpass merging is implementation-dependent; `VK_EXT_subpass_merge_feedback` reports it (S) | Why render passes/subpasses existed |
| Arm/Khronos 2026 course | https://docs.vulkan.org/tutorial/latest/00_Introduction.html (listed) | "SIGGRAPH 2026: How to write a Vulkan application in 2026" (title only, V) | content **UNVERIFIED** |

## 3. Rust

Source: `https://crates.io/api/v1/crates/<name>` (V, fetched 2026-10-10).

| Crate | Stable max | Max incl. pre | updated_at |
|---|---|---|---|
| ash | 0.38.0+1.3.281 | 0.38.0+1.3.281 | 2024-04-01 |
| ash-window | 0.13.0 | 0.13.0 | 2024-04-01 |
| winit | 0.30.13 | 0.31.0-beta.3 | 2026-09-04 |
| raw-window-handle | 0.6.2 | 0.6.2 | 2024-05-17 |
| gpu-allocator | 0.28.0 | 0.28.0 | 2025-09-26 |
| vk-mem | 0.5.0 | 0.5.0 | 2025-08-12 |
| glam | 0.34.1 | 0.34.1 | 2026-10-06 |
| shaderc | 0.10.1 | 0.10.1 | 2025-09-06 |
| vulkanalia | 0.35.0 | 0.35.0 | 2026-02-15 |
| slang (Rust bindings) | 0.0.2 | 0.0.2 | 2019-03-22 (stale) |

- **ash covers Vulkan 1.3, not 1.4 (V):** latest release 0.38.0, "Update Vulkan-Headers to 1.3.281" (https://github.com/ash-rs/ash/releases). ash master status: **UNVERIFIED**.
- **ash-window 0.13.0 deps (V):** `ash` 0.38, `raw-window-handle` 0.6; winit only as dev-dep 0.29 (https://docs.rs/crate/ash-window/0.13.0/source/Cargo.toml.orig).
- **winit 0.30.13 (V):** default uses `raw-window-handle` 0.6 (`rwh_06`) (https://docs.rs/crate/winit/0.30.13/source/Cargo.toml.orig).
- **Compatibility:** ash-window 0.13 and winit 0.30 both use raw-window-handle 0.6, so they should match. Inference from manifests, not compiled. winit 0.31 beta: **UNVERIFIED**.
- **Rust tutorials:**
  - vulkanalia tutorial, https://kylemayes.github.io/vulkanalia/ (V): 38 chapters (Introduction, Setup, Presentation, Pipeline, Drawing, Swapchain, Vertex buffers, Uniform buffers, Texture mapping, Model, Rendering quality, Dynamic scenes, Conclusion); Rust 1.88+; chapters from Push Constants on marked experimental; Vulkan version not stated.
  - adrien-ben/vulkan-tutorial-rs, https://github.com/adrien-ben/vulkan-tutorial-rs (S): ash port of vulkan-tutorial.com; status **UNVERIFIED**.
  - ash examples: `cargo run -p ash-examples --bin triangle` (S, https://docs.rs/crate/ash/0.38.0+1.3.281).

## 4. Shaders

- **V** Slang under Khronos: https://docs.vulkan.org/guide/latest/slang.html: Slang "is under the governance of Khronos"; Slang compilers "are also part of the Vulkan SDK".
- **S** Khronos Slang Initiative launched 2024-11-21, Apache 2.0 on GitHub, NVIDIA-contributed: https://www.khronos.org/news/press/khronos-group-launches-slang-initiative-hosting-open-source-compiler-contributed-by-nvidia ; https://www.phoronix.com/news/Khronos-Slang
- **V** Compile: `slangc texture.slang -target spirv -o texture.vert.spv`; explicit `vk::binding` "usually unnecessary" (same docs page).
- **V** SDK 1.4.341.1 ships slangc, glslc, dxc.
- **Recommendation (judgment):** Slang as primary path for a 2026 course, matching the Khronos tutorial; GLSL as reference (much existing material); HLSL via dxc only for D3D ports.

## 5. Core promotions and extension status

**Core 1.3 (V, https://docs.vulkan.org/spec/latest/appendices/versions.html):** VK_KHR_dynamic_rendering, VK_KHR_synchronization2, VK_KHR_copy_commands2, VK_KHR_maintenance4, VK_KHR_zero_initialize_workgroup_memory, VK_KHR_shader_terminate_invocation, VK_EXT_extended_dynamic_state, VK_EXT_extended_dynamic_state2 (patch-control-points and logic-op not promoted), VK_EXT_pipeline_creation_cache_control, VK_EXT_private_data, others.

**Timeline semaphores:** listed under 1.2 on the same page (V), not 1.3.

**Core 1.4 (V, same page):** VK_KHR_dynamic_rendering_local_read, VK_KHR_maintenance5, VK_KHR_maintenance6, VK_KHR_push_descriptor (gated by `pushDescriptor` feature), VK_KHR_map_memory2, VK_KHR_index_type_uint8, VK_KHR_line_rasterization, VK_KHR_load_store_op_none, VK_KHR_shader_expect_assume, VK_KHR_shader_float_controls2, VK_KHR_shader_subgroup_rotate, VK_KHR_vertex_attribute_divisor, VK_EXT_host_image_copy (optional under some queue conditions), VK_EXT_pipeline_protected_access, VK_EXT_pipeline_robustness.

- **1.4 release (V, Khronos press release, Dec 2024):** push descriptors, dynamic rendering local read, scalar block layout mandatory; maintenance up to maintenance6 in core. https://www.khronos.org/news/press/khronos-streamlines-development-and-deployment-of-gpu-accelerated-applications-with-vulkan-1.4
- **Roadmap 2026 (V, Khronos blog, 2026-01-23):** https://www.khronos.org/blog/vulkan-introduces-roadmap-2026-and-new-descriptor-heap-extension . Required: variable rate shading, shader clock queries, host image copies, compute shader derivatives, swapchain improvements, higher limits.
- **Vulkan 1.4.340 (S):** 2026-01-23 (https://www.phoronix.com/news/Vulkan-1.4.340-Descriptor-Heap).
- **VK_KHR_unified_image_layouts (V):** ratified, ext number 528, rev 1; removes most layout transitions; feature flags `unifiedImageLayouts`, `unifiedImageLayoutsVideo`. Not in the 1.4 core list. Sources: https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_unified_image_layouts.html ; https://www.khronos.org/blog/so-long-image-layouts-simplifying-vulkan-synchronisation . Known issue: Dear ImGui #9373 validation errors (S).
- **VK_EXT_descriptor_heap (V):** EXT-level; blog: "intended to completely replace Vulkan's existing descriptor set mechanism", feedback sought "before finalizing the API as a KHR extension". Present in SDK headers. NVIDIA driver 610+ (S, https://developer.nvidia.com/blog/streamlining-resource-binding-with-end-to-end-support-for-vulkan-descriptor-heaps/). Replacing VK_EXT_descriptor_buffer: **UNVERIFIED**.
- **VK_EXT_shader_object:** in SDK headers; promotion status **UNVERIFIED**.
- **VK_EXT_descriptor_buffer:** status **UNVERIFIED** beyond samples.

## 6. GPU hardware background

| Topic | Source | Status |
|---|---|---|
| Queues, queue families, heaps/types, command pools | Sawicki, "Understanding Vulkan Objects" (2017): https://gpuopen.com/learn/understanding-vulkan-objects/ | V |
| Device-local heap not mappable; numbering not guaranteed, use property flags | AMD GPUOpen, "Using Vulkan Device Memory": https://gpuopen.com/learn/vulkan-device-memory/ | V (upload-heap section not confirmed) |
| Memory types on PC | https://asawicki.info/news_1740_vulkan_memory_types_on_pc_and_how_to_use_them | UNVERIFIED (not fetched) |
| Command processor, GPU memory | fgiesen, "A trip through the Graphics Pipeline 2011" index: https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/ (part 2) | V (index) |
| Load/store ops on tile GPUs | Arm: https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/picking-the-most-efficient-load-store-operations | S |
| Queue family to HW engine | https://docs.vulkan.org/guide/latest/queues.html | S; vendor-specific, no authoritative table found |
| Raph Levien writing | none found | UNVERIFIED |

## 7. Gaps / UNVERIFIED

1. Full official tutorial chapter list.
2. Whether the main tutorial uses synchronization2, CMake or vcpkg in text.
3. Exact launch date of Building a Simple Engine.
4. ash master beyond 0.38.
5. VMA, bindless, BDA in vkguide.
6. Promotion status of VK_EXT_shader_object, VK_EXT_descriptor_buffer, VK_KHR_unified_image_layouts.
7. vulkanalia and vulkan-tutorial-rs: Vulkan target and last-commit status.
8. HW queue-to-engine mapping; Raph Levien sources.
9. Two URLs returned 404 (a GPUOpen URL and the GitHub tree of Vulkan-Tutorial/en/latest).
