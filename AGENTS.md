# vulkan-course — working notes for agents

Keep this file current: structure, decisions, traps, TODOs (`[ ]` / `[x]`). `CLAUDE.md` only imports it.
The engine lives in a separate repo (`../course-kit`, `ssoj13/course-kit`); read its AGENTS.md before touching
anything generic (sidebar, links, progress, mind maps, quiz).

## What this is

Interactive, visualization-first textbook "Vulkan from zero" (English), published on GitHub Pages at
https://ssoj13.github.io/vulkan-course/ (repo `ssoj13/vulkan-course`). Format: roadmap.sh-style course map,
concept mind maps everywhere, live scrubbable visualizations of Vulkan objects, frames and synchronization.

Reader profile: CG/VFX developer (Maya, Houdini, shaders), Rust and C++; wants very gentle, step-by-step
explanations. Explain through CG analogies and always show the "depth stack": concept → API object →
code (Rust and C++) → what the driver and the GPU actually do.

## Decisions (2026-10-10)

- Language: English. Hosting: GitHub Pages. Toolchain: bun 1.4.3.
- Engine: `course-kit` (separate repo), consumed as a git dependency `github:ssoj13/course-kit#<tag>`;
  local co-development via `bun link`. Everything course-agnostic goes there, never into this repo.
- Code examples: BOTH Rust (`ash`) and C++ (Vulkan-Hpp), in this repo: `examples/rust` (cargo workspace),
  `examples/cpp` (CMake + vcpkg). Lessons show code only through region includes from these files,
  so every snippet compiles (missing region = build error).
- Vulkan cannot run in the browser: live parts are simulations of Vulkan state (object graph, frame timeline,
  barriers, layouts, memory) driven by the same data the lesson text uses; real programs run locally.
- Local SDK: Vulkan SDK 1.4.341.1 at `C:\Programs\VulkanSDK\1.4.341.1`.
- Target **Vulkan 1.4** (user: "учим последнее"). Rust uses ash from git pinned to master commit
  f4c2ca3e4f6b998d5254ad101a32f024d87cdec2 (Vulkan-Headers 1.4.352): the last crates.io release 0.38 (2024) stops at
  1.3.281. Switch to ash 0.39 when it is released. C++: vk::ApiVersion14.
- Research notes: `notes/research-vulkan-ecosystem.md` (versions, references, sources).

## Core principles

1. Every concept has a place on the object map and in the term graph (`src/data/terms.json`): one source
   for the roadmap, per-lesson mind maps, genealogy ("X is built on Y") and the glossary.
2. Every animation is a scrubbable timeline (step / back / scrub), never a video.
3. No fake behavior: simulators model real Vulkan rules (valid usage, stage/access scopes, layout transitions);
   validation-style messages quote the real VUID text where one exists.
4. Each lesson: intuition (CG analogy) → objects and their parents → code (Rust / C++ tabs) → what happens
   underneath (loader, ICD, command processor, HW queues, caches) → self-check quiz.

## Curriculum (draft)

1. Foundations: what a GPU is; how a program talks to it (driver, commands, queues); why Vulkan (explicit vs GL
   state machine); the map of Vulkan objects.
2. Instance: toolchain (SDK, vulkaninfo, vkconfig, Rust/C++ projects); loader and instance; validation layers
   and debug messenger.
3. Devices: physical devices (properties, features, limits); queue families = hardware engines; logical device
   and queues.  ← first reference section ends here (review by the user)
4. Presentation: window and surface; swapchain and present modes; image views.
5. Pipeline: shaders and SPIR-V (Slang); graphics pipeline stages; pipeline layout; dynamic rendering.
6. Commands and the first triangle: command pools/buffers, recording, submission.
7. Synchronization: fences vs semaphores, frames in flight, pipeline barriers (sync2), image layouts,
   timeline semaphores, swapchain recreation.
8. Resources and memory: heaps and types, buffers, staging, allocators (VMA / gpu-allocator), vertex/index data.
9. Descriptors: uniform buffers, descriptor sets/layouts/pools, push constants, buffer device address, bindless.
10. Images: textures, samplers, depth, mipmaps, MSAA.
11. Compute: compute pipelines, particles, async compute.
Advanced: debugging/profiling (RenderDoc, Nsight, timestamps), pipeline cache, multithreaded recording,
shader objects, mesh shaders, ray tracing, tile-based mobile GPUs. Capstone: mini renderer (glTF + PBR).

## Visualizations (planned, Vulkan-specific, live in this repo)

- ObjectGraph: build the Vulkan object tree by "creating" objects in order; wrong order → validation message.
- FrameTimeline: CPU/GPU lanes, frames in flight, fences/semaphores, scrubbable.
- BarrierLab: pipeline stages with src/dst stage+access masks; shows what a barrier blocks.
- LayoutMachine: image layout state machine with transitions.
- MemoryHeaps: heaps/types of real GPUs, allocation, staging, ReBAR.
- DescriptorWiring: shader set/binding ↔ resources.

## State (2026-10-10)

- Engine: `github:ssoj13/course-kit#v0.1.0` (public). Bump the tag in package.json for engine changes.
- examples 01-05 in Rust (ash git master, Vulkan 1.4) and C++ (vk::raii, ApiVersion14, C++23 for std::print):
  built and run on RTX 3080 Ti; reviewed by two independent reviewers (code vs spec, lesson facts) and fixed.
  Both languages print the same output; step 05 chains Vulkan13Features directly onto VkDeviceCreateInfo
  (ash master's `push()` panics on a struct that already has its own chain).
- Debug messenger: WARNING|ERROR only (INFO = 100+ loader lines). The dev machine has 8 implicit layers (6 from apps,
  2 from the NVIDIA driver); VK_LAYER_bandicam_helper prints one expected warning, used as the live example.
- ObjectGraph: data `src/data/objects.json`, VUID texts `src/data/vuids.json` (generated: `bun scripts/sync-vuids.ts`),
  simulator `src/lib/objects.ts` (tests in tests/), view `src/components/ObjectGraph{.astro,View.tsx}`.
- Quizzes: the engine fails the build if more than 1/3 of a quiz has the correct option as the longest.
- Repo `ssoj13/vulkan-course` (public, Pages via Actions). `task.md` (private notes) and `.omh/` are gitignored and
  were never published; the pre-publication local history was deleted.

## TODO

- [x] Stage 1: course-kit engine; [x] site on the kit; [x] examples 01-05 both languages; [x] lessons 1-10
- [x] ObjectGraph map + build modes; [x] two reviewers, findings fixed; [x] CI (tests, clippy/fmt, Pages)
- [ ] C++ examples in CI (needs Vulkan SDK headers 1.4.341 with vulkan_raii.hpp on the runner)
- [ ] Decide with the user: term maps put foundations at the bottom, ObjectGraph puts the root on top
- [ ] User review, then modules 4+ (presentation, pipeline, commands, sync, memory, descriptors, images, compute)
