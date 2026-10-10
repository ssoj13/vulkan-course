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

- Language: English. Hosting: GitHub Pages. Toolchain: bun 1.3.12.
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

- course-kit pushed (public, github:ssoj13/course-kit#main; tag v0.1.0 once this site builds on it).
- examples/rust (ash 0.38, API 1.3) and examples/cpp (vk::raii, C++23 for std::print) steps 01-05 written by agents,
  built and run on RTX 3080 Ti, region contract checked. NOT yet independently reviewed.
- Debug messenger: WARNING|ERROR only (INFO = 100+ loader lines). VK_LAYER_bandicam_helper (implicit layer on the
  dev machine) prints one expected warning: use it as the live example in the loader lesson.
- ObjectGraph: data `src/data/objects.json`, VUID texts `src/data/vuids.json` (generated: `bun scripts/sync-vuids.ts`),
  simulator `src/lib/objects.ts` (11 tests pass), view `src/components/ObjectGraph{.astro,View.tsx}` NOT yet built/viewed.
- terms.json seeded for modules 1-3; course.json has modules 1-3.

## TODO

- [x] Stage 1: course-kit engine
- [ ] Commit kit change: `layout()` takes `LayoutInput` (uncommitted in ../course-kit), push
- [ ] Add `@xyflow/react` + `@dagrejs/dagre` as explicit deps of this site; `bun install`
- [ ] Home page src/content/docs/index.mdx + the 10 lessons (modules 1-3) -> site builds (kit checks every lesson has a page)
- [ ] Build + look at ObjectGraph (map and build modes)
- [ ] Two independent reviewers for examples/rust and examples/cpp
- [ ] CI: Pages deploy (bun 1.4.3), cargo check, cmake build; create ssoj13/vulkan-course
- [ ] Queue-family lesson: use the real RTX 3080 Ti table (6 families) recorded by both examples
- [ ] User review, then remaining modules
