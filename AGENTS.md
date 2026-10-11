# vulkan-course — working notes for agents

Keep this file current: structure, decisions, traps, TODOs (`[ ]` / `[x]`). `CLAUDE.md` only imports it.
The engine lives in a separate repo (`../course-kit`, `ssoj13/course-kit`); read its AGENTS.md before touching
anything generic (sidebar, links, progress, mind maps, quiz).

## What this is

Interactive, visualization-first textbook "Vulkan from zero" (English + Russian), published on GitHub Pages at
https://ssoj13.github.io/vulkan-course/ (repo `ssoj13/vulkan-course`). Format: roadmap.sh-style course map,
concept mind maps everywhere, live scrubbable visualizations of Vulkan objects, frames and synchronization.

Reader profile: CG/VFX developer (Maya, Houdini, shaders), Rust and C++; wants very gentle, step-by-step
explanations. Explain through CG analogies and always show the "depth stack": concept → API object →
code (Rust and C++) → what the driver and the GPU actually do.

## Decisions (2026-10-10)

- Languages: English (base path) and Russian (`/ru/`), see "Languages" below. Hosting: GitHub Pages. Toolchain: bun 1.4.3.
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

## Languages

- `astro.config.mjs` `languages` lists them; the first is the default. Engine texts come from course-kit (`i18n.mjs`).
- Per language: `src/data/course.<lang>.json`, `terms.<lang>.json` (overlays: texts only, complete or the build
  fails), `src/data/objects.<lang>.json` (object notes; checked by `overlayErrors()` in tests and in ObjectGraph.astro),
  lessons under `src/content/docs/<lang>/` with the same slugs.
- This course's widget texts (ObjectGraph, simulator remarks): `src/i18n.ts` (`courseText(lang)`, parity test
  `tests/i18n.test.ts`). The simulator returns structured remarks (`SimMsg` + vars); validation reports stay verbatim
  English (that is what the real layer prints).

## Writing (every lesson, every language)

The user's standard: serious things explained properly - the big picture, what follows from what, how concepts
relate, hierarchies of concepts ("mutex -> semaphore -> barrier, cond var, atomics"), mind maps. Each language
version is AUTHORED for its reader, never translated sentence by sentence: same facts, same code regions, its own
wording, examples and emphasis.

Lesson shape:
1. The question the lesson answers (1-3 sentences).
2. "Where this fits" (ru: "Место в курсе"): the layer, what came before, what this unlocks, and the concept
   chain in one line (e.g. GPU -> driver -> loader -> VkInstance -> VkPhysicalDevice -> VkDevice -> VkQueue).
3. Body: CG analogies; every new term defined as "X is Y built on Z" and linked (`term:` / `<Term>`).
4. Code through `<Step file region>`: say what the region does and why, not just what it is.
5. `<MindMap />` of the lesson and "Underneath" (loader, driver, hardware).
6. Quiz (`course-kit` model): 3-4 questions, each option `{ text, why }` - `why` names the misconception and
   corrects it (wrong) or states the reason (right), never just "wrong"; `ref` = `#anchor` of the heading that
   explains it; the correct option is the longest in at most a third of the questions.

Facts: verified against the spec registry (`validusage.json`, `vk.xml`) or real runs on the dev machine (quote
those as real output); never invented. Check content with `bun run lint` before a build.

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

- Engine: `github:ssoj13/course-kit#v0.3.4` (public). Bump the tag in package.json for engine changes.
- examples 01-05 in Rust (ash git master, Vulkan 1.4) and C++ (vk::raii, ApiVersion14, C++23 for std::print):
  built and run on RTX 3080 Ti; reviewed by two independent reviewers (code vs spec, lesson facts) and fixed.
  Rust and C++ print identical output except handle addresses (verified 2026-10-10 by diffing real runs; before that
  01/02/04/05 had drifted). Program output quoted in lessons is pasted from real runs, cuts marked "(shortened)".
  Step 05 prints only what it needs (first GRAPHICS family), not the whole 03/04 report. Step 03 chains
  Vulkan13Features only on devices >= 1.3 (a pNext struct must belong to a supported version). Step 05 chains
  Vulkan13Features directly onto VkDeviceCreateInfo (ash master's `push()` panics on a struct that already has its own chain).
- Debug messenger: WARNING|ERROR only (INFO = 100+ loader lines). The dev machine has 8 implicit layers (6 from apps,
  2 from the NVIDIA driver); VK_LAYER_bandicam_helper prints one expected warning, used as the live example.
- ObjectGraph: data `src/data/objects.json`, VUID texts `src/data/vuids.json` (generated: `bun scripts/sync-vuids.ts`),
  simulator `src/lib/objects.ts` (tests in tests/), view `src/components/ObjectGraph{.astro,View.tsx}`.
- GPU snapshot for the device/queue/memory labs: `src/data/gpu.json` (RTX 3080 Ti, driver 616.64) from `bun run sync:gpu`
  (`scripts/sync-gpu.ts`: vulkaninfo `--json=0` for device/features/limits/queues + `--text` for memory heaps/types,
  which the profile JSON lacks); `tests/gpu.test.ts` pins it to the facts the lessons quote. Rerun after a driver update.
- Quizzes: graded on click; each option carries a `why`, each question a `ref` to the section that explains it.
  `bun run lint` (scripts/lint-content.ts) applies the engine's `quizErrors` (model; the longest and the shortest
  option may each be the answer in at most a third of a quiz) and checks `#anchor` refs, for every language, before a
  build; the engine checks again at build time.
- Queue order wording, course-wide: submission order is the logical order barriers and semaphores refer to; by itself
  it orders neither start nor finish, even within one queue (exception: batches of one vkQueueSubmit begin in order);
  between queues there is not even that.
- Repo `ssoj13/vulkan-course` (public, Pages via Actions). `task.md` (private notes) and `.omh/` are gitignored and
  were never published; the pre-publication local history was deleted.

## TODO

- [x] Stage 1: course-kit engine; [x] site on the kit; [x] examples 01-05 both languages; [x] lessons 1-10
- [x] ObjectGraph map + build modes; [x] two reviewers, findings fixed; [x] CI (tests, clippy/fmt, Pages)
- [x] i18n (2026-10-10, deployed): EN lessons upgraded ("Where this fits", teaching quizzes), RU authored (terms.ru.json, index, 10 lessons),
  reviewed by independent readers, built, screenshots in both languages, deployed
- [ ] C++ examples in CI (needs Vulkan SDK headers 1.4.341 with vulkan_raii.hpp on the runner)
- [ ] Decide with the user: term maps put foundations at the bottom, ObjectGraph puts the root on top
- [ ] User review, then modules 4+ (presentation, pipeline, commands, sync, memory, descriptors, images, compute)
