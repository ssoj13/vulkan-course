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
- Builds share the machine with other sessions (2026-10-11: parallel builds exhausted memory and disk). Run every
  cargo command (examples/rust, tools/vk-probe) and the MSVC C++ build through the machine-wide lock, at most 6 jobs:
  `$env:CARGO_BUILD_JOBS=6; python C:/projects/projects.rust.cg/cglibs/.buildlock/buildlock.py cargo ...`.
  Running built binaries needs no lock. Keep target/ dirs small; never touch repos under cglibs/ without asking.

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

## Labs (12, 2026-10-10)

Interactive widgets with a full page each (`/labs/<slug>`, `/ru/labs/<slug>`; catalog and sidebar from the engine).
Manifest: `labs` in `src/data/course.json` / `course.ru.json` (slug, lesson, title, summary). Brief the lab agents
followed: one question per lab, no fake behavior (rules from the spec/registry, numbers from the data files below or
labeled illustrative, every simplification stated on the page), scrubbable timelines, keyboard/aria, mobile width,
authored (not translated) Russian.

Files per lab `<slug>`: model `src/lib/labs/<slug>.ts` (pure, no DOM), view `src/components/labs/<Name>.astro`
(build-time data checks, text fallback) + `<Name>View.tsx` (React island, `client:only="react"`), texts
`src/components/labs/<slug>.text.ts` (`TEXT.en/ru`, picked by `pick()` from `src/lib/text.ts`), styles
`src/components/labs/<slug>.css`, tests `tests/labs/<slug>.test.ts` (model facts + `textErrors`), pages
`src/content/docs/{,ru/}labs/<slug>.mdx`. Engine primitives: `course-kit/components/lab/{LabFrame,Scrubber,useTimeline}`.

| Lab | Component | Lesson (embedded EN+RU) |
|---|---|---|
| warp-divergence | WarpDivergence | foundations/gpu |
| memory-path | MemoryPath | foundations/gpu |
| command-path | CommandPath | foundations/talking-to-gpu |
| gl-vs-vulkan | GlVsVulkan | foundations/why-vulkan |
| loader-chain | LoaderChain | instance/loader-and-instance |
| messenger-filter | MessengerFilter | instance/validation |
| device-picker | DevicePicker | devices/physical-devices |
| queue-planner | QueuePlanner | devices/queue-families |
| object-builder | ObjectGraph (`src/components/ObjectGraph*`, model `src/lib/objects.ts`) | devices/logical-device |
| queue-sync | QueueSync | sync/fences-and-semaphores (EN + RU) |
| graphics-pipeline | GraphicsPipeline | pipeline/graphics-pipeline (EN + RU) |
| buffer-memory | BufferMemory | memory/memory-types (EN + RU) |

Widget texts: fill placeholders ONLY with `tfmt(lang, text, vars)` from `src/lib/text.ts` (never the engine `fmt`).
Counts use ICU-style plural placeholders `{n|# очередь|# очереди|# очередей}` (ru: one/few/many, en: one/other, `#` =
the number; a form without `#` is the word alone, e.g. units after a shown value). `textErrors` checks the form count
per language; no "X: {n}" or "(s)" workarounds. `bun run lint` also fails on «лаба»/«лаборатория» (RU word is
«интерактив», masculine) and on « - » used as a dash in ru/ pages and `ru` text tables.

Shared modules: `src/lib/labs/memory.ts` (heaps/types of gpu.json and the ONE find-memory-type rule `findType(bits,
flags)`, used by memory-path and buffer-memory), `src/lib/vuids.ts` (`vuid(id)`: the only way to read a VUID text;
throws for an id that was never synced).

Real data and how to refresh it (all from the dev machine: RTX 3080 Ti, driver 616.64, SDK 1.4.341.1):

| File | What | Refresh |
|---|---|---|
| `src/data/gpu.json` | vulkaninfo snapshot: device, features, limits, queue families, memory heaps/types | `bun run sync:gpu` |
| `src/data/buffer-requirements.json` | vkGetBufferMemoryRequirements for 63 usage combinations x 5 sizes (memoryTypeBits = 59: type 2 never allowed for buffers) | `bun run sync:buffers` (tools/vk-probe) |
| `src/data/validation-log.json` | step 02's messages at every severity: totals, leak/dab experiments, exact sample | `bun run sync:validation-log` (tools/vk-probe) |
| `src/data/dynamic-states.json` | core VkDynamicState values with their version, from vk.xml | `bun run sync:dynamic-states` |
| `src/data/vuids.json` | registry texts of every `VUID-...` id mentioned anywhere in src/ (scripts/vuid-refs.ts finds them) | `bun run sync:vuids` |

- VUIDs: write the id in a model, data file or lesson, run `bun run sync:vuids`; never paste a VUID text.
  `tests/vuids.test.ts` fails for an id without its synced text.
- `tools/vk-probe` (cargo, ash at the examples' pin): binaries `buffer-requirements` and `validation-log`, see its
  README. Its output is machine-specific by design; handle values in messages change every run (the lesson quotes
  its own run). Every capture's `date` is the UTC calendar date (vk-probe and scripts/sync-gpu.ts alike; the
  current gpu.json predates the rule and carries the local date of its run). CI lints it (clippy/fmt) next to examples/rust; running needs a GPU.
- `REBAR.md`: Resizable BAR on the dev machine (off now: the DEVICE_LOCAL|HOST_VISIBLE heap is 214 MiB). If it is
  switched on, rerun sync:gpu and sync:buffers, then fix the facts the memory labs, lessons and tests state.

## State (2026-10-10)

- Engine: `github:ssoj13/course-kit#v0.4.1` (public). Bump the tag in package.json for engine changes. v0.4.1 (prepared
  locally, not released yet) ships `src/*.d.mts` declarations for the engine's .mjs modules; until it is released,
  `bun run check` passes only with those files copied into node_modules/course-kit/src. v0.4.2 (prepared locally, not committed/tagged): /labs catalog h2 ids (module slug, `ahead`) + "On this page"; its labs.astro is copied into node_modules for local builds until released.
- examples 01-05 in Rust (ash git master, Vulkan 1.4) and C++ (vk::raii, ApiVersion14, C++23 for std::print):
  built and run on RTX 3080 Ti; reviewed by two independent reviewers (code vs spec, lesson facts) and fixed.
  Rust and C++ print identical output except handle addresses (verified 2026-10-10 by diffing real runs; before that
  01/02/04/05 had drifted). Program output quoted in lessons is pasted from real runs, cuts marked "(shortened)".
  Step 05 prints only what it needs (first GRAPHICS family), not the whole 03/04 report. Step 03 chains
  Vulkan13Features only on devices >= 1.3 (a pNext struct must belong to a supported version). Step 05 chains
  Vulkan13Features directly onto VkDeviceCreateInfo (ash master's `push()` panics on a struct that already has its own chain).
- Debug messenger: WARNING|ERROR only (INFO = 100+ loader lines). The dev machine has 8 registered implicit layers (only NV_optimus, NV_present and bandicam_helper load
  unconditionally; RenderDoc, EOS, Steam overlay, Fossilize and Nsight need their enable_environment variable) (6 from apps,
  2 from the NVIDIA driver); VK_LAYER_bandicam_helper prints one expected warning, used as the live example.
- ObjectGraph: data `src/data/objects.json`, VUID texts via `src/lib/vuids.ts` (`bun run sync:vuids`),
  simulator `src/lib/objects.ts` (tests in tests/), view `src/components/ObjectGraph{.astro,View.tsx}`.
- GPU snapshot for the device/queue/memory labs: `src/data/gpu.json` (RTX 3080 Ti, driver 616.64) from `bun run sync:gpu`
  (`scripts/sync-gpu.ts`: vulkaninfo `--json=0` for device/features/limits/queues + `--text` for memory heaps/types,
  which the profile JSON lacks); `tests/gpu.test.ts` pins it to the facts the lessons quote. Rerun after a driver update.
- Type checking: `bun run check` (astro check, TS 6.0.x like the engine) must report 0 errors/warnings/hints;
  tsconfig `types: ["bun"]` for tests/scripts; engine lib modules are imported with their extension
  (`course-kit/lib/course.ts`), like the components.
- Quizzes: graded on click; each option carries a `why`, each question a `ref` to the section that explains it.
  `bun run lint` (scripts/lint-content.ts) applies the engine's `quizErrors` (model; the longest and the shortest
  option may each be the answer in at most a third of a quiz) and checks `#anchor` refs, for every language, before a
  build; the engine checks again at build time.
- Queue order wording, course-wide: submission order is the logical order barriers and semaphores refer to; by itself
  it orders neither start nor finish, even within one queue (exception: batches of one vkQueueSubmit begin in order);
  between queues there is not even that.
- Repo `ssoj13/vulkan-course` (public, Pages via Actions). `task.md` (private notes) and `.omh/` are gitignored and
  were never published; the pre-publication local history was deleted.
- examples 06-09 (modules 4-6, 2026-10-10): 06_window_surface, 07_swapchain, 08_pipeline, 09_triangle in both
  languages; run on the RTX 3080 Ti, stdout identical (no handles printed), validation clean (core + sync + best
  practices; only the bandicam implicit-layer warning). Rust: winit =0.30.13 (ApplicationHandler; Vulkan setup in
  `Vk::new` from `resumed`), ash-window from the SAME ash git rev (crates.io 0.13 = ash 0.38 types). C++: GLFW 3.5.1
  via vcpkg manifest (`examples/cpp/vcpkg.json`, presets set the vcpkg toolchain: old build dirs need `--fresh`);
  short enum names from `vk_enum_string_helper.h` so output matches ash's Debug. Every region name exists in both
  files of a step (Step.astro shows both tabs). Window fixed 800x600 physical px; minimizing gives OUT_OF_DATE on this
  machine and step 09 stops with a message (recreation = module 7). 09: one frame in flight, one render-finished
  semaphore PER SWAPCHAIN IMAGE, barriers chained to the semaphore stages (COLOR_ATTACHMENT_OUTPUT). Shaders: one
  source `shaders/*.slang` -> committed `shaders/spv/*.spv` + `manifest.json` (`bun run sync:shaders`, slangc
  -fvk-use-entrypoint-name, spirv-val vulkan1.4); `tests/shaders.test.ts` catches stale SPIR-V. Slang SV_VertexID =
  VertexIndex-BaseVertex (needs DrawParameters), so triangle.slang uses SV_VulkanVertexID. Screenshot:
  `examples/screenshots/09_triangle.png` (Rust; C++ renders the same). NVIDIA best-practices layer flags the non-0/1
  clear color (no fast clear) - a possible lesson aside. NOT reproduced on 2026-10-10 (env VK_KHRONOS_VALIDATION_VALIDATE_BEST_PRACTICES[_NVIDIA]=true,
  messenger WARNING|ERROR: only the pageable-device-local-memory PERFORMANCE warning), so no lesson mentions it.
- Lessons of modules 4-6 (EN, 2026-10-10): presentation/{window-and-surface,swapchain,image-views},
  pipeline/{shaders-and-spirv,graphics-pipeline,pipeline-layout,dynamic-rendering}, commands/{command-buffers,first-triangle};
  every 06-09 region is shown once (09 begin-rendering in dynamic-rendering, 09 submit in command-buffers). Program
  output quoted from real Rust runs (window closed via WM_CLOSE). Screenshot `src/assets/09_triangle.png` shown with
  `<Image format="png">`: the lesson quotes measured pixels (bg 39,39,48 = sRGB of 0.02/0.03; centroid 156,157,156 =
  sRGB of 1/3), and the default WebP would change them. RU pages of these 9 lessons are PLACEHOLDERS. 32 terms added
  (fence/semaphore/image-layout/frame-loop point at first-triangle until module 7 owns them; vk-descriptor-set-layout
  has no lesson yet). objects.json: `term` links for all presentation..sync objects, and ParentRef `createOnly`
  (shader-module -> pipeline: destroying the module leaves the pipeline valid, spec of vkDestroyShaderModule).
  ObjectGraph.astro draws a parent that is outside the scope through the in-scope objects that are a kind of it (`isA`).

- examples 10-12 (module 7, 2026-10-11): 10_frames_in_flight (per-slot cmd/fence/acquire semaphore, render_finished per
  image; --frames/--cpu-ms/--draws/--seconds), 11_resize, 12_timeline; CMake STEPS/WINDOW_STEPS include 11, 12.
  Measured 10 (2026-10-11, RELEASE both languages, validation on, sync validation off, 4 s, default 2 ms CPU + 5000
  instances, shared machine, medians of 5 alternating runs): Rust/C++ 1 frame 227/228 fps, fence wait 2.03/2.07 ms
  (46-47%); 2 frames 419/432 fps, 0.048 ms; 3 frames 420/429 fps, 0.047/0.046 ms. "Rest" of the CPU frame ~0.35 ms at
  F=1 and F=2. Sync validation (F=2, 5 alternating pairs): Rust 419 -> 408, C++ 432 -> 418 fps.
  12: timeline counter replaces fences (frame n signals n at ALL_COMMANDS; host waits n - frames); binary
  image_available/render_finished for acquire/present (VUID-vkAcquireNextImageKHR-semaphore-03265,
  VUID-vkQueuePresentKHR-pWaitSemaphores-03267); one offscreen 400x300 B8G8R8A8_SRGB image (memory type 1, DEVICE_LOCAL),
  NEAREST blit to the swapchain; acquire semaphore waited at BLIT, render_finished signaled at BLIT; region `barriers` =
  5 sync2 barriers. TRAP (fixed 2026-10-11 after review): a barrier whose srcStageMask overlaps a semaphore wait's
  stage chains behind that wait, with everything it holds back. The old barrier 1 (offscreen, src BLIT) chained the
  rendering to the acquire. Now: barrier 1 src COLOR_ATTACHMENT_OUTPUT/NONE (WAW with the previous color writes, made
  available by barrier 3), barrier 5 at the END of the frame on the offscreen image (BLIT/NONE -> COLOR_ATTACHMENT_OUTPUT/
  NONE, layout unchanged) = WAR on the blit read, chained to the next frame's barrier 1. Sync validation proven live:
  removing BLIT from barrier 5 gives "WRITE_AFTER_READ hazard ... vkCmdPipelineBarrier2 writes to VkImage ..., which
  was previously read by vkCmdBlitImage2" (quoted in the lesson). GPU timestamps (regions `timestamps`,
  `read-timestamps`, 3 per frame, printed once per second as `[timing]`): render done -> blit done = blit time
  (~7 us) in MAILBOX and in a forced-FIFO test, i.e. the overlap is NOT measurable on the RTX 3080 Ti; the lesson
  says so. Swapchain usage is TRANSFER_DST +
  COLOR_ATTACHMENT: with TRANSFER_DST only, VK_LAYER_bandicam_helper creates views/framebuffers on the swapchain images
  and validation reports VUID-VkImageViewCreateInfo-image-04441 + VUID-VkFramebufferCreateInfo-pAttachments-00877
  (gone with VK_LAYER_bandicam_helper_DEBUG_1=1, the layer's disable_environment; VK_LOADER_LAYERS_DISABLE did not
  disable it). Counter values after the host wait are `[timing]` (frame 3 waits for 1, reads 2).
  11: VK_KHR_swapchain_maintenance1 (+ instance VK_KHR_surface_maintenance1 + VK_KHR_get_surface_capabilities2) is used
  on the dev GPU: present fence per slot, waited with a 1 s timeout (spec not explicit on fences of a present that
  returned OUT_OF_DATE: its queue operations are "still considered to be enqueued"; timeout -> vkDeviceWaitIdle, fence
  abandoned; at exit a present fence that never signaled is NOT destroyed (VUID-vkDestroyFence-fence-01120), Rust skips
  the destroy, C++ raii `release()`; never triggered in runs). Wording course-wide: vkDeviceWaitIdle waits for queue
  work (presents' queue operations included), not for the presentation engine; present fences exist to know when a
  present's resources can be released. Spec facts checked in vkspec 1.4.365 (all extensions); the core-only spec has
  "Regarding interaction with vkQueueWaitIdle, vkQueuePresentKHR behaves as if it signals a fence payload ...", absent
  from the all-extensions build. D3D12: the timeline-semaphore extension issues call D3D12 fences "semantically
  close/identical"; external_semaphore_win32 imports a D3D12 fence into a VkSemaphore (no "recommends"). Retired chain destroyed
  once frame >= retired_at + FRAMES - 1 has waited its slot's present fence. `--no-present-fences` = vkDeviceWaitIdle
  before each recreation. `--demo-resize`: 1 s 1000x700, 2 s 640x480, 3 s minimize, 4 s restore; 3 recreations in every
  run, all triggered by resize events (no OUT_OF_DATE/SUBOPTIMAL was returned on this machine in these runs). winit
  delivers a stale Resized (2564x984) right after creation, so 11 compares the window's current inner_size with the
  swapchain extent. VUID for the 0x0 surface: VUID-VkSwapchainCreateInfoKHR-imageExtent-01689. Run with --seconds
  closing the window itself; no `sync:vuids` run yet for the VUIDs named in examples (they are in code comments only).
- examples 13-16 (module 8, 2026-10-11): 13_memory_types (no window; heaps/types/limits, requirements of 6 usages,
  find_type = memory.ts `findType` + required/preferred two-pass `choose_type`, allocate/bind/map, two buffers in one
  allocation), 14_vertex_buffer (indexed hexagon 7 verts/18 u16 indices, HOST_VISIBLE|COHERENT type 3, new
  `shaders/mesh.slang` with `[[vk::location]]`, FRAMES_IN_FLIGHT = 2, `--seconds`), 15_staging (default: staging type 3
  -> DEVICE_LOCAL type 1 + vkCmdCopyBuffer on the graphics queue + global barrier COPY/TRANSFER_WRITE ->
  VERTEX_ATTRIBUTE_INPUT|INDEX_INPUT; `--transfer-queue`: dedicated family 1, release/acquire buffer barriers + binary
  semaphore at ALL_COMMANDS (spec: required unless the barriers set
  VK_DEPENDENCY_QUEUE_FAMILY_OWNERSHIP_TRANSFER_USE_ALL_STAGES_BIT_KHR, maintenance8 feature only); `--bar`: DEVICE_LOCAL|HOST_VISIBLE|
  HOST_COHERENT type 5, heap 2; `--mib N` pads the vertex buffer; heap-size check before vkAllocateMemory =
  VUID-vkAllocateMemory-pAllocateInfo-01713, `--bar --mib 300` fails cleanly), 16_allocator (no window; `--count`,
  default 3000; raw vkAllocateMemory per buffer vs. sub-allocation). Real facts: maxMemoryAllocationCount =
  4294967295 on this driver (NOT 4096; the spec minimum is 4096); buffer memoryTypeBits 0b111011; INDEX_BUFFER
  alignment 4, UNIFORM 64, others 16. Rust 16: own LinearAllocator (64 MiB blocks, bump, reset/destroy only) because
  gpu-allocator 0.28.0 (latest, also main) needs crates.io ash 0.38; C++ 16: VMA 3.4.0 from vcpkg (`<vk_mem_alloc.h>`;
  the SDK ships VMA 3.3.0 as `<vma/vk_mem_alloc.h>`, unused), preferredLargeHeapBlockSize 64 MiB, pfnAllocate counter.
  Lines about allocator internals start with `[lib]` (Rust: 2 blocks = 128 MiB; VMA: 4 blocks 8+16+32+64 MiB = its
  1/8, 1/4, 1/2 first-block rule); all other stdout identical. Measured (release, validation on, sync off, shared
  machine): 15 --mib 64: staging CPU memcpy ~6-8 ms (~9-11 GB/s), GPU copy ~5.3-6.7 ms (~10-13 GB/s) on either queue;
  --bar CPU writes ~15-17 ms (~4 GB/s; one C++ run 67 ms); 176-byte upload: copy round trip ~0.9-1.7 ms, BAR write
  ~0.02 ms. The old "~70 ms host writes with VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true" had no log and did NOT reproduce
  (re-measured 2026-10-11, see "Module 8 review fixes, round 2"). 16 (3000 buffers): raw ~130-170 us per create+allocate+bind, ~330 us per destroy+free; allocator
  ~3 us and ~2-3 us. Validation does NOT catch a missing post-copy barrier (presumably because a host fence wait sits between upload
  and draws, which sync validation treats as synchronizing; not confirmed) nor a missing acquire barrier: tried 2026-10-11, both silent - the
  barriers follow the spec and the Khronos synchronization examples, not a validation message.
- Module 8 review fixes (2026-10-11): 13 no longer evaluates LAZILY_ALLOCATED against buffer memoryTypeBits (spec: never
  for buffers; one stdout line points to images with TRANSIENT_ATTACHMENT, module 10); "uncached (typically
  write-combined)"; HOST_COHERENT writes = available in the host domain, the submit makes them VISIBLE (14/15
  write_mapped). 15: comments on fence = available not visible (the barrier/acquire makes the copy visible; validation
  is silent because sync validation is off by default and a host fence wait closes the batch's hazards), on the
  ownership transfer being required even though the semaphore wait already gives visibility (spec quote, "Resource
  Sharing Mode"; corrected 2026-10-11 in both comments to "Queue Family Ownership Transfer", where the sentence really is); `--bar` checks vertex+index together via vkGetDeviceBufferMemoryRequirements before allocating
  (`--bar --mib 214` = heap size 224395264 exactly now fails with "vertex + index buffers need 224395300 bytes ...";
  heap size is an upper bound, not the budget - VK_EXT_memory_budget). Rust 15 upload_staged: `UploadScratch` guard
  (device_wait_idle + destroy on drop = C++ IdleGuard + RAII). 16: `--count` checked against the heap before
  allocating (`--count 1000000` needs 23295976960 bytes > heap 0 12673089536 -> clean error); Rust raw and allocator
  paths destroy what was created on an error midway (C++: RAII / release()). Rust `Vertex` has a size assert (20).
- examples 17-20 (module 9, 2026-10-11): all start from 14's frame loop (2 frames in flight, HOST_VISIBLE|COHERENT
  mesh), regular hexagon radius 0.6 (aspect handled by the transform/shader, not the mesh). New shaders:
  `transform.slang` (FrameUniforms {float4x4 transform; float3 tint; float time} = 80 bytes, offsets 0/64/76;
  entry points vertexUniform = ConstantBuffer set 0 binding 0 -> SPIR-V struct `FrameUniforms_std140`, vertexPush =
  [[vk::push_constant]] -> `FrameUniforms_std430`, same offsets), `device_address.slang` (`Instance*` in a 16-byte
  push block; pointee layout named `Instance_natural`, stride 32), `bindless.slang` (`StructuredBuffer<Instance>
  objects[]` + NonUniformResourceIndex). Verified facts: Slang default matrix layout = column-major in memory (SPIR-V
  shows `RowMajor` + MatrixStride 16 and OpVectorTimesMatrix for mul(M, v): HLSL rows map to SPIR-V columns), so the
  host uploads columns; SV_InstanceID needs DrawParameters (BaseInstance) like SV_VertexID, so the shaders use
  SV_VulkanInstanceID. bindless SPIR-V (review fix 2026-10-11): Slang 2026.1 (SDK 1.4.341.1) did NOT declare
  `StorageBufferArrayNonUniformIndexing` (5308) for the non-uniform storage-buffer array index = UB per
  VUID-RuntimeSpirv-StorageBufferArrayNonUniformIndexing-10136 (worked only because NVIDIA is native). Slang has no
  capability atom for it (`slangc -h capability`: only spvShaderNonUniform[EXT]), so bindless.slang states it with
  `spirv_asm { OpCapability StorageBufferArrayNonUniformIndexing; }` (slangc hoists it). Second bug: Slang decorated
  only the descriptor access chain NonUniform, not the derived chain the OpLoad uses (VUID-RuntimeSpirv-subgroupSize-
  10149 wants the exact pointer operand; Mesa does not chase chains); Slang -O2 drops NonUniform entirely. Upstream
  shader-slang/slang#10525 (merged 2026-06) - tested Slang 2026.19 (scratch download): emits 5308 itself, but its
  member-wise loads still go through undecorated derived chains. Fix: `decorateNonUniform` in sync-shaders.ts adds
  NonUniform to every OpAccessChain/InBounds/Ptr/CopyObject/OpLoad result derived from a NonUniform id (result =
  glslang's pattern: chain + load decorated); tests pin each .spv's exact capability list (REQUIRED in
  tests/shaders.test.ts, a new shader must be added) and assert no missing derived NonUniform. Warning 41012 stays
  silenced: NonUniformResourceIndex is declared with the whole HLSL SM capability set (SPV_GOOGLE_user_type,
  DerivativeControl, ImageQuery, ...), none of which reaches the SPIR-V; 2026.19 warns the same.
  17: per-slot uniform buffer, persistently mapped (never unmapped; vkFreeMemory unmaps), set layout/pool (maxSets 2,
  no FREE_DESCRIPTOR_SET)/2 sets/one vkUpdateDescriptorSets; `reflect()` parses the SPIR-V (OpName/OpMemberName/
  OpTypePointer/OpVariable/OpDecorate set+binding/OpMemberDecorate Offset, plus RowMajor 4/ColMajor 5/MatrixStride 7)
  and the program refuses to start on a mismatch with offset_of!/offsetof or if `transform` is not RowMajor +
  MatrixStride 16 (negative test: transform.spv built with -matrix-layout-row-major -> "layout mismatch: shader
  transform is ColMajor MatrixStride 16, host expects RowMajor MatrixStride 16"). transform.slang lists the three
  std140/std430 differences, measured with slangc: array stride rounded to 16; nested struct aligned/sized to 16;
  MatrixStride 16 vs 8 for matrices with 2-float columns (float2x2, float2x3). C++ keeps plain set handles via raii `release()` (raii DescriptorSet frees in
  its destructor = needs FREE_DESCRIPTOR_SET). 18: `--push-descriptor`; push constants default (80-byte range,
  maxPushConstantsSize 256 here); push descriptors need Vulkan14Features.pushDescriptor enabled
  (VUID-vkCmdPushDescriptorSet-None-10356), maxPushDescriptors 32; device feature enabled only on that path.
  19: bufferDeviceAddress, buffer usage SHADER_DEVICE_ADDRESS only (no STORAGE_BUFFER), MemoryAllocateFlagsInfo
  DEVICE_ADDRESS; addresses printed on `[addr]` lines (new stdout exception next to [timing]/[lib]; on this machine
  both languages got 0x6f94000 every run, not guaranteed). 20: layout capacity 4096 (checked against UAB limits),
  set variable count 256, 63 written at start, the 64th written at frame 120 while the other slot may be pending;
  binding flags VARIABLE_DESCRIPTOR_COUNT|PARTIALLY_BOUND|UPDATE_AFTER_BIND|UPDATE_UNUSED_WHILE_PENDING, layout
  UPDATE_AFTER_BIND_POOL, pool UPDATE_AFTER_BIND. Capacity checked against min(maxPerStageDescriptorUAB
  StorageBuffers, maxDescriptorSetUAB StorageBuffers, maxPerStageUpdateAfterBindResources); pool's 256 against
  maxUpdateAfterBindDescriptorsInAllPools (no VUID: the spec says pool creation may fail beyond it). Limits on the
  RTX 3080 Ti: classic and UAB storage-buffer limits all 1048576, maxPerStageUpdateAfterBindResources and
  maxUpdateAfterBindDescriptorsInAllPools 4294967295, shaderStorageBufferArrayNonUniformIndexingNative true.
  18: "at most one push-descriptor set per pipeline layout" = VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00293
  (00365 is the vkCmdPushDescriptorSet set-number rule). 19: DEVICE_ADDRESS allocation is required for binding a
  SHADER_DEVICE_ADDRESS buffer (VUID-vkBindBufferMemory-bufferDeviceAddress-03339; the flag needs the feature,
  VUID-VkMemoryAllocateInfo-flags-03331). TRAP: VUID-vkUpdateDescriptorSets-None-03047 is satisfied by UPDATE_AFTER_BIND alone, so validation is silent
  without UPDATE_UNUSED_WHILE_PENDING (tried) although only that flag's definition covers updates while pending.
  GPU-AV (VK_KHRONOS_VALIDATION_GPUAV_ENABLE=true): 17/19/20 clean apart from GPU-AV's own setup warnings (it forces
  features like shaderInt64/bufferDeviceAddress and lowers maxUpdateAfterBindDescriptorsInAllPools to 4194304, so 20's
  printed limit changes under GPU-AV). Negative tests (temporary edits, reverted): 19 drawing 65 instances -> GPU-AV
  "Out of bounds access: 32 bytes read at buffer device address 0x6f94800" (VUID-RuntimeSpirv-PhysicalStorageBuffer64-11819);
  20 drawing one instance more than written (unwritten partially-bound element) -> VK_ERROR_DEVICE_LOST; GPU-AV
  printed nothing before the loss in that run (an observation on this machine, not something GPU-AV guarantees). Measured (release, sync validation on, 3 s, MAILBOX, not a benchmark): 2000-2900 fps for
  all four. Logs: scratchpad m9/ (session-local).
- Module 9 review fixes (2026-10-11, EN lessons + 17-20 comments): UpdateAfterBind limits count descriptors of ALL
  layouts ("with or without" UPDATE_AFTER_BIND_POOL), classic ones only layouts without it; step 20 labels now
  "(only layouts without UPDATE_AFTER_BIND_POOL)" / "(all sets, incl. UPDATE_AFTER_BIND)" (Rust = C++, rerun). Facts
  fixed in lessons: 03324 covers only vkGetBufferDeviceAddress (flag = VkMemoryAllocateInfo-flags-03331, no VUID ties
  SHADER_DEVICE_ADDRESS usage to the feature); pipeline/set match = layout-07988 (stage) + 07990 (type); disturbing
  rule = binding set N disturbs lower incompatible sets / all above N, binding a pipeline disturbs nothing; pool
  overflow "may fail"; push descriptors update incrementally; descriptor_heap "was developed based on issues discovered
  with VK_EXT_descriptor_buffer" (not "replaces"); descriptor sizes on the RTX 3080 Ti (descriptor_buffer props): UBO 8,
  SSBO 16, sampled image 4 bytes. Terms: subgroup has no lockstep guarantee; push-constant 128 (1.0) / 256 (1.4).
- examples 21-24 (module 10, 2026-10-11): all start from 17's frame loop (2 frames in flight, HOST_VISIBLE|COHERENT mesh,
  per-slot persistently mapped UBO + descriptor set). New shaders `textured.slang` (Camera {float4x4 mvp} at set 0
  binding 0, `Sampler2D albedo` = COMBINED_IMAGE_SAMPLER at binding 1; vertex pos3+uv2, stride 20; used by 21, 23) and
  `cube.slang` (Scene {float4x4 mvp[2]}, std140 ArrayStride 64; SV_VulkanInstanceID picks the matrix, second cube gets
  `.brg` colors; vertex pos3+color3, stride 24; used by 22, 24); both declare only `Shader` (pinned in
  tests/shaders.test.ts). 21_texture: 256x256 RGBA8 generated with integer math (FNV-1a 0xb2e278e1 in both languages;
  why: no decoder dependency, identical bytes), R8G8B8A8_SRGB (UNORM also printed), format check via
  vkGetPhysicalDeviceFormatProperties2 (SAMPLED_IMAGE, FILTER_LINEAR, TRANSFER_DST), image memoryTypeBits 0b11 ->
  type 1, size 262144 alignment 1024; upload = staging + barrier NONE/NONE -> COPY/TRANSFER_WRITE (UNDEFINED ->
  TRANSFER_DST) + vkCmdCopyBufferToImage2 + COPY/TRANSFER_WRITE -> FRAGMENT_SHADER/SHADER_SAMPLED_READ (-> SHADER_READ_ONLY),
  Rust UploadScratch guard / C++ IdleGuard declared after the RAII objects; sampler `--nearest` (anisotropy off then),
  `--address repeat|mirror|clamp|border` (quad UVs 0..2), anisotropy min(16, maxSamplerAnisotropy = 16). samplerAnisotropy
  (and sampleRateShading) are in the spec's "If Vulkan 1.4 is supported, the following features must be supported"
  list (docs.vulkan.org features chapter, checked 2026-10-11) - still enabled explicitly via pEnabledFeatures.
  22_depth: candidates D32_SFLOAT/D24_UNORM_S8_UINT/D16_UNORM all "yes" here, pick D32_SFLOAT (layout
  DEPTH_ATTACHMENT_OPTIMAL; a stencil format would use DEPTH_STENCIL_ATTACHMENT_OPTIMAL + both aspects + pStencilAttachment,
  path run once by forcing the candidate order, clean); spec guarantee quoted: D16_UNORM, one of X8_D24/D32_SFLOAT, one of D24S8/D32S8.
  Depth image 800x600 = 2048000 bytes, type 1, SHARED by both frames in flight: per-frame barrier EARLY|LATE_FRAGMENT_TESTS/
  DEPTH_STENCIL_ATTACHMENT_WRITE -> same stages/READ|WRITE, UNDEFINED -> attachment layout. Negative test (temporary edit,
  reverted): src NONE/NONE gives "WRITE_AFTER_WRITE hazard detected. vkCmdPipelineBarrier2 ... writes to VkImage ...,
  which was previously written by vkCmdEndRendering" (sync validation). Projection printed (fovy 60, aspect 1.3333334,
  near 0.1, far 100; y_clip = -f*y, z 0..1); depth after divide 0.0000/0.9009/0.9724/1.0000 at 0.1/1/3.5/100;
  reversed-Z only mentioned. Front face COUNTER_CLOCKWISE (CCW from outside, the y-flipping projection keeps it CCW on
  screen). 23_mipmaps: 512x512 checker (16-texel cells, FNV-1a 0xef4a0dc5), features + BLIT_SRC/BLIT_DST, mipLevels =
  32 - leading_zeros / std::bit_width = 10, image memory 1399808 bytes (level 0 = 1048576); one command buffer: all
  levels UNDEFINED -> TRANSFER_DST (dst COPY or BLIT), copy level 0, per level: (i-1) TRANSFER_DST -> TRANSFER_SRC,
  vkCmdBlitImage2 LINEAR, (i-1) -> SHADER_READ_ONLY (src BLIT/NONE: WAR), last level from BLIT/TRANSFER_WRITE. Spec quote
  used in comments: blits convert sRGB to linear before filtering ("Image Copies With Scaling"). Floor 40 x 205 units, UVs
  /2 (REPEAT), camera 1 up, pitch 0.2 rad, floor moves by time mod 2. `--no-mips` = maxLod 0, `--anisotropy`.
  24_msaa: framebufferColor/DepthSampleCounts 1, 2, 4, 8 here; `--samples 1|2|4|8` (default 4; 1 = no MSAA, render to
  the swapchain image); attachments TRANSIENT_ATTACHMENT, image-format sampleCounts checked with
  vkGetPhysicalDeviceImageFormatProperties2; no LAZILY_ALLOCATED memory type on this GPU (count printed: 0) -> DEVICE_LOCAL
  type 1 fallback; 4x color = 4x depth = 8192000 bytes. Resolve: color resolveMode AVERAGE, resolveImageView = swapchain
  view, MSAA store DONT_CARE; depth not resolved. Spec (renderpass chapter): resolve runs in COLOR_ATTACHMENT_OUTPUT, reads
  the MSAA image as COLOR_ATTACHMENT_READ, writes the single-sample one as COLOR_ATTACHMENT_WRITE. MSAA color image shared
  by the frames in flight: barrier COLOR_ATTACHMENT_OUTPUT/WRITE -> COLOR_ATTACHMENT_OUTPUT/READ|WRITE; negative test (src
  NONE) -> the same WRITE_AFTER_WRITE message. sampleRateShading mentioned in a comment only. All four: fmt + clippy -D
  warnings, C++ /W4 clean, sync validation clean (only bandicam), stdout identical without `[timing]` for 21 (default,
  --nearest --address clamp, mirror, border), 22 (default, --no-depth), 23 (default, --no-mips, --anisotropy), 24
  (--samples 1/2/4/8). Release, validation + sync on, 3 s, MAILBOX: 1900-2850 fps (not a benchmark). Logs: scratchpad
  m10/logs (session-local).
- Module 10 review fixes (2026-10-11, examples 21-24, both languages, checked against vkspec 1.4.365 + validusage.json):
  21 UNORM comment = double sRGB encode (too bright/washed out) + filtering of non-linear values, not "too dark";
  21/23 `--no-mips`: maxLod 0 kept (stdout unchanged), comments say lambda <= 0 -> always magnified -> magFilter, harmless
  because min == mag (spec Note "Mapping of OpenGL to Vulkan Filter Modes"; GL non-mip min filter = maxLod 0.25);
  24 AVERAGE = the only fixed-function resolve for non-integer color (NONE/CUSTOM_BIT_EXT also allowed,
  VUID-VkRenderingAttachmentInfo-imageView-06129); 24 swapchain barrier dst access now COLOR_ATTACHMENT_READ|WRITE
  ("Access to the single sample attachment can be synchronized with ..._READ_BIT and ..._WRITE_BIT", "Render Pass
  Multisample Resolve Operations"; corrects "writes the single-sample one as WRITE" above); 22 depth is affine in
  1/distance (far/(far-near)*(1-near/d)); 23 UV = (x+20)/2, (5-z)/2; 23 integer formats: no SAMPLED_IMAGE_FILTER_LINEAR
  on the RTX 3080 Ti (vulkaninfo --show-formats; only D24S8/D32S8 depth aspects have it) -> VUID-VkBlitImageInfo2-filter-02001;
  24 "interiors cost no extra shading (still N samples stored)"; C++ 21 now destroys fence/cmd/pool/staging before
  printing "staging buffer destroyed" (= Rust UploadScratch drop). fmt/clippy/W4 clean, sync validation clean (bandicam
  only), stdout identical without [timing] (21 default/--nearest, 22, 23 default/--no-mips, 24 default/--samples 1).
  Examples are not scanned by scripts/vuid-refs.ts (src/ only): no sync:vuids. Logs: scratchpad m10fix/.
- Module 8 review fixes, round 2 (2026-10-11, EN lessons; the RU lessons written afterwards carry the same facts):
  - gpu-allocator experiment (scratch cargo project, deleted after; logs check*.log in the session scratchpad gpualloc/):
    gpu-allocator =0.28.0 (features std+vulkan) + `[patch.crates-io] ash = { git = ash-rs/ash, rev = f4c2ca3 }`.
    The patch WORKS for type unification (Cargo.lock has ONE ash, version `0.38.0+1.4.352` from git, which satisfies
    `ash = "0.38"`), but gpu-allocator then fails to compile: `error[E0599]: no method named push_next found for struct
    MemoryAllocateInfo<'a>` at gpu-allocator-0.28.0/src/vulkan/mod.rs:372 (ash master has only the `push` trait method,
    ash/src/vk.rs:84). Outcome B: allocators.mdx quotes the error, quiz Q3 says "does not compile, even via [patch]".
  - Sync validation re-measured (Rust release 15, `--mib 64` and `--bar --mib 64`, `--seconds 1`, 3 runs each, logs
    m8lesson/sync.*): host writes 7.4-13.4 ms (type 3) and 13.1-16.4 ms (type 5) = same as without it; fps drops from
    2616-2816 to 1530-1991 (proves the layer setting was active). staging-uploads now says so.
  - Step 16 ratios per run (raw / allocator): create+bind 26-53x, destroy+free 98-182x (quiz Q1 of allocators).
  - VUID facts: vkAllocateMemory's "Exceeding this limit will result in undefined behavior..." is DESCRIPTION text, not
    VUID-04101 (whose text is "There must be less than ...maxMemoryAllocationCount..."). The maintenance8 feature rule
    for VK_DEPENDENCY_QUEUE_FAMILY_OWNERSHIP_TRANSFER_USE_ALL_STAGES_BIT_KHR exists for vkCmdPipelineBarrier (10206)
    and vkCmdWaitEvents2 (10205) only, not for vkCmdPipelineBarrier2 (validusage.json, SDK 1.4.341).

## TODO

- [x] Module 7 examples 10-12 (2026-10-11; resumed from WIP 7b36597): fmt + clippy -D warnings, C++ /W4 clean, run on
  the RTX 3080 Ti with VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true: zero messages except the bandicam API-version warning;
  stdout identical Rust vs C++ without `[timing]` lines (10: --frames 1/2/3; 11: both present paths; 12: --frames 1/2/3).
  Details in State "examples 10-12". Logs: scratchpad m7/ (session-local).
- [ ] Screenshots of 10/11/12: NOT taken. On 2026-10-11 ~03:00 every Vulkan window (also step 09, which worked on 10-10)
  showed a white client area in GDI/PrintWindow/MCP captures and in a full-screen capture; retake when the desktop shows
  them again (`examples/screenshots/{10_frames_in_flight,11_resize,12_timeline}.png`, 11 after --demo-resize).
- [x] Module 7 EN lessons (2026-10-11): sync/{fences-and-semaphores (embeds QueueSync), frames-in-flight,
  barriers-and-layouts, timeline-semaphores, swapchain-recreation}; RU pages are placeholders. Regions shown once:
  10 sync-objects/submit (fences), 10 cli/frame-resources/command-buffers/wait-fence/cpu-work/animate/advance-frame/
  cleanup (frames), 12 offscreen-image/barriers/render/blit/timestamps/read-timestamps (barriers), 12 pick-device/timeline-semaphore/
  binary-semaphores/frame-resources/wait-timeline/submit/cleanup (timeline), 11 chain/create-chain/recreate/
  instance-extensions/device-extensions/frame-resources/present/wait-present-fence/destroy-retired/recreate-triggers/
  event-loop/cleanup + 12 swapchain-usage (recreation; Bandicam story). Not shown: 10's 09-identical regions, 11 cli/demo,
  12 cli/event-loop. 13 terms added (execution-dependency, binary-semaphore, frames-in-flight, data-hazard,
  memory-dependency, availability-visibility, stage-mask, access-mask, layout-transition, timeline-semaphore,
  swapchain-recreation, old-swapchain, present-fence); vk-fence/vk-semaphore -> fences-and-semaphores, image-layout/
  pipeline-barrier -> barriers-and-layouts. frame-loop STAYS on first-triangle: <MindMap /> throws for a lesson that
  owns no term. The Rust runs of 10 in m7/ came from the debug build (m7run.ps1), C++ from release.
- [x] Module 7: review of 10-12 + EN lessons (fixed: barrier 5, release medians, fence analogy), RU lessons (2026-10-11).
- [ ] Independent review of the RU lessons of modules 4-7 (language + facts vs EN).
- Step numbering: 10-12 = module 7, so module 8+ examples shift by one vs the course plan (12_vertex_buffer -> 13_...).
  Module 8 = 13-16 (13_memory_types added), so module 9 starts at 17.
- [x] Module 8 examples 13-16 (2026-10-11): fmt + clippy -D warnings, C++ /W4 clean (clean rebuild of all 16), run on
  the RTX 3080 Ti with VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true: only the bandicam warning; stdout identical Rust vs C++
  without `[timing]`/`[lib]` lines (13; 14; 15 default/--transfer-queue/--bar, also with --mib 64; 16 default and
  --count 5000). New shader shaders/mesh.slang (+ spv, manifest; tests/shaders.test.ts passes). Details in State
  "examples 13-16". Logs: scratchpad m8/ (session-local).
- [ ] Screenshots of 14/15: NOT taken - on 2026-10-11 (module 8 session) the capture of the 14 window was white again (same as
  10-12). Retake `examples/screenshots/{14_vertex_buffer,15_staging}.png` when the desktop shows Vulkan windows.
- [x] Module 8 EN lessons (2026-10-11): memory/{memory-types (embeds BufferMemory + RebarCheck, ObjectGraph build with
  device-memory/buffer), vertex-and-index-buffers, staging-uploads, allocators}; RU lessons written 2026-10-11 (same regions,
  outputs, measurements, VUIDs; memory-types embeds BufferMemory, RebarCheck, ObjectGraph). Regions shown once: 13 all; 14 vertex/mesh/vertex-input/create-buffer/vertex-buffer/
  index-buffer/bind-and-draw (+ `<Snippet src="shaders/mesh.slang">`); 15 cli/mesh-bytes/staging-buffer/
  device-local-buffers/record-copy/copy-barrier/finish-upload/transfer-family/release-ownership/acquire-ownership/
  bar-write; 16 allocation-limit/cli/test-buffers/raw/allocator/sub-allocate. Not shown: 14/15 regions identical to
  09/10 (frame loop), 15 find-memory-type/create-buffer/vertex/mesh/vertex-input (= 14), 16 cleanup. 22 terms added
  (memory-heap, memory-type, device-local, host-visible, host-coherent, host-cached, lazily-allocated, bar,
  vk-device-memory, vk-buffer, memory-requirements, memory-type-bits, vertex-buffer, index-buffer,
  vertex-input-binding, vertex-input-attribute, staging-buffer, queue-family-ownership-transfer, sub-allocation,
  linear-allocator, vma, buffer-image-granularity) + RU overlay; objects.json device-memory/buffer got `term`.
  Older EN lessons now link here instead of "module 8" (fences-and-semaphores x2, shaders-and-spirv, barriers).
  Lesson measurements (scratchpad m8lesson/, session-local; release, validation on, sync off, 3 runs per language
  alternating): 15 --mib 64 staging CPU 5.9-9.8 ms, GPU copy 5.1-6.7 ms (graphics 5.3-6.7, transfer 5.1-6.1), BAR
  13.0-16.0 ms (4.2-5.2 GB/s); 176 B: graphics copy 0.83-0.98 ms, transfer 1.19-1.74 ms, BAR 0.017-0.027 ms;
  16: raw 127-150 us create / 316-368 us destroy per buffer, allocator 2.8-4.8 / 1.9-3.8 us. 14 vs 15 fps (Rust,
  2 s x3): 2903-2999 vs 2907-2936 (no measurable difference for 176 bytes). vulkaninfo 2026-10-11: type 2 "usable
  for" no image either; heap budgets quoted in memory-types. Heap 1 = exactly half of the 68630528000 bytes of RAM.
- [x] Module 8 RU lessons (2026-10-11): ru/memory/{memory-types, vertex-and-index-buffers, staging-uploads,
  allocators}, authored from the round-2 EN facts; lint, tests, astro check clean.
- [ ] Module 8: independent review of the EN lessons (facts vs spec/runs, quizzes), then of the RU lessons.
- [x] RebarCheck (EN+RU, 2026-10-11) presents ReBAR ON as a possible misconfiguration too (CSM/MBR boot, NVIDIA
  per-app profiles and measured regressions, uncached BAR heap vs readbacks); sources in REBAR.md.
- [x] Module 9 examples 17-20 (2026-10-11): fmt + clippy -D warnings, C++ /W4 clean (clean rebuild of all 20), run on
  the RTX 3080 Ti with VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true: only the bandicam warning; stdout identical Rust vs C++
  without `[timing]`/`[addr]` lines (17; 18 default and --push-descriptor; 19; 20). New shaders transform/device_address/
  bindless.slang (+ spv, manifest; tests/shaders.test.ts passes). Details in State "examples 17-20".
- [ ] Screenshots of 17-20: NOT taken - the capture of the 17 window was white again on 2026-10-11 (same as 10-16).
  Retake `examples/screenshots/{17_uniforms,18_push_constants,19_buffer_device_address,20_bindless}.png`; the
  column-major matrix upload (17/18) is verified from the SPIR-V only, not visually.
- [ ] Module 9: independent reviews of 17-20 (code vs spec), then the EN lessons descriptors/{uniform-buffers,
  descriptor-sets, push-constants, buffer-device-address, bindless}; `sync:vuids` for the VUIDs named in 17-20 comments.
- [x] Module 9 EN lessons (2026-10-11): module `descriptors` ("Descriptors and shader data") in course.json/course.ru.json;
  RU pages written afterwards (next item). Regions shown once: 17 mesh/uniform-struct/transform/reflect/check-layout/frame-resources/
  uniform-buffers/update-uniforms (uniform-buffers, + Snippet transform.slang), 17 set-layout/pipeline-layout/
  descriptor-pool/allocate-sets/write-sets/bind-and-draw/cleanup (descriptor-sets, ObjectGraph build with pool/set),
  18 cli/limits/push-constant-range/push-constants/frame-resources/device-features/push-descriptor-layout/
  push-descriptor/compare (push-constants), 19 instance/push-data/device-features/create-buffer/instance-buffer/
  get-address/push-and-draw (+ Snippet device_address.slang), 20 sizes/device-features/limits/set-layout/pool-and-set/
  object-buffers/write-descriptors/initial-writes/stream-in/bind-and-draw (+ Snippet bindless.slang). Not shown: 18
  cleanup, 19 pipeline-layout/cleanup, 20 cleanup. 27 terms added (scratchpad m9lesson/addterms9.py) + RU overlay;
  vk-descriptor-set-layout now owned by descriptor-sets; objects.json descriptor-pool/descriptor-set got `term`.
  Outputs quoted from Rust release runs 2026-10-11 (scratchpad m9lesson/, identical to C++ without [timing]/[addr]).
  Measured for the lesson (slangc 2026.1, spirv-dis): one probe struct as std140 / std430 / -fvk-use-scalar-layout,
  table in uniform-buffers (float3 after float3: 32/32/28; float[2] stride 16/4/4; float2x2 MatrixStride 16/8/8).
  Spec facts (vkspec 1.4.365): maxPushConstantsSize min 128 core, **256 for Vulkan 1.4**; maxUniformBufferRange 16384
  core / 65536 1.4; maxBoundDescriptorSets 4 / 7 (1.4); 1.4 requires pushDescriptor + scalarBlockLayout support, 1.3
  bufferDeviceAddress, 1.2 uniformBufferStandardLayout; descriptor indexing is NOT required by core 1.4 (Roadmap 2022 only).
- [x] Module 9 RU lessons (2026-10-11): ru/descriptors/* authored (same regions, VUIDs, outputs, quiz facts as EN; RU term
  labels buffer-device-address/subgroup/update-after-bind/partially-bound/non-uniform-indexing now the English terms).
  [ ] Independent review of them.
- [x] Module 10 examples 21-24 (2026-10-11): see State "examples 21-24"; CMake STEPS/WINDOW_STEPS and READMEs updated.
- [ ] Screenshots of 21-24: NOT taken - the 21 window was white again in window and full-screen captures on 2026-10-11
  (same as 10-20). Retake `examples/screenshots/{21_texture,22_depth,23_mipmaps,24_msaa}.png` (23: one with and one with
  `--no-mips`; 24: `--samples 1` vs 4, zoomed edge). Nothing visual was verified: winding/culling, texture orientation,
  mip quality and the resolve are checked only by validation and code review.
- [ ] Module 10: independent reviews of 21-24 (code vs spec), then the EN lessons images/{textures-and-samplers, depth,
  mipmaps, msaa}; `sync:vuids` for the VUIDs named in 21-24 comments (VkSamplerCreateInfo-anisotropyEnable-01071,
  VkImageMemoryBarrier2-image-03320).
- [x] 22/24 D24_UNORM_S8_UINT branch (DEPTH_STENCIL_ATTACHMENT_OPTIMAL, both aspects, pStencilAttachment): tested
  2026-10-11 by temporarily putting it first in the candidate list (both languages, reverted): runs, sync validation clean.
- [ ] Proposal: a step 16b using gpu-allocator through `[patch.crates-io]` once a gpu-allocator release (or its main)
  builds against ash master (`push_next` -> `push`); 0.28.0 does not. Needs a Cargo.toml change: coordinate with the
  owner of examples/rust/Cargo.toml. Also update the Cargo.toml comment (lines 18-21) to mention the [patch] result.
- [ ] examples/cpp/src/15_staging.cpp ~346 still says "only 256 MiB of address space"; mirror the Rust doc comment
  ("a 256 MiB BAR, of which the driver exposes 214 MiB as heap 2").

- [x] Stage 1: course-kit engine; [x] site on the kit; [x] examples 01-05 both languages; [x] lessons 1-10
- [x] ObjectGraph map + build modes; [x] two reviewers, findings fixed; [x] CI (tests, clippy/fmt, Pages)
- [x] i18n (2026-10-10, deployed): EN lessons upgraded ("Where this fits", teaching quizzes), RU authored (terms.ru.json, index, 10 lessons),
  reviewed by independent readers, built, screenshots in both languages, deployed
- [x] 12 labs (2026-10-10); consolidation: tools/vk-probe, VUID single source, shared find-memory-type rule, astro check clean
- [ ] Release course-kit v0.4.1 (declarations), bump the tag here, then add `bun run check` to the CI test job
- [x] Embed queue-sync and buffer-memory in their lessons (EN: fences-and-semaphores, memory-types; RU pending for buffer-memory)
- [x] RU lessons of modules 4-6 incl. GraphicsPipeline in ru/pipeline/graphics-pipeline
- [ ] labs/graphics-pipeline.mdx (EN+RU) still says it previews the pipeline module: reword now that the lesson exists
- [ ] C++ examples in CI (needs Vulkan SDK headers 1.4.341 with vulkan_raii.hpp on the runner)
- [ ] Decide with the user: term maps put foundations at the bottom, ObjectGraph puts the root on top
- [ ] After enabling ReBAR on the dev machine (steps in REBAR.md): capture a second snapshot (`sync:gpu` ->
  `gpu-rebar.json` or a profile field) and let the memory labs switch between OFF (measured) and ON (measured).
  `src/data/gpu.json` stays the ReBAR-off snapshot; until then the ON state is described in words only
  (`src/components/RebarCheck.astro`, lab notes).
- [ ] User review, then modules 4+ (presentation, pipeline, commands, sync, memory, descriptors, images, compute)
