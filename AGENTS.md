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
| buffer-memory | BufferMemory | none yet (module 8) |

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
  ~0.02 ms. With VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true the 64 MiB host writes take ~70 ms (sync validation cost,
  not the hardware). 16 (3000 buffers): raw ~130-170 us per create+allocate+bind, ~330 us per destroy+free; allocator
  ~3 us and ~2-3 us. Validation does NOT catch a missing post-copy barrier (presumably because a host fence wait sits between upload
  and draws, which sync validation treats as synchronizing; not confirmed) nor a missing acquire barrier: tried 2026-10-11, both silent - the
  barriers follow the spec and the Khronos synchronization examples, not a validation message.
- Module 8 review fixes (2026-10-11): 13 no longer evaluates LAZILY_ALLOCATED against buffer memoryTypeBits (spec: never
  for buffers; one stdout line points to images with TRANSIENT_ATTACHMENT, module 10); "uncached (typically
  write-combined)"; HOST_COHERENT writes = available in the host domain, the submit makes them VISIBLE (14/15
  write_mapped). 15: comments on fence = available not visible (the barrier/acquire makes the copy visible; validation
  is silent because sync validation is off by default and a host fence wait closes the batch's hazards), on the
  ownership transfer being required even though the semaphore wait already gives visibility (spec quote, "Resource
  Sharing Mode"); `--bar` checks vertex+index together via vkGetDeviceBufferMemoryRequirements before allocating
  (`--bar --mib 214` = heap size 224395264 exactly now fails with "vertex + index buffers need 224395300 bytes ...";
  heap size is an upper bound, not the budget - VK_EXT_memory_budget). Rust 15 upload_staged: `UploadScratch` guard
  (device_wait_idle + destroy on drop = C++ IdleGuard + RAII). 16: `--count` checked against the heap before
  allocating (`--count 1000000` needs 23295976960 bytes > heap 0 12673089536 -> clean error); Rust raw and allocator
  paths destroy what was created on an error midway (C++: RAII / release()). Rust `Vertex` has a size assert (20).

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
- [ ] Module 8: independent reviews of 13-16 (code vs spec), then the EN lessons memory/{memory-types,
  vertex-and-index-buffers, staging-uploads, allocators}; embed the buffer-memory lab in memory-types.

- [x] Stage 1: course-kit engine; [x] site on the kit; [x] examples 01-05 both languages; [x] lessons 1-10
- [x] ObjectGraph map + build modes; [x] two reviewers, findings fixed; [x] CI (tests, clippy/fmt, Pages)
- [x] i18n (2026-10-10, deployed): EN lessons upgraded ("Where this fits", teaching quizzes), RU authored (terms.ru.json, index, 10 lessons),
  reviewed by independent readers, built, screenshots in both languages, deployed
- [x] 12 labs (2026-10-10); consolidation: tools/vk-probe, VUID single source, shared find-memory-type rule, astro check clean
- [ ] Release course-kit v0.4.1 (declarations), bump the tag here, then add `bun run check` to the CI test job
- [ ] Embed queue-sync and buffer-memory in their lessons once modules 7/8 exist ([x] graphics-pipeline, EN)
- [x] RU lessons of modules 4-6 incl. GraphicsPipeline in ru/pipeline/graphics-pipeline
- [ ] labs/graphics-pipeline.mdx (EN+RU) still says it previews the pipeline module: reword now that the lesson exists
- [ ] C++ examples in CI (needs Vulkan SDK headers 1.4.341 with vulkan_raii.hpp on the runner)
- [ ] Decide with the user: term maps put foundations at the bottom, ObjectGraph puts the root on top
- [ ] After enabling ReBAR on the dev machine (steps in REBAR.md): capture a second snapshot (`sync:gpu` ->
  `gpu-rebar.json` or a profile field) and let the memory labs switch between OFF (measured) and ON (measured).
  `src/data/gpu.json` stays the ReBAR-off snapshot; until then the ON state is described in words only
  (`src/components/RebarCheck.astro`, lab notes).
- [ ] User review, then modules 4+ (presentation, pipeline, commands, sync, memory, descriptors, images, compute)
