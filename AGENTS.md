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
12. Advanced (plan: `notes/module-12-plan.md`, decisions and verified device facts): debugging-and-profiling
    (28_debug_labels, 29_gpu_queries), pipeline-cache (30), multithreaded-recording (31), mesh-shaders (32), ray-query
    (33), ray-tracing-pipeline (34), capstone-renderer (35, glTF + PBR, multi-file). Not planned yet: shader objects,
    tile-based mobile GPUs.

Status (2026-10-11): modules 1-11 published EN + RU (examples 01-27, 12 labs, screenshots 09-27 pixel-verified).
Module 12 in progress.

## Visualizations (planned, Vulkan-specific, live in this repo)

- ObjectGraph: build the Vulkan object tree by "creating" objects in order; wrong order → validation message.
- FrameTimeline: CPU/GPU lanes, frames in flight, fences/semaphores, scrubbable.
- BarrierLab: pipeline stages with src/dst stage+access masks; shows what a barrier blocks.
- LayoutMachine: image layout state machine with transitions.
- MemoryHeaps: heaps/types of real GPUs, allocation, staging, ReBAR.
- DescriptorWiring: shader set/binding ↔ resources.

## Trainers (engine v0.7.0): how to add one

"Build it yourself" exercises (user, 2026-10-11: "это же как тренажёр, ровно то как надо всё делать") are an ENGINE
feature: course-kit `lib/trainer.ts` (pure core) + `components/trainer/Trainer.tsx` (UI on TreeView, all four views,
undo, start over, goals, console, run stored per page). Read course-kit AGENTS.md "Trainers" first. A new trainer here
(frame assembly with fences/semaphores, barriers, image layouts, descriptor wiring, pipeline creation) is a scenario,
never a new widget. The ObjectGraph build mode is the reference:
1. Model (pure, Vulkan rules): `src/lib/<name>.ts` with its state type and functions that return calls/refusals as
   structured lines (`src/lib/objects.ts`: `blockers`, `destroyBlockers`, `create`, `destroy`). Facts only from the spec
   registry or real runs. VUID texts ONLY through `vuid(id)` (`src/lib/vuids.ts`); write the id, run `bun run sync:vuids`.
2. Scenario (pure, deterministic): `src/lib/<name>-trainer.ts` exporting `(ids, lang, version) => Scenario<State>`
   (`src/lib/object-trainer.ts`): `version` = `ruleVersion([raw sources])` computed in the .astro wrapper from `?raw`
   imports of the data and rule code (a change makes stored runs start fresh); verbs and options with STABLE ids (stored
   in readers' browsers), options of kind flag/choice/number/item, global steps as `actions`, instances that come and
   go as `items: (state) => ids`, a state word on the card as `badge` (e.g. the image layout), `check` = model refusals mapped to entries (`entryOf`: validation
   reports verbatim English with `ref` = VUID, simulator remarks worded with `tfmt` from `src/i18n.ts`), `apply` = model
   transition, `status` -> locked/ready/done/error, `goals` (`allDone(text)` or custom predicates).
3. Texts: course strings in `src/i18n.ts` (EN + RU, `tests/i18n.test.ts` parity); generic buttons/labels are the engine's.
4. View: a React island (`client:only="react"`) that builds `TreeItem`s and `TreeGroup`s (colors `groupColor(i)`, never
   literal colors: site themes recolor through `--ck-*` tokens) and renders `<Trainer ... />` with `body` (details),
   `statusText`/`statusTag`/`progressText`/`startHint` if the course words them; an .astro wrapper validates data at build time and
   gives the text fallback (`src/components/ObjectGraph.astro` + `ObjectGraphView.tsx`).
5. Tests: model facts (tests/objects.test.ts style) + the scenario through the engine core (`act`, `replay`, `undo`,
   `why`, `complete`, `consistency` over reached states, `encode`/`decode` with a stale version, replay stopping at a
   `StaleAct` after a rule change; tests/object-trainer.test.ts style), then `bun run lint`, `bun run check`,
   `bun run build` and a browser pass (refusal shows the VUID, undo, reload restores the run).

### Frame assembly trainer (2026-10-11)

Step 10's frame loop (F=2 slots, 3 images, minImageCount 2, one queue) as a trainer: model `src/lib/frames.ts` (rules +
sources in its header), scenario `src/lib/frame-trainer.ts`, texts `src/components/labs/frame-assembly.text.ts`, view
`FrameAssembly{.astro,View.tsx}`, tests `tests/labs/frame-assembly.test.ts` (every rule accepted/refused, full run,
random walks from prefixes of the correct run with `consistency`, stored runs/versions). Version = `ruleVersion` of
frames.ts + frame-trainer.ts (texts excluded). Shared helper `vuReport(id, call)` in `src/lib/vuids.ts` (objects.ts
uses it too).
- Real validation evidence (2026-10-11, temporary probe bin = copy of step 10 with env-selected faults, built under
  the lock, its source then moved out of examples/; logs scratchpad frame/probe*.log, session-local), with
  VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true: render_finished indexed by slot -> VUID-vkQueueSubmit2-semaphore-03868
  "... may still be in use by VkSwapchainKHR ... Swapchain image 0 was presented but was not re-acquired ..."; wait
  stage BOTTOM_OF_PIPE -> "WRITE_AFTER_READ hazard ... previously read by vkAcquireNextImageKHR" (NOT
  WRITE_AFTER_PRESENT); TOP_OF_PIPE -> clean; present without wait semaphore -> "PRESENT_AFTER_WRITE hazard" and then
  03868 "previously signaled ... and has not since been waited on"; no fence wait -> 01779, 01123, 00045, 00049,
  03875, 01430. Not reproduced live (registry + spec only): 04894, 04895, 03873, 01286, 07783, 03268, 03874.
- Model simplifications (text `simplified`, the figcaption of FrameAssembly.astro, so EVERY embedding states them; the
  lab page no longer repeats them): batches complete in submission order; FIFO presentation engine returning images in
  present order; the host learns of completions only through vkWaitForFences (timeout 0 = vkGetFenceStatus); no
  OUT_OF_DATE. A host wait (fence, acquire of a not-yet-shown image) runs GPU + presentation engine until it returns; a
  wait that never could is refused as a labeled deadlock.
- Review fixes (2026-10-11, tests "review fixes" in tests/labs/frame-assembly.test.ts):
  - GPU truth vs host knowledge: `Batch.done`/`Sig.fired`/`Sig.used`/`Fence.signaled` = what executed; `Batch.seen` =
    a fence wait returned VK_SUCCESS for it or a later batch (`see()`). Host-facing rules read the host view: cb pending
    (00045/00049/03875) and fence in use (01123/04895) until seen; a semaphore keeps a batch's wait until seen (01779).
    The "GPU" button alone never frees anything for the host.
  - Semaphores are lists of signal operations (`Sem.sigs`, each with its claim): a batch may signal a semaphore whose
    current signal an earlier pending batch waits on (03868 holds: that wait executes first, submission order); a
    present's claim still refuses (released only by re-acquiring the image).
  - Acquire hazard: `Img.acqWaited` only from a chained batch, `Img.broken` ORs every batch's hazard since the acquire
    (a later batch cannot hide it). Present hazard = broken, or the waited signal comes from a batch BEFORE the image's
    last rendering batch (a later batch's signal covers it by submission order).
  - Recording for any image is valid (recording does not touch it); `notHeld` refuses the submit. Item options left ''
    (stored runs) are refused with `noPick`, never thrown. Present's default wait = the rendering batch's signal.
- [x] Engine gap (course-kit): the console stayed at the top; fixed in course-kit v0.7.1 (follows the newest line,
  stays put while the reader scrolled up), `lines={8}` removed. Verified headless 2026-10-11.

### Barriers trainer (2026-10-11)

Seven puzzles (`PUZZLES` in `src/lib/barriers.ts`: upload RAW (step 15), particles WAR (26), copy WAW, compute RAW (26),
render-then-sample RAW + layout, texture upload with 2 barriers (21), depth across frames (22)); each a command stream
on one queue with barrier slots. Model `src/lib/barriers.ts` (rules + spec quotes in its header), scenario
`src/lib/barrier-trainer.ts`, texts `src/components/labs/barriers.text.ts`, view `Barriers{.astro,View.tsx}`, tests
`tests/labs/barriers.test.ts`, lab `labs/barriers` (EN+RU), embedded in sync/barriers-and-layouts (EN+RU, section
"Write the barriers yourself" / «Напишите барьеры сами»). Version = `ruleVersion` of barriers.ts + barrier-trainer.ts.
- Editor = barrier card: one verb per mask (`srcStage`, `srcAccess`, `dstStage`, `dstAccess`: flag options), `layout`
  (choices), `submit` (also the puzzle card's primary verb). Engine has no multi-select / option groups, so one fieldset
  per mask; "Set" stores the struct, the rules run only at the submit (editing a struct is not a Vulkan call). Global
  action `all` switches the editors to every stage/access; flags set but not shown are kept.
- Rules: execution scopes with logical order (graphics/compute lists; transfer stages unordered), TOP/BOTTOM per side,
  access scopes only for EXPLICITLY named stages, WAR = execution only, RAW/WAW = memory dependency, a layout transition
  = a write between the scopes, equal layouts ignored (sync2), UNDEFINED where contents are needed = valid but open
  (`discard`). Render passes face outside work only through load/store ops (spec Note), so step 22's barrier
  (EARLY|LATE, READ|WRITE) grades as correct but wider than needed: tight = src LATE / DS_WRITE -> dst EARLY / DS_WRITE.
- VUIDs: access-mask ids are built from `SUPPORT` numbers and must be listed literally in `ACCESS_VUIDS` (vuid-refs.ts
  finds ids by text); layout ids 01197/01198/01208/01211/01212/01213/04066; later-command layout 09600/00180/09588.
- Grading (`extras`): drop each flag (broadest first); split meta flags (ALL_COMMANDS, ALL_GRAPHICS, ALL_TRANSFER, VERTEX_INPUT,
  MEMORY_*, SHADER_READ); try a stage with a strictly smaller execution scope (`narrow`); TOP in dst / BOTTOM in src
  always reported. Tight (★) = none.
- Hazard lines are the simulator's own wording with the syncval hazard id as `ref`; real layer texts only in
  `Puzzle.evidence`, matched by (slot, hazard, before, after, optional miss). Probe runs 2026-10-11 (copies of steps 21/22/26 built
  OUTSIDE the repo under the lock, env PROBE faults, VALIDATE_SYNC + SHADER_ACCESSES_HEURISTIC; logs scratchpad
  barriers/runs, session-local): 22 `min` (src LATE/DS_WRITE -> dst EARLY/DS_WRITE) CLEAN = the depth tight answer is
  real; 22 src EARLY only or src access NONE -> WAW "vkCmdPipelineBarrier2 writes ... previously written by
  vkCmdEndRendering ... prior write (DEPTH_STENCIL_ATTACHMENT_WRITE) at LATE_FRAGMENT_TESTS"; 22 dst access NONE ->
  WAW "vkCmdBeginRendering clears pDepthAttachment ... written during an image layout transition"; 21 barrier 1
  dstAccess NONE -> WAW copy after transition; 21 barrier 2 srcAccess NONE -> WAW transition after copy; 26 WAR barrier
  removed -> WRITE_AFTER_READ ("an execution dependency is sufficient"); 26 compute->compute removed -> silent (as
  compute/particles.mdx says); 26 compute->vertex removed -> READ_AFTER_WRITE (not a puzzle). `discard` is a `warn`.
- Review fixes (2026-10-11, independent reviewer): MEMORY_* split only into the parts valid on the mask's stages (spec:
  "all READ access flags that are valid where it is used"), greedy `drop` (one of two mutually covering flags),
  narrowest `narrow` candidate, "Set" without a change keeps the verdict, depth texts quote the normative
  "happens-after/-before any recorded command which accesses the sample" (the "only needs to synchronize" lines are Notes).
- Review fixes, round 2 (2026-10-11, tests in tests/labs/barriers.test.ts; reviewer's brute-force scripts rerun: 0
  disagreements): `extras` tries each mask's flags broadest first (stages by execScope size, accesses by expansion) and
  drops TOP(dst)/BOTTOM(src) from the working mask when the rest suffices, so COPY|ALL_COMMANDS names ALL_COMMANDS,
  never the needed flag. `Evidence.fault` = what the probe changed ('removed' or `<field> = <value>`, worded by
  `fault_removed`/`fault_mask`); `Evidence.miss` limits it to the misses the run really had: step 26 without its
  compute->compute barrier still chained the dispatches through the draw's barriers (before_draw + next frame's WAR) and
  made the writes available, so only visibility (acc2) was missing; the silent remark shows only for acc2 and no longer
  guesses a cause (also fixed in compute/particles.mdx EN+RU, text + quiz whys). Caption states the adjacent-commands-only
  and stage/access-subset simplifications.
- Finding (not changed, examples are another owner's): step 22's depth barrier and its comment use EARLY|LATE on both
  sides and DS_READ|WRITE as destination; the probe proves LATE/DS_WRITE -> EARLY/DS_WRITE is enough (the trainer grades
  step 22's barrier as correct but wider). No evidence (no real run) for puzzles upload, waw, sample.
- [ ] Engine gap (course-kit): option groups or a multi-select option kind with a visible legend would let one verb
  edit a whole side (stages + accesses) without four "Set" buttons.

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
| object-builder | ObjectGraph (`src/components/ObjectGraph*`, model `src/lib/objects.ts`, scenario `src/lib/object-trainer.ts`) | devices/logical-device |
| queue-sync | QueueSync | sync/fences-and-semaphores (EN + RU) |
| graphics-pipeline | GraphicsPipeline | pipeline/graphics-pipeline (EN + RU) |
| buffer-memory | BufferMemory | memory/memory-types (EN + RU) |
| frame-assembly | FrameAssembly (engine Trainer; model `src/lib/frames.ts`, scenario `src/lib/frame-trainer.ts`) | sync/frames-in-flight (EN + RU) |
| barriers | Barriers (engine Trainer; model `src/lib/barriers.ts`, scenario `src/lib/barrier-trainer.ts`) | sync/barriers-and-layouts (EN + RU) |

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
| `src/data/pipeline-stats.json` | VK_KHR_pipeline_executable_properties of saxpy/reduce/particles/cube + register-pressure series, with timings | `bun run sync:pipeline-stats` (compiles tools/vk-probe/shaders/pressure.slang, runs vk-probe) |

- VUIDs: write the id in a model, data file or lesson, run `bun run sync:vuids`; never paste a VUID text.
  `tests/vuids.test.ts` fails for an id without its synced text.
- `tools/vk-probe` (cargo, ash at the examples' pin): binaries `buffer-requirements`, `validation-log` and `pipeline-stats`, see its
  README. Its output is machine-specific by design; handle values in messages change every run (the lesson quotes
  its own run). Every capture's `date` is the UTC calendar date (vk-probe and scripts/sync-gpu.ts alike; the
  current gpu.json predates the rule and carries the local date of its run). CI lints it (clippy/fmt) next to examples/rust; running needs a GPU.
- `REBAR.md`: Resizable BAR on the dev machine (off now: the DEVICE_LOCAL|HOST_VISIBLE heap is 214 MiB). If it is
  switched on, rerun sync:gpu and sync:buffers, then fix the facts the memory labs, lessons and tests state.

## State (2026-10-10)

- Engine: `github:ssoj13/course-kit#v0.7.1` (v0.7.1 2026-10-11: trainer console follows its newest line; v0.7.0
  released 2026-10-11: trainer API v2 = StaleAct, scenario version,
  option kinds, global actions, card status words; one-`:root` theme contract, `--ck-on-fill`, `--ck-mono`. v0.6.0 color
  tokens `--ck-*` + trainers; v0.5.0 concept
  trees, v0.4.1 declarations and v0.4.2 labs catalog are in it). Bump the tag in package.json for engine changes, then `bun install`. node_modules/course-kit files are
  hard links into bun's cache: for local co-development delete a file before copying a new one over it.
- Concept maps (engine v0.5.0, user-approved "D+E" + roadmap.sh design): MindMap and ObjectGraph are trees on the
  engine's TreeView with four views chosen by the reader and remembered site-wide (Spine = default, Tree + collapse,
  Tree, Outline); details open in a floating draggable window (not inside the card: the user did not want the map to
  move). The react-flow look is gone from both (the course roadmap page keeps react-flow).
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
  simulator `src/lib/objects.ts` (tests in tests/), view `src/components/ObjectGraph{.astro,View.tsx}` + `objects.css`.
  Tree: each object hangs off its FIRST parent in objects.json; the others (descriptor set: + layout; pipeline: +
  layout, + shader module (only while creating); swapchain: + surface; image view: + image) are dashed wires or "+ from"
  notes (`scopedParents`: isA substitution for parents outside the scope). Groups/colors = stage in OBJ_GROUPS order
  (instance, devices, presentation, memory, images, pipeline, descriptors, commands, sync), colors = engine palette
  entries (`STAGE_COLOR` in ObjectGraph.astro -> `groupColor(i)` = `--ck-group-N`; no course color variables any more);
  names = `stage*` texts in src/i18n.ts. Build mode (engine v0.7.0) = the engine Trainer on `src/lib/object-trainer.ts`
  (`objectScenario(ids, lang, version)`, version = `ruleVersion` of objects.json + objects.ts + object-trainer.ts, from
  `?raw` imports in ObjectGraph.astro), card words `tag*` texts: statuses locked/ready/done/error (= old ghost/ready/alive/dangling) in every view; the
  window holds note, call, parents, glossary link (`body`), state (`state*` texts), the blocking VUID messages, the
  primary verb (Create/Allocate/Enumerate/Get, + the VK_EXT_debug_utils checkbox on the instance) and Destroy, the last
  result; under the map: progress (`progress` text), goal (`goalAll`), Undo, Start over, console. The reader's run is
  stored per page and scope (localStorage) and replayed on load. Map mode opens the device's children.
- Lab colors (2026-10-11): no literal colors in lab CSS; series = engine palette (`--lab-series-*` alias `--ck-group-*`),
  queue-sync stage/link colors = `--ck-group-N`, text or marks on a solid fill = `--ck-on-fill`, code font `--ck-mono`.
  graphics-pipeline vertex colors stay literal (they are data). `@dagrejs/dagre` and `@xyflow/react` were removed from
  package.json (unused here; the engine depends on @xyflow/react itself).
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
- examples 25-27 (module 11 "compute", 2026-10-11, both languages; stdout identical without `[timing]`). New shaders
  `saxpy.slang` (result = a*x + y; StructuredBuffer x/y at bindings 0/1, RWStructuredBuffer result at 2, push {float a;
  uint count} 8 bytes), `reduce.slang` (groupshared uint[256] tree sum, GroupMemoryBarrierWithGroupSync = OpControlBarrier
  Workgroup/Workgroup/AcquireRelease|WorkgroupMemory (264); out-of-range invocations load 0 instead of returning, so all
  reach every barrier), `particles.slang` (init/simulate/advance compute + vertexMain/fragmentMain; Particle {float2 pos,
  vel} std430 stride 16; push {dt, time, count, aspect} 16 bytes; writes SV_PointSize 1.0 =
  VUID-VkGraphicsPipelineCreateInfo-topology-08773 without maintenance5). All three declare only `Shader` (pinned).
  Slang emits `OpExecutionMode LocalSize 256 1 1`; the programs read it (OpEntryPoint name -> id -> LocalSize).
  25_compute (no window): family 0 (first COMPUTE), limits printed (RTX 3080 Ti: invocations 1024, size 1024x1024x64,
  count 2147483647x65535x65535, shared 49152, subgroupSize 32; spec: Vulkan 1.4 min invocations 256 and size
  (256,256,64), core 128/(128,128,64), shared min 16384). Default N = 1000000 -> 3907 groups (last has 64), `--count`
  1..2^23 (2N-1 < 2^24 keeps floats exact; OpFAdd/OpFMul "Correctly rounded" table), `--uncached`. Inputs type 3
  (HOST_VISIBLE|COHERENT), outputs choose_type(HOST_VISIBLE, preferred HOST_CACHED) = type 4 (coherent: no invalidate;
  code invalidates if not). Barrier COMPUTE_SHADER/SHADER_STORAGE_WRITE -> HOST/HOST_READ, quoting the fence Note
  ("Signaling a fence and waiting on the host does not guarantee ... visible to the host"). Measured (release, sync
  validation on): saxpy 530-720 us = 17-22 GB/s (inputs/outputs in system RAM, PCIe), saxpy+reduce 750-940 us
  (8M elements: 5.6 ms / 7.6-8.8 ms); CPU copy of 4 MB results: HOST_CACHED type 4 0.8-1.2 ms (3.4-5.0 GB/s) vs
  `--uncached` type 3 23.5-25.2 ms (0.16-0.17 GB/s), ~25x. Verified exactly vs CPU (all results, all group sums,
  total 499500000).
  26_particles: 2^20 particles (`--count`, max 2^24) in ONE DEVICE_LOCAL buffer (type 1) STORAGE_BUFFER|VERTEX_BUFFER;
  init by a compute dispatch (fence-waited startup submit); per frame in one command buffer on family 0 (checked for
  GRAPHICS|COMPUTE): barrier [VERTEX_ATTRIBUTE_INPUT/NONE -> COMPUTE_SHADER/NONE (WAR vs frame n-1's draw) +
  COMPUTE_SHADER/STORAGE_WRITE -> COMPUTE_SHADER/STORAGE_READ|WRITE (RAW)], dispatch simulate (in place), barrier
  COMPUTE_SHADER/STORAGE_WRITE -> VERTEX_ATTRIBUTE_INPUT/VERTEX_ATTRIBUTE_READ, draw POINT_LIST. Single buffer chosen:
  GPU-only data needs barriers, not per-frame copies (copies are for CPU-written data, step 17). dt clamped to 1/60.
  27_async_compute: compute on family 2 (COMPUTE without GRAPHICS), one queue each; ping-pong P[0]/P[1] (step n writes
  P[n%2] from P[(n-1)%2]; frame n draws P[n%2]); sharing CONCURRENT [0, 2] (EXCLUSIVE would need release/acquire per
  switch and serialize the overlap; spec: no transfer -> contents undefined for the second family's reads; CONCURRENT
  "may result in lower performance access"); `--same-queue` = one queue of family 0, EXCLUSIVE (CONCURRENT needs >= 2
  families). Timelines `computed` (init state = 0, step n signals n) and `rendered` (frame n signals n); graphics n
  waits computed >= n, compute n waits rendered >= n-2 (WAR), host waits rendered >= n-F / computed >= n-F for slot
  reuse; semaphore waits/signals at ALL_COMMANDS so the TOP_OF_PIPE start timestamp lies inside the wait's scope
  (interpretation: ALL_COMMANDS covers every operation of the batch). Compute step n+1 submitted right after frame n's
  present. Timestamps compared across queues: all device timestamps are in VK_TIME_DOMAIN_DEVICE_KHR (spec,
  vkCmdWriteTimestamp2). MEASURED (release, validation on, sync off, 3 alternating runs x 2 languages, 4 s, MAILBOX,
  2^20 particles): per frame graphics 272-282 us, compute 45 us; overlap async 0.8-3.0 us (2-7% of the compute time),
  same-queue 0.0 us; fps async 648-1055 vs same-queue 824-1135 (noise, CPU/present bound: ~1 ms per frame vs ~0.32 ms
  GPU). Async compute gives NO measurable gain on the RTX 3080 Ti here; the program prints the numbers as they are.
  Screenshots taken 2026-10-11 (window capture works again): examples/screenshots/{26_particles,27_async_compute}.png.
  SYNC VALIDATION TRAP (SDK 1.4.341): VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true alone does NOT track shader
  storage-buffer accesses (setting syncval_shader_accesses_heuristic, default false); with
  VK_KHRONOS_VALIDATION_SYNCVAL_SHADER_ACCESSES_HEURISTIC=true: removing 26's compute->vertex barrier -> "vkCmdDraw():
  READ_AFTER_WRITE hazard detected. vkCmdDraw reads vertex VkBuffer ..., which was previously written by vkCmdDispatch";
  removing 26's WAR barrier -> "vkQueueSubmit2(): WRITE_AFTER_READ hazard detected. vkCmdDispatch ... writes to
  VkBuffer ..., which was previously read by vkCmdDraw ... Vulkan insight: an execution dependency is sufficient";
  removing 27's computed wait in the graphics submit -> "vkQueueSubmit2(): READ_RACING_WRITE hazard detected. vkCmdDraw
  ... reads VkBuffer ..., which was previously written by vkCmdDispatch (... submitted on VkQueue ...)". NOT detected
  even with the heuristic: 26 without the compute->compute RAW barrier, 25 with saxpy dispatched twice (WAW), 27
  without the rendered >= n-2 WAR wait, 25 without the COMPUTE->HOST barrier (host reads are not tracked; results still
  correct on this machine). Earlier "sync validation clean" claims for storage buffers (modules 9-10) were made without
  the heuristic. All clean runs of 25-27 used both env vars: only the bandicam warning. Logs: scratchpad m11/.
- Module 11 review fixes (2026-10-11, both languages, checked in vkspec 1.4.365 + validusage.json SDK 1.4.341):
  26/27 refuse a particle buffer larger than maxStorageBufferRange (whole-buffer STORAGE_BUFFER descriptor =
  VUID-VkWriteDescriptorSet-descriptorType-00333; spec minimum 2^27, no 1.4 bump; 2^24 x 16 B = 2^28 > 2^27). Not
  reachable on the RTX 3080 Ti (limit 4294967295); on a minimum-limit device `--count` > 2^23 hits it. No stdout change.
  25 `--uncached` rules HOST_CACHED types out (`types_with` mask on memoryTypeBits; still type 3 here); spec:
  "uncached memory is always host coherent". 25 result/groupSums share one memory type by the spec ("memoryTypeBits is
  identical for all VkBuffer objects created with the same value for the flags and usage"); the program checks it.
  25 comment: 06429/06432 are shader (pipeline) rules, only groupCountX-00386 is a vkCmdDispatch rule; the 1-D check
  now precedes the printed subgroup count (no x*y*z product). 26/27 Rust init: device_wait_idle before destroying the
  init fence if the wait failed (VUID-vkDestroyFence-fence-01120); C++ 26/27: init fence hoisted and the IdleGuard
  declared right after it, before init (it used to start only at the frame loop). 27 graphics submit waits
  image_available at ALL_COMMANDS (was COLOR_ATTACHMENT_OUTPUT), so the graphics interval excludes the acquire wait
  (cost: vertex work can no longer start before the image is acquired); the to_attachment barrier still chains (src
  COLOR_ATTACHMENT_OUTPUT is inside ALL_COMMANDS). 27 GpuTimes: all four ticks as signed offsets from g0 modulo
  2^validBits (shift-left/arith-shift-right), so a counter wrap below 64 bits cannot break min/max (both families have
  64 bits here). Runs (release, VALIDATE_SYNC + shader heuristic): 25 default/--uncached/--count 8388608, 26, 27,
  27 --same-queue, both languages: exit 0, only the bandicam warning, stdout identical without [timing].
  RE-MEASURED 27 (release, validation on, sync off, 3 alternating runs x 2 languages, 4 s, MAILBOX, 2^20 particles):
  graphics 269-277 us, compute 44.8-46.5 us in both modes (graphics time unchanged by the ALL_COMMANDS wait: the
  acquire wait was negligible in MAILBOX); overlap async 2.1-5.8 us (5-12% of the compute time), same-queue 0.0 us;
  fps async 895-1328 vs same-queue 1113-1400 (noise, CPU/present bound). Still no measurable gain. Logs: scratchpad m11fix/.
- Sync validation re-check 09-24 with shader-access heuristic (2026-10-11): the setting's own description
  (SDK Bin/VkLayer_khronos_validation.json, key `syncval_shader_accesses_heuristic`, default false, needs validate_sync):
  "Take into account memory accesses performed by the shader based on SPIR-V static analysis. Warning: can produce
  false-positives, can ignore certain types of accesses, does not support VK_EXT_descriptor_buffer." Fresh release builds
  of both languages, VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true + VK_KHRONOS_VALIDATION_SYNCVAL_SHADER_ACCESSES_HEURISTIC=true,
  33 variants x 2 languages (09; 10 --frames 1/2/3; 11 --demo-resize with/without --no-present-fences; 12 --frames 1/2/3;
  13; 14; 15 default/--transfer-queue/--bar; 16; 17; 18 default/--push-descriptor; 19; 20 (4 s, past the frame-120
  write); 21 default/--nearest --address clamp/mirror/border; 22 default/--no-depth; 23 default/--no-mips/--anisotropy;
  24 --samples 1/2/4/8): all exit 0, stderr = only the bandicam warning. So the earlier "clean" claims hold with the
  heuristic too. Heuristic proven active (temporary edit of Rust 21, reverted, sha256 identical, rebuilt): after the draw,
  a texture layout transition SHADER_READ_ONLY -> GENERAL with src NONE gives, with the heuristic, "vkCmdPipelineBarrier2():
  WRITE_AFTER_READ hazard detected. vkCmdPipelineBarrier2 performs image layout transition on the VkImage ..., which was
  previously read by vkCmdDrawIndexed."; WITHOUT it: silent. I.e. sampled-image (descriptor) reads are not tracked by
  plain validate_sync either. Uniform-buffer reads were not tested separately (17-21 UBOs are host-written only, and host
  accesses are not tracked). Logs: scratchpad syncrecheck/.
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
- Screenshots (2026-10-11): originals in `examples/screenshots/` (800x600 client area, Rust release), captured with
  PrintWindow PW_CLIENTONLY|PW_RENDERFULLCONTENT (screen grabs can catch overlapping windows); lesson copies plus
  `*_zoom.png` crops (100x75 region, 4x nearest upscale, made with Pillow) in `src/assets/`, embedded only through
  `src/components/Shots.astro` (PNG on purpose: captions quote measured pixels; verified the built PNGs are
  pixel-identical). 17's binary predates later comment-only changes; the image is unaffected. 15, 18,
  21_texture_nearest and 24 full frames are captured but not embedded.
- Source links (2026-10-11): `Step` renders its own Starlight Tabs (same `syncKey` "code-lang" as course-kit CodeTabs)
  with `SourceLine` under each tab: GitHub link to the region's lines (`#L<a>-L<b>`, repo from course-kit `REPO` =
  astro.config `repo`) + run command (`src/lib/source.ts`: `stepSources`, `regionLines`, `CPP_PRESET`). Lessons import
  the course's `src/components/Snippet.astro` (engine Snippet + whole-file link), not the engine one. `regionLines`
  copies the engine's marker regex; tests/source.test.ts proves it matches `extractRegion` for every shown region
  (TODO: expose line numbers from course-kit lib/regions.ts and drop the copy). Texts: `srcLines/srcFile/srcRun` in
  src/i18n.ts. README.md "Build the examples" has the step -> lesson table (generated from course.json + `<Step file>`
  usages; regenerate when steps are added).
- C++ on Linux + CI (2026-10-11): presets `msvc-*` (cl, Windows only) and `linux-*` (compiler from `CXX`, Linux
  only) share a hidden `base` (Ninja, vcpkg toolchain from `$env{VCPKG_ROOT}`). CMakeLists fails at configure time
  without C++23 `<print>` (HAVE_STD_PRINT; Ubuntu 24.04's default g++ 13 lacks it -> `CXX=g++-14`). GCC 14 / Clang 18
  -Wextra (-Wmissing-field-initializers) flag every member a designated initializer omits unless it has a default
  member initializer (MSVC /W4 does not): fixed at the source - own structs get NSDMIs (18 `Frame::uniform =
  std::nullopt`, 27 `GpuTimes::current{}/total{}`), 16 names every VmaAllocatorCreateInfo/VmaAllocationCreateInfo
  member and builds VmaVulkanFunctions by value-init + assignment (its member set depends on VMA's config macros).
  Rule for new code: a struct used with designated initializers gives every member a default, or the initializer
  names them all. No Win32-only code was found; SHADER_DIR is an absolute CMake path. `.github/workflows/cpp.yml`
  (push/PR touching examples/cpp, shaders or itself; workflow_dispatch): windows-latest (MSVC 14.51, VS 18) and
  ubuntu-24.04 (g++-14); Vulkan SDK 1.4.341.1 via jakoch/install-vulkan-sdk-action v1.7.0 (needs `github_token`:
  its lavapipe lookup hit the anonymous API rate limit); vcpkg cloned at master e456309 (2026-10-10: glfw3 3.5.1,
  VMA 3.4.0; release 2026.07.29 still has glfw3 3.4), its binary cache in actions/cache saved right after configure.
  Linux needs X11 dev headers + autotools (vcpkg builds libx11). Smoke test: 01, 03, 13, 16, 25 run on lavapipe
  (Windows: Mesa 25.2.5 prebuilt; Linux: apt Mesa, LLVM 20) with the validation layer, clean. TRAPS: the Windows
  runner is elevated, so the loader ignores VK_LAYER_PATH/VK_ADD_LAYER_PATH and a cached SDK has no registry
  entries - the workflow registers VkLayer_khronos_validation.json under HKLM ExplicitLayers; the Windows runs use
  the runner's System32 loader 1.3.301 (enough: 1.4 instances and devices work through it).

- Module 12 shader check (2026-10-11, slangc 2026.1 from SDK 1.4.341.1, flags of sync-shaders.ts, spirv-val vulkan1.4,
  scratch shaders in scratchpad m12plan/): all four compile and validate, default and -O2.
  (a) mesh + task: OpCapability MeshShadingEXT + SPV_EXT_mesh_shader; groupshared payload -> TaskPayloadWorkgroupEXT
  (task and the `in payload` mesh parameter), DispatchMesh -> OpEmitMeshTasksEXT, SetMeshOutputCounts ->
  OpSetMeshOutputsEXT, OutputVertices/Indices/Primitives -> OutputVertices/OutputPrimitivesEXT/OutputTrianglesEXT,
  SV_CullPrimitive gets CullPrimitiveEXT + PerPrimitiveEXT. TWO BUGS with user per-primitive attributes: (1) the
  fragment-shader input that reads a per-primitive value is NOT decorated PerPrimitiveEXT (interface matching needs
  "equivalent decorations", spec 19.1.3), and Slang 2026.1 has no syntax for it (`perprimitive`/`perprimitiveEXT` are
  GLSL-only keywords, parse errors in .slang); (2) an explicit [[vk::location(N)]] on an OutputPrimitives member is shifted
  by the number of per-vertex locations (vertex 3 + primitive 5 -> emitted Location 9; vertex 0 + primitive 1 -> 2).
  Workarounds for step 32: per-meshlet data as a flat per-vertex attribute, or SV_PrimitiveID (builtin; Slang then
  declares the Geometry capability = geometryShader feature). (b) ray query: RayQueryKHR + SPV_KHR_ray_query; warning
  41012 (profile auto-extended) unless the entry point has `[require(spvRayQueryKHR)]` (silences it, same SPIR-V).
  (c) RT pipeline: RayTracingKHR + SPV_KHR_ray_tracing; payloads RayPayloadKHR/IncomingRayPayloadKHR, attributes
  HitAttributeKHR, TraceRay flags folded to constants (12 = ACCEPT_FIRST_HIT|SKIP_CLOSEST_HIT). A plain
  RWTexture2D<float4> adds StorageImageReadWithoutFormat + WriteWithoutFormat; `[[vk::image_format("rgba8")]]` removes
  both (image format Rgba8). (d) printf: NonSemantic.DebugPrintf + SPV_KHR_non_semantic_info, only the Shader
  capability; the OpString and the OpExtInst survive -O2.
- examples 28-29 (module 12, 2026-10-11, both languages; fmt + clippy -D warnings, C++ /W4 clean; release runs with
  VALIDATE_SYNC + SYNCVAL_SHADER_ACCESSES_HEURISTIC: only the bandicam warning; stdout identical without `[timing]` for
  28, 28 --printf (with the printf env vars, printf lines included), 29 default / --frames 1 / --no-stats --frames 3).
  Logs: scratchpad m12/. New shader `debug_printf.slang` (fragmentPrintf; pinned ['Shader']). CMake STEPS/WINDOW_STEPS
  and both READMEs list 28/29.
  28_debug_labels: scene of 22, 40 named objects (42 with --printf); cube 0/1 drawn as two draws (firstInstance 0/1)
  under their own labels; cmd labels frame > attachments, main pass > cube 0 | cube 1, inserted "to present"; queue
  label "frame N" around submit + present (closed on every path). The messenger prints pObjects (type, handle, name)
  and the cmd/queue labels (stderr only). VUID facts (validusage.json): vkEndCommandBuffer-commandBuffer-01815 is for
  SECONDARY command buffers only; primaries must balance per queue (vkCmdEndDebugUtilsLabelEXT-commandBuffer-01912;
  spec: "When viewed from the linear series of submissions to a single queue ... matched and balanced"; a region may
  span command buffers); queue labels: vkQueueEndDebugUtilsLabelEXT-None-01911. TRAP: naming the
  VkDebugUtilsMessengerEXT crashes (access violation inside vkSetDebugUtilsObjectNameEXT, also with bandicam disabled).
  Cause = Vulkan-Loader: terminator_CreateDebugUtilsMessengerEXT (loader/debug_utils.c) returns its own pNextIndex
  allocation as the handle, and terminator_SetDebugUtilsObjectNameEXT (loader/generated/vk_loader_extensions.c)
  translates only PHYSICAL_DEVICE, SURFACE_KHR and INSTANCE handles, so the driver gets a handle it never created
  -> the messenger is not named (comment in both files). VUIDs for names: 07872 instance-level, 07873
  physical-device-level, 07874 device-level objects.
  printf: with only VK_KHRONOS_VALIDATION_PRINTF_ENABLE=true NOTHING is printed (the layer warns "DebugPrintf logs to
  the Information message severity"; this messenger is WARNING|ERROR); with VK_KHRONOS_VALIDATION_PRINTF_TO_STDOUT=true
  as well the layer writes to stdout (and warns it skips the debug callback). Real output (scene at time 0):
  "pixel (400, 300): depth 0.967634, color (0.250000, 0.400000, 0.950000)" (cube 0, +z face, drawn first) and
  "pixel (400, 300): depth 0.966468, color (0.950000, 0.250000, 0.400000)" (cube 1 = .brg of blue, nearer, wins);
  identical in Rust and C++. With printf on, the layer forces device features (fragmentStoresAndAtomics,
  vertexPipelineStoresAndAtomics, shaderInt64, timelineSemaphore, vulkanMemoryModel, bufferDeviceAddress,
  scalarBlockLayout, 8/16-bit storage) and reports it as a WARNING.
  Named-object validation message (temporary edit of Rust 28: depth barrier src NONE/NONE; reverted, sha256 identical,
  rebuilt): "vkQueueSubmit2(): WRITE_AFTER_WRITE hazard detected. vkCmdPipelineBarrier2[frame::attachments] (from
  VkCommandBuffer 0x...[frame 1 command buffer] submitted on the current VkQueue 0x...[graphics + present queue]) writes
  to VkImage 0x1e000000001e[depth image], which was previously written by vkCmdEndRendering[frame::main pass] (from
  VkCommandBuffer 0x...[frame 0 command buffer] submitted on VkQueue 0x...[graphics + present queue])." + pObjects
  QUEUE "graphics + present queue", COMMAND_BUFFER "frame 1 command buffer", queue labels "frame 1" (m12/neg28.txt).
  The layer writes names in [brackets] after handles and cmd labels as [frame::attachments].
  RenderDoc 1.39 (portable, registered implicit layer C:\Programs\Ntutil\apps\prog\ide\renderdoc\renderdoc.json,
  enable_environment ENABLE_VULKAN_RENDERDOC_CAPTURE=1, layer api_version 1.3.131): with the layer the device reports
  apiVersion 1.3.0 (1.4.351 without), so every 1.4 step refuses to start: `renderdoccmd capture -w -c <file>
  28_debug_labels.exe --seconds 3` -> "No Vulkan 1.4 GPU ...", no capture. vulkaninfo --text diff (instance + device
  extension names, m12/vi_*.txt): 343 -> 216 entries; hidden e.g. pipeline_binary, maintenance6-11, device_fault,
  swapchain_maintenance1, surface_maintenance1, host_image_copy, descriptor_buffer/heap, layer_settings,
  cooperative_matrix, video_*, NV_device_diagnostic_checkpoints; still present: mesh_shader, ray_query,
  acceleration_structure, ray_tracing_pipeline, calibrated_timestamps, dynamic_rendering, debug_utils, swapchain.
  Nsight Systems 2026.1.3 (`nsys profile --trace=vulkan,vulkan-annotations`, 2 s of 28): works; with `--trace=vulkan`
  alone `nsys stats` reports "does not contain Vulkan Debug Extension (Vulkan Debug Util) data". vulkan_gpu_marker_sum
  (medians): main pass 7.4 us, cube 0 6.0 us, cube 1 0.065 us, attachments 0.065 us; vulkan_marker_sum (CPU) has the
  cmd ranges and one range per "frame N" queue label.
  29_gpu_queries: timestamps TOP_OF_PIPE + 3 x ALL_COMMANDS per frame (4 per slot), pipeline statistics (IA vertices,
  IA primitives, VS, clipping invocations/primitives, FS) around the rendering, queries reset on the host after reading
  (hostQueryReset), calibration (DEVICE + QPC) right before vkQueueSubmit2. The animation clock starts at frame 0, so
  frame 0's statistics are deterministic: IA 72 / 24, VS 48 (= 24 unique x 2 instances; range 48..72), clipping 24 / 24,
  FS 60794, identical in both languages and all runs. Calibrateable domains: DEVICE, QUERY_PERFORMANCE_COUNTER,
  PRESENT_STAGE_LOCAL_EXT; timestampPeriod 1, 64 valid bits on family 0. Measured (release, sync validation on, MAILBOX,
  3 s): GPU attachments 0.0 us, main pass 6.7-10 us, to present 0.0 us; submit -> first timestamp 95-160 us average
  (min ~50-70, max up to 4 ms), maxDeviation ~2.0 us, about the same for --frames 1/2/3 (GPU nearly idle); QPC rate from
  the first and last calibration: 9999702-9999706 ticks per device second (QPC is 10 MHz: the device clock runs ~30 ppm
  fast against QPC). FS invocations average ~57000-64000 per frame (one second of a later run: 70081). Screenshots
  examples/screenshots/{28_debug_labels,29_gpu_queries}.png (Rust release, PrintWindow client area, ~2.5 s in).
  Review fixes 2026-10-11: printed predictions now say "48..72 if each index is shaded at most once; reuse is
  implementation-dependent" (48..72 is NOT a spec bound: replaying GPUs may count more) and clipping primitives "at
  least 1 per triangle with a vertex inside the clip volume (culled or covering a sample)" (spec: "not rasterized" =
  culled primitives count too); pick-device comment: hostQueryReset is required for every 1.2+ device (spec feature
  requirements), so the check cannot fail on 1.4, only enabling matters. Rebuilt (fmt/clippy/MSVC clean), stdout
  identical Rust vs C++ w/o [timing]. Calibration VUIDs: domain must be reported = VkCalibratedTimestampInfoKHR-
  timeDomain-02354; vkGetCalibratedTimestampsKHR-timeDomain-09246 = each domain unique. GPU CLOCK STATES (nvidia-smi
  --query-gpu=clocks.gr,pstate -lms 500 alongside a 6 s run, scratchpad m12fix/): idle P8 210 MHz -> P2 1665 MHz at
  start (main pass 8.0/8.1 us) -> after ~3 s the driver drops to P3 1035-840, P5 750-690 MHz (main pass 11.0, 13.8,
  16.4 us) -> back up at the end (11.7). Whole-run averages mix clock states; maxDeviation per second 2.2-2.7 us
  (reviewer saw 3.2). Validation layer manifest: printf_to_stdout default TRUE (vkconfig applies it), printf_buffer_size
  1024 B per command buffer ("about 50 bytes" per message); with env vars only, stdout redirection stays off.
  TRAP: `vkconfig settings ...` (SDK Bin/vkconfig.exe CLI) WROTE %LOCALAPPDATA%/LunarG/vulkan/vk_loader_settings.json
  (HKLM LoaderSettings points there): validation forced "on" for every app + Nsight Systems layers as implicit ->
  every C++ Vulkan step crashed with 0xC0000005 (Rust survived). Deleted 2026-10-11 17:4x (copy in scratchpad
  vkcfg/). Never run vkconfig CLI commands other than read-only help; check that file if Vulkan apps misbehave.
  RenderDoc capture: the 12 names missing from cap28.xml are window surface, swapchain view 1/2, frame 0 scene
  uniforms/memory/descriptor set, frame 0 + frame 1 command buffer, frame 0 image available/in flight, render
  finished 1/2 (not "unused objects": frame 1's command buffer was used); "Ref All Resources" (--opt-ref-all-resources)
  not tried.
- pipeline-stats probe (2026-10-11, for tools/occupancy-and-registers): `tools/vk-probe` binary `pipeline-stats`,
  shader `tools/vk-probe/shaders/pressure.slang` (+ committed .spv, embedded with include_bytes!; entry points p8..p256 =
  N floats in a fully unrolled local array, 8 rounds; d64 = 64 floats indexed at run time = local memory control),
  `scripts/sync-pipeline-stats.ts` (slangc flags read from shaders/spv/manifest.json), IR dumps -> tools/vk-probe/out/
  (gitignored). NVIDIA 616.64 facts: every executable is named "CS"/"VS"/"FS" (descriptions "Compute Shader" etc.),
  subgroupSize 32; statistics, all UINT64: Register Count ("Number of temporary registers used by this shader stage"),
  Binary Size, Stack Size, Local Memory Size ("Size in bytes required per-thread for local memory"), then per stage
  Shared Memory Size (CS), Input Count/Output Count (VS), Input Count/Output Colors (FS). NO internal representations
  (count 0 with CAPTURE_INTERNAL_REPRESENTATIONS). TRAP: Local Memory Size is 68719476736 (= 16 << 32) for every
  shader and 68719476992 (= 16 << 32 | 256) for d64: only the LOW 32 bits are meaningful (d64: 256 B = 64 floats);
  the lesson must say so, never quote the raw value as bytes. VS Output Count 0 for the cube looks wrong too.
  Registers: saxpy/reduce/particles 16, cube VS 25 / FS 16, p8 19, p16 37, p32 70, p64 118, p128 230, p192 242,
  p256 255 + Shared Memory Size 15360 (= 60 B/thread; the shader declares no groupshared: presumably spilled to shared
  memory, an interpretation), d64 40 regs. Times (4 Mi invocations, median of 5, shared machine, run 2): p8 0.047,
  p16 0.085, p32 0.167, p64 0.348, p128 0.847, p192 1.205, p256 1.866, d64 ~317 ms. Validation: bandicam warning only.
## TODO

- [ ] Concept trees (engine v0.5.0): a reader check in a real browser (keyboard through all four views, window
  drag/resize, phones); headless Chrome verified click, Create, content swap, Escape and focus return in all four
  views (2026-10-11). Screenshots: scratchpad nodeview/final/ (session-local).

Current plan (2026-10-11), in order:
1. Module 12 per `notes/module-12-plan.md`: per pair of steps code (both languages) -> independent code review -> fixes
   -> EN lessons -> review -> fixes + RU -> push. Wave 1 = Slang check (mesh/task, ray query, RT pipeline, printf) +
   28/29 (in progress); then 30, 31, 32, 33+34, 35.
2. Open reviews: RU lessons of modules 4-11 (language + facts vs EN); second look at the module 11 EN fixes.
3. Course-wide cleanups listed below (Vk::new guard pattern, compute pipeline object in objects.json, README .spv list,
   15_staging.cpp comment, labs/graphics-pipeline wording, course-kit v0.4.1 release + `bun run check` in CI, C++ in CI).
4. With the user: term-map direction; ReBAR-on snapshot after they enable it; optional RenderDoc 1.46 / Nsight Graphics
   install (not downloaded without the user).

- [x] Module 12 wave 1 (2026-10-11): Slang check (State "Module 12 shader check") + examples 28/29 (State "examples 28-29").
- [x] 28/29 code review (ba3a454). EN lesson advanced/debugging-and-profiling written 2026-10-11 (module `advanced`
  "Advanced" / «Продвинутые темы» in course.json/course.ru.json, only this lesson listed; RU page = placeholder
  «Русская версия урока готовится.»). Regions: all 8 of 28 and all 11 top-level of 29; 29 `latency` and `statistics`
  are NESTED in `read-results` and shown through it (markers stripped by extractRegion). Snippet debug_printf.slang;
  Shots 28 only (src/assets/28_debug_labels.png; 29 is the same scene, not embedded). 9 terms (debug-object-name,
  debug-label, debug-printf, query-pool, timestamp-query, pipeline-statistics-query, calibrated-timestamps,
  time-domain, frame-capture) + RU overlay, 20 edges (scratchpad m12lesson/addterms12.py). Runs 2026-10-11 (scratchpad
  m12lesson/, release, validation on, sync OFF): 28 default Rust 2032 / C++ 2155 fps, stdout identical w/o [timing];
  28 --printf with PRINTF_ENABLE only: no shader output + "DebugPrintf logs to the Information message severity"
  warning; with PRINTF_TO_STDOUT too: the two pixel lines (Rust = C++), landing after "Window closed" in a redirected
  log; printf cost NOT measurable (3 alternating 3 s runs x 4 variants: 1142-1948 fps overall, overlapping). 29 x3
  alternating per language, 4 s: main pass 10.5-13.4 us, submit -> GPU start avg 70.3-77.5 us (min 46-50, max
  1.4-2.4 ms), maxDeviation 2.2-2.5 us, QPC 9999690-9999720 ticks per device second (~28-31 ppm; QPF = 10000000),
  1457-2070 fps, stdout identical w/o [timing], frame-0 statistics as before. RenderDoc 1.46 capture of Rust 28
  (`renderdoccmd capture -w -c <dir>\cap28 28_debug_labels.exe --seconds 12`, F12 sent to the window): frame 885,
  176844 bytes, device apiVersion 1.4.351 under the layer; `renderdoccmd convert -c xml`: all 8 labels of the frame
  and 28 of 40 names (frame-slot-1 objects, view 0, instance/device/queue...; not frame slot 0, views 1-2, surface,
  command buffers). [x] Independent review of the EN lesson, fixes applied 2026-10-11 (19 points; see State
  "examples 28-29" review fixes). [x] RU lesson written 2026-10-11 (authored; same Steps/Shots/VUIDs/outputs/quiz facts;
  anchors checked in dist). [ ] Independent review of the RU lesson.
- [ ] Step 32 (mesh shaders): avoid user per-primitive outputs (Slang 2026.1 bugs, State "Module 12 shader check") or
  post-process the SPIR-V like decorateNonUniform; try a newer Slang before deciding.
- [ ] 28 --printf: VK_EXT_layer_settings (printf_enable + printf_to_stdout set by the program) would remove the two env
  vars; not done (the brief asked for env vars).

- [x] Module 7 examples 10-12 (2026-10-11; resumed from WIP 7b36597): fmt + clippy -D warnings, C++ /W4 clean, run on
  the RTX 3080 Ti with VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true: zero messages except the bandicam API-version warning;
  stdout identical Rust vs C++ without `[timing]` lines (10: --frames 1/2/3; 11: both present paths; 12: --frames 1/2/3).
  Details in State "examples 10-12". Logs: scratchpad m7/ (session-local).
- [x] Screenshots of 10/11/12 (2026-10-11, see State "Screenshots"): 10 in frames-in-flight, 11 (640x480 after
  --demo-resize) in swapchain-recreation, 12 + zoom in barriers-and-layouts (EN + RU).
- [x] Module 7 EN lessons (2026-10-11): sync/{fences-and-semaphores (embeds QueueSync), frames-in-flight,
  barriers-and-layouts, timeline-semaphores, swapchain-recreation}; RU lessons written later the same day. Regions shown once:
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
- [x] Screenshots of 14/15 (2026-10-11): identical pixel for pixel; 14 in vertex-and-index-buffers, staging-uploads cites
  the identity in text (EN + RU).
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
- [x] Module 8: EN lessons reviewed (two rounds, fixes in State). [ ] Independent review of the RU lessons.
- [x] RebarCheck (EN+RU, 2026-10-11) presents ReBAR ON as a possible misconfiguration too (CSM/MBR boot, NVIDIA
  per-app profiles and measured regressions, uncached BAR heap vs readbacks); sources in REBAR.md.
- [x] Module 9 examples 17-20 (2026-10-11): fmt + clippy -D warnings, C++ /W4 clean (clean rebuild of all 20), run on
  the RTX 3080 Ti with VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true: only the bandicam warning; stdout identical Rust vs C++
  without `[timing]`/`[addr]` lines (17; 18 default and --push-descriptor; 19; 20). New shaders transform/device_address/
  bindless.slang (+ spv, manifest; tests/shaders.test.ts passes). Details in State "examples 17-20".
- [x] Screenshots of 17-20 (2026-10-11): 17 in uniform-buffers (regular hexagon = column-major upload verified
  visually too), 19 in buffer-device-address, 20 in bindless (EN + RU); 18 captured, not embedded (same picture as 17).
- [x] Module 9: reviews of 17-20 (code vs spec) and of the EN lessons done, fixes committed (bc7a8fd, 241fc0f).
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
- [x] Screenshots of 21-24 (2026-10-11): pixel-verified (texture orientation, sRGB round trip 40 -> 40, depth vs a
  CPU reference, mips vs --no-mips neighbor differences, 55 vs 7 colors at 4x/1x); embedded as pairs in
  textures-and-samplers (+ LINEAR/NEAREST zooms), depth, mipmaps, msaa (4x/1x zooms), EN + RU. Not captured:
  23 `--anisotropy`, temporal shimmer.
- [x] Module 10: review of 21-24 (code vs spec) done, fixes committed (d759d61); EN lesson review below.
- [x] 22/24 D24_UNORM_S8_UINT branch (DEPTH_STENCIL_ATTACHMENT_OPTIMAL, both aspects, pStencilAttachment): tested
  2026-10-11 by temporarily putting it first in the candidate list (both languages, reverted): runs, sync validation clean.
- [x] Module 10 EN lessons (2026-10-11): module `images` ("Images") in course.json/course.ru.json; RU pages are
  placeholders («Русская версия урока готовится.»). Regions shown once: 21 all except frame-resources
  (textures-and-samplers, + Snippet textured.slang, ObjectGraph build with image/device-memory/image-view/sampler/
  descriptor objects), 22 all except frame-resources (depth, + Snippet cube.slang), 23 all (mipmaps), 24 all (msaa).
  Not shown: 21 and 22 frame-resources (= step 17's Frame). 29 terms added (scratchpad m10lesson/addterms10.py) + RU
  overlay: texture, vk-sampler, image-tiling, format-features, buffer-to-image-copy, combined-image-sampler,
  texture-filtering, address-mode, anisotropic-filtering, srgb-format, texture-unit, depth-buffer, depth-format,
  depth-test, clip-space, perspective-projection, reversed-z, early-z, mipmap, lod, vk-cmd-blit-image,
  trilinear-filtering, texture-aliasing, msaa, sample-count, msaa-resolve, transient-attachment, sample-shading,
  supersampling; objects.json sampler got `term`. Older EN lessons link here instead of "module 10" (bindless,
  descriptor-sets x2, memory-types, staging-uploads, graphics-pipeline, image-views). Outputs quoted from Rust release
  runs 2026-10-11 (scratchpad m10lesson/, `--seconds 2`, validation on; C++ identical without [timing] for all 16
  configurations: 21 default/--nearest/--address mirror|clamp|border/--nearest --address clamp, 22 default/--no-depth,
  23 default/--no-mips/--anisotropy, 24 --samples 1/2/4/8 + default; stderr only the bandicam warning). Measured for
  the lesson: 23 three alternating 3 s runs each: default 2646-2881 fps, --no-mips 2773-2847, --anisotropy 2722-2817
  (no measurable cost). Real facts used: image memoryTypeBits 0b11 (no HOST_VISIBLE type for an optimal image),
  mip chain 1399808 bytes vs 1398100 texel bytes; BC7_SRGB_BLOCK has BLIT_SRC but not BLIT_DST (vulkaninfo
  --show-formats); spec: linear tiling guaranteed only for 2D/1 mip/1 layer/1 sample/TRANSFER usage; blits need a
  graphics queue (VUID-vkCmdBlitImage2-commandBuffer-cmdpool). Depth precision numbers are computed from the formula,
  not measured. The sync-validation negative tests of 22/24 are described from the State notes (full message not
  logged); a 2026-10-11 rerun with VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true printed only bandicam but showed no fps drop,
  so it did not prove the setting was active. Visual claims verified from screenshots later (2026-10-11, State "Screenshots").
- [x] Module 10: independent review of the EN lessons done; fixes applied 2026-10-11 (checked in vkspec 1.4.365):
  textures (maxLod 0 makes the rate irrelevant), depth (discard forces the depth write, often the test, late), mipmaps
  (bias clamp = clamp(mipLodBias + shader Bias, ±maxSamplerLodBias) per the limit's definition; BC "cannot be blit
  targets"; chains from CPU/offline tool/compute; blit VUIDs 00233 + 00234; implicit LOD also in compute/mesh/task via
  VK_KHR_compute_shader_derivatives; spec: "the derivative and LOD values are undefined" for quad-divergent
  image/sampler unless quadDivergentImplicitLod), msaa ("may never need backing memory"; hazard message from project
  notes; framebufferColorSampleCounts = float/fixed-point formats, integer -> framebufferIntegerColorSampleCounts);
  terms: texture-aliasing def, clip-space lost the wrong "NDC" alias, RU mipmap def reworded.
- [x] Module 10 RU lessons (2026-10-11): ru/images/{textures-and-samplers, depth, mipmaps, msaa} authored (same regions,
  outputs, VUIDs, measurements, ObjectGraph build and quiz facts as EN, with the review fixes).
- [ ] Module 10: independent review of the RU lessons (language + facts vs EN).
- [x] Module 11 examples 25-27 (2026-10-11): see State "examples 25-27"; CMake STEPS/WINDOW_STEPS (SHADER_DIR now
  defined for every step: 25 has no window but loads SPIR-V) and READMEs updated. fmt + clippy -D warnings, C++ /W4 clean.
- [x] Module 11: review of 25-27 (code vs spec) done; `sync:vuids` is not needed for examples (src/ only).
  Review 1 findings fixed 2026-10-11 (State "Module 11 review fixes"); lessons quoting 27's submit-graphics region or
  its measurements must say image_available is waited at ALL_COMMANDS and use the re-measured numbers.
- [x] Module 11 EN lessons (2026-10-11): module `compute` ("Compute") in course.json/course.ru.json; slugs
  compute/{compute-pipelines, particles, async-compute} (NOT compute-shaders); RU pages are placeholders («Русская
  версия урока готовится.»). Regions shown once: 25 all except cleanup (compute-pipelines, + Snippets saxpy.slang,
  reduce.slang), 26 all except cleanup (particles, + Snippet particles.slang), 27 all except pipelines (= 26's with
  `advance`, named in the text) and cleanup (async-compute). No ObjectGraph build: objects.json has no compute
  pipeline object (`pipeline` = vkCreateGraphicsPipelines); adding one is a TODO below. 15 terms added (scratchpad
  m11lesson/addterms11.py) + RU overlay: compute-shader, compute-pipeline, dispatch, workgroup, invocation,
  shared-memory, workgroup-barrier, parallel-reduction, readback (compute-pipelines), gpu-particles, point-primitive
  (particles), async-compute, queue-concurrency, sharing-mode, ping-pong-buffer (async-compute); storage-buffer and
  subgroup stay owned by bindless; vk-pipeline def (EN+RU) widened to compute. Older EN lessons now link here
  (queue-families x2, fences-and-semaphores, window-and-surface, bindless, mipmaps, descriptor-sets); their RU
  counterparts got the same links on 2026-10-11 (see TODO). VUIDs named (synced): RuntimeSpirv-x-06429/06432, vkCmdDispatch-groupCountX-00386,
  vkCmdDispatch-None-10672, VkComputePipelineCreateInfo-stage-00701, VkWriteDescriptorSet-descriptorType-00333,
  VkGraphicsPipelineCreateInfo-topology-08773, VkBufferCreateInfo-sharingMode-00914. Runs for the lesson (scratchpad
  m11lesson/, fresh release binaries of 10:23, validation on, sync off; stdout Rust == C++ without [timing] for 25
  default/--uncached/--count 8388608, 26, 27 async and --same-queue x3): 25 saxpy 0.55-0.65 ms (first runs up to
  0.96 ms), CPU copy HOST_CACHED 0.80-1.21 ms (2.69 ms in the first C++ run) vs --uncached 23.5-29.9 ms, 8M saxpy
  5.1-5.9 ms; 26 1530/1866 fps (Rust/C++, 3 s); 27 second series (3 alternating runs x 2 languages, 4 s): graphics
  273.8-276.8 us, compute 45.3-48.2 us, overlap async 3.2-9.4 us (7-20%), same-queue 0.0, fps async 1130-1546 vs
  same 1114-1580 (lesson table shows it next to the "Module 11 review fixes" series). Sync re-check (both env vars,
  25/26/27/27 --same-queue, both languages): only bandicam. `spirv-dis reduce.spv`: `OpControlBarrier %uint_2 %uint_2
  %uint_264` (quoted). Spec quotes used: graphics => some family has graphics+compute; largePoints; CONCURRENT "may
  result in lower performance access"; EXCLUSIVE without transfer "undefined for any read accesses performed by the
  second queue family"; VK_TIME_DOMAIN_DEVICE_KHR "comparable with device timestamp values captured using
  vkCmdWriteTimestamp or vkCmdWriteTimestamp2"; OpFAdd/OpFMul "Correctly rounded"; HOST_CACHED read speed; fence Note.
  Not sourced, hedged in text: NVIDIA shared memory = L1 array, workgroup stays on one SM, hardware compute queues.
- [x] Screenshots of 26/27 (replaced 2026-10-11 by the 800x600 client-area captures, commit 8778b73). Old note: the first `examples/screenshots/{26_particles,27_async_compute}.png` were 795x628 window grabs WITH the
  title bar (not the 800x600 client-area convention; their colors are off the shader ramp, e.g. B 66 < 89, so they were
  resampled). The lesson copies in src/assets were recaptured 2026-10-11 with scratchpad cap.py (PrintWindow client
  area, Rust release, `--seconds 4`, ~2.5 s): 800x600, every lit color within 0.56 of the sRGB-encoded blue->orange
  ramp, 26: 175375 lit px, 359 colors, disk bbox 479x480; 27: 177328 lit px, 356 colors. Owner of examples/: replace
  the originals with these (copies in scratchpad m11lesson/).
- [x] Module 11 EN review fixes (2026-10-11, vkspec 1.4.365 + validusage.json SDK 1.4.341): push constants are per
  shader stage, not bind-point state ("Binding a pipeline with a layout that is not compatible with the push constant
  layout does not disturb the push constant values"; lesson, terms.json/ru `compute-pipeline`, 25 record comment both
  languages); HOST_CACHED Note quoted; vkCmdWriteTimestamp2 Note "may write the timestamp at any stage that is
  logically later than stage" (compute-pipelines; async-compute: 0.0 us on one queue is an observation, a late start
  timestamp could hide overlap); COMPUTE -> HOST barrier = AVAILABLE in the host domain, visible via HOST_COHERENT or
  invalidate (lesson, quiz Q3, 25 readback-barrier + read-results comments both languages); skipping a workgroup
  barrier = non-uniform control flow, undefined (hang OR wrong sums); integrator = semi-implicit (symplectic) Euler
  (particles.mdx, 26 dt comment both languages, particles.slang comment; sync:shaders: particles.spv bytes unchanged,
  only manifest sourceSha256); particles "The program states the arrangement at start-up"; "dark streak" hedged;
  async-compute cites sharingMode-00914 + -01419 (synced), first-series footnote (one graphics/compute range for both
  modes), pipelines region = 26's + `advance` + binding 1 `previous`; terms `async-compute` "mainly when" (+ru).
  RE-MEASURED 25 default vs --uncached (release, validation on, sync off, 6 alternating pairs per language, 12 runs per
  variant, logs scratchpad m11rev/): CPU copy type 4 0.81-1.34 ms (median 1.17) vs type 3 23.6-25.4 ms (median 24.4),
  18-31x per pair (median ~21x); SAXPY GPU 590-954 us (median 676) vs 568-1260 us (median 697), uncached slower in 8
  of 12 pairs, spread >> difference: the lesson says the GPU-side cost is not shown either way (old "same time" claim
  removed). Rust fmt/clippy -D warnings, C++ release build of 25/26 clean.
- [x] Module 11 RU lessons written (2026-10-11): ru/compute/{compute-pipelines, particles, async-compute}, authored
  (Houdini wrangle/OpenCL SOP, POP solver, SOP Solver Prev_Frame, TOP/farm analogies), same Steps/Snippets/Shots/VUIDs/
  program output/measurements (incl. the review fixes and the new 25 table), quizzes pass lint; all 616 lesson links
  of both languages resolve (scratchpad m11rev/links.ts). RU links to compute added in ru/{devices/queue-families (2,
  also "пригодятся в продвинутой части" -> staging + async compute), sync/fences-and-semaphores (also the stale "появится
  вместе с async compute"), presentation/window-and-surface, descriptors/bindless (also textures link), images/mipmaps,
  descriptors/descriptor-sets}.
- [ ] Module 11: independent review of the RU lessons (language + facts vs EN), and a second review of the EN fixes.
- [ ] objects.json: a compute pipeline object (vkCreateComputePipelines, parents device + pipeline-layout + shader
  module createOnly) would let compute-pipelines show an ObjectGraph build; needs ObjectGraph/test review.
- [ ] Course-wide (Rust windowed steps): `Vk::new` returns through `?` after creating objects, leaking them on an
  error midway (no Drop guard until `Self` exists; C++ RAII has no such gap). Harmless at process exit, but not what
  the course teaches; decide on a guard pattern (as in 25's `Objects`) for all `Vk::new` before changing them.
- [x] Re-run steps 09-24 with VK_KHRONOS_VALIDATION_SYNCVAL_SHADER_ACCESSES_HEURISTIC=true (2026-10-11): clean, see
  State "Sync validation re-check 09-24". Lessons not edited.
- [x] Lessons qualify what plain VALIDATE_SYNC tracks (2026-10-11, EN+RU): barriers-and-layouts has the one aside
  `### What it sees, and what it does not` (ru `### Что она видит, а что нет`; API-visible accesses by default, shader
  descriptor accesses only with SYNCVAL_SHADER_ACCESSES_HEURISTIC, setting text quoted, step 21 proof, host accesses
  untracked, "clean" in the course = both variables, 09-24 re-run); the intro and Underneath bullet link to it;
  uniform-buffers, push-constants, bindless, timeline-semaphores name both variables / link the aside. Not changed
  (checked, correct as written): staging-uploads (copy + vertex/index input, host writes), images/depth and msaa
  (attachments), buffer-device-address (GPU-AV only). New lesson claims of "sync-clean" must link the aside.
- [ ] Async compute shows no gain on the RTX 3080 Ti with 2^20 particles (CPU/present bound); a lesson experiment
  could add a heavier kernel or FIFO + a GPU-bound load to show when overlap helps - measure before claiming.
- [ ] READMEs: the "Steps 08+ use triangle.spv / mesh.spv" sentence predates steps 17-27 shaders; list all .spv files.
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
- [x] C++ examples in CI (2026-10-11): `.github/workflows/cpp.yml`, Windows MSVC + Ubuntu GCC 14, build of all 27
  steps + lavapipe smoke test of the windowless ones; Linux presets. See State "C++ on Linux + CI".
- [ ] Decide with the user: term maps put foundations at the bottom, ObjectGraph puts the root on top
- [ ] After enabling ReBAR on the dev machine (steps in REBAR.md): capture a second snapshot (`sync:gpu` ->
  `gpu-rebar.json` or a profile field) and let the memory labs switch between OFF (measured) and ON (measured).
  `src/data/gpu.json` stays the ReBAR-off snapshot; until then the ON state is described in words only
  (`src/components/RebarCheck.astro`, lab notes).
- [x] Modules 4-11 (presentation, pipeline, commands, sync, memory, descriptors, images, compute), EN + RU, published.
- [ ] User review of modules 4-11.
