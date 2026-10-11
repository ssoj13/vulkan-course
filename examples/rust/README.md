# Vulkan examples (Rust, ash)

Companion code for the Vulkan textbook. Each lesson step is one self-contained
binary in `src/bin/`: it repeats the previous steps, so you can read a single
file top to bottom. Code embedded in the lessons is marked with
`// #region <name>` ... `// #endregion` comments.

## Requirements

- Rust 1.85+ (edition 2024)
- A GPU driver with Vulkan 1.4 support (the examples refuse older devices)
- The [Vulkan SDK](https://vulkan.lunarg.com) - it provides the validation layer
  `VK_LAYER_KHRONOS_validation`, which steps 02-05 require

The crate uses [`ash`](https://github.com/ash-rs/ash) pinned to a master commit
(`f4c2ca3`, Vulkan-Headers 1.4.352) instead of the crates.io release: the latest
release, 0.38 (2024), only has Vulkan 1.3 bindings. Master renames a few APIs
(`push_next` is now `push` from `vk::TaggedStructure`, extension `::new` is
`::load`). It loads `vulkan-1.dll` at runtime, so no Vulkan SDK is needed to build.

Steps 06+ open a window with [`winit`](https://github.com/rust-windowing/winit) `=0.30.13`
(the latest stable release; ApplicationHandler API: winit calls the program back, and
windows are created in `resumed`) and create the Vulkan surface with `ash-window`,
taken from the same ash git commit (the crates.io ash-window 0.13 is built against
crates.io ash 0.38, whose types would not match).

Steps 08+ use the SPIR-V in `../../shaders/spv/`, compiled from `../../shaders/*.slang`.
It is committed, and embedded with `include_bytes!`, so building needs no shader compiler.
After editing a `.slang` file run `bun run sync:shaders` in the repository root (needs the
Vulkan SDK: slangc + spirv-val); `tests/shaders.test.ts` fails if the .spv is stale.

## Run

```
cargo run --bin 01_instance
```

## Steps

| Binary | What it does |
|---|---|
| `01_instance` | Load the Vulkan loader, list extensions, create and destroy an instance |
| `02_validation` | Enable the validation layer and a debug messenger |
| `03_physical_devices` | Enumerate GPUs, print properties, features and limits, pick one |
| `04_queue_families` | List queue families, find graphics / transfer / compute families |
| `05_logical_device` | Create a logical device with Vulkan 1.3 features (on a Vulkan 1.4 GPU) and fetch a queue |
| `06_window_surface` | Window (winit), surface extensions, `VkSurfaceKHR`, graphics + present family, surface capabilities, formats and present modes |
| `07_swapchain` | Device with `VK_KHR_swapchain`; choose format, present mode, extent, image count; swapchain and image views |
| `08_pipeline` | Slang shaders as SPIR-V, shader module, empty pipeline layout, graphics pipeline with dynamic rendering |
| `09_triangle` | Command pool and buffer, fence + semaphores, the frame loop (acquire, record, submit, present): the first triangle |
| `10_frames_in_flight` | One command buffer, fence and acquire semaphore per frame in flight (`--frames 1\|2\|3`); simulated CPU/GPU work (`--cpu-ms`, `--draws`) and fence-wait statistics |
| `11_resize` | Resizable window: swapchain recreation on resize / OUT_OF_DATE / SUBOPTIMAL with `oldSwapchain`, present fences (`VK_KHR_swapchain_maintenance1`) and a retire list, or `--no-present-fences` (vkDeviceWaitIdle); minimized = paused; `--demo-resize` |
| `12_timeline` | One timeline semaphore instead of the fences; binary semaphores stay for acquire/present; offscreen 400x300 image blitted to the swapchain with four exact sync2 barriers |

Steps 06-12 keep their window open until you close it; steps 10-12 also take
`--seconds S` to close it after S seconds. Lines that depend on timing start with
`[timing]` (drop them to compare the Rust and C++ output). Steps 06-10 and 12 have a
fixed-size window: resizing (and minimizing, which makes the swapchain out of date)
needs swapchain recreation, which is step 11 - the others stop cleanly with a message.
Module 8 (memory) continues at step 13.

![Step 09](../screenshots/09_triangle.png)

The debug messenger in steps 02-12 reports only `[WARNING]` and `[ERROR]`.
A warning from a third-party implicit layer (screen recorders, overlays) is
not about our code: implicit layers are explained in the loader lesson.
