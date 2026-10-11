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

Steps 06-09 keep their window open until you close it. The window has a fixed size:
resizing (and minimizing, which makes the swapchain out of date) needs swapchain
recreation, which is module 7 - step 09 stops cleanly with a message instead.

![Step 09](../screenshots/09_triangle.png)

The debug messenger in steps 02-09 reports only `[WARNING]` and `[ERROR]`.
A warning from a third-party implicit layer (screen recorders, overlays) is
not about our code: implicit layers are explained in the loader lesson.
