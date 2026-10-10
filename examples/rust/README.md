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
`::load`). It `vulkan-1.dll` at runtime, so no Vulkan SDK is needed to build.

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

The debug messenger in steps 02-05 reports only `[WARNING]` and `[ERROR]`.
A warning from a third-party implicit layer (screen recorders, overlays) is
not about our code: implicit layers are explained in the loader lesson.
