# Vulkan textbook - C++ examples

Companion code for the lessons. One self-contained program per step; each step
repeats the setup of the previous ones, so a single file shows the whole program.
Style: Vulkan-Hpp RAII (`vk::raii::*`) with designated initializers
(`VULKAN_HPP_NO_STRUCT_CONSTRUCTORS`), like the Khronos Vulkan Tutorial.

## Requirements

- Vulkan SDK 1.4.x (`VULKAN_SDK` set; provides headers, loader, validation layer)
- A Vulkan 1.4 GPU driver with `dynamicRendering` and `synchronization2` (the examples refuse older devices)
- CMake 3.25+, Ninja, MSVC (Visual Studio 2022 17.7+ / toolset 14.51). Run from a
  developer prompt (or any shell where `cl.exe` is on `PATH`).
- C++23 mode is used only for `std::print`; the code is otherwise plain C++20.

## Build and run

```powershell
cmake --preset msvc-debug            # build dir: build/msvc-debug
cmake --build --preset msvc-debug
.\build\msvc-debug\01_instance.exe
```

`msvc-release` works the same way (`build/msvc-release`). No LTO is used.

## Steps

| Step | File | New in this step |
|------|------|------------------|
| 01 | `src/01_instance.cpp` | loader, `ApplicationInfo`, instance extensions, `Instance` |
| 02 | `src/02_validation.cpp` | validation layer, debug callback, messenger (also chained into instance creation) |
| 03 | `src/03_physical_devices.cpp` | enumerate GPUs, properties, features, limits, pick one |
| 04 | `src/04_queue_families.cpp` | queue families: graphics, dedicated transfer, async compute |
| 05 | `src/05_logical_device.cpp` | queue create info, enabling Vulkan 1.3 features on a 1.4 device, `Device`, `Queue` |

Lessons embed code by `// #region <name>` ... `// #endregion` markers; the names
match the Rust examples.

## Notes

- Debug messenger severities are Warning | Error only: Info makes the loader print
  100+ lines. Some machines have third-party implicit layers that warn once (for
  example `VK_LAYER_bandicam_helper` reporting API 1.2 < 1.4); that is not a bug here.
