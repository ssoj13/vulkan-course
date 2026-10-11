# vk-probe

Captures the real Vulkan data the labs show, so every number and message in `src/data/` can be reproduced instead of
copied by hand. Each binary runs real Vulkan calls on GPU 0 and writes one JSON file in the format the labs read.

| Binary | Writes | Run from the repo root | What it does |
|---|---|---|---|
| `buffer-requirements` | `src/data/buffer-requirements.json` | `bun run sync:buffers` | `vkGetBufferMemoryRequirements` for every non-empty combination of six usage flags at five sizes, plus the memory types' raw property flags |
| `validation-log` | `src/data/validation-log.json` | `bun run sync:validation-log` | step 02's instance with a messenger for every severity, run plain, with the messenger leaked, and with `DEVICE_ADDRESS_BINDING_EXT`; per-type totals and a sample of exact messages |

Requirements: a Vulkan 1.4 driver; `validation-log` also needs the Vulkan SDK (`VK_LAYER_KHRONOS_validation`).
Building needs neither (ash loads the loader at runtime), so CI only lints this crate.

Notes:
- The output is machine-specific by design: it is the course machine (see `gpu.json`). `validation-log` picks its
  sample by message text (`PICKS`) and fails if a message is missing, e.g. on a machine without the Bandicam layer.
- Handle values in messages (`VkInstance 0x...`) change on every run; the lesson quotes its own run.
- `date` in the metadata is the UTC date of the capture.
- ash is pinned to the same git commit as `examples/rust` (Vulkan 1.4 bindings are not on crates.io yet).
