# Resizable BAR on the course dev machine

Why this matters for the course: with Resizable BAR (ReBAR) off, the CPU can map only a 256 MiB window of VRAM, so the
Vulkan heap that is both `DEVICE_LOCAL` and `HOST_VISIBLE` is tiny (214 MiB in `src/data/gpu.json`). With ReBAR on, that
window covers the whole VRAM and the CPU can write straight into most of the 12 GB. The `memory-path` and
`buffer-memory` labs show the current state; this file says how to switch it on and how to check it.

## The machine (checked 2026-10-10)

| Part | Value | How it was read |
| --- | --- | --- |
| Motherboard | Gigabyte X570 AORUS ELITE WIFI, BIOS F41a (2026-04-13) | `Get-CimInstance Win32_BaseBoard`, `Win32_BIOS` |
| CPU | AMD Ryzen 9 5900X (Zen 3: supported) | `Win32_Processor` |
| GPU | RTX 3080 Ti, VBIOS 94.02.71.00.ca, driver 616.64, PCIe 4.0 x16 | `nvidia-smi --query-gpu=...` |
| Boot | UEFI (`winload.efi`), Secure Boot on, so CSM is already off | `bcdedit`, `Confirm-SecureBootUEFI` |
| ReBAR now | **off**: BAR1 total = 256 MiB | `nvidia-smi -q -d MEMORY` |

NVIDIA's requirements (support article 5165): a supported CPU, motherboard firmware and GPU VBIOS, driver 465.89 or
newer, UEFI boot with CSM disabled, a GPT system disk, and "Above 4G Decoding" enabled. Everything except the two BIOS
options is already in place here. The RTX 3080 Ti came out after the ReBAR rollout (spring 2021); NVIDIA leaves the
VBIOS question to the card vendor, so if the steps below don't turn it on, run the vendor's VBIOS update tool (it
reports whether an update is needed).

## Turn it on

1. Reboot into the BIOS (press `Del` during boot), switch to Advanced Mode (`F2`).
2. `Settings` -> `IO Ports`:
   - `Above 4G Decoding` -> **Enabled**
   - `Re-Size BAR Support` -> **Auto** (or Enabled)
   On some Gigabyte BIOS versions `IO Ports` sits under `Miscellaneous`. The `Re-Size BAR Support` item only appears
   after `Above 4G Decoding` is enabled.
3. `Boot` -> `CSM Support` must stay **Disabled** (it is: Secure Boot works only without CSM).
4. `F10`, save and exit.

## Check it

```powershell
nvidia-smi -q -d MEMORY        # BAR1 Memory Usage -> Total: now 256 MiB; with ReBAR on it covers the VRAM (a power of two, e.g. 16 GiB)
```

Or NVIDIA Control Panel -> System Information -> "Resizable BAR: Yes".

Then refresh the course data and look at the heaps:

```powershell
bun run sync:gpu               # rewrites src/data/gpu.json from vulkaninfo
bun test tests                 # tests/gpu.test.ts pins today's heap layout; update it to the new one
```

Expected: the `DEVICE_LOCAL` + `HOST_VISIBLE` heap grows from 214 MiB to (almost) the whole VRAM. Not verified on this
machine yet: NVIDIA enables ReBAR per application profile for games; whether its Vulkan driver exposes the large heap to
any Vulkan program is exactly what `vulkaninfo` will show after the switch - record the result here.

## Course decision after switching

Decided (2026-10-10): the course teaches ReBAR OFF vs ON as a common misconfiguration the reader should check on their
own machine (it was off here and nobody noticed until a glitch). `src/data/gpu.json` stays the measured ReBAR-off
snapshot; the memory labs and the gpu lesson explain both states and how to check. After enabling ReBAR, capture a
second snapshot (ReBAR on) and let the memory labs switch between the two measured profiles; until then the ON state is
described qualitatively, never with invented numbers.

Both states are presented as possible misconfigurations (`src/components/RebarCheck.astro`, 2026-10-11). ON-side
pitfalls stated there: UEFI/CSM-off requirement breaks legacy (MBR) installs; NVIDIA enables ReBAR per application
profile because some games regress (TechSpot measured regressions, e.g. Watch Dogs Legion -10% at 1080p:
https://www.techspot.com/review/2234-nvidia-resizable-bar/), forcing it globally with profile tools can regress; in
Vulkan code a large DEVICE_LOCAL|HOST_VISIBLE heap is uncached for CPU reads, so readbacks belong in HOST_CACHED memory.

## Undo

Set `Re-Size BAR Support` back to Disabled (and `Above 4G Decoding` if wanted). If the machine does not boot after
enabling it (rare, usually a CSM/legacy boot issue), clear CMOS or load optimized defaults and re-enable only
`Above 4G Decoding`.

## Sources

- NVIDIA: "GeForce RTX 30 Series Performance Accelerates With Resizable BAR Support", support article 5165:
  https://nvidia.custhelp.com/app/answers/detail/a_id/5165
- Gigabyte BIOS menu location (Settings -> IO Ports), third-party guides:
  https://pool.kryptex.com/pt-br/articles/rebar-4gdecoding-csm-mining-pt-br,
  https://www.antwort.net/a/tutorial/anleitung-resizable-bar-auf-dem-gigabyte-x570-aorus-elite-wifi-aktivieren.html
- Vulkan-only ReBAR in the NVIDIA Linux driver and the BAR1 check: https://www.phoronix.com/review/rtx3080-linux-rebar
