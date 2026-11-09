# Test environment: RX 5500 XT on Windows

The AMD changes whose comments cite this file were tested on the single PC below, on one person's own copy of the game.
Numbers come from `TOTK_TRACE=perf` logs and from Windows' GPU memory counters.

## Hardware and system

| Component | Value |
|---|---|
| GPU | AMD Radeon RX 5500 XT, 4 GB (PCI 1002:7340), no `VK_EXT_mesh_shader` |
| Driver | AMD Software: Adrenalin Edition 26.10.1, Windows driver 32.0.21045.16010 (2026-10-15) |
| Vulkan | API 1.4.315, driver version 2.0.353, `VK_DRIVER_ID_AMD_PROPRIETARY`, "AMD proprietary shader compiler" (`vk_caps.amd_sc`) |
| Memory heaps | device-local 3824 MiB (budget 2.7 to 3.4 GiB, lower with other programs open), host 8158 MiB, BAR 256 MiB |
| CPU | AMD Ryzen 5 2600, 6 cores / 12 threads, 3.4 GHz |
| RAM | 16 GB DDR4, 2 x 8 GB TeamGroup UD4-2666, running at 2400 MT/s |
| Motherboard | ASRock A320M-HD, BIOS P5.40 (2019-07-08) |
| Storage | Kingston A400 480 GB SATA SSD |
| Display | 1920x1080, 75 Hz |
| Operating system | Windows 11 Pro 25H2, build 26200, 64-bit |
| Page file | 32768 MB initial, 49152 MB maximum (commit limit 49 GB) |

## Software

| Item | Value |
|---|---|
| Base | v0.2.8 (`90000ee`), translator id `16b59e1dc1658e0a` |
| Game | Ocarina of Time for Switch 2, version 1.1.1 (`oot111`) |
| Toolchain | clang 23.1.4 (x86_64-w64-windows-gnu), Python 3.14.9, ninja 1.13.2 |
| Settings | docked, `oot_resolution=auto`, `astc_bc=1`, `vk_present=fifo`, FSR upscaler, fullscreen |
| Environment | `DKB_SKIP_CS` with the ten giant compute programs off (the list used before v0.2.7), `TOTK_PREP_CS_MAX_WORDS=200000` |

## Play sessions (2026-11-08)

All of them load a save in Kokiri Forest. Windows' GPU memory counters reach up to about 4.0 GB dedicated and 7.2 GB shared.

| Build | What was played | Length | Result |
|---|---|---|---|
| v0.2.8 and all the changes | Forest, a house, the F1 menu opened 12 times with the VRAM full | 5 min | Clean exit. Forest at about 12 fps, GPU 41 to 52 ms per frame |
| v0.2.8, all but the menu change | Forest | 1 min | Crash the moment F1 opened: `amdvlk64.dll+0x245360`, read of -1, thread `vk present` |
| v0.2.7 and the earlier changes, handheld | Loading the save | none in play | Driver abort at `amdvlk64.dll+0x2eed041` (0xc0000409) during the load, VRAM at 100% |
| v0.2.7 and the earlier changes, docked | Forest, 3D at 50% | 7 min | Clean exit. 9 to 14 fps |
| v0.2.7 and the earlier changes, docked | Start of the game, forest | 13 min | Clean exit. 55 to 57 fps, then 15 to 18 fps |

After the first row, the code changed only where it does nothing on this card: guards for drivers without pipeline binaries and for GPUs on other driver branches, and comments. Those builds passed `--selftest`, `--probe-vkmem` and two preparation runs, one of them with `TOTK_PIPE_BINARIES=0`, and were not played.

Shader preparation on v0.2.8 with the game closed: 21 minutes, 1380 graphics pipelines, 50 big compute shaders built with the optimizer and none failed. Free RAM never went below 2.3 GB. The 132,185-word compute shader took 82 s, and the forest's 153,000 to 186,000-word ones 5 to 13 s each. The 29 pipelines that failed all use mesh shaders.

## Still wrong on this card

These are not fixed by the changes that cite this file:

- The forest runs at about 12 fps. The GPU is the limit, and the costliest program takes 7 of the 44 ms the profiler timed per frame.
- Draws that need mesh shaders are dropped, because the card has none (29 pipelines).
- Pots in the houses and a lifted bush render overexposed. Teal blocks appear now and then against the sky and the fog.
- The driver abort at `amdvlk64.dll+0x2eed041` can still happen while a map loads with the VRAM at 100%.

## Not tested

- Any other GPU, driver or operating system, including Linux.
- Areas beyond Kokiri Forest and sessions longer than 13 minutes.
- Each change on its own in play: they were played together, and each branch alone was only compile-checked.
