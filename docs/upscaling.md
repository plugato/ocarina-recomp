# Upscaling and frame generation

How Ocarina's picture reaches the window, what each option does, and why some well-known ones are not here.

## The path of a frame

1. The game draws its 3D scene at 1280x720 (its dynamic resolution may lower it) and tone maps it.
2. The game's own upscaler (compute `fd029a317a9a4b94`, 1280x720 to 1706x960) does not run: it lost the device. The
   runtime stands in for it (`vk_exec_draw.c` `big_compute_blit`). That upscaler is spatial: its only input is the
   tone-mapped colour.
3. The game draws its UI over the 1706x960 picture and the present scales it to the window (`TOTK_UPSCALER`: FSR 1 or
   bilinear, `vk_fsr.c`).

## Options

| Setting | Where | What |
|---|---|---|
| `scene_upscaler = easu` (default) | scene, step 2 | AMD FSR 1 EASU (`oot_scene_easu.comp`) |
| `scene_upscaler = bilinear` | scene, step 2 | the bilinear sample of `oot_scene.comp` |
| `scene_upscaler = taa` | scene, step 2 | temporal anti-aliasing (`oot_taa.comp`) before EASU, with camera motion vectors |
| `scene_upscaler = fsr2` | scene, step 2 | AMD FSR 2.2 (`vk_fsr2.cpp`, `runtime/third_party/fsr2`, MIT) with camera motion vectors; RCAS at `sharpness` |
| `upscaler = fsr` / `bilinear` | present, step 3 | AMD FSR 1 EASU + RCAS to the window (`sharpness`) |
| `framegen = 2..4` | present | frame interpolation (`vk_framegen.c`, after NotRexed/Hookshot); 1: off |

All of them are in `[display]` of `totk.ini` and in the in-game menu (Screen, Picture), and apply at the next frame.

The motion vectors come from the scene depth and the camera of each frame, taken from the game's sun-visibility probe
(`oot_motion.c`). They are exact for the camera's motion and the static world; moving objects (Link, foliage in the
wind) have none of their own, so the temporal passes fall back to the current frame where the history disagrees.

The game does not jitter its camera between frames, so FSR 2 and the TAA add stability (less shimmer on foliage and
edges) but no detail beyond the render resolution. Jittering the game's projection is possible future work.

## Not included

- **NVIDIA DLSS.** NVIDIA's RTX SDK licence (section 4e) does not allow the SDK to be made subject to an open-source
  licence, and this repository is GPL-3.0 with GPL code from others (Ryujinx, Eglantine): DLSS cannot ship in the
  releases or the repository. On Windows the SDK's link libraries are MSVC builds (`nvsdk_ngx_s.lib`), which this
  project's llvm-mingw build does not link. The Linux port of a contributor builds DLSS from a user's own SDK copy
  (`TOTK_NGX_DIR`, `libnvsdk_ngx.a`); that stays a personal build.
- **AMD FSR 3.1 / FSR 4 (AMD FSR SDK 2.x).** Its licence forbids reverse engineering and its upscalers ship as signed
  DLLs rather than shader sources, which does not sit well with GPL-3.0; FSR 4 also needs an RDNA 4 GPU. FSR 2.2, the
  last release with MIT sources, is used instead.
