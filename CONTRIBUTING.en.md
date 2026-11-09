# Contributing to Ocarina Recomp

[Leia em português](CONTRIBUTING.md)

All help counts, even if you do not program. This guide explains how to test, how to build, how to investigate a graphics bug and how to send your contribution.

## The most important rule

**Nothing that comes from the game goes into this repository.** That includes:

- the game, keys, `exefs`, `romfs` and the `.nup`;
- `ocarina.exe`, the code in `generated/` and everything in `build/`;
- saves, the shader cache, SPIR-V of the game's shaders, frame dumps and screenshots.

Hashes, addresses (`main+0x6e6dc4`) and descriptions of what the code does are fine.

`tools/check_content.py` checks this on every push. Turn on the local check once per clone:

```powershell
git config core.hooksPath .githooks
```

Never use `--no-verify`. Do not ask for or share the game in issues or pull requests.

## Without programming: test

Build using the [step-by-step guide](docs/build-by-hand.md), play, and open an [issue](https://github.com/sostenesfreitas/ocarina-recomp/issues) with:

- graphics card, driver version, processor and amount of RAM;
- where in the game it happened and how to repeat it;
- what you expected to see and what appeared;
- `ultima-partida-erros.log` from the install folder.

**Most useful right now:** tests on AMD and Intel cards, on weaker PCs and in areas beyond Kokiri village.

## Set up your environment

Follow the [step-by-step build guide](docs/build-by-hand.md): it leaves the tools, `.venv` and `build/` ready in `C:\OcarinaRecomp`. Then make your fork and point `origin` at it:

```bash
git remote rename origin upstream
git remote add origin https://github.com/YOUR-USER/ocarina-recomp
```

To rebuild, use a terminal with `C:\msys64\ucrt64\bin` on `PATH`:

```bash
# The runtime changed: incremental and quick
.venv/Scripts/python tools/build_game.py --version oot111 --gen generated/oot111 --out build/oot111 \
  --expected tools/data/expected-oot111.json --exefs build/exefs --romfs build/romfs

# The shader translator changed: run this, then the command above
.venv/Scripts/python tools/build_xlate.py

# The lifter (recomp/) changed: generate again, then run the first command
.venv/Scripts/python recomp/gen.py --exefs build/exefs --out generated/oot111 --modules main,sdk,subsdk0
.venv/Scripts/python tools/oot/patch_generated.py generated/oot111
```

To open the game, use `jogar\jogar-ocarina.ps1`.

## Where things are

| Folder | Contents |
|---|---|
| `recomp/` | ARM64 to C translator |
| `runtime/` | The "console": kernel, system, audio, input and NVN on Vulkan |
| `runtime/vk_exec_draw.c` | Runs draws and dispatches, including mesh |
| `runtime/vk_pipeline.c` | Pipeline creation |
| `runtime/ov_*.c` | Replacements for SDK functions (`HOST_OVERRIDE`) |
| `runtime/third_party/eglantine/.../ampere/` | Ampere to Maxwell shader lowering |
| `tools/xlate/patches/` | Shader translator patches. The expected count is in `tools/build_xlate.py` |
| `tools/oot/` | Parts specific to Ocarina |
| `instalador/`, `tools/oot/make_setup.py` | The windowed installer and the script that builds the ZIP |
| `jogar/` | The launcher and the initial settings |
| `verificar.ps1`, `compilar.ps1` | The build scripts. When you change a command, change the installer and the guides in `docs/` too |

The `TOTK_` and `DKB_` prefixes come from the source projects. They stay so the code remains easy to compare with those projects. Comments that mention other games record where a behavior was discovered.

## Code rules

- **Do not edit generated code by hand.** It gets recreated. Use `recomp/patches.py` or, if that does not fit, `tools/oot/patch_generated.py`.
- **`LOCAL TEST` marker:** changes made during investigation that have no final form yet. When you improve one, remove the marker.
- **Translator patches:** create them with a diff over `build/xlate/ryujinx` and update the count in `tools/build_xlate.py`. Changing the translator recompiles every shader on the next run.
- **Windows:** `long` is 32 bits. A `1L << 40` dropped every indirect mesh draw for days.
- **Copied code:** keep its header and add a row to `THIRD_PARTY_NOTICES.md` with the source commit.
- **Commits:** use your own git identity. Never push directly to `main`: open a pull request.

## How to investigate a graphics bug

What solved the biggest bugs so far was **measuring and comparing against a reference**, not guessing.

1. **Make a repeatable scene.** Use an input script (`TOTK_INPUT_SCRIPT`) with `TOTK_INPUT=none`, otherwise your keyboard moves Link while the test window has focus:
   ```text
   700 A 8
   2300 RSTICK 20000 0 150
   2600 LSTICK 0 32000 240
   ```
2. **Capture.** `ocarina.exe --seconds 120 --shot output.png`. With `TOTK_SHOTS=lo-hi:step`, it saves thumbnails along the way.
3. **Isolate.** Turn off one program at a time until the defect goes away, for example with `OOT_MESH_DENY=4049,4047`.
4. **Compare with Lichii.** With `EDEN_NX2_RENDER_TRACE=1`, it logs its draws, and you can compare them with ours.
5. **Read the shader.** Use `TOTK_DUMP=<frame>` and `TOTK_DUMP_DIR=<folder>`, decompile with `spirv-cross` and compare with the native code. An absurd value usually points at the cause: that is how the register bank collision was found.

### Diagnostic tools

| Variable | What it does |
|---|---|
| `OOT_DRAW_ARGS=<frame>` | Logs draw kinds, arguments, buffers per stage and per-program statistics |
| `OOT_MESH_DENY=<a,b,...>` | Drops indirect mesh draws of those programs |
| `OOT_NO_CULL=1`, `OOT_NO_CULL_PROGS=lo-hi` | Turns off face culling |
| `OOT_FOLLOW_DISP=<frame>:<k>:<c0>` | Follows a dispatch, together with `TOTK_DUMP_BUF=0:<size>` |
| `TOTK_XLATE_IR=<folder>` | The translator writes the blocks and IR of each shader with a subroutine |
| `TOTK_SPV_OVERRIDE=<folder>` | Replaces shaders with `sh_<hash>.spv` files. The hash changes when the translator changes, and the old file silently stops applying |
| `OOT_BIG_CS_WORDS`, `OOT_BIG_CS_OPT=1`, `OOT_BIG_CS_THREADS`, `OOT_BIG_CS_HUGE` | Control background compilation of large compute shaders: the size from which they go to the background, optimization, how many build at once (default 2 with 24 GB of RAM or more, else 1) and the size from which only one builds at a time (default 262144 words). The smallest queued one goes first |
| `TOTK_TRACE=perf` | Logs memory traffic per frame. Start here when the frame rate is low |

**When reading GPU buffers:** read after `VK_MEM_USE`. Before it, you see an old CPU copy.

## Good places to start

- Why the second pass through an area still builds new pipelines.
- Testing and fixing on AMD and Intel.
- Checking the door of Link's house, breaking pots and playing the ocarina.
- Checking the fog in Hyrule Field.
- Checking the New Game opening movie.
- Evaluating Lichii PRs #224, #353, #337, #215 and #345.
- Replacing the pointer candidate list with nxnative's raw pool table (`TryGenerateRawStorageOp`), which would shrink the giant compute shaders.

## Send a pull request

1. Create a branch from `main`.
2. Make one change per pull request.
3. Run `python tools/check_content.py`. If you changed the lifter, also run the tests in `tests/`.
4. Test in game. Say in the pull request which card and driver you used, in which scene and what changed. If you can measure it, even better: "fps from X to Y", or "program Z went from 0 to N triangles".
5. Say what you did **not** test. A "not tested on AMD" prevents a lot of surprises.

If you use AI to help, review the result and confirm it works before sending. See also the [code of conduct](CODE_OF_CONDUCT.md).
