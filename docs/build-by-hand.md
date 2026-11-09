# How to build

[Leia em português](compilar-manual.md)

The game is built on your PC from your copy. These are all the steps. After step 2, `compilar.ps1` can run steps 3 to 8 for you: see the [README](../README.en.md#how-to-build).

On Linux, `./compilar.sh --jogo <your game>` runs these steps with POSIX paths; see
[building.md](building.md#linux-in-progress) for the packages.

You need your own copy of the game at **version 1.1.1** and 60 GB free, plus about 15 GB on C: for the tools. We recommend 16 GB of RAM: we never tested with less, and the build or the game may fail. Use a folder with no spaces or accents. All commands are for **PowerShell**.

## 1. Install the tools

Skip what you already have:

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e
winget install --id MSYS2.MSYS2 -e
winget install --id Microsoft.DotNet.SDK.10 -e
winget install --id Microsoft.VisualStudio.BuildTools -e --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

If you already have Visual Studio, open the **Visual Studio Installer**, click **Modify** and select **Desktop development with C++**.

Close and reopen PowerShell. Then install the compiler. The first command may close MSYS2 midway through the update: that is normal, which is why it appears twice.

```powershell
$env:MSYSTEM = "UCRT64"; $env:CHERE_INVOKING = "1"
C:\msys64\usr\bin\bash.exe -lc "pacman -Syuu --noconfirm"
C:\msys64\usr\bin\bash.exe -lc "pacman -Syuu --noconfirm"
C:\msys64\usr\bin\bash.exe -lc "pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-{clang,lld,ninja,gcc,make,pkgconf,sdl3,opus,libvpx,dav1d,zstd,vulkan-devel,vulkan-memory-allocator,vulkan-validation-layers,shaderc,spirv-tools} base-devel git unzip"
```

## 2. Get the code

```powershell
git clone https://github.com/sostenesfreitas/ocarina-recomp C:\OcarinaRecomp
cd C:\OcarinaRecomp
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r tools\requirements-build.txt
```

**In every new PowerShell** from here on, start with:

```powershell
cd C:\OcarinaRecomp
$env:PATH = "C:\msys64\ucrt64\bin;C:\Program Files (x86)\Microsoft Visual Studio\Installer;$env:PATH"
$env:PYTHONUTF8 = "1"
```

## 3. Extract the game's executable

**If your game is a `.nup`**, list the packages inside it:

```powershell
.venv\Scripts\python.exe tools\extract_nup.py "C:\path\your-game.nup" --list
```

If the first line says `1 package(s)`, use `--package 0` in this guide's `extract_nup.py` commands. If there is more than one package, use `--overlay-update`.

```powershell
.venv\Scripts\python.exe tools\extract_nup.py "C:\path\your-game.nup" --out build\dump --only exefs --package 0
New-Item -ItemType Directory -Force build\exefs
Copy-Item build\dump\exefs\* build\exefs -Force
```

**If your game is a folder**, copy its `exefs` to `build\exefs`. If the path has brackets, rename the folder first, because PowerShell treats them as wildcards:

```powershell
New-Item -ItemType Directory -Force build\exefs
Copy-Item "C:\path\to\game\exefs\*" build\exefs -Force
```

## 4. Check the version and decompress if needed

```powershell
(Get-FileHash build\exefs\main).Hash
```

The right result is:

```text
F90499EEB596D67C730FAC9B7E247B3A46D51D6F334C767AEF071C9A783CCCEF
```

If it differs, `main` is probably compressed. Decompress it with [zbic](https://github.com/kinnay/zbic), replacing `<source>` with `build\dump\exefs` or your game's `exefs` folder:

```powershell
git clone https://github.com/kinnay/zbic build\zbic
git -C build\zbic checkout 11b08f2712264bbed731545085cbd9702096ceb7
clang -O2 -w -Ibuild\zbic\src tools\oot\unzbic.c -o build\unzbic.exe
build\unzbic.exe <source>\main build\exefs\main
(Get-FileHash build\exefs\main).Hash
```

If the hash still differs, this game is not version 1.1.1 and the rest will not work.

## 5. Build the shader translator

```powershell
.venv\Scripts\python.exe tools\build_xlate.py
```

## 6. Convert the game code

```powershell
.venv\Scripts\python.exe recomp\gen.py --exefs build\exefs --out generated\oot111 --modules main,sdk,subsdk0
.venv\Scripts\python.exe tools\oot\patch_generated.py generated\oot111
```

## 7. Prepare the game files

**With a `.nup`**, the extracted files and the image take about 26 GB each, but `--consume` deletes each extracted file as it goes into the image. Use it only on the copy extracted to `build\dump`:

If `--list` showed more than one package, replace `--package 0` with `--overlay-update` below too. With the base package alone the game still runs, but with the 1.0 shaders under the 1.1.1 code: an orange and purple picture with no vegetation. The log then says `RomFS: WARNING: this RomFS is the base game's`.

```powershell
.venv\Scripts\python.exe tools\extract_nup.py "C:\path\your-game.nup" --out build\dump --only romfs --package 0
.venv\Scripts\python.exe tools\pack_romfs.py --dir build\dump\romfs --out build\romfs --consume
Remove-Item build\dump -Recurse -Force
```

**With a folder**, do not use `--consume`, because it would delete your files:

```powershell
.venv\Scripts\python.exe tools\pack_romfs.py --dir "C:\path\to\game\romfs" --out build\romfs
```

## 8. Build the game

This is the longest part. With little RAM, use a lower `--jobs`: at most 1 for every 3 GB.

```powershell
.venv\Scripts\python.exe tools\build_game.py --version oot111 --gen generated\oot111 --out build\oot111 --expected tools\data\expected-oot111.json --exefs build\exefs --romfs build\romfs --jobs 8
Copy-Item jogar\totk.ini build\oot111\totk.ini
```

## 9. Play

```powershell
powershell -ExecutionPolicy Bypass -File jogar\jogar-ocarina.ps1
```

The first launch takes a while, and new areas may stutter a little while shaders are prepared.

## To update

```powershell
git pull
```

Then repeat steps 5, 6 and 8. Steps 3, 4 and 7 only need redoing if you delete `build\exefs` or `build\romfs`.

## Common problems

| Message | What to do |
|---|---|
| `clang` not found | Run the `$env:PATH` commands from step 2 in this PowerShell |
| `vswhere` or `link.exe` error in step 5 | Visual Studio's C++ tools are missing: see step 1 |
| `patch_generated: ... found 0 times` | The game is not version 1.1.1 |
| Out of memory in step 8 | Use a lower `--jobs` |

Do not share `ocarina.exe`, `generated\` or `build\`: they come from your game. To contribute, read the [contributing guide](../CONTRIBUTING.en.md).
