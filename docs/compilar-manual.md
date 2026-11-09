# Como compilar

[Read in English](build-by-hand.md)

O jogo é compilado no seu PC, a partir da sua cópia. Estes são todos os passos. Depois do passo 2, o `compilar.ps1` pode rodar os passos 3 a 8 por você: veja o [README](../README.md#como-compilar).

No Linux, `./compilar.sh --jogo <seu jogo>` faz estes passos com caminhos POSIX; veja
[building.md](building.md#linux-in-progress) para os pacotes.

Você precisa da sua própria cópia do jogo na **versão 1.1.1** e de 60 GB livres, mais uns 15 GB no disco C: para as ferramentas. Recomendamos 16 GB de RAM: com menos, nunca testamos, e a compilação ou o jogo podem falhar. Use uma pasta sem espaços nem acentos. Todos os comandos são para o **PowerShell**.

## 1. Instale as ferramentas

Pule o que você já tiver:

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e
winget install --id MSYS2.MSYS2 -e
winget install --id Microsoft.DotNet.SDK.10 -e
winget install --id Microsoft.VisualStudio.BuildTools -e --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Se você já tem o Visual Studio, abra o **Visual Studio Installer**, clique em **Modificar** e marque **Desenvolvimento para desktop com C++**.

Feche e abra o PowerShell. Depois instale o compilador. O primeiro comando pode fechar o MSYS2 no meio da atualização: é normal, e por isso ele aparece duas vezes.

```powershell
$env:MSYSTEM = "UCRT64"; $env:CHERE_INVOKING = "1"
C:\msys64\usr\bin\bash.exe -lc "pacman -Syuu --noconfirm"
C:\msys64\usr\bin\bash.exe -lc "pacman -Syuu --noconfirm"
C:\msys64\usr\bin\bash.exe -lc "pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-{clang,lld,ninja,gcc,make,pkgconf,sdl3,opus,libvpx,dav1d,zstd,vulkan-devel,vulkan-memory-allocator,vulkan-validation-layers,shaderc,spirv-tools} base-devel git unzip"
```

## 2. Baixe o código

```powershell
git clone https://github.com/sostenesfreitas/ocarina-recomp C:\OcarinaRecomp
cd C:\OcarinaRecomp
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r tools\requirements-build.txt
```

**Em todo PowerShell novo**, a partir daqui, comece por:

```powershell
cd C:\OcarinaRecomp
$env:PATH = "C:\msys64\ucrt64\bin;C:\Program Files (x86)\Microsoft Visual Studio\Installer;$env:PATH"
$env:PYTHONUTF8 = "1"
```

## 3. Extraia o executável do jogo

**Se o seu jogo é um `.nup`**, veja os pacotes dentro dele:

```powershell
.venv\Scripts\python.exe tools\extract_nup.py "C:\caminho\seu-jogo.nup" --list
```

Se a primeira linha disser `1 package(s)`, use `--package 0` nos comandos de `extract_nup.py` deste guia. Se houver mais de um pacote, use `--overlay-update`.

```powershell
.venv\Scripts\python.exe tools\extract_nup.py "C:\caminho\seu-jogo.nup" --out build\dump --only exefs --package 0
New-Item -ItemType Directory -Force build\exefs
Copy-Item build\dump\exefs\* build\exefs -Force
```

**Se o seu jogo é uma pasta**, copie o `exefs` dela para `build\exefs`. Se o caminho tiver colchetes, renomeie a pasta antes, porque o PowerShell os trata como curinga:

```powershell
New-Item -ItemType Directory -Force build\exefs
Copy-Item "C:\caminho\do\jogo\exefs\*" build\exefs -Force
```

## 4. Confira a versão e descomprima, se precisar

```powershell
(Get-FileHash build\exefs\main).Hash
```

O resultado certo é:

```text
F90499EEB596D67C730FAC9B7E247B3A46D51D6F334C767AEF071C9A783CCCEF
```

Se for diferente, o `main` provavelmente está comprimido. Descomprima com o [zbic](https://github.com/kinnay/zbic), trocando `<origem>` por `build\dump\exefs` ou pela pasta `exefs` do seu jogo:

```powershell
git clone https://github.com/kinnay/zbic build\zbic
git -C build\zbic checkout 11b08f2712264bbed731545085cbd9702096ceb7
clang -O2 -w -Ibuild\zbic\src tools\oot\unzbic.c -o build\unzbic.exe
build\unzbic.exe <origem>\main build\exefs\main
(Get-FileHash build\exefs\main).Hash
```

Se o hash continuar diferente, este jogo não é a versão 1.1.1, e o resto não vai funcionar.

## 5. Compile o tradutor de shaders

```powershell
.venv\Scripts\python.exe tools\build_xlate.py
```

## 6. Converta o código do jogo

```powershell
.venv\Scripts\python.exe recomp\gen.py --exefs build\exefs --out generated\oot111 --modules main,sdk,subsdk0
.venv\Scripts\python.exe tools\oot\patch_generated.py generated\oot111
```

## 7. Prepare os arquivos do jogo

**Com um `.nup`**, a extração e a imagem ocupam cerca de 26 GB cada, mas o `--consume` apaga cada arquivo extraído ao colocá-lo na imagem. Use-o só na cópia extraída em `build\dump`:

Se o `--list` mostrou mais de um pacote, troque `--package 0` por `--overlay-update` aqui também. Só com o pacote base o jogo ainda roda, mas com os shaders da 1.0 sob o código 1.1.1: imagem laranja e roxa e sem vegetação. O log avisa com `RomFS: WARNING: this RomFS is the base game's`.

```powershell
.venv\Scripts\python.exe tools\extract_nup.py "C:\caminho\seu-jogo.nup" --out build\dump --only romfs --package 0
.venv\Scripts\python.exe tools\pack_romfs.py --dir build\dump\romfs --out build\romfs --consume
Remove-Item build\dump -Recurse -Force
```

**Com uma pasta**, não use `--consume`, porque ele apagaria os seus arquivos:

```powershell
.venv\Scripts\python.exe tools\pack_romfs.py --dir "C:\caminho\do\jogo\romfs" --out build\romfs
```

## 8. Compile o jogo

É a parte mais demorada. Com pouca RAM, use um `--jobs` menor: no máximo 1 para cada 3 GB.

```powershell
.venv\Scripts\python.exe tools\build_game.py --version oot111 --gen generated\oot111 --out build\oot111 --expected tools\data\expected-oot111.json --exefs build\exefs --romfs build\romfs --jobs 8
Copy-Item jogar\totk.ini build\oot111\totk.ini
```

## 9. Jogue

```powershell
powershell -ExecutionPolicy Bypass -File jogar\jogar-ocarina.ps1
```

A primeira abertura demora, e áreas novas podem dar umas travadinhas enquanto os shaders são preparados.

## Para atualizar

```powershell
git pull
```

Depois repita os passos 5, 6 e 8. Os passos 3, 4 e 7 só precisam ser refeitos se você apagar `build\exefs` ou `build\romfs`.

## Problemas comuns

| Mensagem | O que fazer |
|---|---|
| `clang` não encontrado | Rode os comandos de `$env:PATH` do passo 2 neste PowerShell |
| Erro de `vswhere` ou de `link.exe` no passo 5 | Faltam as ferramentas de C++ do Visual Studio: veja o passo 1 |
| `patch_generated: ... found 0 times` | O jogo não é a versão 1.1.1 |
| Falta de memória no passo 8 | Use um `--jobs` menor |

Não compartilhe o `ocarina.exe`, o `generated\` nem o `build\`: eles saem do seu jogo. Para contribuir, leia o [guia de contribuição](../CONTRIBUTING.md).
