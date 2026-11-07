#!/usr/bin/env bash
# Builds Ocarina Recomp on Linux from your own copy of the game: the commands of docs/building.md, shown before they run.
#
#   ./compilar.sh --jogo /path/game.nup|folder [--simular] [--sem-perguntar] [--jobs N]
#
# --simular       shows every command and runs none of them
# --sem-perguntar does not ask before each step
#
# It needs no root and installs nothing system-wide. It writes only inside this folder and never changes your game:
# --consume below only runs on the copy extracted to build/dump. It goes online only for the Python packages (uv/pip),
# the shader translator's .NET packages (build_xlate.py) and, when main is compressed, kinnay/zbic.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

case "${LC_ALL:-${LANG:-}}" in
    pt*) PT=1 ;;
    *) PT=0 ;;
esac
L() { if [ "$PT" = 1 ]; then printf '%s' "$1"; else printf '%s' "$2"; fi; }

JOGO="" SIMULAR=0 PERGUNTAR=1 JOBS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --jogo|-j) JOGO="${2:-}"; shift 2 ;;
        --simular|-n) SIMULAR=1; shift ;;
        --sem-perguntar|-y) PERGUNTAR=0; shift ;;
        --jobs) JOBS="${2:-}"; shift 2 ;;
        -h|--help) awk 'NR > 1 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done
if [ -z "$JOGO" ]; then
    echo "missing --jogo (your .nup or a folder with exefs and romfs)" >&2
    exit 2
fi
[ "$SIMULAR" = 1 ] && echo "=== SIMULATION: nothing will run ==="

# A step: its title and what it does, then a question unless --sem-perguntar or --simular.
etapa() {
    printf '\n[%s/7] %s\n  %s\n' "$1" "$2" "$3"
    if [ "$SIMULAR" = 0 ] && [ "$PERGUNTAR" = 1 ]; then
        read -r -p "  Continue? (Y/N) " r
        case "$r" in [sSyY]*) ;; *) echo "Stopped. Run again to continue from here."; exit 0 ;; esac
    fi
}
run() {
    printf '  > %s\n' "$*"
    [ "$SIMULAR" = 1 ] && return 0
    "$@"
}
# Runs one program, or only shows it with --simular. A failure stops the build.
pulo() { printf '  %s\n' "$*"; }

# ---------------------------------------------------------------- preflight (looks only)
faltam=0
precisa() {
    if command -v "$1" >/dev/null 2>&1; then
        printf '  [ok]    %s\n' "$1"
    else
        printf '  [falta] %s  (%s)\n' "$1" "$2"
        faltam=$((faltam + 1))
    fi
}
pkg() {
    if pkg-config --exists "$1" 2>/dev/null; then
        printf '  [ok]    %s %s\n' "$1" "$(pkg-config --modversion "$1")"
    else
        printf '  [falta] %s  (%s)\n' "$1" "$2"
        faltam=$((faltam + 1))
    fi
}
echo
echo "$(L 'Ocarina Recomp: verificando este PC (nada será alterado)' 'Ocarina Recomp: checking this PC (nothing will be changed)')"
echo
precisa git 'install git'
precisa clang 'install clang'
precisa ld.lld 'install lld'
precisa ninja 'install ninja'
precisa pkg-config 'install pkgconf'
precisa sha256sum 'coreutils'
if command -v python3.12 >/dev/null 2>&1 || command -v uv >/dev/null 2>&1; then
    printf '  [ok]    python 3.12 (or uv)\n'
else
    printf '  [falta] python 3.12 or uv  (install python3.12 or astral-sh/uv)\n'
    faltam=$((faltam + 1))
fi
if command -v dotnet >/dev/null 2>&1 && dotnet --list-sdks 2>/dev/null | grep -q '^10\.'; then
    printf '  [ok]    .NET SDK 10\n'
else
    printf '  [falta] .NET SDK 10  (the shader translator; https://dot.net)\n'
    faltam=$((faltam + 1))
fi
pkg sdl3 'install sdl3'
pkg opus 'install opus'
pkg dav1d 'install dav1d'
pkg vpx 'install libvpx'
if [ -f /usr/include/zstd.h ]; then
    printf '  [ok]    zstd (header)\n'
else
    printf '  [falta] zstd  (install zstd)\n'
    faltam=$((faltam + 1))
fi
# The loader's vulkan.pc is not the header: on Arch and CachyOS they are separate packages (issue #41)
vk_inc=$(pkg-config --variable=includedir vulkan 2>/dev/null)
if [ -f /usr/include/vulkan/vulkan.h ] || { [ -n "$vk_inc" ] && [ -f "$vk_inc/vulkan/vulkan.h" ]; }; then
    printf '  [ok]    vulkan (header)\n'
else
    printf '  [falta] vulkan/vulkan.h  (install vulkan-headers)\n'
    faltam=$((faltam + 1))
fi
if [ "$faltam" -gt 0 ]; then
    echo
    echo "$(L "Faltam $faltam item(ns). Instale o que está acima e rode de novo." "$faltam item(s) missing. Install what is above and run this again.")" >&2
    [ "$SIMULAR" = 1 ] || exit 1
fi
echo

# A folder with just exefs and romfs, or a folder holding a base/ tree (a dump).
eh_nup=0
case "$JOGO" in *.nup) eh_nup=1 ;; esac
base="$JOGO"
[ -f "$JOGO/base/exefs/main" ] && base="$JOGO/base"
if [ "$eh_nup" = 0 ] && [ ! -f "$base/exefs/main" ]; then
    echo "not a .nup and no exefs/main under $base" >&2
    exit 1
fi
if [ "$eh_nup" = 1 ] && [ ! -f "$JOGO" ]; then
    echo "game not found: $JOGO" >&2
    exit 1
fi

# ---------------------------------------------------------------- 1. Python
etapa 1 "$(L 'Preparar o Python' 'Prepare Python')" \
    "$(L 'Cria .venv e instala tools/requirements-build.txt. Baixa da internet.' 'Creates .venv and installs tools/requirements-build.txt. Downloads from the internet.')"
if [ ! -x .venv/bin/python ]; then
    if command -v uv >/dev/null 2>&1; then
        run uv venv --python 3.12 .venv
    else
        run python3.12 -m venv .venv
    fi
fi
if command -v uv >/dev/null 2>&1; then
    run uv pip install --python .venv/bin/python -q -r tools/requirements-build.txt
else
    run .venv/bin/python -m pip install -q --disable-pip-version-check -r tools/requirements-build.txt
fi
PY=.venv/bin/python

# ---------------------------------------------------------------- 2. ExeFS
etapa 2 "$(L 'Extrair e conferir o executável do jogo' "Extract and check the game's executable")" \
    "$(L 'Copia o exefs para build/exefs e confere a versão 1.1.1 pelo hash. Se o main estiver comprimido, baixa o zbic.' 'Copies the exefs to build/exefs and checks version 1.1.1 by its hash. If main is compressed, downloads zbic.')"
hash_ok() {
    [ -f build/exefs/main ] || return 1
    local certo atual
    certo=$($PY -c 'import json; print(json.load(open("tools/data/expected-oot111.json"))["exefs_sha256"]["main"])')
    atual=$(sha256sum build/exefs/main | cut -d' ' -f1)
    [ "$atual" = "$certo" ]
}
opcoes=(--package 0)
if [ "$eh_nup" = 1 ] && [ "$SIMULAR" = 0 ] \
    && ! "$PY" tools/extract_nup.py "$JOGO" --list | grep -q ': 1 package'; then
    opcoes=(--overlay-update)
fi
if [ "$SIMULAR" = 0 ] && hash_ok; then
    pulo "$(L 'Já feito: build/exefs é a versão 1.1.1.' 'Already done: build/exefs is version 1.1.1.')"
else
    origem="$base/exefs"
    if [ "$eh_nup" = 1 ]; then
        run "$PY" tools/extract_nup.py "$JOGO" --out build/dump --only exefs --quiet "${opcoes[@]}"
        origem=build/dump/exefs
    fi
    printf '  > copy %s/* -> build/exefs\n' "$origem"
    if [ "$SIMULAR" = 0 ]; then
        mkdir -p build/exefs
        cp -f "$origem"/* build/exefs/
    fi
    if [ "$SIMULAR" = 1 ] || ! hash_ok; then
        [ "$SIMULAR" = 1 ] && pulo "$(L '(só se o main estiver comprimido:)' '(only if main is compressed:)')"
        if [ ! -d build/zbic/.git ]; then
            run git clone https://github.com/kinnay/zbic build/zbic
        fi
        run git -C build/zbic checkout -q 11b08f2712264bbed731545085cbd9702096ceb7
        run clang -O2 -w -Ibuild/zbic/src tools/oot/unzbic.c -o build/unzbic
        run build/unzbic "$origem/main" build/exefs/main
        if [ "$SIMULAR" = 0 ] && ! hash_ok; then
            echo "$(L 'Este jogo não é a versão 1.1.1, a única que funciona por enquanto.' 'This game is not version 1.1.1, the only one that works for now.')" >&2
            exit 1
        fi
    fi
    if [ "$eh_nup" = 1 ] && [ "$SIMULAR" = 0 ]; then rm -rf build/dump; fi
fi

# ---------------------------------------------------------------- 3. Shader translator
etapa 3 "$(L 'Compilar o tradutor de shaders' 'Build the shader translator')" \
    "$(L 'Compila o tradutor em build/xlate. Baixa pacotes .NET da internet.' 'Builds the translator in build/xlate. Downloads .NET packages from the internet.')"
run "$PY" tools/build_xlate.py

# ---------------------------------------------------------------- 4. Lift
etapa 4 "$(L 'Converter o código do jogo' 'Convert the game code')" \
    "$(L 'Traduz o código do jogo para C em generated/oot111.' 'Translates the game code to C in generated/oot111.')"
run "$PY" recomp/gen.py --exefs build/exefs --out generated/oot111 --modules main,sdk,subsdk0
run "$PY" tools/oot/patch_generated.py generated/oot111

# ---------------------------------------------------------------- 5. RomFS
etapa 5 "$(L 'Preparar os arquivos do jogo' 'Prepare the game files')" \
    "$(L 'Monta a imagem de 26 GB em build/romfs. O seu arquivo original não é alterado.' 'Builds the 26 GB image in build/romfs. Your original file is not changed.')"
if [ "$SIMULAR" = 0 ] && [ -f build/romfs/index.json ]; then
    pulo "$(L 'Já feito: build/romfs está pronto.' 'Already done: build/romfs is ready.')"
elif [ "$eh_nup" = 1 ]; then
    run "$PY" tools/extract_nup.py "$JOGO" --out build/dump --only romfs "${opcoes[@]}"
    # --consume deletes each loose file once it is in the image: only ever on the copy extracted just above
    run "$PY" tools/pack_romfs.py --dir build/dump/romfs --out build/romfs --consume
    [ "$SIMULAR" = 0 ] && rm -rf build/dump || true
else
    run "$PY" tools/pack_romfs.py --dir "$base/romfs" --out build/romfs
fi

# ---------------------------------------------------------------- 6. Build
etapa 6 "$(L 'Compilar o jogo' 'Build the game')" \
    "$(L 'A parte mais demorada. Pode levar mais de uma hora e deixar o PC lento.' 'The longest part. It can take over an hour and slow your PC down.')"
args=(--version oot111 --gen generated/oot111 --out build/oot111 --expected tools/data/expected-oot111.json \
      --exefs build/exefs --romfs build/romfs)
[ "$JOBS" -gt 0 ] && args+=(--jobs "$JOBS")
run "$PY" tools/build_game.py "${args[@]}"

# ---------------------------------------------------------------- 7. Settings
etapa 7 "$(L 'Configuração inicial' 'Initial settings')" \
    "$(L 'Copia jogar/totk.ini, se ainda não houver uma configuração.' 'Copies jogar/totk.ini, if there are no settings yet.')"
printf '  > copy jogar/totk.ini -> build/oot111/totk.ini\n'
if [ "$SIMULAR" = 0 ] && [ ! -f build/oot111/totk.ini ]; then
    mkdir -p build/oot111
    cp -f jogar/totk.ini build/oot111/totk.ini
fi

echo
if [ "$SIMULAR" = 1 ]; then
    echo "$(L 'Fim da simulação. Nada foi executado.' 'End of simulation. Nothing was run.')"
else
    echo "$(L 'Pronto! Para jogar:' 'Done! To play:')"
    echo "  ./jogar/jogar-ocarina.sh"
    echo "$(L 'Não compartilhe o ocarina, generated/ nem build/: eles saem do seu jogo.' 'Do not share ocarina, generated/ or build/: they come from your game.')"
fi
