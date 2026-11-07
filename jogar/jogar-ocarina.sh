#!/usr/bin/env bash
# Opens Ocarina Recomp on Linux. Build it first with ./compilar.sh --jogo <your game>.
#
# From a terminal it shows the output; when started from a desktop shortcut it writes ultima-partida.log and
# ultima-partida-erros.log next to the project, like the Windows launcher.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The same switches as jogar-ocarina.ps1. TOTK_GPL=0: bindless set 2 + an allocated vertex set in an independent
# layout loses the device. TOTK_ASYNC_PREEMPT=0: the old scheduling exactly. DKB_SKIP_CS: the game's upscaler, giant
# shaders that take many minutes to compile.
export TOTK_GPL=0
export TOTK_ASYNC_PREEMPT=0
export DKB_SKIP_CS="3bdfe449de832903,909139c13b510f1f,753f8ab880fe7923,c968ad44272dcbb4,785a41d2df6dc095,db8c39bdb420e068,dec1feb3c7fa967a,09931d0c57c6b67f,13cab66ee383c4f4,3f6f2b1d671badcd"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
    export TOTK_GPU_NAME=NVIDIA
fi
export TOTK_PROFILE_DIR="$root/perfil"

# A fullscreen window on a tiling Wayland compositor (niri, sway) takes the only output and the keyboard, and the
# player cannot reach the chat or the terminal to send a command. Start windowed unless asked otherwise: F11 toggles
# fullscreen, and TOTK_FULLSCREEN=1 ./jogar/jogar-ocarina.sh starts fullscreen. A Steam Deck keeps its profile.
if [ -z "${TOTK_FULLSCREEN:-}" ] && [ -z "${SteamDeck:-}" ]; then
    export TOTK_FULLSCREEN=0
fi

if [ ! -x "$root/build/oot111/ocarina" ]; then
    echo "build/oot111/ocarina not found: run ./compilar.sh --jogo <your game> first" >&2
    exit 1
fi
cd "$root/build/oot111"
if [ -t 1 ]; then
    exec ./ocarina "$@"
else
    # The previous run's logs go to logs/ (the newest 10 runs), as in jogar-ocarina.ps1.
    if [ -s "$root/ultima-partida-erros.log" ]; then
        mkdir -p "$root/logs"
        when="$(date -r "$root/ultima-partida-erros.log" +%Y-%m-%d_%H-%M-%S)"
        mv -f "$root/ultima-partida-erros.log" "$root/logs/partida-$when-erros.log"
        [ -f "$root/ultima-partida.log" ] && mv -f "$root/ultima-partida.log" "$root/logs/partida-$when.log"
        ls -1 "$root/logs"/partida-*-erros.log | sort -r | tail -n +11 | while read -r old; do
            rm -f "$old" "${old%-erros.log}.log"
        done
    fi
    exec ./ocarina "$@" >"$root/ultima-partida.log" 2>"$root/ultima-partida-erros.log"
fi
