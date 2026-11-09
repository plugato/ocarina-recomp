# LOCAL TEST (Ocarina, AMD): abre o jogo com o perfil de GPU e grava os botões apertados.
# - perf: linhas "perf: frame" no log
# - OOT_GPU_PROF*: a cada 3600 frames (~1-6 min), o custo de GPU de cada pass, dispatch e draw
# - TOTK_INPUT_RECORD: grava a entrada para repetir o mesmo caminho depois (TOTK_INPUT_SCRIPT)
# O log fica em ultima-partida-erros.log; feche pelo X da janela para salvar o cache de pipelines.
$raiz = Split-Path $PSScriptRoot -Parent
$env:TOTK_TRACE = 'perf'
$env:OOT_GPU_PROF = '1200'
$env:OOT_GPU_PROF_EVERY = '3600'
$env:OOT_GPU_PROF_DRAWS = '1'
$env:TOTK_INPUT_RECORD = Join-Path $raiz "entrada-gravada.txt"
& (Join-Path $PSScriptRoot "jogar-ocarina.ps1") @args
