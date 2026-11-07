# Resume o desempenho de uma partida a partir do log, sem abrir o jogo.
#   powershell -ExecutionPolicy Bypass -File jogar\medir-partida.ps1                 a última partida
#   powershell -ExecutionPolicy Bypass -File jogar\medir-partida.ps1 -Log <arquivo>  um log qualquer
#   powershell -ExecutionPolicy Bypass -File jogar\medir-partida.ps1 -Comparar       todas as partidas guardadas em logs\
#   -Csv <arquivo> grava as janelas de 60 s (ou, com -Comparar, uma linha por partida) para abrir numa planilha.
# Lê as linhas "frametime:" que o jogo escreve a cada 60 s, as esperas por pipeline, os crashes e a quarentena.
param([string]$Log, [switch]$Comparar, [string]$Csv)

$raiz = Split-Path $PSScriptRoot -Parent
$inv = [Globalization.CultureInfo]::InvariantCulture
function Num($s) { return [double]::Parse($s, $inv) }

function Ler($caminho) {
    $r = [ordered]@{ Arquivo = $caminho; Janelas = @(); Piores = @(); Esperas = 0; EsperaMs = 0.0; Crash = @(); Diario = @() }
    foreach ($l in [IO.File]::ReadLines($caminho)) {
        if ($l -match 'frametime: (\d+) s: (\d+) frames in ([\d.]+) s, mean ([\d.]+) fps, 1% low ([\d.]+) fps, over 25 ms (\d+), 50 ms (\d+), 100 ms (\d+)') {
            $r.Janelas += [pscustomobject]@{ Janela = $r.Janelas.Count + 1; Frames = [int]$Matches[2]; Segundos = (Num $Matches[3])
                Fps = (Num $Matches[4]); Low1 = (Num $Matches[5]); Acima25 = [int]$Matches[6]; Acima50 = [int]$Matches[7]; Acima100 = [int]$Matches[8] }
        } elseif ($l -match 'worst frame (\d+): ([\d.]+) ms; shader translation ([\d.]+), pipeline build ([\d.]+)') {
            $r.Piores += [pscustomobject]@{ Frame = [int]$Matches[1]; Ms = (Num $Matches[2]); Shader = (Num $Matches[3]); Pipeline = (Num $Matches[4]) }
        } elseif ($l -match 'a draw waited ([\d.]+) ms for its pipeline') {
            $r.Esperas++; $r.EsperaMs += (Num $Matches[1])
        } elseif ($l -match 'FATAL:|^\s+while compiling|no pipeline was being compiled') {
            $r.Crash += $l.Trim()
        } elseif ($l -match 'pipe_journal:|in quarantine') {
            $r.Diario += ($l -replace '^\[host\] ', '').Trim()
        }
    }
    $frames = ($r.Janelas | Measure-Object Frames -Sum).Sum
    $seg = ($r.Janelas | Measure-Object Segundos -Sum).Sum
    $pior = $r.Piores | Sort-Object Ms -Descending | Select-Object -First 1
    [pscustomobject]@{
        Partida   = (Get-Item -LiteralPath $caminho).LastWriteTime.ToString('dd/MM HH:mm')
        Minutos   = [math]::Round($seg / 60, 1)
        FpsMedio  = if ($seg) { [math]::Round($frames / $seg, 1) } else { $null }
        Low1Pior  = ($r.Janelas | Measure-Object Low1 -Minimum).Minimum
        Acima50   = ($r.Janelas | Measure-Object Acima50 -Sum).Sum
        Acima100  = ($r.Janelas | Measure-Object Acima100 -Sum).Sum
        PiorMs    = if ($pior) { $pior.Ms } else { $null }
        PipelineS = [math]::Round((($r.Piores | Measure-Object Pipeline -Sum).Sum) / 1000, 1)
        ShaderS   = [math]::Round((($r.Piores | Measure-Object Shader -Sum).Sum) / 1000, 1)
        Esperas   = $r.Esperas
        Crash     = if ($r.Crash.Count) { 'sim' } else { '' }
        Dados     = $r
    }
}

if ($Comparar) {
    $arquivos = @()
    $pasta = Join-Path $raiz 'logs'
    if (Test-Path -LiteralPath $pasta) { $arquivos += Get-ChildItem -LiteralPath $pasta -Filter 'partida-*-erros.log' | Sort-Object Name | ForEach-Object { $_.FullName } }
    $atual = Join-Path $raiz 'ultima-partida-erros.log'
    if ((Test-Path -LiteralPath $atual) -and (Get-Item -LiteralPath $atual).Length) { $arquivos += $atual }
    if (-not $arquivos) { Write-Host 'Nenhum log encontrado.'; exit 1 }
    $linhas = $arquivos | ForEach-Object { Ler $_ } | Select-Object Partida, Minutos, FpsMedio, Low1Pior, Acima50, Acima100, PiorMs, PipelineS, ShaderS, Esperas, Crash
    $linhas | Format-Table -AutoSize
    Write-Host 'Low1Pior: o pior "1% low" entre as janelas de 60 s. PipelineS e ShaderS: segundos parados compilando, somados nos piores frames que o log lista.'
    if ($Csv) { $linhas | Export-Csv -LiteralPath $Csv -NoTypeInformation -Encoding UTF8 -Delimiter ';'; Write-Host "CSV: $Csv" }
    exit 0
}

if (-not $Log) { $Log = Join-Path $raiz 'ultima-partida-erros.log' }
if (-not (Test-Path -LiteralPath $Log)) { Write-Host "Não achei $Log"; exit 1 }
$s = Ler $Log
$d = $s.Dados
Write-Host "Partida: $Log ($($s.Partida), $($s.Minutos) min medidos)" -ForegroundColor Cyan
if (-not $d.Janelas.Count) { Write-Host 'O log não tem nenhuma janela de 60 s: a partida durou menos de um minuto.' }
else {
    Write-Host ("fps médio {0}   pior 1% low {1}   frames acima de 50 ms: {2}   acima de 100 ms: {3}" -f $s.FpsMedio, $s.Low1Pior, $s.Acima50, $s.Acima100)
    $p = $d.Piores | Sort-Object Ms -Descending | Select-Object -First 1
    if ($p) { Write-Host ("Pior travada: {0} ms no frame {1} ({2} ms compilando pipelines, {3} ms traduzindo shaders)" -f $p.Ms, $p.Frame, $p.Pipeline, $p.Shader) }
    Write-Host ("Nos piores frames listados: {0} s compilando pipelines, {1} s traduzindo shaders" -f $s.PipelineS, $s.ShaderS)
    Write-Host ("Draws que esperaram o pipeline: {0}, somando {1} s" -f $d.Esperas, [math]::Round($d.EsperaMs / 1000, 1))
    $d.Janelas | Format-Table Janela, Fps, Low1, Acima25, Acima50, Acima100 -AutoSize
    if ($Csv) { $d.Janelas | Export-Csv -LiteralPath $Csv -NoTypeInformation -Encoding UTF8 -Delimiter ';'; Write-Host "CSV: $Csv" }
}
if ($d.Crash.Count) { Write-Host 'Crash:' -ForegroundColor Red; $d.Crash | ForEach-Object { Write-Host "  $_" } }
if ($d.Diario.Count) { Write-Host 'Diário de compilação e quarentena:' -ForegroundColor Yellow; $d.Diario | Select-Object -Unique | ForEach-Object { Write-Host "  $_" } }
