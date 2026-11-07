# Abre o Ocarina Recomp.
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
if (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' }) { $env:TOTK_GPU_NAME = "NVIDIA" }
$raiz = Split-Path $PSScriptRoot -Parent
$env:TOTK_PROFILE_DIR = Join-Path $raiz "perfil"
# The previous run's logs go to logs\ (the newest 10 runs) before this one overwrites them: a crash log used to be lost
# as soon as the game was opened again. jogar\medir-partida.ps1 reads them.
$saida = Join-Path $raiz "ultima-partida.log"
$erros = Join-Path $raiz "ultima-partida-erros.log"
if ((Test-Path -LiteralPath $erros) -and (Get-Item -LiteralPath $erros).Length -gt 0) {
    try {
        $logs = Join-Path $raiz "logs"
        New-Item -ItemType Directory -Force -Path $logs | Out-Null
        $quando = (Get-Item -LiteralPath $erros).LastWriteTime.ToString('yyyy-MM-dd_HH-mm-ss')
        Move-Item -LiteralPath $erros -Destination (Join-Path $logs "partida-$quando-erros.log") -Force -ErrorAction Stop
        if (Test-Path -LiteralPath $saida) { Move-Item -LiteralPath $saida -Destination (Join-Path $logs "partida-$quando.log") -Force }
        Get-ChildItem -LiteralPath $logs -Filter 'partida-*-erros.log' | Sort-Object Name -Descending | Select-Object -Skip 10 | ForEach-Object {
            Remove-Item -LiteralPath $_.FullName -Force
            $par = $_.FullName -replace '-erros\.log$', '.log'
            if (Test-Path -LiteralPath $par) { Remove-Item -LiteralPath $par -Force }
        }
    } catch {
        # the game is still open and holds the log: leave it, Start-Process below says so
    }
}
Set-Location (Join-Path $raiz "build\oot111")
$extras = if ($args.Count) { @{ ArgumentList = $args } } else { @{} }
Start-Process .\ocarina.exe @extras -RedirectStandardOutput $saida -RedirectStandardError $erros
