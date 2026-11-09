# Prepara os shaders que o jogo já encontrou e que ainda não estão no disco, com o jogo fechado.
#
# Por quê: na RX 5500 XT (16 GB de RAM), alguns shaders do Ocarina levam de 1 a 7 minutos e vários GB de RAM para
# compilar no driver da AMD. Com o jogo aberto não sobra RAM para isso (o PC reiniciou assim em 2026-11-07). Então o
# jogo deixa esses shaders para depois e pula os desenhos deles naquela partida. Aqui o ocarina.exe só compila
# (TOTK_PREWARM_ONLY=1, sem abrir o jogo) e guarda o resultado em build\oot111\pipeline-binaries. Depois disso, o jogo
# carrega cada um em milissegundos.
#
# O jogar-ocarina.ps1 roda isto antes de abrir o jogo, toda vez. Sem nada na fila leva 2 a 4 s e nenhuma janela
# aparece. Com algo na fila, abre uma janela com o progresso e o log ao vivo. Fechar a janela pula o preparo desta vez
# (o que já ficou pronto fica guardado). Também pode ser rodado sozinho: -SempreMostrar abre a janela desde o início.
param([string]$Raiz = (Split-Path $PSScriptRoot -Parent), [switch]$SempreMostrar)

if (Get-Process ocarina -ErrorAction SilentlyContinue) { return }   # o jogo está aberto: não mexe nos arquivos dele
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
$pasta = Join-Path $Raiz "build\oot111"
$saida = Join-Path $Raiz "preparo-shaders.log"
$erros = Join-Path $Raiz "preparo-shaders-erros.log"
$env:TOTK_PREWARM_ONLY = "1"
$prep = $null
try {
    $prep = Start-Process (Join-Path $pasta "ocarina.exe") -WorkingDirectory $pasta -PassThru `
        -RedirectStandardOutput $saida -RedirectStandardError $erros
} catch { }
Remove-Item Env:TOTK_PREWARM_ONLY -ErrorAction SilentlyContinue
if (-not $prep) { return }

Add-Type -AssemblyName Microsoft.VisualBasic
$pc = New-Object Microsoft.VisualBasic.Devices.ComputerInfo
$inicio = Get-Date
$form = $null; $topo = $null; $caixa = $null
$lido = 0L; $resto = ""
$fase = ""; $prontos = 0; $total = 0; $grandes = 0; $compilando = 0; $ultimoAguardo = [datetime]::MinValue; $erro = $false
$fila = New-Object System.Collections.Generic.List[string]   # linhas do log antes de a janela abrir

function Tempo([double]$s) {
    if ($s -ge 60) { "{0} min {1:00} s" -f [int][math]::Floor($s / 60), [int]($s % 60) } else { "{0:N0} s" -f $s }
}
function Palavras([string]$n) { "{0:N0}" -f [int64]$n }
function Anota([string]$t) {
    $linha = "[{0:HH:mm:ss}] {1}" -f (Get-Date), $t
    if ($caixa) { $caixa.AppendText($linha + "`r`n") } else { $fila.Add($linha) }
}
function AbreJanela {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    $script:form = New-Object System.Windows.Forms.Form
    $form.Text = "Ocarina Recomp - preparando shaders"
    $form.Width = 720; $form.Height = 460
    $form.StartPosition = "CenterScreen"
    $script:topo = New-Object System.Windows.Forms.Label
    $topo.Dock = "Top"; $topo.Height = 96
    $topo.Padding = New-Object System.Windows.Forms.Padding(12, 10, 12, 0)
    $topo.Font = New-Object System.Drawing.Font("Segoe UI", 10)
    $script:caixa = New-Object System.Windows.Forms.TextBox
    $caixa.Multiline = $true; $caixa.ReadOnly = $true; $caixa.ScrollBars = "Vertical"; $caixa.Dock = "Fill"
    $caixa.Font = New-Object System.Drawing.Font("Consolas", 9.5)
    $rodape = New-Object System.Windows.Forms.Label
    $rodape.Dock = "Bottom"; $rodape.Height = 40
    $rodape.Padding = New-Object System.Windows.Forms.Padding(12, 4, 12, 4)
    $rodape.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    $rodape.Text = "Os shaders grandes levam de 1 a 7 minutos cada; vários compilam juntos enquanto a RAM permitir. " +
        "O jogo abre sozinho no fim. Fechar esta janela pula o preparo desta vez (o que já ficou pronto fica guardado)."
    $form.Controls.Add($caixa); $form.Controls.Add($topo); $form.Controls.Add($rodape)
    $form.Show()
    foreach ($l in $fila) { $caixa.AppendText($l + "`r`n") }
    $fila.Clear()
}

while ($true) {
    $acabou = $prep.HasExited
    # linhas novas do log do ocarina.exe desde a última leitura
    try {
        $fs = [IO.File]::Open($erros, 'Open', 'Read', 'ReadWrite')
        if ($fs.Length -gt $lido) {
            [void]$fs.Seek($lido, 'Begin')
            $sr = New-Object IO.StreamReader($fs)
            $novo = $sr.ReadToEnd()
            $lido = $fs.Length
            $partes = @(($resto + $novo) -split "`r?`n")
            $resto = $partes[-1]
            $completas = if ($partes.Count -gt 1) { $partes[0..($partes.Count - 2)] } else { @() }
            foreach ($l in $completas) {
                if ($l -match 'prewarm: \d+ records in .*: (\d+) pipelines built, (\d+) failed') {
                    Anota ("Pipelines gráficas: {0} prontas{1}." -f $Matches[1], $(if ([int]$Matches[2]) { ", $($Matches[2]) falharam" } else { "" }))
                } elseif ($l -match 'prewarm: (\d+) of (\d+) (pipelines|big compute shaders)') {
                    $prontos = [int]$Matches[1]; $total = [int]$Matches[2]
                    $f = if ($Matches[3] -eq 'pipelines') { "pipelines gráficas" } else { "compute shaders grandes" }
                    if ($f -ne $fase) { $fase = $f; Anota "Fase: $fase" }
                } elseif ($l -match 'compiling a big (graphics pipeline|compute shader): (\d+) words') {
                    $grandes++; $compilando++
                    $t = if ($Matches[1] -eq 'compute shader') { "compute shader" } else { "shader gráfico" }
                    Anota ("Compilando {0} grande nº {1} ({2} palavras)..." -f $t, $grandes, (Palavras $Matches[2]))
                } elseif ($l -match 'slow graphics pipeline: ([\d.]+) s, (\d+) words') {
                    $compilando = [math]::Max(0, $compilando - 1)
                    Anota ("Pronto: shader gráfico de {0} palavras em {1}." -f (Palavras $Matches[2]), (Tempo ([double]$Matches[1])))
                } elseif ($l -match 'compute pipeline binary (kept|NOT kept) after a ([\d.]+) s compile, (\d+) words') {
                    $compilando = [math]::Max(0, $compilando - 1)
                    $ok = if ($Matches[1] -eq 'kept') { "" } else { " (não foi possível guardar)" }
                    Anota ("Pronto: compute shader de {0} palavras em {1}{2}." -f (Palavras $Matches[3]), (Tempo ([double]$Matches[2])), $ok)
                } elseif ($l -match 'compile held: (\d+) MB of RAM free.*this one is big') {
                    if (((Get-Date) - $ultimoAguardo).TotalSeconds -ge 60) {
                        $ultimoAguardo = Get-Date
                        Anota ("Aguardando RAM livre para começar o próximo shader grande ({0:N1} GB livres)." -f ([int]$Matches[1] / 1024))
                    }
                } elseif ($l -match 'big compute shaders ready \((\d+) from their binary, (\d+) failed\)') {
                    Anota ("Compute shaders grandes: todos prontos{0}." -f $(if ([int]$Matches[2]) { ", $($Matches[2]) falharam" } else { "" }))
                } elseif ($l -match 'prewarm only: done') {
                    Anota "Preparo concluído."
                } elseif ($l -match 'graphics pipeline skipped: the driver crashed (\d+) time') {
                    Anota ("Pulado: um shader que já derrubou o driver da AMD {0} vez(es) ao compilar." -f $Matches[1])
                } elseif ($l -match 'unhandled exception|FATAL') {
                    $erro = $true
                    Anota "O driver da AMD fechou o preparo ao compilar um shader. Esse shader fica de fora nas próximas vezes; o resto continua na próxima abertura."
                }
            }
            $sr.Close()
        } else { $fs.Close() }
    } catch { }

    $seg = ((Get-Date) - $inicio).TotalSeconds
    if (-not $form -and -not $acabou -and ($SempreMostrar -or $grandes -gt 0 -or $seg -ge 8)) { AbreJanela }
    if ($form) {
        if ($form.IsDisposed) {   # fechou a janela: pula o preparo e segue para o jogo
            Stop-Process -Id $prep.Id -Force -ErrorAction SilentlyContinue
            break
        }
        $livre = $pc.AvailablePhysicalMemory / 1GB
        $prog = if ($total) { "$fase`: $prontos de $total" } else { "carregando a lista de shaders" }
        $topo.Text = "Preparando shaders novos antes de abrir o jogo.`n" +
            "$prog   |   compilando agora: $compilando   |   shaders grandes até aqui: $grandes`n" +
            ("RAM livre: {0:N1} GB   |   tempo: {1}" -f $livre, (Tempo $seg))
        [System.Windows.Forms.Application]::DoEvents()
    }
    if ($acabou) { break }
    Start-Sleep -Milliseconds 400
}
if ($form -and -not $form.IsDisposed) {
    if ($erro) { Anota "Preparo interrompido pelo driver. Abrindo o jogo..." }
    else { Anota ("Tudo pronto em {0}. Abrindo o jogo..." -f (Tempo ((Get-Date) - $inicio).TotalSeconds)) }
    [System.Windows.Forms.Application]::DoEvents()
    Start-Sleep -Seconds $(if ($erro) { 6 } else { 2 })
    $form.Close()
}
