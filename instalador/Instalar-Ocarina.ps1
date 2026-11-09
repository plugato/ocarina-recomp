# Instalador do Ocarina Recomp: uma janela, sem comandos, sem administrador e sem instalar nada no Windows. Usa as ferramentas que vem
# no ZIP e compila o jogo a partir do arquivo da propria pessoa. Tudo fica na pasta de instalacao: para desinstalar, basta apaga-la.
# Rodar de novo continua de onde parou. Testes: OCARINA_PASTA e OCARINA_JOGO evitam os dialogos e OCARINA_AUTO=1 inicia sozinho.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
[Windows.Forms.Application]::EnableVisualStyles()

$SETUP = $env:OCARINA_SETUP   # the folder of Instalar-Ocarina.exe, set by instalador/launcher.c
$HASH_MAIN = 'F90499EEB596D67C730FAC9B7E247B3A46D51D6F334C767AEF071C9A783CCCEF'   # o main da versao 1.1.1, sem compressao
$GB_NECESSARIOS = 40
$LOG = Join-Path $env:TEMP 'ocarina-instalador.log'
$TIGELA = [char]::ConvertFromUtf32(0x1F35C)

$script:emPT = (Get-UICulture).TwoLetterISOLanguageName -eq 'pt'   # not $PT: names are case-insensitive, so it would be the $pt parameter of L
# while the window is built, L keeps both texts joined by a control character and Idioma picks one
$SEP = [char]1
$script:marcar = $true
function L($pt, $en) {
    if ($script:marcar) { return "$pt$SEP$en" }
    if ($script:emPT) { return $pt } return $en
}
function Pick([string]$m) { $p = $m.Split($SEP); if ($script:emPT) { return $p[0] } return $p[1] }

$script:cancelar = $false
$script:rodando = $false
$script:passoIni = [DateTime]::Now
$script:estimativa = 1
$script:feito = 0.0
$script:pesoAtual = 1.0
$script:proc = $null

function Dicas() { return @(
    (L 'Um miojo fica pronto em 3 minutos. Isto aqui leva um pouco mais.' 'Instant noodles take 3 minutes. This takes a little longer.'),
    (L 'A barra quase parada não é travamento: enquanto o relógio acima andar, está tudo funcionando.' 'A nearly still bar is not a freeze: while the clock above moves, everything is working.'),
    (L 'Pode usar o PC enquanto instala, só vai ficar um pouco mais lento.' 'You can use your PC during the install, it will just be a little slower.'),
    (L 'Não precisa olhar: quando terminar, o botão Jogar aparece aqui.' 'No need to watch: the Play button shows up here when it is done.'),
    (L 'Se fechar ou faltar luz, é só abrir o instalador de novo: ele continua de onde parou.' 'If it closes or the power goes out, just open the installer again: it picks up where it stopped.'),
    (L 'O jogo não vem daqui: ele é montado no seu PC, a partir do seu próprio arquivo.' 'The game does not come from here: it is built on your PC from your own copy.'),
    (L 'Dá tempo de fazer uns vinte miojos. Com tempero, por favor.' 'There is time to cook about twenty bowls of noodles. Seasoning included, please.'),
    (L 'Na primeira vez jogando, áreas novas podem dar umas travadinhas. Depois melhora.' 'The first time you play, new areas may stutter a little. It gets better after that.')
) }

# ---- a janela
function Rotulo($texto, $x, $y, $l, $a, $tam = 10, $negrito = $false) {
    $c = New-Object Windows.Forms.Label
    $c.Text = $texto; $c.Left = $x; $c.Top = $y; $c.Width = $l; $c.Height = $a
    $estilo = if ($negrito) { [Drawing.FontStyle]::Bold } else { [Drawing.FontStyle]::Regular }
    $c.Font = New-Object Drawing.Font('Segoe UI', $tam, $estilo)
    return $c
}
function Botao($texto, $x, $y, $l, $a = 36) {
    $c = New-Object Windows.Forms.Button
    $c.Text = $texto; $c.Left = $x; $c.Top = $y; $c.Width = $l; $c.Height = $a
    $c.Font = New-Object Drawing.Font('Segoe UI', 10)
    return $c
}

$form = New-Object Windows.Forms.Form
$form.Text = (L 'Ocarina Recomp - Instalador' 'Ocarina Recomp - Installer')
$form.ClientSize = New-Object Drawing.Size(640, 560)
$form.FormBorderStyle = 'FixedSingle'; $form.MaximizeBox = $false; $form.StartPosition = 'CenterScreen'
$form.KeyPreview = $true

$titulo = Rotulo 'Ocarina Recomp' 24 16 480 40 20 $true
$sub = Rotulo (L 'Ocarina of Time nativo no Windows, sem emulador. Experimental.' 'Ocarina of Time running natively on Windows, no emulator. Experimental.') 26 56 590 24
$btIdioma = Botao '' 516 18 100 30
$btIdioma.Font = New-Object Drawing.Font('Segoe UI', 9)
$form.Controls.AddRange(@($titulo, $sub, $btIdioma))

# Tela 1: escolher o jogo
$tela1 = New-Object Windows.Forms.Panel
$tela1.SetBounds(0, 86, 640, 430)
$t1a = Rotulo (L "O que este instalador faz:`n  1. Copia as ferramentas que vêm no ZIP para a pasta de instalação. Não instala nada no Windows.`n  2. Monta o jogo no seu PC a partir do SEU arquivo do jogo.`n  3. Cria um atalho na Área de Trabalho.`n`nLeva de 30 minutos a 2 horas, e você pode usar o PC enquanto isso. Para desinstalar, apague a pasta." "What this installer does:`n  1. Copies the tools from the ZIP to the install folder. It installs nothing in Windows.`n  2. Builds the game on your PC from YOUR copy of the game.`n  3. Creates a shortcut on your desktop.`n`nIt takes 30 minutes to 2 hours, and you can use your PC in the meantime. To uninstall, delete the folder.") 26 0 590 130
$t1b = Rotulo (L 'Passo 1: mostre onde está o seu jogo (versão 1.1.1)' 'Step 1: show where your game is (version 1.1.1)') 26 136 590 24 11 $true
$btArq = Botao (L 'Escolher arquivo .nup' 'Choose .nup file') 26 164 280
$btPasta = Botao (L 'Escolher pasta com exefs e romfs' 'Choose folder with exefs and romfs') 322 164 292
$lbJogo = Rotulo '' 26 206 590 42
$t1c = Rotulo (L 'Passo 2: confira onde instalar' 'Step 2: check where to install') 26 252 590 24 11 $true
$lbPasta = Rotulo '' 26 280 440 24
$btTrocar = Botao (L 'Trocar pasta' 'Change folder') 482 274 132 32
$lbAviso = Rotulo '' 26 314 590 44
$lbAviso.ForeColor = [Drawing.Color]::DarkOrange
$btInstalar = Botao (L 'Instalar' 'Install') 214 366 212 48
$btInstalar.Font = New-Object Drawing.Font('Segoe UI', 12, [Drawing.FontStyle]::Bold)
$btInstalar.Enabled = $false
$tela1.Controls.AddRange(@($t1a, $t1b, $btArq, $btPasta, $lbJogo, $t1c, $lbPasta, $btTrocar, $lbAviso, $btInstalar))

# Tela 2: progresso
$tela2 = New-Object Windows.Forms.Panel
$tela2.SetBounds(0, 86, 640, 430)
$tela2.Visible = $false
$lista = New-Object Windows.Forms.ListBox
$lista.SetBounds(26, 0, 588, 210)
$lista.Font = New-Object Drawing.Font('Segoe UI', 10)
$lista.SelectionMode = 'None'
$barra = New-Object Windows.Forms.ProgressBar
$barra.SetBounds(26, 220, 588, 26)
$barra.Maximum = 1000
$lbPct = Rotulo '' 26 250 588 22 10 $true
$lbLinha = Rotulo '' 26 274 588 40 8
$lbLinha.ForeColor = [Drawing.Color]::Gray
$lbDica = Rotulo '' 26 318 588 44 10
$btJogar = Botao (L 'Jogar' 'Play') 26 370 180 44
$btJogar.Font = New-Object Drawing.Font('Segoe UI', 12, [Drawing.FontStyle]::Bold)
$btJogar.Visible = $false
$btImportar = Botao (L 'Importar save de emulador' 'Import emulator save') 222 370 228 44
$btImportar.Visible = $false
$btFechar = Botao (L 'Fechar' 'Close') 466 370 148 44
$btFechar.Visible = $false
$tela2.Controls.AddRange(@($lista, $barra, $lbPct, $lbLinha, $lbDica, $btJogar, $btImportar, $btFechar))
$form.Controls.AddRange(@($tela1, $tela2))

$rodape = Rotulo (L "Feito com $TIGELA por Miojo" "Made with $TIGELA by Miojo") 24 526 300 26 9
$rodape.Font = New-Object Drawing.Font('Segoe UI Emoji', 9)
$rodape.Cursor = [Windows.Forms.Cursors]::Hand
$btLog = New-Object Windows.Forms.LinkLabel
$btLog.Text = (L 'Ver registro da instalação' 'View install log'); $btLog.SetBounds(446, 528, 180, 22); $btLog.TextAlign = 'MiddleRight'
$form.Controls.AddRange(@($rodape, $btLog))
# Notepad can be missing from Windows: show the file in Explorer then. An error here would take down the window.
function AbrirRegistro() { try { Start-Process notepad.exe $LOG } catch { Start-Process "$env:windir\explorer.exe" "/select,`"$LOG`"" } }
$btLog.add_LinkClicked({ if (Test-Path $LOG) { AbrirRegistro } })

# ---- o idioma: o botao do canto troca entre portugues e ingles, os dois textos ficam no Tag
function Idioma([Windows.Forms.Control]$c) {
    if ($c.Text.Contains($SEP)) { $c.Tag = $c.Text }
    if ($c.Tag -is [string] -and $c.Tag.Contains($SEP)) { $c.Text = Pick $c.Tag }
    foreach ($f in $c.Controls) { Idioma $f }
}
# texts that change while running are redone from these states
$script:jogoEstado = 'nada'; $script:segredo = $false; $script:fim = $null; $script:passoAtual = 0
function TextoJogo() {
    switch ($script:jogoEstado) {
        'nada' { $lbJogo.Text = (L 'Nada escolhido ainda.' 'Nothing chosen yet.'); $lbJogo.ForeColor = [Drawing.Color]::Empty }
        'pronta' { $lbJogo.Text = (L 'Esta pasta já tem os arquivos do jogo convertidos: não precisa escolher o jogo de novo.' 'This folder already has the converted game files: no need to choose the game again.'); $lbJogo.ForeColor = [Drawing.Color]::DarkGreen }
        'jogo' { $lbJogo.Text = (L "Jogo: $($script:jogo)" "Game: $($script:jogo)"); $lbJogo.ForeColor = [Drawing.Color]::DarkGreen }
        'invalido' { $lbJogo.Text = (L 'Essa pasta não tem as subpastas exefs e romfs do jogo. Escolha a pasta certa.' 'This folder does not have the game''s exefs and romfs subfolders. Choose the right folder.'); $lbJogo.ForeColor = [Drawing.Color]::Firebrick }
    }
}
function TextoSegredo() {
    $titulo.Text = "Ocarina Recomp $TIGELA"
    $sub.Text = (L 'Edição especial do Miojo: 3 minutos de fogo e várias horas de recompilação.' 'Miojo special edition: 3 minutes on the stove and many hours of recompiling.')
}
function TextoFim() {
    if ($script:fim -eq 'ok') {
        $lbPct.Text = (L 'Pronto! 100%' 'Done! 100%')
        $lbDica.Text = (L "Conecte um controle e bom jogo!`nNa primeira vez, a vegetação pode levar alguns minutos para aparecer e áreas novas podem travar um pouco: é normal. Não compartilhe o jogo montado, os saves nem o cache: eles saem do seu jogo." "Plug in a controller and have fun!`nThe first time, plants can take a few minutes to appear and new areas may stutter a little: that is normal. Do not share the built game, the saves or the cache: they come from your game.")
    } elseif ($script:fim -eq 'erro') {
        $lbPct.Text = (L 'A instalação parou.' 'The installation stopped.')
    }
}
function RefrescarLista() {
    $p = Passos
    for ($i = 0; $i -lt $p.Count -and $i -lt $lista.Items.Count; $i++) {
        $lista.Items[$i] = if ($i -lt $script:passoAtual) { " OK  $($p[$i][0])" } elseif ($i -eq $script:passoAtual) { " >  $($p[$i][0])..." } else { "      $($p[$i][0])" }
    }
}
function AplicarIdioma() {
    Idioma $form
    $btIdioma.Text = if ($script:emPT) { 'English' } else { 'Português' }
    if ($script:segredo) { TextoSegredo }
    Validar
    if ($lista.Items.Count) { RefrescarLista }
    if ($script:fim) { TextoFim }
    if ($script:fim -ne 'ok' -and -not $script:rodando) { $lbDica.Text = (Dicas)[0] }
}
$btIdioma.add_Click({ $script:emPT = -not $script:emPT; AplicarIdioma })

# ---- o segredo do Miojo: digitar "miojo" na janela, ou clicar 5 vezes na assinatura
$script:digitado = ''
$script:cliques = 0
function Segredo() {
    $script:segredo = $true; TextoSegredo
    [Windows.Forms.MessageBox]::Show((L "Você achou o miojo escondido!`n`nMiojo manda um salve para quem lê até o rodapé.`nTem outro escondido dentro do jogo: tente digitar o mesmo nome lá." "You found the hidden noodles!`n`nMiojo says hi to everyone who reads the footer.`nThere is another one inside the game: try typing the same name there."), (L "$TIGELA Segredo do Miojo" "$TIGELA Miojo's secret")) | Out-Null
}
$form.add_KeyPress({
    param($s, $e)
    $script:digitado = ($script:digitado + [string]$e.KeyChar).ToLower()
    if ($script:digitado.Length -gt 5) { $script:digitado = $script:digitado.Substring($script:digitado.Length - 5) }
    if ($script:digitado -eq 'miojo') { $script:digitado = ''; Segredo }
})
$rodape.add_Click({ $script:cliques++; if ($script:cliques -eq 5) { $script:cliques = 0; Segredo } })

# ---- mostrar o andamento e rodar programas sem janela preta
# The log may be held for a moment by a program that just wrote to it: losing one line must not stop the install.
function Registrar($t) {
    foreach ($tentativa in 1..20) {
        try { [IO.File]::AppendAllText($LOG, "[$(Get-Date -Format 'HH:mm:ss')] $t`r`n"); return } catch { Start-Sleep -Milliseconds 100 }
    }
}

function UltimaLinha() {
    try {
        $fs = [IO.File]::Open($LOG, 'Open', 'Read', 'ReadWrite,Delete')
        $n = [Math]::Min(4096, $fs.Length)
        $fs.Seek(-$n, 'End') | Out-Null
        $b = New-Object byte[] $n
        $fs.Read($b, 0, $n) | Out-Null
        $fs.Close()
        $linhas = [Text.Encoding]::UTF8.GetString($b) -split "[`r`n]+" | Where-Object { $_.Trim() }
        if ($linhas) { return ($linhas | Select-Object -Last 1) }
    } catch {}
    return ''
}

function Atualizar() {
    $l = UltimaLinha
    $texto = $l -replace '[^\x20-\x7E]', ''
    $lbLinha.Text = if ($texto.Length -gt 180) { $texto.Substring(0, 180) } else { $texto }
    # andamento de verdade quando o programa informa ("[12/660]" ou "20000/186587 files"), senao pelo tempo estimado
    $frac = [Math]::Min(0.95, ([DateTime]::Now - $script:passoIni).TotalMinutes / $script:estimativa)
    if ($l -match '\[(\d+)/(\d+)\]|(\d+)/(\d+) files') {
        $a = if ($Matches[1]) { [double]$Matches[1] } else { [double]$Matches[3] }
        $b = if ($Matches[2]) { [double]$Matches[2] } else { [double]$Matches[4] }
        if ($b -gt 0 -and $a -le $b) { $frac = $a / $b }
    }
    $total = $script:feito + $script:pesoAtual * $frac
    $barra.Value = [Math]::Min(1000, [int](1000 * $total))
    $minutos = [int][Math]::Floor(([DateTime]::Now - $script:passoIni).TotalMinutes)
    $segundos = ([DateTime]::Now - $script:passoIni).Seconds
    $lbPct.Text = (L '{0:0}% concluído. Nesta etapa há {1} min {2:00} s.' '{0:0}% done. On this step for {1} min {2:00} s.') -f (100 * $total), $minutos, $segundos
    $dica = [int]([long]([DateTime]::Now.Ticks / 250000000) % (Dicas).Count)
    $lbDica.Text = (Dicas)[$dica]
    [Windows.Forms.Application]::DoEvents()
}

Add-Type -Namespace Ocarina -Name Espera -MemberDefinition '[DllImport("kernel32.dll")] public static extern void Sleep(uint ms);'
function Exec([string]$exe, [string[]]$argumentos, [switch]$PodeFalhar, [string]$Saida) {
    # the full path: cmd.exe does not have to search PATH for it
    if ($exe -notmatch '\\') {
        $achado = Get-Command $exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($achado) { $exe = $achado.Source }
    }
    $q = ($argumentos | ForEach-Object { if ($_ -match '[\s&()^]') { '"' + $_ + '"' } else { $_ } }) -join ' '
    $destino = if ($Saida) { "> `"$Saida`"" } else { ">> `"$LOG`"" }
    Registrar "> $exe $q"
    $psi = New-Object Diagnostics.ProcessStartInfo
    # ComSpec can be missing from the environment (issue #102: "nao foi fornecido um nome de arquivo")
    $psi.FileName = if ($env:ComSpec -and (Test-Path -LiteralPath $env:ComSpec)) { $env:ComSpec } else { Join-Path $env:SystemRoot 'System32\cmd.exe' }
    $psi.Arguments = "/d /s /c `"`"$exe`" $q $destino 2>&1`""
    $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true
    if (Test-Path $script:pasta) { $psi.WorkingDirectory = $script:pasta }
    $script:proc = [Diagnostics.Process]::Start($psi)
    while (-not $script:proc.HasExited) {
        Atualizar
        if ($script:cancelar) {
            & taskkill.exe /T /F /PID $script:proc.Id 2>&1 | Out-Null
            throw 'CANCELADO'
        }
        # kernel32 Sleep, not Start-Sleep: a managed wait on this STA thread pumps messages itself and loses window commands,
        # so a window could not be minimized or restored during a step
        foreach ($n in 1..3) { [Ocarina.Espera]::Sleep(50); [Windows.Forms.Application]::DoEvents() }
    }
    $cod = $script:proc.ExitCode
    if ($cod -ne 0 -and -not $PodeFalhar) { throw (L "O programa '$([IO.Path]::GetFileName($exe))' falhou (código $cod)." "The program '$([IO.Path]::GetFileName($exe))' failed (code $cod).") }
    return $cod
}

function Ferramenta($sub) { return Join-Path $script:pasta "ferramentas\$sub" }
# The bundled tools and the Windows system folders, nothing from the PC: a long PATH or another compiler cannot get in the way.
function AtualizarPath() {
    $win = $env:SystemRoot
    $env:PATH = "$(Ferramenta 'bin');$(Ferramenta 'llvm\bin');$win\System32;$win;$win\System32\Wbem;$win\System32\WindowsPowerShell\v1.0"
    $env:PYTHONUTF8 = '1'; $env:PYTHONUNBUFFERED = '1'   # progress reaches the log as it happens, not in big blocks
}
# ---- os passos: nome, minutos estimados e o que fazem
function P-Copiar() {
    # Ninja compares dates: every ZIP gives runtime\cpu.h a new one, and an update recompiled all 660 objects of the game
    # (about an hour on a slow PC) with no header changed. A runtime file the update leaves byte-identical gets its old
    # date back; a changed one gets now, since the ZIP's date can be older than the last build.
    $rt = Join-Path $script:pasta 'runtime'
    $antes = @{}
    if (Test-Path -LiteralPath $rt) {
        Get-ChildItem -LiteralPath $rt -Recurse -File | ForEach-Object { $antes[$_.FullName] = @((Get-FileHash -LiteralPath $_.FullName -Algorithm MD5).Hash, $_.LastWriteTimeUtc) }
    }
    # robocopy: fewer than 8 is success. /MIR on the project would delete the game's build output, so it only adds and replaces.
    foreach ($par in @(@('projeto', $script:pasta), @('ferramentas', (Join-Path $script:pasta 'ferramentas')))) {
        $cod = Exec 'robocopy.exe' @((Join-Path $SETUP $par[0]), $par[1], '/E', '/NFL', '/NDL', '/NJH', '/NJS', '/NP', '/R:2', '/W:2') -PodeFalhar
        if ($cod -ge 8) { throw (L "Não consegui copiar os arquivos para $($script:pasta)." "Could not copy the files to $($script:pasta).") }
    }
    $agora = [DateTime]::UtcNow
    foreach ($k in $antes.Keys) {
        if (-not (Test-Path -LiteralPath $k)) { continue }
        $f = Get-Item -LiteralPath $k
        $f.LastWriteTimeUtc = if ((Get-FileHash -LiteralPath $k -Algorithm MD5).Hash -eq $antes[$k][0]) { $antes[$k][1] } else { $agora }
    }
    AtualizarPath
    $script:py = Ferramenta 'python\python.exe'
    # the translator goes where the game looks for it, as tools/build_xlate.py leaves it
    New-Item -ItemType Directory -Force (Build 'xlate') | Out-Null
    foreach ($f in 'totk_xlate.exe', 'totk_xlate.id') { Copy-Item -LiteralPath (Ferramenta "bin\$f") -Destination (Build "xlate\$f") -Force }
}
function Build($sub) { return Join-Path $script:pasta "build\$sub" }

function OpcoesNup() {
    $saida = Build 'lista-nup.txt'
    New-Item -ItemType Directory -Force (Join-Path $script:pasta 'build') | Out-Null
    Exec $script:py @('tools\extract_nup.py', $script:jogo, '--list') -Saida $saida | Out-Null
    $script:nupOpcoes = if ((Get-Content $saida -Raw) -match ': 1 package') { @('--package', '0') } else { @('--overlay-update') }
}

function P-Exefs() {
    $exefs = Build 'exefs'
    if ((Test-Path "$exefs\main") -and (Get-FileHash "$exefs\main").Hash -eq $HASH_MAIN) { return }
    if (-not $script:jogo) { throw 'VERSAO' }
    $origem = $null
    if ($script:jogoNup) {
        OpcoesNup
        Exec $script:py (@('tools\extract_nup.py', $script:jogo, '--out', (Build 'dump'), '--only', 'exefs', '--quiet') + $script:nupOpcoes) | Out-Null
        $origem = Build 'dump\exefs'
    } else {
        $origem = Join-Path $script:jogo 'exefs'
    }
    New-Item -ItemType Directory -Force $exefs | Out-Null
    Get-ChildItem -LiteralPath $origem -File | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $exefs $_.Name) -Force }
    if ((Get-FileHash "$exefs\main").Hash -ne $HASH_MAIN) {
        # o main costuma vir comprimido com ZBIC
        Exec (Ferramenta 'bin\unzbic.exe') @("$origem\main", "$exefs\main") -PodeFalhar | Out-Null
        if ((Get-FileHash "$exefs\main").Hash -ne $HASH_MAIN) { throw 'VERSAO' }
    }
    if ($script:jogoNup) { Remove-Item (Build 'dump') -Recurse -Force }
}
function P-Gerar() {
    Exec $script:py @('recomp\gen.py', '--exefs', (Build 'exefs'), '--out', 'generated\oot111', '--modules', 'main,sdk,subsdk0') | Out-Null
    Exec $script:py @('tools\oot\patch_generated.py', 'generated\oot111') | Out-Null
}
function P-Romfs() {
    if (Test-Path (Build 'romfs\index.json')) { return }
    if ($script:jogoNup) {
        OpcoesNup
        Exec $script:py (@('tools\extract_nup.py', $script:jogo, '--out', (Build 'dump'), '--only', 'romfs') + $script:nupOpcoes) | Out-Null
        # --consume apaga cada arquivo solto ao coloca-lo na imagem: so na copia extraida aqui, nunca na pasta da pessoa
        Exec $script:py @('tools\pack_romfs.py', '--dir', (Build 'dump\romfs'), '--out', (Build 'romfs'), '--consume') | Out-Null
        Remove-Item (Build 'dump') -Recurse -Force
    } else {
        Exec $script:py @('tools\pack_romfs.py', '--dir', (Join-Path $script:jogo 'romfs'), '--out', (Build 'romfs')) | Out-Null
    }
}
function P-Exe() {
    # A new version recompiles the runtime: the copied sources keep the ZIP's dates, which can be older than the last build's
    # objects, and Ninja then kept a runtime that did not match the new generated code (FATAL: cfg: ... no row).
    $versao = (Get-Content -LiteralPath (Join-Path $script:pasta 'VERSION') -Raw).Trim()
    $marca = Build 'oot111\versao-compilada.txt'
    $antes = if (Test-Path -LiteralPath $marca) { (Get-Content -LiteralPath $marca -Raw).Trim() } else { '' }
    if ($antes -ne $versao) {
        $objRt = Build 'oot111\obj\runtime'
        if (Test-Path -LiteralPath $objRt) { Registrar "versão ${antes} -> ${versao}: o runtime é recompilado"; Remove-Item -LiteralPath $objRt -Recurse -Force }
    }
    $ramGb = [Math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    $jobs = [Math]::Max(2, [Math]::Min([Environment]::ProcessorCount, [int]($ramGb / 3)))
    Exec $script:py @('tools\build_game.py', '--version', 'oot111', '--gen', 'generated\oot111', '--out', (Build 'oot111'), '--expected', 'tools\data\expected-oot111.json', '--exefs', (Build 'exefs'), '--romfs', (Build 'romfs'), '--jobs', "$jobs", '--toolchain', (Ferramenta 'llvm'), '--deps', (Ferramenta 'deps'), '--ninja', (Ferramenta 'bin\ninja.exe')) | Out-Null
    Set-Content -LiteralPath $marca -Value $versao -Encoding ascii
}
# Shader preparation: the player's own shaders, extracted from the game files, lowered and translated for the variants
# our seed lists (tools/data/oot-seed, answers and pipeline keys recorded in play), so a first start does not stop to
# build them. Not needed to play: if it fails, the game still works and builds them as before.
function P-Shaders() {
    $seed = Join-Path $script:pasta 'tools\data\oot-seed'
    $dir = Build 'oot111'
    if (-not (Test-Path (Join-Path $seed 'seed-answers.bin'))) { Registrar 'shaders: sem semente, passo pulado'; return }
    # Prepared again when the version or the translator changes: the lowering (runtime) and the translator make the
    # shaders, so the earlier preparation no longer matches and the first start would build them in play again.
    $marca = Join-Path $dir 'shader-prepared.txt'
    $idx = Join-Path $script:pasta 'build\xlate\totk_xlate.id'
    $chave = (Get-Content -LiteralPath (Join-Path $script:pasta 'VERSION') -Raw).Trim() + ' ' + $(if (Test-Path -LiteralPath $idx) { (Get-Content -LiteralPath $idx -Raw).Trim() } else { '' })
    $feita = if (Test-Path -LiteralPath $marca) { (Get-Content -LiteralPath $marca -Raw).Trim() } else { '' }
    if ((Test-Path (Join-Path $dir 'shader-prepared.bin')) -and $feita -eq $chave) { return }
    $pack = Build 'shader-stages.pack'
    $exe = Join-Path $dir 'ocarina.exe'
    try {
        Exec $script:py @('-I', 'tools\oot\shader_extract.py', '--romfs-image', (Build 'romfs\romfs.bin'), '--index', (Build 'romfs\index.json'), '--pack', $pack) | Out-Null
        Exec $exe @('--lower-stages', $pack, $dir) | Out-Null
        Copy-Item -LiteralPath (Join-Path $seed 'seed-answers.bin') -Destination (Join-Path $dir 'seed-answers.bin') -Force
        # the player's own keys, from earlier play, are kept and the seed's are added to them: an update's seed names
        # pipelines (the vegetation's) that a player's earlier keys do not, and they were built in play then
        $chaves = Join-Path $dir 'pipeline-keys.bin'
        if (Test-Path -LiteralPath $chaves) { Exec $script:py @('-I', 'tools\oot\merge_records.py', $chaves, $chaves, (Join-Path $seed 'pipeline-keys.bin')) | Out-Null }
        else { Copy-Item -LiteralPath (Join-Path $seed 'pipeline-keys.bin') -Destination $chaves }
        Exec $exe @('--prepare-shaders') | Out-Null
        Set-Content -LiteralPath $marca -Value $chave -Encoding ascii
    } catch {
        if ("$($_.Exception.Message)" -eq 'CANCELADO') { throw }
        Registrar "shaders: a preparação falhou, o jogo monta os shaders ao jogar: $($_.Exception.Message)"
    } finally {
        foreach ($f in $pack, (Join-Path $dir 'seed-programs.bin')) { if (Test-Path -LiteralPath $f) { Remove-Item -LiteralPath $f -Force } }
    }
}
function P-Finalizar() {
    $ini = Build 'oot111\totk.ini'
    if (-not (Test-Path $ini)) { Copy-Item (Join-Path $script:pasta 'jogar\totk.ini') $ini }
    if ($env:OCARINA_AUTO) { return }
    $ws = New-Object -ComObject WScript.Shell
    $atalho = $ws.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) 'Ocarina Recomp.lnk'))
    $atalho.TargetPath = 'powershell.exe'
    $atalho.Arguments = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script:pasta\jogar\jogar-ocarina.ps1`""
    $atalho.WorkingDirectory = $script:pasta
    $atalho.IconLocation = (Build 'oot111\ocarina.exe')
    $atalho.Save()
}

function Passos() { return @(
    @((L 'Copiar as ferramentas' 'Copy the tools'), 3, 'P-Copiar'),
    @((L 'Conferir o seu jogo' 'Check your game'), 2, 'P-Exefs'),
    @((L 'Converter o código do jogo' 'Convert the game code'), 2, 'P-Gerar'),
    @((L 'Preparar os arquivos do jogo (pode levar mais de 1 hora em HD)' 'Prepare the game files (can take over 1 hour on a hard drive)'), 25, 'P-Romfs'),
    @((L 'Compilar o jogo (a parte mais demorada)' 'Build the game (the longest part)'), 35, 'P-Exe'),
    @((L 'Preparar os shaders (evita travadas na primeira vez)' 'Prepare the shaders (avoids stutter the first time)'), 3, 'P-Shaders'),
    @((L 'Criar o atalho' 'Create the shortcut'), 1, 'P-Finalizar')
) }

function Explicar($erro) {
    $trecho = ''
    try { $trecho = (Get-Content $LOG -Tail 40 -ErrorAction SilentlyContinue) -join "`n" } catch {}
    if ($erro -eq 'VERSAO') { return (L "Este jogo não é a versão 1.1.1, a única que funciona por enquanto.`n`nConfira o arquivo escolhido e tente de novo." "This game is not version 1.1.1, the only one that works for now.`n`nCheck the file you chose and try again.") }
    if ($trecho -match 'No space left|not enough space|espa.o insuficiente|Disk full') { return (L "O disco ficou cheio.`n`nLibere espaço e abra este instalador de novo: ele continua de onde parou." "The disk is full.`n`nFree some space and open this installer again: it picks up where it stopped.") }
    if ($trecho -match 'Could not resolve|Failed to connect|timed out|unable to access|Connection') { return (L "Parece ser um problema de internet.`n`nConfira a conexão e abra este instalador de novo: ele continua de onde parou." "This looks like an internet problem.`n`nCheck your connection and open this installer again: it picks up where it stopped.") }
    return (L "$erro`n`nAbra o instalador de novo: muitas vezes ele resolve sozinho. Se repetir, abra um pedido de ajuda no GitHub do projeto com o registro da instalação." "$erro`n`nOpen the installer again: this often fixes it. If it happens again, open an issue on the project's GitHub with the install log.")
}

function FimDoRegistro() {
    try {
        $linhas = Get-Content -LiteralPath $LOG -Tail 8 -ErrorAction Stop | Where-Object { $_.Trim() } | ForEach-Object { if ($_.Length -gt 110) { $_.Substring(0, 110) } else { $_ } }
        return "`n`n" + (L 'Últimas linhas do registro:' 'Last lines of the log:') + "`n" + ($linhas -join "`n")
    } catch { return '' }
}

function Instalar() {
    $tela1.Visible = $false; $tela2.Visible = $true
    $script:rodando = $true
    $PASSOS = Passos
    $total = ($PASSOS | ForEach-Object { $_[1] } | Measure-Object -Sum).Sum
    foreach ($p in $PASSOS) { [void]$lista.Items.Add('') }
    $script:passoAtual = 0; $script:fim = $null; RefrescarLista
    Set-Content -Path $LOG -Value "Ocarina Recomp, instalador, $(Get-Date)" -Encoding UTF8
    try {
        for ($i = 0; $i -lt $PASSOS.Count; $i++) {
            $p = $PASSOS[$i]
            $script:passoAtual = $i; RefrescarLista
            $script:passoIni = [DateTime]::Now; $script:estimativa = $p[1]; $script:pesoAtual = $p[1] / $total
            Registrar "=== $($p[0])"
            $script:passoNome = $p[0]
            if ($i -ge 3 -or (Test-Path $script:pasta)) { if (Test-Path $script:pasta) { Set-Location $script:pasta } }
            & $p[2]
            $script:feito += $script:pesoAtual
            $script:passoAtual = $i + 1; RefrescarLista
        }
        $barra.Value = 1000
        $script:fim = 'ok'
        $lbLinha.Text = ''
        TextoFim
        $btJogar.Visible = $true; $btImportar.Visible = $true; $btFechar.Visible = $true
        $script:rodando = $false
        if ($env:OCARINA_AUTO) { Registrar 'INSTALACAO CONCLUIDA'; $form.Close() }
    } catch {
        $script:rodando = $false
        $msg = "$($_.Exception.Message)"
        Registrar "ERRO: $msg"
        if ($msg -eq 'CANCELADO') { $form.Close(); return }
        $script:fim = 'erro'; TextoFim
        $btFechar.Visible = $true
        if ($env:OCARINA_AUTO) { $form.Close(); return }
        $r = [Windows.Forms.MessageBox]::Show((L "Etapa: $($script:passoNome)" "Step: $($script:passoNome)") + "`n`n" + (Explicar $msg) + (FimDoRegistro) + (L "`n`nQuer ver o registro da instalação?" "`n`nDo you want to see the install log?"), 'Ocarina Recomp', 'YesNo', 'Warning')
        if ($r -eq 'Yes') { AbrirRegistro }
    }
}

# ---- tela 1: logica
# An install that already has the game's files: it needs neither the game file nor the long steps again.
function JaInstalada($d) { return (Test-Path -LiteralPath (Join-Path $d 'build\romfs\index.json')) -and (Test-Path -LiteralPath (Join-Path $d 'build\exefs\main')) }
function PastaPadrao() {
    $livre = { param($d) [Math]::Floor((Get-PSDrive $d).Free / 1GB) }
    if ((& $livre 'C') -ge $GB_NECESSARIOS) { return 'C:\OcarinaRecomp' }
    $melhor = Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Free -and $_.Name.Length -eq 1 } | Sort-Object Free -Descending | Select-Object -First 1
    return "$($melhor.Name):\OcarinaRecomp"
}
function Validar() {
    $problema = @()
    $unid = $script:pasta.Substring(0, 1)
    # DriveInfo, not Get-PSDrive: an F: drive with 403 GB free read as 0 GB and Instalar stayed off (issue #114).
    # When the size cannot be read the install is not blocked: the step that runs out of space says so.
    $livre = try { $di = New-Object IO.DriveInfo $unid; if ($di.IsReady) { [Math]::Floor($di.AvailableFreeSpace / 1GB) } else { -1 } } catch { -1 }
    $parcial = Test-Path (Join-Path $script:pasta 'build\romfs\index.json')
    if ($script:pasta -match '[^\x21-\x7E]|[\[\]%&]') { $problema += (L 'A pasta de instalação não pode ter espaços nem acentos.' 'The install folder cannot have spaces or accents.') }
    elseif (-not $parcial -and $livre -ge 0 -and $livre -lt $GB_NECESSARIOS) { $problema += (L "Faltam $($GB_NECESSARIOS - $livre) GB livres em ${unid}: (precisa de $GB_NECESSARIOS GB)." "You need $($GB_NECESSARIOS - $livre) GB more free on ${unid}: ($GB_NECESSARIOS GB required).") }
    $txt = if ($livre -ge 0) { "$livre" } else { "?" }
    $lbPasta.Text = (L "$($script:pasta)   ($txt GB livres)" "$($script:pasta)   ($txt GB free)")
    $lbAviso.Text = ($problema + @($script:avisos | ForEach-Object { Pick $_ })) -join "`n"
    $pronta = JaInstalada $script:pasta
    if ($pronta -and -not $script:jogo) { $script:jogoEstado = 'pronta' }
    TextoJogo
    $btInstalar.Enabled = ($problema.Count -eq 0) -and ([bool]$script:jogo -or $pronta) -and -not $script:bloqueio
}
function EscolherJogo([string]$c) {
    if ((Test-Path -LiteralPath $c -PathType Container) -and (JaInstalada $c)) { $script:pasta = $c; $script:jogo = $null; Validar; return }
    # -LiteralPath: game files are often named with [title id], and [ ] are wildcards to the plain -Path
    if ($c -match '\.nup$' -and (Test-Path -LiteralPath $c -PathType Leaf)) {
        $script:jogo = $c; $script:jogoNup = $true; $script:jogoEstado = 'jogo'
    } else {
        $b = if (Test-Path -LiteralPath (Join-Path $c 'base\exefs\main')) { Join-Path $c 'base' } else { $c }
        if ((Test-Path -LiteralPath (Join-Path $b 'exefs\main')) -and (Test-Path -LiteralPath (Join-Path $b 'romfs'))) {
            $script:jogo = $b; $script:jogoNup = $false; $script:jogoEstado = 'jogo'
        } else {
            $script:jogo = $null; $script:jogoEstado = 'invalido'
        }
    }
    Validar
}
$btArq.add_Click({
    $d = New-Object Windows.Forms.OpenFileDialog
    $d.Title = (L 'Escolha o arquivo do seu jogo' 'Choose your game file'); $d.Filter = (L 'Jogo (*.nup)|*.nup' 'Game (*.nup)|*.nup')
    if ($d.ShowDialog($form) -eq 'OK') { EscolherJogo $d.FileName }
})
$btPasta.add_Click({
    $d = New-Object Windows.Forms.FolderBrowserDialog
    $d.Description = (L 'Escolha a pasta do jogo, a que tem exefs e romfs dentro' 'Choose the game folder, the one with exefs and romfs inside')
    if ($d.ShowDialog($form) -eq 'OK') { EscolherJogo $d.SelectedPath }
})
$btTrocar.add_Click({
    $d = New-Object Windows.Forms.FolderBrowserDialog
    $d.Description = (L 'Escolha onde instalar (sem espaços nem acentos no caminho)' 'Choose where to install (no spaces or accents in the path)')
    if ($d.ShowDialog($form) -eq 'OK') {
        $sel = $d.SelectedPath
        $script:pasta = if ((JaInstalada $sel) -or (Split-Path $sel -Leaf) -eq 'OcarinaRecomp') { $sel } else { Join-Path $sel 'OcarinaRecomp' }
        Validar
    }
})
$btInstalar.add_Click({ Instalar })
$btJogar.add_Click({
    Start-Process "$PSHOME\powershell.exe" -ArgumentList "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script:pasta\jogar\jogar-ocarina.ps1`""
    $form.Close()
})
$btFechar.add_Click({ $form.Close() })
$btImportar.add_Click({ Start-Process "$PSHOME\powershell.exe" -ArgumentList "-NoProfile -Sta -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script:pasta\jogar\importar-save.ps1`"" })
$form.add_FormClosing({
    param($s, $e)
    if (-not $script:rodando) { return }
    $e.Cancel = $true
    $r = [Windows.Forms.MessageBox]::Show((L "Parar a instalação agora?`n`nDepois é só abrir o instalador de novo: ele continua de onde parou." "Stop the installation now?`n`nYou can just open the installer again later: it picks up where it stopped."), 'Ocarina Recomp', 'YesNo', 'Question')
    if ($r -eq 'Yes') { $script:cancelar = $true }
})

# ---- verificacoes do PC, antes de qualquer coisa
$script:avisos = @()
$script:bloqueio = $false
$script:jogo = $null; $script:jogoNup = $false; $script:nupOpcoes = @('--overlay-update')
$script:pasta = if ($env:OCARINA_PASTA) { $env:OCARINA_PASTA } else { PastaPadrao }
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { $script:avisos += (L 'Este instalador só funciona em Windows de 64 bits.' 'This installer only works on 64-bit Windows.'); $script:bloqueio = $true }
$ram = [Math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
# Commit limit (RAM plus page file): on a first start the game commits about 24 GB (its 9 GB of console memory, the
# driver's backing for video memory, shader builds). Windows closes it when the limit is lower: a 16 GB PC with the
# page file the system manages (issue #72, limit 19.6 GB) froze and closed at the Deku Tree scene.
$commitGb = [Math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVirtualMemorySize / 1MB)
if ($commitGb -lt 26) { $script:avisos += (L "O Windows deixa usar até $commitGb GB de memória (RAM + arquivo de paginação) e o jogo pode precisar de uns 24 GB na primeira vez. Aumente o arquivo de paginação: Configurações avançadas do sistema > Desempenho > Avançado > Memória virtual > Alterar > tamanho personalizado, inicial 16384 MB e máximo 32768 MB, e reinicie o PC." "Windows allows up to $commitGb GB of memory (RAM + page file) and the game may need about 24 GB the first time. Enlarge the page file: Advanced system settings > Performance > Advanced > Virtual memory > Change > custom size, initial 16384 MB and maximum 32768 MB, then restart the PC.") }
if ($ram -lt 15) { $script:avisos += (L "Este PC tem $ram GB de RAM. Com menos de 16 GB, nunca testamos: a compilação ou o jogo podem falhar." "This PC has $ram GB of RAM. We never tested with less than 16 GB: the build or the game may fail.") }
$gpus = (Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }) -join ', '
Add-Type -Namespace Ocarina -Name Cpu -MemberDefinition '[DllImport("kernel32.dll")] public static extern bool IsProcessorFeaturePresent(uint f);'
# PF_AVX2 (40) is answered from Windows 10 2004 on; older Windows says 0 for it and for SSSE3 (36), which every x64 CPU has
if (-not [Ocarina.Cpu]::IsProcessorFeaturePresent(40)) {
    if ([Ocarina.Cpu]::IsProcessorFeaturePresent(36)) { $script:avisos += (L 'Este processador não tem AVX2, então o jogo não deve funcionar.' 'This processor has no AVX2, so the game should not work.'); $script:bloqueio = $true }
    else { $script:avisos += (L 'Esta versão do Windows não informa se o processador tem AVX2. O jogo precisa de AVX2.' 'This Windows version does not say whether the processor has AVX2. The game needs it.') }
}
if ($gpus -notmatch 'NVIDIA') { $script:avisos += (L "Só testamos em placa NVIDIA. Sua placa ($gpus) pode não funcionar." "We only tested on NVIDIA cards. Your card ($gpus) may not work.") }
if (-not $SETUP -or -not (Test-Path -LiteralPath (Join-Path $SETUP 'ferramentas\python\python.exe'))) {
    $script:avisos += (L 'Extraia o ZIP inteiro para uma pasta e abra o instalador de dentro dela.' 'Unpack the whole ZIP to a folder and open the installer from inside it.')
    $script:bloqueio = $true
}
$script:marcar = $false
AplicarIdioma
if ($env:OCARINA_JOGO) { EscolherJogo $env:OCARINA_JOGO }
if ($env:OCARINA_AUTO) { if (-not $script:jogo -and -not (JaInstalada $script:pasta)) { throw 'OCARINA_JOGO: jogo nao encontrado' }; $form.add_Shown({ Instalar }) }

try { [void][Windows.Forms.Application]::Run($form) }
catch {
    Registrar "ERRO GERAL: $($_.Exception.Message)"
    [Windows.Forms.MessageBox]::Show((L "O instalador encontrou um erro:`n`n$($_.Exception.Message)`n`nRegistro: $LOG" "The installer hit an error:`n`n$($_.Exception.Message)`n`nLog: $LOG"), 'Ocarina Recomp', 'OK', 'Error') | Out-Null
}
