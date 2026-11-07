# Imports a save of Ocarina of Time from Lichii, Eden or a similar emulator. The save files are the same in both.
# It only reads the emulator's folder. The current save is kept in perfil\save-antes-de-importar-<date> first.
#
#   powershell -ExecutionPolicy Bypass -File jogar\importar-save.ps1 [-Origem <folder with account.sav>] [-Perfil <folder>] [-SemPerguntar]
param([string]$Origem, [string]$Perfil, [switch]$SemPerguntar)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
[Windows.Forms.Application]::EnableVisualStyles()
$PT = (Get-UICulture).TwoLetterISOLanguageName -eq 'pt'
function L($textoPt, $textoEn) { if ($PT) { return $textoPt } return $textoEn }   # not $pt: names are case-insensitive, it would hide $PT
$TITULO = '0400486024DEC000'
$NOME = 'Ocarina Recomp'

function Aviso($texto, $icone = 'Information') {
    if ($SemPerguntar) { Write-Output $texto } else { [void][Windows.Forms.MessageBox]::Show($texto, $NOME, 'OK', $icone) }
}
function EhSave($d) { return (Test-Path -LiteralPath (Join-Path $d 'account.sav')) -or (Test-Path -LiteralPath (Join-Path $d 'slot_00')) }

# The usual places of the emulators' nand folder; a portable emulator is chosen by hand.
function Procurar() {
    $achados = @()
    foreach ($nome in 'lichii', 'Lichii', 'eden', 'Eden', 'eglantine', 'Eglantine', 'citron', 'sudachi') {
        $base = Join-Path $env:APPDATA "$nome\nand\user\save"
        if (-not (Test-Path -LiteralPath $base)) { continue }
        $achados += Get-ChildItem -LiteralPath $base -Directory -Recurse -Depth 2 -Filter $TITULO -ErrorAction SilentlyContinue |
            Where-Object { EhSave $_.FullName }
    }
    return $achados | Sort-Object { (Get-ChildItem -LiteralPath $_.FullName -Recurse -File | Measure-Object LastWriteTime -Maximum).Maximum } -Descending
}

try {
    if (-not $Perfil) { $Perfil = Join-Path (Split-Path $PSScriptRoot -Parent) 'perfil' }
    if (-not $Origem) {
        $achados = @(Procurar)
        if ($achados.Count -gt 0) {
            $Origem = $achados[0].FullName
        } else {
            $d = New-Object Windows.Forms.FolderBrowserDialog
            $d.Description = (L "Não achei o save sozinho. Escolha a pasta do save, a que termina em $TITULO e tem account.sav dentro. Num emulador portátil, fica em user\nand\user\save." "Could not find the save by myself. Choose the save folder, the one ending in $TITULO with account.sav inside. In a portable emulator it is in user\nand\user\save.")
            if ($d.ShowDialog() -ne 'OK') { exit 0 }
            $Origem = $d.SelectedPath
        }
    }
    if (-not (EhSave $Origem)) { throw (L "Essa pasta não tem um save do jogo (account.sav): $Origem" "This folder has no save of the game (account.sav): $Origem") }

    if (-not $SemPerguntar) {
        # A player imported an old emulator save over newer progress by mistake: when the current save is newer, say so and
        # make No the default button. The current save is kept in a copy either way.
        $aviso = ''
        $uidAtual = Join-Path $Perfil 'account.uid'
        if (Test-Path -LiteralPath $uidAtual) {
            $u = [IO.File]::ReadAllBytes($uidAtual)
            if ($u.Length -eq 16) {
                [Array]::Reverse($u)
                $atual = Join-Path $Perfil ("save\" + (-join ($u | ForEach-Object { $_.ToString('x2') })))
                $quando = { param($d) (Get-ChildItem -LiteralPath $d -Recurse -File -ErrorAction SilentlyContinue | Measure-Object LastWriteTime -Maximum).Maximum }
                if (Test-Path -LiteralPath $atual) {
                    $tAtual = & $quando $atual; $tOrigem = & $quando $Origem
                    if ($tAtual -and $tOrigem -and $tAtual -gt $tOrigem) {
                        $aviso = (L "ATENÇÃO: o seu save atual ($($tAtual.ToString('g'))) é MAIS NOVO que o que vai ser importado ($($tOrigem.ToString('g'))). Importar troca o seu progresso pelo mais antigo.`n`n" "WARNING: your current save ($($tAtual.ToString('g'))) is NEWER than the one being imported ($($tOrigem.ToString('g'))). Importing replaces your progress with the older one.`n`n")
                    }
                }
            }
        }
        $icone = if ($aviso) { 'Warning' } else { 'Question' }
        $padrao = if ($aviso) { 'Button2' } else { 'Button1' }
        $r = [Windows.Forms.MessageBox]::Show($aviso + (L "Importar este save?`n`n$Origem`n`nO seu save atual fica guardado em uma cópia em $Perfil\save-antes-de-importar-<data>. Feche o jogo antes de continuar." "Import this save?`n`n$Origem`n`nYour current save is kept in a copy in $Perfil\save-antes-de-importar-<date>. Close the game before continuing."), $NOME, 'YesNo', $icone, $padrao)
        if ($r -ne 'Yes') { exit 0 }
    }
    if (Get-Process ocarina -ErrorAction SilentlyContinue) { throw (L 'Feche o jogo e tente de novo.' 'Close the game and try again.') }

    # The game names the save folder after the account: account.uid's 16 bytes, reversed, in hex.
    New-Item -ItemType Directory -Force $Perfil | Out-Null
    $uid = Join-Path $Perfil 'account.uid'
    if (-not (Test-Path -LiteralPath $uid)) {
        $b = New-Object byte[] 16
        [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b)
        [IO.File]::WriteAllBytes($uid, $b)
    }
    $b = [IO.File]::ReadAllBytes($uid)
    if ($b.Length -ne 16) { throw "account.uid: $($b.Length) bytes" }
    [Array]::Reverse($b)
    $pasta = -join ($b | ForEach-Object { $_.ToString('x2') })
    $destino = Join-Path $Perfil "save\$pasta"

    if (Test-Path -LiteralPath $destino) {
        $copia = Join-Path $Perfil ("save-antes-de-importar-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
        Move-Item -LiteralPath $destino -Destination $copia
    } else { $copia = $null }
    # the working copy of the last session would replace the imported save at the next mount
    $tx = Join-Path $Perfil 'save-tx'
    if (Test-Path -LiteralPath $tx) { Remove-Item -LiteralPath $tx -Recurse -Force }
    New-Item -ItemType Directory -Force $destino | Out-Null
    Copy-Item -Path (Join-Path ([Management.Automation.WildcardPattern]::Escape($Origem)) '*') -Destination $destino -Recurse -Force
    $extra = if ($copia) { (L "`n`nSeu save anterior ficou em:`n$copia" "`n`nYour previous save is in:`n$copia") } else { '' }
    Aviso ((L "Save importado! Abra o jogo e escolha Continuar." "Save imported! Open the game and choose Continue.") + $extra)
} catch {
    Aviso ((L 'Não consegui importar o save: ' 'Could not import the save: ') + $_.Exception.Message) 'Error'
    exit 1
}
