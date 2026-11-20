#!/usr/bin/env pwsh
# setup-cowork-skills.ps1 - port PowerShell de setup-cowork-skills.sh (para Windows nativo).
# Lo ejecuta el TECNICO en el equipo del FUNCIONAL, EN EL MISMO entorno/usuario donde corre su Claude Code.
# Deja el Claude/Cowork del funcional listo para la toma de requisitos con "Aura":
#   1) instala deanpeters/Product-Manager-Skills (skills interactivas/advisor, ideales para Cowork),
#   2) recuerda pegar las instrucciones de Aura como Instructions del Project de Cowork,
#   3) lista los comandos que el funcional podra usar.
#
# Por que dean aqui (y phuryn en la factoria): dean coachea con preguntas, una por turno -> encaja con
# una persona no tecnica en Cowork. La factoria (agente autonomo) usa phuryn (ver setup.sh, paso 2).
#
# Uso:  pwsh -File .\.factory\cowork\setup-cowork-skills.ps1
#   o, ya dentro de PowerShell:  .\.factory\cowork\setup-cowork-skills.ps1
# Equivale a setup-cowork-skills.sh: usa el .ps1 en Windows nativo, el .sh en WSL/Git Bash.

function Say($m)  { Write-Output $m }
function Have($c) { $null -ne (Get-Command $c -ErrorAction SilentlyContinue) }

$Here      = $PSScriptRoot
$Instr     = Join-Path $Here 'PROJECT-INSTRUCTIONS.md'
$SkillsDir = Join-Path (Join-Path $HOME '.claude') 'skills'

Say "== Cowork del funcional - skills de PM (dean) + Aura =="
Say ""

# 1) Skills de PM para el funcional -> deanpeters/Product-Manager-Skills
Say "[1/2] Instalando deanpeters/Product-Manager-Skills (para el Claude del funcional)..."
if ((Test-Path (Join-Path $SkillsDir 'prd-development')) -or (Test-Path (Join-Path $SkillsDir 'problem-framing-canvas'))) {
    Say "  OK - ya presentes en ~/.claude/skills"
}
elseif (Have npx) {
    npx -y skills add deanpeters/Product-Manager-Skills -a claude-code -g
    if ($LASTEXITCODE -eq 0) { Say "  OK - instaladas" }
    else { Say "  ! fallo - manual: npx skills add deanpeters/Product-Manager-Skills -a claude-code -g" }
}
else {
    Say "  ! npx no disponible."
    Say "    - Claude Code:     npx skills add deanpeters/Product-Manager-Skills -a claude-code -g"
    Say "    - Claude Desktop/Cowork (no tecnico): descarga el 'pm-skills-starter-pack.zip' de"
    Say "      https://github.com/deanpeters/Product-Manager-Skills/releases/latest y subelo en la app."
}

Say ""
# 2) Aura como Instructions del Project de Cowork
Say "[2/2] Configura a AURA en el Project de Cowork del funcional:"
if (Test-Path $Instr) {
    Say "  - Abre:  $Instr"
    Say "  - Copia TODO su contenido y pegalo en el Project de Cowork -> 'Instructions'"
    Say "    (o como custom instructions del Project en claude.ai web)."
}
else {
    Say "  ! No encuentro PROJECT-INSTRUCTIONS.md junto a este script."
}

Say ""
Say "Listo. El funcional ya puede trabajar SOLO. Comandos que puede usar en su Claude/Cowork:"
Say "  /discover       -> arranca la toma de requisitos guiada"
Say "  /write-prd      -> redacta el documento de requisitos (PRD)"
Say "  /prioritize     -> ayuda a priorizar funcionalidades (MVP vs despues)"
Say "  /plan-roadmap   -> ordena el trabajo en una hoja de ruta"
Say "  /strategy       -> encuadre estrategico del producto"
Say "  ...o simplemente hablarle a AURA con sus palabras: ella guia el resto."
Say ""
Say "Cuando termine, su brief queda en .factory/intake/ y se lo entrega a la factoria"
Say "(Forma A: ZIP al tecnico - Forma B: ya esta en el repo conectado)."
