#!/usr/bin/env bash
# setup-cowork-skills.sh — lo ejecuta el TÉCNICO en el equipo del FUNCIONAL.
# Deja el Claude/Cowork del funcional listo para la toma de requisitos con "Aura":
#   1) instala deanpeters/Product-Manager-Skills (skills interactivas/advisor, ideales para Cowork),
#   2) recuerda pegar las instrucciones de Aura como Instructions del Project de Cowork,
#   3) lista los comandos que el funcional podrá usar.
#
# Por qué dean aquí (y phuryn en la factoría): dean coachea con preguntas, una por turno → encaja con
# una persona no técnica en Cowork. La factoría (agente autónomo) usa phuryn (ver setup.sh, paso 2).
#
# Uso:  bash .factory/cowork/setup-cowork-skills.sh
set -u
say(){ printf '%s\n' "$*"; }
have(){ command -v "$1" >/dev/null 2>&1; }

HERE="$(cd "$(dirname "$0")" && pwd)"
INSTR="$HERE/PROJECT-INSTRUCTIONS.md"
SKILLS_DIR="$HOME/.claude/skills"

say "== Cowork del funcional — skills de PM (dean) + Aura =="
say ""

# 1) Skills de PM para el funcional → deanpeters/Product-Manager-Skills
say "[1/2] Instalando deanpeters/Product-Manager-Skills (para el Claude del funcional)…"
if [ -d "$SKILLS_DIR/prd-development" ] || [ -d "$SKILLS_DIR/problem-framing-canvas" ]; then
  say "  ✓ ya presentes en ~/.claude/skills"
elif have npx; then
  npx -y skills add deanpeters/Product-Manager-Skills -a claude-code -g </dev/null \
    && say "  ✓ instaladas" \
    || say "  ⚠ falló — manual: npx skills add deanpeters/Product-Manager-Skills -a claude-code -g"
else
  say "  ⚠ npx no disponible."
  say "    · Claude Code:     npx skills add deanpeters/Product-Manager-Skills -a claude-code -g"
  say "    · Claude Desktop/Cowork (no técnico): descarga el 'pm-skills-starter-pack.zip' de"
  say "      https://github.com/deanpeters/Product-Manager-Skills/releases/latest y súbelo en la app."
fi

say ""
# 2) Aura como Instructions del Project de Cowork
say "[2/2] Configura a AURA en el Project de Cowork del funcional:"
if [ -f "$INSTR" ]; then
  say "  · Abre:  $INSTR"
  say "  · Copia TODO su contenido y pégalo en el Project de Cowork → 'Instructions'"
  say "    (o como custom instructions del Project en claude.ai web)."
else
  say "  ⚠ No encuentro PROJECT-INSTRUCTIONS.md junto a este script."
fi

say ""
say "Listo. El funcional ya puede trabajar SOLO. Comandos que puede usar en su Claude/Cowork:"
say "  /discover       → arranca la toma de requisitos guiada"
say "  /write-prd      → redacta el documento de requisitos (PRD)"
say "  /prioritize     → ayuda a priorizar funcionalidades (MVP vs después)"
say "  /plan-roadmap   → ordena el trabajo en una hoja de ruta"
say "  /strategy       → encuadre estratégico del producto"
say "  …o simplemente hablarle a AURA con sus palabras: ella guía el resto."
say ""
say "Cuando termine, su brief queda en .factory/intake/ y se lo entrega a la factoría"
say "(Forma A: ZIP al técnico · Forma B: ya está en el repo conectado)."
