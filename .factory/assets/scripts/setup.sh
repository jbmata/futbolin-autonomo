#!/usr/bin/env bash
# Software Factory — setup local: las HERRAMIENTAS y SKILLS que usa la factoría, para cada runtime presente
# (Claude Code, GitHub Copilot CLI, opencode). Las skills se instalan por referencia a su repositorio de origen
# (marketplace + plugin, o `npx skills add -g`); nunca se copian al repo. Y el cableado obligatorio de las house skills (caveman,
# ponytail, rtk, ccc). El motor SDD y el tracker son DECISIONES del proyecto: /factory-init las explica,
# el usuario elige y entonces se instalan con setup-sdd.sh / setup-tracker.sh (este script los llama solo
# si ya están elegidos: por flag o en .factory/config.yml). Nunca elige un default.
# Idempotente y best-effort: cada fallo degrada a un comando manual impreso.
#
# Uso:     setup.sh [--sdd gsd|spec-kit|openspec] [--tracker github|azure-devops|hub] [--skip-index] [TARGET_DIR]
# Config:  .factory/config.yml → skills.* · house_skills.* · sdd.engine · tracker.platform
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
. "$SCRIPT_DIR/_lib.sh"

SKIP_INDEX=0; TARGET=""; FLAG_SDD=""; FLAG_TRACKER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-index) SKIP_INDEX=1; shift ;;
    --sdd)        FLAG_SDD="${2:-}"; shift 2 ;;
    --tracker)    FLAG_TRACKER="${2:-}"; shift 2 ;;
    -*) say "Flag desconocido: $1   (uso: setup.sh [--sdd E] [--tracker P] [--skip-index] [TARGET_DIR])"; exit 2 ;;
    *)  TARGET="$1"; shift ;;
  esac
done
TARGET="$(resolve_target "$TARGET")" || { say "Target dir not found"; exit 1; }
CONFIG_FILE="$TARGET/.factory/config.yml"

SDD_ENGINE="$FLAG_SDD";        [ -z "$SDD_ENGINE" ] && SDD_ENGINE="$(cfg_get sdd.engine)"
TRACKER="$FLAG_TRACKER";       [ -z "$TRACKER" ] && TRACKER="$(cfg_get tracker.platform)"

detect_runtimes
say "════════════════════════════════════════════════════════"
say " Software Factory — setup (herramientas + skills + house skills)"
say " repo:     $TARGET"
say " runtimes: ${RUNTIMES:-ninguno detectado}"
say " SDD:      ${SDD_ENGINE:-(sin elegir — se elige en /factory-init)}"
say " tracker:  ${TRACKER:-(sin elegir — se elige en /factory-init)}"
say "════════════════════════════════════════════════════════"

# ── 1. Runtimes + herramientas base ────────────────────────────────────────────
say ""; say "[1/6] Runtimes + herramientas base"
if [ "$HAVE_CLAUDE" = 1 ];   then say "  ✓ claude   (Claude Code)";        else say "  ○ claude   no encontrado — instala: npm i -g @anthropic-ai/claude-code"; fi
if [ "$HAVE_COPILOT" = 1 ];  then say "  ✓ copilot  (GitHub Copilot CLI)"; else say "  ○ copilot  no encontrado — instala: npm i -g @github/copilot   (Node 22+, luego 'copilot' y /login)"; fi
if [ "$HAVE_OPENCODE" = 1 ]; then say "  ✓ opencode";                      else say "  ○ opencode no encontrado — instala: curl -fsSL https://opencode.ai/install | bash"; fi
miss=0
for t in git node npx python3; do
  if have "$t"; then say "  ✓ $t"; else say "  ✗ $t  (falta — instálalo)"; miss=1; fi
done
if have gh; then
  if gh auth status >/dev/null 2>&1; then say "  ✓ gh (autenticado)"; else say "  ⚠ gh sin autenticar — ejecuta: gh auth login"; fi
else
  say "  ✗ gh  (https://cli.github.com — lo necesitan el tracker github y factory-sync)"
fi
if have docker; then
  say "  ✓ docker"
  if docker compose version >/dev/null 2>&1; then say "  ✓ docker compose"; else say "  ✗ docker compose plugin"; fi
else
  say "  ⚠ docker no encontrado (lo necesita el gate docker-run)"
fi
have pipx || { say "  ✗ pipx (instala: python3 -m pip install --user pipx && python3 -m pipx ensurepath)"; miss=1; }
[ "$miss" = 1 ] && say "  → Instala lo que falte y vuelve a ejecutar; el resto continúa best-effort."

# ── 2. Skills desde su repositorio de origen (nunca copiadas en la factoría) ───
# Las factory-* son el plugin `software-factory` del marketplace `fastex-factory` (el repo de la factoría) y las PM
# skills de la factoría son plugins de phuryn/pm-skills. Claude Code y Copilot CLI los instalan por referencia
# (marketplace + plugin); opencode y Codex no tienen marketplace, así que `npx skills add -g` trae los mismos
# SKILL.md desde el origen al directorio de usuario (~/.agents/skills), nunca al repo.
# OJO: deanpeters/Product-Manager-Skills NO va aquí — es para el Cowork del FUNCIONAL (.factory/cowork/).
SKILLS_MARKETPLACE="$(cfg_get skills.marketplace)"; SKILLS_SOURCE="$(cfg_get skills.source)"
SKILLS_CROSS_AGENT="$(cfg_get skills.cross_agent)"
[ -z "$SKILLS_MARKETPLACE" ] && SKILLS_MARKETPLACE="fastex-factory"
[ -z "$SKILLS_SOURCE" ]      && SKILLS_SOURCE="https://github.com/Fastex-Soluciones-IT/factoria-software"
[ -z "$SKILLS_CROSS_AGENT" ] && SKILLS_CROSS_AGENT="opencode,codex"
PM_PLUGINS="pm-product-discovery pm-product-strategy pm-execution pm-market-research"
slug(){ local s="${1#https://github.com/}"; printf '%s' "${s%.git}"; }
installed_claude(){  grep -q "\"$1\"" "$HOME/.claude/plugins/installed_plugins.json" 2>/dev/null; }
installed_copilot(){ [ -d "$HOME/.copilot/installed-plugins/${1#*@}/${1%@*}" ]; }

# from_origin <marketplace-source> <marketplace-name> <plugin>... — register the marketplace, install each plugin.
from_origin(){
  local src="$1" mkt="$2" csrc p; shift 2
  case "$src" in https://github.com/*) csrc="$(slug "$src")" ;; *) csrc="$src" ;; esac
  if [ "$HAVE_CLAUDE" = 1 ]; then
    claude plugin marketplace add "$src" </dev/null >/dev/null 2>&1
    for p in "$@"; do
      if installed_claude "$p@$mkt"; then say "  ✓ Claude Code: $p@$mkt"
      elif claude plugin install "$p@$mkt" --scope user </dev/null >/dev/null 2>&1 && installed_claude "$p@$mkt"; then say "  ✓ Claude Code: $p@$mkt instalado"
      else say "  ⚠ Claude Code: manual: claude plugin marketplace add $src && claude plugin install $p@$mkt"; fi
    done
  fi
  if [ "$HAVE_COPILOT" = 1 ]; then
    copilot plugin marketplace add "$csrc" </dev/null >/dev/null 2>&1
    for p in "$@"; do
      if installed_copilot "$p@$mkt"; then say "  ✓ Copilot CLI: $p@$mkt"
      # copilot exits 0 even when the install fails, so success is what lands on disk.
      elif copilot plugin install "$p@$mkt" </dev/null >/dev/null 2>&1; installed_copilot "$p@$mkt"; then say "  ✓ Copilot CLI: $p@$mkt instalado"
      else say "  ⚠ Copilot CLI: manual: copilot plugin marketplace add $csrc && copilot plugin install $p@$mkt"; fi
    done
  fi
}
# cross_agent <source> — the same SKILL.md files for opencode/Codex, user level (-g: never into the repo).
cross_agent(){
  local agents="" a
  case "$1" in https://github.com/*) ;; *) say "  ⚠ fuente local — omito npx skills add (espera owner/repo)"; return ;; esac
  [ -n "$SKILLS_CROSS_AGENT" ] && have npx || return
  for a in ${SKILLS_CROSS_AGENT//,/ }; do agents="$agents -a $a"; done
  # shellcheck disable=SC2086
  if npx -y skills add "$(slug "$1")" -g $agents -y </dev/null >/dev/null 2>&1; then
    say "  ✓ $SKILLS_CROSS_AGENT: $(slug "$1") en ~/.agents/skills"
  else say "  ⚠ manual: npx skills add $(slug "$1") -g$agents -y"; fi
}

say ""; say "[2/6] Skills desde su origen — software-factory@$SKILLS_MARKETPLACE ($SKILLS_SOURCE) + phuryn/pm-skills"
from_origin "$SKILLS_SOURCE" "$SKILLS_MARKETPLACE" software-factory
cross_agent "$SKILLS_SOURCE"
# shellcheck disable=SC2086
from_origin https://github.com/phuryn/pm-skills pm-skills $PM_PLUGINS
cross_agent https://github.com/phuryn/pm-skills
[ "$HAVE_CLAUDE$HAVE_COPILOT$HAVE_OPENCODE" = 000 ] && say "  ○ sin runtimes: vuelve a ejecutar setup.sh cuando instales uno"

# Older setups COPIED the phuryn skills into ~/.claude/skills and listed them in a manifest. Once the plugins are
# installed those copies only duplicate them, so they go (only what the manifest lists, nothing else).
PM_MANIFEST="$SKILLS_DIR/.factory-pm-skills"
if [ -f "$PM_MANIFEST" ] && installed_claude "pm-product-discovery@pm-skills"; then
  while IFS= read -r s; do
    [ -n "$s" ] || continue
    [ -L "$COPILOT_SKILLS_DIR/$s" ] && rm -f "$COPILOT_SKILLS_DIR/$s"
    [ -d "$SKILLS_DIR/$s" ] && [ ! -L "$SKILLS_DIR/$s" ] && rm -rf "${SKILLS_DIR:?}/$s"
  done < "$PM_MANIFEST"
  rm -f "$PM_MANIFEST"
  say "  ✓ copias antiguas de phuryn retiradas de ~/.claude/skills (ahora llegan como plugins)"
fi

# ── 3. CocoIndex Code (CLI + skill + MCP ×3 con ruta absoluta + índice) ────────
say ""; say "[3/6] CocoIndex Code (búsqueda semántica de código)"
if [ "$(cfg_get house_skills.ccc)" = off ]; then
  say "  ○ house_skills.ccc: off — omitido"
else
  ARGS=("$TARGET"); [ "$SKIP_INDEX" = 1 ] && ARGS=(--skip-index "$TARGET")
  bash "$SCRIPT_DIR/setup-cocoindex.sh" "${ARGS[@]}" 2>&1 | sed 's/^/  /'
fi

# ── 4. House skills (obligatorias): plugins por máquina + cableado del proyecto ─
say ""; say "[4/6] House skills — caveman, ponytail, rtk, ccc (ver AGENTS.md § House skills)"
plugin_claude(){   # <plugin> <owner/repo>
  installed=$(grep -c "\"$1@$1\"" "$HOME/.claude/plugins/installed_plugins.json" 2>/dev/null)
  if [ "${installed:-0}" -gt 0 ]; then say "  ✓ Claude Code: $1@$1 ya instalado"
  elif claude plugin marketplace add "$2" </dev/null >/dev/null 2>&1; claude plugin install "$1@$1" --scope user </dev/null >/dev/null 2>&1; then
    say "  ✓ Claude Code: $1@$1 instalado"
  else say "  ⚠ Claude Code: manual: claude plugin marketplace add $2 && claude plugin install $1@$1"; fi
}
if [ "$HAVE_CLAUDE" = 1 ]; then
  plugin_claude ponytail DietrichGebert/ponytail
  plugin_claude caveman JuliusBrussee/caveman
fi
if [ "$HAVE_COPILOT" = 1 ]; then
  if [ -d "$HOME/.copilot/installed-plugins/ponytail/ponytail" ]; then say "  ✓ Copilot CLI: ponytail@ponytail ya instalado"
  elif copilot plugin marketplace add DietrichGebert/ponytail </dev/null >/dev/null 2>&1; copilot plugin install ponytail@ponytail </dev/null >/dev/null 2>&1; then
    say "  ✓ Copilot CLI: ponytail@ponytail instalado"
  else say "  ⚠ Copilot CLI: manual: copilot plugin marketplace add DietrichGebert/ponytail && copilot plugin install ponytail@ponytail"; fi
fi
if [ "$HAVE_OPENCODE" = 1 ] && have npx; then
  npx -y github:JuliusBrussee/caveman -- --only opencode </dev/null >/dev/null 2>&1 \
    && say "  ✓ opencode: caveman (plugin nativo)" \
    || say "  ⚠ opencode: manual: npx -y github:JuliusBrussee/caveman -- --only opencode"
fi
if have rtk; then
  say "  ✓ rtk ($(command -v rtk))"
  # opencode has no hook file: rtk ships an opencode plugin, but `rtk init --opencode` also writes a GLOBAL
  # Claude Code hook that is not PATH-guarded, so it is left to the user instead of run here.
  if [ "$HAVE_OPENCODE" = 1 ] && [ "$(cfg_get house_skills.rtk)" != off ]; then
    say "  ○ opencode: para rtk ejecuta una vez 'rtk init -g --opencode' (añade también un hook global de Claude Code)"
  fi
else say "  ○ rtk no está en el PATH — el hook queda desactivado (sin errores). Instálalo: https://github.com/rtk-ai/rtk"; fi
bash "$SCRIPT_DIR/house-skills.sh" apply "$TARGET" 2>&1 | sed 's/^/  /'

# ── 5. Motor SDD (solo si ya está elegido) ─────────────────────────────────────
say ""; say "[5/6] Motor SDD: ${SDD_ENGINE:-(sin elegir)}"
if [ -n "$SDD_ENGINE" ]; then bash "$SCRIPT_DIR/setup-sdd.sh" "$SDD_ENGINE" "$TARGET" 2>&1 | sed 's/^/  /'
else say "  ⏭  sin elegir — /factory-init lo explica y pregunta (gsd | spec-kit | openspec); no instalo ninguno por defecto."; fi

# ── 6. Tracker (solo si ya está elegido) ───────────────────────────────────────
say ""; say "[6/6] Tracker: ${TRACKER:-(sin elegir)}"
if [ -n "$TRACKER" ]; then bash "$SCRIPT_DIR/setup-tracker.sh" "$TRACKER" "$TARGET" 2>&1 | sed 's/^/  /'
else say "  ⏭  sin elegir — /factory-init lo explica y pregunta (github | azure-devops)."; fi

# ── Readiness ───────────────────────────────────────────────────────────────────
say ""
say "════════════════════════════════════════════════════════"
say " Readiness — $(date '+%Y-%m-%d')"
say "════════════════════════════════════════════════════════"
printf "  %-12s %s\n" "runtimes:"  "${RUNTIMES:-ninguno detectado}"
printf "  %-12s %s\n" "skills:"    "software-factory@$SKILLS_MARKETPLACE + phuryn pm-skills, por referencia (sin copias en el repo)"
printf "  %-12s %s\n" "SDD:"       "${SDD_ENGINE:-sin elegir}"
printf "  %-12s %s\n" "tracker:"   "${TRACKER:-sin elegir}"
say "  house skills:"
bash "$SCRIPT_DIR/house-skills.sh" check "$TARGET"
say ""
say "Listo. Abre el repo en Claude Code, opencode o GitHub Copilot CLI y escribe 'guíame' (o /factory-status)."
