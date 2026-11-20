#!/usr/bin/env bash
# shellcheck disable=SC2034  # variables set here are consumed by the scripts that source this file
# Shared helpers for the Software Factory setup scripts (setup.sh, setup-sdd.sh, setup-tracker.sh).
# SOURCE it, never execute it:   . "$SCRIPT_DIR/_lib.sh"
# Sourcing has no side effects: it defines functions and two paths.

say(){ printf '%s\n' "$*"; }
have(){ command -v "$1" >/dev/null 2>&1; }

SKILLS_DIR="$HOME/.claude/skills"            # read natively by Claude Code AND opencode
COPILOT_SKILLS_DIR="$HOME/.copilot/skills"   # read by GitHub Copilot CLI (also ~/.agents/skills)

# resolve_target [DIR] — print the absolute path of the target repo (default: $PWD); status 1 if missing.
resolve_target(){ ( cd "${1:-$PWD}" 2>/dev/null && pwd ); }

# cfg_get <dot.path> — reads a simple scalar from $CONFIG_FILE (grep/awk — no yq). Up to 3 levels:
#   cfg_get sdd.engine            (2-level)
#   cfg_get sdd.gsd.source        (3-level)
#   cfg_get tracker.azure_devops.org
cfg_get(){
  local parts raw; IFS='.' read -ra parts <<< "$1"
  case "${#parts[@]}" in
    3)
      raw="$(awk "
        /^${parts[0]}:/{f=1;next}
        f && /^[^ ]/{f=0}
        f && /^  ${parts[1]}:/{g=1;next}
        f && g && /^    ${parts[2]}:/{sub(/^[[:space:]]*${parts[2]}:[[:space:]]*/,\"\"); sub(/#.*/,\"\"); sub(/[[:space:]]+\$/,\"\"); print; exit}
        f && g && /^  [^ ]/{g=0}
      " "$CONFIG_FILE" 2>/dev/null)"
      ;;
    2)
      raw="$(awk "
        /^${parts[0]}:/{f=1;next}
        f && /^[^ ]/{f=0}
        f && /^  ${parts[1]}:/{sub(/^[[:space:]]*${parts[1]}:[[:space:]]*/,\"\"); sub(/#.*/,\"\"); sub(/[[:space:]]+\$/,\"\"); print; exit}
      " "$CONFIG_FILE" 2>/dev/null)"
      ;;
    *)
      raw="$(grep -m1 "^${1}:" "$CONFIG_FILE" 2>/dev/null | sed "s/^${1}:[[:space:]]*//" | sed 's/#.*//' | sed 's/[[:space:]]*$//')"
      ;;
  esac
  # strip one layer of surrounding double/single quotes
  raw="${raw%\"}"; raw="${raw#\"}"; raw="${raw%\'}"; raw="${raw#\'}"
  printf '%s' "$raw"
}

# detect_runtimes — HAVE_CLAUDE / HAVE_COPILOT / HAVE_OPENCODE = 1|0 ; RUNTIMES = "claude copilot opencode" subset
detect_runtimes(){
  HAVE_CLAUDE=0; HAVE_COPILOT=0; HAVE_OPENCODE=0; RUNTIMES=""
  have claude   && { HAVE_CLAUDE=1;   RUNTIMES="$RUNTIMES claude"; }
  have copilot  && { HAVE_COPILOT=1;  RUNTIMES="$RUNTIMES copilot"; }
  have opencode && { HAVE_OPENCODE=1; RUNTIMES="$RUNTIMES opencode"; }
  RUNTIMES="${RUNTIMES# }"
}

# expose_to_copilot <skill-dir>... — make skills installed under ~/.claude/skills visible to Copilot CLI
# by symlinking them into ~/.copilot/skills. Idempotent; a REAL directory already there (e.g. gsd's own
# Copilot install) is left alone.
expose_to_copilot(){
  local d n
  mkdir -p "$COPILOT_SKILLS_DIR"
  for d in "$@"; do
    [ -d "$d" ] || continue
    n="$(basename "$d")"
    if [ -e "$COPILOT_SKILLS_DIR/$n" ] && [ ! -L "$COPILOT_SKILLS_DIR/$n" ]; then continue; fi
    ln -sfn "$d" "$COPILOT_SKILLS_DIR/$n"
  done
}
