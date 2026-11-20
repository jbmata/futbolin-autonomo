#!/usr/bin/env bash
# Software Factory — push CI/CD variables & secrets to a GitHub repo from a filled .env.cicd file.
#
# Reads a .env.cicd (see factory/templates/cicd/<starterkit>/.env.cicd.example) split into two sections:
#   # --- VARIABLES ---   keys below go to GitHub  *Variables*  (gh variable set)  — NOT sensitive
#   # --- SECRETS ---     keys below go to GitHub  *Secrets*    (gh secret set)    — sensitive, never echoed
# Empty values are skipped. A value of the form `@/path/to/file` is read from that file (use this for
# multiline keys like SSH_PRIVATE_KEY: `SSH_PRIVATE_KEY=@~/.ssh/deploy_key`).
#
# Usage:
#   set-github-secrets.sh [--repo owner/name] [--env dev|prod] [--file .env.cicd] [--create-env] [--dry-run]
#
#   --repo        target repo (default: auto-detected via `gh repo view`)
#   --env         GitHub Environment to scope vars/secrets to (omit = repo-level)
#   --file        the filled env file (default: .env.cicd)
#   --create-env  create the GitHub Environment first if it doesn't exist
#   --dry-run     print what would be set (names only, never values), change nothing
#
# Requires: gh (authenticated — `gh auth login`). Secret VALUES are never printed.
set -uo pipefail

REPO=""; ENVIRONMENT=""; FILE=".env.cicd"; CREATE_ENV=0; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --env) ENVIRONMENT="$2"; shift 2 ;;
    --file) FILE="$2"; shift 2 ;;
    --create-env) CREATE_ENV=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "Flag desconocido: $1" >&2; exit 2 ;;
  esac
done

say(){ printf '%s\n' "$*"; }
have(){ command -v "$1" >/dev/null 2>&1; }

have gh || { say "✗ Falta 'gh' (GitHub CLI). Instálalo y haz 'gh auth login'."; exit 1; }
gh auth status >/dev/null 2>&1 || { say "✗ gh no autenticado. Ejecuta: gh auth login"; exit 1; }
[ -f "$FILE" ] || { say "✗ No existe el fichero '$FILE'. Copia el .env.cicd.example, rellénalo y pásalo con --file."; exit 1; }

# Resolve repo
if [ -z "$REPO" ]; then
  REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)"
  [ -n "$REPO" ] || { say "✗ No pude detectar el repo. Pasa --repo owner/name."; exit 1; }
fi
say "Repo:        $REPO"
say "Environment: ${ENVIRONMENT:-<repo-level>}"
say "Fichero:     $FILE"
[ "$DRY" = 1 ] && say "Modo:        DRY-RUN (no se cambia nada)"
say ""

# Ensure environment exists if requested
if [ -n "$ENVIRONMENT" ] && [ "$CREATE_ENV" = 1 ] && [ "$DRY" = 0 ]; then
  gh api -X PUT "repos/$REPO/environments/$ENVIRONMENT" >/dev/null 2>&1 \
    && say "✓ Environment '$ENVIRONMENT' asegurado" \
    || say "⚠ No pude crear el environment '$ENVIRONMENT' (¿permisos?); continúo."
fi

resolve_value(){ # $1=raw value → echoes resolved (handles @file); returns 1 if @file missing
  local v="$1"
  if [ "${v:0:1}" = "@" ]; then
    local p="${v:1}"; p="${p/#\~/$HOME}"
    [ -f "$p" ] || { return 1; }
    cat "$p"; return 0
  fi
  printf '%s' "$v"
}

gh_scope_args(){ printf -- '--repo %s' "$REPO"; [ -n "$ENVIRONMENT" ] && printf -- ' --env %s' "$ENVIRONMENT"; }

section=""; nvar=0; nsec=0; nskip=0
while IFS= read -r line || [ -n "$line" ]; do
  # section markers
  case "$line" in
    *'--- VARIABLES ---'*) section="var"; continue ;;
    *'--- SECRETS ---'*)   section="sec"; continue ;;
  esac
  # strip leading space; skip blanks and full-line comments
  trimmed="${line#"${line%%[![:space:]]*}"}"
  [ -z "$trimmed" ] && continue
  [ "${trimmed:0:1}" = "#" ] && continue
  case "$trimmed" in *=*) ;; *) continue ;; esac
  key="${trimmed%%=*}"; key="${key%"${key##*[![:space:]]}"}"   # rtrim key
  raw="${trimmed#*=}"
  # strip a quoted value's quotes, else strip a trailing "  # comment"
  if [ "${raw:0:1}" = '"' ]; then
    raw="${raw%\"*}"; raw="${raw#\"}"
  else
    raw="$(printf '%s' "$raw" | sed -E 's/[[:space:]]+#.*$//')"
    raw="${raw%"${raw##*[![:space:]]}"}"   # rtrim
  fi
  [ -z "$raw" ] && { nskip=$((nskip+1)); continue; }
  [ -z "$section" ] && { say "⚠ '$key' fuera de sección VARIABLES/SECRETS — se omite"; continue; }

  if ! value="$(resolve_value "$raw")"; then
    say "✗ '$key': fichero referenciado '${raw:1}' no existe — se omite"; continue
  fi

  if [ "$section" = "var" ]; then
    if [ "$DRY" = 1 ]; then say "  [dry-run] gh variable set $key  ($(printf '%s' "$value" | head -c 40 | tr -d '\n')…)"
    else printf '%s' "$value" | gh variable set "$key" $(gh_scope_args) >/dev/null 2>&1 \
           && say "  ✓ variable $key" || say "  ✗ variable $key (falló)"; fi
    nvar=$((nvar+1))
  else
    if [ "$DRY" = 1 ]; then say "  [dry-run] gh secret set $key  (valor oculto)"
    else printf '%s' "$value" | gh secret set "$key" $(gh_scope_args) >/dev/null 2>&1 \
           && say "  ✓ secret   $key" || say "  ✗ secret   $key (falló)"; fi
    nsec=$((nsec+1))
  fi
done < "$FILE"

say ""
say "Resumen: $nvar variable(s), $nsec secret(s)$([ "$nskip" -gt 0 ] && echo ", $nskip vacío(s) omitido(s)")."
[ "$DRY" = 1 ] && say "(dry-run — nada se aplicó realmente)"
say "Recuerda repetir por environment (--env dev / --env prod) con los valores de cada uno."
