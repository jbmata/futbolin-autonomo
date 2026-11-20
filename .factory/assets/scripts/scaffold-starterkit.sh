#!/usr/bin/env bash
# Scaffold a greenfield project by pulling (cloning) a starterkit into the target dir.
# Used by factory-init step 5 after the user chooses FastAPI, OpenSaaS, Frappe or go-service.
#
# Usage:  scaffold-starterkit.sh <fastapi|opensaas|frappe|go-service|URL> [TARGET_DIR]
#
# Copies the template's files into TARGET WITHOUT overwriting existing files
# (so the factory's .claude/.opencode/.factory/QUICKSTART and the repo's own .git
# are preserved) and WITHOUT bringing the template's git history.
set -uo pipefail

ID="${1:-}"; TARGET="${2:-$PWD}"
[ -z "$ID" ] && { echo "uso: scaffold-starterkit.sh <fastapi|opensaas|frappe|go-service|URL> [DIR]"; exit 2; }
TARGET="$(cd "$TARGET" 2>/dev/null && pwd)" || { echo "target dir not found"; exit 1; }

case "$ID" in
  fastapi)  REPO="https://github.com/fastapi/full-stack-fastapi-template"; SUBDIR="" ;;
  opensaas) REPO="https://github.com/wasp-lang/open-saas";                 SUBDIR="auto" ;;
  frappe)   REPO="https://github.com/frappe/frappe_docker";                SUBDIR="" ;;
  go-service) REPO="https://github.com/Fastex-Soluciones-IT/starterkit-go-service"; SUBDIR="" ;;
  http*://*) REPO="$ID"; SUBDIR="" ;;
  *) echo "starterkit desconocido: $ID  (usa fastapi | opensaas | frappe | go-service | una URL git)"; exit 2 ;;
esac

command -v git >/dev/null || { echo "git no encontrado"; exit 1; }
echo "Scaffolding '$ID' → $TARGET"
echo "  fuente: $REPO"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
git clone --depth 1 "$REPO" "$TMP/src" >/dev/null 2>&1 || {
  echo "  ✗ clone falló ($REPO)"
  echo "    si el repo es privado (go-service lo es): gh auth login && gh auth setup-git, con acceso de lectura al repo"
  exit 1; }
rm -rf "$TMP/src/.git"                       # never bring the template's history

SRC="$TMP/src"
if [ "$SUBDIR" = "auto" ] && [ -d "$TMP/src/template/app" ]; then
  SRC="$TMP/src/template/app"; echo "  usando subdir template/app (layout OpenSaaS/Wasp)"
elif [ -n "$SUBDIR" ] && [ -d "$TMP/src/$SUBDIR" ]; then
  SRC="$TMP/src/$SUBDIR"
fi

# Copy without clobbering anything that already exists in TARGET.
if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude='.git' --ignore-existing "$SRC"/ "$TARGET"/
else
  cp -rn "$SRC"/. "$TARGET"/ 2>/dev/null || true
fi

echo "  ✓ scaffolding hecho (archivos existentes preservados)"
echo ""
echo "Siguientes pasos:"
echo "  - Revisa README/.env del template y copia .env.example → .env"
if [ "$ID" = opensaas ]; then
  echo "  - OpenSaaS corre con Wasp:  wasp start   (no con el gate docker-compose por defecto)"
elif [ "$ID" = frappe ]; then
  echo "  - Frappe (frappe_docker): levanta el stack con  docker compose -f pwd.yml up -d"
  echo "    crea el sitio con bench y añade tu app:  bench get-app <repo> && bench --site <site> install-app <app>"
elif [ "$ID" = go-service ]; then
  echo "  - Sin Docker:  make run   (Go 1.26+ y Node 22+);  con Docker:  docker compose up -d --build"
  echo "  - Comprueba:   curl localhost:8080/health"
else
  echo "  - Levanta en local:  docker compose up -d   (o /factory-run)"
fi
