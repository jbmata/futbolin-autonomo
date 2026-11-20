# Tracker Adapter: GitHub Issues (via `gh` CLI)

> Playbook concreto para el backend `github` del eje de tracking de la Software Factory.
> `factory-sync` y cualquier skill de orquestación cargan **este archivo** cuando
> `tracker.platform: github` en `.factory/config.yml`.  
> No contiene secretos — la auth va en el shell (`gh auth status`).

- **Status:** proven
- **Last updated:** 2026-06-07
- **Used in:** repos con `tracker.platform: github`

---

## 1. Propósito / When to use

Este adapter traduce las 6 operaciones abstractas de espejado de la factoría en comandos `gh`
concretos. El resultado es un espejo de issues en GitHub que sigue el roadmap/planning (fuente de
verdad en SDD), con idempotencia garantizada vía `.factory/issue-map.json`.

**Cuándo aplica:** siempre que `.factory/config.yml` contenga:

```yaml
tracker:
  platform: github
  github:
    repo: <owner/name>   # identidad del adapter
```

---

## 2. Identidad y configuración mínima

| Campo config | Uso |
|---|---|
| `tracker.github.repo` | Repositorio destino (`owner/name`). Requerido. |
| `tracker.platform` | Debe ser `github` para que este adapter se cargue. |

El campo `tracker.github.repo` es la única configuración que este adapter necesita. No hay
variables de entorno propias: la auth la gestiona `gh` (ver sección 3).

**Compatibilidad hacia atrás:** si el config solo tiene el bloque antiguo `github.repo` sin la
clave `tracker:`, leer `github.repo` como fallback.

---

## 3. Auth / prerequisitos

### Verificar auth

```bash
gh auth status
```

Si el comando falla o devuelve "not logged in":

1. Pedir al usuario que abra una terminal y ejecute:
   ```bash
   gh auth login
   ```
   (seleccionar GitHub.com → HTTPS → authenticate with browser)
2. Volver a verificar con `gh auth status`.
3. Si sigue fallando, **detener** y mostrar el error. No continuar con operaciones de escritura.

### Verificar acceso al repo

```bash
gh repo view <owner/name> --json name,isPrivate
```

Si falla (404 / auth insuficiente), notificar al usuario y detener.

---

## 4. Shape de `issue-map.json` (neutral)

Cada entrada en `.factory/issue-map.json` sigue el contrato neutral de la factoría:

```json
{
  "phase-1": {
    "platform": "github",
    "id": 42,
    "url": "https://github.com/owner/repo/issues/42"
  },
  "phase-1.task-3": {
    "platform": "github",
    "id": 57,
    "url": "https://github.com/owner/repo/issues/57"
  }
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `platform` | string | Siempre `"github"` para este adapter. |
| `id` | integer | Número de issue de GitHub (`#N`). |
| `url` | string | URL canónica del issue. |

**Clave de planning:** string estable derivada del SDD activo. Ejemplos:
- gsd: `phase-1`, `phase-2`, `phase-1.task-3`
- spec-kit: `feature-auth`, `feature-auth.task-2`

La clave nunca cambia una vez asignada; es el join entre el planning y el issue-map.

---

## 5. Las 6 operaciones

### 5.1 `ensure-types` — Labels y milestone

Crear labels y milestone si no existen. **Idempotente** (usar `--force`).

```bash
REPO="<tracker.github.repo>"

# Labels (colores distinctivos, --force actualiza si ya existe)
gh label create "factory:phase"   --color "0075ca" --description "Factory phase (epic)" --force --repo "$REPO"
gh label create "factory:task"    --color "e4e669" --description "Factory task"         --force --repo "$REPO"
gh label create "factory:blocked" --color "d93f0b" --description "Blocked item"         --force --repo "$REPO"
gh label create "factory:done"    --color "0e8a16" --description "Completed item"       --force --repo "$REPO"

# Milestone — derivar nombre del milestone activo en el SDD (ej. "M1 — Foundation")
MILESTONE_TITLE="<milestone-name-from-sdd>"

# Crear si no existe; capturar su número para usar en --milestone al crear issues
MILESTONE_NUMBER=$(gh api "repos/$REPO/milestones" \
  --jq ".[] | select(.title == \"$MILESTONE_TITLE\") | .number" 2>/dev/null)

if [ -z "$MILESTONE_NUMBER" ]; then
  MILESTONE_NUMBER=$(gh api "repos/$REPO/milestones" \
    --method POST \
    --field title="$MILESTONE_TITLE" \
    --field state=open \
    --jq ".number")
fi
```

**Cuándo ejecutar:** al inicio de cada sync, antes de crear o actualizar issues.

---

### 5.2 `create` — Crear issue

Llamar solo si la clave de planning **no existe** en `issue-map.json`.

```bash
# Phase issue (epic)
ISSUE_URL=$(gh issue create \
  --repo "$REPO" \
  --title "[Phase N] <título-de-la-fase>" \
  --body "<descripción-de-la-fase>

**Planning key:** phase-N
**Source of truth:** .planning/ (SDD)" \
  --label "factory:phase" \
  --milestone "$MILESTONE_TITLE" \
  --json url --jq ".url")

ISSUE_NUMBER=$(echo "$ISSUE_URL" | grep -o '[0-9]*$')

# Task issue (child)
ISSUE_URL=$(gh issue create \
  --repo "$REPO" \
  --title "<título-de-la-tarea>" \
  --body "<descripción-de-la-tarea>

**Planning key:** phase-N.task-M
**Parent:** #<phase-issue-number>" \
  --label "factory:task" \
  --milestone "$MILESTONE_TITLE" \
  --json url --jq ".url")
```

**Guardrail:** si se van a crear >10 issues en un solo sync, confirmar con el usuario antes de
proceder. No continuar si el usuario dice que no.

**Anti-dup:** comprobar siempre `issue-map.json` antes de llamar a `gh issue create`. Si la clave
ya tiene un `id`, usar `update` en vez de `create`.

**No secretos:** títulos y cuerpos nunca contienen tokens, contraseñas ni credenciales.

---

### 5.3 `update` — Actualizar issue existente

Llamar cuando el planning key existe en el mapa pero el título/body/labels han derivado.

```bash
ISSUE_NUMBER=<id-del-mapa>

gh issue edit "$ISSUE_NUMBER" \
  --repo "$REPO" \
  --title "<nuevo-título>" \
  --body "<nuevo-body>"

# Re-aplicar labels si faltan (edit no quita labels existentes)
gh issue edit "$ISSUE_NUMBER" --repo "$REPO" --add-label "factory:phase"
```

**Nota:** `gh issue edit` es seguro de ejecutar si nada cambió (es idempotente en la práctica).
Para evitar edits innecesarios, comparar título y body con el read-back antes de llamar.

---

### 5.4 `close` — Cerrar issue completado

Llamar cuando el item está marcado como done en el SDD.

```bash
ISSUE_NUMBER=<id-del-mapa>

gh issue close "$ISSUE_NUMBER" \
  --repo "$REPO" \
  --comment "Cerrado automáticamente por factory-sync: item completado en el planning."

gh issue edit "$ISSUE_NUMBER" \
  --repo "$REPO" \
  --add-label "factory:done"
```

**Item eliminado del roadmap (no completado):** cerrar con comentario distinto que indique que fue
eliminado del scope, no completado.

```bash
gh issue close "$ISSUE_NUMBER" \
  --repo "$REPO" \
  --comment "Eliminado del scope en el planning. Cerrado por factory-sync."
```

No borrar la entrada de `issue-map.json` — preservar el histórico.

---

### 5.5 `read-back` — Leer estado actual

Obtener estado de todos los issues de la factoría para detectar cambios hechos por humanos.

```bash
# Leer phases
gh issue list \
  --repo "$REPO" \
  --label "factory:phase" \
  --state all \
  --json number,title,state,labels,url \
  --limit 200

# Leer tasks
gh issue list \
  --repo "$REPO" \
  --label "factory:task" \
  --state all \
  --json number,title,state,labels,url \
  --limit 200
```

**Interpretar el read-back:**

| Estado del issue | Labels adicionales | Acción sugerida |
|---|---|---|
| `closed` + `factory:done` | — | Item completado: ya sincronizado. |
| `closed` sin `factory:done` | — | Cerrado manualmente: registrar en el reporte. |
| `open` + `factory:blocked` | — | Marcar como bloqueado en el reporte al usuario. |
| `open` | — | Normal. |

El read-back es **informativo** por defecto. No reescribir `.planning/` desde los issues salvo que el
usuario lo pida explícitamente.

---

### 5.6 `blocked` — Marcar item bloqueado

Aplicar cuando el planning detecta un bloqueo (dependencia no resuelta, decisión pendiente, etc.).

```bash
ISSUE_NUMBER=<id-del-mapa>

gh issue edit "$ISSUE_NUMBER" \
  --repo "$REPO" \
  --add-label "factory:blocked"

# Comentar el motivo (sin secretos)
gh issue comment "$ISSUE_NUMBER" \
  --repo "$REPO" \
  --body "**Bloqueado:** <motivo-del-bloqueo>"
```

Para desbloquear (cuando el bloqueo se resuelve):

```bash
gh issue edit "$ISSUE_NUMBER" \
  --repo "$REPO" \
  --remove-label "factory:blocked"
```

---

## 6. Flujo completo de sync (orden de operaciones)

```
1. Verificar auth         → gh auth status
2. Verificar repo         → gh repo view <repo>
3. ensure-types           → labels + milestone (idempotente)
4. Leer issue-map.json    → mapa actual de claves → ids
5. Para cada item del planning:
   a. Si clave no en mapa → create (con guardrail >10)
   b. Si clave en mapa, abierto, contenido derivó → update
   c. Si clave en mapa, done en planning → close + label factory:done
6. read-back              → detectar cambios humanos → reporte
7. blocked                → aplicar/quitar factory:blocked según planning
8. Escribir issue-map.json actualizado
9. Imprimir resumen: creados / actualizados / cerrados / bloqueados + URL del milestone
```

---

## 7. Guardrails y reglas de idempotencia

| Guardrail | Implementación |
|---|---|
| **No duplicar issues** | Siempre consultar `issue-map.json` antes de `create`. |
| **Confirmar bulk** | Si se van a crear >10 issues, pedir confirmación al usuario. |
| **No secretos** | Títulos, bodies y comentarios nunca contienen tokens ni credenciales. |
| **No borrar issues** | Cerrar con comentario; nunca usar `gh issue delete`. |
| **One-way por defecto** | `.planning/` → issues; read-back es solo lectura salvo petición explícita. |
| **Auth primero** | Si `gh auth status` falla, detener antes de cualquier escritura. |
| **Re-run seguro** | Ejecutar el flujo completo dos veces seguidas produce el mismo resultado. |

---

## 8. Ejemplos de `issue-map.json` tras un sync

```json
{
  "phase-1": {
    "platform": "github",
    "id": 1,
    "url": "https://github.com/acme/my-app/issues/1"
  },
  "phase-1.task-1": {
    "platform": "github",
    "id": 2,
    "url": "https://github.com/acme/my-app/issues/2"
  },
  "phase-1.task-2": {
    "platform": "github",
    "id": 3,
    "url": "https://github.com/acme/my-app/issues/3"
  },
  "phase-2": {
    "platform": "github",
    "id": 4,
    "url": "https://github.com/acme/my-app/issues/4"
  }
}
```

---

## 9. Gotchas y lecciones

- `gh label create` sin `--force` falla si el label ya existe. Usar siempre `--force`.
- `gh api` para milestones devuelve un array; filtrar con `--jq` o en el script para evitar
  ambigüedades si hay varios milestones con nombre similar.
- `gh issue list --limit` por defecto es 30. Para repos con muchos issues, usar `--limit 200`
  o iterar con `--skip`.
- Los issues de GitHub no tienen campo "parent" nativo: usar la referencia `#<número>` en el body
  y el mismo milestone para agrupar phase + tasks.
- El número de issue (`id` en el mapa) es el entero al final de la URL, no el `nodeId` de GraphQL.
- Si `gh` no está en el PATH del agente en Claude Code, guiar al usuario a instalarlo:
  `https://cli.github.com/`

---

## 10. Changelog

- 2026-06-07 — Playbook inicial: las 6 operaciones completas, shape neutral de issue-map.json,
  guardrails de idempotencia, flujo de sync completo. Reproduce el comportamiento de factory-sync
  expresado como adapter intercambiable.
