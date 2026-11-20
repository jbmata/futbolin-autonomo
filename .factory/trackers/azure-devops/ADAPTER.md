# Tracker Adapter: Azure DevOps Work Items

> **Adapter del tracker para la Software Factory.** Traduce las 6 operaciones de espejado abstractas
> de `factory-sync` en llamadas concretas al MCP oficial `microsoft/azure-devops-mcp`.
> Este archivo es la fuente de verdad para el eje `tracker.platform: azure-devops`.

- **Status:** proven
- **Last updated:** 2026-06-07
- **Platform:** Azure DevOps Work Items (via `microsoft/azure-devops-mcp`)
- **Auth:** Azure CLI (`az login`) o interactive (browser / device-code) — sin PAT en fichero
- **Loaded by:** `factory-sync`, `factory-build` (step 4/8), `factory-status`

---

## 0. Prerrequisitos / Prerequisites

| Requisito | Verificación |
|---|---|
| MCP `azure-devops` registrado en `.mcp.json` (Claude Code + Copilot CLI) | Ver Sección 1 |
| MCP `azure-devops` registrado en `opencode.json` (opencode) | Ver Sección 1 |
| Auth activa (Azure CLI / interactive) | `core_list_projects` devuelve lista sin error |
| `.factory/config.yml` con bloque `tracker.azure_devops` | Ver Sección 2 |
| `.factory/issue-map.json` inicializado como `{}` | Creado por `factory-init` |

---

## 1. MCP config — lo que `setup-tracker.sh azure-devops` escribe

El paquete oficial es **`@azure-devops/mcp`** (npm). Toma el **nombre** del org como argumento
posicional, más `-a <auth>` y `-d <domains>`. Claude Code y opencode usan **schemas MCP distintos**
— por eso se escriben los dos (requisito dual-tool).

### 1a · Claude Code + Copilot CLI — `.mcp.json` (schema: `mcpServers` → `command` + `args`)

`setup-tracker.sh` hace un **merge no destructivo** (preserva otras entradas):

```json
{
  "mcpServers": {
    "azure-devops": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@azure-devops/mcp@latest", "<org>", "-a", "interactive", "-d", "all"]
    }
  }
}
```

### 1b · opencode — `opencode.json` (schema: `type: local` + `command` ARRAY)

Merge no destructivo en `opencode.json` (raíz del repo). **Ojo:** opencode usa `command` como array
completo y `enabled`, no `args`:

```json
{
  "mcp": {
    "azure-devops": {
      "type": "local",
      "command": ["npx", "-y", "@azure-devops/mcp@latest", "<org>", "-a", "interactive", "-d", "all"],
      "enabled": true
    }
  }
}
```

Reemplaza `<org>` con `tracker.azure_devops.org` (el **nombre** del org, p.ej. `contoso` — no la URL).
`-d all` habilita todos los dominios de tools (work-items incluidos); puedes acotarlo a
`-d "work work-items core"`. **Nunca escribir un PAT aquí ni en config.**

### 1c · Auth — sin fichero de secretos

`@azure-devops/mcp` se autentica con la **Azure CLI** o de forma **interactiva** (`-a interactive`):

- **Azure CLI (recomendado):** `az login` una vez; el MCP reutiliza esa credencial. CI: `az login`
  con service principal / managed identity.
- **Interactiva:** con `-a interactive` el MCP lanza un prompt de login (browser / device-code) la
  primera vez que se usa una tool.

**No hace falta un PAT en fichero** — por eso este adapter **no** usa `secrets.env`. No escribas
tokens en `mcp.json`, `opencode.json` ni `config.yml`.

### 1d · Verificación de la conexión

Tras configurar, verifica que el MCP responde correctamente:

```
# En Claude Code / opencode — carga el tool via ToolSearch si no está ya en contexto:
ToolSearch: select:mcp__azure-devops__core_list_projects

# Llama sin argumentos:
core_list_projects {}
```

Debe devolver la lista de proyectos de la organización. Si falla con 401/403, ejecuta `az login`
(o re-lanza el prompt interactivo) y reintenta.

---

## 2. Identidad del tracker en `config.yml`

```yaml
tracker:
  platform: azure-devops
  azure_devops:
    org: "myorg"                 # parte de https://dev.azure.com/<org>
    project: "MyProject"        # nombre exacto del project (sensible a mayúsculas)
    area_path: "MyProject\\Backend"   # opcional; si vacío usa el area raíz del project
```

El adapter lee estos tres valores en cada operación. `area_path` vacío = raíz del proyecto.

---

## 3. Mapa de verbos — las 6 operaciones de espejado

`factory-sync` llama operaciones abstractas. Este adapter las traduce así:

| Operación factory | Tool MCP | Detalle |
|---|---|---|
| `ensure-types` | `wit_list_work_item_types` + (sin crear tipos) | Ver §3.1 |
| `create` | `wit_create_work_item` | Ver §3.2 |
| `update` | `wit_update_work_item` | Ver §3.3 |
| `close` | `wit_update_work_item` (State → Done/Closed) | Ver §3.4 |
| `read-back` | `wit_my_work_items` / `wit_get_work_items_batch_by_ids` | Ver §3.5 |
| `blocked` | `wit_update_work_item` (tag/State Blocked) | Ver §3.6 |

> **Runtime note:** los tools MCP se descubren en ejecución vía `ToolSearch`.
> Antes de la primera llamada en una sesión, carga el schema con:
> `ToolSearch: select:mcp__azure-devops__<tool_name>`

---

### 3.1 `ensure-types` — Verificar Work Item Types y tag `factory`

Azure DevOps define Work Item Types a nivel de **process template** (Agile, Scrum, CMMI…).
**No se crean tipos nuevos** — se usan los estándar y se mapean:

| Planning key | Work Item Type (Azure DevOps) | Tipo de elemento |
|---|---|---|
| `phase-N` (fase) | `Epic` o `Feature` | Contenedor de fase |
| `phase-N.task-M` (tarea) | `Task` | Ítem de trabajo atómico |

El tipo concreto (`Epic` vs `Feature`) se elige según el process template del proyecto:
- **Agile / Scrum:** fase → `Feature`, tarea → `Task`
- **CMMI:** fase → `Feature`, tarea → `Task`
- Si se prefiere agrupar por épicas: fase → `Epic`, sub-fase → `Feature`, tarea → `Task`

Anota el tipo elegido en `.factory/config.yml` bajo `tracker.azure_devops.phase_type` (default: `Feature`).

```yaml
tracker:
  azure_devops:
    phase_type: "Feature"   # Epic | Feature — según el process template del proyecto
    task_type:  "Task"
```

Todos los Work Items creados por la factoría llevan el tag `factory`. El adapter comprueba que el
project responde antes de operar: llama `wit_list_work_item_types` y verifica que el tipo de fase y
tarea existen; si no, alerta al usuario y para.

```
ToolSearch: select:mcp__azure-devops__wit_list_work_item_types

wit_list_work_item_types {
  "project": "<tracker.azure_devops.project>"
}
# → array de tipos; verificar que "Feature" (o "Epic") y "Task" están presentes
```

---

### 3.2 `create` — Crear Work Item (fase o tarea)

**Paso 1 — comprobar idempotencia:** buscar la `planning_key` en `.factory/issue-map.json`.
Si ya existe → no crear; ir a `update`.

**Paso 2 — crear el Work Item:**

```
ToolSearch: select:mcp__azure-devops__wit_create_work_item

# Fase (Feature/Epic):
wit_create_work_item {
  "project":  "<tracker.azure_devops.project>",
  "type":     "<tracker.azure_devops.phase_type>",
  "title":    "<phase title>",
  "description": "<phase goal / summary>",
  "areaPath": "<tracker.azure_devops.area_path>",
  "tags":     "factory"
}

# Tarea (Task) — con parent link a la fase:
wit_create_work_item {
  "project":  "<tracker.azure_devops.project>",
  "type":     "Task",
  "title":    "<task title>",
  "description": "<task body>",
  "areaPath": "<tracker.azure_devops.area_path>",
  "tags":     "factory",
  "parentId": <work_item_id_de_la_fase_padre>
}
```

`parentId` toma el ID numérico de la fase padre (ya en `issue-map.json`). El link que se crea es
de tipo `System.LinkTypes.Hierarchy-Reverse` (Child of), que es el vínculo padre-hijo estándar en
Azure DevOps.

**Paso 3 — guardar en `issue-map.json`:**

```json
{
  "phase-1": { "platform": "azure-devops", "id": 42, "url": "https://dev.azure.com/org/project/_workitems/edit/42" },
  "phase-1.task-3": { "platform": "azure-devops", "id": 57, "url": "https://dev.azure.com/org/project/_workitems/edit/57" }
}
```

Formato neutral `{ "platform": "azure-devops", "id": <int>, "url": "<string>" }`.

---

### 3.3 `update` — Actualizar título / descripción / tags

```
ToolSearch: select:mcp__azure-devops__wit_update_work_item

wit_update_work_item {
  "id":          <work_item_id>,
  "title":       "<nuevo título si cambió>",
  "description": "<nuevo cuerpo si cambió>",
  "tags":        "factory"
}
```

Solo actualiza los campos que hayan derivado. Compara el estado actual (via read-back §3.5) antes
de escribir para evitar escrituras no necesarias.

---

### 3.4 `close` — Marcar como completado

```
wit_update_work_item {
  "id":    <work_item_id>,
  "state": "Done"
}
```

Si el process template usa estados distintos (e.g. Scrum usa `Done`; Agile usa `Closed`),
el adapter detecta el estado final correcto en `ensure-types` (buscando el estado que es `completed`
en el ciclo de vida del tipo) y lo cachea en config. Default: `"Done"` para Agile/Scrum.

No borrar el Work Item — solo cambiar su estado.

---

### 3.5 `read-back` — Leer estado actual

**Por IDs conocidos** (de `issue-map.json`):

```
ToolSearch: select:mcp__azure-devops__wit_get_work_items_batch_by_ids

wit_get_work_items_batch_by_ids {
  "ids":    [42, 57, 83],
  "fields": ["System.Id","System.Title","System.State","System.Tags","System.WorkItemType"]
}
```

**Fallback — elementos propios del agente** (cuando el mapa está vacío o incompleto):

```
ToolSearch: select:mcp__azure-devops__wit_my_work_items

wit_my_work_items {
  "project": "<tracker.azure_devops.project>"
}
# Filtra por tag "factory" en los resultados
```

El adapter refleja cambios humanos: si un Work Item está en `Done`/`Closed` en Azure pero no en
`.planning/` → lo anota en el reporte de `factory-sync` y, solo si el usuario lo pide, propaga el
cierre al plan. Si está en `Blocked` → lo reporta como bloqueador.

---

### 3.6 `blocked` — Marcar como bloqueado

Azure DevOps no tiene un estado `Blocked` universal, pero sí el tag y (en Agile) el campo
`Blocked` (Yes/No). El adapter usa **ambas vías** para máxima visibilidad:

```
wit_update_work_item {
  "id":    <work_item_id>,
  "tags":  "factory; blocked",
  "state": "Active"
}
```

Si el process template tiene un estado explícito `Blocked` (personalizado), se usa en lugar de
solo el tag. Al desbloquear, elimina el tag `blocked` y deja el estado en `Active`.

En `factory-build`, al detectar un bloqueador, este adapter también crea un comentario en el Work
Item con el contexto del error (sin secretos).

---

## 4. Idempotencia y guardrails

Las mismas reglas que el adapter de GitHub — **no relajar ninguna:**

| Guardrail | Implementación |
|---|---|
| No duplicar | Siempre consultar `issue-map.json` antes de crear |
| Bulk confirm | Si se van a crear > 10 Work Items de golpe, pedir confirmación al usuario |
| No secretos | Títulos, descripciones y comentarios NUNCA deben contener tokens, contraseñas o claves |
| No borrar | Al eliminar una fase del roadmap → cerrar el WI con comentario "dropped", no borrar |
| Unidireccional | `.planning/` → Azure DevOps es la dirección por defecto; read-back es solo lectura salvo que el usuario pida lo contrario |
| Resiliente | Si el MCP no responde, loguear el error y continuar con las operaciones que sí funcionen; reportar al final |

---

## 5. Flujo completo de ejemplo

```
factory-sync con tracker: azure-devops
─────────────────────────────────────────────────────────
[1] Resolver adapter  ← leer config.yml, cargar este ADAPTER.md
[2] ensure-types      ← wit_list_work_item_types → confirmar Feature + Task existen
[3] Leer roadmap      ← .planning/ROADMAP.md → planning keys + titles + status
[4] Reconciliar
    phase-1     → en map? No  → wit_create_work_item(type=Feature, tags=factory)
                              → map: {phase-1: {platform:azure-devops, id:101, url:…}}
    phase-1.task-1 → No  → wit_create_work_item(type=Task, parentId=101, tags=factory)
                              → map: {phase-1.task-1: {id:102, …}}
    phase-1.task-2 → Sí  → wit_update_work_item si drift en título/body
    phase-2     → Sí, done en .planning → wit_update_work_item(state=Done)
[5] Read-back         ← wit_get_work_items_batch_by_ids([101,102,…])
                      → refleja: 103 está Blocked → reportar bloqueador
[6] Persist           ← escribir .factory/issue-map.json actualizado
[7] Reporte
    ✓ Creados: 2  Actualizados: 1  Cerrados: 1  Bloqueadores: 1
    🔗 https://dev.azure.com/myorg/MyProject/_workitems
```

---

## 6. Gotchas y lecciones

- **Process templates importan:** Agile, Scrum y CMMI tienen estados distintos (`Closed` vs `Done`).
  Siempre detectar el estado final correcto en `ensure-types`.
- **`parentId` requiere que la fase exista primero:** crear fases antes que tareas en la
  reconciliación (ordenar por nivel: fases → tareas).
- **Tags son strings libres:** `"factory; blocked"` funciona; no usar comas — Azure los separa con
  punto y coma.
- **AreaPath case-sensitive:** `"MyProject\\Team A"` != `"myproject\\team a"`. Copiar exacto del UI.
- **Rate limiting:** Azure DevOps limita a ~200 req/min en plans grandes; en bulk, añadir delay de
  ~300 ms entre creates. `wit_get_work_items_batch_by_ids` acepta hasta 200 IDs por llamada.
- **Permisos mínimos del PAT:** `Work Items (Read & Write)`, `Project and Team (Read)`.
  Con Entra, el scope equivalente es `499b84ac-1321-427f-aa17-267ca6975798/.default`.
- **`wit_my_work_items` solo devuelve WIs asignados al usuario actual:** para WIs de equipo, usar
  la consulta WIQL o `wit_get_work_items_batch_by_ids` con los IDs del mapa.
- **URLs de WI:** formato `https://dev.azure.com/<org>/<project>/_workitems/edit/<id>`.
  El MCP devuelve `url` en la respuesta de `wit_create_work_item`; usarla directamente.

---

## 7. Changelog

- 2026-06-07 — Playbook inicial: 6 operaciones completas, MCP config dual-target (CC + opencode),
  idempotencia, guardrails, gotchas, ejemplo de flujo.
