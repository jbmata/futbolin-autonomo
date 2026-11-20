# SDD Adapter: openspec

> Playbook para usar **OpenSpec** (Fission-AI) como motor SDD de la Software Factory.
> Cárgalo cuando `sdd.engine: openspec` esté en `.factory/config.yml`.
> The next agent (or developer) should be able to install, configure, and operate OpenSpec from this file alone.

- **Status:** proven
- **Last updated:** 2026-06-24
- **Spec:** docs/superpowers/specs/2026-06-24-openspec-engine-copilot-runtime-design.md §A

---

## 1. Contrato de 5 verbos / 5-verb contract

`factory-build` invoca verbos abstractos. Este adapter los traduce al CLI de OpenSpec (driving
**headless**, sin slash commands — así corre igual en Claude Code, opencode y Copilot CLI).

| Verbo factory | Mapeo OpenSpec (CLI) | Notas |
|---|---|---|
| `discover` | `openspec new change <phase-slug>` → `openspec instructions proposal --change <phase-slug> --json` y `openspec instructions specs --change <phase-slug> --json` | Crea la carpeta de cambio y siembra `proposal.md` (qué/por qué) + deltas `specs/`. En modo autónomo pasa `discovery.md`/intake como contexto; en ultra-autónomo no pregunta y anota los supuestos en `PENDIENTES.md`. |
| `plan` | `openspec instructions design --change <phase-slug> --json` | Escribe `design.md` (el cómo). |
| `decompose` | `openspec instructions tasks --change <phase-slug> --json` | Escribe `tasks.md` (checklist `- [ ]`). |
| `execute` | `openspec instructions apply --change <phase-slug> --json` | Implementa tareas una a una, commits atómicos, marca checkboxes. |
| `verify` | `openspec validate <phase-slug> --strict` **+** gate compuesto de la factoría → `openspec archive <phase-slug> --yes` | OpenSpec no trae verbo de "tests de código": la verificación de comportamiento la hace el gate compuesto (§5). `archive` SOLO tras validación + gate OK. |
| **Source of truth** | `openspec/changes/<phase-slug>/{proposal,design,tasks}.md` + deltas `specs/`; canónico acumulado en `openspec/specs/` | Equivale a `.planning/ROADMAP.md` (gsd) / `specs/<feature>/` (spec-kit). |

**Driving híbrido:** el bucle autónomo usa el CLI (arriba). Interactivamente, un humano puede usar los
skills/commands `/opsx:*` que `openspec init` instala (p.ej. `/opsx:propose`, `/opsx:apply`,
`/opsx:archive`) para una UX más rica. Misma fuente de verdad.

---

## 2. Fuente de verdad y layout / Source of truth

```
openspec/
  specs/<domain>/spec.md          ← canónico (comportamiento actual; acumula al archivar)
  changes/<phase-slug>/
    proposal.md                   ← /discover  (qué y por qué)
    design.md                     ← /plan      (cómo)
    tasks.md                      ← /decompose (checklist accionable)
    specs/<domain>/spec.md        ← deltas: ## ADDED / MODIFIED / REMOVED Requirements
  changes/archive/[date-<phase-slug>]/   ← cambios completados (histórico)
  project.md                      ← contexto de proyecto que el agente lee (opcional)
```

**Formato de spec (hardcodeado en OpenSpec — generar EXACTO):**
- Requisito: `### Requirement: <nombre>` + verbos RFC 2119 (SHALL/MUST/SHOULD/MAY).
- Escenario: `#### Scenario: <nombre>` + viñetas `- GIVEN / WHEN / THEN / AND`.
- Delta: `## ADDED Requirements`, `## MODIFIED Requirements`, `## REMOVED Requirements`
  (MODIFIED reescribe el requisito completo; REMOVED solo lo nombra).

Reglas:
- Cada `### Requirement:` tiene **≥1** `#### Scenario:` (lo exige `openspec validate`).
- Nunca editar `openspec/specs/**` a mano: se actualiza solo al `archive` (merge de deltas).

---

## 3. Mapa de roadmap / Roadmap map

OpenSpec es change-céntrico; la factoría razona por fases. Equivalencia 1:1 (como spec-kit):

```
phase-1  ←→  openspec/changes/phase-1-<slug>/
phase-N  ←→  openspec/changes/phase-N-<slug>/
```

- **Clave de planning estable:** `phase-N` / `phase-N.task-M` (tareas parseadas de los checkboxes de
  `tasks.md`). `issue-map.json` y el tracker NO cambian.
- No hay `ROADMAP.md`: la lista de fases vive en `FACTORY.md`/`discovery.md`; cada fase se materializa
  como un `change` al construirse.
- `factory-sync` lee `openspec/changes/<phase>/tasks.md` de la fase activa y proyecta tareas a issues
  con la clave `phase-N.task-M`.

```jsonc
// .factory/issue-map.json — entrada típica con openspec
{ "phase-1": { "platform": "github", "id": 42, "url": "…" },
  "phase-1.task-3": { "platform": "github", "id": 45, "url": "…" } }
```

---

## 4. Instalación / INSTALL

**Requisito:** Node ≥ 20.19.0. Paquete: `@fission-ai/openspec`.

### 4a. Fuente pública (default) — multi-runtime

```bash
npm install -g @fission-ai/openspec@latest
openspec init --tools "<lista>"   # p.ej. claude,opencode,github-copilot
```

`<lista>` se deriva de los runtimes de la factoría (Claude Code, opencode, Copilot) o de
`sdd.openspec.tools` si está fijado. `openspec init` es idempotente (refresca sin borrar tu config).

### 4b. Fork o fuente privada

Si `sdd.openspec.source` tiene valor:

```bash
SRC="$(yq '.sdd.openspec.source' .factory/config.yml)"
# git+ssh://  → usa la clave SSH del sistema:
npm install -g "$SRC"
# git+https:// → inyecta GSD_GIT_TOKEN desde .factory/sdd/secrets.env:
source .factory/sdd/secrets.env
npm install -g "$(echo "$SRC" | sed "s|https://|https://${GSD_GIT_TOKEN}@|")"
# npm privado → npm install -g "$SRC"
openspec init --tools "<lista>"
```

---

## 5. Verbo `verify` — gate compuesto + archive / Composite verify gate

```bash
# 1. Validación estructural de OpenSpec
openspec validate <phase-slug> --strict     # exit 1 si falla
# 2. Tests del stack
bash .factory/assets/scripts/test-gate.sh run   # gates.tests.cmd / coverage_min; sin cmd, el test_cmd del starterkit
# 3. Docker gate (si gates.docker_run: true)
docker compose up --build -d
curl -sf "http://localhost:${PORT:-8000}/health" || exit 1
docker compose down
# 4. Solo si 1–3 pasan: archivar (merge de deltas a openspec/specs/)
openspec archive <phase-slug> --yes
```

Si algún paso falla: capturar salida, volver a `execute`, reintentar (máx. 3 sobre el mismo fallo —
stop conditions de `factory-build`). **Nunca** `archive` con validación o gate en rojo.

---

## 6. Validación / Validation

| Comando | Acción |
|---|---|
| `openspec validate --all` | Valida todos los changes y specs |
| `openspec validate <name> --strict` | Validación reforzada de un change (usar en el gate) |
| `openspec list [--changes\|--specs]` | Lista changes activos o specs |
| `openspec show <name> [--json]` | Muestra un change/spec |
| `openspec status [--change <id>] [--json]` | Estado de completitud de artefactos |
| `openspec instructions <proposal\|specs\|design\|tasks\|apply> --change <id> --json` | Instrucciones machine-readable para generar/aplicar un artefacto. **`--change <id>` es OBLIGATORIO** — el CLI no auto-detecta el change aunque solo haya uno (lanza error si falta). |

Flags útiles para CI/headless: `--json`, `--no-interactive`. `DO_NOT_TRACK=1` u `OPENSPEC_TELEMETRY=0`
desactivan telemetría.

---

## 7. Variables y secretos / Credentials

| Variable | Requerida | Para qué |
|---|---|---|
| `GSD_GIT_TOKEN` | Solo con `openspec.source` privado vía `git+https://` | PAT para `npm install -g` del fork privado |

Orígenes `git+ssh://` usan la clave SSH del sistema (sin variable). Nunca en `config.yml` ni en issues.

---

## 8. Gotchas y lecciones / Gotchas & lessons

- **Sin MCP, sin API keys:** OpenSpec es solo CLI + Markdown. No añade MCP (a diferencia del tracker).
- **Node ≥ 20.19.0:** verificar antes de instalar; avisar si falta.
- **`instructions` exige `--change <phase-slug>`:** el CLI **no** auto-detecta el change activo (lanza
  `Missing required option --change` aunque solo exista uno). Pasa siempre `--change <phase-slug>` (el
  mismo slug de `openspec new change`). `validate`/`archive` sí aceptan el slug **posicional**.
- **Deltas, no rewrites:** un change registra solo ADDED/MODIFIED/REMOVED relativo a `openspec/specs/`.
  No reescribir el spec entero; el merge ocurre en `archive`.
- **`archive` es el commit semántico:** solo tras validación + gate OK. Es irreversible (mueve a
  `changes/archive/` y modifica `openspec/specs/`).
- **Slug vs clave de planning:** el slug de carpeta (`phase-1-auth`) puede cambiar; la clave
  (`phase-1`) es el ancla de `issue-map.json` — mantenerlas separadas.
- **`openspec init` por herramienta:** emite skills/commands por runtime; en Copilot CLI los
  `/opsx-*` (prompts) no son invocables (son de VS Code Chat), por eso el bucle usa el CLI.

---

## 9. Testing / Verificación del adapter

```bash
openspec --version                            # CLI disponible (Node ≥ 20.19)
mkdir -p /tmp/osp-probe && (cd /tmp/osp-probe && git init -q && openspec init --tools claude && openspec validate --all)
openspec list --changes --json                # parseable
```

---

## 10. Changelog

- 2026-06-24 — Playbook inicial: contrato de 5 verbos (CLI headless + híbrido `/opsx:*`), layout y
  formato de spec, mapa de roadmap, instalación pública/privada, gate `verify`+`archive`, validación.
