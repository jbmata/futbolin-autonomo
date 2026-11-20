# SDD Adapter: spec-kit

> Playbook para usar **spec-kit** (GitHub's specification-driven development toolkit) como motor SDD
> de la Software Factory. Cárgalo cuando `sdd.engine: spec-kit` esté en `.factory/config.yml`.
>
> The next agent (or developer) should be able to install, configure, and operate spec-kit from this
> file alone.

- **Status:** proven
- **Last updated:** 2026-06-07
- **Spec:** docs/superpowers/specs/2026-06-07-pluggable-sdd-and-tracker-design.md §2a

---

## 1. Contrato de 5 verbos / 5-verb contract

`factory-build` invoca verbos abstractos. Este adapter los traduce a comandos `specify`.

| Verbo factory | Comando spec-kit | Notas |
|---|---|---|
| `discover` | `/constitution` → `/specify` | Constitution primero (baseline de calidad); `specify` genera el spec de la feature. En modo autónomo pasa la salida de discovery.md como contexto inicial; en ultra-autónomo, además, las `[NEEDS CLARIFICATION]` se resuelven con un supuesto anotado en `PENDIENTES.md` en vez de preguntar. |
| `plan` | `/plan` | Produce el plan de implementación para el spec activo. |
| `decompose` | `/tasks` | Descompone el plan en tareas accionables (spec-kit las escribe en `specs/<feature>/tasks.md`). |
| `execute` | `/implement` | Implementa las tareas una a una; usa commits atómicos. |
| `verify` | gate de tests + docker | spec-kit **no incluye un verbo `verify` propio** — la verificación se delega al gate de la factoría: `pytest` / la suite del stack + `docker compose up` + smoke check (ver §5). |
| **Source of truth** | `specs/<feature>/{spec,plan,tasks}.md` | Estos archivos son el equivalente de `.planning/ROADMAP.md` para este adapter. |

---

## 2. Fuente de verdad y layout de specs / Source of truth

```
specs/
  <feature>/
    spec.md      ← salida de /specify  (qué construir y por qué)
    plan.md      ← salida de /plan     (cómo construirlo)
    tasks.md     ← salida de /tasks    (lista accionable de tareas)
```

Reglas:
- Nunca editar `spec.md` a mano fuera del flujo `specify`. Es el documento de requisitos canónico.
- `plan.md` y `tasks.md` se pueden ajustar con `/plan` o `/tasks` si el contexto cambia.
- Los archivos se versionan en git; cada fase del roadmap tiene su propia carpeta feature.

---

## 3. Mapa de roadmap / Roadmap map

spec-kit razona por *feature*; la factoría razona por *fases de roadmap*. Este adapter define la
equivalencia uno-a-uno:

```
Fase del roadmap  ←→  carpeta specs/<feature>/
─────────────────────────────────────────────────────────
phase-1           ←→  specs/phase-1-<slug>/
phase-2           ←→  specs/phase-2-<slug>/
phase-N           ←→  specs/phase-N-<slug>/
```

**Clave de planning estable:** `phase-N` (igual que en gsd). `factory-sync` e `issue-map.json`
usan esta clave para mapear fases y tareas a issues/work items — el formato neutral no cambia.

```jsonc
// .factory/issue-map.json — entrada típica con spec-kit
{
  "phase-1": { "platform": "github", "id": 42, "url": "https://github.com/…/issues/42" },
  "phase-1.task-3": { "platform": "github", "id": 45, "url": "…" }
}
```

**Flujo de sincronización:**
1. `factory-sync` lee `specs/<feature>/tasks.md` de la fase activa.
2. Extrae las tareas (formato `- [ ] tarea` o numeradas) y las proyecta a issues usando la clave
   `phase-N.task-M`.
3. El read-back (issue cerrado, label `factory:blocked`) se refleja en `tasks.md` como comentario
   de estado — no sobreescribe el contenido generado por spec-kit.

**Slug:** se deriva del título de la fase (snake_case, máx. 40 caracteres). El slug NO forma parte
de la clave de planning — solo del nombre de carpeta para legibilidad.

---

## 4. Instalación / INSTALL

### 4a. Fuente pública (default) — dual-target (Claude Code + opencode)

spec-kit escribe los comandos slash **por agente** (`.claude/commands/` para Claude Code,
`.opencode/command/` para opencode). Para que la factoría funcione en **ambos runtimes**, se ejecuta
`specify init` una vez por agente con `--here --ai <agent>` (el `--here` inicializa en el directorio
actual; cada agente escribe en su propia carpeta de comandos, así que es aditivo):

```bash
# Requiere Node ≥ 18
specify init --here --ai claude      # comandos en .claude/commands/
specify init --here --ai opencode    # comandos en .opencode/command/
```

`specify init` crea `.specify/` (templates + config) — compartido e idéntico para ambos agentes — y
deja los comandos `/constitution`, `/specify`, `/plan`, `/tasks`, `/implement` disponibles en los dos.
`setup-sdd.sh spec-kit` hace exactamente esto (variable `SPECKIT_AGENTS="claude opencode copilot"`).

Si `sdd.speckit.source` en `config.yml` está vacío o es `""`, usar el paquete público por defecto.

### 4b. Fork o fuente privada

Si `sdd.speckit.source` tiene valor:

```bash
# Formato: git+https://… | git+ssh://… | paquete npm privado
SPECKIT_SOURCE="$(yq '.sdd.speckit.source' .factory/config.yml)"
npm install -g "$SPECKIT_SOURCE"
specify init --here --ai claude   --from "$SPECKIT_SOURCE"
specify init --here --ai opencode --from "$SPECKIT_SOURCE"
```

Para fuentes `git+ssh://`, el sistema usa la clave SSH del agente (no se inyecta token).
Para fuentes `git+https://`, inyectar `GSD_GIT_TOKEN` desde `.factory/sdd/secrets.env`:

```bash
# .factory/sdd/secrets.env (git-ignorado — NUNCA commitear)
GSD_GIT_TOKEN=ghp_…        # PAT con acceso de lectura al repo del fork
```

```bash
source .factory/sdd/secrets.env
npm install -g "$(echo "$SPECKIT_SOURCE" | sed "s|https://|https://${GSD_GIT_TOKEN}@|")"
```

---

## 5. Verbo `verify` — gate compuesto / Composite verify gate

Dado que spec-kit no incluye un verbo `verify` nativo, `factory-build` ejecuta la verificación así:

```bash
# 1. Tests del stack
bash .factory/assets/scripts/test-gate.sh run
# (gates.tests.cmd / gates.tests.coverage_min en .factory/config.yml; sin cmd usa el test_cmd del starterkit)

# 2. Docker gate (si gates.docker_run: true)
docker compose up --build -d
# smoke check: espera a que el health endpoint responda OK
curl -sf http://localhost:${PORT:-8000}/health || exit 1
docker compose down
```

Si cualquier paso falla: capturar la salida, volver a `execute` para corregir, reintentar (máximo
3 veces consecutivas sobre el mismo fallo — ver stop conditions en factory-build).

---

## 6. Extensiones / Extensions

### 6a. Instalación de extensiones definidas en config.yml

Cada entrada de `sdd.speckit.extensions[]` en `.factory/config.yml` es `{name, from, subdir}`:

```bash
# from vacío → catálogo público o configurado, resuelto por nombre:
specify extension add <name>

# from = fuente no-git (URL de catálogo, referencia de paquete):
specify extension add <name> --from <from>

# from = URL git (la raíz del paquete puede ser un SUBDIRECTORIO del repo):
git clone --depth 1 <from> .factory/sdd/vendor/<name>
specify extension add --dev .factory/sdd/vendor/<name>/<subdir>
```

`setup-sdd.sh spec-kit` (lo lanza `factory-init`) recorre `extensions[]` y ejecuta el comando que toque. El
clon de `.factory/sdd/vendor/` se **conserva** (git-ignored; `--dev` puede apuntar a esa ruta) y en re-ejecuciones
hace `git pull --ff-only`. Los clones https privados inyectan `GSD_GIT_TOKEN` desde `secrets.env`.

### 6a-bis. Catálogo curado (menú del onboarding)

`factory/sdd/spec-kit/extensions-catalog.yaml` (instalado en `.factory/sdd/spec-kit/extensions-catalog.yaml`) es el
menú que `factory-init` presenta **siempre** que el motor es spec-kit: las extensiones **brownfield** de la
comunidad (raíz del paquete `extension/`), una URL git propia o ninguna. Cada entrada trae `description`/`note`
(se muestran tal cual al usuario), pasos `post_install` opcionales y directorios `skills_copy` (copiados a
`.claude/skills/`). Para proponer una extensión curada nueva, añade una entrada — ver `factory/sdd/README.md`.

### 6b. Catálogo privado

Si `sdd.speckit.catalog_url` tiene valor, inyectarlo como variable de entorno **antes** de cada
comando `specify extension`:

```bash
source .factory/sdd/secrets.env   # carga SPECKIT_CATALOG_TOKEN
export SPECKIT_CATALOG_URL="$(yq '.sdd.speckit.catalog_url' .factory/config.yml)"
```

Para registrar el catálogo privado en la configuración local del proyecto:

```bash
specify extension catalog add "$SPECKIT_CATALOG_URL" --name private-catalog
```

### 6c. Referencia de comandos de extensiones / Extension command reference

| Comando | Acción |
|---|---|
| `specify extension search [query]` | Buscar extensiones disponibles en el catálogo |
| `specify extension add <name> [--from URL]` | Instalar una extensión |
| `specify extension remove <name>` | Desinstalar una extensión |
| `specify extension list` | Listar extensiones instaladas |
| `specify extension info <name>` | Ver detalles de una extensión |
| `specify extension update [name]` | Actualizar una extensión (o todas si se omite el nombre) |
| `specify extension enable <name>` | Habilitar una extensión deshabilitada |
| `specify extension disable <name>` | Deshabilitar sin desinstalar |
| `specify extension set-priority <name> <N>` | Ajustar orden de resolución |
| `specify extension catalog add <url> --name NAME` | Registrar un catálogo adicional |

### 6d. Orden de resolución de catálogos / Catalog resolution order

Cuando spec-kit resuelve el origen de una extensión, lo busca en este orden (primero gana):

1. **`SPECKIT_CATALOG_URL`** — variable de entorno (inyectada desde `secrets.env`).
2. **`.specify/extension-catalogs.yml`** — catálogo(s) registrados a nivel de proyecto.
3. **`~/.specify/extension-catalogs.yml`** (o `~/.specify/…`) — catálogos a nivel de usuario.
4. **Catálogo público por defecto** de spec-kit.

Implicación práctica: para forzar un catálogo privado en CI/CD sin modificar archivos del repo,
basta con exportar `SPECKIT_CATALOG_URL` + `SPECKIT_CATALOG_TOKEN` en el entorno.

---

## 7. Variables y secretos / Credentials & variables

Todos los secretos van en `.factory/sdd/secrets.env` (git-ignorado). Nunca en `config.yml` ni en
issues/work items.

| Variable | Requerida | Para qué |
|---|---|---|
| `SPECKIT_CATALOG_TOKEN` | Solo con catálogo privado | Autenticar peticiones al catálogo de extensiones privado (`sdd.speckit.catalog_url`) |
| `GSD_GIT_TOKEN` | Solo con `speckit.source` privado (HTTPS) | PAT para clonar/instalar el fork privado de spec-kit vía `npm install -g git+https://…` |

Para orígenes `git+ssh://`, el sistema usa la clave SSH del agente — no se necesita variable.

---

## 8. Gotchas y lecciones / Gotchas & lessons

- **`specify init` es idempotente**: si `.specify/` ya existe, lo actualiza sin borrar la config.
  Útil en setups CI/CD repetidos.
- **Nombre de carpeta vs. clave de planning**: el slug del nombre de carpeta (`phase-1-auth`) puede
  cambiar si se renombra la fase; la clave de planning (`phase-1`) no debe cambiar — es el ancla de
  `issue-map.json`. Mantenerlas separadas evita IDs huérfanos.
- **spec-kit y git**: `specify` puede abrir PRs automáticamente si así se configura. En la factoría,
  desactivar esa integración para que `factory-build` controle cuándo se abre una PR (guardrail de
  no push sin confirmación).
- **`/implement` hace commits**: si el adapter ejecuta `/implement`, los commits ya existen cuando
  llega el `verify` gate. No hacer commit adicional vacío.
- **Catálogo privado en WSL/Windows**: `SPECKIT_CATALOG_URL` puede necesitar un proxy si el
  catálogo es una URL interna no accesible desde WSL2 directamente.
- **Versión de spec-kit**: el CLI `specify` puede cambiar la sintaxis de `extension` entre versiones
  mayores. Pinear la versión en `sdd.speckit.source` si la estabilidad es crítica:
  `specify@1.x` o `git+https://github.com/github/spec-kit.git#v1.2.3`.

---

## 9. Testing / Verificación del adapter

Para confirmar que el adapter está operativo tras la instalación:

```bash
# 1. CLI disponible
specify --version

# 2. Extensiones instaladas
specify extension list

# 3. Crear un spec de prueba (sin commitear)
mkdir -p specs/_test-probe && specify --help | grep -q 'plan' && echo "OK"

# 4. Catálogo privado accesible (solo si catalog_url está configurado)
SPECKIT_CATALOG_URL="$SPECKIT_CATALOG_URL" specify extension search --limit 1
```

---

## 10. Changelog

- 2026-06-07 — Playbook inicial: contrato de 5 verbos, mapa de roadmap, instalación pública/privada,
  extensiones y catálogos (públicos y privados), gate `verify` compuesto, referencia de comandos.
