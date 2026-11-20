# 🚀 QUICKSTART — Empezar de cero con la Software Factory

> Para alguien que **no sabe nada** del proyecto. Plan: **clonar el repo → arrancar tu agente
> (Claude Code, opencode o GitHub Copilot CLI) → decir "guíame"**. El agente hace el resto y te va
> llevando paso a paso.
>
> En **GitHub Copilot CLI** no hay slash commands: en vez de `/factory-init`, invoca el agente
> equivalente — `copilot --agent factory-init` (o el selector `/agent`, o pídelo en lenguaje natural).
> Los agentes se generan en `.github/agents/factory-*` al instalar.

---

## TL;DR (3 pasos)

```bash
git clone <URL-del-repo> && cd <repo>
claude            # o tu comando de opencode
```
Y dentro del agente, escribe:
```
guíame
```
El agente lee este QUICKSTART, comprueba tu entorno, instala lo que falte y te conduce por el flujo.
**No necesitas memorizar comandos.**

---

## 0. Qué necesitas instalado (una vez por máquina)

Solo necesitas las **herramientas base**; el resto (PM Skills, CocoIndex Code y las house skills: caveman,
ponytail, rtk y ccc) lo instala el agente por ti con **un único script** (`setup.sh`). El motor SDD y el
tracker se eligen en `/factory-init`, que te los explica antes de instalarlos.

| Herramienta base | Para qué | Comprobar / instalar |
|---|---|---|
| **Claude Code** *(u opencode, o GitHub Copilot CLI)* | El agente que te guía | `claude --version` · Copilot: `npm i -g @github/copilot` (Node 22+), luego `copilot` + `/login` |
| **git** | Clonar y versionar | `git --version` |
| **GitHub CLI (`gh`)** | Issues como panel de control | `gh --version` · login: `gh auth login` |
| **Docker + Compose** | Levantar la app en local (gate) | `docker --version` · `docker compose version` |
| **Python + pipx** | Base para la CLI de indexado (ccc) | `pipx --version` (si no: `python3 -m pip install --user pipx && python3 -m pipx ensurepath`) |
| **Node + npx** | Base para instalar skills | `npx --version` |

Lo que **el agente instala por ti** (un solo paso, ver §3):
- **gsd** — motor de fases
- **phuryn/pm-skills** — discovery / PRD / priorización (para el agente de factoría)
- **CocoIndex Code** — búsqueda semántica: CLI + extensión (skill) + MCP + índice del repo

No hace falta instalar nada de eso a mano si dejas que el agente lo haga.

---

## 1. Clona el repo y entra

```bash
git clone <URL-del-repo>
cd <repo>
```

## 2. Autentícate en GitHub (para los Issues)

```bash
gh auth login           # elige GitHub.com, HTTPS, y la cuenta con acceso al repo
```
> Si tienes varias cuentas: `gh auth switch --user <cuenta>` antes de trabajar.

## 3. Arranca el agente y deja que te guíe

```bash
claude                  # o opencode
```
Dentro, escribe **`guíame`** (o `/factory-init`). El agente entonces:

1. **Comprueba el entorno** (git, gh, docker, node, python) y te ayuda con lo que falte.
2. **Instala todo lo necesario, de una** — ejecuta el setup completo (te pide confirmación):
   ```bash
   .factory/assets/scripts/setup.sh .
   ```
   Esto instala **gsd** (motor de fases) + **phuryn/pm-skills** (para el agente de factoría) + **CocoIndex Code** (CLI +
   extensión/skill + MCP en Claude y opencode) e **indexa el repo** para búsqueda semántica
   (≈70% menos tokens). *(La primera vez, `ccc init` puede pedirte elegir proveedor de embeddings una sola vez.)*
3. **Hace el discovery**: si el repo ya existe, lo analiza; si es nuevo, te ofrece un starterkit. Te
   hace un breve **cuestionario** y escribe `FACTORY.md` (objetivos + criterios de éxito).
4. **Siembra el roadmap** (gsd) y **crea las tareas como GitHub Issues**.
5. **Construye** por fases hasta cumplir los criterios, **levantando `docker compose` en local** como
   prueba de cada fase. Te pausa al final de cada fase para que revises.

---

## Alternativa: instalar con APM

Si tu equipo usa [APM](https://github.com/microsoft/apm), las skills, los comandos y los hooks de las house skills se
instalan como un paquete con lockfile, en vez de `/plugin install` o `npx skills add`. Es una segunda vía: la de arriba
no cambia.

```yaml
# apm.yml en la raíz del repo (ejemplo: examples/apm/apm.yml del repo de la factoría)
name: mi-proyecto
version: 1.0.0
targets: [claude, copilot, opencode]
dependencies:
  apm:
    - Fastex-Soluciones-IT/factoria-software/factory#v3.6.0
```

```bash
apm install
```

APM no ejecuta scripts: el runtime `.factory/` (scripts, plantillas, adaptadores) lo sigue poniendo `install.sh`, que
ejecutas tú desde un clon del repo de la factoría en la misma etiqueta. Con las dos cosas la sección de house skills
llega una sola vez al agente. Dentro de la Factoría v3 no escribes este `apm.yml`: lo compone el Hub.

## Toma de requisitos con cowork (no técnicos)

¿Tienes requisitos anotados en una **libreta, Figma, o notas sueltas**? Un usuario no técnico (product owner, cliente, PM) puede capturarlos de forma **autónoma y sin jerga** en:

- **Claude Cowork** (recomendado): app de escritorio con acceso a la carpeta del repo. Los requisitos se escriben directamente en `.factory/intake/`.
- **Proyecto de claude.ai web**: redactar + subir diseños/capturas (visión Opus), luego exportar manualmente a `.factory/intake/`.

**El flujo:**

1. **Aura** (facilitador integrado) guía con **una sola pregunta por turno**: visión → usuarios → funcionalidades → diseños → criterios → restricciones (10 etapas, no jerga).
2. Los artefactos salen en `.factory/intake/` (un brief ejecutivo, catálogo de funciones, pantallas, `intake.yml`).
3. **Tú o tu agente**: escribe `/factory-intake` para validar + presentar el gate de sign-off (objetivos, criterios medibles, no-goals, roadmap propuesto) **antes** de empezar la construcción.

**Una línea:** *"da acceso a la carpeta del repo y escribe `guíame`"* en Claude Cowork. Aura hace el resto; al terminar, tú lanzas `/factory-intake`.

---

## Elegir SDD y plataforma de tracking

Durante `/factory-init`, el agente te pregunta:

1. **Motor SDD**: ¿`gsd` (GSD Core), `spec-kit` (GitHub Spec Kit) u `openspec`?
   - `gsd`: fases ejecutables por skills, roadmap en `.planning/`. Versión fijada en `sdd.gsd.version`; necesita
     Node 24 (el setup lo propone y pide confirmación). Si tenías el antiguo `get-shit-done-cc`, el setup te enseña
     qué quitaría y lo cambia por GSD Core solo si dices que sí; tu `.planning/` se conserva.
   - `spec-kit`: especificaciones por feature, con extensiones opcionales.
2. **Plataforma de tracking**: ¿`github` (Issues) o `azure-devops` (Work Items)?
   - `github`: Issues espejados desde el roadmap (defecto).
   - `azure-devops`: Work Items mapeados automáticamente.

La factoría instala y configura solo el backend elegido. Si necesitas cambiar luego, re-ejecuta `/factory-init`.

---

## Proyecto NUEVO (greenfield) vs proyecto EN DESARROLLO (brownfield)

La factoría detecta el caso sola en `factory-init`. Lo único distinto es el arranque.

> **Meter la factoría en el repo** (si no la lleva ya): clónala una vez y usa su instalador en cualquier repo:
> ```bash
> git clone https://github.com/Fastex-Soluciones-IT/factoria-software.git ~/factoria-software
> ~/factoria-software/factory/install.sh /ruta/al/repo
> ```

### 🌱 Greenfield (de cero)
```bash
mkdir mi-proyecto && cd mi-proyecto && git init
~/factoria-software/factory/install.sh .
gh auth login
claude            # u opencode → escribe: guíame
```
El agente: `setup.sh` (PM + ccc + house skills) → `factory-init` detecta **greenfield** y te ofrece un
**starterkit** (FastAPI, Open SaaS, Frappe o go-service) con scaffolding → cuestionario → `FACTORY.md`
→ `gsd-new-project` (no hay `.planning/`) → `/factory-sync` → `/factory-build` → `docker compose up`.

### 🏗️ Brownfield (proyecto ya existente)
```bash
git clone <URL-del-repo> && cd <repo>
~/factoria-software/factory/install.sh .       # añade la factoría SIN tocar el código
gh auth login
claude            # u opencode → escribe: guíame
```
El agente: `setup.sh` (PM + ccc + house skills) **e indexa tu código** → `factory-init` detecta **código
existente** (sin scaffolding, detecta stack) → `factory-discover` analiza con `gsd-map-codebase` +
`ccc search` y define el **siguiente milestone** (`gsd-new-milestone` si ya había `.planning/`, si no
`gsd-new-project`) → `/factory-sync` → `/factory-build` **sin romper contratos/tests existentes**.

| | Greenfield | Brownfield |
|---|---|---|
| Punto de partida | repo vacío | repo con código |
| `factory-init` | ofrece **starterkit** + scaffolding | detecta stack, sin scaffolding |
| `factory-discover` | objetivos desde tu idea (`gsd-new-project`) | **mapea + indexa el código** → milestone (`gsd-new-milestone`) |
| Cuidado especial | — | no romper contratos/tests existentes |

## El flujo (comandos) — por si quieres lanzarlos tú

| Comando | Qué hace |
|---|---|
| `/factory-init` | Prepara el repo: entorno, cocoindex, cuestionario → `FACTORY.md` + `.factory/config.yml` |
| `/factory-discover` | Define objetivos medibles y siembra el roadmap |
| `/factory-sync` | Espeja el roadmap en GitHub Issues |
| `/factory-build` | Loop autónomo: fases (gsd) + gate `docker compose` hasta cumplir criterios |
| `/factory-run` | `docker compose up` + smoke check (gate) |
| `/factory-deploy` | Despliegue guiado (te pide confirmación) |
| `/factory-status` | Panel: objetivos, fases, Issues, salud del último run |

Empieza siempre por **`/factory-status`** para ver dónde estás.

---

## Cosas que SOLO tú puedes aportar

- **Decisiones en los checkpoints**: en modo `checkpoint` el agente pausa al final de cada fase.
- **Respuestas en `PENDIENTES.md`**: en modo `ultra-autonomous` el agente no pregunta ni para; deja dudas,
  decisiones y aprobaciones en `PENDIENTES.md` y sigue. Responde allí (o en el chat) cuando quieras: la siguiente
  fase aplica tus respuestas. Nunca hace solo merge/push a `main`, despliegues, secretos ni borrados.
- **Claves/secretos**: van en `.env` (nunca se commitea). P. ej. `ANTHROPIC_API_KEY` si una fase usa
  la API de Claude. El agente te dirá cuándo hace falta.
- **Confirmación de acciones hacia fuera**: push, crear muchos Issues, o desplegar.

---

## ¿Dónde está la info? (no tienes que re-explicar nada)

Todo el contexto vive **en el repo** y el agente lo lee solo:
`QUICKSTART.md` (esto) · `FACTORY.md` (objetivo + criterios) · `.factory/config.yml` (config) ·
`.planning/` (roadmap de gsd) · `AGENTS.md`/`CLAUDE.md` (reglas). Los **GitHub Issues** son un
espejo de seguimiento. Una sesión nueva **no recuerda conversaciones previas** — por eso todo está
escrito aquí.

---

## Problemas frecuentes

- **`gh` da 404 / sin permiso** → cuenta equivocada: `gh auth switch --user <cuenta-correcta>`.
- **Puerto ocupado al `docker compose up`** → el agente detecta y cambia de puerto; o libera el puerto.
- **Falta gsd / phuryn pm-skills / `ccc`** → re-ejecuta el setup completo: `.factory/assets/scripts/setup.sh .`
- **Una house skill sale ❌ en `/factory-status`** → `.factory/assets/scripts/setup.sh .` (o `house-skills.sh check` para ver cuál).
- **El agente no parece "saber" del proyecto** → asegúrate de arrancarlo **dentro** de la carpeta del repo.

---

**Resumen:** clona → `claude` → escribe `guíame`. El agente te lleva de la mano hasta tener la app
corriendo en local. 🚀
