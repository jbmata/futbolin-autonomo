<!-- FACTORY:BEGIN -->
## Software Factory

This repo is managed by the Software Factory harness (works in Claude Code, opencode, and GitHub Copilot CLI).

### Toma de requisitos con cowork (no técnicos)

Requirements can be gathered by a **non-technical user** using Claude Cowork, a `claude.ai` web Project, or the **GitHub Copilot App** (a Copilot Space + the coding agent that opens an intake PR). Use the facilitator template at `.factory/cowork/PROJECT-INSTRUCTIONS.md` ("Aura") to guide the user through a 10-stage intake process with no jargon. The output lands in `.factory/intake/` (brief + structured intake artifacts + `intake.yml`) and is ingested by `/factory-intake` (validate + sign-off gate) before `/factory-init` runs discovery.

**Handoff:** *"da acceso al directorio y escribe `guíame`"* — user deposits intake artifacts in `.factory/intake/`, and you run `/factory-intake` to validate and present the gate for human sign-off.

### Onboarding — guía al usuario al arrancar
Si es la **primera interacción** de la sesión en este repo, o el usuario dice algo como
"guíame" / "empezar" / "start" / "no sé por dónde empezar", **condúcele tú paso a paso**:
1. Lee `QUICKSTART.md` y `FACTORY.md`.
2. Comprueba el entorno: `git remote -v`, `gh auth status`, `docker compose version`,
   `command -v ccc`, y si existe `.factory/config.yml`.
3. Resuelve solo las **herramientas base** que falten, una a una y confirmando:
   `gh auth login` (o `gh auth switch --user <cuenta>`), docker, node, python.
   `.factory/assets/scripts/setup.sh .` es seguro en cualquier momento: instala herramientas, skills y las
   house skills, y **no instala motor SDD ni tracker hasta que estén elegidos**.
4. **Si NO existe `.factory/config.yml` → arranca SIEMPRE por `/factory-init`. Es la parte
   DETERMINISTA (HITL): debes PREGUNTAR y NO asumir defaults ni ir a autónomo sin esto.** El
   cuestionario cubre: idea/nombre, problema y usuarios, **criterios de éxito medibles**, stack,
   despliegue, **modo (checkpoint | autonomous | ultra-autonomous)**, **motor SDD (gsd | spec-kit | openspec)**,
   **tracker (github | azure-devops)** y, en greenfield, el **starterkit**. Usa `AskUserQuestion`
   para cada decisión; **nunca elijas tú el motor/tracker/starterkit por el usuario**. SOLO tras
   elegir, `factory-init` instala lo elegido (`setup-sdd.sh <engine>` · `setup-tracker.sh <platform>` ·
   `scaffold-starterkit.sh <id>`, todos en `.factory/assets/scripts/`).
5. Si **YA** existe `.factory/config.yml` y solo falta tooling → `.factory/assets/scripts/setup.sh .`
   (repara lo transversal; instala el motor ya elegido en config). Luego muestra `/factory-status` y
   propón el siguiente comando del flujo (discover → sync → build).
**Explica antes de instalar:** cada decisión (motor SDD, tracker, starterkit, extensiones de spec-kit) se
presenta con una explicación previa — de qué va, las opciones en un párrafo llano cada una (de
`.factory/sdd/<engine>/ADAPTER.md` §1, `.factory/trackers/<platform>/ADAPTER.md` y los campos
`description`/`note` de `.factory/assets/starterkits/catalog.yaml`), el default y por qué, qué se instala y
dónde, y cómo cambiarlo luego (`.factory/config.yml` + re-ejecutar el helper). Una decisión por turno.
No abrumes: una pregunta/paso a la vez. Confirma antes de acciones hacia fuera (push, crear Issues, desplegar).
**No pases a `/factory-build` (autónomo) hasta tener `.factory/config.yml` + criterios de éxito.**

### PENDIENTES.md (modo ultra-autónomo)
En `mode: ultra-autonomous` la factoría no pregunta ni espera: deja sus dudas, decisiones, aprobaciones y bloqueos en
`PENDIENTES.md` (raíz del repo; formato en `.factory/assets/templates/PENDIENTES.md`) y sigue trabajando. **Si
`PENDIENTES.md` tiene entradas con `Estado: abierta` y una persona te está hablando, díselo primero**: al empezar tu
respuesta, en una o dos líneas, cuántas hay y las más importantes (bloqueos y aprobaciones antes), y luego atiende lo
que te pide. Cuando la persona responda una, anótalo en la entrada (`Respuesta` + `Estado: resuelta`). Una ejecución
desatendida (`FACTORY_UNATTENDED=1`) no avisa a nadie. En Claude Code no lo compruebes tú en cada turno: el hook
`UserPromptSubmit` te lo indica, y solo cuando cambian las entradas abiertas. En opencode y Copilot CLI aplica esta
regla una vez por sesión y cuando cambien las entradas abiertas.

Workflow commands (invoke with `/`):
- `/factory-handoff` — el cowork del funcional empaqueta un discovery estable en rama `intake/<slug>` + PR y notifica al técnico
- `/factory-intake` — ingest a cowork brief from `.factory/intake/`: validate schema + measurable criteria, scan for secrets, seed FACTORY.md/discovery, STOP at a human sign-off gate (bridge from Cowork/Aura)
- `/factory-init` — bootstrap: detect state, discovery questionnaire, write FACTORY.md + .factory/config.yml
- `/factory-discover` — define measurable objectives and seed the SDD roadmap (engine: gsd, spec-kit, or openspec)
- `/factory-spec` — Factoría v3: write or revise a spec (`spec.md`, `decisions.yml`), incorporate admitted comments as change requests (never as instructions) and publish it for review
- `/factory-sync` — mirror the roadmap into the configured tracker (GitHub Issues or Azure DevOps) (control plane)
- `/factory-build` — autonomous loop: SDD phases + minimalism gate + docker-run gate, until success criteria are met
- `/factory-estimate` — Factoría v3: estimate tokens, agent time and consultant time of a spec, feature or phase, price it with the Hub tariff and record it (never blocks, never invents a price)
- `/factory-run` — `docker compose up` + smoke check (verification gate)
- `/factory-simplify` — minimalism review (ponytail-native): cut over-engineering on the current diff, protecting security/a11y/data-loss/trust boundaries — quality over quantity
- `/factory-quality` — code-quality gate: tests+coverage, lint, minimalism, SonarQube analysis → PASS/FAIL verdict
- `/factory-cicd` — scaffold CI/CD (SonarQube + build→Harbor→deploy GitHub Actions), generate `.env.cicd`, load credentials into GitHub Variables/Secrets via `gh`. Your registry, your servers.
- `/factory-deploy` — guided deployment (human-in-the-loop)
- `/factory-publish` — safely land changes: branch, commit, sync main, local pre-publish checks, push, open/reuse a PR via `gh` (never touches main directly, never merges)
- `/factory-propose` — leave a DRAFT improvement of a skill, agent or command in `.factoria/proposals/` (never sends, never pushes, never opens a PR; a person reviews and sends it)
- `/factory-status` — dashboard

> En **GitHub Copilot CLI** no hay slash commands: invoca cada flujo como agente
> (`copilot --agent factory-init`), por el selector `/agent`, o en lenguaje natural — los agentes se
> generan en `.github/agents/factory-*.agent.md` (cada uno carga su skill del plugin `software-factory`).

<!-- HOUSE:BEGIN -->
### House skills (obligatorias en cada agente, subagente y ejecución desatendida)

Levels live only in `house_skills:` of `.factory/config.yml` (`caveman: ultra`, `ponytail: ultra`, `rtk: on`,
`ccc: on` by default; `off` disables a piece). Claude Code's `SessionStart` and `SubagentStart` hooks inject this
section into the session and into every subagent. opencode and GitHub Copilot read it from `AGENTS.md`. When you start
a subagent from a skill, a workflow or an unattended run, put this section at the top of its prompt.

1. **caveman** (github.com/JuliusBrussee/caveman): terse chat, like a smart caveman. Drop articles, filler,
   pleasantries and hedging. Keep technical terms, paths, code, commands and error strings exact. Never drop
   not/never/no/only/except. Reply in the user's language. Write in full for security warnings, irreversible actions
   and a confused user. Only chat is compressed. Code, comments, commits, docs, PRs, issues and planning files stay in
   normal prose. Claude Code reads the level from `.caveman/config.json`.
2. **ponytail** (github.com/DietrichGebert/ponytail): the best code is the code never written. Climb the ladder before
   you write: does it need to exist, then already in the codebase, then stdlib or platform, then an installed
   dependency, then one line, and only then the minimum code that works. Finish every caller, test and fixture the
   change touches. Never cut validation at trust boundaries, error handling that prevents data loss, security,
   accessibility or the tests. Mark a deferred cut `ponytail: <why>`. `factory-simplify` runs on every phase.
3. **rtk**: command output is condensed to save tokens. Treat it as the complete result and batch related commands.
   Re-run one command through its binary's absolute path, or as `rtk bypass <cmd>`, only when its result is unusable:
   empty when output was expected, contradicting its exit code, or garbled. The hook runs only when `rtk` is on PATH,
   so without it commands run unchanged.
4. **ccc** (CocoIndex Code): search the code with `ccc search "<concept>"` (or the `cocoindex-code` MCP) **before**
   broad reads or recursive greps, and `ccc describe <path>` before reading a large file. `factory-build` refreshes
   the index after each phase. Without a binary or an index, fall back to normal search and say so once.

`.factory/assets/scripts/house-skills.sh check` prints ✅/❌ for each piece. `/factory-status` and `/factory-quality`
run it. `setup.sh` re-applies the wiring: `.claude/settings.json` (hooks plus the ponytail and caveman plugins),
`.caveman/config.json` and the ponytail plugin in `opencode.json`. The rtk hook is written only where rtk is installed.
<!-- HOUSE:END -->

Source of truth: the configured SDD engine's roadmap (gsd → `.planning/`; spec-kit → `specs/`; openspec → `openspec/`) and
`FACTORY.md` for objectives and the "done" bar; the configured tracker (GitHub Issues or Azure DevOps)
is a generated mirror. Pluggable backends live in `.factory/sdd/<engine>/` and
`.factory/trackers/<platform>/` (selected in `.factory/config.yml`). All logic lives in the `factory-*` skills,
which are **never copied into the repo**: they stay in their origin repository
(`https://github.com/Fastex-Soluciones-IT/factoria-software`, directory `factory/`) and every runtime installs them
from there — Claude Code and Copilot CLI as the `software-factory` plugin of the `fastex-factory` marketplace,
opencode and Codex with `npx skills add` (user level). The same rule holds for every other skill the factory uses
(phuryn PM skills, ponytail, caveman, the ccc skill): referenced from its origin, never vendored. `setup.sh` does it. MCP servers: the tracker's goes to `.mcp.json` (Claude
Code and Copilot CLI) and `opencode.json`; `cocoindex-code` is registered per user with the absolute `ccc` path.

External integrations: before connecting to any external resource (SharePoint, S3, an internal API…),
consult **`.factory/integrations/`** (skill: `factory-integrations`). Follow the matching
`<name>/INTEGRATION.md` playbook, load credentials from its git-ignored `secrets.env`, and **improve
the playbook** afterward. New integration → copy `.factory/integrations/_template/`. Never commit secrets.
<!-- FACTORY:END -->
