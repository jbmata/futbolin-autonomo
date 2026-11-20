# SDD Adapter: gsd (GSD Core)

> Adapter playbook for the **gsd** SDD engine. `factory-build` and related skills read this file when
> `sdd.engine: gsd` is set in `.factory/config.yml`. The next developer (or agent) can implement
> or debug the full integration from this document alone.

- **Engine:** [GSD Core](https://github.com/open-gsd/gsd-core) — npm package `@opengsd/gsd-core`, successor of
  `get-shit-done-cc` (deprecated on npm)
- **Version:** pinned in `sdd.gsd.version` (`1.20.1` today), never `@latest`
- **Requires:** Node >= 24 and npm >= 10; `jq` and `gh` for `gsd-ship`
- **Status:** proven
- **Last updated:** 2026-11-06
- **Activado con / activated by:** `sdd.engine: gsd` en `.factory/config.yml`

---

## 1. Propósito / Purpose

GSD Core is a skill set for Claude Code, opencode, GitHub Copilot CLI (and other runtimes) that drives structured
software development through a **phase engine**: discuss → plan → execute → verify. It keeps the source of truth in
`.planning/` (Markdown files plus `config.json`), so it is portable and auditable without any external service.

Use this adapter when:
- The team works in Claude Code, opencode or Copilot CLI.
- The source of truth must live in the repo (`.planning/`), not in an external project-management tool.
- You want the full PM-skills integration (problem framing, JTBD, PRD, roadmap, atomic commits).

---

## 2. Contrato de verbos / Verb contract

`factory-build` (and `factory-discover`) call abstract verbs; this adapter translates each to the concrete GSD Core
skill. **Do not call gsd skills directly from `factory-build`: always go through the verb table, so swapping adapters
requires zero changes in the build loop.** `N` is the phase number from `.planning/ROADMAP.md`.

| Verbo factory / Factory verb | Skill invocada / Skill invoked | Notas / Notes |
|---|---|---|
| `discover` | `/gsd-discuss-phase N` (+ PM Skills when available) | Writes `NN-CONTEXT.md`. In `autonomous`, `--auto` picks the recommended defaults; in `ultra-autonomous` never ask (assumptions go to `PENDIENTES.md`) |
| `plan` | `/gsd-plan-phase N` | Writes `.planning/phases/NN-slug/NN-MM-PLAN.md` (one per plan, checked by `gsd-plan-checker`) |
| `decompose` | _(incluido en `/gsd-plan-phase`)_ | Tasks live inside each PLAN; no separate call |
| `execute` | `/gsd-execute-phase N` (N is required) | Atomic commits, one `NN-MM-SUMMARY.md` per plan, `NN-VERIFICATION.md` when the phase completes |
| `verify` | `/gsd-verify-work N` | UAT: `NN-UAT.md`; returns pass, fail (with fix plans) or `human_needed` |
| `status` | `bash .factory/assets/scripts/gsd-status.sh` | Factory status JSON (§2b). Read-only |

All six commands exist with these names in GSD Core 1.20.1 (`docs/COMMANDS.md` at tag `v1.20.1`). Behavior that
changed against `get-shit-done-cc` and matters to the loop:
- `/gsd-plan-phase` orders tasks as a tracer bullet by default (`--no-tracer` turns it off) and inserts a
  `checkpoint:decision` before irreversible decisions (`--no-reversibility-gates`). The factory keeps both defaults.
- `/gsd-execute-phase` never auto-approves a `blocking-human` checkpoint, and a failed package install stops at
  `checkpoint:human-verify` (§5c).
- `/gsd-verify-work` auto-passes deliverables proven by tests (the SUMMARY `coverage:` block) and asks a human for the
  rest; it can abstain with `human_needed`, which the factory treats like a `blocking-human` checkpoint.

### Cómo `factory-build` resuelve y despacha / How `factory-build` resolves and dispatches

```text
1. Read .factory/config.yml → sdd.engine = "gsd"
2. Load this ADAPTER.md (verb table above)
3. For each roadmap phase N not done in the status JSON (§2b):
   a. verb:discover  → /gsd-discuss-phase N
   b. verb:plan      → /gsd-plan-phase N        (skip if status shows tasks_total > 0)
   c. verb:decompose → (no-op: tasks are already in the plans)
   d. verb:execute   → /gsd-execute-phase N     (skip if status shows tasks_done == tasks_total)
   e. verb:verify    → /gsd-verify-work N       → pass | fail | human_needed
4. After verify:
   - PASS → run the test gate (`test-gate.sh run`; gsd's verify does not run the project tests), then the
     docker gate (factory-run), then factory-sync to close issues
   - FAIL → return to step b (plan) or step d (execute) as appropriate; bounded retries
   - human_needed → stop and ask (both modes)
```

**Autonomous mode chains by itself.** With `workflow.auto_advance: true` (written in autonomous mode, §5b),
`/gsd-discuss-phase` launches `/gsd-plan-phase --auto`, which launches `/gsd-execute-phase --no-transition`. One
`discover` call can therefore leave the phase planned and executed; steps b and d see it in the status JSON and are
skipped, and factory-build's "Sync (plan)" step runs after execute. Nothing moves to the next phase on its own
(`--no-transition`): factory-build still runs verify, the gates and sync for every phase.

The build loop's idempotency, stop conditions, model routing, and docker gate remain in `factory-build`; this adapter
owns only the verb→skill mapping.

### 2b. Verbo `status` / `status` verb

`factory-status` (and `factory-build`, to pick the next phase) runs one command instead of reading files:

```bash
# github tracker: blocked issues come from gh; azure-devops: pass the Blocked work items as a JSON array instead
gh issue list --label factory:blocked --state open --json number,title,url \
  | bash .factory/assets/scripts/gsd-status.sh --blocked - .
```

The script calls three `gsd-tools` commands in the repo root (`roadmap analyze --json`, `state json`,
`progress --json`), reads the `## Success criteria` checkboxes of `FACTORY.md`, and prints one JSON document:

```json
{
  "engine": "gsd",
  "version": "1.20.1",
  "milestone": { "version": "v1.0", "name": "MVP" },
  "status": "executing",
  "current_phase": "phase-2",
  "phases": [
    { "key": "phase-1", "number": "1", "name": "Foundation", "status": "done",
      "disk_status": "complete", "tasks_done": 1, "tasks_total": 1 },
    { "key": "phase-2", "number": "2", "name": "API", "status": "in-progress",
      "disk_status": "planned", "tasks_done": 0, "tasks_total": 1 }
  ],
  "totals": { "phases_done": 1, "phases_total": 2, "tasks_done": 1, "tasks_total": 2, "percent": 50 },
  "criteria": [ { "text": "GET /health returns 200", "done": true } ],
  "blocked": [ { "number": 7, "title": "Missing staging DB credential", "url": "https://github.com/…/issues/7" } ]
}
```

- **Tasks** are GSD plans: `tasks_total` counts `NN-MM-PLAN.md`, `tasks_done` the plans with a `SUMMARY.md`.
- **Phase status:** `done` when GSD's `disk_status` is `complete` (plans summarized and a passing
  `NN-VERIFICATION.md`); `in-progress` for the current phase, a phase with any summary, or `disk_status: executed`;
  otherwise `pending`. GSD derives it from disk, not from the ROADMAP checkboxes: a ticked `[x]` has no authority
  (see `checkbox_conflict` in `roadmap analyze`).
- **Phase numbers:** `roadmap analyze` prints them without padding (`"1"`, `"2.1"`) and `progress --json` with it
  (`"01"`, `"02.1"`). The script strips the leading zeros of every segment and keys each phase `phase-N`, the key
  `factory-sync` and `.factory/issue-map.json` use.
- **`percent`** comes from `progress --json` and is `null` when GSD cannot scope it; it is never guessed.
- **`blocked`** is whatever the tracker passes with `--blocked FILE|-` (`[]` without it). GSD Core 1.20.1 has no
  reliable blocker list in these commands (`state json` has none, and `state-snapshot` reads only a `## Blockers`
  section, not the `### Blockers/Concerns` of its own STATE template).
- Exit codes: 0 JSON on stdout · 1 a gsd-tools call failed · 2 no `.planning/ROADMAP.md` · 3 GSD Core not installed.
  `GSD_TOOLS=<path>` overrides the lookup of `gsd-tools.cjs`.

---

## 3. Fuente de verdad / Source of truth

| Artefacto / Artifact | Ruta / Path | Quién escribe / Writer | Quién lee / Reader |
|---|---|---|---|
| Roadmap de fases / Phase roadmap | `.planning/ROADMAP.md` (checklist + Progress table) | `gsd-new-project` / `gsd-new-milestone`, `gsd-phase` | `factory-build`, `factory-sync`, `gsd-status.sh` |
| Plan de fase / Phase plan | `.planning/phases/NN-slug/NN-MM-PLAN.md` | `gsd-plan-phase` | `gsd-execute-phase`, `gsd-verify-work` |
| Tareas de fase / Phase tasks | inside each PLAN | `gsd-plan-phase` | `factory-sync` (issue mirroring) |
| Resumen y verificación | `NN-MM-SUMMARY.md`, `NN-VERIFICATION.md`, `NN-UAT.md` (same folder) | `gsd-execute-phase`, `gsd-verify-work` | `gsd-status.sh`, `gsd-ship` |
| Estado / State | `.planning/STATE.md` | every gsd skill | `gsd-status.sh` (`state json`) |
| Configuración gsd | `.planning/config.json` | `gsd-new-project`, `factory-models` (§5b) | every gsd skill |
| Objetivos y criterios de éxito | `FACTORY.md` | `factory-discover` | `factory-build` (stop condition), `gsd-status.sh` |
| Discovery notes | `.factory/discovery.md` | `factory-discover` | `factory-discover` (idempotent) |
| Milestone gsd | `.planning/PROJECT.md`, `.planning/REQUIREMENTS.md` | `gsd-new-project` | every gsd skill |

The `NN-MM-PLAN.md` naming is mandatory: GSD Core finds plans by it. With workstreams the same tree lives under
`.planning/workstreams/<active>/`.

**Regla de oro / Golden rule:** `.planning/ROADMAP.md` es la única fuente de verdad de fases. Los
issues de GitHub (o Azure DevOps) son un espejo generado por `factory-sync` — nunca al revés.

---

## 4. Instalación / Install

`setup-sdd.sh gsd` (called by `setup.sh` and `factory-init`) does every step below. It is idempotent: with the
pinned version already installed for every target runtime it changes nothing.

### 4a. Origen público (por defecto) / Public source (default)

`sdd.gsd.version` holds the exact version and `sdd.gsd.source` stays empty. The installer takes one flag per entry
of `runtime_targets` in the same run:

```bash
npx -y --package=@opengsd/gsd-core@1.20.1 -- gsd-core --claude --opencode --copilot --global
```

Never `--all`: in GSD Core it installs all 18 supported runtimes, not the factory's three. To move to another
version, change `sdd.gsd.version` and re-run `setup-sdd.sh gsd`; do not use `/gsd-update`, which installs `@latest`
and breaks the pin.

Where it lands (global): Claude Code `~/.claude/skills/gsd-*/`, `~/.claude/agents/gsd-*.md` and the engine in
`~/.claude/gsd-core/` (`gsd-core/VERSION` holds the version); opencode `~/.config/opencode/` (`commands/gsd-*.md`,
`skills/gsd-*/`, `agents/`, `gsd-core/`); Copilot CLI `~/.copilot/skills/gsd-*/` and `agents/`. Skills keep the
`gsd-` prefix and are invoked as `/gsd-xxx`.

**Node >= 24.** GSD Core declares `engines.node >= 24`, and Claude Code runs the GSD hooks with the node that is active when it
starts. Without Node 24, `setup-sdd.sh` proposes installing it at user level (fnm if present, else nvm, installing
nvm v0.40.3 if needed), keeps the current Node, makes 24 the default for new shells, and installs only after
confirmation (`--install-node`). Restart the terminal and Claude Code afterwards.

### 4b. Migración desde get-shit-done-cc / Migrating from get-shit-done-cc

An old install is detected by `get-shit-done/VERSION` in `~/.claude`, `~/.config/opencode` or `~/.copilot`. Before
changing anything, `setup-sdd.sh` shows two plans:

1. GSD Core's own legacy cleanup, `gsd-core <runtimes> --global --dry-run` (it only previews the cleanup and exits).
2. The scope of the old package's uninstaller, `get-shit-done-cc@<installed version> --<runtime> --global
   --uninstall`: `skills/gsd-*`, `agents/gsd-*`, `hooks/gsd-*`, `commands/gsd/`, `get-shit-done/` (it keeps
   `USER-PROFILE.md`), its hooks and statusline in `settings.json`, and `gsd-file-manifest.json`.

Only after the user confirms (`--migrate`, or `s` on a terminal) it uninstalls the old package and then installs GSD
Core. The order matters: run after GSD Core, the old uninstaller would delete the new `skills/gsd-*`, which share
their names. Nothing outside what GSD manages is removed, and `.planning/` is never touched (the installers only
write under the runtime config dirs).

After installing it reports leftovers without deleting them: a `get-shit-done/` dir (usually just your
`USER-PROFILE.md`) and hooks, commands or `settings.json` entries that still name `get-shit-done-cc`. GSD Core
1.20.1's cleanup misses those hooks: it searches for the wrong package name.

Run by an agent (no terminal), `setup-sdd.sh` prints the plan and exits 3; the agent shows it to the user and re-runs
with `--migrate` and/or `--install-node` once the user agrees.

### 4c. Origen privado/custom / Private or custom source

When `sdd.gsd.source` has a value, `setup-sdd.sh` passes it to `npx --package=` instead of the public package, with
the same runtime flags. It must be an npm package spec whose bin is `gsd-core` (a GSD Core fork keeps it), and
`sdd.gsd.version` must match the version it installs. Tokens live in `.factory/sdd/secrets.env` (git-ignored),
**never in `config.yml` or in issues**.

```yaml
# .factory/config.yml
sdd:
  engine: gsd
  gsd:
    version: "1.20.1"
    source: "git+ssh://git@github.com/miorg/gsd-core-fork.git#v1.20.1"   # system SSH key, no token
    # source: "git+https://github.com/miorg/gsd-core-fork.git#v1.20.1"  # GSD_GIT_TOKEN in secrets.env
    # source: "@miorg/gsd-core@1.20.1"                                  # private registry, see below
```

- **git+ssh:** uses the system SSH key; no token.
- **git+https:** `setup-sdd.sh` injects `GSD_GIT_TOKEN` into the URL for that one `npx` call.
- **Private npm:** authenticate the registry for your user first (`npm login --registry …` or a scoped entry in
  `~/.npmrc`); `setup-sdd.sh` does not write tokens into `~/.npmrc`.

### 4d. Variables de entorno / Environment variables

Residen en `.factory/sdd/secrets.env` (git-ignorado). Nombres de referencia:

| Variable | Cuándo se necesita / When needed | Descripción |
|---|---|---|
| `GSD_GIT_TOKEN` | `git+https` a repo privado | PAT con permisos de lectura al repo GSD Core custom |
| `GSD_NPM_TOKEN` | paquete npm privado/scoped | npm auth token for the registry (configure it in your own `~/.npmrc`) |

El fichero `secrets.env` **nunca** se commitea. El `.gitignore` del repo raíz ya lo excluye
(patrón `*.env`, `secrets.env`).

---

## 5. Verificación post-instalación / Post-install verification

```bash
cat ~/.claude/gsd-core/VERSION                                       # = sdd.gsd.version
ls ~/.claude/skills/ | grep -E "gsd-(discuss|plan|execute|verify)-"
node ~/.claude/gsd-core/bin/gsd-tools.cjs roadmap analyze --json     # in a repo with .planning/
bash .factory/assets/scripts/gsd-status.sh .                         # the status verb (§2b)
# (Claude Code / opencode) /gsd-health   → reports the .planning/ health; --repair is no longer destructive
```

`gsd-tools` is not on PATH unless GSD Core was installed with `npm install -g`; call it through
`node <config dir>/gsd-core/bin/gsd-tools.cjs` (the old path was `~/.claude/get-shit-done/bin/`). The workflows
themselves use `gsd_run query <cmd>`.

### 5b. Modelos y autonomía en `.planning/config.json` / Models and autonomy

GSD Core picks its subagents' models from `.planning/config.json`. `factory-models` (step 4) and `factory-init`
write this policy, aligned with the `models:` block of `.factory/config.yml`. Merge these keys into the file and keep
every other key (a missing file is created by `gsd-new-project`; write the policy right after it):

```json
{
  "model_profile": "balanced",
  "model_policy": { "provider": "anthropic" },
  "models": { "planning": "sonnet", "discuss": "sonnet", "research": "sonnet",
              "execution": "sonnet", "verification": "sonnet", "completion": "sonnet" },
  "model_overrides": { "gsd-nyquist-auditor": "haiku" },
  "dynamic_routing": { "enabled": false },
  "workflow": { "auto_advance": false }
}
```

- No `model_policy.runtime_tiers`: GSD Core 1.20.1 does not know the 5.5 ids, so pinning them makes every
  `resolve-model` warn and fall back to the tier alias anyway. The aliases (`haiku`, `sonnet`, `opus`) resolve to
  Claude Code's current models, the 5.5 ones. Add `runtime_tiers.claude` with the `models.ids.claude` ids once GSD
  Core's catalog lists them.
- `models.<phase_type>` takes only `opus`, `sonnet`, `haiku` or `inherit`, from the factory roles: `planning` ←
  `planning`; `discuss` and `research` ← `discovery`; `execution` ← `execution`; `verification` and `completion` ←
  `review`; with `fast` → `haiku`, `balanced` → `sonnet`, `strong` → `opus`. The default policy gives `sonnet`
  everywhere; `opus` appears only when the factory config sets a role to `strong` explicitly. (`discuss` and
  `completion` are accepted but have no subagent yet.)
- `gsd-nyquist-auditor` generates tests, so it runs as `testing` (`haiku`); `dynamic_routing` stays off because the
  escalation to Opus goes through the factory with a recorded justification.
- `workflow.auto_advance` is `true` when `mode: autonomous` or `ultra-autonomous` and `false` in `checkpoint`. `mode: "yolo"` alone does
  not make `/gsd-execute-phase` approve checkpoints: its auto mode (`gsd-tools check auto-mode`) is on only with the
  internal `--auto` chain or `workflow.auto_advance: true`.
- Precedence in GSD Core: `model_overrides` > `dynamic_routing` > `models[phase_type]` > `model_profile`.

Check it: `node ~/.claude/gsd-core/bin/gsd-tools.cjs resolve-model gsd-planner` must print `"model": "sonnet"` (with
`balanced` alone it would be `opus`), and `gsd-nyquist-auditor` must print `haiku`.

### 5c. Checkpoints y gates humanos / Checkpoints and human gates

GSD Core plans can stop at checkpoints. Since 1.7.0 a checkpoint with `gate="blocking-human"` is **never
auto-approved**, in any mode and with no config switch (`references/checkpoints.md` rule 6; `execute-phase`
`checkpoint_handling`, marked `gsd:protected`). An unmet `<precondition>` and a failed or unverified package install
report that gate.

| Checkpoint | `checkpoint` mode | `autonomous` mode (`auto_advance: true`) |
|---|---|---|
| `human-verify` | asks | auto-approved, logged `⚡ Auto-approved checkpoint` |
| `decision` with `auto_select` | asks | picks that option |
| `decision` without `auto_select` | asks | asks (escalates to a human) |
| `human-action` (auth, a step only a human can do) | asks | asks |
| any `gate="blocking-human"` | asks | asks, logged `⛔ blocking-human gate — auto-mode suspended` |

`/gsd-autonomous` (not used by the factory) stops a step after 3 failed fix attempts as `needs_human` and records it in
the `## Needs Human` section of `STATE.md`. What `factory-build` does at each stop is in its "Human gates" section;
the factory never skips these checkpoints.

---

## 6. Ciclo completo de ejemplo / End-to-end example

```text
# Estado inicial: factory-init + factory-discover ya corrieron
# .factory/config.yml:  sdd.engine: gsd · sdd.gsd.version: "1.20.1" · mode: checkpoint

/factory-build

  → adapter resolve: sdd.engine = gsd → load .factory/sdd/gsd/ADAPTER.md
  → status: gsd-status.sh → phase-1 pending
  → phase-1: "Scaffold API"
      discover  → /gsd-discuss-phase 1  → 01-CONTEXT.md        (checkpoint: ¿ok para continuar?)
      plan      → /gsd-plan-phase 1     → .planning/phases/01-scaffold-api/01-01-PLAN.md, 01-02-PLAN.md
      decompose → (incluido en los PLAN)
      /factory-sync                     → crea issues GitHub/Azure
      execute   → /gsd-execute-phase 1  → commits atómicos, 01-01-SUMMARY.md, 01-02-SUMMARY.md, 01-VERIFICATION.md
                                          (a blocking-human checkpoint stops here and asks)
      verify    → /gsd-verify-work 1    → 01-UAT.md → PASS
      gate      → test-gate.sh run + /factory-run → docker compose up → 200 OK
      /factory-sync                     → cierra issues de phase-1
  → phase-2: "Auth layer" … (ídem)
  → criterios de éxito evaluados (status JSON "criteria" + evidencia) → DONE
```

---

## 7. Gotchas y lecciones / Gotchas & lessons

- **gsd necesita `.planning/` inicializado.** Si no existe, `factory-discover` debe correr primero
  (`gsd-new-project` / `gsd-new-milestone`). `factory-build` debe verificar esto al arrancar.
- **`decompose` es un no-op en gsd.** `gsd-plan-phase` ya emite tareas dentro de cada PLAN. No llamar a
  ningún skill adicional para este verbo o se duplicará trabajo.
- **Modo autónomo vs. checkpoint.** `gsd-discuss-phase` puede ser interactivo. En `mode: autonomous`,
  pasarle el contexto suficiente (FACTORY.md + discovery.md) para que salte las preguntas, y recordar que con
  `workflow.auto_advance: true` encadena plan y execute (§2).
- **`gsd-ship` needs `jq` and `gh`.** Before the PR it checks with `roadmap analyze` that the phase is complete and
  recorded in ROADMAP.md (`roadmap_record`) and fails with `PHASE_NOT_COMPLETE` / `PHASE_NOT_RECORDED` otherwise.
  Keep the ROADMAP checklist and Progress table up to date; `setup-sdd.sh` reports a missing `jq` and never installs
  it with sudo.
- **`gsd-pr-branch` rejects a dirty working tree.** Commit or stash first. `planning.pr_strict: true` keeps STATE,
  ROADMAP and PROJECT in the PR branch.
- **`gsd-phase add|insert` also writes the ROADMAP row**, and `gsd-progress` / `gsd-stats` may report a `null`
  percent: show it as unknown.
- **`config-set` rejects unknown keys.** Merge the §5b keys by editing the JSON. A key GSD does not know, written by
  hand, is ignored silently, so check with `resolve-model`.
- **Model ids GSD does not know.** 1.20.1 does not list the 5.5 ids yet: a `runtime_tiers.claude` entry with them makes
  `resolve-model` warn `has no Claude agent alias; using "sonnet" instead` on every call. That is why §5b uses only the
  tier aliases.
- **Tokens en URL de clonado.** The git+https token goes only into the one `npx` call; never into config, logs or a
  persisted remote.
- **No `/gsd-update` under a pin.** It moves to `@latest`. Change `sdd.gsd.version` and re-run `setup-sdd.sh gsd`.
- **WSL / Docker.** If hooks fail because of paths, re-install with `--portable-hooks` (manual run of the command in
  §4a with that flag).
- **Modo ultra-autónomo y checkpoints de GSD Core.** En `mode: ultra-autonomous` nadie responde a un checkpoint de
  `/gsd-execute-phase`. Un `human-verify` normal lo cubren los tests y los gates; un `checkpoint:decision` con
  `auto_select` o con un default seguro y reversible se toma y se anota como `decisión` en `PENDIENTES.md`; un
  `blocking-human`, un `human-action` o un verify que se abstiene con `human_needed` (login, acción física,
  credenciales, una decisión sin default seguro) aparca esa tarea y sus dependientes con una entrada en
  `PENDIENTES.md`, y el bucle sigue con la siguiente tarea o fase independiente (`factory-build`, sección
  Ultra-autonomous mode). GSD Core no los aprueba solos en ningún modo, y la factoría tampoco los salta.

---

## 8. Changelog

> Cada implementación que mejore o ajuste este adapter añade una línea aquí.

- 2026-06-07 — Playbook inicial: contrato de 5 verbos, fuente de verdad, install público/privado
  (SSH, HTTPS+PAT, npm privado), lógica de despacho de `factory-build`, gotchas.
- 2026-11-06 — Perfil de modelos de gsd alineado con la política 5.5 de `factory-models` (Sonnet por defecto,
  Haiku para tests, sin `dynamic_routing`).
- 2026-11-06 — Motor cambiado a GSD Core 1.20.1 (`@opengsd/gsd-core`), fijado en `sdd.gsd.version`: instalación
  multi-runtime, Node 24, migración confirmada desde `get-shit-done-cc`, rutas `.planning/phases/NN-slug/`, verbo
  `status` (`gsd-status.sh`), `model_policy` en `.planning/config.json` y gates `blocking-human`.
