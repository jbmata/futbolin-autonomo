# SDD Adapter: `<engine-name>`

> Reference template for plugging a new SDD (Spec-Driven Development) engine into the Software Factory.
> Copy this folder to `.factory/sdd/<engine>/ADAPTER.md` and fill in the contract table and install sections.

- **Status:** template (ready to customize)
- **Last updated:** (date of this adapter)
- **Engines:** see `factory-build`, `factory-discover`, `factory-sync`

## 1. Purpose / when to use

Define how the factory's five core verbs (discover / plan / decompose / execute / verify) map to
your SDD engine's commands or skills. Use this when you want to swap `gsd` or `spec-kit` for your own
metaprompting layer, custom specify fork, or third-party specification tool.

## 2. Verb contract

Each verb is invoked by `factory-build` and `factory-discover`. Map each to your engine's command,
skill invocation, or CLI call. Include required inputs and expected outputs (where planning state is
written).

| Factory verb | Your engine | Inputs | Outputs / Planning update |
|---|---|---|---|
| `discover` | (your discover command) | roadmap objective (string) | `.planning/ROADMAP.md` + initial phase structure |
| `plan` | (your plan command) | phase context, goals | `.planning/phase-N/plan.md` + task list |
| `decompose` | (your decompose command) | feature/task description | `.planning/*/tasks.md` + breakdowns |
| `execute` | (your execute command) | plan + codebase context | execution in `.planning/*/execute/`, code changes |
| `verify` | (your verify command) | success criteria from planning | test results, gate pass/fail in `.planning/*/verify/` |

## 3. Source of truth

Specify where your engine writes the canonical planning state:

- **Planning root:** `.planning/` or `specs/` (must be parseable by `factory-sync` for issue-mapping).
- **Roadmap file:** `ROADMAP.md` path and format (markdown outline or custom).
- **Phase/feature folders:** how phases map to folders (e.g., `phase-N/`, `specs/<feature-name>/`).
- **Task keys:** stable identifier format for `issue-map.json` (e.g., `phase-1.task-3`, `feature-auth.task-login`).

### Issue-mapping bridge

If your engine uses a different folder/file structure than gsd, document the mapping so `factory-sync`
and `issue-map.json` can match a planning item (e.g., a task) to its mirror issue number:

```
your-engine folder    ↔    planning key (issue-map.json)    ↔    GitHub Issue / Work Item
specs/auth/tasks.md   ↔    feature-auth.task-login           ↔    #42
```

## 4. Install: public source

Describe how to install your engine from a **public** location (registry, github release, npm, pypi,
etc.). This runs in `setup-sdd.sh` (called by `setup.sh`) when `config.yml` has `sdd.engine: <your-engine>` and no private source
is set.

```bash
# Example: install from npm registry
npm install -g @your-org/your-engine@latest

# Or: clone from public git
git clone https://github.com/your-org/your-engine.git ~/.your-engine
export PATH="~/.your-engine/bin:$PATH"

# Or: python package
pip install your-engine-package
```

## 5. Install: private source

If `config.yml` has `sdd.<engine>.source: git+https://... | git+ssh://...`, `setup.sh` injects auth
from `.factory/sdd/secrets.env` and installs the private version. Document:

- **Token variable names** expected in `secrets.env` (e.g., `GSD_GIT_TOKEN`, `SPECKIT_CATALOG_TOKEN`).
- **Cloning method**: if `git+ssh://`, uses system SSH key; if `git+https://`, needs `<TOKEN>@` in URL.
- **Additional setup** (e.g., private npm registry, custom catalog URLs).

```bash
# If source is git+https://github.com/yourprivate/engine.git
if [ -n "$GSD_GIT_TOKEN" ]; then
    git clone https://${GSD_GIT_TOKEN}@github.com/yourprivate/engine.git
else
    git clone git+ssh://git@github.com/yourprivate/engine.git
fi

# If source is an npm private registry
NPM_REGISTRY_TOKEN="$(grep '^NPM_REGISTRY_TOKEN=' .factory/sdd/secrets.env | cut -d= -f2)"
npm config set //registry.example.com/:_authToken="$NPM_REGISTRY_TOKEN"
npm install -g @yourprivate/engine@latest
```

## 6. Integration checklist

When integrating this adapter into `factory-*` skills and `setup-sdd.sh`:

- [ ] **`factory-init`**: add `sdd.<engine>` section to discovery questionnaire and `config.yml` template.
- [ ] **`setup-sdd.sh`**: add install logic (public + private; pass token from `secrets.env` if needed).
- [ ] **`factory-discover`**: load this ADAPTER.md; call the `discover` verb with roadmap objective.
- [ ] **`factory-build`**: in the loop, call `plan` → `decompose` → `execute` → `verify` verbs in order.
- [ ] **`factory-sync`**: parse planning output to extract phase/task keys; map to `issue-map.json`.
- [ ] **`factory-status`**: show this engine as active in the readiness table.
- [ ] **Docs**: update `AGENTS.md` and `QUICKSTART.md` if this is a built-in option.
- [ ] **Secrets template**: if using private sources, add token variable names to `.factory/sdd/secrets.env.example`.

## 7. Testing

- **Offline smoke**: verify verb commands/skill invocations parse and produce minimal output.
- **Roadmap generation**: run `discover` with a sample objective; confirm `.planning/ROADMAP.md` is valid.
- **Sync**: confirm `factory-sync` can read and map planning items to stable keys.
- **CI gate**: gate behind an env flag (`FACTORY_TEST_<ENGINE>=1`) so tests run only with secrets present.

## 8. Changelog

- (date) — initial adapter template.
