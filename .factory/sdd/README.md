# SDD engine registry

A living knowledge base of **SDD engine adapters** (gsd, spec-kit, openspec, …). The factory's
skills call five abstract verbs — `discover / plan / decompose / execute / verify` — and the
adapter playbook translates them into the chosen engine's real commands. `factory-build`,
`factory-discover`, `factory-sync` and `factory-status` read the active adapter from
`.factory/sdd/<engine>/ADAPTER.md` (selected by `sdd.engine` in `.factory/config.yml`). An adapter may add a
`status` verb that prints the factory status JSON (gsd does, with `gsd-status.sh`); `factory-status` and
`factory-build` use it instead of reading the engine's files.

## Layout

```
.factory/sdd/
├── README.md                     # this file
├── _template/                    # copy this to start a new engine adapter
│   ├── ADAPTER.md                # the playbook (verb contract, install, checklist, changelog)
│   └── variables.env.example     # names of the env vars needed (committed, NO secrets)
├── <engine>/                     # one directory per engine (gsd/, spec-kit/, openspec/)
│   └── ADAPTER.md
├── spec-kit/extensions-catalog.yaml   # curated spec-kit extensions (factory-init menu)
├── vendor/                       # cloned extension repos — git-ignored
└── secrets.env                   # REAL tokens — git-ignored, never committed
```

## Adding a new SDD engine

```bash
cp -r .factory/sdd/_template .factory/sdd/<engine>
# edit <engine>/ADAPTER.md — fill the 5-verb contract, source of truth, install and testing
```

Then wire the engine (the template's §6 Integration checklist walks every step):
1. `factory-init` — add the engine to questionnaire Q7 and the `sdd.<engine>:` config block.
2. `setup-sdd.sh` — add an `install_<engine>()` helper and dispatch it from the engine case (`setup.sh`
   installs tools only; the engine is chosen and explained in `factory-init`).
3. `factory-discover` / `factory-build` / `factory-sync` / `factory-status` — verb dispatch and
   readiness live in the adapter; verify they resolve it.
4. Docs — update `AGENTS.md` and `QUICKSTART.md`.

Remember the generated-trees invariant: edit `factory/sdd/…`, then run
`bash factory/install.sh --self` and commit both trees in the same PR.

## Adding a spec-kit extension to the curated catalog

`factory/sdd/spec-kit/extensions-catalog.yaml` is the menu `factory-init` presents when the
engine is spec-kit. Add an entry with:

| Key | Required | Meaning |
|---|---|---|
| `id` | yes | Extension id as `specify` knows it |
| `name` | yes | Display name (shown at onboarding) |
| `description` | yes | What it is/does — shown to the user verbatim |
| `source` | yes | Git URL to clone (`builtin` only for the `none` sentinel) |
| `subdir` | yes | Package root inside the repo (`""` = repo root; must contain `extension.yml`) |
| `requires` | no | Prerequisites (informational) |
| `post_install` | no | Steps the agent walks after `specify extension add` |
| `skills_copy` | no | Repo-relative dirs copied into `.claude/skills/` |
| `note` | no | Auth / stability warnings — shown to the user |

Add new ids to `primary:` to have them offered by default. CI validates the catalog schema
(`Factory structure validation` job). Install mechanics: `factory/sdd/spec-kit/ADAPTER.md` §6.

## Security rules

- **Never commit `secrets.env`** (or any real token). Only variable *names* are committed
  (`variables.env.example`, or the table in the adapter's Credentials section).
- Tokens for private engine sources / extension repos / catalogs go in `.factory/sdd/secrets.env`
  (`GSD_GIT_TOKEN`, `SPECKIT_CATALOG_TOKEN`) — never in `config.yml`, `FACTORY.md`, or issues.
- Adapters describe *how* to integrate and *which* variables are needed — not secret values.
