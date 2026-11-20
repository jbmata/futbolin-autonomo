# Integrations registry

A living knowledge base of **external-resource integrations** (SharePoint, S3, Salesforce, an internal
API, …). When a build needs to connect to an external system, the factory reads the matching playbook
here — reusing prior knowledge and the project's credentials — instead of figuring it out from scratch.

## Layout

```
.factory/integrations/
├── README.md                     # this file
├── _template/                    # copy this to start a new integration
│   ├── INTEGRATION.md            # the playbook (instructions, auth, code patterns, gotchas, changelog)
│   ├── variables.env.example     # names of the env vars needed (committed, NO secrets)
│   └── .gitignore                # ignores secrets.env
└── <name>/                       # one directory per integration (e.g. sharepoint/)
    ├── INTEGRATION.md
    ├── variables.env.example
    └── secrets.env               # REAL credentials — git-ignored, never committed
```

## How the factory uses it (see the `factory-integrations` skill)

1. **Detect the need** — a task requires an external resource (e.g. "read files from SharePoint").
2. **Read the playbook** — `.factory/integrations/<name>/INTEGRATION.md` (auth, SDK, steps, gotchas).
3. **Load credentials** — from `<name>/secrets.env` (git-ignored) or the process env. Names are
   documented in `variables.env.example`.
4. **Implement** following the playbook.
5. **Improve the playbook** — append what you learned (working snippet refs, pitfalls, version notes)
   and bump the changelog. Each implementation makes the next one easier.

## Adding a new integration

```bash
cp -r .factory/integrations/_template .factory/integrations/<name>
# edit <name>/INTEGRATION.md and <name>/variables.env.example
cp .factory/integrations/<name>/variables.env.example .factory/integrations/<name>/secrets.env
# fill secrets.env with real values (it is git-ignored)
```

## Security rules
- **Never commit `secrets.env`** (or any real credential). Only `*.env.example` (names + docs) is committed.
- Prefer a secret manager in production; `secrets.env` is for local development convenience.
- Playbooks describe *how* to integrate and *which* variables are needed — not secret values.
