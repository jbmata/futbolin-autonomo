# Tracker adapters registry

A living knowledge base of **issue-tracking platform adapters** (GitHub, Azure DevOps, Jira,
Linear, etc.). When `factory-sync` needs to mirror work onto an external system, it reads the
matching adapter playbook here — reusing prior knowledge and the project's credentials — instead
of figuring it out from scratch.

## Layout

```
.factory/trackers/
├── README.md                     # this file
├── _template/                    # copy this to start a new tracker adapter
│   ├── ADAPTER.md                # the playbook (6-op contract, auth, code pattern, gotchas, changelog)
│   ├── variables.env.example     # names of the env vars needed (committed, NO secrets)
│   └── .gitignore                # ignores secrets.env
└── <platform>/                   # one directory per tracker (e.g. github/, azure-devops/, hub/)
    ├── ADAPTER.md
    ├── variables.env.example
    └── secrets.env               # REAL credentials — git-ignored, never committed
```

## How the factory uses it

1. **Select the platform** — in `factory-init`, choose which tracker (github, azure-devops, …).
2. **Configure** — write `tracker.<platform>: { org, project, … }` to `.factory/config.yml`.
3. **Authenticate** — `setup.sh` installs/verifies the platform's CLI or MCP server (gh, azure-devops MCP, …).
4. **Sync** — `factory-sync` reads the active platform from `config.yml`, loads `.factory/trackers/<platform>/ADAPTER.md`,
   and executes the 6 operations (ensure-types, create, update, close, read-back, blocked) against it.
5. **Track in `issue-map.json`** — a neutral map of planning keys → `{platform, id, url}` ensures idempotency
   and portability across platforms.

## The 6-operation contract

Every tracker adapter implements these operations; `factory-sync` calls them idempotently:

| Operation | Purpose |
|---|---|
| `ensure-types` | Create labels/tags/work-item-types (e.g., `factory:phase`, `factory:task`, `factory:done`). Runs once per sync. |
| `create` | Create a new issue/work-item. Return `{id, url}` to store in `issue-map.json`. |
| `update` | Update title, body, labels on an existing issue/work-item. |
| `close` | Close/resolve an issue or move work-item to `Done` state. Tag as done. |
| `read-back` | Query platform for current state of tracked items (by mapped ID or by tags). Return `{id, state, title, url}` for each. |
| `blocked` | Mark an issue/work-item as blocked (label or state change). |

See the template's § 5 for the exact input/output shapes and idempotency notes.

## `issue-map.json` — platform-neutral tracking

`factory-sync` maintains `.factory/issue-map.json` mapping planning keys to platform-specific IDs:

```json
{
  "phase-1": {
    "platform": "github",
    "id": "42",
    "url": "https://github.com/owner/repo/issues/42"
  },
  "phase-1.task-3": {
    "platform": "azure-devops",
    "id": "5678",
    "url": "https://dev.azure.com/org/project/_workitems/edit/5678"
  }
}
```

This ensures:
- **Idempotency:** skip create if key already mapped.
- **Portability:** different repos can use different platforms.
- **Validation:** read-back by mapped ID ensures current state matches planning.

## Adding a new tracker

```bash
cp -r .factory/trackers/_template .factory/trackers/<platform>
# edit <platform>/ADAPTER.md and <platform>/variables.env.example
cp .factory/trackers/<platform>/variables.env.example .factory/trackers/<platform>/secrets.env
# fill secrets.env with real values (it is git-ignored)
```

Then update:
1. `.factory/config.yml` — add template entry under `tracker.<platform>:` (ask `factory-init` to generate).
2. `setup.sh` — add verification/install step for the platform's CLI or MCP server.
3. `factory-sync` — dispatch to the adapter's operations based on `config.yml` `tracker.platform`.
4. `factory-status` — report readiness of the active tracker.

See the template's § 8 (Integration checklist) for a full walkthrough.

## Security rules

- **Never commit `secrets.env`** (or any real credential). Only `*.env.example` (names + docs) is committed.
- Prefer a secret manager in production; `secrets.env` is for local development convenience.
- Adapters describe *how* to integrate and *which* variables are needed — not secret values.
- Never include auth tokens or sensitive data in issue/work-item titles or bodies.

## Reference adapters (coming soon)

- `github/ADAPTER.md` — mirrors to GitHub Issues (via `gh` CLI).
- `azure-devops/ADAPTER.md` — mirrors to Azure DevOps Work Items (via MCP `microsoft/azure-devops-mcp`).
- `hub/ADAPTER.md` — Factoría v3: every operation is a no-op; the child syncs through the hub.
