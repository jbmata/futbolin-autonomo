# Starterkit catalog registry

`catalog.yaml` lists the templates that `factory-init` (step 5) offers for greenfield scaffolding and that
`scaffold-starterkit.sh <id>` clones (FastAPI full-stack, Open SaaS, Frappe, go-service, or none).

## Catalog entry schema

| Key | Required | Meaning |
|---|---|---|
| `id` | yes | Short id used by `scaffold-starterkit.sh <id>` and `starterkit:` in config.yml |
| `name` | yes | Display name |
| `stack` | yes | Tech tags (list) |
| `description` | yes | What the template is — shown at onboarding |
| `health` | no | Smoke-check endpoint for the docker-compose gate (e.g. `"GET /health"`) |
| `source` | yes | Git URL to clone (`builtin` only for the `none` sentinel) |
| `subdir` | yes | Template root inside the repo (`""` = repo root, `auto` = `template/app` when present) |
| `default_port` | yes | Port the compose gate probes |
| `note` | no | Auth / usage warnings — shown to the user |
| `cicd` | no | `factory-cicd` wiring: `template` → `factory/templates/cicd/<name>/`, images, compose, deploy |
| `quality` | no | `factory-quality` wiring: sonar sources/coverage, `test_cmd` |

## Adding a new starterkit

1. Add an entry to `catalog.yaml` with the schema above (CI validates it — `Factory structure
   validation` job). Add its `id` to `primary:` only if it is buildable under the docker-compose
   gate (that list is what `factory-init` presents).
2. If it needs CI/CD scaffolding, add a template dir under `factory/templates/cicd/<name>/` and
   reference it from `cicd.template`.
3. Regenerate the trees (`bash factory/install.sh --self`) and commit both in the same PR.

Private repos: `scaffold-starterkit.sh` clones with the user's credentials — keep the `note`
field explicit about authentication.
