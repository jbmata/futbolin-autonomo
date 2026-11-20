# Tracker adapter: `hub` (Factoría v3)

`tracker.platform: hub` is the only tracker value of a Factoría v3 project (design §8, "Una sola vía de sincronización").
In this mode the harness **never writes to any tracker**. The child instance syncs specs, phases, decisions and
comments with Azure DevOps through the hub of the parent factory; the hub owns that connection, its credentials and the
rules of who may change which field. A second writer in the harness would race the hub and break its revision checks.

## Contract

`factory-sync` resolves this adapter like any other and gets the six operations. Every one of them is a **no-op**:

| OPERATION | In `hub` mode |
|---|---|
| `ensure-types` | Does nothing. The hub creates the process, types and fields in Azure DevOps. |
| `create` | Does nothing and returns no id. `issue-map.json` is not written. |
| `update` | Does nothing. |
| `close` | Does nothing. A done phase reaches Azure DevOps through the child's sync, not through the harness. |
| `read-back` | Does nothing. State comes back through `project_context()` of the local MCP, not from a tracker. |
| `blocked` | Does nothing. A blocker is reported to the user and recorded in the SDD files (`.planning/`), not mirrored. |

`factory-sync` in this mode ends after the resolve step with the report `tracker hub: nothing written` and creates no
`gh`, `wit_*` or HTTP call, no label, milestone, issue, work item or `.mcp.json` entry.

## What is not in this adapter

- No auth: no `gh auth`, no PAT, no Entra login, no MCP server. `setup-tracker.sh hub` configures nothing.
- No `secrets.env` and no `variables.env.example`: the harness holds no tracker credential in this mode.
- The type mapping that Azure DevOps needs lives in the `azuredevops` module of the hub (`server/`), reused from
  `trackers/azure-devops/ADAPTER.md`, never applied from here.

## Changelog

- 0.1 (P1): first version. All operations are no-ops; covered by the tests of the Factoría v3 repository.
