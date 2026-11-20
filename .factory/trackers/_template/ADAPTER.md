# Tracker Adapter: <NAME>

> Copy this directory to `.factory/trackers/<name>/` and fill it in. This adapter translates factory
> sync operations into platform-specific commands. See the specification at
> `docs/superpowers/specs/2026-06-07-pluggable-sdd-and-tracker-design.md` § 2b for the contract.

- **Status:** draft | proven
- **Platform:** <issue tracking system name>
- **Last updated:** <date>

## 1. Purpose / when to use

Describe the external tracking system (GitHub Issues, Azure DevOps Work Items, Jira, Linear, etc.)
and how the factory mirrors work onto it. A factory roadmap phase → this platform's concept;
a factory task → this platform's concept; a blocked task → a platform state/tag.

## 2. Identity & configuration

Define which config vars in `.factory/config.yml` under `tracker.<platform>:` identify the target
system. Examples:

**GitHub:** `tracker.github.repo` (owner/name)  
**Azure DevOps:** `tracker.azure_devops.org` (tenant), `tracker.azure_devops.project` (project name),
optionally `tracker.azure_devops.area_path` (area under project)  
**Custom:** (describe your platform's identity)

Document in a table:

| Config key | Required | Description | Example |
|---|---|---|---|
| `tracker.<platform>.<key1>` | yes | … | … |
| `tracker.<platform>.<key2>` | no | … | … |

## 3. Credentials & variables

List env vars (names only; secrets go in `.factory/trackers/<name>/secrets.env`, git-ignored).
Copy names to `variables.env.example` in this directory.

| Variable | Required | Where to get it |
|---|---|---|
| `TRACKER_<PLATFORM>_TOKEN` | yes | … |
| `TRACKER_<PLATFORM>_OTHER` | no | … |

## 4. Auth method

Describe how to authenticate (API token, OAuth, MCP server config, CLI login check, etc.),
including any consent / admin onboarding step.

**Example — GitHub:**  
- User runs `gh auth login` and selects a protocol (HTTPS + personal access token, SSH, or GitHub CLI app).
- `factory-init` / `setup.sh` verifies with `gh auth status`.
- No extra token in env vars; `gh` CLI handles it.

**Example — Azure DevOps:**  
- Azure DevOps MCP server is configured in `.mcp.json` (Claude Code + Copilot CLI) and `opencode.json`
  (for opencode).
- Auth via Entra ID (interactive login) or Personal Access Token (PAT) in MCP config.
- Verify with `core_list_projects` from the MCP.

## 5. Operations contract — 6-operation guarantee

`factory-sync` calls these operations idempotently. Each operation works on **stable keys** from
`.planning/ROADMAP.md` (or `specs/`): e.g., `phase-1`, `phase-1.task-3`. Implement these exactly:

| Operation | Input | Output / side effect | Idempotency notes |
|---|---|---|---|
| **`ensure-types`** | None (runs once per sync) | Creates labels/tags/work-item-types needed by the platform (e.g., labels `factory:phase`, `factory:task`, `factory:done`, `factory:blocked`; or Work Item Types `Epic`/`Feature`/`Task`). | Safe to call multiple times; skips if already present. |
| **`create`** | `{key, title, body, parent_key, labels_or_tags, is_phase}` | Creates a new issue/work-item. Returns `{id, url}` to write into `issue-map.json`. Attach parent link if `parent_key` exists. | Checked via `issue-map.json`; skip if key already mapped. Do not re-create. |
| **`update`** | `{id, title, body, labels_or_tags}` | Updates title, body, tags on an existing issue/work-item. | Safe to call multiple times; merge tags, overwrite text. No op if unchanged. |
| **`close`** | `{id}` | Closes/resolves the issue or moves work-item to `Done`/`Closed` state. Tag as `factory:done`. | Safe to call multiple times; already-closed remains closed. |
| **`read-back`** | `{ids or query}` | Query the platform for all tracked items (by `issue-map.json` IDs or by factory labels/tags). Return `{id, state, title, url, tags}` for each. | Safe; returns current state. Used to validate mirroring. |
| **`blocked`** | `{id}` | Mark an issue/work-item as blocked (e.g., add label `factory:blocked` or move to `Blocked` state). | Safe to call multiple times; toggle or set state once. |

## 6. `issue-map.json` entry shape

When an issue/work-item is created, `factory-sync` writes a neutral entry:

```json
{
  "<planning-key>": {
    "platform": "<platform-name>",
    "id": 0,
    "url": "<full-url-to-the-issue>"
  }
}
```

**Example:**

```json
{
  "phase-1": {
    "platform": "github",
    "id": 42,
    "url": "https://github.com/owner/repo/issues/42"
  },
  "phase-1.task-3": {
    "platform": "azure-devops",
    "id": 5678,
    "url": "https://dev.azure.com/org/project/_workitems/edit/5678"
  }
}
```

This neutral map allows:
- **Portability:** different repos can use different platforms.
- **Idempotency:** `factory-sync` looks up the mapping before create/update.
- **Verification:** read-back queries the mapped IDs to validate state.

## 7. Implementation pattern

Provide a minimal, working code example in your stack (Python, bash, or reference the MCP/CLI
command). Include:
- Auth token acquisition or CLI verification.
- One example from the contract (e.g., create an issue).
- Error handling (timeouts, rate-limit, not-found, auth failures).

```python
# minimal pattern — adapt to your platform
```

## 8. Integration checklist

After implementing this adapter, wire it into the factory with these steps:

### A. Configuration
- [ ] Documented `tracker.<platform>.*` keys in `.factory/config.yml` template (ask `factory-init` to add examples).
- [ ] If using env tokens: created `variables.env.example` with var names + docs.

### B. Setup & credentials  
- [ ] `setup-tracker.sh` has a branch to install/verify this platform's CLI or MCP server.
  - For a CLI tool (like `gh`): verify login (e.g., `gh auth status`).
  - For an MCP server: update `.mcp.json` and `opencode.json` with the server config + org/project from `config.yml`.
- [ ] If using secrets: `secrets.env` exists (git-ignored) with real values populated locally.

### C. Skills integration
- [ ] `factory-sync` reads `tracker.platform` from `config.yml` and dispatches to this adapter's operations.
- [ ] `factory-status` reports this platform as active (readiness check: auth working, config valid).

### D. Testing
- [ ] Unit: mock the CLI or MCP calls; verify operation contracts (ensure-types, create, update, close, read-back, blocked).
- [ ] Integration: a test/sandbox project configured in `config.yml`; gate behind an env flag if credentials unavailable.
- [ ] Smoke test: `factory-init` → choose this platform → `setup-tracker.sh` → `factory-status` shows green.

## 9. Gotchas & lessons (grows over time)

- (Will be populated by first implementation.)

## 10. Changelog

- <date> — initial template.
