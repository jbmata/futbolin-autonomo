#!/usr/bin/env bash
# Software Factory — configure the chosen tracker (github | azure-devops). The tracker is a PROJECT
# DECISION: /factory-init explains both, the user chooses, then the agent runs this helper.
#   github        → checks the gh CLI and its authentication (auth never leaves the shell)
#   azure-devops  → registers the official @azure-devops/mcp server for the runtimes:
#                   .mcp.json (workspace file read by Claude Code AND GitHub Copilot CLI) + opencode.json
# Idempotent (non-destructive JSON merges) and best-effort. No PAT is ever written.
#
#   hub           → Factoría v3: nothing to configure; the harness never writes to a tracker (trackers/hub/ADAPTER.md)
#
# Usage:  setup-tracker.sh [github|azure-devops|hub] [TARGET_DIR]   (platform defaults to tracker.platform in config.yml)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
. "$SCRIPT_DIR/_lib.sh"

PLATFORM=""; TARGET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    github|azure-devops|hub) PLATFORM="$1"; shift ;;
    -*) say "Unknown flag: $1   (usage: setup-tracker.sh [github|azure-devops|hub] [TARGET_DIR])"; exit 2 ;;
    *)  if [ -d "$1" ]; then TARGET="$1"; shift
        else say "Unknown tracker or directory: '$1'   (trackers: github | azure-devops | hub)"; exit 2; fi ;;
  esac
done
TARGET="$(resolve_target "$TARGET")" || { say "Target dir not found"; exit 1; }
CONFIG_FILE="$TARGET/.factory/config.yml"

if [ -z "$PLATFORM" ] && [ -f "$CONFIG_FILE" ]; then
  PLATFORM="$(cfg_get tracker.platform)"
  # backward-compat: a top-level github: block without a tracker block means github
  if [ -z "$PLATFORM" ] && grep -q "^github:" "$CONFIG_FILE" 2>/dev/null; then PLATFORM="github"; fi
fi
if [ -z "$PLATFORM" ]; then
  say "No tracker chosen. Choose it in /factory-init (the agent explains GitHub Issues and Azure DevOps Work Items),"
  say "or run: setup-tracker.sh <github|azure-devops> [TARGET_DIR]"
  exit 2
fi
case "$PLATFORM" in
  github|azure-devops|hub) ;;
  *) say "Unknown tracker '$PLATFORM' (options: github | azure-devops | hub)"; exit 2 ;;
esac

AZDO_ORG=""; AZDO_PROJECT=""
if [ -f "$CONFIG_FILE" ]; then
  AZDO_ORG="$(cfg_get tracker.azure_devops.org)"
  AZDO_PROJECT="$(cfg_get tracker.azure_devops.project)"
fi

say "════════════════════════════════════════════════════════"
say " Software Factory — tracker: $PLATFORM"
say " repo: $TARGET"
say "════════════════════════════════════════════════════════"

# ── Helper: setup_tracker_azdo ─────────────────────────────────────────────────
setup_tracker_azdo(){
  say "  configuring Azure DevOps MCP…"
  ORG="${AZDO_ORG:-}"
  PROJECT="${AZDO_PROJECT:-}"
  [ -z "$ORG" ] && say "  ⚠ tracker.azure_devops.org not set in config.yml — edit the config and re-run"

  # The official Azure DevOps MCP package is @azure-devops/mcp (npm). It takes the org NAME as a
  # positional arg, with -a <auth> and -d <domains>. Auth defaults to Azure CLI / interactive — no
  # PAT file needed. Domains: 'all' (or a space-separated subset like 'work work-items core').
  [ -z "$ORG" ] && ORG="<org>"   # placeholder so the entry is written; user fills it in later

  # .mcp.json (repo root) — the WORKSPACE MCP file that both Claude Code and GitHub Copilot CLI read
  # (schema: mcpServers → type stdio + command + args). Non-destructive merge.
  WS_MCP="$TARGET/.mcp.json"
  python3 - "$WS_MCP" "$ORG" <<'PY'
import json, os, sys
p, org = sys.argv[1], sys.argv[2]
data = {}
if os.path.exists(p):
    try: data = json.load(open(p))
    except Exception: data = {}
servers = data.setdefault("mcpServers", {})
servers["azure-devops"] = {
    "type": "stdio",
    "command": "npx",
    "args": ["-y", "@azure-devops/mcp@latest", org, "-a", "interactive", "-d", "all"],
}
json.dump(data, open(p, "w"), indent=2)
open(p, "a").write("\n")
print(f"  merged 'azure-devops' server into {p}")
PY

  # opencode.json (repo root) — opencode schema: type=local + command ARRAY + enabled. Non-destructive merge.
  OCJSON="$TARGET/opencode.json"
  python3 - "$OCJSON" "$ORG" <<'PY'
import json, os, sys
p, org = sys.argv[1], sys.argv[2]
data = {}
if os.path.exists(p):
    try: data = json.load(open(p))
    except Exception: data = {}
data.setdefault("$schema", "https://opencode.ai/config.json")
mcp = data.setdefault("mcp", {})
mcp["azure-devops"] = {
    "type": "local",
    "command": ["npx", "-y", "@azure-devops/mcp@latest", org, "-a", "interactive", "-d", "all"],
    "enabled": True,
}
json.dump(data, open(p, "w"), indent=2)
open(p, "a").write("\n")
print(f"  merged 'azure-devops' server into {p}")
PY

  say "  ✓ MCP azure-devops written to .mcp.json (Claude Code + GitHub Copilot CLI) and opencode.json (opencode)"
  [ "$ORG" = "<org>" ] && say "  ⚠ edit tracker.azure_devops.org in config.yml and re-run (org placeholder '<org>')"
  say "  Auth: 'az login' (Azure CLI) or the MCP interactive prompt — no PAT file needed"
  say "  Verify with the MCP tool: core_list_projects (load it with ToolSearch if not in context)"
  [ -n "$PROJECT" ] && say "  Expected project: $PROJECT"
}

# ── Configure the chosen tracker ───────────────────────────────────────────────
say ""; say "[1/1] Tracker: $PLATFORM"
case "$PLATFORM" in
  github)
    if have gh; then
      if gh auth status >/dev/null 2>&1; then
        say "  ✓ gh authenticated"
      else
        say "  ⚠ gh not authenticated — run: gh auth login"
        say "    factory-sync needs gh auth to create/update issues"
      fi
    else
      say "  ✗ gh CLI not found — install from https://cli.github.com and run gh auth login"
    fi
    ;;
  azure-devops)
    setup_tracker_azdo
    ;;
  hub)
    say "  ✓ hub: nothing to configure (the child syncs through the hub; the harness writes to no tracker)"
    ;;
esac

# ── Readiness ──────────────────────────────────────────────────────────────────
say ""
say "════════════════════════════════════════════════════════"
say " Tracker readiness — $PLATFORM — $(date '+%Y-%m-%d')"
say "════════════════════════════════════════════════════════"
case "$PLATFORM" in
  github)
    printf "  %-22s %s\n" "gh status:" "$(have gh && gh auth status >/dev/null 2>&1 && echo 'authenticated' || echo 'not authenticated / not installed')"
    ;;
  azure-devops)
    printf "  %-22s %s\n" "mcp azure-devops:" ".mcp.json + opencode.json written (verify with core_list_projects)"
    ;;
  hub)
    printf "  %-22s %s\n" "hub:" "no tracker credential or MCP server (factory-sync writes nothing)"
    ;;
esac
say ""
say "Done. The adapter playbook is $TARGET/.factory/trackers/$PLATFORM/ADAPTER.md — /factory-sync and /factory-status read it."
