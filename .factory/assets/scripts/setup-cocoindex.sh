#!/usr/bin/env bash
# Prepare a repo locally with CocoIndex Code (ccc) — semantic code intelligence for the dev agent.
# Tri-target: registers the MCP in Claude Code, GitHub Copilot CLI and opencode when present, installs the skill
# ("extension"), and indexes the repo. Idempotent and best-effort: anything that needs an interactive
# step prints clear guidance instead of hanging. Re-running it repairs a registration that points at a bare
# or stale `ccc` (the "Executable not found in $PATH: ccc" failure). factory-init runs it; so does setup.sh.
#
# Usage:  factory/scripts/setup-cocoindex.sh [TARGET_DIR]      (default: current dir)
#         factory/scripts/setup-cocoindex.sh --skip-index DIR  (set up tooling, don't index now)
set -uo pipefail

SKIP_INDEX=0
TARGET=""
for a in "$@"; do
  case "$a" in
    --skip-index) SKIP_INDEX=1 ;;
    *) TARGET="$a" ;;
  esac
done
TARGET="${TARGET:-$PWD}"; TARGET="$(cd "$TARGET" 2>/dev/null && pwd)" || { echo "Target dir not found"; exit 1; }
say(){ printf '%s\n' "$*"; }
have(){ command -v "$1" >/dev/null 2>&1; }
ccc_bin(){ command -v ccc 2>/dev/null || { [ -x "$HOME/.local/bin/ccc" ] && printf '%s' "$HOME/.local/bin/ccc"; }; }

say "CocoIndex Code setup → $TARGET"
say ""

# --- 1. CLI (ccc) via pipx -----------------------------------------------------
say "[1/6] CLI (ccc)"
if [ -n "$(ccc_bin)" ]; then
  say "  ccc present — upgrading"; pipx upgrade cocoindex-code >/dev/null 2>&1 || true
elif have pipx; then
  say "  installing cocoindex-code[full] via pipx"; pipx install 'cocoindex-code[full]'
else
  say "  ⚠ pipx not found. Install it, then re-run:"
  say "      python3 -m pip install --user pipx && python3 -m pipx ensurepath"
  say "      pipx install 'cocoindex-code[full]'"
fi

# --- 2. Skill / "extension" — user-level, for every runtime the cross-agent installer knows -------
say "[2/6] Skill (extension)"
if [ -d "$HOME/.claude/skills/ccc" ] || [ -d "$HOME/.agents/skills/ccc" ] || [ -d "$HOME/.config/opencode/skills/ccc" ]; then
  say "  ccc skill already installed"
elif have npx; then
  say "  installing via: npx skills add cocoindex-io/cocoindex-code -g -a claude-code -a opencode -a github-copilot -a codex -y"
  npx -y skills add cocoindex-io/cocoindex-code -g -a claude-code -a opencode -a github-copilot -a codex -y </dev/null >/dev/null 2>&1 \
    || say "  ⚠ skill install failed — run manually: npx skills add cocoindex-io/cocoindex-code -g -a claude-code -a opencode -a github-copilot -a codex -y"
else
  say "  ⚠ npx not found — install the skill manually: npx skills add cocoindex-io/cocoindex-code -g -a claude-code -a opencode -a github-copilot -a codex -y"
fi

# The USER-LEVEL registrations (Claude Code, Copilot CLI) use the resolved binary path: the runtimes do not
# always inherit the shell PATH that pipx extends (~/.local/bin), and a bare "ccc" then fails with ENOENT.
# The project file opencode.json keeps a portable "ccc" (it may be committed and shared).
CCC_BIN="$(ccc_bin)"
# mcp_cmd <json-file> — the command the cocoindex-code server is registered with in that file (empty if none).
mcp_cmd(){ python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("mcpServers",{}).get("cocoindex-code",{}).get("command",""))' "$1" 2>/dev/null; }

# --- 3. MCP in Claude Code (user scope: the CLI is machine-wide, so is its MCP) -----------------
say "[3/6] MCP · Claude Code"
if have claude && [ -n "$CCC_BIN" ]; then
  CUR="$(mcp_cmd "$HOME/.claude.json")"
  if [ "$CUR" = "$CCC_BIN" ]; then
    say "  already registered ($CCC_BIN)"
  else
    # A bare or stale command (or an old local-scope entry) is replaced by the absolute path.
    claude mcp remove --scope local cocoindex-code >/dev/null 2>&1 || true
    [ -n "$CUR" ] && claude mcp remove --scope user cocoindex-code >/dev/null 2>&1
    claude mcp add --scope user cocoindex-code -- "$CCC_BIN" mcp >/dev/null 2>&1 \
      && say "  + claude mcp add --scope user cocoindex-code -- $CCC_BIN mcp${CUR:+   (was: $CUR)}" \
      || say "  ⚠ run manually: claude mcp add --scope user cocoindex-code -- $CCC_BIN mcp"
  fi
elif have claude; then
  say "  ⚠ ccc not installed yet — re-run this script after step 1 succeeds"
else
  say "  claude CLI not found — skipping (run later: claude mcp add --scope user cocoindex-code -- ccc mcp)"
fi

# --- 4. MCP in GitHub Copilot CLI (user config ~/.copilot/mcp-config.json) ----------------------
say "[4/6] MCP · GitHub Copilot CLI"
if have copilot && [ -n "$CCC_BIN" ]; then
  CUR="$(mcp_cmd "$HOME/.copilot/mcp-config.json")"
  if [ "$CUR" = "$CCC_BIN" ]; then
    say "  already registered ($CCC_BIN)"
  else
    [ -n "$CUR" ] && copilot mcp remove cocoindex-code >/dev/null 2>&1
    copilot mcp add cocoindex-code -- "$CCC_BIN" mcp >/dev/null 2>&1 \
      && say "  + copilot mcp add cocoindex-code -- $CCC_BIN mcp${CUR:+   (was: $CUR)}" \
      || say "  ⚠ run manually: copilot mcp add cocoindex-code -- $CCC_BIN mcp"
  fi
elif have copilot; then
  say "  ⚠ ccc not installed yet — re-run this script after step 1 succeeds"
else
  say "  copilot CLI not found — skipping (run later: copilot mcp add cocoindex-code -- ccc mcp)"
fi

# --- 5. MCP in opencode (merge opencode.json at repo root) ---------------------
say "[5/6] MCP · opencode"
OCJSON="$TARGET/opencode.json"
python3 - "$OCJSON" <<'PY' && say "  merged mcp.cocoindex-code into opencode.json" || say "  ⚠ could not write opencode.json — add an mcp entry { type:local, command:[ccc,mcp] }"
import json, os, sys
p = sys.argv[1]
data = {}
if os.path.exists(p):
    try: data = json.load(open(p))
    except Exception: data = {}
data.setdefault("$schema", "https://opencode.ai/config.json")
mcp = data.setdefault("mcp", {})
mcp["cocoindex-code"] = {"type": "local", "command": ["ccc", "mcp"], "enabled": True}   # project file: keep it portable
json.dump(data, open(p, "w"), indent=2)
open(p, "a").write("\n")
PY

# --- 6. Initialize + index the repo -------------------------------------------
say "[6/6] Index the repo"
if [ -z "$CCC_BIN" ]; then
  say "  ⚠ ccc not on PATH yet — after installing, run from the repo:  ccc init && ccc index"
elif [ "$SKIP_INDEX" = 1 ]; then
  say "  skipped (--skip-index). Run later:  cd $TARGET && ccc init && ccc index"
else
  ( cd "$TARGET"
    if "$CCC_BIN" status >/dev/null 2>&1; then
      say "  project already initialized — refreshing index"; "$CCC_BIN" index || true
    else
      # First-ever run prompts for an embedding provider (interactive). Try non-interactive;
      # if it needs a TTY, guide the user.
      if "$CCC_BIN" init -f >/dev/null 2>&1; then "$CCC_BIN" index || true; say "  initialized + indexed"
      else say "  ⚠ 'ccc init' needs a one-time interactive choice. Run once:  cd $TARGET && ccc init && ccc index"
      fi
    fi )
fi

say ""
say "Done. The dev agent can now use semantic search (ccc / the ccc skill / cocoindex-code MCP)."
