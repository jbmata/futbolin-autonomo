#!/usr/bin/env bash
# House skills: caveman, ponytail, rtk and ccc stay active in every agent, subagent and unattended run of a
# factory project. There is one source of truth. The levels live in `house_skills:` in .factory/config.yml
# and the rules live in the "House skills" section of AGENTS.md. This script only wires them into each runtime.
#
# Usage:
#   house-skills.sh apply [DIR]    write the project wiring: .claude/settings.json, .caveman/config.json, opencode.json
#                                  (settings.json also references the factory plugin at its origin marketplace)
#   house-skills.sh hook <Event>   Claude Code hook (SessionStart | SubagentStart): the rules as additionalContext.
#                                  UserPromptSubmit: the open items of PENDIENTES.md, only when that set changed since
#                                  it was last shown, and never in an unattended run (FACTORY_UNATTENDED=1)
#   house-skills.sh rtk            Claude Code PreToolUse(Bash) hook: hand off to `rtk hook claude` only when on + installed
#   house-skills.sh check [DIR]    print ✅/❌ for each piece; exit 1 when a required piece is missing
#
# Works offline and never prints secrets. The hooks cannot break a session: any problem means no output and exit 0.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
. "$SCRIPT_DIR/_lib.sh"

CMD="${1:-}"; [ $# -gt 0 ] && shift
case "$CMD" in
  hook|rtk) TARGET="$(resolve_target "${CLAUDE_PROJECT_DIR:-$SCRIPT_DIR/../../..}")" || exit 0 ;;
  apply|check) TARGET="$(resolve_target "${1:-}")" || { say "Target dir not found"; exit 1; } ;;
  *) say "usage: house-skills.sh apply [DIR] | hook <SessionStart|SubagentStart|UserPromptSubmit> | rtk | check [DIR]"; exit 2 ;;
esac
CONFIG_FILE="$TARGET/.factory/config.yml"

# Double injection guard. The same hooks reach a project from two places: install.sh writes them into the project
# settings (pointing at .factory/assets/scripts), and the plugin or an APM package brings them again (hooks/hooks.json, or
# a copy of this script under .claude/). A copy of this script that is not the project's own stays silent when the
# project settings already register the project's own copy for the same hook, so the rules reach the agent once.
if [ "$CMD" = hook ] || [ "$CMD" = rtk ]; then
  # Not a factory project: a plugin enabled for the user brings these hooks into every repo, so stay silent where
  # there is no .factory/ (the rules would be imposed on repos that never asked for them, and rtk would rewrite commands).
  [ -d "$TARGET/.factory" ] || exit 0
  if [ "$(cd "$TARGET/.factory/assets/scripts" 2>/dev/null && pwd)" != "$SCRIPT_DIR" ]; then
    for f in "$TARGET/.claude/settings.json" "$TARGET/.claude/settings.local.json"; do
      grep -qE "\.factory/assets/scripts/house-skills\.sh[^ ]* $CMD${1:+ $1}( |\"|\$)" "$f" 2>/dev/null && exit 0
    done
  fi
fi

# UserPromptSubmit runs before every prompt a person types, so it is answered first and cheaply, before any config
# parsing. PENDIENTES.md is the ultra-autonomous mode's channel to people (format: .factory/assets/templates/).
if [ "$CMD" = hook ] && [ "${1:-}" = UserPromptSubmit ]; then
  case "${FACTORY_UNATTENDED:-0}" in 0|"") ;; *) exit 0 ;; esac   # unattended run: nobody is reading
  PEND="$TARGET/PENDIENTES.md"; SHOWN="$TARGET/.factory/state/pendientes-shown"
  grep -qiE 'estado:\*\*[[:space:]]*abierta' "$PEND" 2>/dev/null || { rm -f "$SHOWN" 2>/dev/null; exit 0; }
  python3 - "$PEND" "$SHOWN" <<'PY' 2>/dev/null || exit 0
import json, os, re, sys
path, shown = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    text = f.read().replace("\r\n", "\n")
text = re.sub(r"<!--.*?-->", "", text, flags=re.S)   # the template keeps its example entry inside a comment

def field(body, key):
    m = re.search(rf"^\s*-\s*\*\*{key}:\*\*[ \t]*(.*)$", body, re.M | re.I)
    return m.group(1).strip() if m else ""

items = []
for m in re.finditer(r"^## (P-\d+)[ \t]*[·:—-]?[ \t]*([^\n]*)\n(.*?)(?=^## |\Z)", text, re.S | re.M):
    pid, title, body = m.groups()
    if field(body, "Estado").lower().startswith("abierta"):
        items.append((pid, title.strip(), field(body, "Tipo").lower(), field(body, "Fase/tarea")))
key = ",".join(sorted(i[0] for i in items))
seen = ""
if os.path.exists(shown):
    with open(shown, encoding="utf-8") as f:
        seen = f.read().strip()
if not items or seen == key:
    sys.exit(0)
rank = {"bloqueo": 0, "aprobación": 1, "aprobacion": 1, "decisión": 2, "decision": 2, "duda": 3}
items.sort(key=lambda i: rank.get(i[2], 4))   # stable: same type keeps file order (oldest first)
top = [f"- {pid} [{tipo or '?'}] {title}" + (f" ({where})" if where else "") for pid, title, tipo, where in items[:5]]
if len(items) > 5:
    top.append(f"- ... and {len(items) - 5} more in PENDIENTES.md")
note = (f"PENDIENTES.md has {len(items)} open item(s) that the factory left for a person while it worked alone "
        "(mode: ultra-autonomous). Start your reply with a short note in the user's language: how many are open and "
        "the top ones below. Then answer the user's message as usual. When the user answers an item, record it in "
        "PENDIENTES.md: the answer under Respuesta and Estado: resuelta. factory-build applies the answers on its "
        "next run or phase.\n" + "\n".join(top))
print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": note}}))
try:
    os.makedirs(os.path.dirname(shown), exist_ok=True)
    with open(shown, "w", encoding="utf-8") as f:
        f.write(key + "\n")
except OSError:
    pass   # read-only checkout: the note repeats on the next prompt, it never breaks one
PY
  exit 0
fi

# level <piece> — the house_skills value, or the factory default when config.yml does not set it (they are a must).
level(){
  local v; v="$(cfg_get "house_skills.$1")"
  if [ -z "$v" ]; then
    case "$1" in caveman|ponytail) v=ultra ;; *) v=on ;; esac
  fi
  printf '%s' "$v"
}
# ccc_bin — absolute path of ccc. pipx puts it in ~/.local/bin, which runtimes do not always have on PATH.
ccc_bin(){ command -v ccc 2>/dev/null || { [ -x "$HOME/.local/bin/ccc" ] && printf '%s' "$HOME/.local/bin/ccc"; }; }

CAVEMAN="$(level caveman)"; PONYTAIL="$(level ponytail)"; RTK="$(level rtk)"; CCC="$(level ccc)"
# The factory-* skills are referenced at their origin (marketplace + plugin), never copied into the project.
MKT="$(cfg_get skills.marketplace)"; MKT="${MKT:-fastex-factory}"
SRC="$(cfg_get skills.source)"; SRC="${SRC:-https://github.com/Fastex-Soluciones-IT/factoria-software}"
HAVE_RTK=0; have rtk && HAVE_RTK=1

case "$CMD" in
rtk)
  # Rewrite the call only when rtk is on, on PATH (the rewritten command runs `rtk …`) and no user-level hook
  # already does it. Anything else means the command runs unchanged.
  [ "$RTK" = on ] && [ "$HAVE_RTK" = 1 ] || exit 0
  grep -q 'rtk hook' "$HOME/.claude/settings.json" 2>/dev/null && exit 0
  exec rtk hook claude
  ;;

hook)
  python3 - "${1:-SubagentStart}" "$TARGET" "$CAVEMAN" "$PONYTAIL" "$RTK" "$CCC" "$HAVE_RTK" "$(ccc_bin)" <<'PY' 2>/dev/null || exit 0
import json, pathlib, re, sys
event, target, caveman, ponytail, rtk, ccc, have_rtk, ccc_bin = sys.argv[1:]
root = pathlib.Path(target)
agents = (root / "AGENTS.md").read_text(encoding="utf-8") if (root / "AGENTS.md").exists() else ""
m = re.search(r"<!-- HOUSE:BEGIN -->\n(.*?)<!-- HOUSE:END -->", agents, re.S)
rules = m.group(1).strip() if m else "Read the House skills section of AGENTS.md before you start."
state = [f"caveman={caveman}", f"ponytail={ponytail}", f"rtk={rtk}", f"ccc={ccc}"]
notes = []
if rtk == "on" and have_rtk != "1":
    notes.append("rtk is not on PATH here: run commands normally, never prefix them with rtk.")
if ccc == "on" and not ccc_bin:
    notes.append("ccc is not installed here: use your normal search tools until setup.sh installs it.")
elif ccc == "on" and not (root / ".cocoindex_code" / "target_sqlite.db").exists():
    notes.append("ccc has no index for this repo yet: run `ccc index` once, then search with it.")
text = f"House skills active (house_skills in .factory/config.yml): {' · '.join(state)}. A piece set to off does not apply.\n"
text += "".join(f"- {n}\n" for n in notes) + "\n" + rules
print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}))
PY
  ;;

apply)
  python3 - "$TARGET" "$CAVEMAN" "$PONYTAIL" "$RTK" "$HAVE_RTK" "$MKT" "$SRC" <<'PY'
import json, os, sys
target, caveman, ponytail, rtk, have_rtk, mkt, src = sys.argv[1:]
MARK = "house-skills.sh"
SCRIPT = '"$CLAUDE_PROJECT_DIR/.factory/assets/scripts/house-skills.sh"'

def load(p):
    try:
        with open(p, encoding="utf-8") as f: return json.load(f)
    except (OSError, ValueError): return {}

def save(p, data):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2); f.write("\n")
    print(f"  ~ {os.path.relpath(p, target)}")

def set_hook(hooks, event, command, matcher=None):
    # Drop our previous entry (idempotent), keep everybody else's, then add ours when wanted.
    groups = [g for g in hooks.get(event, []) if not any(MARK in h.get("command", "") for h in g.get("hooks", []))]
    if command:
        g = {"hooks": [{"type": "command", "command": command}]}
        if matcher: g = {"matcher": matcher, **g}
        groups.append(g)
    if groups: hooks[event] = groups
    else: hooks.pop(event, None)

# Claude Code: house rules for the main session and every subagent, rtk only where the binary exists, plugins.
p = os.path.join(target, ".claude", "settings.json")
s = load(p)
hooks = s.setdefault("hooks", {})
for ev in ("SessionStart", "SubagentStart", "UserPromptSubmit"):
    set_hook(hooks, ev, f"bash {SCRIPT} hook {ev} 2>/dev/null || true")
set_hook(hooks, "PreToolUse", f"bash {SCRIPT} rtk 2>/dev/null || true" if rtk == "on" and have_rtk == "1" else None, "Bash")
if not hooks: s.pop("hooks")
mk = s.setdefault("extraKnownMarketplaces", {})
mk["ponytail"] = {"source": {"source": "github", "repo": "DietrichGebert/ponytail"}}
mk["caveman"] = {"source": {"source": "github", "repo": "JuliusBrussee/caveman"}}
plugins = s.setdefault("enabledPlugins", {})
# The factory plugin from its origin repo: Claude Code offers to install it when the repo is trusted.
if src.startswith("https://github.com/"):
    mk[mkt] = {"source": {"source": "github", "repo": src[len("https://github.com/"):].removesuffix(".git")}}
elif "://" in src:
    mk[mkt] = {"source": {"source": "git", "url": src}}
plugins[f"software-factory@{mkt}"] = True
plugins["ponytail@ponytail"] = ponytail != "off"
plugins["caveman@caveman"] = caveman != "off"
# ponytail reads its level only from PONYTAIL_DEFAULT_MODE or a user-level file, never from the project: the env
# block of the project settings carries house_skills.ponytail into every Claude Code session and its hooks.
env = s.setdefault("env", {})
if ponytail != "off": env["PONYTAIL_DEFAULT_MODE"] = ponytail
else: env.pop("PONYTAIL_DEFAULT_MODE", None)
if not env: s.pop("env")
save(p, s)

# caveman: the level every Claude Code session starts with.
p = os.path.join(target, ".caveman", "config.json")
c = load(p); c["defaultMode"] = caveman
save(p, c)

# opencode reads AGENTS.md natively in every agent and subagent, so only the ponytail plugin is wired here.
p = os.path.join(target, "opencode.json")
o = load(p); o.setdefault("$schema", "https://opencode.ai/config.json")
lst = [x for x in o.get("plugin", []) if x != "@dietrichgebert/ponytail"]
if ponytail != "off": lst.append("@dietrichgebert/ponytail")
if lst: o["plugin"] = lst
else: o.pop("plugin", None)
save(p, o)
PY
  if [ "$RTK" = on ] && [ "$HAVE_RTK" = 0 ]; then
    say "  ⚠ rtk is on but not installed: no rtk hook was written. Install rtk and re-run to enable it."
  fi
  ;;

check)
  python3 - "$TARGET" "$CAVEMAN" "$PONYTAIL" "$RTK" "$CCC" "$HAVE_RTK" "$(ccc_bin)" "$(grep -c '^house_skills:' "$CONFIG_FILE" 2>/dev/null)" "$MKT" <<'PY'
import glob, json, os, sys
target, caveman, ponytail, rtk, ccc, have_rtk, ccc_bin, cfg, mkt = sys.argv[1:]
home = os.path.expanduser("~")
failed = False

def load(p):
    try:
        with open(p, encoding="utf-8") as f: return json.load(f)
    except (OSError, ValueError): return {}

def row(ok, piece, detail, required=True):
    global failed
    mark = "✅" if ok else ("❌" if required else "⚠️ ")
    failed |= required and not ok
    print(f"  {mark} {piece:<10} {detail}")

def cmds(settings, event):
    return [h.get("command", "") for g in settings.get("hooks", {}).get(event, []) for h in g.get("hooks", [])]

s = load(os.path.join(target, ".claude", "settings.json"))
agents = open(os.path.join(target, "AGENTS.md"), encoding="utf-8").read() if os.path.exists(os.path.join(target, "AGENTS.md")) else ""
row(cfg != "0" and cfg != "", "config", f"house_skills in .factory/config.yml: caveman={caveman} ponytail={ponytail} rtk={rtk} ccc={ccc}")
row("<!-- HOUSE:BEGIN -->" in agents, "rules", "House skills section in AGENTS.md")
vendored = sorted(os.path.basename(d) for d in glob.glob(os.path.join(target, ".claude", "skills", "factory-*")))
row(s.get("enabledPlugins", {}).get(f"software-factory@{mkt}") is True and not vendored, "skills",
    f"software-factory@{mkt} referenced at its origin, no copies in the repo" if not vendored
    else f"copied skills in .claude/skills ({', '.join(vendored)}): re-run install.sh, they come from the plugin")
hooked = all(any("house-skills.sh" in c for c in cmds(s, ev)) for ev in ("SessionStart", "SubagentStart"))
row(hooked and os.path.exists(os.path.join(target, ".factory", "assets", "scripts", "house-skills.sh")),
    "hooks", "SessionStart + SubagentStart give the rules to the session and every subagent")
row(any("house-skills.sh" in c for c in cmds(s, "UserPromptSubmit")), "pendientes",
    "UserPromptSubmit tells a person about open PENDIENTES.md items (silent when FACTORY_UNATTENDED=1)")
if ponytail != "off":
    row(s.get("enabledPlugins", {}).get("ponytail@ponytail") is True, "ponytail", "plugin enabled in .claude/settings.json")
    row(s.get("env", {}).get("PONYTAIL_DEFAULT_MODE") == ponytail, "ponytail", f"PONYTAIL_DEFAULT_MODE={ponytail} in .claude/settings.json")
    row("@dietrichgebert/ponytail" in load(os.path.join(target, "opencode.json")).get("plugin", []), "ponytail", "plugin listed in opencode.json")
if caveman != "off":
    row(load(os.path.join(target, ".caveman", "config.json")).get("defaultMode") == caveman, "caveman", f".caveman/config.json defaultMode={caveman}")
    row(s.get("enabledPlugins", {}).get("caveman@caveman") is True, "caveman", "plugin enabled in .claude/settings.json")
if rtk == "on":
    project_hook = any("house-skills.sh" in c for c in cmds(s, "PreToolUse"))
    if have_rtk == "1":
        user_hook = any("rtk hook" in c for c in cmds(load(os.path.join(home, ".claude", "settings.json")), "PreToolUse"))
        row(project_hook or user_hook, "rtk", "installed, PreToolUse(Bash) hook active" if project_hook or user_hook else "installed, hook missing: re-run house-skills.sh apply")
    else:
        row(not project_hook, "rtk", "not on PATH: hook switched off, commands run unchanged" if not project_hook
            else "not on PATH but the project hook is still written: re-run house-skills.sh apply", required=project_hook)
    for c in cmds(load(os.path.join(home, ".claude", "settings.json")), "PreToolUse"):
        if "rtk hook" in c and have_rtk != "1":
            row(False, "rtk", f"user hook `{c}` rewrites commands to `rtk …` but rtk is not on PATH: every command fails with `rtk: command not found`. Add its directory to PATH or remove the hook from ~/.claude/settings.json", required=False)
if ccc == "on":
    row(bool(ccc_bin), "ccc", f"binary {ccc_bin}" if ccc_bin else "binary not installed: run setup-cocoindex.sh")
    row(os.path.exists(os.path.join(target, ".cocoindex_code", "target_sqlite.db")), "ccc", "repo index (.cocoindex_code/)" if os.path.exists(os.path.join(target, ".cocoindex_code", "target_sqlite.db")) else "no index: run `ccc init && ccc index`")
    cj = load(os.path.join(home, ".claude.json"))
    srv = cj.get("mcpServers", {}).get("cocoindex-code") or cj.get("projects", {}).get(target, {}).get("mcpServers", {}).get("cocoindex-code") \
        or load(os.path.join(target, ".mcp.json")).get("mcpServers", {}).get("cocoindex-code")
    cmd = (srv or {}).get("command", "")
    row(bool(cmd) and os.path.isabs(cmd) and os.path.exists(cmd), "ccc",
        f"MCP cocoindex-code → {cmd}" if cmd and os.path.isabs(cmd) and os.path.exists(cmd)
        else (f"MCP cocoindex-code uses `{cmd}`, which fails with 'Executable not found in $PATH': re-run setup-cocoindex.sh" if cmd else "MCP cocoindex-code not registered: run setup-cocoindex.sh"))
sys.exit(1 if failed else 0)
PY
  ;;
esac
