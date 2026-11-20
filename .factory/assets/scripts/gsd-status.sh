#!/usr/bin/env bash
# Software Factory — `status` verb of the gsd adapter (.factory/sdd/gsd/ADAPTER.md §2b).
# Builds the factory status JSON from three GSD Core gsd-tools outputs (roadmap analyze --json, state json,
# progress --json) plus the success criteria in FACTORY.md and the blocked items the tracker reports.
# Read-only: gsd-tools does not write in these three commands, and this script writes nothing.
#
# Usage:   gsd-status.sh [--blocked FILE|-] [TARGET_DIR]
#          --blocked: JSON array of the tracker's blocked items (e.g. `gh issue list --label factory:blocked --json
#          number,title,url`), read from FILE or from stdin with "-". Without it, "blocked" is [].
# Tools:   GSD_TOOLS=<path to gsd-tools.cjs> overrides the lookup (gsd-tools on PATH, then the gsd-core/ dir of a
#          local or global install for Claude Code, opencode or Copilot CLI).
# Exit:    0 JSON on stdout | 1 gsd-tools failed | 2 usage or no .planning/ROADMAP.md | 3 GSD Core not installed
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
. "$SCRIPT_DIR/_lib.sh"

BLOCKED=""; TARGET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --blocked) BLOCKED="${2:-}"; shift 2 ;;
    -*) say "usage: gsd-status.sh [--blocked FILE|-] [TARGET_DIR]" >&2; exit 2 ;;
    *)  TARGET="$1"; shift ;;
  esac
done
TARGET="$(resolve_target "$TARGET")" || { say "Target dir not found" >&2; exit 2; }
[ -f "$TARGET/.planning/ROADMAP.md" ] || { say "no .planning/ROADMAP.md in $TARGET — run /factory-discover first" >&2; exit 2; }
have node || { say "node not found — GSD Core needs Node >= 24" >&2; exit 3; }

TOOLS="${GSD_TOOLS:-}"
if [ -z "$TOOLS" ]; then
  if have gsd-tools; then TOOLS="$(command -v gsd-tools)"
  else
    for d in "$TARGET/.claude" "${CLAUDE_CONFIG_DIR:-$HOME/.claude}" \
             "${OPENCODE_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/opencode}" \
             "${COPILOT_CONFIG_DIR:-${COPILOT_HOME:-$HOME/.copilot}}"; do
      [ -f "$d/gsd-core/bin/gsd-tools.cjs" ] && { TOOLS="$d/gsd-core/bin/gsd-tools.cjs"; break; }
    done
  fi
fi
[ -n "$TOOLS" ] || { say "GSD Core not installed (no gsd-tools) — run .factory/assets/scripts/setup-sdd.sh gsd" >&2; exit 3; }
# An installed gsd-core/ dir carries VERSION; an npm package (gsd-tools on PATH) only its package.json.
BIN_DIR="$(dirname "$(readlink -f "$TOOLS")")"
VERSION="$(cat "$BIN_DIR/../VERSION" 2>/dev/null || node -p "require('$BIN_DIR/../../package.json').version" 2>/dev/null || true)"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
run(){ ( cd "$TARGET" && node "$TOOLS" "$@" ) >"$TMP/$1.json" || { say "gsd-tools $* failed" >&2; exit 1; }; }
run roadmap analyze --json
run state json
run progress --json
case "$BLOCKED" in
  "") echo '[]' >"$TMP/blocked.json" ;;
  -)  cat >"$TMP/blocked.json" ;;
  *)  cp "$BLOCKED" "$TMP/blocked.json" || exit 2 ;;
esac
[ -f "$TARGET/FACTORY.md" ] && cp "$TARGET/FACTORY.md" "$TMP/FACTORY.md"

node - "$TMP" "$VERSION" <<'JS'
const fs = require('fs'), path = require('path');
const [dir, version] = process.argv.slice(2);
const read = f => JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'));
const roadmap = read('roadmap.json'), state = read('state.json'), progress = read('progress.json');
const blocked = read('blocked.json');
if (!Array.isArray(blocked)) { console.error('--blocked must be a JSON array'); process.exit(2); }

// `roadmap analyze` prints phase numbers without padding ("1", "2.1"); `progress --json` pads them ("01", "02.1").
// Strip the leading zeros of every segment so both map to the same key, phase-N, the key factory-sync uses.
const norm = n => (n == null ? null : String(n).replace(/(^|\.)0+(?=\d)/g, '$1'));
const phases = new Map();
for (const p of roadmap.phases || []) {
  phases.set(norm(p.number), { number: norm(p.number), name: p.name, disk_status: p.disk_status,
    tasks_done: p.summary_count, tasks_total: p.plan_count });
}
for (const p of progress.phases || []) {          // phases on disk that ROADMAP.md does not list yet
  const k = norm(p.number);
  if (!phases.has(k)) phases.set(k, { number: k, name: p.name, disk_status: p.status === 'Complete' ? 'complete' : null,
    tasks_done: p.summaries, tasks_total: p.plans });
}
const current = norm(state.current_phase ?? roadmap.current_phase);
const out = [...phases.values()].map(p => ({
  key: `phase-${p.number}`, number: p.number, name: p.name,
  status: p.disk_status === 'complete' ? 'done'
    : (p.number === current || p.tasks_done > 0 || p.disk_status === 'executed') ? 'in-progress' : 'pending',
  disk_status: p.disk_status, tasks_done: p.tasks_done, tasks_total: p.tasks_total,
}));

// FACTORY.md: the checkbox list under "## Success criteria".
const criteria = [];
const factoryMd = path.join(dir, 'FACTORY.md');
if (fs.existsSync(factoryMd)) {
  let inside = false;
  for (const line of fs.readFileSync(factoryMd, 'utf8').split('\n')) {
    if (/^##\s/.test(line)) inside = /^##\s+Success criteria/i.test(line);
    const m = inside && line.match(/^\s*[-*]\s+\[([ xX])\]\s+(.*\S)/);
    if (m) criteria.push({ text: m[2], done: m[1] !== ' ' });
  }
}

const sum = k => out.reduce((a, p) => a + (p[k] || 0), 0);
console.log(JSON.stringify({
  engine: 'gsd',
  version: version.trim() || null,
  milestone: { version: state.milestone ?? progress.milestone_version ?? null,
    name: state.milestone_name ?? progress.milestone_name ?? null },
  status: state.status ?? null,
  current_phase: current ? `phase-${current}` : null,
  phases: out,
  totals: { phases_done: out.filter(p => p.status === 'done').length, phases_total: out.length,
    tasks_done: sum('tasks_done'), tasks_total: sum('tasks_total'),
    percent: progress.percent ?? roadmap.progress_percent ?? null },   // null when GSD cannot scope it; never guessed
  criteria,
  blocked,
}, null, 2));
JS
