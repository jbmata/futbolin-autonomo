#!/usr/bin/env bash
# Test gate: one deterministic place that runs the project's tests and enforces a minimum coverage.
# factory-build (after `verify`, and again after `factory-simplify`) and factory-quality call it, so "the tests pass"
# is an exit code, not a judgement of the agent.
#
# Usage:
#   test-gate.sh run [--dry-run] [DIR]
#
# What it runs, in order of precedence:
#   1. gates.tests.cmd in .factory/config.yml            one command, run at the project root
#   2. test_cmd of the active starterkit (catalog.yaml)  one command per component, run in <DIR>/<component>
#                                                        (component "." = the project root)
# gates.tests.coverage_min (percent of lines, 0 = report only) is checked against the coverage.xml / lcov.info files
# and Go coverage.out profiles (statements) that the run produced. `gates.tests: false` skips the gate.
#
# Exit: 0 pass (or skipped) | 1 a test run failed or coverage is below the minimum | 2 usage | 3 nothing to run
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
. "$SCRIPT_DIR/_lib.sh"

CMD="${1:-}"; [ $# -gt 0 ] && shift
[ "$CMD" = run ] || { say "usage: test-gate.sh run [--dry-run] [DIR]"; exit 2; }
DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }
TARGET="$(resolve_target "${1:-}")" || { say "Target dir not found"; exit 2; }
CONFIG_FILE="$TARGET/.factory/config.yml"
CATALOG="$TARGET/.factory/assets/starterkits/catalog.yaml"

[ "$(cfg_get gates.tests)" = false ] && { say "test gate: disabled (gates.tests: false)"; exit 0; }

# jobs: one "<dir><TAB><command>" line per test run.
JOBS="$(cfg_get gates.tests.cmd)"
if [ -n "$JOBS" ]; then
  JOBS="$(printf '.\t%s' "$JOBS")"
else
  SK="$(cfg_get starterkit)"
  [ -n "$SK" ] && [ -f "$CATALOG" ] && JOBS="$(awk -v id="$SK" '
    /^  - id:/ { inkit = ($3 == id); intc = 0; next }
    inkit && /^      test_cmd:/ { intc = 1; next }
    intc && /^        [A-Za-z0-9_.-]+:/ {
      key = $1; sub(/:$/, "", key)
      val = $0; sub(/^[^:]*:[[:space:]]*/, "", val); gsub(/^"|"$/, "", val)
      print key "\t" val; next }
    intc { intc = 0 }
  ' "$CATALOG")"
fi
[ -n "$JOBS" ] || { say "test gate: no test command. Set gates.tests.cmd in .factory/config.yml (or a starterkit with test_cmd)."; exit 3; }

MARK="$(mktemp)"; trap 'rm -f "$MARK"' EXIT
FAILED=0
while IFS=$'\t' read -r dir cmd; do
  say "▶ ${dir}: ${cmd}"
  [ "$DRY" = 1 ] && continue
  if [ ! -d "$TARGET/$dir" ]; then say "❌ ${dir}: directory not found"; FAILED=1; continue; fi
  if (cd "$TARGET/$dir" && bash -c "$cmd"); then say "✅ ${dir}: tests passed"; else say "❌ ${dir}: tests failed (exit $?)"; FAILED=1; fi
done <<< "$JOBS"
[ "$DRY" = 1 ] && exit 0

# Coverage: only files this run wrote count, so a stale report from an earlier run never satisfies the gate.
MIN="$(cfg_get gates.tests.coverage_min)"; MIN="${MIN:-0}"
if ! python3 - "$TARGET" "$MARK" "$MIN" <<'PY'
import os, re, sys, xml.etree.ElementTree as ET
target, mark, minimum = sys.argv[1], os.path.getmtime(sys.argv[2]), float(sys.argv[3])
skip = {"node_modules", ".venv", "venv", ".git"}
covered = valid = 0
for root, dirs, files in os.walk(target):
    dirs[:] = [d for d in dirs if d not in skip]
    for f in files:
        p = os.path.join(root, f)
        if f not in ("coverage.xml", "lcov.info", "coverage.out") or os.path.getmtime(p) < mark:
            continue
        if f == "coverage.out":  # go test -coverprofile: "file:start,end statements count"; a block can repeat
            blocks = {}
            for line in open(p, encoding="utf-8", errors="replace"):
                parts = line.split()
                if len(parts) != 3 or parts[0] == "mode:":
                    continue
                n, hit = int(parts[1]), int(parts[2]) > 0
                blocks[parts[0]] = (n, hit or blocks.get(parts[0], (0, False))[1])
            covered += sum(n for n, hit in blocks.values() if hit)
            valid += sum(n for n, _ in blocks.values())
        elif f == "lcov.info":
            txt = open(p, encoding="utf-8", errors="replace").read()
            covered += sum(map(int, re.findall(r"^LH:(\d+)", txt, re.M)))
            valid += sum(map(int, re.findall(r"^LF:(\d+)", txt, re.M)))
        else:
            r = ET.parse(p).getroot()
            if r.get("lines-valid"):
                covered += int(r.get("lines-covered")); valid += int(r.get("lines-valid"))
            else:  # cobertura without line counts: weigh the rate as 100 lines
                covered += float(r.get("line-rate", 0)) * 100; valid += 100
pct = 100 * covered / valid if valid else None
if pct is None:
    print(f"coverage: no report produced by this run (min {minimum:g}%)")
    sys.exit(1 if minimum > 0 else 0)
ok = pct >= minimum
print(f"{'✅' if ok else '❌'} coverage: {pct:.1f}% of lines (min {minimum:g}%)")
sys.exit(0 if ok else 1)
PY
then FAILED=1; fi

if [ "$FAILED" = 0 ]; then say "TEST GATE: PASS"; else say "TEST GATE: FAIL"; fi
exit "$FAILED"
