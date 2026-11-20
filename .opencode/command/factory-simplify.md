---
description: Minimalism review of the current diff (ponytail-native) — flag and cut over-engineering, never weakening security / accessibility / data-loss / trust-boundary code. Quality over quantity.
agent: build
argument-hint: "[optional: a path, or 'repo' for a full audit instead of just the diff]"
---

Invoke the **factory-simplify** skill to review the current change for over-engineering and cut it to the
minimum that works (quality over quantity), protecting security / accessibility / data-loss / trust
boundaries. If an argument is given, scope to it (e.g. `repo` for a full-repo audit): $ARGUMENTS

House skills (AGENTS.md § House skills, levels in `house_skills:` of `.factory/config.yml`): caveman, ponytail, rtk
and ccc apply to you and to every subagent you start; put that section at the top of each subagent prompt.
