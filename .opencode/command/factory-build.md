---
description: Run the autonomous build loop — drive gsd phases, gate each on docker compose run, sync issues, until success criteria are met.
agent: build
argument-hint: "[optional: phase name, 'autonomous' or 'ultra-autonomous']"
---

Invoke the **factory-build** skill and follow it end to end.

If the user specified a phase or mode, honor it: $ARGUMENTS

House skills (AGENTS.md § House skills, levels in `house_skills:` of `.factory/config.yml`): caveman, ponytail, rtk
and ccc apply to you and to every subagent you start; put that section at the top of each subagent prompt.
