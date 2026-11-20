---
description: Package a complete, confirmed discovery into branch intake/<slug> + a PR with the technical owner as reviewer — the functional → technical handoff.
agent: build
argument-hint: "[optional feature name or path to intake subfolder, e.g. .factory/intake/<slug>]"
---

Invoke the **factory-handoff** skill to package the confirmed discovery and open (or update) its Pull Request.

Check the trigger gate first (complete intake.yml, no secrets, explicit user confirmation); then create branch `intake/<slug>`, push, and open a PR with the technical owner as reviewer: $ARGUMENTS

House skills (AGENTS.md § House skills, levels in `house_skills:` of `.factory/config.yml`): caveman, ponytail, rtk
and ccc apply to you and to every subagent you start; put that section at the top of each subagent prompt.
