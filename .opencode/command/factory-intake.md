---
description: Ingest cowork intake from .factory/intake/ — validate schema, pre-fill factory-init, run sign-off gate.
agent: build
argument-hint: "[optional path to intake directory, defaults to .factory/intake/]"
---

Invoke the **factory-intake** skill to validate the intake artifact bundle and run the sign-off gate.

If the user passed a path, use it; otherwise ingest from `.factory/intake/`: $ARGUMENTS

House skills (AGENTS.md § House skills, levels in `house_skills:` of `.factory/config.yml`): caveman, ponytail, rtk
and ccc apply to you and to every subagent you start; put that section at the top of each subagent prompt.
