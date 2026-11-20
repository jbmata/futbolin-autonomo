---
description: Bootstrap the Software Factory into this repo — detect state, run discovery questionnaire, scaffold starterkit, write FACTORY.md + .factory/config.yml.
agent: build
argument-hint: "[optional one-line project idea]"
---

Invoke the **factory-init** skill and follow it end to end.

If the user passed an idea, use it to pre-fill the discovery questionnaire: $ARGUMENTS

House skills (AGENTS.md § House skills, levels in `house_skills:` of `.factory/config.yml`): caveman, ponytail, rtk
and ccc apply to you and to every subagent you start; put that section at the top of each subagent prompt.
