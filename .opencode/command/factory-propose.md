---
description: Prepare a DRAFT improvement to a factory skill, agent or command from this project — finds the installed file, makes the smallest change and leaves it in .factoria/proposals/. Never sends, pushes or opens a PR.
agent: build
argument-hint: "[optional: the skill, agent or command to improve, and why]"
---

Invoke the **factory-propose** skill and follow it end to end: ask what to improve (one question at a time), find the
installed file, make the smallest change and leave a draft in `.factoria/proposals/<date>-<slug>/`. Never send it,
never push, never open a pull request: a person reviews and sends it. $ARGUMENTS

House skills (AGENTS.md § House skills, levels in `house_skills:` of `.factory/config.yml`): caveman, ponytail, rtk
and ccc apply to you and to every subagent you start; put that section at the top of each subagent prompt.
