---
name: factory-sync
description: "Mirror the gsd roadmap/phases into GitHub Issues (idempotent) and read issue state back."
tools: [read, write, edit, search, execute]
---

Load the `factory-sync` skill from the installed `software-factory` plugin (marketplace `fastex-factory`; if it is
missing, run `bash .factory/assets/scripts/setup.sh .`) and follow its SKILL.md end to end. The House skills section of `AGENTS.md`
(caveman, ponytail, rtk, ccc at the levels in `house_skills:` of `.factory/config.yml`) applies to you and to
every subagent you start: put it at the top of each subagent prompt.

If the user passed arguments, honor them: $ARGUMENTS
