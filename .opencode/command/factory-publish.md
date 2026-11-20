---
description: Safely land changes — branch, commit, sync main, run local pre-publish checks, push, and open (or reuse) a GitHub pull request via gh. Never touches main directly, never force-pushes, never merges.
agent: build
argument-hint: "[optional: short change description, or 'commit-only' to skip the push/PR]"
---

Invoke the **factory-publish** skill and follow it end to end: check prerequisites, branch (or
stay on the current branch in commit-only mode), commit, sync `main` with a verified no-content-lost
merge, run the local pre-publish checks, push, and open or reuse the pull request. Never commit
directly to `main`, never force-push, never merge the PR yourself. $ARGUMENTS

House skills (AGENTS.md § House skills, levels in `house_skills:` of `.factory/config.yml`): caveman, ponytail, rtk
and ccc apply to you and to every subagent you start; put that section at the top of each subagent prompt.
