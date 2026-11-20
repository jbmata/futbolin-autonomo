---
description: Code-quality gate — run tests+coverage, lint, ponytail minimalism, and SonarQube analysis, then emit a PASS/FAIL verdict with the top fixes.
agent: build
argument-hint: "[optional: 'sonar' to force a local scan, or a path/component to scope]"
---

Invoke the **factory-quality** skill to validate code quality (tests + coverage, lint/format, minimalism,
SonarQube) and report a quality-gate verdict. Scope to the argument if given: $ARGUMENTS

House skills (AGENTS.md § House skills, levels in `house_skills:` of `.factory/config.yml`): caveman, ponytail, rtk
and ccc apply to you and to every subagent you start; put that section at the top of each subagent prompt.
