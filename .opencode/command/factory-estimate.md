---
description: Estimate the cost (tokens), agent time and all the human work (functional, approver, consultant, admin) of a spec, feature or phase in the assisted and autonomous scenarios, price it with the Hub tariff and record it with the local MCP.
agent: build
argument-hint: "[optional: spec slug, feature or phase key; empty = the whole project]"
---

Invoke the **factory-estimate** skill and follow it end to end. An estimate never blocks the work and never invents a price.

If the user specified a spec, feature or phase, honor it: $ARGUMENTS

House skills (AGENTS.md § House skills, levels in `house_skills:` of `.factory/config.yml`): caveman, ponytail, rtk
and ccc apply to you and to every subagent you start; put that section at the top of each subagent prompt.

With `tracker.platform: hub` the estimate reaches the Hub only through the local MCP `factory` (`tariff_get`,
`estimate_list`, `estimate_submit`). `factory-spec` asks for the estimate of a spec before it is ready, and
`factory-build` re-estimates a phase when its real size differs a lot from the estimate.
