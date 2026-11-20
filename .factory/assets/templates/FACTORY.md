<!-- Managed by the Software Factory. Edit objectives/criteria freely; the factory reads this file. -->
# FACTORY · {{PROJECT_NAME}}

> Single source of *why* and *done*. The roadmap lives in `.planning/` (gsd); tasks mirror to GitHub Issues.

- **Created:** {{DATE}}
- **Mode:** {{MODE}}  <!-- checkpoint | autonomous | ultra-autonomous -->
- **Stack:** {{STACK}}
- **GitHub:** {{GH_REPO}}
- **Deploy target:** {{DEPLOY_TARGET}}

## Objective
{{OBJECTIVE}}

## Success criteria (the loop's stop condition)
<!-- Must be measurable/observable. The build loop runs until every box can be checked with evidence. -->
- [ ] {{CRITERION_1}}
- [ ] {{CRITERION_2}}
- [ ] {{CRITERION_3}}

## Users & problem
{{PROBLEM_STATEMENT}}

## Riskiest assumption
{{RISKIEST_ASSUMPTION}}
<!-- The first phase should be designed to test this cheaply. -->

## Non-goals (YAGNI)
- {{NON_GOAL_1}}

## Current state
{{CURRENT_STATE}}  <!-- greenfield + chosen starterkit, or summary of the existing codebase -->

---
*Workflow:* `/factory-init` → `/factory-discover` → `/factory-sync` → `/factory-build` → `/factory-run` → `/factory-deploy`. Dashboard: `/factory-status`.
