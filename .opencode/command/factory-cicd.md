---
description: Scaffold CI/CD — SonarQube + build→Harbor→deploy GitHub Actions for the starterkit, generate the .env.cicd to fill, and load credentials into GitHub Variables/Secrets via gh. Your registry, your servers.
agent: build
argument-hint: "[optional: target repo owner/name, or an environment dev|prod]"
---

Invoke the **factory-cicd** skill to set up the quality + build + deploy pipeline (Sonar + Harbor + SSH
deploy) for this project's starterkit, ask the user to fill `.env.cicd`, and load the credentials into
the repo's GitHub Variables/Secrets. Confirm before any outward-facing action: $ARGUMENTS

House skills (AGENTS.md § House skills, levels in `house_skills:` of `.factory/config.yml`): caveman, ponytail, rtk
and ccc apply to you and to every subagent you start; put that section at the top of each subagent prompt.
