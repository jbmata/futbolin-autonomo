# Azure Pipelines variables and Key Vault secret names (OpenSaaS / Wasp starterkit)

This file replaces `.env.cicd.example` for Azure Pipelines. There is no secret loader for Azure DevOps and there must
not be one: secrets are read from Key Vault at run time (`service-connection.md`). Below, only **non-secret**
variables hold values; secrets appear as **names** only.

An Azure Pipelines macro `$(NAME)` that has no variable stays as the literal text `$(NAME)`: define every variable
of the lists below (an empty value is fine for the optional ones).

## Pipeline variables (Pipelines, Edit, Variables; not secret)

| Variable | Used by | Example |
|---|---|---|
| `KEY_VAULT_NAME` | every `AzureKeyVault@2` step | `kv-factory-<child>` |
| `SONAR_PROJECT_KEY` | Sonar | `my-saas` |
| `HARBOR_REGISTRY_URL` | Build, Deploy | `harbor.example.com` |
| `HARBOR_REGISTRY_USERNAME` | Build, Deploy | `robot$my-saas` |
| `HARBOR_BACKEND_IMAGE_REPOSITORY` | Build, Deploy | `my-saas/server` |
| `HARBOR_FRONTEND_IMAGE_REPOSITORY` | Build, Deploy | `my-saas/client` |
| `REACT_APP_API_URL` | client image build, Deploy | `https://api.example.com` |

The SonarQube server URL and token are not variables: they belong to the SonarQube service connection.

## Variable group per environment (Library; the `variableGroup` parameter picks it; not secret)

Create `factoria-dev` and `factoria-prod` with the same names and each environment's values. Never mark a value as
secret and never link the group to Key Vault.

| Variable | Required | Example |
|---|---|---|
| `SSH_HOST` | yes | `203.0.113.10` |
| `SSH_PORT` | yes | `22` |
| `SSH_USER` | yes | `deploy` |
| `DEPLOY_PATH` | yes | `/srv/my-saas` |
| `WASP_SERVER_URL`, `WASP_WEB_CLIENT_URL` | yes | `https://api.example.com`, `https://app.example.com` |
| `ADMIN_EMAILS`, `PROJECT_NAME` | no (may be empty) | `admin@example.com` |
| `POSTGRES_DB`, `POSTGRES_USER` | yes | `app` |
| `POSTGRES_HOST`, `POSTGRES_PORT` | no (may be empty; defaults `db`, `5432`) | `db`, `5432` |

## Key Vault secrets (names only, in the child's vault)

The real name is `<secretPrefix><name>`: set the `secretPrefix` parameter (for example `mysaas-prod-`) when the
vault serves several projects or environments. Create each secret in Key Vault yourself (portal, `az keyvault secret
set` with the value read from a file); never type a value in a chat, in the pipeline or in the repo.

| Name | Used by |
|---|---|
| `harbor-registry-password` | Build (`docker login`), Deploy (remote `docker login`) |
| `ssh-private-key` | Deploy |
| `ssh-known-hosts` | Deploy (output of `ssh-keyscan -p <port> <host>`) |
| `jwt-secret` | Deploy (`JWT_SECRET`) |
| `postgres-password` | Deploy (`POSTGRES_PASSWORD`, and inside the derived `DATABASE_URL`) |
| `smtp-password` | Deploy (`SMTP_PASSWORD`; create it with a placeholder value if the app sends no mail) |

`DATABASE_URL` is derived from `postgres-password` inside the deploy step and marked `issecret=true`. Payments, AI or
storage keys of your app follow the same rule: create the secret in Key Vault, add its name to the `SecretsFilter` of
the Deploy `AzureKeyVault@2` step, and map it in that step's `env:` and in the env file the step writes.

## Tests and coverage

The gate runs `npm run test --if-present -- --coverage` in `app`. `PublishCodeCoverageResults@2` reads Cobertura, so
the Quality stage converts `app/coverage/lcov.info` with the `lcov_cobertura` Python package (needs PyPI access).
Sonar reads `lcov.info` directly. For test results, configure the JUnit reporter of the project's test runner.
