# Azure Pipelines variables and Key Vault secret names (FastAPI starterkit)

This file replaces `.env.cicd.example` for Azure Pipelines. There is no secret loader for Azure DevOps and there must
not be one: secrets are read from Key Vault at run time (`service-connection.md`). Below, only **non-secret**
variables hold values; secrets appear as **names** only.

An Azure Pipelines macro `$(NAME)` that has no variable stays as the literal text `$(NAME)`: define every variable
of the lists below (an empty value is fine for the optional ones).

## Pipeline variables (Pipelines, Edit, Variables; not secret)

| Variable | Used by | Example |
|---|---|---|
| `KEY_VAULT_NAME` | every `AzureKeyVault@2` step | `kv-factory-<child>` |
| `SONAR_PROJECT_KEY` | Sonar | `my-project` |
| `HARBOR_REGISTRY_URL` | Build, Deploy | `harbor.example.com` |
| `HARBOR_REGISTRY_USERNAME` | Build, Deploy | `robot$my-project` |
| `HARBOR_BACKEND_IMAGE_REPOSITORY` | Build, Deploy | `my-project/backend` |
| `HARBOR_FRONTEND_IMAGE_REPOSITORY` | Build, Deploy | `my-project/frontend` |
| `VITE_API_URL` | frontend image build | `https://api.example.com` |

The SonarQube server URL and token are not variables: they belong to the SonarQube service connection.

## Variable group per environment (Library; the `variableGroup` parameter picks it; not secret)

Create `factoria-dev` and `factoria-prod` with the same names and each environment's values. Never mark a value as
secret and never link the group to Key Vault.

| Variable | Required | Example |
|---|---|---|
| `SSH_HOST` | yes | `203.0.113.10` |
| `SSH_PORT` | yes | `22` |
| `SSH_USER` | yes | `deploy` |
| `DEPLOY_PATH` | yes | `/srv/my-project` |
| `ENVIRONMENT` | yes | `production` |
| `PUBLIC_HOST`, `DOMAIN`, `STACK_NAME`, `PROJECT_NAME` | no (may be empty) | `app.example.com` |
| `BACKEND_CORS_ORIGINS` | no | `https://app.example.com` |
| `FIRST_SUPERUSER` | yes | `admin@example.com` |
| `SMTP_HOST`, `SMTP_USER`, `SMTP_PORT`, `SMTP_TLS`, `SMTP_SSL`, `EMAILS_FROM_EMAIL` | no | `smtp.example.com` |
| `POSTGRES_DB`, `POSTGRES_USER` | yes | `app` |
| `POSTGRES_PORT`, `FRONTEND_PORT`, `BACKEND_PORT`, `ADMINER_PORT` | no | `5432`, `80`, `8000`, `8080` |

## Key Vault secrets (names only, in the child's vault)

The real name is `<secretPrefix><name>`: set the `secretPrefix` parameter (for example `myproject-prod-`) when the
vault serves several projects or environments. Create each secret in Key Vault yourself (portal, `az keyvault secret
set` with the value read from a file); never type a value in a chat, in the pipeline or in the repo.

| Name | Used by |
|---|---|
| `harbor-registry-password` | Build (`docker login`), Deploy (remote `docker login`) |
| `ssh-private-key` | Deploy |
| `ssh-known-hosts` | Deploy (output of `ssh-keyscan -p <port> <host>`) |
| `secret-key` | Deploy (`SECRET_KEY`) |
| `first-superuser-password` | Deploy (`FIRST_SUPERUSER_PASSWORD`) |
| `postgres-password` | Deploy (`POSTGRES_PASSWORD`) |
| `smtp-password` | Deploy (`SMTP_PASSWORD`; create it with a placeholder value if the app sends no mail) |

An optional secret that the app needs (for example `SENTRY_DSN`) is added in three places: create it in Key Vault,
add its name to the `SecretsFilter` of the Deploy `AzureKeyVault@2` step, and map it in that step's `env:`.
