# Azure Pipelines variables and Key Vault secret names (go-service starterkit)

This file replaces `.env.cicd.example` for Azure Pipelines. There is no secret loader for Azure DevOps and there must
not be one: secrets are read from Key Vault at run time (`service-connection.md`). Below, only **non-secret**
variables hold values; secrets appear as **names** only.

An Azure Pipelines macro `$(NAME)` that has no variable stays as the literal text `$(NAME)`: define every variable
of the lists below (an empty value is fine for the optional ones).

## Pipeline variables (Pipelines, Edit, Variables; not secret)

| Variable | Used by | Example |
|---|---|---|
| `KEY_VAULT_NAME` | every `AzureKeyVault@2` step | `kv-factory-<child>` |
| `SONAR_PROJECT_KEY` | Sonar | `my-service` |
| `HARBOR_REGISTRY_URL` | Build, Deploy | `harbor.example.com` |
| `HARBOR_REGISTRY_USERNAME` | Build, Deploy | `robot$my-service` |
| `HARBOR_IMAGE_REPOSITORY` | Build, Deploy | `my-project/app` |

The SonarQube server URL and token are not variables: they belong to the SonarQube service connection.

## Variable group per environment (Library; the `variableGroup` parameter picks it; not secret)

Create `factoria-dev` and `factoria-prod` with the same names and each environment's values. Never mark a value as
secret and never link the group to Key Vault.

| Variable | Required | Example |
|---|---|---|
| `SSH_HOST` | yes | `203.0.113.10` |
| `SSH_PORT` | yes | `22` |
| `SSH_USER` | yes | `deploy` |
| `DEPLOY_PATH` | yes | `/srv/my-service` |
| `BIND_ADDR` | no (may be empty; default `0.0.0.0`) | `127.0.0.1` |
| `APP_PORT` | no (may be empty; default `8080`) | `8080` |

## Key Vault secrets (names only, in the child's vault)

The real name is `<secretPrefix><name>`: set the `secretPrefix` parameter (for example `myservice-prod-`) when the
vault serves several projects or environments. Create each secret in Key Vault yourself (portal, `az keyvault secret
set` with the value read from a file); never type a value in a chat, in the pipeline or in the repo.

| Name | Used by |
|---|---|
| `harbor-registry-password` | Build (`docker login`), Deploy (remote `docker login`) |
| `ssh-private-key` | Deploy |
| `ssh-known-hosts` | Deploy (output of `ssh-keyscan -p <port> <host>`) |

The app's own secrets are not in the compose file of this starterkit. If yours needs one, add it in three places:
create it in Key Vault, add its name to the `SecretsFilter` of the Deploy `AzureKeyVault@2` step, and map it in that
step's `env:` and in the `.env` the step writes.

## Test and coverage conversions (Quality stage)

`go test` writes neither JUnit nor Cobertura, and `PublishCodeCoverageResults@2` reads Cobertura only. The step
"Convert Go log and coverage to JUnit and Cobertura" runs, after the gate:

- `go run github.com/jstemmer/go-junit-report/v2@v2.1.0` over the gate log (the gate runs with `GOFLAGS=-v`), to
  `test-results/go.xml`;
- `go run github.com/boumenot/gocover-cobertura@v1.3.0` over `coverage.out`, to `coverage-cobertura.xml`;
- `lcov_cobertura` (Python package) over `web/coverage/lcov.info`.

They need access to the Go module proxy and PyPI. Without network access, mirror the modules or drop the step; the
gate verdict does not depend on it. Sonar reads `coverage.out` and `lcov.info` directly.
