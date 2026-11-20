# Key Vault access for Azure Pipelines: service connection with workload identity federation

Secrets live only in the child's Azure Key Vault. The pipeline reads them at run time with `AzureKeyVault@2`
through an Azure Resource Manager service connection that uses **workload identity federation**: no client secret,
no certificate and no password is stored in Azure DevOps, in the repo or in a variable group. Azure DevOps asks
Entra for a short-lived token with its own signed identity token, and Entra trusts it because of a federated
credential you create once.

Prerequisite: the Azure DevOps organization must be backed by Microsoft Entra, and the Entra tenant must be the one
that holds the Key Vault (or a tenant where the app can be consented). Not verified against a real organization:
the Fastex test organization is not Entra-backed (`docs/spikes/03-identidades-y-licencias.md` of the Factoría v3 repository).

## 1. Entra application

Create an app registration in the tenant of the Key Vault, for example `factoria-<child>-ci`. Note its
**Application (client) ID** and **Directory (tenant) ID**. Do not create a client secret or certificate for it.

## 2. Service connection in Azure DevOps

Project settings, Service connections, New service connection, **Azure Resource Manager**, then
**Workload identity federation (manual)** (identity type: app registration, manual).

- Name: the value of the `keyVaultServiceConnection` parameter of the pipeline (default `factoria-keyvault`).
- Subscription ID, subscription name, tenant ID and client ID: those of the vault's subscription and of step 1.
- Azure DevOps then shows the two values Entra needs:
  - **Issuer**: `https://vstoken.dev.azure.com/<organization-id>` (copy the value shown; older organizations show
    `https://app.vstoken.visualstudio.com`).
  - **Subject identifier**: `sc://<organization>/<project>/<service-connection-name>`.
- Leave **Grant access permission to all pipelines** off. In the connection's Security, authorize only this
  pipeline, and add an approval check on the connection if production secrets sit behind it.

## 3. Federated credential in Entra

App registration, Certificates and secrets, Federated credentials, Add credential, scenario **Other issuer**:

| Field | Value |
|---|---|
| Issuer | the Issuer from step 2, exactly |
| Subject identifier | the Subject identifier from step 2, exactly |
| Audience | `api://AzureADTokenExchange` |
| Name | `azure-devops-<project>` |

Then save the service connection. Save without verifying if Azure DevOps wants a subscription-level role to verify:
do not widen the permissions of step 4 for that; the first pipeline run proves the connection.

## 4. Exact permission on the vault

Give the app's service principal the role **Key Vault Secrets User** on the child's vault only. The vault must use
the Azure RBAC permission model (no access policies).

```bash
az role assignment create \
  --assignee <client-id> \
  --role "Key Vault Secrets User" \
  --scope /subscriptions/<subscription-id>/resourceGroups/<group>/providers/Microsoft.KeyVault/vaults/<vault-name>
```

No other role, no role on the subscription or the resource group, no write role on the vault. The role reads secret
values; it cannot create, change or delete them. The same Entra group "Factoria Operadores <child>" that holds the
operators keeps its own role; do not add the pipeline to it.

## 5. What the pipeline reads

`AzureKeyVault@2` takes `KeyVaultName` (pipeline variable `KEY_VAULT_NAME`, not secret) and a `SecretsFilter` with the
secret **names** listed in `pipeline-variables.example.md`. Key Vault secret names allow only letters, digits and
dashes. A name missing from the vault fails the step. Each value becomes a masked pipeline variable named like the
secret and is mapped to an environment variable only in the step that needs it.

## 6. Other things you create by hand in Azure DevOps

- The pipeline: Pipelines, New pipeline, Azure Repos Git, Existing Azure Pipelines YAML file, `/azure-pipelines.yml`.
- Environments `dev` and `prod` (Pipelines, Environments), each with **Approvals and checks** (an approver for `prod`).
- A variable group per environment, for example `factoria-prod`, with the non-secret variables only. Never mark
  a value as secret there and never link the group to the vault: a secret in a variable group is a copy.
- The pipeline variables of `pipeline-variables.example.md` (non-secret).
- A **SonarQube service connection** (SonarQube extension from the Marketplace) named like the `sonarServiceConnection`
  parameter (default `sonarqube`). It holds the Sonar token inside Azure DevOps' service connection store: create the
  token in SonarQube with the minimum scope, and rotate it there. It is the only credential outside Key Vault.
- Pull requests only run Quality, which reads no secret, so a pull request from a branch never reaches Key Vault.

## Checks

- Federation: a run of the `Build` stage reads the secret and `docker login` works without any stored password.
- Least privilege: `az role assignment list --assignee <client-id> --all` shows one assignment, Key Vault Secrets
  User on the vault.
- No copy: Azure DevOps Library has no variable group with secret values and no `.env.cicd` exists for this project.
