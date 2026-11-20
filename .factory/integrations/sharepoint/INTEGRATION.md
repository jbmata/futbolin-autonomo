# Integration: SharePoint (Microsoft Graph, app-only)

> Reference integration. Connects to SharePoint Online document libraries to list / download / upload
> files using **application permissions** (no signed-in user), via Microsoft Graph.

- **Status:** proven (reference)
- **Last updated:** 2026-05-30
- **Used in:** (example)

## 1. Purpose / when to use
Read or write files in a SharePoint Online site's document library from a backend service
(unattended). Typical: ingest spreadsheets/documents, drop generated reports back.

## 2. Credentials & variables
Values go in the git-ignored `secrets.env`; names are in `variables.env.example`.

| Variable | Required | Where to get it |
|----------|----------|-----------------|
| `SHAREPOINT_TENANT_ID` | yes | Azure Portal → Entra ID → Overview → Tenant ID |
| `SHAREPOINT_CLIENT_ID` | yes | Entra ID → App registrations → your app → Application (client) ID |
| `SHAREPOINT_CLIENT_SECRET` | yes | App registration → Certificates & secrets → new client secret |
| `SHAREPOINT_SITE_URL` | yes | e.g. `https://<tenant>.sharepoint.com/sites/<site>` |
| `SHAREPOINT_DRIVE_ID` | no | specific document library drive id (else use the default drive) |

## 3. Auth method
**OAuth2 client-credentials (app-only)** against Entra ID (Azure AD):
1. Register an app in Entra ID. Add **Microsoft Graph → Application permissions** `Sites.Read.All`
   (or `Sites.ReadWrite.All` for writes). **Grant admin consent.**
2. Create a client secret.
3. Acquire a token from `https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token` with
   `scope=https://graph.microsoft.com/.default`.

> Tighter security: prefer `Sites.Selected` + per-site grants over tenant-wide `Sites.Read.All`.

## 4. Library / SDK
- **MSAL** (`msal`) for the token, plus `requests` to call Graph — minimal, pure-Python. **Preferred.**
- Or `msgraph-sdk` (official, heavier), or `Office365-REST-Python-Client` for the SharePoint REST API.

## 5. Implementation steps
```python
import os, msal, requests

def _token() -> str:
    app = msal.ConfidentialClientApplication(
        os.environ["SHAREPOINT_CLIENT_ID"],
        authority=f"https://login.microsoftonline.com/{os.environ['SHAREPOINT_TENANT_ID']}",
        client_credential=os.environ["SHAREPOINT_CLIENT_SECRET"],
    )
    r = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    if "access_token" not in r:
        raise RuntimeError(f"SharePoint auth failed: {r.get('error_description')}")
    return r["access_token"]

def _site_id(token: str) -> str:
    # from https://<tenant>.sharepoint.com/sites/<site>  ->  graph site id
    url = os.environ["SHAREPOINT_SITE_URL"].replace("https://", "")
    host, _, path = url.partition("/")
    resp = requests.get(
        f"https://graph.microsoft.com/v1.0/sites/{host}:/{path}",
        headers={"Authorization": f"Bearer {token}"}, timeout=30)
    resp.raise_for_status()
    return resp.json()["id"]

def download_file(item_path: str) -> bytes:
    token = _token(); site = _site_id(token)
    drive = os.environ.get("SHAREPOINT_DRIVE_ID")
    base = (f"https://graph.microsoft.com/v1.0/drives/{drive}" if drive
            else f"https://graph.microsoft.com/v1.0/sites/{site}/drive")
    r = requests.get(f"{base}/root:/{item_path}:/content",
                     headers={"Authorization": f"Bearer {token}"}, timeout=60)
    r.raise_for_status()
    return r.content
```
- List a folder: `GET {base}/root:/{folder}:/children`.
- Upload (small ≤4 MB): `PUT {base}/root:/{path}:/content`. Larger: createUploadSession (chunks).

## 6. Gotchas & lessons (grows over time)
- App-only tokens have **no `/me`** — always go via `/sites/{id}/drive`.
- Admin consent is required or every call 403s.
- Graph throttles → handle **429** with `Retry-After` backoff.
- Site URL → site id needs the `host:/sites/path` form shown above.
- Files > 4 MB **must** use an upload session; a plain PUT fails.

## 7. Testing
- Unit: mock `msal` + `requests` (no network).
- Integration: a dedicated **test site/library**; gate behind an env flag so CI can skip when secrets absent.

## 8. Changelog
- 2026-05-30 — initial reference playbook (MSAL + Graph, app-only): auth, site-id, download, list, upload notes.
