# Integration: <NAME>

> Copy this directory to `.factory/integrations/<name>/` and fill it in. Keep it concrete: the next
> developer (or agent) should be able to implement the integration from this file alone.

- **Status:** draft | proven
- **Last updated:** <date>
- **Used in:** <repos/phases that implemented it>

## 1. Purpose / when to use
What external system this connects to and what capability it provides (read files, send messages, query, …).

## 2. Credentials & variables
Document the env vars (names only — values go in the git-ignored `secrets.env`). See `variables.env.example`.

| Variable | Required | Where to get it |
|----------|----------|-----------------|
| `EXAMPLE_BASE_URL` | yes | … |
| `EXAMPLE_API_KEY`  | yes | … |

## 3. Auth method
How authentication works (API key, OAuth client-credentials, app registration, signed requests…),
including any consent/admin step.

## 4. Library / SDK
Preferred package(s) and why. Pin versions when known. Note pure-Python vs native deps.

## 5. Implementation steps
Concrete steps + a minimal working code pattern (token acquisition, the core call, error handling).

```python
# minimal pattern — adapt to the project's stack
```

## 6. Gotchas & lessons (grows over time)
- …

## 7. Testing
How to test without hitting prod (mocking, a sandbox/test tenant, fixtures).

## 8. Changelog
> Every implementation appends a line here with what changed/improved.
- <date> — initial draft.
