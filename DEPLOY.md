# Deploying the site

The API runs as a Render web service, reading a Render Postgres database through a
read-only role. The frontend is a static Vite build on Vercel. Everything below that
creates a service, changes a hosted setting, or handles a credential is done in a
dashboard or a terminal by the account owner. **No credential belongs in this repository,
a commit, a chat message or a log.** Connection URLs are entered at a silent prompt
(`read -rs`), never typed into a command.

## The bill

Render's pricing page and Vercel's Hobby plan, read 2026-10-01:

| Item | Monthly |
|---|---|
| Render workspace, Hobby | $0 |
| API web service, `0.5c-512mb` (512 MB RAM; always on, no spin-down) | $7 |
| Postgres, `0.1c-256mb` (256 MB RAM, 100 connections) | $6 |
| Postgres storage, 1 GB at $0.30/GB (the page also lists "1 GB SSD storage included") | $0–0.30 |
| Render bandwidth: 5 GB/month included, then $0.15/GB | $0 at this size |
| Vercel Hobby: free, "non-commercial, personal use only", 100 GB transfer | $0 |
| **Total** | **about $13.00–13.30** |

## Steps, in order

Do them one at a time; each one's check comes before the next.

### 1. The database: read its expiry, then upgrade it in place

A Free Render Postgres database expires 30 days after creation and is deleted 14 days
after that. Read the date on the database's **Info** page first.

Upgrade it on its **Plan** page → **Compute** → **Edit** → `0.1c-256mb` → **Save**. The
database is unavailable for a few minutes, so do this before the API exists. Render's
free-tier docs describe this as the way to lift the free limits, the expiry included, and
the data stays in place. Check afterwards that the Info page no longer shows an expiry.

**Fallback, only if the in-place upgrade fails:** create a new `0.1c-256mb` Postgres,
version 17, in the same region (Oregon), and restore the dump from the Mac with
Homebrew's `postgresql@17` client:

```
read -rs AV_STRESS_DATABASE_URL && export AV_STRESS_DATABASE_URL
psql "$AV_STRESS_DATABASE_URL" -c "CREATE EXTENSION IF NOT EXISTS postgis"
/opt/homebrew/opt/postgresql@17/bin/pg_restore --no-owner --no-privileges \
    -d "$AV_STRESS_DATABASE_URL" av_stress_amax12.dump
```

The dump was made without privileges, so step 2 must run after a restore as well.

### 2. A read-only role for the API

The API only reads. It connects as a role that can only `SELECT`, so neither a bug nor a
leaked API credential can change the dataset. Create it in SQL. **Not** with the
dashboard's "+ New default credential", which creates a new *default* user with full
rights. Render documents that users created with `CREATE USER` are not managed by Render
and do not replace the default user.

From the Mac, connected as the database's default user through its **external** URL:

```
read -rs AV_STRESS_DATABASE_URL && export AV_STRESS_DATABASE_URL
psql "$AV_STRESS_DATABASE_URL"
```

```sql
SELECT rolcreaterole FROM pg_roles WHERE rolname = current_user;   -- must be t
CREATE ROLE av_api_ro LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
\password av_api_ro
GRANT CONNECT ON DATABASE <the database name> TO av_api_ro;
GRANT USAGE ON SCHEMA public TO av_api_ro;
GRANT SELECT ON scenario_scores, scenario_agents, perturbed_paths TO av_api_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO av_api_ro;
ALTER ROLE av_api_ro SET default_transaction_read_only = on;
```

- `\password` prompts without echoing and sends only a hash: the password never appears
  in the SQL, the shell history or the server log. Keep it in a password manager and
  nowhere else.
- The three tables are exactly the ones `src/api` reads
  (`tests/test_deploy_contract.py` fails if the API starts reading another one and this
  grant is not updated).
- `ALTER DEFAULT PRIVILEGES` covers tables created **later**, but only tables created by
  the role that runs it. Run it as the default user, which is the owner the notebook
  writes as, so a table a later pass creates is readable by the API without a new grant.
- The `GRANT` is the guarantee. `default_transaction_read_only` is a second layer, which a
  session could switch off.
- The geometry endpoints call PostGIS functions and read `spatial_ref_sys`, both of which
  PostGIS makes available to every role. `SELECT` on the three tables should be enough;
  check 4 below is the proof. **A 500 on `/scenarios/<id>/trajectories` after deploying
  most likely means a missing grant.**

Check it, connected as `av_api_ro` (same host and database, user `av_api_ro`,
`sslmode=require`; psql prompts for the password):

```sql
SELECT count(*) FROM scenario_scores;                              -- 100
UPDATE scenario_scores SET n_agents = n_agents WHERE false;        -- must fail
CREATE TABLE t (i int);                                            -- must fail
```

### 3. The API: a Render web service

**+ New** → **Web Service** → this GitHub repository, branch `main`:

| Setting | Value |
|---|---|
| Region | **Oregon**, the database's region (the internal URL works only within a region) |
| Runtime | **Python 3** (not Docker: the untracked `Dockerfile` is unrelated and unused) |
| Root directory | blank (the repository root) |
| Build command | `pip install -r requirements-api.txt` |
| Start command | `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT` |
| Instance type | `0.5c-512mb`, $7/month |
| Health check path | `/health` (Advanced) |

Render reads `.python-version` (3.11.15, the version the tests run on). Without it,
Render's default would be Python 3.14.3.

**Environment variables.** The **internal** connection details, which the database's Info
page lists one by one. `src/api/config.py` reads separate libpq variables, not a URL.

| Variable | Value |
|---|---|
| `PGHOST` | the database's **internal** hostname |
| `PGPORT` | `5432` |
| `PGDATABASE` | the database name |
| `PGUSER` | `av_api_ro` |
| `PGPASSWORD` | the read-only role's password, entered as a secret |
| `PGSSLMODE` | `require` (internal connections only *optionally* use TLS; this forbids plaintext) |
| `API_CORS_ORIGINS` | `http://localhost:5173` for now; step 5 replaces it |

The pool defaults (`API_POOL_MIN=1`, `API_POOL_MAX=8`) stay: the database allows 100
connections, and one instance uses at most 8 (16 while a deploy overlaps the old one).

Render redeploys on every push to `main`. A build filter limited to `src/api/**` and
`requirements-api.txt` is optional.

Check: `https://<the service>.onrender.com/health` returns
`{"status":"ok","database":"connected","postgis":"3.6 …"}`. If `/health` fails, Render
keeps routing to the previous version and cancels the deploy after 15 minutes.

### 4. The frontend: a Vercel project

**Add New** → **Project** → this GitHub repository:

| Setting | Value |
|---|---|
| Root directory | `frontend` |
| Framework preset | Vite (detected) |
| Build command | the default, `npm run build` (its `prebuild` generates the types from the committed `openapi.json`) |
| Output directory | `dist` |
| Environment variable | `VITE_API_BASE_URL` = `https://<the service>.onrender.com`, for Production |

The build **refuses** to run without `VITE_API_BASE_URL`, or with an `http://` URL to
anything but localhost (`frontend/src/build/requireApiBaseUrl.ts`). The variable is baked
in at build time, so changing it needs a redeploy.

`frontend/vercel.json` rewrites every path to `/index.html`, so a direct visit or refresh
on `/scenarios/<id>` or `/stats` loads (Vercel's Vite guide: deep links do not work out
of the box). Vercel serves real files first, so `/assets/*` is unaffected.

### 5. CORS: let the site, and only the site, read the API

On Render, set `API_CORS_ORIGINS` to the Vercel production origin, exactly as Vercel shows
it, with `https://` and no trailing slash (for example `https://<project>.vercel.app`).
Saving redeploys the API. A custom domain added later is appended, comma-separated.

**Never `*`, and never a `*.vercel.app` pattern:** every Vercel project lives under that
domain, so a pattern is `*` in practice. Preview deployments are not allowed. Their
per-commit URLs contain a random hash, and this repository commits straight to `main`, so
every push is a production deploy anyway. If a preview is ever wanted, its branch URL
(`<project>-git-<branch>-<scope>.vercel.app`) is stable and can be added explicitly.

## Checks, in order

1. `GET /health`: `status: ok`, `database: connected`, `postgis` set.
2. `GET /stats`: `total_scenarios: 100`, `stress_tested: 18`, `collisions_found: 18`,
   `replay_infeasible: 2`, `no_collision_found: 0`, `with_geometry: 20`.
   (`stress_tested` counts stored search results; the 2 drift refusals are
   `replay_infeasible`. `with_geometry` is the run's Pass 3 `exported: 20`.)
3. The list pages through all 100 scenarios.
4. A detail page draws its scene and plays back, for example `8ec2910b`, one of the six the
   2 m backstop admitted. A 500 here most likely means a missing grant (step 2).
5. A direct link to a detail page loads, and so does a refresh on `/stats`.
6. The browser console shows no CORS errors.

If the API is unreachable, the site shows "Loading…" through two retries, then "Could not
reach the API at <URL>". That is accurate: a Render error page carries no CORS headers, so
the browser reports a network failure.
