"""
config.py — Phase 6.

All configuration in one place, read from the environment exactly once at import.

Why environment variables rather than a config file: the database credentials must
never be committed, and the PG* names are the standard libpq variables that Phase 5
already uses (src/scoring/db.py). Speaking the same dialect means pointing this API
at a hosted Postgres later is purely an environment change — no code edit, no new
config format to learn.

Read once, at import, rather than per request: these values cannot change while the
process runs, and re-reading os.environ on every request would be a syscall's worth
of pointless work in the hot path.
"""

import os


class Settings:
    """Environment-driven settings. Instantiated once as the module-level `settings`."""

    def __init__(self):
        # ── database (standard libpq names, same defaults as Phase 5's db.py) ──
        self.pg_database = os.environ.get('PGDATABASE', 'av_stress')
        self.pg_user = os.environ.get('PGUSER', 'avi')
        self.pg_password = os.environ.get('PGPASSWORD', '')
        self.pg_host = os.environ.get('PGHOST', 'localhost')
        self.pg_port = int(os.environ.get('PGPORT', 5432))

        # ── connection pool sizing ────────────────────────────────────────────
        # The ceiling is deliberately small. Free-tier hosted Postgres caps total
        # connections aggressively (often ~20 across ALL clients), and a pool that
        # cheerfully opens 50 connections will exhaust the server's limit and lock
        # out everything else — including the psql session you would use to debug
        # it. Eight is plenty for an API whose queries take single-digit ms.
        self.pool_min = int(os.environ.get('API_POOL_MIN', 1))
        self.pool_max = int(os.environ.get('API_POOL_MAX', 8))

        # How long a request may wait for a free connection before the server admits
        # it is overloaded (audit B18). ThreadedConnectionPool.getconn does not block —
        # it raises the moment the pool is at its ceiling — so without a wait a brief
        # burst becomes a 503 for a request that would have been served milliseconds
        # later. The queries here are single-digit milliseconds, so 250 ms absorbs
        # roughly thirty query-durations of queueing; past that the server really is
        # overloaded and 503 is the honest answer. Bounded, so a worker thread can
        # never park indefinitely waiting for a connection that is not coming.
        self.pool_wait_ms = int(os.environ.get('API_POOL_WAIT_MS', 250))

        # Server-side ceiling on any single statement (audit B18). Without it, one
        # pathological query holds a pooled connection for as long as it runs, and with
        # a ceiling of 8 it takes only eight of those to starve the API — the same
        # failure the putconn-in-finally rule exists to prevent, arrived at from the
        # other direction.
        #
        # Applied at CONNECT time via libpq's `options`, so every pooled connection
        # carries it with no per-request round trip. 5 s is ~1000x the normal query
        # cost, so it can only fire on something genuinely wrong.
        #
        # Deliberately NOT applied to src/scoring/db.py's connections: Pass 1/2/3 run
        # long batch work and must not be interrupted by a limit sized for a dashboard.
        self.statement_timeout_ms = int(os.environ.get('API_STATEMENT_TIMEOUT_MS', 5000))

        # ── CORS ──────────────────────────────────────────────────────────────
        # Explicit origins, comma-separated. Defaults cover the usual Vite (5173)
        # and CRA/Next (3000) dev servers.
        #
        # This must NEVER become "*". A GET-only API is not automatically safe to
        # expose to every origin: "*" means any page on the internet can read this
        # data through a visitor's browser. The read-only-ness limits what an
        # attacker can *change*, not what they can *see*.
        raw_origins = os.environ.get(
            'API_CORS_ORIGINS',
            'http://localhost:5173,http://localhost:3000',
        )
        self.cors_origins = [o.strip() for o in raw_origins.split(',') if o.strip()]

        # ── paging ────────────────────────────────────────────────────────────
        # A hard ceiling matters: without it, `?limit=1000000` is a trivial way for
        # a client to make the server materialize the entire table into JSON.
        self.page_size = int(os.environ.get('API_PAGE_SIZE', 25))
        self.max_page_size = int(os.environ.get('API_MAX_PAGE_SIZE', 200))

    def dsn_kwargs(self) -> dict:
        """psycopg2 connection kwargs. One place that knows the mapping."""
        return {
            'dbname': self.pg_database,
            'user': self.pg_user,
            'password': self.pg_password,
            'host': self.pg_host,
            'port': self.pg_port,
            # libpq passes this through to the backend at startup, so the timeout is
            # in force for every statement on every pooled connection.
            'options': f'-c statement_timeout={self.statement_timeout_ms}',
        }


settings = Settings()
