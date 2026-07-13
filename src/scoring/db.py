"""
db.py — Phase 5.

PostgreSQL persistence for scenario scores. One table is enough for this phase;
Phase 6 (FastAPI backend) will read from it, and PostGIS columns can be added
later without breaking this schema.

Design decisions (be ready to defend each):

  * scenario_id is the PRIMARY KEY and all writes are UPSERTS
    (INSERT ... ON CONFLICT DO UPDATE). Re-running a batch is therefore
    idempotent — you can re-score a shard after a code fix and rows update in
    place instead of duplicating. Idempotent writes are what make batch
    pipelines safely re-runnable.

  * fragility_score has a DESCENDING index because the pipeline's hottest
    query — "give me the top-N most fragile scenarios" — is an ORDER BY
    fragility_score DESC LIMIT N. The index turns that from a full-table sort
    into an index scan.

  * min_perturbation / delta / collision_timestep are NULLable: PASS 1 rows
    don't have them yet. scored_at vs stress_tested_at are separate timestamps
    because the two passes run at different times (possibly days apart).

  * delta is stored as DOUBLE PRECISION[] — Postgres native arrays keep the
    perturbation vector queryable without a join table for a fixed-dim vector.

Connection settings come from arguments or the standard PG* environment
variables (PGHOST, PGDATABASE, PGUSER, PGPASSWORD), so no credentials live in code.
"""

import os
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scenario_scores (
    scenario_id        TEXT PRIMARY KEY,
    shard              TEXT,
    n_agents           INTEGER,
    min_ttc            DOUBLE PRECISION,
    min_pet            DOUBLE PRECISION,
    fragility_score    DOUBLE PRECISION NOT NULL,
    min_perturbation   DOUBLE PRECISION,      -- NULL until Phase 4 pass runs
    delta              DOUBLE PRECISION[],    -- perturbation vector (Phase 4)
    collision_timestep INTEGER,               -- first colliding t (Phase 4)
    stress_method      TEXT,                  -- 'de' or 'de+autograd'
    scored_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    stress_tested_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_scenario_scores_fragility
    ON scenario_scores (fragility_score DESC);
"""


def get_connection(dbname=None, user=None, password=None, host=None, port=None):
    """
    Open a connection. Falls back to PG* environment variables, then defaults.
    """
    return psycopg2.connect(
        dbname=dbname or os.environ.get('PGDATABASE', 'av_stress'),
        user=user or os.environ.get('PGUSER', 'avi'),
        password=password or os.environ.get('PGPASSWORD', ''),
        host=host or os.environ.get('PGHOST', 'localhost'),
        port=port or os.environ.get('PGPORT', 5432),
    )


def init_schema(conn):
    """Create the table and index if they don't exist. Safe to call every run."""
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def upsert_scores(conn, records):
    """
    Bulk-write PASS 1 records. Idempotent: re-writing the same scenario_id
    updates the row (and refreshes scored_at) instead of duplicating it.
    Phase 4 columns are NOT touched here, so a re-score never wipes out an
    earlier stress-test result.
    """
    if not records:
        return 0
    rows = [(r['scenario_id'], r.get('shard'), r.get('n_agents'),
             r['min_ttc'], r['min_pet'], r['fragility_score'])
            for r in records]
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO scenario_scores
                (scenario_id, shard, n_agents, min_ttc, min_pet, fragility_score)
            VALUES %s
            ON CONFLICT (scenario_id) DO UPDATE SET
                shard           = EXCLUDED.shard,
                n_agents        = EXCLUDED.n_agents,
                min_ttc         = EXCLUDED.min_ttc,
                min_pet         = EXCLUDED.min_pet,
                fragility_score = EXCLUDED.fragility_score,
                scored_at       = now()
        """, rows)
    conn.commit()
    return len(rows)


def update_stress_results(conn, results):
    """
    Write PASS 2 (Phase 4) results onto existing rows.

    Args:
        results: dict scenario_id -> phase 4 result dict
                 (as returned by batch_scorer.stress_test_scenarios)
    Returns:
        number of rows updated.
    """
    n = 0
    with conn.cursor() as cur:
        for sid, r in results.items():
            if r.get('status') != 'ok':
                continue
            min_pert = r['min_perturbation']
            # store NULL (not +inf) for robustly-safe scenarios — SQL has no inf
            min_pert = None if min_pert == float('inf') else float(min_pert)
            delta = [float(x) for x in r['delta']] if r.get('delta') is not None else None
            t_hit = r.get('collision_timestep')
            t_hit = None if t_hit is None or t_hit < 0 else int(t_hit)
            cur.execute("""
                UPDATE scenario_scores SET
                    min_perturbation   = %s,
                    delta              = %s,
                    collision_timestep = %s,
                    stress_method      = %s,
                    stress_tested_at   = now()
                WHERE scenario_id = %s
            """, (min_pert, delta, t_hit, r.get('method'), sid))
            n += cur.rowcount
    conn.commit()
    return n


def fetch_top(conn, n=20):
    """
    The pipeline's hottest read: top-N most fragile scenarios.
    Served by the descending index — no full-table sort.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT * FROM scenario_scores
            ORDER BY fragility_score DESC, scenario_id
            LIMIT %s
        """, (n,))
        return [dict(r) for r in cur.fetchall()]


def fetch_scenario(conn, scenario_id):
    """Fetch one scenario's full row (or None)."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM scenario_scores WHERE scenario_id = %s",
                    (scenario_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def count_rows(conn):
    """Total rows, and how many have been stress-tested."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT count(*),
                   count(*) FILTER (WHERE stress_tested_at IS NOT NULL)
            FROM scenario_scores
        """)
        total, stressed = cur.fetchone()
        return {'total': total, 'stress_tested': stressed}