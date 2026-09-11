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

import hashlib
import os

import psycopg2
from psycopg2.extras import execute_values, Json, RealDictCursor


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

-- Phase 3 rework: min_ttc / min_pet now hold SDC-restricted values and
-- fragility_score is computed from them. These carry the old all-pairs
-- "scene density" numbers as a diagnostic. Additive so a database already
-- populated by an earlier run (e.g. the Colab validation shard) upgrades
-- in place — the CREATE TABLE above has already ensured the table exists.
ALTER TABLE scenario_scores ADD COLUMN IF NOT EXISTS min_ttc_all_pairs DOUBLE PRECISION;
ALTER TABLE scenario_scores ADD COLUMN IF NOT EXISTS min_pet_all_pairs DOUBLE PRECISION;

-- Batch 2 (audit B04, B14): say what the stress pass actually established.
--
-- Additive, like the Phase 3 columns above, so a database populated by an earlier
-- run upgrades in place. Nothing is dropped or retyped; existing rows get NULL,
-- which means "unknown outcome" and must NOT be read as any particular one — an old
-- row with stress_tested_at set and min_perturbation NULL could have been a
-- completed search OR a scenario that would now be refused as replay_infeasible.
ALTER TABLE scenario_scores ADD COLUMN IF NOT EXISTS stress_outcome TEXT;
ALTER TABLE scenario_scores ADD COLUMN IF NOT EXISTS stress_attempted_at TIMESTAMPTZ;
ALTER TABLE scenario_scores ADD COLUMN IF NOT EXISTS challengers_total INTEGER;
ALTER TABLE scenario_scores ADD COLUMN IF NOT EXISTS challengers_searched INTEGER;
ALTER TABLE scenario_scores ADD COLUMN IF NOT EXISTS search_provenance JSONB;
ALTER TABLE scenario_scores ADD COLUMN IF NOT EXISTS stress_run_id TEXT;

-- What the MOST RECENT pass concluded, as opposed to what produced the stored
-- result. These are different runs whenever a pass fails to reproduce an earlier
-- success, and one column cannot honestly hold both: a row reading
-- stress_outcome='replay_infeasible' beside a real min_perturbation asserts
-- something false by juxtaposition, which is the robustly_safe defect (B04) wearing
-- different field names. stress_outcome travels WITH the result it describes;
-- last_attempt_outcome travels with stress_attempted_at.
ALTER TABLE scenario_scores ADD COLUMN IF NOT EXISTS last_attempt_outcome TEXT;

-- The safety certificate, kept deliberately as a column that nothing sets.
--
-- robustly_safe is derived from this. No code path writes TRUE, because no method
-- in this pipeline establishes it: Differential Evolution is a stochastic global
-- optimizer whose convergence test is population spread, not a proof that no
-- feasible point exists, and _stress_one searches ONE heuristically-chosen
-- challenger out of N. Storing the falseness rather than hardcoding it keeps the
-- claim visible in the data, and lets a method that genuinely certifies
-- infeasibility flip a value instead of editing the API.
ALTER TABLE scenario_scores
    ADD COLUMN IF NOT EXISTS search_certifies_infeasibility BOOLEAN NOT NULL DEFAULT FALSE;
"""


# The closed set of things a Phase 4 pass can conclude. Kept here, next to the
# column, so the vocabulary has exactly one definition.
#
# The distinction that matters, and the reason this column exists at all:
# NO_COLLISION_FOUND means a search ran to completion and found nothing within its
# budget and bounds. REPLAY_INFEASIBLE means no search ran at all, because the
# scenario's own zero-perturbation replay was not faithful enough to measure a
# perturbation against (Batch 1, audit B03). Collapsing the second into the first
# would report a scenario that was never testable as one that came back clean.
OUTCOME_COLLISION_FOUND    = 'collision_found'
OUTCOME_NO_COLLISION_FOUND = 'no_collision_found'
OUTCOME_REPLAY_INFEASIBLE  = 'replay_infeasible'
OUTCOME_NO_CHALLENGER      = 'no_challenger'
OUTCOME_ERROR              = 'error'

# Outcomes for which a search actually ran, and therefore the only ones that set
# stress_tested_at. Everything else sets stress_attempted_at only.
OUTCOMES_SEARCH_RAN = frozenset({OUTCOME_COLLISION_FOUND, OUTCOME_NO_COLLISION_FOUND})


def compute_stress_run_id(scenario_id, target_idx, delta, method) -> str:
    """
    A short, deterministic identifier for one stress-test run (audit B14).

    Stamped on the scenario_scores row and on the perturbed_paths row exported from
    it, so a result and the geometry drawn beside it can be proven to describe the
    same run rather than merely assumed to.

    Deliberately a CONTENT HASH and not a UUID. The project forbids behaviour that
    depends on wall-clock or randomness, and a content hash additionally gives
    idempotence for free: re-running the identical delta produces the identical id,
    so a re-export is a no-op, while any change to the delta produces a different id
    and is therefore detectable.

    delta components are formatted with repr(float(...)), which round-trips a
    binary64 exactly — the same rule _linestring_m_wkt uses for coordinates — so the
    id cannot drift with float formatting or numpy scalar repr.

    Uses hashlib, NOT the builtin hash(), which is salted per process by
    PYTHONHASHSEED and would produce a different id on every run.
    """
    components = '|'.join(repr(float(x)) for x in (delta if delta is not None else ()))
    canonical = f'{scenario_id}|{int(target_idx)}|{components}|{method or ""}'
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]


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
             r['min_ttc'], r['min_pet'], r['fragility_score'],
             r.get('min_ttc_all_pairs'), r.get('min_pet_all_pairs'))
            for r in records]
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO scenario_scores
                (scenario_id, shard, n_agents, min_ttc, min_pet, fragility_score,
                 min_ttc_all_pairs, min_pet_all_pairs)
            VALUES %s
            ON CONFLICT (scenario_id) DO UPDATE SET
                shard             = EXCLUDED.shard,
                n_agents          = EXCLUDED.n_agents,
                min_ttc           = EXCLUDED.min_ttc,
                min_pet           = EXCLUDED.min_pet,
                fragility_score   = EXCLUDED.fragility_score,
                min_ttc_all_pairs = EXCLUDED.min_ttc_all_pairs,
                min_pet_all_pairs = EXCLUDED.min_pet_all_pairs,
                scored_at         = now()
        """, rows)
    conn.commit()
    return len(rows)


def resolve_outcome(result) -> str:
    """
    Map one batch_scorer result dict onto the outcome vocabulary.

    Preferred source is result['outcome'], which _stress_one sets directly. The
    derivation below is the fallback for a result dict built elsewhere (the audit
    fixtures, and any caller predating this column), so the vocabulary still has one
    definition rather than two that can drift.
    """
    outcome = result.get('outcome')
    if outcome:
        return outcome
    status = result.get('status')
    if status == 'ok':
        return (OUTCOME_COLLISION_FOUND if result.get('collision')
                else OUTCOME_NO_COLLISION_FOUND)
    if status == OUTCOME_REPLAY_INFEASIBLE:
        return OUTCOME_REPLAY_INFEASIBLE
    if status == OUTCOME_NO_CHALLENGER:
        return OUTCOME_NO_CHALLENGER
    return OUTCOME_ERROR


def update_stress_results(conn, results):
    """
    Write PASS 2 (Phase 4) results onto existing rows.

    EVERY outcome is persisted now, not only status='ok' (audit B04). Previously a
    scenario whose pass ended in 'no_challenger' or 'error' was simply never written,
    so it read back through the API as "not yet tested" — indistinguishable from one
    the pass had never reached. With Batch 1's replay_infeasible added to that set,
    silence became actively misleading: a scenario refused because its own replay was
    untrustworthy would have looked untested.

    A row can describe TWO different runs, so it carries two of everything:

        the latest pass          last_attempt_outcome, stress_attempted_at
        the stored result        stress_outcome, stress_tested_at, min_perturbation,
                                 delta, collision_timestep, stress_method,
                                 stress_run_id, search_provenance,
                                 challengers_total, challengers_searched

    They diverge exactly when a later pass fails to reproduce an earlier success. The
    result group is then PRESERVED, not overwritten — see the column-ownership comment
    on the UPDATE below for why, and why one column could not honestly hold both.

    Also invalidates stale geometry (audit B14): a perturbed_paths entry was exported
    from the PREVIOUS delta, so committing a NEW result deletes it in the SAME
    transaction. That covers the case where the later geometry export fails or never
    runs. Only a pass that produced a result does this — a refused re-run supersedes
    nothing, so the preserved result keeps its matching path.

    Args:
        results: dict scenario_id -> phase 4 result dict
                 (as returned by batch_scorer.stress_test_scenarios)
    Returns:
        number of rows updated.
    """
    n = 0
    with conn.cursor() as cur:
        # Checked ONCE, up front, and not inside the DELETE. A guard in the DELETE's
        # WHERE clause would not help: Postgres resolves table names when it parses
        # the statement, so `DELETE FROM perturbed_paths` raises on a database where
        # Pass 3 has never run, whatever the WHERE says. Pass 2 legitimately runs
        # before the geometry tables exist.
        cur.execute("SELECT to_regclass('perturbed_paths') IS NOT NULL")
        geometry_exists = bool(cur.fetchone()[0])

        for sid, r in results.items():
            outcome = resolve_outcome(r)
            search_ran = outcome in OUTCOMES_SEARCH_RAN

            min_pert = r.get('min_perturbation') if search_ran else None
            # store NULL (not +inf) when no collision was found — SQL has no inf
            if min_pert is not None and min_pert == float('inf'):
                min_pert = None
            min_pert = None if min_pert is None else float(min_pert)

            delta = ([float(x) for x in r['delta']]
                     if search_ran and r.get('delta') is not None else None)
            t_hit = r.get('collision_timestep') if search_ran else None
            t_hit = None if t_hit is None or t_hit < 0 else int(t_hit)
            method = r.get('method') if search_ran else None
            target_idx = r.get('target_idx')

            run_id = (compute_stress_run_id(sid, target_idx, delta, method)
                      if search_ran and target_idx is not None else None)

            provenance = r.get('search_provenance')

            # Two groups of columns, with two different owners.
            #
            # THE LATEST PASS (always written): last_attempt_outcome,
            # stress_attempted_at.
            #
            # THE LAST VERIFIED RESULT (written only by a pass that PRODUCED one):
            # min_perturbation, delta, collision_timestep, stress_method,
            # stress_run_id, search_provenance, stress_outcome, stress_tested_at,
            # challengers_total, challengers_searched.
            #
            # challengers_total/searched are in the second group because they are part
            # of the SAME concept as search_provenance — "how was the stored result
            # searched", i.e. 1 challenger out of N. They began in the first group,
            # which split one concept across both owners and would have nulled them out
            # on an errored re-run (_stress_one returns neither field for 'error')
            # while min_perturbation beside them was being carefully preserved.
            #
            # stress_outcome and stress_tested_at sit in the SECOND group deliberately.
            # An earlier version of this function put them in the first, which made a
            # refused re-run produce a row reading stress_outcome='replay_infeasible'
            # beside min_perturbation=0.11 — the outcome field and the score fields
            # describing two different runs, asserting something false by juxtaposition.
            # That is the robustly_safe defect (B04) with different field names, and
            # traceability via stress_run_id is not a defence: the invariant is that no
            # field asserts more than its method established, not that a careful reader
            # can reconstruct the truth. Grouped this way the row is coherent BY
            # CONSTRUCTION — stress_outcome always describes exactly the run that
            # min_perturbation and delta came from.
            #
            # The preservation rule itself is Block 5 section 22's column ownership,
            # applied to the case it did not originally anticipate. That rule keeps
            # Pass 1 from wiping out stress results that cost hours of optimizer time;
            # the same principle says a later Pass 2 that could not run a search has
            # not superseded an earlier one that did, and must not destroy it. A
            # scenario that succeeded today and is refused tomorrow by a stricter
            # max_baseline_drift would otherwise lose the only successful result it
            # ever produced, silently.
            #
            # Retaining it misrepresents nothing, which is the other half of the trade:
            # last_attempt_outcome and stress_attempted_at report the refusal in full,
            # so the row says "here is a verified result, and here is what the most
            # recent pass concluded" rather than conflating the two. An overwrite would
            # make the result unrecoverable AND erase the difference between "re-
            # verified as no longer true" and "we failed to check this time".
            #
            # stress_tested_at belongs to the result for the same reason: an earlier
            # version cleared it, which silently dropped a real, persisted,
            # exact-SAT-verified collision out of /stats' collisions_found the moment
            # an unrelated later pass failed. It timestamps the stored result, so it
            # travels with the stored result.
            cur.execute("""
                UPDATE scenario_scores SET
                    min_perturbation     = CASE WHEN %s THEN %s ELSE min_perturbation END,
                    delta                = CASE WHEN %s THEN %s ELSE delta END,
                    collision_timestep   = CASE WHEN %s THEN %s ELSE collision_timestep END,
                    stress_method        = CASE WHEN %s THEN %s ELSE stress_method END,
                    stress_run_id        = CASE WHEN %s THEN %s ELSE stress_run_id END,
                    search_provenance    = CASE WHEN %s THEN %s ELSE search_provenance END,
                    stress_outcome       = CASE WHEN %s THEN %s ELSE stress_outcome END,
                    challengers_total    = CASE WHEN %s THEN %s ELSE challengers_total END,
                    challengers_searched = CASE WHEN %s THEN %s ELSE challengers_searched END,
                    stress_tested_at     = CASE WHEN %s THEN now() ELSE stress_tested_at END,
                    last_attempt_outcome = %s,
                    stress_attempted_at  = now()
                WHERE scenario_id = %s
            """, (search_ran, min_pert,
                  search_ran, delta,
                  search_ran, t_hit,
                  search_ran, method,
                  search_ran, run_id,
                  search_ran, Json(provenance) if provenance is not None else None,
                  search_ran, outcome,
                  search_ran, r.get('challengers_total'),
                  search_ran, r.get('challengers_searched'),
                  search_ran,
                  outcome,
                  sid))
            updated = cur.rowcount
            n += updated

            if updated and search_ran and geometry_exists:
                # Only a pass that produced a NEW result invalidates the geometry, for
                # the same reason. A refused re-run supersedes nothing, so the
                # preserved result keeps its matching path — they still share a
                # stress_run_id, so the read-side join continues to pair them
                # correctly. Deleting here would strand the preserved result without
                # its picture for no gain.
                cur.execute("DELETE FROM perturbed_paths WHERE scenario_id = %s",
                            (sid,))
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