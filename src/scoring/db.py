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

-- WHICH CHALLENGER the stored result was searched against (audit R06).
--
-- It was already folded into stress_run_id's hash and duplicated on
-- perturbed_paths, but never recoverable from the score row itself — so a scenario
-- that had been stress-tested and not yet geometry-exported could not answer "which
-- agent was this?", even though _stress_one knew.
--
-- IT SITS IN THE RESULT GROUP, not the attempt group, by Batch 2's own column
-- ownership rule rather than a new argument: target_idx is an input to
-- compute_stress_run_id alongside delta and stress_method, both of which are in the
-- result group. A row whose delta is preserved from an earlier run while target_idx
-- described a later REFUSED attempt would be the same juxtaposition defect the
-- split exists to prevent.
--
-- The refused attempt's own challenger is not lost; it goes to
-- last_attempt_diagnostics below. target_idx therefore partitions cleanly across
-- the two owners: the result group says which agent the STORED RESULT describes,
-- the attempt group says which agent the LATEST PASS selected.
ALTER TABLE scenario_scores ADD COLUMN IF NOT EXISTS target_idx INTEGER;

-- What the most recent attempt CONCLUDED, beyond its one-word outcome (audit R07).
--
-- ReplayFidelityError carries reason, baseline_replay_error and
-- baseline_replay_collides all the way to _stress_one (Batch 1), and persistence
-- used to keep none of it: a collision-refusal and a drift-refusal became
-- indistinguishable the moment the in-memory dict went out of scope. That defeats
-- the stated purpose of measuring baseline_replay_error at all, which is to
-- establish the real distribution before defending the 0.5 m default.
--
-- JSONB and not columns, for the same reason search_provenance is JSONB: these are
-- fields of one concept ("what did the latest attempt conclude"), they travel
-- together, and the set will grow as more refusal gates are added.
--
-- WRITTEN UNCONDITIONALLY, NULL when the attempt had nothing to report. Writing it
-- only on refusals would leave a stale diagnostic from an earlier refusal sitting
-- beside a later successful attempt — the same defect in a new column.
ALTER TABLE scenario_scores ADD COLUMN IF NOT EXISTS last_attempt_diagnostics JSONB;

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

-- Batch 11 (audit A02, A12, A05): WHICH SCENE this row describes.
--
-- scenario_id is a stable natural key from WOMD, but the states/validity/types arrays
-- behind it are not. A parser fix, a reinterpreted protobuf, or simply a different
-- shard carrying the same id — upsert_scores conflicts on scenario_id ALONE and
-- overwrites `shard`, so that collision is structural, not hypothetical — produces a
-- genuinely different scene under an unchanged key. Every identifier this project had
-- until now (stress_run_id, and the B14 join built on it) is a function of the
-- PERTURBATION, never of the scene, so none of them could see it.
--
-- A THIRD OWNER, and this is the part worth reading twice. Batch 2 established two
-- column groups on this table: the latest pass, and the last verified result. This
-- belongs to NEITHER. A fingerprint describes the INPUT, not the search — so it sits
-- with shard, n_agents and min_ttc: written once by Pass 1 through upsert_scores, and
-- never touched by update_stress_results. Putting it in the result group would say a
-- scene is a property of a search, which is backwards, and would make a refused
-- re-run preserve a fingerprint for a scene it never looked at.
--
-- NULL means NOT RECORDED and must not be read as a mismatch, exactly as B14 reasons
-- about stress_run_id. Rows written before this column keep exporting.
ALTER TABLE scenario_scores ADD COLUMN IF NOT EXISTS scene_fingerprint TEXT;

-- Fix F02 (independent review): WHICH AGENT is the SDC, alongside scene_fingerprint,
-- not folded into it.
--
-- compute_scene_fingerprint hashes states/validity/types only. sdc_track_index is a
-- separate scalar field in the WOMD protobuf (ScenarioParser.get_sdc_index reads it
-- directly, not derived from the arrays) — so two parses can produce a BYTE-IDENTICAL
-- array hash while disagreeing about which agent the SDC even is. Every collision
-- check in this project is anchored on sdc_idx, so that disagreement is exactly the
-- kind of "different scene under an unchanged key" A02 was about, invisible to the
-- fingerprint alone.
--
-- A SEPARATE COLUMN, NOT A CHANGE TO compute_scene_fingerprint's OWN HASH. That
-- function's contract is versioned (the 'scenefp1' prefix) and already relied on by
-- Pass 1 and the whole A02/A05/A12 test family; folding sdc_idx into it would need a
-- version bump and would invalidate every already-stored fingerprint until Pass 1
-- re-runs shard-wide — a much bigger blast radius than one write-path check.
--
-- Same ownership as scene_fingerprint: written once by Pass 1, describes the input,
-- never touched by update_stress_results. NULL means NOT RECORDED, same carve-out.
ALTER TABLE scenario_scores ADD COLUMN IF NOT EXISTS sdc_idx INTEGER;
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


def compute_scene_fingerprint(states, validity, types) -> str:
    """
    A short, deterministic identifier for the SCENE ITSELF (audit A02).

    compute_stress_run_id above identifies a perturbation applied to a scene. Nothing
    identified the scene, so a re-parse under a stable scenario_id could produce a
    matching run id against different content — the id was never a function of content
    in the first place. This closes that: geometry is published only if the arrays it
    was built from are the arrays the stored result was computed against.

    WHAT IS HASHED: the whole scenario, every agent. Narrower options exist — the SDC
    and target are the only two the collision check reads — but the question this
    answers is "is this the same scene", and scenario_agents exports EVERY agent, so
    the baseline the dashboard draws genuinely depends on all of them. A corrected
    bystander coordinate does therefore invalidate a result that never touched that
    agent; that is deliberate, and it surfaces as an explicit refusal rather than being
    silently decided either way.

    WHAT IS NOT HASHED: anything recomputed. Hashing PerturbationSpace.apply output
    would let export_perturbed_path verify the target's own kinematics from its
    mandatory arguments, which would close the legacy-signature gap completely — but it
    would compare floating-point results recomputed on two platforms, and this project
    runs Pass 2 on Colab's Python 3.8 and everything else on 3.11. A check that fails
    closed on valid input is worse than the defect it replaces (R10, R11). Raw parsed
    input only.

    CANONICALIZATION, and every part of it is load-bearing:

      - dtypes and BYTE ORDER are forced ('<f4', 'bool', '<i4'). The same scene must
        fingerprint identically on the Colab half and the local half of this project's
        documented split, and tobytes() is byte-order-dependent.
      - ascontiguousarray, so a view or a transposed-then-restored array hashes as the
        array it represents rather than as its memory layout.
      - shapes are hashed alongside the bytes, so a reshape cannot collide with a
        genuinely different scene that happens to share a byte string.
      - a version prefix, so this algorithm can change without a new value silently
        colliding with an old one.

    sha256 truncated to 16 hex, matching compute_stress_run_id: 64 bits is far beyond
    what accidental change requires, and the adversarial case is not the threat model.

    -0.0 and 0.0 are numerically equal and hash differently. Left alone deliberately:
    normalizing would mean rewriting the array before hashing it, and a parser that
    starts emitting negative zero HAS changed its output.
    """
    import numpy as np

    parts = [b'scenefp1']
    for array, dtype in ((states, '<f4'), (validity, 'bool'), (types, '<i4')):
        canonical = np.ascontiguousarray(np.asarray(array), dtype=dtype)
        parts.append(repr(canonical.shape).encode('ascii'))
        parts.append(canonical.tobytes())
    return hashlib.sha256(b'|'.join(parts)).hexdigest()[:16]


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
    # scene_fingerprint and sdc_idx join the SCENARIO-OWNED columns here (audit A02,
    # fix F02), not the result or attempt groups — they describe the input Pass 1
    # read, so Pass 1 writes them and update_stress_results never touches them.
    # r.get(), not r[...], so a caller that predates either column still writes a
    # row; NULL means "not recorded".
    rows = [(r['scenario_id'], r.get('shard'), r.get('n_agents'),
             r['min_ttc'], r['min_pet'], r['fragility_score'],
             r.get('min_ttc_all_pairs'), r.get('min_pet_all_pairs'),
             r.get('scene_fingerprint'), r.get('sdc_idx'))
            for r in records]
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO scenario_scores
                (scenario_id, shard, n_agents, min_ttc, min_pet, fragility_score,
                 min_ttc_all_pairs, min_pet_all_pairs, scene_fingerprint, sdc_idx)
            VALUES %s
            ON CONFLICT (scenario_id) DO UPDATE SET
                shard             = EXCLUDED.shard,
                n_agents          = EXCLUDED.n_agents,
                min_ttc           = EXCLUDED.min_ttc,
                min_pet           = EXCLUDED.min_pet,
                fragility_score   = EXCLUDED.fragility_score,
                min_ttc_all_pairs = EXCLUDED.min_ttc_all_pairs,
                min_pet_all_pairs = EXCLUDED.min_pet_all_pairs,
                -- COALESCE, not a bare overwrite: a re-score by a caller that does
                -- not compute a fingerprint/sdc_idx must not ERASE one that was
                -- recorded. Dropping it would silently reopen the carve-out for that
                -- row and make every later export/write unguarded.
                scene_fingerprint = COALESCE(EXCLUDED.scene_fingerprint,
                                             scenario_scores.scene_fingerprint),
                sdc_idx           = COALESCE(EXCLUDED.sdc_idx,
                                             scenario_scores.sdc_idx),
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


# The fields ReplayFidelityError carries, in the order its own docstring lists them.
# Named here so "everything the refusal knew" has one definition rather than being
# re-enumerated at the call site (audit R07).
_ATTEMPT_DIAGNOSTIC_FIELDS = (
    'reason',
    'baseline_replay_error',
    # REDUNDANT TODAY, STORED ANYWAY. There are exactly two raise sites in
    # PerturbationSpace — ('collision', err, True) and ('drift', err, False) — so
    # this boolean is currently derivable from `reason`. It is persisted because a
    # future gate could raise 'drift' with collides=True, and a reader who had been
    # taught to derive it would have no way to learn the mapping had stopped being
    # two-way. The project already makes this trade for min_ttc_all_pairs and
    # challengers_total: record the fact, do not make a later reader re-derive an
    # invariant from code structure that can silently change. One boolean in a JSONB
    # blob is free; a silently-wrong derivation is not.
    'baseline_replay_collides',
    # Which challenger THIS attempt selected, which is not necessarily the one the
    # stored result describes — see the target_idx column comment.
    'target_idx',
    # WHY THE ATTEMPT FAILED, when it failed for a reason this project did not
    # anticipate (audit A06). This tuple was built around what ReplayFidelityError
    # carries, so the 'error' outcome — the catch-all for everything unexpected — had
    # nothing in the allowlist and was dropped the moment the in-memory dict went out
    # of scope. The row said 'error' and could not say why, which is the one outcome
    # where "why" is the entire content.
    #
    # TWO FIELDS, NOT ONE. stress_test_scenarios has always built
    # f'{type(e).__name__}: {e}', fusing both facts into a string the notebook then
    # takes apart again with .split(':')[0]. A type is a category you can GROUP BY; a
    # message is prose. Storing them separately is what makes the first possible.
    'error_type',
    'error_message',
)


class UpdateReport(int):
    """
    How many rows were updated, plus which scenarios could not be written.

    AN int SUBCLASS, so every existing caller is untouched (audit A03). Forty-five
    call sites take this return value; exactly one asserts on it
    (test_claims_contract.py's `== 1`) and one assigns it (the notebook's n_updated).
    Both keep working because this IS an int — the same trick StressResults used in
    Batch 5, and for the same reason: a richer return that breaks its callers is a
    second defect, not a fix.

    `.failures` is a list of {scenario_id, error} records rather than a count, matching
    export_shard_geometry's perturbed_stale and Block 5 Concept 19: an operator looking
    at a spike needs to know WHICH scenarios and why, and a tally answers neither.
    """

    def __new__(cls, count, failures=()):
        report = super().__new__(cls, count)
        report.failures = list(failures)
        return report


_NONFINITE_KEY = 'nonfinite_fields'


def _describe_nonfinite(value: float) -> str:
    """'inf', '-inf' or 'nan' — the three things JSON cannot spell."""
    if value != value:
        return 'nan'
    return 'inf' if value > 0 else '-inf'


def _sanitize_for_json(value, path=''):
    """
    Replace every non-finite float with None, and report what was replaced.

    Returns (cleaned, {dotted_path: 'inf'|'-inf'|'nan'}).

    WHY THIS EXISTS AT THE BOUNDARY RATHER THAN PER FIELD (audit A03). Python's
    json.dumps emits the non-standard tokens Infinity / -Infinity / NaN, which
    PostgreSQL's JSONB parser correctly rejects:

        psycopg2.errors.InvalidTextRepresentation: invalid input syntax for type json

    This project has met that twice already and fixed it pointwise both times: Block 6
    Concept 24 established NULL-not-infinity for min_perturbation, and Batch 9 wrapped
    target_min_speed in np.isfinite for exactly this reason. Both are correct and both
    are per-field vigilance, which does not generalise to the next field somebody adds
    — and A06 adds two in this same batch. This is the same decision made once, at the
    only place every JSONB value must pass through.

    Recursive over dicts and lists because search_provenance is not flat all the way
    down: `bounds` is a list of [lo, hi] pairs, and a non-finite bound would be just as
    unstorable as a scalar.

    bool is checked BEFORE float. In Python bool is a subclass of int, not float, so it
    would not be caught here anyway — but isinstance(True, int) surprises people, and
    the explicit skip stops a later edit from "simplifying" this into a numeric check
    that silently rewrites booleans.
    """
    import math

    if isinstance(value, dict):
        cleaned, report = {}, {}
        for key, item in value.items():
            sub, sub_report = _sanitize_for_json(item, f'{path}.{key}' if path else str(key))
            cleaned[key] = sub
            report.update(sub_report)
        return cleaned, report

    if isinstance(value, (list, tuple)):
        cleaned, report = [], {}
        for index, item in enumerate(value):
            sub, sub_report = _sanitize_for_json(item, f'{path}.{index}' if path else str(index))
            cleaned.append(sub)
            report.update(sub_report)
        return cleaned, report

    if isinstance(value, bool):
        return value, {}

    if isinstance(value, float) and not math.isfinite(value):
        return None, {path: _describe_nonfinite(value)}

    return value, {}


def _json_safe(blob):
    """
    A JSONB-storable version of `blob`, with any non-finite value replaced by null and
    ONE extra key naming what was replaced (audit A03).

    NULL PLUS A RECORD, NOT A BARE NULL, and the distinction is the whole point. The
    value cannot persist as a number — JSON has no infinity literal — but a bare null
    is indistinguishable from "this attempt recorded no such diagnostic", which
    _attempt_diagnostics' own docstring is careful to mean. So:

        {'baseline_replay_error': None,
         'nonfinite_fields': {'baseline_replay_error': 'inf'}}

    One key however many fields were affected, purely additive so no existing reader
    changes, and the numeric fields stay numeric-or-null rather than sometimes-a-string
    — the notebook's drift-distribution study reads baseline_replay_error and should
    not have to type-check it.
    """
    if blob is None:
        return None
    cleaned, report = _sanitize_for_json(blob)
    if report and isinstance(cleaned, dict):
        cleaned[_NONFINITE_KEY] = report
    return cleaned


def _attempt_diagnostics(result) -> dict:
    """
    What the latest attempt concluded beyond its outcome word, or None if it had
    nothing to report (audit R07).

    None rather than an empty dict, so the column reads NULL — "this attempt
    recorded no diagnostics" — instead of "{}", which invites a reader to conclude
    the attempt recorded an empty set of them.
    """
    present = {k: result[k] for k in _ATTEMPT_DIAGNOSTIC_FIELDS if result.get(k) is not None}
    return present or None


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

        the latest pass          last_attempt_outcome, stress_attempted_at,
                                 last_attempt_diagnostics
        the stored result        stress_outcome, stress_tested_at, min_perturbation,
                                 delta, collision_timestep, stress_method,
                                 stress_run_id, search_provenance, target_idx,
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
    failures = []
    with conn.cursor() as cur:
        # Checked ONCE, up front, and not inside the DELETE. A guard in the DELETE's
        # WHERE clause would not help: Postgres resolves table names when it parses
        # the statement, so `DELETE FROM perturbed_paths` raises on a database where
        # Pass 3 has never run, whatever the WHERE says. Pass 2 legitimately runs
        # before the geometry tables exist.
        cur.execute("SELECT to_regclass('perturbed_paths') IS NOT NULL")
        geometry_exists = bool(cur.fetchone()[0])

        for sid, r in results.items():
            # ── PER-SCENARIO ISOLATION (audit A03) ──────────────────────────────
            #
            # R02's rule — one bad record must never kill the batch — applied to the
            # stage that never got it. Pass 1 and Pass 2 got it in Batch 5 and Pass 3
            # has had it from the start; persistence, the ONE loop whose failure
            # discards work that has already been COMPUTED, had none. Measured before
            # this savepoint existed: three results in one call, one unserializable,
            # and all three came back last_attempt_outcome=None — including the valid
            # one processed FIRST, whose UPDATE had already executed.
            #
            # A SAVEPOINT and not a commit-per-row. Committing inside the loop would
            # give up the property that a call either advances a scenario or leaves it
            # alone, and would multiply fsyncs across a shard-sized batch. ROLLBACK TO
            # SAVEPOINT discards one scenario's statements and leaves the surrounding
            # transaction usable, which is exactly the granularity wanted.
            #
            # THE SAVEPOINT IS TAKEN BEFORE THE PARAMETERS ARE BUILT, AND THE HANDLER
            # CATCHES Exception RATHER THAN psycopg2.Error. The first version did
            # neither, and this batch's own isolation test caught it: a delta of
            # ['not','a','number'] raises ValueError inside [float(x) for x in
            # r['delta']] — Python-level, before any SQL reaches the database — so a
            # narrow handler wrapped around the statements alone let a malformed
            # result dict kill the batch exactly as before. "A malformed result" is
            # the category this finding is about, so the isolation has to cover the
            # whole per-scenario body. score_shard and export_shard_geometry already
            # catch Exception here with the same deliberate breadth.
            #
            # The sanitizer above stops the trigger this was found through; this stops
            # the CLASS. That distinction earns its keep in this very batch: A06 is
            # adding arbitrary exception text to the same JSONB blob.
            cur.execute('SAVEPOINT scenario_write')
            try:
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

                # Sanitized at the ONE place every JSONB value passes through (audit
                # A03), rather than trusting each producer to have remembered.
                provenance = _json_safe(r.get('search_provenance'))
                diagnostics = _json_safe(_attempt_diagnostics(r))

                # WHAT THIS ATTEMPT WAS SEARCHED AGAINST (fix F02), captured by
                # _stress_one at entry — see its own comment. None for every outcome
                # that never reaches a search (no_challenger, replay_infeasible,
                # error) and for any caller that predates this field; None on EITHER
                # side is read as "not recorded", never as a mismatch, below.
                captured_fp = r.get('scene_fingerprint')
                captured_sdc = r.get('sdc_idx')

                # Two groups of columns, with two different owners.
                #
                # THE LATEST PASS (always written): last_attempt_outcome,
                # stress_attempted_at, last_attempt_diagnostics.
                #
                # THE LAST VERIFIED RESULT (written only by a pass that PRODUCED one):
                # min_perturbation, delta, collision_timestep, stress_method,
                # stress_run_id, search_provenance, stress_outcome, stress_tested_at,
                # target_idx, challengers_total, challengers_searched.
                #
                # target_idx joined the second group in Batch 7 (audit R06) because it is
                # an input to compute_stress_run_id beside delta and stress_method, which
                # are already there. last_attempt_diagnostics joined the first (audit R07)
                # and is written on EVERY pass, NULL included — a diagnostic left behind
                # by an earlier refusal sitting beside a later successful attempt is the
                # same juxtaposition this grouping exists to prevent.
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
                #
                # ── fix F02: THE SCENE-IDENTITY GUARD ────────────────────────────
                #
                # Until this fix, every CASE above read `search_ran` alone — this
                # function asked "did a search run", never "did it run against the
                # scene this row NOW describes". Pass 1 can re-run between a search
                # starting and this call persisting it (upsert_scores has no guard;
                # by design, per Block 5 v4 — Pass 1 owns this column outright), and
                # nothing stopped a stale search's collision_found from being
                # attached to a row that had since moved on. The DELETE below
                # already compares scene_fingerprint this way, for perturbed_paths;
                # this closes the identical gap for the row's own result columns.
                #
                # READ IN THE SAME STATEMENT, not a Python-side SELECT-then-UPDATE —
                # the DELETE's own comment explains why: this function must not take
                # a second opinion on a column Pass 1 owns, and a separate read would
                # reopen exactly the race a single statement closes.
                #
                # BOTH scene_fingerprint AND sdc_idx, not the fingerprint alone.
                # compute_scene_fingerprint hashes states/validity/types only;
                # sdc_track_index is a separate WOMD field two parses could disagree
                # on while every array still hashes identically, and every collision
                # check is anchored on sdc_idx — invisible to the fingerprint alone.
                #
                # NULL ON EITHER SIDE MEANS "NOT RECORDED", NOT "MISMATCH" — the
                # same carve-out A12 established for the export path, extended here.
                # The alternative (refuse whenever the CAPTURED side is NULL) would
                # break the fixture shape most of this project's own test suite
                # already uses — synthetic result dicts built by hand, with no
                # scene_fingerprint key — for a check aimed at real Pass 2 traffic,
                # which always carries one now. Refusing only requires BOTH sides
                # present and disagreeing.
                cur.execute("""
                    WITH row_check AS (
                        SELECT scene_fingerprint AS stored_scene_fingerprint,
                               sdc_idx AS stored_sdc_idx,
                               (scene_fingerprint IS NOT DISTINCT FROM %s
                                OR scene_fingerprint IS NULL OR %s IS NULL)
                               AND
                               (sdc_idx IS NOT DISTINCT FROM %s
                                OR sdc_idx IS NULL OR %s IS NULL) AS scene_matches
                          FROM scenario_scores
                         WHERE scenario_id = %s
                    )
                    UPDATE scenario_scores SET
                        min_perturbation     = CASE WHEN %s AND (SELECT scene_matches FROM row_check) THEN %s ELSE min_perturbation END,
                        delta                = CASE WHEN %s AND (SELECT scene_matches FROM row_check) THEN %s ELSE delta END,
                        collision_timestep   = CASE WHEN %s AND (SELECT scene_matches FROM row_check) THEN %s ELSE collision_timestep END,
                        stress_method        = CASE WHEN %s AND (SELECT scene_matches FROM row_check) THEN %s ELSE stress_method END,
                        stress_run_id        = CASE WHEN %s AND (SELECT scene_matches FROM row_check) THEN %s ELSE stress_run_id END,
                        search_provenance    = CASE WHEN %s AND (SELECT scene_matches FROM row_check) THEN %s ELSE search_provenance END,
                        target_idx           = CASE WHEN %s AND (SELECT scene_matches FROM row_check) THEN %s ELSE target_idx END,
                        stress_outcome       = CASE WHEN %s AND (SELECT scene_matches FROM row_check) THEN %s ELSE stress_outcome END,
                        challengers_total    = CASE WHEN %s AND (SELECT scene_matches FROM row_check) THEN %s ELSE challengers_total END,
                        challengers_searched = CASE WHEN %s AND (SELECT scene_matches FROM row_check) THEN %s ELSE challengers_searched END,
                        stress_tested_at     = CASE WHEN %s AND (SELECT scene_matches FROM row_check) THEN now() ELSE stress_tested_at END,
                        last_attempt_outcome = %s,
                        -- The outcome word stays honest even when refused (fix F02):
                        -- the search DID reach collision_found/no_collision_found,
                        -- unlike replay_infeasible where none ran at all. What did
                        -- NOT happen is that outcome being attached to the row, and
                        -- that goes in diagnostics, not a relabelled outcome.
                        last_attempt_diagnostics = CASE
                            WHEN %s AND NOT (SELECT scene_matches FROM row_check)
                            THEN COALESCE(%s::jsonb, '{}'::jsonb) || jsonb_build_object(
                                    'scene_mismatch', true,
                                    'captured_scene_fingerprint', %s,
                                    'stored_scene_fingerprint',
                                        (SELECT stored_scene_fingerprint FROM row_check),
                                    'captured_sdc_idx', %s,
                                    'stored_sdc_idx',
                                        (SELECT stored_sdc_idx FROM row_check)
                                 )
                            ELSE %s
                        END,
                        stress_attempted_at  = now()
                    WHERE scenario_id = %s
                    RETURNING (SELECT scene_matches FROM row_check)
                """, (captured_fp, captured_fp, captured_sdc, captured_sdc, sid,
                      search_ran, min_pert,
                      search_ran, delta,
                      search_ran, t_hit,
                      search_ran, method,
                      search_ran, run_id,
                      search_ran, Json(provenance) if provenance is not None else None,
                      search_ran, None if target_idx is None else int(target_idx),
                      search_ran, outcome,
                      search_ran, r.get('challengers_total'),
                      search_ran, r.get('challengers_searched'),
                      search_ran,
                      outcome,
                      search_ran,
                      Json(diagnostics) if diagnostics is not None else None,
                      captured_fp,
                      captured_sdc,
                      Json(diagnostics) if diagnostics is not None else None,
                      sid))
                updated = cur.rowcount
                row = cur.fetchone()
                scene_matches = bool(row[0]) if row and row[0] is not None else True

                if updated and search_ran and scene_matches and geometry_exists:
                    # scene_matches ADDED (fix F02), read back from the UPDATE's own
                    # RETURNING rather than recomputed: when the guard above refused
                    # to attach this attempt's result, there is no NEW result for
                    # this DELETE to invalidate anything for, and running it anyway
                    # would compare stress_run_id against a run_id THIS statement
                    # just decided not to trust — computed from the rejected
                    # attempt's own delta/method/target, not from whatever the
                    # preserved row's geometry actually pairs with. Skipping it
                    # entirely leaves the preserved result's geometry exactly where
                    # A05 already established it belongs: untouched.
                    #
                    # Only a pass that produced a NEW result invalidates the geometry, for
                    # the same reason. A refused re-run supersedes nothing, so the
                    # preserved result keeps its matching path — they still share a
                    # stress_run_id, so the read-side join continues to pair them
                    # correctly. Deleting here would strand the preserved result without
                    # its picture for no gain.
                    #
                    # CONDITIONAL, NOT UNCONDITIONAL (audit A05). This used to delete
                    # whenever a search ran, without ever asking whether the new identity
                    # DIFFERED from the stored one — so a bit-for-bit identical retry (same
                    # delta, same method, same target, therefore the same run id by
                    # construction) destroyed working geometry for nothing. Measured: rows
                    # 1 -> 0 with the stored run id unchanged at 6854242eaa4550ec.
                    #
                    # BOTH IDENTITIES, AND THE FINGERPRINT IS NOT DECORATIVE HERE. A
                    # re-parsed scene that happens to yield the SAME optimal delta produces
                    # the SAME stress_run_id, so a run-id-only comparison would preserve a
                    # path built from the old scene and the read-side join would serve it —
                    # fixing A05 by reopening A02 one table over. The scene is read from the
                    # row in the same statement rather than passed in, because Pass 1 owns
                    # that column and this function must not take a second opinion on it.
                    #
                    # IS DISTINCT FROM is the exact inverse of the IS NOT DISTINCT FROM the
                    # read side uses, so the NULL semantics are the ones B14 reasoned about:
                    # a legacy path (NULL) against a new run still deletes, as it does today.
                    cur.execute("""
                        DELETE FROM perturbed_paths pp
                         WHERE pp.scenario_id = %s
                           AND (pp.stress_run_id IS DISTINCT FROM %s
                                OR pp.scene_fingerprint IS DISTINCT FROM (
                                    SELECT ss.scene_fingerprint FROM scenario_scores ss
                                     WHERE ss.scenario_id = %s))
                    """, (sid, run_id, sid))

            except Exception as exc:  # noqa: BLE001 — deliberate: isolate per scenario
                # RECORDED AND THE WALK CONTINUES, the same shape and the same
                # deliberate breadth score_shard and export_shard_geometry already use.
                # Narrower was tried and was wrong — see the savepoint comment above.
                cur.execute('ROLLBACK TO SAVEPOINT scenario_write')
                failures.append({'scenario_id': sid,
                                 'error': f'{type(exc).__name__}: {exc}'.strip()})
            else:
                cur.execute('RELEASE SAVEPOINT scenario_write')
                n += updated
    conn.commit()
    return UpdateReport(n, failures)


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