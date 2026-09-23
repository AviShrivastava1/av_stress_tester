"""
test_f07_notebook_write_confirmation.py — fix F07 (independent review, notebook cell).

notebooks/colab_validation_run.ipynb's "11. update_stress_results, then re-read" cell
used to bind the return value to a bare int and never look at it again —
`n_updated = db.update_stress_results(conn, stress_results)` — even though that
function returns an UpdateReport specifically so a caller can see which scenarios
failed (`.failures`) and whether perturbed_paths' schema was too old for the call to
fully do its job (`.geometry_schema_incomplete`, fix F08). Every assertion in the cell
only checked that PASS-1 columns (fragility_score, min_ttc, ...) survived unchanged —
nothing checked that the PASS-2 columns this call is supposed to write
(stress_tested_at, min_perturbation, stress_outcome) actually landed with the
submitted values. The final "Confirmed: Phase 4 columns landed" print was earned by
proving Phase 1 columns didn't move, which is true whether or not Phase 4 wrote
anything at all.

Tests the REAL cell source, extracted from the notebook by content (never by index —
this project's own established rule, see test_audit3_regressions.py's
_cell_by_content), exec'd against a real, disposable database. A hand-written mirror
of the cell's logic could drift from what is actually in the notebook and prove
nothing about it; this cannot.

Needs a DISPOSABLE Postgres/PostGIS database and skips unless AV_CLAIMS_DB=1, same
requirement as every other DB-gated file in this project.

Run:
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_f07_notebook_write_confirmation.py -q
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring import db

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The same shape used throughout this project's other test files (test_batch11_
# contract.py, test_audit3_regressions.py, ...) — a completed, successful search.
_RESULT = {'status': 'ok', 'outcome': 'collision_found', 'collision': True,
           'min_perturbation': 0.5, 'delta': [-1.0, 0.0, 0.0, 0.0],
           'collision_timestep': 9, 'target_idx': 1, 'method': 'de'}


def _notebook_cells():
    path = os.path.join(PROJECT, 'notebooks', 'colab_validation_run.ipynb')
    return json.load(open(path))['cells']


def _cell_by_content(*required):
    """Located by CONTENT, never by index — this project's own rule after repeated
    index-drift breakage (see test_audit3_regressions.py's own _cell_by_content)."""
    hits = [''.join(c['source']) for c in _notebook_cells()
            if c['cell_type'] == 'code'
            and all(token in ''.join(c['source']) for token in required)]
    assert len(hits) == 1, f'expected exactly one cell matching {required}, got {len(hits)}'
    return hits[0]


def _write_cell():
    """cell 42, '11. update_stress_results, then re-read' — located by a token
    combination unique to it (the two print substrings it alone contains)."""
    return _cell_by_content('Confirmed: Phase 4 columns landed', 'fetch_top')


def _seed(conn, sid):
    db.upsert_scores(conn, [dict(scenario_id=sid, shard='x', n_agents=2, min_ttc=9.0,
                                 min_pet=9.0, fragility_score=1.0)])


@pytest.fixture
def dconn():
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, '
                    'scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    yield connection
    connection.rollback()
    connection.close()


@pytest.fixture
def migrated_conn():
    from src.scoring.export_geometry import init_geometry_schema
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, '
                    'scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    init_geometry_schema(connection)
    yield connection
    connection.rollback()
    connection.close()


@pytest.fixture
def stale_geom_conn():
    """
    A perturbed_paths table shaped as it was before the stress_run_id/
    scene_fingerprint ALTERs existed (same fixture shape as
    test_f08_geometry_schema_migration.py's stale_geom_conn, duplicated here per this
    project's own per-file convention). With fix F08 applied, a call against this
    connection succeeds and reports geometry_schema_incomplete=True rather than
    crashing — the state this test needs to force the flag genuinely true.
    """
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, '
                    'scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    with connection.cursor() as cur:
        cur.execute("""
            CREATE TABLE perturbed_paths (
                scenario_id TEXT PRIMARY KEY
                            REFERENCES scenario_scores(scenario_id) ON DELETE CASCADE,
                target_idx  INTEGER NOT NULL,
                n_points    INTEGER NOT NULL,
                exported_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
    connection.commit()
    yield connection
    connection.rollback()
    connection.close()


@requires_db
def test_F07_a_scenario_whose_write_failed_is_not_certified_confirmed(dconn):
    """
    THE PRIMARY REPRO. Two genuine successes and one write that fails for real —
    reusing test_batch12_contract.py's own malformed-delta trigger
    (delta=['not','a','number','!'], which the sanitizer cannot save and the
    per-scenario savepoint isolates) verbatim, rather than a hand-picked trigger that
    could accidentally stop reproducing the defect as the sanitizer evolves.

    'bad's Pass-1 columns genuinely do not move (upsert_scores seeded them, the
    rolled-back savepoint leaves them exactly there) — which is precisely why the
    OLD cell's Pass-1-only assertions were satisfied and printed "Confirmed" over a
    real, recorded failure.
    """
    records = [
        dict(scenario_id='good_a', shard='x', n_agents=2, min_ttc=9.0, min_pet=9.0,
             fragility_score=1.0),
        dict(scenario_id='bad', shard='x', n_agents=2, min_ttc=9.0, min_pet=9.0,
             fragility_score=1.0),
        dict(scenario_id='good_c', shard='x', n_agents=2, min_ttc=9.0, min_pet=9.0,
             fragility_score=1.0),
    ]
    db.upsert_scores(dconn, records)

    stress_results = {
        'good_a': dict(_RESULT),
        'bad': {'status': 'ok', 'outcome': 'collision_found', 'collision': True,
                'min_perturbation': 0.5, 'collision_timestep': 3, 'target_idx': 1,
                'method': 'de', 'delta': ['not', 'a', 'number', '!']},
        'good_c': dict(_RESULT),
    }

    env = dict(db=db, conn=dconn, stress_results=stress_results, records=records,
              TOP_N=3)
    with pytest.raises(AssertionError) as excinfo:
        exec(compile(_write_cell(), 'notebook_cell_f07', 'exec'), env)

    assert "'bad'" in str(excinfo.value), (
        f'the failed scenario must be named in the failure, not a generic message: '
        f'{excinfo.value}'
    )


@requires_db
def test_F07_a_stale_geometry_schema_is_surfaced_not_silently_confirmed(stale_geom_conn):
    """
    THE SECOND FLAG. A genuinely successful write (no scenario in .failures) against
    a database whose perturbed_paths predates the columns this call needs must still
    refuse to print "Confirmed" — that call silently skipped part of its own
    documented behaviour (stale-geometry invalidation), which is exactly the kind of
    thing this cell exists to catch.
    """
    records = [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0, min_pet=9.0,
                    fragility_score=1.0)]
    db.upsert_scores(stale_geom_conn, records)
    stress_results = {'s': dict(_RESULT)}

    env = dict(db=db, conn=stale_geom_conn, stress_results=stress_results,
              records=records, TOP_N=1)
    with pytest.raises(AssertionError) as excinfo:
        exec(compile(_write_cell(), 'notebook_cell_f07', 'exec'), env)

    assert 'schema is incomplete' in str(excinfo.value), excinfo.value


@requires_db
def test_F07_a_fully_successful_write_still_confirms(migrated_conn):
    """
    THE ORDINARY PATH MUST NOT BECOME A TAX. Genuine results, fully migrated schema,
    nothing failed — the fixed cell must still reach "Confirmed", and the Pass-2
    columns it now checks must genuinely match what was submitted.
    """
    import io
    import contextlib

    records = [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0, min_pet=9.0,
                    fragility_score=1.0)]
    db.upsert_scores(migrated_conn, records)
    stress_results = {'s': dict(_RESULT)}

    env = dict(db=db, conn=migrated_conn, stress_results=stress_results,
              records=records, TOP_N=1)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(_write_cell(), 'notebook_cell_f07', 'exec'), env)

    assert 'Confirmed: Phase 4 columns landed' in out.getvalue()
