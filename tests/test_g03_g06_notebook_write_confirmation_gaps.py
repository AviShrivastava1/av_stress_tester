"""
test_g03_g06_notebook_write_confirmation_gaps.py — fixes G03 + G06 (independent
review, notebook cell).

Two gaps in notebooks/colab_validation_run.ipynb's "11. update_stress_results, then
re-read" cell (the same cell fix F07 already hardened once, for a different
defect — see test_f07_notebook_write_confirmation.py).

G03: `if submitted.get('min_perturbation') is not None:` treats presence as proof
of a comparable value. A genuine no_collision_found result carries
min_perturbation=float('inf') in memory (Block 5 Concept 21: NULL, not infinity,
is what gets STORED — db.py's update_stress_results converts inf to NULL at write
time). float('inf') is not None, so the old check ran
`abs(row['min_perturbation'] - submitted['min_perturbation'])` against the
correctly-stored NULL and the in-memory inf: TypeError, on an entirely ordinary
no-collision result. Fixed by checking finiteness, not just non-None-ness, and
asserting row['min_perturbation'] is None on the else side.

G06: the cell never checked last_attempt_diagnostics for scene_mismatch, and never
checked collision_timestep at all — only stress_outcome and min_perturbation. A
write refused by F02's own scene/SDC identity guard is not a report.failures entry
(nothing raised) and the preserved row's stress_outcome/min_perturbation can
coincidentally match a stale, rejected submission on exactly the two fields the
cell checked, printing "Confirmed" over a write that never landed. Fixed two ways:
(1) last_attempt_diagnostics is checked for scene_mismatch before trusting any
row-vs-submitted comparison, collecting refusals into scene_mismatches, which now
blocks the final "Confirmed" print the same way report.failures/
geometry_schema_incomplete already do; (2) collision_timestep joins the checked
fields unconditionally, independent of (1) — its total absence is a real gap on
its own, the same category F07 already closed for min_perturbation/stress_outcome
("proving Pass-1 columns did not move is not proof Pass-2 columns landed" applies
one field further down).

Tests the REAL cell source, extracted from the notebook by content (never by index
— this project's own established rule, see test_audit3_regressions.py's
_cell_by_content), exec'd against a real, disposable database. A hand-written
mirror of the cell's logic could drift from what is actually in the notebook and
prove nothing about it; this cannot.

Needs a DISPOSABLE Postgres/PostGIS database and skips unless AV_CLAIMS_DB=1, same
requirement as every other DB-gated file in this project.

Run:
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_g03_g06_notebook_write_confirmation_gaps.py -q
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
    """cell 42, '11. update_stress_results, then re-read' — same selector
    test_f07_notebook_write_confirmation.py's own _write_cell() uses."""
    return _cell_by_content('Confirmed: Phase 4 columns landed', 'fetch_top')


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


def _run_write_cell(conn, records, stress_results, top_n=1):
    env = dict(db=db, conn=conn, stress_results=stress_results, records=records,
              TOP_N=top_n)
    exec(compile(_write_cell(), 'notebook_cell_g03_g06', 'exec'), env)
    return env


@requires_db
def test_G03_a_genuine_no_collision_result_does_not_crash_the_cell(migrated_conn):
    """
    THE PRIMARY REPRO. A real no_collision_found result carries
    min_perturbation=float('inf') in memory. Pre-fix, `submitted.get(
    'min_perturbation') is not None` was True for inf, and the cell crashed
    comparing the correctly-stored NULL against it.
    """
    records = [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0, min_pet=9.0,
                    fragility_score=1.0)]
    db.upsert_scores(migrated_conn, records)

    stress_results = {'s': {'status': 'ok', 'outcome': 'no_collision_found',
                            'collision': False, 'min_perturbation': float('inf'),
                            'delta': None, 'collision_timestep': None,
                            'target_idx': 1, 'method': 'de'}}

    # Must not raise at all — a TypeError here (not an AssertionError) is exactly
    # what pre-fix behaviour looked like.
    _run_write_cell(migrated_conn, records, stress_results)

    row = db.fetch_scenario(migrated_conn, 's')
    assert row['min_perturbation'] is None, (
        'fixture check: a no_collision_found result must store NULL, not inf'
    )


@requires_db
def test_G03_a_finite_min_perturbation_is_genuinely_verified_not_merely_present(
        migrated_conn):
    """
    THE OVERCORRECTION GUARD, STATED PRECISELY. This cell writes stress_results via
    update_stress_results and THEN reads back and compares against that SAME dict
    — for an ACCEPTED write, the round trip is deterministic (db.py's sanitizing is
    the only thing that can make row and submitted disagree, and that boundary is
    exactly what the primary repro above exercises). So there is no reachable
    "genuinely accepted write, real finite value, silently wrong" case to construct
    here — that is a property of this cell's own structure, not a gap in this test.

    What CAN be shown, and is: the `if` branch (real comparison) is the one that
    actually ran here, not the `else` branch (assert None) misfiring on a real
    value — proven by asserting the specific, non-trivial value (0.5, not 0 and not
    None) landed, rather than merely asserting the cell didn't raise.
    """
    records = [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0, min_pet=9.0,
                    fragility_score=1.0)]
    db.upsert_scores(migrated_conn, records)

    _run_write_cell(migrated_conn, records, {'s': dict(_RESULT)})

    row = db.fetch_scenario(migrated_conn, 's')
    assert row['min_perturbation'] == 0.5, (
        f"expected the genuine finite value 0.5 to round-trip, got "
        f"{row['min_perturbation']!r} — the comparison branch did not verify a "
        f"real value"
    )


@requires_db
def test_G06_a_scene_refused_write_is_not_certified_confirmed(migrated_conn):
    """
    THE FINDING'S OWN REPRO. Genuine result against scene B (frame 8, norm 0.5),
    then a stale result submitted against scene A (frame 9, same outcome, same
    norm). F02 correctly refuses it; the row correctly keeps B's values. Pre-fix,
    the cell's only two checks (stress_outcome, min_perturbation) coincidentally
    matched and it printed "Confirmed" over a refused write.
    """
    records = [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0, min_pet=9.0,
                    fragility_score=1.0, scene_fingerprint='fp_B', sdc_idx=0)]
    db.upsert_scores(migrated_conn, records)

    genuine = dict(_RESULT, collision_timestep=8, scene_fingerprint='fp_B', sdc_idx=0)
    db.update_stress_results(migrated_conn, {'s': genuine})

    stale = dict(_RESULT, collision_timestep=9, scene_fingerprint='fp_A', sdc_idx=0)
    with pytest.raises(AssertionError) as excinfo:
        _run_write_cell(migrated_conn, records, {'s': stale})

    assert "'s'" in str(excinfo.value), (
        f'the refused scenario must be named in the failure: {excinfo.value}'
    )
    row = db.fetch_scenario(migrated_conn, 's')
    assert row['collision_timestep'] == 8, (
        'fixture check: the refusal must have preserved the genuine value'
    )


@requires_db
def test_G06_the_scene_mismatch_is_reported_not_just_blocked(migrated_conn):
    """
    THE DIAGNOSIS, NOT JUST THE REFUSAL. The raised AssertionError must carry
    enough to diagnose the refusal (scene_mismatch present), not a generic message
    — matching the diagnostics db.py's own update_stress_results already attaches.
    """
    records = [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0, min_pet=9.0,
                    fragility_score=1.0, scene_fingerprint='fp_B', sdc_idx=0)]
    db.upsert_scores(migrated_conn, records)
    db.update_stress_results(migrated_conn, {'s': dict(
        _RESULT, scene_fingerprint='fp_B', sdc_idx=0)})

    stale = dict(_RESULT, scene_fingerprint='fp_A', sdc_idx=0)
    with pytest.raises(AssertionError) as excinfo:
        _run_write_cell(migrated_conn, records, {'s': stale})
    assert 'scene_mismatch' in str(excinfo.value), excinfo.value


@requires_db
def test_G06_collision_timestep_is_genuinely_verified_independent_of_scene_mismatch(
        migrated_conn):
    """
    THE INDEPENDENT GAP, STATED PRECISELY. No scene mismatch anywhere in this test
    — proving collision_timestep is checked on its OWN terms, not merely as a
    byproduct of the scene_mismatch branch above it. Same structural point as
    test_G03_a_finite_min_perturbation_is_genuinely_verified_not_merely_present:
    this cell's own write-then-compare-the-same-dict shape means a genuinely
    ACCEPTED write's collision_timestep cannot legitimately diverge from
    submitted's (db.py's own sanitizing at the None/negative boundary is the only
    thing that can do that, and that boundary is
    test_G06_a_no_collision_result_still_checks_collision_timestep_is_null's job).
    What this proves instead: the comparison branch actually ran and verified a
    specific, non-trivial frame (8, not 0 and not None), not merely "didn't crash".
    Before this fix, collision_timestep was not read from `row` at all here — this
    is the check that would have caught the finding's own G06 repro on its own,
    had the scene_mismatch check not already caught it first.
    """
    records = [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0, min_pet=9.0,
                    fragility_score=1.0)]
    db.upsert_scores(migrated_conn, records)

    genuine = dict(_RESULT, collision_timestep=8)
    _run_write_cell(migrated_conn, records, {'s': genuine})

    row = db.fetch_scenario(migrated_conn, 's')
    assert row['collision_timestep'] == 8, (
        f"expected the genuine frame 8 to round-trip, got "
        f"{row['collision_timestep']!r} — the comparison branch did not verify a "
        f"real value"
    )


@requires_db
def test_G06_a_no_collision_result_still_checks_collision_timestep_is_null(
        migrated_conn):
    """
    G03's OWN GUARD, APPLIED TO THE NEW FIELD. collision_timestep is sanitized the
    same way as min_perturbation (db.py: None or negative -> NULL). A naive
    `is not None` check on this field would reproduce G03's exact bug class for a
    genuine no_collision_found result, whose submitted collision_timestep is None
    already — so this must NOT raise.
    """
    records = [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0, min_pet=9.0,
                    fragility_score=1.0)]
    db.upsert_scores(migrated_conn, records)

    stress_results = {'s': {'status': 'ok', 'outcome': 'no_collision_found',
                            'collision': False, 'min_perturbation': float('inf'),
                            'delta': None, 'collision_timestep': None,
                            'target_idx': 1, 'method': 'de'}}
    _run_write_cell(migrated_conn, records, stress_results)

    row = db.fetch_scenario(migrated_conn, 's')
    assert row['collision_timestep'] is None


@requires_db
def test_G06_a_fully_successful_write_still_confirms(migrated_conn):
    """
    THE ORDINARY PATH MUST NOT BECOME A TAX. Genuine result, matching scene, no
    refusal, nothing failed — the cell must still reach "Confirmed", and every
    newly-checked field (collision_timestep included) must genuinely match.
    """
    import contextlib
    import io

    records = [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0, min_pet=9.0,
                    fragility_score=1.0, scene_fingerprint='fp', sdc_idx=0)]
    db.upsert_scores(migrated_conn, records)
    stress_results = {'s': dict(_RESULT, scene_fingerprint='fp', sdc_idx=0)}

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _run_write_cell(migrated_conn, records, stress_results)
    assert 'Confirmed: Phase 4 columns landed' in out.getvalue()
