"""
test_f02_scene_identity_guard.py — fix F02 (independent review, post-Batch-11).

Batch 11 (audit A02, A12, A05) gave scenario_scores a scene_fingerprint, but only
taught TWO consumers to read it before trusting a write: export_perturbed_path (A12)
and the perturbed_paths DELETE inside update_stress_results itself (A05, "fixing A05
by reopening A02 one table over"). The UPDATE nine lines above that DELETE, in the
SAME function, writing min_perturbation/delta/collision_timestep/stress_outcome onto
the row, never got the check — its WHERE clause is `scenario_id = %s` alone. A stale
Pass 2 search, persisted after Pass 1 re-ran under a changed scene for the same
scenario_id, silently relabelled its result onto a row it no longer describes.

This file holds the write-path guard's properties, the same relationship
test_batch11_contract.py has to test_audit3_regressions.py's A02/A05/A12 repros:
reconstructed fixtures, not the audit's own bundle (none exists for this finding
either — see test_audit3_regressions.py's own header for that provenance note, which
applies here unchanged).

Needs a DISPOSABLE Postgres/PostGIS database and skips unless AV_CLAIMS_DB=1, same
requirement and same reason as test_audit3_regressions.py's A02/A05/A12 section: this
is exactly the class of defect nothing outside a real database exercises.

Run:
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_f02_scene_identity_guard.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

SID = 'f02_scene_identity'

# Same shape as test_audit3_regressions.py's own _RESULT — a completed, successful
# search — extended with the two fields _stress_one now captures at entry (fix F02).
# Tests that want the OLD, pre-fix shape (no scene_fingerprint/sdc_idx keys) build
# their own dict from scratch rather than starting here and deleting keys, so the
# fixture reads honestly as "what a caller that predates this fix sends".
_RESULT = {'status': 'ok', 'outcome': 'collision_found', 'collision': True,
           'min_perturbation': 0.5, 'delta': [-1.0, 0.0, 0.0, 0.0],
           'collision_timestep': 9, 'target_idx': 1, 'method': 'de'}


def _scene_a():
    """A vehicle SDC at the origin and a pedestrian walking in — 'scene A'."""
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 5:7] = [0.6, 0.6]
    states[1, :, 0] = 5.0 - 2.0 * np.arange(10) * 0.1
    states[1, :, 2] = -2.0
    states[1, :, 4] = np.pi
    return states, np.ones((2, 10), dtype=bool), np.array([1, 2])


def _scene_b():
    """
    A DIFFERENT scene under the same scenario_id — 'scene B'. Not a shift of scene A
    (test_audit3_regressions.py's _pedestrian_scene(shift=...) pattern): a genuinely
    different pedestrian position and speed, so scene A's own captured fingerprint
    could not coincidentally match it.
    """
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 5:7] = [0.6, 0.6]
    states[1, :, 1] = 6.0 + np.arange(10) * 0.1
    states[1, :, 3] = 1.0
    states[1, :, 4] = -np.pi / 2
    return states, np.ones((2, 10), dtype=bool), np.array([1, 2])


def _fp(states, validity, types):
    from src.scoring.db import compute_scene_fingerprint
    return compute_scene_fingerprint(states, validity, types)


@pytest.fixture
def sconn():
    """Same shape as test_audit3_regressions.py's own sconn fixture."""
    from src.scoring import db
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


def _seed_pass1(conn, states, validity, types, sdc_idx=0, sid=SID):
    """Pass 1 only — through the real write path, upsert_scores."""
    from src.scoring import db
    db.upsert_scores(conn, [dict(
        scenario_id=sid, shard='synthetic', n_agents=int(states.shape[0]),
        min_ttc=9.0, min_pet=9.0, fragility_score=1.0,
        scene_fingerprint=_fp(states, validity, types), sdc_idx=sdc_idx,
    )])


# ── the reproduction, matching the finding's own repro exactly ──────────────────

@requires_db
def test_F02_a_stale_search_cannot_overwrite_a_result_for_a_changed_scene(sconn):
    """
    THE FINDING'S OWN REPRODUCTION. Pass 1 seeds scene A. A Pass 2 search runs
    against scene A and captures its identity (this is what _stress_one now does at
    entry). Before that result is persisted, Pass 1 runs AGAIN — reparsed shard,
    moved challenger — under scene B, for the SAME scenario_id. Persisting the scene
    A search's result must not attach a collision_found outcome, or any of
    min_perturbation/delta/collision_timestep/stress_outcome, to a row that now
    describes scene B.
    """
    from src.scoring import db

    states_a, validity_a, types_a = _scene_a()
    states_b, validity_b, types_b = _scene_b()
    fp_a = _fp(states_a, validity_a, types_a)
    fp_b = _fp(states_b, validity_b, types_b)
    assert fp_a != fp_b, 'fixture regressed: scenes A and B must not collide'

    _seed_pass1(sconn, states_a, validity_a, types_a, sdc_idx=0)
    # ...Pass 1 races ahead, same scenario_id, now describing scene B.
    _seed_pass1(sconn, states_b, validity_b, types_b, sdc_idx=0)

    before = db.fetch_scenario(sconn, SID)
    assert before['scene_fingerprint'] == fp_b, 'fixture regressed: Pass 1 must win the race'
    assert before['min_perturbation'] is None, 'fixture regressed: no prior Pass 2 result'

    stale_result = dict(_RESULT, scene_fingerprint=fp_a, sdc_idx=0)
    report = db.update_stress_results(sconn, {SID: stale_result})
    assert not report.failures, f'unexpected failure: {report.failures}'

    after = db.fetch_scenario(sconn, SID)
    assert after['min_perturbation'] is None, (
        f"a stale search's result was attached: min_perturbation="
        f"{after['min_perturbation']!r}"
    )
    assert after['delta'] is None
    assert after['collision_timestep'] is None
    assert after['stress_outcome'] is None
    assert after['stress_tested_at'] is None

    # The latest-pass group still reports the truth: a search DID run and DID find
    # a collision. What did not happen is that finding being attached to the row.
    assert after['last_attempt_outcome'] == 'collision_found'
    assert after['stress_attempted_at'] is not None
    diagnostics = after['last_attempt_diagnostics']
    assert diagnostics is not None, 'the refusal reason was not recorded'
    assert diagnostics['scene_mismatch'] is True
    assert diagnostics['captured_scene_fingerprint'] == fp_a
    assert diagnostics['stored_scene_fingerprint'] == fp_b


@requires_db
def test_F02_sdc_idx_alone_can_trigger_the_guard(sconn):
    """
    ISOLATES sdc_idx FROM scene_fingerprint. Identical states/validity/types (so the
    array hash matches exactly) with only sdc_idx differing between the captured
    value and the stored one — the exact gap compute_scene_fingerprint's own hash
    cannot see, since sdc_track_index is a separate WOMD field. Confirms the guard
    checks sdc_idx independently rather than relying on the fingerprint alone.
    """
    from src.scoring import db

    states, validity, types = _scene_a()
    fp = _fp(states, validity, types)

    _seed_pass1(sconn, states, validity, types, sdc_idx=0)
    stale_result = dict(_RESULT, scene_fingerprint=fp, sdc_idx=1)
    db.update_stress_results(sconn, {SID: stale_result})

    after = db.fetch_scenario(sconn, SID)
    assert after['min_perturbation'] is None, (
        'an sdc_idx mismatch, with the fingerprint matching exactly, still wrote '
        'the result — the array hash alone is not covering this'
    )
    diagnostics = after['last_attempt_diagnostics']
    assert diagnostics['scene_mismatch'] is True
    assert diagnostics['captured_sdc_idx'] == 1
    assert diagnostics['stored_sdc_idx'] == 0


# ── the hole this fix must not open ──────────────────────────────────────────────

@requires_db
def test_F02_a_genuinely_matching_scene_still_updates(sconn):
    """
    THE MIRROR OF THE ABOVE, and the one a fix this narrow could get wrong in the
    other direction: refusing EVERYTHING would also stop the finding from mattering,
    by making update_stress_results useless. A captured identity that genuinely
    matches the stored one must write normally — the ordinary case, exercised
    thousands of times by the rest of this project's own test suite, still has to
    hold here explicitly.
    """
    from src.scoring import db

    states, validity, types = _scene_a()
    fp = _fp(states, validity, types)

    _seed_pass1(sconn, states, validity, types, sdc_idx=0)
    matching_result = dict(_RESULT, scene_fingerprint=fp, sdc_idx=0)
    report = db.update_stress_results(sconn, {SID: matching_result})
    assert report == 1, f'expected exactly one row updated, got {int(report)}'

    after = db.fetch_scenario(sconn, SID)
    assert after['min_perturbation'] == pytest.approx(0.5)
    assert after['delta'] == [-1.0, 0.0, 0.0, 0.0]
    assert after['collision_timestep'] == 9
    assert after['stress_outcome'] == 'collision_found'
    assert after['stress_tested_at'] is not None
    diagnostics = after['last_attempt_diagnostics']
    assert not diagnostics or 'scene_mismatch' not in diagnostics


# ── the DELETE-gating interaction ────────────────────────────────────────────────

@requires_db
def test_F02_a_refused_write_does_not_delete_the_preserved_results_geometry(sconn):
    """
    THE SECOND-ORDER HOLE THIS FIX COULD OPEN, if the guard above were not ALSO
    threaded into the DELETE's own trigger condition. A scene-mismatched attempt
    still computes a fresh stress_run_id from its own (rejected) delta/method/
    target — if the DELETE ran anyway, it would compare the PRESERVED result's
    geometry against a run_id that statement just decided not to trust, and delete
    geometry that still correctly pairs with the untouched, preserved result. A05's
    own test (test_A05_a_genuinely_new_result_still_invalidates_geometry) checks the
    mirror of this for an ordinary accepted result; this is the same shape for a
    refused one.
    """
    from src.scoring import db
    from src.scoring.export_geometry import export_scenario_agents, export_perturbed_path

    states_a, validity_a, types_a = _scene_a()
    states_b, validity_b, types_b = _scene_b()
    fp_a = _fp(states_a, validity_a, types_a)

    _seed_pass1(sconn, states_a, validity_a, types_a, sdc_idx=0)
    good_result = dict(_RESULT, scene_fingerprint=fp_a, sdc_idx=0)
    db.update_stress_results(sconn, {SID: good_result})

    export_scenario_agents(sconn, SID, states_a, validity_a, types_a, 0)
    export_perturbed_path(sconn, SID, states_a, validity_a, 1,
                          delta=good_result['delta'], method=good_result['method'],
                          scene_fingerprint=fp_a)
    with sconn.cursor() as cur:
        cur.execute('SELECT count(*) FROM perturbed_paths WHERE scenario_id = %s', (SID,))
        before = cur.fetchone()[0]
    assert before == 1, 'fixture regressed: geometry was not exported'

    # Pass 1 races ahead to scene B, then a STALE search (against scene A, the
    # scene the exported geometry actually describes) tries to persist.
    _seed_pass1(sconn, states_b, validity_b, types_b, sdc_idx=0)
    stale_result = dict(_RESULT, delta=[-9.0, 0.0, 0.0, 0.0],
                        scene_fingerprint=fp_a, sdc_idx=0)
    db.update_stress_results(sconn, {SID: stale_result})

    with sconn.cursor() as cur:
        cur.execute('SELECT count(*) FROM perturbed_paths WHERE scenario_id = %s', (SID,))
        after = cur.fetchone()[0]
    assert after == before, (
        f'a refused write deleted the preserved result\'s geometry anyway: '
        f'{before} -> {after} rows'
    )


# ── NULL policy, both directions ─────────────────────────────────────────────────

@requires_db
def test_F02_a_captured_identity_of_null_still_updates(sconn):
    """
    THE POLICY THAT KEEPS THE REST OF THIS PROJECT'S TEST SUITE GREEN. 55 existing
    call sites build synthetic result dicts with no scene_fingerprint/sdc_idx key at
    all — the shape every caller had before this fix, and the shape any caller that
    does not capture scene identity will keep sending. NULL on the CAPTURED side
    must be read as "not recorded", not as a mismatch, or this fix breaks every one
    of them for a check they were never in a position to satisfy.
    """
    from src.scoring import db

    states, validity, types = _scene_a()
    _seed_pass1(sconn, states, validity, types, sdc_idx=0)

    legacy_shaped_result = dict(_RESULT)   # no scene_fingerprint, no sdc_idx key
    assert 'scene_fingerprint' not in legacy_shaped_result
    report = db.update_stress_results(sconn, {SID: legacy_shaped_result})
    assert report == 1

    after = db.fetch_scenario(sconn, SID)
    assert after['min_perturbation'] == pytest.approx(0.5)
    assert after['stress_outcome'] == 'collision_found'


@requires_db
def test_F02_a_stored_identity_of_null_still_updates(sconn):
    """
    THE MIRROR: a row Pass 1 never fingerprinted (or one written before this fix's
    columns existed) must not become permanently unwritable. Same carve-out A12
    established for the export path, extended here rather than re-litigated.
    """
    from src.scoring import db

    # Pass 1 WITHOUT scene identity — the pre-Batch-11 shape.
    db.upsert_scores(sconn, [dict(scenario_id=SID, shard='synthetic', n_agents=2,
                                  min_ttc=9.0, min_pet=9.0, fragility_score=1.0)])
    before = db.fetch_scenario(sconn, SID)
    assert before['scene_fingerprint'] is None, 'fixture regressed'
    assert before['sdc_idx'] is None, 'fixture regressed'

    states, validity, types = _scene_a()
    fp = _fp(states, validity, types)
    captured_result = dict(_RESULT, scene_fingerprint=fp, sdc_idx=0)
    report = db.update_stress_results(sconn, {SID: captured_result})
    assert report == 1, (
        'a legacy row with no recorded scene identity was refused — the carve-out closed'
    )

    after = db.fetch_scenario(sconn, SID)
    assert after['min_perturbation'] == pytest.approx(0.5)
