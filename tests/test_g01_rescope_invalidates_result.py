"""
test_g01_rescope_invalidates_result.py — fix G01 (independent review).

upsert_scores's ON CONFLICT DO UPDATE SET never touched the Phase 4 result-group
columns (collision_timestep, stress_outcome, min_perturbation, delta, stress_run_id,
stress_method, search_provenance, target_idx, challengers_total,
challengers_searched, stress_tested_at) or the attempt-group columns
(last_attempt_outcome, last_attempt_diagnostics, stress_attempted_at). When Pass 1
rescopes a scenario_id to a genuinely different scene or a different SDC, an
already-persisted result from the OLD identity simply survived, untouched, with
nothing flagging it as stale — served as-is by every consumer that reads
scenario_scores directly (the list/detail endpoints, /stats, get_perturbed's own
score_row).

update_stress_results's own F02 guard does not cover this: it refuses to ATTACH a
NEW result to a row whose identity has since moved, which protects a write still in
flight. It has nothing to say about a write that already landed and is never
revisited by a later Phase 2 call.

The fix (db.py's upsert_scores, _RESCOPED_UPDATE) makes upsert_scores's own
ON CONFLICT statement atomically null the whole Phase 4 state whenever this call's
identity genuinely disagrees with what's stored — checked independently for
scene_fingerprint and sdc_idx (a fingerprint alone cannot see the SDC axis; see
db.py's own comment), using the SAME NULL-means-not-recorded carve-out F02 already
established: only refuse to keep the old state when BOTH sides are non-NULL and
disagree. UpsertReport.rescoped reports which scenario_ids this fired for, and
what they moved from/to.

Needs a DISPOSABLE Postgres database and skips unless AV_CLAIMS_DB=1, same
requirement as every other DB-gated file in this project.

Run:
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_g01_rescope_invalidates_result.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

_RESULT = {'status': 'ok', 'outcome': 'collision_found', 'collision': True,
           'min_perturbation': 0.5, 'delta': [-1.0, 0.0, 0.0, 0.0],
           'collision_timestep': 9, 'target_idx': 1, 'method': 'de'}

_RESULT_GROUP = ('collision_timestep', 'stress_outcome', 'min_perturbation',
                 'delta', 'stress_run_id', 'stress_method', 'search_provenance',
                 'target_idx', 'challengers_total', 'challengers_searched',
                 'stress_tested_at')
_ATTEMPT_GROUP = ('last_attempt_outcome', 'last_attempt_diagnostics',
                  'stress_attempted_at')

# _RESULT carries no challengers_total/challengers_searched/search_provenance key,
# so update_stress_results writes NULL for all three even on a genuine success —
# those columns have nothing to discriminate a rescope by in THIS fixture. Used
# only for the "did the fixture actually persist a result" sanity check below; the
# "did the rescope null EVERYTHING in the result group" assertions still use the
# full _RESULT_GROUP, which is a valid (if partly redundant) check either way.
_RESULT_GROUP_THIS_FIXTURE_POPULATES = tuple(
    c for c in _RESULT_GROUP
    if c not in ('challengers_total', 'challengers_searched', 'search_provenance')
)


@pytest.fixture
def dconn():
    from src.scoring import db
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, '
                    'scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    yield connection
    connection.rollback()
    connection.close()


def _score_and_stress(conn, sid, fp, sdc_idx):
    from src.scoring import db
    db.upsert_scores(conn, [dict(scenario_id=sid, shard='x', n_agents=2, min_ttc=9.0,
                                 min_pet=9.0, fragility_score=1.0,
                                 scene_fingerprint=fp, sdc_idx=sdc_idx)])
    db.update_stress_results(conn, {sid: dict(
        _RESULT, scene_fingerprint=fp, sdc_idx=sdc_idx)})


def _all_null(row, columns):
    return all(row[c] is None for c in columns)


def _none_null(row, columns):
    return all(row[c] is not None for c in columns)


@requires_db
def test_G01_a_genuine_scene_rescope_nulls_the_stale_phase4_result(dconn):
    """
    THE FINDING'S OWN REPRO. Score A, run real Phase 2, persist a collision.
    Rescope the same scenario_id to a genuinely different scene (B, 10m away). Pre-
    fix, upsert_scores's ON CONFLICT clause never mentioned the result-group columns
    at all, so A's collision_timestep/stress_outcome/min_perturbation/delta/
    stress_run_id simply kept being served under B's identity.
    """
    from src.scoring import db

    _score_and_stress(dconn, 's', 'fp_A', 0)
    before = db.fetch_scenario(dconn, 's')
    assert _none_null(before, _RESULT_GROUP_THIS_FIXTURE_POPULATES), (
        'fixture regressed: nothing persisted'
    )

    report = db.upsert_scores(dconn, [dict(scenario_id='s', shard='x', n_agents=2,
                                           min_ttc=9.0, min_pet=9.0,
                                           fragility_score=1.0,
                                           scene_fingerprint='fp_B', sdc_idx=0)])

    after = db.fetch_scenario(dconn, 's')
    assert after['scene_fingerprint'] == 'fp_B', 'the rescope itself must still land'
    assert _all_null(after, _RESULT_GROUP), (
        f"A's Phase 4 result survived a genuine rescope to a different scene: {after}"
    )
    assert _all_null(after, _ATTEMPT_GROUP), (
        f"the attempt group survived a genuine rescope too: {after}"
    )
    assert report.rescoped == [{
        'scenario_id': 's',
        'old_scene_fingerprint': 'fp_A', 'new_scene_fingerprint': 'fp_B',
        'old_sdc_idx': 0, 'new_sdc_idx': 0,
    }], report.rescoped


@requires_db
def test_G01_an_sdc_only_rescope_also_nulls_the_result(dconn):
    """
    THE AXIS A FINGERPRINT CANNOT SEE. compute_scene_fingerprint hashes
    states/validity/types only (see F02's own comment in db.py) — sdc_track_index is
    a separate WOMD field two parses can disagree on while every array still hashes
    identically. scene_fingerprint stays THE SAME here; only sdc_idx moves.
    """
    from src.scoring import db

    _score_and_stress(dconn, 's', 'fp_stable', 0)
    before = db.fetch_scenario(dconn, 's')
    assert _none_null(before, _RESULT_GROUP_THIS_FIXTURE_POPULATES), (
        'fixture regressed: nothing persisted'
    )

    report = db.upsert_scores(dconn, [dict(scenario_id='s', shard='x', n_agents=2,
                                           min_ttc=9.0, min_pet=9.0,
                                           fragility_score=1.0,
                                           scene_fingerprint='fp_stable', sdc_idx=3)])

    after = db.fetch_scenario(dconn, 's')
    assert after['scene_fingerprint'] == 'fp_stable', 'fixture check: fp must not move'
    assert after['sdc_idx'] == 3, 'the rescope itself must still land'
    assert _all_null(after, _RESULT_GROUP), (
        f"an sdc-only rescope (scene_fingerprint unchanged) left the stale result "
        f"in place: {after}"
    )
    assert len(report.rescoped) == 1 and report.rescoped[0]['scenario_id'] == 's'
    assert report.rescoped[0]['old_scene_fingerprint'] == 'fp_stable'
    assert report.rescoped[0]['old_sdc_idx'] == 0
    assert report.rescoped[0]['new_sdc_idx'] == 3


@requires_db
def test_G01_a_first_time_fingerprint_is_not_a_rescope(dconn):
    """
    THE A12/F02 CARVE-OUT, PRESERVED. A legacy row (no scene_fingerprint/sdc_idx
    recorded at all) getting its FIRST-EVER identity must not be read as "changed
    from something" — there was nothing to change from. Refusing here would silently
    reopen exactly the carve-out A12 established for the export path.
    """
    from src.scoring import db

    db.upsert_scores(dconn, [dict(scenario_id='s', shard='x', n_agents=2,
                                  min_ttc=9.0, min_pet=9.0, fragility_score=1.0)])
    db.update_stress_results(dconn, {'s': dict(_RESULT)})
    before = db.fetch_scenario(dconn, 's')
    assert before['scene_fingerprint'] is None, 'fixture regressed'
    assert _none_null(before, _RESULT_GROUP_THIS_FIXTURE_POPULATES), (
        'fixture regressed: nothing persisted'
    )

    report = db.upsert_scores(dconn, [dict(scenario_id='s', shard='x', n_agents=2,
                                           min_ttc=9.0, min_pet=9.0,
                                           fragility_score=1.0,
                                           scene_fingerprint='fp_first', sdc_idx=0)])

    after = db.fetch_scenario(dconn, 's')
    assert after['scene_fingerprint'] == 'fp_first'
    assert _none_null(after, _RESULT_GROUP_THIS_FIXTURE_POPULATES), (
        f"a first-ever fingerprint was treated as a rescope and wiped a real "
        f"result: {after}"
    )
    assert report.rescoped == [], report.rescoped


@requires_db
def test_G01_an_ordinary_rescore_with_the_same_identity_preserves_the_result(dconn):
    """
    THE ORDINARY PATH MUST NOT BECOME A TAX. Re-scoring the SAME scene/SDC (the
    common case — a shard re-scored after a code fix, per this file's own Idempotent
    docstring) must not touch the Phase 4 result at all.
    """
    from src.scoring import db

    _score_and_stress(dconn, 's', 'fp_A', 0)
    before = db.fetch_scenario(dconn, 's')

    report = db.upsert_scores(dconn, [dict(scenario_id='s', shard='x', n_agents=2,
                                           min_ttc=8.0, min_pet=8.0,
                                           fragility_score=2.0,
                                           scene_fingerprint='fp_A', sdc_idx=0)])

    after = db.fetch_scenario(dconn, 's')
    assert after['fragility_score'] == 2.0, 'the ordinary Pass 1 columns must update'
    for col in _RESULT_GROUP:
        assert after[col] == before[col], (
            f'{col} changed on a same-identity re-score: {before[col]!r} -> '
            f'{after[col]!r}'
        )
    assert report.rescoped == [], report.rescoped


@requires_db
def test_G01_upsert_scores_still_returns_a_plain_int_count(dconn):
    """
    UpsertReport IS AN int SUBCLASS, not a new return shape. Every existing caller
    (the whole rest of the test suite, the notebook's n1/n2 idempotency check)
    compares or assigns this as a plain count and must keep working.
    """
    from src.scoring import db

    report = db.upsert_scores(dconn, [
        dict(scenario_id='a', shard='x', n_agents=2, min_ttc=9.0, min_pet=9.0,
             fragility_score=1.0),
        dict(scenario_id='b', shard='x', n_agents=2, min_ttc=9.0, min_pet=9.0,
             fragility_score=1.0),
    ])
    assert report == 2
    assert int(report) == 2
    assert isinstance(report, int)
