"""
test_g04_sdc_index_identity_guard.py — fix G04 (independent review).

sdc_idx was checked for Phase 4 result writes (fix F02) but not by
export_scenario_agents or get_trajectories. F02's own comment in db.py already
established why a fingerprint cannot stand in for this: compute_scene_fingerprint
hashes states/validity/types only, so two parses can disagree about which track is
the SDC while every array still hashes identically.

TWO HALVES, BOTH TESTED HERE, mirroring exactly how scene_fingerprint already has
both (see test_batch11_contract.py's own
test_a_stale_baseline_is_not_served_after_the_cross_function_window, whose sdc-only
analogue this file's last test is):

  * WRITE-TIME (export_scenario_agents, export_geometry.py): the same FOR UPDATE
    read that already checks scene_fingerprint now also checks sdc_idx, raising the
    new SdcIndexChangedError on a genuine mismatch, with the same NULL-means-
    not-recorded carve-out F02/A12 already established.

  * READ-TIME (get_trajectories, routes.py): scenario_agents now carries the sdc_idx
    it was verified against (mirroring the existing scene_fingerprint column), and
    get_trajectories compares it against scenario_scores.sdc_idx on every read —
    which is what makes a LATER sdc-only rescope (fix G01) self-correcting without
    any extra invalidation logic, exactly as scene_fingerprint already is.

Needs a DISPOSABLE Postgres/PostGIS database and skips unless AV_CLAIMS_DB=1, same
requirement as every other DB-gated file in this project.

Run:
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_g04_sdc_index_identity_guard.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring.db import compute_scene_fingerprint

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

_RESULT = {'status': 'ok', 'outcome': 'collision_found', 'collision': True,
           'min_perturbation': 0.5, 'delta': [-1.0, 0.0, 0.0, 0.0],
           'collision_timestep': 9, 'target_idx': 1, 'method': 'de'}


def _ped(shift=0.0):
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 5:7] = [0.6, 0.6]
    states[1, :, 0] = 5.0 - 2.0 * np.arange(10) * 0.1 + shift
    states[1, :, 2] = -2.0
    states[1, :, 4] = np.pi
    return states, np.ones((2, 10), dtype=bool), np.array([1, 2])


@pytest.fixture
def bconn():
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


@requires_db
def test_G04_export_refuses_a_parsed_sdc_that_disagrees_with_the_stored_one(bconn):
    """
    THE FINDING'S OWN REPRO, write-time half. scenario_scores records sdc_idx=0;
    export_scenario_agents is handed sdc_idx=1 for the SAME arrays. Pre-fix, this
    silently wrote is_sdc computed from the WRONG index, no different from
    scene_fingerprint's own pre-A02 behaviour.
    """
    from src.scoring import db
    from src.scoring.export_geometry import export_scenario_agents, SdcIndexChangedError

    states, validity, types = _ped()
    fp = compute_scene_fingerprint(states, validity, types)
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp, sdc_idx=0)])

    with pytest.raises(SdcIndexChangedError) as excinfo:
        export_scenario_agents(bconn, 's', states, validity, types, 1)
    assert excinfo.value.stored_sdc_idx == 0
    assert excinfo.value.computed_sdc_idx == 1

    with bconn.cursor() as cur:
        cur.execute("SELECT count(*) FROM scenario_agents WHERE scenario_id = 's'")
        assert cur.fetchone()[0] == 0, (
            'nothing should have been written for a refused sdc_idx mismatch'
        )


@requires_db
def test_G04_a_scene_match_does_not_excuse_an_sdc_mismatch(bconn):
    """
    NOT FOLDED INTO scene_fingerprint's OWN CHECK. The scene here is untouched
    (same states/validity/types) — only sdc_idx disagrees — proving this is a
    genuinely independent check, not one riding on the fingerprint comparison.
    """
    from src.scoring import db
    from src.scoring.export_geometry import export_scenario_agents, SdcIndexChangedError

    states, validity, types = _ped()
    fp = compute_scene_fingerprint(states, validity, types)
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp, sdc_idx=0)])

    with pytest.raises(SdcIndexChangedError):
        export_scenario_agents(bconn, 's', states, validity, types, 1)


@requires_db
def test_G04_export_still_publishes_when_the_parsed_sdc_matches(bconn):
    """
    THE OVERCORRECTION GUARD. A fix that refused every export regardless of the
    actual sdc_idx would also make the repro above pass — for the wrong reason.
    Proves the ordinary, matching case still writes, and stamps sdc_idx=0 onto
    every agent row (agent 0 IS the SDC here).
    """
    from src.scoring import db
    from src.scoring.export_geometry import export_scenario_agents

    states, validity, types = _ped()
    fp = compute_scene_fingerprint(states, validity, types)
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp, sdc_idx=0)])

    written, skipped, stored_fp = export_scenario_agents(
        bconn, 's', states, validity, types, 0
    )
    assert written == 2 and skipped == 0

    with bconn.cursor() as cur:
        cur.execute('SELECT agent_idx, is_sdc, sdc_idx FROM scenario_agents '
                    "WHERE scenario_id = 's' ORDER BY agent_idx")
        rows = cur.fetchall()
    assert rows == [(0, True, 0), (1, False, 0)], rows


@requires_db
def test_G04_a_legacy_row_with_no_recorded_sdc_still_exports(bconn):
    """
    THE A12/F02 CARVE-OUT, EXTENDED TO THIS AXIS. A row that never recorded an
    sdc_idx (Pass 1 predates the column, or never supplied one) must not be refused
    just because a caller now supplies a real one — NULL means "not recorded", not
    "mismatch". Mirrors test_batch11_contract.py's own
    test_F05_a_legacy_scenario_still_publishes_through_the_batch_pass for the
    fingerprint axis.
    """
    from src.scoring import db
    from src.scoring.export_geometry import export_scenario_agents

    states, validity, types = _ped()
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0)])
    assert db.fetch_scenario(bconn, 's')['sdc_idx'] is None, 'fixture regressed'

    written, skipped, stored_fp = export_scenario_agents(
        bconn, 's', states, validity, types, 0
    )
    assert written == 2, 'a legacy (never-sdc-recorded) scenario was refused to export'

    with bconn.cursor() as cur:
        cur.execute('SELECT sdc_idx FROM scenario_agents '
                    "WHERE scenario_id = 's' AND agent_idx = 0")
        stored_row_sdc_idx = cur.fetchone()[0]
    assert stored_row_sdc_idx is None, (
        'a legacy row must stamp NULL, not the freshly parsed sdc_idx — storing a '
        'real value here would make get_trajectories compare it against '
        "scenario_scores.sdc_idx's own NULL and wrongly refuse this legacy row"
    )


@requires_db
def test_G04_a_stale_is_sdc_flag_is_not_served_after_an_sdc_only_rescope(bconn):
    """
    THE READ-TIME HALF, mirroring test_batch11_contract.py's own
    test_a_stale_baseline_is_not_served_after_the_cross_function_window exactly,
    but for the axis a scene_fingerprint check cannot see: the scene here NEVER
    changes, only sdc_idx does. Before this fix, get_trajectories had no sdc_idx
    predicate at all, so this exact race left the OLD is_sdc flags being served
    forever, with the scene_fingerprint predicate passing throughout and hiding it.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.scoring import db
    from src.scoring.export_geometry import export_scenario_agents

    states, validity, types = _ped()
    fp = compute_scene_fingerprint(states, validity, types)
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp, sdc_idx=0)])
    export_scenario_agents(bconn, 's', states, validity, types, 0)

    with TestClient(app) as client:
        before = client.get('/scenarios/s/trajectories').json()
    assert before['agents'], 'fixture regressed: must serve before the rescope'
    before_sdc = next(a for a in before['agents'] if a['agent_idx'] == 0)
    assert before_sdc['is_sdc'] is True, 'fixture regressed: agent 0 must be the SDC'

    # The rescope: a concurrent Pass 1 changes ONLY sdc_idx. scene_fingerprint is
    # re-supplied unchanged, so the scene-match predicate keeps passing throughout —
    # isolating the sdc_idx axis specifically.
    other = db.get_connection()
    db.upsert_scores(other, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp, sdc_idx=1)])
    other.close()
    assert db.fetch_scenario(bconn, 's')['scene_fingerprint'] == fp, (
        'fixture check: the scene predicate must still pass throughout'
    )

    # The stale rows are still there, just inert — export_scenario_agents was never
    # asked to re-run, so nothing has re-verified or overwritten them.
    with bconn.cursor() as cur:
        cur.execute("SELECT count(*) FROM scenario_agents WHERE scenario_id = 's'")
        assert cur.fetchone()[0] == 2, (
            'fixture regressed: the stale rows should still be present, just inert'
        )

    with TestClient(app) as client:
        after = client.get('/scenarios/s/trajectories').json()
    assert after['agents'] == [], (
        'geometry whose is_sdc flags were verified against the OLD sdc_idx was '
        'served as current after an sdc-only rescope — the scene_fingerprint '
        'predicate alone cannot see this axis'
    )
