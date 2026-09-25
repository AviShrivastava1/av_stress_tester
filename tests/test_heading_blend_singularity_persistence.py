"""
test_heading_blend_singularity_persistence.py — independent review, 2026-09-24.

Companion to tests/test_heading_blend_singularity_refusal.py, which stays
DB-free per its own stated scope. This file drives the REAL persistence and API
path end to end: a genuine HeadingBlendSingularityError refusal, through the
real src.scoring.batch_scorer._stress_one, through the real
src.scoring.db.update_stress_results, read back through the real /stats
endpoint -- proving OUTCOME_HEADING_BLEND_SINGULARITY is not just a value that
exists, but one every layer between the search and the dashboard actually
carries.

Also the shard-wide measurement raised alongside the original finding:
HEADING_BLEND_SINGULARITY_MARGIN's own comment in perturbation_space.py says
plainly it is "an order-of-magnitude, reasoned default, not yet validated
against real WOMD data" -- /stats' new heading_blend_singularity count is what
lets a real shard run say how often this actually fires, the same role
replay_infeasible's own count already plays for ReplayFidelityError.

Needs a DISPOSABLE Postgres database and skips unless AV_CLAIMS_DB=1, same
requirement as every other DB-gated file in this project.

Run:
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_heading_blend_singularity_persistence.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring import db

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)


def _near_sdc_repro():
    """Identical to test_heading_blend_singularity_refusal.py's own helper of
    the same name -- the finding's own exact geometry, duplicated here per this
    project's own per-file convention."""
    T, dt = 10, 0.1
    vx, vy = -0.524999976, 0.000009975
    x0, y0 = 1.6, 1.5

    states = np.zeros((2, T, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 5:7] = [1.8, 0.5]
    states[1, :, 4] = 0.0
    for t in range(T):
        states[1, t, 0] = x0 + vx * dt * t
        states[1, t, 1] = y0 + vy * dt * t
        states[1, t, 2] = vx
        states[1, t, 3] = vy
    validity = np.ones((2, T), dtype=bool)
    types = np.array([1, 3])
    return states, validity, types


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


@requires_db
def test_the_refusal_persists_with_its_structured_fields_not_a_generic_error(dconn):
    """
    THE FULL PATH: real _stress_one -> real update_stress_results. The row must
    read back as last_attempt_outcome='heading_blend_singularity', carrying the
    exception's own fields in last_attempt_diagnostics -- not
    last_attempt_outcome='error' with a stringified message, which is exactly
    what an unwrapped exception in _stress_one would have produced.
    """
    from src.scoring.batch_scorer import _stress_one

    states, validity, types = _near_sdc_repro()
    result = _stress_one(states, validity, types, 0)

    db.upsert_scores(dconn, [dict(scenario_id='s', shard='x', n_agents=2,
                                  min_ttc=9.0, min_pet=9.0, fragility_score=1.0)])
    report = db.update_stress_results(dconn, {'s': result})
    assert not report.failures, report.failures

    row = db.fetch_scenario(dconn, 's')
    assert row['last_attempt_outcome'] == 'heading_blend_singularity'
    # NOT a search that ran and found nothing -- these must stay NULL, the same
    # invariant OUTCOME_REPLAY_INFEASIBLE already has to hold.
    assert row['stress_outcome'] is None
    assert row['stress_tested_at'] is None
    assert row['stress_attempted_at'] is not None

    diagnostics = row['last_attempt_diagnostics']
    assert diagnostics['baseline_heading_blend_min_magnitude'] == pytest.approx(
        9.606e-06, rel=1e-2
    )
    assert diagnostics['margin'] == 0.5
    assert diagnostics['target_idx'] == 1


@requires_db
def test_the_refusal_is_counted_in_stats_not_silently_missing(dconn):
    """
    THE SHARD-WIDE MEASUREMENT. /stats' new heading_blend_singularity count is
    what lets a real run say how often HEADING_BLEND_SINGULARITY_MARGIN actually
    fires -- the same role replay_infeasible's own count already plays. Driven
    through the real FastAPI app, not a hand-built response.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.scoring.batch_scorer import _stress_one

    states, validity, types = _near_sdc_repro()
    result = _stress_one(states, validity, types, 0)

    db.upsert_scores(dconn, [dict(scenario_id='s', shard='x', n_agents=2,
                                  min_ttc=9.0, min_pet=9.0, fragility_score=1.0)])
    db.update_stress_results(dconn, {'s': result})

    with TestClient(app) as client:
        stats = client.get('/stats').json()

    assert stats['heading_blend_singularity'] == 1, stats
    # Must not ALSO be counted as replay_infeasible or an error -- one refusal,
    # one bucket.
    assert stats['replay_infeasible'] == 0, stats
    assert stats['stress_errors'] == 0, stats
