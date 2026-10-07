"""
test_json_safe_numpy_db.py — a result whose search_provenance holds NumPy scalars is stored.

Before the conversion at the JSONB boundary, update_stress_results reported a failure for
any result carrying a NumPy scalar in search_provenance (json.dumps raised TypeError inside
the per-scenario savepoint) and stored nothing. Expected values are derived here from
float(np.float32(x)) and the like; nothing is typed in from a run.

Needs a DISPOSABLE Postgres/PostGIS database and skips unless AV_CLAIMS_DB=1, the same
requirement as every other DB-gated file. It drops and recreates the project's tables.

Run:
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_json_safe_numpy_db.py -q
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


@pytest.fixture
def conn():
    from src.scoring import db
    c = db.get_connection()
    with c.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, scenario_scores CASCADE')
    c.commit()
    db.init_schema(c)
    yield c
    c.rollback()
    c.close()


@requires_db
def test_a_result_with_numpy_scalars_in_its_provenance_is_stored_and_reads_back_equal(conn):
    from src.scoring import db

    db.upsert_scores(conn, [{'scenario_id': 'np-prov', 'shard': 'synthetic', 'n_agents': 2, 'min_ttc': 1.0,
                             'min_pet': 1.0, 'fragility_score': 1.0, 'sdc_idx': 0}])
    provenance = {
        'heading_speed_floor': np.float32(0.5),
        'a_max': np.float64(12.0),
        'de_popsize': np.int64(15),
        'bounds': [[np.float32(-2.0), np.float32(2.0)]],
        'target_has_interior_gap': np.bool_(True),
        'baseline_replay_error': np.float32('inf'),
    }
    report = db.update_stress_results(conn, {'np-prov': {
        'status': 'ok', 'outcome': 'collision_found', 'collision': True, 'min_perturbation': 0.5,
        'delta': [-1.0, 0.0, 0.0, 0.0], 'collision_timestep': 9, 'target_idx': 1, 'method': 'de',
        'search_provenance': provenance, 'sdc_idx': 0,
    }})

    assert report.failures == [], report.failures
    assert int(report) == 1
    stored = db.fetch_scenario(conn, 'np-prov')['search_provenance']
    assert stored['heading_speed_floor'] == float(np.float32(0.5))
    assert stored['a_max'] == float(np.float64(12.0))
    assert stored['de_popsize'] == 15 and isinstance(stored['de_popsize'], int)
    assert stored['bounds'] == [[float(np.float32(-2.0)), float(np.float32(2.0))]]
    assert stored['target_has_interior_gap'] is True
    assert stored['baseline_replay_error'] is None
    assert stored['nonfinite_fields'] == {'baseline_replay_error': 'inf'}


@requires_db
def test_a_float32_that_is_not_exactly_representable_is_stored_as_its_widened_value(conn):
    from src.scoring import db

    db.upsert_scores(conn, [{'scenario_id': 'np-wide', 'shard': 'synthetic', 'n_agents': 2, 'min_ttc': 1.0,
                             'min_pet': 1.0, 'fragility_score': 1.0, 'sdc_idx': 0}])
    value = 0.1                                          # no float32 holds this exactly
    report = db.update_stress_results(conn, {'np-wide': {
        'status': 'ok', 'outcome': 'collision_found', 'collision': True, 'min_perturbation': 0.5,
        'delta': [-1.0, 0.0, 0.0, 0.0], 'collision_timestep': 9, 'target_idx': 1, 'method': 'de',
        'search_provenance': {'width': np.float32(value)}, 'sdc_idx': 0,
    }})

    assert report.failures == []
    stored = db.fetch_scenario(conn, 'np-wide')['search_provenance']['width']
    assert stored == float(np.float32(value)) and stored != value
