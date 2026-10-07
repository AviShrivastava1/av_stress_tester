"""
test_run_identity_v2_db.py — the writer and the exporter derive the same run id.

update_stress_results stamps scenario_scores.stress_run_id from the result's search provenance;
export_perturbed_path derives the id again from the provenance it is handed and publishes
geometry only if that id is still the stored one. The two must agree by CONSTRUCTION, including
when the exporter is handed the provenance as it comes back from the JSONB column (floats such
as 0.05, 12.0 and 1e-05 must survive the round trip unchanged), and including when a value was
non-finite (stored as null with a record of what it was).

A row written before this change carries the previous identity, which does not include the
provenance. The exporter refuses it with a message that names that identity and says what to do.

Needs a DISPOSABLE Postgres/PostGIS database and skips unless AV_CLAIMS_DB=1, the same requirement as
every other DB-gated file. It drops and recreates the project's tables.

Run:
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_run_identity_v2_db.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_run_identity_v2 import pre_v2_id

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

SID = 'run-identity'
DELTA = [0.1, 0.0, 0.0, 0.0]
PROVENANCE = {
    'replay_model_version': 'kinematic-replay-v1',
    'collision_geometry_version': 'oriented-box-float64-v1',
    'delta_parameterization': 'bicycle',
    'heading_speed_floor': 0.5,
    'heading_transition_width': 0.05,
    'a_max': 12.0,
    'de_tol': 1e-05,
    'de_popsize': 15,
    'de_seed': 7,
    'bounds': [[-3.0, 3.0], [-0.2, 0.2]],
    'target_has_interior_gap': False,
    'max_speed_step': None,
}


@pytest.fixture
def world():
    from src.scoring import db
    from src.scoring.export_geometry import init_geometry_schema

    conn = db.get_connection()
    with conn.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, scenario_scores CASCADE')
    conn.commit()
    db.init_schema(conn)
    init_geometry_schema(conn)

    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 0] = np.arange(10, dtype=np.float32)
    states[1, :, 0] = np.arange(10, dtype=np.float32) + 4.0
    states[:, :, 5:7] = [4.5, 2.0]
    validity = np.ones((2, 10), dtype=bool)
    fingerprint = db.compute_scene_fingerprint(states, validity, np.array([1, 1]))
    db.upsert_scores(conn, [{'scenario_id': SID, 'shard': 'synthetic', 'n_agents': 2, 'min_ttc': 1.0, 'min_pet': 1.0,
                             'fragility_score': 1.0, 'scene_fingerprint': fingerprint, 'sdc_idx': 0}])
    yield conn, states, validity, fingerprint
    conn.rollback()
    conn.close()


def _store(conn, fingerprint, provenance, delta=DELTA, method='de'):
    from src.scoring import db
    result = {'status': 'ok', 'outcome': 'collision_found', 'collision': True, 'min_perturbation': 0.1, 'delta': delta,
              'collision_timestep': 5, 'target_idx': 1, 'method': method, 'scene_fingerprint': fingerprint, 'sdc_idx': 0}
    if provenance is not None:
        result['search_provenance'] = provenance
    report = db.update_stress_results(conn, {SID: result})
    assert report.failures == [], report.failures
    return db.fetch_scenario(conn, SID)


def _export(conn, states, validity, fingerprint, provenance, delta=DELTA, method='de'):
    from src.scoring.export_geometry import export_perturbed_path
    extra = {} if provenance is None else {'search_provenance': provenance}      # no provenance: the call as it was before
    return export_perturbed_path(conn, SID, states, validity, 1, delta=delta, method=method,
                                 scene_fingerprint=fingerprint, **extra)


def _path_run_id(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT stress_run_id FROM perturbed_paths WHERE scenario_id = %s', (SID,))
        row = cur.fetchone()
    return row[0] if row else None


# ── the writer and the exporter agree, through a real JSONB round trip ──────────────


@requires_db
def test_the_stored_id_is_the_one_derived_from_the_in_memory_and_from_the_read_back_provenance(world):
    from src.scoring.db import compute_stress_run_id
    conn, states, validity, fingerprint = world

    row = _store(conn, fingerprint, PROVENANCE)
    stored_id, read_back = row['stress_run_id'], row['search_provenance']

    # The assumption the equality below rests on: JSONB hands every value back as it was written.
    assert read_back == PROVENANCE
    assert {k: type(v) for k, v in read_back.items()} == {k: type(v) for k, v in PROVENANCE.items()}
    assert repr(read_back['a_max']) == '12.0' and repr(read_back['de_tol']) == '1e-05' and repr(read_back['heading_transition_width']) == '0.05'

    assert stored_id != pre_v2_id(SID, 1, DELTA, 'de')
    assert stored_id == compute_stress_run_id(SID, 1, DELTA, 'de', PROVENANCE)
    assert stored_id == compute_stress_run_id(SID, 1, DELTA, 'de', read_back)
    # and the exporter, handed either form, publishes geometry stamped with that same id
    assert _export(conn, states, validity, fingerprint, PROVENANCE) is True
    assert _path_run_id(conn) == stored_id
    assert _export(conn, states, validity, fingerprint, read_back) is True
    assert _path_run_id(conn) == stored_id


@requires_db
def test_a_non_finite_provenance_value_gives_the_same_id_on_both_sides(world):
    from src.scoring.db import compute_stress_run_id
    conn, states, validity, fingerprint = world
    raw = {**PROVENANCE, 'baseline_replay_error': float('inf'), 'held_speed_excess': float('nan')}

    row = _store(conn, fingerprint, raw)
    read_back = row['search_provenance']

    assert read_back['baseline_replay_error'] is None and read_back['held_speed_excess'] is None
    assert read_back['nonfinite_fields'] == {'baseline_replay_error': 'inf', 'held_speed_excess': 'nan'}
    assert row['stress_run_id'] == compute_stress_run_id(SID, 1, DELTA, 'de', raw) == compute_stress_run_id(SID, 1, DELTA, 'de', read_back)
    assert _export(conn, states, validity, fingerprint, raw) is True
    assert _export(conn, states, validity, fingerprint, read_back) is True
    assert _path_run_id(conn) == row['stress_run_id']


@requires_db
def test_numpy_scalars_in_the_provenance_give_the_same_stored_id_as_python_values(world):
    from src.scoring.db import compute_stress_run_id
    conn, states, validity, fingerprint = world
    numpy_form = {**PROVENANCE, 'a_max': np.float64(12.0), 'de_seed': np.int64(7), 'target_has_interior_gap': np.bool_(False)}

    row = _store(conn, fingerprint, numpy_form)

    assert row['stress_run_id'] == compute_stress_run_id(SID, 1, DELTA, 'de', PROVENANCE)
    assert _export(conn, states, validity, fingerprint, PROVENANCE) is True


@requires_db
def test_a_result_without_provenance_is_stamped_with_the_previous_id(world):
    # Passes on main on purpose: the compatibility path.
    conn, states, validity, fingerprint = world

    row = _store(conn, fingerprint, None)

    assert row['stress_run_id'] == pre_v2_id(SID, 1, DELTA, 'de')


# ── the contract: the exporter must be told the provenance ──────────────────────────


@requires_db
def test_an_export_that_omits_the_provenance_of_a_provenance_stamped_result_is_refused(world):
    from src.scoring.export_geometry import StaleExportError
    conn, states, validity, fingerprint = world
    _store(conn, fingerprint, PROVENANCE)

    with pytest.raises(StaleExportError):
        _export(conn, states, validity, fingerprint, None)

    assert _path_run_id(conn) is None


@requires_db
def test_an_export_from_a_different_delta_is_refused_without_claiming_an_old_identity(world):
    from src.scoring.export_geometry import StaleExportError
    conn, states, validity, fingerprint = world
    _store(conn, fingerprint, PROVENANCE)

    with pytest.raises(StaleExportError) as refused:
        _export(conn, states, validity, fingerprint, PROVENANCE, delta=[0.2, 0.0, 0.0, 0.0])

    assert 'pre-v2' not in str(refused.value)
    assert _path_run_id(conn) is None


# ── rows written before this change ─────────────────────────────────────────────────


def _make_it_a_pre_v2_row(conn):
    with conn.cursor() as cur:
        cur.execute('UPDATE scenario_scores SET stress_run_id = %s WHERE scenario_id = %s',
                    (pre_v2_id(SID, 1, DELTA, 'de'), SID))
    conn.commit()


@requires_db
def test_a_row_with_the_previous_identity_is_refused_with_a_message_that_says_what_to_do(world):
    from src.scoring.export_geometry import StaleExportError
    conn, states, validity, fingerprint = world
    _store(conn, fingerprint, PROVENANCE)
    _make_it_a_pre_v2_row(conn)

    with pytest.raises(StaleExportError) as refused:
        _export(conn, states, validity, fingerprint, PROVENANCE)

    message = str(refused.value)
    assert 'pre-v2 run identity' in message and 'Rerun Pass 2' in message
    assert refused.value.current_run_id == pre_v2_id(SID, 1, DELTA, 'de')
    assert _path_run_id(conn) is None


@requires_db
def test_the_provenance_free_call_still_exports_a_row_with_the_previous_identity(world):
    # Passes on main on purpose. A caller that supplies no provenance derives the previous
    # identity, so a hand-built export against a row that carries it still works. The strict
    # refusal above applies when the caller says the result carries provenance.
    conn, states, validity, fingerprint = world
    _store(conn, fingerprint, None)

    assert _export(conn, states, validity, fingerprint, None) is True
    assert _path_run_id(conn) == pre_v2_id(SID, 1, DELTA, 'de')
