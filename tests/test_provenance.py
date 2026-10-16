"""
GET /scenarios/{id}/provenance: where a stored result came from.

The route serves the run id, the scene fingerprint, the SDC index, three timestamps and an allow-listed subset of the stored
`search_provenance` (src/api/provenance.py). What is pinned:

  * the allow-list: the served keys, their types, their order, that the response model carries exactly them, that nothing else in
    the stored blob is served, and that a value of the wrong type is null rather than repaired or a 500;
  * the route, on a real stored result (the real write path, _stress_one -> update_stress_results): every served value is the
    stored one to the last digit, the timestamps are UTC, an unknown scenario is a 404, a scenario with nothing stored is 200
    with nulls, a row whose provenance is NULL has a null `search`, and a key added to the blob later is not served;
  * what it reads: scenario_scores only.

The database tests need a DISPOSABLE database and skip unless AV_ROBUSTNESS_DB=1 (they drop and recreate the project's tables).

Run:
    AV_ROBUSTNESS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_provenance.py -q
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import db_pool, main as api_main, provenance as prov, routes
from src.api.models import DELTA_LABELS, LINEAR_DELTA_LABELS, ProvenanceResponse, SearchRecord
from src.scoring import db
from src.scoring.batch_scorer import _stress_one
from src.scoring.export_geometry import init_geometry_schema

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

requires_db = pytest.mark.skipif(
    os.environ.get('AV_ROBUSTNESS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

DE_KWARGS = {'popsize': 8, 'maxiter': 40, 'seed': 1}

#: A provenance as the pipeline writes it, with every served key present and one key that is not served.
FULL = {
    'replay_model_version': 'kinematic-replay-v1', 'collision_geometry_version': 'oriented-box-float64-v1',
    'delta_parameterization': 'bicycle', 'challenger_selection': 'nearest_by_min_center_distance', 'selected_stage': 'de',
    'de_popsize': 15, 'de_maxiter': 200, 'de_tol': 1e-3, 'de_seed': 1, 'de_n_iter': 31, 'de_n_eval': 3360,
    'de_candidate_source': 'de_winner', 'de_archive_covered_all_evaluations': True, 'refine_n_iters': None,
    'bounds': [[-2.0, 2.0], [-0.1, 0.1], [-1.0, 1.0], [-0.05, 0.05]],
    'max_baseline_drift': 1.5, 'max_speed_step': 2.0, 'speed_step_hold_s': 0.5, 'held_speed_excess': 0.25, 'a_max': 12.0,
    'heading_speed_floor': 0.5, 'heading_transition_width': 0.25, 'target_min_speed': 3.5,
    'frames_observed': 91, 'frames_below_heading_floor': 0, 'frames_in_heading_transition_band': 2, 'target_has_interior_gap': False,
    'baseline_replay_error': 0.4, 'baseline_offset_at_collision': 0.2, 'not_served': 'secret',
}


# ── the allow-list ──────────────────────────────────────────────────────────────────────────


def test_the_served_keys_are_exactly_the_response_models_fields_in_the_same_order():
    assert prov.KEYS == list(SearchRecord.model_fields)
    assert len(set(prov.KEYS)) == len(prov.KEYS) == 27


def test_every_served_key_is_returned_as_stored_and_nothing_else():
    served = prov.select_search_record(FULL)
    assert list(served) == prov.KEYS
    for key in prov.KEYS:
        assert served[key] == FULL[key], key
    for not_served in ('not_served', 'baseline_replay_error', 'baseline_offset_at_collision'):
        assert not_served not in served


def test_the_figures_already_on_the_detail_are_not_repeated_here():
    assert 'baseline_replay_error' not in prov.KEYS and 'baseline_offset_at_collision' not in prov.KEYS


def test_no_provenance_is_none_and_an_empty_one_is_all_null():
    for nothing in (None, [], 'x', 3, ['a']):
        assert prov.select_search_record(nothing) is None
    empty = prov.select_search_record({})
    assert list(empty) == prov.KEYS and all(v is None for v in empty.values())


def test_a_missing_key_is_null_and_the_others_are_kept():
    partial = {k: v for k, v in FULL.items() if k != 'de_seed'}
    served = prov.select_search_record(partial)
    assert served['de_seed'] is None and served['de_popsize'] == 15


@pytest.mark.parametrize('key,bad', [
    ('de_popsize', '15'), ('de_popsize', 15.0), ('de_popsize', True), ('de_popsize', [15]),
    ('de_seed', 1.5), ('de_seed', False), ('de_n_eval', '3360'),
    ('de_tol', '0.001'), ('de_tol', True), ('de_tol', float('inf')), ('de_tol', float('nan')), ('de_tol', None),
    ('a_max', 'fast'), ('a_max', {'v': 1}), ('target_min_speed', float('-inf')),
    ('replay_model_version', 1), ('replay_model_version', ['v1']), ('selected_stage', None),
    ('de_archive_covered_all_evaluations', 1), ('de_archive_covered_all_evaluations', 'true'),
    ('target_has_interior_gap', 0),
    ('bounds', [[1, 2, 3]]), ('bounds', [[1]]), ('bounds', 'x'), ('bounds', [[1, 'a']]), ('bounds', [[1, float('inf')]]),
    ('bounds', [1, 2]), ('bounds', {'a': 1}), ('bounds', [[True, False]]),
])
def test_a_value_of_the_wrong_type_is_null_not_repaired(key, bad):
    served = prov.select_search_record({**FULL, key: bad})
    assert served[key] is None, (key, bad)
    assert served['replay_model_version'] == ('kinematic-replay-v1' if key != 'replay_model_version' else None)


def test_integers_are_numbers_where_a_number_is_expected_and_floats_are_not_counts():
    served = prov.select_search_record({**FULL, 'a_max': 12, 'de_tol': 1})
    assert served['a_max'] == 12 and served['de_tol'] == 1
    assert prov.select_search_record({**FULL, 'de_n_iter': 31.0})['de_n_iter'] is None


def test_a_zero_and_a_false_are_values_not_nulls():
    served = prov.select_search_record({**FULL, 'frames_below_heading_floor': 0, 'target_has_interior_gap': False, 'de_seed': 0})
    assert served['frames_below_heading_floor'] == 0 and served['target_has_interior_gap'] is False and served['de_seed'] == 0


def test_the_model_accepts_what_the_allow_list_returns():
    SearchRecord(**prov.select_search_record(FULL))
    SearchRecord(**prov.select_search_record({}))


def test_the_contract_has_the_route_and_its_two_models():
    schema = api_main.create_app().openapi()
    assert 'get' in schema['paths']['/scenarios/{scenario_id}/provenance']
    assert set(schema['components']['schemas']['ProvenanceResponse']['properties']) == {
        'scenario_id', 'stress_run_id', 'scene_fingerprint', 'sdc_idx', 'scored_at', 'stress_attempted_at', 'stress_tested_at', 'search',
        'delta_labels', 'run'}
    assert set(schema['components']['schemas']['SearchRecord']['properties']) == set(prov.KEYS)
    assert 'provenance' not in schema['components']['schemas']['ScenarioDetail']['properties']


def test_the_route_reads_scenario_scores_and_nothing_else():
    text = open(os.path.join(PROJECT, 'src', 'api', 'routes.py')).read()
    start = text.index("def get_provenance")
    body = text[start:text.index("\n@router.", start)]       # this route's own function, up to the next route
    tables = set(re.findall(r'\b(?:FROM|JOIN)\s+([a-z_]+)', body))
    assert tables == {'scenario_scores'}


def test_the_module_needs_nothing_the_deployed_api_lacks():
    imports = set(re.findall(r'^(?:import|from)\s+([a-z_]+)', open(os.path.join(PROJECT, 'src', 'api', 'provenance.py')).read(), re.M))
    assert imports <= {'math'}


# ── the route, on a real stored result ──────────────────────────────────────────────────────


@pytest.fixture
def conn():
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    init_geometry_schema(connection)
    yield connection
    connection.rollback()
    connection.close()


@pytest.fixture
def client(conn, monkeypatch):
    monkeypatch.setattr(api_main, 'init_pool', lambda: None)
    monkeypatch.setattr(api_main, 'close_pool', lambda: None)
    app = api_main.create_app()
    app.dependency_overrides[db_pool.get_db] = lambda: conn
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _scene():
    t = np.arange(91) * 0.1
    states = np.zeros((2, 91, 7), dtype=np.float32)
    states[0, :, 0] = -20.0 + 10.0 * t
    states[0, :, 2] = 10.0
    states[1, :, 1] = -28.0 + 10.0 * t
    states[1, :, 3] = 10.0
    states[1, :, 4] = np.pi / 2
    states[:, :, 5:7] = [4.5, 2.0]
    return states, np.ones((2, 91), dtype=bool), np.array([1, 1])


def _seed(connection, sid='scene', with_scene=True):
    """What Pass 1 writes: the scores, and, with the scene, its fingerprint and the SDC's index."""
    record = dict(scenario_id=sid, shard='synthetic', n_agents=2, min_ttc=9.0, min_pet=9.0, fragility_score=1.0)
    if with_scene:
        record.update(scene_fingerprint=db.compute_scene_fingerprint(*_scene()), sdc_idx=0)
    db.upsert_scores(connection, [record])


def _store(connection, sid='scene'):
    _seed(connection, sid)
    result = _stress_one(*_scene(), 0, de_kwargs=DE_KWARGS)
    db.update_stress_results(connection, {sid: result})
    return result


def _stored(connection, sid='scene'):
    with connection.cursor() as cur:
        cur.execute('SELECT search_provenance, stress_run_id, scene_fingerprint, sdc_idx, scored_at, stress_attempted_at, '
                    'stress_tested_at FROM scenario_scores WHERE scenario_id = %s', (sid,))
        return cur.fetchone()


def _set(connection, sql, sid='scene'):
    with connection.cursor() as cur:
        cur.execute(f'UPDATE scenario_scores SET {sql} WHERE scenario_id = %s', (sid,))
    connection.commit()


@requires_db
def test_a_real_stored_result_serves_every_served_key_exactly_as_stored(conn, client):
    result = _store(conn)
    assert result['outcome'] == 'collision_found', f'fixture regressed: {result}'
    stored, run_id, fingerprint, sdc_idx, *_ = _stored(conn)
    body = client.get('/scenarios/scene/provenance').json()
    assert body['scenario_id'] == 'scene'
    assert body['stress_run_id'] == run_id and run_id
    assert body['scene_fingerprint'] == fingerprint and fingerprint
    assert body['sdc_idx'] == sdc_idx == 0
    assert list(body['search']) == prov.KEYS
    for key in prov.KEYS:
        expected = stored.get(key)
        assert body['search'][key] == expected, key
    assert body['search']['de_seed'] == 1 and body['search']['de_popsize'] == 8 and body['search']['de_maxiter'] == 40
    assert body['search']['delta_parameterization'] == 'bicycle'
    assert body['search']['bounds'] == stored['bounds']
    assert body['search']['replay_model_version'] == 'kinematic-replay-v1'


@requires_db
def test_the_response_has_no_key_the_allow_list_does_not_name(conn, client):
    _store(conn)
    _set(conn, "search_provenance = search_provenance || '{\"zz_secret\": \"x\", \"zz_number\": 7}'::jsonb")
    raw = client.get('/scenarios/scene/provenance').text
    assert 'zz_secret' not in raw and 'zz_number' not in raw
    assert 'baseline_replay_error' not in client.get('/scenarios/scene/provenance').json()['search']


@requires_db
def test_values_round_trip_to_the_identical_double(conn, client):
    _store(conn)
    awkward = 0.1 + 0.2
    _set(conn, "search_provenance = search_provenance || '%s'::jsonb" % json.dumps({'a_max': awkward, 'de_tol': 1.2345678901234567e-7}))
    search = client.get('/scenarios/scene/provenance').json()['search']
    assert search['a_max'] == awkward and search['de_tol'] == 1.2345678901234567e-7


@requires_db
def test_a_wrong_type_stored_in_the_blob_is_null_in_the_response(conn, client):
    _store(conn)
    _set(conn, """search_provenance = search_provenance || '{"de_popsize": "many", "a_max": true, "bounds": [[1]], "de_seed": 2.5}'::jsonb""")
    response = client.get('/scenarios/scene/provenance')
    assert response.status_code == 200
    search = response.json()['search']
    assert (search['de_popsize'], search['a_max'], search['bounds'], search['de_seed']) == (None, None, None, None)
    assert search['de_maxiter'] == 40


@requires_db
def test_a_row_whose_provenance_is_null_has_a_null_search_and_keeps_the_rest(conn, client):
    _store(conn)
    _set(conn, 'search_provenance = NULL')
    body = client.get('/scenarios/scene/provenance').json()
    assert body['search'] is None
    assert body['stress_run_id'] and body['scene_fingerprint'] and body['sdc_idx'] == 0


@requires_db
def test_a_scenario_with_no_stored_search_is_200_with_nulls(conn, client):
    _seed(conn, 'fresh', with_scene=False)
    response = client.get('/scenarios/fresh/provenance')
    assert response.status_code == 200
    body = response.json()
    assert body['scenario_id'] == 'fresh'
    assert (body['stress_run_id'], body['scene_fingerprint'], body['sdc_idx'], body['search']) == (None, None, None, None)
    assert body['stress_attempted_at'] is None and body['stress_tested_at'] is None
    assert body['scored_at'] is not None


@requires_db
def test_an_unknown_scenario_is_a_404_and_a_bad_id_is_a_422(conn, client):
    r = client.get('/scenarios/nope/provenance')
    assert r.status_code == 404 and r.json() == {'detail': 'Unknown scenario: nope'}
    assert client.get('/scenarios/' + 'x' * 5000 + '/provenance').status_code == 422
    assert client.get('/scenarios/a%00b/provenance').status_code == 422


# ── the labels of the perturbation's components, under the parameterization the provenance records ────────────────


def test_the_labels_come_from_the_one_table_the_perturbed_route_uses():
    assert routes._provenance_delta_labels({'delta_parameterization': 'bicycle'}) == DELTA_LABELS
    assert routes._provenance_delta_labels({'delta_parameterization': 'linear'}) == LINEAR_DELTA_LABELS
    assert DELTA_LABELS != LINEAR_DELTA_LABELS


@pytest.mark.parametrize('search', [None, {'delta_parameterization': None}, {'delta_parameterization': 'tricycle'}, {'delta_parameterization': ''}])
def test_nothing_recorded_or_not_known_gives_no_labels(search):
    assert routes._provenance_delta_labels(search) is None


def test_the_labels_are_a_copy_not_the_tables_own_list():
    labels = routes._provenance_delta_labels({'delta_parameterization': 'bicycle'})
    labels.append('changed')
    assert 'changed' not in DELTA_LABELS and len(routes._provenance_delta_labels({'delta_parameterization': 'bicycle'})) == len(DELTA_LABELS)


@requires_db
def test_a_stored_bicycle_result_serves_the_labels_perturbed_serves_for_it(conn, client):
    _store(conn)
    body = client.get('/scenarios/scene/provenance').json()
    assert body['search']['delta_parameterization'] == 'bicycle'
    assert body['delta_labels'] == DELTA_LABELS
    assert body['delta_labels'] == client.get('/scenarios/scene/perturbed').json()['delta_labels']


@requires_db
def test_the_labels_follow_the_recorded_parameterization_and_are_null_otherwise(conn, client):
    _store(conn)
    _set(conn, """search_provenance = search_provenance || '{"delta_parameterization": "linear"}'::jsonb""")
    assert client.get('/scenarios/scene/provenance').json()['delta_labels'] == LINEAR_DELTA_LABELS
    _set(conn, """search_provenance = search_provenance || '{"delta_parameterization": "tricycle"}'::jsonb""")
    body = client.get('/scenarios/scene/provenance').json()
    assert body['delta_labels'] is None and body['search']['delta_parameterization'] == 'tricycle'
    _set(conn, """search_provenance = search_provenance || '{"delta_parameterization": 7}'::jsonb""")
    body = client.get('/scenarios/scene/provenance').json()
    assert body['delta_labels'] is None and body['search']['delta_parameterization'] is None
    _set(conn, "search_provenance = search_provenance - 'delta_parameterization'")
    assert client.get('/scenarios/scene/provenance').json()['delta_labels'] is None
    _set(conn, 'search_provenance = NULL')
    assert client.get('/scenarios/scene/provenance').json()['delta_labels'] is None


@requires_db
def test_a_scenario_with_no_stored_search_has_no_labels(conn, client):
    _seed(conn, 'fresh', with_scene=False)
    assert client.get('/scenarios/fresh/provenance').json()['delta_labels'] is None


@requires_db
def test_the_timestamps_are_utc_and_are_the_stored_instants(conn, client):
    _store(conn)
    _, _, _, _, scored, attempted, tested = _stored(conn)
    body = client.get('/scenarios/scene/provenance').json()
    for field, expected in (('scored_at', scored), ('stress_attempted_at', attempted), ('stress_tested_at', tested)):
        assert expected is not None, field
        served = datetime.fromisoformat(body[field].replace('Z', '+00:00'))
        assert served.utcoffset() == timedelta(0), (field, body[field])
        assert served == expected, field


@requires_db
def test_the_timestamps_are_utc_whatever_zone_the_session_uses(conn, client):
    _store(conn)
    with conn.cursor() as cur:
        cur.execute("SET TIME ZONE 'America/Los_Angeles'")
    conn.commit()
    body = client.get('/scenarios/scene/provenance').json()
    for field in ('scored_at', 'stress_attempted_at', 'stress_tested_at'):
        assert body[field].endswith('Z') or body[field].endswith('+00:00'), body[field]


@requires_db
def test_the_detail_does_not_carry_provenance_and_the_provenance_route_does_not_change_the_row(conn, client):
    _store(conn)
    before = _stored(conn)
    assert 'search' not in client.get('/scenarios/scene').json()
    client.get('/scenarios/scene/provenance')
    assert _stored(conn) == before
