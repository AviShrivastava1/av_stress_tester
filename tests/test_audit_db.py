"""SQL/API regression expectations. Requires a DISPOSABLE Postgres/PostGIS DB.

AV_AUDIT_DB=1 is required. These tests clear the three project tables in that DB.
The bundled runner creates a fresh in-memory PGlite instance for this purpose.
"""
import base64
import json
import os
from pathlib import Path
import sys

import numpy as np
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(os.environ['AV_AUDIT_PROJECT']).resolve()))
from src.scoring import db
from src.scoring.export_geometry import (
    init_geometry_schema, export_scenario_agents, export_perturbed_path,
)
from src.optimization.perturbation_space import PerturbationSpace
from src.danger.collision_detector import check_collision_trajectory
from src.api import main as api_main
from src.api import db_pool

pytestmark = pytest.mark.skipif(os.environ.get('AV_AUDIT_DB') != '1',
                                reason='Requires explicit disposable-DB test runner')


@pytest.fixture
def conn():
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    yield connection
    connection.rollback()
    connection.close()


@pytest.fixture
def client(conn, monkeypatch):
    monkeypatch.setattr(api_main, 'init_pool', lambda: None)
    monkeypatch.setattr(api_main, 'close_pool', lambda: None)
    app = api_main.create_app()
    app.dependency_overrides[db_pool.get_db] = lambda: conn
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def seed(conn, sid='audit_scene', n=2):
    db.upsert_scores(conn, [dict(scenario_id=sid, shard='synthetic', n_agents=n,
                                min_ttc=999., min_pet=999., fragility_score=0.)])


def pedestrian_scene():
    states = np.zeros((2,10,7), dtype=np.float32)
    states[0,:,5:7] = [4.5,2.]
    states[1,:,5:7] = [.6,.6]
    states[1,:,0] = 5. - 2.*np.arange(10)*.1
    states[1,:,2] = -2.
    states[1,:,4] = np.pi
    return states, np.ones((2,10), dtype=bool), np.array([1,2])


def store_result(conn, space, delta, export=False):
    delta = np.asarray(delta, dtype=np.float32)
    perturbed = space.apply(delta)
    collision, t = check_collision_trajectory(perturbed, space.validity, 0, 1)
    assert collision
    result = dict(status='ok', collision=True, min_perturbation=space.weighted_norm(delta),
                  delta=delta, collision_timestep=t, target_idx=1, method='de')
    db.update_stress_results(conn, {'audit_scene':result})
    if export:
        export_perturbed_path(conn, 'audit_scene', perturbed, space.validity, 1)
    return perturbed


@pytest.mark.parametrize('suffix', ['trajectories','perturbed'])
def test_B12_missing_geometry_tables_are_a_normal_unexported_state(conn, client, suffix):
    seed(conn)
    response = client.get(f'/scenarios/audit_scene/{suffix}')
    assert response.status_code == 200, (
        f'Pass 1 only, before geometry schema: /{suffix} returns '
        f'{response.status_code}, expected a normal empty response'
    )


def test_B13_pedestrian_delta_labels_use_linear_units(conn, client):
    seed(conn)
    init_geometry_schema(conn)
    states, validity, types = pedestrian_scene()
    export_scenario_agents(conn, 'audit_scene', states, validity, types, 0)
    space = PerturbationSpace(states,validity,types,0,1)
    store_result(conn,space,[-1,0,0,0],export=True)
    response = client.get('/scenarios/audit_scene/perturbed')
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['baseline']['agent_type'] == 2
    assert not any('rad' in label for label in data['delta_labels']), data['delta_labels']


def test_B14_new_delta_does_not_get_paired_with_an_old_export(conn, client):
    seed(conn)
    init_geometry_schema(conn)
    s,v,types = pedestrian_scene()
    export_scenario_agents(conn, 'audit_scene', s,v,types,0)
    space = PerturbationSpace(s,v,types,0,1)
    store_result(conn,space,[-1,0,0,0],export=True)
    new_states = store_result(conn,space,[-2,0,0,0],export=False)
    response = client.get('/scenarios/audit_scene/perturbed')
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['delta'] == [-2.,0.,0.,0.]
    if data['perturbed'] is not None:
        actual = np.asarray(data['perturbed']['path'])
        wanted = new_states[1,:,:2]
        deviation = float(np.abs(actual-wanted).max())
        assert np.allclose(actual,wanted,atol=1e-7), (
            f'New delta is returned with the previous run path; max mismatch={deviation:.6f} m'
        )


def test_B15_reexport_removes_agents_that_are_now_skipped(conn, client):
    seed(conn,n=3)
    init_geometry_schema(conn)
    s = np.zeros((3,3,7),dtype=np.float32)
    s[:,:,5:7] = [4.5,2.]
    s[:,:,0] = np.arange(3)
    v = np.ones((3,3),dtype=bool)
    types = np.ones(3,dtype=int)
    export_scenario_agents(conn,'audit_scene',s,v,types,0)
    v[2,1:] = False
    assert export_scenario_agents(conn,'audit_scene',s,v,types,0) == (2,1)
    response = client.get('/scenarios/audit_scene/trajectories')
    assert response.status_code == 200, response.text
    actual = [agent['agent_idx'] for agent in response.json()['agents']]
    assert actual == [0,1], f'Export reports 2 written/1 skipped, but API still serves {actual}'


@pytest.mark.parametrize('location',['path','cursor'])
def test_B17_nul_in_client_input_is_rejected_as_a_client_error(conn,client,location):
    seed(conn)
    if location == 'path':
        response = client.get('/scenarios/%00')
    else:
        cursor = base64.urlsafe_b64encode(json.dumps({'f':1.,'s':'\0'}).encode()).decode()
        response = client.get('/scenarios',params={'cursor':cursor})
    assert 400 <= response.status_code < 500, (
        f'Malformed {location} input returns {response.status_code}, expected 4xx'
    )


def test_B18_exhausted_pool_has_an_explicit_overload_response(conn,monkeypatch):
    pool = ThreadedConnectionPool(1,1,**api_main.settings.dsn_kwargs())
    borrowed = pool.getconn()
    monkeypatch.setattr(db_pool, '_pool', pool)
    app = api_main.create_app()
    try:
        # Without a context manager, lifespan does not replace/close this controlled pool.
        test_client = TestClient(app,raise_server_exceptions=False)
        response = test_client.get('/health')
        assert response.status_code == 503, (
            f'Valid pool configuration, one connection in use: HTTP {response.status_code}'
        )
    finally:
        pool.putconn(borrowed)
        pool.closeall()


def test_control_geometry_retains_real_timestep_gaps(conn,client):
    seed(conn)
    init_geometry_schema(conn)
    s,v,types = pedestrian_scene()
    v[1] = False
    v[1,[0,3,9]] = True
    export_scenario_agents(conn,'audit_scene',s,v,types,0)
    response = client.get('/scenarios/audit_scene/trajectories')
    assert response.status_code == 200, response.text
    agent = response.json()['agents'][1]
    assert agent['timesteps'] == [0.,3.,9.]
    np.testing.assert_allclose(agent['path'],s[1,[0,3,9],:2],atol=1e-8)


def test_control_unknown_id_is_404_and_sql_is_parameterized(conn,client):
    seed(conn)
    response = client.get('/scenarios/' + "x'; DELETE FROM scenario_scores;--")
    assert response.status_code == 404
    assert db.count_rows(conn)['total'] == 1
