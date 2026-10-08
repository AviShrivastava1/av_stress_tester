"""
test_api_per_frame_sizes.py — A-2: the API serves each agent's box size at every exported frame.

scenario_agents has stored two arrays since A-1: element i of lengths_m / widths_m is the size at
vertex i of `path`, built from the same list of valid frames as the path and the headings. The
verifier builds an agent's box from the size at THAT frame, so a client that draws only the scalar
length_m / width_m draws boxes that can fail to touch for a collision the verifier found. These tests
pin what the two routes that return an AgentTrack (/trajectories and /perturbed) now serve:

  * the arrays, exactly as stored, aligned with `timesteps` and not with the frame index;
  * null for an element whose recorded size was unusable, and strict JSON throughout;
  * null arrays for a row exported before the columns existed;
  * a 200 with null arrays, not a 500, when the database has no such columns (a restore of an older
    backup under a running API), and recovery when they return;
  * a 500 that names the problem, never a pad, a truncation or a guess, for an array whose length
    differs from the path, for exactly one array being NULL, for a stored element that is not
    finite, and for sizes borrowed by the perturbed track that do not belong to its frames.

Expected values are computed from the inputs through float(np.float32(...)), the way the export
stores them; none is copied from a run.

Needs a DISPOSABLE Postgres/PostGIS database; skips unless AV_ROBUSTNESS_DB=1 (CI sets it).

Run:
    AV_ROBUSTNESS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_api_per_frame_sizes.py -q
"""

import json
import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from psycopg2.pool import ThreadedConnectionPool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import db_pool, main as api_main
from src.scoring import db
from src.scoring.export_geometry import (
    export_perturbed_path, export_scenario_agents, init_geometry_schema,
)

requires_db = pytest.mark.skipif(
    os.environ.get('AV_ROBUSTNESS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

SID = 'sizes_scene'
T = 10
TARGET = 1

# Valid frames per agent. Agent 0 starts late, agent 1 (the perturbed target) has two interior
# gaps, so an array aligned with the frame index instead of with the valid frames cannot match.
VALID = {0: [2, 3, 4, 5, 6, 7, 8, 9], 1: [0, 1, 4, 5, 9]}
BOTH = ('lengths_m', 'widths_m')


def _length(agent, t):
    return 4.0 + 0.37 * agent + 0.013 * t


def _width(agent, t):
    return 1.8 + 0.21 * agent + 0.007 * t


def _f32(x):
    return float(np.float32(x))


def _usable(x):
    x = _f32(x)
    return bool(np.isfinite(x) and x > 0.0)


def _expected(agent, fn):
    """What the export stores for this agent's valid frames: float32-widened, None where unusable."""
    return [_f32(fn(agent, t)) if _usable(fn(agent, t)) else None for t in VALID[agent]]


def _scene(length=_length, width=_width):
    """Two agents, T frames. Invalid frames hold a junk size (99.0) that must never be served."""
    states = np.zeros((2, T, 7), dtype=np.float32)
    validity = np.zeros((2, T), dtype=bool)
    for agent, frames in VALID.items():
        for t in range(T):
            states[agent, t, 0] = 10.0 * agent + t
            states[agent, t, 1] = 3.0 * agent
            states[agent, t, 4] = 0.01 * t
            if t in frames:
                validity[agent, t] = True
                states[agent, t, 5] = length(agent, t)
                states[agent, t, 6] = width(agent, t)
            else:
                states[agent, t, 5] = 99.0
                states[agent, t, 6] = 99.0
    return states, validity, np.array([1, 2])


# ── fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def conn():
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, '
                    'scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    init_geometry_schema(connection)
    db.upsert_scores(connection, [dict(scenario_id=SID, shard='synthetic', n_agents=2,
                                       min_ttc=9.0, min_pet=9.0, fragility_score=1.0)])
    yield connection
    connection.rollback()
    connection.close()


def _app(monkeypatch):
    monkeypatch.setattr(api_main, 'init_pool', lambda: None)
    monkeypatch.setattr(api_main, 'close_pool', lambda: None)
    return api_main.create_app()


@pytest.fixture
def client(conn, monkeypatch):
    app = _app(monkeypatch)
    app.dependency_overrides[db_pool.get_db] = lambda: conn
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _export(conn, scene=None, perturbed_validity=None):
    states, validity, types = scene if scene is not None else _scene()
    export_scenario_agents(conn, SID, states, validity, types, 0)
    export_perturbed_path(conn, SID, states,
                          validity if perturbed_validity is None else perturbed_validity, TARGET)


def _sql(conn, statement, params=()):
    with conn.cursor() as cur:
        cur.execute(statement, params)
    conn.commit()


def _trajectories(client):
    response = client.get(f'/scenarios/{SID}/trajectories')
    return response


def _perturbed(client):
    return client.get(f'/scenarios/{SID}/perturbed')


def _by_agent(response):
    assert response.status_code == 200, response.text
    return {a['agent_idx']: a for a in response.json()['agents']}


def _strictly_json(response):
    """The bytes carry no NaN/Infinity token, and the parsed body re-serialises with allow_nan=False."""
    def refuse(token):
        raise AssertionError(f'non-JSON constant {token} in the response body')
    json.loads(response.content, parse_constant=refuse)
    json.dumps(response.json(), allow_nan=False)


# ── 1-3. the stored values, exactly ───────────────────────────────────────────────

@requires_db
def test_trajectories_serve_the_stored_arrays_and_leave_the_scalars_alone(conn, client):
    _export(conn)
    agents = _by_agent(_trajectories(client))
    assert set(agents) == {0, 1}
    for agent, track in agents.items():
        assert track['lengths_m'] == _expected(agent, _length)
        assert track['widths_m'] == _expected(agent, _width)
        # the scalars are still read at the first valid frame, as before
        assert track['length_m'] == _f32(_length(agent, VALID[agent][0]))
        assert track['width_m'] == _f32(_width(agent, VALID[agent][0]))


@requires_db
def test_the_arrays_follow_timesteps_not_the_frame_index(conn, client):
    _export(conn)
    for agent, track in _by_agent(_trajectories(client)).items():
        assert len(track['timesteps']) < T, 'premise: the agent has a validity gap'
        assert len(track['lengths_m']) == len(track['widths_m']) == len(track['timesteps']) \
            == len(track['path']) == len(track['headings'])
        for k, frame in enumerate(track['timesteps']):
            assert track['lengths_m'][k] == _f32(_length(agent, int(frame))), (agent, k, frame)
            assert track['widths_m'][k] == _f32(_width(agent, int(frame))), (agent, k, frame)
        assert 99.0 not in track['lengths_m'] and 99.0 not in track['widths_m']


@requires_db
def test_perturbed_serves_the_target_agents_arrays_on_both_tracks(conn, client):
    _export(conn)
    response = _perturbed(client)
    assert response.status_code == 200, response.text
    body = response.json()
    for name in ('baseline', 'perturbed'):
        track = body[name]
        assert track is not None, name
        assert track['agent_idx'] == TARGET
        assert track['lengths_m'] == _expected(TARGET, _length), name
        assert track['widths_m'] == _expected(TARGET, _width), name
    assert body['baseline']['timesteps'] == body['perturbed']['timesteps']


# ── 4. unusable sizes arrive as null; the body is strict JSON ──────────────────────

def _bad_length(agent, t):
    if agent == TARGET:
        return {4: float('nan'), 9: 0.0}.get(t, _length(agent, t))
    return _length(agent, t)


def _bad_width(agent, t):
    if agent == TARGET:
        return {5: float('inf'), 1: -1.0}.get(t, _width(agent, t))
    return _width(agent, t)


@requires_db
@pytest.mark.parametrize('route', ['trajectories', 'perturbed'])
def test_unusable_sizes_arrive_as_null_and_the_body_is_strict_json(conn, client, route):
    _export(conn, _scene(_bad_length, _bad_width))
    response = client.get(f'/scenarios/{SID}/{route}')
    assert response.status_code == 200, response.text
    _strictly_json(response)

    tracks = ([_by_agent(response)[TARGET]] if route == 'trajectories'
              else [response.json()['baseline'], response.json()['perturbed']])
    for track in tracks:
        assert track['lengths_m'] == _expected(TARGET, _bad_length)
        assert track['widths_m'] == _expected(TARGET, _bad_width)
        assert track['lengths_m'].count(None) == 2 and track['widths_m'].count(None) == 2
        assert len(track['lengths_m']) == len(track['path'])      # nulls keep their place
    # the other agent has no unusable size
    if route == 'trajectories':
        assert None not in _by_agent(response)[0]['lengths_m']


# ── 5. a row exported before the columns existed ───────────────────────────────────

@requires_db
def test_a_legacy_row_serves_null_arrays_and_unchanged_scalars(conn, client):
    _export(conn)
    _sql(conn, 'UPDATE scenario_agents SET lengths_m = NULL, widths_m = NULL')
    for agent, track in _by_agent(_trajectories(client)).items():
        assert track['lengths_m'] is None and track['widths_m'] is None
        assert track['length_m'] == _f32(_length(agent, VALID[agent][0]))
        assert track['width_m'] == _f32(_width(agent, VALID[agent][0]))
    body = _perturbed(client).json()
    for name in ('baseline', 'perturbed'):
        assert body[name]['lengths_m'] is None and body[name]['widths_m'] is None
        assert body[name]['length_m'] == _f32(_length(TARGET, VALID[TARGET][0]))
        assert body[name]['path']


# ── 6. a table without the columns ────────────────────────────────────────────────

def _without_arrays(value):
    """A response body with the two array fields forced to None, everything else untouched."""
    if isinstance(value, dict):
        return {k: (None if k in BOTH else _without_arrays(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_without_arrays(v) for v in value]
    return value


@requires_db
@pytest.mark.parametrize('dropped', [BOTH, ('lengths_m',), ('widths_m',)], ids='drop-{}'.format)
def test_a_table_without_the_columns_serves_200_with_null_arrays(conn, client, dropped):
    _export(conn)
    with_columns = {r: client.get(f'/scenarios/{SID}/{r}').json() for r in ('trajectories', 'perturbed')}
    assert any(a['lengths_m'] is not None for a in with_columns['trajectories']['agents']), 'premise'

    _sql(conn, 'ALTER TABLE scenario_agents ' + ', '.join(f'DROP COLUMN {c}' for c in dropped))

    for route in ('trajectories', 'perturbed'):
        for attempt in (1, 2):               # the second request proves the connection was not poisoned
            response = client.get(f'/scenarios/{SID}/{route}')
            assert response.status_code == 200, (route, attempt, response.text)
            assert response.json() == _without_arrays(with_columns[route]), (route, attempt)
    assert client.get('/health').status_code == 200


@requires_db
def test_a_restore_of_an_older_backup_under_a_running_api_does_not_break_the_routes(conn, monkeypatch):
    """
    The rollback sequence. A pooled connection is open and has served requests; the columns are
    dropped underneath it (what restoring the pre-A-1 backup does); the same app is asked again
    and must answer 200 with null arrays, and must serve the arrays again once they are back.
    """
    _export(conn)
    pool = ThreadedConnectionPool(1, 1, **api_main.settings.dsn_kwargs())    # one connection: the same one every time
    monkeypatch.setattr(db_pool, '_pool', pool)
    try:
        with TestClient(_app(monkeypatch), raise_server_exceptions=False) as client:
            first = _by_agent(_trajectories(client))
            assert first[TARGET]['lengths_m'] == _expected(TARGET, _length)

            _sql(conn, 'ALTER TABLE scenario_agents DROP COLUMN lengths_m, DROP COLUMN widths_m')
            for route in ('trajectories', 'perturbed'):
                response = client.get(f'/scenarios/{SID}/{route}')
                assert response.status_code == 200, (route, response.text)
            assert _by_agent(_trajectories(client))[TARGET]['lengths_m'] is None
            body = _perturbed(client).json()
            assert body['baseline']['lengths_m'] is None and body['perturbed']['widths_m'] is None

            init_geometry_schema(conn)            # the new schema returns (a restore of the new dump)
            _export(conn)
            assert _by_agent(_trajectories(client))[TARGET]['lengths_m'] == _expected(TARGET, _length)
            assert _perturbed(client).json()['perturbed']['widths_m'] == _expected(TARGET, _width)
    finally:
        pool.closeall()


# ── 7-9. refusals, named ──────────────────────────────────────────────────────────

@requires_db
@pytest.mark.parametrize('column', BOTH)
@pytest.mark.parametrize('route, label', [('trajectories', f'{SID} agent {TARGET}'),
                                          ('perturbed', f'{SID} agent {TARGET} (baseline)')])
def test_an_array_that_is_shorter_than_the_path_is_refused_by_name(conn, client, column, route, label):
    _export(conn)
    _sql(conn, f'UPDATE scenario_agents SET {column} = {column}[1:3] WHERE agent_idx = %s', (TARGET,))
    response = client.get(f'/scenarios/{SID}/{route}')
    assert response.status_code == 500, f'a short {column} was served as HTTP {response.status_code}'
    detail = response.json()['detail']
    assert label in detail and f'5 coordinates' in detail and f'3 {column}' in detail, detail
    assert 'Refusing to pad or truncate' in detail


@requires_db
@pytest.mark.parametrize('column', BOTH)
def test_an_array_that_is_longer_than_the_path_is_refused_by_name(conn, client, column):
    _export(conn)
    _sql(conn, f'UPDATE scenario_agents SET {column} = {column} || 4.5::float8 WHERE agent_idx = %s', (TARGET,))
    response = _trajectories(client)
    assert response.status_code == 500
    assert f'6 {column}' in response.json()['detail']


@requires_db
@pytest.mark.parametrize('null_column, other', [('lengths_m', 'widths_m'), ('widths_m', 'lengths_m')])
@pytest.mark.parametrize('route', ['trajectories', 'perturbed'])
def test_exactly_one_array_null_is_refused_by_name(conn, client, null_column, other, route):
    _export(conn)
    _sql(conn, f'UPDATE scenario_agents SET {null_column} = NULL WHERE agent_idx = %s', (TARGET,))
    response = client.get(f'/scenarios/{SID}/{route}')
    assert response.status_code == 500, f'half of the per-frame sizes was served as HTTP {response.status_code}'
    detail = response.json()['detail']
    assert f'{null_column} is NULL' in detail and other in detail, detail


@requires_db
@pytest.mark.parametrize('column', BOTH)
@pytest.mark.parametrize('stored', ['NaN', 'Infinity', '-Infinity'])
@pytest.mark.parametrize('route', ['trajectories', 'perturbed'])
def test_a_stored_non_finite_element_is_refused_by_name_and_not_converted(conn, client, column, stored, route):
    _export(conn)
    _sql(conn, f"UPDATE scenario_agents SET {column}[2] = '{stored}'::float8 WHERE agent_idx = %s", (TARGET,))
    response = client.get(f'/scenarios/{SID}/{route}')
    assert response.status_code == 500, f'{stored} was served as HTTP {response.status_code}'
    detail = response.json()['detail']
    assert f'{column}[1]' in detail and 'finite' in detail, detail       # SQL element 2 is index 1


# ── 10. the perturbed track without a baseline row ────────────────────────────────

@requires_db
@pytest.mark.parametrize('how', ['no-scenario_agents-table', 'scene-mismatch'])
def test_the_perturbed_track_has_null_arrays_when_there_is_no_baseline_row(conn, client, how):
    _export(conn)
    if how == 'no-scenario_agents-table':
        _sql(conn, 'DROP TABLE scenario_agents CASCADE')
    else:
        _sql(conn, "UPDATE scenario_agents SET scene_fingerprint = 'a-scene-that-no-longer-exists'")
    response = _perturbed(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['baseline'] is None
    perturbed = body['perturbed']
    assert perturbed is not None and perturbed['path']
    assert perturbed['length_m'] is None and perturbed['width_m'] is None
    assert perturbed['lengths_m'] is None and perturbed['widths_m'] is None


# ── 11. sizes borrowed by the perturbed track must belong to its frames ───────────

def _validity_with_target_frames(frames):
    _, validity, _ = _scene()
    validity = validity.copy()
    validity[TARGET] = False
    validity[TARGET, frames] = True
    return validity


@requires_db
def test_borrowed_arrays_with_a_different_count_than_the_perturbed_path_are_refused(conn, client):
    _export(conn, perturbed_validity=_validity_with_target_frames([0, 1, 4, 5]))
    response = _perturbed(client)
    assert response.status_code == 500, response.text
    detail = response.json()['detail']
    assert f'{SID} agent {TARGET} (perturbed)' in detail, detail
    assert '4 coordinates' in detail and '5 lengths_m' in detail, detail


@requires_db
def test_borrowed_arrays_for_other_frames_are_refused_with_the_first_differing_index(conn, client):
    # same count (5), but the last frame is 8, not 9: the sizes would be attached to the wrong moment
    _export(conn, perturbed_validity=_validity_with_target_frames([0, 1, 4, 5, 8]))
    response = _perturbed(client)
    assert response.status_code == 500, response.text
    detail = response.json()['detail']
    assert f'{SID} agent {TARGET} (perturbed)' in detail, detail
    assert 'index 4' in detail and 'baseline' in detail, detail


@requires_db
def test_a_legacy_row_with_no_arrays_still_serves_whatever_frames_the_perturbed_path_has(conn, client):
    _export(conn, perturbed_validity=_validity_with_target_frames([0, 1, 4, 5, 8]))
    _sql(conn, 'UPDATE scenario_agents SET lengths_m = NULL, widths_m = NULL')
    response = _perturbed(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['perturbed']['lengths_m'] is None and body['perturbed']['timesteps'][-1] == 8.0


# ── 12. the contract ───────────────────────────────────────────────────────────────

def test_the_openapi_schema_declares_the_two_arrays_as_nullable_optional_number_or_null_lists():
    schema = api_main.app.openapi()['components']['schemas']['AgentTrack']
    assert 'lengths_m' not in schema.get('required', []) and 'widths_m' not in schema.get('required', [])
    for name in BOTH:
        variants = schema['properties'][name]['anyOf']
        assert {'type': 'null'} in variants, name
        arrays = [v for v in variants if v.get('type') == 'array']
        assert len(arrays) == 1, name
        item_types = sorted(v['type'] for v in arrays[0]['items']['anyOf'])
        assert item_types == ['null', 'number'], name
