"""
What the exporter refuses to store as a track's geometry.

validate_track_geometry runs on the write path only (export_scenario_agents and
export_perturbed_path). Before it existed, a NaN coordinate and a non-finite heading were
stored as they were, an infinite coordinate failed with PostGIS's own parse error, and the
measures were never checked. Each refusal here names the scenario, the track, the field and
the index, and carries the one offending value, never the arrays.

The accepted side is pinned with the production-shaped records the repository has: the 18
scenarios in frontend/src/test/realCollisionFrames.json, each the SDC's track and the
perturbed challenger's track exactly as the API served them (36 tracks), read from the
file as it stands. Nothing may be refused that those pass.

The tests that write to a database run against the disposable database named by the PG*
variables (AV_CLAIMS_DB=1); the rest need no database.
"""
import json
import os
import sys

import numpy as np
import psycopg2.extensions
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring import db
from src.scoring import export_geometry as eg

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(PROJECT, 'frontend', 'src', 'test', 'realCollisionFrames.json')

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

SID = 'geometry_write'


def _check(xs, ys, ms, hs, label='agent 3'):
    eg.validate_track_geometry(xs, ys, ms, hs, scenario_id=SID, label=label)


def _track(n=6):
    """A clean track: distinctive values so that a payload dump would be recognisable."""
    xs = [1000.5 + i for i in range(n)]
    ys = [2000.25 + i for i in range(n)]
    ms = [float(i) for i in range(n)]
    hs = [0.5 + 0.01 * i for i in range(n)]
    return xs, ys, ms, hs


# ── the accepted side ────────────────────────────────────────────────────────────────


def _real_tracks():
    with open(REAL, encoding='utf-8') as f:
        scenarios = json.load(f)['scenarios']
    for entry in scenarios:
        for role in ('sdc', 'perturbed'):
            yield entry['scenario_id'], role, entry[role]


def test_the_real_records_are_all_there():
    tracks = list(_real_tracks())
    assert len({sid for sid, _, _ in tracks}) == 18
    assert len(tracks) == 36
    assert all(len(t['path']) >= 2 for _, _, t in tracks)


@pytest.mark.parametrize('sid,role,track', [pytest.param(sid, role, track, id=f'{sid}-{role}')
                                            for sid, role, track in _real_tracks()])
def test_every_production_shaped_track_still_validates(sid, role, track):
    xs = [p[0] for p in track['path']]
    ys = [p[1] for p in track['path']]
    eg.validate_track_geometry(xs, ys, track['timesteps'], track['headings'],
                               scenario_id=sid, label=f'{role} agent {track["agent_idx"]}')


@pytest.mark.parametrize('ms', [
    [0, 1, 2, 3, 4, 5],           # every frame observed
    [2, 3, 4, 5, 6, 7],           # starts late
    [0, 1, 4, 5, 9, 12],          # gaps between observed frames
    [0, 90],                      # two vertices, far apart
])
def test_ordinary_measures_are_accepted(ms):
    n = len(ms)
    _check([float(i) for i in range(n)], [0.0] * n, ms, [0.0] * n)


def test_numpy_inputs_are_accepted_as_the_exporter_passes_them():
    states = np.zeros((1, 10, 7), dtype=np.float32)
    ts = np.array([2, 3, 4, 7])
    _check(states[0, ts, 0], states[0, ts, 1], ts.astype(float), [float(h) for h in states[0, ts, 4]])


# ── the refused side: one per form ───────────────────────────────────────────────────

BAD = [
    # name,                       field,       index, mutate,                                   fragment of the problem
    ('NaN x',                     'x',         2,     lambda t: t[0].__setitem__(2, float('nan')),  'nan'),
    ('infinite x',                'x',         0,     lambda t: t[0].__setitem__(0, float('inf')),  'inf'),
    ('NaN y',                     'y',         5,     lambda t: t[1].__setitem__(5, float('nan')),  'nan'),
    ('negative infinite y',       'y',         3,     lambda t: t[1].__setitem__(3, float('-inf')), '-inf'),
    ('infinite heading',          'headings',  4,     lambda t: t[3].__setitem__(4, float('inf')),  'inf'),
    ('negative infinite heading', 'headings',  1,     lambda t: t[3].__setitem__(1, float('-inf')), '-inf'),
    ('NaN heading',               'headings',  2,     lambda t: t[3].__setitem__(2, float('nan')),  'nan'),
    ('NaN measure',               'measures',  3,     lambda t: t[2].__setitem__(3, float('nan')),  'not a finite number'),
    ('infinite measure',          'measures',  5,     lambda t: t[2].__setitem__(5, float('inf')),  'not a finite number'),
    ('fractional measure',        'measures',  2,     lambda t: t[2].__setitem__(2, 2.5),           'not a whole number'),
    ('negative measure',          'measures',  0,     lambda t: t[2].__setitem__(0, -1.0),          'negative'),
    ('duplicate measure',         'measures',  3,     lambda t: t[2].__setitem__(3, 2.0),           'the same as measures[2]'),
    ('out-of-order measure',      'measures',  3,     lambda t: t[2].__setitem__(3, 1.0),           'before measures[2]'),
]


@pytest.mark.parametrize('name,field,index,mutate,problem', BAD, ids=[b[0] for b in BAD])
def test_each_bad_form_is_refused_by_name(name, field, index, mutate, problem):
    track = _track()
    mutate(track)
    with pytest.raises(eg.GeometryRefusedError) as raised:
        _check(*track)
    message = str(raised.value)
    assert f'geometry refused for {SID} agent 3: {field}[{index}] ' in message
    assert problem in message
    assert (raised.value.scenario_id, raised.value.label, raised.value.field, raised.value.index) == (
        SID, 'agent 3', field, index)


def test_a_refusal_is_a_value_error_and_not_the_whole_payload():
    n = 400
    xs, ys, ms, hs = _track(n)
    hs[137] = float('inf')
    with pytest.raises(eg.GeometryRefusedError) as raised:
        _check(xs, ys, ms, hs)
    assert isinstance(raised.value, ValueError)
    message = str(raised.value)
    assert len(message) < 140
    # The payload's distinctive values are not in it: not the coordinates, not the other headings.
    assert '1000.5' not in message and '2000.25' not in message and '0.5' not in message.replace('headings[137]', '')


def test_the_first_bad_index_is_the_one_named():
    xs, ys, ms, hs = _track()
    xs[4] = float('nan')
    xs[2] = float('nan')
    with pytest.raises(eg.GeometryRefusedError) as raised:
        _check(xs, ys, ms, hs)
    assert raised.value.index == 2


@pytest.mark.parametrize('which,field', [(0, 'x'), (1, 'y'), (2, 'measures'), (3, 'headings')])
def test_arrays_of_different_lengths_are_refused_naming_all_four(which, field):
    track = list(_track())
    track[which] = track[which][:-1]
    with pytest.raises(eg.GeometryRefusedError) as raised:
        _check(*track)
    assert raised.value.field == 'lengths'
    counts = {'x': 6, 'y': 6, 'measures': 6, 'headings': 6}
    counts[field] = 5
    assert str(raised.value).endswith('lengths differ: ' + ', '.join(f'{f} {n}' for f, n in counts.items()))


def test_a_value_that_is_not_a_number_is_refused_by_field():
    xs, ys, ms, hs = _track()
    xs[1] = 'abc'
    with pytest.raises(eg.GeometryRefusedError) as raised:
        _check(xs, ys, ms, hs)
    assert raised.value.field == 'x'
    assert 'abc' not in str(raised.value)


def test_nothing_on_the_read_side_uses_the_validator():
    """A stored row cannot start failing on read: the API module never refers to it."""
    with open(os.path.join(PROJECT, 'src', 'api', 'routes.py'), encoding='utf-8') as f:
        source = f.read()
    assert 'validate_track_geometry' not in source
    assert 'GeometryRefusedError' not in source


# ── through the real exporters (disposable database) ────────────────────────────────

T = 6


def _scene():
    """Two agents, T frames, all observed."""
    states = np.zeros((2, T, 7), dtype=np.float32)
    validity = np.ones((2, T), dtype=bool)
    for agent in range(2):
        for t in range(T):
            states[agent, t] = [10.0 * agent + t, 3.0 * agent, 0.0, 0.0, 0.01 * t, 4.5, 2.0]
    return states, validity, np.array([1, 1])


@pytest.fixture
def conn():
    c = db.get_connection()
    with c.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, scenario_scores CASCADE')
    c.commit()
    db.init_schema(c)
    eg.init_geometry_schema(c)
    db.upsert_scores(c, [dict(scenario_id=SID, shard='x', n_agents=2, min_ttc=9.0,
                              min_pet=9.0, fragility_score=1.0)])
    yield c
    c.rollback()
    c.close()


def _count(conn, table):
    with conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM {table} WHERE scenario_id = %s', (SID,))
        return cur.fetchone()[0]


@requires_db
def test_a_clean_scene_is_written_as_before(conn):
    written, skipped, _ = eg.export_scenario_agents(conn, SID, *_scene(), 0)
    assert (written, skipped) == (2, 0)
    assert _count(conn, 'scenario_agents') == 2


@requires_db
@pytest.mark.parametrize('column,value,field', [
    (0, float('nan'), 'x'),
    (0, float('inf'), 'x'),
    (1, float('nan'), 'y'),
    (1, float('-inf'), 'y'),
    (4, float('inf'), 'headings'),
    (4, float('nan'), 'headings'),
])
def test_export_refuses_a_bad_value_and_leaves_nothing_behind(conn, column, value, field):
    states, validity, types = _scene()
    # Agent 1 is the bad one, so agent 0 has already been written when it is reached: the
    # refusal must take that row back too.
    states[1, 3, column] = value
    with pytest.raises(eg.GeometryRefusedError) as raised:
        eg.export_scenario_agents(conn, SID, states, validity, types, 0)
    assert str(raised.value).startswith(f'geometry refused for {SID} agent 1: {field}[3] ')
    assert conn.get_transaction_status() == psycopg2.extensions.TRANSACTION_STATUS_IDLE
    assert _count(conn, 'scenario_agents') == 0
    # The refusal rolled back: no transaction is left open holding the row lock, and the
    # next export works.
    written, _, _ = eg.export_scenario_agents(conn, SID, *_scene(), 0)
    assert written == 2


def _perturbed(conn, mutate=None):
    states, validity, _ = _scene()
    if mutate:
        mutate(states)
    return eg.export_perturbed_path(conn, SID, states, validity, 1)


@requires_db
def test_a_clean_perturbed_path_is_written_as_before(conn):
    assert _perturbed(conn) is True
    assert _count(conn, 'perturbed_paths') == 1


@requires_db
@pytest.mark.parametrize('column,value,field', [
    (0, float('nan'), 'x'),
    (1, float('inf'), 'y'),
    (4, float('inf'), 'headings'),
])
def test_perturbed_export_refuses_a_bad_value_and_writes_nothing(conn, column, value, field):
    with pytest.raises(eg.GeometryRefusedError) as raised:
        _perturbed(conn, lambda s: s.__setitem__((1, 2, column), value))
    assert str(raised.value).startswith(f'geometry refused for {SID} perturbed target 1: {field}[2] ')
    assert conn.get_transaction_status() == psycopg2.extensions.TRANSACTION_STATUS_IDLE
    assert _count(conn, 'perturbed_paths') == 0
