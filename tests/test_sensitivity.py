"""
The sensitivity slice (#15): the offline runner (src/scoring/sensitivity.py), the allow-list (src/api/sensitivity.py) and
GET /scenarios/{id}/slice.

Pinned here:

  * the request: two different components 0 to 3, 2 to 101 points per axis, finite ranges with low below high, checked before anything
    is read or evaluated; a grid that contains the stored value exactly;
  * the runner: a slice of the stored result's scene and challenger, the other two components held at the stored values, the stored point
    a cell that collides; it writes only `sensitivity_slices`; it refuses a scene or SDC index that is not the one searched, a range
    outside the search's bounds or not containing the stored value, a stored point that does not collide; a slice already recorded for
    the same result and axes is left as it is and reported, with nothing evaluated;
  * the allow-list: each field's type, a wrong type is null, a grid whose arrays do not fit is not a grid;
  * the route: only slices drawn around the stored result (its `stress_run_id`), the identity gate, a fixed order, a limit that says when
    it cut (`truncated`), labels from each slice's own model, an empty list for every state with nothing to list (a role that cannot read
    the table: one fixed log line), and 404 only for an unknown scenario.

The database tests build a throw-away database and drop it after. They skip unless AV_ROBUSTNESS_DB=1.

Run:
    AV_ROBUSTNESS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_sensitivity.py -q
"""

import gzip
import logging
import math
import os
import sys

import numpy as np
import psycopg2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import sensitivity as served
from src.api.models import DELTA_LABELS_BY_PARAMETERIZATION, SensitivityResponse, SensitivitySlice
from src.scoring import sensitivity as runner
from src.scoring.manifest import ManifestTablesMissing
from src.scoring.search_runs import TraceRerunRefused

from tests.test_trace import (  # noqa: F401  (the fixtures are used by name)
    BUDGET, _client, _count, _fingerprint_of_the_results, _insert_scenario, _scene, _server, _stored, bare_db, deny_role, requires_db, stored, tr_db,
)

# ── the request ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize('kwargs', [
    dict(axis_x=0, axis_y=0), dict(axis_x=-1, axis_y=1), dict(axis_x=0, axis_y=4), dict(axis_x=True, axis_y=1), dict(axis_x=0, axis_y=1.0),
    dict(axis_x=True, axis_y=2), dict(axis_x=0, axis_y=True), dict(axis_x=2, axis_y=False), dict(axis_x=False, axis_y=3),
    dict(axis_x='0', axis_y=1), dict(axis_x=None, axis_y=1),
    dict(axis_x=0, axis_y=1, nx=1), dict(axis_x=0, axis_y=1, ny=1), dict(axis_x=0, axis_y=1, nx=102), dict(axis_x=0, axis_y=1, ny=102),
    dict(axis_x=0, axis_y=1, nx=True), dict(axis_x=0, axis_y=1, nx=5.0), dict(axis_x=0, axis_y=1, ny=None),
    dict(axis_x=0, axis_y=1, x_range=(1, 1)), dict(axis_x=0, axis_y=1, x_range=(2, 1)), dict(axis_x=0, axis_y=1, y_range=(0, float('inf'))),
    dict(axis_x=0, axis_y=1, y_range=(float('nan'), 1)), dict(axis_x=0, axis_y=1, x_range=(0,)), dict(axis_x=0, axis_y=1, x_range='ab'),
    dict(axis_x=0, axis_y=1, x_range=(True, 2)), dict(axis_x=0, axis_y=1, y_range=(0, '1')),
])
def test_a_bad_request_is_refused_before_anything_is_read(kwargs):
    args = {'nx': 5, 'ny': 5, 'x_range': None, 'y_range': None, **kwargs}
    with pytest.raises(runner.SliceRefused):
        runner.check_request(args['axis_x'], args['axis_y'], args['nx'], args['ny'], args['x_range'], args['y_range'])


@pytest.mark.parametrize('nx,ny', [(2, 2), (101, 101), (2, 101), (21, 21)])
def test_the_limits_themselves_are_allowed(nx, ny):
    runner.check_request(0, 3, nx, ny, (-1.0, 1.0), None)


def test_a_slice_refusal_is_a_trace_rerun_refusal_for_callers_that_catch_the_family():
    assert issubclass(runner.SliceRefused, TraceRerunRefused)


@pytest.mark.parametrize('stored_value', [0.0, 0.3, -0.7, 1.0, -1.0, 0.123456789])
@pytest.mark.parametrize('n', [2, 3, 5, 21, 101])
def test_the_grid_contains_the_stored_value_exactly_and_runs_from_low_to_high(stored_value, n):
    grid = runner.grid_through(stored_value, -1.0, 1.0, n)
    assert stored_value in grid and len(grid) == n
    assert all(a < b for a, b in zip(grid, grid[1:]))
    assert abs(grid[0] - (-1.0)) < 0.5 * (2.0 / (n - 1)) + 1e-12 and abs(grid[-1] - 1.0) < 0.5 * (2.0 / (n - 1)) + 1e-12


def test_a_stored_value_that_is_already_a_grid_point_leaves_the_grid_even():
    assert runner.grid_through(0.0, -1.0, 1.0, 5) == [-1.0, -0.5, 0.0, 0.5, 1.0]


@pytest.mark.parametrize('stored_value', [1.5, -1.5, 1.0000001])
def test_a_range_that_does_not_contain_the_stored_value_is_refused(stored_value):
    with pytest.raises(runner.SliceRefused, match='does not contain'):
        runner.grid_through(stored_value, -1.0, 1.0, 5)


def test_the_stored_value_at_the_end_of_the_range_is_allowed():
    assert runner.grid_through(1.0, -1.0, 1.0, 5)[-1] == 1.0


# ── the allow-list ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

GOOD = {'slice_id': 3, 'axis_x': 0, 'axis_y': 2, 'delta_parameterization': 'bicycle', 'x_values': [-1.0, 0.0, 1.0], 'y_values': [-2.0, 2.0],
        'weighted_norm': [0.5, 0.1, 0.5, 0.6, 0.4, 0.6], 'collides': [False, True, False, False, True, False], 'held_delta': [0.0, 0.1, 2.0, 0.2]}


def test_the_served_fields_are_exactly_the_models():
    assert list(served.select_slice(GOOD)) == [f for f in SensitivitySlice.model_fields if f not in ('axis_x_label', 'axis_y_label', 'delta_labels')]
    assert set(SensitivityResponse.model_fields) == {'scenario_id', 'stress_run_id', 'stored_delta_parameterization', 'slices', 'truncated'}


def test_a_good_row_is_served_as_it_is_with_the_stored_cell_found():
    out = served.select_slice(GOOD)
    assert {k: v for k, v in out.items() if not k.startswith('stored_')} == GOOD
    assert (out['stored_x_index'], out['stored_y_index']) == (1, 1)


def test_the_stored_cell_is_null_when_the_stored_value_is_not_on_the_grid():
    out = served.select_slice({**GOOD, 'held_delta': [0.5, 0.1, 2.0, 0.2]})
    assert out['stored_x_index'] is None and out['stored_y_index'] == 1


@pytest.mark.parametrize('field,bad', [
    ('slice_id', -1), ('slice_id', True), ('slice_id', 1.5), ('slice_id', '3'),
    ('axis_x', 4), ('axis_x', -1), ('axis_x', True), ('axis_x', 1.0), ('axis_x', '0'),
    ('delta_parameterization', 'quartic'), ('delta_parameterization', 3), ('delta_parameterization', None),
    ('held_delta', [0, 0, 0]), ('held_delta', [0, 0, 0, 0, 0]), ('held_delta', [0, 0, 0, float('nan')]), ('held_delta', [0, 0, 0, True]), ('held_delta', 'abcd'),
])
def test_a_field_of_the_wrong_type_is_null_and_the_others_are_kept(field, bad):
    out = served.select_slice({**GOOD, field: bad})
    assert out[field] is None
    unaffected = {'slice_id', 'delta_parameterization', 'held_delta'} if field in ('slice_id', 'delta_parameterization', 'held_delta') else {'axis_x'}
    for key in unaffected - {field}:
        assert out[key] == GOOD[key]


@pytest.mark.parametrize('axis', ['x_values', 'y_values'])
@pytest.mark.parametrize('bad', [[1.0], [0.0] * 102, [float(i) for i in range(102)], [0.0, float('nan')], [0.0, float('inf')], [1.0, 0.0], [0.0, 0.0], [0.0, True], ['a', 'b'], 'ab', None, 3])
def test_an_axis_that_is_not_an_increasing_list_of_numbers_makes_the_grid_null(axis, bad):
    out = served.select_slice({**GOOD, axis: bad})
    assert out[axis] is None and out['weighted_norm'] is None and out['collides'] is None
    assert out['held_delta'] == GOOD['held_delta']


@pytest.mark.parametrize('axis', ['x_values', 'y_values'])
def test_an_axis_of_the_most_values_the_table_allows_is_served_and_one_more_is_not(axis):
    # increasing numbers, so that only the length decides; the other axis has the two values the minimum allows
    other = 'y_values' if axis == 'x_values' else 'x_values'
    for n, served_as_a_grid in ((served.MAX_POINTS, True), (served.MAX_POINTS + 1, False)):
        values = [float(i) for i in range(n)]
        cells = n * 2
        out = served.select_slice({**GOOD, axis: values, other: [0.0, 1.0], 'weighted_norm': [0.5] * cells, 'collides': [True] * cells})
        if served_as_a_grid:
            assert out[axis] == values and len(out['weighted_norm']) == cells == len(out['collides'])
        else:
            assert out[axis] is None and out['weighted_norm'] is None and out['collides'] is None
        assert out[other] == [0.0, 1.0]


@pytest.mark.parametrize('field,bad', [('weighted_norm', [0.5] * 5), ('weighted_norm', [0.5] * 7), ('weighted_norm', 'abc'), ('weighted_norm', None),
                                       ('collides', [True] * 5), ('collides', [True] * 7), ('collides', 'abc'), ('collides', None)])
def test_a_grid_with_the_wrong_number_of_cells_is_null_on_its_own(field, bad):
    out = served.select_slice({**GOOD, field: bad})
    other = 'collides' if field == 'weighted_norm' else 'weighted_norm'
    assert out[field] is None and out[other] == GOOD[other]


def test_a_cell_that_is_not_a_number_or_not_a_verdict_is_null_and_the_others_are_kept():
    out = served.select_slice({**GOOD, 'weighted_norm': [0.5, float('nan'), -1.0, float('inf'), True, 'x'], 'collides': [True, 1, 'yes', None, False, True]})
    assert out['weighted_norm'] == [0.5, None, None, None, None, None]
    assert out['collides'] == [True, None, None, None, False, True]


def test_one_component_against_itself_is_not_a_slice():
    out = served.select_slice({**GOOD, 'axis_x': 1, 'axis_y': 1})
    assert out['axis_x'] is None and out['axis_y'] is None


def test_a_missing_column_is_null_and_an_extra_one_is_never_served():
    out = served.select_slice({'axis_x': 0, 'pipeline_run_id': 'x', 'scene_fingerprint': 'f'})
    assert out['x_values'] is None and 'pipeline_run_id' not in out and 'scene_fingerprint' not in out


def test_the_module_needs_nothing_the_deployed_api_lacks():
    import ast
    tree = ast.parse(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'api', 'sensitivity.py')).read())
    imported = {n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    imported |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert imported <= {'math'}


def test_the_limits_agree_with_the_tables_checks():
    assert (served.MIN_POINTS, served.MAX_POINTS, served.COMPONENTS) == (runner.MIN_POINTS, runner.MAX_POINTS, runner.COMPONENTS) == (2, 101, 4)
    assert served.MAX_SLICES == 12


# ── the runner, on a database ───────────────────────────────────────────────────────────────────────────────────────────────────


def _true_norm(scene, stored_row):
    """The weighted norm of the stored perturbation as the stored result recorded it: the same arithmetic the search ended with."""
    from src.optimization.perturbation_space import PerturbationSpace
    states, validity, types = scene
    return float(PerturbationSpace(states, validity, types, 0, stored_row['target_idx']).weighted_norm(np.array(stored_row['delta'], dtype=np.float32)))


def _insert(c, s, scene=None, **over):
    """_insert_scenario with the stored result's own weighted norm as its min_perturbation (the default there is a placeholder)."""
    if 'min_perturbation' not in over:
        over['min_perturbation'] = _true_norm(scene or _scene(), s)
    return _insert_scenario(c, s, **over)


@pytest.fixture
def conn(tr_db, stored):
    _, c = tr_db
    with c.cursor() as cur:
        cur.execute('DELETE FROM sensitivity_slices')
        cur.execute('DELETE FROM search_runs')
        cur.execute('DELETE FROM scenario_scores')
    _insert(c, stored)
    return c


def _slices(c):
    with c.cursor() as cur:
        cur.execute('SELECT slice_id, scenario_id, stress_run_id, scene_fingerprint, sdc_idx, axis_x, axis_y, delta_parameterization, x_values, y_values, '
                    'weighted_norm, collides, held_delta, pipeline_run_id FROM sensitivity_slices ORDER BY axis_x, axis_y')
        return cur.fetchall()


@requires_db
def test_a_slice_is_recorded_around_the_stored_result_with_the_stored_point_as_a_cell(conn, stored):
    out = runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=5, ny=7)
    assert out['status'] == 'recorded' and out['cells'] == 35 and out['stored_point_collides'] is True
    [r] = _slices(conn)
    assert r[1:3] == ('t1', stored['stress_run_id']) and r[3] == stored['fingerprint'] and r[4] == 0 and r[5:8] == (0, 1, 'bicycle')
    xs, ys, norms, collides, held = r[8], r[9], r[10], r[11], r[12]
    assert (len(xs), len(ys), len(norms), len(collides)) == (5, 7, 35, 35)
    assert held == stored['delta']
    ix, iy = xs.index(stored['delta'][0]), ys.index(stored['delta'][1])        # the stored point is a grid point, exactly
    assert collides[iy * 5 + ix] is True
    assert out['collisions'] == sum(collides) and 0 < out['collisions'] < out['cells']      # neither none of the cells nor all of them
    assert xs[0] == pytest.approx(-3.0, abs=0.5) and xs[-1] == pytest.approx(3.0, abs=0.5)     # the search's bounds for component 0, give or take half a step
    assert all(n >= 0 and math.isfinite(n) for n in norms) and any(not c for c in collides)


@requires_db
def test_a_cell_is_the_weighted_norm_and_the_exact_verdict_of_that_perturbation(conn, stored):
    from src.danger.collision_detector import check_collision_trajectory
    from src.optimization.perturbation_space import PerturbationSpace
    states, validity, types = _scene()
    space = PerturbationSpace(states, validity, types, 0, stored['target_idx'])
    runner.record_sensitivity_slice(conn, 't1', states, validity, types, 0, axis_x=1, axis_y=3, nx=4, ny=3, x_range=(-0.1, 0.2), y_range=(-0.1, 0.1))
    [r] = _slices(conn)
    xs, ys, norms, collides = r[8], r[9], r[10], r[11]
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            delta = np.array(stored['delta'], dtype=np.float32)
            delta[1], delta[3] = x, y
            assert collides[j * len(xs) + i] == bool(check_collision_trajectory(space.apply(delta), validity, 0, stored['target_idx'])[0])
            assert norms[j * len(xs) + i] == pytest.approx(space.weighted_norm(delta), abs=1e-12)


@requires_db
def test_the_two_components_not_on_an_axis_are_held_at_the_stored_values(conn, stored, monkeypatch):
    from src.optimization.perturbation_space import PerturbationSpace
    seen = []
    real = PerturbationSpace.apply
    monkeypatch.setattr(PerturbationSpace, 'apply', lambda self, d: seen.append(np.asarray(d, dtype=np.float32).copy()) or real(self, d))
    runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=3, nx=3, ny=3)
    assert not seen[0].any()                       # the space's own zero-perturbation baseline replay, made when it is built
    seen = seen[1:]
    assert len(seen) == 1 + 9                      # the stored point, then the nine cells
    held = np.array(stored['delta'], dtype=np.float32)
    for d in seen:
        assert d[1] == held[1] and d[2] == held[2]


@requires_db
def test_the_default_range_is_the_searchs_bounds_for_that_component(conn):
    runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=2, axis_y=3, nx=3, ny=3)
    [r] = _slices(conn)
    assert r[8][0] == pytest.approx(-1.5, abs=0.8) and r[9][-1] == pytest.approx(0.1, abs=0.06)


@requires_db
def test_it_writes_only_sensitivity_slices(conn):
    before = _fingerprint_of_the_results(conn)
    runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3)
    assert _fingerprint_of_the_results(conn) == before
    assert _count(conn, 'sensitivity_slices') == 1
    for table in ('search_runs', 'search_traces', 'pipeline_runs', 'pipeline_run_results', 'shards'):
        assert _count(conn, table) == 0, table


@requires_db
def test_the_same_axes_again_are_left_alone_and_nothing_is_evaluated(conn, monkeypatch):
    from src.optimization.perturbation_space import PerturbationSpace
    first = runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3)
    before = _slices(conn)
    calls = []
    real = PerturbationSpace.apply
    monkeypatch.setattr(PerturbationSpace, 'apply', lambda self, d: calls.append(1) or real(self, d))
    again = runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=7, ny=5, x_range=(-1, 1))
    assert again['status'] == 'conflict' and again['slice_id'] == first['slice_id'] and calls == []
    assert _slices(conn) == before


@requires_db
def test_a_second_writer_that_gets_in_between_the_check_and_the_insert_wins_and_the_run_is_reported_as_a_conflict(conn, tr_db, stored):
    name, _ = tr_db
    other = psycopg2.connect(**{**_server(), 'dbname': name})
    raced = []

    class Proxy:
        """The runner's connection, except that just before its INSERT another writer commits the same (scenario, result, axes)."""
        def __init__(self, c):
            self._c = c

        def __getattr__(self, attr):
            return getattr(self._c, attr)

        def cursor(self, *args, **kwargs):
            class Spy(psycopg2.extensions.cursor):
                def execute(spy, query, vars=None):
                    if 'INSERT INTO sensitivity_slices' in query and not raced:
                        raced.append(_put(other, stored['stress_run_id'], axes=(0, 1)))
                        other.commit()
                        raced.append(_slices(other))
                        other.commit()
                    return super().execute(query, vars)
            return self._c.cursor(*args, cursor_factory=Spy, **kwargs)

    try:
        out = runner.record_sensitivity_slice(Proxy(conn), 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=5, ny=5)
        winner_id, winner_rows = raced
        assert out['status'] == 'conflict' and out['slice_id'] is None
        assert 'recorded meanwhile' in out['reason']
        assert len(winner_rows) == 1 and winner_rows[0][0] == winner_id
        assert _slices(conn) == winner_rows                       # the other writer's row, untouched: its 3 by 2 grid, not this run's 5 by 5
        assert _slices(conn)[0][8] == [-1.0, 0.0, 1.0]
    finally:
        other.close()


@requires_db
def test_the_axes_the_other_way_round_are_another_slice(conn):
    runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3)
    out = runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=1, axis_y=0, nx=3, ny=3)
    assert out['status'] == 'recorded' and _count(conn, 'sensitivity_slices') == 2


@requires_db
def test_a_slice_of_a_result_that_has_been_replaced_does_not_block_a_slice_of_the_new_one(conn, stored):
    runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3)
    with conn.cursor() as cur:
        cur.execute("UPDATE scenario_scores SET stress_run_id = 'ffffffffffffffff' WHERE scenario_id = 't1'")
    out = runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3)
    assert out['status'] == 'recorded' and sorted(r[2] for r in _slices(conn)) == sorted([stored['stress_run_id'], 'ffffffffffffffff'])


class _Boom(AssertionError):
    pass


def _no_cell_may_be_evaluated(monkeypatch):
    """Any perturbation but the space's own zero-perturbation baseline replay (made when it is built) fails the test."""
    from src.optimization.perturbation_space import PerturbationSpace
    real = PerturbationSpace.apply

    def apply(self, d):
        if np.any(np.asarray(d)):
            raise _Boom('a cell was evaluated')
        return real(self, d)
    monkeypatch.setattr(PerturbationSpace, 'apply', apply)


@requires_db
@pytest.mark.parametrize('kwargs,message', [
    (dict(axis_x=1, axis_y=1), 'different'), (dict(axis_x=0, axis_y=1, nx=1), 'grid points'), (dict(axis_x=0, axis_y=1, ny=500), 'grid points'),
    (dict(axis_x=0, axis_y=1, x_range=(1, 0)), 'low below high'), (dict(axis_x=0, axis_y=1, x_range=(-10, 10)), 'outside the search bounds'),
    (dict(axis_x=0, axis_y=1, y_range=(-5, 5)), 'outside the search bounds'),
    (dict(axis_x=0, axis_y=1, x_range=(-3.5, 1.0)), 'outside the search bounds'), (dict(axis_x=0, axis_y=1, x_range=(0.0, 3.5)), 'outside the search bounds'),
    (dict(axis_x=3, axis_y=2, y_range=(-1.0, 1.6)), 'outside the search bounds'), (dict(axis_x=3, axis_y=2, y_range=(-1.6, 1.0)), 'outside the search bounds'), (dict(axis_x=0, axis_y=1, x_range=(1.5, 2.5)), 'does not contain'),
    (dict(axis_x=3, axis_y=2, y_range=(0.5, 1.0)), 'does not contain'),
])
def test_a_request_that_cannot_be_drawn_is_refused_before_one_cell_is_evaluated(conn, monkeypatch, kwargs, message):
    _no_cell_may_be_evaluated(monkeypatch)
    with pytest.raises(runner.SliceRefused, match=message):
        runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, **{'nx': 3, 'ny': 3, **kwargs})
    assert _count(conn, 'sensitivity_slices') == 0


@requires_db
def test_a_scene_that_is_not_the_one_searched_is_refused(conn, monkeypatch):
    _no_cell_may_be_evaluated(monkeypatch)
    states, validity, types = _scene()
    states[1, :, 1] = 2.2
    with pytest.raises(TraceRerunRefused, match='scene'):
        runner.record_sensitivity_slice(conn, 't1', states, validity, types, 0, axis_x=0, axis_y=1, nx=3, ny=3)
    with pytest.raises(TraceRerunRefused, match='SDC'):
        runner.record_sensitivity_slice(conn, 't1', *_scene(), 1, axis_x=0, axis_y=1, nx=3, ny=3)
    assert _count(conn, 'sensitivity_slices') == 0


@requires_db
def test_no_stored_search_and_no_such_scenario_are_refused(conn):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO scenario_scores (scenario_id, shard, fragility_score) VALUES ('bare', 'a', 1.0)")
    with pytest.raises(TraceRerunRefused, match='no stored search'):
        runner.record_sensitivity_slice(conn, 'bare', *_scene(), 0, axis_x=0, axis_y=1)
    with pytest.raises(TraceRerunRefused, match='no such scenario'):
        runner.record_sensitivity_slice(conn, 'nope', *_scene(), 0, axis_x=0, axis_y=1)


@requires_db
def test_a_stored_result_that_does_not_say_its_model_or_challenger_or_has_the_wrong_number_of_components_is_refused(conn, stored):
    no_model = {k: v for k, v in stored['provenance'].items() if k != 'delta_parameterization'}
    _insert(conn, {**stored, 'provenance': no_model})
    with pytest.raises(runner.SliceRefused, match='which model'):
        runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3)
    _insert(conn, stored, target_idx=None)
    with pytest.raises(runner.SliceRefused, match='which challenger'):
        runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3)
    _insert(conn, {**stored, 'delta': stored['delta'][:3]}, min_perturbation=0.5)
    with pytest.raises(runner.SliceRefused, match='3 components'):
        runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3)
    assert _count(conn, 'sensitivity_slices') == 0


@requires_db
def test_a_stored_point_that_does_not_collide_in_this_code_is_refused_and_nothing_is_written(conn, stored):
    _insert(conn, {**stored, 'delta': [0.0, 0.0, 0.0, 0.0]}, min_perturbation=0.0)
    with pytest.raises(runner.SliceRefused, match='does not collide'):
        runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3)
    assert _count(conn, 'sensitivity_slices') == 0


@requires_db
def test_a_replay_that_is_not_faithful_is_refused(conn, monkeypatch):
    from src.optimization import perturbation_space as ps
    def refuse(*a, **k):
        raise ps.ReplayFidelityError('drift', 9.0, False)
    monkeypatch.setattr(ps, 'PerturbationSpace', refuse)
    with pytest.raises(runner.SliceRefused, match='not faithful'):
        runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3)


@requires_db
def test_missing_table_is_a_clear_error(bare_db):
    _, c = bare_db
    with c.cursor() as cur:
        cur.execute('DELETE FROM scenario_scores')
    with pytest.raises(ManifestTablesMissing, match='sensitivity_slices|search_runs'):
        runner.record_sensitivity_slice(c, 't1', *_scene(), 0, axis_x=0, axis_y=1)


@requires_db
def test_the_table_only_slice_without_search_runs_is_a_clear_error_too(tr_db, stored):
    name, owner = tr_db
    with owner.cursor() as cur:
        cur.execute('ALTER TABLE sensitivity_slices RENAME TO sensitivity_slices_away')
    try:
        with pytest.raises(ManifestTablesMissing, match='sensitivity_slices'):
            runner.record_sensitivity_slice(owner, 't1', *_scene(), 0, axis_x=0, axis_y=1)
    finally:
        with owner.cursor() as cur:
            cur.execute('ALTER TABLE sensitivity_slices_away RENAME TO sensitivity_slices')


@requires_db
def test_it_commits_on_a_connection_that_is_not_autocommit(conn, tr_db):
    name, _ = tr_db
    own = psycopg2.connect(**{**_server(), 'dbname': name})
    try:
        runner.record_sensitivity_slice(own, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3)
        assert _count(conn, 'sensitivity_slices') == 1
    finally:
        own.close()


@requires_db
def test_the_pipeline_run_it_belongs_to_is_recorded_when_given(conn):
    from src.scoring import manifest
    run = manifest.begin_run(conn, 'rerun', command_line='x')
    runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3, pipeline_run_id=run)
    with conn.cursor() as cur:
        cur.execute('SELECT pipeline_run_id::text FROM sensitivity_slices')
        assert cur.fetchone()[0] == run


@requires_db
def test_the_largest_grid_the_table_allows_is_recorded(conn):
    out = runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=101, ny=2)
    assert out['status'] == 'recorded' and out['cells'] == 202


# ── the route ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch, conn):
    with _client(monkeypatch, conn) as c:
        yield c


def _get(client, scenario='t1'):
    return client.get(f'/scenarios/{scenario}/slice')


def _put(c, stress_run_id, *, scenario='t1', axes=(0, 1), fingerprint='STORED', sdc=0, parameterization='bicycle', xs=(-1.0, 0.0, 1.0), ys=(-1.0, 1.0),
         held=(0.0, 1.0, 0.5, 0.25), stored_row=None):
    fp = (stored_row or _stored())['fingerprint'] if fingerprint == 'STORED' else fingerprint
    n = len(xs) * len(ys)
    with c.cursor() as cur:
        cur.execute("""INSERT INTO sensitivity_slices (scenario_id, stress_run_id, scene_fingerprint, sdc_idx, axis_x, axis_y, delta_parameterization,
                                                       x_values, y_values, weighted_norm, collides, held_delta)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING slice_id""",
                    (scenario, stress_run_id, fp, sdc, axes[0], axes[1], parameterization, list(xs), list(ys),
                     [0.1 * (k + 1) for k in range(n)], [k % 2 == 0 for k in range(n)], list(held)))
        return cur.fetchone()[0]


@requires_db
def test_with_nothing_recorded_the_list_is_empty_and_the_status_is_200(client, stored):
    r = _get(client)
    assert r.status_code == 200
    assert r.json() == {'scenario_id': 't1', 'stress_run_id': stored['stress_run_id'],
                        'stored_delta_parameterization': stored['provenance']['delta_parameterization'], 'slices': [], 'truncated': False}


@requires_db
def test_an_unknown_scenario_is_404(client):
    assert _get(client, 'nope').status_code == 404


@requires_db
def test_a_recorded_slice_is_served_with_its_own_labels_the_flat_layout_and_the_stored_cell(conn, client, stored):
    runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=3, nx=5, ny=3)
    [s] = _get(client).json()['slices']
    assert set(s) == set(SensitivitySlice.model_fields)
    labels = DELTA_LABELS_BY_PARAMETERIZATION['bicycle']
    assert (s['axis_x'], s['axis_y'], s['axis_x_label'], s['axis_y_label']) == (0, 3, labels[0], labels[3])
    assert s['delta_labels'] == list(labels)
    assert len(s['x_values']) == 5 and len(s['y_values']) == 3 and len(s['weighted_norm']) == 15 == len(s['collides'])
    assert s['held_delta'] == stored['delta']
    assert s['x_values'][s['stored_x_index']] == stored['delta'][0] and s['y_values'][s['stored_y_index']] == stored['delta'][3]
    assert s['collides'][s['stored_y_index'] * 5 + s['stored_x_index']] is True


@requires_db
def test_a_slice_of_a_result_that_has_been_replaced_is_not_listed(conn, client, stored):
    _put(conn, stored['stress_run_id'])
    _put(conn, '0123456789abcdef', axes=(1, 0))
    assert [s['axis_x'] for s in _get(client).json()['slices']] == [0]
    with conn.cursor() as cur:
        cur.execute("UPDATE scenario_scores SET stress_run_id = 'ffffffffffffffff' WHERE scenario_id = 't1'")
    assert _get(client).json()['slices'] == []


@requires_db
def test_a_result_with_no_run_id_lists_nothing_and_asks_the_table_nothing(conn, client, monkeypatch):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO scenario_scores (scenario_id, shard, fragility_score) VALUES ('bare', 'a', 1.0)")
    from src.api import routes
    asked = []
    real = routes._table_exists
    monkeypatch.setattr(routes, '_table_exists', lambda cur, table: asked.append(table) or real(cur, table))
    body = _get(client, 'bare').json()
    assert body['slices'] == [] and body['truncated'] is False and asked == []


@requires_db
def test_another_scenarios_slices_are_not_listed(conn, client, stored):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO scenario_scores (scenario_id, shard, fragility_score, stress_run_id) VALUES ('other', 'a', 1.0, %s)", (stored['stress_run_id'],))
    _put(conn, stored['stress_run_id'], scenario='other')
    assert _get(client).json()['slices'] == [] and len(_get(client, 'other').json()['slices']) == 1


@requires_db
def test_a_slice_of_another_scene_or_sdc_index_is_not_listed(conn, client, stored):
    _put(conn, stored['stress_run_id'], axes=(0, 1), fingerprint='some-other-scene')
    _put(conn, stored['stress_run_id'], axes=(0, 2), sdc=3)
    _put(conn, stored['stress_run_id'], axes=(0, 3))
    assert [(s['axis_x'], s['axis_y']) for s in _get(client).json()['slices']] == [(0, 3)]


@requires_db
def test_a_result_stored_without_a_fingerprint_lists_what_is_there(conn, client, stored):
    with conn.cursor() as cur:
        cur.execute("UPDATE scenario_scores SET scene_fingerprint = NULL, sdc_idx = NULL WHERE scenario_id = 't1'")
    _put(conn, stored['stress_run_id'], fingerprint='whatever', sdc=7)
    assert len(_get(client).json()['slices']) == 1


@requires_db
def test_a_slice_recorded_without_a_fingerprint_or_sdc_index_is_listed(conn, client, stored):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO sensitivity_slices (scenario_id, stress_run_id, axis_x, axis_y, delta_parameterization, x_values, y_values, weighted_norm, "
                    "collides, held_delta) VALUES ('t1', %s, 0, 1, 'bicycle', '{0,1}', '{0,1}', '{1,2,3,4}', '{t,f,f,t}', '{0,0,0,0}')", (stored['stress_run_id'],))
    [s] = _get(client).json()['slices']
    assert s['stored_x_index'] == 0 and s['stored_y_index'] == 0 and s['collides'] == [True, False, False, True]


@requires_db
def test_the_order_is_by_axes(conn, client, stored):
    for axes in [(2, 3), (0, 1), (1, 0), (0, 3), (3, 2)]:
        _put(conn, stored['stress_run_id'], axes=axes)
    assert [(s['axis_x'], s['axis_y']) for s in _get(client).json()['slices']] == [(0, 1), (0, 3), (1, 0), (2, 3), (3, 2)]


@requires_db
def test_the_labels_come_from_each_slices_own_model(conn, client, stored):
    _put(conn, stored['stress_run_id'], axes=(0, 1), parameterization='bicycle')
    _put(conn, stored['stress_run_id'], axes=(0, 2), parameterization='linear')
    a, b = _get(client).json()['slices']
    assert a['axis_y_label'] == DELTA_LABELS_BY_PARAMETERIZATION['bicycle'][1]
    assert b['axis_y_label'] == DELTA_LABELS_BY_PARAMETERIZATION['linear'][2] and b['delta_parameterization'] == 'linear'
    assert a['delta_labels'] == list(DELTA_LABELS_BY_PARAMETERIZATION['bicycle']) and b['delta_labels'] == list(DELTA_LABELS_BY_PARAMETERIZATION['linear'])


@requires_db
def test_the_stored_results_model_is_served_only_if_it_is_one_the_api_knows(conn, client):
    with conn.cursor() as cur:
        cur.execute("UPDATE scenario_scores SET search_provenance = jsonb_set(search_provenance, '{delta_parameterization}', '\"quartic\"') WHERE scenario_id = 't1'")
    assert _get(client).json()['stored_delta_parameterization'] is None


@requires_db
def test_every_ordered_pair_of_components_is_listed_and_that_is_not_truncated(conn, client, stored):
    pairs = [(a, b) for a in range(4) for b in range(4) if a != b]
    assert len(pairs) == served.MAX_SLICES == 12
    for axes in pairs:
        _put(conn, stored['stress_run_id'], axes=axes)
    body = _get(client).json()
    assert [(s['axis_x'], s['axis_y']) for s in body['slices']] == sorted(pairs) and body['truncated'] is False


@requires_db
def test_the_same_axes_cannot_be_recorded_twice_for_one_result_so_the_limit_cannot_be_passed(conn, stored):
    _put(conn, stored['stress_run_id'], axes=(0, 1))
    with pytest.raises(psycopg2.errors.UniqueViolation):
        _put(conn, stored['stress_run_id'], axes=(0, 1))


@requires_db
def test_thirteen_rows_list_twelve_and_say_it_cut_one_when_the_unique_key_is_lifted(tr_db, conn, client, stored):
    # The table's unique key makes 13 slices around one stored result impossible, so the real limit is tested on a copy of the situation in which
    # the key is lifted for the length of the test: the twelve ordered pairs plus a second slice of the first pair.
    _, owner = tr_db
    with owner.cursor() as cur:
        cur.execute("SELECT conname FROM pg_constraint WHERE conrelid = 'sensitivity_slices'::regclass AND contype = 'u'")
        [(key,)] = cur.fetchall()
        cur.execute(f'ALTER TABLE sensitivity_slices DROP CONSTRAINT "{key}"')
    try:
        pairs = [(a, b) for a in range(4) for b in range(4) if a != b]
        for axes in pairs:
            _put(conn, stored['stress_run_id'], axes=axes)
        body = _get(client).json()
        assert len(body['slices']) == 12 and body['truncated'] is False                 # twelve rows: nothing cut
        _put(conn, stored['stress_run_id'], axes=(0, 1))                                 # the thirteenth
        body = _get(client).json()
        assert len(body['slices']) == 12 and body['truncated'] is True
        assert [(s['axis_x'], s['axis_y']) for s in body['slices']] == sorted(pairs + [(0, 1)])[:12]      # the first twelve in the fixed order
    finally:
        with owner.cursor() as cur:
            cur.execute('DELETE FROM sensitivity_slices')
            cur.execute(f'ALTER TABLE sensitivity_slices ADD CONSTRAINT "{key}" UNIQUE (scenario_id, stress_run_id, axis_x, axis_y)')


@requires_db
def test_the_guard_says_it_cut_when_more_matched_than_the_limit(conn, client, stored, monkeypatch):
    # With the limit at every ordered pair the guard cannot fire (the table's unique key); with a smaller one it shows what it does.
    monkeypatch.setattr(served, 'MAX_SLICES', 3)
    pairs = [(a, b) for a in range(4) for b in range(4) if a != b]
    for axes in pairs[:3]:
        _put(conn, stored['stress_run_id'], axes=axes)
    body = _get(client).json()
    assert len(body['slices']) == 3 and body['truncated'] is False
    _put(conn, stored['stress_run_id'], axes=pairs[3])
    body = _get(client).json()
    assert len(body['slices']) == 3 and body['truncated'] is True
    assert [(s['axis_x'], s['axis_y']) for s in body['slices']] == sorted(pairs[:4])[:3]


@requires_db
def test_the_query_asks_for_one_slice_more_than_the_limit_and_no_more(monkeypatch, conn, stored):
    asked = []

    class Proxy:
        def __init__(self, c):
            self._c = c

        def __getattr__(self, name):
            return getattr(self._c, name)

        def cursor(self, *args, cursor_factory=None, **kwargs):
            class Spy(cursor_factory or psycopg2.extensions.cursor):
                def execute(spy, query, vars=None):
                    if vars and 'FROM sensitivity_slices' in query:
                        asked.append(vars[-1])
                    return super().execute(query, vars)
            return self._c.cursor(*args, cursor_factory=Spy, **kwargs)

    with _client(monkeypatch, Proxy(conn)) as c:
        _get(c)
    assert asked == [13]                             # exactly one more than the 12 it lists
    assert served.MAX_SLICES + 1 == 13


@requires_db
def test_values_that_are_not_what_they_should_be_are_null_not_an_error(conn, client, stored):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO sensitivity_slices (scenario_id, stress_run_id, axis_x, axis_y, delta_parameterization, x_values, y_values, weighted_norm, "
                    "collides, held_delta) VALUES ('t1', %s, 0, 1, 'linear', '{1,0}', '{0,1}', '{1,2,3,4}', '{t,f,f,t}', '{0,0,0,\"NaN\"}')", (stored['stress_run_id'],))
        cur.execute("INSERT INTO sensitivity_slices (scenario_id, stress_run_id, axis_x, axis_y, delta_parameterization, x_values, y_values, weighted_norm, "
                    "collides, held_delta) VALUES ('t1', %s, 1, 2, 'bicycle', '{0,1}', '{0,1}', '{1,\"NaN\",3,4}', '{t,f,f,t}', '{0,0,0,0}')", (stored['stress_run_id'],))
    r = _get(client)
    assert r.status_code == 200
    a, b = r.json()['slices']
    assert a['x_values'] is None and a['weighted_norm'] is None and a['collides'] is None and a['held_delta'] is None and a['y_values'] == [0.0, 1.0]
    assert b['weighted_norm'] == [1.0, None, 3.0, 4.0] and b['collides'] == [True, False, False, True]


@requires_db
def test_the_response_claims_nothing_about_safety_regions_or_boundaries(conn, client, stored):
    _put(conn, stored['stress_run_id'])
    text = _get(client).text.lower()
    for word in ('safe', 'safety', 'region', 'boundary', 'probab', 'robust', 'smallest', 'minimum'):
        assert word not in text, word


@requires_db
def test_without_the_table_the_list_is_empty_not_an_error(monkeypatch, bare_db, stored):
    _, c = bare_db
    _insert_scenario(c, stored)
    with _client(monkeypatch, c) as client:
        r = _get(client)
    assert r.status_code == 200 and r.json()['slices'] == [] and r.json()['truncated'] is False


@requires_db
def test_the_read_only_role_can_read_them(monkeypatch, conn, stored):
    _put(conn, stored['stress_run_id'])
    with conn.cursor() as cur:
        cur.execute('SET ROLE av_api_ro')
    try:
        with _client(monkeypatch, conn) as client:
            assert len(_get(client).json()['slices']) == 1
    finally:
        with conn.cursor() as cur:
            cur.execute('RESET ROLE')


@requires_db
def test_a_role_without_select_on_the_table_gets_an_empty_list_one_clean_log_line_and_a_usable_connection(monkeypatch, conn, deny_role, stored, caplog):
    _put(conn, stored['stress_run_id'])
    c = psycopg2.connect(**deny_role)
    try:
        with _client(monkeypatch, c) as client, caplog.at_level(logging.WARNING, logger='src.api.routes'):
            r = _get(client)
            assert r.status_code == 200, r.text
            assert r.json()['slices'] == [] and r.json()['truncated'] is False and r.json()['stress_run_id'] == stored['stress_run_id']
            assert _get(client).status_code == 200
        warnings = [x for x in caplog.records if x.levelno == logging.WARNING and x.name == 'src.api.routes']
        assert len(warnings) == 2
        message = warnings[0].getMessage()
        assert 'The sensitivity-slice table cannot be read by the API role' in message and 'DEPLOY.md' in message
        for secret in (deny_role['password'], deny_role['user'], 'sensitivity_slices', 'SELECT', 'host=', 'postgres'):
            assert secret not in message, secret
    finally:
        c.close()


# ── the stored result's model and norm ───────────────────────────────────────────────────────────────────────────────────────────


def _linear_scene():
    """A pedestrian challenger: the linear model, replayed (and searched) with the linear space."""
    states, validity, types = _scene()
    states[1, :, 5:7] = [0.8, 0.8]
    return states, validity, np.array([1, 2])


def _stored_linear():
    from src.scoring.batch_scorer import _stress_one
    from src.scoring import db
    states, validity, types = _linear_scene()
    result = _stress_one(states, validity, types, 0, de_kwargs=dict(BUDGET))
    assert result['status'] == 'ok' and result['collision'] and result['search_provenance']['delta_parameterization'] == 'linear'
    prov = result['search_provenance']
    return {
        'scenario_id': 't1', 'delta': [float(x) for x in result['delta']], 'target_idx': result['target_idx'], 'stress_method': result['method'],
        'provenance': prov, 'n_iter': prov['de_n_iter'], 'n_eval': prov['de_n_eval'],
        'stress_run_id': db.compute_stress_run_id('t1', result['target_idx'], result['delta'], result['method'], prov),
        'fingerprint': db.compute_scene_fingerprint(states, validity, types),
    }


@requires_db
def test_a_linear_stored_result_is_sliced_with_the_linear_model_and_labelled_as_linear(conn, monkeypatch):
    s = _stored_linear()
    _insert(conn, s, scene=_linear_scene())
    out = runner.record_sensitivity_slice(conn, 't1', *_linear_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3)
    assert out['status'] == 'recorded' and out['stored_point_collides'] is True
    [r] = _slices(conn)
    assert r[7] == 'linear'
    with _client(monkeypatch, conn) as c:
        [served_slice] = _get(c).json()['slices']
    assert served_slice['delta_parameterization'] == 'linear'
    assert served_slice['axis_x_label'] == DELTA_LABELS_BY_PARAMETERIZATION['linear'][0] and served_slice['delta_labels'] == list(DELTA_LABELS_BY_PARAMETERIZATION['linear'])


@requires_db
@pytest.mark.parametrize('recorded,scene_of,space_model', [('bicycle', _linear_scene, 'linear'), ('linear', _scene, 'bicycle')])
def test_a_stored_result_recorded_in_one_model_is_refused_when_its_challenger_is_replayed_in_the_other(conn, monkeypatch, recorded, scene_of, space_model):
    # the stored numbers are those of the model the space really has; only the recorded label is the other one
    s = _stored_linear() if space_model == 'linear' else _stored()
    wrong = {**s, 'provenance': {**s['provenance'], 'delta_parameterization': recorded}}
    _insert(conn, wrong, scene=scene_of())
    _no_cell_may_be_evaluated(monkeypatch)
    with pytest.raises(runner.SliceRefused, match=f'recorded as the {recorded} model.*{space_model} model') as e:
        runner.record_sensitivity_slice(conn, 't1', *scene_of(), 0, axis_x=0, axis_y=1, nx=3, ny=3)
    assert 'labels on the other' in str(e.value)
    assert _count(conn, 'sensitivity_slices') == 0


@requires_db
@pytest.mark.parametrize('factor,accepted', [(1.0, True), (1 + 5e-7, True), (1 - 5e-7, True), (1 + 2e-6, False), (1 - 2e-6, False), (1.01, False), (0.5, False), (2.0, False)])
def test_the_weighted_norm_of_the_stored_point_must_be_the_stored_one(conn, stored, factor, accepted):
    true = _true_norm(_scene(), stored)
    _insert(conn, stored, min_perturbation=true * factor)
    if accepted:
        assert runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3)['status'] == 'recorded'
    else:
        with pytest.raises(runner.SliceRefused, match='weighted norm of the stored perturbation'):
            runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3)
        assert _count(conn, 'sensitivity_slices') == 0


@requires_db
def test_a_stored_result_that_records_no_norm_is_refused(conn, stored):
    with conn.cursor() as cur:
        cur.execute("UPDATE scenario_scores SET min_perturbation = NULL WHERE scenario_id = 't1'")
    with pytest.raises(runner.SliceRefused, match='does not record its weighted norm'):
        runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3)
    with conn.cursor() as cur:
        cur.execute("UPDATE scenario_scores SET min_perturbation = 'NaN' WHERE scenario_id = 't1'")
    with pytest.raises(runner.SliceRefused, match='does not record its weighted norm'):
        runner.record_sensitivity_slice(conn, 't1', *_scene(), 0, axis_x=0, axis_y=1, nx=3, ny=3)
    assert _count(conn, 'sensitivity_slices') == 0


# ── the size of the response ─────────────────────────────────────────────────────────────────────────────────────────────────────


def _fill(c, stored_row, pairs, *, long_floats):
    """Slices of the largest size the table allows (101 by 101), with norms that print as long as a float64 can."""
    import random
    random.seed(7)
    xs = [-3 + 6 * i / 100 for i in range(101)]
    ys = [-1.5 + 3 * i / 100 for i in range(101)]
    n = 101 * 101
    with c.cursor() as cur:
        for a, b in pairs:
            norms = [random.random() * (1e-5 if long_floats else 1.0) for _ in range(n)]
            cur.execute("""INSERT INTO sensitivity_slices (scenario_id, stress_run_id, scene_fingerprint, sdc_idx, axis_x, axis_y, delta_parameterization,
                                                           x_values, y_values, weighted_norm, collides, held_delta)
                           VALUES ('t1', %s, %s, 0, %s, %s, 'bicycle', %s, %s, %s, %s, %s)""",
                        (stored_row['stress_run_id'], stored_row['fingerprint'], a, b, xs, ys, norms, [random.random() < 0.3 for _ in range(n)], [0.0, 0.1, 0.2, 0.3]))


@requires_db
def test_the_response_for_the_largest_slices_the_table_allows_stays_within_a_stated_size(conn, client, stored):
    """
    MEASURED on the test client (the bytes of the JSON response body, in process, on one machine; no network, no compression): one 101 by 101
    slice is 257 743 bytes with norms printed as ordinary floats and 277 050 with the longest ones (exponent form, 17 digits); the twelve
    slices the API lists at most (every ordered pair of components, 122 412 cells) are 3 091 219 and 3 322 799 bytes, built in about 0.09 s.
    The bounds below are those worst cases with a margin. They are not a limit the API enforces; they fail if a field is added that makes the
    response grow.
    """
    pairs = [(a, b) for a in range(4) for b in range(4) if a != b]
    _fill(conn, stored, pairs[:1], long_floats=True)
    r = _get(client)
    assert r.status_code == 200 and len(r.json()['slices']) == 1
    assert len(r.content) < 330_000
    assert len(gzip.compress(r.content)) < 150_000          # NOT what the API sends (it does not compress; see ROUTE-READS.md): what a proxy that compresses would
    _fill(conn, stored, pairs[1:], long_floats=True)
    r = _get(client)
    body = r.json()
    assert len(body['slices']) == 12 and body['truncated'] is False and all(len(s['weighted_norm']) == 10201 for s in body['slices'])
    assert len(r.content) < 4_000_000
