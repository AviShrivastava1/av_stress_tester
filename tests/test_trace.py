"""
The convergence trace, end to end: the offline recorder (src/scoring/search_runs.py), the rules for what is served (src/api/trace.py) and
GET /scenarios/{id}/trace.

Pinned here:

  * the rules: the tolerance (1e-6, strict), which differences are usable, how points are cleaned;
  * the recorder: it re-runs the stored search with the stored seed and budget, records a row of kind trace_rerun that names the stored
    result and its distance from it, records one point per generation, refuses a scene that is not the one searched, and changes no
    table but its own two;
  * the route: a trace only for the stored result that was re-run, only within the tolerance, only for the same scene, the newest of
    several; null (200) for every other state; 404 for an unknown scenario; and that it works as the read-only role and degrades, with
    one log line and no 500, for a role that cannot read the tables.

The database tests build a throw-away database and drop it after. They skip unless AV_ROBUSTNESS_DB=1.

Run:
    AV_ROBUSTNESS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_trace.py -q
"""

import json
import logging
import math
import os
import secrets
import sys
import uuid

import numpy as np
import psycopg2
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import db_pool, main as api_main, routes, trace as rules
from src.api.config import settings
from src.api.models import TraceResponse
from src.optimization.perturbation_space import PerturbationSpace
from src.scoring import db, search_runs
from src.scoring.batch_scorer import _stress_one
from src.scoring.manifest import ManifestTablesMissing

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(PROJECT, 'deploy', 'rerun_class_additive.sql')

requires_db = pytest.mark.skipif(
    os.environ.get('AV_ROBUSTNESS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

# ── the rules ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_the_tolerance_is_a_millionth_of_a_unit_and_the_reason_is_written_down():
    assert rules.TRACE_DELTA_TOLERANCE == 1e-6
    assert 'float32' in rules.__doc__ and 'spacing' in rules.__doc__ and 'a gate and not a measurement' in rules.__doc__


def test_points_come_back_in_generation_order_with_plain_types():
    out = rules.select_points([(3, 90, 1.5), (1, 30, None), (2, 60, 2.5)])
    assert out == [{'generation': 1, 'n_eval_cumulative': 30, 'best_weighted_norm': None},
                   {'generation': 2, 'n_eval_cumulative': 60, 'best_weighted_norm': 2.5},
                   {'generation': 3, 'n_eval_cumulative': 90, 'best_weighted_norm': 1.5}]


@pytest.mark.parametrize('best', [float('nan'), float('inf'), float('-inf'), 'x', True, [1.0], None])
def test_a_best_norm_that_is_not_a_finite_number_is_null(best):
    assert rules.select_points([(1, 10, best)])[0]['best_weighted_norm'] is None


@pytest.mark.parametrize('point', [(-1, 10, 1.0), (1, -10, 1.0), (1.5, 10, 1.0), (1, 10.5, 1.0), (True, 10, 1.0), (1, True, 1.0), ('1', 10, 1.0), (None, 10, 1.0)])
def test_a_point_without_whole_counts_is_dropped(point):
    assert rules.select_points([point, (2, 20, 1.0)]) == [{'generation': 2, 'n_eval_cumulative': 20, 'best_weighted_norm': 1.0}]


def test_no_more_points_than_the_pipeline_can_write():
    assert len(rules.select_points([(g, g, 1.0) for g in range(rules.MAX_POINTS + 50)])) == rules.MAX_POINTS


def test_the_module_needs_nothing_the_deployed_api_lacks():
    import ast
    tree = ast.parse(open(os.path.join(PROJECT, 'src', 'api', 'trace.py')).read())
    imported = {n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    imported |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert imported <= {'math'}


def test_the_differences_the_recorder_writes():
    assert search_runs.max_abs_delta_difference([1, 2, 3, 4], [1, 2, 3, 4]) == 0.0
    assert search_runs.max_abs_delta_difference([1, 2, 3, 4], [1, 2.5, 3, 3]) == 1.0
    assert search_runs.max_abs_delta_difference([1, 2, 3, 4], [1, 2, 3]) == math.inf
    assert search_runs.max_abs_delta_difference([], []) == math.inf


# ── a stored result, produced by the pipeline itself ─────────────────────────────────────────────────────────────────────────────

BUDGET = {'popsize': 5, 'maxiter': 4, 'tol': 1e-3, 'seed': 2}


def _scene():
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[1, :, 1] = 2.1
    return states, np.ones((2, 10), dtype=bool), np.ones(2, dtype=int)


def _stored(scenario_id='t1', use_autograd=False, **kw):
    """What the pipeline would have stored for the scene above, as the columns of a scenario_scores row."""
    states, validity, types = _scene()
    result = _stress_one(states, validity, types, 0, use_autograd=use_autograd, de_kwargs={**BUDGET, **kw})
    assert result['status'] == 'ok' and result['collision']
    prov = result['search_provenance']
    return {
        'scenario_id': scenario_id, 'delta': [float(x) for x in result['delta']], 'target_idx': result['target_idx'],
        'stress_method': result['method'], 'provenance': prov, 'n_iter': prov['de_n_iter'], 'n_eval': prov['de_n_eval'],
        'stress_run_id': db.compute_stress_run_id(scenario_id, result['target_idx'], result['delta'], result['method'], prov),
        'fingerprint': db.compute_scene_fingerprint(states, validity, types),
    }


def _server():
    return settings.dsn_kwargs()


def _make_db(with_tables):
    base = _server()
    name = f'{base["dbname"]}_tr_{uuid.uuid4().hex[:8]}'
    admin = psycopg2.connect(**base)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = 'av_api_ro'")
        made_role = cur.fetchone() is None
        cur.execute(f'CREATE DATABASE "{name}" TEMPLATE template0')
    conn = psycopg2.connect(**{**base, 'dbname': name})
    conn.autocommit = True
    with conn.cursor() as cur:
        if made_role:
            cur.execute('CREATE ROLE av_api_ro LOGIN NOSUPERUSER')
        cur.execute('GRANT USAGE ON SCHEMA public TO av_api_ro')
        cur.execute('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO av_api_ro')
    db.init_schema(conn)
    if with_tables:
        with conn.cursor() as cur:
            cur.execute(open(SCRIPT).read())
    return admin, name, conn, made_role


def _drop_db(admin, name, conn, made_role):
    conn.close()
    with admin.cursor() as cur:
        cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        if made_role:
            cur.execute('DROP ROLE IF EXISTS av_api_ro')
    admin.close()


def _insert_scenario(conn, s, **over):
    row = {'scenario_id': s['scenario_id'], 'shard': 'shard-a', 'fragility_score': 1.0, 'stress_run_id': s['stress_run_id'],
           'scene_fingerprint': s['fingerprint'], 'sdc_idx': 0, 'delta': s['delta'], 'target_idx': s['target_idx'],
           'stress_method': s['stress_method'], 'min_perturbation': 0.5, 'collision_timestep': 3}
    row.update(over)
    names = ', '.join([*row, 'search_provenance', 'stress_tested_at'])
    marks = ', '.join(['%s'] * len(row) + ['%s::jsonb', 'now()'])
    with conn.cursor() as cur:
        cur.execute('DELETE FROM scenario_scores WHERE scenario_id = %s', (s['scenario_id'],))
        cur.execute(f'INSERT INTO scenario_scores ({names}) VALUES ({marks})',
                    [*row.values(), json.dumps(db._json_safe(s['provenance']))])


@pytest.fixture(scope='module')
def tr_db():
    admin, name, conn, made_role = _make_db(True)
    yield name, conn
    _drop_db(admin, name, conn, made_role)


@pytest.fixture(scope='module')
def bare_db():
    admin, name, conn, made_role = _make_db(False)
    yield name, conn
    _drop_db(admin, name, conn, made_role)


@pytest.fixture(scope='module')
def stored():
    return _stored()


@pytest.fixture
def conn(tr_db, stored):
    _, c = tr_db
    with c.cursor() as cur:
        cur.execute('DELETE FROM search_runs')       # search_traces go with them
        cur.execute('DELETE FROM scenario_scores')
    _insert_scenario(c, stored)
    return c


def _count(c, table):
    with c.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM {table}')
        return cur.fetchone()[0]


def _fingerprint_of_the_results(c):
    """A digest of every row of scenario_scores, and the row counts of the geometry tables where they exist (Pass 3 makes them)."""
    with c.cursor() as cur:
        cur.execute("SELECT md5(string_agg(t::text, E'\\n' ORDER BY scenario_id)) FROM scenario_scores t")
        digest = cur.fetchone()[0]
    counts = []
    for table in ('scenario_agents', 'perturbed_paths'):
        if _exists(c, table):
            with c.cursor() as cur:
                cur.execute(f'SELECT count(*) FROM {table}')
                counts.append(cur.fetchone()[0])
    return digest, counts


def _exists(c, table):
    with c.cursor() as cur:
        cur.execute('SELECT to_regclass(%s) IS NOT NULL', (table,))
        return cur.fetchone()[0]


# ── the recorder ────────────────────────────────────────────────────────────────────────────────────────────────────────────────


@requires_db
def test_the_rerun_follows_the_stored_search_and_is_recorded_with_its_distance_from_it(conn, stored):
    out = search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    assert out['max_abs_delta_difference'] == 0.0 and out['matches_stored'] is True and out['inserted'] is True and out['will_be_served'] is True
    with conn.cursor() as cur:
        cur.execute("SELECT kind, reruns_stress_run_id, max_abs_delta_difference, scenario_id, scene_fingerprint, sdc_idx, target_idx, "
                    "seed, popsize, maxiter, n_iter, n_eval, delta, outcome, delta_parameterization FROM search_runs")
        [r] = cur.fetchall()
    assert r[:4] == ('trace_rerun', stored['stress_run_id'], 0.0, 't1')
    assert r[4:7] == (stored['fingerprint'], 0, stored['target_idx'])
    assert r[7:10] == (BUDGET['seed'], BUDGET['popsize'], BUDGET['maxiter'])
    assert r[10:12] == (stored['n_iter'], stored['n_eval']) and r[12] == stored['delta']
    assert r[13] == 'collision_found' and r[14] == stored['provenance']['delta_parameterization']


@requires_db
def test_one_point_per_generation_is_recorded_and_the_last_total_is_the_searchs(conn, stored):
    out = search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    with conn.cursor() as cur:
        cur.execute('SELECT generation, n_eval_cumulative, best_weighted_norm FROM search_traces ORDER BY generation')
        rows = cur.fetchall()
    assert [g for g, _, _ in rows] == list(range(1, stored['n_iter'] + 1)) and out['generations'] == len(rows)
    assert rows[-1][1] == stored['n_eval']
    norms = [b for _, _, b in rows if b is not None]
    assert norms and all(y <= x for x, y in zip(norms, norms[1:]))


@requires_db
def test_the_recorder_writes_nothing_but_its_two_tables(conn):
    before = _fingerprint_of_the_results(conn)
    search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    assert _fingerprint_of_the_results(conn) == before
    assert _count(conn, 'search_runs') == 1 and _count(conn, 'search_traces') > 0
    for table in ('pipeline_runs', 'pipeline_run_results', 'sensitivity_slices', 'shards'):
        assert _count(conn, table) == 0, table


@requires_db
def test_recording_the_same_rerun_twice_leaves_one_row(conn):
    search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    n = _count(conn, 'search_traces')
    again = search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    assert again['inserted'] is False and _count(conn, 'search_runs') == 1 and _count(conn, 'search_traces') == n


@requires_db
def test_a_scene_that_is_not_the_one_searched_is_refused_and_nothing_is_written(conn):
    states, validity, types = _scene()
    states[1, :, 1] = 2.2
    with pytest.raises(search_runs.TraceRerunRefused, match='scene'):
        search_runs.record_trace_rerun(conn, 't1', states, validity, types, 0)
    assert _count(conn, 'search_runs') == 0 and _count(conn, 'search_traces') == 0


@requires_db
def test_another_sdc_index_is_refused(conn):
    with pytest.raises(search_runs.TraceRerunRefused, match='SDC'):
        search_runs.record_trace_rerun(conn, 't1', *_scene(), 1)
    assert _count(conn, 'search_runs') == 0


@requires_db
def test_no_stored_search_and_no_such_scenario_are_refused(conn):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO scenario_scores (scenario_id, shard, fragility_score) VALUES ('bare', 'a', 1.0)")
    with pytest.raises(search_runs.TraceRerunRefused, match='no stored search'):
        search_runs.record_trace_rerun(conn, 'bare', *_scene(), 0)
    with pytest.raises(search_runs.TraceRerunRefused, match='no such scenario'):
        search_runs.record_trace_rerun(conn, 'nope', *_scene(), 0)
    assert _count(conn, 'search_runs') == 0


@requires_db
@pytest.mark.parametrize('key', ['de_seed', 'de_popsize', 'de_maxiter', 'de_tol'])
def test_a_stored_search_that_does_not_say_its_seed_and_budget_is_refused(conn, stored, key):
    prov = {k: v for k, v in stored['provenance'].items() if k != key}
    _insert_scenario(conn, {**stored, 'provenance': prov})
    with pytest.raises(search_runs.TraceRerunRefused, match='seed and the budget'):
        search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    assert _count(conn, 'search_runs') == 0


@requires_db
def test_a_rerun_that_ended_elsewhere_is_recorded_with_its_difference_not_hidden(conn, stored):
    _insert_scenario(conn, {**stored, 'delta': [x + 0.01 for x in stored['delta']]})
    out = search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    assert abs(out['max_abs_delta_difference'] - 0.01) < 1e-6 and out['inserted'] is True and out['will_be_served'] is False
    with conn.cursor() as cur:
        cur.execute('SELECT max_abs_delta_difference FROM search_runs')
        assert abs(cur.fetchone()[0] - 0.01) < 1e-6


@requires_db
def test_the_rerun_names_the_stored_result_it_re_ran_even_when_its_own_id_is_another(conn, stored):
    # a stored id from another machine's session: the re-run's content hash differs, and the row still points at the stored one
    other = 'ffffffffffffffff'
    _insert_scenario(conn, {**stored, 'stress_run_id': other})
    out = search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    assert out['search_run_id'] == stored['stress_run_id'] and out['matches_stored'] is False and out['max_abs_delta_difference'] == 0.0
    assert out['will_be_served'] is True      # it names the stored result (the other id) and ended where it ended
    with conn.cursor() as cur:
        cur.execute('SELECT search_run_id, reruns_stress_run_id FROM search_runs')
        assert cur.fetchall() == [(stored['stress_run_id'], other)]


@requires_db
def test_a_row_of_another_kind_with_the_same_id_means_nothing_is_written_and_nothing_is_served(conn, stored):
    # search_run_id is the primary key: a runner's row (kind 'seed') that already has the id keeps it. The re-run's row is not inserted,
    # its trace is not written, and the recorder must not say the trace will be shown: nothing in the database would show it.
    with conn.cursor() as cur:
        cur.execute("INSERT INTO search_runs (search_run_id, scenario_id, target_idx, kind, seed, outcome) VALUES (%s, 't1', 1, 'seed', 2, 'collision_found')",
                    (stored['stress_run_id'],))
    out = search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    assert out['search_run_id'] == stored['stress_run_id']
    assert out['inserted'] is False and out['will_be_served'] is False
    assert _count(conn, 'search_traces') == 0 and _count(conn, 'search_runs') == 1
    with conn.cursor() as cur:
        cur.execute('SELECT kind, reruns_stress_run_id FROM search_runs')
        assert cur.fetchall() == [('seed', None)]


@requires_db
def test_the_recorder_commits_on_a_connection_that_is_not_autocommit(conn, tr_db):
    name, _ = tr_db
    own = psycopg2.connect(**{**_server(), 'dbname': name})        # a transaction per call, as a notebook's connection is
    try:
        out = search_runs.record_trace_rerun(own, 't1', *_scene(), 0)
        assert out['inserted'] is True and out['will_be_served'] is True
        assert _count(conn, 'search_runs') == 1 and _count(conn, 'search_traces') == out['generations']    # seen from another connection
    finally:
        own.close()


@requires_db
def test_a_second_identical_rerun_inserts_nothing_and_the_first_is_still_served(conn, client):
    first = search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    assert first['inserted'] is True and first['will_be_served'] is True
    n = _count(conn, 'search_traces')
    second = search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    assert second['inserted'] is False and second['will_be_served'] is True
    assert _count(conn, 'search_runs') == 1 and _count(conn, 'search_traces') == n
    assert client.get('/scenarios/t1/trace').json()['trace'] is not None


@requires_db
def test_will_be_served_is_what_the_route_then_does(conn, client, stored):
    # recorded but ended elsewhere: not served, and said so
    _insert_scenario(conn, {**stored, 'delta': [x + 0.01 for x in stored['delta']]})
    out = search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    assert out['inserted'] is True and out['will_be_served'] is False
    assert client.get('/scenarios/t1/trace').json()['trace'] is None


@requires_db
def test_a_row_with_a_negative_difference_is_not_served_and_is_not_said_to_be(conn, client, stored):
    # a difference is a largest absolute value, never negative; a row that says otherwise is not one this pipeline wrote
    with conn.cursor() as cur:
        cur.execute("INSERT INTO search_runs (search_run_id, scenario_id, target_idx, kind, outcome, reruns_stress_run_id, max_abs_delta_difference) "
                    "VALUES (%s, 't1', 1, 'trace_rerun', 'collision_found', %s, -1e-9)", (stored['stress_run_id'], stored['stress_run_id']))
        cur.execute("INSERT INTO search_traces VALUES (%s, 1, 10, 1.0)", (stored['stress_run_id'],))
    assert client.get('/scenarios/t1/trace').json()['trace'] is None
    out = search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    assert out['inserted'] is False and out['will_be_served'] is False


@requires_db
def test_a_row_with_the_right_id_and_no_trace_points_is_not_served(conn, stored):
    # the row is there, the points are not (a recorder that failed between the two writes): not served
    with conn.cursor() as cur:
        cur.execute("INSERT INTO search_runs (search_run_id, scenario_id, target_idx, kind, outcome, reruns_stress_run_id, max_abs_delta_difference) "
                    "VALUES (%s, 't1', 1, 'trace_rerun', 'collision_found', %s, 0)", (stored['stress_run_id'], stored['stress_run_id']))
    out = search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    assert out['inserted'] is False and out['will_be_served'] is False


@requires_db
def test_a_stored_result_that_the_gradient_stage_refined_is_re_run_with_the_gradient_stage(conn):
    pytest.importorskip('torch')
    refined = _stored(use_autograd=True)
    if refined['provenance'].get('refine_n_iters') is None:
        pytest.skip('the gradient stage did not run on this scene')
    _insert_scenario(conn, refined)
    out = search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    assert out['max_abs_delta_difference'] == 0.0 and out['matches_stored'] is True


@requires_db
def test_missing_tables_are_a_clear_error(bare_db, stored):
    _, c = bare_db
    with c.cursor() as cur:
        cur.execute('DELETE FROM scenario_scores')
    with pytest.raises(ManifestTablesMissing, match='search_runs'):
        search_runs.record_trace_rerun(c, 't1', *_scene(), 0)


@requires_db
def test_the_stored_results_identity_is_untouched_by_a_trace_in_the_budget():
    # Passing a trace through de_kwargs must not enter the provenance the run id hashes.
    plain = _stored()
    traced = _stored(trace=[])
    assert traced['stress_run_id'] == plain['stress_run_id'] and traced['provenance'] == plain['provenance']
    assert 'trace' not in json.dumps(traced['provenance'])


# ── the route ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def _client(monkeypatch, db_conn):
    monkeypatch.setattr(api_main, 'init_pool', lambda: None)
    monkeypatch.setattr(api_main, 'close_pool', lambda: None)
    app = api_main.create_app()
    app.dependency_overrides[db_pool.get_db] = lambda: db_conn
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client(monkeypatch, conn):
    with _client(monkeypatch, conn) as c:
        yield c


def _put_rerun(c, *, scenario='t1', reruns='STORED', difference=0.0, fingerprint='STORED', sdc=0, created='2026-10-10T10:00:00+00',
               points=((1, 20, None), (2, 40, 3.0), (3, 60, 2.0)), stored=None):
    stored = stored or _stored()
    run_id = secrets.token_hex(8)
    reruns = stored['stress_run_id'] if reruns == 'STORED' else reruns
    fingerprint = stored['fingerprint'] if fingerprint == 'STORED' else fingerprint
    with c.cursor() as cur:
        cur.execute("""INSERT INTO search_runs (search_run_id, scenario_id, scene_fingerprint, sdc_idx, target_idx, kind, outcome,
                                                reruns_stress_run_id, max_abs_delta_difference, created_at)
                       VALUES (%s, %s, %s, %s, 1, 'trace_rerun', 'collision_found', %s, %s, %s)""",
                    (run_id, scenario, fingerprint, sdc, reruns, difference, created))
        for g, n, b in points:
            cur.execute('INSERT INTO search_traces VALUES (%s, %s, %s, %s)', (run_id, g, n, b))
    return run_id


@requires_db
def test_a_recorded_trace_is_served_with_its_difference_and_the_tolerance(conn, client, stored):
    search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    body = client.get('/scenarios/t1/trace').json()
    assert set(body) == set(TraceResponse.model_fields)
    assert body['scenario_id'] == 't1' and body['stress_run_id'] == stored['stress_run_id']
    assert body['max_abs_delta_difference'] == 0.0 and body['tolerance'] == 1e-6 and body['recorded_at'].endswith('Z') or '+00:00' in body['recorded_at']
    assert [p['generation'] for p in body['trace']] == list(range(1, stored['n_iter'] + 1))
    assert body['trace'][-1]['n_eval_cumulative'] == stored['n_eval']
    assert set(body['trace'][0]) == {'generation', 'n_eval_cumulative', 'best_weighted_norm'}


@requires_db
def test_with_nothing_recorded_the_trace_is_null_and_the_status_is_200(client, stored):
    r = client.get('/scenarios/t1/trace')
    assert r.status_code == 200
    assert r.json() == {'scenario_id': 't1', 'stress_run_id': stored['stress_run_id'], 'trace': None, 'max_abs_delta_difference': None,
                        'tolerance': 1e-6, 'recorded_at': None}


@requires_db
def test_a_result_with_no_run_id_does_not_ask_the_trace_tables_anything(conn, client, monkeypatch):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO scenario_scores (scenario_id, shard, fragility_score) VALUES ('bare', 'a', 1.0)")
    asked = []
    real = routes._table_exists
    monkeypatch.setattr(routes, '_table_exists', lambda cur, table: asked.append(table) or real(cur, table))
    assert client.get('/scenarios/bare/trace').json()['trace'] is None
    assert asked == []
    client.get('/scenarios/t1/trace')
    assert 'search_runs' in asked


@requires_db
def test_an_unknown_scenario_is_404(client):
    assert client.get('/scenarios/nope/trace').status_code == 404


@requires_db
def test_a_result_with_no_run_id_has_no_trace_even_if_a_rerun_names_null(conn, client):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO scenario_scores (scenario_id, shard, fragility_score) VALUES ('bare', 'a', 1.0)")
    r = client.get('/scenarios/bare/trace')
    assert r.status_code == 200 and r.json()['trace'] is None and r.json()['stress_run_id'] is None


@requires_db
@pytest.mark.parametrize('difference,served', [
    (0.0, True), (5e-7, True), (9.99e-7, True), (1e-6, False), (1.5e-6, False), (2e-6, False), (0.01, False), (None, False), (float('nan'), False),
])
def test_a_trace_is_served_only_below_the_tolerance(conn, client, difference, served):
    _put_rerun(conn, difference=difference)
    body = client.get('/scenarios/t1/trace').json()
    assert (body['trace'] is not None) is served
    assert body['max_abs_delta_difference'] == (difference if served else None)


@requires_db
def test_a_trace_that_re_ran_another_result_is_not_served(conn, client):
    _put_rerun(conn, reruns='0123456789abcdef')
    assert client.get('/scenarios/t1/trace').json()['trace'] is None


@requires_db
def test_a_trace_that_belongs_to_another_scenario_is_not_served(conn, client):
    _put_rerun(conn, scenario='someone-else')
    assert client.get('/scenarios/t1/trace').json()['trace'] is None


@requires_db
def test_when_the_stored_result_is_replaced_the_old_trace_stops_being_served(conn, client, stored):
    _put_rerun(conn)
    assert client.get('/scenarios/t1/trace').json()['trace'] is not None
    with conn.cursor() as cur:
        cur.execute("UPDATE scenario_scores SET stress_run_id = 'ffffffffffffffff' WHERE scenario_id = 't1'")
    assert client.get('/scenarios/t1/trace').json()['trace'] is None


@requires_db
def test_a_trace_searched_against_another_scene_or_sdc_is_not_served(conn, client):
    _put_rerun(conn, fingerprint='some-other-scene')
    _put_rerun(conn, sdc=3)
    assert client.get('/scenarios/t1/trace').json()['trace'] is None


@requires_db
def test_a_result_stored_without_a_fingerprint_accepts_a_trace_as_the_read_side_does_everywhere_else(conn, client, stored):
    with conn.cursor() as cur:
        cur.execute("UPDATE scenario_scores SET scene_fingerprint = NULL, sdc_idx = NULL WHERE scenario_id = 't1'")
    _put_rerun(conn, fingerprint='whatever', sdc=7)
    assert client.get('/scenarios/t1/trace').json()['trace'] is not None


@requires_db
def test_the_newest_qualifying_trace_is_served(conn, client):
    _put_rerun(conn, created='2026-10-01T10:00:00+00', points=((1, 10, 9.0),))
    _put_rerun(conn, created='2026-10-12T10:00:00+00', points=((1, 10, 8.0), (2, 20, 7.0)))
    _put_rerun(conn, created='2026-10-14T10:00:00+00', difference=0.5, points=((1, 10, 1.0),))   # newer, but it ended elsewhere
    body = client.get('/scenarios/t1/trace').json()
    assert [p['best_weighted_norm'] for p in body['trace']] == [8.0, 7.0]


@requires_db
def test_a_rerun_with_no_points_serves_no_trace(conn, client):
    _put_rerun(conn, points=())
    assert client.get('/scenarios/t1/trace').json()['trace'] is None


@requires_db
def test_a_best_norm_that_is_not_finite_is_served_as_null(conn, client):
    _put_rerun(conn, points=((1, 10, float('nan')), (2, 20, 2.0)))
    assert [p['best_weighted_norm'] for p in client.get('/scenarios/t1/trace').json()['trace']] == [None, 2.0]


@requires_db
def test_without_the_tables_the_trace_is_null_not_an_error(monkeypatch, bare_db, stored):
    _, c = bare_db
    _insert_scenario(c, stored)
    with _client(monkeypatch, c) as client:
        r = client.get('/scenarios/t1/trace')
    assert r.status_code == 200 and r.json()['trace'] is None


@requires_db
def test_the_read_only_role_can_read_a_trace(monkeypatch, tr_db, conn):
    _put_rerun(conn)
    with conn.cursor() as cur:
        cur.execute('SET ROLE av_api_ro')
    try:
        with _client(monkeypatch, conn) as client:
            assert client.get('/scenarios/t1/trace').json()['trace'] is not None
    finally:
        with conn.cursor() as cur:
            cur.execute('RESET ROLE')


# ── a role that cannot read the tables ──────────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope='module')
def deny_role(tr_db):
    name, owner = tr_db
    role, password = f'av_tr_deny_{uuid.uuid4().hex[:6]}', secrets.token_hex(16)
    with owner.cursor() as cur:
        cur.execute(f"CREATE ROLE {role} LOGIN NOSUPERUSER PASSWORD %s", (password,))
        cur.execute(f'GRANT USAGE ON SCHEMA public TO {role}')
        cur.execute(f'GRANT SELECT ON scenario_scores TO {role}')
    yield {**_server(), 'dbname': name, 'user': role, 'password': password}
    with owner.cursor() as cur:
        cur.execute(f'REASSIGN OWNED BY {role} TO CURRENT_USER')
        cur.execute(f'DROP OWNED BY {role}')
        cur.execute(f'DROP ROLE IF EXISTS {role}')


@requires_db
def test_a_role_without_select_on_the_trace_tables_gets_no_trace_not_a_500_and_one_clean_log_line(monkeypatch, conn, deny_role, stored, caplog):
    _put_rerun(conn)
    c = psycopg2.connect(**deny_role)
    try:
        with _client(monkeypatch, c) as client, caplog.at_level(logging.WARNING, logger='src.api.routes'):
            r = client.get('/scenarios/t1/trace')
            assert r.status_code == 200, r.text
            assert r.json()['trace'] is None and r.json()['stress_run_id'] == stored['stress_run_id']
            assert client.get('/scenarios/t1/trace').status_code == 200      # the connection is usable after the rollback
        warnings = [x for x in caplog.records if x.levelno == logging.WARNING and x.name == 'src.api.routes']
        assert len(warnings) == 2
        message = warnings[0].getMessage()
        assert 'cannot be read by the API role' in message and 'DEPLOY.md' in message
        for secret in (deny_role['password'], deny_role['user'], 'search_runs', 'SELECT', 'host=', 'postgres'):
            assert secret not in message, secret
    finally:
        c.close()
