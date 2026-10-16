"""
The other searches of a stored result: the offline runners (src/scoring/other_searches.py), the rules for what is served
(src/api/other_searches.py) and GET /scenarios/{id}/other-searches.

Pinned here:

  * the runners: another seed (#16) or another budget (#17) of the stored search, recorded as kind 'seed' or 'budget' with the stored
    scene, challenger and model; they write only `search_runs` (no trace, nothing of the stored result); a configuration that is the
    stored search is skipped (before it runs when its seed and budget are the stored ones, after it when its id is the stored
    `stress_run_id`); an id that is already there is left as it is and reported as a conflict with the kind that holds it; they refuse
    a scene or SDC index that is not the one searched; bad configurations are refused before anything runs;
  * the allow-list: each field's type, a wrong type is null, an unserved kind is not served;
  * the route: only the three kinds, never a trace re-run, never the stored search, only rows whose scene and SDC index do not conflict
    with the stored result's, a fixed order that is not a ranking, labels from each search's own model, an empty list for every state
    with nothing to list (including a role that cannot read the table: one fixed log line), and 404 only for an unknown scenario.

The database tests build a throw-away database and drop it after. They skip unless AV_ROBUSTNESS_DB=1.

Run:
    AV_ROBUSTNESS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_other_searches.py -q
"""

import logging
import math
import os
import sys

import numpy as np
import psycopg2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import other_searches as served, routes
from src.api.models import DELTA_LABELS_BY_PARAMETERIZATION, OtherSearch, OtherSearchesResponse
from src.scoring import other_searches as runner, search_runs
from src.scoring.manifest import ManifestTablesMissing
from src.scoring.search_runs import TraceRerunRefused

from tests.test_trace import (  # noqa: F401  (the fixtures are used by name)
    BUDGET, _client, _count, _fingerprint_of_the_results, _insert_scenario, _scene, _server, _stored, bare_db, client, deny_role,
    requires_db, stored, tr_db,
)


@pytest.fixture
def conn(tr_db, stored):
    _, c = tr_db
    with c.cursor() as cur:
        cur.execute('DELETE FROM search_runs')
        cur.execute('DELETE FROM scenario_scores')
    _insert_scenario(c, stored)
    return c


def _rows(c, where=''):
    with c.cursor() as cur:
        cur.execute(f'SELECT search_run_id, kind, seed, popsize, maxiter, tol, outcome, scene_fingerprint, sdc_idx, target_idx, '
                    f'delta_parameterization, delta, n_iter, n_eval, reruns_stress_run_id FROM search_runs {where} ORDER BY kind, seed, popsize, maxiter')
        return cur.fetchall()


def _search(c, search_run_id, *, scenario='t1', kind='seed', seed=1, popsize=5, maxiter=4, tol=1e-3, outcome='collision_found', minp=0.5,
            fingerprint='STORED', sdc=0, parameterization='bicycle', delta=(0.1, 0.0, 0.0, 0.0), rank=None, stored_row=None):
    fp = (stored_row or _stored())['fingerprint'] if fingerprint == 'STORED' else fingerprint
    with c.cursor() as cur:
        cur.execute("""INSERT INTO search_runs (search_run_id, scenario_id, scene_fingerprint, sdc_idx, target_idx, delta_parameterization, kind,
                                                challenger_rank, seed, popsize, maxiter, tol, outcome, min_perturbation, collision_timestep, delta,
                                                n_iter, n_eval)
                       VALUES (%s, %s, %s, %s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s, 3, %s, 4, 99)""",
                    (search_run_id, scenario, fp, sdc, parameterization, kind, rank, seed, popsize, maxiter, tol, outcome, minp, list(delta)))


# ── the allow-list ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

GOOD = {'search_run_id': '0123456789abcdef', 'kind': 'seed', 'challenger_rank': None, 'target_idx': 1, 'seed': 3, 'popsize': 15, 'maxiter': 200,
        'tol': 0.001, 'outcome': 'collision_found', 'min_perturbation': 0.25, 'collision_timestep': 40, 'delta': [0.1, 0.2, 0.3, 0.4],
        'delta_parameterization': 'bicycle', 'n_iter': 12, 'n_eval': 900}


def test_the_served_fields_are_exactly_the_models():
    assert list(served.select_search(GOOD)) == [f for f in OtherSearch.model_fields if f != 'delta_labels']
    assert set(OtherSearchesResponse.model_fields) == {'scenario_id', 'stress_run_id', 'stored_delta_parameterization', 'searches', 'truncated'}


def test_a_good_row_is_served_as_it_is():
    assert served.select_search(GOOD) == GOOD


@pytest.mark.parametrize('kind', ['trace_rerun', 'other', '', None, 3, ['seed']])
def test_a_kind_that_is_not_served_is_not_served(kind):
    assert served.select_search({**GOOD, 'kind': kind}) is None


@pytest.mark.parametrize('kind', ['seed', 'budget', 'challenger'])
def test_the_three_kinds_are_served(kind):
    assert served.select_search({**GOOD, 'kind': kind})['kind'] == kind


@pytest.mark.parametrize('field,bad', [
    ('search_run_id', 'xyz'), ('search_run_id', 'A' * 16), ('search_run_id', '0' * 15), ('search_run_id', 7),
    ('seed', -1), ('seed', True), ('seed', 1.5), ('seed', '3'),
    ('popsize', 0), ('popsize', False), ('maxiter', 0), ('maxiter', 2.0), ('target_idx', -1), ('challenger_rank', 0), ('collision_timestep', -2),
    ('tol', -1e-3), ('tol', float('nan')), ('tol', float('inf')), ('tol', True), ('tol', '1e-3'),
    ('outcome', 'replay_infeasible'), ('outcome', 'smallest'), ('outcome', 3), ('outcome', None),
    ('min_perturbation', -0.1), ('min_perturbation', float('nan')), ('min_perturbation', float('inf')), ('min_perturbation', True), ('min_perturbation', '0.2'),
    ('delta', []), ('delta', [0.1] * 9), ('delta', [0.1, float('nan')]), ('delta', [0.1, True]), ('delta', ['a']), ('delta', 'abc'), ('delta', 3),
    ('delta_parameterization', 'quartic'), ('delta_parameterization', 1), ('delta_parameterization', None),
    ('n_iter', -1), ('n_eval', True), ('n_eval', 1.5),
])
def test_a_field_of_the_wrong_type_is_null_not_repaired_and_the_others_are_kept(field, bad):
    out = served.select_search({**GOOD, field: bad})
    assert out[field] is None
    assert {k: v for k, v in out.items() if k != field} == {k: v for k, v in GOOD.items() if k != field}


def test_a_missing_column_is_null():
    assert served.select_search({'kind': 'seed'})['min_perturbation'] is None


def test_an_extra_column_is_never_served():
    assert 'search_provenance' not in served.select_search({**GOOD, 'search_provenance': {'a': 1}, 'pipeline_run_id': 'x'})


def test_a_whole_number_is_a_number():
    assert served.select_search({**GOOD, 'min_perturbation': 1, 'tol': 0})['tol'] == 0.0


def test_the_limit_is_two_hundred():
    assert served.MAX_SEARCHES == 200


def test_the_parameterizations_are_the_two_the_api_has_labels_for():
    assert set(served.PARAMETERIZATIONS) == set(DELTA_LABELS_BY_PARAMETERIZATION)


def test_the_module_needs_nothing_the_deployed_api_lacks():
    import ast
    tree = ast.parse(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'api', 'other_searches.py')).read())
    imported = {n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    imported |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert imported <= {'math', 're'}


# ── the runners: what they validate before they run anything ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize('seeds', [[-1], [True], ['1'], [1.5], [None], [1, -2]])
def test_a_seed_that_is_not_a_non_negative_integer_is_refused_before_anything_runs(seeds, monkeypatch):
    called = []
    monkeypatch.setattr(runner, '_run', lambda *a, **k: called.append(1))
    with pytest.raises(ValueError, match='seed'):
        runner.run_seed_searches(None, 't1', *_scene(), 0, seeds=seeds)
    assert called == []


@pytest.mark.parametrize('budget,message', [
    ({'popsize': 1}, 'popsize'), ({'popsize': True}, 'popsize'), ({'popsize': 5.5}, 'popsize'), ({'maxiter': 0}, 'maxiter'),
    ({'maxiter': -3}, 'maxiter'), ({'tol': -1}, 'tol'), ({'tol': float('nan')}, 'tol'), ({'tol': float('inf')}, 'tol'), ({'tol': '1e-3'}, 'tol'),
    ({'tol': True}, 'tol'), ({'seed': 4}, 'popsize, maxiter and tol'), ({'popsize': 5, 'population': 5}, 'popsize, maxiter and tol'),
])
def test_a_bad_budget_is_refused(budget, message):
    with pytest.raises(ValueError, match=message):
        runner._budgets([budget], {'popsize': 15, 'maxiter': 200, 'tol': 1e-3})


def test_a_budget_that_leaves_things_out_takes_the_stored_values_and_repeats_are_dropped():
    stored_budget = {'popsize': 15, 'maxiter': 200, 'tol': 1e-3}
    out = runner._budgets([{'maxiter': 50}, {'maxiter': 50, 'popsize': 15, 'tol': 0.001}, {}], stored_budget)
    assert out == [{'popsize': 15, 'maxiter': 50, 'tol': 0.001}, {'popsize': 15, 'maxiter': 200, 'tol': 0.001}]


def test_repeated_seeds_are_run_once_in_the_order_given():
    assert runner._seeds([3, 1, 3, 2, 1]) == [3, 1, 2]


# ── the runners, on a database ──────────────────────────────────────────────────────────────────────────────────────────────────


@requires_db
def test_other_seeds_are_recorded_as_separate_searches_of_the_same_scene_and_challenger(conn, stored):
    report = runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[3, 4])
    assert [r['status'] for r in report['recorded']] == ['recorded', 'recorded'] and report['skipped'] == [] and report['conflicts'] == []
    rows = _rows(conn)
    assert [(r[1], r[2], r[3], r[4], r[5]) for r in rows] == [('seed', 3, BUDGET['popsize'], BUDGET['maxiter'], BUDGET['tol']),
                                                              ('seed', 4, BUDGET['popsize'], BUDGET['maxiter'], BUDGET['tol'])]
    for r in rows:
        assert r[0] not in (None, stored['stress_run_id'])
        assert r[6] in ('collision_found', 'no_collision_found')
        assert r[7] == stored['fingerprint'] and r[8] == 0 and r[9] == stored['target_idx']
        assert r[10] == stored['provenance']['delta_parameterization'] and len(r[11]) == 4 and r[14] is None
        assert r[12] is not None and r[13] is not None


@requires_db
def test_other_budgets_use_the_stored_seed_and_the_budget_given(conn, stored):
    report = runner.run_budget_searches(conn, 't1', *_scene(), 0, budgets=[{'maxiter': 3}, {'popsize': 6, 'maxiter': 5, 'tol': 1e-2}])
    assert len(report['recorded']) == 2
    rows = _rows(conn)
    assert [(r[1], r[2], r[3], r[4], r[5]) for r in rows] == [('budget', BUDGET['seed'], BUDGET['popsize'], 3, BUDGET['tol']),
                                                              ('budget', BUDGET['seed'], 6, 5, 1e-2)]


@requires_db
def test_the_runners_write_only_search_runs_and_nothing_of_the_stored_result_changes(conn):
    before = _fingerprint_of_the_results(conn)
    runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[3])
    runner.run_budget_searches(conn, 't1', *_scene(), 0, budgets=[{'maxiter': 3}])
    assert _fingerprint_of_the_results(conn) == before
    assert _count(conn, 'search_runs') == 2 and _count(conn, 'search_traces') == 0
    for table in ('pipeline_runs', 'pipeline_run_results', 'sensitivity_slices', 'shards'):
        assert _count(conn, table) == 0, table


@requires_db
def test_the_stored_seed_and_budget_is_the_stored_search_so_it_is_skipped_and_not_even_run(conn, monkeypatch):
    from src.scoring import batch_scorer
    calls = []
    real = batch_scorer._stress_one
    monkeypatch.setattr(batch_scorer, '_stress_one', lambda *a, **k: calls.append(k['de_kwargs']) or real(*a, **k))
    report = runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[BUDGET['seed'], 3])
    assert [s['reason'] for s in report['skipped']] == ['same configuration as the stored result']
    assert [r['seed'] for r in report['recorded']] == [3] and [c['seed'] for c in calls] == [3]
    report = runner.run_budget_searches(conn, 't1', *_scene(), 0, budgets=[{}, {'maxiter': 3}])
    assert [s['reason'] for s in report['skipped']] == ['same configuration as the stored result'] and len(report['recorded']) == 1
    assert _count(conn, 'search_runs') == 2


@requires_db
def test_a_search_whose_id_is_the_stored_results_id_is_skipped_after_it_ran(conn, stored):
    # The stored id is made to be the id of the search with seed 5: running seed 5 reproduces the stored result, so it is not an alternative.
    first = runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[5])
    same = first['recorded'][0]['search_run_id']
    with conn.cursor() as cur:
        cur.execute('DELETE FROM search_runs')
        cur.execute("UPDATE scenario_scores SET stress_run_id = %s WHERE scenario_id = 't1'", (same,))
    report = runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[5, 6])
    assert [(s['seed'], s['reason'], s['search_run_id']) for s in report['skipped']] == [(5, 'same as the stored result', same)]
    assert [r['seed'] for r in report['recorded']] == [6]
    assert [r[2] for r in _rows(conn)] == [6]


@requires_db
def test_an_id_that_is_already_there_is_left_alone_and_reported_with_the_kind_that_holds_it(conn):
    first = runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[3])
    sid = first['recorded'][0]['search_run_id']
    with conn.cursor() as cur:
        cur.execute("UPDATE search_runs SET kind = 'budget', n_eval = 7777 WHERE search_run_id = %s", (sid,))
    before = _rows(conn)
    again = runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[3])
    assert again['recorded'] == [] and again['skipped'] == []
    assert [(c['search_run_id'], c['status'], c['held_by_kind']) for c in again['conflicts']] == [(sid, 'conflict', 'budget')]
    assert _rows(conn) == before


@requires_db
def test_a_second_identical_run_records_nothing_new(conn):
    runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[3, 4])
    again = runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[3, 4])
    assert again['recorded'] == [] and [c['held_by_kind'] for c in again['conflicts']] == ['seed', 'seed'] and _count(conn, 'search_runs') == 2


@requires_db
def test_a_trace_rerun_that_holds_the_id_is_a_conflict_too_and_its_trace_is_untouched(conn, stored):
    rid = runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[8])['recorded'][0]['search_run_id']
    with conn.cursor() as cur:
        cur.execute('DELETE FROM search_runs')
        cur.execute("INSERT INTO search_runs (search_run_id, scenario_id, target_idx, kind, outcome, reruns_stress_run_id, max_abs_delta_difference) "
                    "VALUES (%s, 't1', 1, 'trace_rerun', 'collision_found', %s, 0)", (rid, stored['stress_run_id']))
        cur.execute('INSERT INTO search_traces VALUES (%s, 1, 10, 0.5)', (rid,))
    before = _rows(conn)
    report = runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[8])
    assert [(c['search_run_id'], c['held_by_kind']) for c in report['conflicts']] == [(rid, 'trace_rerun')] and report['recorded'] == []
    assert _rows(conn) == before and _count(conn, 'search_traces') == 1


@requires_db
def test_a_scene_that_is_not_the_one_searched_is_refused_and_nothing_is_written(conn):
    states, validity, types = _scene()
    states[1, :, 1] = 2.2
    with pytest.raises(TraceRerunRefused, match='scene'):
        runner.run_seed_searches(conn, 't1', states, validity, types, 0, seeds=[3])
    with pytest.raises(TraceRerunRefused, match='scene'):
        runner.run_budget_searches(conn, 't1', states, validity, types, 0, budgets=[{'maxiter': 3}])
    assert _count(conn, 'search_runs') == 0


@requires_db
def test_another_sdc_index_and_a_scenario_with_no_stored_search_are_refused(conn):
    with pytest.raises(TraceRerunRefused, match='SDC'):
        runner.run_seed_searches(conn, 't1', *_scene(), 1, seeds=[3])
    with conn.cursor() as cur:
        cur.execute("INSERT INTO scenario_scores (scenario_id, shard, fragility_score) VALUES ('bare', 'a', 1.0)")
    with pytest.raises(TraceRerunRefused, match='no stored search'):
        runner.run_seed_searches(conn, 'bare', *_scene(), 0, seeds=[3])
    with pytest.raises(TraceRerunRefused, match='no such scenario'):
        runner.run_budget_searches(conn, 'nope', *_scene(), 0, budgets=[{}])
    assert _count(conn, 'search_runs') == 0


@requires_db
def test_a_stored_search_that_does_not_say_its_seed_is_refused(conn, stored):
    _insert_scenario(conn, {**stored, 'provenance': {k: v for k, v in stored['provenance'].items() if k != 'de_seed'}})
    with pytest.raises(TraceRerunRefused, match='seed and the budget'):
        runner.run_budget_searches(conn, 't1', *_scene(), 0, budgets=[{'maxiter': 3}])


@requires_db
def test_more_configurations_than_the_limit_are_refused_before_any_runs(conn):
    with pytest.raises(ValueError, match='at most'):
        runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=range(100, 100 + runner.MAX_CONFIGURATIONS + 1))
    assert _count(conn, 'search_runs') == 0


def _fake(monkeypatch, **changes):
    """_stress_one as it is, with the fields of its result changed: for the results the small test scene does not produce."""
    from src.scoring import batch_scorer
    real = batch_scorer._stress_one
    monkeypatch.setattr(batch_scorer, '_stress_one', lambda *a, **k: {**real(*a, **k), **changes})


@requires_db
def test_a_search_that_found_no_collision_is_recorded_with_no_norm_and_no_frame(conn, monkeypatch):
    _fake(monkeypatch, collision=False, min_perturbation=float('inf'), collision_timestep=-1, outcome='no_collision_found')
    report = runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[3])
    assert report['recorded'][0]['outcome'] == 'no_collision_found'
    with conn.cursor() as cur:
        cur.execute('SELECT outcome, min_perturbation, collision_timestep FROM search_runs')
        assert cur.fetchall() == [('no_collision_found', None, None)]


@requires_db
def test_a_search_that_found_no_collision_has_no_norm_even_if_its_result_carries_one(conn, monkeypatch):
    _fake(monkeypatch, collision=False, min_perturbation=0.5, collision_timestep=7, outcome='no_collision_found')
    runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[3])
    with conn.cursor() as cur:
        cur.execute('SELECT outcome, min_perturbation, collision_timestep FROM search_runs')
        assert cur.fetchall() == [('no_collision_found', None, None)]


@requires_db
def test_a_search_that_did_not_run_is_refused_and_nothing_after_it_is_run(conn, monkeypatch):
    _fake(monkeypatch, status='replay_infeasible', delta=None)
    with pytest.raises(TraceRerunRefused, match='did not run'):
        runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[3, 4])
    assert _count(conn, 'search_runs') == 0


@requires_db
def test_a_search_that_chose_another_challenger_is_refused_not_recorded(conn, monkeypatch, stored):
    _fake(monkeypatch, target_idx=stored['target_idx'] + 1)
    with pytest.raises(TraceRerunRefused, match='another challenger'):
        runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[3])
    assert _count(conn, 'search_runs') == 0


@requires_db
def test_exactly_the_limit_of_configurations_is_allowed(conn, monkeypatch):
    monkeypatch.setattr(runner, 'MAX_CONFIGURATIONS', 2)
    report = runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[BUDGET['seed'], 3])
    assert len(report['skipped']) == 1 and len(report['recorded']) == 1
    with pytest.raises(ValueError, match='at most 2'):
        runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[3, 4, 5])


@requires_db
def test_missing_tables_are_a_clear_error(bare_db):
    _, c = bare_db
    with c.cursor() as cur:
        cur.execute('DELETE FROM scenario_scores')
    with pytest.raises(ManifestTablesMissing, match='search_runs'):
        runner.run_seed_searches(c, 't1', *_scene(), 0, seeds=[3])


@requires_db
def test_the_gradient_stage_runs_for_an_alternative_when_it_ran_for_the_stored_search(conn):
    pytest.importorskip('torch')
    refined = _stored(use_autograd=True)
    if refined['provenance'].get('refine_n_iters') is None:
        pytest.skip('the gradient stage did not run on this scene')
    _insert_scenario(conn, refined)
    runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[3])
    with conn.cursor() as cur:
        cur.execute("SELECT search_provenance -> 'refine_n_iters' FROM search_runs")
        assert cur.fetchone()[0] is not None
    _insert_scenario(conn, _stored())                                       # and not when it did not
    with conn.cursor() as cur:
        cur.execute('DELETE FROM search_runs')
    runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[3])
    with conn.cursor() as cur:
        cur.execute("SELECT search_provenance -> 'refine_n_iters' = 'null'::jsonb FROM search_runs")
        assert cur.fetchone()[0] is True


@requires_db
def test_the_runner_commits_on_a_connection_that_is_not_autocommit(conn, tr_db):
    name, _ = tr_db
    own = psycopg2.connect(**{**_server(), 'dbname': name})
    try:
        runner.run_seed_searches(own, 't1', *_scene(), 0, seeds=[3])
        assert _count(conn, 'search_runs') == 1
    finally:
        own.close()


@requires_db
def test_the_pipeline_run_it_belongs_to_is_recorded_when_given(conn):
    from src.scoring import manifest
    run = manifest.begin_run(conn, 'rerun', command_line='x')
    runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[3], pipeline_run_id=run)
    with conn.cursor() as cur:
        cur.execute('SELECT pipeline_run_id::text FROM search_runs')
        assert cur.fetchone()[0] == run


# ── the route ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch, conn):
    with _client(monkeypatch, conn) as c:
        yield c


def _get(client, scenario='t1'):
    return client.get(f'/scenarios/{scenario}/other-searches')


@requires_db
def test_with_nothing_recorded_the_list_is_empty_and_the_status_is_200(client, stored):
    r = _get(client)
    assert r.status_code == 200
    assert r.json() == {'scenario_id': 't1', 'stress_run_id': stored['stress_run_id'],
                        'stored_delta_parameterization': stored['provenance']['delta_parameterization'], 'searches': [], 'truncated': False}


@requires_db
def test_an_unknown_scenario_is_404(client):
    assert _get(client, 'nope').status_code == 404


@requires_db
def test_the_recorded_searches_are_listed_with_their_own_labels(conn, client):
    runner.run_seed_searches(conn, 't1', *_scene(), 0, seeds=[3])
    runner.run_budget_searches(conn, 't1', *_scene(), 0, budgets=[{'maxiter': 3}])
    body = _get(client).json()
    assert [(s['kind'], s['seed'], s['maxiter']) for s in body['searches']] == [('seed', 3, BUDGET['maxiter']), ('budget', BUDGET['seed'], 3)]
    for s in body['searches']:
        assert set(s) == set(OtherSearch.model_fields)
        assert s['delta_parameterization'] == 'bicycle' and s['delta_labels'] == list(DELTA_LABELS_BY_PARAMETERIZATION['bicycle'])
        assert len(s['delta']) == 4 and s['search_run_id'] is not None and s['outcome'] in ('collision_found', 'no_collision_found')


@requires_db
def test_the_labels_come_from_each_searchs_own_model_not_the_stored_results(conn, client):
    _search(conn, 'aaaaaaaaaaaaaaa1', seed=1, parameterization='bicycle')
    _search(conn, 'aaaaaaaaaaaaaaa2', seed=2, parameterization='linear')
    _search(conn, 'aaaaaaaaaaaaaaa3', seed=3, parameterization=None)
    s = _get(client).json()['searches']
    assert [x['delta_parameterization'] for x in s] == ['bicycle', 'linear', None]
    assert s[0]['delta_labels'] == list(DELTA_LABELS_BY_PARAMETERIZATION['bicycle'])
    assert s[1]['delta_labels'] == list(DELTA_LABELS_BY_PARAMETERIZATION['linear'])
    assert s[2]['delta_labels'] is None


@requires_db
def test_the_stored_results_model_is_served_only_if_it_is_one_the_api_knows(conn, client):
    with conn.cursor() as cur:
        cur.execute("UPDATE scenario_scores SET search_provenance = jsonb_set(search_provenance, '{delta_parameterization}', '\"quartic\"') WHERE scenario_id = 't1'")
    assert _get(client).json()['stored_delta_parameterization'] is None
    with conn.cursor() as cur:
        cur.execute("UPDATE scenario_scores SET search_provenance = jsonb_set(search_provenance, '{delta_parameterization}', '\"linear\"') WHERE scenario_id = 't1'")
    assert _get(client).json()['stored_delta_parameterization'] == 'linear'


@requires_db
def test_a_trace_rerun_with_an_id_of_its_own_is_not_listed_and_does_not_count_towards_the_limit(conn, client, stored):
    _many(conn, served.MAX_SEARCHES)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO search_runs (search_run_id, scenario_id, target_idx, kind, seed, outcome, reruns_stress_run_id, max_abs_delta_difference) "
                    "VALUES ('9999999999999999', 't1', 1, 'trace_rerun', 0, 'collision_found', %s, 0.5)", (stored['stress_run_id'],))
    body = _get(client).json()
    assert len(body['searches']) == served.MAX_SEARCHES and body['truncated'] is False
    assert {s['kind'] for s in body['searches']} == {'seed'}


@requires_db
def test_the_order_is_fixed_and_is_not_a_ranking(conn, client):
    # norms fall as the seeds rise, budgets and challengers are mixed in: the order stays seeds, budgets, challengers, then by seed or budget
    _search(conn, 'bbbbbbbbbbbbbbb1', kind='challenger', seed=None, popsize=None, maxiter=None, tol=None, rank=2, minp=0.01)
    _search(conn, 'bbbbbbbbbbbbbbb2', kind='budget', seed=0, popsize=10, maxiter=50, minp=0.02)
    _search(conn, 'bbbbbbbbbbbbbbb3', kind='seed', seed=9, minp=0.03)
    _search(conn, 'bbbbbbbbbbbbbbb4', kind='seed', seed=2, minp=0.9)
    _search(conn, 'bbbbbbbbbbbbbbb5', kind='budget', seed=0, popsize=5, maxiter=50, minp=0.5)
    _search(conn, 'bbbbbbbbbbbbbbb6', kind='challenger', seed=None, popsize=None, maxiter=None, tol=None, rank=1, minp=0.5)
    order = [(s['kind'], s['seed'], s['popsize'], s['challenger_rank']) for s in _get(client).json()['searches']]
    assert order == [('seed', 2, 5, None), ('seed', 9, 5, None), ('budget', 0, 5, None), ('budget', 0, 10, None),
                     ('challenger', None, None, 1), ('challenger', None, None, 2)]


@requires_db
def test_a_trace_rerun_is_not_an_other_search(conn, client):
    search_runs.record_trace_rerun(conn, 't1', *_scene(), 0)
    assert _get(client).json()['searches'] == []


@requires_db
def test_a_row_with_the_stored_results_id_is_never_listed(conn, client, stored):
    _search(conn, stored['stress_run_id'], seed=7)
    _search(conn, 'cccccccccccccccc', seed=8)
    assert [s['seed'] for s in _get(client).json()['searches']] == [8]


@requires_db
def test_another_scenarios_searches_are_not_listed(conn, client, stored):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO scenario_scores (scenario_id, shard, fragility_score) VALUES ('other', 'a', 1.0)")
    _search(conn, 'dddddddddddddddd', scenario='other', seed=1)
    assert _get(client).json()['searches'] == [] and len(_get(client, 'other').json()['searches']) == 1


@requires_db
def test_a_search_of_another_scene_or_sdc_index_is_not_listed(conn, client):
    _search(conn, 'eeeeeeeeeeeeeee1', seed=1, fingerprint='some-other-scene')
    _search(conn, 'eeeeeeeeeeeeeee2', seed=2, sdc=3)
    _search(conn, 'eeeeeeeeeeeeeee3', seed=3)
    assert [s['seed'] for s in _get(client).json()['searches']] == [3]


@requires_db
def test_a_result_stored_without_a_fingerprint_lists_what_is_there_as_the_trace_route_does(conn, client):
    with conn.cursor() as cur:
        cur.execute("UPDATE scenario_scores SET scene_fingerprint = NULL, sdc_idx = NULL WHERE scenario_id = 't1'")
    _search(conn, 'ffffffffffffff01', seed=1, fingerprint='whatever', sdc=7)
    assert [s['seed'] for s in _get(client).json()['searches']] == [1]


@requires_db
def test_a_search_recorded_without_a_fingerprint_or_sdc_index_is_listed(conn, client):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO search_runs (search_run_id, scenario_id, target_idx, kind, seed, outcome) "
                    "VALUES ('1111111111111111', 't1', 1, 'seed', 4, 'collision_found')")
    [s] = _get(client).json()['searches']
    assert s['seed'] == 4 and s['min_perturbation'] is None and s['delta'] is None and s['delta_parameterization'] is None


@requires_db
def test_values_that_are_not_what_they_should_be_are_null_not_an_error(conn, client):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO search_runs (search_run_id, scenario_id, target_idx, kind, seed, outcome, min_perturbation, delta) "
                    "VALUES ('2222222222222222', 't1', 1, 'seed', 4, 'a-word-we-do-not-serve', 'NaN', ARRAY[1.0, 'NaN'::double precision])")
        cur.execute("INSERT INTO search_runs (search_run_id, scenario_id, target_idx, kind, seed, outcome) VALUES ('not-a-hex-id', 't1', 1, 'seed', 5, 'collision_found')")
    r = _get(client)
    assert r.status_code == 200
    a, b = r.json()['searches']
    assert (a['outcome'], a['min_perturbation'], a['delta']) == (None, None, None) and a['seed'] == 4
    assert b['search_run_id'] is None and b['seed'] == 5


def _many(conn, n):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO search_runs (search_run_id, scenario_id, target_idx, kind, seed, outcome) "
                    "SELECT lpad(to_hex(g), 16, '0'), 't1', 1, 'seed', g, 'collision_found' FROM generate_series(1, %s) g", (n,))


@requires_db
def test_exactly_the_limit_is_listed_whole_and_is_not_truncated(conn, client):
    _many(conn, served.MAX_SEARCHES)
    body = _get(client).json()
    assert [s['seed'] for s in body['searches']] == list(range(1, served.MAX_SEARCHES + 1))
    assert body['truncated'] is False


@requires_db
def test_one_more_than_the_limit_lists_the_first_in_the_fixed_order_and_says_it_cut_one_off(conn, client):
    _many(conn, served.MAX_SEARCHES + 1)
    body = _get(client).json()
    assert [s['seed'] for s in body['searches']] == list(range(1, served.MAX_SEARCHES + 1))
    assert body['truncated'] is True


@requires_db
def test_the_rows_that_are_cut_off_are_the_last_in_the_order_not_an_arbitrary_few(conn, client):
    # seeds come before budgets in the fixed order: with a full list of seeds and one budget, the budget is the one left out
    _many(conn, served.MAX_SEARCHES)
    _search(conn, 'f0f0f0f0f0f0f0f0', kind='budget', seed=0, popsize=9, maxiter=9)
    body = _get(client).json()
    assert {s['kind'] for s in body['searches']} == {'seed'} and len(body['searches']) == served.MAX_SEARCHES and body['truncated'] is True


@requires_db
def test_many_more_than_the_limit_is_still_200_and_truncated(conn, client):
    _many(conn, served.MAX_SEARCHES + 50)
    body = _get(client).json()
    assert len(body['searches']) == served.MAX_SEARCHES and body['truncated'] is True


@requires_db
def test_a_search_the_gate_leaves_out_does_not_count_towards_the_limit(conn, client):
    # 200 listed plus one of another scene: nothing was cut off, so nothing is said to be
    _many(conn, served.MAX_SEARCHES)
    _search(conn, 'e0e0e0e0e0e0e0e0', seed=999, fingerprint='another-scene')
    body = _get(client).json()
    assert len(body['searches']) == served.MAX_SEARCHES and body['truncated'] is False


@requires_db
def test_the_query_asks_for_one_row_more_than_the_limit_and_no_more(monkeypatch, conn):
    # the extra row is read only to learn that the list was cut; asking for more would read rows nobody can see
    asked = []

    class Proxy:
        def __init__(self, c):
            self._c = c

        def __getattr__(self, name):
            return getattr(self._c, name)

        def cursor(self, *args, cursor_factory=None, **kwargs):
            class Spy(cursor_factory or psycopg2.extensions.cursor):
                def execute(spy, query, vars=None):
                    if vars and 'FROM search_runs' in query:
                        asked.append(vars[-1])
                    return super().execute(query, vars)
            return self._c.cursor(*args, cursor_factory=Spy, **kwargs)

    _many(conn, served.MAX_SEARCHES + 3)
    with _client(monkeypatch, Proxy(conn)) as c:
        body = _get(c).json()
    assert asked == [served.MAX_SEARCHES + 1]
    assert len(body['searches']) == served.MAX_SEARCHES and body['truncated'] is True


@requires_db
def test_none_recorded_is_not_truncated(client):
    assert _get(client).json()['truncated'] is False


@requires_db
def test_the_response_marks_no_search_as_the_smallest(conn, client):
    _search(conn, 'abababababababa1', seed=1, minp=0.9)
    _search(conn, 'abababababababa2', seed=2, minp=0.1)
    text = _get(client).text.lower()
    for word in ('smallest', 'best', 'minimum', 'winner'):
        assert word not in text


@requires_db
def test_without_the_table_the_list_is_empty_not_an_error(monkeypatch, bare_db, stored):
    _, c = bare_db
    _insert_scenario(c, stored)
    with _client(monkeypatch, c) as client:
        r = _get(client)
    assert r.status_code == 200 and r.json()['searches'] == [] and r.json()['truncated'] is False


@requires_db
def test_the_read_only_role_can_read_them(monkeypatch, conn):
    _search(conn, 'cdcdcdcdcdcdcdc1', seed=1)
    with conn.cursor() as cur:
        cur.execute('SET ROLE av_api_ro')
    try:
        with _client(monkeypatch, conn) as client:
            assert len(_get(client).json()['searches']) == 1
    finally:
        with conn.cursor() as cur:
            cur.execute('RESET ROLE')


@requires_db
def test_a_role_without_select_on_the_table_gets_an_empty_list_one_clean_log_line_and_a_usable_connection(monkeypatch, conn, deny_role, stored, caplog):
    _search(conn, 'dededededededed1', seed=1)
    c = psycopg2.connect(**deny_role)
    try:
        with _client(monkeypatch, c) as client, caplog.at_level(logging.WARNING, logger='src.api.routes'):
            r = _get(client)
            assert r.status_code == 200, r.text
            assert r.json()['searches'] == [] and r.json()['truncated'] is False and r.json()['stress_run_id'] == stored['stress_run_id']
            assert _get(client).status_code == 200
        warnings = [x for x in caplog.records if x.levelno == logging.WARNING and x.name == 'src.api.routes']
        assert len(warnings) == 2
        message = warnings[0].getMessage()
        assert 'The other-searches table cannot be read by the API role' in message and 'DEPLOY.md' in message
        for secret in (deny_role['password'], deny_role['user'], 'search_runs', 'SELECT', 'host=', 'postgres'):
            assert secret not in message, secret
    finally:
        c.close()
