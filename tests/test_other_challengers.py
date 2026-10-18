"""
Other challengers (#1): the ranking (src/optimization/perturbation_space.py rank_challengers), the optional `challenger_rank` of
`_stress_one`, and the offline runner `run_challenger_searches` (src/scoring/other_searches.py).

Pinned here:

  * the ranking is the one the pipeline picks its one challenger from: the nearest is the first of it, on every scene (held equal
    against the implementation that was there before the ranking existed), ties go to the lower index, an agent that shares no valid
    timestep with the SDC is not ranked;
  * `challenger_rank` is the smallest optional parameter: left out, or 1, the search, its provenance and its stress_run_id are what
    they were (two ids recorded from the code before the parameter existed are pinned); k > 1 searches the k-th nearest and says so;
  * the runner: one search_runs row per challenger of kind 'challenger' with its rank (1 = nearest), target_idx, model, delta, outcome,
    n_iter and n_eval, with the stored seed and budget; the nearest, searched with the stored configuration, is the stored result and is
    skipped by its id; a rerun writes nothing; an id that is already there is left as it is and reported; k that is not an integer from
    1 to the number of challengers that can be ranked, a scene that is not the one searched and a missing table are refused before any
    search runs; a challenger whose search does not run is reported and the others go on; a pedestrian is recorded in the linear model;
    scenario_scores and every other table are untouched.

The database tests build a throw-away database and drop it after. They skip unless AV_ROBUSTNESS_DB=1.

Run:
    AV_ROBUSTNESS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_other_challengers.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import other_searches as served
from src.api.models import DELTA_LABELS_BY_PARAMETERIZATION
from src.optimization.perturbation_space import pick_nearest_challenger, rank_challengers
from src.scoring import db, other_searches as runner
from src.scoring.batch_scorer import _stress_one
from src.scoring.manifest import ManifestTablesMissing
from src.scoring.search_runs import TraceRerunRefused

from tests.test_trace import (  # noqa: F401  (the fixtures are used by name)
    BUDGET, _count, _fingerprint_of_the_results, _insert_scenario, _scene, bare_db, client, requires_db, tr_db,
)

#: stress_run_id of the default path on tests.test_trace._scene() with BUDGET, and on _multi_scene() with BUDGET, as the code computed them
#: BEFORE `challenger_rank` existed
GOLDEN_ID_STANDARD_SCENE = 'f1e54296099c0236'
GOLDEN_ID_MULTI_SCENE = 'aa2ad7303c69945b'


def _multi_scene():
    """
    The SDC (agent 0) and four others, whose order by distance is not their order by index: agent 2 is the nearest (a vehicle, 2.1 m
    to the side), then agent 3 (a pedestrian, 4.0 m), then agent 1 (a vehicle, 7.5 m); agent 4 would be nearer than agent 1 but is
    never valid at the same time as the SDC, so it cannot be ranked.
    """
    states = np.zeros((5, 10, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[1, :, 1] = 7.5
    states[2, :, 1] = 2.1
    states[3, :, 1] = 4.0
    states[3, :, 5:7] = [0.8, 0.8]
    states[4, :, 1] = 3.0
    validity = np.ones((5, 10), dtype=bool)
    validity[4, :] = False
    return states, validity, np.array([1, 1, 1, 2, 1])


RANKED = [2, 3, 1]
NEAREST_REASON = 'the nearest challenger, searched with the stored seed and budget, is the stored result; not searched again'


# ── the ranking ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def _previous_pick_nearest_challenger(states, validity, sdc_idx) -> int:
    """The implementation as it was before the ranking existed, kept here to hold the new one equal to it."""
    N, T, _ = states.shape
    best_idx, best_d = -1, np.inf
    sdc_xy = states[sdc_idx, :, :2]
    for j in range(N):
        if j == sdc_idx:
            continue
        shared = validity[sdc_idx] & validity[j]
        if not shared.any():
            continue
        d = np.linalg.norm(sdc_xy[shared] - states[j, shared, :2], axis=1).min()
        if d < best_d:
            best_d, best_idx = d, j
    return best_idx


def test_the_challengers_are_ranked_nearest_first_whatever_their_index_and_one_that_cannot_be_ranked_is_left_out():
    assert rank_challengers(*_multi_scene()[:2], 0) == RANKED
    assert pick_nearest_challenger(*_multi_scene()[:2], 0) == RANKED[0]


def test_a_tie_goes_to_the_lower_index_and_the_sdc_is_never_ranked():
    states = np.zeros((4, 5, 7), dtype=np.float32)
    states[1, :, 1] = 3.0
    states[2, :, 1] = -3.0
    states[3, :, 1] = 3.0
    validity = np.ones((4, 5), dtype=bool)
    assert rank_challengers(states, validity, 0) == [1, 2, 3]
    assert rank_challengers(states, validity, 2) == [0, 1, 3]                  # seen from agent 2: agent 0 is 3 m away, 1 and 3 are 6 m (a tie); agent 2 is left out


def test_the_distance_is_the_closest_approach_over_the_timesteps_both_are_valid_only():
    states = np.zeros((3, 4, 7), dtype=np.float32)
    states[1, :, 1] = [1.0, 9.0, 9.0, 9.0]           # agent 1 is close only at step 0
    states[2, :, 1] = 5.0
    validity = np.ones((3, 4), dtype=bool)
    assert rank_challengers(states, validity, 0) == [1, 2]
    validity[1, 0] = False                            # now its only close step does not count
    assert rank_challengers(states, validity, 0) == [2, 1]


def test_nothing_is_ranked_when_nothing_shares_a_valid_step_or_the_distance_is_not_a_number():
    states = np.zeros((3, 4, 7), dtype=np.float32)
    validity = np.ones((3, 4), dtype=bool)
    validity[1:, :] = False
    assert rank_challengers(states, validity, 0) == [] and pick_nearest_challenger(states, validity, 0) == -1
    validity[:] = True
    states[1, :, 0] = np.nan
    assert rank_challengers(states, validity, 0) == [2] and pick_nearest_challenger(states, validity, 0) == 2


def test_the_nearest_is_what_the_pipeline_picked_before_the_ranking_existed_on_many_scenes():
    rng = np.random.default_rng(11)
    for _ in range(300):
        n, t = int(rng.integers(1, 8)), int(rng.integers(1, 12))
        states = rng.integers(-4, 5, size=(n, t, 7)).astype(np.float32)          # integers, so that ties are common
        validity = rng.random((n, t)) < 0.7
        if rng.random() < 0.2:
            states[int(rng.integers(0, n)), :, :2] = np.nan
        sdc = int(rng.integers(0, n))
        assert pick_nearest_challenger(states, validity, sdc) == _previous_pick_nearest_challenger(states, validity, sdc)
        ranked = rank_challengers(states, validity, sdc)
        assert sdc not in ranked and len(set(ranked)) == len(ranked)


# ── challenger_rank of _stress_one ──────────────────────────────────────────────────────────────────────────────────────────────


def _id_of(result, scenario_id='t1'):
    return db.compute_stress_run_id(scenario_id, result['target_idx'], result['delta'], result['method'], result['search_provenance'])


def test_the_default_path_is_byte_identical_to_what_it_was():
    standard = _stress_one(*_scene(), 0, de_kwargs=dict(BUDGET))
    multi = _stress_one(*_multi_scene(), 0, de_kwargs=dict(BUDGET))
    assert _id_of(standard) == GOLDEN_ID_STANDARD_SCENE and _id_of(multi) == GOLDEN_ID_MULTI_SCENE
    assert standard['search_provenance']['challenger_selection'] == 'nearest_by_min_center_distance'
    assert 'challenger_rank' not in standard


def test_the_nearest_by_rank_is_the_default_search_and_has_its_id_the_next_ones_say_which_rank_they_are():
    default = _stress_one(*_multi_scene(), 0, de_kwargs=dict(BUDGET))
    first = _stress_one(*_multi_scene(), 0, de_kwargs=dict(BUDGET), challenger_rank=1)
    assert _id_of(first) == _id_of(default) == GOLDEN_ID_MULTI_SCENE and first['target_idx'] == RANKED[0]
    for k in (2, 3):
        r = _stress_one(*_multi_scene(), 0, de_kwargs=dict(BUDGET), challenger_rank=k)
        assert r['status'] == 'ok' and r['target_idx'] == RANKED[k - 1]
        assert r['search_provenance']['challenger_selection'] == f'rank_{k}_by_min_center_distance'
        assert _id_of(r) != GOLDEN_ID_MULTI_SCENE
    assert _stress_one(*_multi_scene(), 0, de_kwargs=dict(BUDGET), challenger_rank=3)['search_provenance']['delta_parameterization'] == 'bicycle'
    assert _stress_one(*_multi_scene(), 0, de_kwargs=dict(BUDGET), challenger_rank=2)['search_provenance']['delta_parameterization'] == 'linear'


@pytest.mark.parametrize('rank', [0, -1, -3, 4, 5, 99])
def test_a_rank_beyond_the_challengers_that_can_be_ranked_is_no_challenger(rank):
    out = _stress_one(*_multi_scene(), 0, de_kwargs=dict(BUDGET), challenger_rank=rank)
    assert out['status'] == 'no_challenger' and out['challengers_searched'] == 0


# ── the runner: what it validates before it runs anything ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize('k', [0, -1, 4, 99, True, False, 1.0, 2.5, '2', None, [2]])
def test_a_k_that_is_not_an_integer_from_one_to_the_number_of_rankable_challengers_is_refused_before_anything_runs(k, monkeypatch):
    called = []
    monkeypatch.setattr(runner, '_run', lambda *a, **kw: called.append(1))
    with pytest.raises(ValueError, match='k is'):
        runner.run_challenger_searches(None, 't1', *_multi_scene(), 0, k=k)
    assert called == []


@pytest.mark.parametrize('k', [1, 2, 3])
def test_a_k_up_to_the_number_of_rankable_challengers_is_passed_on_with_one_rank_each(k, monkeypatch):
    seen = []
    monkeypatch.setattr(runner, '_run', lambda conn, kind, *a, **kw: seen.append((kind, a[-1]({'popsize': 5, 'maxiter': 4, 'tol': 0.001, 'seed': 2}))))
    runner.run_challenger_searches(None, 't1', *_multi_scene(), 0, k=k)
    [(kind, configs)] = seen
    assert kind == 'challenger' and [c['challenger_rank'] for c in configs] == list(range(1, k + 1))
    assert all(c['seed'] == 2 and c['popsize'] == 5 and c['maxiter'] == 4 for c in configs)


# ── the runner, on a database ───────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope='module')
def multi_stored():
    states, validity, types = _multi_scene()
    result = _stress_one(states, validity, types, 0, de_kwargs=dict(BUDGET))
    assert result['status'] == 'ok' and result['collision'] and result['target_idx'] == RANKED[0]
    prov = result['search_provenance']
    return {
        'scenario_id': 't1', 'delta': [float(x) for x in result['delta']], 'target_idx': result['target_idx'], 'stress_method': result['method'],
        'provenance': prov, 'n_iter': prov['de_n_iter'], 'n_eval': prov['de_n_eval'],
        'stress_run_id': db.compute_stress_run_id('t1', result['target_idx'], result['delta'], result['method'], prov),
        'fingerprint': db.compute_scene_fingerprint(states, validity, types),
    }


@pytest.fixture
def conn(tr_db, multi_stored):
    _, c = tr_db
    with c.cursor() as cur:
        cur.execute('DELETE FROM search_runs')
        cur.execute('DELETE FROM scenario_scores')
    _insert_scenario(c, multi_stored)
    return c


def _rows(c):
    with c.cursor() as cur:
        cur.execute('SELECT search_run_id, kind, challenger_rank, target_idx, delta_parameterization, seed, popsize, maxiter, tol, outcome, delta, '
                    'n_iter, n_eval, scene_fingerprint, sdc_idx, reruns_stress_run_id FROM search_runs ORDER BY challenger_rank NULLS FIRST, search_run_id')
        return cur.fetchall()


@requires_db
def test_the_nearest_is_the_stored_result_and_is_skipped_the_others_are_recorded_in_rank_order(conn, multi_stored):
    report = runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=3)
    assert [(s['challenger_rank'], s['target_idx'], s['reason'], s['search_run_id']) for s in report['skipped']] == [
        (1, RANKED[0], NEAREST_REASON, multi_stored['stress_run_id'])]
    assert [(r['challenger_rank'], r['target_idx'], r['status']) for r in report['recorded']] == [(2, 3, 'recorded'), (3, 1, 'recorded')]
    assert report['conflicts'] == [] and report['not_run'] == [] and report['kind'] == 'challenger'
    rows = _rows(conn)
    assert [(r[1], r[2], r[3]) for r in rows] == [('challenger', 2, 3), ('challenger', 3, 1)]
    for r in rows:
        assert r[0] not in (None, multi_stored['stress_run_id']) and r[5:9] == (BUDGET['seed'], BUDGET['popsize'], BUDGET['maxiter'], BUDGET['tol'])
        assert r[9] in ('collision_found', 'no_collision_found') and len(r[10]) == 4 and r[11] is not None and r[12] is not None
        assert r[13] == multi_stored['fingerprint'] and r[14] == 0 and r[15] is None


@requires_db
def test_a_challenger_of_another_type_is_recorded_in_its_own_model(conn):
    runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=3)
    assert [(r[2], r[3], r[4]) for r in _rows(conn)] == [(2, 3, 'linear'), (3, 1, 'bicycle')]      # the pedestrian is in the linear model


def _spy(monkeypatch):
    """The ranks _stress_one is asked for, in order."""
    from src.scoring import batch_scorer
    real = batch_scorer._stress_one
    calls = []
    monkeypatch.setattr(batch_scorer, '_stress_one', lambda *a, **kw: calls.append(kw.get('challenger_rank')) or real(*a, **kw))
    return calls


def _store_as(conn, **columns):
    with conn.cursor() as cur:
        for name, value in columns.items():
            cur.execute(f"UPDATE scenario_scores SET {name} = %s WHERE scenario_id = 't1'", (value,))


@requires_db
def test_the_nearest_is_not_searched_at_all_when_it_is_the_stored_target_and_the_others_are(conn, multi_stored, monkeypatch):
    calls = _spy(monkeypatch)
    report = runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=3)
    assert calls == [2, 3]
    assert report['skipped'] == [{'seed': BUDGET['seed'], 'popsize': BUDGET['popsize'], 'maxiter': BUDGET['maxiter'], 'tol': BUDGET['tol'],
                                  'challenger_rank': 1, 'status': 'skipped', 'target_idx': RANKED[0], 'search_run_id': multi_stored['stress_run_id'],
                                  'reason': NEAREST_REASON}]
    assert [r['challenger_rank'] for r in report['recorded']] == [2, 3]


@requires_db
def test_a_k_of_one_is_only_the_nearest_which_is_the_stored_result_so_no_search_runs_and_nothing_is_recorded(conn, monkeypatch):
    calls = _spy(monkeypatch)
    report = runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=1)
    assert calls == [] and len(report['skipped']) == 1 and report['recorded'] == [] and _count(conn, 'search_runs') == 0


@requires_db
def test_when_the_nearest_is_not_the_stored_target_it_is_searched_and_recorded_with_every_other_rank(conn, multi_stored, monkeypatch):
    _store_as(conn, target_idx=1, stress_run_id='ffffffffffffffff')
    calls = _spy(monkeypatch)
    report = runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=3)
    assert calls == [1, 2, 3]
    assert report['skipped'] == [] and [(r['challenger_rank'], r['target_idx']) for r in report['recorded']] == [(1, 2), (2, 3), (3, 1)]
    assert [(r[2], r[3]) for r in _rows(conn)] == [(1, 2), (2, 3), (3, 1)]
    assert _rows(conn)[0][0] == multi_stored['stress_run_id']             # the nearest, with the stored configuration, has the stored result's own id


@requires_db
def test_a_stored_result_that_does_not_say_its_target_is_not_taken_for_the_nearest_so_its_id_decides_after_the_search(conn, multi_stored, monkeypatch):
    _store_as(conn, target_idx=None)
    calls = _spy(monkeypatch)
    report = runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=2)
    assert calls == [1, 2]
    assert [(s['challenger_rank'], s['target_idx'], s['reason'], s['search_run_id']) for s in report['skipped']] == [
        (1, RANKED[0], 'same as the stored result', multi_stored['stress_run_id'])]
    assert [r['challenger_rank'] for r in report['recorded']] == [2]


@requires_db
def test_another_rank_that_turns_out_to_be_the_stored_result_is_skipped_after_it_ran(conn, monkeypatch):
    first = runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=3)
    rank_two = first['recorded'][0]['search_run_id']
    with conn.cursor() as cur:
        cur.execute('DELETE FROM search_runs')
    _store_as(conn, stress_run_id=rank_two)
    calls = _spy(monkeypatch)
    report = runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=3)
    assert calls == [2, 3]               # rank one is the stored target and is not searched; rank two is, and turns out to be the stored result
    assert [(s['challenger_rank'], s['target_idx'], s['reason'], s['search_run_id']) for s in report['skipped']] == [
        (1, RANKED[0], NEAREST_REASON, rank_two), (2, RANKED[1], 'same as the stored result', rank_two)]
    assert [r['challenger_rank'] for r in report['recorded']] == [3]


@requires_db
def test_a_second_run_records_nothing_and_changes_nothing(conn):
    runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=3)
    before = _rows(conn)
    again = runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=3)
    assert again['recorded'] == [] and [c['held_by_kind'] for c in again['conflicts']] == ['challenger', 'challenger'] and len(again['skipped']) == 1
    assert _rows(conn) == before and _count(conn, 'search_traces') == 0


@requires_db
def test_an_id_that_is_already_there_is_left_as_it_is_and_no_trace_points_are_written(conn):
    first = runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=2)
    sid = first['recorded'][0]['search_run_id']
    with conn.cursor() as cur:
        cur.execute("UPDATE search_runs SET kind = 'budget', challenger_rank = NULL, n_eval = 7777 WHERE search_run_id = %s", (sid,))
        cur.execute('INSERT INTO search_traces VALUES (%s, 1, 10, 0.5)', (sid,))
    before = _rows(conn)
    again = runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=2)
    assert again['recorded'] == [] and [(c['search_run_id'], c['status'], c['held_by_kind'], c['challenger_rank']) for c in again['conflicts']] == [
        (sid, 'conflict', 'budget', 2)]
    assert _rows(conn) == before and _count(conn, 'search_traces') == 1


@requires_db
def test_the_reports_of_the_seed_and_budget_runners_are_what_they_were_and_have_no_not_run(conn):
    for report in (runner.run_seed_searches(conn, 't1', *_multi_scene(), 0, seeds=[3]),
                   runner.run_budget_searches(conn, 't1', *_multi_scene(), 0, budgets=[{'maxiter': 3}])):
        assert set(report) == {'kind', 'scenario_id', 'recorded', 'skipped', 'conflicts'}
        assert all('challenger_rank' not in entry and 'target_idx' not in entry for entry in report['recorded'])


@requires_db
def test_a_search_that_does_not_run_is_reported_and_the_others_go_on(conn, monkeypatch):
    from src.scoring import batch_scorer
    real = batch_scorer._stress_one

    def fake(*a, **kw):
        if kw.get('challenger_rank') == 2:
            return {'status': 'replay_infeasible', 'outcome': 'replay_infeasible', 'target_idx': 3}
        return real(*a, **kw)
    monkeypatch.setattr(batch_scorer, '_stress_one', fake)
    report = runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=3)
    assert report['not_run'] == [{'seed': BUDGET['seed'], 'popsize': BUDGET['popsize'], 'maxiter': BUDGET['maxiter'], 'tol': BUDGET['tol'],
                                  'challenger_rank': 2, 'status': 'not_run', 'target_idx': 3, 'reason': 'the search did not run (replay_infeasible)'}]
    assert [r['challenger_rank'] for r in report['recorded']] == [3] and [(r[2], r[3]) for r in _rows(conn)] == [(3, 1)]


@requires_db
def test_it_writes_only_search_runs_and_never_the_stored_result(conn):
    before = _fingerprint_of_the_results(conn)
    runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=3)
    assert _fingerprint_of_the_results(conn) == before
    assert _count(conn, 'search_runs') == 2
    for table in ('search_traces', 'pipeline_runs', 'pipeline_run_results', 'sensitivity_slices', 'shards'):
        assert _count(conn, table) == 0, table


def _no_search_may_run(monkeypatch):
    from src.scoring import batch_scorer

    def boom(*a, **kw):
        raise AssertionError('a search was run')
    monkeypatch.setattr(batch_scorer, '_stress_one', boom)


@requires_db
@pytest.mark.parametrize('k', [0, True, 4, 2.0, '2'])
def test_a_bad_k_is_refused_with_the_database_given_before_a_search_runs(conn, monkeypatch, k):
    _no_search_may_run(monkeypatch)
    with pytest.raises(ValueError, match='k is'):
        runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=k)
    assert _count(conn, 'search_runs') == 0


@requires_db
def test_a_scene_that_is_not_the_one_searched_and_another_sdc_and_no_stored_search_are_refused_before_a_search_runs(conn, monkeypatch):
    _no_search_may_run(monkeypatch)
    states, validity, types = _multi_scene()
    states[1, :, 1] = 7.6
    with pytest.raises(TraceRerunRefused, match='scene'):
        runner.run_challenger_searches(conn, 't1', states, validity, types, 0, k=2)
    with pytest.raises(TraceRerunRefused, match='SDC'):
        runner.run_challenger_searches(conn, 't1', *_multi_scene(), 1, k=2)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO scenario_scores (scenario_id, shard, fragility_score) VALUES ('bare', 'a', 1.0)")
    with pytest.raises(TraceRerunRefused, match='no stored search'):
        runner.run_challenger_searches(conn, 'bare', *_multi_scene(), 0, k=2)
    with pytest.raises(TraceRerunRefused, match='no such scenario'):
        runner.run_challenger_searches(conn, 'nope', *_multi_scene(), 0, k=2)
    assert _count(conn, 'search_runs') == 0


@requires_db
def test_a_missing_search_runs_table_is_a_clear_error(bare_db, monkeypatch):
    _no_search_may_run(monkeypatch)
    _, c = bare_db
    with c.cursor() as cur:
        cur.execute('DELETE FROM scenario_scores')
    with pytest.raises(ManifestTablesMissing, match='search_runs'):
        runner.run_challenger_searches(c, 't1', *_multi_scene(), 0, k=2)


# ── what the API serves of them ─────────────────────────────────────────────────────────────────────────────────────────────────

CHALLENGER_ROW = {'search_run_id': '0123456789abcdef', 'kind': 'challenger', 'challenger_rank': 2, 'target_idx': 5, 'seed': 2, 'popsize': 5,
                  'maxiter': 4, 'tol': 0.001, 'outcome': 'collision_found', 'min_perturbation': 0.25, 'collision_timestep': 4,
                  'delta': [0.1, 0.2, 0.3, 0.4], 'delta_parameterization': 'linear', 'n_iter': 3, 'n_eval': 90}


def test_a_challenger_row_is_served_with_its_rank_and_its_agent():
    assert served.select_search(CHALLENGER_ROW) == CHALLENGER_ROW


@pytest.mark.parametrize('field,bad', [
    ('challenger_rank', 0), ('challenger_rank', -3), ('challenger_rank', True), ('challenger_rank', 1.0), ('challenger_rank', '2'),
    ('challenger_rank', [2]), ('challenger_rank', object()), ('challenger_rank', float('nan')),
    ('target_idx', -1), ('target_idx', True), ('target_idx', 5.0), ('target_idx', '5'), ('target_idx', {}), ('target_idx', object()),
])
def test_a_rank_or_an_agent_of_the_wrong_type_is_null_and_the_rest_of_the_row_is_kept_never_an_error(field, bad):
    out = served.select_search({**CHALLENGER_ROW, field: bad})
    assert out[field] is None
    assert {k: v for k, v in out.items() if k != field} == {k: v for k, v in CHALLENGER_ROW.items() if k != field}


def _get(client):
    return client.get('/scenarios/t1/other-searches')


@requires_db
def test_the_route_serves_the_recorded_challengers_in_rank_order_each_in_its_own_model(conn, client, multi_stored):
    runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=3)
    body = _get(client).json()
    assert body['stored_delta_parameterization'] == 'bicycle' and body['truncated'] is False
    rows = body['searches']
    assert [(r['kind'], r['challenger_rank'], r['target_idx'], r['delta_parameterization']) for r in rows] == [
        ('challenger', 2, 3, 'linear'), ('challenger', 3, 1, 'bicycle')]
    assert [r['delta_labels'] for r in rows] == [list(DELTA_LABELS_BY_PARAMETERIZATION['linear']), list(DELTA_LABELS_BY_PARAMETERIZATION['bicycle'])]
    for r in rows:
        assert (r['seed'], r['popsize'], r['maxiter'], r['tol']) == (BUDGET['seed'], BUDGET['popsize'], BUDGET['maxiter'], BUDGET['tol'])
        assert r['search_run_id'] != multi_stored['stress_run_id'] and r['n_eval'] is not None and len(r['delta']) == 4


@requires_db
def test_the_nearest_challenger_searched_with_the_stored_configuration_is_never_listed_because_it_is_the_stored_result(conn, client, multi_stored):
    _store_as(conn, target_idx=1, stress_run_id='ffffffffffffffff')
    runner.run_challenger_searches(conn, 't1', *_multi_scene(), 0, k=3)
    assert [r['challenger_rank'] for r in _get(client).json()['searches']] == [1, 2, 3]
    _store_as(conn, target_idx=RANKED[0], stress_run_id=multi_stored['stress_run_id'])
    assert [r['challenger_rank'] for r in _get(client).json()['searches']] == [2, 3]
