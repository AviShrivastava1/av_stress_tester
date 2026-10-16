"""
The convergence trace of a Differential Evolution search (optimize_scenario(..., trace=[])).

The trace is a record of one generation after another: how many evaluations had been spent and the smallest weighted norm of an
exactly verified colliding delta found so far. It must be a record only. The first group of tests runs the same search twice, with
and without a trace, and compares what comes back bit for bit; the second group checks that the trace says what it claims to.
"""
import numpy as np
import pytest

from src.optimization.scipy_optimizer import optimize_scenario
from src.optimization.perturbation_space import PerturbationSpace


def _scene(n=2, t=10):
    states = np.zeros((n, t, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    return states, np.ones((n, t), dtype=bool), np.ones(n, dtype=int)


def _colliding_space():
    # the challenger runs 2.1 m beside the SDC: a small push sideways collides
    states, validity, types = _scene(t=10)
    states[1, :, 1] = 2.1
    return PerturbationSpace(states, validity, types, 0, 1)


def _far_space():
    # 20 m ahead and stationary: no collision inside the default bounds
    states, validity, types = _scene(t=2)
    states[1, :, 0] = 200.0
    return PerturbationSpace(states, validity, types, 0, 1)


BUDGETS = [(0, 5, 3), (1, 8, 4), (7, 6, 5), (42, 15, 6)]  # seed, popsize, maxiter
RESULT_KEYS = ['collision', 'min_perturbation', 'delta', 'collision_timestep', 'margin', 'n_iter', 'n_eval', 'candidate_source']


def _same(a, b):
    for k in RESULT_KEYS:
        x, y = a[k], b[k]
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype and x.tobytes() == y.tobytes(), k
        else:
            assert repr(x) == repr(y), (k, x, y)


@pytest.mark.parametrize('seed,popsize,maxiter', BUDGETS)
def test_a_trace_changes_nothing_that_comes_back(seed, popsize, maxiter):
    plain = optimize_scenario(_colliding_space(), popsize=popsize, maxiter=maxiter, seed=seed)
    traced = optimize_scenario(_colliding_space(), popsize=popsize, maxiter=maxiter, seed=seed, trace=[])
    _same(plain, traced)


@pytest.mark.parametrize('seed,popsize,maxiter', BUDGETS)
def test_a_trace_changes_nothing_when_no_collision_is_found(seed, popsize, maxiter):
    plain = optimize_scenario(_far_space(), popsize=popsize, maxiter=maxiter, seed=seed)
    traced = optimize_scenario(_far_space(), popsize=popsize, maxiter=maxiter, seed=seed, trace=[])
    _same(plain, traced)
    assert plain['collision'] is False


def test_the_same_search_twice_with_a_trace_gives_the_same_trace():
    a, b = [], []
    optimize_scenario(_colliding_space(), popsize=6, maxiter=5, seed=3, trace=a)
    optimize_scenario(_colliding_space(), popsize=6, maxiter=5, seed=3, trace=b)
    assert a == b and a


def test_one_point_per_generation_numbered_from_one():
    trace = []
    res = optimize_scenario(_colliding_space(), popsize=6, maxiter=5, seed=3, trace=trace)
    assert [p['generation'] for p in trace] == list(range(1, len(trace) + 1))
    assert len(trace) == res['n_iter']


def test_evaluations_only_go_up_and_the_last_point_is_the_search_total():
    trace = []
    res = optimize_scenario(_colliding_space(), popsize=6, maxiter=5, seed=3, trace=trace)
    counts = [p['n_eval_cumulative'] for p in trace]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)
    assert counts[-1] == res['n_eval']


def test_the_best_norm_never_rises_and_ends_at_what_the_archive_found():
    trace = []
    res = optimize_scenario(_colliding_space(), popsize=10, maxiter=12, seed=1, trace=trace)
    seen = [p['best_weighted_norm'] for p in trace if p['best_weighted_norm'] is not None]
    assert seen and all(b <= a for a, b in zip(seen, seen[1:]))
    # None only before the first verified collision, never after it
    first = next(i for i, p in enumerate(trace) if p['best_weighted_norm'] is not None)
    assert all(p['best_weighted_norm'] is not None for p in trace[first:])
    assert res['collision'] and res['min_perturbation'] <= seen[-1] + 1e-12


def test_no_collision_means_no_best_norm_at_any_generation():
    trace = []
    optimize_scenario(_far_space(), popsize=5, maxiter=4, seed=0, trace=trace)
    assert trace and all(p['best_weighted_norm'] is None for p in trace)


def test_every_point_has_exactly_the_three_documented_keys_with_plain_types():
    trace = []
    optimize_scenario(_colliding_space(), popsize=6, maxiter=4, seed=0, trace=trace)
    for p in trace:
        assert set(p) == {'generation', 'n_eval_cumulative', 'best_weighted_norm'}
        assert type(p['generation']) is int and type(p['n_eval_cumulative']) is int
        assert p['best_weighted_norm'] is None or type(p['best_weighted_norm']) is float


def test_a_trace_needs_one_worker_and_says_so_rather_than_recording_nothing():
    with pytest.raises(ValueError, match='workers=1'):
        optimize_scenario(_colliding_space(), popsize=5, maxiter=2, seed=0, workers=2, trace=[])


def test_without_a_trace_the_search_is_called_with_no_callback(monkeypatch):
    import src.optimization.scipy_optimizer as mod
    seen = {}
    real = mod.differential_evolution

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(mod, 'differential_evolution', spy)
    optimize_scenario(_colliding_space(), popsize=5, maxiter=2, seed=0)
    assert seen['callback'] is None


def test_the_counting_wrapper_returns_exactly_what_the_objective_returns_for_the_same_argument():
    # Not rounded, not cast: a float64 candidate that float32 cannot hold exactly gives the same value, and the counter counts one each.
    from src.optimization.scipy_optimizer import LAMBDA, _Counted, _DEObjective
    space = _colliding_space()
    rng = np.random.default_rng(5)
    direct, wrapped = _DEObjective(space, LAMBDA), _DEObjective(space, LAMBDA)
    counted = _Counted(wrapped)
    for i in range(40):
        delta = rng.uniform(-0.3, 0.3, 4) * (1 + 1e-9 * rng.random(4))
        assert repr(counted(delta)) == repr(direct(delta))
        assert counted.n == i + 1
    assert wrapped.best_norm == direct.best_norm and repr(wrapped.best_norm) == repr(direct.best_norm)
