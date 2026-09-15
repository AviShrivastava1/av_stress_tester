"""
test_batch6_contract.py — Batch 6 contract tests (audit R04, R05).

The two findings share one idea, which is the reason they are one batch: A CANDIDATE
THAT EXACT-VERIFIES AS A COLLISION MUST NEVER BE DISCARDED IN FAVOUR OF ONE THAT DOES
NOT, ANYWHERE IN THE SEARCH — and the candidate that survives must be a legal member
of the space that was searched.

  R04  the DE stage threw away collisions it had already evaluated, because the
       penalty objective ranks CLOSER above FURTHER, not feasible above infeasible
  R05  the refiner's retained warm start was the one candidate never checked against
       the bounds every other candidate is clamped to

The audit's own repros live in tests/test_audit2_regressions.py, reconstructed and
labelled as such. THIS file holds the properties: the things that must be true of any
correct fix, including the ones the repro cannot see.

Pure numpy/torch/shapely, no database. Run:
    ./venv/bin/python -m pytest tests/test_batch6_contract.py -q
"""

import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.danger.collision_detector import check_collision_trajectory
from src.optimization import autograd_optimizer
from src.optimization.autograd_optimizer import refine_scenario
from src.optimization.perturbation_space import PerturbationSpace
from src.optimization.scipy_optimizer import (
    _DEObjective, keeps_challenger, optimize_scenario,
)


def _scene(n=2, t=10, lateral=2.1):
    states = np.zeros((n, t, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[1, :, 1] = lateral
    return states, np.ones((n, t), dtype=bool), np.ones(n, dtype=int)


def _space(lateral=2.1, bounds=None):
    states, validity, types = _scene(lateral=lateral)
    return PerturbationSpace(states, validity, types, 0, 1, bounds=bounds), validity


# ── one predicate, not three ────────────────────────────────────────────────────

# Every combination that matters, with the two tie directions spelled out rather than
# left to one representative case. `_stress_one`'s inline expression is reproduced
# verbatim from src/scoring/batch_scorer.py as the reference; if either drifts, the
# table below stops agreeing.
_PREDICATE_CASES = [
    # (challenger_collides, challenger_norm, incumbent_collides, incumbent_norm)
    (True,  0.5, True,  1.0),   # feasible and smaller -> replace
    (True,  1.5, True,  1.0),   # feasible and larger  -> keep incumbent
    (True,  1.0, True,  1.0),   # exact tie            -> keep incumbent
    (True,  9.9, False, 0.1),   # feasibility beats a smaller infeasible norm
    (False, 0.1, True,  9.9),   # infeasible never replaces feasible
    (False, 0.1, False, 9.9),   # infeasible never replaces anything
    (True,  0.5, False, float('inf')),   # the empty-incumbent case
    (False, 0.5, False, float('inf')),
]


def _stress_one_rule(challenger_collides, challenger_norm,
                     incumbent_collides, incumbent_norm):
    """
    batch_scorer._stress_one's DE-vs-refined comparison, transcribed:

        if refined['collision'] and (
            not result['collision']
            or refined['min_perturbation'] < result['min_perturbation']
        ):
    """
    refined = {'collision': challenger_collides, 'min_perturbation': challenger_norm}
    result = {'collision': incumbent_collides, 'min_perturbation': incumbent_norm}
    return bool(refined['collision'] and (
        not result['collision']
        or refined['min_perturbation'] < result['min_perturbation']
    ))


@pytest.mark.parametrize('case', _PREDICATE_CASES)
def test_the_de_selection_rule_matches_stress_one(case):
    """
    The DE-side sibling of Batch 5's test_the_selection_rule_matches_stress_one.

    The archive's internal choice and _stress_one's choice are the same comparison.
    An archive that kept a LARGER-norm collision than DE's own winner would hand
    _stress_one something it throws away; one that replaced on ties would churn.
    """
    assert keeps_challenger(*case) == _stress_one_rule(*case), (
        f'keeps_challenger and _stress_one disagree on {case}'
    )


def test_the_refiner_and_the_searcher_use_one_predicate():
    """
    Not "they behave the same" — they ARE the same object.

    Two copies that agree today are two copies that can drift tomorrow, and the
    drift is silent. This is the assertion that a re-inlined duplicate fails.
    """
    from src.optimization import scipy_optimizer
    assert autograd_optimizer.keeps_challenger is scipy_optimizer.keeps_challenger, (
        'the refiner has its own copy of the selection rule'
    )


# ── R04: the archive ────────────────────────────────────────────────────────────

@pytest.mark.parametrize('seed', [0, 1, 3, 5, 7])
def test_the_archive_never_returns_a_worse_answer_than_de(seed):
    """
    MONOTONICITY, which is what makes the fix safe to land under every existing test
    that runs the real DE path: R04 can only ever turn collision False -> True and
    can only ever lower min_perturbation. It can never lose a collision DE found, and
    never raise the reported norm.

    Checked against DE's own raw winner, recomputed here, rather than against a
    remembered value — so this cannot go red on a machine whose DE arithmetic
    differs.
    """
    from scipy.optimize import differential_evolution

    space, validity = _space(2.6)
    result = optimize_scenario(space, popsize=6, maxiter=25, seed=seed)

    # What DE alone would have returned. The archive does not change the objective's
    # return value, so this run walks the identical trajectory and res.x is exactly
    # the pre-R04 answer — derived on this machine, not remembered from another.
    reference = differential_evolution(
        _DEObjective(space, 1.0e3), [tuple(b) for b in space.bounds],
        popsize=6, mutation=(0.5, 1.0), recombination=0.7, maxiter=25, tol=1e-3,
        seed=seed, workers=1, polish=False, init='latinhypercube', disp=False,
    )
    de_delta = reference.x.astype(np.float32)
    de_hit, _ = check_collision_trajectory(space.apply(de_delta), validity, 0, 1)
    de_norm = space.weighted_norm(de_delta) if de_hit else float('inf')

    assert result['collision'] >= de_hit, (
        f'seed {seed}: DE alone found a collision and the fix lost it'
    )
    assert result['min_perturbation'] <= de_norm, (
        f"seed {seed}: reported {result['min_perturbation']} against DE's own "
        f'{de_norm}'
    )

    # and whatever won, every field describes it
    hit, t_hit = check_collision_trajectory(space.apply(result['delta']), validity, 0, 1)
    assert hit == result['collision'] and t_hit == result['collision_timestep']
    if result['collision']:
        assert result['min_perturbation'] == pytest.approx(
            space.weighted_norm(result['delta'])), (
            'min_perturbation describes a different delta than the one returned'
        )
    assert result['candidate_source'] in ('de_winner', 'archive')


def test_the_archive_only_ever_holds_a_verified_collision():
    """
    `g < 0` is claimed to BE the exact check, not a surrogate for it. If that claim
    were ever wrong the archive would hold a non-colliding candidate, so it is
    asserted directly against Shapely on every candidate the archive accepts.
    """
    space, validity = _space(2.1)   # a scene where random deltas collide often
    objective = _DEObjective(space, 1.0e3)

    rng = np.random.default_rng(0)
    accepted = 0
    for _ in range(300):
        delta = rng.uniform(space.bounds[:, 0], space.bounds[:, 1])
        before = objective.best_delta
        objective(delta)
        if objective.best_delta is not before:
            accepted += 1
            hit, _ = check_collision_trajectory(
                space.apply(objective.best_delta), validity, 0, 1)
            assert hit, (
                f'the archive accepted {objective.best_delta.tolist()}, which does '
                'not collide under the exact check'
            )
    assert accepted > 0, 'fixture regressed: the archive accepted nothing to check'


def test_the_archive_does_not_change_what_de_computes():
    """
    The archive remembers; it must not steer. The objective's RETURN VALUE is what
    DE sees, and it has to be the same number it was before — otherwise Batch 5's
    bit-identity guarantee for the workers=1 path is quietly gone.
    """
    from src.optimization.scipy_optimizer import _signed_gap
    space, _ = _space(2.6)
    lam = 1.0e3
    objective = _DEObjective(space, lam)

    rng = np.random.default_rng(1)
    for _ in range(25):
        delta = rng.uniform(space.bounds[:, 0], space.bounds[:, 1]).astype(np.float32)
        expected = (space.weighted_norm(delta) ** 2
                    + lam * max(0.0, _signed_gap(space.apply(delta), space.validity,
                                                 space.sdc_idx, space.target_idx)))
        assert objective(delta) == expected


def test_the_objective_still_pickles_with_an_archive():
    """
    B10's guarantee, re-checked against the new mutable state. An archive holding a
    numpy array must not be what finally breaks the pickling that B10 fixed.
    """
    space, _ = _space(2.6)
    objective = _DEObjective(space, 1.0e3)
    objective(np.array([2.9, 0.19, 0.5, -0.09], dtype=np.float32))
    revived = pickle.loads(pickle.dumps(objective))
    assert revived.best_norm == objective.best_norm
    probe = np.array([0.5, 0.01, 0.1, 0.01], dtype=np.float32)
    assert revived(probe) == _DEObjective(space, 1.0e3)(probe)


def test_the_workers_gap_is_recorded_not_hidden():
    """
    The archive cannot see what a child process evaluated, and that is stated in the
    result rather than left for a reader to discover. This is the honest half of the
    decision to scope the R04 guarantee to workers=1.

    differential_evolution is stubbed out, exactly as the audit's own B10 fixture
    does it, so no process pool is ever built — the point is the reported flag, not
    the parallel execution this environment cannot run anyway.
    """
    from types import SimpleNamespace
    from src.optimization import scipy_optimizer

    space, _ = _space(2.6)
    captured = {}

    def stub(function, bounds, **kwargs):
        captured['function'] = function
        return SimpleNamespace(x=np.zeros(4), nit=0, nfev=0)

    real = scipy_optimizer.differential_evolution
    scipy_optimizer.differential_evolution = stub
    try:
        parallel = optimize_scenario(space, workers=2)
        serial = optimize_scenario(space, workers=1)
    finally:
        scipy_optimizer.differential_evolution = real

    assert parallel['archive_covered_all_evaluations'] is False
    assert serial['archive_covered_all_evaluations'] is True
    pickle.dumps(captured['function'])  # B10 still holds with the archive attached


# ── R05: the retained warm start must be legal ──────────────────────────────────

def test_the_rejection_names_the_offending_component():
    """
    Which fix was chosen, pinned. The audit repro accepts a clamp too; this does not.

    A rejection that says only "out of bounds" makes the caller find it themselves,
    so the message has to carry the component, its value and its bound.
    """
    bounds = np.array([[0, 0], [-0.01, 0.01], [0, 0], [0, 0]], dtype=np.float32)
    space, _ = _space(2.1, bounds=bounds)

    with pytest.raises(ValueError) as excinfo:
        refine_scenario(space, delta_init=np.array([0.0, 0.15, 0.0, 0.0], np.float32),
                        n_iters=5)

    message = str(excinfo.value)
    assert 'component 1' in message, message
    assert '0.15' in message, message
    assert '0.00999' in message, message


def test_every_offending_component_is_named_not_just_the_first():
    """A caller fixing one violation at a time re-runs the search to learn the next."""
    bounds = np.array([[-0.5, 0.5], [-0.01, 0.01], [-0.5, 0.5], [-0.01, 0.01]],
                      dtype=np.float32)
    space, _ = _space(2.1, bounds=bounds)

    with pytest.raises(ValueError) as excinfo:
        refine_scenario(space, delta_init=np.array([0.0, 0.15, 0.0, -0.9], np.float32),
                        n_iters=5)

    message = str(excinfo.value)
    assert 'component 1' in message and 'component 3' in message, message


def test_a_default_zero_warm_start_outside_the_box_is_also_rejected():
    """
    The check runs on the value ACTUALLY USED, after delta_init has defaulted. A box
    that excludes the origin makes the implicit zero warm start illegal, and that is
    as much a caller error as an explicit one — the alternative is a default that
    silently gets a licence no caller-supplied value gets, which is the exact shape
    of the defect R05 describes.
    """
    bounds = np.array([[0.1, 0.2], [0, 0], [0, 0], [0, 0]], dtype=np.float32)
    space, _ = _space(2.1, bounds=bounds)

    with pytest.raises(ValueError) as excinfo:
        refine_scenario(space, n_iters=5)
    assert 'component 0' in str(excinfo.value)


def test_an_in_bounds_warm_start_is_unaffected():
    """
    The fix must not become a tax on the normal path. The default bounds allow a
    heading offset of 0.20, so Batch 5's B09 warm start of 0.15 is legal and must
    behave exactly as it did.
    """
    space, validity = _space(2.1)
    warm = np.array([0.0, 0.15, 0.0, 0.0], dtype=np.float32)
    assert warm[1] <= space.bounds[1, 1], 'fixture regressed: the warm start is illegal'

    result = refine_scenario(space, delta_init=warm, n_iters=100)

    assert result['collision']
    assert result['min_perturbation'] <= space.weighted_norm(warm) + 1e-9
    delta = np.asarray(result['delta'], np.float32)
    assert np.all(delta >= space.bounds[:, 0]) and np.all(delta <= space.bounds[:, 1])


def test_a_warm_start_exactly_on_the_boundary_is_legal():
    """
    Closed bounds, not open ones. DE returns points ON its bounds routinely, so an
    exclusive comparison would reject the pipeline's own warm starts — turning a
    dead-in-practice contract defect into a live production failure.
    """
    bounds = np.array([[0, 0], [0, 0.2], [0, 0], [0, 0]], dtype=np.float32)
    space, _ = _space(2.1, bounds=bounds)
    for value in (0.0, 0.2):
        warm = np.array([0.0, value, 0.0, 0.0], dtype=np.float32)
        refine_scenario(space, delta_init=warm, n_iters=2)  # must not raise


def test_the_pipelines_own_warm_start_is_always_legal():
    """
    The trace behind calling R05 dead-in-practice, asserted rather than described:
    every delta optimize_scenario returns — DE's winner or an archived candidate —
    is inside the bounds, so _stress_one's handoff to refine_scenario cannot raise.
    """
    for seed in (0, 3, 5):
        space, _ = _space(2.6)
        result = optimize_scenario(space, popsize=6, maxiter=25, seed=seed)
        delta = np.asarray(result['delta'], np.float32)
        assert np.all(delta >= space.bounds[:, 0]) and np.all(delta <= space.bounds[:, 1]), (
            f"seed {seed} returned {delta.tolist()} outside {space.bounds.tolist()} "
            f"(source={result['candidate_source']})"
        )
        refine_scenario(space, delta_init=result['delta'], n_iters=2)  # must not raise
