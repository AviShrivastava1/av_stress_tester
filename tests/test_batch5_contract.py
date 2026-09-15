"""
test_batch5_contract.py — Batch 5 regression tests (audit B09, B10, B11, B16, B20).

THIS BATCH HAS NO SHARED INVARIANT, and that is the accurate description rather than
a gap in the writing. Batches 1, 2 and 4 each had one real unifying idea. These are
five independent narrow fixes that happened to be what was left, and pretending
otherwise would be inventing a story to match a format.

  B09  refinement kept the last iterate instead of the best verified one
  B10  the DE objective was a closure and could not be pickled for process workers
  B11  TFRecord framing errors were read as clean end-of-file
  B16  check_any_collision returned the first agent, not the earliest collision
  B20  one-sided bounds produced a meaningless weighted norm

B19 lives in the notebook and is covered by the audit's own fixture.

Pure numpy/torch/shapely, no database. Run:
    ./venv/bin/python -m pytest tests/test_batch5_contract.py -q
"""

import os
import pickle
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.danger.collision_detector import check_any_collision, check_collision_trajectory
from src.data.loader import ShardLoader, _crc32c, _masked_crc32c
from src.optimization.autograd_optimizer import refine_scenario
from src.optimization.perturbation_space import PerturbationSpace
from src.optimization.scipy_optimizer import (
    _DEObjective, keeps_challenger, optimize_scenario,
)


def _scene(n=2, t=10):
    states = np.zeros((n, t, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    return states, np.ones((n, t), dtype=bool), np.ones(n, dtype=int)


# ── B09: refinement keeps the best verified candidate ───────────────────────────

def _b09_space():
    states, validity, types = _scene(t=10)
    states[1, :, 1] = 2.1
    return PerturbationSpace(states, validity, types, 0, 1), validity


def test_refinement_never_returns_worse_than_its_warm_start():
    """
    The audit's scene, stated as the property rather than the symptom.

    The warm start exact-verifies as colliding at weighted norm 0.75. Refinement used
    to return only its final iterate, which does not collide, so a feasible answer
    handed in for free came back as collision=False / inf. Measured after the fix:
    it returns norm 0.229093 — better than the warm start, because an iterate along
    the way was better than both it and the endpoint.
    """
    space, validity = _b09_space()
    warm = np.array([0.0, 0.15, 0.0, 0.0], dtype=np.float32)
    assert check_collision_trajectory(space.apply(warm), validity, 0, 1)[0], (
        'fixture regressed: the warm start must itself collide'
    )
    warm_norm = space.weighted_norm(warm)

    result = refine_scenario(space, delta_init=warm, n_iters=100)

    assert result['collision'], 'a verified colliding warm start was discarded'
    assert result['min_perturbation'] <= warm_norm + 1e-9, (
        f"returned {result['min_perturbation']} against a warm start of {warm_norm}"
    )


def test_the_returned_fields_describe_the_returned_delta():
    """
    Whichever candidate wins, every field must describe IT.

    Reporting a collision_timestep or smooth_margin computed from the final iterate
    beside a delta from iteration 12 is the same juxtaposition defect Batch 2 removed
    from scenario_scores — a row whose fields silently describe two different runs.
    """
    space, validity = _b09_space()
    result = refine_scenario(space, delta_init=np.array([0.0, 0.15, 0.0, 0.0], np.float32),
                             n_iters=100)

    collided, t_hit = check_collision_trajectory(
        space.apply(result['delta']), validity, 0, 1)
    assert collided == result['collision']
    assert t_hit == result['collision_timestep'], (
        'collision_timestep belongs to a different delta than the one returned'
    )
    assert result['min_perturbation'] == pytest.approx(
        space.weighted_norm(result['delta'])), (
        'min_perturbation belongs to a different delta than the one returned'
    )


def test_the_warm_start_itself_is_a_candidate():
    """
    The warm start must be in the candidate set, not merely usually beaten by one.

    This gap was found by a mutation check, not by reading: on the audit's scene an
    ITERATE (norm 0.229) beats the warm start (0.750), so deleting the warm start from
    the candidate set changed nothing and every test still passed. The claim "never
    worse than what DE handed in" was therefore untested.

    This fixture makes the warm start the ONLY colliding candidate. Every dimension
    except heading is pinned to zero, and heading is bounded one-sided [0, 0.2] so Adam
    cannot sweep down through zero into the mirror-image collision band on the negative
    side. The collision threshold was located by bisection at theta* = 0.044908; the
    warm start sits 1e-4 above it, and one Adam step is ~lr = 0.02, so the very first
    iterate lands near 0.025 — below theta*, and every iterate after it is smaller
    still. Measured: 0 of 25 iterates collide.

    So if the warm start is not a candidate, this returns collision=False.
    """
    states, validity, types = _scene(t=10)
    states[1, :, 1] = 2.1
    bounds = np.array([[0, 0], [0, 0.2], [0, 0], [0, 0]], dtype=np.float32)
    space = PerturbationSpace(states, validity, types, 0, 1, bounds=bounds)

    warm = np.array([0.0, 0.045008, 0.0, 0.0], dtype=np.float32)
    assert check_collision_trajectory(space.apply(warm), validity, 0, 1)[0], (
        'fixture regressed: the warm start must sit just inside the collision band'
    )

    result = refine_scenario(space, delta_init=warm, n_iters=25)

    assert result['collision'], 'the warm start was dropped from the candidate set'
    assert result['delta'] == pytest.approx(warm, abs=1e-6), (
        'expected the warm start itself to be returned, as no iterate collides here'
    )
    assert result['min_perturbation'] == pytest.approx(space.weighted_norm(warm))


def test_a_non_colliding_warm_start_still_reports_honestly():
    """
    The fix must not manufacture a collision. With a scene whose agents are far apart
    and a zero warm start, no candidate can verify, and the honest answer is unchanged
    from before: collision=False, min_perturbation=inf.
    """
    states, validity, types = _scene(t=10)
    states[1, :, 1] = 60.0          # far away, nothing reachable inside the bounds
    space = PerturbationSpace(states, validity, types, 0, 1)
    result = refine_scenario(space, delta_init=np.zeros(4, np.float32), n_iters=20)
    assert result['collision'] is False
    assert result['min_perturbation'] == float('inf')


def test_the_selection_rule_matches_stress_one():
    """
    refine_scenario's internal rule and batch_scorer._stress_one's DE-vs-refined rule
    are the same comparison and must not drift: feasibility first, then smaller
    weighted norm. If refinement returned a colliding delta with a LARGER norm than
    the warm start, _stress_one would discard the whole refinement — so an internal
    rule that disagreed would silently waste the work.
    """
    space, _ = _b09_space()
    warm = np.array([0.0, 0.15, 0.0, 0.0], dtype=np.float32)
    result = refine_scenario(space, delta_init=warm, n_iters=100)

    # _stress_one keeps `refined` iff refined['collision'] and its norm is smaller.
    de_like = {'collision': True, 'min_perturbation': space.weighted_norm(warm)}
    kept = result['collision'] and (not de_like['collision']
                                    or result['min_perturbation'] < de_like['min_perturbation'])
    assert kept, (
        'refinement returned something _stress_one would throw away, which means the '
        'two disagree about which candidate is better'
    )


# ── B10: the DE objective must pickle ───────────────────────────────────────────

def _b10_space():
    states, validity, types = _scene(t=2)
    states[1, :, 0] = 20.0
    return PerturbationSpace(states, validity, types, 0, 1)


def test_the_objective_pickles():
    """
    scipy services workers != 1 with a process pool, which pickles the objective by
    qualified name. A nested def is `optimize_scenario.<locals>.objective`, which no
    fresh interpreter can look up — so every documented value of `workers` except the
    default died at serialization (audit B10).
    """
    objective = _DEObjective(_b10_space(), 1.0e3)
    revived = pickle.loads(pickle.dumps(objective))
    delta = np.array([0.5, 0.01, 0.1, 0.01], dtype=np.float32)
    assert revived(delta) == objective(delta), (
        'the objective survives pickling but stops computing the same value'
    )


def test_workers_one_is_bit_identical_after_the_refactor():
    """
    The real risk in B10 is not the parallel path — it is the refactor silently
    changing the DEFAULT path, which everything else in this project uses and which
    no test exercised for exact values.

    COMPARED AGAINST A RECONSTRUCTED PRISTINE CLOSURE, NOT AGAINST HARDCODED
    LITERALS. The first version of this test pinned the delta, margin and evaluation
    counts that differential_evolution converged to on one machine. A reviewer ran the
    identical fixture and seed on a second machine and got 2.8292112350463867 where
    this box produces 2.751762866973877 — DE's arithmetic is not bit-portable, so
    those literals asserted a property of the hardware rather than a property of the
    refactor. Exactly the failure Batch 4 hit with `x ** 2`, and the same fix: derive
    the reference on the machine running the test instead of freezing one machine's
    answer into the file.

    This is also a STRICTLY STRONGER test. The literals could only catch drift if the
    drift happened to move the converged point; running both objectives through DE
    under identical settings catches any arithmetic difference between them on this
    machine's own numbers, and it cannot go red for being run somewhere else.

    AMENDED BY BATCH 6, AND FOR THE SAME REASON THE LITERALS WERE REMOVED. The final
    assertion used to read

        assert np.array_equal(result['delta'], reference.x.astype(np.float32))

    which pinned optimize_scenario's OUTPUT to differential_evolution's raw winner.
    Audit R04 changed that relationship by definition: optimize_scenario now returns
    the better of DE's winner and the best exact-verified collision the search
    evaluated, so the two coincide only when DE happens to converge onto its own best
    feasible point. On this machine, at this fixture and seed, it does — which is
    exactly the trap. The assertion would have stayed green here and gone red on a
    machine whose DE arithmetic put the archive ahead, reporting a portability
    accident as a regression. Identical shape to the literals it replaced.

    So the assertion below tests the post-R04 contract instead: the returned delta is
    DE's raw winner UNLESS the archive beat it, in which case it must be a genuinely
    better verified collision. That is a real constraint in both branches, and it
    cannot be satisfied by an implementation that has stopped wiring the objective in
    — which is what this half of the test is for, and which nit/nfev also pin.
    """
    from scipy.optimize import differential_evolution
    from src.optimization.scipy_optimizer import _signed_gap

    states, validity, types = _scene(t=10)
    states[1, :, 1] = 2.6
    space = PerturbationSpace(states, validity, types, 0, 1)
    lam = 1.0e3

    # The closure exactly as it read before the refactor, rebuilt here so the
    # comparison is against the code that was replaced rather than against a memory
    # of what it produced.
    sdc, tgt = space.sdc_idx, space.target_idx
    space_validity = space.validity

    def original_objective(delta):
        pert = space.apply(delta)
        g = _signed_gap(pert, space_validity, sdc, tgt)
        norm = space.weighted_norm(delta)
        return norm * norm + lam * max(0.0, g)

    de_kwargs = dict(
        popsize=6, mutation=(0.5, 1.0), recombination=0.7,
        maxiter=25, tol=1e-3, seed=3, workers=1,
        polish=False, init='latinhypercube', disp=False,
    )
    bounds = [tuple(b) for b in space.bounds]

    reference = differential_evolution(original_objective, bounds, **de_kwargs)
    refactored = differential_evolution(_DEObjective(space, lam), bounds, **de_kwargs)

    assert np.array_equal(reference.x, refactored.x), (
        f'the refactor moved the converged point: {reference.x!r} -> {refactored.x!r}'
    )
    assert reference.nit == refactored.nit
    assert reference.nfev == refactored.nfev
    assert reference.fun == refactored.fun

    # ...and that optimize_scenario actually wires the new objective in, rather than
    # the two merely agreeing in isolation.
    result = optimize_scenario(space, lam=lam, popsize=6, maxiter=25, seed=3)
    assert result['n_iter'] == reference.nit
    assert result['n_eval'] == reference.nfev

    de_delta = reference.x.astype(np.float32)
    if result['candidate_source'] == 'de_winner':
        assert np.array_equal(result['delta'], de_delta), (
            f"claims to have returned DE's winner {de_delta!r} but returned "
            f"{result['delta']!r}"
        )
    else:
        # The archive won. It has to have won on the merits: a verified collision,
        # strictly smaller than DE's own answer under the shared predicate.
        de_hit, _ = check_collision_trajectory(space.apply(de_delta), space.validity,
                                               space.sdc_idx, space.target_idx)
        de_norm = space.weighted_norm(de_delta)
        hit, _ = check_collision_trajectory(space.apply(result['delta']),
                                            space.validity, space.sdc_idx,
                                            space.target_idx)
        assert keeps_challenger(hit, space.weighted_norm(result['delta']),
                                de_hit, de_norm), (
            f"the archive replaced DE's winner without being better: returned "
            f"{result['delta']!r} (collides={hit}) over {de_delta!r} "
            f"(collides={de_hit}, norm={de_norm})"
        )


def test_the_objective_matches_the_closure_it_replaced():
    """The arithmetic, compared term for term against the original expression."""
    space = _b10_space()
    lam = 1.0e3
    objective = _DEObjective(space, lam)
    from src.optimization.scipy_optimizer import _signed_gap

    rng = np.random.default_rng(0)
    for _ in range(25):
        delta = rng.uniform(space.bounds[:, 0], space.bounds[:, 1]).astype(np.float32)
        pert = space.apply(delta)
        g = _signed_gap(pert, space.validity, space.sdc_idx, space.target_idx)
        norm = space.weighted_norm(delta)
        assert objective(delta) == norm * norm + lam * max(0.0, g)


# ── B11: framing always, checksums on request ───────────────────────────────────

def _write_tfrecord(path, records):
    blob = b''
    for record in records:
        header = struct.pack('<Q', len(record))
        blob += (header + struct.pack('<I', _masked_crc32c(header))
                 + record + struct.pack('<I', _masked_crc32c(record)))
    path.write_bytes(blob)
    return blob


def test_crc32c_matches_the_published_vector():
    """
    CRC-32C (Castagnoli), not the CRC-32 (IEEE) zlib computes — a different
    polynomial, so zlib's answer is a different number rather than a close one. An
    unverified checksum implementation is worse than no checksum: it fails closed on
    good data.
    """
    assert _crc32c(b'123456789') == 0xE3069283
    assert _crc32c(b'') == 0x00000000


def test_a_valid_shard_still_reads_with_the_flag_either_way(tmp_path):
    """The regression risk in B11: breaking the files that are actually fine."""
    records = [b'first', b'second-record', b'', b'x' * 300]
    path = tmp_path / 'good.tfrecord'
    _write_tfrecord(path, records)
    for verify in (False, True):
        assert list(ShardLoader(str(path), verify_crc=verify)) == records


@pytest.mark.parametrize('contents,label', [
    (b'abc', 'partial header'),
    (struct.pack('<Q', 10) + b'\0' * 4 + b'ab', 'truncated payload'),
    (struct.pack('<Q', 3) + b'\0' * 4 + b'abc', 'missing trailing checksum'),
])
def test_framing_errors_raise_regardless_of_the_crc_flag(tmp_path, contents, label):
    """
    Framing is a STRUCTURAL defect and is checked unconditionally: a read that returns
    fewer bytes than the format demands means the file stops mid-record. This is
    separate from checksums, costs one length comparison, and is always an error.
    """
    path = tmp_path / 'corrupt.tfrecord'
    path.write_bytes(contents)
    for verify in (False, True):
        with pytest.raises(ValueError):
            list(ShardLoader(str(path), verify_crc=verify))


def test_a_bad_checksum_is_accepted_by_default_and_refused_on_request(tmp_path):
    """
    THE DELIBERATE TRADEOFF, asserted in both directions.

    Block 1 Concept 2 decided not to checksum every record: the shard is a local file
    from a trusted source and verification costs a full pass on every read. Audit B11
    flagged that; the decision stands, and the flag makes it an explicit choice
    instead of an unexamined default. The audit's own fixture expects the DEFAULT
    path to reject this, and that parametrization is marked xfail(strict=True) in
    tests/test_audit_core.py with this reasoning attached.
    """
    path = tmp_path / 'bad_payload.tfrecord'
    blob = bytearray(_write_tfrecord(path, [b'payload-bytes']))
    blob[12] ^= 0xFF                 # flip a bit inside the payload, framing intact
    path.write_bytes(bytes(blob))

    assert len(list(ShardLoader(str(path)))) == 1, (
        'the default path must still accept this — see Block 1 Concept 2'
    )
    with pytest.raises(ValueError, match='checksum mismatch'):
        list(ShardLoader(str(path), verify_crc=True))


def test_an_empty_file_is_a_clean_end_not_an_error(tmp_path):
    """Zero records is a valid shard. The framing check must not turn it into a fault."""
    path = tmp_path / 'empty.tfrecord'
    path.write_bytes(b'')
    assert list(ShardLoader(str(path))) == []


# ── B16: earliest collision, not the first agent ────────────────────────────────

def test_any_collision_returns_the_earliest_in_time():
    """
    Agent 1 collides at frame 4, agent 2 at frame 1. Index order reported (True, 1, 4)
    — a later event named as the first one.
    """
    states, validity, _ = _scene(n=3, t=5)
    states[1:, :, 0] = 100.0
    states[1, 4, 0] = 0.0
    states[2, 1, 0] = 0.0
    assert check_any_collision(states, validity, 0) == (True, 2, 1)


def test_a_tie_keeps_the_lower_agent_index():
    """
    Two agents striking in the same frame is ordinary. Leaving the winner to iteration
    order would make the answer depend on how the scene happened to be numbered — the
    same class of defect as an unstable sort key in pagination.
    """
    states, validity, _ = _scene(n=3, t=5)
    states[1:, :, 0] = 100.0
    states[1, 2, 0] = 0.0
    states[2, 2, 0] = 0.0
    assert check_any_collision(states, validity, 0) == (True, 1, 2)


def test_no_collision_is_unchanged():
    states, validity, _ = _scene(n=3, t=5)
    states[1:, :, 0] = 100.0
    assert check_any_collision(states, validity, 0) == (False, -1, -1)


# ── B20: the weighted norm must respect one-sided bounds ────────────────────────

def test_a_fully_spent_one_sided_budget_has_norm_one():
    """
    A braking-only bound [-3, 0] is legitimate. Weighting by abs(high) alone makes its
    denominator the 1e-6 floor, so spending exactly the 3 m/s the bound allows scored
    3e6 instead of 1 — the optimizers minimize this, so the dimension was priced as
    unusable rather than as a full budget.
    """
    states, validity, types = _scene(t=2)
    states[1, :, 0] = 20.0
    bounds = np.array([[-3, 0], [-.2, .2], [-1.5, 1.5], [-.1, .1]], dtype=np.float32)
    space = PerturbationSpace(states, validity, types, 0, 1, bounds=bounds)
    assert space.weighted_norm([-3, 0, 0, 0]) == pytest.approx(1.0)


def test_symmetric_bounds_are_untouched():
    """
    The fix must be inert for every existing caller. All default bounds are symmetric,
    so max(|low|, |high|) == |high| and the weights are identical term for term —
    which is what lets Batch 1's whole suite stay green without re-deriving anything.
    """
    for kinds in (np.ones(2, dtype=int), np.full(2, 2, dtype=int)):
        states, validity, _ = _scene(t=2)
        states[1, :, 0] = 20.0
        space = PerturbationSpace(states, validity, kinds, 0, 1)
        legacy = 1.0 / np.maximum(np.abs(space.bounds[:, 1]), 1e-6)
        np.testing.assert_array_equal(space.weights, legacy)


def test_a_zero_width_dimension_does_not_divide_by_zero():
    """
    A dimension pinned to one value has NO budget, so any nonzero delta there is
    outside the box. A huge norm says 'infinitely far outside its budget', which is
    true; a weight of zero would price it as free, which is not.
    """
    states, validity, types = _scene(t=2)
    states[1, :, 0] = 20.0
    bounds = np.array([[0, 0], [-.2, .2], [-1.5, 1.5], [-.1, .1]], dtype=np.float32)
    space = PerturbationSpace(states, validity, types, 0, 1, bounds=bounds)
    assert np.isfinite(space.weights).all()
    assert space.weighted_norm([0, 0, 0, 0]) == 0.0
    assert space.weighted_norm([1, 0, 0, 0]) > 1e5


def test_an_asymmetric_two_sided_bound_uses_the_larger_side():
    """[-4, 1]: the budget is 4, so a delta of -4 is exactly one unit of it."""
    states, validity, types = _scene(t=2)
    states[1, :, 0] = 20.0
    bounds = np.array([[-4, 1], [-.2, .2], [-1.5, 1.5], [-.1, .1]], dtype=np.float32)
    space = PerturbationSpace(states, validity, types, 0, 1, bounds=bounds)
    assert space.weighted_norm([-4, 0, 0, 0]) == pytest.approx(1.0)
    assert space.weighted_norm([1, 0, 0, 0]) == pytest.approx(0.25)
