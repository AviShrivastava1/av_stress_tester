"""
test_audit3_regressions.py — third audit, finding A01.

PROVENANCE, STATED PLAINLY: reconstructed from the audit's description, NOT copied
from its own code. No third-audit bundle exists — searched exhaustively on 2026-09-15
across ~/Downloads, ~/Desktop, ~/Documents and /tmp: the only audit archive on disk is
av_stress_tester_audit_bundle.zip (2026-09-10), which is the FIRST, B-numbered audit,
and nothing anywhere contains a `test_A0*` name. av_stress_tester-main.zip is a source
snapshot of commit 2b71cb7 going OUT to the auditor, not a bundle coming back.

Same standard as the second audit's reconstructions: each test was run against the
UNFIXED code first and shown to reproduce the audit's SPECIFIC claim.

    A01  stationary cyclist at (0, 1.8), 2.0 x 0.6, heading 0, beside a 4.5 x 2.0 SDC
         at the origin; 10 valid frames; linear model (type 3)
           -> delta [0, 1e-9, 0, 0] rotated the footprint to arctan2(1e-9, 0) = pi/2
              EXACTLY, colliding at frame 0 with the centre unmoved, at weighted norm
              5.000000e-10. Identical at 1e-12, 1e-30 and 1e-45 — atan2(+x, 0) is
              pi/2 for every positive x, so more precision makes this worse, not
              better.
           -> at dvy0 <= 1e-30 the weighted norm was EXACTLY 0.0, because the float32
              square underflows while atan2 does not: a verified exact-SAT collision
              reported as caused by a perturbation of magnitude zero.
           -> a real DE run (popsize=4, maxiter=20, seed=1) found it unprompted at
              min_perturbation = 0.003215756034478545, matching the audit's
              0.003215756. Reachable by the ordinary pipeline:
              pick_nearest_challenger has no speed filter and selects this agent.

    A01b THE DEFECT THE AUDIT DID NOT REPORT, found while reproducing A01 and worse
         because it needs no perturbation at all: for a stationary agent vx = vy = 0
         in the LOGGED data too, so arctan2(0, 0) = 0 fired on the ZERO-delta replay.
         A parked agent logged facing 0.7 rad replayed facing 0.0000 — a 40-degree
         footprint rotation applied by a delta of exactly zero — while
         baseline_replay_error reported 0.0000 m, because _measure_baseline_replay
         compares states[..., :2] and heading is column 4.

THE DE FIGURE ABOVE IS NOT ASSERTED ANYWHERE. It matched the audit to every printed
digit on this machine, and it is still a constant obtained by running the code, which
Batch 5's review established twice is a portability hazard. It is recorded here as
evidence and derived at run time where a test needs it.

Run:
    ./venv/bin/python -m pytest tests/test_audit3_regressions.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.danger.collision_detector import check_collision_trajectory
from src.optimization.perturbation_space import PerturbationSpace


def _stationary_cyclist(heading=0.0, lateral=1.8):
    """The audit's fixture: a parked cyclist beside a stationary SDC."""
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]          # SDC at the origin
    states[1, :, 1] = lateral
    states[1, :, 5:7] = [2.0, 0.6]
    states[1, :, 4] = heading
    #                        3 = cyclist -> linear model, heading derived not carried
    return states, np.ones((2, 10), dtype=bool), np.array([1, 3])


@pytest.mark.parametrize('dvy0', [1e-9, 1e-12, 1e-30, 1e-45])
def test_A01_a_vanishing_delta_cannot_rotate_a_stationary_footprint(dvy0):
    """
    The audit's claim, parametrized over the magnitudes that prove it is not a
    precision artifact: atan2(+x, 0) is pi/2 for every positive x, so shrinking the
    delta makes the exploit cheaper without making the rotation smaller.

    Asserted as the INVARIANT rather than the mechanism — a delta this small must not
    produce a verified collision, however the implementation chooses to achieve that.
    """
    states, validity, types = _stationary_cyclist()
    space = PerturbationSpace(states, validity, types, 0, 1)
    assert not space.is_vehicle, 'fixture regressed: this must be a linear-model agent'

    delta = np.array([0.0, dvy0, 0.0, 0.0])
    perturbed = space.apply(delta)
    collided, frame = check_collision_trajectory(perturbed, validity, 0, 1)

    assert not collided, (
        f'dvy0={dvy0:g} produced a verified collision at frame {frame} for weighted '
        f'norm {space.weighted_norm(delta)!r}; heading went '
        f'{float(states[1, 0, 4]):.6f} -> {float(perturbed[1, 0, 4]):.6f} with the '
        f'centre unmoved'
    )


def test_A01_no_collision_is_ever_reported_at_zero_weighted_norm():
    """
    The sharpest form of the finding. At dvy0 <= 1e-30 the float32 square underflows
    and weighted_norm returns EXACTLY 0.0 while atan2 still returns pi/2 — so the
    project's headline number, the smallest perturbation that causes a collision,
    evaluated to zero for a real exact-SAT collision.

    Stated as a property of the whole search space, not of one delta: no delta whose
    weighted norm is zero may collide, because such a delta has by definition changed
    nothing this project can charge for.
    """
    states, validity, types = _stationary_cyclist()
    space = PerturbationSpace(states, validity, types, 0, 1)

    zero_norm_deltas = [
        np.zeros(4),
        np.array([0.0, 1e-30, 0.0, 0.0]),
        np.array([0.0, -1e-30, 0.0, 0.0]),
        np.array([1e-30, 1e-30, 1e-30, 1e-30]),
    ]
    # THE LOOP BELOW SKIPS ANY DELTA WHOSE NORM IS NOT ZERO, SO SOMETHING OUTSIDE IT
    # HAS TO CONFIRM THE INTERESTING CASE EXISTS. Without this the test would pass
    # having asserted nothing if float32 underflow behaved differently anywhere. The
    # standing lesson from Batch 4's `finite > 100` and Batch 6's `checked > 10000`:
    # a conditional `continue` is only a real test when its condition is shown to fire.
    assert any(space.weighted_norm(d) == 0.0 for d in zero_norm_deltas), (
        'no candidate delta produced a weighted norm of exactly 0.0 on this machine, '
        'so the property this test exists to check was never exercised'
    )
    for delta in zero_norm_deltas:
        norm = space.weighted_norm(delta)
        if norm != 0.0:
            continue                      # not a zero-norm delta on this machine
        collided, frame = check_collision_trajectory(
            space.apply(delta), validity, 0, 1)
        assert not collided, (
            f'delta {delta.tolist()} has weighted norm exactly {norm!r} and still '
            f'collides at frame {frame}: a collision caused by nothing'
        )


@pytest.mark.parametrize('seed', range(8))
def test_A01_a_real_search_must_pay_for_the_rotation_it_buys(seed):
    """
    Reachability, which is what separates this from a constructed curiosity: DE found
    the exploit on its own, at norm 0.003215756034478545 pre-fix.

    AN EARLIER VERSION OF THIS TEST ASSERTED THE WRONG THING, and it is recorded
    rather than quietly replaced because the mistake is the instructive part. It read
    `moved > 1e-6` — "post-fix, a collision must involve displacement" — and passed.
    It passed on seed 1 by luck. Sweeping seeds 0-11 showed seeds 6, 7 and 8 collide
    at frame 0 with EXACTLY zero displacement, and correctly so: apply() perturbs the
    initial VELOCITY, not the initial position, so at frame 0 the agent sits exactly
    where it was logged while carrying enough speed for its heading to be derived.
    The fix never promised rotation would be impossible. It promised rotation would be
    PAID FOR, and a test asserting the stronger thing was asserting something false.

    So this asserts the guarantee the design actually makes, with the bound DERIVED
    from the space's own weights rather than hardcoded — DE constants are not
    bit-portable (Batch 5) and a ratio against the pre-fix answer swings 6x to 71x
    across seeds, which would be just as fragile.

    THE BOUND. Rotation requires speed >= floor at the colliding frame. Speed can be
    bought with initial velocity (weight w_v) or with an acceleration bias acting over
    the horizon T (weight w_a), and combining them is cheaper than either alone
    because the norm adds in quadrature while speed adds linearly. Minimising
    ||W d|| subject to  dv + a*T = floor  is the distance from the origin to a line:

        min_norm = floor / sqrt((1/w_v)^2 + (T/w_a)^2)

    which is 0.5 / sqrt(4 + 0.81) = 0.227980 here. Measured: DE's cheapest answer over
    12 seeds was 0.228815, within 0.4% of that bound — it finds the optimum almost
    exactly. Pre-fix the most expensive answer over the same seeds was 0.042078, which
    is 0.185x the bound, so the separation is not marginal.
    """
    from src.optimization.scipy_optimizer import optimize_scenario

    states, validity, types = _stationary_cyclist()
    space = PerturbationSpace(states, validity, types, 0, 1)
    result = optimize_scenario(space, popsize=4, maxiter=20, seed=seed)

    assert result['collision'], (
        f'seed {seed}: the search found nothing at all, so this run is evidence about '
        'the budget rather than about the fix. Measured 12/12 seeds colliding at this '
        'budget post-fix; if that has changed, raise the budget rather than passing.'
    )

    w_v, w_a = float(space.weights[1]), float(space.weights[3])
    horizon = (states.shape[1] - 1) * float(space.dt)
    from src.physics.linear_model import V_HEADING_MIN
    min_norm = V_HEADING_MIN / np.sqrt((1.0 / w_v) ** 2 + (horizon / w_a) ** 2)

    assert result['min_perturbation'] >= 0.98 * min_norm, (
        f"seed {seed}: DE collided for weighted norm {result['min_perturbation']}, "
        f'below the {min_norm:.6f} that reaching the heading floor costs — so it '
        f'bought a rotation it did not pay for'
    )

    # ...and that the same search WITH THE FLOOR OFF still reproduces the exploit,
    # on this machine, so the comparison is against measured pre-fix behaviour rather
    # than against a remembered number.
    unguarded = PerturbationSpace(states, validity, types, 0, 1,
                                  heading_speed_floor=None)
    before = optimize_scenario(unguarded, popsize=4, maxiter=20, seed=seed)
    assert before['collision'] and before['min_perturbation'] < min_norm, (
        f'seed {seed}: with the floor off the pre-A01 exploit must still be findable '
        f'and must cost less than {min_norm:.6f}; got {before}'
    )


@pytest.mark.parametrize('logged_heading', [0.0, 0.7, np.pi / 2, 2.5, -1.2])
def test_A01b_the_zero_delta_replay_preserves_the_logged_heading(logged_heading):
    """
    The unreported half: the identity perturbation must reproduce the logged track,
    and heading is part of the logged track.

    Pre-fix every non-zero heading here came back as 0.0000 with
    baseline_replay_error reporting 0.0000 m, because the gate compares positions
    only. That is a Batch 1 / audit B03 violation in a column B03 never inspected.

    Laterally separated so the rotated footprint does not overlap the SDC — otherwise
    the fixture is refused by the replay-fidelity gate before it can be measured,
    which is itself correct behaviour and is pinned separately in
    tests/test_batch9_contract.py.
    """
    states, validity, types = _stationary_cyclist(heading=logged_heading, lateral=6.0)
    space = PerturbationSpace(states, validity, types, 0, 1)

    replayed = space.apply(np.zeros(4, dtype=np.float32))
    for frame in range(10):
        assert replayed[1, frame, 4] == pytest.approx(logged_heading, abs=1e-6), (
            f'frame {frame}: logged heading {logged_heading} replayed as '
            f'{float(replayed[1, frame, 4])} under a ZERO perturbation'
        )
    assert space.baseline_replay_error == pytest.approx(0.0, abs=1e-9)
