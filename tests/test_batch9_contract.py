"""
test_batch9_contract.py — Batch 9 contract tests (audit A01).

A01 is a different class of finding from everything in Batches 1-7. Those clamped,
gated, or archived a value that was well-defined and wrong. This one asks whether
ORIENTATION IS EVEN DEFINED for the state this project uses: the linear model's state
vector is [x, y, vx, vy], it has no heading, and every consumer that needs one
recovers it as arctan2(vy, vx) — which is singular exactly where a parked car or a
standing pedestrian lives.

The fix derives heading where the velocity direction is observable and holds the
LOGGED heading where it is not, mirroring invert_bicycle's `if abs(v) < 1e-3` guard.

The audit's own repro lives in tests/test_audit3_regressions.py, reconstructed and
labelled. THIS file holds the properties — including the ones about what the fix
deliberately does NOT do.

Pure numpy/shapely, no database. Run:
    ./venv/bin/python -m pytest tests/test_batch9_contract.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.danger.collision_detector import check_collision_trajectory
from src.optimization.perturbation_space import (
    PerturbationSpace, ReplayFidelityError,
)
from src.physics.linear_model import V_HEADING_MIN


def _scene(agent_type=3, heading=0.0, lateral=1.8, speed=0.0, direction=(1.0, 0.0)):
    """SDC at the origin; one challenger, stationary unless `speed` says otherwise."""
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 1] = lateral
    states[1, :, 5:7] = [2.0, 0.6]
    states[1, :, 4] = heading
    if speed:
        ux, uy = direction
        states[1, :, 2] = speed * ux
        states[1, :, 3] = speed * uy
        states[1, :, 0] = speed * ux * 0.1 * np.arange(10)
        states[1, :, 1] = lateral + speed * uy * 0.1 * np.arange(10)
        states[1, :, 4] = np.arctan2(speed * uy, speed * ux)
    return states, np.ones((2, 10), dtype=bool), np.array([1, agent_type])


# ── the floor only touches what it is supposed to touch ─────────────────────────

def test_a_vehicle_challenger_is_untouched():
    """
    The bicycle model carries theta as real STATE, so its heading is written from the
    rollout and was never derived. A01 cannot apply to it, and the fix must not
    reach it — if it did, the floor would be silently overriding a real state
    variable with a logged one.
    """
    states, validity, types = _scene(agent_type=1, heading=0.7, lateral=6.0)
    space = PerturbationSpace(states, validity, types, 0, 1)
    assert space.is_vehicle

    for delta in (np.zeros(4), np.array([0.0, 0.15, 0.0, 0.0])):
        replayed = space.apply(delta)
        expected = 0.7 + float(delta[1])          # dtheta0 is a real state offset
        assert float(replayed[1, 0, 4]) == pytest.approx(expected, abs=1e-5)


def test_a_moving_linear_agent_still_derives_its_heading():
    """
    Above the floor nothing changes — the derived heading is exactly arctan2 as
    before. A fix that held the logged heading for everyone would freeze orientation
    for every pedestrian and cyclist in the corpus.
    """
    states, validity, types = _scene(speed=3.0, direction=(0.0, 1.0), lateral=6.0)
    space = PerturbationSpace(states, validity, types, 0, 1)

    delta = np.array([2.0, 0.0, 0.0, 0.0])        # push vx, so the direction turns
    replayed = space.apply(delta)
    vx, vy = replayed[1, 5, 2], replayed[1, 5, 3]
    assert float(np.hypot(vx, vy)) > V_HEADING_MIN, 'fixture regressed: below the floor'
    assert float(replayed[1, 5, 4]) == pytest.approx(
        float(np.arctan2(vy, vx)), abs=1e-6), (
        'an observable heading must still be derived from the velocity direction'
    )


def test_the_held_heading_is_per_frame_not_the_t0_heading():
    """
    An agent parked but slowly rotating in place — a cyclist turning its bars, a
    pedestrian pivoting — has a DIFFERENT logged heading at every frame, and the
    zero-delta replay has to reproduce each of them. Holding the t0 heading for the
    whole window would be a subtler version of the same defect.
    """
    states, validity, types = _scene(lateral=6.0)
    states[1, :, 4] = 0.1 * np.arange(10)         # stationary, rotating in place
    space = PerturbationSpace(states, validity, types, 0, 1)

    replayed = space.apply(np.zeros(4))
    for frame in range(10):
        assert float(replayed[1, frame, 4]) == pytest.approx(0.1 * frame, abs=1e-6), (
            f'frame {frame} came back with the wrong held heading'
        )


# ── what the floor costs, and what it guarantees ────────────────────────────────

def test_rotating_a_resting_agent_at_frame_zero_costs_at_least_the_floor():
    """
    THE GUARANTEE, stated as arithmetic rather than as a hope — and scoped to FRAME
    ZERO, which is narrower than an earlier version of this docstring claimed.

    At frame 0 the only thing that can lift the agent to the floor speed is the
    initial velocity offset, because no acceleration has had time to act. Reaching
    speed f from rest needs |dv| = f in any direction, and the weighted norm of that
    is f * weight_vy regardless of direction, so the frame-0 cost is exactly
    V_HEADING_MIN * weight_vy = 0.25.

    IT IS NOT THE BOUND FOR LATER FRAMES, and saying so would be the same overclaim
    this batch already corrected once. Over a horizon T an acceleration bias can
    contribute to the speed too, and combining the two is cheaper than either alone
    because the norm adds in quadrature while speed adds linearly — the true
    multi-frame minimum is floor / sqrt((1/w_v)^2 + (T/w_a)^2) = 0.227980, and DE
    finds 0.228815. That bound is asserted in
    tests/test_audit3_regressions.py::test_A01_a_real_search_must_pay_for_the_rotation_it_buys.

    The count guard below is the standing rule this project has now rediscovered four
    times (Batches 4, 5, 6, and here): a loop whose property is checked under `if`
    proves nothing unless something outside it confirms the `if` fired.
    """
    states, validity, types = _scene(lateral=6.0)
    space = PerturbationSpace(states, validity, types, 0, 1)
    logged = float(states[1, 0, 4])

    weight_vy = float(space.weights[1])
    predicted_floor_cost = V_HEADING_MIN * weight_vy

    rng = np.random.default_rng(0)
    rotated_count = 0
    for _ in range(400):
        delta = rng.uniform(space.bounds[:, 0], space.bounds[:, 1])
        replayed = space.apply(delta)
        rotated = abs(float(replayed[1, 0, 4]) - logged) > 1e-9
        if rotated:
            rotated_count += 1
            assert space.weighted_norm(delta) >= predicted_floor_cost - 1e-6, (
                f'delta {delta.tolist()} rotated the footprint for weighted norm '
                f'{space.weighted_norm(delta)}, under the predicted floor cost '
                f'{predicted_floor_cost}'
            )

    # Measured: 379 of 400 draws rotate, because the default +/-2.0 velocity bounds
    # are four times the floor, so most uniform draws clear it. Asserted at a margin
    # well below that rather than at the measured figure, which would pin one
    # machine's RNG stream.
    assert rotated_count > 50, (
        f'only {rotated_count} of 400 draws rotated the footprint, so the assertion '
        'inside the loop was barely exercised and this test is close to vacuous'
    )


def test_the_floor_is_reachable_not_prohibitive():
    """
    The complement, and the thing that stops the fix from being a way to make every
    linear scenario unsearchable: a delta that genuinely accelerates the agent to the
    floor DOES rotate it. The orientation is expensive now, not forbidden.
    """
    states, validity, types = _scene(lateral=6.0)
    space = PerturbationSpace(states, validity, types, 0, 1)

    delta = np.array([0.0, V_HEADING_MIN, 0.0, 0.0])
    replayed = space.apply(delta)
    assert float(replayed[1, 0, 4]) == pytest.approx(np.pi / 2, abs=1e-5), (
        'a delta that reaches the floor must derive the heading normally'
    )
    assert space.weighted_norm(delta) == pytest.approx(
        V_HEADING_MIN * float(space.weights[1]), rel=1e-6)


def test_what_the_fix_does_not_fix_is_true_as_written():
    """
    The plan claims the 1/|v| sensitivity is BOUNDED, not eliminated, and quantifies
    it. A claim that specific should be checked, not just written down — if the real
    sensitivity at the floor were far from the stated figure, the honest description
    of this batch would be wrong.

    At |v| = floor, a perpendicular velocity change dv turns the heading by about
    dv/|v|, and costs dv * weight_vy of norm, so the exchange rate is
    1 / (floor * weight_vy) rad per unit norm. The assertion is that measurement
    matches that arithmetic.
    """
    states, validity, types = _scene(speed=V_HEADING_MIN, direction=(1.0, 0.0),
                                     lateral=6.0)
    space = PerturbationSpace(states, validity, types, 0, 1)
    base = float(space.apply(np.zeros(4))[1, 0, 4])

    dv = 0.01
    delta = np.array([0.0, dv, 0.0, 0.0])
    turned = abs(float(space.apply(delta)[1, 0, 4]) - base)
    cost = space.weighted_norm(delta)

    measured_rate = turned / cost
    predicted_rate = 1.0 / (V_HEADING_MIN * float(space.weights[1]))
    assert measured_rate == pytest.approx(predicted_rate, rel=0.05), (
        f'measured {measured_rate:.3f} rad per unit norm against a predicted '
        f'{predicted_rate:.3f} — the batch writeup describes a different system'
    )
    # ...and that this really is ~10x cheaper than a vehicle's priced dtheta0.
    v_states, v_validity, v_types = _scene(agent_type=1, heading=0.0, lateral=6.0)
    v_space = PerturbationSpace(v_states, v_validity, v_types, 0, 1)
    vehicle_rate = 1.0 / float(v_space.weights[1])
    assert measured_rate > 5 * vehicle_rate, (
        'the stated asymmetry with the vehicle model is not real'
    )


# ── the escape hatch, and the measurement that justifies the constant ───────────

def test_the_floor_can_be_switched_off_and_restores_the_old_behaviour():
    """
    Same escape hatch max_baseline_drift has, and for the same stated purpose: so the
    real distribution can be measured before 0.5 m/s is defended as final. Switching
    it off must restore the PRE-FIX behaviour exactly, not something approximating it
    — otherwise a measurement taken with the floor off is measuring a third thing.
    """
    states, validity, types = _scene()
    space = PerturbationSpace(states, validity, types, 0, 1,
                              heading_speed_floor=None)

    delta = np.array([0.0, 1e-9, 0.0, 0.0])
    replayed = space.apply(delta)
    assert float(replayed[1, 0, 4]) == pytest.approx(np.pi / 2, abs=1e-6)
    assert check_collision_trajectory(replayed, validity, 0, 1)[0], (
        'with the floor off, the pre-A01 exploit must be reproducible — that is what '
        'makes the switch a measurement tool rather than a second code path'
    )


@pytest.mark.parametrize('floor', [V_HEADING_MIN, None, 0.1])
def test_the_measurement_is_recorded_whatever_the_floor_is(floor):
    """
    has_interior_gap's precedent: recorded whether or not it changes behaviour,
    because a number that only exists where it already mattered cannot say how often
    it matters. Switching the floor off to measure must not switch off the
    measurement.
    """
    states, validity, types = _scene(lateral=6.0)
    space = PerturbationSpace(states, validity, types, 0, 1,
                              heading_speed_floor=floor)

    assert space.target_min_speed == pytest.approx(0.0)
    assert space.frames_observed == 10
    assert space.frames_below_heading_floor == 10, (
        'a fully stationary challenger must be counted as below the floor even when '
        'the floor is switched off'
    )


def test_the_measurement_distinguishes_moving_from_parked():
    """A counter that says the same thing about every scenario measures nothing."""
    parked = PerturbationSpace(*_scene(lateral=6.0), 0, 1)
    moving = PerturbationSpace(*_scene(speed=3.0, direction=(1.0, 0.0), lateral=6.0),
                               0, 1)

    assert parked.frames_below_heading_floor == 10
    assert moving.frames_below_heading_floor == 0
    assert moving.target_min_speed == pytest.approx(3.0, rel=1e-5)


def test_a_vehicle_scenario_is_measured_too():
    """
    Recorded for vehicles as well, even though the floor never applies to them. The
    question the Colab pass has to answer is "how many challengers are near
    stationary", and filtering that to linear agents only would answer a different
    one.
    """
    space = PerturbationSpace(*_scene(agent_type=1, lateral=6.0), 0, 1)
    assert space.is_vehicle
    assert space.frames_observed == 10
    assert space.frames_below_heading_floor == 10


def test_the_measurement_reaches_search_provenance():
    """
    The one reason this batch touches src/scoring/ at all: the validation notebook
    reads search_provenance, not PerturbationSpace instances, so a measurement that
    stops at the object cannot be queried over a shard.
    """
    import json
    from src.scoring.batch_scorer import _stress_one

    states, validity, types = _scene(lateral=6.0)
    result = _stress_one(states, validity, types, 0,
                         de_kwargs={'popsize': 4, 'maxiter': 5, 'seed': 1})
    provenance = result['search_provenance']

    assert provenance['target_min_speed'] == pytest.approx(0.0)
    assert provenance['frames_below_heading_floor'] == 10
    assert provenance['frames_observed'] == 10
    assert provenance['heading_speed_floor'] == V_HEADING_MIN
    # JSON has no infinity literal (audit B04) and this dict is written to JSONB.
    json.dumps(provenance)


# ── the knock-on the fix creates, pinned deliberately ───────────────────────────

def test_preserving_the_heading_lets_the_gate_see_a_real_baseline_overlap():
    """
    A CONSEQUENCE, NOT AN ACCIDENT, and worth pinning because it changes which
    scenarios get refused.

    A parked agent logged at 0.7 rad beside the SDC genuinely overlaps it. Pre-fix the
    zero-delta replay straightened it to 0.0, the overlap vanished, and the scenario
    was accepted for search — so the replay-fidelity gate was being protected from a
    real baseline collision by the very defect A01 describes. With the logged heading
    preserved the gate sees the true footprint and refuses, which is what B03 exists
    to do.
    """
    states, validity, types = _scene(heading=0.7, lateral=1.8)
    assert check_collision_trajectory(states, validity, 0, 1)[0], (
        'fixture regressed: the logged configuration must genuinely overlap'
    )

    with pytest.raises(ReplayFidelityError) as excinfo:
        PerturbationSpace(states, validity, types, 0, 1)
    assert excinfo.value.reason == 'collision'

    # ...and with the floor off, the old straightening hides it again.
    space = PerturbationSpace(states, validity, types, 0, 1,
                              heading_speed_floor=None)
    assert not check_collision_trajectory(
        space.apply(np.zeros(4)), validity, 0, 1)[0], (
        'with the floor off the baseline overlap must be hidden again, which is the '
        'defect this test exists to document'
    )


def test_the_perturbation_space_shape_is_unchanged():
    """
    The cost-elsewhere guarantee, asserted rather than asserted-in-prose: A01 changes
    no dimension, weight or bound, so every `len(delta) == 4` in the API layer and the
    delta column's arity are safe by construction.
    """
    space = PerturbationSpace(*_scene(lateral=6.0), 0, 1)
    assert space.dim == 4
    assert space.bounds.shape == (4, 2)
    assert np.allclose(space.weights, [0.5, 0.5, 1.0, 1.0])
    assert len(space.apply(np.zeros(4))[1, 0]) == 7
