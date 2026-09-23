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
    linear scenario unsearchable: a delta that genuinely accelerates the agent past
    the transition band DOES rotate it. The orientation is expensive now, not
    forbidden.

    UPDATED for the transition-width fix (independent review, post-A01): the band is
    ANCHORED at V_HEADING_MIN and extends upward, deliberately, so that "nothing
    below the floor ever derives" stays true — see perturbation_space.py's own
    docstring for why anchored beats centred against exactly that guarantee. One
    consequence: reaching V_HEADING_MIN itself no longer derives "normally" the way
    it did pre-fix — it is the START of the ramp (w=0 there, by construction), not
    the finish. Full derivation now needs V_HEADING_MIN + HEADING_TRANSITION_WIDTH,
    which is what this test targets instead.
    """
    from src.optimization.perturbation_space import HEADING_TRANSITION_WIDTH

    states, validity, types = _scene(lateral=6.0)
    space = PerturbationSpace(states, validity, types, 0, 1)

    target_speed = V_HEADING_MIN + HEADING_TRANSITION_WIDTH
    delta = np.array([0.0, target_speed, 0.0, 0.0])
    replayed = space.apply(delta)
    assert float(replayed[1, 0, 4]) == pytest.approx(np.pi / 2, abs=1e-5), (
        'a delta that clears the whole transition band must derive the heading '
        'normally'
    )
    assert space.weighted_norm(delta) == pytest.approx(
        target_speed * float(space.weights[1]), rel=1e-6)


def test_what_the_fix_does_not_fix_is_true_as_written():
    """
    The plan claims the 1/|v| sensitivity is BOUNDED, not eliminated, and quantifies
    it. A claim that specific should be checked, not just written down — if the real
    sensitivity at the floor were far from the stated figure, the honest description
    of this batch would be wrong.

    MEASURED ABOVE THE BAND, NOT AT THE FLOOR — updated for the transition-width fix.
    _smoothstep is C1 (zero slope at both ends by construction, a free property of
    the cubic Hermite form), so the instantaneous rotation rate AT V_HEADING_MIN
    itself is now ~0, not the 1/|v| rate this test exists to check — that classic
    arctan2-derivative regime only resumes once the band is fully cleared, at
    V_HEADING_MIN + HEADING_TRANSITION_WIDTH and beyond, which is where a real DE
    search spends most of its non-transitional time above the floor. Measuring
    there is truer to what the original A01 finding was about (the derived-heading
    regime's own sensitivity) than measuring inside a ramp this fix deliberately
    added.

    At |v| = floor + width, a perpendicular velocity change dv turns the heading by
    about dv/|v|, and costs dv * weight_vy of norm, so the exchange rate is
    1 / ((floor + width) * weight_vy) rad per unit norm. The assertion is that
    measurement matches that arithmetic.
    """
    from src.optimization.perturbation_space import HEADING_TRANSITION_WIDTH

    above_band = V_HEADING_MIN + HEADING_TRANSITION_WIDTH
    states, validity, types = _scene(speed=above_band, direction=(1.0, 0.0),
                                     lateral=6.0)
    space = PerturbationSpace(states, validity, types, 0, 1)
    base = float(space.apply(np.zeros(4))[1, 0, 4])

    dv = 0.01
    delta = np.array([0.0, dv, 0.0, 0.0])
    turned = abs(float(space.apply(delta)[1, 0, 4]) - base)
    cost = space.weighted_norm(delta)

    measured_rate = turned / cost
    predicted_rate = 1.0 / (above_band * float(space.weights[1]))
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


# ── the boundary cliff itself, and its fix (independent review, post-A01) ───────
#
# A01 fixed the singularity at v=0. It left a second one, found later: the SWITCH
# ITSELF, at v=heading_speed_floor, was a single-point discontinuity — continuous
# everywhere except exactly there, because np.where(speed >= floor, derived, logged)
# is exactly the old arctan2-at-the-origin problem relocated, not removed, and
# moving the floor cannot fix it since the argument does not depend on where the
# floor sits. Measured directly (not assumed), both directions, on a fixture where
# logged and derived genuinely disagree (routine for a real, noisy slow-moving
# agent): a weighted-norm delta of ~5e-7 flipped a verified collision by crossing
# it, upward and downward alike. See perturbation_space.py::_linear_heading's own
# docstring for the fix (a transition band, anchored at heading_speed_floor and
# extending upward, so "nothing below the floor ever derives" stays exactly true).

def _disagreeing_scene(vx, vy, logged_heading, lateral=1.8):
    """
    A challenger whose recorded heading and velocity-implied direction disagree —
    the configuration the boundary cliff needs in order to be reachable at all. At
    lateral=1.8 (A01's own fixture geometry), heading=0 does not overlap the SDC and
    heading=pi/2 does — the same two values the ordinary A01 exploit used, here
    used to make crossing the boundary matter rather than to make v=0 matter.

    Position is propagated to match velocity (same pattern _scene's `speed` branch
    above uses), not left static — a static position under a nonzero logged
    velocity is an internally inconsistent fixture that the replay-fidelity gate
    (audit B03) will refuse once the implied drift clears max_baseline_drift, which
    a large enough vx/vy here would. Frame 0's position is unaffected either way
    (t=0 makes the propagation term zero), so this changes nothing about what the
    tests below check.
    """
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 5:7] = [2.0, 0.6]
    states[1, :, 4] = logged_heading
    states[1, :, 2] = vx
    states[1, :, 3] = vy
    states[1, :, 0] = vx * 0.1 * np.arange(10)
    states[1, :, 1] = lateral + vy * 0.1 * np.arange(10)
    return states, np.ones((2, 10), dtype=bool), np.array([1, 3])


def test_crossing_the_floor_upward_no_longer_flips_for_free():
    """
    UPWARD. Logged heading 0.0 (safe at this geometry), velocity direction implying
    pi/2 (collides at this geometry), logged speed held just BELOW the floor — so at
    delta=0 the safe logged heading is what replays, and the baseline is clean.
    Pre-fix, dvy0=0.05 (weighted norm 0.025) flipped this to a verified collision;
    the exact boundary (dvy0=0.049999 -> 0.05) flipped for a weighted-norm
    difference of ~5e-7. Post-fix, that same tiny delta must change nothing
    measurable.
    """
    states, validity, types = _disagreeing_scene(vx=0.0, vy=0.45, logged_heading=0.0)
    space = PerturbationSpace(states, validity, types, 0, 1)
    assert not space.baseline_replay_collides, 'fixture regressed: baseline must be safe'

    tiny = np.array([0.0, 1e-6, 0.0, 0.0])
    replayed = space.apply(tiny)
    collided, frame = check_collision_trajectory(replayed, validity, 0, 1)
    assert not collided, (
        f'a weighted-norm-{space.weighted_norm(tiny):.2e} delta produced a verified '
        f'collision at frame {frame} by crossing the floor upward — the cliff is back'
    )

    # ...and the exact old flip point (dvy0 0.049999 -> 0.05) is smooth now, not a step.
    just_below = space.apply(np.array([0.0, 0.049999, 0.0, 0.0]))
    just_at = space.apply(np.array([0.0, 0.05, 0.0, 0.0]))
    assert float(just_below[1, 0, 4]) == pytest.approx(float(just_at[1, 0, 4]), abs=1e-6), (
        'heading still jumps across the old flip point'
    )


def test_crossing_the_floor_downward_no_longer_flips_for_free():
    """
    DOWNWARD, the mirror image. Logged heading pi/2 (collides here), velocity
    direction implying 0 (safe), logged speed held ABOVE THE WHOLE BAND (not just
    above the floor — post-fix, anything inside [floor, floor+width) is itself a
    blend, so the baseline needs to clear floor+width to be purely-derived/safe) —
    so at delta=0 the safe derived heading is what replays. A delta that
    decelerates the agent back below the floor pre-fix flipped this to a collision
    for the same ~5e-7 weighted norm, measured independently from the upward case.
    """
    states, validity, types = _disagreeing_scene(vx=0.56, vy=0.0,
                                                  logged_heading=np.pi / 2)
    space = PerturbationSpace(states, validity, types, 0, 1)
    assert not space.baseline_replay_collides, 'fixture regressed: baseline must be safe'

    # Straddle the OLD cliff (v == 0.5 exactly, reached at dvx0 = 0.5 - 0.56 = -0.06)
    # with a delta difference of 1e-6 either side of it.
    just_above = space.apply(np.array([-0.06 + 1e-6, 0.0, 0.0, 0.0]))
    just_below = space.apply(np.array([-0.06 - 1e-6, 0.0, 0.0, 0.0]))
    assert float(just_above[1, 0, 4]) == pytest.approx(
        float(just_below[1, 0, 4]), abs=1e-6), (
        'heading still jumps across the old cliff at v == floor, crossing downward'
    )

    # And a delta that only nudges toward the floor without reaching the band at
    # all must leave heading exactly at derived (safe) — not flip early.
    still_above_band = space.apply(np.array([-0.005, 0.0, 0.0, 0.0]))
    collided, frame = check_collision_trajectory(still_above_band, validity, 0, 1)
    assert not collided, (
        f'a delta that never reached the transition band produced a verified '
        f'collision at frame {frame}'
    )


def test_heading_is_continuous_across_the_whole_band():
    """
    The property itself, not just its two probed consequences above: sampling
    heading at 1000 points across and around the transition band must never show a
    jump larger than a small multiple of the typical per-sample step — the
    signature of a removed discontinuity, on the exact fixture the boundary cliff
    was measured on.
    """
    from src.optimization.perturbation_space import HEADING_TRANSITION_WIDTH

    states, validity, types = _disagreeing_scene(vx=0.0, vy=0.45, logged_heading=0.0)
    space = PerturbationSpace(states, validity, types, 0, 1)

    dvy_values = np.linspace(0.0, 2 * HEADING_TRANSITION_WIDTH + 0.1, 1000)
    headings = np.array([
        float(space.apply(np.array([0.0, dv, 0.0, 0.0]))[1, 0, 4])
        for dv in dvy_values
    ])
    jumps = np.abs(np.diff(headings))
    nonzero = jumps[jumps > 1e-12]
    assert len(nonzero) > 10, 'the sweep never left the flat region — test is vacuous'
    assert jumps.max() < 50 * np.median(nonzero), (
        f'largest per-sample heading jump {jumps.max():.4f} is far above the '
        f'typical {np.median(nonzero):.4f} — a discontinuity survived'
    )


def test_heading_stays_exactly_logged_below_the_floor_under_arbitrary_search():
    """
    Not just 'small deltas near the boundary are safe': NOTHING strictly below the
    floor should ever produce anything but the logged heading, for any delta in the
    search box — confirming the anchored-band guarantee
    (perturbation_space.py::_linear_heading's own docstring) holds under a
    randomised search, not only under the two hand-picked probes above.
    """
    states, validity, types = _disagreeing_scene(vx=0.0, vy=0.3, logged_heading=0.0)
    space = PerturbationSpace(states, validity, types, 0, 1)

    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(3000):
        delta = rng.uniform(space.bounds[:, 0], space.bounds[:, 1])
        replayed = space.apply(delta)
        speed0 = float(np.hypot(replayed[1, 0, 2], replayed[1, 0, 3]))
        if speed0 >= V_HEADING_MIN:
            continue
        checked += 1
        assert float(replayed[1, 0, 4]) == pytest.approx(0.0, abs=1e-9), (
            f'delta {delta.tolist()} left frame-0 speed {speed0} below the floor '
            f'but did not replay the logged heading exactly'
        )
    assert checked > 50, (
        f'only {checked}/3000 draws stayed below the floor — barely exercised'
    )


def test_the_antipodal_tie_break_is_logged_not_arctan2_of_near_zero():
    """
    THE DEGENERATE CASE, PINNED AS POLICY. Logged and derived exactly pi apart, at
    the band's midpoint speed where the vector blend cancels toward (0, 0) —
    arctan2(~0, ~0) would return an angle unrelated to either input, the same
    silent-default failure mode A01 itself fixed (arctan2(0,0)=0 silently
    overwriting a parked car's logged heading), relocated rather than removed. Must
    fall back to logged, exactly, not to whatever float noise survives the blend.

    Also the case that caught two real bugs during development, both found by
    independent review rather than in-house. First: computing `derived` at float32
    precision (as the pre-existing code above does, for the heading_speed_floor=None
    path) carried ~2.4e-7 rad of error near pi, enough on its own to skip a naive
    1e-9 fallback threshold. Fixed by recomputing `derived` at float64 in the blend.
    Second, found AFTER that fix shipped: even at float64, this exact fixture's
    real |blend| floor is NOT the idealized ~1e-16 an exact-arithmetic hand check
    predicts — it measures ~1.4e-6, because `speed` itself (float32-derived from
    the rollout) never lands on the smoothstep's exact analytic midpoint, and that
    small error gets amplified by the smoothstep's own slope there. The fallback
    threshold (perturbation_space.py::_linear_heading) is sized against this
    measured floor, not the idealized one — see its own comment for the
    derivation, and test_heading_is_continuous_near_but_not_at_the_antipodal_point
    below for why the threshold cannot simply be made larger to compensate.
    """
    states, validity, types = _disagreeing_scene(
        vx=-0.525, vy=0.0, logged_heading=0.0, lateral=6.0)
    space = PerturbationSpace(states, validity, types, 0, 1)

    replayed = space.apply(np.zeros(4))
    assert float(replayed[1, 0, 4]) == pytest.approx(0.0, abs=1e-9), (
        f'antipodal tie-break did not fall back to logged: got '
        f'{float(replayed[1, 0, 4])!r}'
    )


def test_heading_is_continuous_near_but_not_at_the_antipodal_point():
    """
    THE REGRESSION THIS TEST EXISTS TO CATCH: nothing before this test checked the
    zone the antipodal fallback's own threshold names in its docstring, so nothing
    caught the threshold itself being too wide. Independent review found it: at the
    ORIGINAL 1e-3 threshold, a fixture 0.057 deg off exact antipodal (well inside
    that threshold's own ~0.1146 deg danger zone) jumped 2.625 rad for a
    weighted-norm-equivalent change of ~2.6e-7 m/s — the fallback branch forcing
    `logged` at points where the raw, un-fallen-back arctan2(by, bx) was already
    smooth. The corrected threshold (1e-5) shrinks that zone to ~0.001146 deg; this
    fixture, at 0.1 deg off exact antipodal, sits well outside the new zone and
    well inside the old one — exactly the case that would have caught the
    regression.

    CHECKED BY SHRINKING EPSILON, NOT BY A FIXED SAMPLING GRID — a first version of
    this test swept a uniform grid across the whole band and failed: the true
    transition here is legitimately steep (a near-antipodal pair still swings
    close to pi of heading across a very narrow speed window, ~1.76e-5 wide at this
    angle) so a coarse fixed grid aliases against it and reports a fake 1.08 rad
    'jump' that is really several under-sampled steps. That is a real property of
    vector-blend interpolation near ANY near-antipodal pair (documented in
    _linear_heading's own 'NOT fixed' list) and not itself a defect. The
    discriminating test for a genuine discontinuity — the one the original defect
    and the 1e-3 regression both would fail — is whether shrinking the straddle
    around the point of maximum sensitivity shrinks the output difference too. A
    real jump does not shrink no matter how small the straddle gets (measured
    directly for the two bugs above: both stayed near a fixed nonzero value at
    every epsilon tried). This one does, monotonically, down through 1e-9.
    """
    delta_angle = np.radians(0.1)
    states = np.zeros((2, 2, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 1] = 6.0
    states[1, :, 5:7] = [2.0, 0.6]
    states[1, :, 4] = 0.0
    validity = np.ones((2, 2), dtype=bool)
    types = np.array([1, 3])
    space = PerturbationSpace(states, validity, types, 0, 1)

    # The theoretical point of maximum sensitivity for ANY antipodal-ish pair is
    # the band's own midpoint (w=0.5), independent of the disagreement angle — see
    # _linear_heading's docstring on why the degenerate point is always there.
    midpoint = space.heading_speed_floor + space.heading_transition_width / 2

    diffs = []
    for half_width in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
        speeds = np.array([midpoint - half_width, midpoint + half_width],
                          dtype=np.float32)
        vx = (speeds * np.cos(np.pi - delta_angle)).astype(np.float32)
        vy = (speeds * np.sin(np.pi - delta_angle)).astype(np.float32)
        h = np.array(space._linear_heading(vx, vy, 0, 2))
        diffs.append(abs(float(h[1]) - float(h[0])))

    diffs = np.array(diffs)
    assert diffs[0] > 0.5, (
        f'the widest straddle (half_width=1e-3) produced diff={diffs[0]:.4f} — too '
        f'small to show the swing this fixture is meant to sweep through; fixture '
        f'regressed'
    )
    assert diffs[-1] < 1e-3, (
        f'the narrowest straddle (half_width=1e-9) still produced diff='
        f'{diffs[-1]:.4f} — a genuine discontinuity does not shrink with epsilon'
    )
    # THE ASSERTION THAT ACTUALLY DISCRIMINATES A JUMP FROM A TOO-WIDE FALLBACK
    # ZONE — added after the first version of this test was shown not to catch its
    # own regression. Reverting the threshold to the old 1e-3 and running this test
    # still passed diffs[-1] < 1e-3 and the monotonic check below, because BOTH
    # straddled points at the smaller half-widths land inside the too-wide
    # forced-logged zone and tie at exactly 0.0 — a sequence of trailing exact
    # zeros is trivially "shrinking" and trivially small, even though it reflects a
    # clamp, not a limit. A genuine shrink-with-epsilon, by contrast, stays
    # meaningfully nonzero at every half-width except (possibly) the very last,
    # where float32 quantisation can legitimately tie the two straddled points —
    # measured on the current (1e-5) threshold: diffs[:-1] range from 3.11 down to
    # 4.1e-3, four to six orders of magnitude above this bound, nothing near it.
    assert np.all(diffs[:-1] > 1e-9), (
        f'diff sequence {diffs.tolist()} contains a near-zero entry before the '
        f'last half-width — looks like a too-wide fallback zone is clamping '
        f'nearby points to the same value rather than the heading genuinely '
        f'converging as epsilon shrinks'
    )
    # Monotonically non-increasing, with slack for the float32 quantisation visible
    # at the smallest half-widths (vx/vy are cast to float32 above, matching the
    # real rollout, so the very last steps can tie rather than strictly decrease).
    assert np.all(np.diff(diffs) <= 1e-6), (
        f'diff sequence {diffs.tolist()} is not shrinking as epsilon shrinks — '
        f'looks like a discontinuity survives somewhere in this range'
    )


def test_zero_or_none_transition_width_reproduces_the_pre_fix_step():
    """
    The escape hatch, mirroring heading_speed_floor's own. A future Colab pass
    needs to A/B this fix against real data, which requires the OLD behaviour to
    still be reachable, exactly, not approximated.
    """
    states, validity, types = _disagreeing_scene(vx=0.0, vy=0.45, logged_heading=0.0)

    for width in (0.0, None):
        space = PerturbationSpace(states, validity, types, 0, 1,
                                  heading_transition_width=width)
        just_below = space.apply(np.array([0.0, 0.049999, 0.0, 0.0]))
        just_at = space.apply(np.array([0.0, 0.05, 0.0, 0.0]))
        assert float(just_below[1, 0, 4]) == pytest.approx(0.0, abs=1e-9)
        assert float(just_at[1, 0, 4]) == pytest.approx(np.pi / 2, abs=1e-6), (
            f'width={width!r} did not reproduce the pre-fix step at the floor'
        )


def test_transition_width_defaults_to_nonzero_so_the_fix_is_not_opt_in():
    """
    THE FIRST QUESTION THIS BATCH HAD TO ANSWER BEFORE ANY DIFF: every production
    call site (_stress_one, export_perturbed_path's geometry rebuild) constructs
    PerturbationSpace positionally, with no heading_speed_floor or
    heading_transition_width keyword — so the constructor's OWN default is the only
    thing that reaches optimize_scenario and stress_test_scenarios. A correctness
    fix that defaulted to None (reproducing the pre-fix step) would never actually
    apply in production. Pinned here so a future edit cannot silently regress it by
    changing the default.
    """
    import inspect
    sig = inspect.signature(PerturbationSpace.__init__)
    default = sig.parameters['heading_transition_width'].default
    assert default not in (None, 0, 0.0), (
        f'heading_transition_width defaults to {default!r} — the boundary-cliff '
        f'fix would be off by default in every production call site'
    )
