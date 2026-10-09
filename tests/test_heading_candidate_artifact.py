"""
test_heading_candidate_artifact.py — a SYNTHETIC reproduction of the "candidate-time" heading artifact
reported in independent review, Round 1. A fixture that documents an OPEN limitation; it does not claim
the gate is fixed, and it does not claim the report is false.

WHAT WAS REPORTED (Round 1): the baseline-only singularity gate does not protect every candidate the
search generates later. For an admitted cyclist scene whose baseline blend magnitude is 0.516673 (above
the 0.5 gate), the delta [1.3709068e-6, 0, -0.001, 0] (weighted norm 0.0010000003) rotates the frame-90
footprint to 2.11335 rad and produces a verified collision; holding the same simulated positions and
velocities at the logged heading produces none.

WHAT THIS IS: a SYNTHETIC cyclist fitted to those numbers, NOT the original scene, which is not in the
repository. Baseline blend magnitude 0.516673 and the delta match the report; the candidate's frame-90
heading is 2.11333 here against 2.11335 reported (the logged heading was chosen to land there); the
logged heading, the velocity direction and where the SDC stands are this fixture's own choices, so the
frame at which contact first happens (86 here) is this fixture's, not the report's.

WHAT IT SHOWS: with default settings the space is built (the gate passes), the candidate's blend
magnitude stays at the baseline's 0.516673 at every frame (so a check on blend magnitude alone would not
see it), and a bias of -0.001 m/s^2 over 9 s lowers the speed by about 0.008 m/s, which moves the heading
by 0.67 rad because the whole logged/derived transition band is only 0.05 m/s wide. The centre moves 4 cm.
The contact needs that heading: with the candidate's positions at the LOGGED heading (control A) or at
the BASELINE heading (control B) there is none.

THE CODE ALREADY SAYS THIS IS OPEN, in src/optimization/perturbation_space.py:
  * _linear_heading's docstring: "NOT fixed — above the floor the 1/|v| sensitivity remains. At v = 0.5 a
    unit of weighted norm still buys about 2 rad, where a vehicle's dtheta0 buys 0.2, so a linear agent's
    orientation stays roughly ten times cheaper to rotate than a vehicle's. Bounded, not equalised;"
  * HEADING_BLEND_SINGULARITY_MARGIN's comment: "WHAT THIS DOES NOT CLOSE, stated rather than implied: this
    gates on the BASELINE trajectory's own minimum magnitude only. A larger delta — via the constant
    accel-bias dimensions, over the whole rollout — could in principle steer a DIFFERENT, currently-safe
    frame into a near-antipodal alignment the baseline never reached."

TWO TESTS:
  * test_the_reproduction_still_reproduces  pins the numbers above, so the fixture cannot silently stop
    reproducing (for example if the transition band, the floor or the margin changed).
  * test_a_tiny_delta_does_not_create_contact_that_the_baseline_heading_would_not  is a STRICT xfail
    stating what a fix would give. When a fix lands it flips to XPASS and the run FAILS until the marker is
    removed, which is the point: nothing changes this behaviour unnoticed.

Pure numpy/shapely, no database. Run:
    ./venv/bin/python -m pytest tests/test_heading_candidate_artifact.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.danger.collision_detector import check_collision_trajectory
from src.optimization.perturbation_space import (
    HEADING_BLEND_SINGULARITY_MARGIN,
    HeadingBlendSingularityError,
    PerturbationSpace,
)

T, DT = 91, 0.1
SPEED = 0.525                                   # the middle of the transition band [0.5, 0.55]
BASELINE_BLEND = 0.516673                       # reported, and above the 0.5 gate
# logged vs derived heading that gives |blend| = BASELINE_BLEND at the band's middle (w = 0.5)
ANGLE_LOGGED_TO_DERIVED = 2 * np.arccos(BASELINE_BLEND)
LOGGED_HEADING = 2.47506                        # chosen so the candidate's frame-90 heading is 2.11333
VELOCITY_DIRECTION = LOGGED_HEADING - ANGLE_LOGGED_TO_DERIVED
DELTA = np.array([1.3709068e-6, 0.0, -0.001, 0.0], dtype=np.float32)    # as reported
SDC_OFFSET = (-2.7, 1.8)                        # from the candidate's frame-90 centre; the SDC faces +x


def _scene(sdc_xy):
    states = np.zeros((2, T, 7), dtype=np.float32)
    states[0, :, 0:2] = sdc_xy
    states[0, :, 5:7] = [4.5, 2.0]
    vx, vy = SPEED * np.cos(VELOCITY_DIRECTION), SPEED * np.sin(VELOCITY_DIRECTION)
    for t in range(T):
        states[1, t, 0] = vx * DT * t
        states[1, t, 1] = vy * DT * t
    states[1, :, 2], states[1, :, 3] = vx, vy
    states[1, :, 4] = LOGGED_HEADING
    states[1, :, 5:7] = [1.8, 0.5]
    return states, np.ones((2, T), dtype=bool), np.array([1, 3])      # SDC a vehicle, the target a cyclist


def _space():
    # where the candidate's frame-90 centre is, with the SDC out of the way, to place the SDC relative to it
    far, validity, types = _scene((1000.0, 1000.0))
    probe = PerturbationSpace(far, validity, types, 0, 1).apply(DELTA)
    centre = (float(probe[1, 90, 0]), float(probe[1, 90, 1]))
    states, validity, types = _scene((centre[0] + SDC_OFFSET[0], centre[1] + SDC_OFFSET[1]))
    return PerturbationSpace(states, validity, types, 0, 1), validity


def _contact(states, validity):
    return check_collision_trajectory(states, validity, 0, 1)


def test_the_reproduction_still_reproduces():
    space, validity = _space()                          # constructing it IS the gate: it did not refuse

    # the gate was passed, with the reported baseline magnitude
    assert space.baseline_replay_collides is False
    assert space.baseline_replay_error < 1e-4
    assert space.baseline_heading_blend_min_magnitude == pytest.approx(BASELINE_BLEND, abs=1e-6)
    assert space.baseline_heading_blend_min_magnitude >= HEADING_BLEND_SINGULARITY_MARGIN
    # the reported delta, at the reported size
    assert space.weighted_norm(DELTA) == pytest.approx(0.0010000003, rel=1e-6)

    baseline = space.apply(np.zeros(4, dtype=np.float32))
    candidate = space.apply(DELTA)
    candidate_min_blend = space._last_heading_blend_min_magnitude      # set by the apply() just above

    # headings at frame 90: baseline, candidate, logged
    assert float(baseline[1, 90, 4]) == pytest.approx(1.44723, abs=1e-4)
    assert float(candidate[1, 90, 4]) == pytest.approx(2.11333, abs=1e-4)
    assert LOGGED_HEADING == pytest.approx(2.47506, abs=1e-5)
    # the candidate never gets near the singularity: its minimum blend magnitude is the baseline's
    assert candidate_min_blend == pytest.approx(BASELINE_BLEND, abs=1e-6)
    # and its centre barely moves: 4 cm at frame 90
    assert float(np.hypot(*(candidate[1, 90, :2] - baseline[1, 90, :2]))) < 0.05

    # contact, and the two controls that say it comes from the heading
    assert _contact(baseline, validity) == (False, -1)
    assert _contact(candidate, validity) == (True, 86)
    held_logged = candidate.copy()
    held_logged[1, :, 4] = LOGGED_HEADING                                   # control A
    held_baseline = candidate.copy()
    held_baseline[1, :, 4] = baseline[1, :, 4]                              # control B
    assert _contact(held_logged, validity) == (False, -1)
    assert _contact(held_baseline, validity) == (False, -1)


@pytest.mark.xfail(
    strict=True,
    reason='OPEN LIMITATION, documented in perturbation_space.py ("NOT fixed", "WHAT THIS DOES NOT CLOSE"): a '
           'candidate delta of weighted norm 1e-3 swings a linear challenger\'s heading by 0.67 rad through the '
           'transition band and creates contact that the same positions at the baseline heading do not. '
           'A fix (a refusal at construction, or a candidate-time check) flips this to XPASS; remove the marker then.',
)
def test_a_tiny_delta_does_not_create_contact_that_the_baseline_heading_would_not():
    try:
        space, validity = _space()
    except HeadingBlendSingularityError:
        return                                          # a gate that refuses this baseline is one way to fix it
    assert space.weighted_norm(DELTA) <= 1e-3 + 1e-9
    try:
        candidate = space.apply(DELTA)
    except ValueError:
        return                                          # so is a search that refuses the candidate
    baseline = space.apply(np.zeros(4, dtype=np.float32))
    at_baseline_heading = candidate.copy()
    at_baseline_heading[1, :, 4] = baseline[1, :, 4]
    assert _contact(candidate, validity)[0] == _contact(at_baseline_heading, validity)[0]
