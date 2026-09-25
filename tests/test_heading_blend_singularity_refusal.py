"""
test_heading_blend_singularity_refusal.py — independent review, 2026-09-24.

_linear_heading's own docstring in perturbation_space.py already documents the
antipodal degenerate case as "NOT fixed, AND STRUCTURAL RATHER THAN AN OVERSIGHT":
straight-line vector averaging between two near-exactly-opposite unit vectors is
topologically undefined, and the 1e-3 -> 1e-5 fallback-threshold correction only
narrowed the window this lives in, exactly as that correction's own comment says
it would.

Reproduced live, exactly, before this fix: a cyclist at speed 0.525 m/s, logged
heading 0, velocity [-0.524999976, 0.000009975] (nearly antipodal to logged, at
very close to the transition band's own midpoint speed). Clean baseline
(baseline_replay_error=3.77e-7 m, no baseline_replay_collides). A delta of
[-1.192e-07, 0, 0, 0] -- weighted norm 5.96e-08 -- flips heading to 2.1133502 rad
and produces a verified frame-0 collision.

THE FIX: PerturbationSpace now refuses to construct (HeadingBlendSingularityError)
when the baseline replay's own logged/derived blend magnitude is already close to
the degenerate point (HEADING_BLEND_SINGULARITY_MARGIN = 0.5, i.e. logged and the
velocity-implied heading disagree by more than 120 degrees at a frame whose speed
is in or near the transition band) -- mirroring ReplayFidelityError's own
precedent (refuse ill-posed input rather than silently search it) rather than a
third narrowing of the same fallback threshold.

CALIBRATED, NOT GUESSED: HEADING_BLEND_SINGULARITY_MARGIN's own comment in
perturbation_space.py carries the full measured table this margin was chosen
from -- 0.5 corresponds to 60 degrees of separation from exact antipodal, and
guarantees every UNREFUSED scenario requires at least ~0.006 weighted norm to
exploit this mechanism, comfortably above the 0.0032 weighted norm the ORIGINAL
audit finding (F01) already flagged as clearly too small to be real.

Pure numpy/shapely, no database. Run:
    ./venv/bin/python -m pytest tests/test_heading_blend_singularity_refusal.py -q
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


def _near_sdc_repro():
    """
    THE FINDING'S OWN GEOMETRY. SDC stationary at (0, 0), cyclist at (1.6, 1.5) --
    close enough that a heading of 2.1133502 rad overlaps it, while heading 0.0
    (the logged value) does not. Every number here is the actual reproduction,
    not a constructed-for-the-test approximation.
    """
    T, dt = 10, 0.1
    vx, vy = -0.524999976, 0.000009975
    x0, y0 = 1.6, 1.5

    states = np.zeros((2, T, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 5:7] = [1.8, 0.5]
    states[1, :, 4] = 0.0
    for t in range(T):
        states[1, t, 0] = x0 + vx * dt * t
        states[1, t, 1] = y0 + vy * dt * t
        states[1, t, 2] = vx
        states[1, t, 3] = vy
    validity = np.ones((2, T), dtype=bool)
    types = np.array([1, 3])
    return states, validity, types


def _isolated_scene(vx, vy, x0=1000.0, y0=1000.0, T=10, dt=0.1):
    """Cyclist far from any SDC -- isolates the heading-blend mechanism from
    collision geometry, for the boundary/overcorrection tests below."""
    states = np.zeros((2, T, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 5:7] = [1.8, 0.5]
    states[1, :, 4] = 0.0
    for t in range(T):
        states[1, t, 0] = x0 + vx * dt * t
        states[1, t, 1] = y0 + vy * dt * t
        states[1, t, 2] = vx
        states[1, t, 3] = vy
    validity = np.ones((2, T), dtype=bool)
    types = np.array([1, 3])
    return states, validity, types


def _velocity_at_separation(offset_deg, speed=0.525):
    """vx, vy whose derived heading is `offset_deg` away from exact antipodal to
    a logged heading of 0.0 -- the same construction HEADING_BLEND_SINGULARITY_
    MARGIN's own calibration table was measured with."""
    theta = np.pi - np.radians(offset_deg)
    return speed * np.cos(theta), speed * np.sin(theta)


def test_the_exact_repro_is_refused_at_construction():
    """
    THE PRIMARY REPRO. Before this fix, this exact geometry produced a verified
    frame-0 collision from a weighted-norm 5.96e-08 delta. Now it must never reach
    a search at all.
    """
    states, validity, types = _near_sdc_repro()
    with pytest.raises(HeadingBlendSingularityError) as excinfo:
        PerturbationSpace(states, validity, types, 0, 1)
    # The measured baseline magnitude from this exact fixture, confirmed live
    # during investigation: ~9.6e-6, far below the 0.5 margin.
    assert excinfo.value.baseline_heading_blend_min_magnitude < 1e-4
    assert excinfo.value.margin == HEADING_BLEND_SINGULARITY_MARGIN


def test_the_exact_repro_trips_the_refusal_through_the_real_stress_one_path():
    """
    NOT JUST DIRECT PerturbationSpace CONSTRUCTION (independent review,
    2026-09-24, raised on the first version of this file): the caller that
    actually matters is src.scoring.batch_scorer._stress_one, which wraps
    PerturbationSpace(...) in its own try/except and previously caught only
    ReplayFidelityError. HeadingBlendSingularityError is a SIBLING of
    ReplayFidelityError, not a subclass, by design (see its own docstring) — so
    an unwrapped except clause here would have let it fall through to
    stress_test_scenarios' generic handler exactly the way this file's own
    comment warns against: 'arrive as status=error with a stringified message,
    throwing away the structured reason/error'. This drives the REAL
    _stress_one on the finding's own exact geometry and checks the result dict
    it actually returns, not a hand-built stand-in for what it should return.
    """
    from src.scoring.batch_scorer import _stress_one
    from src.scoring.db import OUTCOME_HEADING_BLEND_SINGULARITY, OUTCOMES_SEARCH_RAN

    states, validity, types = _near_sdc_repro()
    result = _stress_one(states, validity, types, 0)

    assert result['status'] == OUTCOME_HEADING_BLEND_SINGULARITY, result
    assert result['outcome'] == OUTCOME_HEADING_BLEND_SINGULARITY, result
    assert result['outcome'] not in OUTCOMES_SEARCH_RAN, (
        'this outcome must not be read as "a search ran" -- it would set '
        'stress_tested_at on a scenario that was never actually searched'
    )
    # EVERY field the exception carries reaches the result dict, not a selection
    # of them (audit R07's own standard, applied to this exception too).
    assert result['baseline_heading_blend_min_magnitude'] == pytest.approx(
        9.606e-06, rel=1e-2
    )
    assert result['margin'] == HEADING_BLEND_SINGULARITY_MARGIN
    assert result['target_idx'] == 1
    assert result['challengers_searched'] == 0


def test_the_exact_repro_still_collides_when_the_check_is_bypassed():
    """
    PROOF THE REFUSAL IS THE ONLY THING STOPPING THIS, not some other change.
    Monkeypatches HEADING_BLEND_SINGULARITY_MARGIN to -inf for the duration of
    ONE construction, so __init__'s new gate never fires (the comparison is
    always false) while everything else __init__ does runs completely normally
    -- then replays the exact delta from the finding and confirms the exact
    reported heading and the verified frame-0 collision it produces, the same
    live reproduction the plan for this fix was built on. Pinned so a future
    change to the gate cannot silently stop testing what it is actually
    guarding.
    """
    states, validity, types = _near_sdc_repro()

    # Build a real, ungated space by monkeypatching the margin to -inf for the
    # duration of this one construction -- simpler and less fragile than
    # reimplementing __init__ by hand, and it exercises the exact same code path
    # apply() below will run.
    import src.optimization.perturbation_space as ps_module
    original_margin = ps_module.HEADING_BLEND_SINGULARITY_MARGIN
    ps_module.HEADING_BLEND_SINGULARITY_MARGIN = float('-inf')
    try:
        space = PerturbationSpace(states, validity, types, 0, 1)
    finally:
        ps_module.HEADING_BLEND_SINGULARITY_MARGIN = original_margin

    assert space.baseline_replay_collides is False
    assert space.baseline_replay_error < 1e-5

    delta = np.array([-1.192e-07, 0.0, 0.0, 0.0], dtype=np.float32)
    assert space.weighted_norm(delta) == pytest.approx(5.96e-08, rel=1e-2)

    perturbed = space.apply(delta)
    assert float(perturbed[1, 0, 4]) == pytest.approx(2.1133502, abs=1e-4), (
        f'expected the exact reported heading flip, got {perturbed[1, 0, 4]!r}'
    )
    collided, t_hit = check_collision_trajectory(perturbed, validity, 0, 1)
    assert collided and t_hit == 0, (
        f'fixture regressed: expected a verified frame-0 collision, got '
        f'collided={collided} t_hit={t_hit}'
    )


def test_a_90_degree_disagreement_still_constructs_and_searches_normally():
    """
    THE OVERCORRECTION GUARD. 90 degrees of separation from exact antipodal is a
    comfortably ordinary case -- a real cyclist whose recorded facing and recorded
    motion direction disagree by a quarter turn is unremarkable. Measured during
    calibration: baseline magnitude 0.707, and no delta up to weighted norm 2.0
    (DE's entire search range) produces the kind of heading jump this fix guards
    against. Must construct without raising anything.
    """
    vx, vy = _velocity_at_separation(90.0)
    states, validity, types = _isolated_scene(vx, vy)
    space = PerturbationSpace(states, validity, types, 0, 1)
    assert space.baseline_heading_blend_min_magnitude == pytest.approx(0.7071, abs=1e-3)


def test_a_separation_just_inside_the_margin_is_refused():
    """
    THE BOUNDARY, FROM THE DANGEROUS SIDE. 55 degrees of separation from exact
    antipodal measures a baseline magnitude below 0.5 (cos(62.5 deg) = 0.4617) --
    must refuse.
    """
    vx, vy = _velocity_at_separation(55.0)
    states, validity, types = _isolated_scene(vx, vy)
    with pytest.raises(HeadingBlendSingularityError) as excinfo:
        PerturbationSpace(states, validity, types, 0, 1)
    assert excinfo.value.baseline_heading_blend_min_magnitude < HEADING_BLEND_SINGULARITY_MARGIN


def test_a_separation_just_outside_the_margin_constructs():
    """
    THE BOUNDARY, FROM THE SAFE SIDE. 65 degrees of separation from exact
    antipodal measures a baseline magnitude above 0.5 (cos(57.5 deg) = 0.5373) --
    must NOT refuse. Proves the margin has a real, non-trivial safe side and is
    not accidentally refusing everything.
    """
    vx, vy = _velocity_at_separation(65.0)
    states, validity, types = _isolated_scene(vx, vy)
    space = PerturbationSpace(states, validity, types, 0, 1)
    assert space.baseline_heading_blend_min_magnitude >= HEADING_BLEND_SINGULARITY_MARGIN


def test_a_vehicle_challenger_is_never_affected():
    """
    _linear_heading IS NEVER CALLED FOR VEHICLES (the bicycle model carries theta
    as real state) -- baseline_heading_blend_min_magnitude must stay at its
    initial float('inf'), and construction must never raise
    HeadingBlendSingularityError for a vehicle challenger. An ordinary, cleanly
    replayable constant-velocity vehicle track is enough to prove this; the point
    is that the NEW attribute is untouched, not that this fixture is itself
    near-antipodal (a vehicle's heading is real state, not derived from velocity
    at all, so "near-antipodal to velocity" is not even a meaningful risk for it).
    """
    T, dt = 10, 0.1
    v = 5.0
    states = np.zeros((2, T, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[0, :, 1] = 1000.0
    states[1, :, 5:7] = [4.5, 2.0]
    states[1, :, 4] = 0.0
    for t in range(T):
        states[1, t, 0] = 1.0 + v * dt * t
        states[1, t, 2] = v
        states[1, t, 3] = 0.0
    validity = np.ones((2, T), dtype=bool)
    types = np.array([1, 1])  # both vehicles

    space = PerturbationSpace(states, validity, types, 0, 1)
    assert space.baseline_heading_blend_min_magnitude == float('inf')
