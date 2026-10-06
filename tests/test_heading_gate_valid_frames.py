"""
test_heading_gate_valid_frames.py — the heading-singularity gate reads valid frames only.

PerturbationSpace refuses a non-vehicle challenger whose baseline replay sits near the
degenerate point of the logged/derived heading blend (HeadingBlendSingularityError; see
HEADING_BLEND_SINGULARITY_MARGIN). The minimum it compares against the margin used to be
taken over EVERY frame of the replay slice, including the frames where the challenger was
not observed. Those frames carry whatever the file stored there: 0, -1.0 or NaN in each
field. A padded heading is not an observation, so:

  * a padding of 0 or -1.0 produced a small blend magnitude on a perfectly sound track
    and refused it, and
  * a NaN padding made the minimum NaN, and `NaN < margin` is False, so the gate
    admitted the track, hiding a genuine singularity on the observed frames.

The gate now takes the minimum over the challenger's valid frames of the same t0:end
slice, and applies only where the blend was actually computed (a non-vehicle target, a
heading_speed_floor, a non-zero heading_transition_width). A minimum that is not finite
on those frames is refused too.

No test here hardcodes a number obtained by running the code. Where a minimum is
asserted it is either analytic (logged and derived heading parallel gives a blend of
magnitude 1) or recomputed at test time from the zero-delta replay by the helper below,
which is an independent re-statement of the blend, not a call into the code under test.

Pure numpy/shapely, no database. Run:
    ./venv/bin/python -m pytest tests/test_heading_gate_valid_frames.py -q
"""

import os
import sys

import numpy as np
import pytest
from shapely.errors import GEOSException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.optimization.perturbation_space as perturbation_space
from src.optimization.perturbation_space import (
    HEADING_BLEND_SINGULARITY_MARGIN,
    HEADING_TRANSITION_WIDTH,
    V_HEADING_MIN,
    HeadingBlendSingularityError,
    PerturbationSpace,
)

# The three values an unobserved frame has been seen to carry, written into EVERY field.
PADDINGS = [
    pytest.param(0.0, id='pad_0'),
    pytest.param(np.nan, id='pad_nan'),
    pytest.param(-1.0, id='pad_minus_one'),
]
NON_VEHICLE_TYPES = [pytest.param(2, id='pedestrian'), pytest.param(3, id='cyclist')]
VEHICLE = 1
SDC_IDX, TARGET_IDX = 0, 1

# Midpoint of the speed band: the blend weight is 0.5 there, so a logged heading
# antipodal to the velocity cancels almost exactly.
BAND_MID = V_HEADING_MIN + HEADING_TRANSITION_WIDTH / 2.0


def _scene(challenger_type=3, speed=BAND_MID, motion_angle=np.pi, logged=None,
           singular_offset=None, n=12, y0=50.0):
    """
    SDC stationary at the origin, challenger far away moving at constant velocity, so the
    zero-delta replay is exact and no replay-fidelity gate interferes.

    By default the logged heading equals the motion direction, i.e. the logged and
    derived headings are parallel and the blend magnitude is exactly 1 on every frame the
    challenger was observed. With singular_offset=eps the logged heading is antipodal to
    the motion direction up to eps radians: a real singularity at the band midpoint.
    """
    if logged is None:
        logged = motion_angle if singular_offset is None else motion_angle + np.pi + singular_offset
    vx, vy = speed * np.cos(motion_angle), speed * np.sin(motion_angle)
    t = np.arange(n) * 0.1
    states = np.zeros((2, n, 7), dtype=np.float32)
    states[SDC_IDX, :, 5:7] = [4.5, 2.0]
    states[TARGET_IDX, :, 5:7] = [0.6, 0.6] if challenger_type == 2 else [2.0, 0.6]
    states[TARGET_IDX, :, 0] = vx * t
    states[TARGET_IDX, :, 1] = y0 + vy * t
    states[TARGET_IDX, :, 2] = vx
    states[TARGET_IDX, :, 3] = vy
    states[TARGET_IDX, :, 4] = logged
    validity = np.ones((2, n), dtype=bool)
    types = np.array([VEHICLE, challenger_type])
    return states, validity, types


def _singular_scene(challenger_type=3, **kw):
    return _scene(challenger_type, singular_offset=2e-5, **kw)


def _pad(states, validity, agent, frames, value):
    """Mark `frames` of `agent` as not observed and write `value` into every field."""
    frames = list(frames)
    validity[agent, frames] = False
    states[agent, frames, :] = value


def _build(states, validity, types, **kw):
    return PerturbationSpace(states, validity, types, SDC_IDX, TARGET_IDX, **kw)


def _refused(states, validity, types, **kw):
    try:
        _build(states, validity, types, **kw)
    except HeadingBlendSingularityError as err:
        return err
    return None


def _smoothstep(x, edge0, edge1):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _independent_minima(space, states, validity):
    """
    (min over ALL frames of the slice, min over the challenger's VALID frames) of the
    logged/derived blend magnitude, recomputed here from the zero-delta replay. Float64
    throughout, from the module's published floor and width.
    """
    t0 = int(np.flatnonzero(validity[TARGET_IDX])[0])
    replay = space.apply(np.zeros(4, dtype=np.float32))
    vx = replay[TARGET_IDX, t0:, 2].astype(np.float64)
    vy = replay[TARGET_IDX, t0:, 3].astype(np.float64)
    logged = states[TARGET_IDX, t0:, 4].astype(np.float64)
    speed = np.hypot(vx, vy)
    w = _smoothstep(speed, V_HEADING_MIN, V_HEADING_MIN + HEADING_TRANSITION_WIDTH)
    derived = np.arctan2(vy, vx)
    bx = (1.0 - w) * np.cos(logged) + w * np.cos(derived)
    by = (1.0 - w) * np.sin(logged) + w * np.sin(derived)
    magnitude = np.hypot(bx, by)
    return float(np.min(magnitude)), float(np.min(magnitude[validity[TARGET_IDX, t0:]]))


# ── 1. a sound track is not refused for the padding after it ─────────────────────────


@pytest.mark.parametrize('challenger_type', NON_VEHICLE_TYPES)
@pytest.mark.parametrize('pad', PADDINGS)
def test_a_sound_track_with_an_unobserved_tail_is_admitted(challenger_type, pad):
    states, validity, types = _scene(challenger_type)
    _pad(states, validity, TARGET_IDX, range(5, 12), pad)

    space = _build(states, validity, types)

    # Finite and above the margin: a NaN minimum does not qualify, even though the old
    # gate let it through.
    assert np.isfinite(space.baseline_heading_blend_min_magnitude)
    assert space.baseline_heading_blend_min_magnitude >= HEADING_BLEND_SINGULARITY_MARGIN


# ── 2. the recorded minimum is over observed frames ─────────────────────────────────


@pytest.mark.parametrize('pad', PADDINGS)
def test_the_recorded_minimum_is_over_the_observed_frames(pad):
    states, validity, types = _scene()
    _pad(states, validity, TARGET_IDX, range(5, 12), pad)

    space = _build(states, validity, types)
    _, observed_min = _independent_minima(space, states, validity)

    # Analytic: logged and derived headings are parallel on every observed frame, so the
    # blend magnitude is 1 there. The independent recomputation must agree with it.
    assert observed_min == pytest.approx(1.0, abs=1e-6)
    assert space.baseline_heading_blend_min_magnitude == pytest.approx(observed_min, abs=1e-6)


# ── 3. a real singularity is still refused, with padding after it ───────────────────


@pytest.mark.parametrize('challenger_type', NON_VEHICLE_TYPES)
@pytest.mark.parametrize('pad', PADDINGS)
def test_a_real_singularity_is_refused_with_an_unobserved_tail(challenger_type, pad):
    # Passes on main on purpose for pads 0 and -1.0, because the old minimum is tiny either
    # way; fails on main only for NaN padding, which hides that minimum.
    states, validity, types = _singular_scene(challenger_type)
    _pad(states, validity, TARGET_IDX, range(8, 12), pad)

    refusal = _refused(states, validity, types)

    assert refusal is not None
    assert refusal.baseline_heading_blend_min_magnitude < 1e-3


def test_a_real_singularity_is_refused_with_no_padding_at_all():
    # Passes on main on purpose: nothing is padded, so old and new minima coincide.
    states, validity, types = _singular_scene()
    assert _refused(states, validity, types) is not None


# ── 4. padding inside the track cannot hide a singular observed heading ─────────────


@pytest.mark.parametrize('pad', PADDINGS)
def test_padding_in_an_interior_gap_cannot_hide_a_singular_observed_heading(pad):
    states, validity, types = _singular_scene()
    _pad(states, validity, TARGET_IDX, range(4, 6), pad)

    refusal = _refused(states, validity, types)

    assert refusal is not None
    assert np.isfinite(refusal.baseline_heading_blend_min_magnitude)
    assert refusal.baseline_heading_blend_min_magnitude < 1e-3


# ── 5-6. a challenger that appears late: the slice starts at its first valid frame ──


@pytest.mark.parametrize('pad', PADDINGS)
def test_a_late_challenger_with_an_unobserved_head_only(pad):
    # Passes on main on purpose: the frames before t0 are outside the t0:end slice, so
    # padding there never reached the old minimum either. This guards the slice start.
    states, validity, types = _scene()
    _pad(states, validity, TARGET_IDX, range(0, 3), pad)
    assert _refused(states, validity, types) is None

    states, validity, types = _singular_scene()
    _pad(states, validity, TARGET_IDX, range(0, 3), pad)
    assert _refused(states, validity, types) is not None


@pytest.mark.parametrize('pad', PADDINGS)
def test_a_late_challenger_with_an_unobserved_head_and_tail(pad):
    states, validity, types = _scene()
    _pad(states, validity, TARGET_IDX, range(0, 3), pad)
    _pad(states, validity, TARGET_IDX, range(8, 12), pad)

    space = _build(states, validity, types)

    assert space.t0 == 3
    assert np.isfinite(space.baseline_heading_blend_min_magnitude)
    assert space.baseline_heading_blend_min_magnitude == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize('pad', PADDINGS)
def test_a_late_challenger_singularity_is_found_at_the_right_offset(pad):
    # The singular frame is the first observed one (global frame 3); a mask shifted by
    # t0 would look at padding instead and miss it.
    states, validity, types = _scene()
    # parallel everywhere except the first observed frame, which is antipodal
    states[TARGET_IDX, 3, 4] = states[TARGET_IDX, 3, 4] + np.pi + 2e-5
    _pad(states, validity, TARGET_IDX, range(0, 3), pad)
    _pad(states, validity, TARGET_IDX, range(8, 12), pad)

    refusal = _refused(states, validity, types)

    assert refusal is not None
    assert refusal.baseline_heading_blend_min_magnitude < 1e-3


# ── 7. a vehicle target is untouched ─────────────────────────────────────────────────


@pytest.mark.parametrize('pad', PADDINGS)
def test_a_vehicle_target_is_unchanged_and_its_minimum_stays_infinite(pad):
    # Passes on main on purpose: a vehicle never reaches _linear_heading, so its minimum
    # is the initial inf on both. This guards the non-vehicle condition of the gate; a
    # gate that treated inf as non-finite would refuse every vehicle.
    states, validity, types = _scene(VEHICLE)
    _pad(states, validity, TARGET_IDX, range(5, 12), pad)

    space = _build(states, validity, types)

    assert space.baseline_heading_blend_min_magnitude == float('inf')


# ── 8. the escape hatches ────────────────────────────────────────────────────────────


@pytest.mark.parametrize('pad', PADDINGS)
@pytest.mark.parametrize('escape', [
    pytest.param({'heading_speed_floor': None}, id='no_floor'),
    pytest.param({'heading_transition_width': 0}, id='width_zero'),
    pytest.param({'heading_transition_width': None}, id='width_none'),
])
def test_an_escape_hatch_skips_the_gate_even_for_a_singular_padded_scene(escape, pad):
    # Passes on main on purpose: with the floor or the band switched off no blend is
    # computed, the minimum stays inf, and inf < margin is False. This guards the
    # escape-hatch conditions of the gate; without them the inf would be refused as
    # non-finite.
    states, validity, types = _singular_scene()
    _pad(states, validity, TARGET_IDX, range(8, 12), pad)

    space = _build(states, validity, types, **escape)

    assert space.baseline_heading_blend_min_magnitude == float('inf')


# ── 9-10. a non-finite value on an observed frame ────────────────────────────────────


@pytest.mark.parametrize('challenger_type', NON_VEHICLE_TYPES)
@pytest.mark.parametrize('bad', [np.nan, np.inf], ids=['nan', 'inf'])
def test_a_non_finite_heading_on_an_observed_frame_is_refused(bad, challenger_type):
    # The collision check skips frames where the SDC is invalid, so the first thing that
    # would trip over the value never sees it. The blend minimum is then non-finite.
    states, validity, types = _scene(challenger_type)
    states[TARGET_IDX, 4, 4] = bad
    validity[SDC_IDX, 4] = False

    refusal = _refused(states, validity, types)

    assert refusal is not None
    assert not np.isfinite(refusal.baseline_heading_blend_min_magnitude)


@pytest.mark.parametrize('bad', [np.nan, np.inf], ids=['nan', 'inf'])
def test_a_non_finite_heading_where_the_sdc_is_observed_still_raises_in_geos(bad):
    # Passes on main on purpose: documents behaviour this change does not touch. With
    # the SDC valid at that frame the exact collision check evaluates the corners first
    # and GEOS raises before the gate is reached.
    states, validity, types = _scene()
    states[TARGET_IDX, 4, 4] = bad

    with pytest.raises(GEOSException):
        _build(states, validity, types)


def test_a_non_finite_velocity_or_a_vehicle_still_raises_in_geos():
    # Passes on main on purpose: same as above, for a NaN velocity on a non-vehicle and
    # a NaN heading on a vehicle. Neither reaches the gate and neither is changed here.
    states, validity, types = _scene()
    states[TARGET_IDX, 4, 2] = np.nan
    validity[SDC_IDX, 4] = False
    with pytest.raises(GEOSException):
        _build(states, validity, types)

    states, validity, types = _scene(VEHICLE)
    states[TARGET_IDX, 4, 4] = np.nan
    validity[SDC_IDX, 4] = False
    with pytest.raises(GEOSException):
        _build(states, validity, types)


def test_a_blend_minimum_that_was_never_recorded_is_refused_for_a_non_vehicle(monkeypatch):
    # Defensive branch: _linear_heading leaves the minimum at inf only if the slice held no
    # observed frame, which a constructed space cannot produce (t0 is a valid frame). If it
    # ever happened, a non-vehicle with the band active must be refused, not admitted by
    # `inf < margin` being False. Reached here by stubbing the heading function out.
    # Fails on main by design: the old gate admits inf.
    def unrecorded(self, vx, vy, t0, end):
        return np.arctan2(vy, vx)

    monkeypatch.setattr(PerturbationSpace, '_linear_heading', unrecorded)
    states, validity, types = _scene()

    refusal = _refused(states, validity, types)

    assert refusal is not None
    assert refusal.baseline_heading_blend_min_magnitude == float('inf')


def test_a_minimum_exactly_at_the_margin_is_admitted(monkeypatch):
    # Passes on main on purpose: the comparison is strict on both. Guards the boundary.
    # The margin is set to the scene's own minimum, read at test time.
    states, validity, types = _scene()
    with monkeypatch.context() as patched:
        patched.setattr(perturbation_space, 'HEADING_BLEND_SINGULARITY_MARGIN', 0.0)
        at_margin = _build(states, validity, types).baseline_heading_blend_min_magnitude
    assert np.isfinite(at_margin)

    with monkeypatch.context() as patched:
        patched.setattr(perturbation_space, 'HEADING_BLEND_SINGULARITY_MARGIN', at_margin)
        assert _refused(states, validity, types) is None


# ── 11. the new gate admits a superset of the old one ───────────────────────────────

# Fixed seed so the scene set is the same on every run. The counts below are minimums
# stated in advance, not read off a run: if the generator ever stops producing them the
# test fails instead of passing vacuously.
SEED = 20261006
N_SCENES = 200
MIN_ADMITTED_BECAUSE_PADDING_WAS_IGNORED = 10   # old_min < margin <= new_min
MIN_NAN_OLD_MINIMUM = 5                         # a NaN-padded scene whose old minimum is NaN
MIN_GENUINE_REFUSALS = 10                       # new_min < margin
MIN_ADMITTED_BY_BOTH = 10                       # old_min >= margin


def _random_scene(rng):
    n = 12
    challenger_type = int(rng.choice([2, 3]))
    motion_angle = rng.uniform(-np.pi, np.pi)
    pad = [0.0, np.nan, -1.0][int(rng.integers(0, 3))]
    kind = int(rng.choice([0, 1, 2, 2, 2, 3, 3, 3]))
    if kind == 0:       # logged heading parallel to the motion: sound
        speed = rng.uniform(0.3, 0.8)
        logged = motion_angle
    elif kind == 3:     # sound, moving roughly against the padded heading, at a speed in the
        # band: the replay keeps that speed over the first unobserved frames, so the padded
        # heading and the motion direction are nearly antipodal where the blend is active
        speed = rng.uniform(V_HEADING_MIN, V_HEADING_MIN + HEADING_TRANSITION_WIDTH)
        motion_angle = (0.0 if np.isnan(pad) else pad) + np.pi + rng.uniform(-0.5, 0.5)
        logged = motion_angle
    elif kind == 1:     # logged heading anywhere, speed around the band
        speed = rng.uniform(0.48, 0.58)
        logged = rng.uniform(-np.pi, np.pi)
    else:               # logged heading near antipodal, speed inside the band
        speed = rng.uniform(V_HEADING_MIN + 0.1 * HEADING_TRANSITION_WIDTH,
                            V_HEADING_MIN + 0.9 * HEADING_TRANSITION_WIDTH)
        logged = motion_angle + np.pi + rng.uniform(-0.5, 0.5)
    states, validity, types = _scene(challenger_type, speed=speed, motion_angle=motion_angle,
                                     logged=logged, n=n)
    head = int(rng.integers(0, 4))
    tail = int(rng.integers(0, 5))
    if head:
        _pad(states, validity, TARGET_IDX, range(0, head), pad)
    if tail:
        _pad(states, validity, TARGET_IDX, range(n - tail, n), pad)
    if rng.random() < 0.3 and n - tail - 3 > head + 2:
        start = int(rng.integers(head + 2, n - tail - 3))
        _pad(states, validity, TARGET_IDX, range(start, start + int(rng.integers(1, 3))), pad)
    return states, validity, types


def test_the_new_gate_admits_a_superset_of_the_old_one(monkeypatch):
    rng = np.random.default_rng(SEED)
    margin = HEADING_BLEND_SINGULARITY_MARGIN
    flipped_by_padding = nan_old_minimum = genuine_refusals = admitted_by_both = 0

    for k in range(N_SCENES):
        states, validity, types = _random_scene(rng)

        # Build with the margin out of the way so the space exists whichever way the
        # gate would have gone, and read both minima from the replay itself.
        with monkeypatch.context() as patched:
            patched.setattr(perturbation_space, 'HEADING_BLEND_SINGULARITY_MARGIN', 0.0)
            space = _build(states, validity, types)
        old_min, new_min = _independent_minima(space, states, validity)

        # Observed frames are a subset of all frames, so where both are finite the
        # observed minimum cannot be lower.
        if np.isfinite(old_min):
            assert new_min >= old_min - 1e-9, k
        # The scene set must not sit on the margin, or the tolerance below means nothing.
        assert abs(new_min - margin) > 1e-6, k
        if np.isfinite(old_min):
            assert abs(old_min - margin) > 1e-6, k

        admitted = _refused(states, validity, types) is None

        # The new gate decides on the observed frames alone.
        assert admitted == (np.isfinite(new_min) and new_min >= margin), k
        # Superset: whatever the old gate admitted on a finite minimum, the new one does.
        # (A NaN old minimum was admitted by accident; that case is decided on new_min.)
        if np.isfinite(old_min) and old_min >= margin:
            assert admitted, k

        flipped_by_padding += bool(old_min < margin <= new_min)
        nan_old_minimum += bool(np.isnan(old_min))
        genuine_refusals += bool(new_min < margin)
        admitted_by_both += bool(np.isfinite(old_min) and old_min >= margin)

    assert flipped_by_padding >= MIN_ADMITTED_BECAUSE_PADDING_WAS_IGNORED
    assert nan_old_minimum >= MIN_NAN_OLD_MINIMUM
    assert genuine_refusals >= MIN_GENUINE_REFUSALS
    assert admitted_by_both >= MIN_ADMITTED_BY_BOTH
