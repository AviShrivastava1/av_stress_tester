"""
test_collision_geometry_float64.py — oriented-box corners are built in float64.

The state tensor is float32. At the kilometre-scale local coordinates the Waymo data
uses, one float32 step is about 1.2e-4 m near x = 1865 and grows with distance. Adding
a half-length to a float32 position in float32 rounds the facing edge a second time, so
two boxes that are genuinely separated by less than a step can round onto the same
coordinate, and Shapely's exact predicate then reports contact. get_corners now promotes
its inputs to float64 before the half-extents are added and the box is rotated. That
cannot recover precision lost when the state was stored; it only avoids a second,
avoidable rounding.

Three things must agree at every contact, whatever its size:
  * check_collision / check_collision_trajectory, the exact check the search verifies with,
  * scipy_optimizer._signed_gap, whose sign the DE archive uses as a certificate
    (g < 0 is "collides"), and
  * the arithmetic truth: for axis-aligned boxes, contact iff the float64 gap between the
    facing edges, computed from the STORED float32 values, is <= 0.

Coordinates in the fixed cases below (near x = 1865) are chosen test inputs: they are not
taken from any named scenario. Nothing here hardcodes a number obtained by running the
code: gaps, steps and edge positions are derived in the test from its own float32 inputs.

Pure numpy/shapely/scipy, no database. Run:
    ./venv/bin/python -m pytest tests/test_collision_geometry_float64.py -q
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.danger.collision_detector import (
    check_collision,
    check_collision_trajectory,
    get_corners,
)
from src.optimization.scipy_optimizer import _signed_gap

F32 = np.float32

# Chosen test inputs near x = 1865 m (float32, as stored in the state tensor).
X_A, Y = F32(1863.1277), F32(6.280408)
LEN_A, WID_A = F32(4.9857497), F32(2.5677965)
X_B = F32(1867.9562)
LEN_B, WID_B = F32(4.6710963), F32(2.328627)


def _states(x_a, x_b, y=Y, len_a=LEN_A, wid_a=WID_A, len_b=LEN_B, wid_b=WID_B):
    """Two axis-aligned agents, one frame, float32 exactly as the state tensor holds them."""
    st = np.zeros((2, 1, 7), dtype=np.float32)
    st[0, 0] = [x_a, y, 0, 0, 0, len_a, wid_a]
    st[1, 0] = [x_b, y, 0, 0, 0, len_b, wid_b]
    return st, np.ones((2, 1), dtype=bool)


def _edge_gap(st):
    """
    The arithmetic truth for axis-aligned boxes, in float64, from the STORED float32
    values: B's rear edge minus A's front edge. Positive = separated, <= 0 = contact.
    """
    xa, la = float(st[0, 0, 0]), float(st[0, 0, 5])
    xb, lb = float(st[1, 0, 0]), float(st[1, 0, 5])
    return (xb - lb / 2) - (xa + la / 2)


def _step_at_edge(st):
    """One float32 step at the coordinate of the facing edges."""
    return float(np.spacing(F32(float(st[0, 0, 0]) + float(st[0, 0, 5]) / 2)))


def _verdicts(st, va):
    hit = bool(check_collision_trajectory(st, va, 0, 1)[0])
    return hit, _signed_gap(st, va, 0, 1)


# ── 1. a gap smaller than one float32 step is not a collision ────────────────────────


def test_a_positive_gap_below_one_float32_step_is_no_collision_for_both_checks():
    st, va = _states(X_A, X_B)
    gap = _edge_gap(st)
    # The premise, derived here rather than asserted from a printed number.
    assert 0.0 < gap < _step_at_edge(st), (gap, _step_at_edge(st))

    hit, g = _verdicts(st, va)

    assert hit is False
    assert g > 0.0, g
    assert g == pytest.approx(gap, abs=1e-9)        # g is the real distance between the boxes


# ── 2. real overlaps are collisions for both checks ──────────────────────────────────


def test_a_real_overlap_is_a_collision_for_both_checks():
    # Passes on main on purpose: float32 rounding only ever pushed separated boxes INTO
    # contact, never an overlapping pair out of it. Guards the other side of the boundary.
    x_a = float(X_A)
    x_b = F32(x_a + float(LEN_A) / 2 + float(LEN_B) / 2 - 1e-3)      # about 1 mm of overlap
    st, va = _states(X_A, x_b)
    assert _edge_gap(st) < -5e-4

    hit, g = _verdicts(st, va)

    assert hit is True
    assert g < 0.0, g


def test_an_overlap_below_one_float32_step_is_a_collision_for_both_checks():
    # Passes on main on purpose, same reason as above: the mirror image of test 1.
    x_b = X_B
    st, va = _states(X_A, x_b)
    while _edge_gap(st) >= 0.0:                      # step B left, one float32 value at a time
        x_b = np.nextafter(x_b, F32(0.0))
        st, va = _states(X_A, x_b)
    assert -_step_at_edge(st) < _edge_gap(st) < 0.0

    hit, g = _verdicts(st, va)

    assert hit is True
    assert g < 0.0, g


# ── 3. g is never exactly 0, and the checks match the arithmetic across a sweep ─────

SEED = 20261006
N_CASES = 6000
EXCLUDE_ABS_GAP = 1e-9           # |float64 edge gap| below this: contact vs separation is not decidable by sign
MIN_PER_SIDE = 1000              # stated in advance: separated cases and overlapping cases the sweep must contain
MAX_EXCLUDED_SHARE = 0.01        # the exclusion margin may not be what makes the test pass


@pytest.fixture(scope='module')
def sweep():
    """
    Seeded near-contact cases at coordinates 100 m to 15 km: B's rear edge is placed
    within +-2 float32 steps of A's front edge, so many cases have |gap| below one step.
    Returns (gap, hit, g) per case.
    """
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(N_CASES):
        x = float(rng.uniform(100.0, 15000.0))
        y = float(rng.uniform(-15000.0, 15000.0))
        len_a, wid_a = F32(rng.uniform(1.5, 6.0)), F32(rng.uniform(0.5, 2.6))
        len_b, wid_b = F32(rng.uniform(1.5, 6.0)), F32(rng.uniform(0.5, 2.6))
        x_a = F32(x)
        step = float(np.spacing(F32(x + float(len_a) / 2)))
        x_b = F32(float(x_a) + float(len_a) / 2 + float(len_b) / 2 + rng.uniform(-2.0, 2.0) * step)
        st, va = _states(x_a, x_b, F32(y), len_a, wid_a, len_b, wid_b)
        hit, g = _verdicts(st, va)
        out.append((_edge_gap(st), hit, g))
    return out


def test_g_is_never_exactly_zero_and_its_sign_is_the_exact_check(sweep):
    # Passes on main on purpose: _signed_gap and the exact check have always shared one
    # get_corners, so they agree with each other on main as well. This is the property the
    # DE archive relies on, and it must survive the arithmetic change.
    assert all(g != 0.0 for _, _, g in sweep)
    assert all((g < 0.0) == hit for _, hit, g in sweep)


def test_the_exact_check_matches_the_float64_edge_gap_across_the_sweep(sweep):
    decided = [(gap, hit) for gap, hit, _ in sweep if abs(gap) >= EXCLUDE_ABS_GAP]
    excluded = len(sweep) - len(decided)
    separated = sum(1 for gap, _ in decided if gap > 0.0)
    overlapping = sum(1 for gap, _ in decided if gap < 0.0)
    print(f'\nsweep: {len(sweep)} cases, {excluded} excluded for |gap| < {EXCLUDE_ABS_GAP:g} m, '
          f'{separated} separated, {overlapping} overlapping')

    # The sweep must reach both sides of contact, and the margin must not be doing the work.
    assert separated >= MIN_PER_SIDE and overlapping >= MIN_PER_SIDE, (separated, overlapping)
    assert excluded <= MAX_EXCLUDED_SHARE * len(sweep), excluded

    wrong = [(gap, hit) for gap, hit in decided if hit != (gap <= 0.0)]
    assert not wrong, f'{len(wrong)} of {len(decided)} cases disagree with the float64 gap, e.g. {wrong[:3]}'


# ── 4. the single-pair float64 check ─────────────────────────────────────────────────


def test_check_collision_does_not_round_a_sub_step_gap_into_contact():
    # Chosen test inputs near x = 1865 m, float32 as stored; not from a named scenario.
    gap = float(X_B) - float(LEN_B) / 2 - (float(X_A) + float(LEN_A) / 2)
    assert 0.0 < gap < float(np.spacing(F32(float(X_A) + float(LEN_A) / 2)))
    assert not check_collision(
        X_A, Y, F32(0.0), LEN_A, WID_A,
        X_B, Y, F32(0.0), LEN_B, WID_B,
    )


# ── 5. get_corners ───────────────────────────────────────────────────────────────────


def _reference_corners(x, y, theta, length, width):
    """Independent pure-python statement of the documented corner order, in float64."""
    half_l, half_w = length / 2.0, width / 2.0
    c, s = math.cos(theta), math.sin(theta)
    local = [(half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)]
    return np.array([[x + c * px - s * py, y + s * px + c * py] for px, py in local])


def test_get_corners_returns_float64_for_float32_inputs():
    corners = get_corners(F32(1865.1234), F32(-3.25), F32(0.7), F32(4.9857497), F32(2.5677965))
    assert corners.dtype == np.float64
    assert corners.shape == (4, 2)


def test_get_corners_returns_float64_for_python_floats():
    # Passes on main on purpose: python floats were already float64 there. Guards the dtype
    # for callers that never hand it a NumPy scalar.
    assert get_corners(1865.1234, -3.25, 0.7, 4.9857497, 2.5677965).dtype == np.float64


@pytest.mark.parametrize('theta', [0.0, 0.7, -2.4, 3.1])
def test_get_corners_matches_an_independent_float64_computation_at_large_coordinates(theta):
    x, y = F32(1865.1234), F32(-12583.25)
    theta32, length, width = F32(theta), F32(4.9857497), F32(2.5677965)

    got = get_corners(x, y, theta32, length, width)
    want = _reference_corners(float(x), float(y), float(theta32), float(length), float(width))

    assert np.max(np.abs(got - want)) < 1e-9


@pytest.mark.parametrize('theta', [0.0, 0.7, -2.4])
def test_get_corners_keeps_full_float64_precision_for_float64_inputs(theta):
    # Passes on main on purpose: python floats were never narrowed there. Guards the other
    # direction, a float32 dtype forced onto the local corners or the translation, which is
    # invisible for float32-valued inputs (their halves and sums are exactly representable)
    # and costs about 1e-7 m on anything finer.
    x, y, length, width = 1865.123456789, -12583.987654321, 4.123456789012, 2.098765432101

    got = get_corners(x, y, theta, length, width)
    want = _reference_corners(x, y, theta, length, width)

    assert np.max(np.abs(got - want)) < 1e-9


# ── 6. the version constant ──────────────────────────────────────────────────────────


def test_the_collision_geometry_version_is_defined():
    from src.danger import collision_detector

    assert getattr(collision_detector, 'COLLISION_GEOMETRY_VERSION', None) == 'oriented-box-float64-v1'
