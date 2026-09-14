"""
test_danger_contract.py — Batch 4 regression tests (audit B06, B07).

One invariant, and it is what makes these two findings one batch:

    A danger signal reports what the trajectories actually do, not what a
    simplifying model extrapolates them to do.

B06 extrapolated OCCUPANCY — two separate visits to a conflict zone became one
continuous span. B07 extrapolated MOTION — a curved relative approach became a
radial one. Both manufacture false alarms, and neither can be caught by the
fixtures that were already in the suite.

Pure numpy/shapely, no database. Run:
    ./venv/bin/python -m pytest tests/test_danger_contract.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.danger.pet_engine import (
    DT, PET_INFINITY, _occupancy_visits, compute_pet_pair, get_path_polygon,
)
from src.danger.ttc_engine import (
    TTC_INFINITY, compute_effective_radius, compute_min_ttc_sdc,
    compute_ttc_all_pairs, compute_ttc_pair,
)


# ── helpers ─────────────────────────────────────────────────────────────────────

def _stack(occupied_frames, T):
    """
    A two-agent scene in which every agent sits at the SAME place with the same box,
    so their swept paths coincide and the conflict zone is that box. Occupancy is
    then controlled purely through validity, which is how the audit's own B06 fixture
    builds its visits — and the only way to place visits at exact frames without
    also having to reason about polygon geometry.
    """
    n = len(occupied_frames)
    states = np.zeros((n, T, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    validity = np.zeros((n, T), dtype=bool)
    for i, frames in enumerate(occupied_frames):
        validity[i, list(frames)] = True
    return states, validity


# ── B06: separate visits stay separate ──────────────────────────────────────────

def test_the_audit_repro_reports_a_positive_gap():
    """
    The audit's own scene, restated here so this file documents the defect it fixes.
    A occupies 0,1,5,6; B occupies only 3. No frame has both present, yet the merged
    spans (0,6) and (3,3) overlap and the old code returned -0.3 s.
    """
    states, validity = _stack([[0, 1, 5, 6], [3]], T=7)
    assert not (validity[0] & validity[1]).any(), 'fixture must share no frame'
    assert compute_pet_pair(states, validity, 0, 1) == pytest.approx(0.2)


def test_a_visit_is_broken_by_a_validity_gap_not_only_by_leaving():
    """
    THE decision this fix rests on, pinned so it cannot be quietly optimized away.

    A visit ends at an UNOBSERVED frame, not only at a frame where the agent has
    moved out of the zone. Bridging invalid frames would reintroduce exactly the
    fictitious continuity B06 is about, sourced from missing data instead of real
    departure — and it fails the audit's own fixture, whose two visits are created
    by a validity gap rather than by any motion.
    """
    states, validity = _stack([[0, 1, 5, 6], [3]], T=7)
    zone = get_path_polygon(states, 0, validity).intersection(
        get_path_polygon(states, 1, validity))
    assert _occupancy_visits(states, validity, 0, zone) == [(0, 1), (5, 6)]
    assert _occupancy_visits(states, validity, 1, zone) == [(3, 3)]


def test_the_minimum_can_require_a_later_visit_pair():
    """
    Multi-visit pairing AND the crossing-order fix in one fixture.

    A visits (0,0) and (10,10); B visits (5,5) and (11,11). The four pairings give
    0.5, 1.1, 0.5 and 0.1 — so the answer is 0.1 and it lives in the LAST pair.
    Three independent ways to fail, which is the point of the fixture:

      * scanning only the first visit pair       -> 0.5
      * merging each agent into one span (B06)   -> -0.5
      * dropping the max() ordering fix (v3)     -> -0.5

    The middle pairing A2xB1 is one where B cleared the zone before A arrived, so
    the ordering-safe term is the one that has to be selected; that is what makes
    this fixture exercise both fixes at once rather than in sequence.
    """
    states, validity = _stack([[0, 10], [5, 11]], T=13)
    pet = compute_pet_pair(states, validity, 0, 1)
    assert pet == pytest.approx(0.1), (
        f'expected the last visit pairing to win; got {pet}'
    )
    assert compute_pet_pair(states, validity, 1, 0) == pytest.approx(0.1), (
        'visit pairing broke the a/b symmetry guarantee'
    )


def test_a_single_visit_each_reduces_to_the_previous_formula():
    """
    The generalization must be a strict superset, not a replacement. With one visit
    per agent the cross product is a single pair and the expression is bit-identical
    to `max(enter_b - exit_a, enter_a - exit_b) * DT`, so Block 3 v3's crossing-order
    fix is PRESERVED rather than re-decided.
    """
    for frames_a, frames_b in ([[0, 1, 2], [5, 6]], [[4, 5], [0, 1]],
                               [[2, 3], [2, 3]], [[0], [8]]):
        states, validity = _stack([frames_a, frames_b], T=10)
        enter_a, exit_a = min(frames_a), max(frames_a)
        enter_b, exit_b = min(frames_b), max(frames_b)
        expected = max(enter_b - exit_a, enter_a - exit_b) * DT
        assert compute_pet_pair(states, validity, 0, 1) == pytest.approx(expected), (
            f'single-visit case diverged from the v3 formula: {frames_a} {frames_b}'
        )


def test_negative_still_means_genuine_simultaneous_occupancy():
    """
    The invariant v3 established has to survive the generalization. Overlapping
    visits must still produce a negative PET — if the fix turned every negative into
    a positive it would 'pass' B06 by destroying the signal instead of correcting it.
    """
    states, validity = _stack([[0, 1, 2], [1, 2, 3]], T=6)
    assert compute_pet_pair(states, validity, 0, 1) == pytest.approx(-0.1)


def test_an_agent_that_never_reaches_the_zone_is_infinity():
    """No visits at all is still PET_INFINITY, not an empty-sequence crash."""
    states, validity = _stack([[0, 1], [2, 3]], T=6)
    zone = get_path_polygon(states, 0, validity).intersection(
        get_path_polygon(states, 1, validity))
    validity[1, :] = False
    assert _occupancy_visits(states, validity, 1, zone) == []
    assert compute_pet_pair(states, validity, 0, 1) == PET_INFINITY


# ── B07: TTC is a quadratic root ────────────────────────────────────────────────

def test_a_non_radial_miss_is_infinite():
    """
    The audit's repro. A parked at the origin, B passing 3 m to the side with the
    circles needing 2 m to touch. The linear extrapolation reported 1.762387740 s.
    """
    assert compute_ttc_pair(0, 0, 0, 0, 1, 10, 3, -5, 0, 1) == TTC_INFINITY


def test_the_head_on_case_is_where_old_and_new_agree_exactly():
    """
    Documents why tests/test_audit_core.py's control case cannot catch B07.

    With dy = 0 the relative motion is purely radial, the quadratic degenerates to
    the linear projection, and both implementations return exactly 1.6. A test that
    cannot distinguish two implementations is not evidence for either — this asserts
    the degeneracy on purpose so the next reader does not mistake that control case
    for coverage.
    """
    quadratic = compute_ttc_pair(0, 0, 0, 0, 1, 10, 0, -5, 0, 1)
    dist, safe, closing = 10.0, 2.0, 5.0
    linear = (dist - safe) / closing          # the formula B07 replaced
    assert quadratic == 1.6
    assert quadratic == linear, 'the head-on degeneracy no longer holds'


def test_a_grazing_approach_that_does_touch_is_still_finite():
    """
    The fix must not answer 'infinity' to everything non-radial. B passes 1.5 m to
    the side and the circles need 2 m, so they DO meet — and before the true closest
    approach, which is what the smaller root means.
    """
    ttc = compute_ttc_pair(0, 0, 0, 0, 1, 10, 1.5, -5, 0, 1)
    assert ttc < TTC_INFINITY
    # closest approach is at t = 2.0 s; first contact must precede it
    assert 0 < ttc < 2.0, ttc


def test_diverging_and_stationary_pairs_are_infinite():
    """The branches B07 did NOT break, asserted so the rewrite cannot regress them."""
    # moving apart
    assert compute_ttc_pair(0, 0, 0, 0, 1, 10, 0, 5, 0, 1) == TTC_INFINITY
    # zero relative velocity, separated
    assert compute_ttc_pair(0, 0, 3, 0, 1, 10, 0, 3, 0, 1) == TTC_INFINITY


def test_already_overlapping_is_zero():
    """c <= 0. Unchanged behaviour, and the branch the saturation work will build on."""
    assert compute_ttc_pair(0, 0, 0, 0, 1, 1.0, 0, -5, 0, 1) == 0.0


def test_the_scenario_level_signal_sees_the_fix():
    """
    B07 has to reach the number Phase 5 ranks on, not just the pair function.

    The SDC drives east along y = 0; a second agent crosses well to the north on a
    path that never brings the boxes within a safe distance. The linear formula
    reported a finite TTC for this the entire time they were closing.
    """
    T = 91
    t = np.arange(T) * 0.1
    states = np.zeros((2, T, 7), dtype=np.float32)
    states[0, :, 0] = -20.0 + 10.0 * t
    states[0, :, 2] = 10.0
    states[1, :, 0] = 60.0 - 10.0 * t
    states[1, :, 1] = 25.0
    states[1, :, 2] = -10.0
    states[:, :, 5:7] = [4.5, 2.0]
    validity = np.ones((2, T), dtype=bool)
    assert compute_min_ttc_sdc(states, validity, 0) == TTC_INFINITY


# ── B07: the two implementations must not drift ─────────────────────────────────

def _scalar_from_states(states, i, j, t):
    """compute_ttc_pair called on the same two agents the matrix entry [i, j] covers."""
    def radius(k):
        return compute_effective_radius(float(states[k, t, 5]), float(states[k, t, 6]))
    return compute_ttc_pair(
        float(states[j, t, 0]), float(states[j, t, 1]),
        float(states[j, t, 2]), float(states[j, t, 3]), radius(j),
        float(states[i, t, 0]), float(states[i, t, 1]),
        float(states[i, t, 2]), float(states[i, t, 3]), radius(i),
    )


def test_scalar_and_vectorized_agree_exactly():
    """
    EXACT equality, not allclose — the same discipline as Batch 1's NumPy/Torch
    parity gate. Two implementations of one formula that agree only to a tolerance
    have a bug in one of them that the other does not, which is worse than either
    alone because it hides behind the tolerance.

    Exactness here is not free, and it is why ttc_engine spells every square `x * x`:
    Python's `float ** 2` dispatches to the C library's pow(), NumPy rewrites
    `arr ** 2` as `arr * arr`, and the two disagree in the last bit for roughly 1
    value in 1150. Written identically and computed in float64 on both sides, the
    paths agree bit-for-bit. If this test ever needs a tolerance to pass, the fix is
    to find which expression stopped matching — not to add the tolerance.

    The scene deliberately contains every branch: converging pairs, a non-radial
    near-miss, an already-overlapping pair, a zero-relative-velocity pair, and an
    invalid agent.
    """
    rng = np.random.default_rng(4)
    checked = finite = 0

    for trial in range(400):
        n = int(rng.integers(4, 9))
        states = np.zeros((n, 1, 7), dtype=np.float32)
        if trial % 2 == 0:
            # A REGULAR RING, deliberately: its near-cancelling coordinate differences
            # are a rich source of values whose square rounds differently under pow()
            # than under multiplication, which is what gives this fixture teeth on the
            # `x * x` spelling. WHICH values those are is a property of the host's
            # libm, so this is one contributor to the sweep and not a guarantee on any
            # particular machine — the breadth below is what actually does the work.
            angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
            radius = np.full(n, 40.0)
        else:
            angle = rng.uniform(0, 2 * np.pi, n)
            radius = rng.uniform(15, 60, n)
        states[:, 0, 0] = radius * np.cos(angle)
        states[:, 0, 1] = radius * np.sin(angle)
        speed = rng.uniform(3, 14, n)
        # a jittered inward heading: mostly converging, some genuinely missing
        aim = angle + rng.uniform(-0.45, 0.45, n)
        states[:, 0, 2] = -speed * np.cos(aim)
        states[:, 0, 3] = -speed * np.sin(aim)
        states[:, 0, 5] = rng.uniform(2, 6, n)
        states[:, 0, 6] = rng.uniform(1, 3, n)

        # the audit's exact non-radial miss geometry, planted in every scene
        states[0, 0, :4] = [0.0, 0.0, 0.0, 0.0]
        states[1, 0, :4] = [10.0, 3.0, -5.0, 0.0]
        states[0, 0, 5:7] = states[1, 0, 5:7] = np.float32(2.0 / np.sqrt(2.0))
        # an already-overlapping pair
        states[2, 0, :4] = [100.0, 100.0, 1.0, 0.0]
        states[3, 0, :4] = [100.5, 100.0, -1.0, 0.0]

        validity = np.ones((n, 1), dtype=bool)
        if trial % 5 == 0:
            validity[int(rng.integers(0, n)), 0] = False

        matrix = compute_ttc_all_pairs(states, validity, 0)
        assert matrix.shape == (n, n)

        for i in range(n):
            assert matrix[i, i] == TTC_INFINITY, 'diagonal must be infinity'
            for j in range(n):
                if i == j:
                    continue
                if not (validity[i, 0] and validity[j, 0]):
                    assert matrix[i, j] == TTC_INFINITY, 'invalid pair must be infinity'
                    continue
                scalar = _scalar_from_states(states, i, j, 0)
                checked += 1
                if 0 < scalar < TTC_INFINITY:
                    finite += 1
                assert scalar == matrix[i, j], (
                    f'trial {trial} pair ({i},{j}): scalar {scalar!r} != '
                    f'matrix {matrix[i, j]!r}'
                )

    # A parity test over pairs that all came back infinite would pass vacuously.
    assert finite > 100, f'fixture stopped exercising the collision branch: {finite}'

    # The sweep is this wide ON PURPOSE. A spelling divergence between `x * x` and
    # `x ** 2` reaches the FINAL returned value for only about 1 pair in 1600 — most
    # last-bit differences in `a` or `c` are rounded away again by the sqrt and the
    # division. At a few hundred pairs this test passed happily with the two paths
    # spelled differently, which made its exactness claim decorative. Measured: this
    # fixture fails under that mutation, and a much smaller one does not.
    assert checked > 10000, checked


def test_the_matrix_is_symmetric():
    """
    Entry [i, j] labels i relative to j, the opposite of compute_ttc_pair's
    convention. Negating d and v together leaves every term unchanged and does so
    EXACTLY — (-x)*(-y) == x*y bit-for-bit — so the matrix must be exactly
    symmetric, not symmetric to a tolerance.
    """
    rng = np.random.default_rng(11)
    n = 6
    states = np.zeros((n, 1, 7), dtype=np.float32)
    states[:, 0, :4] = rng.uniform(-30, 30, (n, 4))
    states[:, 0, 5] = 4.5
    states[:, 0, 6] = 2.0
    matrix = compute_ttc_all_pairs(states, np.ones((n, 1), dtype=bool), 0)
    assert np.array_equal(matrix, matrix.T)


def test_the_x_times_x_spelling_is_the_one_that_matches():
    """
    Guards the reason ttc_engine writes `x * x` everywhere instead of `x ** 2`.

    Python's `float ** 2` goes through the platform C library's pow(); NumPy lowers
    `arr ** 2` to `arr * arr`. Where pow() is not correctly rounded the two differ in
    the last bit, so mixing the spellings across the scalar and vectorized paths makes
    exact parity impossible.

    The subtlety, which cost a mutation check to find: a float32 input squared is
    EXACT in float64 (a 24-bit mantissa squares into 48 bits, which fits in 53), so
    pow and multiply agree on it and no divergence is observable. What diverges is the
    square of a DIFFERENCE of two float32 values, which can need far more than 24
    bits. That is exactly what the engines compute — dx, dy, dvx, dvy are all
    differences.

    SEARCHES for a divergent value rather than hardcoding one. An earlier version
    asserted that one specific ring-geometry float diverged; that value diverges under
    macOS libm and NOT under the reviewer's glibc, so the test failed on a correct
    codebase on a second machine. Which exact values round differently is a property
    of the host's pow(), not of this project, and a test that encodes one machine's
    libm is noise in every other environment.

    If no candidate diverges, this platform's pow() is correctly rounded for these
    inputs, the hazard does not arise here, and the test SKIPS rather than fails —
    nothing is wrong in that case. The invariant itself is guarded by
    test_scalar_and_vectorized_agree_exactly regardless; this test only explains why
    the spelling was chosen, and verifies the explanation still applies here.
    """
    rng = np.random.default_rng(0)
    divergent = None
    for _ in range(20000):
        pair = np.array(rng.uniform(-80, 80, 2), dtype=np.float32)
        x = float(pair[0]) - float(pair[1])
        if x * x != x ** 2:
            divergent = x
            break

    if divergent is None:
        pytest.skip(
            "this platform's pow() agrees with multiplication on every candidate, so "
            "the x*x/x**2 hazard does not arise here; parity is still guarded by "
            'test_scalar_and_vectorized_agree_exactly'
        )

    # NumPy lowers arr ** 2 to arr * arr, so the array path produces the MULTIPLY
    # result. Python's ** produces the pow() result. `x * x` is therefore the only
    # spelling under which the scalar function agrees with the matrix one.
    numpy_square = float((np.array([divergent], dtype=np.float64) ** 2)[0])
    assert divergent * divergent == numpy_square, (
        'x * x no longer matches NumPy; the engines are spelled the wrong way'
    )
    assert divergent ** 2 != numpy_square, (
        'x ** 2 now matches NumPy on a value where it diverges from multiplication, '
        'which is self-contradictory — investigate before trusting this suite'
    )
