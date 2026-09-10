"""
test_danger.py — Phase 3 regression tests.

Pure numpy/shapely, no database — runnable anywhere, unlike test_api.py which
needs PostGIS. Each test encodes a specific bug the Colab validation run on real
WOMD data surfaced, and each would fail against the pre-rework code.

Run:
    ./venv/bin/python tests/test_danger.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.danger.pet_engine import compute_pet_pair, PET_INFINITY
from src.danger.danger_score import score_scenario, compute_danger_score, TTC_FLOOR


T = 91           # WOMD scenario length
DT = 0.1         # seconds per timestep
SPEED = 10.0     # m/s
CAR_L, CAR_W = 4.5, 2.0


def _crossing(challenger_y0):
    """
    Two vehicles on a perpendicular collision course through the origin.

    Agent 0 drives east along y=0, starting at x=-20, so it reaches the origin
    at t=2.0s. Agent 1 drives north through x=0, starting at y=`challenger_y0`.
    Lowering `challenger_y0` makes the challenger arrive at the origin earlier.
    """
    t = np.arange(T) * DT
    s = np.zeros((2, T, 7), dtype=np.float32)

    s[0, :, 0] = -20.0 + SPEED * t
    s[0, :, 1] = 0.0
    s[0, :, 2] = SPEED
    s[0, :, 4] = 0.0

    s[1, :, 0] = 0.0
    s[1, :, 1] = challenger_y0 + SPEED * t
    s[1, :, 3] = SPEED
    s[1, :, 4] = np.pi / 2

    s[:, :, 5] = CAR_L
    s[:, :, 6] = CAR_W
    v = np.ones((2, T), dtype=bool)
    return s, v


def check(label, condition, detail=""):
    if not condition:
        raise AssertionError(f"{label} FAILED {detail}")
    print(f"  ok  {label}")


def test_pet_ordering_challenger_crosses_first():
    """
    Finding 5: compute_pet_pair used to evaluate only `enter_b - exit_a`, so when
    agent b cleared the conflict zone before agent a arrived, it returned a
    spurious NEGATIVE PET — indistinguishable from a genuine collision — for a
    scenario that was actually safely sequenced.

    Here the challenger (agent 1) starts at y0=-5, clearing the origin at ~0.9s,
    while the SDC (agent 0) does not reach it until ~2.0s. A gap of ~0.9s. The
    old code reported -2.1s; the corrected `max(enter_b - exit_a, enter_a - exit_b)`
    reports +0.9s.
    """
    s, v = _crossing(challenger_y0=-5.0)
    pet = compute_pet_pair(s, v, 0, 1)
    check("challenger-crosses-first PET is positive (was -2.1s under the old formula)",
          pet > 0, f"got {pet:.3f}s")
    check("challenger-crosses-first PET is a plausible near-miss gap",
          0.3 < pet < 2.0, f"got {pet:.3f}s")


def test_pet_symmetry():
    """
    compute_min_pet_scenario only ever calls compute_pet_pair with i < j. That is
    safe only if the function is symmetric — swapping a and b swaps the two max
    operands, so the result must be identical either way.
    """
    for y0 in (-5.0, -33.0, -50.0):
        s, v = _crossing(challenger_y0=y0)
        pet_ab = compute_pet_pair(s, v, 0, 1)
        pet_ba = compute_pet_pair(s, v, 1, 0)
        check(f"compute_pet_pair symmetric at challenger_y0={y0}",
              pet_ab == pet_ba, f"{pet_ab!r} != {pet_ba!r}")


def test_pet_negative_only_on_genuine_overlap():
    """
    After the fix, a negative PET is returned only when both agents genuinely
    occupy the conflict zone at overlapping times. Two vehicles starting
    equidistant from the origin at the same speed arrive simultaneously.
    """
    s, v = _crossing(challenger_y0=-20.0)   # mirror of the SDC's -20 start
    pet = compute_pet_pair(s, v, 0, 1)
    check("simultaneous arrival yields PET <= 0 (genuine overlap)",
          pet <= 0, f"got {pet:.3f}s")


def test_pet_no_spatial_overlap_is_infinity():
    """Parallel, non-crossing paths never share a conflict zone."""
    t = np.arange(T) * DT
    s = np.zeros((2, T, 7), dtype=np.float32)
    s[0, :, 0] = -20.0 + SPEED * t
    s[0, :, 1] = 0.0
    s[0, :, 2] = SPEED
    s[1, :, 0] = -20.0 + SPEED * t
    s[1, :, 1] = 50.0                        # 50m to the side, never crosses
    s[1, :, 2] = SPEED
    s[:, :, 5] = CAR_L
    s[:, :, 6] = CAR_W
    v = np.ones((2, T), dtype=bool)
    pet = compute_pet_pair(s, v, 0, 1)
    check("non-crossing paths return PET_INFINITY",
          pet == PET_INFINITY, f"got {pet}")


def test_rank_divergence_sdc_vs_all_pairs():
    """
    Finding 1: score_scenario had no sdc_index, so it minimised TTC/PET over ALL
    agent pairs. On real WOMD data (~57 agents/scene) that saturated min_ttc to
    0.0 for 100/100 scenarios — any two agents passing close pin it, regardless of
    whether the SDC was ever at risk. All-pairs vs SDC-only ranking on the same 50
    scenarios gave Spearman rho = -0.0152: the cheap filter was uncorrelated with
    what Phase 4 optimises.

    Fixture: the SDC drives east along y=0 from x=-20, alone. Two parked vehicles
    sit 4.5 m apart at y=200 — inside the ~4.92 m circumscribed safe_dist for two
    4.5x2.0 boxes, so the non-SDC pair drives all-pairs TTC to exactly 0.0. The
    SDC stays ~200 m from everything.

    Pre-verified against the engines:
        all-pairs: min_ttc=0.0     min_pet=-9.0   -> compute_danger_score = 100.0
        SDC-only:  min_ttc=161.67  min_pet=999.0  -> compute_danger_score = 0.0035
    """
    t = np.arange(T) * DT
    s = np.zeros((3, T, 7), dtype=np.float32)
    # agent 0 = SDC, east along y=0 from x=-20
    s[0, :, 0] = -20.0 + SPEED * t
    s[0, :, 1] = 0.0
    s[0, :, 2] = SPEED
    s[0, :, 4] = 0.0
    # agents 1, 2 = parked 4.5 m apart at y=200, far from the SDC
    s[1, :, 0] = 0.0
    s[1, :, 1] = 200.0
    s[2, :, 0] = 4.5
    s[2, :, 1] = 200.0
    s[:, :, 5] = CAR_L
    s[:, :, 6] = CAR_W
    v = np.ones((3, T), dtype=bool)

    rec = score_scenario(s, v, "syn_rankdiv", sdc_index=0)

    check("all-pairs TTC saturates (<= TTC_FLOOR) on the parked non-SDC pair",
          rec['min_ttc_all_pairs'] <= TTC_FLOOR,
          f"got {rec['min_ttc_all_pairs']}")
    check("SDC-restricted TTC stays far above the floor",
          rec['min_ttc'] > 100.0, f"got {rec['min_ttc']}")
    check("fragility_score (SDC-restricted) is LOW — correctly ranks this as safe",
          rec['fragility_score'] < 0.1, f"got {rec['fragility_score']}")

    # the divergence is the point: feed the all-pairs values through the SAME
    # scoring function and it screams danger for a scenario where the SDC is 200 m
    # from the nearest agent.
    old_score = compute_danger_score(rec['min_ttc_all_pairs'],
                                     rec['min_pet_all_pairs'])
    check("all-pairs scoring would have ranked this scenario MAX danger",
          old_score > 50.0, f"got {old_score}")
    check("SDC-aware scoring is >1000x lower than the all-pairs score it replaced",
          old_score / max(rec['fragility_score'], 1e-9) > 1000,
          f"ratio {old_score / max(rec['fragility_score'], 1e-9):.0f}")


def test_score_scenario_requires_sdc_index():
    """A missing sdc_index must be a loud TypeError, not a silent fallback."""
    s, v = _crossing(challenger_y0=-33.0)
    try:
        score_scenario(s, v, "syn_no_sdc")   # type: ignore  — intentionally wrong
    except TypeError:
        print("  ok  score_scenario without sdc_index raises TypeError")
        return
    raise AssertionError("score_scenario accepted a call with no sdc_index")


def main():
    print("Phase 3 danger-engine regression tests")
    print("=" * 60)
    print("\nFix 1 — PET ordering (finding 5)")
    test_pet_ordering_challenger_crosses_first()
    test_pet_symmetry()
    test_pet_negative_only_on_genuine_overlap()
    test_pet_no_spatial_overlap_is_infinity()
    print("\nFix 2 — SDC-aware scoring (finding 1)")
    test_rank_divergence_sdc_vs_all_pairs()
    test_score_scenario_requires_sdc_index()
    print("\n" + "=" * 60)
    print("ALL PHASE 3 REGRESSION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
