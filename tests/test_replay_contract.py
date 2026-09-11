"""
test_replay_contract.py — Batch 1 regression tests (audit B01, B02, B03, B08).

The contract this file pins down:

    For any challenger — any agent type, any validity pattern — apply(zeros) must
    reproduce the logged trajectory over that challenger's valid timesteps, or the
    scenario must be refused out loud; every returned state must respect the declared
    kinematic bounds at every step INCLUDING the first; and all of it must hold
    identically in the numpy path and the differentiable torch path.

The audit's own fixtures (tests/test_audit_core.py) cover the four findings on the
numpy side. These add what they cannot: the turning case, the numpy/torch parity that
keeps DE and the autograd refiner solving the same problem, and the fidelity gate.

Run:
    ./venv/bin/python -m pytest tests/test_replay_contract.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimization.perturbation_space import (
    BASELINE_DRIFT_REFUSE_M, PerturbationSpace, ReplayFidelityError,
)
from src.physics.bicycle_model import V_MAX
from src.physics.linear_model import A_MAX as LINEAR_A_MAX, linear_step

DT = 0.1
CAR_L, CAR_W = 4.5, 2.0

# Drift tolerance for fixtures that TURN, and why it is not tighter. A discrete step
# advances along the CHORD of the arc, length 2R*sin(w*dt/2), while the model advances
# v*dt = R*w*dt along the arc. That leaves a relative shortfall of (w*dt/2)^2/6 = 1.7e-5
# per step, which over the ~90 m of path in these fixtures accumulates to ~1.5e-3 m. It
# is an inherent property of integrating a curve at 10 Hz, not a defect, and no choice
# of midpoint removes it. Measured: 1.33e-3 m over the full 91 frames.
#
# 5e-3 sits above that residual and 100x below the 0.5 m fidelity gate, so it pins the
# fix (the pre-fix integrator drifts 0.783 m here, 150x this bound) without pinning
# float noise. Shared by every turning fixture so the bound is chosen once rather than
# re-guessed per test against whatever margin that test happens to have.
TURN_TOLERANCE_M = 5e-3


def _scene(n=2, t=91):
    """n agents, all vehicles, all frames valid, with real dimensions."""
    states = np.zeros((n, t, 7), dtype=np.float32)
    states[:, :, 5] = CAR_L
    states[:, :, 6] = CAR_W
    return states, np.ones((n, t), dtype=bool), np.ones(n, dtype=int)


def _park_sdc_far_away(states):
    """Keep the SDC 1 km off so no fixture trips the collision gate by accident."""
    states[0, :, 1] = 1000.0


def _turning_track(states, idx, v=10.0, omega=0.2, x0=0.0, y0=0.0):
    """
    Write a constant-speed, constant-turn-rate arc — a circle of radius v/omega.

    This is the case the audit never exercised. Every audit fixture drives in a
    straight line, so a position update that takes its direction from the PRE-step
    heading looks correct in all of them while drifting 0.78 m over 9 s here.
    """
    t = np.arange(states.shape[1]) * DT
    theta = omega * t
    r = v / omega
    states[idx, :, 0] = x0 + r * np.sin(theta)
    states[idx, :, 1] = y0 + r * (1.0 - np.cos(theta))
    states[idx, :, 2] = v * np.cos(theta)
    states[idx, :, 3] = v * np.sin(theta)
    states[idx, :, 4] = theta
    return states


# ── replay fidelity ──────────────────────────────────────────────────────────────

def test_turning_track_replays_without_drift():
    """
    A turning challenger must replay onto its own logged arc at zero delta.

    Measured against the pre-fix integrator this drifts 0.783 m — larger than the
    0.45 m straight-line drift that was already enough to manufacture a collision in
    audit B03, and invisible to every straight-line fixture in the audit.
    """
    s, v, types = _scene()
    _park_sdc_far_away(s)
    _turning_track(s, 1)

    space = PerturbationSpace(s, v, types, 0, 1)
    replay = space.apply(np.zeros(4))

    drift = np.linalg.norm(replay[1, :, :2] - s[1, :, :2], axis=1).max()
    assert drift <= TURN_TOLERANCE_M, (
        f"turning replay drifted {drift:.6f} m at zero perturbation"
    )
    assert space.baseline_replay_error <= TURN_TOLERANCE_M


def test_accelerating_track_replays_without_drift():
    """The audit's own B03 geometry, asserted on drift rather than only on collision."""
    s, v, types = _scene()
    _park_sdc_far_away(s)
    t = np.arange(91) * DT
    s[1, :, 0] = 0.5 * t * t
    s[1, :, 2] = t

    space = PerturbationSpace(s, v, types, 0, 1)
    drift = np.abs(space.apply(np.zeros(4))[1, :, 0] - s[1, :, 0]).max()
    assert drift <= 1e-3, f"accelerating replay drifted {drift:.6f} m"


def test_late_start_replays_on_its_own_frames():
    """
    A challenger first seen at frame 40 must replay onto frames 40..90, and its
    unobserved past must be left exactly as logged rather than overwritten with a
    simulated history that never happened.
    """
    s, v, types = _scene()
    _park_sdc_far_away(s)
    _turning_track(s, 1)
    logged_before = s[1, :40].copy()
    v[1, :40] = False

    space = PerturbationSpace(s, v, types, 0, 1)
    assert space.t0 == 40

    replay = space.apply(np.zeros(4))
    # This fixture TURNS, so it carries the same chord-vs-arc residual as the one
    # above and must share its bound. It previously asserted 1e-3 against a measured
    # 8.04e-4 — a 1.24x margin that would have failed on any change to t0, for a
    # reason that is not a bug.
    drift = np.linalg.norm(replay[1, 40:, :2] - s[1, 40:, :2], axis=1).max()
    assert drift <= TURN_TOLERANCE_M, f"late-start replay drifted {drift:.6f} m"
    np.testing.assert_array_equal(replay[1, :40], logged_before)


# ── numpy / torch parity — the batch's completion gate ───────────────────────────

@pytest.mark.parametrize('delta', [
    np.zeros(4),
    np.array([3.0, 0.20, 1.5, 0.10]),      # a corner of the default bounds
    np.array([-3.0, -0.20, -1.5, -0.10]),  # the opposite corner
], ids=['zeros', 'upper_corner', 'lower_corner'])
def test_numpy_and_torch_rollouts_agree(delta):
    """
    The two rollouts must produce the same trajectory, step for step.

    If they diverge, Differential Evolution and the autograd refiner are solving
    different problems while reporting into one number — an already-documented
    failure mode (Block 4 §15's 0.123-vs-0.184 overshoot). Every audit fixture
    exercises only the numpy path, so a numpy-only fix passes the whole audit suite
    while making this worse; this test is the thing that would catch that.

    The fixture is deliberately awkward: a late start, a turn, and acceleration at
    once, so an index-alignment error cannot hide behind a straight line starting at
    frame 0.
    """
    torch = pytest.importorskip('torch')
    from src.optimization.autograd_optimizer import _DiffBicycleRollout

    s, v, types = _scene()
    _park_sdc_far_away(s)
    _turning_track(s, 1)
    # layer acceleration on top of the arc
    t = np.arange(91) * DT
    s[1, :, 2] += 0.5 * t * np.cos(0.2 * t)
    s[1, :, 3] += 0.5 * t * np.sin(0.2 * t)
    v[1, :7] = False

    space = PerturbationSpace(s, v, types, 0, 1, max_baseline_drift=None)
    roll = _DiffBicycleRollout(space)

    numpy_states = space.apply(delta.astype(np.float32))
    torch_traj = roll.rollout(torch.tensor(delta, dtype=torch.float64)).detach().numpy()

    assert torch_traj.shape[0] == 91 - space.t0, "torch rollout is not aligned to t0"

    frames = np.arange(space.t0, 91)
    np.testing.assert_allclose(numpy_states[1, frames, 0], torch_traj[:, 0], atol=1e-4)
    np.testing.assert_allclose(numpy_states[1, frames, 1], torch_traj[:, 1], atol=1e-4)
    np.testing.assert_allclose(numpy_states[1, frames, 4], torch_traj[:, 2], atol=1e-4)
    speed = np.hypot(numpy_states[1, frames, 2], numpy_states[1, frames, 3])
    np.testing.assert_allclose(speed, torch_traj[:, 3], atol=1e-4)


def test_torch_margin_reads_the_sdc_at_global_frames():
    """
    The smooth margin must compare rollout row k against SDC frame t0 + k.

    Indexing the logged SDC with the rollout row index is only correct when t0 == 0.
    With a late-starting challenger the margin would otherwise be evaluated against
    an SDC that is t0 frames behind where it actually was.
    """
    torch = pytest.importorskip('torch')
    from src.optimization.autograd_optimizer import _DiffBicycleRollout, _smooth_margin

    s, v, types = _scene(t=40)
    # The SDC drives east past a parked challenger. Its position therefore differs
    # sharply between frame k and frame t0 + k, so reading it at the wrong index
    # changes the margin by metres rather than by rounding.
    s[0, :, 0] = -60.0 + 10.0 * np.arange(40) * DT
    s[0, :, 2] = 10.0
    s[1, :, 0] = 0.0
    s[1, :, 1] = 6.0
    v[1, :20] = False

    space = PerturbationSpace(s, v, types, 0, 1)
    assert space.t0 == 20
    beta = 6.0
    traj = _DiffBicycleRollout(space).rollout(torch.zeros(4, dtype=torch.float64))
    margin = _smooth_margin(space, traj, beta, 1, torch.float64).item()

    # Reference, computed from scratch. n_circles=1 makes each agent a single circle
    # of radius width/2 at its own centre, so the per-frame gap is exact and the only
    # thing left under test is WHICH SDC frame each rollout row is compared against.
    def reference(sdc_frames):
        radii = CAR_W / 2.0 + CAR_W / 2.0
        gaps = np.linalg.norm(
            s[0, sdc_frames, :2] - traj.detach().numpy()[:, :2], axis=1) - radii
        return float(-np.log(np.exp(-beta * gaps).sum()) / beta)

    rollout_rows = np.arange(traj.shape[0])
    aligned = reference(rollout_rows + space.t0)   # correct: row k -> frame t0 + k
    misaligned = reference(rollout_rows)           # the bug: row k -> frame k

    assert margin == pytest.approx(aligned, abs=1e-6), (
        f"smooth margin {margin:.6f} != aligned reference {aligned:.6f}"
    )
    assert abs(aligned - misaligned) > 1.0, "fixture too weak to detect misalignment"


# ── bounds hold at every step, including the first ───────────────────────────────

@pytest.mark.parametrize('base_speed, dv0', [(0.5, -3.0), (39.0, 3.0), (0.0, -3.0)])
def test_vehicle_initial_speed_is_inside_the_bounds(base_speed, dv0):
    s, v, types = _scene(t=30)
    _park_sdc_far_away(s)
    s[1, :, 0] = base_speed * np.arange(30) * DT
    s[1, :, 2] = base_speed

    space = PerturbationSpace(s, v, types, 0, 1)
    replay = space.apply(np.array([dv0, 0.0, 0.0, 0.0], dtype=np.float32))
    speeds = np.hypot(replay[1, :, 2], replay[1, :, 3])
    assert speeds.min() >= -1e-6 and speeds.max() <= V_MAX + 1e-6, (
        f"speeds ran outside [0, {V_MAX}]: min={speeds.min()}, max={speeds.max()}"
    )


def test_torch_initial_speed_is_clamped_like_numpy():
    torch = pytest.importorskip('torch')
    from src.optimization.autograd_optimizer import _DiffBicycleRollout

    s, v, types = _scene(t=30)
    _park_sdc_far_away(s)
    s[1, :, 0] = 39.0 * np.arange(30) * DT
    s[1, :, 2] = 39.0

    space = PerturbationSpace(s, v, types, 0, 1)
    traj = _DiffBicycleRollout(space).rollout(
        torch.tensor([3.0, 0.0, 0.0, 0.0], dtype=torch.float64))
    assert float(traj[0, 3]) <= V_MAX + 1e-9


def test_pedestrian_initial_velocity_is_clamped_by_magnitude():
    """The linear model's V_MAX is a magnitude bound, so the clamp must be too."""
    s, v, types = _scene(t=30)
    _park_sdc_far_away(s)
    types[1] = 2  # pedestrian -> linear model
    s[1, :, 5], s[1, :, 6] = 0.5, 0.5
    s[1, :, 0] = 39.0 * np.arange(30) * DT
    s[1, :, 2] = 39.0

    space = PerturbationSpace(s, v, types, 0, 1, max_baseline_drift=None)
    replay = space.apply(np.array([2.0, 2.0, 0.0, 0.0], dtype=np.float32))
    speeds = np.hypot(replay[1, :, 2], replay[1, :, 3])
    assert speeds.max() <= V_MAX + 1e-4, f"linear speed reached {speeds.max()}"


def test_linear_acceleration_is_bounded_by_magnitude_not_per_component():
    result = linear_step(np.zeros(4), np.array([LINEAR_A_MAX, LINEAR_A_MAX]))
    magnitude = float(np.linalg.norm(result[2:]) / DT)
    assert magnitude <= LINEAR_A_MAX + 1e-4


# ── the fidelity gate ────────────────────────────────────────────────────────────

def _colliding_baseline_scene():
    """
    The audit's B03 geometry, but with the drift re-introduced deliberately: two
    4.5 m vehicles nose-to-tail with a gap far smaller than the replay error the
    scene's inconsistency forces, so the zero-delta replay overlaps the SDC.
    """
    s, v, types = _scene(t=30)
    s[0, :, 0] = 0.0          # SDC parked at the origin
    s[1, :, 0] = 5.0          # challenger parked 5 m ahead of it...
    s[1, :, 4] = np.pi        # ...but logged as FACING the SDC at 20 m/s.
    s[1, :, 2] = -20.0        # vx: heading pi at 20 m/s
    # Logged, the boxes never touch: two 4.5 m vehicles 5 m apart centre to centre.
    # Replayed, the challenger's own logged velocity drives it straight through the
    # SDC within two steps -- a collision produced entirely by replay, with no
    # perturbation applied. That is audit B03's failure mode in its purest form.
    return s, v, types


def test_hard_gate_refuses_a_baseline_that_already_collides():
    s, v, types = _colliding_baseline_scene()
    with pytest.raises(ReplayFidelityError) as excinfo:
        PerturbationSpace(s, v, types, 0, 1, max_baseline_drift=None)

    err = excinfo.value
    assert err.reason == 'collision'
    assert err.baseline_replay_collides is True
    assert np.isfinite(err.baseline_replay_error) and err.baseline_replay_error > 0, (
        "a collision refusal must still report its measured drift"
    )


def test_soft_gate_refuses_excessive_drift_without_a_collision():
    s, v, types = _scene(t=30)
    _park_sdc_far_away(s)      # no collision possible
    s[1, :, 0] = 10.0          # parked, but claiming 20 m/s -> 2 m of drift per step
    s[1, :, 2] = 20.0

    with pytest.raises(ReplayFidelityError) as excinfo:
        PerturbationSpace(s, v, types, 0, 1)

    err = excinfo.value
    assert err.reason == 'drift'
    assert err.baseline_replay_collides is False
    assert err.baseline_replay_error > BASELINE_DRIFT_REFUSE_M


def test_disabling_the_drift_gate_leaves_the_collision_gate_armed():
    """max_baseline_drift=None is for measuring the distribution, not for silencing."""
    s, v, types = _scene(t=30)
    _park_sdc_far_away(s)
    s[1, :, 0] = 10.0
    s[1, :, 2] = 20.0
    space = PerturbationSpace(s, v, types, 0, 1, max_baseline_drift=None)
    assert space.baseline_replay_error > BASELINE_DRIFT_REFUSE_M

    s2, v2, types2 = _colliding_baseline_scene()
    with pytest.raises(ReplayFidelityError) as excinfo:
        PerturbationSpace(s2, v2, types2, 0, 1, max_baseline_drift=None)
    assert excinfo.value.reason == 'collision'


def test_stress_one_reports_replay_infeasible_not_error():
    """
    The refusal must arrive as its own status. status='error' would mean the narrow
    try around the construction call is missing or too broad, and would throw away
    the structured reason the real-shard run needs.
    """
    from src.scoring.batch_scorer import _stress_one

    s, v, types = _colliding_baseline_scene()
    result = _stress_one(s, v, types, 0, de_kwargs={'popsize': 4, 'maxiter': 5, 'seed': 1})

    assert result['status'] == 'replay_infeasible', f"got {result}"
    assert result['reason'] == 'collision'
    assert result['target_idx'] == 1
    assert 'baseline_replay_error' in result


# ── fail loud on unusable geometry ───────────────────────────────────────────────

def test_never_valid_challenger_raises():
    s, v, types = _scene(t=30)
    _park_sdc_far_away(s)
    v[1, :] = False
    with pytest.raises(ValueError, match='no valid timesteps'):
        PerturbationSpace(s, v, types, 0, 1)


@pytest.mark.parametrize('col, name', [(5, 'length'), (6, 'width')])
def test_unusable_dimensions_at_the_first_valid_frame_raise(col, name):
    s, v, types = _scene(t=30)
    _park_sdc_far_away(s)
    s[1, :, col] = 0.0
    with pytest.raises(ValueError, match=name):
        PerturbationSpace(s, v, types, 0, 1)


def test_dimensions_come_from_the_first_valid_frame_not_frame_zero():
    s, v, types = _scene(t=30)
    _park_sdc_far_away(s)
    s[1, :, 0] = 10.0 * np.arange(30) * DT
    s[1, :, 2] = 10.0
    v[1, :5] = False
    s[1, :5, 5:7] = 0.0          # zero-fill in the unobserved past

    space = PerturbationSpace(s, v, types, 0, 1)
    assert space.target_len == CAR_L and space.target_wid == CAR_W
    assert space.simulator.wheelbase > 0
    assert np.isfinite(space.apply(np.zeros(4))[1, v[1]]).all()


# ── determinism ──────────────────────────────────────────────────────────────────

def test_construction_is_deterministic():
    s, v, types = _scene()
    _park_sdc_far_away(s)
    _turning_track(s, 1)
    v[1, :12] = False

    a = PerturbationSpace(s, v, types, 0, 1)
    b = PerturbationSpace(s, v, types, 0, 1)
    assert (a.t0, a.target_len, a.target_wid) == (b.t0, b.target_len, b.target_wid)
    assert a.baseline_replay_error == b.baseline_replay_error
    np.testing.assert_array_equal(a.apply(np.zeros(4)), b.apply(np.zeros(4)))


def test_apply_does_not_mutate_the_original_states():
    s, v, types = _scene()
    _park_sdc_far_away(s)
    _turning_track(s, 1)
    original = s.copy()
    space = PerturbationSpace(s, v, types, 0, 1)
    space.apply(np.array([1.0, 0.05, 0.5, 0.02], dtype=np.float32))
    np.testing.assert_array_equal(s, original)
