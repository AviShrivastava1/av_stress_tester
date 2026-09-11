"""
perturbation_space.py — Phase 4, Concept 17 (Perturbation Space Design).

Defines the space Delta that the optimizers search over. A perturbation delta is
applied to ONE challenger agent's behavior; the SDC keeps its logged trajectory.
The perturbation is injected at the *control* level (and initial state) and then
rolled forward through the Phase 2 ForwardSimulator, which guarantees every
perturbed trajectory is dynamically feasible — a car could actually have driven it.

Parameterization (4 dimensions), for a VEHICLE challenger:
    delta = [dv0, dtheta0, da_bias, ddelta_bias]
        dv0         : offset added to the challenger's initial speed        (m/s)
        dtheta0     : offset added to the challenger's initial heading      (rad)
        da_bias     : constant offset added to every acceleration control   (m/s^2)
        ddelta_bias : constant offset added to every steering control       (rad)

For a PEDESTRIAN / CYCLIST challenger (linear model), the four dimensions are:
    delta = [dvx0, dvy0, dax_bias, day_bias]

Why this parameterization (defend this in an interview):
    * Low-dimensional (4) — cheap for Differential Evolution to search.
    * Expressive — initial-condition + sustained-control changes can reach most
      realistic near-miss-to-collision transitions.
    * Smooth/realistic — constant control biases rolled through the bicycle model
      produce smooth, drivable trajectories, not per-timestep jitter.
    * The kinematic clamps inside bicycle_step / linear_step (DELTA_MAX, A_MAX,
      V_MAX) act as built-in realism guarantees even at the bound.
"""

import numpy as np

from src.data.validity import (
    first_valid_index, has_interior_gap, valid_timesteps,
)
from src.danger.collision_detector import check_collision_trajectory
from src.physics.bicycle_model import (
    V_MAX, extract_state_from_womd as bicycle_extract,
)
from src.physics.linear_model import (
    project_to_magnitude, extract_state_from_womd as linear_extract,
)
from src.physics.simulator import (
    ForwardSimulator, TrajectoryInverter, TYPE_VEHICLE,
)


# Maximum baseline (zero-delta) replay error, in metres, at which a scenario is
# still considered replayable. 0.5 m is a quarter of a standard 2.0 m vehicle
# width — the scale at which box-overlap decisions start to flip.
#
# This number is NOT yet validated against real WOMD tracks; no shard has been
# measured under the corrected integrator. Pass max_baseline_drift=None to record
# the error without refusing on it, which is how the distribution gets measured
# before this constant is defended as final.
BASELINE_DRIFT_REFUSE_M = 0.5


class ReplayFidelityError(ValueError):
    """
    Raised when a challenger's logged trajectory cannot be faithfully reproduced by
    the kinematic model, so no perturbation measured against it would mean anything.

    Carries structured fields rather than only a message, so callers can tell the
    two gates apart without parsing text:

        reason                  'collision' — the ZERO-delta replay already collides
                                  with the SDC. A "collision at ||delta|| = 0" is not
                                  a stress-test result under any reading (audit B03).
                                'drift'     — the zero-delta replay stays clear of the
                                  SDC but wanders further from the logged track than
                                  max_baseline_drift allows.
        baseline_replay_error   measured max positional error, in metres. Populated
                                  for BOTH reasons — a collision refusal still reports
                                  its drift, which is the number needed to ask whether
                                  baseline collisions cluster at high drift.
        baseline_replay_collides  whether the zero-delta replay collided.
    """

    def __init__(self, reason, baseline_replay_error, baseline_replay_collides):
        self.reason = reason
        self.baseline_replay_error = float(baseline_replay_error)
        self.baseline_replay_collides = bool(baseline_replay_collides)
        if reason == 'collision':
            detail = ("the zero-perturbation replay already collides with the SDC, "
                      "so any 'caused' collision would be an artefact of the replay")
        else:
            detail = ("the zero-perturbation replay does not reproduce the logged "
                      "track closely enough to measure a perturbation against it")
        super().__init__(
            f"replay fidelity check failed ({reason}): {detail}; "
            f"baseline_replay_error={self.baseline_replay_error:.4f} m, "
            f"baseline_replay_collides={self.baseline_replay_collides}"
        )

class PerturbationSpace:
    """
    Owns the perturbation parameterization for one (scenario, challenger) pair.

    Construct once per scenario, then call apply(delta) many times inside the
    optimizer loop. apply() is pure: it never mutates the original states.
    """

    def __init__(
        self,
        states: np.ndarray,
        validity: np.ndarray,
        types: np.ndarray,
        sdc_idx: int,
        target_idx: int,
        bounds: np.ndarray = None,
        dt: float = 0.1,
        max_baseline_drift: float = BASELINE_DRIFT_REFUSE_M,
    ):
        """
        Args:
            states:     (N, T, 7) = [x, y, vx, vy, heading, length, width]
            validity:   (N, T) bool
            types:      (N,) int — 1 vehicle, 2 pedestrian, 3 cyclist
            sdc_idx:    index of the self-driving car (kept fixed)
            target_idx: index of the challenger agent we perturb
            bounds:     (4, 2) array of (low, high) per dimension. If None, sane
                        defaults are chosen based on the challenger's agent type.
            dt:         timestep duration (seconds)
            max_baseline_drift:
                        refuse the scenario if the zero-delta replay wanders further
                        than this (metres) from the logged track. None disables the
                        check and records the error only — use that to measure the
                        real distribution, not to silence an inconvenient refusal.

        Raises:
            ValueError:            the challenger is never observed, or its recorded
                                   dimensions at its first valid frame are unusable.
            ReplayFidelityError:   the zero-delta replay is not faithful enough for a
                                   perturbation measured against it to mean anything.
        """
        self.states0  = states.astype(np.float32)
        self.validity = validity
        self.types    = types
        self.sdc_idx  = int(sdc_idx)
        self.target_idx = int(target_idx)
        self.dt = dt

        self.N, self.T, _ = states.shape
        self.target_type = int(types[target_idx])
        self.is_vehicle  = self.target_type == TYPE_VEHICLE

        # ── first valid observation, established BEFORE anything reads the array ──
        # Frame 0 of a late-appearing agent is zero-fill, not data (audit B01/B02):
        # reading dimensions there gives length 0 -> wheelbase 0 -> divide-by-zero in
        # the heading update -> NaN across the whole rollout. Both indices come from
        # the one shared helper so this module, export_geometry and the torch margin
        # cannot drift apart about what "first observation" means.
        self.t0     = first_valid_index(validity, self.target_idx)
        self.sdc_t0 = first_valid_index(validity, self.sdc_idx)
        self.has_interior_gap = has_interior_gap(validity, self.target_idx)

        # physical dimensions, read at each agent's own first valid frame
        self.target_len = self._valid_dimension(self.target_idx, self.t0, 5, 'length')
        self.target_wid = self._valid_dimension(self.target_idx, self.t0, 6, 'width')
        self.sdc_len    = self._valid_dimension(self.sdc_idx, self.sdc_t0, 5, 'length')
        self.sdc_wid    = self._valid_dimension(self.sdc_idx, self.sdc_t0, 6, 'width')

        # baseline controls recovered from the logged trajectory (Phase 2 inversion).
        # Indexed by GLOBAL frame: row t is the transition t -> t+1.
        self.inverter = TrajectoryInverter(self.target_type, self.target_len)
        self.base_controls = self.inverter.invert(
            states, self.target_idx, validity, dt
        )  # (T-1, 2)

        # forward simulator used to replay perturbed controls
        self.simulator = ForwardSimulator(self.target_type, self.target_len)

        # baseline initial state of the challenger ([x,y,theta,v] or [x,y,vx,vy]),
        # taken at t0 — which is why apply() must also START the rollout at t0.
        if self.is_vehicle:
            self.base_init = bicycle_extract(states, self.target_idx, self.t0)
        else:
            self.base_init = linear_extract(states, self.target_idx, self.t0)

        # per-dimension box bounds (used by DE and by gradient clamping)
        self.bounds = self._default_bounds() if bounds is None else np.asarray(bounds, np.float32)

        # per-dimension weights for the norm: 1 / bound_magnitude, so each term is
        # a dimensionless fraction of its allowed budget (Concept 17, weighted norm).
        self.weights = 1.0 / np.maximum(np.abs(self.bounds[:, 1]), 1e-6)

        # ── replay fidelity (audit B03) ──────────────────────────────────────────
        # Measure what a ZERO perturbation actually reproduces before anyone asks
        # this space for a minimum perturbation. Costs one rollout, against the
        # thousands DE is about to run.
        self.baseline_replay_error, self.baseline_replay_collides = \
            self._measure_baseline_replay()

        if self.baseline_replay_collides:
            raise ReplayFidelityError('collision', self.baseline_replay_error, True)
        if (max_baseline_drift is not None
                and not (self.baseline_replay_error <= max_baseline_drift)):
            # `not (<=)` rather than `>` so a non-finite error refuses too.
            raise ReplayFidelityError('drift', self.baseline_replay_error, False)

    # ── public API ──────────────────────────────────────────────────────────

    @property
    def dim(self) -> int:
        return 4

    def _default_bounds(self) -> np.ndarray:
        if self.is_vehicle:
            # [dv0 (m/s), dtheta0 (rad), da_bias (m/s^2), ddelta_bias (rad)]
            return np.array([[-3.0, 3.0],
                             [-0.20, 0.20],
                             [-1.5, 1.5],
                             [-0.10, 0.10]], dtype=np.float32)
        # linear: [dvx0, dvy0, dax_bias, day_bias]
        return np.array([[-2.0, 2.0],
                         [-2.0, 2.0],
                         [-1.0, 1.0],
                         [-1.0, 1.0]], dtype=np.float32)

    def weighted_norm(self, delta: np.ndarray) -> float:
        """
        L2 norm after normalizing each dimension by its bound, so incomparable
        units (m/s vs rad) contribute on equal footing. This is the ||delta|| that
        the optimizers minimize and that feeds Phase 3's danger score.
        """
        delta = np.asarray(delta, np.float32)
        return float(np.sqrt(np.sum((delta * self.weights) ** 2)))

    def apply(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply a perturbation and return a NEW (N, T, 7) states array in which the
        challenger's trajectory has been replaced by the re-simulated, perturbed
        one. The SDC and all other agents are untouched.

        Steps:
            1. slice the baseline controls to start at t0 and add the control biases
            2. copy the baseline initial state, add the offsets, clamp to the bounds
            3. roll forward through ForwardSimulator (enforces kinematic limits)
            4. write the new trajectory back into a copy of states, starting at t0
        """
        delta = np.asarray(delta, np.float32)

        # Controls are indexed by GLOBAL frame, and base_init was taken at t0, so the
        # rollout must start from t0's control too (audit B01). Simulating from index
        # 0 while starting from t0's state replayed the agent at the wrong moment: a
        # challenger first seen at frame 3 came back three frames' worth of motion
        # ahead of where it was actually logged.
        controls = self.base_controls[self.t0:].copy()
        init = self.base_init.copy()

        if self.is_vehicle:
            dv0, dtheta0, da_bias, ddelta_bias = delta
            init[2] = init[2] + dtheta0      # heading
            # Clamp the perturbed initial SPEED to the same range bicycle_step
            # enforces on every later state (audit B08). Without this the first step
            # ran at an out-of-range speed before anything clamped it: 0.5 m/s with a
            # permitted -3 offset drove backwards at -2.5 m/s, and 39 m/s with a +3
            # offset briefly exceeded the 40 m/s cap. Bounds the space advertises
            # have to hold for EVERY returned state, including the first.
            init[3] = np.clip(init[3] + dv0, 0.0, V_MAX)
            controls[:, 0] = controls[:, 0] + ddelta_bias  # steering
            controls[:, 1] = controls[:, 1] + da_bias      # acceleration
        else:
            dvx0, dvy0, dax_bias, day_bias = delta
            # Same gap, and clamped the same way linear_step clamps speed: scale the
            # velocity VECTOR back onto the disc rather than clipping each component,
            # so the space and the model cannot disagree about the bound's shape.
            init[2], init[3] = project_to_magnitude(
                init[2] + dvx0, init[3] + dvy0, V_MAX
            )
            controls[:, 0] = controls[:, 0] + dax_bias
            controls[:, 1] = controls[:, 1] + day_bias

        traj = self.simulator.simulate(init, controls, self.dt)  # (T-t0, state_dim)
        return self._write_back(traj)

    # ── internals ───────────────────────────────────────────────────────────

    def _valid_dimension(self, idx: int, t: int, col: int, name: str) -> float:
        """
        Read one physical dimension at a frame where the agent was observed, and
        insist it is usable.

        Fails loud rather than substituting a guess (audit B02). A zero length here
        becomes a zero wheelbase, and a zero wheelbase divides by zero in the heading
        update, poisoning the entire rollout with NaN — a failure that surfaces far
        downstream as an unexplained failed stress test rather than as bad geometry.
        """
        value = float(self.states0[idx, t, col])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"agent {idx} has unusable {name}={value} at its first valid frame "
                f"{t}; cannot build a kinematic model for it"
            )
        return value

    def _measure_baseline_replay(self):
        """
        Roll the challenger forward at ZERO perturbation and measure how far the
        result strays from what was actually logged.

        Returns (max_positional_error_m, collides_with_sdc).

        This is the honest version of the claim the whole project rests on. The
        optimizer reports "the smallest change that causes a collision", measured
        against the SDC's UNTOUCHED logged trajectory — so any error in reproducing
        the challenger is not common-mode and does not cancel. It shows up as
        perturbation that was never applied.
        """
        replay = self.apply(np.zeros(self.dim, dtype=np.float32))
        ts = valid_timesteps(self.validity, self.target_idx)

        logged = self.states0[self.target_idx, ts, :2]
        got    = replay[self.target_idx, ts, :2]
        offsets = np.linalg.norm(got - logged, axis=1)

        if len(offsets) == 0 or not np.all(np.isfinite(offsets)):
            # Non-finite means the rollout blew up; report it as unbounded drift so
            # the gate refuses rather than comparing NaN and silently passing.
            error = float('inf')
        else:
            error = float(offsets.max())

        collided, _ = check_collision_trajectory(
            replay, self.validity, self.sdc_idx, self.target_idx
        )
        return error, bool(collided)

    def _write_back(self, traj: np.ndarray) -> np.ndarray:
        """
        Insert the re-simulated challenger trajectory into a fresh copy of states,
        aligned so that traj row k lands on GLOBAL frame t0 + k (audit B01).
        Rebuilds [x, y, vx, vy, heading]; length/width are physical constants and
        are left unchanged.

        Frames before t0 keep their logged contents. The agent was not observed
        then, so those frames are invalid and every consumer skips them — leaving
        them alone is more faithful than writing a simulated past that never
        happened.
        """
        states = self.states0.copy()
        i = self.target_idx
        t0 = self.t0
        n = min(self.T - t0, traj.shape[0])
        end = t0 + n

        if self.is_vehicle:
            # traj rows are [x, y, theta, v]
            x, y, theta, v = traj[:n, 0], traj[:n, 1], traj[:n, 2], traj[:n, 3]
            states[i, t0:end, 0] = x
            states[i, t0:end, 1] = y
            states[i, t0:end, 2] = v * np.cos(theta)   # vx
            states[i, t0:end, 3] = v * np.sin(theta)   # vy
            states[i, t0:end, 4] = theta               # heading
        else:
            # traj rows are [x, y, vx, vy]
            x, y, vx, vy = traj[:n, 0], traj[:n, 1], traj[:n, 2], traj[:n, 3]
            states[i, t0:end, 0] = x
            states[i, t0:end, 1] = y
            states[i, t0:end, 2] = vx
            states[i, t0:end, 3] = vy
            states[i, t0:end, 4] = np.arctan2(vy, vx)  # heading from velocity

        return states


def pick_nearest_challenger(states, validity, sdc_idx) -> int:
    """
    Convenience helper: choose the non-SDC agent whose bounding-box center comes
    closest to the SDC at any shared valid timestep. This is usually the most
    interesting challenger to perturb. Returns its index (or -1 if none).
    """
    N, T, _ = states.shape
    best_idx, best_d = -1, np.inf
    sdc_xy = states[sdc_idx, :, :2]
    for j in range(N):
        if j == sdc_idx:
            continue
        shared = validity[sdc_idx] & validity[j]
        if not shared.any():
            continue
        d = np.linalg.norm(sdc_xy[shared] - states[j, shared, :2], axis=1).min()
        if d < best_d:
            best_d, best_idx = d, j
    return best_idx