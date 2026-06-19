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

from src.physics.simulator import (
    ForwardSimulator, TrajectoryInverter, TYPE_VEHICLE,
)
from src.physics.bicycle_model import (
    extract_state_from_womd as bicycle_extract,
)
from src.physics.linear_model import (
    extract_state_from_womd as linear_extract,
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
        """
        self.states0  = states.astype(np.float32)
        self.validity = validity
        self.types    = types
        self.sdc_idx  = int(sdc_idx)
        self.target_idx = int(target_idx)
        self.dt = dt

        self.N, self.T, _ = states.shape
        self.target_type = int(types[target_idx])
        self.target_len  = float(states[target_idx, 0, 5])
        self.is_vehicle  = self.target_type == TYPE_VEHICLE

        # baseline controls recovered from the logged trajectory (Phase 2 inversion)
        self.inverter = TrajectoryInverter(self.target_type, self.target_len)
        self.base_controls = self.inverter.invert(
            states, self.target_idx, validity, dt
        )  # (T-1, 2)

        # forward simulator used to replay perturbed controls
        self.simulator = ForwardSimulator(self.target_type, self.target_len)

        # baseline initial state of the challenger ([x,y,theta,v] or [x,y,vx,vy])
        self.t0 = self._first_valid_t(self.target_idx)
        if self.is_vehicle:
            self.base_init = bicycle_extract(states, self.target_idx, self.t0)
        else:
            self.base_init = linear_extract(states, self.target_idx, self.t0)

        # per-dimension box bounds (used by DE and by gradient clamping)
        self.bounds = self._default_bounds() if bounds is None else np.asarray(bounds, np.float32)

        # per-dimension weights for the norm: 1 / bound_magnitude, so each term is
        # a dimensionless fraction of its allowed budget (Concept 17, weighted norm).
        self.weights = 1.0 / np.maximum(np.abs(self.bounds[:, 1]), 1e-6)

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
            1. copy the baseline controls and add the constant control biases
            2. copy the baseline initial state and add the initial-state offsets
            3. roll forward through ForwardSimulator (enforces kinematic limits)
            4. write the new trajectory back into a copy of states
        """
        delta = np.asarray(delta, np.float32)

        controls = self.base_controls.copy()
        init = self.base_init.copy()

        if self.is_vehicle:
            dv0, dtheta0, da_bias, ddelta_bias = delta
            init[3] = init[3] + dv0          # speed
            init[2] = init[2] + dtheta0      # heading
            controls[:, 0] = controls[:, 0] + ddelta_bias  # steering
            controls[:, 1] = controls[:, 1] + da_bias      # acceleration
        else:
            dvx0, dvy0, dax_bias, day_bias = delta
            init[2] = init[2] + dvx0
            init[3] = init[3] + dvy0
            controls[:, 0] = controls[:, 0] + dax_bias
            controls[:, 1] = controls[:, 1] + day_bias

        traj = self.simulator.simulate(init, controls, self.dt)  # (T, state_dim)
        return self._write_back(traj)

    # ── internals ───────────────────────────────────────────────────────────

    def _first_valid_t(self, idx: int) -> int:
        valid_ts = np.where(self.validity[idx])[0]
        return int(valid_ts[0]) if len(valid_ts) else 0

    def _write_back(self, traj: np.ndarray) -> np.ndarray:
        """
        Insert the re-simulated challenger trajectory into a fresh copy of states.
        Rebuilds [x, y, vx, vy, heading]; length/width are physical constants and
        are left unchanged.
        """
        states = self.states0.copy()
        i = self.target_idx
        T = min(self.T, traj.shape[0])

        if self.is_vehicle:
            # traj rows are [x, y, theta, v]
            x, y, theta, v = traj[:T, 0], traj[:T, 1], traj[:T, 2], traj[:T, 3]
            states[i, :T, 0] = x
            states[i, :T, 1] = y
            states[i, :T, 2] = v * np.cos(theta)   # vx
            states[i, :T, 3] = v * np.sin(theta)   # vy
            states[i, :T, 4] = theta               # heading
        else:
            # traj rows are [x, y, vx, vy]
            x, y, vx, vy = traj[:T, 0], traj[:T, 1], traj[:T, 2], traj[:T, 3]
            states[i, :T, 0] = x
            states[i, :T, 1] = y
            states[i, :T, 2] = vx
            states[i, :T, 3] = vy
            states[i, :T, 4] = np.arctan2(vy, vx)  # heading from velocity

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