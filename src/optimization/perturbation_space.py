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
    V_HEADING_MIN,
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

# Width (m/s) of the speed band, anchored at heading_speed_floor and extending
# upward, over which a linear-model challenger's heading ramps smoothly from the
# LOGGED value to the DERIVED one (independent review, 2026-09-18: the pre-fix
# np.where(speed >= floor, derived, logged) was a hard step exactly at the floor,
# so a delta of ~5e-7 weighted norm could flip a verified collision by crossing
# it — measured in both directions, not assumed). Same status as V_HEADING_MIN
# and BASELINE_DRIFT_REFUSE_M: an order-of-magnitude, defensible default (10% of
# V_HEADING_MIN), not yet validated against a real shard.
HEADING_TRANSITION_WIDTH = 0.05


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


def _smoothstep(x, edge0, edge1):
    """
    Cubic Hermite smoothstep: 0 at/below edge0, 1 at/above edge1, continuous (and
    zero-slope at both ends, i.e. C1) in between. Standard formula, `t*t*(3-2t)`
    with `t` clamped to [0, 1] first — used here only for `_linear_heading`'s
    logged/derived blend, over the band [heading_speed_floor,
    heading_speed_floor + heading_transition_width].
    """
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


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
        heading_speed_floor: float = V_HEADING_MIN,
        heading_transition_width: float = HEADING_TRANSITION_WIDTH,
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
            heading_speed_floor:
                        speed (m/s) below which a LINEAR-model challenger's heading is
                        held at its logged value instead of being derived from the
                        velocity direction (audit A01). Ignored for vehicles, whose
                        heading is real state. None disables the floor and restores
                        the pre-A01 behaviour — same escape hatch, and the same
                        warning, as max_baseline_drift: it exists so the real
                        distribution can be measured, not so a refusal can be
                        silenced. The measurement is recorded either way; see
                        `frames_below_heading_floor` and `target_min_speed`.
            heading_transition_width:
                        width (m/s) of the band, ANCHORED AT heading_speed_floor and
                        extending upward, over which heading ramps smoothly from
                        logged to derived instead of switching at a single point
                        (independent review, post-A01: the switch itself was a hard
                        step, so a delta of order 1e-6 weighted norm could cross it
                        for a free footprint rotation — a different defect from A01's
                        singularity-at-rest, same family). Anchoring at the floor
                        rather than centring on it means speed < heading_speed_floor
                        keeps using logged EXACTLY, same as before this parameter
                        existed — nothing below the floor derives, full stop; only
                        the room needed to eliminate the step is added above it.
                        Ignored when heading_speed_floor is None (no floor, no band).
                        0 or None reproduces the pre-fix step exactly — the same
                        escape hatch as heading_speed_floor, for measuring the real
                        distribution before this width is defended as final. See
                        `frames_in_heading_transition_band`.

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

        # ── heading observability for the linear model (audit A01) ──────────────
        #
        # RECORDED ON EVERY SCENARIO, INCLUDING WHEN THE FLOOR IS OFF AND INCLUDING
        # FOR VEHICLES, for the same reason has_interior_gap is recorded whether or
        # not it changes anything: a number that only exists when it already mattered
        # cannot tell anyone how often it matters. These two are what a Colab pass
        # needs to turn V_HEADING_MIN from a proposal into a defended default.
        #
        # Measured against the LOGGED velocities, not a perturbed replay, so it
        # describes the scenario rather than one search's path through it.
        self.heading_speed_floor = heading_speed_floor
        self.heading_transition_width = heading_transition_width
        _ts = valid_timesteps(self.validity, self.target_idx)
        if len(_ts):
            _logged_speed = np.hypot(
                self.states0[self.target_idx, _ts, 2].astype(np.float64),
                self.states0[self.target_idx, _ts, 3].astype(np.float64),
            )
            self.target_min_speed = float(_logged_speed.min())
            # counted against the ACTIVE floor when there is one, and against the
            # module default when the floor is off, so switching it off to measure
            # does not also switch off the measurement.
            _floor = V_HEADING_MIN if heading_speed_floor is None else heading_speed_floor
            self.frames_below_heading_floor = int((_logged_speed < _floor).sum())
            # Same idea, for the band the transition-width fix added (independent
            # review, post-A01): a LOGGED frame that lands inside [floor, floor+width)
            # no longer replays its exact logged heading under a zero-delta
            # perturbation — a narrower version of the fidelity gap A01b closed at
            # v=0. Recorded whether or not the width is currently zero, for the same
            # reason frames_below_heading_floor is: this is what turns
            # HEADING_TRANSITION_WIDTH from a proposal into a defended default.
            _width = (HEADING_TRANSITION_WIDTH if not heading_transition_width
                     else heading_transition_width)
            self.frames_in_heading_transition_band = int(
                ((_logged_speed >= _floor) & (_logged_speed < _floor + _width)).sum()
            )
            self.frames_observed = int(len(_ts))
        else:
            self.target_min_speed = float('inf')
            self.frames_below_heading_floor = 0
            self.frames_in_heading_transition_band = 0
            self.frames_observed = 0

        # per-dimension box bounds (used by DE and by gradient clamping)
        self.bounds = self._default_bounds() if bounds is None else np.asarray(bounds, np.float32)

        # per-dimension weights for the norm: 1 / bound_magnitude, so each term is
        # a dimensionless fraction of its allowed budget (Concept 17, weighted norm).
        #
        # THE MAGNITUDE IS THE LARGER SIDE, NOT THE UPPER BOUND (audit B20). Using
        # abs(high) alone assumes every bound straddles zero symmetrically, which the
        # defaults below do — but a one-sided bound is legitimate and the obvious one
        # is braking-only, [-3, 0]. There abs(high) is 0, the 1e-6 floor takes over,
        # and the weight becomes 1e6: a delta that spends exactly the 3 m/s the bound
        # actually allows reports a norm of 3e6 instead of 1. The optimizers minimize
        # this norm, so a one-sided dimension would be priced as unusable rather than
        # as a full budget.
        #
        # Inert for every existing caller: all default bounds are symmetric, so
        # max(|low|, |high|) == |high| and the weights are unchanged term for term.
        # Nothing in src/ or tests/ passes custom bounds except the audit's own B20
        # fixture.
        #
        # The 1e-6 floor stays, and now guards a genuinely degenerate case: a
        # dimension pinned to a single value ([0, 0]) has NO budget, so any nonzero
        # delta there is outside the box. A huge norm says "infinitely far outside its
        # budget", which is true; a weight of zero would price it as free, which is
        # not.
        bound_magnitude = np.maximum(np.abs(self.bounds[:, 0]),
                                     np.abs(self.bounds[:, 1]))
        self.weights = 1.0 / np.maximum(bound_magnitude, 1e-6)

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
            states[i, t0:end, 4] = self._linear_heading(vx, vy, t0, end)

        return states

    def _linear_heading(self, vx, vy, t0, end) -> np.ndarray:
        """
        The heading to write for a linear-model challenger: derived from the velocity
        direction where that direction is observable, and the LOGGED heading where it
        is not (audit A01).

        THE OLD LINE WAS `np.arctan2(vy, vx)`, UNCONDITIONALLY, and it had two
        defects rather than the one the audit reported.

        The reported one: atan2 is singular at the origin, and not in a way more
        precision helps. atan2(eps, 0) is EXACTLY pi/2 for every positive eps down to
        the smallest float32 subnormal, so a stationary challenger's footprint could
        be rotated a quarter turn — into a verified, exact-SAT collision — by a delta
        too small to be worth anything. Measured on a stationary cyclist 1.8 m from
        the SDC: dvy0 = 1e-9 collided at weighted norm 5.0e-10, and dvy0 = 1e-30
        collided at a weighted norm of EXACTLY 0.0, because the float32 square
        underflows while atan2 does not. The centre never moved. A real DE run found
        it unprompted at norm 0.003215756, so this was reachable by the ordinary
        pipeline and not a constructed curiosity — pick_nearest_challenger has no
        speed filter, and a parked bicycle beside the SDC is an entirely ordinary
        challenger.

        The UNREPORTED one, found while reproducing the first, is worse because it
        needs no perturbation at all: for a stationary agent vx = vy = 0 in the
        LOGGED data too, so arctan2(0, 0) = 0 fired on the zero-delta replay and
        silently rewrote the agent's logged orientation to zero. A parked car facing
        0.7 rad replayed facing 0.0 — a 40-degree footprint rotation applied by a
        delta of exactly zero. That is a Batch 1 / audit B03 violation ("the
        zero-perturbation replay must reproduce the logged track") living in a column
        B03's own measurement never inspected: _measure_baseline_replay compares
        states[..., :2], and heading is column 4, so it reported 0.0000 m of error
        while the orientation was being destroyed.

        Holding the LOGGED heading FRAME BY FRAME, rather than the t0 heading, is
        what makes the zero-delta replay exact everywhere instead of merely at the
        start — an agent that was logged rotating in place keeps doing so.

        THE BOUNDARY ITSELF WAS A THIRD DEFECT, found by independent review after
        this fix shipped (2026-09-18), same family as the first two: a hard switch is
        continuous everywhere except at the switch. `np.where(speed >= floor, derived,
        logged)` is exactly `arctan2`'s old singularity relocated from v=0 to
        v=heading_speed_floor rather than removed — and moving the floor cannot fix
        this, because the argument does not depend on where the floor is set.
        Measured, both directions, on a fixture where logged and derived genuinely
        disagree (routine for a slow real agent, whose recorded heading and
        velocity-implied direction need not agree): crossing UPWARD, dvy0=0.049999
        left speed at 0.499999 with no collision; dvy0=0.05 put speed at exactly
        0.5 and produced a VERIFIED COLLISION, for a weighted-norm difference of
        ~5e-7. Crossing DOWNWARD reproduced the same cliff in the other direction at
        the same magnitude. Neither the frame-0-from-rest guarantee two paragraphs
        up nor A01's own tests caught this, because both are scoped to v=0 — this
        defect lives entirely at v=heading_speed_floor, a different point.

        THE FIX: heading ramps smoothly across [heading_speed_floor,
        heading_speed_floor + heading_transition_width] instead of switching at one
        point — see _smoothstep. ANCHORED at the floor rather than centred on it,
        deliberately: centring would let speeds BELOW heading_speed_floor blend in a
        derived component, relaxing "nothing below the floor ever derives" — a real
        guarantee the pre-fix code already had (by construction, `speed >= floor` is
        the only branch that reaches `derived`) and which A01's rest-fixture tests
        depend on. Anchoring instead makes that guarantee STRICTLY STRONGER: with
        w = smoothstep(speed, floor, floor + width), w is exactly 0 for every
        speed < floor as before, AND now also exactly 0 AT speed == floor (where the
        old code jumped to derived) — so the two sides of the old cliff now agree,
        and full derivation is deferred until floor + width, not floor.

        Blended as UNIT VECTORS, not raw angles: averaging angles directly breaks at
        the +-pi wraparound (-179 deg and +179 deg would average to ~0, not +-180).
        Working in the plane the angle lives on sidesteps this entirely and is exact.

        THE DEGENERATE CASE, HANDLED DELIBERATELY: when logged and derived are
        exactly pi apart, the blended vector can land at (0, 0) — at whatever weight
        makes (1-w)*u + w*(-u) = 0, generically the band's midpoint. arctan2(0, 0)
        returns 0.0 by silent convention, an angle unrelated to either input — which
        is exactly the kind of unexamined default that made the ORIGINAL defect this
        function fixes (arctan2(0,0)=0 silently overwriting a parked car's logged
        0.7 rad, see above) rather than a new one. Falls back to `logged` explicitly:
        the conservative choice, since it is the value already in effect before any
        derivation began and hands an optimiser nothing extra at the degenerate
        point, unlike `derived` (a third free-rotation case) or an unexamined
        arctan2(0,0) (unpredictable).

        WHAT THIS FIXES AND WHAT IT DOES NOT, stated rather than implied:

          * fixed — orientation is preserved exactly under the identity perturbation,
            for any frame whose LOGGED speed is < heading_speed_floor (the vast
            majority — a frame whose logged speed itself lands inside the transition
            band no longer replays byte-exact under delta=0; see
            `frames_in_heading_transition_band`, measured whether or not this
            matters, same reason `frames_below_heading_floor` is);
          * fixed — rotation is no longer free, AND the old single-point cliff at
            v=heading_speed_floor is gone: heading is now continuous in speed (hence
            in delta) everywhere. Rotating a resting linear agent from TRUE rest
            still requires reaching floor + heading_transition_width, which costs
            slightly MORE than the pre-fix bound, not less. A challenger whose
            LOGGED speed already sits near the floor pays less — as little as
            heading_transition_width * weight_vy to swing fully across the band —
            which is smaller than the rest-to-floor cost but, unlike before, never
            zero and never a discontinuous jump;
          * NOT fixed — above the floor the 1/|v| sensitivity remains. At v = 0.5 a
            unit of weighted norm still buys about 2 rad, where a vehicle's dtheta0
            buys 0.2, so a linear agent's orientation stays roughly ten times cheaper
            to rotate than a vehicle's. Bounded, not equalised;
          * NOT fixed — rotation is still not a priced DIMENSION. The cost function
            sees it only through the velocity change that caused it. Pricing it
            directly means a fifth perturbation dimension, which is rejected on arity
            (four separate assertions plus the delta column) rather than on merit;
          * NOT addressed — whether a cyclist can physically rotate in place at all
            is a modelling question this does not open;
          * NOT fixed, AND STRUCTURAL RATHER THAN AN OVERSIGHT — the antipodal
            fallback above has its own residual cliff, measured rather than assumed
            away. Straight-line vector averaging between two EXACTLY opposite unit
            vectors is not just numerically delicate but topologically undefined —
            there is no continuous way to pick which side a 180-degree tie resolves
            to, so any tie-break rule has a jump SOMEWHERE. This design confines it:
            verified numerically, |blend| only drops below the 1e-5 fallback
            threshold when logged and derived disagree by more than
            179.998854 degrees (within ~0.001146 deg of exactly antipodal) AND
            speed sits in the resulting sub-1e-5-wide window near the band's
            midpoint — both simultaneously, not either alone. (An earlier version
            of this threshold, 1e-3, was 100x too generous — it was sized against
            the float32-precision bug fixed above rather than re-measured
            afterward, and independent review found it forcing `logged` on pairs
            merely 0.057 deg off exact antipodal, recreating a smaller version of
            the very cliff this function exists to remove; see the threshold's own
            comment below for the corrected, measured derivation.) Finding this
            narrower window requires an adversarial search to hit two
            independently narrow targets at once, unlike the single-dimension,
            one-sided cliff this function was written to remove. Not eliminated
            because it cannot be, short of abandoning vector-blend interpolation
            entirely; bounded and quantified instead.

        Vehicles never reach this function: the bicycle model carries theta as real
        state, so its heading is written from the rollout and was never derived.
        """
        derived = np.arctan2(vy, vx)
        if self.heading_speed_floor is None:
            return derived
        # np.hypot in float64 rather than sqrt(vx*vx + vy*vy) in float32. DEFENSIVE,
        # NOT LOAD-BEARING, and the distinction is recorded because the obvious story
        # about it is wrong: the float32 square is what underflows in weighted_norm,
        # so it is tempting to say this comparison would inherit the same blind spot.
        # It would not. v*v reaches exactly zero in float32 only below
        # sqrt(smallest float32 subnormal) = sqrt(1.4013e-45) = 3.7434e-23, and any
        # floor worth setting is twenty-odd orders of magnitude above that, so both
        # spellings agree such a speed is below it. Mutation-tested: swapping this for
        # the float32 form changed no test.
        #
        # (Derived, not recalled. A previous version of this comment said 1e-19, which
        # is sqrt of the smallest NORMAL — wrong by four orders, because the square
        # stays representable as a subnormal well past that point. The conclusion
        # survived the error, which is exactly why the number had to be checked.)
        #
        # Kept because it costs nothing and the float64 form is the one whose
        # correctness does not depend on where the floor happens to be set.
        vx64 = vx.astype(np.float64)
        vy64 = vy.astype(np.float64)
        speed = np.hypot(vx64, vy64)
        logged = self.states0[self.target_idx, t0:end, 4].astype(np.float64)

        if not self.heading_transition_width:
            # width == 0 (or explicitly None) reproduces the pre-fix step exactly —
            # the same measurement escape hatch heading_speed_floor itself has, for
            # an A/B comparison against real data before HEADING_TRANSITION_WIDTH is
            # defended as final.
            return np.where(speed >= self.heading_speed_floor, derived, logged)

        # RECOMPUTED IN FLOAT64, NOT THE `derived` ABOVE. Measured, not assumed: the
        # float32 arctan2 carries ~2.4e-7 rad of error near pi, which is small on
        # its own but is exactly the kind of error the blend below is sensitive to
        # near an antipodal pair (see the degenerate-case note). Confirmed by
        # running this fixture with the float32 `derived`: a logged/derived pair
        # exactly pi apart, at the band midpoint where blend cancellation should be
        # near-exact, came back 0.0527 rad off instead of falling to the degenerate
        # branch below — the float32 precision loss alone was enough to walk the
        # blend vector's magnitude from ~1e-16 out to ~1.4e-6, comfortably past a
        # naively-small epsilon. Recomputing here removes that specific source;
        # `w` still will not land on EXACTLY the analytic midpoint for a real
        # speed, which is why the epsilon below is sized in real margin, not at
        # machine precision.
        derived64 = np.arctan2(vy64, vx64)
        w = _smoothstep(speed, self.heading_speed_floor,
                        self.heading_speed_floor + self.heading_transition_width)
        bx = (1.0 - w) * np.cos(logged) + w * np.cos(derived64)
        by = (1.0 - w) * np.sin(logged) + w * np.sin(derived64)
        magnitude = np.hypot(bx, by)
        # THE DEGENERATE-CASE THRESHOLD — CORRECTED A SECOND TIME, BY INDEPENDENT
        # REVIEW, AFTER THE FIRST FIX SHIPPED WITHOUT RE-MEASURING IT.
        #
        # 1e-3 was the first number tried, and it was wrong in the SAME family of
        # mistake this whole function exists to fix: a threshold sized for the
        # threat that had just been found (float32 `derived` precision) rather than
        # re-measured after that threat was removed. It was never checked against a
        # NEAR-but-not-exactly-antipodal pair — one delta_angle, of a fixture 0.057
        # deg off exact antipodal, reaches |blend| as low as 5.08e-4 at some point
        # during an ordinary speed sweep across the band. 1e-3 is BIGGER than that,
        # so it wrongly forced `logged` at that point too — recreating the original
        # discontinuity, smaller and relocated rather than removed: measured, a
        # weighted-norm-equivalent change of 2.6e-7 m/s produced a 2.625 rad jump.
        #
        # The number that actually matters is the REAL noise floor at TRUE exact
        # antipodal, measured through the real rollout (not the idealized w=0.5,
        # derived=pi-exactly hand calculation, which understates it): |blend| came
        # out to 5.9e-7 to 1.4e-6 across four different antipodal directions,
        # dominated not by arctan2's own precision (fixed above) but by `speed`
        # itself never landing on the smoothstep's exact analytic midpoint — a
        # float32-scale error in `speed` (~2.4e-8) gets amplified by the
        # smoothstep's own slope there, 6 / heading_transition_width (currently
        # 6/0.05=120, i.e. dw/dspeed=30 at t=0.5, and |blend|~=2*dw), to ~1.4e-6.
        # THIS THRESHOLD SCALES WITH heading_transition_width — a materially
        # smaller width would raise this noise floor proportionally and this
        # number would need re-deriving, not reused.
        #
        # 1e-5 sits about 7x above that measured noise floor and about 50x below
        # the smallest near-antipodal danger-zone minimum measured above — margin
        # verified on both sides, not assumed. In degrees, this confines the
        # forced-logged zone to within ~0.001146 deg of exact antipodal (was
        # ~0.1146 deg at 1e-3 — about 100x narrower), which
        # test_heading_is_continuous_near_but_not_at_the_antipodal_point in
        # tests/test_batch9_contract.py checks directly, inside what used to be the
        # 1e-3 danger zone and is well clear of the new one.
        return np.where(magnitude < 1e-5, logged, np.arctan2(by, bx))


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