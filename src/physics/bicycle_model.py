import numpy as np


# ── Kinematic constraints ──────────────────────────────────────────────────────
DELTA_MAX = 0.5    # max steering angle (radians, ~30 degrees)
A_MAX     = 12.0   # max acceleration magnitude (m/s^2)
V_MAX     = 40.0   # max speed (m/s, ~90 mph)

# A_MAX is 12.0, about 1.2 g: hard braking on a good tyre. Raised from 5.0 on a
# measurement, not a preference. Over every valid vehicle-agent transition in a real
# 496-scenario shard, |dv/dt| is p50 0.00, p90 3.25, p99 13.4 m/s^2, and 33.1% of
# vehicle agents have at least one transition above 5.0. invert_bicycle clipped those
# to 5.0 and bicycle_step replayed the clipped value, so a logged hard stop replayed
# as a gentle one and the zero-delta replay drifted ahead of the log. A counterfactual
# sweep of that mechanism put the scenarios drifting over 5 m at 32 (A=5), 13 (A=10),
# 9 (A=12) and 10 (A=15): the return flattens at 12, which is also the shard's p99.
#
# ONE CONSTANT, THREE ENFORCEMENT SITES, AND THEY CANNOT BE DECOUPLED: bicycle_step's
# clamp, invert_bicycle's clip, and autograd_optimizer's torch rollout, which must
# mirror bicycle_step exactly. Raising only the inverter's clip does nothing, because
# the zero-delta replay runs through bicycle_step, which clips it straight back down.
#
# So this also RAISES THE CEILING ON WHAT PHASE 4 MAY COMMAND: a perturbed search can
# now ask a vehicle for up to 12 m/s^2, where it used to be capped at 5. That widens
# the realism guarantee ("a manoeuvre a real driver could execute") by one notch, from
# firm braking to hard braking. It is deliberate, not a side effect.
#
# linear_model.A_MAX is a SEPARATE constant and stays 5.0. The measurement above
# covers vehicles only; nothing has measured pedestrians or cyclists, and raising
# their bound on vehicle data would present an assumption as a measurement.
#
# Every Pass 2 result records the cap it was searched under (search_provenance
# 'a_max'), and export_shard_geometry refuses to replay a stored delta under a
# different one. See simulator.replay_a_max.
DT        = 0.1    # timestep duration (seconds, WOMD is 10 Hz)


def get_wheelbase(length: float) -> float:
    """
    Estimate wheelbase from agent length.
    Wheelbase = distance from rear axle to front axle.
    Approximated as 60% of total vehicle length.
    """
    return 0.6 * length


def bicycle_step(
    state: np.ndarray,
    control: np.ndarray,
    wheelbase: float,
    dt: float = DT
) -> np.ndarray:
    """
    Propagate a vehicle state forward by one timestep using the
    bicycle kinematic model.

    Args:
        state:     [x, y, theta, v] — position, heading, speed
        control:   [delta, a] — steering angle, acceleration
        wheelbase: distance from rear to front axle (meters)
        dt:        timestep duration (seconds)

    Returns:
        next_state: [x, y, theta, v] after one timestep
    """
    x, y, theta, v = state
    delta, a = control

    # enforce kinematic constraints on control inputs
    delta = np.clip(delta, -DELTA_MAX, DELTA_MAX)
    a     = np.clip(a, -A_MAX, A_MAX)

    # heading update — turning rate = (v / L) * tan(delta).
    # Uses the PRE-step speed, which is what invert_bicycle assumes when it recovers
    # delta from a heading change. Forward and inverse must agree here or a replayed
    # heading drifts from the logged one; do not "improve" this without changing
    # invert_bicycle in the same edit.
    theta_next = theta + (v / wheelbase) * np.tan(delta) * dt

    # speed update — clamp to [0, V_MAX], cars don't go backwards
    v_next = np.clip(v + a * dt, 0.0, V_MAX)

    # position update — trapezoidal, from the MIDPOINT of the step (audit B03).
    #
    # This used to integrate with the pre-step speed and pre-step heading:
    #     x_next = x + v * cos(theta) * dt
    # which is wrong in two independent ways, and both matter. An accelerating car
    # really covers v*dt + 0.5*a*dt^2 in one step, so the old form lagged a logged
    # 1 m/s^2 track by 0.45 m over 9 s. A turning car really moves along the chord,
    # whose direction is the mid-step heading, so the old form drifted 0.78 m over
    # 9 s on a 50 m-radius turn at 10 m/s.
    #
    # Neither error cancels against the SDC, because apply() replaces only the
    # challenger and leaves the SDC's logged positions untouched — so this drift was
    # enough to make the exact SAT check report a collision at ZERO perturbation.
    #
    # The midpoints are taken from the CLAMPED next state, not from `a` and `delta`
    # directly, so a step that saturates V_MAX (or floors at 0) integrates the speed
    # the car actually reached rather than the one it was commanded toward.
    v_mid     = 0.5 * (v + v_next)
    theta_mid = 0.5 * (theta + theta_next)

    x_next = x + v_mid * np.cos(theta_mid) * dt
    y_next = y + v_mid * np.sin(theta_mid) * dt

    return np.array([x_next, y_next, theta_next, v_next], dtype=np.float32)


def invert_bicycle(
    state_t: np.ndarray,
    state_next: np.ndarray,
    wheelbase: float,
    dt: float = DT
) -> np.ndarray:
    """
    Recover the control inputs [delta, a] that produced the transition
    from state_t to state_next under the bicycle model.

    This is trajectory inversion — going from observed states back to controls.
    Used in Phase 4 to get the baseline control sequence before perturbation.

    Args:
        state_t:    [x, y, theta, v] at time t
        state_next: [x, y, theta, v] at time t+1
        wheelbase:  distance from rear to front axle (meters)
        dt:         timestep duration (seconds)

    Returns:
        control: [delta, a]
    """
    _, _, theta, v       = state_t
    _, _, theta_next, v_next = state_next

    # recover acceleration from speed change
    a = (v_next - v) / dt

    # recover steering angle from heading change
    # guard against near-zero speed to avoid divide-by-zero
    if abs(v) < 1e-3:
        delta = 0.0
    else:
        # wrap heading difference to [-pi, pi] to handle angle wraparound
        dtheta = theta_next - theta
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        delta = np.arctan(dtheta * wheelbase / (v * dt))

    # clip recovered controls to valid range
    delta = np.clip(delta, -DELTA_MAX, DELTA_MAX)
    a     = np.clip(a, -A_MAX, A_MAX)

    return np.array([delta, a], dtype=np.float32)


def extract_state_from_womd(states: np.ndarray, agent_idx: int, t: int) -> np.ndarray:
    """
    Pull a bicycle model state vector from the parsed WOMD states array.

    Args:
        states:    shape (N, T, 7) from ScenarioParser.get_agent_states()
        agent_idx: which agent
        t:         which timestep

    Returns:
        state: [x, y, theta, v] where v = sqrt(vx^2 + vy^2)
    """
    x       = states[agent_idx, t, 0]
    y       = states[agent_idx, t, 1]
    vx      = states[agent_idx, t, 2]
    vy      = states[agent_idx, t, 3]
    theta   = states[agent_idx, t, 4]
    speed   = np.sqrt(vx**2 + vy**2)

    return np.array([x, y, theta, speed], dtype=np.float32)