import numpy as np


# ── Kinematic constraints ──────────────────────────────────────────────────────
DELTA_MAX = 0.5    # max steering angle (radians, ~30 degrees)
A_MAX     = 5.0    # max acceleration magnitude (m/s^2)
V_MAX     = 40.0   # max speed (m/s, ~90 mph)
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

    # position update — car moves in the direction it is heading
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt

    # heading update — turning rate = (v / L) * tan(delta)
    theta_next = theta + (v / wheelbase) * np.tan(delta) * dt

    # speed update — clamp to [0, V_MAX], cars don't go backwards
    v_next = np.clip(v + a * dt, 0.0, V_MAX)

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