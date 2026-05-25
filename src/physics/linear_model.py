import numpy as np


A_MAX = 5.0    # max acceleration magnitude (m/s^2)
V_MAX = 40.0   # max speed (m/s)
DT    = 0.1    # timestep duration (seconds)


def linear_step(
    state: np.ndarray,
    control: np.ndarray,
    dt: float = DT
) -> np.ndarray:
    """
    Propagate a pedestrian or cyclist state forward by one timestep
    using the linear kinematic model.

    Args:
        state:   [x, y, vx, vy] — position and velocity components
        control: [ax, ay] — acceleration components
        dt:      timestep duration (seconds)

    Returns:
        next_state: [x, y, vx, vy] after one timestep
    """
    x, y, vx, vy = state
    ax, ay = control

    # enforce acceleration constraints
    ax = np.clip(ax, -A_MAX, A_MAX)
    ay = np.clip(ay, -A_MAX, A_MAX)

    # position update — includes acceleration term for accuracy
    x_next = x + vx * dt + 0.5 * ax * dt**2
    y_next = y + vy * dt + 0.5 * ay * dt**2

    # velocity update
    vx_next = vx + ax * dt
    vy_next = vy + ay * dt

    # enforce speed constraint
    speed = np.sqrt(vx_next**2 + vy_next**2)
    if speed > V_MAX:
        scale = V_MAX / speed
        vx_next *= scale
        vy_next *= scale

    return np.array([x_next, y_next, vx_next, vy_next], dtype=np.float32)


def invert_linear(
    state_t: np.ndarray,
    state_next: np.ndarray,
    dt: float = DT
) -> np.ndarray:
    """
    Recover control inputs [ax, ay] from consecutive observed states.

    Args:
        state_t:    [x, y, vx, vy] at time t
        state_next: [x, y, vx, vy] at time t+1
        dt:         timestep duration (seconds)

    Returns:
        control: [ax, ay]
    """
    _, _, vx, vy         = state_t
    _, _, vx_next, vy_next = state_next

    ax = (vx_next - vx) / dt
    ay = (vy_next - vy) / dt

    ax = np.clip(ax, -A_MAX, A_MAX)
    ay = np.clip(ay, -A_MAX, A_MAX)

    return np.array([ax, ay], dtype=np.float32)


def extract_state_from_womd(
    states: np.ndarray,
    agent_idx: int,
    t: int
) -> np.ndarray:
    """
    Pull a linear model state vector from the parsed WOMD states array.

    Args:
        states:    shape (N, T, 7) from ScenarioParser.get_agent_states()
        agent_idx: which agent
        t:         which timestep

    Returns:
        state: [x, y, vx, vy]
    """
    x  = states[agent_idx, t, 0]
    y  = states[agent_idx, t, 1]
    vx = states[agent_idx, t, 2]
    vy = states[agent_idx, t, 3]

    return np.array([x, y, vx, vy], dtype=np.float32)