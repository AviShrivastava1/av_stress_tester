import numpy as np


A_MAX = 5.0    # max acceleration magnitude (m/s^2)
V_MAX = 40.0   # max speed (m/s)
DT    = 0.1    # timestep duration (seconds)

# The speed below which this model's HEADING IS NOT OBSERVABLE (audit A01).
#
# The linear state vector is [x, y, vx, vy]. It has no heading, so every consumer
# that needs one recovers it as arctan2(vy, vx) — and that recovery is singular at
# the origin. atan2(eps, 0) is exactly pi/2 for EVERY positive eps, down to the
# smallest float32 subnormal, so the derived orientation of a stationary agent is
# not approximately undefined, it is a step function of an unmeasurable quantity.
#
# Measured on a stationary cyclist beside the SDC: dvy0 = 1e-9 rotated the footprint
# a quarter turn into a verified collision at weighted norm 5e-10, and dvy0 = 1e-30
# did the same at a weighted norm of EXACTLY 0.0 — the float32 square underflows
# while atan2 does not care how small the input was. A real DE run found the exploit
# unprompted at norm 0.003215756.
#
# Same shape as invert_bicycle's `if abs(v) < 1e-3: delta = 0.0` guard, and for the
# same reason: a quantity recovered by dividing by v is not recoverable as v -> 0,
# so return the honest value rather than one amplified by 1/|v|. The constants
# differ deliberately. That one only has to avoid a division by zero; this one has
# to bound a CONDITION NUMBER, and at 1e-3 m/s the sensitivity is still 1000 rad per
# m/s, which leaves the exploit merely six times more expensive instead of eight
# orders of magnitude.
#
# 0.5 m/s IS A DEFAULT TO BE MEASURED, NOT A CONSTANT THAT HAS BEEN DEFENDED. It has
# exactly the status BASELINE_DRIFT_REFUSE_M has: PerturbationSpace takes
# heading_speed_floor=None to switch it off, and records how much of the challenger
# falls below it on EVERY scenario whether it is switched on or not, so the real
# distribution of near-stationary challengers can be measured on a real shard before
# this number is argued for.
V_HEADING_MIN = 0.5


def project_to_magnitude(u: float, v: float, limit: float):
    """
    Scale the vector (u, v) back onto the disc of radius `limit`, preserving its
    direction. Vectors already inside the disc are returned untouched.

    Shared by linear_step and invert_linear so the forward and inverse models can
    never disagree about whether a bound is on the magnitude or on the components.
    """
    magnitude = np.sqrt(u * u + v * v)
    if magnitude > limit:
        scale = limit / magnitude
        return u * scale, v * scale
    return u, v


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

    # enforce the acceleration constraint as a MAGNITUDE bound (audit B08).
    # A_MAX is documented as "max acceleration magnitude", but this used to clip the
    # two components independently, which bounds a SQUARE and not a disc: control
    # [5, 5] against a declared 5.0 limit produced |a| = 7.07 m/s^2. Projecting the
    # vector back onto the disc enforces the limit that is actually written down,
    # and preserves the commanded direction while doing it.
    ax, ay = project_to_magnitude(ax, ay, A_MAX)

    # velocity update, including the speed constraint, BEFORE position — the
    # position update below integrates the velocity the agent actually ends at.
    vx_next = vx + ax * dt
    vy_next = vy + ay * dt
    vx_next, vy_next = project_to_magnitude(vx_next, vy_next, V_MAX)

    # position update — trapezoidal, from the midpoint of the clamped velocities.
    #
    # Unsaturated this is algebraically identical to the previous
    # `x + vx*dt + 0.5*ax*dt**2`, so for essentially every real pedestrian and
    # cyclist nothing changes. It differs only when A_MAX or V_MAX actually binds,
    # which is exactly where the old form silently stopped being trapezoidal: it
    # integrated the PRE-scaling velocity, so a saturated step moved the agent
    # further than its own clamped velocity allowed.
    #
    # Matching bicycle_step's rule here is the point. "The two kinematic models
    # integrate position differently" is the asymmetry that became audit B03; a
    # second instance of it is not worth keeping.
    x_next = x + 0.5 * (vx + vx_next) * dt
    y_next = y + 0.5 * (vy + vy_next) * dt

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

    # Same magnitude projection linear_step applies (audit B08). Inversion and the
    # forward step MUST clamp by the same rule: if invert clipped per-component and
    # the step projected by magnitude, a recovered control would be re-clamped on
    # replay and the pedestrian/cyclist path would acquire drift that the replay
    # fidelity gate then refuses — a bug manufactured purely by the two functions
    # disagreeing about what A_MAX means.
    ax, ay = project_to_magnitude(ax, ay, A_MAX)

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