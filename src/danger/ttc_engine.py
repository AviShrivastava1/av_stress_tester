import numpy as np


TTC_INFINITY = 999.0   # sentinel value for diverging agents
TTC_THRESHOLD = 3.0    # seconds — pre-filter threshold for Phase 5


def compute_effective_radius(length: float, width: float) -> float:
    """
    Approximate an agent as a circle for fast TTC computation.
    The radius is half the diagonal of the bounding box.
    This overestimates size slightly — safe for pre-filtering.
    """
    return np.sqrt(length**2 + width**2) / 2


def compute_ttc_pair(
    x_a: float, y_a: float, vx_a: float, vy_a: float, r_a: float,
    x_b: float, y_b: float, vx_b: float, vy_b: float, r_b: float
) -> float:
    """
    Compute TTC between a single pair of agents at one timestep.
    Uses circular approximation for speed.

    Args:
        x_a, y_a:   position of agent A
        vx_a, vy_a: velocity of agent A
        r_a:        effective radius of agent A
        x_b, y_b:   position of agent B
        vx_b, vy_b: velocity of agent B
        r_b:        effective radius of agent B

    Returns:
        TTC in seconds. TTC_INFINITY if agents are diverging.
    """
    # relative position (B relative to A)
    dx = x_b - x_a
    dy = y_b - y_a
    dist = np.sqrt(dx**2 + dy**2)

    # safe distance — sum of effective radii
    safe_dist = r_a + r_b

    # already overlapping
    if dist <= safe_dist:
        return 0.0

    # relative velocity (B relative to A)
    dvx = vx_b - vx_a
    dvy = vy_b - vy_a

    # closing speed — rate at which distance is decreasing
    # dot product of relative position unit vector with relative velocity
    closing_speed = -(dx * dvx + dy * dvy) / dist

    # agents are diverging or stationary relative to each other
    if closing_speed <= 0:
        return TTC_INFINITY

    return (dist - safe_dist) / closing_speed


def compute_ttc_all_pairs(
    states: np.ndarray,
    validity: np.ndarray,
    t: int
) -> np.ndarray:
    """
    Compute TTC for all agent pairs at a single timestep using NumPy broadcasting.
    This is the vectorized version — no Python loops over agent pairs.

    Args:
        states:   shape (N, T, 7)
        validity: shape (N, T)
        t:        timestep index

    Returns:
        ttc_matrix: shape (N, N) where ttc_matrix[i, j] is TTC between agent i and j
                    diagonal is TTC_INFINITY (agent with itself)
                    invalid pairs are TTC_INFINITY
    """
    N = states.shape[0]

    # extract positions and velocities at timestep t
    pos = states[:, t, :2]    # shape (N, 2) — x, y
    vel = states[:, t, 2:4]   # shape (N, 2) — vx, vy
    lengths = states[:, t, 5] # shape (N,)
    widths  = states[:, t, 6] # shape (N,)
    valid   = validity[:, t]   # shape (N,)

    # effective radii for all agents
    radii = np.array([compute_effective_radius(l, w)
                      for l, w in zip(lengths, widths)])  # shape (N,)

    # relative positions — broadcasting to get (N, N) matrices
    dx = pos[:, None, 0] - pos[None, :, 0]   # shape (N, N)
    dy = pos[:, None, 1] - pos[None, :, 1]   # shape (N, N)

    # relative velocities
    dvx = vel[:, None, 0] - vel[None, :, 0]  # shape (N, N)
    dvy = vel[:, None, 1] - vel[None, :, 1]  # shape (N, N)

    # distances
    dist = np.sqrt(dx**2 + dy**2)             # shape (N, N)

    # safe distances — sum of radii for each pair
    safe_dist = radii[:, None] + radii[None, :]  # shape (N, N)

    # closing speeds
    # avoid division by zero on diagonal (dist=0)
    dist_safe = np.where(dist > 0, dist, 1.0)
    closing_speed = -(dx * dvx + dy * dvy) / dist_safe  # shape (N, N)

    # compute TTC
    ttc = np.full((N, N), TTC_INFINITY)

    # already overlapping
    overlapping = dist <= safe_dist
    ttc[overlapping] = 0.0

    # approaching and not overlapping
    approaching = (closing_speed > 0) & ~overlapping
    ttc[approaching] = (dist[approaching] - safe_dist[approaching]) / closing_speed[approaching]

    # mask invalid agents
    valid_pair = valid[:, None] & valid[None, :]  # shape (N, N)
    ttc[~valid_pair] = TTC_INFINITY

    # diagonal — agent with itself
    np.fill_diagonal(ttc, TTC_INFINITY)

    return ttc


def compute_min_ttc_scenario(
    states: np.ndarray,
    validity: np.ndarray
) -> float:
    """
    Compute the minimum TTC across all agent pairs and all timesteps
    for an entire scenario. This is the scenario's TTC danger signal.

    Args:
        states:   shape (N, T, 7)
        validity: shape (N, T)

    Returns:
        min_ttc: scalar — minimum TTC in seconds across the whole scenario
    """
    T = states.shape[1]
    min_ttc = TTC_INFINITY

    for t in range(T):
        ttc_matrix = compute_ttc_all_pairs(states, validity, t)
        min_ttc = min(min_ttc, ttc_matrix.min())

    return float(min_ttc)