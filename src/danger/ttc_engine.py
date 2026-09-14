import numpy as np


TTC_INFINITY = 999.0   # sentinel value for diverging agents
TTC_THRESHOLD = 3.0    # seconds — pre-filter threshold for Phase 5


def compute_effective_radius(length, width):
    """
    Approximate an agent as a circle for fast TTC computation.
    The radius is half the diagonal of the bounding box.
    This overestimates size slightly — safe for pre-filtering.

    Accepts scalars or NumPy arrays; the expression is elementwise either way,
    which is what lets compute_ttc_all_pairs build its radii without a loop.
    """
    return np.sqrt(length**2 + width**2) / 2


# ── why TTC is a quadratic root and not a division ──────────────────────────────
#
# TTC used to be computed as `(dist - safe_dist) / closing_speed`: project the
# CURRENT closing speed forward and ask when the gap reaches zero. That is a linear
# extrapolation along the line of sight, and it is only correct when the relative
# motion is purely radial. For any relative velocity with a lateral component it
# reports a collision that never happens (audit B07).
#
# The audit's repro: A parked at the origin, B at (10, 3) moving (-5, 0), radius 1
# each. B passes 3 m to the side of A, and the circles need to be within 2 m to
# touch, so they never intersect. The old formula reported TTC = 1.762387740 s.
#
# The exact question is: at what time does the distance between two points moving at
# constant velocity first equal the sum of the radii? With d = relative position,
# v = relative velocity and R = safe_dist, solve |d + v*t|^2 = R^2, which expands to
# the quadratic a*t^2 + b*t + c = 0 with
#
#     a = |v|^2              always >= 0
#     b = 2 * (d . v)        negative exactly when the agents are currently closing
#     c = |d|^2 - R^2        negative exactly when they already overlap
#
# The smaller root (-b - sqrt(D)) / (2a) is the FIRST contact, which is the one TTC
# means. Taking the larger root would report the moment they separate again.
#
# THE OLD BRANCH FOR DIVERGING AGENTS WAS NEVER WRONG, and it is worth knowing why
# this bug is one-directional. closing_speed <= 0 means d . v >= 0, so b >= 0; with
# a > 0 and c > 0 the parabola's vertex sits at t = -b/(2a) <= 0, so f is
# non-decreasing over t >= 0 and never returns to zero. Infinity was correct. The
# defect fires ONLY in the currently-closing branch, and it can only ever manufacture
# a finite TTC for a pair that actually misses — a false alarm, never a missed
# danger. That lands on the same side as the circle oversizing above, though for an
# unrelated reason: oversizing is a deliberate margin, this was an error that
# happened to point the safe way.
#
# WHY THE EXISTING CONTROL TEST DOES NOT CATCH THIS. tests/test_audit_core.py
# asserts compute_ttc_pair(0,0,0,0,1, 10,0,-5,0,1) == 1.6. That fixture has dy = 0 —
# a pure head-on approach, where the relative motion IS radial and the quadratic
# degenerates to exactly the linear answer. It returns 1.6 under both the old code
# and this one. A test that cannot distinguish the two implementations cannot be
# evidence for either, which is why B07 needs a fixture with lateral offset.
#
# EVERY SQUARE BELOW IS WRITTEN `x * x`, NOT `x ** 2`, AND IN THE SCALAR FUNCTION
# THAT IS LOAD-BEARING. Python's float ** 2 dispatches to the platform C library's
# pow(), a genuinely different code path from multiplication that disagrees with it in
# the last bit for roughly 1 value in 1150. NumPy rewrites arr ** 2 as arr * arr, so in
# compute_ttc_all_pairs the two spellings are the same operation and the choice there is
# only consistency — measured: mutating the matrix path to ** 2 changes nothing, while
# mutating the scalar path breaks parity.
#
# The hazard is specific, and it is not where you would first look. A float32 input
# squared is EXACT in float64 (a 24-bit mantissa squares into 48 bits, which fits in
# 53), so pow and multiply agree on it and no divergence is visible. What diverges is
# the square of a DIFFERENCE of two float32 values, which can need far more than 24
# bits — and dx, dy, dvx and dvy are all differences. Even then the divergence reaches
# the returned value for only about 1 pair in 1600, because the sqrt and the division
# usually round it away again.
#
# That rarity is why the parity test in tests/test_danger_contract.py sweeps tens of
# thousands of pairs rather than a handful: at a few hundred it passed with the two
# paths spelled differently, which made its EXACT-equality claim decorative. Do not
# "tidy" the scalar function into ** 2, and do not shrink that sweep.


def compute_ttc_pair(
    x_a: float, y_a: float, vx_a: float, vy_a: float, r_a: float,
    x_b: float, y_b: float, vx_b: float, vy_b: float, r_b: float
) -> float:
    """
    Compute TTC between a single pair of agents at one timestep.
    Uses a circular approximation for speed — see the derivation above.

    Args:
        x_a, y_a:   position of agent A
        vx_a, vy_a: velocity of agent A
        r_a:        effective radius of agent A
        x_b, y_b:   position of agent B
        vx_b, vy_b: velocity of agent B
        r_b:        effective radius of agent B

    Returns:
        TTC in seconds, or TTC_INFINITY if the circles never meet.
    """
    # relative position and velocity (B relative to A)
    dx = x_b - x_a
    dy = y_b - y_a
    dvx = vx_b - vx_a
    dvy = vy_b - vy_a

    # safe distance — sum of effective radii
    safe_dist = r_a + r_b

    a = dvx * dvx + dvy * dvy
    b = 2.0 * (dx * dvx + dy * dvy)
    c = dx * dx + dy * dy - safe_dist * safe_dist

    # already overlapping — unchanged from the previous implementation, and not
    # the branch B07 is about.
    if c <= 0:
        return 0.0

    # zero relative velocity: the distance is constant and currently positive.
    # a == 0 implies b == 0 (b is twice a dot product with the zero vector), so
    # there is no linear term to fall back on either.
    if a == 0:
        return TTC_INFINITY

    disc = b * b - 4.0 * a * c
    if disc < 0:
        # the paths never bring the circles within safe_dist of each other
        return TTC_INFINITY

    t = (-b - np.sqrt(disc)) / (2.0 * a)

    # A NEGATIVE SMALLER ROOT MEANS BOTH ROOTS ARE NEGATIVE — the contact happened
    # in the past and the agents are now separating. The roots cannot straddle zero
    # here: an upward parabola with f(0) = c > 0 has roots whose product is c/a > 0,
    # so they share a sign. Verified by brute force over 400k random configurations
    # as well as algebraically.
    if t < 0:
        return TTC_INFINITY

    return float(t)


def compute_ttc_all_pairs(
    states: np.ndarray,
    validity: np.ndarray,
    t: int
) -> np.ndarray:
    """
    Compute TTC for all agent pairs at a single timestep using NumPy broadcasting.
    This is the vectorized version — no Python loops over agent pairs.

    Solves the same quadratic as compute_ttc_pair, elementwise over an (N, N)
    matrix. The two implementations are held to EXACT equality by
    tests/test_danger_contract.py, so any change here must be mirrored there and
    spelled identically — see the `x * x` note in the block comment above.

    Everything is computed in float64. `states` is float32, and the discriminant
    b*b - 4ac suffers catastrophic cancellation near a grazing contact: in float32
    it can cross zero and flip a near-miss into a collision or back. The upcast also
    matches the float64 that scalar callers pass in, which is the other half of what
    makes exact parity achievable.

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

    # extract positions and velocities at timestep t, in float64 — see docstring
    pos = states[:, t, :2].astype(np.float64)    # shape (N, 2) — x, y
    vel = states[:, t, 2:4].astype(np.float64)   # shape (N, 2) — vx, vy
    valid = validity[:, t]                       # shape (N,)

    # effective radii for all agents — elementwise, no Python loop. This used to be
    # a list comprehension over zip(lengths, widths), which is a per-agent loop
    # inside the function whose entire reason for existing is that it has none.
    radii = compute_effective_radius(states[:, t, 5].astype(np.float64),
                                     states[:, t, 6].astype(np.float64))

    # relative positions and velocities — broadcasting to (N, N).
    # Entry [i, j] is i relative to j, which is the opposite labelling from
    # compute_ttc_pair's "B relative to A". It does not matter: negating d and v
    # together leaves every term below unchanged, and does so EXACTLY, since
    # (-x)*(-y) == x*y bit-for-bit in IEEE-754.
    dx = pos[:, None, 0] - pos[None, :, 0]    # shape (N, N)
    dy = pos[:, None, 1] - pos[None, :, 1]    # shape (N, N)
    dvx = vel[:, None, 0] - vel[None, :, 0]   # shape (N, N)
    dvy = vel[:, None, 1] - vel[None, :, 1]   # shape (N, N)

    # safe distances — sum of radii for each pair
    safe_dist = radii[:, None] + radii[None, :]  # shape (N, N)

    a = dvx * dvx + dvy * dvy                                  # shape (N, N)
    b = 2.0 * (dx * dvx + dy * dvy)                            # shape (N, N)
    c = dx * dx + dy * dy - safe_dist * safe_dist              # shape (N, N)
    disc = b * b - 4.0 * a * c                                 # shape (N, N)

    # Both guards below compute a value for EVERY cell and then mask, rather than
    # branching: np.sqrt of a negative and division by a zero `a` would each emit a
    # warning and produce nan/inf in cells the mask is about to discard anyway.
    disc_safe = np.where(disc >= 0, disc, 0.0)
    a_safe = np.where(a > 0, a, 1.0)
    root = (-b - np.sqrt(disc_safe)) / (2.0 * a_safe)

    ttc = np.full((N, N), TTC_INFINITY)

    # the collision branch — mirrors compute_ttc_pair's four conditions in order
    hits = (c > 0) & (a > 0) & (disc >= 0) & (root >= 0)
    ttc[hits] = root[hits]

    # already overlapping. Applied AFTER the hits mask so it wins outright, matching
    # the scalar function's early return.
    ttc[c <= 0] = 0.0

    # mask invalid agents
    valid_pair = valid[:, None] & valid[None, :]  # shape (N, N)
    ttc[~valid_pair] = TTC_INFINITY

    # diagonal — agent with itself. c = -safe_dist^2 < 0 there, so without this the
    # overlap branch above would report 0.0 for every agent against itself.
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

def compute_min_ttc_sdc(
    states: np.ndarray,
    validity: np.ndarray,
    sdc_index: int
) -> float:
    """
    Minimum TTC across timesteps, restricted to pairs that involve the SDC.

    This is the signal Phase 5 ranks on. compute_min_ttc_scenario (all pairs)
    saturates on real WOMD data — a dense urban scene has ~57 agents and any two
    passing within a few metres drives its scene-wide min TTC to zero, telling us
    nothing about whether the *self-driving car* was ever in danger. On the
    validation shard that was 100/100 scenarios. Phase 4 optimises SDC-vs-challenger
    collisions, so the filter feeding it must measure the same thing.

    Reuses compute_ttc_all_pairs so the pair math is defined once. The TTC matrix
    is symmetric, but both the SDC row and the SDC column are minimised over as a
    defensive measure (matching the validation notebook's SDC-only experiment).

    Args:
        states:     shape (N, T, 7)
        validity:   shape (N, T)
        sdc_index:  index of the self-driving car

    Returns:
        min_ttc: minimum SDC-involving TTC in seconds, or TTC_INFINITY if the
                 SDC is never in a closing pair (or sdc_index is out of range).
    """
    N, T = states.shape[0], states.shape[1]
    if not (0 <= sdc_index < N):
        return TTC_INFINITY

    min_ttc = TTC_INFINITY
    for t in range(T):
        ttc_matrix = compute_ttc_all_pairs(states, validity, t)
        row_min = ttc_matrix[sdc_index, :].min()
        col_min = ttc_matrix[:, sdc_index].min()
        min_ttc = min(min_ttc, row_min, col_min)

    return float(min_ttc)
