import numpy as np
from src.danger.collision_detector import get_corners
from shapely.geometry import Polygon, LineString


PET_INFINITY = 999.0    # sentinel for pairs whose paths never cross
PET_THRESHOLD = 2.0     # seconds — pre-filter threshold for Phase 5
DT = 0.1                # seconds per timestep


def get_path_polygon(states: np.ndarray, agent_idx: int,
                     validity: np.ndarray) -> Polygon:
    """
    Build a Shapely polygon representing the full swept area of an agent
    across its entire trajectory. This is the union of all its bounding
    boxes at every valid timestep.

    Used to find whether two agents' paths ever cross spatially.
    """
    from shapely.ops import unary_union

    boxes = []
    T = states.shape[1]

    for t in range(T):
        if not validity[agent_idx, t]:
            continue

        x, y     = states[agent_idx, t, 0], states[agent_idx, t, 1]
        theta    = states[agent_idx, t, 4]
        length   = states[agent_idx, t, 5]
        width    = states[agent_idx, t, 6]

        corners = get_corners(x, y, theta, length, width)
        boxes.append(Polygon(corners))

    if not boxes:
        return None

    return unary_union(boxes)


def compute_pet_pair(
    states: np.ndarray,
    validity: np.ndarray,
    agent_a: int,
    agent_b: int
) -> float:
    """
    Compute Post-Encroachment Time between two agents.

    PET = time between when the first agent leaves the conflict zone
    and when the second agent enters it.

    Small PET = near-miss. Negative PET = actual collision (simultaneous occupancy).

    Args:
        states:   shape (N, T, 7)
        validity: shape (N, T)
        agent_a:  index of first agent
        agent_b:  index of second agent

    Returns:
        PET in seconds. PET_INFINITY if paths never cross.
    """
    # find the spatial conflict zone — where both agents' swept paths overlap
    path_a = get_path_polygon(states, agent_a, validity)
    path_b = get_path_polygon(states, agent_b, validity)

    if path_a is None or path_b is None:
        return PET_INFINITY

    conflict_zone = path_a.intersection(path_b)

    if conflict_zone.is_empty:
        return PET_INFINITY  # paths never cross spatially

    T = states.shape[1]

    # find last timestep agent_a occupies the conflict zone
    t_exit_a = -1
    for t in range(T):
        if not validity[agent_a, t]:
            continue
        x, y   = states[agent_a, t, 0], states[agent_a, t, 1]
        theta  = states[agent_a, t, 4]
        length = states[agent_a, t, 5]
        width  = states[agent_a, t, 6]
        box_a  = Polygon(get_corners(x, y, theta, length, width))
        if box_a.intersects(conflict_zone):
            t_exit_a = t

    # find first timestep agent_b occupies the conflict zone
    t_enter_b = -1
    for t in range(T):
        if not validity[agent_b, t]:
            continue
        x, y   = states[agent_b, t, 0], states[agent_b, t, 1]
        theta  = states[agent_b, t, 4]
        length = states[agent_b, t, 5]
        width  = states[agent_b, t, 6]
        box_b  = Polygon(get_corners(x, y, theta, length, width))
        if box_b.intersects(conflict_zone):
            t_enter_b = t
            break

    if t_exit_a == -1 or t_enter_b == -1:
        return PET_INFINITY

    # PET = time gap between first agent leaving and second agent entering
    pet = (t_enter_b - t_exit_a) * DT

    # if pet < 0, agents were simultaneously in conflict zone — actual collision
    return float(pet)


def compute_min_pet_scenario(
    states: np.ndarray,
    validity: np.ndarray,
    max_pairs: int = 50
) -> float:
    """
    Compute minimum PET across all agent pairs in a scenario.
    Limits to max_pairs for computational efficiency.

    Args:
        states:    shape (N, T, 7)
        validity:  shape (N, T)
        max_pairs: maximum number of pairs to check

    Returns:
        min_pet: scalar minimum PET in seconds
    """
    N = states.shape[0]
    min_pet = PET_INFINITY
    pairs_checked = 0

    for i in range(N):
        for j in range(i+1, N):
            if pairs_checked >= max_pairs:
                return float(min_pet)

            pet = compute_pet_pair(states, validity, i, j)
            min_pet = min(min_pet, pet)
            pairs_checked += 1

    return float(min_pet)