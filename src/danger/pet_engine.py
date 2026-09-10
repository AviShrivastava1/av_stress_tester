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


def _occupancy_interval(states: np.ndarray, validity: np.ndarray,
                        agent: int, conflict_zone) -> tuple:
    """
    First and last timestep at which `agent`'s bounding box intersects the
    conflict zone.

    Returns (first, last), or (-1, -1) if the agent never occupies the zone.
    One pass over the trajectory yields both boundaries, so callers do not scan
    the agent twice.
    """
    T = states.shape[1]
    first = last = -1
    for t in range(T):
        if not validity[agent, t]:
            continue
        box = Polygon(get_corners(states[agent, t, 0], states[agent, t, 1],
                                  states[agent, t, 4], states[agent, t, 5],
                                  states[agent, t, 6]))
        if box.intersects(conflict_zone):
            if first == -1:
                first = t
            last = t
    return first, last


def compute_pet_pair(
    states: np.ndarray,
    validity: np.ndarray,
    agent_a: int,
    agent_b: int
) -> float:
    """
    Compute Post-Encroachment Time between two agents.

    PET is the time gap between one agent clearing the shared conflict zone and
    the other agent entering it. A small positive PET is a near-miss.

    Both crossing orders are evaluated and the larger (safer) gap is returned:

        pet = max(enter_b - exit_a, enter_a - exit_b) * DT

    Whichever agent actually crossed first contributes the meaningful positive
    gap; the other ordering contributes a negative number that only reflects the
    arbitrary a/b labelling. Taking the max makes the result independent of which
    agent is passed as `agent_a` (see `compute_pet_pair(a, b) == compute_pet_pair(b, a)`).

    A negative result means BOTH terms are negative — `enter_b < exit_a` and
    `enter_a < exit_b` — which is exactly the condition for the two agents'
    occupancy intervals to overlap, i.e. genuine simultaneous presence in the
    conflict zone. (The previous implementation computed only `enter_b - exit_a`
    and so reported a spurious negative whenever `b` happened to cross first; on
    a real WOMD shard that was 56% of sampled crossing pairs, a third of which
    were safely sequenced.)

    Args:
        states:   shape (N, T, 7)
        validity: shape (N, T)
        agent_a:  index of one agent
        agent_b:  index of the other agent

    Returns:
        PET in seconds. PET_INFINITY if the agents' swept paths never overlap or
        if either agent never actually enters the spatial conflict zone.
    """
    # spatial conflict zone — where both agents' swept paths overlap
    path_a = get_path_polygon(states, agent_a, validity)
    path_b = get_path_polygon(states, agent_b, validity)

    if path_a is None or path_b is None:
        return PET_INFINITY

    conflict_zone = path_a.intersection(path_b)

    if conflict_zone.is_empty:
        return PET_INFINITY  # paths never cross spatially

    enter_a, exit_a = _occupancy_interval(states, validity, agent_a, conflict_zone)
    enter_b, exit_b = _occupancy_interval(states, validity, agent_b, conflict_zone)

    if enter_a == -1 or enter_b == -1:
        # swept paths overlap, but at least one agent's box never actually
        # reaches the zone (thin slivers from the polygon intersection)
        return PET_INFINITY

    pet = max(enter_b - exit_a, enter_a - exit_b) * DT
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