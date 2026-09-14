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


def _occupancy_visits(states: np.ndarray, validity: np.ndarray,
                      agent: int, conflict_zone) -> list:
    """
    Every SEPARATE visit `agent` makes to the conflict zone, as [(enter, exit), ...]
    in ascending time. Empty if the agent never occupies the zone.

    This replaces _occupancy_interval, which returned only the first and last
    occupied frame. That single span silently merged an agent that entered the zone,
    LEFT, and came back into one continuous occupancy it never had (audit B06). The
    audit's repro: A occupies frames 0, 1, 5, 6 and B occupies only frame 3. There is
    no frame where both are present, yet the merged spans (0, 6) and (3, 3) overlap,
    and PET was reported as -0.3 s — the signature of a genuine simultaneous
    encroachment, invented out of a gap.

    A VISIT IS A MAXIMAL RUN OF CONSECUTIVE FRAMES WHERE THE AGENT IS BOTH VALID AND
    INSIDE THE ZONE, and the "valid" half of that is not a stylistic choice. An
    unobserved frame breaks the run. The alternative — treating an invalid frame as
    "unknown, probably still here" and bridging it — reintroduces exactly the
    fictitious continuity this function exists to remove, just sourced from missing
    data instead of real departure. It is also what the audit's own fixture demands:
    its two visits are created by a VALIDITY gap (v[0] = [1,1,0,0,0,1,1]), not by the
    agent moving away, so bridging invalid frames fails B06 outright.

    That choice has a cost, stated rather than hidden: across a long unobserved
    stretch an agent that never actually left will be recorded as two visits, and the
    gap between them reported as a safe positive PET. That is the direction Batch 1
    already committed to in validity.has_interior_gap — a gap means consecutive valid
    frames are not consecutive in time, so two observations cannot be joined into one
    continuous thing. How often it happens on real data is measured by the validation
    notebook rather than assumed here.

    One pass over the trajectory, same cost as the single-span version it replaces:
    the Polygon construction per frame dominates, and that is unchanged.
    """
    T = states.shape[1]
    visits = []
    for t in range(T):
        inside = False
        if validity[agent, t]:
            box = Polygon(get_corners(states[agent, t, 0], states[agent, t, 1],
                                      states[agent, t, 4], states[agent, t, 5],
                                      states[agent, t, 6]))
            inside = box.intersects(conflict_zone)

        if not inside:
            continue
        if visits and visits[-1][1] == t - 1:
            visits[-1][1] = t          # extend the run in progress
        else:
            visits.append([t, t])      # a new, separate visit

    return [(int(enter), int(leave)) for enter, leave in visits]


def compute_pet_pair(
    states: np.ndarray,
    validity: np.ndarray,
    agent_a: int,
    agent_b: int,
    path_a=None,
    path_b=None,
) -> float:
    """
    Compute Post-Encroachment Time between two agents.

    PET is the time gap between one agent clearing the shared conflict zone and
    the other agent entering it. A small positive PET is a near-miss.

    EACH AGENT OCCUPIES THE ZONE AS A SET OF DISJOINT VISITS, NOT ONE SPAN, and the
    result is the minimum over every pairing of one A-visit with one B-visit:

        pet = min over (i, j) of  max(enter_Bj - exit_Ai, enter_Ai - exit_Bj) * DT

    The inner max is unchanged from the previous version and is what makes the
    result independent of which agent is passed as `agent_a`: whichever agent
    actually crossed first contributes the meaningful positive gap, and the other
    ordering contributes a negative number that only reflects the arbitrary a/b
    labelling. (Evaluating only `enter_b - exit_a` reported a spurious negative
    whenever `b` happened to cross first; on a real WOMD shard that was 56% of
    sampled crossing pairs, a third of which were safely sequenced.)

    The outer min over visit pairs is the audit B06 fix. With a single visit each it
    collapses to exactly the expression above, so nothing about the crossing-order
    correction is re-litigated — it is that formula applied to every pair of visits
    instead of to one merged span per agent.

    WHY MIN IS THE RIGHT AGGREGATOR, since this is the load-bearing claim:

      * Per pair, `max(...) <= 0` iff `enter_Bj <= exit_Ai` AND `enter_Ai <= exit_Bj`,
        which is exactly the condition for those two closed intervals to overlap. So
        a negative per-pair value means genuine simultaneous presence, with no gap in
        the logic.
      * Visits PARTITION each agent's presence in the zone, so if any instant has
        both agents present it falls inside exactly one visit of each, that pair
        overlaps, and the min is negative. If no such instant exists, no pair
        overlaps, every pair is non-negative, and so is the min. The guarantee that
        "negative means genuine overlap" therefore transfers exactly, not
        approximately.
      * A temporally distant, irrelevant visit pair cannot spuriously win the min:
        mismatched pairs produce one large positive and one large negative term, and
        the inner max always selects the large positive one. Distant pairs can only
        produce large uninteresting values, never small ones. The search space is
        safely oversized.

    Args:
        states:   shape (N, T, 7)
        validity: shape (N, T)
        agent_a:  index of one agent
        agent_b:  index of the other agent
        path_a:   optional pre-built swept polygon for agent_a. When a caller
                  compares one agent against many others (compute_min_pet_sdc),
                  passing that agent's polygon in avoids rebuilding it — a
                  unary_union of up to 91 boxes — once per pair.
        path_b:   optional pre-built swept polygon for agent_b.

    Returns:
        PET in seconds — the smallest gap over all visit pairings. Negative means
        the two agents were genuinely in the zone at the same time. PET_INFINITY if
        the agents' swept paths never overlap or if either agent never actually
        enters the spatial conflict zone.
    """
    # spatial conflict zone — where both agents' swept paths overlap
    if path_a is None:
        path_a = get_path_polygon(states, agent_a, validity)
    if path_b is None:
        path_b = get_path_polygon(states, agent_b, validity)

    if path_a is None or path_b is None:
        return PET_INFINITY

    conflict_zone = path_a.intersection(path_b)

    if conflict_zone.is_empty:
        return PET_INFINITY  # paths never cross spatially

    visits_a = _occupancy_visits(states, validity, agent_a, conflict_zone)
    visits_b = _occupancy_visits(states, validity, agent_b, conflict_zone)

    if not visits_a or not visits_b:
        # swept paths overlap, but at least one agent's box never actually
        # reaches the zone (thin slivers from the polygon intersection)
        return PET_INFINITY

    pet = min(
        max(enter_b - exit_a, enter_a - exit_b)
        for enter_a, exit_a in visits_a
        for enter_b, exit_b in visits_b
    ) * DT
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


def compute_min_pet_sdc(
    states: np.ndarray,
    validity: np.ndarray,
    sdc_index: int,
    max_pairs: int = None
) -> float:
    """
    Minimum PET across pairs that involve the SDC.

    This is the signal Phase 5 ranks on. compute_min_pet_scenario (all pairs,
    capped at 50) has two problems on real data that this avoids:

      * the cap is spent walking agent 0's pairs first, so in a scene with 51+
        agents it never looks at any pair not involving agent 0 — which is
        rarely the SDC.
      * like TTC, an all-pairs minimum answers "did any two agents nearly
        collide", not "was the SDC in danger".

    No pair cap by default: there are only N-1 SDC pairs and N is at most a few
    hundred, versus the combinatorial blow-up the all-pairs cap defends against.
    An explicit max_pairs is honoured if a caller wants one.

    The SDC's swept polygon is built once and passed into every compute_pet_pair
    call rather than rebuilt per pair.

    Args:
        states:     shape (N, T, 7)
        validity:   shape (N, T)
        sdc_index:  index of the self-driving car
        max_pairs:  optional cap on SDC pairs checked (default: no cap)

    Returns:
        min_pet: minimum SDC-involving PET in seconds, or PET_INFINITY if the
                 SDC never shares a conflict zone with another agent (or
                 sdc_index is out of range, or the SDC has no valid timesteps).
    """
    N = states.shape[0]
    if not (0 <= sdc_index < N):
        return PET_INFINITY

    sdc_path = get_path_polygon(states, sdc_index, validity)
    if sdc_path is None:
        return PET_INFINITY

    min_pet = PET_INFINITY
    pairs_checked = 0
    for j in range(N):
        if j == sdc_index:
            continue
        if max_pairs is not None and pairs_checked >= max_pairs:
            break
        pet = compute_pet_pair(states, validity, sdc_index, j, path_a=sdc_path)
        min_pet = min(min_pet, pet)
        pairs_checked += 1

    return float(min_pet)
