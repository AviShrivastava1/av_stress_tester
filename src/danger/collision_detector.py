import numpy as np
from shapely.geometry import Polygon


def get_corners(x: float, y: float, theta: float, length: float, width: float) -> np.ndarray:
    """
    Compute the 4 corners of an agent's oriented bounding box in global coordinates.

    The car's local frame has its center at origin, pointing along the x-axis.
    Corners in local frame:
        front-right: (+L/2, +W/2)
        front-left:  (+L/2, -W/2)
        rear-left:   (-L/2, -W/2)
        rear-right:  (-L/2, +W/2)

    We rotate these by the agent's heading theta and translate to (x, y).

    Args:
        x, y:   agent center position
        theta:  heading angle in radians
        length: agent length (meters)
        width:  agent width (meters)

    Returns:
        corners: shape (4, 2) — four (x, y) corner points in global frame
    """
    L, W = length / 2, width / 2

    # corners in local frame (car points along x-axis)
    local_corners = np.array([
        [ L,  W],   # front right
        [ L, -W],   # front left
        [-L, -W],   # rear left
        [-L,  W],   # rear right
    ])

    # rotation matrix for heading theta
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ])

    # rotate corners and translate to global frame
    global_corners = (R @ local_corners.T).T + np.array([x, y])
    return global_corners


def corners_to_polygon(corners: np.ndarray) -> Polygon:
    """Convert (4, 2) corner array to a Shapely Polygon."""
    return Polygon(corners)


def check_collision(
    x_a: float, y_a: float, theta_a: float, length_a: float, width_a: float,
    x_b: float, y_b: float, theta_b: float, length_b: float, width_b: float
) -> bool:
    """
    Check if two agents' oriented bounding boxes overlap at a single timestep.

    Uses Shapely polygon intersection — robust implementation of SAT under the hood.

    Args:
        x_a, y_a, theta_a, length_a, width_a: agent A state
        x_b, y_b, theta_b, length_b, width_b: agent B state

    Returns:
        True if the bounding boxes overlap (collision), False otherwise
    """
    corners_a = get_corners(x_a, y_a, theta_a, length_a, width_a)
    corners_b = get_corners(x_b, y_b, theta_b, length_b, width_b)

    poly_a = corners_to_polygon(corners_a)
    poly_b = corners_to_polygon(corners_b)

    return poly_a.intersects(poly_b)


def check_collision_trajectory(
    states: np.ndarray,
    validity: np.ndarray,
    agent_a: int,
    agent_b: int
) -> tuple:
    """
    Check for collision between two agents across their full trajectories.

    Args:
        states:   shape (N, T, 7) from ScenarioParser
        validity: shape (N, T) validity flags
        agent_a:  index of first agent
        agent_b:  index of second agent

    Returns:
        (collision_occurred, first_collision_timestep)
        first_collision_timestep is -1 if no collision
    """
    T = states.shape[1]

    for t in range(T):
        # skip if either agent is invalid at this timestep
        if not validity[agent_a, t] or not validity[agent_b, t]:
            continue

        x_a, y_a     = states[agent_a, t, 0], states[agent_a, t, 1]
        theta_a       = states[agent_a, t, 4]
        length_a      = states[agent_a, t, 5]
        width_a       = states[agent_a, t, 6]

        x_b, y_b     = states[agent_b, t, 0], states[agent_b, t, 1]
        theta_b       = states[agent_b, t, 4]
        length_b      = states[agent_b, t, 5]
        width_b       = states[agent_b, t, 6]

        if check_collision(x_a, y_a, theta_a, length_a, width_a,
                          x_b, y_b, theta_b, length_b, width_b):
            return True, t

    return False, -1


def check_any_collision(
    states: np.ndarray,
    validity: np.ndarray,
    sdc_idx: int
) -> tuple:
    """
    Check if the SDC collides with any other agent across the full trajectory.
    This is the key function for Phase 4 — after perturbation, did we cause a crash?

    Args:
        states:   shape (N, T, 7)
        validity: shape (N, T)
        sdc_idx:  index of the self-driving car

    Returns:
        (collision_occurred, colliding_agent_idx, first_collision_timestep)
    """
    N = states.shape[0]

    for agent_idx in range(N):
        if agent_idx == sdc_idx:
            continue

        collision, t = check_collision_trajectory(
            states, validity, sdc_idx, agent_idx
        )

        if collision:
            return True, agent_idx, t

    return False, -1, -1