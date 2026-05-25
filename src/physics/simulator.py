import numpy as np
from src.physics.bicycle_model import (
    bicycle_step, invert_bicycle, extract_state_from_womd as bicycle_extract,
    get_wheelbase
)
from src.physics.linear_model import (
    linear_step, invert_linear, extract_state_from_womd as linear_extract
)

# agent type constants from WOMD
TYPE_VEHICLE    = 1
TYPE_PEDESTRIAN = 2
TYPE_CYCLIST    = 3


class ForwardSimulator:
    """
    Simulates an agent's trajectory forward in time given control inputs.
    Dispatches to the correct kinematic model based on agent type.
    Vehicles use the bicycle model. Pedestrians and cyclists use the linear model.
    """

    def __init__(self, agent_type: int, agent_length: float):
        self.agent_type = agent_type
        self.wheelbase  = get_wheelbase(agent_length)

    def step(self, state: np.ndarray, control: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        Propagate state forward by one timestep.

        Args:
            state:   [x, y, theta, v] for vehicles OR [x, y, vx, vy] for peds/cyclists
            control: [delta, a] for vehicles OR [ax, ay] for peds/cyclists
            dt:      timestep duration

        Returns:
            next_state: same shape as state
        """
        if self.agent_type == TYPE_VEHICLE:
            return bicycle_step(state, control, self.wheelbase, dt)
        else:
            return linear_step(state, control, dt)

    def simulate(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        dt: float = 0.1
    ) -> np.ndarray:
        """
        Simulate a full trajectory from initial state given a sequence of controls.

        Args:
            initial_state: state at t=0
            controls:      shape (T-1, 2) — one control per timestep transition
            dt:            timestep duration

        Returns:
            trajectory: shape (T, state_dim) — states at every timestep including t=0
        """
        T = len(controls) + 1
        state_dim = len(initial_state)
        trajectory = np.zeros((T, state_dim), dtype=np.float32)
        trajectory[0] = initial_state

        for t in range(len(controls)):
            trajectory[t+1] = self.step(trajectory[t], controls[t], dt)

        return trajectory


class TrajectoryInverter:
    """
    Recovers control sequences from observed WOMD trajectories.
    Used in Phase 4 to get the baseline controls before perturbation.
    """

    def __init__(self, agent_type: int, agent_length: float):
        self.agent_type = agent_type
        self.wheelbase  = get_wheelbase(agent_length)

    def invert(
        self,
        states: np.ndarray,
        agent_idx: int,
        validity: np.ndarray,
        dt: float = 0.1
    ) -> np.ndarray:
        """
        Recover control inputs from an observed trajectory.

        Args:
            states:    shape (N, T, 7) from ScenarioParser
            agent_idx: which agent to invert
            validity:  shape (N, T) validity flags
            dt:        timestep duration

        Returns:
            controls: shape (T-1, 2) — one control per timestep transition
        """
        T = states.shape[1]
        controls = np.zeros((T-1, 2), dtype=np.float32)

        for t in range(T-1):
            # skip invalid timesteps — use zero control
            if not validity[agent_idx, t] or not validity[agent_idx, t+1]:
                continue

            if self.agent_type == TYPE_VEHICLE:
                state_t    = bicycle_extract(states, agent_idx, t)
                state_next = bicycle_extract(states, agent_idx, t+1)
                controls[t] = invert_bicycle(state_t, state_next, self.wheelbase, dt)
            else:
                state_t    = linear_extract(states, agent_idx, t)
                state_next = linear_extract(states, agent_idx, t+1)
                controls[t] = invert_linear(state_t, state_next, dt)

        return controls