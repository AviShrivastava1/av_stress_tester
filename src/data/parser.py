import numpy as np
from waymo_open_dataset.protos import scenario_pb2


class ScenarioParser:
    """
    Takes raw serialized bytes from ShardLoader and decodes them into
    clean NumPy arrays ready for the physics engine and danger computation.
    """

    def __init__(self, raw_bytes: bytes):
        # deserialize raw bytes into a Scenario protobuf object
        self.scenario = scenario_pb2.Scenario()
        self.scenario.ParseFromString(raw_bytes)

    def get_scenario_id(self) -> str:
        """unique string id for this scenario"""
        return self.scenario.scenario_id

    def get_agent_states(self) -> np.ndarray:
        """
        Extract all agent states into a 3D NumPy array.

        Returns:
            states: shape (N, T, 7) where
                N = number of agents
                T = number of timesteps (91 for a 9-second scenario)
                7 features = [x, y, vx, vy, heading, length, width]
        """
        tracks = self.scenario.tracks
        N = len(tracks)
        T = 91

        # initialize with zeros — invalid timesteps stay zero
        states = np.zeros((N, T, 7), dtype=np.float32)

        for i, track in enumerate(tracks):
            for t, state in enumerate(track.states):
                if t >= T:
                    break
                states[i, t, 0] = state.center_x    # x position
                states[i, t, 1] = state.center_y    # y position
                states[i, t, 2] = state.velocity_x  # x velocity
                states[i, t, 3] = state.velocity_y  # y velocity
                states[i, t, 4] = state.heading      # heading angle (radians)
                states[i, t, 5] = state.length       # agent length (meters)
                states[i, t, 6] = state.width        # agent width (meters)

        return states

    def get_agent_validity(self) -> np.ndarray:
        """
        Extract validity flags — whether each agent exists at each timestep.
        An agent not yet on the road or that has left is marked invalid.

        Returns:
            validity: shape (N, T) — True if agent present, False otherwise
        """
        tracks = self.scenario.tracks
        N = len(tracks)
        T = 91

        validity = np.zeros((N, T), dtype=bool)

        for i, track in enumerate(tracks):
            for t, state in enumerate(track.states):
                if t >= T:
                    break
                validity[i, t] = state.valid

        return validity

    def get_agent_types(self) -> np.ndarray:
        """
        Extract agent type for each track.
        1 = vehicle, 2 = pedestrian, 3 = cyclist

        Returns:
            types: shape (N,) — one integer type per agent
        """
        return np.array(
            [track.object_type for track in self.scenario.tracks],
            dtype=np.int32
        )

    def get_sdc_index(self) -> int:
        """
        Returns the index of the self-driving car in the tracks array.
        This is the Waymo vehicle whose behavior we are stress-testing.
        """
        return self.scenario.sdc_track_index