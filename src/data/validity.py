"""
validity.py — one definition of "first valid observation", shared by every module
that needs it.

WOMD tracks do not all start at frame zero. `ScenarioParser.get_agent_states()`
zero-fills every timestep an agent was not observed, so frame 0 of a late-appearing
agent is not "the agent at rest" — it is *no data at all*, indistinguishable from a
real reading of (0, 0) with zero length and zero width.

Three modules used to answer "where does this agent start?" independently, and they
did not agree: `export_geometry` took the first valid frame, `PerturbationSpace`
took the first valid frame for its initial state but frame 0 for its dimensions, and
`autograd_optimizer` took frame 0 for everything. That disagreement is audit findings
B01 and B02. One definition, imported everywhere, is the fix.

No waymo import here, deliberately: `src/data/__init__.py` is empty, so importing
this module never drags in `parser.py`'s module-level `scenario_pb2` and this stays
usable on machines without the Waymo package.
"""

import numpy as np


def valid_timesteps(validity: np.ndarray, agent_idx: int) -> np.ndarray:
    """
    Indices of the timesteps where this agent was actually observed.

    Args:
        validity:  (N, T) bool
        agent_idx: which agent

    Returns:
        1-D int array of global frame indices, possibly empty.
    """
    return np.where(validity[agent_idx])[0]


def first_valid_index(validity: np.ndarray, agent_idx: int) -> int:
    """
    The first global frame at which this agent was observed.

    Raises ValueError if the agent is never valid. That is deliberate: the previous
    implementation returned 0 for a never-observed agent, which reads as "starts at
    the beginning" and silently hands zero-filled garbage to the simulator. An agent
    with no observations has no initial state, and the caller must say so out loud
    rather than simulate a fiction (the project's no-fallback rule).
    """
    ts = valid_timesteps(validity, agent_idx)
    if len(ts) == 0:
        raise ValueError(
            f"agent {agent_idx} has no valid timesteps; it has no observed initial "
            f"state and cannot be simulated"
        )
    return int(ts[0])


def has_interior_gap(validity: np.ndarray, agent_idx: int) -> bool:
    """
    True if the agent's observations are interrupted — valid, then invalid, then
    valid again.

    A leading or trailing run of invalid frames is not a gap: the agent simply had
    not appeared yet, or had already left. Only a hole *between* observations breaks
    the assumption that consecutive valid frames are consecutive in time, which is
    what makes a replay across the hole join two incompatible observations.

    Recorded rather than special-cased: a rollout across an interior gap produces
    large baseline replay error, which PerturbationSpace's fidelity gate already
    refuses. This flag exists so the real-shard run can measure how often that
    happens before anyone invents a bespoke rule for it.
    """
    ts = valid_timesteps(validity, agent_idx)
    if len(ts) < 2:
        return False
    return bool(np.any(np.diff(ts) > 1))
