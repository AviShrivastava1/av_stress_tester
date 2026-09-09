"""
models.py — Phase 6.

Pydantic response models: the typed contract with the Phase 7 frontend, and the
source of the OpenAPI schema at /docs. Declaring these explicitly (rather than
returning raw dicts) means the frontend can generate types from the schema, and a
field that changes shape breaks loudly here instead of quietly in a chart.

THE CRITICAL DETAIL: a NULL min_perturbation means two different things
-----------------------------------------------------------------------
In `scenario_scores`, `min_perturbation` is NULL in two completely different
situations, and they are distinguished by `stress_tested_at`:

    stress_tested_at IS NULL
        Phase 4 has not run on this scenario yet. We do not know anything about
        its perturbation distance. "Unknown".

    stress_tested_at IS NOT NULL, min_perturbation IS NULL
        Phase 4 DID run and found NO collision anywhere within the perturbation
        bounds. The scenario is "robustly safe" — a genuine, informative result,
        and arguably the most reassuring one a safety audit can produce.

Those are opposite meanings sharing one NULL. Leaving the frontend to infer which
is which from a second column is exactly how a dashboard ends up displaying "no
collision found" for scenarios that were never tested. So the API resolves it here,
into two explicit booleans — `stress_tested` and `robustly_safe` — and the frontend
never has to reason about it.

This is also precisely why Phase 5 stored NULL rather than infinity for the
robustly-safe case (see db.py's update_stress_results): **JSON has no infinity
literal**. `float('inf')` is not valid JSON — serializers either emit the
non-standard bare token `Infinity`, which a strict parser rejects, or crash. NULL
is the only honest representation that survives the trip to a browser.

One more wrinkle worth knowing: `update_stress_results` only writes rows for
results whose status is 'ok'. A scenario whose Phase 4 pass ended in
'no_challenger' or 'error' never gets `stress_tested_at` set, so it reads through
this API as not-yet-tested. That is the truthful answer — we have no stress-test
result for it — but it does mean `stress_tested=False` covers both "not attempted"
and "attempted and could not run".
"""

from pydantic import BaseModel


# The 4-D perturbation vector from src/optimization/perturbation_space.py, in order.
# Shipped alongside the raw delta so the frontend can label the axes without
# hardcoding the parameterization — if Phase 4's space ever changes, the labels
# travel with the data instead of going stale in a different repo.
DELTA_LABELS = [
    "initial speed (m/s)",
    "initial heading (rad)",
    "acceleration bias (m/s^2)",
    "steering bias (rad)",
]


class ScenarioSummary(BaseModel):
    """One row of the ranked list."""

    scenario_id: str
    shard: str | None = None
    n_agents: int | None = None
    min_ttc: float | None = None
    min_pet: float | None = None
    fragility_score: float
    min_perturbation: float | None = None
    collision_timestep: int | None = None
    stress_method: str | None = None

    # Resolved from (stress_tested_at, min_perturbation) — see module docstring.
    stress_tested: bool
    robustly_safe: bool


class ScenarioDetail(ScenarioSummary):
    """A single scenario, including the raw perturbation vector."""

    delta: list[float] | None = None


class ScenarioPage(BaseModel):
    """
    One page of the ranked list.

    `next_cursor` is opaque: clients pass it back verbatim and must never construct
    or edit one. See routes.py for why its precision is load-bearing.
    """

    items: list[ScenarioSummary]
    next_cursor: str | None = None
    limit: int


class AgentTrack(BaseModel):
    """
    One agent's path through space and time.

    `path`, `timesteps` and `headings` are index-aligned: element i of each
    describes the same vertex. Coordinates are LOCAL PLANAR METRES, not lon/lat.
    """

    agent_idx: int
    agent_type: int | None = None
    is_sdc: bool
    length_m: float | None = None
    width_m: float | None = None
    path: list[list[float]]       # [[x, y], ...]
    timesteps: list[float]        # the M ordinate — which frame each vertex is
    headings: list[float]         # radians; needed to draw a rotated box, not a dot


class TrajectoryResponse(BaseModel):
    """Every agent's logged path for one scenario. Empty if Pass 3 has not run."""

    scenario_id: str
    agents: list[AgentTrack]


class PerturbedResponse(BaseModel):
    """
    The Phase 4 answer, made drawable: the challenger's logged path next to its
    minimally-perturbed one.

    Every field except scenario_id and delta_labels is nullable, because
    "this scenario has not been stress-tested yet" is a normal state that returns
    200, not an error.
    """

    scenario_id: str
    target_idx: int | None = None
    delta: list[float] | None = None
    delta_labels: list[str] = DELTA_LABELS
    min_perturbation: float | None = None
    collision_timestep: int | None = None
    baseline: AgentTrack | None = None
    perturbed: AgentTrack | None = None


class StatsResponse(BaseModel):
    """Corpus-level summary for the dashboard header."""

    total_scenarios: int
    stress_tested: int
    collisions_found: int
    robustly_safe: int
    with_geometry: int
    fragility_min: float | None = None
    fragility_max: float | None = None
    fragility_mean: float | None = None


class HealthResponse(BaseModel):
    """
    Liveness plus a real database round-trip.

    `postgis` is None when the extension is absent — that is degraded, not dead:
    the scores endpoints work fine without it, only geometry is unavailable.
    """

    status: str
    database: str
    postgis: str | None = None
