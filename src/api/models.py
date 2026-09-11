"""
models.py — Phase 6.

Pydantic response models: the typed contract with the Phase 7 frontend, and the
source of the OpenAPI schema at /docs. Declaring these explicitly (rather than
returning raw dicts) means the frontend can generate types from the schema, and a
field that changes shape breaks loudly here instead of quietly in a chart.

THE CRITICAL DETAIL: a NULL min_perturbation is not one state, and not two
--------------------------------------------------------------------------
An earlier version of this docstring said `min_perturbation` was NULL in exactly
two situations — "not tested yet" and "tested, robustly safe" — and that the second
was "a genuine, informative result, and arguably the most reassuring one a safety
audit can produce". **That was wrong, and it was the most consequential wrong thing
in this file** (audit B04). It is corrected here rather than quietly deleted,
because the mistake is instructive: the API was inferring a safety conclusion from
the ABSENCE of a number.

A NULL min_perturbation is consistent with at least four different histories, and no
combination of NULLs distinguishes them. So the pass now records what it concluded,
and the API reports that instead of inferring.

`stress_outcome` — what the run that PRODUCED THE STORED RESULT concluded. Only three
values are reachable here, because the other two produce no result to describe:

    NULL                              No verified result. Either the scenario was
                                      never searched, or written before this column
                                      existed — genuinely unknown, and NOT to be
                                      back-inferred from which columns are NULL.

    'collision_found'                 A collision verified by exact SAT geometry at a
                                      known delta. min_perturbation is set.

    'no_collision_found'              A search ran to completion and found no
                                      collision WITHIN ITS BUDGET AND BOUNDS. This is
                                      a fact about the search, not about the scenario.

`last_attempt_outcome` — what the MOST RECENT pass concluded. All five are reachable,
adding to the two above:

    'replay_infeasible'               No search ran. The scenario's own
                                      zero-perturbation replay was not faithful
                                      enough to measure a perturbation against
                                      (Batch 1, audit B03). Emphatically NOT the same
                                      as "came back clean".

    'no_challenger' / 'error'         Nothing to perturb, or the pass raised.

WHY `robustly_safe` IS EFFECTIVELY ALWAYS FALSE
-----------------------------------------------
It is still exposed, and it is now derived from an explicit
`search_certifies_infeasibility` column that **no code path sets to TRUE**. That is
not an oversight — it is the finding. Establishing "no perturbation within these
bounds causes a collision" requires a method that can certify infeasibility.
Differential Evolution cannot: it is a stochastic global optimizer whose termination
test measures population spread, not the absence of a feasible point. And
`_stress_one` searches ONE challenger, chosen by a nearest-distance heuristic, out of
however many the scene contains.

So `no_collision_found` is the honest field, and the one a dashboard should show.
`robustly_safe` is reserved for a method that earns it.

`stress_tested` means a verified search result EXISTS on this row. It survives a later
pass that could not reproduce it, because the result itself survives — see the
column-ownership comment in `db.update_stress_results`. `stress_attempted` means the
most recent pass reached the scenario at all, which is what makes `replay_infeasible`
visible rather than looking untested.

TWO OUTCOMES, BECAUSE A ROW CAN DESCRIBE TWO DIFFERENT RUNS
-----------------------------------------------------------
`stress_outcome` describes the run that produced the stored result — it always matches
`min_perturbation`, `delta` and `stress_run_id`, by construction.

`last_attempt_outcome` describes the most recent pass, whatever it concluded.

They are the same value almost always, and differ exactly when a later pass failed to
reproduce an earlier success: a stricter `max_baseline_drift`, a transient error, or an
input change that left no challenger. In that case the verified result is deliberately
NOT destroyed (see the column-ownership comment in `db.update_stress_results`) — it
cost thousands of DE evaluations, the refusal did not supersede it, and erasing it
would also erase the difference between "we re-verified this is no longer true" and
"we failed to check this time".

One column cannot hold both honestly. A row reading `stress_outcome='replay_infeasible'`
next to a real `min_perturbation` asserts something false by juxtaposition — the
`robustly_safe` defect wearing different field names. Splitting them makes the row
coherent by construction rather than by a careful reader cross-checking run ids.

So: `stress_outcome` tells you what the stored result IS. `last_attempt_outcome` tells
you what just happened.

This is also precisely why Phase 5 stored NULL rather than infinity for the
robustly-safe case (see db.py's update_stress_results): **JSON has no infinity
literal**. `float('inf')` is not valid JSON — serializers either emit the
non-standard bare token `Infinity`, which a strict parser rejects, or crash. NULL
is the only honest representation that survives the trip to a browser.

That wrinkle is now gone: `update_stress_results` used to write rows only for
results whose status was 'ok', so a pass ending in 'no_challenger' or 'error' left no
trace and read back as not-yet-tested. The old docstring called that "the truthful
answer". It was survivable while the only silent outcomes were "nothing to perturb"
and "it crashed"; it stopped being survivable when Batch 1 added `replay_infeasible`,
because a scenario REFUSED for untrustworthy replay would have been indistinguishable
from one the pass had simply never reached. Every outcome is persisted now.

MIN_TTC / MIN_PET ARE SDC-RESTRICTED (Phase 3 rework)
----------------------------------------------------
`min_ttc` and `min_pet` are the minimum over pairs that involve the self-driving
car — not the whole scene. `fragility_score` is computed from those. The
scene-wide "how crowded was this" values still exist in `scenario_scores` as
`min_ttc_all_pairs` / `min_pet_all_pairs` but are not exposed here yet; add them to
`ScenarioSummary` and `_SCORE_COLUMNS` when the dashboard needs a density view.
This distinction is load-bearing: on real WOMD data the all-pairs minimum TTC was
0.0 for 100/100 scenarios (any two of ~57 agents passing close saturates it),
which is why ranking moved to the SDC-restricted signal.
"""

from pydantic import BaseModel, Field


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
    min_ttc: float | None = Field(
        default=None,
        description="Minimum time-to-collision (s) over pairs involving the SDC. "
                    "999.0 sentinel = the SDC is never in a closing pair.",
    )
    min_pet: float | None = Field(
        default=None,
        description="Minimum post-encroachment time (s) over pairs involving the "
                    "SDC. Negative = the SDC and another agent genuinely occupied "
                    "a conflict zone simultaneously. 999.0 = no shared conflict zone.",
    )
    fragility_score: float
    min_perturbation: float | None = None
    collision_timestep: int | None = None
    stress_method: str | None = None

    # Resolved from stress_outcome — see module docstring for why this is reported
    # rather than inferred from which columns happen to be NULL.
    stress_tested: bool
    stress_attempted: bool
    stress_outcome: str | None
    last_attempt_outcome: str | None
    challengers_total: int | None
    challengers_searched: int | None
    # A completed search found nothing within its budget and bounds.
    no_collision_found: bool
    # A safety certificate. Always False today; see the module docstring.
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
    no_collision_found: int
    replay_infeasible: int
    no_challenger: int
    stress_errors: int
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
