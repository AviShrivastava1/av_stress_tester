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

`last_attempt_outcome` — what the MOST RECENT pass concluded. All six are reachable,
adding to the two above:

    'replay_infeasible'               No search ran. The scenario's own
                                      zero-perturbation replay was not faithful
                                      enough to measure a perturbation against
                                      (Batch 1, audit B03). Emphatically NOT the same
                                      as "came back clean".

    'heading_blend_singularity'       No search ran. The scenario's baseline replay
                                      WAS faithful (unlike replay_infeasible above) —
                                      but its logged/derived heading blend sat close
                                      enough to the antipodal degenerate point
                                      (perturbation_space.py::_linear_heading) that a
                                      vanishingly small perturbation could flip the
                                      challenger's heading to an essentially
                                      arbitrary orientation (independent review,
                                      2026-09-24). A separate value from
                                      replay_infeasible, not a sub-case of it — the
                                      two are refused for genuinely different
                                      reasons, and collapsing them would assert a
                                      replay-fidelity problem that was never
                                      measured.

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

This is also precisely why Phase 5 stored NULL rather than infinity for a search
that COMPLETED AND FOUND NOTHING (see db.py's update_stress_results): **JSON has no
infinity literal**. `float('inf')` is not valid JSON — serializers either emit the
non-standard bare token `Infinity`, which a strict parser rejects, or crash. NULL
is the only honest representation that survives the trip to a browser.

(That clause used to read "for the robustly-safe case", which is the very conflation
the forty lines above it exist to undo — a completed search named after a conclusion
it does not support. It survived two audits because it is hyphenated and both sweeps
looked for the spaced form; found by the exhaustive sweep in Batch 10, audit R08/A09.)

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

from datetime import datetime

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

# The linear model's parameterization, for pedestrian and cyclist challengers
# (audit B13). A delta produced by linear_step is [dvx0, dvy0, dax_bias, day_bias] —
# four components with different meanings and different UNITS from the bicycle set
# above. Labelling a pedestrian's y-velocity offset "initial heading (rad)" is not a
# cosmetic slip: it invites a reader to interpret metres per second as radians.
LINEAR_DELTA_LABELS = [
    "initial vx (m/s)",
    "initial vy (m/s)",
    "x-acceleration bias (m/s^2)",
    "y-acceleration bias (m/s^2)",
]

# Which label set belongs to which recorded parameterization. Keyed by the value
# _stress_one writes into search_provenance, so the mapping has one definition.
DELTA_LABELS_BY_PARAMETERIZATION = {
    'bicycle': DELTA_LABELS,
    'linear': LINEAR_DELTA_LABELS,
}


class FragilityBreakdown(BaseModel):
    """
    The two terms of the stored `fragility_score`, worked out from the stored `min_ttc` and `min_pet`.

    The score ranks the recorded scene only (no perturbation term):

        fragility = ttc_weight * ttc_signal + pet_weight * pet_signal

    with `ttc_weight` 0.4/0.7 and `pet_weight` 0.3/0.7, and each signal 1 / max(value, floor), or 0 when the value is the 999.0
    "none" sentinel. The floors are 0.01 s: a value at or below the floor, a PET of zero or an overlap (negative PET)
    included, contributes the same as 0.01 s. `reproduces_stored_score` says whether the two terms add up to the stored score;
    when it is false the breakdown does not describe that score and is not to be shown as if it did.
    """

    ttc_weight: float
    pet_weight: float
    ttc_signal: float
    pet_signal: float
    ttc_term: float
    pet_term: float
    recomputed_score: float
    ttc_floor_s: float
    pet_floor_s: float
    ttc_none: bool = Field(description="min_ttc is the 999.0 sentinel: the SDC is never in a closing pair; the term is 0.")
    pet_none: bool = Field(description="min_pet is the 999.0 sentinel: no shared conflict zone; the term is 0.")
    ttc_at_floor: bool = Field(description="min_ttc is at or below the floor, so its signal is capped.")
    pet_at_floor: bool = Field(description="min_pet is at or below the floor (an overlap included), so its signal is capped.")
    reproduces_stored_score: bool


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

    # Computed by the API from the stored min_ttc and min_pet (src/danger/fragility_terms.py), not stored. Null when either is
    # null. Only the detail carries it: the ranked list does not.
    fragility_breakdown: FragilityBreakdown | None = None

    # Read from search_provenance, which records them for every searched scenario. Both
    # are distances in metres between the zero-perturbation replay of the challenger and
    # its logged track, so both qualify the stored result. Neither is in ScenarioSummary:
    # the ranked list does not carry them. Null means "not recorded for this result" or,
    # for the offset, "no collision was found"; it is never 0.
    baseline_offset_at_collision: float | None = Field(
        default=None,
        description="Distance in metres, at the collision frame, between the "
                    "zero-perturbation replay of the challenger and its logged "
                    "position. Null when no collision was found or the figure was not "
                    "recorded for the stored result. Not comparable with "
                    "min_perturbation, a weighted norm over components of mixed units.",
    )
    baseline_replay_error: float | None = Field(
        default=None,
        description="Largest distance in metres, over the whole track, between the "
                    "zero-perturbation replay of the challenger and its logged "
                    "position. Null when the figure was not recorded for the stored "
                    "result. Not comparable with min_perturbation, a weighted norm "
                    "over components of mixed units.",
    )


class SearchRecord(BaseModel):
    """
    How the stored result was searched: the keys of `search_provenance` that are public, each as recorded or null.

    Null means "not recorded for this result" (an older result, or a key the search did not write) and is never zero. The keys
    are an allow-list (src/api/provenance.py); nothing else in the stored blob is served. `bounds` has one [low, high] per
    component of the perturbation, in the order of `delta` and `delta_labels`.
    """

    replay_model_version: str | None = None
    collision_geometry_version: str | None = None
    delta_parameterization: str | None = Field(default=None, description="'bicycle' or 'linear': which model's units the four components of the delta carry.")
    challenger_selection: str | None = None
    selected_stage: str | None = Field(default=None, description="Which stage's answer the stored result is: 'de' or 'de+autograd'.")
    de_popsize: int | None = None
    de_maxiter: int | None = None
    de_tol: float | None = None
    de_seed: int | None = None
    de_n_iter: int | None = None
    de_n_eval: int | None = None
    de_candidate_source: str | None = None
    de_archive_covered_all_evaluations: bool | None = None
    refine_n_iters: int | None = None
    bounds: list[list[float]] | None = None
    max_baseline_drift: float | None = None
    max_speed_step: float | None = None
    speed_step_hold_s: float | None = None
    held_speed_excess: float | None = None
    a_max: float | None = None
    heading_speed_floor: float | None = None
    heading_transition_width: float | None = None
    target_min_speed: float | None = None
    frames_observed: int | None = None
    frames_below_heading_floor: int | None = None
    frames_in_heading_transition_band: int | None = None
    target_has_interior_gap: bool | None = None


class PackageVersions(BaseModel):
    """The versions the run recorded: Python and the five packages the pipeline depends on. Null = not recorded."""

    python: str | None = None
    numpy: str | None = None
    scipy: str | None = None
    shapely: str | None = None
    torch: str | None = None
    psycopg2: str | None = None


class RunCounts(BaseModel):
    """What the run reported having done. Null = not reported, never zero."""

    scenarios: int | None = None
    scored: int | None = None
    searched: int | None = None
    refused: int | None = None
    errors: int | None = None
    stored_results: int | None = None
    collisions_found: int | None = None
    no_collision_found: int | None = None


class RunParameters(BaseModel):
    """The parameters that varied a run: seeds, budgets, the number of challengers. Null = not reported."""

    de_seed: int | None = None
    de_popsize: int | None = None
    de_maxiter: int | None = None
    challengers_k: int | None = None
    de_tol: float | None = None
    seeds: list[int] | None = None


class PipelineRun(BaseModel):
    """
    One run of the offline pipeline, as its manifest recorded it (the public columns only: the command line, the host label and the
    notes are never served).

    Every text column is checked before it is served (src/api/manifest.py): `kind` and `status` must be one of the known words, the revision 40 hex
    digits, the checksum 64, the Python version a version string and `shard_name` a file name (letters, digits, `.`, `_`, `-`, at most 200
    characters); anything else is null, not repaired. Null always means "not recorded for this run", never "none". `git_dirty` true means the working tree had uncommitted changes when
    the run started, so the revision alone does not reproduce the code. `package_versions`, `parameters` and `counts` are served through
    allow-lists of keys and types (src/api/manifest.py); a key that is not listed is not served, and a value of the wrong type is null.
    The times are those of the run, in UTC.
    """

    pipeline_run_id: str
    kind: str | None = None
    status: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    git_revision: str | None = None
    git_dirty: bool | None = None
    python_version: str | None = None
    package_versions: PackageVersions | None = None
    shard_name: str | None = None
    shard_sha256: str | None = None
    parameters: RunParameters | None = None
    counts: RunCounts | None = None


class ProvenanceResponse(BaseModel):
    """
    Where a scenario's stored result came from, for the reader who wants to check it or reproduce it.

    Read from `scenario_scores` only. Every field is null when it was not recorded; a scenario with no stored search has
    nulls and `search` null. `stress_run_id` is the hash of the run's identity (the inputs it was searched with and the
    code-contract versions), `scene_fingerprint` the hash of the scene it was searched against, `sdc_idx` which agent was the
    SDC. The times are when the scenario was scored, when a stress-test pass last reached it, and when the stored result was
    written; they are timestamps of the offline runs, in UTC, not of any request to this API.
    """

    scenario_id: str
    stress_run_id: str | None = None
    scene_fingerprint: str | None = None
    sdc_idx: int | None = None
    scored_at: datetime | None = None
    stress_attempted_at: datetime | None = None
    stress_tested_at: datetime | None = None
    search: SearchRecord | None = None
    run: PipelineRun | None = Field(
        default=None,
        description="The manifest of the run that stored this result, found through pipeline_run_results by scenario and "
                    "stress_run_id (the most recent if several are linked). Null when no manifest is recorded for this result: "
                    "the results stored before manifests existed have none, and that is not an error.",
    )
    delta_labels: list[str] | None = Field(
        default=None,
        description="The names, with units, of the components of the perturbation under the parameterization recorded in "
                    "`search`, from the table GET /perturbed uses (DELTA_LABELS_BY_PARAMETERIZATION). Null when no "
                    "parameterization is recorded or it is one this API does not know. A client names `search.bounds` "
                    "with them only when the labels GET /perturbed served are the same list.",
    )


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

    `lengths_m` and `widths_m` are index-aligned with them too: element i is the
    agent's box size at vertex i, which is the size the collision check used at that
    frame. A null ELEMENT means the recorded size at that frame was unusable (not
    finite, or not above zero). A null ARRAY means this agent was exported before
    per-frame sizes were recorded, or the database has no such columns; `length_m` and
    `width_m` (read at the first valid frame) are all there is then. The scalars are
    unchanged and remain what they were.
    """

    agent_idx: int
    agent_type: int | None = None
    is_sdc: bool
    length_m: float | None = None
    width_m: float | None = None
    lengths_m: list[float | None] | None = None
    widths_m: list[float | None] | None = None
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

    Every field except scenario_id is nullable, because "this scenario has not been
    stress-tested yet" is a normal state that returns 200, not an error.

    delta_labels USED to be the second exception, and this docstring said so. It stopped
    being one in Batch 3 (audit B13): a delta can only be labelled if we know which
    kinematic model produced it, and for a legacy row with neither recorded provenance
    nor exported geometry, nothing does. Corrected here rather than quietly reworded,
    because the old sentence is exactly the guarantee a frontend would have been written
    against.
    """

    scenario_id: str
    target_idx: int | None = None
    delta: list[float] | None = None
    # NULLABLE, and null means "we cannot say" (audit B13). A delta is only labellable
    # if we know which parameterization produced it; see the module docstring. Guessing
    # the vehicle set for an unlabellable delta is the B13 defect narrowed, not fixed.
    delta_labels: list[str] | None = None
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
    heading_blend_singularity: int
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

    `status` is "ok" when the database and PostGIS both answered, and "degraded" when the
    database answered but PostGIS did not: `postgis` is then None (the extension is
    missing, or the version call raised). That is degraded, not dead: the scores
    endpoints work fine without it, only geometry is unavailable. Both are HTTP 200.
    """

    status: str
    database: str
    postgis: str | None = None
