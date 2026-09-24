"""
export_geometry.py — Phase 6, Pass 3.

Exports agent trajectories from a WOMD shard into PostGIS so the API server never
needs the shard.

Why this file exists at all
---------------------------
The `.tfrecord` shard is ~1.5 GB and can only be parsed with the Waymo package,
which realistically only installs in Colab (Linux wheels, pinned to an old
TensorFlow). An API process that had to open the shard to answer "draw me this
scenario" would be undeployable. So trajectories are exported ONCE, during a batch
pass, into tables the API can read with nothing but psycopg2.

This is also the first place in the project where a second table is genuinely
justified. Phase 5 got away with one row per scenario because a score is a scalar.
Geometry is 1:many — one scenario has many agents, and each agent has a path
through space and time — so it needs its own table with a foreign key back to
`scenario_scores`.

Four decisions worth defending
------------------------------
1. LINESTRINGM, with the M ordinate carrying the timestep index.
   A trajectory is a space-TIME object. Storing it as a plain LINESTRING would
   throw away *when* the agent was at each vertex, and storing one row per
   (agent, timestep) would turn a 91-point path into 91 rows. LINESTRINGM keeps
   the whole space-time path in one column and PostGIS can still do ordinary
   spatial math on it (ST_Intersects, ST_Distance, ...). The frontend reads M to
   know which timestep each vertex belongs to.

2. SRID 0, deliberately.
   WOMD coordinates are a LOCAL PLANAR FRAME IN METRES, not lon/lat. Tagging them
   4326 would tell PostGIS to interpret metres as degrees, and every distance
   computation would silently return nonsense (a 4-metre car would span several
   hundred kilometres). SRID 0 says "unspecified planar frame", which is exactly
   what this is, and planar math on a local metric frame is the correct math here.

3. Only valid timesteps are exported.
   An agent that appears and disappears mid-scenario produces a linestring whose M
   values jump. That gap is the truth and is represented by MISSING MEASURES —
   never by interpolated fake points. Inventing positions for timesteps where the
   sensor saw nothing would be fabricating data in a safety-auditing tool.

4. Headings live in a parallel DOUBLE PRECISION[], not in the geometry.
   A linestring vertex holds coordinates, not orientation, and the dashboard needs
   heading to draw a rotated bounding box rather than a dot. `headings[i]`
   corresponds to vertex `i` of `path`, and both are built from the same
   valid-timestep list so they cannot drift out of alignment.

Idempotency: every write is an upsert, matching the Phase 5 discipline in db.py.
Re-running the exporter after a code fix updates rows in place.

Error isolation: `export_shard_geometry` wraps EACH scenario in its own try/except.
One corrupt record must never kill a multi-hour batch — the Phase 5 iron rule.

Note the Waymo imports live inside `export_shard_geometry`, not at module level, so
this module imports cleanly on a machine with no Waymo package. Only the
shard-reading path needs it. Same pattern as batch_scorer.py.
"""

import time

import numpy as np

# One shared definition of "where was this agent observed?". PerturbationSpace and
# the torch margin use the same helper; three modules quietly disagreeing about that
# was audit findings B01 and B02.
from src.data.validity import valid_timesteps as _valid_timesteps


# The composite index is additive — it does not redefine anything db.py created.
# It exists to serve the API's ranked ORDER BY (fragility_score DESC, scenario_id),
# which is the ordering ranker.rank_scenarios produces. With it, the paged query is
# an Index Only Scan instead of a sort. See the pagination comment in
# src/api/routes.py for what it does NOT buy us (it is not a seek).
GEOMETRY_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS scenario_agents (
    scenario_id TEXT NOT NULL
                REFERENCES scenario_scores(scenario_id) ON DELETE CASCADE,
    agent_idx   INTEGER NOT NULL,
    agent_type  INTEGER,
    is_sdc      BOOLEAN NOT NULL DEFAULT FALSE,
    length_m    DOUBLE PRECISION,
    width_m     DOUBLE PRECISION,
    n_points    INTEGER NOT NULL,
    headings    DOUBLE PRECISION[],
    path        geometry(LINESTRINGM, 0),
    PRIMARY KEY (scenario_id, agent_idx)
);
CREATE INDEX IF NOT EXISTS idx_scenario_agents_scenario
    ON scenario_agents (scenario_id);

CREATE TABLE IF NOT EXISTS perturbed_paths (
    scenario_id TEXT PRIMARY KEY
                REFERENCES scenario_scores(scenario_id) ON DELETE CASCADE,
    target_idx  INTEGER NOT NULL,
    n_points    INTEGER NOT NULL,
    headings    DOUBLE PRECISION[],
    path        geometry(LINESTRINGM, 0),
    exported_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_scenario_scores_fragility_id
    ON scenario_scores (fragility_score DESC, scenario_id);

-- Audit B14: which stress run this geometry was replayed from. Compared against
-- scenario_scores.stress_run_id on read, so a path exported from an older delta can
-- be DETECTED rather than served as evidence for a newer one.
ALTER TABLE perturbed_paths ADD COLUMN IF NOT EXISTS stress_run_id TEXT;

-- Audit A02: which SCENE this geometry was built from. Symmetric with stress_run_id
-- above, and for the same reason it exists on both tables: a path can be paired with
-- a result only if the RUN and the SCENE both agree. stress_run_id alone is not
-- enough — a re-parsed scene that happens to produce the same optimal delta yields
-- the same run id, and the join would pair a new result with old-scene geometry.
ALTER TABLE perturbed_paths ADD COLUMN IF NOT EXISTS scene_fingerprint TEXT;

-- The same column on the BASELINE side, and it closes a window rather than narrowing
-- one (audit A02). export_scenario_agents commits and releases its row lock before
-- export_perturbed_path opens its own transaction, so a concurrent Pass 1 can move
-- scenario_scores.scene_fingerprint in between. The perturbed half self-guards and
-- refuses; the baseline half was already committed, and without a fingerprint of its
-- own nothing on the read side could tell. get_trajectories reads this table with no
-- other identity check at all, so that stale geometry was being SERVED, not merely
-- left lying around.
--
-- A per-scenario fact on a per-agent table, which is denormalization — accepted for
-- the same reason perturbed_paths.stress_run_id is, and with a stronger guarantee
-- behind it: every agent row for a scenario is written in ONE transaction under the
-- FOR UPDATE lock, so they cannot disagree with each other. That is also what stops
-- get_trajectories from ever seeing a PARTIALLY matching agent set: the comparison
-- passes for a whole scenario or fails for a whole scenario, never for some of it.
--
-- ── THE MIGRATION CONSEQUENCE, STATED BECAUSE IT AFFECTS LIVE DATA ──────────────
--
-- Geometry exported before this column exists carries NULL here. IS NOT DISTINCT FROM
-- carves out the BOTH-NULL case only — never one-NULL-one-real — exactly as B14
-- established for stress_run_id. So the first time Pass 1 re-runs and records a
-- fingerprint on the score row, that pre-existing geometry STOPS SERVING until Pass 3
-- catches up. Measured on a legacy row:
--
--     legacy row, legacy geometry        -> 2 agents served
--     after Pass 1 re-run, before Pass 3 -> 0 agents served
--     after Pass 3 catches up            -> 2 agents served
--
-- Correct and consistent rather than a defect — serving geometry that cannot be shown
-- to describe the current scene is the thing this batch exists to stop — but it means
-- a Pass 1 re-run over an already-exported corpus blanks the dashboard until Pass 3
-- follows. Run them together, or expect the gap.
ALTER TABLE scenario_agents ADD COLUMN IF NOT EXISTS scene_fingerprint TEXT;

-- fix G04: THE SDC AXIS, FOR THE SAME REASON scene_fingerprint IS A COLUMN HERE
-- AND NOT JUST A WRITE-TIME CHECK. export_scenario_agents already verifies sdc_idx
-- against scenario_scores at write time (mirroring the scene_fingerprint check
-- above) — but a write-time check alone protects only the instant of that write.
-- scene_fingerprint additionally persists a per-row snapshot so get_trajectories can
-- keep verifying it on every READ, long after the write, against whatever
-- scenario_scores.scene_fingerprint says NOW — which is exactly what makes a later
-- Pass 1 rescope self-correcting without any extra invalidation logic. sdc_idx had
-- no such snapshot, so an sdc_idx-only rescope (scene_fingerprint unchanged — a real
-- case: compute_scene_fingerprint hashes states/validity/types only, not which
-- track is the SDC) left is_sdc flags computed from the OLD sdc_idx being served
-- forever, with nothing to compare against on read. Same column, same carve-out,
-- same read-side predicate as scene_fingerprint — see get_trajectories.
ALTER TABLE scenario_agents ADD COLUMN IF NOT EXISTS sdc_idx INTEGER;
"""


def init_geometry_schema(conn):
    """Create the geometry tables and the PostGIS extension. Safe to call every run."""
    with conn.cursor() as cur:
        cur.execute(GEOMETRY_SCHEMA_SQL)
    conn.commit()


# ── WKT construction ────────────────────────────────────────────────────────────

def _linestring_m_wkt(xs, ys, ms) -> str:
    """
    Build a `LINESTRING M(x y m, ...)` WKT string.

    Coordinates are formatted with repr(float(...)) rather than a fixed number of
    decimals: repr round-trips a binary64 exactly, so the geometry that comes back
    out of PostGIS is the geometry that went in. Casting to a Python float first
    matters — repr() of a numpy scalar produces 'np.float32(1.0)', which is not WKT.
    """
    pts = ', '.join(
        f'{float(x)!r} {float(y)!r} {float(m)!r}'
        for x, y, m in zip(xs, ys, ms)
    )
    return f'LINESTRING M({pts})'


# ── per-scenario exports ────────────────────────────────────────────────────────

class SceneChangedError(RuntimeError):
    """
    The arrays offered for export are not the scene the stored row describes
    (audit A02).

    A DISTINCT TYPE FROM StaleExportError, and the distinction is the finding rather
    than taxonomy. StaleExportError means A NEWER RESULT EXISTS: the scene is fine, the
    perturbation moved on. This means THE INPUT CHANGED UNDERNEATH A STABLE ID: the
    scenario_id still resolves, the result may be untouched, and the thing the result
    was computed against is gone. Those call for different responses — one is "re-run
    Pass 3", the other is "re-run Pass 1 and Pass 2, your result describes a scene that
    no longer exists" — and one exception type reporting both would assert less than it
    knows, which is the defect class Batch 2 spent two rounds removing from
    scenario_scores.

    Carries both fingerprints so the caller can say WHICH scene was refused against
    WHICH, not merely that something was.
    """

    def __init__(self, scenario_id, stored_fingerprint, computed_fingerprint):
        self.scenario_id = scenario_id
        self.stored_fingerprint = stored_fingerprint
        self.computed_fingerprint = computed_fingerprint
        super().__init__(
            f"refusing to publish geometry for {scenario_id!r}: it was built from "
            f"scene {computed_fingerprint!r}, but the stored row describes scene "
            f"{stored_fingerprint!r}"
        )


class SdcIndexChangedError(RuntimeError):
    """
    The SDC this export was passed does not match the SDC scenario_scores.sdc_idx
    describes (fix G04).

    A DISTINCT TYPE FROM SceneChangedError, for the reason that class's own
    docstring argues for splitting by finding rather than merging: this can fire
    with scene_fingerprint UNCHANGED. compute_scene_fingerprint hashes
    states/validity/types only (see F02's comment in db.py) — sdc_track_index is a
    separate WOMD field two parses can disagree on while every array still hashes
    identically. Reusing SceneChangedError here would report "scene changed" while
    printing two IDENTICAL fingerprints, which asserts something false about which
    axis actually disagreed.

    Carries both indices so the caller can say WHICH sdc_idx was refused against
    WHICH, not merely that something was.
    """

    def __init__(self, scenario_id, stored_sdc_idx, computed_sdc_idx):
        self.scenario_id = scenario_id
        self.stored_sdc_idx = stored_sdc_idx
        self.computed_sdc_idx = computed_sdc_idx
        super().__init__(
            f"refusing to publish geometry for {scenario_id!r}: it was parsed with "
            f"sdc_idx {computed_sdc_idx!r}, but the stored row describes sdc_idx "
            f"{stored_sdc_idx!r}"
        )


def export_scenario_agents(conn, scenario_id, states, validity, types, sdc_idx):
    """
    Write every agent's logged trajectory for one scenario.

    Args:
        states:   (N, T, 7) = [x, y, vx, vy, heading, length, width]
        validity: (N, T) bool
        types:    (N,) int — 1 vehicle, 2 pedestrian, 3 cyclist
        sdc_idx:  index of the self-driving car

    Returns:
        (n_written, n_skipped, stored_fingerprint)

        stored_fingerprint is the value read under FOR UPDATE above and stamped onto
        every row this call writes — None for a legacy/unscored-by-fingerprint row,
        the verified scene fingerprint otherwise. Callers doing a SECOND export in
        the same pass (fix F05) should reuse this rather than re-reading
        scenario_scores: the lock that made this value trustworthy is released the
        moment this function commits, so a fresh read afterward is a new snapshot,
        not a continuation of this one. sdc_idx is not returned the same way (fix
        G04): once this call has not raised, the caller's OWN sdc_idx argument has
        already been proven consistent with scenario_scores.sdc_idx (or the latter
        was never recorded), so there is nothing a second return value would add.

    Raises:
        SceneChangedError:    the parsed scene does not match scenario_scores'
                              stored scene_fingerprint.
        SdcIndexChangedError: the parsed sdc_idx does not match scenario_scores'
                              stored sdc_idx (fix G04) — checked independently of
                              the scene, since a fingerprint cannot see this axis.

    An agent with fewer than 2 valid timesteps cannot form a linestring — a
    LINESTRING needs at least two vertices. Those agents are skipped and counted,
    never raised on: a scenario where one agent blinks in for a single frame is
    normal data, not an error.
    """
    n_agents = states.shape[0]
    written = skipped = 0
    exportable = []

    with conn.cursor() as cur:
        # ── THE SCENE CHECK, AND IT RUNS BEFORE ANY WRITE (audit A02) ────────────
        #
        # This function used to read scenario_scores not at all. It upserted every
        # agent and called conn.commit() in its own body, and export_shard_geometry
        # calls it FIRST — so R01's stale-export guard, which lives inside
        # export_perturbed_path, ran one commit too late. Measured: a refused perturbed
        # export left a baseline from the wrong scene already committed, 3.7000 ->
        # 3.2000, and the StaleExportError handler's rollback had nothing left to undo.
        #
        # ── WHY `FOR UPDATE` AND NOT R01's SINGLE-STATEMENT PATTERN ──────────────
        #
        # R01 guards ONE row with INSERT ... SELECT ... WHERE, so its check and its
        # write are the same statement and no lock is needed. That is not available
        # here: this function writes N agent upserts plus two DELETEs, and guarding
        # each individually gives PER-ROW atomicity, not SET atomicity — agents 0-3
        # could be admitted under the old fingerprint, a concurrent Pass 1 commits, and
        # agents 4-9 are silently refused. Checking rowcount per statement and rolling
        # back narrows the window to "a change committing after the last write but
        # before COMMIT"; it does not close it.
        #
        # FOR UPDATE holds the row lock until COMMIT, so the fingerprint cannot move
        # between this read and the last write. That closes the window rather than
        # shrinking it.
        #
        # This does NOT contradict export_perturbed_path's comment that the project has
        # "no precedent for locking primitives". That sentence continues "which
        # single-statement consistency makes unnecessary HERE". The precedent was never
        # "do not lock" — it was "do not lock when one statement suffices". One
        # statement does not suffice for an N-row agent set.
        #
        # LOCK ORDER, checked rather than assumed: update_stress_results takes its row
        # lock via UPDATE scenario_scores and only then touches perturbed_paths; this
        # locks scenario_scores and only then touches scenario_agents/perturbed_paths.
        # Scores before geometry in both, so no inversion is possible, and each call
        # locks exactly one scenario row. The contention this does create is correct:
        # Pass 2 committing a result while Pass 3 writes that scenario's geometry is
        # precisely the interleaving that should serialize.
        # sdc_idx READ IN THE SAME STATEMENT AS scene_fingerprint, UNDER THE SAME LOCK
        # (fix G04) — not a second query. F02's own reasoning in db.py: sdc_idx is a
        # separate identity axis a fingerprint cannot see, since
        # compute_scene_fingerprint hashes states/validity/types only. A second read
        # here would reopen exactly the window this FOR UPDATE exists to close.
        cur.execute("""
            SELECT scene_fingerprint, sdc_idx FROM scenario_scores
             WHERE scenario_id = %s FOR UPDATE
        """, (scenario_id,))
        row = cur.fetchone()
        # No row is NOT an error here. It means this scenario was never scored, and the
        # foreign key on the INSERT below already refuses it — with a message about the
        # actual problem. Inventing a SceneChangedError for it would report a scene
        # mismatch where the truth is a missing scenario.
        stored_fingerprint = row[0] if row else None
        stored_sdc_idx = row[1] if row else None
        if stored_fingerprint is not None:
            from src.scoring.db import compute_scene_fingerprint
            computed = compute_scene_fingerprint(states, validity, types)
            if computed != stored_fingerprint:
                # Nothing has been written yet, so there is nothing to undo — but the
                # lock is released and the caller gets a clean transaction either way.
                conn.rollback()
                raise SceneChangedError(scenario_id, stored_fingerprint, computed)
        # NULL means NOT RECORDED, not "mismatch" — the same carve-out B14 makes for
        # stress_run_id. Rows written before this column keep exporting exactly as they
        # did, which is what keeps the audit's own B13/B14 fixtures passing unmodified.
        #
        # CHECKED INDEPENDENTLY OF scene_fingerprint, not folded into the branch above
        # (fix G04) — a scene can match while sdc_idx still disagrees (see
        # SdcIndexChangedError's own docstring), so this needs its own comparison and
        # its own carve-out, not a shared one.
        if stored_sdc_idx is not None and int(sdc_idx) != stored_sdc_idx:
            conn.rollback()
            raise SdcIndexChangedError(scenario_id, stored_sdc_idx, int(sdc_idx))

        for i in range(n_agents):
            ts = _valid_timesteps(validity, i)
            if len(ts) < 2:
                skipped += 1
                continue
            exportable.append(int(i))

            xs = states[i, ts, 0]
            ys = states[i, ts, 1]
            # M ordinate = the timestep index itself, so the frontend can map a
            # vertex back to a frame without a second lookup.
            wkt = _linestring_m_wkt(xs, ys, ts.astype(float))
            headings = [float(h) for h in states[i, ts, 4]]

            # length/width are physical constants of the agent; read them from the
            # first valid timestep rather than assuming index 0 was observed.
            t0 = int(ts[0])

            # scene_fingerprint is the value THIS ROW was verified against, taken
            # from the locked read above rather than recomputed — so a legacy row
            # (NULL) stamps NULL and keeps serving, the same carve-out everywhere else.
            #
            # sdc_idx (fix G04) is stored the SAME way, and for the SAME reason —
            # stored_sdc_idx, not the freshly parsed sdc_idx parameter. A legacy row
            # (stored_sdc_idx is None, never checked above) must stamp NULL here too:
            # storing the parsed value instead would make get_trajectories's read-side
            # predicate compare a real integer against scenario_scores.sdc_idx's own
            # NULL and wrongly refuse every legacy row, reopening exactly the carve-out
            # scene_fingerprint already protects. Once stored_sdc_idx IS recorded, the
            # check above has already proven it equals the parsed sdc_idx, so which one
            # is stamped makes no difference there — stored_sdc_idx is used regardless,
            # to keep this column governed by the same rule in both branches.
            cur.execute("""
                INSERT INTO scenario_agents
                    (scenario_id, agent_idx, agent_type, is_sdc,
                     length_m, width_m, n_points, headings, path, scene_fingerprint,
                     sdc_idx)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, ST_GeomFromText(%s, 0), %s, %s)
                ON CONFLICT (scenario_id, agent_idx) DO UPDATE SET
                    agent_type        = EXCLUDED.agent_type,
                    is_sdc            = EXCLUDED.is_sdc,
                    length_m          = EXCLUDED.length_m,
                    width_m           = EXCLUDED.width_m,
                    n_points          = EXCLUDED.n_points,
                    headings          = EXCLUDED.headings,
                    path              = EXCLUDED.path,
                    scene_fingerprint = EXCLUDED.scene_fingerprint,
                    sdc_idx           = EXCLUDED.sdc_idx
            """, (
                scenario_id, int(i), int(types[i]), bool(i == sdc_idx),
                float(states[i, t0, 5]), float(states[i, t0, 6]),
                len(ts), headings, wkt, stored_fingerprint, stored_sdc_idx,
            ))
            written += 1

        # Audit B15: a re-export REPLACES this scenario's agent set, it does not add
        # to it. Upserting alone leaves a stale row behind whenever an agent that was
        # exportable last time is not this time — corrected input, a parser change, or
        # an agent that dropped out of the array entirely. The export then truthfully
        # reports "2 written, 1 skipped" while the API keeps serving all three.
        cur.execute("""
            DELETE FROM scenario_agents
             WHERE scenario_id = %s AND NOT (agent_idx = ANY(%s))
        """, (scenario_id, exportable))

        # A perturbed path pointing at an agent that no longer exists is the same
        # defect one table over: /perturbed would look up a baseline that is gone and
        # quietly return a perturbed path with baseline=None.
        cur.execute("""
            DELETE FROM perturbed_paths
             WHERE scenario_id = %s AND NOT (target_idx = ANY(%s))
        """, (scenario_id, exportable))

    conn.commit()
    return written, skipped, stored_fingerprint


class StaleExportError(RuntimeError):
    """
    The geometry offered for export does not belong to the result currently stored
    (audit R01).

    A DISTINCT EXCEPTION RATHER THAN A FALSY RETURN. export_perturbed_path already
    returns False for "the target had too few valid timesteps", which is a property
    of the input and not a failure. Collapsing "this export is stale" into that same
    False would make the two indistinguishable to every caller — the defect class
    Batch 2 spent two rounds removing from scenario_scores, reappearing in a return
    value.

    Carries both ids so the caller can say WHICH run was refused against WHICH, not
    merely that something was.
    """

    def __init__(self, scenario_id, attempted_run_id, current_run_id):
        self.scenario_id = scenario_id
        self.attempted_run_id = attempted_run_id
        self.current_run_id = current_run_id
        super().__init__(
            f"refusing to publish geometry for {scenario_id!r}: it was built from "
            f"run {attempted_run_id!r}, but the stored result is run "
            f"{current_run_id!r}"
        )


def export_perturbed_path(conn, scenario_id, perturbed_states, validity, target_idx,
                          stress_run_id=None, delta=None, method=None,
                          scene_fingerprint=None):
    """
    Write the challenger's PERTURBED trajectory — the Phase 4 answer, made visible.

    This is the row that lets the dashboard draw "here is what the car actually did,
    and here is the smallest change that would have caused a crash" as two paths on
    one map. Only the challenger is stored: the SDC and every other agent keep their
    logged trajectory by construction (PerturbationSpace.apply only rewrites the
    target), so re-storing them would duplicate scenario_agents for no gain.

    ⚠ PASS `delta` AND `method`. THE STALE-EXPORT PROTECTION IS VACUOUS WITHOUT THEM.
    =================================================================================
    They are optional only for backwards compatibility with callers written before
    audit R01, and a call that omits them gets the OLD, DEFECTIVE behaviour: the run
    id is read from whatever is in scenario_scores at this instant and stamped onto
    whatever trajectory was handed in, so geometry rebuilt from an older result is
    labelled as current and the read-side join cannot see it — the ids genuinely
    match, they are simply attached to the wrong content. The verification below then
    compares a value against itself and always passes.

    BATCH 11 UPDATE (audit A12): that warning was not a caveat, it was a live bypass,
    and it is now closed for any scenario whose scene has been fingerprinted. Measured
    against a scene displaced 99 m, the 8-argument call refused and the 5-argument call
    published. A call that supplies neither `delta` nor `scene_fingerprint` is now
    REFUSED with SceneChangedError whenever the row records a scene_fingerprint —
    which is every row Pass 1 writes from now on. Rows without one keep exporting, so
    legacy data and hand-seeded fixtures are unaffected. The paragraph above still
    describes exactly what the vacuous path DID, and is kept because the reason the
    signature survives at all is that ten callers depend on its defaults.

    FIX F04: the run half of the vacuous path was still open. A12 closed "the caller
    named no scene, the row has one" but left "the caller named a scene, but no run"
    free to fall through to the exact same borrow the paragraph above describes —
    read whatever run id is currently stored and stamp it back onto the geometry in
    hand, so the WHERE clause compares that value against itself. Measured: persist
    delta A, persist delta B against the same scene, export A's geometry with only
    the correct `scene_fingerprint` and nothing else — it published, stamped with
    B's run id. A call that names a scene but no run is now REFUSED with
    StaleExportError under the same condition as the scene refusal (the row records a
    scene_fingerprint); a row with none keeps exporting under the same carve-out.

    With `delta` and `method` supplied, the run id is DERIVED FROM THE CONTENT being
    exported — compute_stress_run_id(scenario_id, target_idx, delta, method), the
    same function and the same four inputs update_stress_results used to stamp the
    result — and the write publishes only if that id is still the persisted one.

    The audit's repro, which this refuses: persist result A, persist a newer result
    B, then export using A's result dict. Before R01 the API served B's delta beside
    A's trajectory, 0.9 m apart at the last frame, with the B14 join passing.

    Args:
        stress_run_id: override the derived/looked-up id. Callers that genuinely know
                       better may still set it; it is checked like any other. It does
                       NOT vouch for the scene — supplying it without
                       `scene_fingerprint` against a fingerprinted row is refused, for
                       the reason spelled out at the refusal below.
        delta:         the perturbation this trajectory was rebuilt from.
        method:        'de' or 'de+autograd', as recorded on the result.
        scene_fingerprint: compute_scene_fingerprint of the scene this trajectory was
                       built from (audit A02). Supplying it is what makes the export
                       verifiable against a re-parse; omitting it on a fingerprinted
                       row is refused.

    Returns True if a row was written, False if the target had too few valid
    timesteps to form a linestring.

    Raises:
        SceneChangedError: the scene this export was built from is not the scene the
                          stored row describes, or the caller named no scene for a row
                          that records one. Nothing is written.

        StaleExportError: the run id this export carries is not the one currently
                          stored for the scenario, or the caller named a scene but no
                          run for a row that records a scene_fingerprint (fix F04).
                          Nothing is written.

                          Its `current_run_id` is READ AFTER the refusal, so it is
                          the current value at report time rather than a guaranteed
                          part of the snapshot the refusal was decided against — a
                          third writer landing in between would change what the
                          message says. That affects the message only. The refusal
                          itself already happened atomically, inside the single
                          INSERT ... SELECT ... WHERE below, and is not re-derived
                          from this read.
    """
    ts = _valid_timesteps(validity, target_idx)
    if len(ts) < 2:
        return False

    # Derive from the content in hand when the caller supplied it (audit R01);
    # otherwise fall back to the pre-R01 lookup, with the caveat in the docstring.
    if stress_run_id is None and delta is not None:
        from src.scoring.db import compute_stress_run_id
        stress_run_id = compute_stress_run_id(scenario_id, target_idx, delta, method)

    with conn.cursor() as cur:
        cur.execute("SELECT stress_run_id, scene_fingerprint FROM scenario_scores "
                    "WHERE scenario_id = %s", (scenario_id,))
        row = cur.fetchone()
    stored_run_id, stored_scene = (row if row else (None, None))

    # ── THE LEGACY PATH IS NO LONGER A FREE PASS (audit A12) ────────────────────
    #
    # Batch 7 kept this signature for backward compatibility and warned the protection
    # was "vacuous without delta/method". It is worse than a caveat: the run id is read
    # from scenario_scores and stamped onto whatever trajectory was handed in, so the
    # WHERE clause below compares a value against itself and ALWAYS passes. Measured
    # against a scene displaced 99 m: the 8-argument call refused and the 5-argument
    # call published, stamped with the stored id so every downstream join agreed.
    #
    # THE RULE: if the row knows which scene it describes, the caller must say which
    # scene its geometry came from. Nothing else — not delta, not an explicitly
    # supplied stress_run_id — vouches for the scene.
    #
    #   - A row with NO fingerprint is legacy data or a hand-seeded fixture. NULL means
    #     "not recorded" and is not evidence of a mismatch, exactly as B14 reasons for
    #     stress_run_id. These keep exporting. That carve-out is what lets the audit's
    #     own B13/B14 fixtures pass unmodified — they seed scenario_scores by hand and
    #     carry no fingerprint, and it was a census of all 26 call sites, not a guess,
    #     that established every existing legacy caller is in that class.
    #   - A row Pass 1 fingerprinted is modern data, and publishing geometry against it
    #     without saying what the geometry was built from is the bypass. Refused.
    #
    # NOT CONDITIONED ON WHETHER THE RUN ID WAS DERIVED, and an earlier version of this
    # was. That version gated the refusal on `derived = stress_run_id is not None or
    # delta is not None`, which reopened A12 through a different door: a caller passing
    # stress_run_id= explicitly — a pattern this function's own docstring invites —
    # skipped the refusal, and the write below then fell back to `stored_scene`, read
    # from the same row moments earlier, so the scene half of the WHERE compared the
    # row against ITSELF. The run half stayed sound; the scene half was vacuous, which
    # is precisely the shape of the finding being fixed.
    #
    # A census found zero callers in that state — `stress_run_id=` appears nowhere in
    # this repository except in this signature — so it was dead in practice and the
    # contract was still wrong. Same treatment B16 and R05 got: closed because closing
    # it was cheaper than describing it, and it DELETES a flag rather than adding a
    # branch.
    if scene_fingerprint is None and stored_scene is not None:
        conn.rollback()
        raise SceneChangedError(scenario_id, stored_scene, None)

    # ── THE RUN CHECK MIRRORS THE SCENE CHECK (fix F04) ─────────────────────────
    #
    # A caller that names no run and supplies no delta used to fall straight into
    # `stress_run_id = stored_run_id` below — reading the row's own current run id
    # and stamping it back onto whatever content was handed in. The INSERT's WHERE
    # then compares that borrowed value against itself: always true. On a row that
    # HAS a fingerprint this is the exact shape A12 already closed on the scene
    # axis and left open here. Measured: persist delta [-1,0,0,0], persist delta
    # [-2,0,0,0] against the same scene, export the first path with only its
    # scene_fingerprint — it published, stamped with the second result's run id.
    #
    # GATED ON stored_scene, NOT stored_run_id, matching the scene check above and
    # deliberately NOT the literal borrow. A row with no fingerprint is exempt from
    # this whole contract already — both axes, the same carve-out — and that is
    # what keeps test_A12_a_row_without_a_fingerprint_still_exports passing
    # unmodified: it scores a row via upsert_scores with no scene_fingerprint, then
    # stress-tests it (so stored_run_id IS set), then exports bare. Gating on
    # stored_run_id instead would refuse that exact call and regress a contract the
    # audit protects as unchanged. A row WITH a fingerprint is Pass-1-verified data;
    # exporting geometry against it without saying which run that geometry came
    # from is the bypass.
    if stress_run_id is None and stored_scene is not None:
        conn.rollback()
        raise StaleExportError(scenario_id, None, stored_run_id)

    if stress_run_id is None:
        stress_run_id = stored_run_id

    xs = perturbed_states[target_idx, ts, 0]
    ys = perturbed_states[target_idx, ts, 1]
    wkt = _linestring_m_wkt(xs, ys, ts.astype(float))
    headings = [float(h) for h in perturbed_states[target_idx, ts, 4]]

    with conn.cursor() as cur:
        # INSERT ... SELECT ... WHERE, so the check and the write are ONE statement.
        # A SELECT-then-INSERT would have a race of its own — exactly the shape of the
        # bug being fixed — and this project has no precedent for locking primitives,
        # which single-statement consistency makes unnecessary here.
        #
        # IS NOT DISTINCT FROM, not `=`, matching the read side in api/routes.py: a
        # legacy scenario with NULL on both sides still exports. `=` is NULL rather
        # than true when either operand is NULL, which would refuse every export for a
        # row that predates stress_run_id.
        # THE SCENE IS CHECKED IN THE SAME STATEMENT AS THE RUN (audit A02), for the
        # same reason the run is: a SELECT-then-INSERT reintroduces the race. The
        # scene is stamped onto the row so the read side can pair on both — a result
        # and a picture belong together only if the RUN and the SCENE agree.
        #
        # IS NOT DISTINCT FROM on the scene too, so a caller that supplies nothing and
        # a row that records nothing still match, which is the legacy carve-out the
        # refusal above already let through.
        cur.execute("""
            INSERT INTO perturbed_paths
                (scenario_id, target_idx, n_points, headings, path, stress_run_id,
                 scene_fingerprint)
            SELECT %s, %s, %s, %s, ST_GeomFromText(%s, 0), %s, ss.scene_fingerprint
            FROM scenario_scores ss
            WHERE ss.scenario_id = %s
              AND ss.stress_run_id IS NOT DISTINCT FROM %s
              AND ss.scene_fingerprint IS NOT DISTINCT FROM %s
            ON CONFLICT (scenario_id) DO UPDATE SET
                target_idx        = EXCLUDED.target_idx,
                n_points          = EXCLUDED.n_points,
                headings          = EXCLUDED.headings,
                path              = EXCLUDED.path,
                stress_run_id     = EXCLUDED.stress_run_id,
                scene_fingerprint = EXCLUDED.scene_fingerprint,
                exported_at       = now()
        """, (scenario_id, int(target_idx), len(ts), headings, wkt, stress_run_id,
              scenario_id, stress_run_id,
              # The caller's scene when it named one; otherwise the row's own, so the
              # predicate is satisfied by construction and this clause changes nothing
              # for a caller that legitimately did not supply it.
              scene_fingerprint if scene_fingerprint is not None else stored_scene))
        published = cur.rowcount

    if not published:
        # Nothing was written, so nothing needs rolling back — but the transaction is
        # left clean for the caller either way.
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SELECT stress_run_id, scene_fingerprint FROM scenario_scores "
                        "WHERE scenario_id = %s", (scenario_id,))
            row = cur.fetchone()
        current_run, current_scene = (row if row else (None, None))
        # WHICH refusal this was, reported as the thing it actually is. A scene
        # mismatch and a superseded result are different situations with different
        # responses, and collapsing them into StaleExportError would make the caller
        # re-run the wrong pass.
        if scene_fingerprint is not None and current_scene != scene_fingerprint:
            raise SceneChangedError(scenario_id, current_scene, scene_fingerprint)
        raise StaleExportError(scenario_id, stress_run_id, current_run)

    conn.commit()
    return True


# ── the batch pass ──────────────────────────────────────────────────────────────

def export_shard_geometry(conn, shard_path, scenario_ids, stress_results=None,
                          verbose=True):
    """
    PASS 3 — walk a shard and export geometry for the requested scenarios.

    Args:
        conn:           open psycopg2 connection
        shard_path:     path to the .tfrecord shard
        scenario_ids:   iterable of scenario IDs to export (typically the ranked
                        top-N that Pass 2 stress-tested, plus whatever the
                        dashboard needs to draw)
        stress_results: optional dict sid -> Phase 4 result, as returned by
                        batch_scorer.stress_test_scenarios. Where a result is 'ok'
                        and carries a delta, the perturbed challenger path is
                        rebuilt and stored too.
        verbose:        print progress lines

    Returns a summary dict: exported, agents_written, agents_skipped,
    perturbed_written, perturbed_stale (list of dicts), scene_changed (list of
    dicts), sdc_changed (list of dicts), errors (list of dicts).

    perturbed_stale RECORDS EACH REFUSAL INDIVIDUALLY, not as a tally. Block 5
    Concept 19: a batch that reports "982 scored, 18 skipped, here is why" is
    trustworthy; one that reports "982 scored" and hides 18 failures is a silent
    data-quality bug. A count answers "how many", and an operator looking at a
    stale-export spike needs "which scenarios, and which run did each one think it
    was" — so every entry carries the scenario id, the run it was built from, and
    the run currently stored.

    Every scenario is isolated in its own try/except. Errors are RECORDED and the
    walk continues — one malformed record must not cost a multi-hour batch.

    NOTE: this function has never been run against real WOMD data. If real
    scenarios violate an assumption here — validity gaps in odd places, pedestrian
    challengers, agents with a single valid timestep — that belongs in the errors
    list where it can be seen and fixed, not papered over with a broad except that
    pretends the export succeeded.
    """
    # Lazy import: only the shard-reading path needs the Waymo package, and it only
    # installs in Colab. Keeping these inside the function is what lets the API
    # process import this module (for the schema constant) on any machine.
    from src.data.loader import ShardLoader
    from src.data.parser import ScenarioParser

    wanted = set(scenario_ids)
    summary = {
        'exported': 0,
        'agents_written': 0,
        'agents_skipped': 0,
        'perturbed_written': 0,
        'perturbed_stale': [],
        # Scenarios whose parsed arrays are not the scene the stored row describes
        # (audit A02). RECORDS, NOT A TALLY, matching perturbed_stale beside it and
        # for Batch 7's reason: an operator looking at a spike needs to know WHICH
        # scenarios and which two scenes disagreed, and a count answers neither.
        'scene_changed': [],
        # Same shape, same reason, for the sdc_idx axis scene_changed cannot see
        # (fix G04) — a scene can match while the parsed SDC disagrees with the
        # stored one. A SEPARATE bucket from scene_changed rather than folded in,
        # for SceneChangedError/SdcIndexChangedError's own reason: printing "scene
        # changed" for an sdc-only mismatch would assert something false.
        'sdc_changed': [],
        'errors': [],
    }
    t_start = time.time()

    # Audit R02, same shape as batch_scorer's two loops: the fetch sits inside a
    # handler so a truncated shard cannot discard geometry already exported, and the
    # completion check runs BEFORE the fetch so an export of N scenarios never reads
    # the N+1'th record.
    reader = iter(ShardLoader(shard_path))
    idx = -1
    while True:
        if not wanted:
            break
        idx += 1
        try:
            raw = next(reader)
        except StopIteration:
            break                      # clean EOF — normal completion, not an error
        except Exception as e:  # noqa: BLE001
            summary['errors'].append({'index': idx, 'scenario_id': None,
                                      'error': f'{type(e).__name__}: {e}',
                                      'kind': 'shard_truncated'})
            if verbose:
                print(f"  [fatal] shard unreadable at record {idx}: {e}")
            break

        sid = None
        try:
            parser = ScenarioParser(raw)
            sid = parser.get_scenario_id()
            if sid not in wanted:
                continue
            wanted.discard(sid)

            states = parser.get_agent_states()
            validity = parser.get_agent_validity()
            types = parser.get_agent_types()
            sdc_idx = parser.get_sdc_index()

            # A scene that no longer matches the stored row is a NORMAL outcome of a
            # re-parse, not a crash — the same reasoning that makes replay_infeasible a
            # status and a stale export a recorded refusal. Recorded in full, and the
            # walk continues. `continue`, not a partial export: the perturbed path
            # below is built from the same rejected arrays, so publishing it would be
            # exactly the half-written state the baseline guard exists to prevent.
            #
            # ── THE TWO GUARDS ARE SEQUENTIAL, AND THAT WINDOW IS NOW CLOSED ────
            #
            # export_scenario_agents commits in its own body, releasing its FOR UPDATE
            # row lock, and export_perturbed_path then opens a separate transaction. A
            # concurrent Pass 1 can move scenario_scores.scene_fingerprint in the gap.
            # Measured before the baseline column existed:
            #
            #     baseline committed under fp_old
            #     concurrent Pass 1: 1d51b8dfabec61a8 -> a35d62644962c405
            #     perturbed REFUSED (stored=a35d..., attempted=1d51...)
            #     final: scenario_agents=2 rows from fp_old, perturbed_paths=0 rows
            #
            # The perturbed half always self-guarded, so no mismatched PAIR was ever
            # published. The baseline half was already committed and carried no
            # fingerprint, and get_trajectories reads that table with no other identity
            # check — so the stale scene was being SERVED.
            #
            # This was briefly written up as an accepted residual. It is not one:
            # scenario_agents now carries scene_fingerprint too, and get_trajectories
            # compares it, so the window closes to zero rather than being documented as
            # narrow. That is R03's precedent applied — binding to an identity already
            # in hand beat narrowing a race there, and it beats narrowing one here.
            # Stale rows can still exist in the table after such an interleaving; they
            # are inert, invisible to every read, and replaced by the next export.
            try:
                written, skipped, stored_fingerprint = export_scenario_agents(
                    conn, sid, states, validity, types, sdc_idx
                )
            except SceneChangedError as changed:
                conn.rollback()
                summary['scene_changed'].append({
                    'scenario_id': changed.scenario_id,
                    'stored_fingerprint': changed.stored_fingerprint,
                    'computed_fingerprint': changed.computed_fingerprint,
                })
                if verbose:
                    print(f"  [scene changed] {sid}: parsed scene "
                          f"{changed.computed_fingerprint} does not match the stored "
                          f"scene {changed.stored_fingerprint}; nothing exported")
                continue
            except SdcIndexChangedError as changed:
                # Same shape as SceneChangedError just above, and the same
                # consequence — nothing for this scenario has been written yet, so
                # `continue` skips it whole, including the PerturbationSpace replay
                # below. That is fix G04's write-time guard AND the closest thing
                # this function has to guarding that replay against sdc_idx drift:
                # it never reaches PerturbationSpace for a scenario refused here.
                conn.rollback()
                summary['sdc_changed'].append({
                    'scenario_id': changed.scenario_id,
                    'stored_sdc_idx': changed.stored_sdc_idx,
                    'computed_sdc_idx': changed.computed_sdc_idx,
                })
                if verbose:
                    print(f"  [sdc changed] {sid}: parsed sdc_idx "
                          f"{changed.computed_sdc_idx} does not match the stored "
                          f"sdc_idx {changed.stored_sdc_idx}; nothing exported")
                continue
            summary['exported'] += 1
            summary['agents_written'] += written
            summary['agents_skipped'] += skipped
            if verbose:
                print(f"  {sid}: {written} agents written, {skipped} skipped")

            result = (stress_results or {}).get(sid)
            if result and result.get('status') == 'ok' \
                    and result.get('delta') is not None \
                    and result.get('target_idx') is not None:
                # Rebuild the perturbed trajectory rather than storing it in Pass 2:
                # the delta is 4 floats, the trajectory is 91 points. Storing the
                # small thing and replaying the physics keeps the DB honest — the
                # path shown is always the path the current simulator produces from
                # the stored delta.
                from src.optimization.perturbation_space import PerturbationSpace

                # THE REPLAY MODEL THIS RESULT WAS ACTUALLY VERIFIED UNDER (fix
                # G08), not whatever heading_speed_floor/heading_transition_width
                # happen to default to right now. _stress_one always builds its
                # search-time PerturbationSpace with no override either (see
                # batch_scorer.py), so these two are today's ambient module
                # constants at the MOMENT the search that produced this delta ran
                # — recorded into search_provenance for exactly this reason (each
                # is "a proposal until a real shard says how often this actually
                # matters"). A Pass 2 and its matching Pass 3 can straddle a later
                # change to either constant; without reusing the recorded value,
                # the SAT-verified collision this delta earned gets silently
                # replayed under a DIFFERENT model than the one that verified it.
                #
                # OMIT THE KWARG WHEN THE KEY IS ABSENT, DO NOT DEFAULT IT TO NONE.
                # PerturbationSpace treats heading_transition_width=None (and
                # heading_speed_floor=None) as a MEANINGFUL value — "the floor/band
                # is off" — not as "use the class default"; that only happens when
                # the keyword is omitted entirely (see its own docstring). A result
                # with no search_provenance at all (a caller predating this field,
                # or an outcome that never reached a search) has nothing recorded
                # to reproduce, so it must fall through to the class default exactly
                # as before this fix — not have every legacy result silently
                # replayed with the band/floor forced off.
                provenance = result.get('search_provenance') or {}
                replay_kwargs = {
                    key: provenance[key]
                    for key in ('heading_speed_floor', 'heading_transition_width')
                    if key in provenance
                }
                space = PerturbationSpace(
                    states, validity, types, sdc_idx, int(result['target_idx']),
                    **replay_kwargs
                )
                perturbed = space.apply(np.asarray(result['delta'], dtype=np.float32))
                try:
                    # delta and method are what make the run id derive from THIS
                    # content rather than from whatever is currently on the score row
                    # (audit R01). Without them the protection is vacuous — see
                    # export_perturbed_path's docstring.
                    #
                    # scene_fingerprint IS stored_fingerprint FROM THE CALL ABOVE, NOT
                    # RECOMPUTED (fix F05). export_scenario_agents already read this
                    # value under FOR UPDATE, already verified it against the parsed
                    # arrays (raising SceneChangedError on a mismatch, which this loop
                    # already catches and skips past), and already stamped this exact
                    # value onto its own rows. Recomputing here used to hand
                    # export_perturbed_path a NON-NULL fingerprint even for a
                    # scenario Pass 1 never fingerprinted (stored_fingerprint is
                    # None): export_perturbed_path's own carve-out only fires on
                    # `scene_fingerprint is None`, so a computed value sailed past it,
                    # the WHERE clause compared NULL against a real hash, and every
                    # legacy scenario's perturbed export was refused with
                    # SceneChangedError for no actual mismatch. Passing the SAME value
                    # export_scenario_agents already verified and stamped keeps the
                    # two tables' carve-out decisions consistent by construction,
                    # rather than by two independent reads that can disagree.
                    #
                    # A fresh read here (instead of reusing stored_fingerprint) would
                    # reopen a window this file already closed once: the FOR UPDATE
                    # lock above is released the moment export_scenario_agents
                    # commits, so a lookup after that point is a new snapshot, not a
                    # continuation of the locked one — exactly the cross-function gap
                    # the comment above this try block documents being closed for the
                    # baseline/perturbed pair. Reusing the value already in hand,
                    # rather than re-reading, is the same R03 precedent that comment
                    # already invokes.
                    written = export_perturbed_path(
                        conn, sid, perturbed, validity, int(result['target_idx']),
                        delta=result['delta'], method=result.get('method'),
                        scene_fingerprint=stored_fingerprint,
                    )
                except StaleExportError as stale:
                    # A refused export is a NORMAL outcome of a delayed or retried
                    # pass, not a crash — the same reasoning that makes
                    # replay_infeasible a status rather than an exception. Recorded
                    # in full and the walk continues.
                    conn.rollback()
                    summary['perturbed_stale'].append({
                        'scenario_id': stale.scenario_id,
                        'attempted_run_id': stale.attempted_run_id,
                        'current_run_id': stale.current_run_id,
                    })
                    if verbose:
                        print(f"    [stale] perturbed path refused: built from run "
                              f"{stale.attempted_run_id}, stored result is run "
                              f"{stale.current_run_id}")
                    written = False
                if written:
                    summary['perturbed_written'] += 1
                    if verbose:
                        print(f"    + perturbed path (target {result['target_idx']})")

        except Exception as e:  # noqa: BLE001 — deliberate: isolate per scenario
            # Roll back so a failed scenario cannot poison the next one's transaction.
            conn.rollback()
            summary['errors'].append({
                'index': idx, 'scenario_id': sid,
                'error': f'{type(e).__name__}: {e}',
            })
            if verbose:
                print(f"  [skip] record {idx} ({sid}): {e}")
            continue

    if verbose:
        if wanted:
            print(f"  warning: {len(wanted)} requested IDs not found in shard")
        print(f"Geometry pass done: {summary['exported']} scenarios, "
              f"{summary['agents_written']} agents, "
              f"{len(summary['perturbed_stale'])} stale exports refused, "
              f"{len(summary['errors'])} errors, "
              f"{time.time() - t_start:.1f}s")
    return summary
