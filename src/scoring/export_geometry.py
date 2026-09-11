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


# Re-exported rather than reimplemented: PerturbationSpace and the torch margin
# answer "where does this agent start?" with the same helper, and three modules
# quietly disagreeing about that was audit findings B01 and B02.
from src.data.validity import valid_timesteps as _valid_timesteps


# ── per-scenario exports ────────────────────────────────────────────────────────

def export_scenario_agents(conn, scenario_id, states, validity, types, sdc_idx):
    """
    Write every agent's logged trajectory for one scenario.

    Args:
        states:   (N, T, 7) = [x, y, vx, vy, heading, length, width]
        validity: (N, T) bool
        types:    (N,) int — 1 vehicle, 2 pedestrian, 3 cyclist
        sdc_idx:  index of the self-driving car

    Returns:
        (n_written, n_skipped)

    An agent with fewer than 2 valid timesteps cannot form a linestring — a
    LINESTRING needs at least two vertices. Those agents are skipped and counted,
    never raised on: a scenario where one agent blinks in for a single frame is
    normal data, not an error.
    """
    n_agents = states.shape[0]
    written = skipped = 0

    with conn.cursor() as cur:
        for i in range(n_agents):
            ts = _valid_timesteps(validity, i)
            if len(ts) < 2:
                skipped += 1
                continue

            xs = states[i, ts, 0]
            ys = states[i, ts, 1]
            # M ordinate = the timestep index itself, so the frontend can map a
            # vertex back to a frame without a second lookup.
            wkt = _linestring_m_wkt(xs, ys, ts.astype(float))
            headings = [float(h) for h in states[i, ts, 4]]

            # length/width are physical constants of the agent; read them from the
            # first valid timestep rather than assuming index 0 was observed.
            t0 = int(ts[0])

            cur.execute("""
                INSERT INTO scenario_agents
                    (scenario_id, agent_idx, agent_type, is_sdc,
                     length_m, width_m, n_points, headings, path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, ST_GeomFromText(%s, 0))
                ON CONFLICT (scenario_id, agent_idx) DO UPDATE SET
                    agent_type = EXCLUDED.agent_type,
                    is_sdc     = EXCLUDED.is_sdc,
                    length_m   = EXCLUDED.length_m,
                    width_m    = EXCLUDED.width_m,
                    n_points   = EXCLUDED.n_points,
                    headings   = EXCLUDED.headings,
                    path       = EXCLUDED.path
            """, (
                scenario_id, int(i), int(types[i]), bool(i == sdc_idx),
                float(states[i, t0, 5]), float(states[i, t0, 6]),
                len(ts), headings, wkt,
            ))
            written += 1

    conn.commit()
    return written, skipped


def export_perturbed_path(conn, scenario_id, perturbed_states, validity, target_idx):
    """
    Write the challenger's PERTURBED trajectory — the Phase 4 answer, made visible.

    This is the row that lets the dashboard draw "here is what the car actually did,
    and here is the smallest change that would have caused a crash" as two paths on
    one map. Only the challenger is stored: the SDC and every other agent keep their
    logged trajectory by construction (PerturbationSpace.apply only rewrites the
    target), so re-storing them would duplicate scenario_agents for no gain.

    Returns True if a row was written, False if the target had too few valid
    timesteps to form a linestring.
    """
    ts = _valid_timesteps(validity, target_idx)
    if len(ts) < 2:
        return False

    xs = perturbed_states[target_idx, ts, 0]
    ys = perturbed_states[target_idx, ts, 1]
    wkt = _linestring_m_wkt(xs, ys, ts.astype(float))
    headings = [float(h) for h in perturbed_states[target_idx, ts, 4]]

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO perturbed_paths
                (scenario_id, target_idx, n_points, headings, path)
            VALUES (%s, %s, %s, %s, ST_GeomFromText(%s, 0))
            ON CONFLICT (scenario_id) DO UPDATE SET
                target_idx  = EXCLUDED.target_idx,
                n_points    = EXCLUDED.n_points,
                headings    = EXCLUDED.headings,
                path        = EXCLUDED.path,
                exported_at = now()
        """, (scenario_id, int(target_idx), len(ts), headings, wkt))
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
    perturbed_written, errors (list of dicts).

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
        'errors': [],
    }
    t_start = time.time()

    for idx, raw in enumerate(ShardLoader(shard_path)):
        if not wanted:
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

            written, skipped = export_scenario_agents(
                conn, sid, states, validity, types, sdc_idx
            )
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

                space = PerturbationSpace(
                    states, validity, types, sdc_idx, int(result['target_idx'])
                )
                perturbed = space.apply(np.asarray(result['delta'], dtype=np.float32))
                if export_perturbed_path(conn, sid, perturbed, validity,
                                         int(result['target_idx'])):
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
              f"{len(summary['errors'])} errors, "
              f"{time.time() - t_start:.1f}s")
    return summary
