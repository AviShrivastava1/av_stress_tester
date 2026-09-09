"""
routes.py — Phase 6.

Six GET endpoints over the Phase 5 scores and the Phase 6 PostGIS geometry.

Read-only by design — see src/api/__init__.py for why nothing here starts a Phase 4
run. Every response model lives in models.py; every row-to-model mapping goes
through one helper here so the summary and detail paths cannot drift apart.
"""

import base64
import json

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.config import settings
from src.api.db_pool import dict_cursor, get_db
from src.api.models import (
    AgentTrack, HealthResponse, PerturbedResponse, ScenarioDetail,
    ScenarioPage, ScenarioSummary, StatsResponse, TrajectoryResponse,
)

router = APIRouter()


# The columns every scenario response needs. Kept in one constant so the list
# query, the detail query and the perturbed query cannot disagree about them.
_SCORE_COLUMNS = """
    scenario_id, shard, n_agents, min_ttc, min_pet, fragility_score,
    min_perturbation, collision_timestep, stress_method, stress_tested_at
"""


# ── row mapping ─────────────────────────────────────────────────────────────────

def _summary_fields(row) -> dict:
    """
    Map a scenario_scores row to the fields shared by ScenarioSummary and
    ScenarioDetail.

    This returns a plain dict, and both response models are constructed from a row
    directly — ScenarioDetail is NEVER built by converting a ScenarioSummary
    instance. Model-to-model conversion would mean calling .model_dump() (Pydantic
    v2 only) or .dict() (v1 only), binding this file to one major version for no
    benefit. A dict is a dict in every version.

    The two booleans are the whole point: see the models.py docstring for why a
    NULL min_perturbation is ambiguous and must be resolved here rather than in the
    frontend.
    """
    stress_tested = row['stress_tested_at'] is not None
    return {
        'scenario_id': row['scenario_id'],
        'shard': row['shard'],
        'n_agents': row['n_agents'],
        'min_ttc': row['min_ttc'],
        'min_pet': row['min_pet'],
        'fragility_score': row['fragility_score'],
        'min_perturbation': row['min_perturbation'],
        'collision_timestep': row['collision_timestep'],
        'stress_method': row['stress_method'],
        'stress_tested': stress_tested,
        # Phase 4 ran, and found no collision within bounds.
        'robustly_safe': stress_tested and row['min_perturbation'] is None,
    }


def _geojson_xy(geojson_str: str) -> list[list[float]]:
    """Pull [[x, y], ...] out of a GeoJSON LineString produced by ST_AsGeoJSON."""
    coords = json.loads(geojson_str)['coordinates']
    return [[float(c[0]), float(c[1])] for c in coords]


def _make_track(row, *, geojson, measures, agent_idx, is_sdc,
                agent_type=None, length_m=None, width_m=None,
                label='') -> AgentTrack:
    """
    Assemble an AgentTrack, refusing to guess if the pieces do not line up.

    `path`, `timesteps` and `headings` describe the SAME vertices and must be the
    same length. If they are not, something upstream is inconsistent — a partially
    written export, a hand-edited row — and there is no safe way to continue.

    THERE IS DELIBERATELY NO FALLBACK HERE, and the temptation is specific: when
    the M ordinate is missing it is very natural to substitute the vertex index
    (`coords[i][2] if len(coords[i]) > 2 else i`). That is wrong in a way designed
    to escape detection. Index equals timestep ONLY when every timestep is valid.
    The moment an agent has a validity gap — which is ordinary in real WOMD data,
    and never happens in synthetic test fixtures where all 91 steps are valid —
    the index and the true timestep diverge, and the endpoint starts returning
    plausible-looking frames that are quietly attached to the wrong moments in
    time. A test built on fully-valid data passes forever while the real data is
    silently corrupted.

    So: raise. A 500 that names the mismatch is strictly better than a chart that
    looks right and is not.
    """
    path = _geojson_xy(geojson)
    timesteps = [float(m) for m in (measures or [])]
    headings = [float(h) for h in (row['headings'] or [])]

    if len(timesteps) != len(path) or len(headings) != len(path):
        raise HTTPException(
            status_code=500,
            detail=(f"Geometry inconsistent for {label}: "
                    f"{len(path)} coordinates, {len(timesteps)} measures, "
                    f"{len(headings)} headings. Refusing to pad or truncate."),
        )

    return AgentTrack(
        agent_idx=agent_idx,
        agent_type=agent_type,
        is_sdc=is_sdc,
        length_m=length_m,
        width_m=width_m,
        path=path,
        timesteps=timesteps,
        headings=headings,
    )


# ── cursor encoding ─────────────────────────────────────────────────────────────
#
# THE CURSOR'S FLOAT MUST ROUND-TRIP BIT-EXACTLY. This is load-bearing, not
# pedantry.
#
# The pagination predicate below contains `fragility_score = %s` — an exact
# IEEE-754 equality against a DOUBLE PRECISION column. That equality is what
# advances the cursor correctly through rows that TIE on fragility_score. If the
# float in the cursor is not bit-identical to the value in the row:
#
#     rounded DOWN -> tied rows match neither branch of the predicate
#                     (not `f < f0`, not `f = f0`) and VANISH from the results
#     rounded UP   -> tied rows match `f < f0` and are RE-SERVED as duplicates
#
# Both failures are silent. No error, no warning — just wrong pages, and only when
# scores happen to tie.
#
# What makes the current design safe: json.dumps/json.loads use Python's
# shortest-repr algorithm, which round-trips every finite binary64 value exactly,
# and psycopg2 adapts a Python float via repr(), so PostgreSQL parses back the
# identical bits. Nothing here needs changing — it needs PROTECTING, because it
# looks like a detail somebody would "tidy up".
#
# Rules for anyone editing this:
#   * take fragility_score VERBATIM from the last row of the page — never
#     recompute it, re-derive it, or re-query it
#   * NEVER round it, truncate it, format it (f"{f:.6f}"), stringify it for
#     "readability", or route it through Decimal
#   * the cursor is opaque to clients precisely so that its precision is nobody
#     else's business

def _encode_cursor(fragility_score: float, scenario_id: str) -> str:
    """Opaque cursor: base64 of {"f": <exact float>, "s": <scenario id>}."""
    payload = json.dumps(
        {'f': fragility_score, 's': scenario_id},
        separators=(',', ':'),
    )
    return base64.urlsafe_b64encode(payload.encode('utf-8')).decode('ascii')


def _decode_cursor(cursor: str) -> tuple[float, str]:
    """
    Decode a cursor, or fail with 422.

    A malformed cursor is a bad REQUEST, not a server fault — the client sent
    something we cannot parse. Letting the base64/JSON error escape would produce a
    500 and page whoever is on call for what is really a client bug.
    """
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode('ascii')))
        return float(payload['f']), str(payload['s'])
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Malformed cursor. Pass back a next_cursor value verbatim.",
        )


# ── endpoints ───────────────────────────────────────────────────────────────────

@router.get('/health', response_model=HealthResponse)
def health(conn=Depends(get_db)):
    """
    Liveness check that actually touches the database.

    A health check that only returns {"status": "ok"} without querying will report
    healthy while the database is unreachable — which is the exact failure it
    exists to catch. So this executes a real round-trip.

    PostGIS being absent is reported as null rather than raised: the scores
    endpoints work fine without it and only geometry is unavailable. That is
    degraded, not dead, and a health check should be able to say the difference.
    """
    with dict_cursor(conn) as cur:
        cur.execute('SELECT 1 AS ok')
        cur.fetchone()

        postgis = None
        try:
            cur.execute('SELECT PostGIS_Version() AS v')
            postgis = cur.fetchone()['v']
        except Exception:
            # The failed statement aborts the transaction; roll back so this
            # connection is usable by the next request that borrows it from the
            # pool. Without this, every later query on it fails with
            # "current transaction is aborted".
            conn.rollback()

    return HealthResponse(status='ok', database='connected', postgis=postgis)


@router.get('/stats', response_model=StatsResponse)
def stats(conn=Depends(get_db)):
    """
    Corpus-level counts for the dashboard header.

    `count(*) FILTER (WHERE ...)` computes every count in ONE pass over the table,
    rather than issuing four separate COUNT queries that each scan it again.
    """
    with dict_cursor(conn) as cur:
        cur.execute("""
            SELECT
                count(*)                                        AS total_scenarios,
                count(*) FILTER (WHERE stress_tested_at IS NOT NULL)
                                                                AS stress_tested,
                count(*) FILTER (WHERE stress_tested_at IS NOT NULL
                                   AND min_perturbation IS NOT NULL)
                                                                AS collisions_found,
                count(*) FILTER (WHERE stress_tested_at IS NOT NULL
                                   AND min_perturbation IS NULL)
                                                                AS robustly_safe,
                min(fragility_score)                            AS fragility_min,
                max(fragility_score)                            AS fragility_max,
                avg(fragility_score)                            AS fragility_mean
            FROM scenario_scores
        """)
        row = cur.fetchone()

        # Pass 3 may never have run — the geometry tables can legitimately not
        # exist yet. That is a normal state for a fresh database, so report 0
        # rather than 500. The rollback clears the aborted transaction so this
        # pooled connection stays usable.
        with_geometry = 0
        try:
            cur.execute(
                'SELECT count(DISTINCT scenario_id) AS n FROM scenario_agents'
            )
            with_geometry = cur.fetchone()['n']
        except Exception:
            conn.rollback()

    return StatsResponse(
        total_scenarios=row['total_scenarios'],
        stress_tested=row['stress_tested'],
        collisions_found=row['collisions_found'],
        robustly_safe=row['robustly_safe'],
        with_geometry=with_geometry,
        fragility_min=row['fragility_min'],
        fragility_max=row['fragility_max'],
        fragility_mean=float(row['fragility_mean']) if row['fragility_mean'] is not None else None,
    )


@router.get('/scenarios', response_model=ScenarioPage)
def list_scenarios(
    limit: int = Query(default=settings.page_size, ge=1,
                       le=settings.max_page_size),
    cursor: str | None = Query(default=None),
    stress_tested_only: bool = Query(default=False),
    conn=Depends(get_db),
):
    """
    The ranked list, most fragile first, keyset-paginated.

    ORDERING
    --------
    `ORDER BY fragility_score DESC, scenario_id` is byte-identical to
    ranker.rank_scenarios' sort key `(-fragility_score, str(scenario_id))`. That
    equivalence is deliberate: if the API and the batch pipeline ordered
    differently, "rank 3" would mean two different scenarios depending on who you
    asked, which is indefensible in a safety audit.

    The scenario_id tiebreak is LOAD-BEARING, not cosmetic. Without a unique second
    key, rows tied on fragility_score have no defined order, and the same row can
    be served on two consecutive pages (or skipped entirely) as the database
    happens to return ties differently between queries.

    WHY KEYSET AND NOT LIMIT/OFFSET
    -------------------------------
    OFFSET is not just slow, it is WRONG under concurrent writes: if a row is
    inserted above the current position between two requests, every subsequent row
    shifts down by one, and the client silently sees a duplicate at the page
    boundary (or, on delete, a gap). Keyset pagination anchors on a value rather
    than a count, so the page boundary stays put no matter what else is written.
    That stability is why this is here.

    WHAT THIS QUERY COSTS — AND WHY IT IS NOT A SEEK
    ------------------------------------------------
    PostgreSQL compiles a ROW COMPARISON, `(a, b) < (c, d)`, into an Index Cond —
    a genuine seek straight to the cursor position. But a row comparison is only
    VALID when every column sorts the same direction, because it is lexicographic
    in one direction by definition. Our ordering is mixed — fragility_score DESC,
    scenario_id ASC — so the correct predicate after a cursor (f0, s0) is:

        f < f0  OR  (f = f0 AND s > s0)

    and an OR-form predicate is NOT recognized as an index condition. It becomes a
    Filter: the scan starts at the top of the index and discards every row before
    the cursor. Measured on a 50k-row table with the composite index present:

        tuple form (f, s) < (f0, s0)   Index Only Scan + Index Cond   0.03 ms
        OR form (what we use)          Index Only Scan + Filter       8.68 ms
                                       (25005 rows removed by filter)

    So: we keep keyset pagination's CORRECTNESS property (stable page boundaries,
    no duplicates, no gaps) and give up its PERFORMANCE property — page N again
    costs proportional to N, the same asymptotics as OFFSET. Do not describe this
    query as a seek.

    Both properties are recoverable by ordering `fragility_score DESC, scenario_id
    DESC` and using the tuple form, which in ranker.py is
    `sorted(records, key=lambda r: (r['fragility_score'], r['scenario_id']),
    reverse=True)` — same direction on both columns. That is the escape hatch if
    deep paging ever becomes real. It is not taken here because it would change
    Phase 5 code to buy nothing at this scale: a shard holds a few hundred
    scenarios, and the dashboard will never page deep enough for a filter over a
    few thousand rows to be noticeable. Matching the ranker's ordering is worth
    more than the seek today.

    The composite index (fragility_score DESC, scenario_id), created in
    export_geometry.GEOMETRY_SCHEMA_SQL, still earns its keep: it serves the
    ORDER BY, so this is an Index Only Scan rather than a sort.
    """
    where = []
    params: list = []

    if cursor is not None:
        f0, s0 = _decode_cursor(cursor)
        # The exact-equality branch is why the cursor's float precision matters —
        # see the encoding comment above.
        where.append('(fragility_score < %s OR (fragility_score = %s AND scenario_id > %s))')
        params.extend([f0, f0, s0])

    if stress_tested_only:
        where.append('stress_tested_at IS NOT NULL')

    where_sql = f"WHERE {' AND '.join(where)}" if where else ''

    # Fetch one extra row to learn whether a further page exists, without a second
    # COUNT query over the whole table.
    params.append(limit + 1)

    with dict_cursor(conn) as cur:
        cur.execute(f"""
            SELECT {_SCORE_COLUMNS}
            FROM scenario_scores
            {where_sql}
            ORDER BY fragility_score DESC, scenario_id
            LIMIT %s
        """, params)
        rows = cur.fetchall()

    has_more = len(rows) > limit
    rows = rows[:limit]

    next_cursor = None
    if has_more and rows:
        last = rows[-1]
        # Verbatim from the row. Not rounded, not reformatted.
        next_cursor = _encode_cursor(last['fragility_score'], last['scenario_id'])

    return ScenarioPage(
        items=[ScenarioSummary(**_summary_fields(r)) for r in rows],
        next_cursor=next_cursor,
        limit=limit,
    )


@router.get('/scenarios/{scenario_id}', response_model=ScenarioDetail)
def get_scenario(scenario_id: str, conn=Depends(get_db)):
    """One scenario's full row, including the raw perturbation vector."""
    with dict_cursor(conn) as cur:
        cur.execute(f"""
            SELECT {_SCORE_COLUMNS}, delta
            FROM scenario_scores
            WHERE scenario_id = %s
        """, (scenario_id,))
        row = cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"Unknown scenario: {scenario_id}")

    # Built from the row, not converted from a ScenarioSummary — see _summary_fields.
    delta = [float(x) for x in row['delta']] if row['delta'] is not None else None
    return ScenarioDetail(**_summary_fields(row), delta=delta)


@router.get('/scenarios/{scenario_id}/trajectories',
            response_model=TrajectoryResponse)
def get_trajectories(scenario_id: str, conn=Depends(get_db)):
    """
    Every agent's logged path for one scenario.

    404 only when the scenario itself is unknown — checked against scenario_scores,
    because "this scenario exists but its geometry has not been exported yet" is a
    normal state that must not look like a missing scenario. That case returns 200
    with an empty agents list.

    ST_AsGeoJSON does the geometry-to-JSON conversion inside PostgreSQL: less data
    on the wire than WKT, no geometry library needed in the API process, and the
    result drops straight into a deck.gl PathLayer.

    BUT GeoJSON has no M dimension, and ST_AsGeoJSON silently DROPS it —
    `LINESTRING M(1 2 0, 3 4 1)` comes back as `[[1,2],[3,4]]` with the measures
    gone. Since M is what tells the frontend which timestep each vertex belongs to,
    it is fetched alongside with ST_DumpPoints + ST_M, ordered by vertex position
    so it stays index-aligned with the coordinates.
    """
    with dict_cursor(conn) as cur:
        cur.execute('SELECT 1 FROM scenario_scores WHERE scenario_id = %s',
                    (scenario_id,))
        if cur.fetchone() is None:
            raise HTTPException(status_code=404,
                                detail=f"Unknown scenario: {scenario_id}")

        cur.execute("""
            SELECT sa.agent_idx, sa.agent_type, sa.is_sdc,
                   sa.length_m, sa.width_m, sa.headings,
                   ST_AsGeoJSON(sa.path) AS geojson,
                   ARRAY(SELECT ST_M(dp.geom)
                           FROM ST_DumpPoints(sa.path) dp
                          ORDER BY dp.path) AS measures
            FROM scenario_agents sa
            WHERE sa.scenario_id = %s
            ORDER BY sa.agent_idx
        """, (scenario_id,))
        rows = cur.fetchall()

    agents = [
        _make_track(
            r,
            geojson=r['geojson'], measures=r['measures'],
            agent_idx=r['agent_idx'], is_sdc=r['is_sdc'],
            agent_type=r['agent_type'],
            length_m=r['length_m'], width_m=r['width_m'],
            label=f"{scenario_id} agent {r['agent_idx']}",
        )
        for r in rows
    ]
    return TrajectoryResponse(scenario_id=scenario_id, agents=agents)


@router.get('/scenarios/{scenario_id}/perturbed',
            response_model=PerturbedResponse)
def get_perturbed(scenario_id: str, conn=Depends(get_db)):
    """
    The challenger's logged path next to its minimally-perturbed one — the Phase 4
    result, made drawable.

    404 ONLY when the scenario is unknown. A scenario that exists but has not been
    stress-tested returns 200 with null paths: "nothing to show yet" is a normal
    state in a pipeline whose expensive pass runs over a chosen subset, and a
    frontend should render an empty panel for it, not an error.
    """
    with dict_cursor(conn) as cur:
        cur.execute("""
            SELECT min_perturbation, collision_timestep, delta
            FROM scenario_scores
            WHERE scenario_id = %s
        """, (scenario_id,))
        score_row = cur.fetchone()

        if score_row is None:
            raise HTTPException(status_code=404,
                                detail=f"Unknown scenario: {scenario_id}")

        cur.execute("""
            SELECT pp.target_idx, pp.headings,
                   ST_AsGeoJSON(pp.path) AS geojson,
                   ARRAY(SELECT ST_M(dp.geom)
                           FROM ST_DumpPoints(pp.path) dp
                          ORDER BY dp.path) AS measures
            FROM perturbed_paths pp
            WHERE pp.scenario_id = %s
        """, (scenario_id,))
        pert_row = cur.fetchone()

        baseline = None
        perturbed = None
        if pert_row is not None:
            # The baseline is the SAME agent's logged path, so the frontend can
            # draw "what happened" against "what nearly happened".
            cur.execute("""
                SELECT sa.agent_idx, sa.agent_type, sa.is_sdc,
                       sa.length_m, sa.width_m, sa.headings,
                       ST_AsGeoJSON(sa.path) AS geojson,
                       ARRAY(SELECT ST_M(dp.geom)
                               FROM ST_DumpPoints(sa.path) dp
                              ORDER BY dp.path) AS measures
                FROM scenario_agents sa
                WHERE sa.scenario_id = %s AND sa.agent_idx = %s
            """, (scenario_id, pert_row['target_idx']))
            base_row = cur.fetchone()

            if base_row is not None:
                baseline = _make_track(
                    base_row,
                    geojson=base_row['geojson'], measures=base_row['measures'],
                    agent_idx=base_row['agent_idx'], is_sdc=base_row['is_sdc'],
                    agent_type=base_row['agent_type'],
                    length_m=base_row['length_m'], width_m=base_row['width_m'],
                    label=f"{scenario_id} agent {base_row['agent_idx']} (baseline)",
                )

            perturbed = _make_track(
                pert_row,
                geojson=pert_row['geojson'], measures=pert_row['measures'],
                agent_idx=pert_row['target_idx'],
                is_sdc=False,          # the SDC is never the perturbed agent
                agent_type=base_row['agent_type'] if base_row else None,
                length_m=base_row['length_m'] if base_row else None,
                width_m=base_row['width_m'] if base_row else None,
                label=f"{scenario_id} agent {pert_row['target_idx']} (perturbed)",
            )

    delta = ([float(x) for x in score_row['delta']]
             if score_row['delta'] is not None else None)

    return PerturbedResponse(
        scenario_id=scenario_id,
        target_idx=pert_row['target_idx'] if pert_row else None,
        delta=delta,
        min_perturbation=score_row['min_perturbation'],
        collision_timestep=score_row['collision_timestep'],
        baseline=baseline,
        perturbed=perturbed,
    )
