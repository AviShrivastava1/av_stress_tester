"""
test_api_robustness.py — Batch 3 regression tests (audit B12, B13, B17, B18).

Two unrelated guarantees, kept separate rather than forced into one story:

  1. A request against a valid scenario in a normal pipeline state never returns 500.
     B12 (geometry before Pass 3), B17 (malformed client input), B18 (pool exhaustion).

  2. A response never labels a value using a parameterization it wasn't produced under.
     B13.

The audit's own fixtures in tests/test_audit_db.py cover the straightforward half of
each. These add the states those fixtures structurally cannot reach — above all a
pedestrian scenario that has been stress-tested but NOT yet geometry-exported, which the
audit's B13 fixture cannot produce because it always exports.

Needs a DISPOSABLE Postgres/PostGIS database; skips unless AV_ROBUSTNESS_DB=1.

Run:
    AV_ROBUSTNESS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_api_robustness.py -q
"""

import base64
import json
import os
import sys
import threading
import time
from urllib.parse import quote

import numpy as np
import pytest
from fastapi.testclient import TestClient
from psycopg2.pool import ThreadedConnectionPool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import db_pool, main as api_main
from src.api.models import DELTA_LABELS, LINEAR_DELTA_LABELS
from src.scoring import db
from src.scoring.batch_scorer import _stress_one
from src.scoring.export_geometry import (
    export_perturbed_path, export_scenario_agents, init_geometry_schema,
)

requires_db = pytest.mark.skipif(
    os.environ.get('AV_ROBUSTNESS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

DE_KWARGS = {'popsize': 8, 'maxiter': 40, 'seed': 1}


# ── fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def bare_conn():
    """Scores table only — the state before Pass 3 has ever run."""
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, '
                    'scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    yield connection
    connection.rollback()
    connection.close()


@pytest.fixture
def conn(bare_conn):
    """Scores plus geometry tables."""
    init_geometry_schema(bare_conn)
    return bare_conn


def _client(connection, monkeypatch):
    monkeypatch.setattr(api_main, 'init_pool', lambda: None)
    monkeypatch.setattr(api_main, 'close_pool', lambda: None)
    app = api_main.create_app()
    app.dependency_overrides[db_pool.get_db] = lambda: connection
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client(conn, monkeypatch):
    with _client(conn, monkeypatch) as c:
        yield c


@pytest.fixture
def bare_client(bare_conn, monkeypatch):
    with _client(bare_conn, monkeypatch) as c:
        yield c


def _seed(connection, sid='scene', n=2):
    db.upsert_scores(connection, [dict(scenario_id=sid, shard='synthetic', n_agents=n,
                                       min_ttc=9.0, min_pet=9.0, fragility_score=1.0)])


def _pedestrian_scene():
    """A vehicle SDC at the origin and a pedestrian walking into it."""
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 5:7] = [0.6, 0.6]
    states[1, :, 0] = 5.0 - 2.0 * np.arange(10) * 0.1
    states[1, :, 2] = -2.0
    states[1, :, 4] = np.pi
    return states, np.ones((2, 10), dtype=bool), np.array([1, 2])


def _vehicle_scene():
    """
    The crossing geometry tests/test_api.py uses for syn_close, which is known to
    produce a verified collision under perturbation.

    An earlier draft used two cars nose-to-tail 6 m apart. That scene's own ZERO-delta
    replay collides — 4.5 m boxes closing to a 4.2 m centre gap — so Batch 1's replay
    fidelity gate refused it outright and no search ever ran. The gate was right and the
    fixture was wrong; noted because it is an easy scene to write by accident.
    """
    t = np.arange(91) * 0.1
    states = np.zeros((2, 91, 7), dtype=np.float32)
    states[0, :, 0] = -20.0 + 10.0 * t
    states[0, :, 2] = 10.0
    states[1, :, 0] = 0.0
    states[1, :, 1] = -28.0 + 10.0 * t
    states[1, :, 3] = 10.0
    states[1, :, 4] = np.pi / 2
    states[:, :, 5:7] = [4.5, 2.0]
    return states, np.ones((2, 91), dtype=bool), np.array([1, 1])


def _stress_and_store(connection, states, validity, types, sid='scene'):
    """
    Drive the REAL write path: _stress_one -> update_stress_results.

    Deliberately not a hand-built result dict. The point of the tests below is that a
    particular database state is reachable by the actual pipeline, and a fabricated row
    could assert that while the pipeline never produces it.
    """
    result = _stress_one(states, validity, types, 0, de_kwargs=DE_KWARGS)
    db.update_stress_results(connection, {sid: result})
    return result


# ── B13: the parameterization must travel with the delta ────────────────────────

@requires_db
def test_pedestrian_labels_are_linear_before_geometry_is_exported(conn, client):
    """
    THE state the audit's own B13 fixture cannot reach.

    That fixture calls store_result(..., export=True), so scenario_agents always exists
    and agent_type is always readable. A scenario that has been stress-tested but not
    yet exported has no agent type to infer from — and its labels have to be right
    anyway. A fix that only reads base_row['agent_type'] passes the audit and fails
    here.

    Runs end to end: _stress_one writes the provenance, update_stress_results persists
    it, and the assertion reads the response over HTTP.
    """
    states, validity, types = _pedestrian_scene()
    _seed(conn)
    result = _stress_and_store(conn, states, validity, types)
    assert result['outcome'] == 'collision_found', f'fixture regressed: {result}'

    # no export_scenario_agents / export_perturbed_path call: geometry is absent
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM scenario_agents WHERE scenario_id='scene'")
        assert cur.fetchone()[0] == 0, 'fixture must have NO exported geometry'

    data = client.get('/scenarios/scene/perturbed').json()

    assert data['delta'] is not None, 'the delta itself must still be served'
    assert data['baseline'] is None and data['perturbed'] is None
    assert data['delta_labels'] == LINEAR_DELTA_LABELS, data['delta_labels']
    assert not any('rad' in label for label in data['delta_labels'])


@requires_db
def test_vehicle_labels_are_bicycle_before_geometry_is_exported(conn, client):
    """The control: the same un-exported state must not label everything linear."""
    states, validity, types = _vehicle_scene()
    _seed(conn)
    _stress_and_store(conn, states, validity, types)

    data = client.get('/scenarios/scene/perturbed').json()
    assert data['delta_labels'] == DELTA_LABELS, data['delta_labels']


@requires_db
def test_provenance_wins_over_agent_type_when_they_disagree(conn, client):
    """
    Priority 1 beats priority 2. Provenance is the model actually dispatched to;
    agent_type is an inference. Where both exist and disagree, the recorded fact wins.
    """
    states, validity, types = _pedestrian_scene()
    _seed(conn)
    _stress_and_store(conn, states, validity, types)
    export_scenario_agents(conn, 'scene', states, validity, types, 0)

    # Force the exported agent row to claim VEHICLE while provenance says linear.
    with conn.cursor() as cur:
        cur.execute("UPDATE scenario_agents SET agent_type = 1 "
                    "WHERE scenario_id='scene' AND agent_idx=1")
        cur.execute("SELECT search_provenance ->> 'delta_parameterization' "
                    "FROM scenario_scores WHERE scenario_id='scene'")
        assert cur.fetchone()[0] == 'linear'
    conn.commit()
    export_perturbed_path(conn, 'scene', states, validity, 1)

    data = client.get('/scenarios/scene/perturbed').json()
    assert data['delta_labels'] == LINEAR_DELTA_LABELS, (
        'agent_type overrode the recorded parameterization'
    )


@requires_db
def test_a_batch2_era_row_with_provenance_but_no_parameterization_key(conn, client):
    """
    provenance IS NULL and provenance EXISTS BUT LACKS THE KEY are different database
    states, and only the first arises naturally from the audit's fixtures.

    A row written by Batch 2 has a fully populated search_provenance — bounds, DE
    budget, baseline_replay_error, challengers — and no delta_parameterization, because
    the key did not exist yet. `->> 'missing_key'` returns SQL NULL exactly as it does
    for a NULL column, so this SHOULD fall through to priority 2 — but "should" is the
    word that precedes most of this project's defects, and nothing else in the suite
    writes this shape on purpose.
    """
    states, validity, types = _pedestrian_scene()
    _seed(conn)
    _stress_and_store(conn, states, validity, types)

    # Strip only the new key, leaving a genuine Batch-2-era provenance blob behind.
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE scenario_scores
               SET search_provenance = search_provenance - 'delta_parameterization'
             WHERE scenario_id = 'scene'
        """)
        cur.execute("SELECT search_provenance FROM scenario_scores "
                    "WHERE scenario_id='scene'")
        provenance = cur.fetchone()[0]
    conn.commit()

    assert provenance is not None and 'delta_parameterization' not in provenance
    assert 'baseline_replay_error' in provenance, (
        'fixture must be a REAL Batch-2 blob, not an empty object'
    )

    # With no geometry either, nothing can establish the parameterization.
    assert client.get('/scenarios/scene/perturbed').json()['delta_labels'] is None

    # Export geometry, and priority 2 resolves it from the agent type.
    export_scenario_agents(conn, 'scene', states, validity, types, 0)
    export_perturbed_path(conn, 'scene', states, validity, 1)
    data = client.get('/scenarios/scene/perturbed').json()
    assert data['delta_labels'] == LINEAR_DELTA_LABELS, data['delta_labels']


@requires_db
@pytest.mark.parametrize('break_path', ['absent', 'stale'])
def test_agent_type_is_reachable_without_the_perturbed_path(conn, client, break_path):
    """
    BATCH 10: priority 2 no longer depends on perturbed_paths.

    _resolve_delta_labels' agent-type fallback reads scenario_agents, but the read was
    nested inside `if pert_row is not None` and indexed by pert_row['target_idx'] — so
    a row with exported geometry lost its labels whenever the perturbed path was
    missing or stale, for a reason that has nothing to do with which units a delta
    carries. R06 put target_idx on scenario_scores, which makes the agent reachable
    directly.

    Both ways the path goes away, because they are different SQL:

      absent  the row is gone — update_stress_results deletes it when a new result
              lands, so this is the window between a re-run Pass 2 and the Pass 3 that
              has not caught up
      stale   the row is there with a different stress_run_id, so B14's
              IS NOT DISTINCT FROM join declines to serve it

    The row is a PEDESTRIAN, so a wrong answer here is not a near miss: the vehicle
    labels would present metres per second as radians, which is B13 exactly.
    """
    states, validity, types = _pedestrian_scene()
    _seed(conn)
    _stress_and_store(conn, states, validity, types)
    export_scenario_agents(conn, 'scene', states, validity, types, 0)
    export_perturbed_path(conn, 'scene', states, validity, 1)

    with conn.cursor() as cur:
        # Make the row legacy: a delta with no provenance, which is the only state in
        # which priority 2 is consulted at all.
        cur.execute("UPDATE scenario_scores SET search_provenance = NULL "
                    "WHERE scenario_id = 'scene'")
        if break_path == 'absent':
            cur.execute("DELETE FROM perturbed_paths WHERE scenario_id = 'scene'")
        else:
            cur.execute("UPDATE perturbed_paths SET stress_run_id = 'deadbeefdeadbeef' "
                        "WHERE scenario_id = 'scene'")
        cur.execute("SELECT target_idx FROM scenario_scores WHERE scenario_id='scene'")
        target_idx = cur.fetchone()[0]
    conn.commit()

    assert target_idx == 1, (
        'fixture regressed: target_idx must be on the score row for this to be '
        'reachable at all'
    )

    data = client.get('/scenarios/scene/perturbed').json()

    assert data['perturbed'] is None, (
        f'fixture regressed: the perturbed path should be unreachable ({break_path})'
    )
    assert data['delta_labels'] == LINEAR_DELTA_LABELS, (
        f'a pedestrian delta lost its labels because the PERTURBED PATH was '
        f'{break_path}: {data["delta_labels"]}'
    )
    assert not any('rad' in label for label in data['delta_labels']), data['delta_labels']

    # NOTHING ELSE ABOUT THE RESPONSE MOVES. The narrow fix resolves a label; it does
    # not start serving a baseline track where none was served before, which is the
    # broader change it was deliberately scoped away from.
    assert data['baseline'] is None, (
        'the label-only lookup started serving a baseline track'
    )
    assert data['target_idx'] == 1


@requires_db
def test_unknown_parameterization_returns_null_labels(conn, client):
    """
    A legacy row: a delta, no provenance at all, no exported geometry. Nothing can say
    which model produced it, so the honest answer is null — not the vehicle set, which
    would be B13 narrowed rather than fixed.
    """
    _seed(conn)
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE scenario_scores
               SET min_perturbation = 0.5, delta = %s, collision_timestep = 3,
                   stress_method = 'de', stress_tested_at = now(),
                   stress_outcome = 'collision_found', search_provenance = NULL
             WHERE scenario_id = 'scene'
        """, ([0.5, 0.0, 0.0, 0.0],))
    conn.commit()

    data = client.get('/scenarios/scene/perturbed').json()
    assert data['delta'] == [0.5, 0.0, 0.0, 0.0], 'the delta is still served'
    assert data['delta_labels'] is None, (
        'an unlabellable delta was labelled anyway'
    )


# ── B12: geometry routes degrade, they do not 500 ───────────────────────────────

@requires_db
def test_perturbed_keeps_the_score_fields_when_geometry_is_absent(bare_conn, bare_client):
    """
    min_perturbation, delta and collision_timestep come from scenario_scores and do not
    depend on the geometry tables at all. Degrading must null the PICTURE, not the
    result.

    AMENDED BY BATCH 7 (audit R06), AND THE AMENDMENT IS THE FINDING. This test used
    to assert `target_idx is None` here, because there was nowhere else for it to come
    from: it lived only on perturbed_paths, so a scenario stress-tested but not yet
    geometry-exported could not say which agent it had been tested against — even
    though _stress_one knew. That is R06, and the old assertion was pinning the defect
    in place rather than testing a contract.

    scenario_scores.target_idx now answers it, so the expectation flips from "null" to
    "the challenger that was actually searched". Strictly stronger: null was satisfied
    by any implementation that lost the value, and 1 is satisfied only by one that
    keeps it. Everything else about the degraded shape is unchanged — the PICTURE is
    still null, which is what this test is chiefly about.
    """
    states, validity, types = _pedestrian_scene()
    _seed(bare_conn)
    _stress_and_store(bare_conn, states, validity, types)

    response = bare_client.get('/scenarios/scene/perturbed')
    assert response.status_code == 200, response.text
    data = response.json()

    assert data['min_perturbation'] is not None
    assert data['delta'] is not None and len(data['delta']) == 4
    assert data['collision_timestep'] is not None
    assert data['baseline'] is None and data['perturbed'] is None
    assert data['target_idx'] == 1, (
        'the searched challenger must be recoverable without geometry (audit R06)'
    )
    assert data['delta_labels'] == LINEAR_DELTA_LABELS, (
        'provenance must still label the delta with no geometry present'
    )


@requires_db
def test_trajectories_is_empty_not_an_error_when_geometry_is_absent(bare_conn,
                                                                   bare_client):
    _seed(bare_conn)
    response = bare_client.get('/scenarios/scene/trajectories')
    assert response.status_code == 200, response.text
    assert response.json()['agents'] == []


@requires_db
def test_only_one_geometry_table_present_still_degrades(conn, client):
    """
    The guards are independent, not one standing in for the other.

    init_geometry_schema creates both tables in one statement, so they are atomic at
    creation — but CREATE TABLE IF NOT EXISTS means a database that acquired one from
    an older schema keeps it, and either can be dropped alone. Checking perturbed_paths
    and assuming scenario_agents is a coupling assumption; this asserts it is not being
    made.
    """
    states, validity, types = _pedestrian_scene()
    _seed(conn)
    _stress_and_store(conn, states, validity, types)
    export_scenario_agents(conn, 'scene', states, validity, types, 0)
    export_perturbed_path(conn, 'scene', states, validity, 1)

    with conn.cursor() as cur:
        cur.execute('DROP TABLE scenario_agents CASCADE')
    conn.commit()

    response = client.get('/scenarios/scene/perturbed')
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['perturbed'] is not None, 'perturbed_paths still exists'
    assert data['baseline'] is None, 'the missing table degrades to no baseline'

    # The CONSEQUENCE of that degradation, pinned rather than left merely true.
    # agent_type, length_m and width_m are read off base_row, so with scenario_agents
    # gone the perturbed track is a path with no box: drawable as a line, not as a
    # vehicle. That is the honest shape, but it is a real contract detail for the
    # frontend and it should break loudly if anyone changes it by accident.
    assert data['perturbed']['agent_type'] is None
    assert data['perturbed']['length_m'] is None
    assert data['perturbed']['width_m'] is None
    assert data['perturbed']['path'], 'the path itself must still be served'

    assert client.get('/scenarios/scene/trajectories').status_code == 200


@requires_db
def test_the_guard_does_not_swallow_a_real_geometry_failure(conn, client):
    """
    The precheck must stay precise. A try/except around these queries would turn an
    inconsistent row into a cheerful empty response — the opposite of _make_track's
    no-fallback argument, and of the guard tests/test_api.py asserts at :333.

    This is the test that catches a broad except creeping back in.
    """
    states, validity, types = _pedestrian_scene()
    _seed(conn)
    export_scenario_agents(conn, 'scene', states, validity, types, 0)

    # headings no longer matches the number of path vertices
    with conn.cursor() as cur:
        cur.execute("UPDATE scenario_agents SET headings = %s "
                    "WHERE scenario_id='scene' AND agent_idx=0", ([0.0, 0.0],))
    conn.commit()

    response = client.get('/scenarios/scene/trajectories')
    assert response.status_code == 500, (
        f'inconsistent geometry was papered over as HTTP {response.status_code}'
    )


# ── B17: malformed input is a client error ──────────────────────────────────────

@requires_db
@pytest.mark.parametrize('bad', ['\x00', '\x01', '\x1f', '\x7f'])
def test_control_characters_in_the_path_are_rejected(conn, client, bad):
    # Percent-encoded, because httpx refuses to construct a URL containing a raw
    # control character — the client would reject it before the server ever saw it,
    # which is why the audit's own fixture writes /scenarios/%00. The server still
    # decodes it back to the real byte, so the handler sees exactly what a hostile
    # client would send.
    response = client.get(f'/scenarios/{quote(bad, safe="")}')
    assert 400 <= response.status_code < 500, (
        f'control character {bad!r} in path returned {response.status_code}'
    )


@requires_db
@pytest.mark.parametrize('bad', ['\x00', '\x01', '\x7f'])
def test_control_characters_in_the_cursor_payload_are_rejected(conn, client, bad):
    """
    The cursor DECODES fine here — the payload is valid base64 and valid JSON. It is the
    decoded scenario id that is unusable, which the pre-Batch-3 code only discovered at
    the psycopg2 layer, as a 500.
    """
    cursor = base64.urlsafe_b64encode(
        json.dumps({'f': 1.0, 's': bad}).encode()).decode()
    response = client.get('/scenarios', params={'cursor': cursor})
    assert response.status_code == 422, response.text


@requires_db
@pytest.mark.parametrize('suffix', ['', '/trajectories', '/perturbed'])
def test_every_scenario_route_validates_its_path_parameter(conn, client, suffix):
    """The dependency is applied to all three, so a new route cannot quietly skip it."""
    response = client.get(f'/scenarios/%00{suffix}')
    assert 400 <= response.status_code < 500, response.status_code


@requires_db
@pytest.mark.parametrize('scenario_id', [
    'a.b.c', 'a-b_c', 'ABC123', 'ünïcødé', 'scenario nested', 'a+b=c&d', 'x' * 200,
])
def test_the_validator_does_not_become_a_format_whitelist(conn, client, scenario_id):
    """
    Defensive, not restrictive. Nothing in this project specifies what a real WOMD
    scenario_id looks like beyond "a globally unique string", so anything that is not
    actively unsafe must still reach the 404 it deserves rather than a 422. A validator
    that fails closed on valid input is worse than the 500 it replaced.

    Percent-encoded, so each case actually REACHES the dependency. An earlier draft
    included a raw 'scenario/nested', which passed for the wrong reason: FastAPI's
    default path converter is `[^/]+`, so a literal slash matches no single-segment
    route at all and 404s at the ROUTER, before _validated_scenario_id ever runs. A
    test that cannot reach the code it names is not testing it, however green it is.
    """
    response = client.get(f'/scenarios/{quote(scenario_id, safe="")}')
    assert response.status_code == 404, (
        f'{scenario_id!r} was rejected as malformed, not merely unknown: '
        f'{response.status_code} {response.text[:120]}'
    )


@requires_db
@pytest.mark.parametrize('raw', ['scenario/nested', 'scenario%2Fnested'])
def test_a_slash_bearing_id_is_unreachable_rather_than_validated(conn, client, raw):
    """
    A scenario_id containing a slash never reaches these routes AT ALL — and, measured
    rather than assumed, percent-encoding does not change that.

    Starlette routes on scope['path'], which its TestClient builds as
    `unquote(request.url.path)`. httpx has already decoded %2F to a literal '/' by the
    time .path is read, so the unquote is a SECOND decode: both spellings arrive as
    three path segments and match no single-segment route. The 404 comes from the
    router, before _validated_scenario_id or any handler runs.

    This test exists because an earlier draft asserted the opposite in two different
    ways — first with a raw slash in the format-whitelist list (which passed for the
    wrong reason), then with an encoded one claimed to reach the dependency (which did
    not). Pinning the real behaviour is worth more than a third guess at it.

    The consequence, flagged and not fixed here: if a real WOMD scenario_id ever
    contained a slash it would be unaddressable through this API. Nothing in the corpus
    suggests they do, and inventing a path converter for a hypothetical is exactly the
    format assumption _reject_unstorable_text refuses to make.
    """
    response = client.get(f'/scenarios/{raw}')
    assert response.status_code == 404, response.status_code
    assert response.json()['detail'] == 'Not Found', (
        f'{raw!r} now reaches a handler; this test documented that it cannot: '
        f'{response.text[:160]}'
    )


@requires_db
def test_an_absurdly_long_scenario_id_is_refused(conn, client):
    """A resource guard, not a format rule."""
    response = client.get('/scenarios/' + 'x' * 9000)
    assert 400 <= response.status_code < 500, response.status_code


# ── B18: pool exhaustion is a 503 ───────────────────────────────────────────────

@requires_db
def test_pool_exhaustion_returns_503_not_500(monkeypatch):
    pool = ThreadedConnectionPool(1, 1, **api_main.settings.dsn_kwargs())
    borrowed = pool.getconn()
    monkeypatch.setattr(db_pool, '_pool', pool)
    monkeypatch.setattr(api_main, 'init_pool', lambda: None)
    monkeypatch.setattr(api_main, 'close_pool', lambda: None)
    try:
        client = TestClient(api_main.create_app(), raise_server_exceptions=False)
        started = time.monotonic()
        response = client.get('/health')
        waited_ms = (time.monotonic() - started) * 1000

        assert response.status_code == 503, response.status_code
        assert response.headers.get('Retry-After') == '1'
        # It waited, and it stopped waiting. Both halves matter: no wait makes a burst
        # a 503, and no bound makes a worker thread hang.
        assert waited_ms >= api_main.settings.pool_wait_ms * 0.5, waited_ms
        assert waited_ms < api_main.settings.pool_wait_ms * 8, waited_ms
    finally:
        pool.putconn(borrowed)
        pool.closeall()


@requires_db
def test_a_connection_returned_during_the_wait_is_picked_up(monkeypatch):
    """
    The bounded wait must actually serve the request it waited for, not merely delay the
    503 — otherwise it is a sleep with extra steps.
    """
    pool = ThreadedConnectionPool(1, 1, **api_main.settings.dsn_kwargs())
    borrowed = pool.getconn()
    monkeypatch.setattr(db_pool, '_pool', pool)
    monkeypatch.setattr(api_main, 'init_pool', lambda: None)
    monkeypatch.setattr(api_main, 'close_pool', lambda: None)

    released = threading.Timer(0.05, lambda: pool.putconn(borrowed))
    released.start()
    try:
        client = TestClient(api_main.create_app(), raise_server_exceptions=False)
        response = client.get('/health')
        assert response.status_code == 200, (
            f'a connection freed mid-wait was not picked up: {response.status_code}'
        )
    finally:
        released.cancel()
        pool.closeall()


@requires_db
def test_pooled_connections_carry_a_statement_timeout(conn):
    """
    One slow query must not be able to hold a pooled connection indefinitely.

    Pinned to the EXACT value, not merely "not zero". A malformed options string that
    left some other timeout in force would satisfy != '0' while the configured ceiling
    was never applied — the assertion would be reporting that libpq did something, not
    that it did the right thing.

    '5s' is how PostgreSQL renders 5000 ms back through SHOW; measured against a live
    connection, not inferred. The settings check below is what fails first, and legibly,
    if the default ever moves — otherwise a changed default would surface here as an
    opaque string mismatch.
    """
    assert api_main.settings.statement_timeout_ms == 5000, (
        'default changed; update the expected SHOW rendering below to match'
    )
    pool = ThreadedConnectionPool(1, 1, **api_main.settings.dsn_kwargs())
    try:
        borrowed = pool.getconn()
        with borrowed.cursor() as cur:
            cur.execute('SHOW statement_timeout')
            assert cur.fetchone()[0] == '5s', 'configured timeout not in force'
        pool.putconn(borrowed)
    finally:
        pool.closeall()
