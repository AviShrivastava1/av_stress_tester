"""
test_batch7_contract.py — Batch 7 contract tests (audit R01, R03, R06, R07).

Batch 2 built stress_run_id and the IS NOT DISTINCT FROM join so a result and its
geometry could never be proven to describe different runs. These four findings are
not gaps in that join's logic — the join is correct in every one of them. They are
two ways of defeating it from outside:

  R01  WRITE-TIME. The wrong content gets stamped with a right-looking id, so the
       read-side check passes on ids that genuinely match and are simply attached to
       different things.
  R03  READ-TIME. Correct writes, compared against an id nobody captured: the handler
       read scenario_scores twice and matched the second read against the second.
  R06  the challenger was recoverable only from geometry, so a stress-tested,
       unexported scenario could not say which agent it had been tested against.
  R07  a refusal's diagnostics never reached the database at all.

R01 and R03 are INDEPENDENT, and that is demonstrated rather than argued: the R03
fixture stamps both of its exports from content (R01 already fixed) and still
reproduces, while the R01 fixture is sequential on one connection and has no
concurrency for R03's fix to touch.

The audit's own repros live in tests/test_audit2_regressions.py, reconstructed and
labelled. THIS file holds the properties.

Requires a disposable database:
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest \
        tests/test_batch7_contract.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring import db
from src.scoring.export_geometry import (
    StaleExportError, export_perturbed_path, export_scenario_agents,
    init_geometry_schema,
)
from src.optimization.perturbation_space import PerturbationSpace

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)


@pytest.fixture
def conn():
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, '
                    'scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    init_geometry_schema(connection)
    yield connection
    connection.rollback()
    connection.close()


def _scene(n_agents=3):
    """SDC, a near challenger (1) and a far one (2)."""
    states = np.zeros((n_agents, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1:, :, 5:7] = [0.6, 0.6]
    states[1, :, 0] = 5.0 - 2.0 * np.arange(10) * 0.1
    states[1, :, 2] = -2.0
    states[1, :, 4] = np.pi
    if n_agents > 2:
        states[2, :, 0] = 30.0
        states[2, :, 1] = 30.0
    types = np.array([1] + [2] * (n_agents - 1))
    return states, np.ones((n_agents, 10), dtype=bool), types


def _seed(conn, sid='scene', n=3):
    db.upsert_scores(conn, [dict(scenario_id=sid, shard='synthetic', n_agents=n,
                                 min_ttc=9.0, min_pet=9.0, fragility_score=1.0)])


def _result(**over):
    base = {'status': 'ok', 'outcome': db.OUTCOME_COLLISION_FOUND, 'collision': True,
            'min_perturbation': 0.5, 'delta': [-1.0, 0.0, 0.0, 0.0],
            'collision_timestep': 3, 'target_idx': 1, 'method': 'de',
            'challengers_total': 2, 'challengers_searched': 1}
    base.update(over)
    return base


# ── R01: the export publishes only what matches ─────────────────────────────────

@requires_db
def test_a_current_export_still_publishes(conn):
    """The guard must not become a tax on the ordinary path."""
    states, validity, types = _scene()
    _seed(conn)
    export_scenario_agents(conn, 'scene', states, validity, types, 0)
    space = PerturbationSpace(states, validity, types, 0, 1)
    result = _result()
    db.update_stress_results(conn, {'scene': result})

    assert export_perturbed_path(
        conn, 'scene', space.apply(np.float32(result['delta'])), validity, 1,
        delta=result['delta'], method='de') is True

    row = db.fetch_scenario(conn, 'scene')
    with conn.cursor() as cur:
        cur.execute("SELECT stress_run_id FROM perturbed_paths WHERE scenario_id='scene'")
        assert cur.fetchone()[0] == row['stress_run_id']


@requires_db
def test_the_derived_id_is_the_one_update_stress_results_computed(conn):
    """
    The exporter and the writer must agree by CONSTRUCTION, not by coincidence: the
    same function over the same four inputs. If they ever diverged, every export
    would be refused as stale and the refusal would be the bug.
    """
    states, validity, types = _scene()
    _seed(conn)
    space = PerturbationSpace(states, validity, types, 0, 1)
    result = _result()
    db.update_stress_results(conn, {'scene': result})

    persisted = db.fetch_scenario(conn, 'scene')['stress_run_id']
    derived = db.compute_stress_run_id('scene', result['target_idx'],
                                       result['delta'], result['method'])
    assert derived == persisted
    export_perturbed_path(conn, 'scene', space.apply(np.float32(result['delta'])),
                          validity, 1, delta=result['delta'], method='de')


@requires_db
def test_a_refused_export_writes_nothing_at_all(conn):
    """
    Not "writes something wrong" — writes NOTHING. The check and the write are one
    statement, so there is no window in which a partial row exists.
    """
    states, validity, types = _scene()
    _seed(conn)
    space = PerturbationSpace(states, validity, types, 0, 1)
    a, b = _result(), _result(delta=[-2.0, 0.0, 0.0, 0.0], min_perturbation=1.0)
    db.update_stress_results(conn, {'scene': a})
    db.update_stress_results(conn, {'scene': b})

    with pytest.raises(StaleExportError) as excinfo:
        export_perturbed_path(conn, 'scene', space.apply(np.float32(a['delta'])),
                              validity, 1, delta=a['delta'], method='de')

    assert excinfo.value.attempted_run_id == db.compute_stress_run_id(
        'scene', 1, a['delta'], 'de')
    assert excinfo.value.current_run_id == db.fetch_scenario(conn, 'scene')['stress_run_id']
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM perturbed_paths WHERE scenario_id='scene'")
        assert cur.fetchone()[0] == 0


@requires_db
def test_stale_is_not_the_same_answer_as_too_short(conn):
    """
    The two reasons an export produces no row must stay distinguishable. Collapsing
    them into one falsy return is the defect class Batch 2 removed from
    scenario_scores, reappearing in a return value.
    """
    states, validity, types = _scene()
    _seed(conn)
    space = PerturbationSpace(states, validity, types, 0, 1)
    db.update_stress_results(conn, {'scene': _result()})

    short = np.ones((3, 10), dtype=bool)
    short[1, :] = False
    short[1, 0] = True                       # one valid frame: no linestring
    assert export_perturbed_path(conn, 'scene', space.apply(np.float32([-1., 0, 0, 0])),
                                 short, 1, delta=[-1.0, 0.0, 0.0, 0.0],
                                 method='de') is False

    db.update_stress_results(conn, {'scene': _result(delta=[-2.0, 0.0, 0.0, 0.0])})
    with pytest.raises(StaleExportError):
        export_perturbed_path(conn, 'scene', space.apply(np.float32([-1., 0, 0, 0])),
                              validity, 1, delta=[-1.0, 0.0, 0.0, 0.0], method='de')


@requires_db
def test_a_legacy_row_with_no_run_id_still_exports(conn):
    """
    IS NOT DISTINCT FROM on the write side too, for the same reason as the read side:
    `=` is NULL rather than true when either operand is NULL, which would refuse
    every export for a scenario that predates stress_run_id.
    """
    states, validity, types = _scene()
    _seed(conn)
    space = PerturbationSpace(states, validity, types, 0, 1)
    # a scored row that no stress pass has touched: stress_run_id IS NULL
    assert db.fetch_scenario(conn, 'scene')['stress_run_id'] is None
    assert export_perturbed_path(conn, 'scene', space.apply(np.zeros(4, np.float32)),
                                 validity, 1) is True


@requires_db
def test_the_batch_records_which_exports_were_refused_not_how_many(conn):
    """
    Block 5 Concept 19: a batch that reports "982 scored, 18 skipped, here is why" is
    trustworthy; one that reports "982 scored" is a silent data-quality bug. An
    operator looking at a stale-export spike needs to know WHICH scenarios and which
    run each one thought it was, which a tally cannot answer.

    export_shard_geometry needs a shard to walk, so the recording shape is asserted
    against the summary contract directly.
    """
    import inspect
    from src.scoring import export_geometry

    source = inspect.getsource(export_geometry.export_shard_geometry)
    assert "'perturbed_stale': []" in source, 'the summary must start it as a list'
    for field in ('scenario_id', 'attempted_run_id', 'current_run_id'):
        assert field in source.split('StaleExportError')[1][:600], (
            f'a refusal entry must carry {field}'
        )


# ── R03: one snapshot ───────────────────────────────────────────────────────────

@requires_db
def test_the_path_query_does_not_reread_scenario_scores(conn):
    """
    The structural assertion behind R03: if the handler joins scenario_scores again,
    the response is the product of two reads and a commit between them makes it
    incoherent. One read, one snapshot.
    """
    import inspect
    from src.api import routes

    source = inspect.getsource(routes.get_perturbed)
    path_query = source.split('FROM perturbed_paths pp')[1].split('"""')[0]
    assert 'scenario_scores' not in path_query, (
        'the perturbed-path query re-reads scenario_scores; it must match against the '
        "run id already captured in score_row (audit R03)"
    )
    assert 'IS NOT DISTINCT FROM' in path_query, (
        'the legacy NULL-NULL carve-out must survive the rebinding'
    )


@requires_db
def test_a_matching_pair_is_still_served(conn):
    """R03's fix must not blank the path for the ordinary consistent case."""
    from fastapi.testclient import TestClient
    from src.api.main import app

    states, validity, types = _scene()
    _seed(conn)
    export_scenario_agents(conn, 'scene', states, validity, types, 0)
    space = PerturbationSpace(states, validity, types, 0, 1)
    result = _result()
    db.update_stress_results(conn, {'scene': result})
    export_perturbed_path(conn, 'scene', space.apply(np.float32(result['delta'])),
                          validity, 1, delta=result['delta'], method='de')

    with TestClient(app) as client:
        data = client.get('/scenarios/scene/perturbed').json()
    assert data['perturbed'] is not None
    assert data['baseline'] is not None
    assert data['target_idx'] == 1


# ── R06 / R07: target_idx partitions across the two owners ──────────────────────

@requires_db
def test_a_refused_rerun_keeps_the_results_challenger_and_reports_its_own(conn):
    """
    THE SHARPEST EDGE THE R06/R07 PARTITION CREATES, and the reason both findings are
    in one batch.

    A stored successful result against challenger X, then a refused re-run that would
    have selected a DIFFERENT challenger Y. Three things must hold at once:

      * the result-group target_idx stays X — Batch 2's preservation rule, the same
        one that keeps delta and min_perturbation;
      * last_attempt_diagnostics records Y, because the refusal is what the LATEST
        pass concluded and Y is the agent it selected;
      * /perturbed serves X, because X is the agent the exported geometry actually
        describes. Serving Y there would label a trajectory with an agent it is not.

    This is Batch 2's test_a_pass_that_ran_no_search_preserves_the_previous_result,
    extended to the field this batch adds. If target_idx had gone into the attempt
    group instead, the first and third assertions would both fail.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app

    states, validity, types = _scene()
    _seed(conn)
    export_scenario_agents(conn, 'scene', states, validity, types, 0)
    space = PerturbationSpace(states, validity, types, 0, 1)

    success = _result(target_idx=1)
    db.update_stress_results(conn, {'scene': success})
    export_perturbed_path(conn, 'scene', space.apply(np.float32(success['delta'])),
                          validity, 1, delta=success['delta'], method='de')
    stored_run = db.fetch_scenario(conn, 'scene')['stress_run_id']

    # the refused re-run picks a DIFFERENT challenger
    db.update_stress_results(conn, {'scene': {
        'status': 'replay_infeasible', 'outcome': db.OUTCOME_REPLAY_INFEASIBLE,
        'target_idx': 2, 'baseline_replay_error': 3.9, 'reason': 'drift',
        'baseline_replay_collides': False,
        'challengers_total': 2, 'challengers_searched': 0}})

    row = db.fetch_scenario(conn, 'scene')
    assert row['target_idx'] == 1, (
        'the stored result was searched against agent 1; a refused re-run supersedes '
        'nothing and must not repoint it at agent 2'
    )
    assert row['delta'] == [-1.0, 0.0, 0.0, 0.0], 'the result group must be preserved'
    assert row['stress_run_id'] == stored_run
    assert row['last_attempt_outcome'] == db.OUTCOME_REPLAY_INFEASIBLE
    assert row['last_attempt_diagnostics']['target_idx'] == 2, (
        "the latest pass selected agent 2 and that must be recorded, separately"
    )
    assert row['last_attempt_diagnostics']['reason'] == 'drift'

    with TestClient(app) as client:
        data = client.get('/scenarios/scene/perturbed').json()
    assert data['target_idx'] == 1, (
        'the served target must be the agent the served geometry describes, not the '
        'one a later refused attempt happened to pick'
    )
    assert data['perturbed'] is not None, (
        'a refused re-run supersedes nothing, so the matching geometry still serves'
    )


@requires_db
def test_the_diagnostics_do_not_outlive_the_attempt_that_produced_them(conn):
    """
    Written unconditionally, NULL included. A diagnostic left behind by an earlier
    refusal sitting beside a later successful attempt is the same juxtaposition the
    two-owner split exists to prevent.
    """
    _seed(conn)
    db.update_stress_results(conn, {'scene': {
        'status': 'replay_infeasible', 'outcome': db.OUTCOME_REPLAY_INFEASIBLE,
        'target_idx': 1, 'baseline_replay_error': 3.9, 'reason': 'drift',
        'baseline_replay_collides': False}})
    assert db.fetch_scenario(conn, 'scene')['last_attempt_diagnostics'] is not None

    db.update_stress_results(conn, {'scene': _result()})
    diagnostics = db.fetch_scenario(conn, 'scene')['last_attempt_diagnostics']
    assert diagnostics is not None and 'reason' not in diagnostics, (
        "a later successful attempt still carries the earlier refusal's reason"
    )
    assert diagnostics['target_idx'] == 1


@requires_db
def test_a_pass_with_no_diagnostics_clears_the_previous_ones(conn):
    """
    THE CASE THAT ACTUALLY TESTS "written unconditionally", found by a mutation that
    the obvious test could not catch.

    test_the_diagnostics_do_not_outlive_the_attempt_that_produced_them follows the
    refusal with a SUCCESSFUL pass, and a successful pass still reports which
    challenger it selected — so the column is overwritten with {'target_idx': ...}
    and never sees NULL. Changing the write to
    `COALESCE(%s, last_attempt_diagnostics)` therefore left every test green while
    reintroducing exactly the defect: a stale diagnostic surviving its attempt.

    `no_challenger` is the pass that genuinely produces none — no target was selected,
    so there is nothing to report — and it is a realistic sequence: a scenario refused
    for drift, re-run after the scene changed and left nothing to perturb. The row
    must not still claim the latest attempt drifted 3.9 m.
    """
    _seed(conn)
    db.update_stress_results(conn, {'scene': {
        'status': 'replay_infeasible', 'outcome': db.OUTCOME_REPLAY_INFEASIBLE,
        'target_idx': 1, 'baseline_replay_error': 3.9, 'reason': 'drift',
        'baseline_replay_collides': False}})
    assert db.fetch_scenario(conn, 'scene')['last_attempt_diagnostics'] is not None

    db.update_stress_results(conn, {'scene': {
        'status': 'no_challenger', 'outcome': db.OUTCOME_NO_CHALLENGER,
        'challengers_total': 0, 'challengers_searched': 0}})

    row = db.fetch_scenario(conn, 'scene')
    assert row['last_attempt_outcome'] == db.OUTCOME_NO_CHALLENGER
    assert row['last_attempt_diagnostics'] is None, (
        "a pass that reported no diagnostics left the previous refusal's behind: "
        f"{row['last_attempt_diagnostics']}"
    )


@requires_db
@pytest.mark.parametrize('reason,collides', [('collision', True), ('drift', False)])
def test_the_two_refusal_reasons_stay_distinguishable(conn, reason, collides):
    """
    The whole point of R07: a collision-refusal and a drift-refusal must not read the
    same once the in-memory dict is gone. Both carry the same outcome word, so the
    outcome alone cannot tell them apart.
    """
    _seed(conn)
    db.update_stress_results(conn, {'scene': {
        'status': 'replay_infeasible', 'outcome': db.OUTCOME_REPLAY_INFEASIBLE,
        'target_idx': 1, 'baseline_replay_error': 0.0 if collides else 3.9,
        'reason': reason, 'baseline_replay_collides': collides}})

    row = db.fetch_scenario(conn, 'scene')
    assert row['last_attempt_outcome'] == db.OUTCOME_REPLAY_INFEASIBLE
    assert row['last_attempt_diagnostics']['reason'] == reason
    assert row['last_attempt_diagnostics']['baseline_replay_collides'] is collides


@requires_db
def test_the_drift_number_survives_for_the_distribution_study(conn):
    """
    baseline_replay_error was collected to measure the real distribution before
    defending the 0.5 m default — still-outstanding Colab work. A number that does not
    survive the process that produced it cannot support that argument.
    """
    _seed(conn)
    for i, err in enumerate([0.31, 1.27, 4.02]):
        db.upsert_scores(conn, [dict(scenario_id=f's{i}', shard='x', n_agents=2,
                                     min_ttc=9.0, min_pet=9.0, fragility_score=1.0)])
        db.update_stress_results(conn, {f's{i}': {
            'status': 'replay_infeasible', 'outcome': db.OUTCOME_REPLAY_INFEASIBLE,
            'target_idx': 1, 'baseline_replay_error': err, 'reason': 'drift',
            'baseline_replay_collides': False}})

    with conn.cursor() as cur:
        cur.execute("""
            SELECT (last_attempt_diagnostics ->> 'baseline_replay_error')::float
            FROM scenario_scores
            WHERE last_attempt_diagnostics ->> 'reason' = 'drift'
            ORDER BY 1
        """)
        assert [r[0] for r in cur.fetchall()] == [0.31, 1.27, 4.02]
