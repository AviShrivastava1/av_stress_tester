"""
test_claims_contract.py — Batch 2 regression tests (audit B04, B05, B14, B15).

The contract this file pins down:

    Every value the API serves is traceable to the run that produced it, and no
    field asserts more than the method behind it actually established.

Concretely: the five stress outcomes stay distinguishable all the way from
_stress_one to the API — `replay_infeasible` in particular must never read as
`no_collision_found`; `robustly_safe` is never True from any current code path; a
record that fails before yielding a scenario id cannot overwrite a result; and a
result and the geometry drawn beside it can be proven to come from the same run.

The DB-backed tests need a DISPOSABLE Postgres/PostGIS database — they drop and
recreate the project's three tables. They skip unless AV_CLAIMS_DB=1 is set, the
same guard the audit's own SQL suite uses.

Run:
    ./venv/bin/python -m pytest tests/test_claims_contract.py -q
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_claims_contract.py -q
"""

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.routes import _summary_fields
from src.scoring import db
from src.scoring.batch_scorer import StressResults, _stress_one
from src.scoring.db import (
    OUTCOME_COLLISION_FOUND, OUTCOME_ERROR, OUTCOME_NO_CHALLENGER,
    OUTCOME_NO_COLLISION_FOUND, OUTCOME_REPLAY_INFEASIBLE,
    compute_stress_run_id, resolve_outcome,
)

ALL_OUTCOMES = [
    OUTCOME_COLLISION_FOUND, OUTCOME_NO_COLLISION_FOUND,
    OUTCOME_REPLAY_INFEASIBLE, OUTCOME_NO_CHALLENGER, OUTCOME_ERROR,
]

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)


# ── B04: the outcome vocabulary ─────────────────────────────────────────────────

def _row(**overrides):
    """A scenario_scores row as _summary_fields receives it."""
    row = dict(scenario_id='s', shard='synthetic', n_agents=3,
               min_ttc=9.0, min_pet=9.0, fragility_score=1.0,
               min_perturbation=None, collision_timestep=None, stress_method=None,
               stress_tested_at=None, stress_attempted_at=None, stress_outcome=None,
               last_attempt_outcome=None, challengers_total=2, challengers_searched=0,
               search_certifies_infeasibility=False)
    row.update(overrides)
    return row


def test_replay_infeasible_is_not_no_collision_found():
    """
    The cross-batch guard, and the reason this batch exists.

    A scenario refused because its own zero-perturbation replay was untrustworthy
    (Batch 1, audit B03) had NO search run against it. Reporting it as "a search ran
    and found nothing" is audit B04's category error one level up — and it is the
    easy mistake, because both leave min_perturbation NULL.
    """
    # These are the shapes update_stress_results ACTUALLY writes. A refusal leaves no
    # stored result, so stress_outcome stays NULL and the refusal lives in
    # last_attempt_outcome; only a completed search sets stress_outcome. Building a row
    # with stress_outcome='replay_infeasible' would test a state the database can no
    # longer produce, and would keep passing through a regression.
    refused = _summary_fields(_row(stress_outcome=None,
                                   last_attempt_outcome=OUTCOME_REPLAY_INFEASIBLE,
                                   stress_attempted_at='t'))
    searched = _summary_fields(_row(stress_outcome=OUTCOME_NO_COLLISION_FOUND,
                                    last_attempt_outcome=OUTCOME_NO_COLLISION_FOUND,
                                    stress_attempted_at='t', stress_tested_at='t'))

    assert refused['last_attempt_outcome'] == OUTCOME_REPLAY_INFEASIBLE
    assert refused['no_collision_found'] is False
    assert searched['no_collision_found'] is True

    # ...and they are not smuggled back together by the booleans either.
    assert refused['stress_tested'] is False, 'no search ran, so nothing was tested'
    assert refused['stress_attempted'] is True, 'but the pass DID reach it'
    assert searched['stress_tested'] is True

    assert refused['min_perturbation'] == searched['min_perturbation'] is None, (
        'both leave min_perturbation NULL — which is exactly why the outcome columns '
        'have to carry the distinction instead'
    )


@pytest.mark.parametrize('outcome', ALL_OUTCOMES)
def test_every_attempt_outcome_is_distinguishable(outcome):
    """All five values are reachable in last_attempt_outcome, and survive the mapping."""
    fields = _summary_fields(_row(last_attempt_outcome=outcome,
                                  stress_attempted_at='t'))
    assert fields['last_attempt_outcome'] == outcome


@pytest.mark.parametrize('outcome', [OUTCOME_COLLISION_FOUND, OUTCOME_NO_COLLISION_FOUND])
def test_stored_result_outcomes_drive_no_collision_found(outcome):
    """
    Only these two are reachable in stress_outcome — it describes a STORED RESULT, and
    the other three produce none.
    """
    fields = _summary_fields(_row(stress_outcome=outcome, last_attempt_outcome=outcome,
                                  stress_attempted_at='t', stress_tested_at='t'))
    assert fields['stress_outcome'] == outcome
    # exactly one of the derived booleans may fire, and only for its own outcome
    assert fields['no_collision_found'] == (outcome == OUTCOME_NO_COLLISION_FOUND)


def test_unknown_outcome_is_not_back_inferred():
    """
    A row written before this column existed is genuinely unknown. It must not be
    reconstructed from which columns happen to be NULL — such a row may have been a
    completed search, or one that would now be refused outright.
    """
    legacy = _summary_fields(_row(stress_outcome=None, stress_tested_at='t',
                                  min_perturbation=None))
    assert legacy['stress_outcome'] is None
    assert legacy['no_collision_found'] is False
    assert legacy['robustly_safe'] is False


@pytest.mark.parametrize('outcome', ALL_OUTCOMES)
def test_robustly_safe_is_never_true_from_any_current_path(outcome):
    """
    No code path sets search_certifies_infeasibility, so no row can claim a safety
    certificate. A completed search that found nothing is included deliberately —
    that is the case the old API called robustly_safe.
    """
    reachable = outcome in (OUTCOME_COLLISION_FOUND, OUTCOME_NO_COLLISION_FOUND)
    fields = _summary_fields(_row(stress_outcome=outcome if reachable else None,
                                  last_attempt_outcome=outcome,
                                  stress_attempted_at='t', stress_tested_at='t'))
    assert fields['robustly_safe'] is False


def test_robustly_safe_would_follow_a_real_certificate():
    """The field is derived, not hardcoded: the data is what makes it False today."""
    fields = _summary_fields(_row(stress_outcome=OUTCOME_NO_COLLISION_FOUND,
                                  stress_tested_at='t',
                                  search_certifies_infeasibility=True))
    assert fields['robustly_safe'] is True


def test_summary_fields_tolerates_a_row_without_the_new_columns():
    """
    The audit's B04 fixture builds a bare dict. Indexing a missing column with [...]
    would raise KeyError — a FAILING test that looks like a passing fix.
    """
    bare = dict(scenario_id='s', shard=None, n_agents=3, min_ttc=1.0, min_pet=1.0,
                fragility_score=0.0, min_perturbation=None, collision_timestep=None,
                stress_method='de', stress_tested_at='completed')
    fields = _summary_fields(bare)
    assert fields['robustly_safe'] is False
    assert fields['stress_outcome'] is None


def _three_agent_scene():
    s = np.zeros((3, 2, 7), dtype=np.float32)
    s[:, :, 5:7] = [4.5, 2.0]
    s[1, :, 1] = 2.6
    s[2, :, 0] = 4.6
    s[2, :, 4] = np.pi
    return s, np.ones((3, 2), dtype=bool), np.ones(3, dtype=int)


def test_stress_one_records_what_the_search_actually_covered():
    """A result must carry its own coverage, not leave a reader to assume it."""
    s, v, types = _three_agent_scene()
    result = _stress_one(s, v, types, 0,
                         de_kwargs={'popsize': 4, 'maxiter': 10, 'seed': 1})

    assert result['challengers_total'] == 2, 'two non-SDC agents were available'
    assert result['challengers_searched'] == 1, 'exactly one was searched'
    provenance = result['search_provenance']
    assert provenance['challenger_selection'] == 'nearest_by_min_center_distance'
    assert provenance['de_maxiter'] == 10 and provenance['de_seed'] == 1
    assert 'baseline_replay_error' in provenance
    assert len(provenance['bounds']) == 4


def test_resolve_outcome_maps_every_status():
    assert resolve_outcome({'status': 'ok', 'collision': True}) == OUTCOME_COLLISION_FOUND
    assert resolve_outcome({'status': 'ok', 'collision': False}) == OUTCOME_NO_COLLISION_FOUND
    assert resolve_outcome({'status': 'replay_infeasible'}) == OUTCOME_REPLAY_INFEASIBLE
    assert resolve_outcome({'status': 'no_challenger'}) == OUTCOME_NO_CHALLENGER
    assert resolve_outcome({'status': 'error'}) == OUTCOME_ERROR
    # an explicit outcome always wins over the derivation
    assert resolve_outcome({'status': 'ok', 'collision': False,
                            'outcome': OUTCOME_REPLAY_INFEASIBLE}) == OUTCOME_REPLAY_INFEASIBLE


# ── B05: a corrupt record must not overwrite a good result ──────────────────────

def _stub_shard(monkeypatch, records, parse):
    """
    Drive stress_test_scenarios without the Waymo package.

    src/data/parser.py imports scenario_pb2 at module scope, and the Waymo wheels are
    manylinux-only, so the audit's own B05 fixture cannot run on macOS. Substituting
    the module in sys.modules exercises the identical bookkeeping path — the failure
    under test is in stress_test_scenarios' error handling, not in protobuf decoding.
    """
    monkeypatch.setattr('src.data.loader.ShardLoader', lambda _: iter(records),
                        raising=False)
    stub = types.ModuleType('src.data.parser')
    stub.ScenarioParser = parse
    monkeypatch.setitem(sys.modules, 'src.data.parser', stub)


def test_corrupt_record_does_not_overwrite_a_previous_success(monkeypatch):
    """
    Process A successfully, then hit a record that fails BEFORE yielding an id.

    The old code left `sid` holding 'A' from the previous iteration, so the error was
    written to results['A'], destroying a result that had already succeeded.
    """
    from src.scoring import batch_scorer

    class Parser:
        def __init__(self, raw):
            if raw == b'corrupt':
                raise ValueError('undecodable record')
            self.raw = raw

        def get_scenario_id(self):
            return self.raw.decode()

        def get_agent_states(self):
            return np.zeros((2, 2, 7), dtype=np.float32)

        def get_agent_validity(self):
            return np.ones((2, 2), dtype=bool)

        def get_agent_types(self):
            return np.ones(2, dtype=int)

        def get_sdc_index(self):
            return 0

    _stub_shard(monkeypatch, [b'A', b'corrupt', b'B'], Parser)
    monkeypatch.setattr(batch_scorer, '_stress_one', lambda *a, **k: {'status': 'ok'})

    results = batch_scorer.stress_test_scenarios('synthetic', ['A', 'B'], verbose=False)

    assert results['A']['status'] == 'ok', f'A was overwritten: {dict(results)}'
    assert results['B']['status'] == 'ok'


def test_unidentified_failure_lands_in_errors_with_its_record_index(monkeypatch):
    """It goes in the separate list, against the index — not into results."""
    from src.scoring import batch_scorer

    class Parser:
        def __init__(self, raw):
            if raw == b'corrupt':
                raise ValueError('undecodable record')
            self.raw = raw

        def get_scenario_id(self):
            return self.raw.decode()

        def get_agent_states(self):
            return np.zeros((2, 2, 7), dtype=np.float32)

        def get_agent_validity(self):
            return np.ones((2, 2), dtype=bool)

        def get_agent_types(self):
            return np.ones(2, dtype=int)

        def get_sdc_index(self):
            return 0

    _stub_shard(monkeypatch, [b'A', b'corrupt', b'B'], Parser)
    monkeypatch.setattr(batch_scorer, '_stress_one', lambda *a, **k: {'status': 'ok'})

    results = batch_scorer.stress_test_scenarios('synthetic', ['A', 'B'], verbose=False)

    assert len(results.errors) == 1
    assert results.errors[0]['record_index'] == 1
    assert results.errors[0]['scenario_id'] is None
    assert 'undecodable' in results.errors[0]['error']
    assert set(results) == {'A', 'B'}, 'errors must not become scenario keys'
    assert 'None' not in results, 'the old code keyed an error under str(None)'


def test_stress_results_is_still_a_dict():
    """Every existing caller indexes the return directly; that must keep working."""
    results = StressResults({'A': {'status': 'ok'}})
    assert results['A']['status'] == 'ok'
    assert isinstance(results, dict) and results.errors == []


# ── B14: a result and its geometry must describe the same run ───────────────────

def test_stress_run_id_is_deterministic_and_delta_sensitive():
    a = compute_stress_run_id('scene', 1, [-1.0, 0.0, 0.0, 0.0], 'de')
    b = compute_stress_run_id('scene', 1, [-1.0, 0.0, 0.0, 0.0], 'de')
    assert a == b, 'same run must hash identically, or re-export can never be a no-op'

    assert a != compute_stress_run_id('scene', 1, [-2.0, 0.0, 0.0, 0.0], 'de')
    assert a != compute_stress_run_id('scene', 2, [-1.0, 0.0, 0.0, 0.0], 'de')
    assert a != compute_stress_run_id('other', 1, [-1.0, 0.0, 0.0, 0.0], 'de')
    assert a != compute_stress_run_id('scene', 1, [-1.0, 0.0, 0.0, 0.0], 'de+autograd')

    # a float32 delta out of the optimizer must hash like the float64 it round-trips to
    assert a == compute_stress_run_id(
        'scene', 1, np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32), 'de')


def test_stress_run_id_does_not_use_the_salted_builtin_hash():
    """
    hash() is salted per process by PYTHONHASHSEED, so an id built from it would
    differ between runs and every re-export would look like a mismatch.
    """
    import subprocess
    code = ("import sys; sys.path.insert(0, '.');"
            "from src.scoring.db import compute_stress_run_id;"
            "print(compute_stress_run_id('scene', 1, [-1.0, 0.0, 0.0, 0.0], 'de'))")
    seen = {
        subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                       env={**os.environ, 'PYTHONHASHSEED': seed},
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                       ).stdout.strip()
        for seed in ('0', '1', '12345')
    }
    assert len(seen) == 1, f'id varies with PYTHONHASHSEED: {seen}'


# ── DB-backed ───────────────────────────────────────────────────────────────────

@pytest.fixture
def conn():
    from src.scoring.export_geometry import init_geometry_schema
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


def _seed(conn, sid='scene', n=3):
    db.upsert_scores(conn, [dict(scenario_id=sid, shard='synthetic', n_agents=n,
                                 min_ttc=9.0, min_pet=9.0, fragility_score=1.0)])


@requires_db
@pytest.mark.parametrize('outcome', ALL_OUTCOMES)
def test_every_outcome_round_trips_through_the_database(conn, outcome):
    _seed(conn)
    result = {'status': 'ok' if 'found' in outcome else outcome,
              'outcome': outcome, 'collision': outcome == OUTCOME_COLLISION_FOUND,
              'min_perturbation': 0.5 if outcome == OUTCOME_COLLISION_FOUND else float('inf'),
              'delta': [0.5, 0.0, 0.0, 0.0], 'collision_timestep': 7,
              'target_idx': 1, 'method': 'de',
              'challengers_total': 2, 'challengers_searched': 1}
    assert db.update_stress_results(conn, {'scene': result}) == 1

    row = db.fetch_scenario(conn, 'scene')
    assert row['last_attempt_outcome'] == outcome, (
        'every outcome must be visible as what the latest pass concluded'
    )
    assert row['stress_attempted_at'] is not None, 'every outcome records an attempt'

    search_ran = outcome in (OUTCOME_COLLISION_FOUND, OUTCOME_NO_COLLISION_FOUND)
    # stress_outcome describes the STORED RESULT. On a fresh row a pass that ran no
    # search leaves none, so it stays NULL rather than borrowing the attempt's outcome.
    assert row['stress_outcome'] == (outcome if search_ran else None)
    assert (row['stress_tested_at'] is not None) == search_ran
    # ...and the coverage columns follow the result, not the attempt: the inputs above
    # supply 2/1 for every outcome, but only a pass that produced a result may write
    # them, so on a fresh row a non-search outcome leaves them NULL.
    assert row['challengers_total'] == (2 if search_ran else None)
    assert row['challengers_searched'] == (1 if search_ran else None)
    assert _summary_fields(row)['robustly_safe'] is False


@requires_db
def test_replay_infeasible_persists_instead_of_vanishing(conn):
    """
    It used to be dropped entirely — update_stress_results skipped any non-'ok'
    status — so a refused scenario read back as never attempted.
    """
    _seed(conn)
    db.update_stress_results(conn, {'scene': {
        'status': 'replay_infeasible', 'outcome': OUTCOME_REPLAY_INFEASIBLE,
        'target_idx': 1, 'baseline_replay_error': 3.9, 'reason': 'drift',
        'challengers_total': 2, 'challengers_searched': 0}})

    fields = _summary_fields(db.fetch_scenario(conn, 'scene'))
    assert fields['last_attempt_outcome'] == OUTCOME_REPLAY_INFEASIBLE
    assert fields['stress_attempted'] is True
    assert fields['stress_tested'] is False
    assert fields['stress_outcome'] is None, 'no search ran, so there is no result'
    assert fields['no_collision_found'] is False


@requires_db
def test_committing_a_new_result_invalidates_the_exported_geometry(conn):
    """
    The B14 failure mode without the API in the way: a new delta must not leave the
    old run's path behind, even if the re-export never happens.
    """
    from src.optimization.perturbation_space import PerturbationSpace
    from src.scoring.export_geometry import (
        export_perturbed_path, export_scenario_agents,
    )

    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 5:7] = [0.6, 0.6]
    states[1, :, 0] = 5.0 - 2.0 * np.arange(10) * 0.1
    states[1, :, 2] = -2.0
    states[1, :, 4] = np.pi
    validity = np.ones((2, 10), dtype=bool)
    types = np.array([1, 2])

    _seed(conn, n=2)
    export_scenario_agents(conn, 'scene', states, validity, types, 0)
    space = PerturbationSpace(states, validity, types, 0, 1)

    first = {'status': 'ok', 'outcome': OUTCOME_COLLISION_FOUND, 'collision': True,
             'min_perturbation': 0.5, 'delta': [-1.0, 0.0, 0.0, 0.0],
             'collision_timestep': 3, 'target_idx': 1, 'method': 'de'}
    db.update_stress_results(conn, {'scene': first})
    export_perturbed_path(conn, 'scene', space.apply(np.float32([-1, 0, 0, 0])),
                          validity, 1)

    with conn.cursor() as cur:
        cur.execute('SELECT stress_run_id FROM perturbed_paths WHERE scenario_id=%s',
                    ('scene',))
        exported_id = cur.fetchone()[0]
    assert exported_id is not None, 'the export must record which run it came from'
    assert exported_id == db.fetch_scenario(conn, 'scene')['stress_run_id']

    # a NEW delta, with no re-export
    second = dict(first, delta=[-2.0, 0.0, 0.0, 0.0], min_perturbation=1.0)
    db.update_stress_results(conn, {'scene': second})

    with conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM perturbed_paths WHERE scenario_id=%s',
                    ('scene',))
        assert cur.fetchone()[0] == 0, (
            'stale geometry survived a new result — it would be served as evidence '
            'for a delta it was not replayed from'
        )
    assert db.fetch_scenario(conn, 'scene')['stress_run_id'] != exported_id


# ── B15: re-export replaces the agent set ───────────────────────────────────────

@requires_db
def test_reexport_removes_an_agent_that_is_no_longer_exportable(conn):
    from src.scoring.export_geometry import export_scenario_agents

    states = np.zeros((3, 3, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[:, :, 0] = np.arange(3)
    validity = np.ones((3, 3), dtype=bool)
    types = np.ones(3, dtype=int)

    _seed(conn)
    assert export_scenario_agents(conn, 'scene', states, validity, types, 0) == (3, 0)

    validity[2, 1:] = False      # agent 2 now has a single valid frame
    assert export_scenario_agents(conn, 'scene', states, validity, types, 0) == (2, 1)

    with conn.cursor() as cur:
        cur.execute('SELECT agent_idx FROM scenario_agents WHERE scenario_id=%s '
                    'ORDER BY agent_idx', ('scene',))
        assert [r[0] for r in cur.fetchall()] == [0, 1]


@requires_db
def test_reexport_removes_a_perturbed_path_for_a_removed_agent(conn):
    """
    The same defect one table over: /perturbed would look up a baseline agent that no
    longer exists and quietly serve a perturbed path with baseline=None.
    """
    from src.scoring.export_geometry import (
        export_perturbed_path, export_scenario_agents,
    )

    states = np.zeros((3, 3, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[:, :, 0] = np.arange(3)
    validity = np.ones((3, 3), dtype=bool)
    types = np.ones(3, dtype=int)

    _seed(conn)
    export_scenario_agents(conn, 'scene', states, validity, types, 0)
    assert export_perturbed_path(conn, 'scene', states, validity, 2) is True

    validity[2, 1:] = False
    export_scenario_agents(conn, 'scene', states, validity, types, 0)

    with conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM perturbed_paths WHERE scenario_id=%s',
                    ('scene',))
        assert cur.fetchone()[0] == 0, (
            'a perturbed path still references agent 2, which no longer exists'
        )


# ── B14: the run-id join, both branches of its NULL carve-out ───────────────────
#
# get_perturbed matches on `IS NOT DISTINCT FROM`, not `=`, so NULL-NULL counts as a
# match. That is a deliberate compatibility carve-out for rows exported before the
# column existed, and it is the one case the check cannot detect a mismatch in — so
# both branches get an explicit test rather than being inferred from the suite
# passing. See the truth table at the SQL site in routes.py.


@pytest.fixture
def client(conn, monkeypatch):
    from src.api import db_pool, main as api_main
    from fastapi.testclient import TestClient
    monkeypatch.setattr(api_main, 'init_pool', lambda: None)
    monkeypatch.setattr(api_main, 'close_pool', lambda: None)
    app = api_main.create_app()
    app.dependency_overrides[db_pool.get_db] = lambda: conn
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _export_a_scenario_with_a_path(conn):
    """Seed a scenario, export its agents, and export one perturbed path."""
    from src.scoring.export_geometry import (
        export_perturbed_path, export_scenario_agents,
    )
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 5:7] = [0.6, 0.6]
    states[1, :, 0] = 5.0 - 2.0 * np.arange(10) * 0.1
    states[1, :, 2] = -2.0
    states[1, :, 4] = np.pi
    validity = np.ones((2, 10), dtype=bool)
    types = np.array([1, 2])

    _seed(conn, n=2)
    export_scenario_agents(conn, 'scene', states, validity, types, 0)
    export_perturbed_path(conn, 'scene', states, validity, 1)
    return states, validity


@requires_db
def test_legacy_rows_without_a_run_id_still_serve(conn, client):
    """
    A scenario exported before stress_run_id existed has NULL on both sides.

    That pair must keep serving exactly what it always served. Using `=` instead of
    IS NOT DISTINCT FROM would evaluate to NULL here and blank the path for every
    pre-Batch-2 row — a silent regression wearing a safety check's clothing.
    """
    _export_a_scenario_with_a_path(conn)

    # Write the score columns directly, WITHOUT a run id, exactly as a row persisted
    # before this column existed would look.
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE scenario_scores
               SET min_perturbation = 0.5, delta = %s, collision_timestep = 3,
                   stress_method = 'de', stress_tested_at = now(),
                   stress_run_id = NULL
             WHERE scenario_id = 'scene'
        """, ([-1.0, 0.0, 0.0, 0.0],))
        cur.execute("UPDATE perturbed_paths SET stress_run_id = NULL "
                    "WHERE scenario_id = 'scene'")
        cur.execute("""SELECT ss.stress_run_id, pp.stress_run_id
                         FROM scenario_scores ss
                         JOIN perturbed_paths pp USING (scenario_id)
                        WHERE ss.scenario_id = 'scene'""")
        assert cur.fetchone() == (None, None), 'fixture must be a genuine legacy pair'
    conn.commit()

    response = client.get('/scenarios/scene/perturbed')
    assert response.status_code == 200, response.text
    assert response.json()['perturbed'] is not None, (
        'a legacy NULL/NULL pair stopped being served — the join is too strict'
    )


@requires_db
def test_a_mismatched_run_id_hides_the_perturbed_path(conn, client):
    """
    The branch that does the actual work: a path stamped with a DIFFERENT run than
    the score row must not be served.

    Mutated directly in SQL rather than via update_stress_results, because that
    function deletes the stale row outright — this test is about the second line of
    defence, for a row written out of band or by a partially-completed export.
    """
    _export_a_scenario_with_a_path(conn)

    with conn.cursor() as cur:
        cur.execute("""
            UPDATE scenario_scores
               SET min_perturbation = 0.5, delta = %s, collision_timestep = 3,
                   stress_method = 'de', stress_tested_at = now(),
                   stress_run_id = 'run_bbbb'
             WHERE scenario_id = 'scene'
        """, ([-2.0, 0.0, 0.0, 0.0],))
        cur.execute("UPDATE perturbed_paths SET stress_run_id = 'run_aaaa' "
                    "WHERE scenario_id = 'scene'")
    conn.commit()

    response = client.get('/scenarios/scene/perturbed')
    assert response.status_code == 200, response.text
    data = response.json()

    assert data['perturbed'] is None, (
        'geometry from run_aaaa was served beside a result from run_bbbb'
    )
    # Absent, not an error, and the RESULT itself is still reported — the delta is
    # known, only the picture of it is missing.
    assert data['delta'] == [-2.0, 0.0, 0.0, 0.0]


@requires_db
def test_a_matching_run_id_is_served(conn, client):
    """The control: identical ids on both sides must still return the path."""
    _export_a_scenario_with_a_path(conn)

    with conn.cursor() as cur:
        cur.execute("""
            UPDATE scenario_scores
               SET min_perturbation = 0.5, delta = %s, collision_timestep = 3,
                   stress_method = 'de', stress_tested_at = now(),
                   stress_run_id = 'run_aaaa'
             WHERE scenario_id = 'scene'
        """, ([-1.0, 0.0, 0.0, 0.0],))
        cur.execute("UPDATE perturbed_paths SET stress_run_id = 'run_aaaa' "
                    "WHERE scenario_id = 'scene'")
    conn.commit()

    response = client.get('/scenarios/scene/perturbed')
    assert response.status_code == 200, response.text
    assert response.json()['perturbed'] is not None


@requires_db
def test_a_stale_export_against_a_new_result_is_hidden(conn, client):
    """
    The ('x', NULL) row of the truth table: the score has been through the new write
    path, the export has not. That is a stale export and must be hidden.
    """
    _export_a_scenario_with_a_path(conn)

    with conn.cursor() as cur:
        cur.execute("""
            UPDATE scenario_scores
               SET min_perturbation = 0.5, delta = %s, stress_tested_at = now(),
                   stress_run_id = 'run_bbbb'
             WHERE scenario_id = 'scene'
        """, ([-2.0, 0.0, 0.0, 0.0],))
        cur.execute("UPDATE perturbed_paths SET stress_run_id = NULL "
                    "WHERE scenario_id = 'scene'")
    conn.commit()

    response = client.get('/scenarios/scene/perturbed')
    assert response.status_code == 200, response.text
    assert response.json()['perturbed'] is None


@requires_db
def test_a_refused_rerun_keeps_the_result_and_reports_the_refusal_separately(conn):
    """
    A scenario tested successfully, then re-run and refused, must do BOTH: keep the
    verified result it already produced, and report that the latest pass refused.

    This test has been rewritten twice, and both rewrites were corrections to it
    encoding a bug as a requirement:

      1. It first asserted min_perturbation was None here — i.e. that a refusal should
         destroy a result costing thousands of DE evaluations.
      2. It then asserted stress_outcome == 'replay_infeasible' AND
         min_perturbation == 0.5 TOGETHER, which is the incoherent row: the outcome
         field and the score fields describing two different runs.

    The row is now coherent by construction. stress_outcome travels with the result;
    last_attempt_outcome carries the refusal.
    """
    _seed(conn, n=2)
    db.update_stress_results(conn, {'scene': {
        'status': 'ok', 'outcome': OUTCOME_COLLISION_FOUND, 'collision': True,
        'min_perturbation': 0.5, 'delta': [0.5, 0.0, 0.0, 0.0],
        'collision_timestep': 7, 'target_idx': 1, 'method': 'de'}})

    db.update_stress_results(conn, {'scene': {
        'status': 'replay_infeasible', 'outcome': OUTCOME_REPLAY_INFEASIBLE,
        'target_idx': 1, 'baseline_replay_error': 3.9, 'reason': 'drift'}})

    fields = _summary_fields(db.fetch_scenario(conn, 'scene'))

    # the stored result, intact and self-consistent
    assert fields['min_perturbation'] == 0.5
    assert fields['stress_outcome'] == OUTCOME_COLLISION_FOUND, (
        'stress_outcome must describe the run that produced min_perturbation'
    )
    assert fields['stress_tested'] is True, 'a verified result still exists'

    # the refusal, reported without overwriting anything
    assert fields['last_attempt_outcome'] == OUTCOME_REPLAY_INFEASIBLE
    assert fields['stress_attempted'] is True


@requires_db
def test_outcome_and_score_always_describe_the_same_run(conn):
    """
    The invariant, stated directly: whenever a score is present, stress_outcome is one
    of the two outcomes that can produce one. It can never read 'replay_infeasible',
    'no_challenger' or 'error' beside a real min_perturbation.
    """
    _seed(conn, n=2)
    db.update_stress_results(conn, {'scene': {
        'status': 'ok', 'outcome': OUTCOME_COLLISION_FOUND, 'collision': True,
        'min_perturbation': 0.11, 'delta': [0.11, 0.0, 0.0, 0.0],
        'collision_timestep': 24, 'target_idx': 1, 'method': 'de'}})

    for outcome in (OUTCOME_REPLAY_INFEASIBLE, OUTCOME_ERROR, OUTCOME_NO_CHALLENGER):
        db.update_stress_results(conn, {'scene': {'status': outcome,
                                                  'outcome': outcome, 'target_idx': 1}})
        row = db.fetch_scenario(conn, 'scene')
        assert row['last_attempt_outcome'] == outcome
        if row['min_perturbation'] is not None:
            assert row['stress_outcome'] in (OUTCOME_COLLISION_FOUND,
                                             OUTCOME_NO_COLLISION_FOUND), (
                f"stress_outcome={row['stress_outcome']!r} sits beside "
                f"min_perturbation={row['min_perturbation']} — two different runs"
            )


@requires_db
@pytest.mark.parametrize('outcome', [OUTCOME_REPLAY_INFEASIBLE, OUTCOME_NO_CHALLENGER,
                                     OUTCOME_ERROR])
def test_a_pass_that_ran_no_search_preserves_the_previous_result(conn, outcome):
    """
    Block 5 section 22's column-ownership rule, applied to Pass 2 overwriting itself.

    A verified result costs thousands of DE evaluations. A later pass that could not
    run a search has not superseded it and must not destroy it — otherwise a stricter
    max_baseline_drift, or one transient failure, becomes silent and expensive data
    loss, with nothing left in the row to distinguish "re-verified as no longer true"
    from "we failed to check this time".
    """
    _seed(conn, n=2)
    db.update_stress_results(conn, {'scene': {
        'status': 'ok', 'outcome': OUTCOME_COLLISION_FOUND, 'collision': True,
        'min_perturbation': 0.11, 'delta': [0.11, 0.0, 0.5, 0.0],
        'collision_timestep': 24, 'target_idx': 1, 'method': 'de',
        'search_provenance': {'de_seed': 1},
        'challengers_total': 7, 'challengers_searched': 1}})
    before = db.fetch_scenario(conn, 'scene')
    assert before['challengers_total'] == 7 and before['challengers_searched'] == 1

    db.update_stress_results(conn, {'scene': {'status': outcome, 'outcome': outcome,
                                              'target_idx': 1}})
    after = db.fetch_scenario(conn, 'scene')

    # Every column in the result group. challengers_total/searched are in this list
    # because they are part of the same concept as search_provenance — how the stored
    # result was searched — and _stress_one returns neither for an 'error' outcome, so
    # under the previous grouping they were nulled out beside a preserved
    # min_perturbation.
    for column in ('min_perturbation', 'delta', 'collision_timestep', 'stress_method',
                   'stress_run_id', 'search_provenance', 'stress_outcome',
                   'stress_tested_at', 'challengers_total', 'challengers_searched'):
        assert after[column] == before[column], f'{column} was destroyed by {outcome}'

    # ...and stress_outcome/stress_tested_at travel WITH it, so the row never
    # describes two runs at once. The refusal surfaces in last_attempt_outcome.
    assert after['stress_outcome'] == before['stress_outcome']
    assert after['stress_tested_at'] == before['stress_tested_at']
    assert after['last_attempt_outcome'] == outcome
    assert _summary_fields(after)['stress_tested'] is True


@requires_db
def test_a_pass_that_did_run_a_search_still_supersedes(conn):
    """The other half: preservation must not become a refusal to ever update."""
    _seed(conn, n=2)
    db.update_stress_results(conn, {'scene': {
        'status': 'ok', 'outcome': OUTCOME_COLLISION_FOUND, 'collision': True,
        'min_perturbation': 0.11, 'delta': [0.11, 0.0, 0.5, 0.0],
        'collision_timestep': 24, 'target_idx': 1, 'method': 'de'}})

    db.update_stress_results(conn, {'scene': {
        'status': 'ok', 'outcome': OUTCOME_NO_COLLISION_FOUND, 'collision': False,
        'min_perturbation': float('inf'), 'delta': None, 'collision_timestep': -1,
        'target_idx': 1, 'method': 'de'}})

    row = db.fetch_scenario(conn, 'scene')
    assert row['stress_outcome'] == OUTCOME_NO_COLLISION_FOUND
    assert row['min_perturbation'] is None, 'a real search must overwrite'
    assert row['delta'] is None
    assert row['stress_tested_at'] is not None


@requires_db
def test_a_pass_that_ran_no_search_keeps_the_matching_geometry(conn):
    """
    A refused re-run supersedes nothing, so it must not strand the preserved result
    without its exported path. They still share a stress_run_id.
    """
    _export_a_scenario_with_a_path(conn)
    db.update_stress_results(conn, {'scene': {
        'status': 'ok', 'outcome': OUTCOME_COLLISION_FOUND, 'collision': True,
        'min_perturbation': 0.5, 'delta': [-1.0, 0.0, 0.0, 0.0],
        'collision_timestep': 3, 'target_idx': 1, 'method': 'de'}})
    from src.scoring.export_geometry import export_perturbed_path
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[1, :, 0] = 5.0 - 2.0 * np.arange(10) * 0.1
    export_perturbed_path(conn, 'scene', states, np.ones((2, 10), dtype=bool), 1)

    db.update_stress_results(conn, {'scene': {'status': OUTCOME_REPLAY_INFEASIBLE,
                                              'outcome': OUTCOME_REPLAY_INFEASIBLE,
                                              'target_idx': 1}})

    with conn.cursor() as cur:
        cur.execute("""SELECT pp.stress_run_id = ss.stress_run_id
                         FROM perturbed_paths pp
                         JOIN scenario_scores ss USING (scenario_id)
                        WHERE pp.scenario_id = 'scene'""")
        row = cur.fetchone()
    assert row is not None, 'the preserved result lost its geometry'
    assert row[0] is True, 'preserved result and its path no longer share a run id'
