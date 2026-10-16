"""
The two terms of the stored fragility score (src/danger/fragility_terms.py, served as ScenarioDetail.fragility_breakdown).

The score ranks the recorded scene only: src/danger/danger_score.py compute_danger_score called with no perturbation, so its
weights are renormalised, 0.4/0.7 for the TTC term and 0.3/0.7 for the PET term, each signal is 1 / max(value, 0.01), and the
999.0 sentinel means "none" and contributes 0. The API cannot import that module (it needs numpy, which the deployed API does not
install), so fragility_terms.py restates the arithmetic; this file is what keeps the two the same:

  * every constant is the scoring code's, and the sentinel is both engines';
  * the recomputed score equals compute_danger_score exactly over a grid of values that includes the sentinel, the floor and
    its neighbours, zero and negatives;
  * the weights, the terms, the "none" and "at the floor" flags, and the tolerance on "reproduces the stored score";
  * the 100 rows the public API serves (results only, tests/data/public_scores_2026-10-15.json): every one reproduces;
  * the module and the API import without numpy;
  * the route: a stored row gives its breakdown, a row without TTC or PET gives none, a row whose stored score is not the
    formula's says so, and the ranked list carries no breakdown.

The database tests need a DISPOSABLE database and skip unless AV_ROBUSTNESS_DB=1 (they drop and recreate the project's tables).

Run:
    AV_ROBUSTNESS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_fragility_terms.py -q
"""

import itertools
import json
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import db_pool, main as api_main
from src.danger import danger_score, fragility_terms as ft, pet_engine, ttc_engine
from src.scoring import db
from src.scoring.export_geometry import init_geometry_schema

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC = os.path.join(PROJECT, 'tests', 'data', 'public_scores_2026-10-15.json')

requires_db = pytest.mark.skipif(
    os.environ.get('AV_ROBUSTNESS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

GRID = [-1e-9, -0.0, 0.0, 0.005, 0.01, 0.0100001, 0.2, 0.5, 1.0, 2.5, 10.0, 60.0, 998.9999, 999.0, 999.0001, 1e6]
NEGATIVE_PET = [-3.6, -0.1, -1e-6]


# ── the same numbers as the scoring code ────────────────────────────────────────────────────


def test_every_constant_is_the_scoring_codes():
    assert ft.TTC_FLOOR == danger_score.TTC_FLOOR == 0.01
    assert ft.PET_FLOOR == danger_score.PET_FLOOR == 0.01
    assert ft.W_TTC == danger_score.W_TTC == 0.4
    assert ft.W_PET == danger_score.W_PET == 0.3
    assert ft.NO_CONFLICT == ttc_engine.TTC_INFINITY == pet_engine.PET_INFINITY == 999.0


def test_the_weights_are_four_sevenths_and_three_sevenths_and_sum_to_one():
    b = ft.breakdown(1.0, 1.0, 1.0)
    assert b['ttc_weight'] == pytest.approx(4 / 7, abs=0, rel=1e-15)
    assert b['pet_weight'] == pytest.approx(3 / 7, abs=0, rel=1e-15)
    assert b['ttc_weight'] + b['pet_weight'] == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize('ttc,pet', list(itertools.product(GRID, GRID + NEGATIVE_PET)))
def test_the_recomputed_score_is_compute_danger_score_with_no_perturbation(ttc, pet):
    expected = danger_score.compute_danger_score(ttc, pet, None)
    b = ft.breakdown(ttc, pet, expected)
    assert b['recomputed_score'] == expected
    assert b['ttc_term'] + b['pet_term'] == expected
    assert b['reproduces_stored_score'] is True


def test_the_terms_for_a_plain_pair_of_values():
    b = ft.breakdown(2.0, 4.0, danger_score.compute_danger_score(2.0, 4.0, None))
    assert b['ttc_signal'] == 0.5 and b['pet_signal'] == 0.25
    assert b['ttc_term'] == pytest.approx(0.5 * 4 / 7)
    assert b['pet_term'] == pytest.approx(0.25 * 3 / 7)
    assert (b['ttc_none'], b['pet_none'], b['ttc_at_floor'], b['pet_at_floor']) == (False, False, False, False)


def test_the_sentinel_contributes_nothing_and_is_flagged_not_floored():
    b = ft.breakdown(999.0, 3.0, 1.0)
    assert (b['ttc_signal'], b['ttc_term'], b['ttc_none'], b['ttc_at_floor']) == (0.0, 0.0, True, False)
    b = ft.breakdown(3.0, 999.0, 1.0)
    assert (b['pet_signal'], b['pet_term'], b['pet_none'], b['pet_at_floor']) == (0.0, 0.0, True, False)
    assert ft.breakdown(998.999, 3.0, 1.0)['ttc_none'] is False


def test_a_value_at_or_below_the_floor_is_capped_and_flagged():
    for value in (0.01, 0.005, 0.0, -0.0):
        b = ft.breakdown(value, 5.0, 1.0)
        assert b['ttc_signal'] == pytest.approx(100.0) and b['ttc_at_floor'] is True, value
    just_above = ft.breakdown(0.0100001, 5.0, 1.0)
    assert just_above['ttc_at_floor'] is False and just_above['ttc_signal'] < 100.0


def test_an_overlap_is_a_pet_at_the_floor_with_a_signal_of_one_hundred():
    for pet in (-3.6, -0.1, -1e-6, 0.0):
        b = ft.breakdown(5.0, pet, 1.0)
        assert b['pet_signal'] == pytest.approx(100.0) and b['pet_at_floor'] is True, pet
    assert ft.breakdown(5.0, 0.01, 1.0)['pet_signal'] == ft.breakdown(5.0, -3.6, 1.0)['pet_signal']


def test_both_at_the_floor_is_exactly_one_hundred():
    b = ft.breakdown(0.0, -2.0, 100.0)
    assert b['recomputed_score'] == pytest.approx(100.0, abs=1e-12)
    assert b['reproduces_stored_score'] is True


def test_a_missing_ttc_or_pet_gives_no_breakdown():
    assert ft.breakdown(None, 1.0, 5.0) is None
    assert ft.breakdown(1.0, None, 5.0) is None
    assert ft.breakdown(None, None, 5.0) is None


def test_the_breakdown_says_when_it_does_not_reproduce_the_stored_score():
    exact = danger_score.compute_danger_score(0.5, 2.0, None)
    assert ft.breakdown(0.5, 2.0, exact)['reproduces_stored_score'] is True
    assert ft.breakdown(0.5, 2.0, exact + 1e-6)['reproduces_stored_score'] is False
    assert ft.breakdown(0.5, 2.0, exact * 1.01)['reproduces_stored_score'] is False
    assert ft.breakdown(0.5, 2.0, 0.0)['reproduces_stored_score'] is False
    # the roundoff headroom: below it the same number, above it not
    assert ft.breakdown(0.5, 2.0, exact * (1 + 5e-10))['reproduces_stored_score'] is True
    assert ft.breakdown(0.5, 2.0, exact * (1 + 5e-9))['reproduces_stored_score'] is False
    # the score the perturbation formula would give is not this score
    with_perturbation = danger_score.compute_danger_score(0.5, 2.0, 0.1)
    assert ft.breakdown(0.5, 2.0, with_perturbation)['reproduces_stored_score'] is False


def test_the_floors_and_weights_are_in_the_breakdown_so_a_reader_need_not_assume_them():
    b = ft.breakdown(1.0, 1.0, 1.0)
    assert (b['ttc_floor_s'], b['pet_floor_s']) == (0.01, 0.01)


# ── the public data ─────────────────────────────────────────────────────────────────────────


def _public():
    with open(PUBLIC, encoding='utf-8') as f:
        return json.load(f)['rows']


def test_the_public_rows_are_all_there():
    rows = _public()
    assert len(rows) == 100 and len({r['scenario_id'] for r in rows}) == 100


def test_every_public_row_reproduces_its_stored_score_exactly():
    kinds = set()
    for r in _public():
        b = ft.breakdown(r['min_ttc'], r['min_pet'], r['fragility_score'])
        assert b is not None, r['scenario_id']
        assert b['reproduces_stored_score'] is True, r['scenario_id']
        assert b['recomputed_score'] == r['fragility_score'], r['scenario_id']
        kinds.add((b['ttc_none'], b['ttc_at_floor'], b['pet_none'], b['pet_at_floor']))
    # the data really has the cases the flags exist for
    assert any(k[0] for k in kinds) and any(k[1] for k in kinds) and any(k[2] for k in kinds) and any(k[3] for k in kinds)


# ── deployable without numpy ────────────────────────────────────────────────────────────────


def _runs_without_numpy(code):
    # -I ignores PYTHONPATH and the working directory, so the project is put on the path by hand; numpy and scipy are made
    # unimportable, as they are on the deployed API.
    guard = f"import sys; sys.path.insert(0, {PROJECT!r}); sys.modules['numpy'] = None; sys.modules['scipy'] = None; "
    return subprocess.run([sys.executable, '-I', '-c', guard + code], capture_output=True, text=True)


def test_the_module_imports_without_numpy():
    run = _runs_without_numpy('import src.danger.fragility_terms as m; print(m.breakdown(1.0, 1.0, 1.0)["ttc_term"])')
    assert run.returncode == 0, run.stderr
    assert float(run.stdout) == pytest.approx(4 / 7)


def test_the_api_imports_without_numpy():
    run = _runs_without_numpy('import src.api.main; print("ok")')
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == 'ok'


# ── the contract ────────────────────────────────────────────────────────────────────────────


def test_only_the_detail_has_a_breakdown_in_the_published_contract():
    schema = api_main.create_app().openapi()['components']['schemas']
    assert 'fragility_breakdown' in schema['ScenarioDetail']['properties']
    assert 'fragility_breakdown' not in schema['ScenarioSummary']['properties']
    assert set(schema['FragilityBreakdown']['properties']) == {
        'ttc_weight', 'pet_weight', 'ttc_signal', 'pet_signal', 'ttc_term', 'pet_term', 'recomputed_score',
        'ttc_floor_s', 'pet_floor_s', 'ttc_none', 'pet_none', 'ttc_at_floor', 'pet_at_floor', 'reproduces_stored_score',
    }
    assert 'fragility_breakdown' not in schema['ScenarioPage']['properties']


# ── the route ───────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def conn():
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    init_geometry_schema(connection)
    yield connection
    connection.rollback()
    connection.close()


@pytest.fixture
def client(conn, monkeypatch):
    monkeypatch.setattr(api_main, 'init_pool', lambda: None)
    monkeypatch.setattr(api_main, 'close_pool', lambda: None)
    app = api_main.create_app()
    app.dependency_overrides[db_pool.get_db] = lambda: conn
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _store(connection, sid, ttc, pet, fragility):
    db.upsert_scores(connection, [dict(scenario_id=sid, shard='synthetic', n_agents=2, min_ttc=ttc, min_pet=pet,
                                       fragility_score=fragility)])


@requires_db
def test_a_stored_row_serves_its_breakdown_and_it_adds_up_to_the_stored_score(conn, client):
    score = danger_score.compute_danger_score(0.5, 2.0, None)
    _store(conn, 'plain', 0.5, 2.0, score)
    body = client.get('/scenarios/plain').json()
    b = body['fragility_breakdown']
    assert b['reproduces_stored_score'] is True
    assert b['ttc_term'] + b['pet_term'] == body['fragility_score'] == score
    assert (b['ttc_signal'], b['pet_signal']) == (2.0, 0.5)
    assert (b['ttc_none'], b['pet_none'], b['ttc_at_floor'], b['pet_at_floor']) == (False, False, False, False)


@requires_db
def test_the_sentinel_and_the_floor_are_served_as_flags(conn, client):
    _store(conn, 'none_ttc', 999.0, -3.6, danger_score.compute_danger_score(999.0, -3.6, None))
    b = client.get('/scenarios/none_ttc').json()['fragility_breakdown']
    assert (b['ttc_none'], b['ttc_term'], b['pet_at_floor']) == (True, 0.0, True)
    assert b['pet_term'] == pytest.approx(100 * 3 / 7)
    assert b['reproduces_stored_score'] is True


@requires_db
def test_a_row_without_ttc_or_pet_has_no_breakdown_and_is_otherwise_served(conn, client):
    _store(conn, 'no_ttc', None, 2.0, 1.0)
    _store(conn, 'no_pet', 1.0, None, 1.0)
    for sid in ('no_ttc', 'no_pet'):
        r = client.get(f'/scenarios/{sid}')
        assert r.status_code == 200
        assert r.json()['fragility_breakdown'] is None
        assert r.json()['fragility_score'] == 1.0


@requires_db
def test_a_stored_score_that_is_not_the_formulas_says_so(conn, client):
    _store(conn, 'odd', 0.5, 2.0, 7.0)
    b = client.get('/scenarios/odd').json()['fragility_breakdown']
    assert b['reproduces_stored_score'] is False
    assert b['recomputed_score'] == danger_score.compute_danger_score(0.5, 2.0, None) != 7.0


@requires_db
def test_the_ranked_list_carries_no_breakdown(conn, client):
    _store(conn, 'plain', 0.5, 2.0, danger_score.compute_danger_score(0.5, 2.0, None))
    row = client.get('/scenarios').json()['items'][0]
    assert 'fragility_breakdown' not in row
