"""
test_api_replay_drift.py — the two replay-drift figures on GET /scenarios/{id}.

search_provenance records, for every searched scenario, how far the zero-perturbation
replay of the challenger sits from its logged track:

  baseline_replay_error          the largest such distance over the whole track, metres
  baseline_offset_at_collision   that distance at the collision frame, metres; None
                                 unless a collision was found

The API has never served either. These tests pin that ScenarioDetail (and ONLY
ScenarioDetail) now does, as nullable metres, read straight from search_provenance,
and that a row without a usable value returns null rather than 0 or a 500.

What is NOT asserted, deliberately: any relationship between these numbers and
min_perturbation. The perturbation is a weighted norm over components of mixed units;
the drift figures are metres. Nothing here, or in the API, may relate them.

Needs a DISPOSABLE Postgres/PostGIS database; skips unless AV_ROBUSTNESS_DB=1, the same
switch tests/test_api_robustness.py uses (these tests drop and recreate the project's
tables).

Run:
    AV_ROBUSTNESS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_api_replay_drift.py -q
"""

import json
import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import db_pool, main as api_main
from src.api.export_openapi import render_schema
from src.scoring import db
from src.scoring.batch_scorer import _stress_one
from src.scoring.export_geometry import init_geometry_schema

requires_db = pytest.mark.skipif(
    os.environ.get('AV_ROBUSTNESS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

OFFSET = 'baseline_offset_at_collision'
DRIFT = 'baseline_replay_error'
DE_KWARGS = {'popsize': 8, 'maxiter': 40, 'seed': 1}


# ── fixtures ────────────────────────────────────────────────────────────────────

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


@pytest.fixture
def client(conn, monkeypatch):
    monkeypatch.setattr(api_main, 'init_pool', lambda: None)
    monkeypatch.setattr(api_main, 'close_pool', lambda: None)
    app = api_main.create_app()
    app.dependency_overrides[db_pool.get_db] = lambda: conn
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _seed(connection, sid='scene'):
    db.upsert_scores(connection, [dict(scenario_id=sid, shard='synthetic', n_agents=2,
                                       min_ttc=9.0, min_pet=9.0, fragility_score=1.0)])


def _lane_scene(lateral_m=None):
    """
    Two vehicles heading +x. With lateral_m=None the challenger crosses the SDC's path
    (the geometry tests/test_api_robustness.py uses: a verified collision under
    perturbation). With lateral_m set it drives that many metres to the side, which no
    bounded perturbation reaches: a completed search that finds no collision.
    """
    t = np.arange(91) * 0.1
    states = np.zeros((2, 91, 7), dtype=np.float32)
    if lateral_m is None:
        states[0, :, 0] = -20.0 + 10.0 * t
        states[0, :, 2] = 10.0
        states[1, :, 1] = -28.0 + 10.0 * t
        states[1, :, 3] = 10.0
        states[1, :, 4] = np.pi / 2
    else:
        states[0, :, 0] = 10.0 * t
        states[0, :, 2] = 10.0
        states[1, :, 0] = 10.0 * t
        states[1, :, 1] = lateral_m
        states[1, :, 2] = 10.0
    states[:, :, 5:7] = [4.5, 2.0]
    return states, np.ones((2, 91), dtype=bool), np.array([1, 1])


def _store(connection, lateral_m=None, sid='scene'):
    """The real write path: _stress_one -> update_stress_results."""
    _seed(connection, sid)
    result = _stress_one(*_lane_scene(lateral_m), 0, de_kwargs=DE_KWARGS)
    db.update_stress_results(connection, {sid: result})
    return result


def _provenance(connection, sid='scene'):
    with connection.cursor() as cur:
        cur.execute('SELECT search_provenance FROM scenario_scores WHERE scenario_id=%s', (sid,))
        return cur.fetchone()[0]


def _set_provenance(connection, provenance_sql, sid='scene'):
    """Overwrite the stored provenance with a JSONB expression, leaving the rest of the result."""
    with connection.cursor() as cur:
        cur.execute(f'UPDATE scenario_scores SET search_provenance = {provenance_sql} '
                    'WHERE scenario_id = %s', (sid,))
    connection.commit()


def _row_fingerprint(connection):
    with connection.cursor() as cur:
        cur.execute('SELECT md5(string_agg(t::text, E\'\\n\' ORDER BY scenario_id)) '
                    'FROM scenario_scores t')
        return cur.fetchone()[0]


# ── a real collision result ─────────────────────────────────────────────────────

@requires_db
def test_a_collision_result_serves_both_figures_exactly_as_stored(conn, client):
    result = _store(conn)
    assert result['outcome'] == 'collision_found', f'fixture regressed: {result}'
    stored = _provenance(conn)
    assert isinstance(stored[OFFSET], float) and isinstance(stored[DRIFT], float)

    body = client.get('/scenarios/scene').json()

    assert body[OFFSET] == stored[OFFSET]
    assert body[DRIFT] == stored[DRIFT]


@requires_db
def test_values_round_trip_to_the_identical_double(conn, client):
    """JSONB keeps the decimal; the API must hand back the very same float."""
    _store(conn)
    awkward_offset, awkward_drift = 0.1 + 0.2, 1.2345678901234567e-7
    _set_provenance(conn, "search_provenance || %s::jsonb"
                    % ("'" + json.dumps({OFFSET: awkward_offset, DRIFT: awkward_drift}) + "'"))

    body = client.get('/scenarios/scene').json()

    assert body[OFFSET] == awkward_offset
    assert body[DRIFT] == awkward_drift


@requires_db
def test_a_zero_offset_is_zero_not_null(conn, client):
    _store(conn)
    _set_provenance(conn, "search_provenance || '{\"%s\": 0.0, \"%s\": 0.0}'::jsonb" % (OFFSET, DRIFT))

    body = client.get('/scenarios/scene').json()

    assert body[OFFSET] == 0.0 and body[OFFSET] is not None
    assert body[DRIFT] == 0.0 and body[DRIFT] is not None


# ── rows that carry no usable value: null, never 0, never a 500 ──────────────────

@requires_db
def test_a_legacy_row_without_the_keys_returns_null(conn, client):
    _store(conn)
    _set_provenance(conn, f"search_provenance - '{OFFSET}' - '{DRIFT}'")

    response = client.get('/scenarios/scene')

    assert response.status_code == 200
    body = response.json()
    assert OFFSET in body and body[OFFSET] is None
    assert DRIFT in body and body[DRIFT] is None


@requires_db
def test_a_stored_result_with_no_provenance_returns_null(conn, client):
    _store(conn)
    _set_provenance(conn, 'NULL')

    response = client.get('/scenarios/scene')

    assert response.status_code == 200
    body = response.json()
    assert OFFSET in body and body[OFFSET] is None
    assert DRIFT in body and body[DRIFT] is None


@requires_db
def test_a_scenario_with_no_stored_result_returns_null(conn, client):
    _seed(conn)

    response = client.get('/scenarios/scene')

    assert response.status_code == 200
    body = response.json()
    assert OFFSET in body and body[OFFSET] is None
    assert DRIFT in body and body[DRIFT] is None


@requires_db
def test_a_json_null_returns_null(conn, client):
    """db._json_safe stores a non-finite float as JSON null plus a marker."""
    _store(conn)
    _set_provenance(conn, "search_provenance || '{\"%s\": null, \"%s\": null, "
                          "\"nonfinite_fields\": {\"%s\": \"inf\"}}'::jsonb" % (OFFSET, DRIFT, DRIFT))

    body = client.get('/scenarios/scene').json()

    assert body[OFFSET] is None and body[DRIFT] is None


@requires_db
@pytest.mark.parametrize('junk', ['"inf"', '"0.4"', 'true', '[0.4]', '{"m": 0.4}'])
def test_a_non_number_returns_null_not_a_500(conn, client, junk):
    _store(conn)
    _set_provenance(conn, "search_provenance || '{\"%s\": %s, \"%s\": %s}'::jsonb"
                    % (OFFSET, junk, DRIFT, junk))

    response = client.get('/scenarios/scene')

    assert response.status_code == 200, response.text
    body = response.json()
    assert body[OFFSET] is None and body[DRIFT] is None


# ── JSON integers ───────────────────────────────────────────────────────────────

@requires_db
@pytest.mark.parametrize('integer', [0, 1, 7])
@pytest.mark.parametrize('key', [OFFSET, DRIFT])
def test_a_json_integer_is_served_as_a_float(conn, client, key, integer):
    """A JSON integer is a number: it comes back as 1.0, not null and not a 500."""
    _store(conn)
    _set_provenance(conn, "search_provenance || '{\"%s\": %d}'::jsonb" % (key, integer))

    response = client.get('/scenarios/scene')

    assert response.status_code == 200, response.text
    served = response.json()[key]
    assert served == float(integer)
    assert isinstance(served, float), f'{served!r} is a {type(served).__name__}'


# ── a completed search that found nothing ────────────────────────────────────────

@requires_db
def test_a_no_collision_result_has_no_offset_but_has_a_drift(conn, client):
    result = _store(conn, lateral_m=300.0)
    assert result['outcome'] == 'no_collision_found', f'fixture regressed: {result}'
    stored = _provenance(conn)
    assert stored[OFFSET] is None and isinstance(stored[DRIFT], float)

    body = client.get('/scenarios/scene').json()

    assert body['stress_outcome'] == 'no_collision_found'
    assert OFFSET in body and body[OFFSET] is None
    assert body[DRIFT] == stored[DRIFT]


# ── what must NOT change ────────────────────────────────────────────────────────

@requires_db
def test_only_the_detail_response_carries_the_figures(conn, client):
    _store(conn)

    listing = client.get('/scenarios').json()['items'][0]
    perturbed = client.get('/scenarios/scene/perturbed').json()
    detail = client.get('/scenarios/scene').json()

    for absent_from in (listing, perturbed):
        assert OFFSET not in absent_from and DRIFT not in absent_from
    assert OFFSET in detail and DRIFT in detail


@requires_db
def test_reading_the_figures_changes_nothing_stored(conn, client):
    _store(conn)
    before = _row_fingerprint(conn)

    for _ in range(3):
        assert client.get('/scenarios/scene').status_code == 200

    assert _row_fingerprint(conn) == before


# ── the published contract (no database) ────────────────────────────────────────

def _schemas():
    return json.loads(render_schema())['components']['schemas']


def test_the_contract_declares_both_as_nullable_numbers_with_units():
    props = _schemas()['ScenarioDetail']['properties']
    for key in (OFFSET, DRIFT):
        assert key in props, f'{key} missing from the ScenarioDetail contract'
        kinds = {branch.get('type') for branch in props[key]['anyOf']}
        assert kinds == {'number', 'null'}, props[key]
        assert 'metre' in props[key]['description'].lower(), props[key]


def test_the_contract_keeps_them_off_the_list_summary():
    props = _schemas()['ScenarioSummary']['properties']
    assert OFFSET not in props and DRIFT not in props


# Words the descriptions must not use, and what they must say. The figures qualify a
# result; they are not an uncertainty on it, a margin, a safety statement, or an
# explanation of what produced the collision.
_BANNED = ('error bar', 'margin', 'uncertainty', 'safe', 'safety', 'cause', 'caused', 'causes')


def test_the_contract_descriptions_use_none_of_the_banned_words():
    import re
    props = _schemas()['ScenarioDetail']['properties']
    for key in (OFFSET, DRIFT):
        text = props[key]['description']
        for word in _BANNED:
            assert not re.search(rf'\b{re.escape(word)}\b', text, re.IGNORECASE), (
                f'{key}: description uses "{word}": {text!r}')


def test_the_contract_descriptions_say_the_figure_is_not_comparable_with_min_perturbation():
    import re
    props = _schemas()['ScenarioDetail']['properties']
    for key in (OFFSET, DRIFT):
        text = props[key]['description']
        assert re.search(r'not comparable', text, re.IGNORECASE), (key, text)
        assert 'min_perturbation' in text, (key, text)
        assert re.search(r'weighted norm', text, re.IGNORECASE), (key, text)
        assert re.search(r'mixed units', text, re.IGNORECASE), (key, text)
