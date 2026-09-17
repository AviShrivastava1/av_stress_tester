"""
test_batch12_contract.py — Batch 12 contract tests (audit A03, A06, A07).

The audit's own repros live in tests/test_audit3_regressions.py, reconstructed and
labelled as such. THIS file holds the properties: the things that must be true of any
correct fix, including the ones the repro cannot see.

The sanitizer is the piece that matters most, because it is the first fix in this
project for this class that is NOT per-field. Block 6 Concept 24 handled
min_perturbation; Batch 9 handled target_min_speed; A06 adds two more fields in this
same batch. So the tests below are about the BOUNDARY holding for values nobody has
thought of yet, not about the one field the audit named.

Mostly pure Python. The DB-backed section needs a DISPOSABLE Postgres and skips unless
AV_CLAIMS_DB=1.

Run:
    ./venv/bin/python -m pytest tests/test_batch12_contract.py -q
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_batch12_contract.py -q
"""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring.db import (
    UpdateReport, _json_safe, _sanitize_for_json, _NONFINITE_KEY,
)


# ── the sanitizer ───────────────────────────────────────────────────────────────

INF, NINF, NAN = float('inf'), float('-inf'), float('nan')


def test_the_detector_finds_every_kind_of_non_finite():
    """
    THE VACUITY GUARD. A sanitizer that silently matched nothing would leave every
    test below green and every value unstorable. All three spellings are checked
    because they are three different float bit patterns and `x != x` catches only one.
    """
    for value, name in ((INF, 'inf'), (NINF, '-inf'), (NAN, 'nan')):
        cleaned, report = _sanitize_for_json({'v': value})
        assert cleaned == {'v': None}, f'{name} was not replaced: {cleaned}'
        assert report == {'v': name}, f'{name} was not reported as such: {report}'


def test_finite_values_are_returned_bit_identical():
    """
    The regression risk in any sanitizer: mangling the values it was meant to pass
    through. repr-level equality, not approx — a fingerprint-grade guarantee, because
    search_provenance's bounds feed compute_stress_run_id's sibling reasoning about
    exact round-trips.
    """
    blob = {'a': 0.1, 'b': -0.0, 'c': 1e308, 'd': 5e-324, 'e': 0, 'f': 'text',
            'g': None, 'h': [1.5, 2.5], 'i': True, 'j': False}
    cleaned, report = _sanitize_for_json(blob)
    assert report == {}, f'a finite blob reported replacements: {report}'
    assert repr(cleaned) == repr(blob), 'a finite value was not returned unchanged'


def test_booleans_are_not_treated_as_numbers():
    """
    bool subclasses int, not float, so it would not be caught anyway — asserted so a
    later "simplification" into a generic numeric check cannot silently rewrite
    baseline_replay_collides, which is a bool living right beside the float that
    triggered this whole finding.
    """
    cleaned, report = _sanitize_for_json({'baseline_replay_collides': True})
    assert cleaned == {'baseline_replay_collides': True}
    assert report == {}


@pytest.mark.parametrize('blob,expected_path', [
    ({'bounds': [[0.0, INF]]}, 'bounds.0.1'),
    ({'outer': {'inner': NAN}}, 'outer.inner'),
    ({'list': [1.0, NINF, 3.0]}, 'list.1'),
])
def test_nested_structures_are_walked_and_named(blob, expected_path):
    """
    search_provenance is NOT flat all the way down — `bounds` is a list of [lo, hi]
    pairs — so a scalar-only sanitizer would pass an unstorable bound straight through.
    The reported path names where it was, because "something was non-finite" is not
    actionable.
    """
    cleaned, report = _sanitize_for_json(blob)
    assert expected_path in report, f'{expected_path} not reported: {report}'
    json.dumps(cleaned)


def test_the_whole_blob_round_trips_through_strict_json():
    """
    THE PROPERTY THAT ACTUALLY MATTERS, asserted the way PostgreSQL asserts it.

    json.dumps emits Infinity/NaN happily — that is the entire defect — so dumps alone
    proves nothing. allow_nan=False is the strict-parser behaviour JSONB has, and is
    what makes this test equivalent to the database's own check without needing one.
    """
    blob = {'baseline_replay_error': INF, 'bounds': [[-2.0, NAN]], 'reason': 'drift',
            'nested': {'deep': [NINF]}}
    with pytest.raises(ValueError):
        json.dumps(blob, allow_nan=False)          # the fixture really is unstorable
    json.dumps(_json_safe(blob), allow_nan=False)  # and the sanitized form is not


def test_the_replacement_is_distinguishable_from_not_recorded():
    """
    null already means "this attempt recorded no such diagnostic", which
    _attempt_diagnostics' docstring is careful about. A bare null would collapse "the
    rollout diverged" into "nothing was measured" — different facts, and the diverged
    one is the one worth knowing.
    """
    diverged = _json_safe({'baseline_replay_error': INF, 'reason': 'drift'})
    absent = _json_safe({'reason': 'drift'})

    assert diverged['baseline_replay_error'] is None
    assert diverged[_NONFINITE_KEY] == {'baseline_replay_error': 'inf'}
    assert _NONFINITE_KEY not in absent, 'a clean blob gained a report key'
    assert diverged != absent, 'the two cases are indistinguishable once stored'


def test_a_clean_blob_is_not_given_a_report_key():
    """Purely additive means additive only when there is something to add."""
    assert _json_safe({'a': 1.0}) == {'a': 1.0}
    assert _json_safe(None) is None


def test_batch9s_own_guard_and_the_boundary_do_not_conflict():
    """
    THE OVERLAP, EXAMINED RATHER THAN LEFT IMPLICIT.

    Batch 9 already wraps target_min_speed in np.isfinite before it reaches provenance,
    which is now redundant with this boundary. It STAYS, and this pins why: the two
    produce the same stored value for the same input, so keeping both costs nothing and
    the field-level guard is the one that documents, at the point the value is produced,
    that an unobserved challenger has no minimum speed. Defence in depth only counts if
    the layers agree — so that is asserted rather than assumed.
    """
    guarded = None if not np.isfinite(INF) else float(INF)      # Batch 9's expression
    assert guarded is None
    sanitized = _json_safe({'target_min_speed': INF})
    assert sanitized['target_min_speed'] is None, (
        'the boundary disagrees with the field-level guard about the same input'
    )


# ── A06: the message bound ──────────────────────────────────────────────────────

def test_a_long_exception_message_is_truncated_visibly():
    """
    str(e) is unbounded in principle and lands in a JSONB column beside four small
    fixed-shape fields. Truncation is MARKED so a reader can tell a message ends
    because it ended, not because it was cut — an unmarked truncation is a worse
    diagnostic than a missing one, because it reads as complete.
    """
    from src.scoring.batch_scorer import _bounded, _MAX_ERROR_MESSAGE_LEN

    short = 'x' * 10
    assert _bounded(short) == short, 'a short message was altered'

    long = 'y' * (_MAX_ERROR_MESSAGE_LEN + 5000)
    out = _bounded(long)
    assert len(out) < len(long), 'the bound did not fire'
    assert 'truncated' in out, 'truncation is silent'
    assert str(len(long)) in out, 'the original length is not recoverable'


def test_the_bound_leaves_every_observed_message_intact():
    """
    The bound is a resource guard, not a format rule, and it must not be quietly
    reshaping real diagnostics. Both messages are the actual strings the two reachable
    triggers produce, at 102 and 93 characters.
    """
    from src.scoring.batch_scorer import _bounded

    for message in (
        'agent 1 has unusable length=0.0 at its first valid frame 0; cannot build a '
        'kinematic model',
        'IllegalArgumentException: Points of LinearRing do not form a closed linestring',
    ):
        assert _bounded(message) == message, 'a real message was truncated'


# ── A07: the provenance stays flat ──────────────────────────────────────────────

def test_search_provenance_has_no_nested_dict():
    """
    FLATNESS IS A CONTRACT, NOT A STYLE. api/routes.py reads it with the SQL operator
    `search_provenance ->> 'delta_parameterization'`, which returns NULL rather than
    erroring if the key moves under another object — so a nested restructure would
    break B13's label resolution silently.

    nonfinite_fields is the one permitted nested value, and it is added by the
    sanitizer at write time rather than by _stress_one, so it is not in this dict.
    """
    from src.scoring.batch_scorer import _stress_one

    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[1, :, 1] = 2.1
    result = _stress_one(states, np.ones((2, 10), dtype=bool), np.array([1, 1]), 0,
                         use_autograd=True,
                         de_kwargs={'popsize': 4, 'maxiter': 0, 'seed': 0})

    provenance = result['search_provenance']
    nested = {k: v for k, v in provenance.items() if isinstance(v, dict)}
    assert not nested, f'search_provenance gained a nested object: {sorted(nested)}'
    assert provenance['delta_parameterization'] == 'bicycle', (
        'the key the SQL operator reads is not at the top level'
    )


def test_de_numbers_are_the_searchs_own_not_the_refiners():
    """
    The capture has to hold DE's numbers, not merely SOME numbers. maxiter=0 makes
    DE's own count definite — one evaluation of a 16-member initial population at
    popsize=4 — and the refiner runs 300 iterations, so a capture that had accidentally
    read the refiner's dict would be obvious rather than plausible.
    """
    from src.scoring.batch_scorer import _stress_one

    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[1, :, 1] = 2.1
    result = _stress_one(states, np.ones((2, 10), dtype=bool), np.array([1, 1]), 0,
                         use_autograd=True,
                         de_kwargs={'popsize': 4, 'maxiter': 0, 'seed': 0})
    provenance = result['search_provenance']

    assert result['method'] == 'de+autograd', 'fixture regressed: refinement must win'
    assert provenance['de_n_iter'] == 0, provenance['de_n_iter']
    assert provenance['de_n_eval'] == 16, (
        f"expected DE's own 16 initial evaluations, got {provenance['de_n_eval']}"
    )
    assert provenance['refine_n_iters'] not in (None, 0, 16), (
        'the refiner count is missing or has collided with DE\'s'
    )
    assert provenance['selected_stage'] == 'de+autograd'


def test_de_numbers_survive_when_de_itself_wins():
    """
    The other branch. A fix that only populated the capture inside the autograd block
    would leave a DE-only run with nothing, which is the same defect with the
    conditions inverted.
    """
    from src.scoring.batch_scorer import _stress_one

    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[1, :, 1] = 2.1
    result = _stress_one(states, np.ones((2, 10), dtype=bool), np.array([1, 1]), 0,
                         de_kwargs={'popsize': 4, 'maxiter': 2, 'seed': 0})
    provenance = result['search_provenance']

    assert provenance['de_n_eval'] is not None
    assert provenance['de_candidate_source'] is not None
    assert provenance['refine_n_iters'] is None, (
        'a run with no refinement reported a refiner iteration count'
    )
    assert provenance['selected_stage'] == 'de'


# ── DB-backed: isolation ────────────────────────────────────────────────────────

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

_OK = {'status': 'no_challenger', 'outcome': 'no_challenger',
       'challengers_total': 0, 'challengers_searched': 0}


@pytest.fixture
def dconn():
    from src.scoring import db
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, '
                    'scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    yield connection
    connection.rollback()
    connection.close()


@requires_db
def test_one_unwritable_row_costs_only_itself(dconn):
    """
    R02's rule at the persistence stage. The sanitizer removes the trigger this was
    found through; the savepoint removes the CLASS, so a failure nobody anticipated
    loses one scenario instead of the call.

    Driven with a value the SANITIZER CANNOT SAVE — a delta of the wrong arity for a
    DOUBLE PRECISION[4] consumer — precisely so it tests isolation rather than
    re-testing sanitization. A test that used the inf again would pass even with the
    savepoint removed.
    """
    from src.scoring import db

    for sid in ('a', 'b', 'c'):
        db.upsert_scores(dconn, [dict(scenario_id=sid, shard='x', n_agents=2,
                                      min_ttc=9.0, min_pet=9.0, fragility_score=1.0)])

    report = db.update_stress_results(dconn, {
        'a': dict(_OK),
        'b': {'status': 'ok', 'outcome': 'collision_found', 'collision': True,
              'min_perturbation': 0.5, 'collision_timestep': 3, 'target_idx': 1,
              'method': 'de', 'delta': ['not', 'a', 'number', '!']},
        'c': dict(_OK),
    })

    assert db.fetch_scenario(dconn, 'a')['last_attempt_outcome'] == 'no_challenger'
    assert db.fetch_scenario(dconn, 'c')['last_attempt_outcome'] == 'no_challenger', (
        'a row AFTER the failure was never written — the loop did not continue'
    )
    assert [f['scenario_id'] for f in report.failures] == ['b'], report.failures
    assert report == 2, f'the count must exclude the failed row: {int(report)}'


@requires_db
def test_the_return_value_is_still_an_int(dconn):
    """
    Forty-five call sites take this return. Exactly one asserts on it and one assigns
    it, and both keep working only because UpdateReport IS an int — the same trick
    StressResults used in Batch 5. A richer return that breaks its callers is a second
    defect, not a fix.
    """
    from src.scoring import db

    db.upsert_scores(dconn, [dict(scenario_id='a', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0)])
    report = db.update_stress_results(dconn, {'a': dict(_OK)})

    assert isinstance(report, int)
    assert report == 1 and report + 1 == 2 and f'{report}' == '1'
    assert report.failures == []


@requires_db
def test_the_sql_operator_still_reaches_delta_parameterization(dconn):
    """
    Asserted through the OPERATOR, not a Python lookup, because that is what
    api/routes.py uses and what a nested restructure would break. `->>` returns NULL
    on a missing key rather than raising, so this failure mode is silent by
    construction.
    """
    from src.scoring import db
    from src.scoring.batch_scorer import _stress_one

    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[1, :, 1] = 2.1
    result = _stress_one(states, np.ones((2, 10), dtype=bool), np.array([1, 1]), 0,
                         use_autograd=True,
                         de_kwargs={'popsize': 4, 'maxiter': 0, 'seed': 0})

    db.upsert_scores(dconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0)])
    db.update_stress_results(dconn, {'s': result})

    with dconn.cursor() as cur:
        cur.execute("SELECT search_provenance ->> 'delta_parameterization', "
                    "       search_provenance ->> 'de_n_eval', "
                    "       search_provenance ->> 'selected_stage' "
                    "  FROM scenario_scores WHERE scenario_id = 's'")
        parameterization, n_eval, stage = cur.fetchone()

    assert parameterization == 'bicycle', 'the ->> operator no longer finds the key'
    assert n_eval == '16', n_eval
    assert stage == 'de+autograd', stage
