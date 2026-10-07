"""
test_run_identity_v2.py — a stress-run id covers the search provenance, not only the delta.

compute_stress_run_id used to hash (scenario, target, delta, method). The same delta replayed
under a different heading blend, acceleration cap, collision predicate, bound or search budget
is a different result, and the read-side join (scenario_scores.stress_run_id against
perturbed_paths.stress_run_id) could not tell: old geometry kept matching a newly verified row
as long as the four delta components happened not to change.

With provenance supplied the id is now a hash of a canonical (sorted-key, compact) JSON object
tagged version 2 that includes the sanitised provenance. With NO provenance it is the previous
spelling byte for byte, which is the compatibility path for hand-built callers and for rows
stored before this change.

Ids are comparable within one session and are not guaranteed across machines: the provenance
holds measured floats (for example baseline_replay_error) that can differ in the last digit
between numpy builds.

Expected values here are derived in the test: the previous spelling from its documented formula
(a separate implementation of it, below), and sensitivity by changing one provenance value at a time.

Pure python/numpy, no database (the writer/exporter agreement tests are in
test_run_identity_v2_db.py). Run:
    ./venv/bin/python -m pytest tests/test_run_identity_v2.py -q
"""

import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring.db import compute_stress_run_id

DELTA = [0.1, 0.0, 0.0, 0.0]
PROVENANCE = {
    'replay_model_version': 'kinematic-replay-v1',
    'collision_geometry_version': 'oriented-box-float64-v1',
    'delta_parameterization': 'bicycle',
    'heading_speed_floor': 0.5,
    'heading_transition_width': 0.05,
    'a_max': 12.0,
    'de_tol': 1e-05,
    'de_seed': 1,
    'bounds': [[-3.0, 3.0], [-0.2, 0.2]],
    'target_has_interior_gap': False,
    'max_speed_step': None,
}


def pre_v2_id(scenario_id, target_idx, delta, method):
    """The previous spelling, restated independently from its documented formula."""
    components = '|'.join(repr(float(x)) for x in (delta if delta is not None else ()))
    canonical = f'{scenario_id}|{int(target_idx)}|{components}|{method or ""}'
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]


# ── no provenance: the previous id, byte for byte ───────────────────────────────────


@pytest.mark.parametrize('delta, method', [
    ([0.1, 0.0, 0.0, 0.0], 'de'), ([-1.192e-07, 0.0, 0.0, 0.0], 'de+autograd'),
    (np.array([0.5, -0.25, 1e-05, 12.0], dtype=np.float32), 'de'), (None, None), ([], ''),
])
def test_without_provenance_the_id_is_the_previous_one(delta, method):
    # Passes on main on purpose: this is the compatibility contract, and the call has only the four arguments.
    assert compute_stress_run_id('scene', 1, delta, method) == pre_v2_id('scene', 1, delta, method)


@pytest.mark.parametrize('delta, method', [([0.1, 0.0, 0.0, 0.0], 'de'), (None, None)])
def test_an_explicit_none_provenance_is_the_previous_id_too(delta, method):
    assert compute_stress_run_id('scene', 1, delta, method, search_provenance=None) == pre_v2_id('scene', 1, delta, method)


# ── with provenance: a different, provenance-sensitive id ───────────────────────────


def version_2_id(scenario_id, target_idx, delta, method, provenance):
    """The documented version-2 spelling, restated independently: a sorted-key, compact JSON object."""
    import json
    canonical = json.dumps(
        {'version': 2, 'scenario_id': str(scenario_id), 'target_idx': int(target_idx),
         'delta': [float(x) for x in delta], 'method': method or '', 'search_provenance': provenance},
        sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]


@pytest.mark.parametrize('delta, method', [
    (DELTA, 'de'), (np.array([0.5, -0.25, 1e-05, 12.0], dtype=np.float32), 'de+autograd'), ([], None),
])
def test_the_version_2_id_is_the_documented_canonical_json(delta, method):
    provenance = {**PROVENANCE, 'unit': 'mètre'}            # non-ASCII text is hashed as written, not escaped
    assert compute_stress_run_id('scene', 1, delta, method, provenance) == version_2_id('scene', 1, delta, method, provenance)


def test_with_provenance_the_id_is_not_the_previous_one():
    assert compute_stress_run_id('scene', 1, DELTA, 'de', PROVENANCE) != pre_v2_id('scene', 1, DELTA, 'de')


def test_an_empty_provenance_is_still_version_2():
    # Absence (None) selects the previous spelling; an empty record does not.
    empty = compute_stress_run_id('scene', 1, DELTA, 'de', {})
    assert empty != pre_v2_id('scene', 1, DELTA, 'de')
    assert empty != compute_stress_run_id('scene', 1, DELTA, 'de', PROVENANCE)


def test_the_id_is_sixteen_lowercase_hex_digits():
    run = compute_stress_run_id('scene', 1, DELTA, 'de', PROVENANCE)
    assert len(run) == 16 and set(run) <= set('0123456789abcdef')


def test_the_same_inputs_give_the_same_id():
    assert compute_stress_run_id('scene', 1, DELTA, 'de', dict(PROVENANCE)) == compute_stress_run_id('scene', 1, DELTA, 'de', PROVENANCE)


@pytest.mark.parametrize('key, other', [
    ('replay_model_version', 'kinematic-replay-v2'),
    ('collision_geometry_version', 'oriented-box-float32-v0'),
    ('heading_speed_floor', 0.75),
    ('heading_transition_width', 0.0),
    ('a_max', 5.0),
    ('de_tol', 1e-03),
    ('de_seed', 2),
    ('bounds', [[-3.0, 3.0], [-0.2, 0.3]]),
    ('target_has_interior_gap', True),
    ('max_speed_step', 2.0),
    ('delta_parameterization', 'linear'),
])
def test_changing_any_one_provenance_value_changes_the_id(key, other):
    changed = {**PROVENANCE, key: other}
    assert compute_stress_run_id('scene', 1, DELTA, 'de', changed) != compute_stress_run_id('scene', 1, DELTA, 'de', PROVENANCE)


def test_adding_or_removing_a_provenance_key_changes_the_id():
    base = compute_stress_run_id('scene', 1, DELTA, 'de', PROVENANCE)
    assert compute_stress_run_id('scene', 1, DELTA, 'de', {**PROVENANCE, 'extra': 1}) != base
    assert compute_stress_run_id('scene', 1, DELTA, 'de', {k: v for k, v in PROVENANCE.items() if k != 'a_max'}) != base


@pytest.mark.parametrize('change', [
    dict(scenario_id='other'), dict(target_idx=2), dict(delta=[0.1, 0.0, 0.0, 1e-09]), dict(method='de+autograd'),
])
def test_the_four_original_inputs_still_matter(change):
    args = dict(scenario_id='scene', target_idx=1, delta=DELTA, method='de')
    assert compute_stress_run_id(**{**args, **change}, search_provenance=PROVENANCE) != compute_stress_run_id(**args, search_provenance=PROVENANCE)


def test_key_order_does_not_matter_at_any_depth():
    reordered = {k: PROVENANCE[k] for k in reversed(list(PROVENANCE))}
    reordered['nested'] = {'b': 1, 'a': {'y': 2.0, 'x': 1}}
    original = {**PROVENANCE, 'nested': {'a': {'x': 1, 'y': 2.0}, 'b': 1}}
    assert compute_stress_run_id('scene', 1, DELTA, 'de', reordered) == compute_stress_run_id('scene', 1, DELTA, 'de', original)


def test_numpy_scalars_in_provenance_give_the_id_of_the_python_values():
    numpy_version = {**PROVENANCE, 'a_max': np.float64(12.0), 'de_seed': np.int64(1), 'target_has_interior_gap': np.bool_(False),
                     'heading_speed_floor': np.float32(0.5), 'bounds': [[np.float32(-3.0), np.float32(3.0)], [np.float32(-0.2), np.float32(0.2)]]}
    python_version = {**PROVENANCE, 'bounds': [[float(np.float32(-3.0)), float(np.float32(3.0))], [float(np.float32(-0.2)), float(np.float32(0.2))]]}
    assert compute_stress_run_id('scene', 1, DELTA, 'de', numpy_version) == compute_stress_run_id('scene', 1, DELTA, 'de', python_version)


def test_a_non_finite_value_is_hashed_as_the_sanitised_record_and_does_not_raise():
    raw = {**PROVENANCE, 'baseline_replay_error': float('inf')}
    from src.scoring.db import _json_safe
    cleaned = _json_safe(raw)                   # what the writer stores: null plus nonfinite_fields

    assert cleaned['baseline_replay_error'] is None and cleaned['nonfinite_fields'] == {'baseline_replay_error': 'inf'}
    # writer (hashes the cleaned record) and exporter (hashes the raw one) agree
    assert compute_stress_run_id('scene', 1, DELTA, 'de', raw) == compute_stress_run_id('scene', 1, DELTA, 'de', cleaned)
    assert compute_stress_run_id('scene', 1, DELTA, 'de', raw) != compute_stress_run_id('scene', 1, DELTA, 'de', {**PROVENANCE, 'baseline_replay_error': 1.0})


def test_a_non_finite_delta_is_refused_when_the_provenance_is_hashed():
    # The version-2 form is strict JSON: a NaN or infinite delta cannot be spelled, so it raises
    # (the writer records that as a per-scenario failure) instead of hashing a non-standard token.
    # The previous spelling, used when there is no provenance, is unchanged and hashes it as text.
    with pytest.raises(ValueError):
        compute_stress_run_id('scene', 1, [float('nan'), 0.0, 0.0, 0.0], 'de', PROVENANCE)
    assert compute_stress_run_id('scene', 1, [float('nan'), 0.0, 0.0, 0.0], 'de') == pre_v2_id('scene', 1, [float('nan'), 0.0, 0.0, 0.0], 'de')


def test_an_int_and_a_float_of_the_same_value_are_different_provenance():
    # 12 and 12.0 are written differently in the canonical JSON. Provenance values are cast with float()
    # where they are floats, and a JSONB round trip keeps 12.0 a float (see the database test), so the
    # distinction is stable in practice; this pins that it exists.
    assert compute_stress_run_id('scene', 1, DELTA, 'de', {'a_max': 12}) != compute_stress_run_id('scene', 1, DELTA, 'de', {'a_max': 12.0})


# ── the version constants and the real provenance ───────────────────────────────────


def test_the_replay_model_version_is_defined():
    from src.physics import simulator
    assert getattr(simulator, 'REPLAY_MODEL_VERSION', None) == 'kinematic-replay-v1'


def test_the_collision_geometry_version_is_unchanged():
    from src.danger import collision_detector
    assert collision_detector.COLLISION_GEOMETRY_VERSION == 'oriented-box-float64-v1'


def _crossing():
    t = np.arange(91) * 0.1
    st = np.zeros((2, 91, 7), dtype=np.float32)
    st[0, :, 0] = -20.0 + 10.0 * t; st[0, :, 2] = 10.0
    st[1, :, 1] = -28.0 + 10.0 * t; st[1, :, 3] = 10.0; st[1, :, 4] = np.pi / 2
    st[:, :, 5:7] = [4.5, 2.0]
    return st, np.ones((2, 91), dtype=bool), np.array([1, 1])


def test_a_real_search_records_both_versions_in_its_provenance_and_the_id_depends_on_them():
    from src.danger.collision_detector import COLLISION_GEOMETRY_VERSION
    from src.physics.simulator import REPLAY_MODEL_VERSION
    from src.scoring.batch_scorer import _stress_one

    states, validity, types = _crossing()
    result = _stress_one(states, validity, types, 0, de_kwargs={'popsize': 8, 'maxiter': 40, 'seed': 1})

    assert result['outcome'] == 'collision_found', result['outcome']
    provenance = result['search_provenance']
    assert provenance['replay_model_version'] == REPLAY_MODEL_VERSION == 'kinematic-replay-v1'
    assert provenance['collision_geometry_version'] == COLLISION_GEOMETRY_VERSION
    run = compute_stress_run_id('x', result['target_idx'], result['delta'], result['method'], provenance)
    bumped = {**provenance, 'collision_geometry_version': 'oriented-box-float64-v2'}
    assert run != compute_stress_run_id('x', result['target_idx'], result['delta'], result['method'], bumped)
