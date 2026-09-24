"""
test_g08_replay_model_identity.py — fix G08 (independent review).

export_shard_geometry's PerturbationSpace(...) call (the one that rebuilds a
perturbed challenger's trajectory for publishing) passed neither
heading_speed_floor nor heading_transition_width, so it always replayed under
whatever the CURRENT module constants (V_HEADING_MIN, HEADING_TRANSITION_WIDTH)
happen to be — not whatever they were when the search that produced this delta
actually ran and verified the collision. _stress_one always builds its own
search-time PerturbationSpace the same unparameterized way (see batch_scorer.py),
so these two are recorded into search_provenance for exactly this reason: each is
"a proposal until a real shard says how often this actually matters", meaning they
are expected to change over the project's life. A Pass 2 and its later Pass 3 can
straddle such a change, and the SAT-verified collision that Pass 2 earned then gets
silently replayed under a DIFFERENT heading model than the one that verified it.

Material only for NON-VEHICLE challengers — heading_speed_floor/
heading_transition_width govern the linear model's heading-observability blend;
"vehicles never reach this function: the bicycle model carries theta as real state"
(PerturbationSpace's own docstring). Every test below uses a pedestrian challenger
(_ped()'s own shape, agent 1, type 2) for that reason.

THE SUBTLETY THIS FILE TAKES CARE TO COVER: PerturbationSpace treats
heading_transition_width=None (and heading_speed_floor=None) as a MEANINGFUL
value — "the band/floor is off" — not as "use the class default"; the default only
applies when the keyword is omitted entirely. A naive fix that always passed
provenance.get('heading_transition_width') would silently force every LEGACY
result (no search_provenance at all) to replay with the band off, which is a
regression of its own. test_G08_a_recorded_none_width_is_reproduced_not_silently_
redefaulted is the one repro that can tell "key absent" and "key present with value
None" apart.

Needs a DISPOSABLE Postgres/PostGIS database and skips unless AV_CLAIMS_DB=1, same
requirement as every other DB-gated file in this project. Reuses
test_batch11_contract.py's _f05_parser SHAPE (a monkeypatched src.data.parser so
this runs through the real export_shard_geometry call site without WOMD installed)
but serves a DIFFERENT, purpose-built scene — see _bad_heading_ped's own docstring
for why _ped() itself cannot discriminate this finding.

Run:
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_g08_replay_model_identity.py -q
"""

import os
import struct
import sys
import types as _types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring.db import compute_scene_fingerprint

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

_RESULT = {'status': 'ok', 'outcome': 'collision_found', 'collision': True,
           'min_perturbation': 0.5, 'delta': [0.0, 0.0, 0.0, 0.0],
           'collision_timestep': 9, 'target_idx': 1, 'method': 'de'}


def _bad_heading_ped():
    """
    Same 2-agent shape as test_batch11_contract.py's own _ped() (agent 0 the SDC/
    vehicle, agent 1 a pedestrian challenger moving at a constant vx=-2.0, vy=0) —
    EXCEPT the challenger's LOGGED heading is 0.0 rather than pi.

    That single change is what makes this fixture able to discriminate G08 at all.
    _ped() sets the logged heading to exactly the DERIVED (velocity-direction)
    heading (both are pi), so "use the logged value" and "use the derived value"
    produce IDENTICAL output there regardless of heading_speed_floor/
    heading_transition_width — a fixture that cannot tell the two replay models
    apart proves nothing about which one export_shard_geometry actually used.
    Here they are deliberately different (0.0 vs derived pi), so which one comes
    out the other end is directly observable.
    """
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 5:7] = [0.6, 0.6]
    states[1, :, 0] = 5.0 - 2.0 * np.arange(10) * 0.1
    states[1, :, 2] = -2.0        # vx: derived heading = atan2(0, -2.0) = pi
    states[1, :, 4] = 0.0         # LOGGED heading, deliberately NOT pi
    return states, np.ones((2, 10), dtype=bool), np.array([1, 2])


def _write_geom_shard(path, scenario_ids):
    """Identical framing to test_batch11_contract.py's own helper of the same
    name — a minimal real tfrecord framing, ScenarioParser alone is stubbed."""
    from src.data.loader import _masked_crc32c

    blob = b''
    for sid in scenario_ids:
        payload = sid.encode()
        header = struct.pack('<Q', len(payload))
        blob += (header + struct.pack('<I', _masked_crc32c(header))
                 + payload + struct.pack('<I', _masked_crc32c(payload)))
    path.write_bytes(blob)
    return str(path)


@pytest.fixture
def _g08_parser(monkeypatch):
    """Same shape as test_batch11_contract.py's own _f05_parser, serving
    _bad_heading_ped() instead of _ped()."""
    module = _types.ModuleType('src.data.parser')

    class ScenarioParser:
        def __init__(self, raw):
            self.sid = raw.decode()

        def get_scenario_id(self):
            return self.sid

        def get_agent_states(self):
            return _bad_heading_ped()[0]

        def get_agent_validity(self):
            return _bad_heading_ped()[1]

        def get_agent_types(self):
            return _bad_heading_ped()[2]

        def get_sdc_index(self):
            return 0

    module.ScenarioParser = ScenarioParser
    monkeypatch.setitem(sys.modules, 'src.data.parser', module)
    yield module


@pytest.fixture
def bconn():
    from src.scoring import db
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


def _exported_headings(conn, sid):
    with conn.cursor() as cur:
        cur.execute('SELECT headings FROM perturbed_paths WHERE scenario_id = %s',
                    (sid,))
        row = cur.fetchone()
        assert row is not None, 'fixture regressed: nothing exported'
        return list(row[0])


@requires_db
def test_G08_export_replays_under_the_recorded_heading_speed_floor(
        bconn, tmp_path, _g08_parser):
    """
    THE PRIMARY REPRO — END TO END, AGAINST A REAL SEARCH, NOT A HAND-BUILT RESULT
    DICT. Independent review of the first version of this file (correctly) pointed
    out that a test built entirely from a synthetic search_provenance dict cannot
    tell "the export-side mechanism works" apart from "the export-side mechanism
    works AND the pipeline actually produces this shape of data" — and that the
    second half matters, because this is the exact gap that let G08 happen in the
    first place.

    So this test drives the REAL src.scoring.batch_scorer._stress_one — a genuine
    DE search, with its own real delta and its own real, live-captured
    search_provenance — and only afterward manufactures the DRIFT the finding is
    about: a change to PerturbationSpace's class default between the search and the
    export. _stress_one's own PerturbationSpace call (batch_scorer.py) never
    overrides heading_speed_floor/heading_transition_width, so the only way to make
    a REAL search run under a non-default floor is to change what "default" means
    at call time — done here by reassigning PerturbationSpace.__init__.__defaults__
    for the duration of the search (Python binds default argument values once, at
    function-definition time, so this is the one honest way to make an unparameterized
    call observe a different default without reloading the module). Restored
    immediately after in a try/finally, so a failure here cannot leak into any other
    test in the same process.

    Confirmed independently, live, before writing this: under
    PerturbationSpace.__init__.__defaults__ mutated so heading_speed_floor=3.0,
    _stress_one(states, validity, types, 0, de_kwargs=...) on this file's own
    _bad_heading_ped() scene returns status='ok' with
    result['search_provenance']['heading_speed_floor'] == 3.0 and
    ['heading_transition_width'] is None — both read back from the real
    PerturbationSpace instance the real search used, not typed by hand.

    THE DRIFT: by the time export_shard_geometry runs (below), the defaults are
    back to their ordinary values (0.5, 0.05) — this challenger's speed (2.0) sits
    BELOW that ordinary floor, which would derive the heading from velocity (pi).
    The search_provenance this REAL search captured says otherwise (floor=3.0,
    above 2.0 m/s, keeps the LOGGED heading (0.0) throughout) — so which one comes
    out of export_shard_geometry is the whole question this finding is about,
    answered here with a real search's own receipts.
    """
    from src.optimization.perturbation_space import PerturbationSpace
    from src.scoring import db
    from src.scoring.batch_scorer import _stress_one
    from src.scoring.export_geometry import export_shard_geometry

    states, validity, types = _bad_heading_ped()

    # The search runs under a floor ABOVE this challenger's speed — simulating "this
    # is what PerturbationSpace's class default was when Pass 2 ran".
    orig_defaults = PerturbationSpace.__init__.__defaults__
    PerturbationSpace.__init__.__defaults__ = orig_defaults[:3] + (3.0, None)
    try:
        result = _stress_one(states, validity, types, 0,
                             de_kwargs={'popsize': 4, 'maxiter': 5, 'seed': 1})
    finally:
        # By the time Pass 3 (export, below) runs, the ambient default has reverted
        # to the ordinary one — the drift this finding is about.
        PerturbationSpace.__init__.__defaults__ = orig_defaults

    assert result['status'] == 'ok', (
        f'fixture regressed: expected a real collision search to succeed, got '
        f'{result}'
    )
    provenance = result['search_provenance']
    assert provenance['heading_speed_floor'] == 3.0, (
        f'fixture regressed: the real search did not capture the mutated default '
        f'— got {provenance}'
    )
    assert provenance['heading_transition_width'] is None, provenance

    fp = compute_scene_fingerprint(states, validity, types)
    result['scene_fingerprint'] = fp
    result['sdc_idx'] = 0
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2,
                                  min_ttc=9.0, min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp, sdc_idx=0)])
    db.update_stress_results(bconn, {'s': dict(result)})

    shard = _write_geom_shard(tmp_path / 'shard.tfrecord', ['s'])
    summary = export_shard_geometry(bconn, shard, ['s'],
                                    stress_results={'s': dict(result)}, verbose=False)
    assert summary['perturbed_written'] == 1, summary

    headings = _exported_headings(bconn, 's')
    assert all(abs(h - 0.0) < 1e-4 for h in headings), (
        f'expected the LOGGED heading (0.0), recorded as verified under a REAL '
        f'search\'s own heading_speed_floor=3.0 — got {headings}, which is the '
        f'now-ambient class-default (0.5) replay instead'
    )


@requires_db
def test_G08_a_legacy_result_with_no_search_provenance_still_uses_the_class_default(
        bconn, tmp_path, _g08_parser):
    """
    THE OVERCORRECTION GUARD, AND THE NO-REGRESSION CASE. _RESULT itself (no
    search_provenance key, the shape every OTHER test file in this project already
    uses for it) must replay exactly as it always did — the class default
    (heading_speed_floor=0.5), which for this challenger's speed (2.0) derives the
    heading from velocity (pi).
    """
    from src.scoring import db
    from src.scoring.export_geometry import export_shard_geometry

    states, validity, types = _bad_heading_ped()
    fp = compute_scene_fingerprint(states, validity, types)
    assert 'search_provenance' not in _RESULT, 'fixture regressed'
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2,
                                  min_ttc=9.0, min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp, sdc_idx=0)])
    db.update_stress_results(bconn, {'s': dict(_RESULT, scene_fingerprint=fp,
                                               sdc_idx=0)})

    shard = _write_geom_shard(tmp_path / 'shard.tfrecord', ['s'])
    summary = export_shard_geometry(
        bconn, shard, ['s'],
        stress_results={'s': dict(_RESULT, scene_fingerprint=fp, sdc_idx=0)},
        verbose=False,
    )
    assert summary['perturbed_written'] == 1, summary

    headings = _exported_headings(bconn, 's')
    assert all(abs(h - np.pi) < 1e-4 for h in headings), (
        f'a legacy result with no search_provenance was NOT replayed under the '
        f'class default — got {headings}, expected the derived heading (pi)'
    )


@requires_db
def test_G08_a_recorded_none_width_is_reproduced_not_silently_redefaulted(
        bconn, tmp_path, _g08_parser):
    """
    THE SUBTLETY. heading_transition_width=None IN search_provenance is a
    MEANINGFUL recorded value (the search ran with the band off, a step function
    at exactly heading_speed_floor) — not "not recorded". A fix that read this key
    with .get(..., default) and treated a None value the same as a missing key
    would silently swap in the class default width (0.05) instead of reproducing
    "off" — observable here because the challenger's speed (2.0) sits EXACTLY at
    the recorded floor (2.0): a pure step function (width off) derives the heading
    (pi) at speed >= floor, while smoothstep's own value at its lower bound is 0
    (pure logged, 0.0) — so which one comes out distinguishes "None reproduced" from
    "None silently redefaulted".
    """
    from src.scoring import db
    from src.scoring.export_geometry import export_shard_geometry

    states, validity, types = _bad_heading_ped()
    fp = compute_scene_fingerprint(states, validity, types)
    result = dict(_RESULT, scene_fingerprint=fp, sdc_idx=0,
                  search_provenance={'heading_speed_floor': 2.0,
                                     'heading_transition_width': None})
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2,
                                  min_ttc=9.0, min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp, sdc_idx=0)])
    db.update_stress_results(bconn, {'s': dict(result)})

    shard = _write_geom_shard(tmp_path / 'shard.tfrecord', ['s'])
    summary = export_shard_geometry(bconn, shard, ['s'],
                                    stress_results={'s': dict(result)}, verbose=False)
    assert summary['perturbed_written'] == 1, summary

    headings = _exported_headings(bconn, 's')
    assert all(abs(h - np.pi) < 1e-4 for h in headings), (
        f'a recorded heading_transition_width=None was silently redefaulted to the '
        f'class default band width instead of being reproduced as "off" — got '
        f'{headings}, expected the pure-step-function derived heading (pi)'
    )
