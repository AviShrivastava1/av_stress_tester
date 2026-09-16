"""
test_audit2_regressions.py — second audit: R01–R12 except R09's siblings, i.e.
R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R11, R12. The third audit's A08,
A09 and A11 are the same findings as R11, R08 and R12 and are covered here rather
than duplicated in test_audit3_regressions.py, which holds A-only findings.

PROVENANCE, STATED PLAINLY: these tests are RECONSTRUCTED FROM THE SECOND AUDIT'S
DESCRIPTIONS, NOT COPIED FROM ITS OWN CODE. The audit's source archive is on disk
(sha256 f308e4b3e1482c9d1d6a65e6d91e0bb5b5d68978e1441dadc4da472a0610f024, which is
commit 8e6cfb7), but its test bundle is not, so the fixtures below were written from
the repro each finding describes. They carry the audit's test names because those are
what the finding is tracked by, and that is the only thing they share with it.

What makes them trustworthy is the same thing that made Batch 2's B05 substitutes
trustworthy: each was run against the UNFIXED code first and shown to reproduce the
audit's SPECIFIC claim, not merely to fail somehow. Recorded here so the claim can be
re-checked rather than taken on faith:

    R02  score_shard over one good record + 3 garbage bytes
           -> ValueError escaped the function; both scored records discarded
         stress_test_scenarios(shard, ['A']) with A as record 0
           -> same ValueError; a one-scenario request still read the corrupt tail
    R09  notebook cell 50, sample of three results with zero collisions
           -> StopIteration at the bare next(), line 8 of the cell
         and with cell 50 alone patched, cell 51 raised StopIteration in turn,
           because target_idx is None and no agent matches it
    R04  optimize_scenario on the two-agent scene, challenger at y = 2.6,
         popsize=6, maxiter=25, seed=5
           -> 624 evaluations, 3 of them exact-verified as colliding, best of those
              at weighted norm 1.687680960; the function returned collision=False,
              min_perturbation=inf at norm 1.652127862 with margin +1.4506e-05
    R05  refine_scenario with bounds [[0,0], [-0.01, 0.01], [0,0], [0,0]] and
         delta_init [0, 0.15, 0, 0]
           -> returned collision=True, delta [0, 0.15000000596046448, 0, 0],
              min_perturbation 15.000000953674316 — a heading offset 15x its
              allowed maximum, reported as a verified answer

    R01  persist result A, persist newer result B, export using A's result dict
           -> A's geometry stamped with B's run id (23eb305e65629e43), so
              GET /perturbed served B's delta [-2,0,0,0] beside A's trajectory:
              path point 9 = 2.299999714 where B's replay is 1.399999380. The B14
              join passed, because both sides genuinely held the same id
    R03  result A with A's geometry, both content-derived and matching; a commit
         landing between get_perturbed's two statements
           -> delta [-1,0,0,0] and min_perturbation 0.5 from result A, served beside
              a path whose point 9 is 0.5 — result B's trajectory
    R06  scenario_scores had no target_idx column at all, so a stress-tested,
         not-yet-exported scenario answered target_idx=None
    R07  after a replay_infeasible attempt, the persisted row held
         last_attempt_outcome and stress_attempted_at and nothing else; reason
         ('collision') and baseline_replay_error were both gone. _stress_one never
         carried baseline_replay_collides out of the exception in the first place

THE AUDIT'S OWN R04 NUMBERS (176 evaluations, 126 colliding, best colliding norm
0.339553833, returned 0.262787908) ARE NOT REPRODUCED HERE and no attempt is made to
pretend otherwise: they belong to a fixture that does not exist on this machine. What
is reproduced is the mechanism, on a fixture of this file's own, with its own numbers
recorded above. The general form was derived independently and predicts the specific
failure: an infeasible point wins whenever its margin satisfies
0 < g < (norm_feasible^2 - norm_infeasible^2) / LAMBDA, and here that window is
(1.687680960^2 - 1.652127862^2) / 1e3 = 1.1874e-04, inside which the measured
g = 1.4506e-05 falls.

Run:
    ./venv/bin/python -m pytest tests/test_audit2_regressions.py -q
"""

import base64
import contextlib
import io
import json
import os
import struct
import subprocess
import sys
import types
import unicodedata

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import _masked_crc32c

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── shard fixtures ──────────────────────────────────────────────────────────────

def _write_shard(path, payloads, tail=b''):
    """
    A real tfrecord file. `tail` is appended raw — the audit's repro uses 3 garbage
    bytes, which is a length header that stops after 3 of its 8 bytes.

    Real framing rather than a stubbed loader, deliberately: R02 is about an exception
    raised INSIDE ShardLoader.__iter__ during the `for` statement's own next() call,
    and a stub that yields from a list cannot reproduce that at all.
    """
    blob = b''
    for payload in payloads:
        header = struct.pack('<Q', len(payload))
        blob += (header + struct.pack('<I', _masked_crc32c(header))
                 + payload + struct.pack('<I', _masked_crc32c(payload)))
    path.write_bytes(blob + tail)
    return str(path)


@pytest.fixture
def stub_parser(monkeypatch):
    """
    Substitute src.data.parser, the technique Batch 2's B05 substitutes established:
    parser.py imports scenario_pb2 at module scope and the Waymo wheels are
    manylinux-only. The payload's CONTENT is irrelevant to R02 — the failure is in
    framing, which the loader handles before any parsing happens.
    """
    module = types.ModuleType('src.data.parser')

    class ScenarioParser:
        def __init__(self, raw):
            self.raw = raw

        def get_scenario_id(self):
            return self.raw.decode()

        def get_agent_states(self):
            states = np.zeros((2, 10, 7), dtype=np.float32)
            states[:, :, 5:7] = [4.5, 2.0]
            t = np.arange(10) * 0.1
            states[0, :, 0] = -20.0 + 10.0 * t
            states[0, :, 2] = 10.0
            states[1, :, 0] = 30.0
            return states

        def get_agent_validity(self):
            return np.ones((2, 10), dtype=bool)

        def get_agent_types(self):
            return np.ones(2, dtype=int)

        def get_sdc_index(self):
            return 0

    module.ScenarioParser = ScenarioParser
    monkeypatch.setitem(sys.modules, 'src.data.parser', module)
    return module


# ── R02: a truncated tail must not destroy completed work ───────────────────────

def test_R02_score_batch_keeps_completed_results_on_truncated_tail(tmp_path, stub_parser):
    """
    Block 5 Concept 19: one bad record must never kill the batch. Batch 5's B11 fix
    made ShardLoader correctly refuse truncated framing — but it raises inside
    __iter__, which the `for` statement calls BEFORE the loop body, so the raise was
    structurally outside the per-record handler and every scored record was lost.
    """
    from src.scoring.batch_scorer import score_shard

    path = _write_shard(tmp_path / 'trunc.tfrecord', [b'A', b'B'], tail=b'abc')
    records, errors = score_shard(path, verbose=False)

    assert len(records) == 2, (
        f'completed work was discarded by the truncated tail: {len(records)} records'
    )
    assert [r['scenario_id'] for r in records] == ['A', 'B']

    fatal = [e for e in errors if e.get('kind') == 'shard_truncated']
    assert len(fatal) == 1, f'the truncation was not recorded distinctly: {errors}'
    assert fatal[0]['scenario_id'] is None
    assert fatal[0]['index'] == 2


def test_R02_a_clean_shard_records_no_fatal_error(tmp_path, stub_parser):
    """
    Beyond the audit, and the easiest thing to get wrong in this restructure:
    StopIteration subclasses Exception, so a handler that does not catch it FIRST and
    separately would log every successful run as a fatal truncation.
    """
    from src.scoring.batch_scorer import score_shard

    path = _write_shard(tmp_path / 'clean.tfrecord', [b'A', b'B'])
    records, errors = score_shard(path, verbose=False)

    assert len(records) == 2
    assert errors == [], f'a clean end of file was reported as an error: {errors}'


@pytest.mark.parametrize('payloads,requested,reads_tail', [
    ([b'A'], ['A'], False),              # the audit's own shape: A, then garbage
    ([b'A', b'B'], ['A', 'B'], False),   # last match is the last record
    ([b'A', b'B'], ['A', 'ZZZ'], True),  # ZZZ is never found, so the shard is read out
], ids=['audit_repro', 'both_present', 'one_missing'])
def test_R02_completed_subset_does_not_read_a_corrupt_next_record(
        tmp_path, stub_parser, payloads, requested, reads_tail):
    """
    The completion check has to run BEFORE the fetch. Asking for scenarios that are
    all found early used to read one record past the last match, so a targeted run
    touched shard data it was never asked for — and died on corruption it had no
    business reaching.

    THE SHARD SHAPE IS PARAMETRIZED TOO, and that is not cosmetic. An earlier draft
    used [A, B] + garbage for every case, which left the single-id case unable to
    catch the defect at all: with the check misplaced, requesting ['A'] reads one past
    A, lands on B, and stops — never reaching the tail at record 2, so the test passed
    either way. Mutation-testing surfaced that. The first case now matches the audit's
    described repro exactly — one record, then 3 garbage bytes — where reading a
    single record too far is precisely what hits the corruption.

    The third case is the control: when a requested id genuinely is not in the shard,
    reading to the end IS correct, and the truncation must then be recorded rather
    than swallowed.
    """
    from src.scoring.batch_scorer import stress_test_scenarios

    path = _write_shard(tmp_path / 'trunc.tfrecord', payloads, tail=b'abc')
    results = stress_test_scenarios(path, requested, verbose=False)

    fatal = [e for e in results.errors if e.get('kind') == 'shard_truncated']
    assert bool(fatal) == reads_tail, (
        f'requested {requested}: expected reads_tail={reads_tail}, '
        f'got errors={results.errors}'
    )
    assert 'A' in results, 'the scenario that WAS found must survive either way'


def test_R02_export_does_not_fetch_beyond_completed_subset(tmp_path, stub_parser, monkeypatch):
    """
    export_shard_geometry has the identical loop shape and needs the identical fix.
    Exercised without a database by stubbing the two writers — the defect under test
    is in the read loop, not in what it writes.
    """
    from src.scoring import export_geometry

    monkeypatch.setattr(export_geometry, 'export_scenario_agents',
                        lambda *a, **k: (2, 0))
    monkeypatch.setattr(export_geometry, 'export_perturbed_path',
                        lambda *a, **k: None)

    path = _write_shard(tmp_path / 'trunc.tfrecord', [b'A', b'B'], tail=b'abc')
    summary = export_geometry.export_shard_geometry(None, path, ['A'], verbose=False)

    assert summary['exported'] == 1
    fatal = [e for e in summary['errors'] if e.get('kind') == 'shard_truncated']
    assert not fatal, (
        f'the export read past the only scenario it was asked for: {summary["errors"]}'
    )


def test_R02_the_truncation_tag_is_additive(tmp_path, stub_parser):
    """
    `kind` is a new key on an existing dict shape, not a replacement for it. Every
    current consumer reads 'index'/'record_index', 'scenario_id' and 'error' by name
    or prints the dict whole, so the older keys must all still be present — a
    consumer that never looks at 'kind' must not notice it exists.

    A string rather than a boolean because this project already made that choice once
    and should stay consistent with itself: Batch 2 built stress_outcome and
    last_attempt_outcome as closed string vocabularies precisely so the taxonomy could
    grow without a redesign. `fatal: True` says the run stopped short; it cannot say
    why, and the next terminal condition would either be jammed into the same
    undifferentiated flag or need a second boolean beside it.
    """
    from src.scoring.batch_scorer import score_shard

    path = _write_shard(tmp_path / 'trunc.tfrecord', [b'A'], tail=b'abc')
    _, errors = score_shard(path, verbose=False)

    entry = errors[0]
    assert set(entry) == {'index', 'scenario_id', 'error', 'kind'}
    assert isinstance(entry['kind'], str)
    assert entry['kind'] == 'shard_truncated'


# ── R09: the notebook must survive a sample with no collision ───────────────────

def _notebook_cells():
    path = os.path.join(PROJECT, 'notebooks', 'colab_validation_run.ipynb')
    return json.load(open(path))['cells']


def _cell_by_content(*required):
    """
    Located by CONTENT, never by index. The audit cites cells 46/47; the Colab prep
    commit inserted four cells ahead of them and they are now 50/51. This is the
    fourth time an index citation in this project has needed re-deriving, which is why
    COLAB_RUNBOOK.md says section headings are the stable reference.
    """
    hits = [''.join(c['source']) for c in _notebook_cells()
            if c['cell_type'] == 'code'
            and all(token in ''.join(c['source']) for token in required)]
    assert len(hits) == 1, f'expected exactly one cell matching {required}, got {len(hits)}'
    return hits[0]


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _fake_httpx(agents):
    def get(url, params=None):
        if url.endswith('/health'):
            return _Resp({'status': 'ok', 'database': 'connected'})
        if url.endswith('/stats'):
            return _Resp({'total_scenarios': 3, 'collisions_found': 0})
        if url.endswith('/scenarios'):
            return _Resp({'items': [], 'limit': 3})
        if 'trajectories' in url:
            return _Resp({'agents': agents})
        if 'perturbed' in url:
            return _Resp({'delta': None, 'collision_timestep': None,
                          'target_idx': None, 'min_perturbation': None})
        return _Resp({})
    return types.SimpleNamespace(get=get)


def test_R09_notebook_api_checks_work_when_no_collision_was_found():
    """
    A Pass 2 sample with no verified collision is an ORDINARY outcome — Batch 1 made
    replay_infeasible routine, and a completed search finding nothing is just a search
    that worked. With TOP_N = 5 it is entirely likely. The bare next() turned it into
    StopIteration after Pass 1, Pass 2 and Pass 3 had already run.

    Both cells are exercised, because fixing only the first moves the crash to the
    second: it reads pert['target_idx'], which is None when no perturbed path exists.
    """
    agents = [{'agent_idx': 0, 'timesteps': [0.0, 1.0, 2.0]}]
    stress_results = {
        'syn_a': {'status': 'ok', 'collision': False, 'outcome': 'no_collision_found'},
        'syn_b': {'status': 'replay_infeasible', 'outcome': 'replay_infeasible'},
        'syn_c': {'status': 'ok', 'collision': False, 'outcome': 'no_collision_found'},
    }
    assert not any(r.get('collision') for r in stress_results.values())

    env = {
        'httpx': _fake_httpx(agents), 'base': 'http://x', 'np': np,
        'stress_results': stress_results, 'ids_to_test': list(stress_results),
        'dump_points_m': lambda *a, **k: (None, 3, None, [0.0, 1.0, 2.0]),
        'conn': object(), 'server': types.SimpleNamespace(should_exit=False),
    }

    api_cell = _cell_by_content('stress_tested_sid', 'GET /health')
    roundtrip_cell = _cell_by_content('http_timesteps')

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(api_cell, 'api_cell', 'exec'), env)
        exec(compile(roundtrip_cell, 'roundtrip_cell', 'exec'), env)
    printed = out.getvalue()

    assert env['COLLISION_AVAILABLE'] is False
    assert env['stress_tested_sid'] in stress_results
    # The round trip is the notebook's most important assertion and must still run.
    assert 'CONFIRMED: HTTP timesteps == Postgres M values' in printed
    # And the absence must be stated, not implied away.
    assert 'COLLISION-SPECIFIC CHECKS ARE SKIPPED' in printed
    assert env['server'].should_exit is True


def test_R09_a_collision_still_takes_the_normal_path():
    """The fix must not change behaviour when a collision IS present."""
    agents = [{'agent_idx': 0, 'timesteps': [0.0, 1.0]},
              {'agent_idx': 1, 'timesteps': [0.0, 1.0]}]

    def get(url, params=None):
        if 'trajectories' in url:
            return _Resp({'agents': agents})
        if 'perturbed' in url:
            return _Resp({'delta': [0.1, 0, 0, 0], 'collision_timestep': 4,
                          'target_idx': 1})
        return _Resp({'status': 'ok'})

    env = {
        'httpx': types.SimpleNamespace(get=get), 'base': 'http://x', 'np': np,
        'stress_results': {'syn_hit': {'status': 'ok', 'collision': True}},
        'ids_to_test': ['syn_hit'],
        'dump_points_m': lambda *a, **k: (None, 2, None, [0.0, 1.0]),
        'conn': object(), 'server': types.SimpleNamespace(should_exit=False),
    }
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(_cell_by_content('stress_tested_sid', 'GET /health'), 'c', 'exec'), env)
        exec(compile(_cell_by_content('http_timesteps'), 'c', 'exec'), env)

    assert env['COLLISION_AVAILABLE'] is True
    assert env['stress_tested_sid'] == 'syn_hit'
    assert env['target_idx_api'] == 1, 'the perturbed target must still be used'
    assert 'COLLISION-SPECIFIC CHECKS ARE SKIPPED' not in out.getvalue()


# ── R04: DE must not discard a collision it already evaluated ───────────────────

def _optimizer_scene(lateral):
    """Two agents, the challenger offset laterally. Bare numpy, no shard."""
    from src.optimization.perturbation_space import PerturbationSpace
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[1, :, 1] = lateral
    validity = np.ones((2, 10), dtype=bool)
    return PerturbationSpace(states, validity, np.ones(2, dtype=int), 0, 1), validity


def test_R04_de_returns_a_collision_if_it_evaluated_one():
    """
    The audit's claim, on this file's own fixture: a search that evaluated a verified
    collision must not report that it found none.

    Pre-fix on this scene (recorded in the module docstring): 3 of 624 evaluations
    exact-verified as colliding, and optimize_scenario returned collision=False /
    min_perturbation=inf, because its winner cleared the challenger by 14.5
    micrometres and the penalty charged only 0.0145 for that against a norm^2
    advantage of 0.119.

    The assertion is written against what the search SAW, discovered at run time by
    replaying every evaluation through the exact check, rather than against the
    literals above — those are this machine's numbers, and DE's arithmetic is not
    bit-portable (the lesson from Batch 5's own review of this project's DE tests).
    """
    from src.danger.collision_detector import check_collision_trajectory
    from src.optimization.scipy_optimizer import _DEObjective, optimize_scenario

    space, validity = _optimizer_scene(2.6)

    seen = []
    pristine = _DEObjective.__call__

    def recording(self, delta):
        value = pristine(self, delta)
        candidate = np.asarray(delta, np.float32)
        hit, _ = check_collision_trajectory(space.apply(candidate), validity, 0, 1)
        if hit:
            seen.append(space.weighted_norm(candidate))
        return value

    _DEObjective.__call__ = recording
    try:
        result = optimize_scenario(space, popsize=6, maxiter=25, seed=5)
    finally:
        _DEObjective.__call__ = pristine

    if not seen:
        pytest.skip('no colliding candidate was evaluated on this machine; '
                    'the finding cannot be exercised by this fixture here')

    assert result['collision'], (
        f'the search evaluated {len(seen)} verified collisions (best weighted norm '
        f'{min(seen)}) and still reported none'
    )
    assert result['min_perturbation'] <= min(seen) + 1e-9, (
        f"returned {result['min_perturbation']} when it had already evaluated "
        f'{min(seen)}'
    )
    # and the answer is the exact one, not the archive taken on trust
    hit, t_hit = check_collision_trajectory(space.apply(result['delta']), validity, 0, 1)
    assert hit and t_hit == result['collision_timestep']


# ── R05: a retained warm start must be inside the box ───────────────────────────

def test_R05_refiner_never_returns_an_out_of_bounds_warm_start():
    """
    The audit's exact repro: bounds of +/-0.01 on heading, a warm start of 0.15.

    STATED AS THE INVARIANT, NOT AS THE MECHANISM. Refusing and clamping are both
    defensible fixes and the audit's own bundle is not here to say which it asserted,
    so this accepts either: raise, or return something inside the box. Pre-fix the
    function did neither — it returned the 15x-over warm start as a verified answer
    with min_perturbation 15.000000953674316 — so this reproduces the finding under
    either reading. Which fix was chosen is pinned separately, in
    tests/test_batch6_contract.py.
    """
    from src.optimization.autograd_optimizer import refine_scenario
    from src.optimization.perturbation_space import PerturbationSpace
    from src.danger.collision_detector import check_collision_trajectory

    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[1, :, 1] = 2.1
    validity = np.ones((2, 10), dtype=bool)
    bounds = np.array([[0, 0], [-0.01, 0.01], [0, 0], [0, 0]], dtype=np.float32)
    space = PerturbationSpace(states, validity, np.ones(2, dtype=int), 0, 1,
                              bounds=bounds)

    warm = np.array([0.0, 0.15, 0.0, 0.0], dtype=np.float32)
    assert check_collision_trajectory(space.apply(warm), validity, 0, 1)[0], (
        'fixture regressed: the out-of-bounds warm start must itself collide, or the '
        'finding is not being exercised'
    )
    assert warm[1] > space.bounds[1, 1], 'fixture regressed: the warm start is legal'

    try:
        result = refine_scenario(space, delta_init=warm, n_iters=50)
    except ValueError:
        return  # refused outright — one of the two acceptable answers

    delta = np.asarray(result['delta'], np.float32)
    assert np.all(delta >= space.bounds[:, 0]) and np.all(delta <= space.bounds[:, 1]), (
        f'returned {delta.tolist()} for bounds {space.bounds.tolist()} — a delta '
        'outside the search space it claims to have searched'
    )


# ── R01 / R03 / R06 / R07: result-and-geometry consistency ─────────────────────
#
# DB-backed, gated on the same AV_CLAIMS_DB flag the rest of the results layer uses
# rather than a new environment variable — these are claims-layer tests and want the
# same explicitly-nominated disposable database.

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)


def _r7_scene():
    """SDC plus a small challenger closing head-on. Two agents, ten frames."""
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 5:7] = [0.6, 0.6]
    states[1, :, 0] = 5.0 - 2.0 * np.arange(10) * 0.1
    states[1, :, 2] = -2.0
    states[1, :, 4] = np.pi
    return states, np.ones((2, 10), dtype=bool), np.array([1, 2])


@pytest.fixture
def conn7():
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


def _seed7(conn, sid='scene'):
    from src.scoring import db
    db.upsert_scores(conn, [dict(scenario_id=sid, shard='synthetic', n_agents=2,
                                 min_ttc=9.0, min_pet=9.0, fragility_score=1.0)])


_RESULT_A = {'status': 'ok', 'outcome': 'collision_found', 'collision': True,
             'min_perturbation': 0.5, 'delta': [-1.0, 0.0, 0.0, 0.0],
             'collision_timestep': 3, 'target_idx': 1, 'method': 'de'}
_RESULT_B = dict(_RESULT_A, delta=[-2.0, 0.0, 0.0, 0.0], min_perturbation=1.0)


@requires_db
def test_R01_a_stale_export_cannot_borrow_the_current_run_id(conn7):
    """
    The audit's repro: persist A, persist a newer B, then export using A's dict.

    Pre-fix the export was stamped with B's id and the API served B's delta beside
    A's trajectory — 0.9 m apart at the last frame — with the B14 join passing,
    because the ids matched. They were simply attached to different content.

    The assertion is the INVARIANT: whatever geometry is served must have been built
    from the delta that is served beside it. Refusing the write and serving no
    picture satisfies that; so would any other correct answer.
    """
    from src.scoring import db
    from src.scoring.export_geometry import (
        export_perturbed_path, export_scenario_agents,
    )
    from src.optimization.perturbation_space import PerturbationSpace

    states, validity, types = _r7_scene()
    _seed7(conn7)
    export_scenario_agents(conn7, 'scene', states, validity, types, 0)
    space = PerturbationSpace(states, validity, types, 0, 1)

    db.update_stress_results(conn7, {'scene': _RESULT_A})
    id_a = db.fetch_scenario(conn7, 'scene')['stress_run_id']
    db.update_stress_results(conn7, {'scene': _RESULT_B})
    id_b = db.fetch_scenario(conn7, 'scene')['stress_run_id']
    assert id_a != id_b, 'fixture regressed: the two results must be different runs'

    stale = space.apply(np.float32(_RESULT_A['delta']))
    try:
        # The content kwargs are what make the id derive from THIS trajectory. Against
        # the pre-R01 signature they do not exist, and the fallback below takes the
        # old path deliberately — otherwise this test would fail pre-fix with a
        # TypeError about a keyword argument, which demonstrates nothing about the
        # finding. With the fallback it fails pre-fix on the ASSERTION, naming the run
        # the geometry was actually published under.
        export_perturbed_path(conn7, 'scene', stale, validity, 1,
                              delta=_RESULT_A['delta'], method=_RESULT_A['method'])
    except TypeError:
        export_perturbed_path(conn7, 'scene', stale, validity, 1)
    except Exception as e:                       # refused outright — one right answer
        assert id_a in str(e) and id_b in str(e), (
            f'the refusal must name both the refused run and the stored one: {e}'
        )
    else:
        conn7.rollback()

    with conn7.cursor() as cur:
        cur.execute("SELECT stress_run_id FROM perturbed_paths WHERE scenario_id='scene'")
        row = cur.fetchone()
    assert row is None or row[0] == id_a, (
        f"A's geometry was published under run {row[0]}, which is not the run it was "
        f'built from ({id_a})'
    )


@requires_db
def test_R03_the_two_reads_are_one_snapshot(conn7, monkeypatch):
    """
    Correct writes throughout — both exports content-derived and matching — read
    inconsistently.

    A commit lands between get_perturbed's two statements. `_table_exists` runs
    exactly there, which makes the interleaving deterministic: no sleeps, no timing.

    NOTE THE FIXTURE STAMPS BOTH EXPORTS FROM CONTENT, i.e. R01 is already fixed
    here. That is deliberate: it is what makes this a test of R03 rather than of R01
    leaking into it.
    """
    from fastapi.testclient import TestClient
    from src.api import routes
    from src.api.main import app
    from src.scoring import db
    from src.scoring.export_geometry import (
        export_perturbed_path, export_scenario_agents,
    )
    from src.optimization.perturbation_space import PerturbationSpace

    states, validity, types = _r7_scene()
    _seed7(conn7)
    export_scenario_agents(conn7, 'scene', states, validity, types, 0)
    space = PerturbationSpace(states, validity, types, 0, 1)

    db.update_stress_results(conn7, {'scene': _RESULT_A})
    export_perturbed_path(conn7, 'scene', space.apply(np.float32(_RESULT_A['delta'])),
                          validity, 1, delta=_RESULT_A['delta'], method='de')

    real = routes._table_exists
    fired = []

    def interfering(cur, name):
        if name == 'perturbed_paths' and not fired:
            fired.append(True)
            other = db.get_connection()
            db.update_stress_results(other, {'scene': _RESULT_B})
            export_perturbed_path(
                other, 'scene', space.apply(np.float32(_RESULT_B['delta'])),
                validity, 1, delta=_RESULT_B['delta'], method='de')
            other.close()
        return real(cur, name)

    monkeypatch.setattr(routes, '_table_exists', interfering)
    with TestClient(app) as client:
        data = client.get('/scenarios/scene/perturbed').json()
    assert fired, 'fixture regressed: the interfering commit never ran'

    if data['perturbed'] is None:
        return                      # a result with no matching picture is consistent

    served_last_x = data['perturbed']['path'][-1][0]
    replay = space.apply(np.float32(data['delta']))
    assert served_last_x == pytest.approx(float(replay[1, 9, 0]), abs=1e-3), (
        f"served delta {data['delta']} against a path ending at x={served_last_x}, "
        f'which that delta replays to x={float(replay[1, 9, 0])} — the response '
        'describes two different runs'
    )


@requires_db
def test_R06_the_searched_challenger_survives_without_geometry(conn7):
    """A stress-tested, not-yet-exported scenario must still say which agent."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.scoring import db

    _seed7(conn7)
    db.update_stress_results(conn7, {'scene': _RESULT_A})

    assert db.fetch_scenario(conn7, 'scene')['target_idx'] == 1, (
        'target_idx is not recoverable from the score row'
    )
    with TestClient(app) as client:
        data = client.get('/scenarios/scene/perturbed').json()
    assert data['perturbed'] is None, 'fixture regressed: no geometry should exist'
    assert data['target_idx'] == 1


@requires_db
def test_R07_a_refusal_keeps_its_diagnostics(conn7):
    """
    reason, baseline_replay_error and baseline_replay_collides all survive
    persistence. Pre-fix the row held last_attempt_outcome and a timestamp, and a
    collision-refusal was indistinguishable from a drift-refusal.
    """
    from src.scoring import db

    _seed7(conn7)
    db.update_stress_results(conn7, {'scene': {
        'status': 'replay_infeasible', 'outcome': 'replay_infeasible',
        'target_idx': 1, 'baseline_replay_error': 3.9, 'reason': 'drift',
        'baseline_replay_collides': False,
        'challengers_total': 2, 'challengers_searched': 0}})

    diagnostics = db.fetch_scenario(conn7, 'scene')['last_attempt_diagnostics']
    assert diagnostics is not None, 'the refusal recorded nothing'
    assert diagnostics['reason'] == 'drift'
    assert diagnostics['baseline_replay_error'] == 3.9
    assert diagnostics['baseline_replay_collides'] is False
    assert diagnostics['target_idx'] == 1


@requires_db
def test_R07_stress_one_carries_every_field_the_exception_holds():
    """
    The half of R07 that happens before the database: _stress_one dropped
    baseline_replay_collides on the floor, so no schema could have persisted it.
    """
    from src.scoring.batch_scorer import _stress_one

    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    result = _stress_one(states, np.ones((2, 10), dtype=bool), np.ones(2, dtype=int),
                         0, de_kwargs={'popsize': 4, 'maxiter': 5, 'seed': 1})

    assert result['status'] == 'replay_infeasible', 'fixture regressed'
    assert result['reason'] == 'collision'
    assert result['baseline_replay_error'] == 0.0
    assert result['baseline_replay_collides'] is True, (
        'the refusal dict still drops a field the exception carried'
    )


# ── R12 / A11: pytest must collect without an environment variable ──────────────
#
# Raised twice: R12 in the second audit, A11 in the third. One finding.
#
# Reconstructed from the description, and the description understates it. The claim
# is "collection requires AV_AUDIT_PROJECT"; what actually happens is that the
# KeyError fires at IMPORT time in two files, pytest reports `2 errors during
# collection`, and INTERRUPTS the entire session. Measured on the unfixed tree:
#
#     $ env -u AV_AUDIT_PROJECT pytest tests/ --collect-only -q
#     ERROR tests/test_audit_core.py - KeyError: 'AV_AUDIT_PROJECT'
#     ERROR tests/test_audit_db.py - KeyError: 'AV_AUDIT_PROJECT'
#     !!!!!! Interrupted: 2 errors during collection !!!!!!
#     228 tests collected, 2 errors in 1.20s
#
# 228 tests collected and zero of them runnable. Not "two files skip".
#
# This file has derived PROJECT from __file__ since it was written (line 80 above),
# as does every other test module here; the two audit files are the outliers.

def _pytest_run(args, drop=(), timeout=300):
    """
    pytest in a subprocess with specific variables removed from the environment.

    A subprocess rather than monkeypatch because the defect is at MODULE IMPORT of a
    file this session has already imported — os.environ cannot be un-read. Same
    technique tests/test_claims_contract.py uses for PYTHONHASHSEED determinism.
    """
    env = {k: v for k, v in os.environ.items() if k not in drop}
    return subprocess.run(
        [sys.executable, '-m', 'pytest', *args, '-p', 'no:cacheprovider'],
        cwd=PROJECT, env=env, capture_output=True, text=True, timeout=timeout,
    )


def test_R12_the_suite_collects_without_the_audit_project_variable():
    """
    Collection is a path question, and the path is knowable from __file__.

    --collect-only, so this does not recursively run the suite (it collects itself,
    which is fine — collection does not execute test bodies).
    """
    proc = _pytest_run(['tests/', '--collect-only', '-q'], drop=('AV_AUDIT_PROJECT',))

    assert 'AV_AUDIT_PROJECT' not in proc.stdout + proc.stderr, (
        'collection still fails on the missing variable:\n'
        + (proc.stdout + proc.stderr)[-2000:]
    )
    assert proc.returncode == 0, (
        f'collection exited {proc.returncode}:\n' + (proc.stdout + proc.stderr)[-2000:]
    )


@pytest.mark.parametrize('flag,path', [
    ('AV_CLAIMS_DB', 'tests/test_claims_contract.py'),
    ('AV_ROBUSTNESS_DB', 'tests/test_api_robustness.py'),
    ('AV_AUDIT_DB', 'tests/test_audit_db.py'),
])
def test_R12_the_disposable_database_opt_in_is_still_required(flag, path):
    """
    THE OTHER HALF, AND IT IS THE HALF WITH TEETH.

    AV_AUDIT_PROJECT gated nothing — it demanded a path the file already knows.
    These three gate something DESTRUCTIVE: each of those suites drops and recreates
    the project's tables in whatever PGDATABASE points at. The explicit opt-in IS the
    safety property, and deriving a default for one of them would be a way to lose a
    real database.

    So this test exists to fail if R12's fix is over-applied. Without the flag the
    gated tests must SKIP — not run, and not error.

    ASSERTED ON THE SKIP REASON, NOT ON THE ABSENCE OF PASSES. An earlier version of
    this test asserted `' passed' not in stdout`, which is wrong and was caught by
    running it: test_claims_contract.py holds 43 tests of which only 20 are gated, so
    23 pure ones pass without any database and always should. The question is not
    whether anything ran, it is whether the GATE FIRED — so the assertion reads the
    -rs report for a skip attributed to the disposable-database marker. Derive the
    flag away and that line disappears, whatever else the file does.
    """
    proc = _pytest_run([path, '-q', '-rs'], drop=(flag,))

    assert proc.returncode == 0, (
        f'{path} without {flag} did not exit clean:\n'
        + (proc.stdout + proc.stderr)[-2000:]
    )
    gated = [line for line in proc.stdout.splitlines()
             if line.startswith('SKIPPED') and 'disposable' in line]
    assert gated, (
        f'{path} reported no disposable-database skip without {flag} — the opt-in '
        f'has been derived away:\n' + proc.stdout[-2000:]
    )


# ── R08 / A09: nothing reports a safety conclusion the pass did not reach ───────
#
# One finding, raised twice, and each audit found it at a DIFFERENT set of lines —
# which is why Batch 10 answered it with a repo-wide sweep and a standing test
# (tests/test_batch10_contract.py) rather than two edits.
#
# The claim: the verbose console line in stress_test_scenarios and the notebook's
# Pass 2 aggregate both call every non-collision outcome "robustly safe", including
# replay_infeasible — a scenario the pipeline REFUSED TO MEASURE. The API and
# models.py stopped making that claim in Batch 2; these two never did.
#
# Measured on the unfixed tree, with the fixtures below:
#
#     stress-testing refused ...
#         robustly safe within bounds (replay_infeasible)   <- no search ran at all
#     stress-testing searched ...
#         robustly safe within bounds (ok)
#     stress-testing solo ...
#         robustly safe within bounds (no_challenger)       <- nothing to perturb

@pytest.fixture
def outcome_parser(monkeypatch):
    """
    A parser stub whose scene depends on the scenario id, so ONE shard can drive
    _stress_one to every outcome it has. Same substitution technique as stub_parser
    above; a different scene table is the only difference.

    Each scene was run through _stress_one and the outcome it produces recorded, so
    this fixture is not asserting an outcome it merely hopes for:

        refused   two 4.5x2.0 boxes both at the origin -> replay_infeasible/collision
        searched  boxes 40 m apart, closing at 10 m/s  -> no_collision_found
        solo      one agent, nothing to perturb        -> no_challenger
        hit       boxes 2.1 m apart laterally          -> collision_found
    """
    def _boxes(n=2):
        states = np.zeros((n, 10, 7), dtype=np.float32)
        states[:, :, 5:7] = [4.5, 2.0]
        return states

    def _searched():
        states = _boxes()
        states[0, :, 0] = -20.0 + 10.0 * (np.arange(10) * 0.1)
        states[0, :, 2] = 10.0
        states[1, :, 0] = 30.0
        return states

    def _hit():
        states = _boxes()
        states[1, :, 1] = 2.1
        return states

    scenes = {'refused': _boxes, 'searched': _searched,
              'solo': lambda: _boxes(n=1), 'hit': _hit}

    module = types.ModuleType('src.data.parser')

    class ScenarioParser:
        def __init__(self, raw):
            self.sid = raw.decode()
            self.states = scenes[self.sid]()

        def get_scenario_id(self):
            return self.sid

        def get_agent_states(self):
            return self.states

        def get_agent_validity(self):
            return np.ones(self.states.shape[:2], dtype=bool)

        def get_agent_types(self):
            return np.ones(self.states.shape[0], dtype=int)

        def get_sdc_index(self):
            return 0

    module.ScenarioParser = ScenarioParser
    monkeypatch.setitem(sys.modules, 'src.data.parser', module)
    return module


_SAFETY_CLAIMS = ('robustly safe', 'certified safe', 'proven safe',
                  'guaranteed safe', 'collision-free')


def _verbose_stress(tmp_path, ids):
    from src.scoring.batch_scorer import stress_test_scenarios

    path = _write_shard(tmp_path / 'outcomes.tfrecord', [i.encode() for i in ids])
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        results = stress_test_scenarios(
            path, ids, verbose=True,
            de_kwargs={'popsize': 4, 'maxiter': 8, 'seed': 1},
        )
    return results, buffer.getvalue()


def test_R08_the_console_does_not_call_a_refused_replay_safe(tmp_path, outcome_parser):
    """
    THE SHARPEST CASE, and the reason this is not cosmetic: replay_infeasible means
    the scenario's own zero-perturbation replay was not faithful enough to measure
    against, so NO SEARCH RAN. There is no budget, no bounds, and no result — and the
    console announced it as safety.
    """
    results, output = _verbose_stress(tmp_path, ['refused'])

    assert results['refused']['status'] == 'replay_infeasible', (
        f"fixture regressed: {results['refused'].get('status')}"
    )
    for claim in _SAFETY_CLAIMS:
        assert claim not in output.lower(), (
            f'the console called a refused replay {claim!r}:\n{output}'
        )
    assert 'replay_infeasible' in output, (
        f'the refusal is not named at all:\n{output}'
    )


@pytest.mark.parametrize('sid,expected_status', [
    ('refused', 'replay_infeasible'),
    ('searched', 'ok'),
    ('solo', 'no_challenger'),
])
def test_R08_every_non_collision_outcome_gets_its_own_console_line(
        tmp_path, outcome_parser, sid, expected_status):
    """
    PARAMETRIZED OVER ALL THREE, because the defect was one `else` branch swallowing
    them. Fixing only the refusal would leave a completed search that found nothing
    still described as a safety certificate, which is the same defect narrowed.

    Each outcome must be NAMED. Asserting only the absence of the banned phrase would
    pass on an empty line.
    """
    results, output = _verbose_stress(tmp_path, [sid])

    assert results[sid]['status'] == expected_status, 'fixture regressed'
    for claim in _SAFETY_CLAIMS:
        assert claim not in output.lower(), f'{sid}: {claim!r} in:\n{output}'
    assert results[sid]['outcome'] in output, (
        f'{sid}: the outcome is not reported:\n{output}'
    )


def test_R08_a_real_collision_still_reports_as_one(tmp_path, outcome_parser):
    """
    The branch that was already correct, pinned so the fix cannot eat it.

    ASSERTED ON THE NUMBERS, NOT ON THE LINE'S SHAPE. An earlier version of this test
    asserted that the outcome name did NOT appear on the collision line, which was a
    guess about formatting rather than about content — and it failed once the fix
    prefixed every line with its outcome, which is the point of the fix. What actually
    has to survive is the norm and the timestep: replacing an informative line with a
    bare outcome name would be a regression dressed as consistency.
    """
    results, output = _verbose_stress(tmp_path, ['hit'])

    assert results['hit']['collision'] is True, 'fixture regressed'
    assert f"{results['hit']['min_perturbation']:.4f}" in output, (
        f'the collision line no longer reports the norm:\n{output}'
    )
    assert f"t={results['hit']['collision_timestep']}" in output, (
        f'the collision line no longer reports the timestep:\n{output}'
    )
    assert results['hit']['method'] in output, (
        f'the collision line no longer reports the method:\n{output}'
    )


def _exec_pass2_aggregate(stress_results):
    """
    Run the notebook's Pass 2 aggregate cell against a supplied result dict.

    ShardLoader is stubbed to an empty iterable, so the cell's re-parse loop does not
    run and no shard or Waymo package is needed — the aggregate block below it, which
    is what the finding is about, runs in full.
    """
    cell = _cell_by_content('PASS 2 AGGREGATES', 'n_no_challenger')
    env = {
        'ShardLoader': lambda path: [], 'SHARD_PATH': 'unused',
        'ScenarioParser': None, 'np': np,
        'stress_results': stress_results, 'ids_to_test': list(stress_results),
        'pass2_elapsed': 1.0,
    }
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(cell, '<cell:pass2_aggregate>', 'exec'), env)
    return buffer.getvalue(), env


_NOTEBOOK_SAMPLE = {
    'syn_hit': {'status': 'ok', 'outcome': 'collision_found', 'collision': True,
                'min_perturbation': 0.5, 'collision_timestep': 3, 'target_idx': 1},
    'syn_searched': {'status': 'ok', 'outcome': 'no_collision_found',
                     'collision': False, 'target_idx': 1},
    'syn_refused': {'status': 'replay_infeasible', 'outcome': 'replay_infeasible',
                    'target_idx': 1},
    'syn_solo': {'status': 'no_challenger', 'outcome': 'no_challenger'},
    'syn_broke': {'status': 'error', 'outcome': 'error', 'error': 'boom'},
}


def test_A09_the_notebook_does_not_count_a_completed_search_as_safe():
    """
    The notebook half. n_safe counts `status == 'ok' and not collided` — a search that
    ran to completion inside ITS budget and ITS bounds against ONE heuristically
    chosen challenger — and prints it as "robustly safe".
    """
    output, _ = _exec_pass2_aggregate(_NOTEBOOK_SAMPLE)

    for claim in _SAFETY_CLAIMS:
        assert claim not in output.lower(), (
            f'the notebook aggregate still reports {claim!r}:\n{output}'
        )


def test_A09_the_notebook_aggregate_accounts_for_every_result():
    """
    A SECOND DEFECT IN THE SAME CELL, not previously cited by either audit.

    The aggregate counts no_challenger, error, and the two `ok` cases. Nothing counts
    replay_infeasible, so the block does not sum to len(stress_results) and says so
    nowhere. Batch 9 makes that worse: scenarios whose logged geometry genuinely
    overlaps now refuse instead of searching, so the uncounted bucket grows.

    Asserted on the CELL'S OWN NAMESPACE rather than on its printed text, so this
    cannot be satisfied by a print that says the right number while the counters
    behind it still lose a row.
    """
    _, env = _exec_pass2_aggregate(_NOTEBOOK_SAMPLE)

    counters = {name: env[name] for name in
                ('n_collision', 'n_no_collision', 'n_replay_infeasible',
                 'n_no_challenger', 'n_error') if name in env}
    missing = {'n_collision', 'n_no_collision', 'n_replay_infeasible',
               'n_no_challenger', 'n_error'} - set(counters)
    assert not missing, f'the aggregate has no counter for {sorted(missing)}'
    assert sum(counters.values()) == len(_NOTEBOOK_SAMPLE), (
        f'counters {counters} sum to {sum(counters.values())}, but the pass produced '
        f'{len(_NOTEBOOK_SAMPLE)} results — some outcome is counted by nothing'
    )


# ── R10: an unpaired surrogate is a client error, not a server fault ────────────
#
# _reject_control_characters tests unicodedata.category(ch) == 'Cc'. A lone surrogate
# is category 'Cs', so it passes validation and reaches psycopg2, which cannot encode
# it and raises — a 500 for what the client sent.
#
# THE CURSOR IS THE REACHABLE VECTOR AND THE PATH IS NOT, measured rather than
# assumed. Starlette and uvicorn both decode the request path with urllib.parse
# .unquote, whose default errors='replace' turns %ED%A0%80 into three U+FFFD, and
# quote('\ud800') raises before a client can even build such a URL. A path-based
# repro would pass vacuously whatever the validator did, so there is not one here.

_LONE_SURROGATE = '\ud800'


def _surrogate_cursor():
    """
    What a hostile client actually puts on the wire: valid base64 of valid JSON whose
    \\ud800 escape decodes to an unpaired surrogate. Nothing in the transport objects.
    """
    return base64.urlsafe_b64encode(
        b'{"f":1.0,"s":"\\ud800"}').decode('ascii')


def test_R10_a_surrogate_cursor_is_rejected_by_the_validator():
    """
    The DB-free half, at the layer the defect is in: _decode_cursor must not hand back
    a string the database cannot store. Pre-fix it returns it happily.
    """
    from fastapi import HTTPException
    from src.api.routes import _decode_cursor

    assert unicodedata.category(_LONE_SURROGATE) == 'Cs', 'fixture regressed'

    with pytest.raises(HTTPException) as caught:
        _decode_cursor(_surrogate_cursor())
    assert caught.value.status_code == 422, caught.value.status_code


@requires_db
def test_R10_a_surrogate_cursor_is_a_422_not_a_500(conn7):
    """
    The audit's own claim, end to end: the request returns a client error rather than
    crashing in psycopg2. raise_server_exceptions=False so the 500 is observable as a
    status code instead of propagating out of the client.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app

    _seed7(conn7)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get('/scenarios', params={'cursor': _surrogate_cursor()})

    assert response.status_code == 422, (
        f'surrogate cursor returned {response.status_code}: {response.text[:300]}'
    )


@pytest.mark.parametrize('legal', [
    'a.b.c', 'a-b_c', 'ABC123', 'ünïcødé', 'scenario nested', 'a+b=c&d', 'x' * 200,
])
def test_R10_the_new_check_rejects_nothing_that_was_legal(legal):
    """
    The regression risk in any validator change: failing closed on valid input.

    Reuses test_api_robustness.test_the_validator_does_not_become_a_format_whitelist's
    fixture list verbatim — the same seven strings, including a non-ASCII one — and
    checks them at the function rather than through HTTP, so this runs without a
    database alongside the surrogate case it is the counterweight to.

    RESOLVES THE VALIDATOR UNDER EITHER NAME, DELIBERATELY. The claim here is "these
    seven were legal before the change and are legal after it", and a test that can
    only run against the fixed tree cannot check the first half — it would fail
    pre-fix with an ImportError, which demonstrates nothing about validation. Batch 7
    hit exactly this: test_R01 failed pre-fix on a TypeError about a new keyword and
    was reported as reproducing a defect it had never reached.
    """
    from src.api import routes

    validator = getattr(routes, '_reject_unstorable_text',
                        getattr(routes, '_reject_control_characters', None))
    assert validator is not None, 'neither validator name exists in src.api.routes'

    assert validator(legal, 'scenario_id') == legal


# ── R11 / A08: the loader must not trust a length it has not checked ────────────
#
# TWO FIXES, DIFFERENT IN KIND, and the module's own docstring already draws the line
# between them: "FRAMING is checked unconditionally... CHECKSUMS are opt-in."
#
#   (1) VERIFICATION, inside the opt-in path. With verify_crc=True the length field's
#       own checksum was verified AFTER the length had already been used to size the
#       read. Ordering only; the flag still gates it.
#   (2) FRAMING, unconditional. A declared length larger than the bytes remaining is a
#       structural defect detectable from the file alone, with no checksum and no
#       constant. Batch 5 self-disclosed this and deferred it.
#
# ASSERTED VIA A CALL RECORDER, NOT BY OBSERVING A MemoryError. How a machine responds
# to a 256 MiB allocation is not a property of this code, and a test that depends on it
# is a test that passes or fails by luck.

_OVERSIZED = 2 ** 28          # 256 MiB claimed by a file of a few bytes


@pytest.fixture
def read_recorder(monkeypatch):
    """Records every byte count _read_exactly is asked for, and calls through."""
    from src.data.loader import ShardLoader

    requested = []
    original = ShardLoader._read_exactly

    def recording(self, f, n, what, record):
        requested.append(n)
        return original(self, f, n, what, record)

    monkeypatch.setattr(ShardLoader, '_read_exactly', recording)
    return requested


def test_R11_a_length_longer_than_the_file_is_refused_before_it_is_read(
        tmp_path, read_recorder):
    """
    The unconditional half: verify_crc is OFF, and the length's checksum is VALID, so
    the framing check is the only thing that can catch this. Pre-fix the loader asks
    for 256 MiB out of a 17-byte file.
    """
    from src.data.loader import ShardLoader

    header = struct.pack('<Q', _OVERSIZED)
    path = tmp_path / 'oversized.tfrecord'
    path.write_bytes(header + struct.pack('<I', _masked_crc32c(header)) + b'short')
    size = path.stat().st_size

    with pytest.raises(ValueError):
        list(ShardLoader(str(path)))

    assert max(read_recorder) <= size, (
        f'asked for {max(read_recorder)} bytes from a {size}-byte file — the declared '
        f'length was used before it was checked against the file'
    )


def test_A08_the_length_checksum_is_verified_before_the_length_is_used(
        tmp_path, read_recorder):
    """
    The opt-in half, which is a DIFFERENT fix: here the length's checksum is WRONG, so
    with verify_crc=True the loader has everything it needs to reject the record
    without touching the payload — and used to size the read first anyway.

    The message assertion matters: post-fix this must be the LENGTH checksum failing,
    not the framing check quietly taking credit for it.
    """
    from src.data.loader import ShardLoader

    header = struct.pack('<Q', _OVERSIZED)
    path = tmp_path / 'bad_length_crc.tfrecord'
    path.write_bytes(header + b'\x00\x00\x00\x00' + b'short')
    size = path.stat().st_size

    with pytest.raises(ValueError, match='length checksum mismatch'):
        list(ShardLoader(str(path), verify_crc=True))

    assert max(read_recorder) <= size, (
        f'asked for {max(read_recorder)} bytes before verifying the length checksum '
        f'that would have rejected the record'
    )
