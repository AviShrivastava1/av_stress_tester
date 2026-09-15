"""
test_audit2_regressions.py — second audit, findings R02, R04, R05 and R09.

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

import contextlib
import io
import json
import os
import struct
import sys
import types

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
