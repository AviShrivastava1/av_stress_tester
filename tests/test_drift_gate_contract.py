"""
test_drift_gate_contract.py — the fixed 0.5 m drift gate replaced by what the data supports.

The A_MAX=12 run measured baseline drift over the whole shard as continuous, with no
valley, so any fixed threshold is a judgement. Measured on that shard, the 0.5 m gate
refused 8 of the 20 Pass 2 slots; six of those had no speed defect at all. What this
file pins:

  1. THE GATES AND THEIR ORDER. The hard gate (zero-delta replay already collides)
     first, unchanged. Then the speed-step refusal: a vehicle challenger whose logged
     speed holds an unreplayable step (speed change beyond A_MAX that the log does not
     take back) of more than SPEED_STEP_REFUSE_MPS for SPEED_STEP_HOLD_S. Then the
     drift backstop at BASELINE_DRIFT_REFUSE_M. All three refuse as
     ReplayFidelityError, so as `replay_infeasible`, each with its own reason. On the
     real shard every speed-step case also drifts past the backstop, so the speed step
     names the cause only because it is checked before drift.

  2. WHAT EVERY ATTEMPT RECORDS. A searched result's search_provenance carries the
     gate settings it ran under, its held speed excess, and baseline_offset_at_collision.
     A refused attempt's numbers go to last_attempt_diagnostics, never
     search_provenance (Batch 2's column ownership).

  3. THAT PASS 3 REPLAYS UNDER THE SEARCH'S OWN GATES. export_shard_geometry passes
     the recorded settings to PerturbationSpace; a result with none recorded predates
     them and replays under the historical 0.5 m and no speed gate; a refusal at export
     lands in its own replay_refused bucket, not in errors.

Fixtures are straight vehicle tracks along x written analytically: a logged SPEED
channel, and positions integrated (trapezoidally, as bicycle_step does) from a
separately chosen POSITION speed. When the position speed is the A_MAX-limited replay
of the logged speed, the replay reproduces the logged positions and drift is ~0; when
positions follow the logged speed itself, an unreplayable step becomes drift.

The DB-backed tests need a DISPOSABLE Postgres/PostGIS database and skip unless
AV_CLAIMS_DB=1. Their parser stub and shard writer have the same shape as
test_a_max_contract.py's.

Run:
    ./venv/bin/python -m pytest tests/test_drift_gate_contract.py -q
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_drift_gate_contract.py -q
"""

import os
import struct
import sys
import types as _types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimization import perturbation_space as ps
from src.optimization.perturbation_space import (
    BASELINE_DRIFT_REFUSE_M, SPEED_STEP_HOLD_S, SPEED_STEP_REFUSE_MPS,
    PerturbationSpace, ReplayFidelityError, held_speed_excess,
)
from src.physics.bicycle_model import A_MAX, DT
from src.physics.simulator import TYPE_PEDESTRIAN, TYPE_VEHICLE
from src.scoring.db import compute_scene_fingerprint

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

CAR_L, CAR_W = 4.5, 2.0
T = 91
HOLD_FRAMES = int(round(SPEED_STEP_HOLD_S / DT))
V0 = 10.0          # cruising speed of every fixture track, m/s
STEP_AT = 20       # the logged step lands on the transition STEP_AT -> STEP_AT+1

# A logged speed jump of S in one frame is an acceleration of S/DT. The replay can follow
# A_MAX*DT of it; the rest is the excess the replay never recovers unless the log takes
# the jump back.
REPLAYABLE = A_MAX * DT


def _scene(t=T, sdc_far=True):
    states = np.zeros((2, t, 7), dtype=np.float32)
    states[:, :, 5] = CAR_L
    states[:, :, 6] = CAR_W
    if sdc_far:
        states[0, :, 1] = 1000.0
    return states, np.ones((2, t), dtype=bool), np.full(2, TYPE_VEHICLE)


def _replayed_speed(logged):
    """What the A_MAX-limited replay does with a logged speed channel."""
    out = np.empty_like(logged)
    out[0] = logged[0]
    for k in range(len(logged) - 1):
        a = np.clip((logged[k + 1] - logged[k]) / DT, -A_MAX, A_MAX)
        out[k + 1] = out[k] + a * DT
    return out


def _track(states, idx, logged_speed, position_speed):
    """Heading 0 along x: the logged speed channel, and positions from position_speed."""
    logged_speed = np.asarray(logged_speed, dtype=np.float64)
    position_speed = np.asarray(position_speed, dtype=np.float64)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (position_speed[:-1] + position_speed[1:]) * DT)])
    states[idx, :, 0] = x
    states[idx, :, 2] = logged_speed
    states[idx, :, 3] = 0.0
    states[idx, :, 4] = 0.0
    return states


def _step(size, lasts=None, t=T):
    """V0, then V0+size from frame STEP_AT+1 on; back to V0 after `lasts` frames if given."""
    v = np.full(t, V0)
    end = t if lasts is None else STEP_AT + 1 + lasts
    v[STEP_AT + 1:end] += size
    return v


def _step_only_scene(size=3.0, lasts=None, t=T):
    """A speed step whose positions follow the REPLAY: refused for the step, drift ~0."""
    s, v, types = _scene(t)
    logged = _step(size, lasts, t)
    return _track(s, 1, logged, _replayed_speed(logged)), v, types


def _step_and_drift_scene(size=3.0):
    """The same step with positions following the LOG: the step becomes metres of drift."""
    s, v, types = _scene()
    logged = _step(size)
    return _track(s, 1, logged, logged), v, types


def _constant_drift_scene(drift_m, t=T, sdc_far=True):
    """Parked positions, logged speed u: the replay drifts exactly u*(t-1)*DT by the end."""
    s, v, types = _scene(t, sdc_far)
    u = drift_m / ((t - 1) * DT)
    return _track(s, 1, np.full(t, u), np.zeros(t)), v, types


# ── the decided values ───────────────────────────────────────────────────────────

def test_the_decided_gate_values():
    """2 m backstop, 1.0 m/s held for 1 s: the values the review decided (2026-09-30)."""
    assert BASELINE_DRIFT_REFUSE_M == 2.0
    assert SPEED_STEP_REFUSE_MPS == 1.0
    assert SPEED_STEP_HOLD_S == 1.0


# ── the held speed excess ────────────────────────────────────────────────────────

def test_an_unreturned_step_holds_its_unreplayable_part():
    s, v, _ = _step_only_scene(size=3.0)
    assert held_speed_excess(s, v, 1, DT) == pytest.approx(3.0 - REPLAYABLE, abs=1e-4)


def test_a_balanced_spike_holds_nothing():
    """+S then -S on the next frame: the replay's two clipped halves cancel."""
    s, v, _ = _step_only_scene(size=3.0, lasts=1)
    assert held_speed_excess(s, v, 1, DT) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize('lasts, held', [(HOLD_FRAMES, False), (HOLD_FRAMES + 1, True)])
def test_the_hold_is_how_long_the_log_keeps_the_step(lasts, held):
    """A step the log takes back within SPEED_STEP_HOLD_S is a glitch reversing, not held."""
    s, v, _ = _step_only_scene(size=3.0, lasts=lasts)
    measured = held_speed_excess(s, v, 1, DT)
    if held:
        assert measured == pytest.approx(3.0 - REPLAYABLE, abs=1e-4)
    else:
        assert measured == pytest.approx(0.0, abs=1e-4)


def test_a_gap_splits_the_track_into_runs():
    """
    No accumulation across an interior gap: each valid run starts from zero. The step
    is held for HOLD_FRAMES - 1 frames before a gap and the speed stays up after it,
    so it would count as held only if the excess were carried across the gap.
    """
    s, v, _ = _step_only_scene(size=3.0)
    gap = STEP_AT + HOLD_FRAMES        # the run before it holds the step HOLD_FRAMES-1 frames
    v[1, gap] = False
    assert held_speed_excess(s, v, 1, DT) == pytest.approx(0.0, abs=1e-4)
    v[1, gap] = True
    assert held_speed_excess(s, v, 1, DT) == pytest.approx(3.0 - REPLAYABLE, abs=1e-4), (
        'fixture regressed: without the gap the same step must be held'
    )


# ── the gates, one at a time ─────────────────────────────────────────────────────

@pytest.mark.parametrize('excess, refused', [
    (SPEED_STEP_REFUSE_MPS - 0.05, False), (SPEED_STEP_REFUSE_MPS + 0.05, True),
])
def test_the_speed_step_threshold(excess, refused):
    s, v, types = _step_only_scene(size=REPLAYABLE + excess)
    if refused:
        with pytest.raises(ReplayFidelityError) as caught:
            PerturbationSpace(s, v, types, 0, 1)
        assert caught.value.reason == 'speed_step'
        assert caught.value.held_speed_excess == pytest.approx(excess, abs=1e-4)
    else:
        space = PerturbationSpace(s, v, types, 0, 1)
        assert space.held_speed_excess == pytest.approx(excess, abs=1e-4)


def test_a_speed_step_refusal_does_not_need_any_drift():
    s, v, types = _step_only_scene()
    with pytest.raises(ReplayFidelityError) as caught:
        PerturbationSpace(s, v, types, 0, 1)
    assert caught.value.reason == 'speed_step'
    assert caught.value.baseline_replay_error < 0.01, 'fixture regressed: it should not drift'
    assert caught.value.baseline_replay_collides is False


@pytest.mark.parametrize('fraction, refused', [(0.9, False), (1.1, True)])
def test_the_drift_backstop(fraction, refused):
    s, v, types = _constant_drift_scene(fraction * BASELINE_DRIFT_REFUSE_M)
    if refused:
        with pytest.raises(ReplayFidelityError) as caught:
            PerturbationSpace(s, v, types, 0, 1)
        assert caught.value.reason == 'drift'
        assert caught.value.held_speed_excess == pytest.approx(0.0, abs=1e-6)
    else:
        space = PerturbationSpace(s, v, types, 0, 1)
        assert 0.5 < space.baseline_replay_error < BASELINE_DRIFT_REFUSE_M, (
            'fixture regressed: the 0.5 m gate would have refused this, 2 m must not'
        )


def test_the_speed_step_applies_to_vehicles_only():
    """Pedestrians and cyclists replay through linear_model, whose cap was never measured."""
    s, v, types = _step_only_scene()
    types[1] = TYPE_PEDESTRIAN
    space = PerturbationSpace(s, v, types, 0, 1, max_baseline_drift=None)
    assert space.held_speed_excess is None


def test_none_disables_the_speed_step_and_still_records_it():
    s, v, types = _step_only_scene()
    space = PerturbationSpace(s, v, types, 0, 1, max_speed_step=None)
    assert space.held_speed_excess == pytest.approx(3.0 - REPLAYABLE, abs=1e-4)
    assert space.max_speed_step is None


# ── the order: hard gate, then speed step, then drift ────────────────────────────

def test_a_step_that_also_drifts_is_refused_as_a_speed_step():
    """
    THE ORDER, pinned. On the real shard all 12 speed-step cases also drift past 2 m;
    the reason names the cause only if the speed step is checked first.
    """
    s, v, types = _step_and_drift_scene()
    with pytest.raises(ReplayFidelityError) as caught:
        PerturbationSpace(s, v, types, 0, 1)
    assert caught.value.baseline_replay_error > BASELINE_DRIFT_REFUSE_M, (
        'fixture regressed: it must trip the drift backstop too'
    )
    assert caught.value.reason == 'speed_step'


def test_a_step_on_a_baseline_that_already_collides_is_refused_as_a_collision():
    s, v, types = _step_only_scene()
    s[0, :, 0] = s[1, :, 0] + 1.0       # the SDC sits on the challenger's logged track
    s[0, :, 1] = 0.0
    with pytest.raises(ReplayFidelityError) as caught:
        PerturbationSpace(s, v, types, 0, 1)
    assert caught.value.reason == 'collision'
    assert caught.value.held_speed_excess == pytest.approx(3.0 - REPLAYABLE, abs=1e-4)


# ── Pass 2: what each attempt records ────────────────────────────────────────────

DE_SMALL = {'popsize': 6, 'maxiter': 15, 'seed': 1}


def test_a_speed_step_refusal_reaches_the_attempt_diagnostics():
    from src.scoring.batch_scorer import _describe_outcome, _stress_one
    from src.scoring.db import _attempt_diagnostics

    s, v, types = _step_only_scene()
    result = _stress_one(s, v, types, 0, de_kwargs=DE_SMALL)
    assert result['outcome'] == 'replay_infeasible'
    assert result['reason'] == 'speed_step'
    assert 'search_provenance' not in result, 'a refusal must not write the result group'

    diagnostics = _attempt_diagnostics(result)
    assert diagnostics['reason'] == 'speed_step'
    assert diagnostics['held_speed_excess'] == pytest.approx(3.0 - REPLAYABLE, abs=1e-4)
    assert 'baseline_replay_error' in diagnostics
    assert 'held speed excess' in _describe_outcome(result)


def _near_pair_with_drift(drift_m=0.9, t=31):
    """Parked challenger 2.1 m beside the SDC, logged as creeping forward: known drift."""
    s, v, types = _constant_drift_scene(drift_m, t=t, sdc_far=False)
    s[1, :, 1] = 2.1
    return s, v, types


def test_search_provenance_records_the_gates_it_ran_under():
    from src.scoring.batch_scorer import _stress_one

    s, v, types = _near_pair_with_drift()
    result = _stress_one(s, v, types, 0, de_kwargs=DE_SMALL)
    assert result['status'] == 'ok', f'fixture regressed: {result}'
    prov = result['search_provenance']
    assert prov['max_baseline_drift'] == BASELINE_DRIFT_REFUSE_M
    assert prov['max_speed_step'] == SPEED_STEP_REFUSE_MPS
    assert prov['speed_step_hold_s'] == SPEED_STEP_HOLD_S
    assert prov['held_speed_excess'] == pytest.approx(0.0, abs=1e-6)


def test_baseline_offset_at_collision_is_the_replay_to_log_distance_at_that_frame():
    """
    Defined as: the distance between the zero-delta replay and the logged challenger at
    collision_timestep. Checked against the analytic replay (parked positions, logged
    speed u, so the replay is u*t*DT ahead at frame t) and against the space's own replay.
    """
    from src.scoring.batch_scorer import _stress_one

    t = 31
    s, v, types = _near_pair_with_drift(drift_m=0.9, t=t)
    result = _stress_one(s, v, types, 0, de_kwargs={'popsize': 10, 'maxiter': 30, 'seed': 1})
    assert result['collision'], f'fixture regressed: no collision found: {result}'
    hit = result['collision_timestep']
    u = 0.9 / ((t - 1) * DT)

    offset = result['search_provenance']['baseline_offset_at_collision']
    assert offset == pytest.approx(u * hit * DT, abs=1e-4)

    replay = PerturbationSpace(s, v, types, 0, 1).apply(np.zeros(4, np.float32))
    assert offset == pytest.approx(float(np.linalg.norm(replay[1, hit, :2] - s[1, hit, :2])),
                                   abs=1e-6)


def test_baseline_offset_at_collision_is_none_without_a_collision():
    from src.scoring.batch_scorer import _stress_one

    s, v, types = _constant_drift_scene(0.9)          # SDC 1 km away: nothing to hit
    result = _stress_one(s, v, types, 0, de_kwargs=DE_SMALL)
    assert result['status'] == 'ok' and not result['collision'], f'fixture regressed: {result}'
    assert result['search_provenance']['baseline_offset_at_collision'] is None


# ── Pass 3: the export replays under the search's own gates (DB-backed) ──────────

def _write_geom_shard(path, scenario_ids):
    """Same framing as test_a_max_contract.py's helper of the same name."""
    from src.data.loader import _masked_crc32c

    blob = b''
    for sid in scenario_ids:
        payload = sid.encode()
        header = struct.pack('<Q', len(payload))
        blob += (header + struct.pack('<I', _masked_crc32c(header))
                 + payload + struct.pack('<I', _masked_crc32c(payload)))
    path.write_bytes(blob)
    return str(path)


SCENES = {'drift': _near_pair_with_drift, 'step': _step_only_scene}


def _assert_today_refuses_the_step_scene():
    """Premise of the two step-scene export tests: without it they pass vacuously."""
    s, v, types = _step_only_scene()
    with pytest.raises(ReplayFidelityError) as caught:
        PerturbationSpace(s, v, types, 0, 1)
    assert caught.value.reason == 'speed_step', 'fixture regressed'


@pytest.fixture
def _scene_parser(monkeypatch):
    """test_a_max_contract's _vehicle_parser shape, serving SCENES by scenario id."""
    module = _types.ModuleType('src.data.parser')

    class ScenarioParser:
        def __init__(self, raw):
            self.sid = raw.decode()

        def get_scenario_id(self):
            return self.sid

        def get_agent_states(self):
            return SCENES[self.sid]()[0]

        def get_agent_validity(self):
            return SCENES[self.sid]()[1]

        def get_agent_types(self):
            return SCENES[self.sid]()[2]

        def get_sdc_index(self):
            return 0

    module.ScenarioParser = ScenarioParser
    monkeypatch.setitem(sys.modules, 'src.data.parser', module)
    yield module


@pytest.fixture
def gconn():
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


def _search_and_store(conn, sid, monkeypatch=None):
    """A REAL search on SCENES[sid], stored the way Pass 2 stores it. With monkeypatch,
    the search runs with the speed-step gate off (the step scene is otherwise refused)."""
    from src.scoring import db
    from src.scoring.batch_scorer import _stress_one

    states, validity, types = SCENES[sid]()
    if monkeypatch is None:
        result = _stress_one(states, validity, types, 0, de_kwargs=DE_SMALL)
    else:
        class NoSpeedGate(ps.PerturbationSpace):
            def __init__(self, *a, **k):
                k['max_speed_step'] = None
                super().__init__(*a, **k)

        with monkeypatch.context() as scoped:      # scoped: the parser stub stays
            scoped.setattr(ps, 'PerturbationSpace', NoSpeedGate)
            result = _stress_one(states, validity, types, 0, de_kwargs=DE_SMALL)
    assert result['status'] == 'ok' and result['delta'] is not None, f'fixture regressed: {result}'
    db.upsert_scores(conn, [dict(scenario_id=sid, shard='x', n_agents=2, min_ttc=9.0,
                                 min_pet=9.0, fragility_score=1.0,
                                 scene_fingerprint=compute_scene_fingerprint(states, validity, types),
                                 sdc_idx=0)])
    db.update_stress_results(conn, {sid: dict(result)})
    return result


def _export(conn, tmp_path, sid, result):
    from src.scoring.export_geometry import export_shard_geometry
    shard = _write_geom_shard(tmp_path / 'shard.tfrecord', [sid])
    return export_shard_geometry(conn, shard, [sid], stress_results={sid: dict(result)},
                                 verbose=False)


def _without_gate_keys(result):
    legacy = dict(result)
    legacy['search_provenance'] = {
        k: val for k, val in result['search_provenance'].items()
        if k not in ('max_baseline_drift', 'max_speed_step', 'speed_step_hold_s')
    }
    return legacy


@requires_db
def test_a_result_searched_under_the_current_gates_is_exported(gconn, tmp_path, _scene_parser):
    """0.9 m of drift: refused by the old 0.5 m gate, searched and exported under 2 m."""
    result = _search_and_store(gconn, 'drift')
    summary = _export(gconn, tmp_path, 'drift', result)
    assert summary['replay_refused'] == [], summary['replay_refused']
    assert summary['errors'] == [], summary['errors']
    assert summary['perturbed_written'] == 1, summary


@requires_db
def test_a_legacy_result_replays_under_the_historical_drift_gate(gconn, tmp_path, _scene_parser):
    """No recorded gates: the search predates them and ran under 0.5 m. Refused into its
    own bucket, not errors, with the baseline agents still exported."""
    from src.scoring.export_geometry import DRIFT_GATE_BEFORE_PROVENANCE

    result = _search_and_store(gconn, 'drift')
    summary = _export(gconn, tmp_path, 'drift', _without_gate_keys(result))
    assert DRIFT_GATE_BEFORE_PROVENANCE == 0.5
    assert [(r['scenario_id'], r['reason'], r['max_baseline_drift'], r['max_speed_step'],
             r['gate_settings_source']) for r in summary['replay_refused']] == [
        ('drift', 'drift', 0.5, None, 'predates_provenance')]
    assert summary['errors'] == [], summary['errors']
    assert summary['perturbed_written'] == 0 and summary['exported'] == 1, summary


@requires_db
def test_a_legacy_result_replays_with_no_speed_gate(gconn, tmp_path, _scene_parser, monkeypatch):
    """The speed gate did not exist before the keys: a legacy result is never refused for it."""
    _assert_today_refuses_the_step_scene()
    result = _search_and_store(gconn, 'step', monkeypatch)
    summary = _export(gconn, tmp_path, 'step', _without_gate_keys(result))
    assert summary['replay_refused'] == [], summary['replay_refused']
    assert summary['perturbed_written'] == 1, summary


@requires_db
def test_a_recorded_speed_gate_is_reproduced_at_export(gconn, tmp_path, _scene_parser, monkeypatch):
    """The recorded settings are what the export replays under, not today's defaults."""
    _assert_today_refuses_the_step_scene()
    result = _search_and_store(gconn, 'step', monkeypatch)
    recorded = dict(result)
    recorded['search_provenance'] = dict(result['search_provenance'],
                                         max_speed_step=SPEED_STEP_REFUSE_MPS)
    summary = _export(gconn, tmp_path, 'step', recorded)
    assert [(r['scenario_id'], r['reason'], r['gate_settings_source'])
            for r in summary['replay_refused']] == [('step', 'speed_step', 'search_provenance')]
    assert summary['errors'] == [], summary['errors']
