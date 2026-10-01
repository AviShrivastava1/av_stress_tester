"""
test_a_max_contract.py — bicycle_model.A_MAX raised from 5.0 to 12.0 m/s^2.

Three things this file pins, none of which any test pinned before this change:

  1. WHY the constant moved (T1). A logged hard stop between the old cap and the new
     one must now replay faithfully at zero perturbation. Under 5.0 the inverter
     clipped the recovered deceleration and bicycle_step replayed the clipped value,
     so the replay drifted ahead of the log.

  2. THAT THE CAP STILL EXISTS (T2), at every one of its three enforcement sites:
     bicycle_step's clamp, invert_bicycle's clip, and the torch rollout in
     autograd_optimizer. Before this file, deleting any of the three clamps left the
     whole suite green: no vehicle fixture delivered more than 2.0 m/s^2 to any of
     them. The realism ceiling was entirely unpinned.

  3. THAT A STORED DELTA IS NOT REPLAYED UNDER A DIFFERENT CAP (the export guard).
     Pass 2 records the cap it searched under into search_provenance['a_max'], and
     export_shard_geometry refuses to replay a delta whose recorded cap differs from
     the current one, into its own a_max_changed bucket. Before this file, no test
     sent a VEHICLE result through export_shard_geometry at all: every existing
     export fixture uses a pedestrian challenger.

Every expected magnitude is written relative to the imported constants, never as a
bare 5.0 or 12.0, except T1's HARD_BRAKE (a fixture input, see its own comment) and
A_MAX_BEFORE_PROVENANCE (a historical fact, imported from where it is defined).

The DB-backed tests need a DISPOSABLE Postgres/PostGIS database and skip unless
AV_CLAIMS_DB=1, same as every other DB-gated file in this project. Their parser stub
and shard writer have the same shape as test_g08_replay_model_identity.py's, serving
a vehicle scene instead of a pedestrian one.

Run:
    ./venv/bin/python -m pytest tests/test_a_max_contract.py -q
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_a_max_contract.py -q
"""

import os
import struct
import sys
import types as _types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimization.perturbation_space import PerturbationSpace
from src.physics import simulator
from src.physics.bicycle_model import A_MAX, DT, bicycle_step, invert_bicycle
from src.physics.linear_model import A_MAX as LINEAR_A_MAX
from src.physics.simulator import (
    TYPE_CYCLIST, TYPE_PEDESTRIAN, TYPE_VEHICLE, replay_a_max,
)
from src.scoring.db import compute_scene_fingerprint
from src.scoring.export_geometry import A_MAX_BEFORE_PROVENANCE

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

CAR_L, CAR_W = 4.5, 2.0

# A real hard stop, strictly between the old cap (5.0) and the new one (12.0): the
# same magnitude 7g-iv's own hard-stop fixture used. A FIXTURE INPUT describing a
# manoeuvre, not a number obtained by running code.
HARD_BRAKE = 8.0

# The drift bound test_replay_contract.py's test_accelerating_track_replays_without_drift
# already holds a straight, uniformly accelerating track to. Trapezoidal integration
# of a linear speed profile is exact, so what is left is float32 noise.
STRAIGHT_TOLERANCE_M = 1e-3


def _vehicle_scene(t):
    """SDC and one vehicle challenger, all frames valid, SDC parked 1 km away."""
    states = np.zeros((2, t, 7), dtype=np.float32)
    states[:, :, 5] = CAR_L
    states[:, :, 6] = CAR_W
    states[0, :, 1] = 1000.0
    return states, np.ones((2, t), dtype=bool), np.full(2, TYPE_VEHICLE)


def _braking_track(states, idx, v0, decel):
    """
    A straight track braking at a constant `decel`, written analytically.
    x = v0*t - decel*t^2/2 and v = v0 - decel*t: exactly what bicycle_step's
    trapezoidal update produces from that deceleration, so a faithful replay has
    nothing left to disagree with but float32.
    """
    t = np.arange(states.shape[1]) * DT
    assert (v0 - decel * t[-1]) > 0, 'fixture regressed: the track must not stop'
    states[idx, :, 0] = v0 * t - 0.5 * decel * t * t
    states[idx, :, 2] = v0 - decel * t
    return states


# ── dispatch ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('agent_type, expected', [
    (TYPE_VEHICLE, A_MAX),
    (TYPE_PEDESTRIAN, LINEAR_A_MAX),
    (TYPE_CYCLIST, LINEAR_A_MAX),
], ids=['vehicle', 'pedestrian', 'cyclist'])
def test_replay_a_max_follows_the_simulators_model_dispatch(agent_type, expected):
    """
    The two A_MAX constants share a name and are independent. replay_a_max is the one
    place that decides which applies, for both the Pass 2 writer and the Pass 3
    checker, so it must pick by the same rule ForwardSimulator.step does.
    """
    assert replay_a_max(agent_type) == expected
    assert isinstance(replay_a_max(agent_type), float)


# ── T1: why the constant moved ───────────────────────────────────────────────────

def test_a_logged_hard_brake_between_the_old_cap_and_the_new_one_replays_faithfully():
    """
    A real 8 m/s^2 stop, from 25 m/s over 3 s, must replay onto its own logged track
    at zero perturbation.

    Measured against A_MAX=5.0 (origin/main at 7286b90, before this change): every
    recovered control was clipped to -5.0, the replay braked at 5 instead of 8, and
    it ran ahead of the log from the start, crossing the 0.5 m fidelity gate at frame
    6 (0.6 s) and finishing 13.5 m ahead: baseline_replay_error = 13.5000 m, which is
    0.5 * (8 - 5) * 3^2 exactly. That is the 7g-iv mechanism, reduced to one track.

    max_baseline_drift=None so a regression reports the drift itself rather than
    dying inside the constructor with ReplayFidelityError.
    """
    s, v, types = _vehicle_scene(31)
    _braking_track(s, 1, v0=25.0, decel=HARD_BRAKE)

    space = PerturbationSpace(s, v, types, 0, 1, max_baseline_drift=None)

    assert space.baseline_replay_error <= STRAIGHT_TOLERANCE_M, (
        f'a logged {HARD_BRAKE} m/s^2 stop drifted {space.baseline_replay_error:.4f} m '
        f'at zero perturbation under A_MAX={A_MAX}'
    )


# ── T2: the cap still exists, at all three sites ─────────────────────────────────

@pytest.mark.parametrize('sign', [1.0, -1.0], ids=['accelerating', 'braking'])
def test_bicycle_step_clamps_a_command_beyond_a_max(sign):
    """Site 1. A command ten times the cap moves the speed by exactly A_MAX*dt."""
    v0 = 20.0
    nxt = bicycle_step(np.array([0.0, 0.0, 0.0, v0]),
                       np.array([0.0, sign * 10.0 * A_MAX]), wheelbase=2.7)
    assert float(nxt[3]) == pytest.approx(v0 + sign * A_MAX * DT, abs=1e-5)


@pytest.mark.parametrize('sign', [1.0, -1.0], ids=['accelerating', 'braking'])
def test_invert_bicycle_clips_a_recovered_acceleration_beyond_a_max(sign):
    """Site 2. A logged speed jump ten times the cap recovers exactly +/-A_MAX."""
    v0 = 20.0
    v1 = v0 + sign * 10.0 * A_MAX * DT
    assert v1 > 0, 'fixture regressed: the jump must not reverse the car'
    control = invert_bicycle(np.array([0.0, 0.0, 0.0, v0]),
                             np.array([0.0, 0.0, 0.0, v1]), wheelbase=2.7)
    assert float(control[1]) == pytest.approx(sign * A_MAX, abs=1e-5)


def test_numpy_and_torch_rollouts_agree_when_the_cap_binds():
    """
    Site 3, and the numpy side of it at the same time. The logged braking is twice
    the cap, so every recovered control sits AT -A_MAX, and a da_bias of -1.5 then
    asks both rollouts for more than the cap on every step. A log no vehicle can
    drive, on purpose, so both soft gates are off: max_speed_step=None because the
    speed-step refusal exists to refuse exactly this log.

    test_replay_contract.py's own parity test never reaches the cap (the largest
    acceleration it delivers to any clamp is 2.0 m/s^2), so the two rollouts could
    disagree about saturation and it would still pass. Here, removing either clamp
    makes one rollout brake harder than the other.
    """
    torch = pytest.importorskip('torch')
    from src.optimization.autograd_optimizer import _DiffBicycleRollout

    s, v, types = _vehicle_scene(12)
    _braking_track(s, 1, v0=35.0, decel=2.0 * A_MAX)
    space = PerturbationSpace(s, v, types, 0, 1, max_baseline_drift=None,
                              max_speed_step=None)
    assert np.allclose(space.base_controls[:-1, 1], -A_MAX), (
        'fixture regressed: the baseline controls must sit at the cap'
    )

    delta = np.array([0.0, 0.0, -1.5, 0.0], dtype=np.float32)
    numpy_states = space.apply(delta)
    torch_traj = _DiffBicycleRollout(space).rollout(
        torch.tensor(delta, dtype=torch.float64)).detach().numpy()

    frames = np.arange(space.t0, s.shape[1])
    speed = np.hypot(numpy_states[1, frames, 2], numpy_states[1, frames, 3])
    np.testing.assert_allclose(speed, torch_traj[:, 3], atol=1e-4)
    np.testing.assert_allclose(numpy_states[1, frames, 0], torch_traj[:, 0], atol=1e-4)
    # And the cap, not the command, is what both of them applied.
    np.testing.assert_allclose(np.diff(speed), -A_MAX * DT, atol=1e-4)


# ── the writer: Pass 2 records the cap it searched under ─────────────────────────

def _near_vehicle_pair():
    """test_batch12_contract.py's vehicle scene: two parked cars 2.1 m apart."""
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[:, :, 5:7] = [CAR_L, CAR_W]
    states[1, :, 1] = 2.1
    return states, np.ones((2, 10), dtype=bool), np.full(2, TYPE_VEHICLE)


def _walking_pedestrian():
    """test_batch7_contract.py's pedestrian: walking at 2 m/s toward the SDC."""
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [CAR_L, CAR_W]
    states[1, :, 5:7] = [0.6, 0.6]
    states[1, :, 0] = 5.0 - 2.0 * np.arange(10) * 0.1
    states[1, :, 2] = -2.0
    states[1, :, 4] = np.pi
    return states, np.ones((2, 10), dtype=bool), np.array([TYPE_VEHICLE, TYPE_PEDESTRIAN])


@pytest.mark.parametrize('scene, expected', [
    (_near_vehicle_pair, A_MAX),
    (_walking_pedestrian, LINEAR_A_MAX),
], ids=['vehicle', 'pedestrian'])
def test_stress_one_records_the_cap_of_the_model_it_searched_under(scene, expected):
    """
    A real search, not a hand-built result: the key the export guard reads must be
    the one the pipeline actually writes, holding the constant of the model the
    search actually dispatched to.
    """
    from src.scoring.batch_scorer import _stress_one

    states, validity, types = scene()
    result = _stress_one(states, validity, types, 0,
                         de_kwargs={'popsize': 4, 'maxiter': 5, 'seed': 1})
    assert result['status'] == 'ok', f'fixture regressed: {result}'
    assert result['search_provenance']['a_max'] == expected


# ── the checker: Pass 3 refuses a replay under a different cap (DB-backed) ───────

def _write_geom_shard(path, scenario_ids):
    """Same framing as test_g08_replay_model_identity.py's helper of the same name."""
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
def _vehicle_parser(monkeypatch):
    """test_g08's _g08_parser shape, serving _near_vehicle_pair()."""
    module = _types.ModuleType('src.data.parser')

    class ScenarioParser:
        def __init__(self, raw):
            self.sid = raw.decode()

        def get_scenario_id(self):
            return self.sid

        def get_agent_states(self):
            return _near_vehicle_pair()[0]

        def get_agent_validity(self):
            return _near_vehicle_pair()[1]

        def get_agent_types(self):
            return _near_vehicle_pair()[2]

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


def _searched_and_stored(conn):
    """A REAL vehicle search on _near_vehicle_pair(), stored the way Pass 2 stores it."""
    from src.scoring import db
    from src.scoring.batch_scorer import _stress_one

    states, validity, types = _near_vehicle_pair()
    result = _stress_one(states, validity, types, 0,
                         de_kwargs={'popsize': 4, 'maxiter': 5, 'seed': 1})
    assert result['status'] == 'ok' and result['delta'] is not None, (
        f'fixture regressed: {result}'
    )
    fp = compute_scene_fingerprint(states, validity, types)
    db.upsert_scores(conn, [dict(scenario_id='s', shard='x', n_agents=2,
                                 min_ttc=9.0, min_pet=9.0, fragility_score=1.0,
                                 scene_fingerprint=fp, sdc_idx=0)])
    db.update_stress_results(conn, {'s': dict(result)})
    return result


def _export(conn, tmp_path, result):
    from src.scoring.export_geometry import export_shard_geometry
    shard = _write_geom_shard(tmp_path / 'shard.tfrecord', ['s'])
    return export_shard_geometry(conn, shard, ['s'],
                                 stress_results={'s': dict(result)}, verbose=False)


def _perturbed_rows(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM perturbed_paths WHERE scenario_id = 's'")
        return cur.fetchone()[0]


@requires_db
def test_a_vehicle_result_searched_under_the_current_cap_is_exported(
        bconn, tmp_path, _vehicle_parser):
    """
    THE NO-REGRESSION CASE, end to end: a real search and its export, under the same
    constants, must publish. This is also the only test in the suite that sends a
    vehicle result through export_shard_geometry's perturbed replay at all.
    """
    result = _searched_and_stored(bconn)
    summary = _export(bconn, tmp_path, result)

    assert summary['a_max_changed'] == [], summary['a_max_changed']
    assert summary['perturbed_written'] == 1, summary
    assert _perturbed_rows(bconn) == 1


@requires_db
def test_a_vehicle_result_is_refused_once_the_cap_has_moved_since_its_search(
        bconn, tmp_path, _vehicle_parser, monkeypatch):
    """
    THE PRIMARY REPRO. A real search records the cap it ran under; then the cap
    moves before Pass 3 (monkeypatched on the module replay_a_max reads, which is the
    constant the checker compares against). The stored delta must not be replayed
    under the new cap: refused into a_max_changed, with the baseline agents still
    exported, since they are logged data rather than a replay.
    """
    result = _searched_and_stored(bconn)
    recorded = result['search_provenance']['a_max']
    moved = recorded + 1.0
    monkeypatch.setattr(simulator, 'BICYCLE_A_MAX', moved)

    summary = _export(bconn, tmp_path, result)

    assert summary['a_max_changed'] == [{
        'scenario_id': 's', 'target_idx': 1,
        'recorded_a_max': recorded, 'recorded_a_max_source': 'search_provenance',
        'current_a_max': moved,
    }], summary['a_max_changed']
    assert summary['perturbed_written'] == 0, summary
    assert _perturbed_rows(bconn) == 0
    assert summary['exported'] == 1, 'the baseline agents must still be exported'
    assert summary['errors'] == [], summary['errors']


@requires_db
def test_a_legacy_vehicle_result_with_no_recorded_cap_is_refused(
        bconn, tmp_path, _vehicle_parser):
    """
    THE CASE G08'S RULE WOULD GET WRONG. A result with no 'a_max' key predates the
    key, so it was searched under A_MAX_BEFORE_PROVENANCE. G08 reads a missing key as
    "today's default", which here is the value this change moved to, and would
    therefore pass the stale vehicle row straight through. (The pedestrian half of
    this rule, a legacy row that MUST still export because linear_model.A_MAX never
    moved, is test_g08's own
    test_G08_a_legacy_result_with_no_search_provenance_still_uses_the_class_default,
    which now runs through this check.)
    """
    assert A_MAX_BEFORE_PROVENANCE != A_MAX, (
        'premise: this test exists because the bicycle cap has moved since the key '
        'was introduced'
    )
    result = _searched_and_stored(bconn)
    legacy = dict(result)
    legacy['search_provenance'] = {k: val for k, val in result['search_provenance'].items()
                                   if k != 'a_max'}

    summary = _export(bconn, tmp_path, legacy)

    assert summary['a_max_changed'] == [{
        'scenario_id': 's', 'target_idx': 1,
        'recorded_a_max': A_MAX_BEFORE_PROVENANCE,
        'recorded_a_max_source': 'predates_provenance',
        'current_a_max': A_MAX,
    }], summary['a_max_changed']
    assert summary['perturbed_written'] == 0, summary
    assert _perturbed_rows(bconn) == 0
