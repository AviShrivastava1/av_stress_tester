"""Independent regression expectations. Known defects FAIL on the uploaded code.

Set AV_AUDIT_PROJECT to the extracted project root before running pytest.
Tests deliberately use short analytical scenes, not the original tests' geometry.
"""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import pickle
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
PROJECT = Path(os.environ['AV_AUDIT_PROJECT']).resolve()
sys.path.insert(0, str(PROJECT))

from src.optimization.perturbation_space import PerturbationSpace
from src.danger.collision_detector import check_collision_trajectory, check_any_collision
from src.danger.pet_engine import compute_pet_pair
from src.danger.ttc_engine import compute_ttc_pair, TTC_INFINITY
from src.data.loader import ShardLoader


def scene(n=2, t=10):
    states = np.zeros((n, t, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    return states, np.ones((n, t), dtype=bool), np.ones(n, dtype=int)


def test_B01_late_start_preserves_first_observed_position():
    s, v, types = scene()
    s[0, :, 1] = 1000
    s[1, :, 0] = 100 + np.arange(10)
    s[1, :, 2] = 10
    v[1, :3] = False
    space = PerturbationSpace(s, v, types, 0, 1)
    perturbed = space.apply(np.zeros(4))
    assert perturbed[1, 3, 0] == s[1, 3, 0], (
        f'First valid frame 3: logged x={s[1,3,0]}, zero-delta x={perturbed[1,3,0]}'
    )


def test_B02_dimensions_come_from_a_valid_observation():
    s, v, types = scene()
    s[0, :, 1] = 1000
    s[1, :, 0] = 100 + np.arange(10)
    s[1, :, 2] = 10
    v[1, :3] = False
    s[1, :3, 5:7] = 0
    space = PerturbationSpace(s, v, types, 0, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        perturbed = space.apply(np.zeros(4))
    assert np.isfinite(perturbed[1, v[1]]).all(), (
        f'Valid positions are nonfinite; selected wheelbase={space.simulator.wheelbase}'
    )


def test_B03_zero_perturbation_does_not_create_a_replay_collision():
    s, v, types = scene(t=91)
    t = np.arange(91) * .1
    s[0, :, 0] = .5 * t*t
    s[1, :, 0] = 4.6 + .5 * t*t
    s[:, :, 2] = t
    assert check_collision_trajectory(s, v, 0, 1) == (False, -1)
    space = PerturbationSpace(s, v, types, 0, 1)
    replay = space.apply(np.zeros(4))
    collision, hit = check_collision_trajectory(replay, v, 0, 1)
    drift = float(replay[1, -1, 0] - s[1, -1, 0])
    assert not collision, f'Zero-delta collision at frame {hit}; end drift={drift:.6f} m'


def test_B04_one_unsuccessful_challenger_search_is_not_a_safety_certificate():
    from src.scoring.batch_scorer import _stress_one
    from src.api.routes import _summary_fields
    s, v, types = scene(n=3, t=2)
    s[1, :, 1] = 2.6  # nearest challenger, cannot reach the SDC in this short horizon
    s[2, :, 0] = 4.6  # farther challenger, a small speed offset CAN produce collision
    s[2, :, 4] = np.pi
    other = PerturbationSpace(s, v, types, 0, 2)
    assert check_collision_trajectory(other.apply([3,0,0,0]), v, 0, 2)[0]
    result = _stress_one(s, v, types, 0, de_kwargs={'popsize':4, 'maxiter':10, 'seed':1})
    assert result['target_idx'] == 1 and not result['collision']
    row = dict(scenario_id='other_agent_collision', shard='synthetic', n_agents=3,
               min_ttc=999., min_pet=999., fragility_score=0., min_perturbation=None,
               collision_timestep=None, stress_method='de', stress_tested_at='completed')
    assert not _summary_fields(row)['robustly_safe'], (
        'API declares robustly_safe=True despite a verified in-bounds collision with agent 2'
    )


def test_B05_corrupt_next_record_does_not_overwrite_previous_success(monkeypatch):
    from src.scoring import batch_scorer
    import src.data.parser as parser_module
    from waymo_open_dataset.protos import scenario_pb2
    records = [scenario_pb2.Scenario(scenario_id='A').SerializeToString(),
               b'\xff', scenario_pb2.Scenario(scenario_id='B').SerializeToString()]
    monkeypatch.setattr('src.data.loader.ShardLoader', lambda _: iter(records))
    monkeypatch.setattr(batch_scorer, '_stress_one', lambda *a, **k: {'status':'ok'})
    result = batch_scorer.stress_test_scenarios('synthetic', ['A','B'], verbose=False)
    assert result['A']['status'] == 'ok', f'Good record A was overwritten: {result}'


def test_B06_pet_keeps_separate_occupancy_visits_separate():
    s, v, _ = scene(t=7)
    v[0] = [1,1,0,0,0,1,1]
    v[1] = [0,0,0,1,0,0,0]
    assert not (v[0] & v[1]).any()
    pet = compute_pet_pair(s, v, 0, 1)
    assert pet == pytest.approx(.2), f'No shared observed frame, but PET={pet}'


def test_B07_ttc_is_infinite_for_a_constant_velocity_miss():
    # A stays at (0,0); B travels west along y=3. Each circle has radius 1.
    # The minimum possible center separation is 3, greater than radius sum 2.
    ttc = compute_ttc_pair(0,0,0,0,1,10,3,-5,0,1)
    assert ttc == TTC_INFINITY, f'Nonintersecting circle paths give TTC={ttc:.9f} s'


@pytest.mark.parametrize('base_speed, offset', [(.5,-3.), (39.,3.)])
def test_B08_initial_speed_obeys_the_same_bounds_as_later_states(base_speed, offset):
    s, v, types = scene(t=2)
    s[1, :, 0] = 10
    # DELIBERATE, NARROW EDIT to the audit fixture's SCENE SETUP — not its assertion.
    # As shipped, this fixture held position constant across both frames while
    # declaring a velocity of `base_speed`, i.e. at base_speed=39 it described a car
    # travelling 39 m/s that never moves. That is internally inconsistent, and the
    # replay-fidelity gate added for audit B03 correctly refuses it
    # (baseline_replay_error = 3.9 m, well past the 0.5 m default), raising before the
    # test could reach what it is actually about.
    #
    # Giving frame 1 the position its own logged velocity implies makes the scene
    # self-consistent; baseline_replay_error drops to 0.0 for both parametrizations.
    # The assertion below is UNCHANGED and still exercises exactly the B08 clamp
    # (39 + 3 -> 42 -> 40). The rejected alternative was passing
    # max_baseline_drift=None here, which also goes green but gets there by switching
    # off the safety check this same change installed.
    s[1, 1, 0] = 10 + base_speed * 0.1
    s[1, :, 2] = base_speed
    space = PerturbationSpace(s, v, types, 0, 1)
    perturbed = space.apply([offset,0,0,0])
    initial_speed = float(perturbed[1, 0, 2])
    assert 0 <= initial_speed <= 40, f'Initial signed speed={initial_speed}; bounds are [0,40]'


def test_B09_refinement_retains_an_already_verified_feasible_candidate():
    from src.optimization.autograd_optimizer import refine_scenario
    s, v, types = scene(t=10)
    s[1, :, 1] = 2.1
    space = PerturbationSpace(s, v, types, 0, 1)
    initial = np.array([0,.15,0,0])
    assert check_collision_trajectory(space.apply(initial), v, 0, 1)[0]
    result = refine_scenario(space, delta_init=initial, n_iters=100)
    assert result['collision'], (
        f'Refinement discarded a verified collision: delta={result["delta"]}, '
        f'smooth_margin={result["smooth_margin"]}'
    )


def test_B10_de_objective_can_be_serialized_for_process_workers(monkeypatch):
    import src.optimization.scipy_optimizer as optimizer
    s, v, types = scene(t=2)
    s[1, :, 0] = 20
    captured = {}
    def capture_objective(function, bounds, **kwargs):
        captured['function'] = function
        return SimpleNamespace(x=np.zeros(4), nit=0, nfev=0)
    monkeypatch.setattr(optimizer, 'differential_evolution', capture_objective)
    optimizer.optimize_scenario(PerturbationSpace(s,v,types,0,1), workers=2)
    pickle.dumps(captured['function'])


@pytest.mark.parametrize('contents', [
    b'abc',  # incomplete header silently looks like EOF
    struct.pack('<Q',10) + b'\0'*4 + b'ab',  # advertised payload is truncated
    struct.pack('<Q',3) + b'\0'*4 + b'abc' + b'\0'*4,  # invalid CRCs accepted
], ids=['partial_header','truncated_payload','invalid_crc'])
def test_B11_loader_rejects_corrupt_tfrecord_framing(tmp_path, contents):
    path = tmp_path/'corrupt.tfrecord'
    path.write_bytes(contents)
    with pytest.raises((ValueError, OSError, EOFError)):
        list(ShardLoader(str(path)))


def test_B16_any_collision_returns_the_earliest_time_across_agents():
    s, v, _ = scene(n=3, t=5)
    s[1:, :, 0] = 100
    s[1, 4, 0] = 0
    s[2, 1, 0] = 0
    assert check_any_collision(s,v,0) == (True,2,1)


def test_B19_notebook_geometry_validation_rejects_missing_valid_agents():
    notebook = json.loads((PROJECT/'notebooks/colab_validation_run.ipynb').read_text())
    code = ''.join(notebook['cells'][39]['source'])
    class Parser:
        def __init__(self, raw): pass
        def get_scenario_id(self): return 'A'
        def get_agent_validity(self): return np.ones((2,3), dtype=bool)
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, *args): pass
        def fetchall(self): return []
    env = dict(np=np, ShardLoader=lambda _: [b'A'], SHARD_PATH='synthetic',
               ScenarioParser=Parser, ids_to_test=['A'],
               conn=SimpleNamespace(cursor=Cursor), summary={'agents_skipped':0})
    with pytest.raises(AssertionError), contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, 'notebook_cell_39', 'exec'), env)


def test_B20_one_sided_bounds_have_a_meaningful_normalized_budget():
    s, v, types = scene(t=2)
    s[1,:,0] = 20
    bounds = np.array([[-3,0],[-.2,.2],[-1.5,1.5],[-.1,.1]])
    space = PerturbationSpace(s,v,types,0,1,bounds=bounds)
    norm = space.weighted_norm([-3,0,0,0])
    assert norm == pytest.approx(1.), f'A full allowed speed budget has norm {norm}, expected 1'


def test_B08_linear_acceleration_respects_the_declared_magnitude_limit():
    from src.physics.linear_model import linear_step, A_MAX
    result = linear_step(np.zeros(4),np.array([A_MAX,A_MAX]))
    acceleration = float(np.linalg.norm(result[2:])/.1)
    assert acceleration <= A_MAX + 1e-6, f'Acceleration magnitude={acceleration}, declared max={A_MAX}'


def test_control_zero_delta_replays_a_valid_constant_speed_track():
    s, v, types = scene()
    s[0, :, 1] = 100
    s[1, :, 0] = np.arange(10)
    s[1, :, 2] = 10
    original = s.copy()
    space = PerturbationSpace(s,v,types,0,1)
    result = space.apply(np.zeros(4))
    np.testing.assert_allclose(result, s, atol=1e-6)
    np.testing.assert_array_equal(s, original)


def test_control_circle_ttc_on_a_direct_collision_course():
    assert compute_ttc_pair(0,0,0,0,1,10,0,-5,0,1) == pytest.approx(1.6)


def test_control_parser_preserves_valid_states_and_flags():
    from src.data.parser import ScenarioParser
    from waymo_open_dataset.protos import scenario_pb2
    scenario = scenario_pb2.Scenario(scenario_id='parser_control', sdc_track_index=0)
    track = scenario.tracks.add(id=1, object_type=1)
    track.states.add(center_x=7., center_y=3., velocity_x=2., heading=.25,
                     length=4.5, width=2., valid=True)
    track.states.add(valid=False)
    parser = ScenarioParser(scenario.SerializeToString())
    assert parser.get_scenario_id() == 'parser_control'
    assert parser.get_sdc_index() == 0
    np.testing.assert_allclose(parser.get_agent_states()[0,0], [7,3,2,0,.25,4.5,2])
    assert parser.get_agent_validity()[0,0] and not parser.get_agent_validity()[0,1]
