"""
test_notebook_drift_gate.py — the notebook and runbook after the drift-gate batch.

src/ now refuses a baseline replay in a fixed order (hard gate, speed step, 2 m drift
backstop) and records drift and the held speed excess on every attempt. What this file
pins on the notebook side:

  * 7g measures with BOTH soft gates off, and records each row's held speed excess,
    including on hard-gate rows, where it is recovered from the exception.
  * Every diagnostic construction that turns the drift gate off also turns the speed
    step off. Otherwise 7g's own assert fires, and 7g-iii/7g-iv silently lose the
    glitch scenarios they exist to study.
  * 7g reports the decision as made, and where the gates fall on the shard.
  * Pass 2's aggregate splits replay_infeasible by reason.
  * The runbook carries the next run's pass criteria and the cell 12 stop line.

Cells are located by content. Notebook code is executed from the REAL source, with
the shard cache replaced by analytic fixture scenes. No database, no Waymo package.

Run:
    ./venv/bin/python -m pytest tests/test_notebook_drift_gate.py -q
"""

import ast
import contextlib
import io
import json
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimization.perturbation_space import (
    BASELINE_DRIFT_REFUSE_M, SPEED_STEP_REFUSE_MPS,
)
from src.physics.bicycle_model import A_MAX, DT
from src.physics.simulator import TYPE_VEHICLE

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(PROJECT, 'notebooks', 'colab_validation_run.ipynb')
RUNBOOK = os.path.join(PROJECT, 'notebooks', 'COLAB_RUNBOOK.md')


def _cells():
    return json.load(open(NOTEBOOK))['cells']


def _cell_by_content(*required, cell_type='code'):
    hits = [''.join(c['source']) for c in _cells()
            if c['cell_type'] == cell_type and all(t in ''.join(c['source']) for t in required)]
    assert len(hits) == 1, f'expected one {cell_type} cell matching {required}, got {len(hits)}'
    return hits[0]


def _run(source, namespace):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, '<notebook cell>', 'exec'), namespace)
    return out.getvalue()


def _flat(text):
    return ' '.join(text.split())


# ── fixture scenes: straight vehicle tracks, the SDC parked 1 km away ────────────

T = 91


def _scene(logged_speed, position_speed):
    states = np.zeros((2, T, 7), dtype=np.float32)
    states[:, :, 5] = 4.5
    states[:, :, 6] = 2.0
    states[0, :, 1] = 1000.0
    p = np.asarray(position_speed, dtype=np.float64)
    states[1, :, 0] = np.concatenate([[0.0], np.cumsum(0.5 * (p[:-1] + p[1:]) * DT)])
    states[1, :, 2] = logged_speed
    return {'states': states, 'validity': np.ones((2, T), dtype=bool),
            'types': np.full(2, TYPE_VEHICLE), 'sdc_idx': 0}


def _replayed(logged):
    out = np.array(logged, dtype=np.float64)
    for k in range(len(out) - 1):
        out[k + 1] = out[k] + np.clip((logged[k + 1] - logged[k]) / DT, -A_MAX, A_MAX) * DT
    return out


def _shard():
    clean = np.full(T, 10.0)
    step = clean.copy()
    step[21:] += 3.0                                   # unreturned: held 3.0 - A_MAX*DT
    creep = np.full(T, 1.0 / ((T - 1) * DT))           # parked, logged at a crawl: 1 m drift
    scenes = {
        'clean': _scene(clean, clean),
        'step_only': _scene(step, _replayed(step)),    # refused for the step, ~0 drift
        'step_and_drift': _scene(step, step),          # the step becomes metres of drift
        'creep': _scene(creep, np.zeros(T)),           # 1 m: under the backstop
    }
    collide = _scene(step, _replayed(step))            # hard gate: SDC on the logged track
    collide['states'][0, :, 0] = collide['states'][1, :, 0] + 1.0
    collide['states'][0, :, 1] = 0.0
    scenes['collide'] = collide
    return [dict(scene, scenario_id=sid) for sid, scene in scenes.items()]


EXCESS = 3.0 - A_MAX * DT


# ── 7g ───────────────────────────────────────────────────────────────────────────

def _7g():
    return _cell_by_content('BASELINE REPLAY DRIFT', 'rows.append')


@pytest.fixture(scope='module')
def ran_7g():
    namespace = {'shard_cache': _shard()}
    out = _run(_7g(), namespace)
    return out, {r['scenario_id']: r for r in namespace['rows']}


def test_7g_measures_every_scenario_with_both_soft_gates_off(ran_7g):
    """A speed-step or drift refusal here would mean a soft gate is still armed."""
    _, rows = ran_7g
    assert set(rows) == {'clean', 'step_only', 'step_and_drift', 'creep', 'collide'}
    assert rows['collide']['collides'] is True
    assert rows['step_and_drift']['drift'] > BASELINE_DRIFT_REFUSE_M


def test_7g_records_the_held_speed_excess_on_every_row_including_hard_gate_rows(ran_7g):
    _, rows = ran_7g
    assert rows['clean']['held_speed_excess'] == pytest.approx(0.0, abs=1e-6)
    assert rows['step_only']['held_speed_excess'] == pytest.approx(EXCESS, abs=1e-4)
    assert rows['collide']['held_speed_excess'] == pytest.approx(EXCESS, abs=1e-4), (
        'the hard-gate row must recover the excess from the exception'
    )


def test_7g_reports_where_the_gates_fall(ran_7g):
    out, _ = ran_7g
    flat = _flat(out)
    assert 'whether max_baseline_drift stays at 0.5 m' not in flat
    assert '<- current default' not in out
    assert re.search(r'drift <= 2\.00 m :.*<- backstop', out), 'the 2 m backstop is not marked'
    # step_only and step_and_drift hold the step; the hard-gate row is not counted.
    assert re.search(r'speed-step refusals: 2 of 4\b', flat), flat
    assert re.search(r'of which also drift past the backstop: 1\b', flat), flat
    assert 'DECIDED' in out


def test_every_gates_off_construction_also_turns_the_speed_step_off():
    found = 0
    for index, cell in enumerate(_cells()):
        if cell['cell_type'] != 'code':
            continue
        # Colab shell and magic lines (!pip, %cd) are not Python.
        source = ''.join(line for line in cell['source'] if not line.lstrip().startswith(('!', '%')))
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'PerturbationSpace'):
                continue
            kw = {k.arg: k.value for k in node.keywords}
            if 'max_baseline_drift' in kw and isinstance(kw['max_baseline_drift'], ast.Constant) \
                    and kw['max_baseline_drift'].value is None:
                found += 1
                assert 'max_speed_step' in kw and kw['max_speed_step'].value is None, (
                    f'cell {index}: max_baseline_drift=None without max_speed_step=None'
                )
    assert found == 3, f'expected the three diagnostic constructions (7g, 7g-iii, 7g-iv), got {found}'


# ── Pass 2's aggregate ───────────────────────────────────────────────────────────

def test_the_pass_2_aggregate_splits_replay_infeasible_by_reason():
    sample = {
        'hit': {'status': 'ok', 'outcome': 'collision_found', 'collision': True,
                'min_perturbation': 0.5, 'collision_timestep': 3, 'target_idx': 1},
        'r1': {'status': 'replay_infeasible', 'reason': 'speed_step', 'target_idx': 1},
        'r2': {'status': 'replay_infeasible', 'reason': 'drift', 'target_idx': 1},
        'r3': {'status': 'replay_infeasible', 'reason': 'drift', 'target_idx': 1},
        'r4': {'status': 'replay_infeasible', 'target_idx': 1},
    }
    env = {'shard_cache': [], 'np': np, 'stress_results': sample,
           'ids_to_test': list(sample), 'pass2_elapsed': 1.0}
    out = _run(_cell_by_content('PASS 2 AGGREGATES', 'n_no_challenger'), env)
    assert env['replay_infeasible_by_reason'] == {'speed_step': 1, 'drift': 2, 'unrecorded': 1}
    assert 'replay_infeasible by reason: drift=2, speed_step=1, unrecorded=1' in out


# ── the markdown and the runbook ─────────────────────────────────────────────────

def test_the_7g_markdown_reports_the_decision_as_made():
    md = _flat(_cell_by_content('## 7g. Baseline replay drift', cell_type='markdown'))
    assert 'before the 0.5 m default is defended as final' not in md
    assert '2 m backstop' in md and 'max_speed_step=None' in md


def _runbook():
    return _flat(open(RUNBOOK).read())


def test_the_runbook_says_to_stop_if_cell_12_is_not_the_render_host():
    runbook = _runbook()
    assert 'External database: host=' in runbook
    assert re.search(r'Postgres running, role=avi`?.{0,40}stop', runbook, re.I), (
        'the runbook does not say to stop when cell 12 prints the local database'
    )


def test_the_runbook_lists_the_next_runs_pass_criteria():
    runbook = _runbook()
    for criterion in ('18 collisions', '58d5f1b9', '2.562', '504dd390', '3.464',
                      'replay_refused: 0', '0 speed-step refusals'):
        assert criterion in runbook, f'pass criterion missing: {criterion}'
    for sid, norm in (('8ec2910b', '0.4034'), ('c302c905', '0.1109'), ('a6bf1ade', '0.0101'),
                      ('b1e5a345', '0.0204'), ('19043d68', '0.0197'), ('38c703d6', '0.0153')):
        assert re.search(sid + r'.{0,60}' + re.escape(norm), runbook), (
            f'{sid} is not listed with its predicted norm {norm}'
        )


def test_the_runbook_no_longer_treats_the_drift_default_as_open():
    runbook = _runbook()
    assert '`max_baseline_drift` default — from 7g' not in runbook
    assert 'Decision this feeds — the `max_baseline_drift` default' not in runbook
