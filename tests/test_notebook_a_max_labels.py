"""
test_notebook_a_max_labels.py — the notebook and runbook no longer describe A_MAX=5.

bicycle_model.A_MAX moved from 5.0 to 12.0 (ccab66d). The 7g-v cells compute with the
IMPORTED constant, so their numbers follow the change automatically; their labels,
variable names and a few hardcoded conclusions did not. A re-run under 12.0 would then
print numbers computed at 12 next to text saying 5, and name "the scenarios that did
not close" from the old run rather than this one. Each test below pins one of those.

Cells are located by content, never by index (this project's rule after repeated
index-drift breakage). Where a test executes notebook or runbook code, it executes the
REAL source extracted from the file, with only its external dependencies stubbed.

No database, no Waymo package.

Run:
    ./venv/bin/python -m pytest tests/test_notebook_a_max_labels.py -q
"""

import ast
import contextlib
import io
import json
import os
import pickle
import re
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.physics.bicycle_model import A_MAX

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(PROJECT, 'notebooks', 'colab_validation_run.ipynb')
RUNBOOK = os.path.join(PROJECT, 'notebooks', 'COLAB_RUNBOOK.md')


def _cells():
    return json.load(open(NOTEBOOK))['cells']


def _cell_by_content(*required, cell_type='code'):
    hits = [''.join(c['source']) for c in _cells()
            if c['cell_type'] == cell_type
            and all(token in ''.join(c['source']) for token in required)]
    assert len(hits) == 1, f'expected exactly one {cell_type} cell matching {required}, got {len(hits)}'
    return hits[0]


def _7g_v_code():
    return _cell_by_content('7g-v. Shard-wide A_MAX calibration', 'GLITCH_THRESHOLD')


def _7g_v_markdown():
    return _cell_by_content('## 7g-v.', cell_type='markdown')


def _run(source, namespace):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, '<notebook cell>', 'exec'), namespace)
    return out.getvalue()


# ── item 1: labels and names follow the imported constant ───────────────────────

def test_the_proxy_label_prints_the_imported_cap_not_a_literal_5():
    source = _7g_v_code()
    assert 'max-norm proxy (A=5' not in source
    assert re.search(r'max-norm proxy \(A=\{A_MAX', source), (
        'the proxy line should print the imported A_MAX'
    )


def test_no_variable_is_named_after_the_old_cap():
    tree = ast.parse(_7g_v_code())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    stale = sorted(n for n in names if re.fullmatch(r'v5b?(_ts)?', n))
    assert not stale, f'names still carry the old cap: {stale}'


def test_the_7g_v_markdown_does_not_claim_a_cap_of_5():
    markdown = _7g_v_markdown()
    assert not re.search(r'A_MAX\s*=\s*5\b', markdown), 'markdown still states A_MAX=5'
    assert 'A=5)' not in markdown, 'markdown still labels the proxy with A=5'


def test_the_glitch_threshold_is_documented_as_independent_of_the_cap():
    line = next(l for l in _7g_v_code().splitlines() if l.startswith('GLITCH_THRESHOLD'))
    assert '#' in line and 'A_MAX' in line.split('#', 1)[1], (
        f'GLITCH_THRESHOLD needs a comment saying it is not coupled to A_MAX: {line!r}'
    )


# ── item 2: item 4 names this run's worst ten, not the old run's two ────────────

def test_no_scenario_id_from_the_old_run_is_hardcoded():
    for source in (_7g_v_code(), _7g_v_markdown()):
        assert '8fd0' not in source and 'ef85' not in source


def _item_4_block():
    source = _7g_v_code()
    marker = '# ── Item 4'
    assert source.count(marker) == 1, 'item 4 block not found exactly once'
    return source[source.index(marker):]


def test_item_4_lists_every_worst_scenario_sorted_by_unexplained_fraction():
    mechanism = [
        {'scenario_id': 'w-low', 'group': 'worst', 'is_vehicle': True, 'unexplained_iv': 0.02},
        {'scenario_id': 'w-high', 'group': 'worst', 'is_vehicle': True, 'unexplained_iv': 0.61},
        {'scenario_id': 'w-mid', 'group': 'worst', 'is_vehicle': True, 'unexplained_iv': 0.30},
        {'scenario_id': 'w-floor', 'group': 'worst', 'is_vehicle': True, 'unexplained_iv': None},
        {'scenario_id': 'w-ped', 'group': 'worst', 'is_vehicle': False},
        {'scenario_id': 'm-1', 'group': 'middle', 'is_vehicle': True, 'unexplained_iv': 0.99},
    ]
    out = _run(_item_4_block(), {'mechanism': mechanism, 'A_MAX': A_MAX})

    order = [sid for sid in ('w-high', 'w-mid', 'w-low', 'w-floor', 'w-ped') if sid in out]
    assert order == ['w-high', 'w-mid', 'w-low', 'w-floor', 'w-ped']
    positions = [out.index(sid) for sid in order]
    assert positions == sorted(positions), 'worst scenarios are not printed largest-first'
    assert 'm-1' not in out, 'a middle-group scenario was listed as worst'
    assert '61.00%' in out and '2.00%' in out
    assert 'not a vehicle challenger' in out
    assert 'not computed in 7g-iv' in out
    assert 'noise floor' not in out, 'the label names a cause 7g-v did not check'


# ── item 5: the Pass 3 summary prints every refusal bucket ──────────────────────

def test_the_pass_3_summary_prints_all_five_refusal_buckets(monkeypatch):
    import src.scoring.export_geometry as export_geometry

    summary = {
        'exported': 3, 'agents_written': 40, 'agents_skipped': 2, 'perturbed_written': 1,
        'perturbed_stale': [{'scenario_id': 'stale-1', 'attempted_run_id': 'a', 'current_run_id': 'b'}],
        'scene_changed': [{'scenario_id': 'scene-1', 'stored_fingerprint': 'f1', 'computed_fingerprint': 'f2'}],
        'sdc_changed': [{'scenario_id': 'sdc-1', 'stored_sdc_idx': 0, 'computed_sdc_idx': 3}],
        'a_max_changed': [{'scenario_id': 'amax-1', 'target_idx': 2, 'recorded_a_max': 5.0,
                           'recorded_a_max_source': 'predates_provenance', 'current_a_max': 12.0}],
        'replay_refused': [{'scenario_id': 'rr-1', 'target_idx': 1, 'reason': 'drift',
                            'baseline_replay_error': 0.9, 'held_speed_excess': 0.0,
                            'max_baseline_drift': 0.5, 'max_speed_step': None,
                            'gate_settings_source': 'predates_provenance'}],
        'errors': [],
    }
    monkeypatch.setattr(export_geometry, 'export_shard_geometry', lambda *a, **k: summary)
    source = _cell_by_content('PASS 3 SUMMARY')
    out = _run(source, {'time': time, 'conn': None, 'SHARD_PATH': 'x',
                        'ids_to_test': [], 'stress_results': {},
                        'refresh_connection': lambda: None})

    for bucket, sid in [('perturbed_stale', 'stale-1'), ('scene_changed', 'scene-1'),
                        ('sdc_changed', 'sdc-1'), ('a_max_changed', 'amax-1'),
                        ('replay_refused', 'rr-1')]:
        line = next((l for l in out.splitlines() if l.startswith(f'{bucket}:')), None)
        assert line is not None, f'{bucket} is not printed'
        assert line.split(':', 1)[1].strip().startswith('1'), f'{bucket} count wrong: {line!r}'
        assert sid in out, f'{bucket} does not name its scenario'
    assert 'searched under 5.0' in out and '12.0' in out


# ── items 3 and 6: the runbook ──────────────────────────────────────────────────

def test_the_hard_stop_fixture_no_longer_claims_to_exceed_the_cap():
    runbook = open(RUNBOOK).read()
    sentence = re.search(r'A hard-stop fixture[^.]*\.', runbook, re.S)
    assert sentence, 'the hard-stop fixture sentence is gone'
    assert 'exceeding' not in sentence.group(0), sentence.group(0)


def _runbook_python_block(*required):
    blocks = re.findall(r'```python\n(.*?)```', open(RUNBOOK).read(), re.S)
    hits = [b for b in blocks if all(t in b for t in required)]
    assert len(hits) == 1, f'expected one runbook code block with {required}, got {len(hits)}'
    return hits[0]


CHECKPOINT = '/content/drive/MyDrive/av_stress_checkpoint.pkl'


def test_the_checkpoint_records_the_cap_it_was_computed_under(tmp_path):
    save = _runbook_python_block('pickle.dump', CHECKPOINT).replace(CHECKPOINT, str(tmp_path / 'c.pkl'))
    _run(save, {'rows': [1], 'decompositions': [2], 'mechanism': [3], 'A_MAX': A_MAX})
    saved = pickle.load(open(tmp_path / 'c.pkl', 'rb'))
    assert saved['a_max'] == A_MAX


@pytest.mark.parametrize('stored,refused', [
    ({'a_max': A_MAX}, False),
    ({'a_max': 5.0}, True),
    ({}, True),
])
def test_resuming_refuses_a_checkpoint_from_another_cap(tmp_path, stored, refused):
    path = tmp_path / 'c.pkl'
    pickle.dump({'rows': ['r'], 'decompositions': ['d'], 'mechanism': ['m'], **stored},
                open(path, 'wb'))
    load = _runbook_python_block('pickle.load', CHECKPOINT).replace(CHECKPOINT, str(path))
    namespace = {'A_MAX': A_MAX}
    if refused:
        with pytest.raises(AssertionError):
            _run(load, namespace)
        assert 'rows' not in namespace, 'a refused checkpoint still bound its names'
    else:
        _run(load, namespace)
        assert namespace['rows'] == ['r'] and namespace['mechanism'] == ['m']
