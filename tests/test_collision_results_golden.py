"""
The results-only golden record of the collision results the public API serves (tests/data/public_collision_results_2026-10-18.json), and
the tool that refreshes it (tools/refresh_collision_results.py).

Pinned here:

  * the record's shape: for each of the 18 collision results, exactly scenario id, target_idx, the four components of the perturbation,
    which model's units they carry, the weighted norm, the collision frame and the run id of the stored result, and nothing else;
  * its values: every field has the type and range the API serves, the ids and collision frames agree with the other committed fixtures,
    and a hash of the rows catches any edit by hand (a refresh changes it on purpose; see the constant below);
  * that it holds no scene geometry, no trajectory and no coordinate of any agent;
  * the tool: it reads only what it needs from a response that carries tracks, drops the tracks, lists only collision results, follows the
    pages, and refuses a base URL that is not on this machine. No test here opens a network connection.

The record is refreshed by hand with tools/refresh_collision_results.py against a local copy of the API (see its docstring), then the
hash below is updated in the same commit.
"""

import hashlib
import json
import math
import os
import re
import sys

import pytest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from tools import refresh_collision_results as tool

GOLDEN = os.path.join(PROJECT, 'tests', 'data', 'public_collision_results_2026-10-18.json')
SCORES = os.path.join(PROJECT, 'tests', 'data', 'public_scores_2026-10-15.json')
FRAMES = os.path.join(PROJECT, 'frontend', 'src', 'test', 'realCollisionFrames.json')

#: sha256 of json.dumps(rows, sort_keys=True, separators=(',', ':')); changes when the record is refreshed, which is the point
ROWS_SHA256 = '2f87b78a2b58e874b130be20c25e64b08c6d24793963c716b0b7b3eb1eec1c7d'

#: keys that would mean geometry or a track is in the record
GEOMETRY_KEYS = {'path', 'x', 'y', 'xs', 'ys', 'headings', 'timesteps', 'lengths_m', 'widths_m', 'baseline', 'perturbed', 'states', 'validity',
                 'agents', 'tracks', 'sdc', 'points', 'vertices', 'polyline', 'position', 'positions'}


def _load():
    with open(GOLDEN, encoding='utf-8') as fh:
        return json.load(fh)


def _walk(value):
    """Every key of every object in a parsed JSON value."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


# ── the record ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_the_record_has_a_note_a_source_and_eighteen_rows_and_nothing_else():
    record = _load()
    assert list(record) == ['note', 'source', 'rows'] and len(record['rows']) == 18
    assert record['note'] == tool.NOTE and 'no coordinate of any agent' in record['note']
    assert 'local scratch database' in record['source'] and 'GET requests only' in record['source']


def test_every_row_has_exactly_the_result_fields_in_order():
    for row in _load()['rows']:
        assert tuple(row) == tool.FIELDS


def test_the_rows_are_sorted_by_scenario_id_and_the_ids_are_unique():
    ids = [r['scenario_id'] for r in _load()['rows']]
    assert ids == sorted(ids) and len(set(ids)) == 18
    assert all(re.fullmatch(r'[0-9a-f]{16}', i) for i in ids)


def test_each_field_has_the_type_and_range_the_api_serves():
    for r in _load()['rows']:
        assert _is_int(r['target_idx']) and r['target_idx'] >= 0
        assert isinstance(r['delta'], list) and len(r['delta']) == 4
        assert all(isinstance(v, float) and math.isfinite(v) for v in r['delta'])
        assert r['delta_parameterization'] in ('bicycle', 'linear')
        assert isinstance(r['min_perturbation'], float) and math.isfinite(r['min_perturbation']) and r['min_perturbation'] > 0
        assert _is_int(r['collision_timestep']) and 0 <= r['collision_timestep'] <= 90
        assert re.fullmatch(r'[0-9a-f]{16}', r['stress_run_id'])


def test_the_ids_and_collision_frames_agree_with_the_other_committed_fixtures():
    rows = _load()['rows']
    with open(FRAMES, encoding='utf-8') as fh:
        frames = {s['scenario_id']: s['collision_timestep'] for s in json.load(fh)['scenarios']}
    assert {r['scenario_id']: r['collision_timestep'] for r in rows} == frames
    with open(SCORES, encoding='utf-8') as fh:
        scored = {r['scenario_id'] for r in json.load(fh)['rows']}
    assert {r['scenario_id'] for r in rows} <= scored


def test_the_rows_are_exactly_the_recorded_ones():
    rows = _load()['rows']
    assert hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == ROWS_SHA256


def test_the_first_row_is_pinned_verbatim():
    assert _load()['rows'][0] == {
        'scenario_id': '1492befc8cd5b7cd', 'target_idx': 69,
        'delta': [-0.005721431691199541, 0.0008577097323723137, -0.0036641627084463835, 0.0021800962276756763],
        'delta_parameterization': 'bicycle', 'min_perturbation': 0.02243385650217533, 'collision_timestep': 55, 'stress_run_id': '939a8b42724f462e'}


def test_there_is_no_geometry_trajectory_or_coordinate_in_it():
    record = _load()
    assert not GEOMETRY_KEYS & set(_walk(record))
    for r in record['rows']:
        for key, value in r.items():
            if key != 'delta':
                assert not isinstance(value, (list, dict)), key
    assert os.path.getsize(GOLDEN) < 20_000


# ── the tool ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

BASE = 'http://127.0.0.1:9'


def _fake_api(extra_items=()):
    """A fake API: two pages of the list (one collision result, one other result, one more collision result), and tracks in /perturbed."""
    items_1 = [{'scenario_id': 'bbbb000000000001', 'stress_outcome': 'collision_found'}, {'scenario_id': 'aaaa000000000002', 'stress_outcome': 'no_collision_found'}]
    items_2 = [{'scenario_id': 'aaaa000000000001', 'stress_outcome': 'collision_found'}, *extra_items]
    calls = []

    def get_json(url):
        calls.append(url)
        if url.startswith(f'{BASE}/scenarios?'):
            return {'items': items_2, 'next_cursor': None} if 'cursor=' in url else {'items': items_1, 'next_cursor': 'abc=='}
        sid = url[len(BASE) + len('/scenarios/'):].split('/')[0]
        if url.endswith('/provenance'):
            return {'scenario_id': sid, 'stress_run_id': '0123456789abcdef', 'search': {'delta_parameterization': 'linear', 'de_seed': 3}}
        if url.endswith('/perturbed'):
            return {'scenario_id': sid, 'target_idx': 7, 'delta': [9, 9, 9, 9], 'baseline': {'path': [[1.5, 2.5]], 'timesteps': [0.0]},
                    'perturbed': {'path': [[3.5, 4.5]], 'timesteps': [0.0]}}
        return {'scenario_id': sid, 'delta': [0.1, 0.2, 0.3, 0.4], 'min_perturbation': 0.5, 'collision_timestep': 12, 'stress_outcome': 'collision_found'}
    return get_json, calls


def test_the_tool_lists_only_collision_results_follows_the_pages_and_sorts_by_id():
    get_json, calls = _fake_api()
    record = tool.build_record(get_json, BASE + '/', 'a test')
    assert [r['scenario_id'] for r in record['rows']] == ['aaaa000000000001', 'bbbb000000000001']
    assert record['source'] == 'a test' and record['note'] == tool.NOTE
    assert any('cursor=abc%3D%3D' in c for c in calls)
    assert [c for c in calls if '/scenarios?' in c][0] == f'{BASE}/scenarios?limit=100&stress_tested_only=true'
    assert tool.PAGE == 100
    assert all(c.startswith(BASE) for c in calls) and not any('//scenarios' in c for c in calls)


def test_the_tool_takes_the_result_fields_and_nothing_of_the_tracks_the_perturbed_response_carries():
    get_json, _ = _fake_api()
    [row, _other] = tool.build_record(get_json, BASE, 's')['rows']
    assert row == {'scenario_id': 'aaaa000000000001', 'target_idx': 7, 'delta': [0.1, 0.2, 0.3, 0.4], 'delta_parameterization': 'linear',
                   'min_perturbation': 0.5, 'collision_timestep': 12, 'stress_run_id': '0123456789abcdef'}
    text = json.dumps(tool.build_record(get_json, BASE, 's'))
    assert '1.5' not in text and '4.5' not in text and not GEOMETRY_KEYS & set(_walk(json.loads(text)))


def test_a_result_with_no_search_record_has_no_model_rather_than_a_guess():
    get_json, _ = _fake_api()

    def bare(url):
        out = get_json(url)
        if url.endswith('/provenance'):
            out = {'scenario_id': 'x', 'stress_run_id': 'ffffffffffffffff', 'search': None}
        return out
    assert tool.build_record(bare, BASE, 's')['rows'][0]['delta_parameterization'] is None


@pytest.mark.parametrize('url,local', [
    ('http://127.0.0.1:8123', True), ('http://localhost:8123/', True), ('http://[::1]:8000', True), ('http://127.1.2.3', True),
    ('https://example.com', False), ('http://10.0.0.5:8000', False), ('https://api.example.invalid/scenarios', False), ('http://localhost.example.com', False),
    ('not a url', False), ('', False),
])
def test_only_a_base_url_on_this_machine_is_local(url, local):
    assert tool.is_local(url) is local


def test_the_tool_refuses_a_remote_base_url_without_opening_a_connection(monkeypatch, capsys, tmp_path):
    def no_network(url):
        raise AssertionError('a network request was made')
    monkeypatch.setattr(tool, 'http_get_json', no_network)
    out = tmp_path / 'x.json'
    assert tool.main(['--base-url', 'https://api.example.invalid', '--out', str(out)]) == 2
    assert 'not on this machine' in capsys.readouterr().err and not out.exists()


def test_the_tool_writes_the_record_for_a_local_base_url_and_for_a_remote_one_only_when_allowed(monkeypatch, tmp_path):
    get_json, _ = _fake_api()
    monkeypatch.setattr(tool, 'http_get_json', lambda url: get_json(url.replace('http://127.0.0.1:9', BASE).replace('https://api.example.invalid', BASE)))
    out = tmp_path / 'local.json'
    assert tool.main(['--base-url', BASE, '--out', str(out)]) == 0
    assert len(json.loads(out.read_text())['rows']) == 2
    monkeypatch.setattr(tool, 'http_get_json', lambda url: get_json(url.replace('https://api.example.invalid', BASE)))
    remote = tmp_path / 'remote.json'
    assert tool.main(['--base-url', 'https://api.example.invalid', '--out', str(remote), '--allow-remote']) == 0 and remote.exists()


def test_neither_this_module_nor_the_tool_names_the_hosted_api():
    for path in (os.path.abspath(__file__), tool.__file__):
        text = open(path, encoding='utf-8').read()
        assert 'onrender.com' not in text.replace("'onrender.com'", '') and 'vercel.app' not in text.replace("'vercel.app'", '')
