"""
tools/build_geometry_fixtures.py: builds small geometry fixtures from a local shard into a directory git ignores, behind the one switch for
Waymo-derived outputs (waymo_derived_outputs.json).

Pinned here:

  * it refuses unless the switch value it is given is exactly True, having written nothing, and the command line passes the value of the file (which is
    false, so the command refuses today); no shard is read when it refuses;
  * it writes only into a directory git ignores (the default is tests/geometry_fixtures_local), and refuses any other, having written nothing;
  * with the switch passed as on: one file per scenario holding the SDC and its nearest challengers in the ranking the pipeline uses, every frame,
    and a manifest with the sha256 of each file;
  * the counts it is given are checked.

These tests write only into pytest's temporary directories; none of them creates a fixture in the repository or reads a shard.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pytest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from tools import build_geometry_fixtures as tool

pytestmark = pytest.mark.filterwarnings('ignore')


def _scene(scenario_id='scene-a', n_agents=6, steps=8):
    """Agent 2 is the SDC; the others sit at distances 5, 1, 3, 2, 4 from it, so the ranking by distance is [3 (1 m), 1... see RANKED."""
    states = np.zeros((n_agents, steps, 7), dtype=np.float32)
    distances = {0: 5.0, 1: 1.0, 3: 3.0, 4: 2.0, 5: 4.0}
    for j, d in distances.items():
        states[j, :, 1] = d
    states[:, :, 5:7] = [4.5, 2.0]
    states[1, 0, 0] = 0.25                                      # a value to find again
    validity = np.ones((n_agents, steps), dtype=bool)
    validity[4, 5:] = False
    types = np.array([1, 1, 1, 2, 3, 1])
    return scenario_id, states, validity, types, 2


RANKED = [1, 4, 3, 5, 0]


@pytest.fixture
def ignored(monkeypatch):
    """Stands in for git's answer, so that a temporary directory counts as an ignored one."""
    monkeypatch.setattr(tool, 'is_git_ignored', lambda path: True)


def test_it_refuses_unless_the_switch_is_exactly_true_and_writes_nothing(tmp_path, ignored):
    out = tmp_path / 'fixtures'
    for value in (False, None, 1, 'true', 'True', [True], 0):
        with pytest.raises(tool.FixturesRefused, match='switched off'):
            tool.build_fixtures('/no/such/shard', str(out), enabled=value, scenes=iter([_scene()]))
        assert not out.exists()


def test_it_does_not_read_the_shard_when_it_refuses(tmp_path, ignored, monkeypatch):
    def boom(*a, **k):
        raise AssertionError('the shard was read')
    monkeypatch.setattr(tool, 'iter_shard_scenes', boom)
    with pytest.raises(tool.FixturesRefused):
        tool.build_fixtures('/no/such/shard', str(tmp_path / 'x'), enabled=False)


def test_the_command_passes_the_value_of_the_switch_file_which_is_off_so_it_refuses(tmp_path, capsys, monkeypatch):
    def boom(*a, **k):
        raise AssertionError('the shard was read')
    monkeypatch.setattr(tool, 'iter_shard_scenes', boom)
    out = tmp_path / 'o'
    assert tool.main(['--shard', '/no/such/shard.tfrecord', '--out', str(out)]) == 2
    assert 'switched off' in capsys.readouterr().err and not out.exists()


def test_the_command_builds_when_the_switch_reads_on(tmp_path, ignored, monkeypatch, capsys):
    from src import waymo_derived
    monkeypatch.setattr(waymo_derived, 'outputs_enabled', lambda: True)
    monkeypatch.setattr(tool, 'iter_shard_scenes', lambda shard, limit: iter([_scene('s1'), _scene('s2')]))
    assert tool.main(['--shard', '/no/such/shard.tfrecord', '--out', str(tmp_path / 'o'), '--max-scenarios', '2']) == 0
    assert sorted(os.listdir(tmp_path / 'o')) == ['MANIFEST.json', 's1.json', 's2.json']
    assert 'wrote 3 files' in capsys.readouterr().out


def test_it_writes_only_where_git_ignores(tmp_path):
    out = tmp_path / 'fixtures'
    with pytest.raises(tool.FixturesRefused, match='not a directory git ignores'):
        tool.build_fixtures('/s', str(out), enabled=True, scenes=iter([_scene()]))
    assert not out.exists()
    with pytest.raises(tool.FixturesRefused, match='not a directory git ignores'):
        tool.build_fixtures('/s', os.path.join(PROJECT, 'tests', 'data'), enabled=True, scenes=iter([_scene()]))


def test_git_ignores_the_default_directory_and_not_the_others():
    assert tool.is_git_ignored(tool.FIXTURE_DIR) is True
    assert tool.is_git_ignored(os.path.join(tool.FIXTURE_DIR, 'one', 'two.json')) is True
    for path in (os.path.join(PROJECT, 'tests'), os.path.join(PROJECT, 'tests', 'data'), os.path.join(PROJECT, 'frontend', 'src'), '/tmp', PROJECT):
        assert tool.is_git_ignored(path) is False, path


def test_the_default_directory_is_the_one_the_gitignore_names():
    text = open(os.path.join(PROJECT, '.gitignore'), encoding='utf-8').read().split('\n')
    assert '/tests/geometry_fixtures_local/' in text
    assert os.path.relpath(tool.FIXTURE_DIR, PROJECT) == os.path.join('tests', 'geometry_fixtures_local')


def test_a_fixture_holds_the_sdc_then_its_nearest_challengers_in_the_pipelines_ranking_with_every_frame(tmp_path, ignored):
    paths = tool.build_fixtures('/s', str(tmp_path), enabled=True, challengers=3, scenes=iter([_scene('scene-a')]))
    assert [os.path.basename(p) for p in paths] == ['scene-a.json', 'MANIFEST.json']
    fixture = json.loads((tmp_path / 'scene-a.json').read_text())
    assert list(fixture) == ['scenario_id', 'sdc_idx', 'agents'] and fixture['scenario_id'] == 'scene-a' and fixture['sdc_idx'] == 2
    assert [a['agent_idx'] for a in fixture['agents']] == [2, *RANKED[:3]]
    assert [a['agent_type'] for a in fixture['agents']] == [1, 1, 3, 2]
    for a in fixture['agents']:
        assert len(a['states']) == 8 and all(len(row) == 7 for row in a['states']) and len(a['valid']) == 8
    one = fixture['agents'][1]
    assert one['states'][0][0] == 0.25 and one['states'][0][1] == 1.0
    assert fixture['agents'][2]['valid'] == [True] * 5 + [False] * 3


def test_the_manifest_has_the_sha256_of_every_file_and_no_other(tmp_path, ignored):
    tool.build_fixtures('/s', str(tmp_path), enabled=True, scenes=iter([_scene('a'), _scene('b')]))
    manifest = json.loads((tmp_path / 'MANIFEST.json').read_text())
    assert sorted(manifest['files']) == ['a.json', 'b.json']
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    assert 'Never commit' in manifest['note']


def test_it_stops_at_the_most_scenarios_asked_for(tmp_path, ignored):
    scenes = (_scene(f's{i}') for i in range(10))
    tool.build_fixtures('/s', str(tmp_path), enabled=True, max_scenarios=3, scenes=scenes)
    assert sorted(os.listdir(tmp_path)) == ['MANIFEST.json', 's0.json', 's1.json', 's2.json']


def test_it_writes_nothing_when_the_shard_gave_no_scenario(tmp_path, ignored):
    with pytest.raises(tool.FixturesRefused, match='no scenario'):
        tool.build_fixtures('/s', str(tmp_path / 'x'), enabled=True, scenes=iter([]))
    assert not (tmp_path / 'x').exists()


@pytest.mark.parametrize('kwargs', [{'max_scenarios': 0}, {'max_scenarios': 51}, {'max_scenarios': True}, {'max_scenarios': 2.0}, {'challengers': 0}, {'challengers': 11}, {'challengers': True}])
def test_the_counts_are_checked_before_anything_is_written(tmp_path, ignored, kwargs):
    with pytest.raises(tool.FixturesRefused, match=r'^(max_scenarios|challengers) is from 1 to (50|10), not '):
        tool.build_fixtures('/s', str(tmp_path / 'x'), enabled=True, scenes=iter([_scene()]), **kwargs)
    assert not (tmp_path / 'x').exists()


def test_the_tool_never_reaches_the_network_or_the_hosted_api():
    text = open(tool.__file__, encoding='utf-8').read()
    for needle in ('urllib', 'requests', 'http://', 'https://', 'socket', 'psycopg2'):
        assert needle not in text, needle
