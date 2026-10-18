"""
The local geometry fixtures (tests/geometry_fixtures_local/, built by tools/build_geometry_fixtures.py behind the switch for Waymo-derived outputs).

The directory is git-ignored and absent from a clean checkout, and so is this whole file: it is skipped when the directory is not there and runs when
it is. What it checks is that what the tool wrote is what the tool says it wrote: the manifest, the shapes, and that the pipeline's own ranking reads
the fixtures. That nothing under the directory is tracked is tests/test_waymo_derived_files_not_tracked.py.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pytest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from tools.build_geometry_fixtures import FIXTURE_DIR, MANIFEST

pytestmark = pytest.mark.skipif(not os.path.isdir(FIXTURE_DIR), reason='tests/geometry_fixtures_local is absent (built by tools/build_geometry_fixtures.py with the switch on)')


def _manifest():
    with open(os.path.join(FIXTURE_DIR, MANIFEST), encoding='utf-8') as fh:
        return json.load(fh)


def _fixtures():
    for name in sorted(_manifest()['files']):
        with open(os.path.join(FIXTURE_DIR, name), encoding='utf-8') as fh:
            yield name, json.load(fh)


def test_the_directory_holds_the_manifest_and_the_files_it_lists_and_nothing_else():
    listed = set(_manifest()['files'])
    assert listed, 'the manifest lists no file'
    assert set(os.listdir(FIXTURE_DIR)) == listed | {MANIFEST}


def test_every_file_is_the_one_the_manifest_hashed():
    for name, digest in _manifest()['files'].items():
        with open(os.path.join(FIXTURE_DIR, name), 'rb') as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == digest, name


def test_every_fixture_has_the_sdc_first_and_agents_of_the_same_length_with_seven_numbers_a_frame():
    for name, f in _fixtures():
        assert list(f) == ['scenario_id', 'sdc_idx', 'agents'], name
        assert name == f"{f['scenario_id']}.json"
        assert f['agents'][0]['agent_idx'] == f['sdc_idx']
        steps = {len(a['states']) for a in f['agents']}
        assert len(steps) == 1 and steps.pop() > 0
        for a in f['agents']:
            assert len(a['valid']) == len(a['states'])
            assert all(len(row) == 7 and all(isinstance(v, float) for v in row) for row in a['states'])
            assert isinstance(a['agent_type'], int) and isinstance(a['agent_idx'], int)


def test_the_pipelines_own_ranking_reads_every_fixture():
    from src.optimization.perturbation_space import rank_challengers
    for name, f in _fixtures():
        states = np.array([a['states'] for a in f['agents']], dtype=np.float32)
        validity = np.array([a['valid'] for a in f['agents']], dtype=bool)
        ranked = rank_challengers(states, validity, 0)
        # the file lists the challengers nearest first, so read back with the SDC as agent 0 they rank in the order they are written
        assert ranked == list(range(1, len(f['agents']))), name
