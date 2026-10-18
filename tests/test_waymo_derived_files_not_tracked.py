"""
Nothing Waymo-derived is tracked by git (the licence reply is outstanding; waymo_derived_outputs.json is the switch).

Pinned here, on the files git tracks (`git ls-files`, the index):

  * nothing under tests/geometry_fixtures_local/, the directory tools/build_geometry_fixtures.py writes into;
  * no file of a type that is a recording or an image of a scene (a shard, an array dump, a pickle, a table, a video, a GIF) and no SVG or PNG
    whose name says it is a scene, a frame, a replay or an export;
  * no JSON file that carries agent geometry: a list of coordinate pairs or a list of numbers under a key such as path, states, headings,
    timesteps, lengths_m or widths_m.

One tracked file is known to match the last rule and was committed before this check (2026-10-08, "Draw each agent's box at its per-frame size"):
frontend/src/test/realCollisionFrames.json, the SDC and perturbed challenger tracks around the collision frame of 18 scenarios. It is listed in KNOWN_TRACKED
so that this test fails for every OTHER file, and fails again if that file is ever removed without removing the line: whether it may stay is a ruling for the
project's author, not something this test decides. The same list is printed in the report that introduced this test.
"""

import json
import os
import re
import subprocess

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: tracked files that carry Waymo-derived geometry and were committed before this check: see the docstring
KNOWN_TRACKED = ['frontend/src/test/realCollisionFrames.json']

FIXTURE_PREFIX = 'tests/geometry_fixtures_local/'
RECORDING_SUFFIXES = ('.npz', '.npy', '.pkl', '.pickle', '.parquet', '.h5', '.hdf5', '.webm', '.mp4', '.gif', '.mov', '.avi')
SCENE_IMAGE = re.compile(r'(scene|frame|replay|export|trajector|track)', re.I)
GEOMETRY_KEYS = {'path', 'states', 'headings', 'timesteps', 'lengths_m', 'widths_m', 'polyline', 'vertices', 'positions', 'trajectory', 'trajectories'}
MAX_JSON_BYTES = 20_000_000


def tracked():
    out = subprocess.run(['git', 'ls-files'], cwd=PROJECT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.split('\n') if f]


def _numbers(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def carries_geometry(value) -> bool:
    """Whether a parsed JSON value has, under a geometry key, a non-empty list of numbers or of lists of numbers."""
    if isinstance(value, dict):
        for key, v in value.items():
            if key in GEOMETRY_KEYS and isinstance(v, list) and v and (all(_numbers(x) for x in v) or all(isinstance(x, list) and x and all(_numbers(y) for y in x) for x in v)):
                return True
            if carries_geometry(v):
                return True
    elif isinstance(value, list):
        return any(carries_geometry(v) for v in value)
    return False


def looks_like_recording_or_scene_image(path: str) -> bool:
    lower = path.lower()
    if '.tfrecord' in lower or lower.endswith(RECORDING_SUFFIXES):
        return True
    return lower.endswith(('.svg', '.png', '.jpg', '.jpeg')) and bool(SCENE_IMAGE.search(os.path.basename(path)))


def suspects(files):
    found = []
    for f in files:
        if f.startswith(FIXTURE_PREFIX) or looks_like_recording_or_scene_image(f):
            found.append(f)
        elif f.endswith('.json'):
            full = os.path.join(PROJECT, f)
            if os.path.isfile(full) and os.path.getsize(full) <= MAX_JSON_BYTES:
                with open(full, encoding='utf-8') as fh:
                    if carries_geometry(json.load(fh)):
                        found.append(f)
    return sorted(found)


def test_nothing_under_the_fixture_directory_is_tracked():
    assert [f for f in tracked() if f.startswith(FIXTURE_PREFIX)] == []


def test_the_only_tracked_files_that_look_like_waymo_derived_geometry_are_the_known_one():
    assert suspects(tracked()) == sorted(KNOWN_TRACKED)


def test_the_known_file_is_still_there_so_the_list_does_not_outlive_it():
    for f in KNOWN_TRACKED:
        assert f in tracked(), f'{f} is no longer tracked: remove it from KNOWN_TRACKED'


# the rule itself, on made-up lists of files, so that it does not rest on what happens to be in the repository today

def test_the_rule_flags_a_fixture_a_recording_a_scene_image_and_a_geometry_json(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, 'isfile', os.path.isfile)
    assert looks_like_recording_or_scene_image('data/x.tfrecord') and looks_like_recording_or_scene_image('a/b.tfrecord-00001-of-01000')
    for name in ('s.npz', 's.npy', 'x.pkl', 'x.parquet', 'r.webm', 'r.mp4', 'r.gif'):
        assert looks_like_recording_or_scene_image(name), name
    for name in ('scene-1.svg', 'frame_12.png', 'export.png', 'replay.jpg', 'trajectories.svg'):
        assert looks_like_recording_or_scene_image(name), name
    for name in ('favicon.svg', 'logo.png', 'README.md', 'package.json', 'schema.d.ts'):
        assert not looks_like_recording_or_scene_image(name), name
    assert suspects(['tests/geometry_fixtures_local/a.json', 'tests/geometry_fixtures_local/MANIFEST.json']) == ['tests/geometry_fixtures_local/MANIFEST.json', 'tests/geometry_fixtures_local/a.json']


def test_the_rule_reads_geometry_in_json_by_its_shape_and_not_by_a_name_alone():
    assert carries_geometry({'a': {'path': [[1.0, 2.0], [3.0, 4.5]]}})
    assert carries_geometry([{'states': [[0, 1, 2, 3, 4, 5, 6]]}])
    assert carries_geometry({'timesteps': [0.0, 1.0, 2.0]})
    assert not carries_geometry({'path': '/api/x'})            # a route, as in an OpenAPI file
    assert not carries_geometry({'path': {'in': 'path'}})
    assert not carries_geometry({'rows': [{'scenario_id': 'a', 'delta': [0.1, 0.2, 0.3, 0.4]}]})
    assert not carries_geometry({'path': []})
    assert not carries_geometry({'path': [True, False]})
    assert not carries_geometry({'headings_note': [1, 2, 3]})
