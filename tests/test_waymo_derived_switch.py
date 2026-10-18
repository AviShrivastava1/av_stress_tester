"""
The one switch for Waymo-derived outputs (waymo_derived_outputs.json, read by src/waymo_derived.py and by frontend/src/waymoDerivedOutputs.ts).

Pinned here: the committed value is false (the licence reply is outstanding), and the test fails if it is ever true; the file is exactly one boolean
and nothing else; a file that is not exactly that is an error and is never read as "on"; there is no second reader of the file and no second
constant on the Python side. The frontend suite has the matching test (frontend/src/waymoDerivedOutputs.test.ts).
"""

import json
import os
import subprocess
import sys

import pytest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from src import waymo_derived

FILE = os.path.join(PROJECT, 'waymo_derived_outputs.json')


def test_the_switch_is_off():
    """If this fails, the switch file says true: Waymo-derived outputs are on, and the licence reply has not been settled."""
    assert waymo_derived.read_switch() is False
    assert waymo_derived.outputs_enabled() is False


def test_the_file_is_exactly_one_boolean_called_enabled():
    with open(FILE, encoding='utf-8') as fh:
        assert json.load(fh) == {'enabled': False}


def test_the_default_path_is_the_file_at_the_repository_root():
    assert os.path.abspath(waymo_derived.SWITCH_FILE) == os.path.abspath(FILE)


@pytest.mark.parametrize('text,expected', [('{"enabled": true}', True), ('{"enabled": false}', False), ('{\n "enabled": true\n}\n', True)])
def test_a_well_formed_file_is_read_as_it_says(tmp_path, text, expected):
    path = tmp_path / 'switch.json'
    path.write_text(text)
    assert waymo_derived.read_switch(str(path)) is expected


@pytest.mark.parametrize('text', [
    '', 'true', '[]', '{}', '{"enabled": 1}', '{"enabled": "true"}', '{"enabled": null}', '{"enabled": true, "other": 1}', '{"Enabled": true}',
    '{"enabled": true,}', 'enabled', '{"enabled": [true]}',
])
def test_a_file_that_is_not_exactly_the_switch_is_an_error_never_on(tmp_path, text):
    path = tmp_path / 'switch.json'
    path.write_text(text)
    with pytest.raises(waymo_derived.SwitchError):
        waymo_derived.read_switch(str(path))


def test_outputs_enabled_follows_the_file_it_is_given(tmp_path):
    on, off = tmp_path / 'on.json', tmp_path / 'off.json'
    on.write_text('{"enabled": true}')
    off.write_text('{"enabled": false}')
    assert waymo_derived.outputs_enabled(str(on)) is True
    assert waymo_derived.outputs_enabled(str(off)) is False


def test_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(waymo_derived.SwitchError):
        waymo_derived.read_switch(str(tmp_path / 'nope.json'))


def _tracked(*roots):
    out = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', *roots], cwd=PROJECT, capture_output=True, text=True, check=True).stdout.split('\n')
    return [f for f in out if f.endswith('.py') and f != 'tests/test_waymo_derived_switch.py']


def _names_the_file_as_a_value(path):
    """Whether the module has the file's name as a string value on its own (a path component handed to open or join), not in a docstring or comment."""
    import ast
    tree = ast.parse(open(os.path.join(PROJECT, path), encoding='utf-8').read())
    return any(isinstance(n, ast.Constant) and n.value == 'waymo_derived_outputs.json' for n in ast.walk(tree))


def test_the_file_is_read_in_one_place_only_on_the_python_side():
    readers = [f for f in _tracked('src', 'tools') if _names_the_file_as_a_value(f)]
    assert readers == ['src/waymo_derived.py'], readers


def test_there_is_no_second_switch_constant_on_the_python_side():
    import re
    pattern = re.compile(r'^\s*(WAYMO_\w*(ENABLED|SWITCH)\w*|\w*waymo_derived\w*enabled\w*)\s*=\s*(True|False)\b', re.I | re.M)
    offenders = [f for f in _tracked('src', 'tools') if pattern.search(open(os.path.join(PROJECT, f), encoding='utf-8').read())]
    assert offenders == [], offenders
