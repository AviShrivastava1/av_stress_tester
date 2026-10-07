"""
test_json_safe_numpy.py — NumPy scalars are normalised at the JSONB boundary.

db._json_safe is the one place every JSONB value passes through before psycopg2's
Json adapter calls json.dumps. The offline search naturally produces NumPy scalars
(float32 from the state tensor, int64 counts, bool_ flags), and json.dumps raises
TypeError on every one of them, so a result carrying one in its search_provenance could
not be written at all. _json_safe now converts a NumPy SCALAR to the Python value it
stands for and leaves everything else alone: arrays are not scalars and are returned
untouched, Python bool stays a bool and does not become 1.

Nothing here hardcodes a number obtained by running the code. Expected values are
derived in the test, from float(np.float32(x)) and the like.

Pure python, numpy, no database (the database test is in test_json_safe_numpy_db.py). Run:
    ./venv/bin/python -m pytest tests/test_json_safe_numpy.py -q
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring.db import _json_safe

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize('value', [0.5, 0.1, 12.0, 1e-05, -3.25, 1865.1234])
def test_a_float32_becomes_the_python_float_of_the_same_value(value):
    out = _json_safe({'x': np.float32(value)})

    assert type(out['x']) is float
    assert out['x'] == float(np.float32(value))        # the widened float32, not the decimal typed in


def test_a_finite_float64_becomes_a_plain_float():
    out = _json_safe({'x': np.float64(0.05)})
    assert type(out['x']) is float and out['x'] == 0.05


@pytest.mark.parametrize('scalar', [np.int64(7), np.int32(-3), np.uint8(200), np.intp(15)])
def test_a_numpy_integer_becomes_a_python_int(scalar):
    out = _json_safe({'n': scalar})
    assert type(out['n']) is int and out['n'] == int(scalar)


def test_numpy_scalars_nested_in_dicts_lists_and_tuples_are_converted_and_serialise():
    provenance = {
        'bounds': [[np.float32(-2.0), np.float32(2.0)], (np.float64(-0.1), np.float64(0.1))],
        'de_popsize': np.int64(15),
        'flags': {'has_gap': np.bool_(False)},
    }
    out = _json_safe(provenance)

    text = json.dumps(out)                            # raises TypeError on any surviving numpy scalar
    assert json.loads(text)['bounds'] == [[-2.0, 2.0], [float(np.float64(-0.1)), float(np.float64(0.1))]]
    assert json.loads(text)['de_popsize'] == 15
    assert json.loads(text)['flags'] == {'has_gap': False}


@pytest.mark.parametrize('scalar, spelled', [
    (np.float32('inf'), 'inf'), (np.float32('-inf'), '-inf'), (np.float32('nan'), 'nan'),
])
def test_a_non_finite_float32_becomes_null_with_a_record_of_what_it_was(scalar, spelled):
    out = _json_safe({'baseline_replay_error': scalar, 'ok': np.float32(1.0)})

    assert out['baseline_replay_error'] is None
    assert out['nonfinite_fields'] == {'baseline_replay_error': spelled}
    assert type(out['ok']) is float and out['ok'] == 1.0


@pytest.mark.parametrize('scalar, spelled', [(np.float64('inf'), 'inf'), (np.float64('-inf'), '-inf'), (np.float64('nan'), 'nan')])
def test_a_non_finite_float64_becomes_null_with_a_record_of_what_it_was(scalar, spelled):
    # Passes on main on purpose: np.float64 is a subclass of Python float, so the existing
    # non-finite check already caught it. Guards that the conversion keeps doing so.
    out = _json_safe({'baseline_replay_error': scalar})

    assert out['baseline_replay_error'] is None
    assert out['nonfinite_fields'] == {'baseline_replay_error': spelled}


def test_the_record_names_the_path_of_a_non_finite_value_nested_in_lists():
    out = _json_safe({'bounds': [[np.float32(-1.0), np.float32('inf')]]})
    assert out['bounds'] == [[-1.0, None]] and type(out['bounds'][0][0]) is float
    assert out['nonfinite_fields'] == {'bounds.0.1': 'inf'}


@pytest.mark.parametrize('flag', [True, False])
def test_a_numpy_bool_becomes_a_python_bool_not_an_int(flag):
    out = _json_safe({'f': np.bool_(flag)})
    assert type(out['f']) is bool and out['f'] is flag


@pytest.mark.parametrize('flag', [True, False])
def test_a_python_bool_stays_a_bool(flag):
    # Passes on main on purpose: guards the order of the checks. bool is an int in Python;
    # a numeric branch ahead of the bool branch would turn True into 1.
    out = _json_safe({'f': flag})
    assert type(out['f']) is bool and out['f'] is flag


def test_plain_python_values_are_unchanged():
    # Passes on main on purpose.
    blob = {'i': 7, 'f': 0.05, 'big': 10 ** 20, 's': 'text', 'n': None, 'l': [1, 2.5, 'a'], 'd': {'k': 1e-05}}
    assert _json_safe(blob) == blob


def test_a_python_int_subclass_becomes_a_plain_int():
    import enum

    class Step(enum.IntEnum):
        THREE = 3

    out = _json_safe({'s': Step.THREE})

    assert type(out['s']) is int and out['s'] == 3


def test_a_rational_becomes_a_float():
    from fractions import Fraction

    out = _json_safe({'r': Fraction(1, 4)})

    assert type(out['r']) is float and out['r'] == 0.25


def test_an_object_that_only_looks_like_a_numpy_scalar_is_left_alone():
    # Passes on main on purpose. A class that is merely NAMED generic, in some other module, is
    # not a NumPy scalar and must not have its .item() called.
    class generic:
        def item(self):
            raise AssertionError('.item() must not be called on a non-NumPy object')

    thing = generic()

    assert _json_safe({'t': thing})['t'] is thing


def test_a_numpy_string_becomes_a_python_str():
    out = _json_safe({'s': np.str_('abc')})
    assert type(out['s']) is str and out['s'] == 'abc'


def test_arrays_are_left_alone():
    # Passes on main on purpose: an array is not a scalar and is not this function's business.
    # A generic "has .item()" test would convert a 0-d array and raise on a longer one.
    vector = np.array([1.0, 2.0, 3.0])
    zero_dim = np.array(3.0)

    out = _json_safe({'v': vector, 'z': zero_dim})

    assert out['v'] is vector and out['z'] is zero_dim


def test_the_conversion_is_idempotent():
    blob = {'a': np.float32(0.5), 'b': np.float64('inf'), 'c': [np.int64(2), np.bool_(True)]}

    once = _json_safe(blob)

    assert _json_safe(once) == once                    # no second report key, no change


def test_importing_the_persistence_module_does_not_import_numpy():
    # Passes on main on purpose. db.py handles numpy scalars without depending on numpy.
    r = subprocess.run([sys.executable, '-c',
                        "import sys; sys.dont_write_bytecode = True; import src.scoring.db; "
                        "sys.exit(1 if 'numpy' in sys.modules else 0)"],
                       cwd=PROJECT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
