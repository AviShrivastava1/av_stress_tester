"""
test_f09_notebook_m_value_precision.py — fix F09 (independent review, notebook cell).

notebooks/colab_validation_run.ipynb's "13. Geometry verification" cell (the one
audit B19 already fixed once, for a different defect in the same loop — see
test_audit_core.py's test_B19_notebook_geometry_validation_rejects_missing_valid_
agents) used to cast the M values read back from PostGIS to int BEFORE comparing
them against the expected valid-timestep indices:

    m_array = np.array(m_values, dtype=int)
    if np.array_equal(m_array, true_valid_ts): ...

A systematic fractional offset — every M value shifted by, say, +0.25 — survives
`dtype=int` (which truncates, it does not round or refuse) and compares equal to the
correct integer sequence. The check whose entire job is proving the M ordinate equals
the true valid-timestep index, element for element, could not see a mistake in
exactly that value.

Modelled directly on test_B19's own technique for this cell: a fully in-memory,
no-database env (ShardLoader/ScenarioParser/cursor all faked), because this defect is
pure Python-level comparison logic — nothing about it depends on what PostGIS itself
can store. Located by the SAME content tokens test_B19 already uses
('n_exact_match', 'dump_points_m(') — two cells in this notebook print
'n_exact_match', and requiring the dump_points_m call is what pins the one that
actually reads geometry back out, not the summary cell that reprints the counter.

WHAT THIS FILE CANNOT DISTINGUISH, STATED PLAINLY. With a +0.25 offset, comparing the
RAW (uncast) values against the expected integer sequence already fails on its own —
the fix's second, independent "are these values integer-valued at all" check adds
nothing OBSERVABLE for that specific input. Its value is in surviving a future edit
that loosens the first comparison (an np.allclose with a generous tolerance, say)
without anyone noticing the underlying contract — M ordinates are exact integers —
quietly stopped being enforced. That is a property of the fix's STRUCTURE, not
something a black-box test of the cell's output can observe from outside; this file
does not claim to prove it.

Needs no database — @requires_db is not used here, matching test_B19.

Run:
    ./venv/bin/python -m pytest tests/test_f09_notebook_m_value_precision.py -q
"""

import contextlib
import io
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _notebook_cells():
    path = os.path.join(PROJECT, 'notebooks', 'colab_validation_run.ipynb')
    return json.load(open(path))['cells']


def _geometry_validation_cell():
    """Same selector test_B19_notebook_geometry_validation_rejects_missing_valid_
    agents already uses, and for the same stated reason."""
    sources = [''.join(c['source']) for c in _notebook_cells()
              if c['cell_type'] == 'code']
    cells = [src for src in sources
             if 'n_exact_match' in src and 'dump_points_m(' in src]
    assert len(cells) == 1, (
        f'expected exactly one geometry-validation cell, found {len(cells)}'
    )
    return cells[0]


class _Parser:
    """One scenario, one agent, three valid timesteps at indices 0, 1, 2."""
    def __init__(self, raw):
        pass

    def get_scenario_id(self):
        return 'A'

    def get_agent_validity(self):
        return np.ones((1, 3), dtype=bool)


class _Cursor:
    """agent 0 IS exported — unlike test_B19's Cursor, whose empty fetchall() is the
    point of ITS test. This one needs to reach dump_points_m at all."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, *args):
        pass

    def fetchall(self):
        return [(0,)]


def _dump_points_m(m_values):
    def dump(conn, table, scenario_id, agent_idx=None):
        headings = [0.0] * len(m_values)
        return (agent_idx, len(m_values), headings, list(m_values))
    return dump


def _run_cell(m_values):
    env = dict(np=np, ShardLoader=lambda _: [b'A'], SHARD_PATH='synthetic',
               ScenarioParser=_Parser, ids_to_test=['A'],
               conn=SimpleNamespace(cursor=_Cursor),
               dump_points_m=_dump_points_m(m_values),
               summary={'agents_skipped': 0})
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(_geometry_validation_cell(), 'notebook_cell_f09', 'exec'), env)
    return out.getvalue()


def test_F09_a_systematic_fractional_offset_is_caught():
    """
    THE PRIMARY REPRO. Every M value shifted by exactly +0.25 — truncates to the
    correct integer sequence under dtype=int, which is exactly what let this defect
    hide behind a passing assertion before.
    """
    with pytest.raises(AssertionError):
        _run_cell([0.25, 1.25, 2.25])


def test_F09_the_printed_mismatch_names_non_integer_values_not_a_generic_mismatch():
    """The raised AssertionError's own message is generic by design (it reports a
    COUNT mismatch, not a per-agent reason) — the per-agent diagnosis is in the
    printed MISMATCH line, captured here directly rather than assumed."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(AssertionError):
        env = dict(np=np, ShardLoader=lambda _: [b'A'], SHARD_PATH='synthetic',
                   ScenarioParser=_Parser, ids_to_test=['A'],
                   conn=SimpleNamespace(cursor=_Cursor),
                   dump_points_m=_dump_points_m([0.25, 1.25, 2.25]),
                   summary={'agents_skipped': 0})
        exec(compile(_geometry_validation_cell(), 'notebook_cell_f09', 'exec'), env)
    assert 'not integer-valued' in out.getvalue(), out.getvalue()


def test_F09_genuine_integer_valued_floats_from_postgis_still_pass():
    """
    THE MUST-NOT-REGRESS CASE. Every real M value read back from PostGIS is a Python
    float — ST_M returns double precision even when the value it holds is
    conceptually an integer. Rejecting [0.0, 1.0, 2.0] against true_valid_ts
    [0, 1, 2] would make the fix reject the ENTIRE ordinary case, not just the
    corrupted one.
    """
    out = _run_cell([0.0, 1.0, 2.0])
    assert 'CONFIRMED' in out, out


def test_F09_an_ordinary_integer_mismatch_is_still_caught():
    """The raw-equality half of the fix must still catch plain wrong values on its
    own, independent of the integer-valued check — removing the premature cast must
    not weaken this side."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(AssertionError):
        env = dict(np=np, ShardLoader=lambda _: [b'A'], SHARD_PATH='synthetic',
                   ScenarioParser=_Parser, ids_to_test=['A'],
                   conn=SimpleNamespace(cursor=_Cursor),
                   dump_points_m=_dump_points_m([0.0, 1.0, 5.0]),
                   summary={'agents_skipped': 0})
        exec(compile(_geometry_validation_cell(), 'notebook_cell_f09', 'exec'), env)
    printed = out.getvalue()
    assert 'values differ from true valid timesteps' in printed, printed
    assert 'not integer-valued' not in printed, (
        'an integer, merely-wrong value was misdiagnosed as non-integer: ' + printed
    )
