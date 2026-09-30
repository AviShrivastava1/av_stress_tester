"""
test_notebook_committed_clean.py — the committed notebook carries no cell outputs.

Once cell 12 can point the notebook at a hosted database, a saved output can contain
that database's host (cell 12 prints it) and anything a later cell prints about the
data. The notebook is committed to a public repository, so "clear all outputs before
committing" cannot be a rule someone remembers; it has to fail the suite.

Execution counts are checked too: a cleared-output notebook that still carries counts
was run and saved, and its outputs were stripped by hand, which is exactly the step
this test exists to stop relying on.

Pure stdlib. No database, no Waymo package, no notebook execution.

Run:
    ./venv/bin/python -m pytest tests/test_notebook_committed_clean.py -q
"""

import json
import os

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(PROJECT, 'notebooks', 'colab_validation_run.ipynb')


def saved_run_state(notebook):
    """(cell index, what was found) for every code cell carrying outputs or a count."""
    found = []
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        if cell.get('outputs'):
            found.append((index, f"{len(cell['outputs'])} output(s)"))
        if cell.get('execution_count') is not None:
            found.append((index, f"execution_count={cell['execution_count']}"))
    return found


def test_the_check_finds_a_planted_output():
    """Vacuity guard: a check that matched nothing would pass on any notebook."""
    planted = {'cells': [
        {'cell_type': 'markdown', 'source': ['# x']},
        {'cell_type': 'code', 'source': ['print(1)'], 'execution_count': 3,
         'outputs': [{'output_type': 'stream', 'name': 'stdout', 'text': ['host: db.example\n']}]},
        {'cell_type': 'code', 'source': ['x = 1'], 'execution_count': None, 'outputs': []},
    ]}
    assert saved_run_state(planted) == [(1, '1 output(s)'), (1, 'execution_count=3')]


def test_the_committed_notebook_has_no_outputs_or_execution_counts():
    notebook = json.load(open(NOTEBOOK))
    found = saved_run_state(notebook)
    assert not found, (
        'the committed notebook carries saved run state. Clear all outputs (Edit > Clear '
        'all outputs in Colab) before committing; an output from an external-database '
        'run can contain the database host:\n'
        + '\n'.join(f'  cell {i}: {what}' for i, what in found)
    )
