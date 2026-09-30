"""
test_notebook_top_n.py — every statement of TOP_N agrees with the config cell, and the
runbook says what the chosen value does and does not mean.

TOP_N went from 5 to 20 for the A_MAX=12 re-run: the deployed site serves what Pass 2
stress-tests, and 20 gives it 20 scenarios with a perturbation and playback instead of 5.
Nothing here pins 20. The tests read the value from cell 6 and check that the notebook's
prose and comments and the runbook agree with it, so the next change of TOP_N is caught
the same way.

No database, no Waymo package.

Run:
    ./venv/bin/python -m pytest tests/test_notebook_top_n.py -q
"""

import ast
import json
import os
import re

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(PROJECT, 'notebooks', 'colab_validation_run.ipynb')
RUNBOOK = os.path.join(PROJECT, 'notebooks', 'COLAB_RUNBOOK.md')

# `TOP_N = 5`, `TOP_N=5`, and the runbook's aligned `TOP_N         = 5`.
STATED = re.compile(r'\bTOP_N\s*=\s*(\d+)')


def _cells():
    return json.load(open(NOTEBOOK))['cells']


def config_top_n():
    """TOP_N as assigned in the config cell (cell 6, located by content)."""
    hits = [''.join(c['source']) for c in _cells() if c['cell_type'] == 'code'
            and 'the only cell you should need to edit' in ''.join(c['source'])]
    assert len(hits) == 1, f'expected one config cell, got {len(hits)}'
    values = [ast.literal_eval(node.value) for node in ast.parse(hits[0]).body
              if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'TOP_N' for t in node.targets)]
    assert len(values) == 1, f'expected one TOP_N assignment in the config cell, got {values}'
    return values[0]


def _flat(text):
    return ' '.join(text.split())


def _step_6_top_n_paragraph():
    runbook = open(RUNBOOK).read()
    step = runbook[runbook.index('## Step 6'):]
    paragraphs = [p for p in step.split('\n\n') if p.startswith('**`TOP_N')]
    assert len(paragraphs) == 1, f'expected one bold TOP_N paragraph in step 6, got {len(paragraphs)}'
    return _flat(paragraphs[0])


def test_every_stated_top_n_in_the_notebook_matches_the_config_cell():
    top_n = config_top_n()
    stale = [(i, m.group(0)) for i, c in enumerate(_cells())
             for m in STATED.finditer(''.join(c['source'])) if int(m.group(1)) != top_n]
    assert not stale, f'cells still state another TOP_N than the config\'s {top_n}: {stale}'


def test_every_stated_top_n_in_the_runbook_matches_the_config_cell():
    top_n = config_top_n()
    stale = [m.group(0) for m in STATED.finditer(open(RUNBOOK).read()) if int(m.group(1)) != top_n]
    assert not stale, f'the runbook still states another TOP_N than the config\'s {top_n}: {stale}'


def test_step_6_gives_the_reason_for_the_value_and_says_what_it_does_not_change():
    paragraph = _step_6_top_n_paragraph()
    assert 'deployed site' in paragraph, 'the reason (the site\'s dataset) is not given'
    assert 'not any measurement' in paragraph, 'it does not say the value serves no measurement'
    assert 'cheap filter' in paragraph, 'the architecture argument is gone rather than answered'
    assert 'S1–S8' in paragraph, 'it does not say the drift checks are independent of TOP_N'


def test_step_6_states_the_tie_caveat():
    paragraph = _step_6_top_n_paragraph()
    top_n = config_top_n()
    assert 'scenario ID' in paragraph and 'tie' in paragraph
    assert '70%' not in paragraph, 'the tie share is the TTC-floor share, not the fragility tie'
    assert 'at least a quarter and fewer than half' in paragraph and 'p75 100' in paragraph, (
        'the tie share is not stated from the Pass 1 fragility percentiles'
    )
    assert 'not re-derived here' in paragraph
    assert f'not the {top_n} most fragile' in paragraph, (
        'it does not say the selection is an ID-ordered slice of the tied scenarios'
    )


def test_the_run_time_estimate_is_restated_for_this_top_n_and_labelled():
    runbook = open(RUNBOOK).read()
    ground_rules = _flat(runbook[runbook.index('## Ground rules'):runbook.index('## Step 1')])
    assert 'estimated at about 17 minutes' not in ground_rules, 'the estimate is still the old one'
    assert 'estimated at about 26–28 minutes' in ground_rules
    assert 'estimates, not measurements' in ground_rules, 'the parts are not labelled as estimates'
