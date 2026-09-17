"""
test_batch13_contract.py — Batch 13 contract tests (audit A04, A13, A14).

The audit's own repros live in tests/test_audit3_regressions.py, reconstructed and
labelled as such. THIS file holds the properties.

The centrepiece is the shadowed-read guard. A14 was one variable in one notebook, and
renaming it fixes that instance — but this notebook has had four cell-index-drift
incidents and now one namespace collision, and a rename protects against neither
recurrence. Batch 10 answered the same shape of problem (a finding two audits kept
finding in different places) with a standing sweep rather than edits, and the same
answer applies here.

Pure stdlib. No database, no Waymo package, no notebook execution.

Run:
    ./venv/bin/python -m pytest tests/test_batch13_contract.py -q
"""

import ast
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(PROJECT, 'notebooks', 'colab_validation_run.ipynb')
RUNBOOK = os.path.join(PROJECT, 'notebooks', 'COLAB_RUNBOOK.md')

_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _code_cells(path=NOTEBOOK):
    cells = json.load(open(path))['cells']
    return [(i, ''.join(c['source'])) for i, c in enumerate(cells)
            if c['cell_type'] == 'code']


def _scoped_names(tree):
    """
    Names that are NOT module-level in this cell: comprehension targets, function
    parameters, and anything assigned inside a function body.

    ALL THREE ARE REQUIRED, and I got this wrong twice while building it. Skipping
    comprehensions entirely lost `traj`, because a generator expression CLOSES OVER
    enclosing names even though its target is local. Skipping function bodies for
    bindings but not for reads produced two false positives (`N` and `j`, both locals
    in the TTC comparison cell's own helper). The guard is only as trustworthy as this
    function, which is why test_the_guard_finds_a_planted_collision exists.
    """
    scoped = set()
    for node in ast.walk(tree):
        if isinstance(node, _COMPREHENSIONS):
            for generator in node.generators:
                for leaf in ast.walk(generator.target):
                    if isinstance(leaf, ast.Name):
                        scoped.add(leaf.id)
        if isinstance(node, _FUNCTIONS):
            args = node.args
            for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
                scoped.add(arg.arg)
            if args.vararg:
                scoped.add(args.vararg.arg)
            if args.kwarg:
                scoped.add(args.kwarg.arg)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        for leaf in ast.walk(target):
                            if isinstance(leaf, ast.Name):
                                scoped.add(leaf.id)
                elif isinstance(sub, ast.For):
                    for leaf in ast.walk(sub.target):
                        if isinstance(leaf, ast.Name):
                            scoped.add(leaf.id)
    return scoped


def _module_bindings(tree, scoped):
    bound = set()

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node): pass
        def visit_AsyncFunctionDef(self, node): pass
        def visit_ClassDef(self, node): pass
        def visit_Lambda(self, node): pass
        def visit_ListComp(self, node): pass
        def visit_SetComp(self, node): pass
        def visit_DictComp(self, node): pass
        def visit_GeneratorExp(self, node): pass

        def visit_Assign(self, node):
            for target in node.targets:
                for leaf in ast.walk(target):
                    if isinstance(leaf, ast.Name):
                        bound.add(leaf.id)
            self.generic_visit(node)

        def visit_AugAssign(self, node):
            if isinstance(node.target, ast.Name):
                bound.add(node.target.id)
            self.generic_visit(node)

        def visit_For(self, node):
            for leaf in ast.walk(node.target):
                if isinstance(leaf, ast.Name):
                    bound.add(leaf.id)
            self.generic_visit(node)

    Visitor().visit(tree)
    return bound - scoped


def shadowed_reads(cells):
    """
    Every (name, binding cells, reading cell) where a name is bound at module scope in
    MORE THAN ONE code cell and then read by a cell that binds it in neither.

    That is the precise shape of A14: two experiments wrote `rho`, the summary read it,
    and which value it got depended on execution order. A name bound in several cells
    that each only read their own is NOT flagged — that is ordinary scratch reuse and
    cannot mislead a later reader.
    """
    binds, reads = {}, {}
    for index, source in cells:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue                      # shell-magic install cells
        scoped = _scoped_names(tree)
        binds[index] = _module_bindings(tree, scoped)
        reads[index] = {n.id for n in ast.walk(tree)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                        and n.id not in scoped}

    found = []
    for name in sorted(set().union(*binds.values())) if binds else []:
        binders = sorted(i for i in binds if name in binds[i])
        if len(binders) < 2:
            continue
        for reader in sorted(reads):
            if name in reads[reader] and reader not in binders and reader > binders[0]:
                found.append((name, binders, reader))
    return found


# ── the standing guard ──────────────────────────────────────────────────────────

def test_the_guard_finds_a_planted_collision():
    """
    THE VACUITY GUARD, AND IT COMES FIRST.

    A shadowed-read detector that matched nothing would leave the sweep below green and
    the notebook unprotected. Planted in the same shape as the real defect: two cells
    binding one name, a third reading it.
    """
    planted = [(0, 'rho, p = spearmanr(a, b)\n'),
               (1, 'rho, p = spearmanr(c, d)\n'),
               (2, 'print(f"result {rho}")\n')]
    hits = shadowed_reads(planted)
    assert [h[0] for h in hits] == ['rho'], f'the detector missed a planted collision: {hits}'
    assert hits[0][1] == [0, 1] and hits[0][2] == 2


@pytest.mark.parametrize('cells,why', [
    ([(0, 'for i in range(3):\n    pass\n'), (1, 'for i in range(4):\n    pass\n')],
     'a loop variable reused in two cells, read by neither'),
    ([(0, 'xs = [q for q in range(3)]\n'), (1, 'ys = [q for q in range(4)]\n'),
      (2, 'print(xs, ys)\n')],
     'a comprehension target is scoped to its comprehension'),
    ([(0, 'def f(n):\n    total = n + 1\n    return total\n'),
      (1, 'def g(n):\n    total = n * 2\n    return total\n'),
      (2, 'print(f(1), g(2))\n')],
     'names assigned inside function bodies are locals, not module bindings'),
    ([(0, 'a = 1\nprint(a)\n'), (1, 'a = 2\nprint(a)\n')],
     'two cells binding and reading their own value cannot mislead a third'),
])
def test_the_guard_does_not_flag_legitimate_reuse(cells, why):
    """
    THE OTHER HALF. A guard that flagged every reused name would be turned off within a
    week, and the three exclusions below are exactly the ones I got wrong while
    building it — each of these cases produced a false positive at some point.
    """
    assert shadowed_reads(cells) == [], why


def test_no_notebook_name_is_read_from_the_wrong_cell():
    """
    THE SWEEP, KEPT SWEPT.

    A14 was one variable, and a rename fixes one variable. This notebook has had four
    cell-index-drift incidents and one namespace collision; what it needs is the
    property enforced, not the instance repaired. Batch 10 reached the same conclusion
    about R08/A09 for the same reason.

    ZERO ALLOWLIST, which is what makes it worth having: measured before the fix at
    exactly one hit (`rho`, bound by the two correlation cells, read by the summary)
    and at zero after.
    """
    cells = _code_cells()
    assert len(cells) >= 30, f'only {len(cells)} code cells found'

    hits = shadowed_reads(cells)
    assert not hits, 'a cell reads a name that another cell rebinds:\n' + '\n'.join(
        f'  {name!r} bound in cells {binders}, read by cell {reader} — it gets '
        f'cell {max(b for b in binders if b < reader)}\'s value'
        for name, binders, reader in hits
    )


# ── the notebook stays parseable and locatable ──────────────────────────────────

def test_every_code_cell_still_parses():
    """
    Structural validation, which is all that is available: there is no Waymo package
    here and no shard, so the cells cannot be executed end to end. Shell-magic cells
    are excluded by name rather than by silently swallowing SyntaxError, so a REAL
    syntax error in a Python cell cannot hide among them.
    """
    unparseable = []
    for index, source in _code_cells():
        first = next((l for l in source.splitlines() if l.strip()), '')
        if first.lstrip().startswith(('!', '%')):
            continue
        try:
            ast.parse(source)
        except SyntaxError as exc:
            unparseable.append((index, str(exc)))
    assert not unparseable, f'cells failed to parse: {unparseable}'


@pytest.mark.parametrize('tokens', [
    ('stress_tested_sid', 'GET /health'),
    ('http_timesteps',),
    ('audit B06 — did separate visits',),
    ('PASS 2 AGGREGATES', 'n_no_challenger'),
    ('COLAB VALIDATION RUN — SUMMARY',),
])
def test_every_cell_any_test_execs_is_still_locatable_by_content(tokens):
    """
    Four index-drift incidents are why every test here locates cells by content. This
    asserts the content anchors themselves still resolve to exactly one cell — an
    anchor that matched two cells, or none, would break the tests that use it in a way
    that reads as a failure of the thing under test rather than of the lookup.
    """
    hits = [i for i, src in _code_cells() if all(t in src for t in tokens)]
    assert len(hits) == 1, f'{tokens} matched cells {hits}'


def test_the_runbook_still_maps_section_headings_to_the_right_cells():
    """
    The runbook's own recount check, run automatically rather than by hand. It records
    a cell count and a table of section-heading indices; this batch edits cells without
    inserting or deleting any, so both must still hold exactly.
    """
    cells = json.load(open(NOTEBOOK))['cells']
    headings = {}
    for index, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            match = re.match(r'##\s+(\S+?)\.?\s', ''.join(cell['source']).strip() + ' ')
            if match:
                headings[match.group(1).rstrip('.')] = index

    runbook = open(RUNBOOK).read()
    recorded = re.search(r'at (\d+) cells', runbook)
    assert recorded, 'the runbook no longer records a cell count'
    assert len(cells) == int(recorded.group(1)), (
        f'notebook has {len(cells)} cells, runbook records {recorded.group(1)}'
    )

    # THREE REFERENCE SHAPES, because the runbook uses three and a pattern that caught
    # only one would under-check while looking thorough. An earlier version matched
    # bolded table rows alone, found three mappings, and its `>= 4` floor failed — the
    # honest fix was to widen the search, not to lower the number until it passed.
    TABLE_ROW = r'\|\s*(\d+)(?:[–-](\d+))?\s*\|\s*\*{0,2}(\w+)\*{0,2}\s*\|'
    SUBHEADING = r'###\s+(\w+)\s+—[^(]*\(cells?\s+(\d+)(?:[–-](\d+))?\)'

    mappings = []
    for match in re.finditer(TABLE_ROW, runbook):
        mappings.append((int(match.group(1)), match.group(2), match.group(3)))
    for match in re.finditer(SUBHEADING, runbook):
        mappings.append((int(match.group(2)), match.group(3), match.group(1)))

    checked = 0
    for low, high, section in mappings:
        if section not in headings:
            continue
        index = headings[section]
        assert index == low or (high and low <= index <= int(high)), (
            f'section {section}: runbook says {low}-{high}, heading is at cell {index}'
        )
        checked += 1
    assert checked >= 6, f'only {checked} section mappings were checked'
