"""
test_batch10_contract.py — Batch 10 contract tests (audit R08/A09, R10, R11/A08).

The audit's own repros live in tests/test_audit2_regressions.py, reconstructed and
labelled as such. THIS file holds the properties: the things that must be true of any
correct fix, including the ones the repro cannot see.

The centrepiece is the standing sweep. R08/A09 was raised by TWO audits at TWO
DIFFERENT sets of lines, which is evidence that the defect is "nobody has looked
everywhere" rather than "these particular lines are wrong". A fix that edits the cited
lines answers the citation, not the finding. So the sweep that Batch 10 ran by hand is
kept as a test, and it fails on the next occurrence wherever it appears.

Pure stdlib plus numpy — no database, no Waymo package. Run:
    ./venv/bin/python -m pytest tests/test_batch10_contract.py -q
"""

import ast
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── R08 / A09: the sweep, kept swept ────────────────────────────────────────────

# Phrases that assert a safety CONCLUSION about a scenario. Not a style rule — every
# one of these claims something no method in this pipeline establishes, for the reason
# api/models.py sets out at length: DE's termination test measures population spread,
# not the absence of a feasible point, and _stress_one searches ONE challenger chosen
# by a nearest-distance heuristic.
#
# `robustly_safe` WITH THE UNDERSCORE IS NOT HERE, deliberately. It is a real API field
# and a real column, derived from search_certifies_infeasibility, which no code path
# sets to TRUE. The field is the honest bookkeeping; the prose was the defect.
BANNED_CLAIMS = (
    'robustly safe',
    'robustly-safe',
    'certified safe',
    'proven safe',
    'provably safe',
    'guaranteed safe',
    'collision-free',
    'collision free',
    'no collision possible',
    'cannot collide',
)


def _docstring_nodes(tree):
    """Every Constant that is a docstring, by identity."""
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, 'body', None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            found.add(id(body[0].value))
    return found


def _claims_in_source(source, label):
    """
    Every banned claim appearing in a STRING LITERAL of this source.

    LITERALS ONLY, AND THAT IS THE WHOLE DESIGN. An earlier version of this test
    scanned raw lines and found eight hits, every one of which was prose ABOUT the
    defect — models.py quoting the wrong docstring it replaced, db.py explaining what
    "came back clean" would have meant, this batch's own comments. Allowlisting eight
    entries keyed on exact line text would have been brittle, would have needed
    re-approving on every edit, and would have been testing the wrong thing.

    The actual invariant is narrower and needs no allowlist: THE PROGRAM MUST NOT SAY
    THESE THINGS. A print, a field label, an error message. Comments do not appear in
    the AST at all, and docstrings are excluded explicitly, so prose that discusses the
    defect in order to prevent it is free to do so — which is the difference between a
    codebase that documents its history and one that repeats it.

    f-strings are covered: ast.walk descends into JoinedStr and visits its Constant
    parts, which is exactly where the original defect lived
    (print(f"    robustly safe within bounds ({r.get('status')})")).
    """
    tree = ast.parse(source)
    skip = _docstring_nodes(tree)
    hits = []
    literals = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in skip:
            continue
        literals += 1
        lowered = node.value.lower()
        for claim in BANNED_CLAIMS:
            if claim in lowered:
                hits.append((label, getattr(node, 'lineno', '?'), claim, node.value))
    return hits, literals


def _python_sources():
    """
    src/ ONLY, plus the notebook — tests/ is deliberately not swept.

    Not a workaround for this file tripping its own check (it does, and so does
    test_audit2_regressions.py's _SAFETY_CLAIMS). The reason is prior to that: the
    invariant is about what the SYSTEM asserts to whoever reads its output. A test that
    names these phrases in order to search for them is machinery, not a claim — and a
    sweep that cannot state its own vocabulary without failing is a sweep that has
    confused the map for the territory.

    The notebook IS in scope and src/ IS in scope because both are things a person
    reads conclusions off: one prints to a console, the other to a Colab cell. That is
    exactly where both audits found this.
    """
    base = os.path.join(PROJECT, 'src')
    for root, _dirs, files in os.walk(base):
        if '__pycache__' in root:
            continue
        for name in sorted(files):
            if name.endswith('.py'):
                path = os.path.join(root, name)
                yield os.path.relpath(path, PROJECT), open(path).read()


def _notebook_code_cells():
    path = os.path.join(PROJECT, 'notebooks', 'colab_validation_run.ipynb')
    cells = json.load(open(path))['cells']
    for index, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            yield index, ''.join(cell['source'])


def test_the_detector_finds_a_planted_claim():
    """
    THE VACUITY GUARD, AND IT COMES FIRST ON PURPOSE.

    A sweep that silently scanned nothing would pass. A sweep whose matcher was subtly
    broken would pass. Neither failure is visible from a green result, so the scanner
    is run against sources that are KNOWN to contain the thing it looks for, in every
    shape the real defect took, before it is trusted to report an absence.

    This project has been caught four times by a loop whose property is only checked
    under a conditional; the standing rule is that something outside the loop must
    confirm the condition fired.
    """
    planted = [
        ('bare print', 'print("robustly safe within bounds")'),
        ('f-string', 'print(f"    robustly safe within bounds ({status})")'),
        ('field label', 'summary = {"verdict": "certified safe"}'),
        ('error message', 'raise ValueError("this scenario is provably safe")'),
    ]
    for label, source in planted:
        hits, _ = _claims_in_source(source, label)
        assert hits, f'the scanner missed a planted claim in the {label} case: {source}'

    # And the exclusions really exclude, or the test above would be passing for the
    # wrong reason and the sweep below would be vacuous in the other direction.
    for label, source in [
        ('comment', '# robustly safe is what this used to say\nx = 1\n'),
        ('docstring', '"""Explains why robustly safe was wrong."""\nx = 1\n'),
        ('identifier', 'robustly_safe = False\n'),
    ]:
        hits, _ = _claims_in_source(source, label)
        assert not hits, f'the scanner flagged a legitimate {label}: {hits}'


def test_no_module_states_a_safety_conclusion_it_cannot_establish():
    """
    THE SWEEP. Every .py under src/, and every notebook code cell. See _python_sources
    for why tests/ is out of scope.

    Two audits found this at two different sets of lines; Batch 10's own by-hand sweep
    then found two more that neither had cited (models.py's hyphenated "robustly-safe
    case", and README's /stats description). The lesson is not about those four places.
    """
    hits = []
    files = 0
    literals = 0

    for label, source in _python_sources():
        found, seen = _claims_in_source(source, label)
        hits.extend(found)
        files += 1
        literals += seen

    cells = 0
    skipped = []
    for index, source in _notebook_code_cells():
        try:
            found, seen = _claims_in_source(source, f'notebook cell {index}')
        except SyntaxError:
            skipped.append((index, source))
            continue
        hits.extend(found)
        cells += 1
        literals += seen

    # Coverage, asserted rather than assumed — see test_the_detector_finds_a_planted_claim.
    # Measured today: 30 files under src/, 600 non-docstring string literals in them,
    # plus the notebook's. Floors set below those rather than at them, so ordinary
    # additions do not trip this while a scan that collapsed to nothing still does.
    assert files >= 25, f'only {files} python files scanned'
    assert literals >= 500, f'only {literals} string literals examined'

    # WHAT IS SKIPPED, NOT JUST HOW MANY PARSED. A bare `cells >= N` threshold is
    # satisfied by a notebook where the skip count has quietly grown, and a cell the
    # scanner cannot parse is a cell the sweep does not cover. Measured today: 31 code
    # cells, 29 parse, 2 skip — both of which are pure shell-magic install cells that
    # are not Python and hold no prose. So the assertion is that every skip is one of
    # those, which fails the moment a real cell stops parsing.
    for index, source in skipped:
        first = next((line for line in source.splitlines() if line.strip()), '')
        assert first.lstrip().startswith(('!', '%')), (
            f'notebook cell {index} is not scanned by this sweep and is not a shell '
            f'magic cell: {first!r}'
        )
    assert cells >= 29, f'only {cells} notebook code cells parsed'

    assert not hits, 'a module states a safety conclusion it cannot establish:\n' + '\n'.join(
        f'  {label}:{line} — {claim!r} in {value!r}' for label, line, claim, value in hits
    )


def test_the_pass_two_aggregate_cell_is_actually_among_the_scanned_cells():
    """
    The sweep above counts cells; this names one. A count can be satisfied by thirty
    cells that do not include either of the two the finding was cited against, which
    would make the sweep green and meaningless for the case it exists for.
    """
    sources = {index: source for index, source in _notebook_code_cells()}
    aggregate = [i for i, s in sources.items() if 'PASS 2 AGGREGATES' in s]
    summary = [i for i, s in sources.items() if 'COLAB VALIDATION RUN — SUMMARY' in s]

    assert len(aggregate) == 1, f'expected one Pass 2 aggregate cell, got {aggregate}'
    assert len(summary) == 1, f'expected one summary cell, got {summary}'

    for index in aggregate + summary:
        ast.parse(sources[index])          # raises if the sweep would have skipped it


# ── R08 / A09: the console names every outcome ──────────────────────────────────

def test_every_outcome_gets_a_distinct_console_description():
    """
    The fix replaced a two-branch if/else with one branch per outcome. What makes that
    a fix rather than a rewording is EXHAUSTIVENESS: the original defect was a single
    `else` absorbing three different outcomes into one sentence, and an `else` that
    absorbs two is the same defect with a smaller blast radius.

    Driven from db.py's OUTCOME_* constants rather than a list written here, so an
    outcome added later and not described lands in this assertion instead of silently
    inheriting whatever the last branch says.
    """
    from src.scoring.batch_scorer import _describe_outcome
    from src.scoring import db

    outcomes = {name: getattr(db, name) for name in dir(db)
                if name.startswith('OUTCOME_')}
    # independent review, 2026-09-24: OUTCOME_HEADING_BLEND_SINGULARITY joined the
    # vocabulary (PerturbationSpace's own antipodal-blend refusal, distinct from
    # ReplayFidelityError's OUTCOME_REPLAY_INFEASIBLE) — this tripwire did exactly
    # its stated job, catching the addition here rather than letting it silently
    # inherit whatever the last branch in _describe_outcome says.
    assert len(outcomes) == 6, f'the outcome vocabulary changed: {sorted(outcomes)}'

    fields = {'min_perturbation': 0.25, 'collision_timestep': 3, 'method': 'de',
              'challengers_searched': 1, 'challengers_total': 4, 'reason': 'drift',
              'baseline_replay_error': 0.4, 'baseline_replay_collides': False,
              'baseline_heading_blend_min_magnitude': 1e-6, 'margin': 0.5,
              'error': 'ValueError: boom'}

    described = {}
    for name, outcome in outcomes.items():
        text = _describe_outcome(dict(fields, outcome=outcome))
        assert outcome in text, f'{name}: the outcome is not named in {text!r}'
        for claim in BANNED_CLAIMS:
            assert claim not in text.lower(), f'{name}: {claim!r} in {text!r}'
        described[outcome] = text

    assert len(set(described.values())) == len(described), (
        f'two outcomes produce the same console line: {described}'
    )


def test_an_unknown_outcome_is_reported_as_unknown_not_absorbed():
    """
    There is no catch-all branch that could describe a new outcome as an existing one.
    A status this function has never heard of must say so.
    """
    from src.scoring.batch_scorer import _describe_outcome

    text = _describe_outcome({'outcome': 'something_new', 'status': 'something_new'})
    assert 'something_new' in text
    assert 'unrecognized' in text.lower(), text
    for claim in BANNED_CLAIMS:
        assert claim not in text.lower(), text


# ── R11 / A08: framing must not over-reject ─────────────────────────────────────

def _tfrecord(records):
    import struct
    from src.data.loader import _masked_crc32c
    blob = b''
    for record in records:
        header = struct.pack('<Q', len(record))
        blob += (header + struct.pack('<I', _masked_crc32c(header))
                 + record + struct.pack('<I', _masked_crc32c(record)))
    return blob


@pytest.mark.parametrize('records', [
    [],                                  # empty file, a valid shard of zero records
    [b''],                               # a zero-length payload: length 0, needs 4
    [b'x'],
    [b'x' * 4096],
    [b'', b'a', b'', b'b' * 100, b''],   # zero-length records interleaved
])
@pytest.mark.parametrize('verify', [False, True])
def test_the_framing_check_accepts_every_well_formed_shard(tmp_path, records, verify):
    """
    THE REGRESSION RISK IN R11, and the reason the check is `>` and not `>=`.

    A valid FINAL record has exactly length + 4 bytes remaining, so an off-by-one in
    the comparison rejects the last record of every shard in the corpus — a validator
    that fails closed on valid input, which is worse than the defect it replaces. The
    zero-length cases are here because `length + 4 > remaining` with length 0 is the
    tightest the check ever gets.
    """
    from src.data.loader import ShardLoader

    path = tmp_path / 'good.tfrecord'
    path.write_bytes(_tfrecord(records))
    assert list(ShardLoader(str(path), verify_crc=verify)) == records


def test_a_shard_with_trailing_bytes_after_the_last_record_still_refuses(tmp_path):
    """
    The framing check must not become a reason to ACCEPT a truncated tail. Batch 5's
    B11 guarantee — a file that stops mid-record is an error, not an EOF — is what R11
    layers on top of, not something it relaxes.
    """
    from src.data.loader import ShardLoader

    path = tmp_path / 'tail.tfrecord'
    path.write_bytes(_tfrecord([b'a', b'b']) + b'abc')
    with pytest.raises(ValueError):
        list(ShardLoader(str(path)))
