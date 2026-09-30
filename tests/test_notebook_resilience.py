"""
test_notebook_resilience.py — the re-run finishes, or resumes, in one Colab session.

Three changes this file pins, each needed by the A_MAX=12 re-run against Render:

  * 10b is skipped by default (RUN_B09 = False). Its result is already recorded, and it
    is by far the longest cell. Skipped means it says so in one line and names where the
    recorded result is; it never runs silently to nothing.
  * The database connection is refreshed before each write pass, because the one cell
    13 opens sits idle through several minutes of diagnostics. `conn` is bound ONLY
    inside cell 13's refresh_connection(), which applies the same external-mode TLS
    check cell 13 does, so a reconnect can never quietly drop encryption.
  * Pass 1 and Pass 2 results can be checkpointed to Drive and resumed. A checkpoint
    loads only if the commit, shard, MAX_SCENARIOS, TOP_N, DE_KWARGS and A_MAX all match
    this session, and a Pass 2 checkpoint only if it was made against the same Pass 1
    ranking, so stress_results can never pair with a different ranking.

Cells are located by content. Notebook and runbook code is executed from the REAL
source, with only external dependencies stubbed. The refresh_connection tests need a
DISPOSABLE Postgres/PostGIS database and skip unless AV_CLAIMS_DB=1.

Run:
    ./venv/bin/python -m pytest tests/test_notebook_resilience.py -q
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_notebook_resilience.py -q
"""

import ast
import contextlib
import io
import json
import os
import pickle
import re
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.physics.bicycle_model as bicycle_model

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(PROJECT, 'notebooks', 'colab_validation_run.ipynb')
RUNBOOK = os.path.join(PROJECT, 'notebooks', 'COLAB_RUNBOOK.md')

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)


def _code_cells():
    return [''.join(c['source']) for c in json.load(open(NOTEBOOK))['cells']
            if c['cell_type'] == 'code']


def _cell_by_content(*required):
    hits = [s for s in _code_cells() if all(t in s for t in required)]
    assert len(hits) == 1, f'expected exactly one cell matching {required}, got {len(hits)}'
    return hits[0]


def _run(source, namespace):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, '<notebook cell>', 'exec'), namespace)
    return out.getvalue()


# ── 10b: skipped by default, and says so ────────────────────────────────────────

def _cell_10b():
    return _cell_by_content('_refine_recording_both')


def test_run_b09_defaults_to_false_in_the_config_cell():
    config = _cell_by_content('the only cell you should need to edit')
    assert re.search(r'^RUN_B09 = False\b', config, re.M), 'RUN_B09 must default to False'


def test_a_skipped_10b_prints_one_line_naming_the_recorded_result_and_runs_nothing():
    namespace = {'RUN_B09': False}
    out = _run(_cell_10b(), namespace)
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) == 1, f'expected exactly one line when skipped, got {lines}'
    line = lines[0]
    assert 'skipped' in line and 'RUN_B09' in line
    assert '6 evaluable' in line and 'warm start' in line, 'the recorded result is not named'
    assert 'runbook' in line, 'the skip line does not say where the result is recorded'
    assert '_refine_recording_both' not in namespace, 'a skipped 10b still defined its helpers'


def test_10b_s_whole_body_is_behind_the_flag():
    tree = ast.parse(_cell_10b())
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.If), (
        '10b must be a single `if` on RUN_B09 so nothing runs, or is imported, when skipped'
    )
    assert 'RUN_B09' in ast.unparse(tree.body[0].test)


def _markdown_10b():
    hits = [''.join(c['source']) for c in json.load(open(NOTEBOOK))['cells']
            if c['cell_type'] == 'markdown' and '## 10b.' in ''.join(c['source'])]
    assert len(hits) == 1, f'expected one 10b markdown cell, got {len(hits)}'
    return hits[0]


def _runbook_10b():
    runbook = open(RUNBOOK).read()
    start = runbook.index('### 10b')
    return runbook[start:runbook.index('\n#', start + 1)]


def test_the_recorded_b09_result_is_not_credited_to_the_run_whose_10b_was_interrupted():
    # The last full run (cache commit 32a2853) interrupted 10b before it finished, so the
    # 6-of-6 result cannot have come from it. The runbook says where it did come from.
    assert '32a2853' not in _markdown_10b(), '10b markdown credits the result to 32a2853'
    section = _runbook_10b()
    assert 'interrupted' in section, 'the runbook does not say the 32a2853 run never finished 10b'
    assert 'not been re-derived' in section, 'the runbook does not state the provenance limit'


def test_the_recorded_b09_result_names_the_run_and_the_physics_it_was_measured_under():
    section = ' '.join(_runbook_10b().split())
    assert 'as relayed in review' not in section, 'the provenance is still unspecific'
    for fact in ('first real-shard run', '496 scenarios', 'freshness guard', '`A_MAX = 5`',
                 '6 of the 25 sampled scenarios were evaluable', 'drift gate refused 18'):
        assert fact in section, f'the runbook does not state: {fact}'
    assert 'evaluate more scenarios' in section, 'it does not say what re-measuring under 12 changes'


def test_the_block_4_note_is_described_as_decided_everywhere():
    runbook = ' '.join(open(RUNBOOK).read().split())
    for stale in ('If it fires on real data', 'this would be the first one still open',
                  "10b's earlier run"):
        assert stale not in runbook, f'the runbook still says {stale!r}'
    markdown = ' '.join(_markdown_10b().split())
    assert 'Nobody knows whether' not in markdown, '10b markdown calls the answer unknown'
    # The note exists: it is drafted in the review session's doctrine corrections.
    for stale in ("Writing it is Avi's", "Avi's to write"):
        assert stale not in runbook, f'the runbook still says {stale!r}'
    assert runbook.count("drafted in the review session's doctrine corrections") == 3, (
        'the decision paragraph, "What to bring back" and "does not do" must all say where the note is'
    )


# ── the connection: refreshed before each write, never without the TLS check ────

WRITE_CELLS = [('n1 = db.upsert_scores',), ('Confirmed: Phase 4 columns landed', 'fetch_top'),
               ('PASS 3 SUMMARY',)]


@pytest.mark.parametrize('anchor', WRITE_CELLS)
def test_every_write_pass_refreshes_the_connection_first(anchor):
    first = ast.parse(_cell_by_content(*anchor)).body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call), ast.unparse(first)
    assert ast.unparse(first.value) == 'refresh_connection()', ast.unparse(first)


def test_conn_is_bound_only_inside_refresh_connection():
    """
    One place creates the connection every later cell uses, so no cell can read a
    connection that skipped the TLS check, and test_batch13's shadowed-read sweep has
    nothing ambiguous to find: no cell binds `conn` at module scope at all.
    """
    for source in _code_cells():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        inside = {id(x) for f in functions for x in ast.walk(f)}
        stray = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Name) and n.id == 'conn' and isinstance(n.ctx, ast.Store)
                 and id(n) not in inside]
        assert not stray, f'a cell binds `conn` outside refresh_connection:\n{source[:200]}'
    schema_cell = _cell_by_content('def refresh_connection')
    fn = next(n for n in ast.walk(ast.parse(schema_cell))
              if isinstance(n, ast.FunctionDef) and n.name == 'refresh_connection')
    assert any(isinstance(n, ast.Global) and 'conn' in n.names for n in ast.walk(fn))


def _schema_cell():
    return _cell_by_content('init_geometry_schema', 'PostGIS_Version()')


def _drop_project_tables():
    from src.scoring import db
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, '
                    'scenario_scores CASCADE')
    connection.commit()
    connection.close()


@requires_db
def test_refresh_connection_replaces_the_connection_and_closes_the_old_one(monkeypatch):
    monkeypatch.delenv('PGSSLMODE', raising=False)
    _drop_project_tables()
    namespace = {'DB_MODE': 'colab_local'}
    _run(_schema_cell(), namespace)
    old = namespace['conn']

    returned = namespace['refresh_connection']()

    assert namespace['conn'] is returned and returned is not old
    assert old.closed, 'the replaced connection was left open'
    with returned.cursor() as cur:
        cur.execute('SELECT 1')
        assert cur.fetchone()[0] == 1
    returned.close()


@requires_db
def test_a_reconnect_in_external_mode_without_tls_raises_like_cell_13(monkeypatch):
    # The disposable server has ssl off: the connection succeeds in plain text, which is
    # exactly what external mode must refuse, on a reconnect as much as in cell 13.
    monkeypatch.delenv('PGSSLMODE', raising=False)
    _drop_project_tables()
    namespace = {'DB_MODE': 'colab_local'}
    _run(_schema_cell(), namespace)
    kept = namespace['conn']

    namespace['DB_MODE'] = 'external'
    with pytest.raises(RuntimeError, match='TLS'):
        namespace['refresh_connection']()

    assert namespace['conn'] is kept, 'a refused reconnect replaced the connection anyway'
    kept.close()


# ── Pass 1 / Pass 2 checkpoints ─────────────────────────────────────────────────

def _run_metadata_source():
    tree = ast.parse(_cell_by_content('Freshness check passed'))
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_metadata']
    assert len(fn) == 1, 'run_metadata() is not defined at the top level of cell 9'
    return ast.unparse(fn[0])


def _session(**overrides):
    namespace = {'subprocess': subprocess, 'REPO_DIR': PROJECT,
                 'SHARD_PATH': '/content/drive/MyDrive/waymo_data/shard-00000',
                 'MAX_SCENARIOS': 100, 'TOP_N': 5,
                 'DE_KWARGS': dict(popsize=15, maxiter=200, tol=1e-3, seed=1)}
    namespace.update(overrides)
    exec(compile(_run_metadata_source(), '<run_metadata>', 'exec'), namespace)
    return namespace


def _block(*required):
    blocks = re.findall(r'```python\n(.*?)```', open(RUNBOOK).read(), re.S)
    hits = [b for b in blocks if all(t in b for t in required)]
    assert len(hits) == 1, f'expected one runbook code block with {required}, got {len(hits)}'
    return hits[0]


PASS1 = '/content/drive/MyDrive/av_stress_pass1.pkl'
PASS2 = '/content/drive/MyDrive/av_stress_pass2.pkl'


def _paths(source, tmp_path):
    return source.replace(PASS1, str(tmp_path / 'p1.pkl')).replace(PASS2, str(tmp_path / 'p2.pkl'))


def _records(bump=None):
    records = [{'scenario_id': f's{i:02d}', 'fragility_score': 1.0 + (i % 3)} for i in range(12)]
    if bump:
        records[0]['fragility_score'] += bump
    return records


def _save_pass1(tmp_path, session, records):
    session.update(records=records, errors=[])
    _run(_paths(_block(PASS1, 'pickle.dump'), tmp_path), session)


def _save_pass2(tmp_path, session, records):
    from src.scoring.ranker import top_n_ids
    ids = top_n_ids(records, session['TOP_N'])
    session.update(records=records, ids_to_test=ids,
                   stress_results={sid: {'status': 'ok'} for sid in ids})
    _run(_paths(_block(PASS2, 'pickle.dump'), tmp_path), session)


def test_run_metadata_records_everything_a_resume_must_match():
    meta = _session()['run_metadata']()
    assert set(meta) == {'commit', 'shard_path', 'max_scenarios', 'top_n', 'de_kwargs', 'a_max'}
    assert re.fullmatch(r'[0-9a-f]{40}', meta['commit'])
    assert meta['a_max'] == bicycle_model.A_MAX


def test_both_checkpoints_resume_in_a_matching_session(tmp_path):
    records = _records()
    _save_pass1(tmp_path, _session(), records)
    _save_pass2(tmp_path, _session(), records)

    resumed = _session()
    _run(_paths(_block(PASS1, 'pickle.load'), tmp_path), resumed)
    assert resumed['records'] == records and resumed['errors'] == []
    _run(_paths(_block(PASS2, 'pickle.load'), tmp_path), resumed)
    assert resumed['ids_to_test'] == ['s02', 's05', 's08', 's11', 's01']
    assert set(resumed['stress_results']) == set(resumed['ids_to_test'])


@pytest.mark.parametrize('key,value', [
    ('commit', '0' * 40), ('shard_path', '/elsewhere'), ('max_scenarios', 50),
    ('top_n', 20), ('de_kwargs', {'popsize': 15}), ('a_max', 5.0),
])
def test_a_pass_1_checkpoint_from_a_different_run_is_refused(tmp_path, key, value):
    _save_pass1(tmp_path, _session(), _records())
    saved = pickle.load(open(tmp_path / 'p1.pkl', 'rb'))
    saved['meta'][key] = value
    pickle.dump(saved, open(tmp_path / 'p1.pkl', 'wb'))

    resumed = _session()
    with pytest.raises(AssertionError):
        _run(_paths(_block(PASS1, 'pickle.load'), tmp_path), resumed)
    assert 'records' not in resumed, 'a refused checkpoint still bound its names'


def test_a_changed_session_setting_refuses_a_checkpoint_made_before_it(tmp_path, monkeypatch):
    _save_pass1(tmp_path, _session(), _records())
    monkeypatch.setattr(bicycle_model, 'A_MAX', 5.0)
    with pytest.raises(AssertionError):
        _run(_paths(_block(PASS1, 'pickle.load'), tmp_path), _session())


def test_a_pass_2_checkpoint_made_against_a_different_ranking_is_refused(tmp_path):
    _save_pass2(tmp_path, _session(), _records())
    resumed = _session(records=_records(bump=5.0))    # same run settings, different Pass 1
    with pytest.raises(AssertionError, match='ranking'):
        _run(_paths(_block(PASS2, 'pickle.load'), tmp_path), resumed)
    assert 'stress_results' not in resumed


def test_a_pass_2_checkpoint_is_refused_before_pass_1_is_in_place(tmp_path):
    _save_pass2(tmp_path, _session(), _records())
    with pytest.raises(AssertionError, match='Pass 1'):
        _run(_paths(_block(PASS2, 'pickle.load'), tmp_path), _session())
