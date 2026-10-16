"""
Run manifests: the offline writer (src/scoring/manifest.py), the API that serves them (GET /pipeline-runs and the `run` object of GET
/scenarios/{id}/provenance) and the allow-lists in between (src/api/manifest.py).

Pinned here:

  * the allow-lists: which keys of package_versions, counts and parameters are served, with which types; that a key that is not listed is
    NOT served (any name, any type) and a value of the wrong type is null, not repaired and not an error;
  * the writer: what it captures (git revision and dirty flag, package versions, shard name and sha256, parameters), what it leaves NULL
    when it cannot know, that credentials are removed from the command line before it is stored, that a run is finished once, that it
    refuses with a clear error when the tables are not there, and that it touches no table but the manifest's own;
  * the API: only the public columns (the command line, host label and notes never appear, whatever they hold), UTC times, newest first,
    a limit, an empty list when nothing is recorded or the tables are absent, the manifest of the run that stored a result found through
    (scenario_id, stress_run_id) and null for a result with none, and that it all works as the read-only role, which has no privilege on
    the raw table.

The database tests build a throw-away database from the real schema and deploy/rerun_class_additive.sql, and drop it after. They skip
unless AV_ROBUSTNESS_DB=1.

Run:
    AV_ROBUSTNESS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_manifest.py -q
"""

import hashlib
import json
import logging
import os
import secrets
import re
import subprocess
import sys
import uuid
from datetime import datetime, timedelta

import psycopg2
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import db_pool, main as api_main, manifest as served, routes
from src.api.config import settings
from src.api.models import PipelineRun, ProvenanceResponse
from src.scoring import db, manifest as writer

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(PROJECT, 'deploy', 'rerun_class_additive.sql')

requires_db = pytest.mark.skipif(
    os.environ.get('AV_ROBUSTNESS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

SECRET = 'hunter2-secret-value'

# ── the allow-lists ─────────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_the_served_lists_are_what_the_plan_says():
    assert served.PACKAGES == ['python', 'numpy', 'scipy', 'shapely', 'torch', 'psycopg2']
    assert served.COUNTS == ['scenarios', 'scored', 'searched', 'refused', 'errors', 'stored_results', 'collisions_found', 'no_collision_found']
    assert served.PARAMETER_INTS == ['de_seed', 'de_popsize', 'de_maxiter', 'challengers_k']
    assert served.PARAMETER_NUMBERS == ['de_tol'] and served.PARAMETER_INT_LISTS == ['seeds'] and served.MAX_SEEDS == 100


def test_the_writer_records_the_libraries_the_api_serves_and_no_others():
    assert list(writer.PACKAGES) == served.PACKAGES[1:]


def test_the_models_carry_exactly_the_served_keys():
    from src.api.models import PackageVersions, RunCounts, RunParameters
    assert list(PackageVersions.model_fields) == served.PACKAGES
    assert list(RunCounts.model_fields) == served.COUNTS
    assert list(RunParameters.model_fields) == served.PARAMETER_INTS + served.PARAMETER_NUMBERS + served.PARAMETER_INT_LISTS


GOOD_VERSIONS = {'python': '3.11.15', 'numpy': '1.26.4', 'scipy': '1.11.4', 'shapely': '2.0.3', 'torch': '2.1.0+cpu', 'psycopg2': '2.9.9'}


def test_package_versions_are_served_as_recorded_and_in_order():
    out = served.select_package_versions({**GOOD_VERSIONS, 'fastapi': '0.1', 'extra': 'x', 'pip': {'a': 1}})
    assert out == GOOD_VERSIONS and list(out) == served.PACKAGES


@pytest.mark.parametrize('key,bad', [
    ('numpy', 1), ('numpy', 1.26), ('numpy', None), ('numpy', ['1.26']), ('numpy', {'v': '1'}), ('numpy', True),
    ('numpy', ''), ('numpy', '1.26 4'), ('numpy', '1.26\n'), ('numpy', 'x' * 65), ('numpy', '1.26;rm'), ('numpy', '<b>1</b>'),
    ('python', 'three'.join(['', ' point ', ''])),
])
def test_a_wrong_package_version_is_null_not_repaired(key, bad):
    out = served.select_package_versions({**GOOD_VERSIONS, key: bad})
    assert out[key] is None
    assert out['scipy'] == '1.11.4'


def test_a_package_version_of_the_longest_allowed_length_is_served():
    assert served.select_package_versions({'numpy': 'a' * 64})['numpy'] == 'a' * 64


@pytest.mark.parametrize('blob', [None, [], 'x', 3, ['numpy']])
def test_no_object_means_no_versions_counts_or_parameters(blob):
    assert served.select_package_versions(blob) is None
    assert served.select_counts(blob) is None
    assert served.select_parameters(blob) is None


def test_an_empty_object_is_all_null_and_an_extra_key_is_never_served():
    for select, names in ((served.select_package_versions, served.PACKAGES), (served.select_counts, served.COUNTS)):
        empty = select({})
        assert list(empty) == names and all(v is None for v in empty.values())
        for value in ('secret', 7, None, ['a'], {'k': 'v'}, True):
            out = select({'zz_extra': value, '__proto__': value, 'Numpy': value})
            assert 'zz_extra' not in out and '__proto__' not in out and 'Numpy' not in out
            assert list(out) == names


def test_counts_are_served_as_recorded_zero_included():
    out = served.select_counts({'scored': 100, 'searched': 18, 'errors': 0})
    assert (out['scored'], out['searched'], out['errors'], out['refused']) == (100, 18, 0, None)


@pytest.mark.parametrize('bad', ['18', 18.0, 18.5, True, False, None, -1, [18], {'n': 18}, float('nan')])
def test_a_wrong_count_is_null(bad):
    assert served.select_counts({'scored': bad})['scored'] is None


def test_parameters_are_served_by_type():
    out = served.select_parameters({'de_seed': 0, 'de_popsize': 15, 'de_maxiter': 200, 'challengers_k': 5, 'de_tol': 0.001, 'seeds': [0, 1, 2]})
    assert out == {'de_seed': 0, 'de_popsize': 15, 'de_maxiter': 200, 'challengers_k': 5, 'de_tol': 0.001, 'seeds': [0, 1, 2]}
    assert served.select_parameters({'de_tol': 1})['de_tol'] == 1, 'an integer is a number'


@pytest.mark.parametrize('key,bad', [
    ('de_seed', '1'), ('de_seed', 1.5), ('de_seed', True), ('de_popsize', None), ('challengers_k', [5]),
    ('de_tol', '0.001'), ('de_tol', True), ('de_tol', float('inf')), ('de_tol', float('nan')), ('de_tol', None),
    ('seeds', 5), ('seeds', '0,1'), ('seeds', [0, 1.5]), ('seeds', [0, '1']), ('seeds', [True]), ('seeds', list(range(101))), ('seeds', {'a': 1}),
])
def test_a_wrong_parameter_is_null(key, bad):
    assert served.select_parameters({key: bad})[key] is None


def test_the_largest_list_of_seeds_is_served_and_an_empty_one_too():
    assert served.select_parameters({'seeds': list(range(100))})['seeds'] == list(range(100))
    assert served.select_parameters({'seeds': []})['seeds'] == []


@pytest.mark.parametrize('name', ['shard-test-00007-of-00123', 'training.tfrecord-00000-of-01000', 'a', 'A_b.c-d', '.', '..', '...hidden', 'x' * 200])
def test_a_shard_name_that_is_a_file_name_is_valid(name):
    assert served.valid_shard_name(name)


@pytest.mark.parametrize('name', [
    '', 'x' * 201, 'x' * 300, '../../etc/passwd', '/data/shards/x.tfrecord', 'dir/file', 'C:\\data\\x', 'two words', ' lead', 'trail ', 'tab\there',
    'new\nline', 'nul\x00byte', 'caf\u00e9', '<script>', 'a;b', 'a$b', 'a|b', "a'b", 'a"b', 'a%20b', 'a:b', None, 7, ['x'], {'x': 1}, True,
])
def test_anything_else_is_not_a_shard_name(name):
    assert not served.valid_shard_name(name)


def test_revisions_checksums_kinds_and_statuses_are_checked_too():
    assert served.valid_revision('a' * 40) and served.valid_sha256('0' * 64)
    for bad in ('a' * 39, 'a' * 41, 'A' * 40, 'g' * 40, ' ' + 'a' * 39, 'a' * 40 + '\n', '', None, 7):
        assert not served.valid_revision(bad), bad
    for bad in ('a' * 63, 'a' * 65, 'F' * 64, 'z' * 64, '', None, 7):
        assert not served.valid_sha256(bad), bad
    for ok in served.KINDS:
        assert served.valid_kind(ok)
    for ok in served.STATUSES:
        assert served.valid_status(ok)
    for bad in ('backfill', 'Stress', 'stress ', '', None, 1, ['stress']):
        assert not served.valid_kind(bad), bad
    for bad in ('done', 'Finished', 'finished\n', '', None, 1):
        assert not served.valid_status(bad), bad


def _view_row(**over):
    row = {'pipeline_run_id': uuid.uuid4(), 'kind': 'stress', 'status': 'finished', 'started_at': datetime(2026, 10, 1, 10, 0, tzinfo=__import__('datetime').timezone.utc),
           'finished_at': None, 'git_revision': 'a' * 40, 'git_dirty': False, 'python_version': '3.11.15', 'package_versions': GOOD_VERSIONS,
           'shard_name': 'shard-a', 'shard_sha256': 'b' * 64, 'parameters': {}, 'counts': {}}
    row.update(over)
    return row


@pytest.mark.parametrize('column,bad', [
    ('shard_name', '../../etc/passwd'), ('shard_name', 'two words'), ('shard_name', 'x' * 300), ('shard_name', '/abs/path'),
    ('git_revision', 'main'), ('git_revision', 'A' * 40), ('git_revision', 'a' * 41),
    ('shard_sha256', 'abc'), ('shard_sha256', 'B' * 64),
    ('kind', 'backfill'), ('kind', '<b>x</b>'), ('status', 'done'), ('status', 'x' * 500),
    ('git_dirty', 'yes'), ('git_dirty', 1), ('python_version', '3 11'),
])
def test_every_free_text_column_of_the_view_is_checked_before_it_is_served(column, bad):
    run = routes._pipeline_run(_view_row(**{column: bad}))
    assert getattr(run, column) is None, column
    others = routes._pipeline_run(_view_row())
    for field in ('kind', 'status', 'git_revision', 'python_version', 'shard_name', 'shard_sha256', 'git_dirty'):
        if field != column:
            assert getattr(run, field) == getattr(others, field), field


def test_a_good_row_is_served_as_it_is():
    run = routes._pipeline_run(_view_row())
    assert (run.kind, run.status, run.git_revision, run.python_version, run.shard_name, run.shard_sha256, run.git_dirty) == (
        'stress', 'finished', 'a' * 40, '3.11.15', 'shard-a', 'b' * 64, False)


def test_the_module_needs_nothing_the_deployed_api_lacks():
    text = open(os.path.join(PROJECT, 'src', 'api', 'manifest.py')).read()
    assert set(re.findall(r'^(?:import|from)\s+([a-z_]+)', text, re.M)) <= {'math', 're'}


# ── the writer, without a database ──────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize('raw,expected', [
    ('python run.py --shard x.tfrecord', 'python run.py --shard x.tfrecord'),
    (f'psql postgres://avi:{SECRET}@host:5432/db -c x', 'psql postgres://avi:***@host:5432/db -c x'),
    (f'python run.py --password {SECRET} --seed 1', 'python run.py --password *** --seed 1'),
    (f'python run.py --password={SECRET}', 'python run.py --password=***'),
    (f'python run.py -p {SECRET}', f'python run.py -p {SECRET}'),
    (f'PGPASSWORD={SECRET} python run.py', 'PGPASSWORD=*** python run.py'),
    (f'GITHUB_TOKEN={SECRET} python run.py --token {SECRET}', 'GITHUB_TOKEN=*** python run.py --token ***'),
    (f'python run.py --api-key={SECRET}', 'python run.py --api-key=***'),
    (f'API_KEY={SECRET} x', 'API_KEY=*** x'),
    ('python run.py --seed 1 --out /tmp/x', 'python run.py --seed 1 --out /tmp/x'),
    ('', ''),
])
def test_credentials_are_removed_from_a_command_line(raw, expected):
    assert writer.redact_command_line(raw) == expected
    if SECRET in raw and '-p ' not in raw:
        assert SECRET not in writer.redact_command_line(raw)


def test_a_short_flag_p_is_not_a_password_it_could_be_anything():
    # Documented limit: only the flags that name a secret are recognised; the command line is stored, never served.
    assert 'pw' in writer.redact_command_line('tool -p pw')


def test_package_versions_are_python_and_the_installed_named_packages():
    v = writer.package_versions()
    assert set(v) <= {'python', *writer.PACKAGES} and 'python' in v
    import importlib.metadata as md
    for name in v:
        if name != 'python':
            assert v[name] == md.version(name)


def test_a_package_that_is_not_installed_is_left_out_not_guessed(monkeypatch):
    import importlib.metadata as md
    real = md.version

    def fake(name):
        if name == 'torch':
            raise md.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(md, 'version', fake)
    assert 'torch' not in writer.package_versions()
    assert 'python' in writer.package_versions()


def test_the_sha256_of_a_file_is_the_files(tmp_path):
    f = tmp_path / 'shard.tfrecord'
    data = os.urandom(3 * (1 << 20) + 17)
    f.write_bytes(data)
    assert writer.sha256_file(str(f)) == hashlib.sha256(data).hexdigest()
    empty = tmp_path / 'empty'
    empty.write_bytes(b'')
    assert writer.sha256_file(str(empty)) == hashlib.sha256(b'').hexdigest()


def _git_available():
    try:
        return subprocess.run(['git', '--version'], capture_output=True).returncode == 0
    except OSError:
        return False


@pytest.mark.skipif(not _git_available(), reason='git is not installed')
def test_the_git_revision_and_the_dirty_flag_of_a_repository(tmp_path):
    def git(*a):
        subprocess.run(['git', '-c', 'user.email=a@b', '-c', 'user.name=x', *a], cwd=tmp_path, check=True, capture_output=True)
    git('init', '-q')
    (tmp_path / 'f.txt').write_text('one')
    git('add', 'f.txt')
    git('commit', '-q', '-m', 'one')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    assert writer.git_state(str(tmp_path)) == {'git_revision': head, 'git_dirty': False}
    (tmp_path / 'f.txt').write_text('two')
    assert writer.git_state(str(tmp_path)) == {'git_revision': head, 'git_dirty': True}
    git('checkout', '--', 'f.txt')
    assert writer.git_state(str(tmp_path))['git_dirty'] is False
    (tmp_path / 'new.txt').write_text('untracked')
    assert writer.git_state(str(tmp_path))['git_dirty'] is True, 'an untracked file is a tree the revision does not reproduce'


@pytest.mark.skipif(not _git_available(), reason='git is not installed')
def test_no_repository_means_no_revision_and_no_dirty_flag(tmp_path):
    assert writer.git_state(str(tmp_path / ('not-a-repo-' + uuid.uuid4().hex))) == {'git_revision': None, 'git_dirty': None}
    empty = tmp_path / 'plain'
    empty.mkdir()
    state = writer.git_state(str(empty))
    assert state['git_revision'] is None or re.fullmatch(r'[0-9a-f]{40}', state['git_revision'])  # a parent directory may be a repository


@pytest.mark.parametrize('printed', ['HEAD', 'fatal: bad revision', 'A' * 40, 'a' * 39, 'a' * 41, 'a' * 64, 'xyz'])
def test_anything_git_prints_that_is_not_a_commit_id_is_no_revision(monkeypatch, printed):
    # git was run and printed something, but not a 40-hex commit: neither the revision nor the dirty flag is recorded
    monkeypatch.setattr(writer, '_git', lambda args, cwd: printed + '\n' if args[0] == 'rev-parse' else '')
    assert writer.git_state('/anywhere') == {'git_revision': None, 'git_dirty': None}
    monkeypatch.setattr(writer, '_git', lambda args, cwd: ('b' * 40 + '\n') if args[0] == 'rev-parse' else '')
    assert writer.git_state('/anywhere') == {'git_revision': 'b' * 40, 'git_dirty': False}


def test_a_repository_with_no_commit_has_no_revision(tmp_path):
    if not _git_available():
        pytest.skip('git is not installed')
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True, capture_output=True)
    assert writer.git_state(str(tmp_path)) == {'git_revision': None, 'git_dirty': None}


# ── a throw-away database ───────────────────────────────────────────────────────────────────────────────────────────────────────


def _server():
    return settings.dsn_kwargs()


def _make_db(with_manifest):
    base = _server()
    name = f'{base["dbname"]}_mf_{uuid.uuid4().hex[:8]}'
    admin = psycopg2.connect(**base)
    admin.autocommit = True
    made_role = False
    with admin.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = 'av_api_ro'")
        made_role = cur.fetchone() is None
        cur.execute(f'CREATE DATABASE "{name}" TEMPLATE template0')
    conn = psycopg2.connect(**{**base, 'dbname': name})
    conn.autocommit = True
    with conn.cursor() as cur:
        if made_role:
            cur.execute('CREATE ROLE av_api_ro LOGIN NOSUPERUSER')
        cur.execute('GRANT USAGE ON SCHEMA public TO av_api_ro')
        cur.execute('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO av_api_ro')
    db.init_schema(conn)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO scenario_scores (scenario_id, shard, fragility_score, stress_run_id, scene_fingerprint, sdc_idx, stress_tested_at) "
                    "VALUES ('s1', 'shard-a', 1.5, 'run-s1', 'fp', 0, now()), ('s2', 'shard-a', 2.5, NULL, NULL, NULL, NULL)")
        if with_manifest:
            cur.execute(open(SCRIPT).read())
    return admin, name, conn, made_role


def _drop_db(admin, name, conn, made_role):
    conn.close()
    with admin.cursor() as cur:
        cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        if made_role:
            cur.execute('DROP ROLE IF EXISTS av_api_ro')
    admin.close()


@pytest.fixture(scope='module')
def mf_db():
    admin, name, conn, made_role = _make_db(True)
    yield name, conn
    _drop_db(admin, name, conn, made_role)


@pytest.fixture(scope='module')
def bare_db():
    admin, name, conn, made_role = _make_db(False)
    yield name, conn
    _drop_db(admin, name, conn, made_role)


@pytest.fixture
def conn(mf_db):
    _, c = mf_db
    with c.cursor() as cur:
        cur.execute('DELETE FROM pipeline_run_results; DELETE FROM pipeline_runs')
    return c


def _row(c, run_id):
    with c.cursor() as cur:
        cur.execute('SELECT kind, status, started_at, finished_at, git_revision, git_dirty, python_version, package_versions, shard_name, '
                    'shard_sha256, parameters, counts, command_line, host_label, notes FROM pipeline_runs WHERE pipeline_run_id = %s', (run_id,))
        cols = [d[0] for d in cur.description]
        r = cur.fetchone()
    return dict(zip(cols, r)) if r else None


def _scenario_fingerprint(c):
    with c.cursor() as cur:
        cur.execute("SELECT md5(string_agg(t::text, E'\\n' ORDER BY scenario_id)) FROM scenario_scores t")
        return cur.fetchone()[0]


# ── the writer, on a database ───────────────────────────────────────────────────────────────────────────────────────────────────


@requires_db
def test_a_run_begins_running_with_what_the_writer_can_know(conn):
    run = writer.begin_run(conn, 'stress', parameters={'de_seed': 3, 'de_popsize': 15}, command_line='python go.py --seed 3', host_label='laptop', notes='n')
    uuid.UUID(run)
    r = _row(conn, run)
    assert (r['kind'], r['status'], r['finished_at'], r['host_label'], r['notes']) == ('stress', 'running', None, 'laptop', 'n')
    assert r['started_at'] is not None and r['command_line'] == 'python go.py --seed 3'
    assert r['parameters'] == {'de_seed': 3, 'de_popsize': 15}
    assert r['python_version'] == sys.version.split()[0] or r['python_version'].startswith(f'{sys.version_info.major}.{sys.version_info.minor}')
    assert r['package_versions'] == writer.package_versions()
    assert r['git_revision'] is None or re.fullmatch(r'[0-9a-f]{40}', r['git_revision'])
    assert r['git_dirty'] is None or isinstance(r['git_dirty'], bool)
    assert (r['git_revision'] is None) == (r['git_dirty'] is None)
    assert (r['shard_name'], r['shard_sha256']) == (None, None)


@requires_db
def test_the_git_fields_are_those_of_the_given_repository(conn, tmp_path):
    if not _git_available():
        pytest.skip('git is not installed')
    def git(*a):
        subprocess.run(['git', '-c', 'user.email=a@b', '-c', 'user.name=x', *a], cwd=tmp_path, check=True, capture_output=True)
    git('init', '-q'); (tmp_path / 'f').write_text('1'); git('add', 'f'); git('commit', '-q', '-m', 'x'); (tmp_path / 'f').write_text('2')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    r = _row(conn, writer.begin_run(conn, 'score', repo_path=str(tmp_path), command_line='x'))
    assert (r['git_revision'], r['git_dirty']) == (head, True)
    plain = tmp_path / ('nope-' + uuid.uuid4().hex)
    r = _row(conn, writer.begin_run(conn, 'score', repo_path=str(plain), command_line='x'))
    assert (r['git_revision'], r['git_dirty']) == (None, None)


@requires_db
def test_the_shard_name_and_checksum_come_from_the_file(conn, tmp_path):
    f = tmp_path / 'training.tfrecord-00000-of-01000'
    f.write_bytes(b'abc' * 1000)
    r = _row(conn, writer.begin_run(conn, 'score', shard_path=str(f), command_line='x'))
    assert (r['shard_name'], r['shard_sha256']) == (f.name, hashlib.sha256(b'abc' * 1000).hexdigest())


@requires_db
def test_a_checksum_that_is_given_is_kept_even_when_a_file_is_named(conn, tmp_path):
    f = tmp_path / 'x.tfrecord'
    f.write_bytes(b'abc')
    r = _row(conn, writer.begin_run(conn, 'score', shard_path=str(f), shard_sha256='c' * 64, command_line='x'))
    assert r['shard_sha256'] == 'c' * 64


@requires_db
def test_a_shard_that_is_not_hashed_has_no_checksum_not_a_made_up_one(conn, tmp_path):
    f = tmp_path / 'big.tfrecord'
    f.write_bytes(b'x')
    r = _row(conn, writer.begin_run(conn, 'score', shard_path=str(f), hash_shard=False, command_line='x'))
    assert (r['shard_name'], r['shard_sha256']) == ('big.tfrecord', None)
    r = _row(conn, writer.begin_run(conn, 'score', shard_name='given-name', shard_sha256='f' * 64, command_line='x'))
    assert (r['shard_name'], r['shard_sha256']) == ('given-name', 'f' * 64)


@requires_db
def test_the_command_line_is_stored_without_its_credentials_and_defaults_to_argv(conn, monkeypatch):
    run = writer.begin_run(conn, 'rerun', command_line=f'python x.py --password {SECRET} postgres://u:{SECRET}@h/db')
    r = _row(conn, run)
    assert SECRET not in r['command_line'] and r['command_line'] == 'python x.py --password *** postgres://u:***@h/db'
    monkeypatch.setattr(sys, 'argv', ['prog', '--token', SECRET])
    r = _row(conn, writer.begin_run(conn, 'rerun'))
    assert SECRET not in r['command_line'] and r['command_line'] == 'prog --token ***'


@requires_db
def test_an_unknown_kind_is_refused_and_nothing_is_written(conn):
    for kind in ('mystery', '', 'backfill', None):
        with pytest.raises(ValueError):
            writer.begin_run(conn, kind, command_line='x')
    with conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM pipeline_runs')
        assert cur.fetchone()[0] == 0


@requires_db
def test_parameters_may_be_absent(conn):
    assert _row(conn, writer.begin_run(conn, 'score', command_line='x'))['parameters'] is None


@requires_db
def test_a_result_is_linked_once_and_relinking_updates_it(conn):
    run = writer.begin_run(conn, 'stress', command_line='x')
    writer.link_result(conn, run, 's1', 'run-s1')
    writer.link_result(conn, run, 's1', 'run-s1')
    with conn.cursor() as cur:
        cur.execute('SELECT scenario_id, stress_run_id FROM pipeline_run_results WHERE pipeline_run_id = %s', (run,))
        assert cur.fetchall() == [('s1', 'run-s1')]
    writer.link_result(conn, run, 's1', 'run-new')
    with conn.cursor() as cur:
        cur.execute('SELECT stress_run_id FROM pipeline_run_results WHERE pipeline_run_id = %s', (run,))
        assert cur.fetchall() == [('run-new',)]


@requires_db
def test_a_run_is_finished_once_with_its_counts(conn):
    run = writer.begin_run(conn, 'stress', command_line='x')
    assert writer.finish_run(conn, run, counts={'scored': 100, 'searched': 18}) is True
    r = _row(conn, run)
    assert (r['status'], r['counts']) == ('finished', {'scored': 100, 'searched': 18})
    assert r['finished_at'] >= r['started_at']
    assert writer.finish_run(conn, run, status='failed', counts={'scored': 1}) is False
    again = _row(conn, run)
    assert (again['status'], again['counts'], again['finished_at']) == ('finished', {'scored': 100, 'searched': 18}, r['finished_at'])


@requires_db
def test_a_run_can_fail_or_be_abandoned_and_an_unknown_run_is_not_finished(conn):
    for status in ('failed', 'abandoned'):
        run = writer.begin_run(conn, 'export', command_line='x')
        assert writer.finish_run(conn, run, status=status) is True
        r = _row(conn, run)
        assert (r['status'], r['counts']) == (status, None) and r['finished_at'] is not None
    assert writer.finish_run(conn, str(uuid.uuid4())) is False
    run = writer.begin_run(conn, 'export', command_line='x')
    for bad in ('running', 'done', ''):
        with pytest.raises(ValueError):
            writer.finish_run(conn, run, status=bad)
    assert _row(conn, run)['status'] == 'running'


@requires_db
def test_the_writer_touches_no_table_but_the_manifests(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM scenario_scores')
        n = cur.fetchone()[0]
    before = _scenario_fingerprint(conn)
    run = writer.begin_run(conn, 'stress', parameters={'de_seed': 1}, command_line='x')
    writer.link_result(conn, run, 's1', 'run-s1')
    writer.finish_run(conn, run, counts={'scored': 2})
    assert _scenario_fingerprint(conn) == before
    with conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM scenario_scores')
        assert cur.fetchone()[0] == n
    text = open(os.path.join(PROJECT, 'src', 'scoring', 'manifest.py')).read()
    code = re.sub(r'"""[\s\S]*?"""', '', text)
    assert not re.search(r'\b(scenario_scores|scenario_agents|perturbed_paths|update_stress_results|search_provenance)\b', code)


@requires_db
def test_without_the_tables_the_writer_says_so_and_changes_nothing(bare_db):
    _, c = bare_db
    before = _scenario_fingerprint(c)
    with pytest.raises(writer.ManifestTablesMissing, match='rerun_class_additive.sql'):
        writer.begin_run(c, 'stress', command_line='x')
    assert _scenario_fingerprint(c) == before
    with c.cursor() as cur:  # the connection is still usable
        cur.execute('SELECT 1')
        assert cur.fetchone() == (1,)
    with c.cursor() as cur:
        cur.execute("SELECT to_regclass('pipeline_runs') IS NULL")
        assert cur.fetchone()[0] is True, 'the writer must not create the tables'


# ── the API ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def _client(monkeypatch, db_conn, role=False):
    if role:
        with db_conn.cursor() as cur:
            cur.execute('SET ROLE av_api_ro')
    monkeypatch.setattr(api_main, 'init_pool', lambda: None)
    monkeypatch.setattr(api_main, 'close_pool', lambda: None)
    app = api_main.create_app()
    app.dependency_overrides[db_pool.get_db] = lambda: db_conn
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client(monkeypatch, conn):
    c = _client(monkeypatch, conn)
    with c:
        yield c


def _insert_run(c, **over):
    cols = {'kind': 'stress', 'status': 'finished', 'started_at': '2026-10-01T10:00:00+00', 'finished_at': '2026-10-01T11:30:00+00', 'git_revision': 'a' * 40,
            'git_dirty': False, 'python_version': '3.11.15', 'package_versions': json.dumps(GOOD_VERSIONS), 'shard_name': 'shard-a',
            'shard_sha256': 'b' * 64, 'parameters': json.dumps({'de_seed': 1, 'seeds': [0, 1]}), 'counts': json.dumps({'scored': 100, 'searched': 18}),
            'command_line': 'cmd', 'host_label': 'host', 'notes': 'notes'}
    cols.update(over)
    run = cols.pop('pipeline_run_id', str(uuid.uuid4()))
    names = ', '.join(['pipeline_run_id', *cols])
    marks = ', '.join(['%s'] + ['%s::jsonb' if k in ('package_versions', 'parameters', 'counts') else '%s' for k in cols])
    with c.cursor() as cur:
        cur.execute(f'INSERT INTO pipeline_runs ({names}) VALUES ({marks})', [run, *cols.values()])
    return run


@requires_db
def test_with_no_run_recorded_the_list_is_empty(client):
    r = client.get('/pipeline-runs')
    assert r.status_code == 200 and r.json() == []


@requires_db
def test_without_the_tables_the_list_is_empty_and_the_provenance_has_no_run(monkeypatch, bare_db):
    _, c = bare_db
    with _client(monkeypatch, c) as client:
        assert client.get('/pipeline-runs').json() == []
        body = client.get('/scenarios/s1/provenance').json()
        assert client.get('/scenarios/s1/provenance').status_code == 200 and body['run'] is None


@requires_db
def test_a_run_is_served_with_exactly_the_public_fields(conn, client):
    run = _insert_run(conn)
    [r] = client.get('/pipeline-runs').json()
    assert set(r) == set(PipelineRun.model_fields)
    assert r['pipeline_run_id'] == run and (r['kind'], r['status']) == ('stress', 'finished')
    assert (r['git_revision'], r['git_dirty'], r['python_version']) == ('a' * 40, False, '3.11.15')
    assert (r['shard_name'], r['shard_sha256']) == ('shard-a', 'b' * 64)
    assert r['package_versions'] == GOOD_VERSIONS
    assert r['counts']['scored'] == 100 and r['counts']['searched'] == 18 and r['counts']['refused'] is None
    assert r['parameters']['de_seed'] == 1 and r['parameters']['seeds'] == [0, 1] and r['parameters']['de_tol'] is None


@requires_db
@pytest.mark.parametrize('name', ['../../etc/passwd', '/data/shards/training.tfrecord', 'a shard name', 'x' * 300, 'dir/file', '\\\\host\\share'])
def test_a_shard_name_with_a_path_a_space_or_too_long_is_served_as_null(conn, client, name):
    _insert_run(conn, shard_name=name)
    [r] = client.get('/pipeline-runs').json()
    assert r['shard_name'] is None
    assert name not in client.get('/pipeline-runs').text
    assert r['shard_sha256'] == 'b' * 64 and r['git_revision'] == 'a' * 40


@requires_db
def test_a_shard_name_that_is_a_file_name_is_served(conn, client):
    _insert_run(conn, shard_name='training.tfrecord-00000-of-01000')
    assert client.get('/pipeline-runs').json()[0]['shard_name'] == 'training.tfrecord-00000-of-01000'


@requires_db
def test_the_command_line_host_label_and_notes_are_never_served(conn, client):
    _insert_run(conn, command_line=f'python x --password {SECRET}', host_label=f'host-{SECRET}', notes=f'notes {SECRET}')
    raw = client.get('/pipeline-runs').text + client.get('/scenarios/s1/provenance').text
    assert SECRET not in raw
    for word in ('command_line', 'host_label', 'notes'):
        assert word not in raw


@requires_db
def test_a_key_added_to_the_json_columns_is_not_served(conn, client):
    extra = {'zz_secret': SECRET, 'zz_number': 7, 'zz_list': [SECRET], 'zz_obj': {'k': SECRET}}
    _insert_run(conn,
                package_versions=json.dumps({**GOOD_VERSIONS, **extra}),
                counts=json.dumps({'scored': 5, **extra}),
                parameters=json.dumps({'de_seed': 2, **extra}))
    raw = client.get('/pipeline-runs').text
    assert SECRET not in raw and 'zz_' not in raw
    [r] = client.get('/pipeline-runs').json()
    assert list(r['package_versions']) == served.PACKAGES and list(r['counts']) == served.COUNTS
    assert list(r['parameters']) == served.PARAMETER_INTS + served.PARAMETER_NUMBERS + served.PARAMETER_INT_LISTS


@requires_db
def test_a_wrong_type_in_the_json_columns_is_null_not_an_error(conn, client):
    _insert_run(conn,
                package_versions=json.dumps({**GOOD_VERSIONS, 'numpy': 126, 'scipy': 'x y'}),
                counts=json.dumps({'scored': '100', 'searched': True, 'errors': -1, 'refused': 4}),
                parameters=json.dumps({'de_seed': 1.5, 'seeds': list(range(101)), 'de_popsize': 15, 'de_tol': 'small'}),
                python_version='3 11 15')
    r = client.get('/pipeline-runs')
    assert r.status_code == 200
    [run] = r.json()
    assert (run['package_versions']['numpy'], run['package_versions']['scipy'], run['package_versions']['torch']) == (None, None, '2.1.0+cpu')
    assert (run['counts']['scored'], run['counts']['searched'], run['counts']['errors'], run['counts']['refused']) == (None, None, None, 4)
    assert (run['parameters']['de_seed'], run['parameters']['seeds'], run['parameters']['de_popsize'], run['parameters']['de_tol']) == (None, None, 15, None)
    assert run['python_version'] is None


@requires_db
def test_the_table_itself_refuses_json_that_is_not_an_object(conn):
    for column, value in (('package_versions', '[1, 2]'), ('counts', '"x"'), ('parameters', '7')):
        with pytest.raises(psycopg2.errors.CheckViolation):
            _insert_run(conn, **{column: value})


@requires_db
def test_a_run_with_nothing_recorded_is_served_with_nulls_not_zeros(conn, client):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO pipeline_runs (pipeline_run_id, kind) VALUES (%s, 'score')", (str(uuid.uuid4()),))
    [r] = client.get('/pipeline-runs').json()
    assert r['status'] == 'running' and r['finished_at'] is None
    for field in ('git_revision', 'git_dirty', 'python_version', 'package_versions', 'shard_name', 'shard_sha256', 'parameters', 'counts'):
        assert r[field] is None, field


@requires_db
def test_the_list_is_newest_first_and_has_a_limit(conn, client):
    ids = [_insert_run(conn, started_at=f'2026-10-0{d}T10:00:00+00', finished_at=f'2026-10-0{d}T11:00:00+00') for d in (3, 1, 2)]
    assert [r['pipeline_run_id'] for r in client.get('/pipeline-runs').json()] == [ids[0], ids[2], ids[1]]
    assert [r['pipeline_run_id'] for r in client.get('/pipeline-runs?limit=2').json()] == [ids[0], ids[2]]
    assert len(client.get('/pipeline-runs?limit=1').json()) == 1
    for bad in ('0', '201', '-1', 'x'):
        assert client.get(f'/pipeline-runs?limit={bad}').status_code == 422, bad
    assert client.get('/pipeline-runs?limit=200').status_code == 200


@requires_db
def test_the_list_defaults_to_twenty(conn, client):
    for d in range(25):
        _insert_run(conn, started_at=f'2026-09-{d + 1:02d}T10:00:00+00', finished_at=f'2026-09-{d + 1:02d}T11:00:00+00')
    assert len(client.get('/pipeline-runs').json()) == 20


@requires_db
def test_the_times_are_utc_whatever_zone_the_session_uses(conn, client):
    _insert_run(conn, started_at='2026-10-01T10:00:00+00', finished_at='2026-10-01T11:30:00+00')
    with conn.cursor() as cur:
        cur.execute("SET TIME ZONE 'America/Los_Angeles'")
    try:
        [r] = client.get('/pipeline-runs').json()
    finally:
        with conn.cursor() as cur:
            cur.execute('RESET TIME ZONE')
    for field, expected in (('started_at', datetime(2026, 10, 1, 10, 0)), ('finished_at', datetime(2026, 10, 1, 11, 30))):
        got = datetime.fromisoformat(r[field].replace('Z', '+00:00'))
        assert got.utcoffset() == timedelta(0) and got.replace(tzinfo=None) == expected, (field, r[field])


@requires_db
def test_the_provenance_of_a_result_carries_the_manifest_of_the_run_that_stored_it(conn, client):
    run = _insert_run(conn)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO pipeline_run_results VALUES (%s, 's1', 'run-s1')", (run,))
    body = client.get('/scenarios/s1/provenance').json()
    assert body['run']['pipeline_run_id'] == run and body['run']['git_revision'] == 'a' * 40
    assert body['run']['package_versions'] == GOOD_VERSIONS
    for private in ('command_line', 'host_label', 'notes'):
        assert private not in body['run']


@requires_db
def test_a_result_with_no_link_has_no_run_and_that_is_not_an_error(conn, client):
    _insert_run(conn)
    for sid in ('s1', 's2'):
        r = client.get(f'/scenarios/{sid}/provenance')
        assert r.status_code == 200 and r.json()['run'] is None


@requires_db
def test_a_run_linked_to_another_result_of_the_same_scenario_is_not_this_ones(conn, client):
    run = _insert_run(conn)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO pipeline_run_results VALUES (%s, 's1', 'an-older-result')", (run,))
    assert client.get('/scenarios/s1/provenance').json()['run'] is None


@requires_db
def test_a_result_with_no_run_id_matches_no_link_even_a_null_one(conn, client):
    run = _insert_run(conn)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO pipeline_run_results VALUES (%s, 's2', NULL)", (run,))
    assert client.get('/scenarios/s2/provenance').json()['run'] is None


@requires_db
def test_a_result_with_no_run_id_does_not_ask_the_manifest_tables_anything(conn, client, monkeypatch):
    # No run id, nothing to look up: the manifest tables are not even checked for. (A lookup by a NULL id could not match anyway; the
    # guard saves the query, and this is where it is held.)
    asked = []
    real = routes._table_exists
    monkeypatch.setattr(routes, '_table_exists', lambda cur, table: asked.append(table) or real(cur, table))
    assert client.get('/scenarios/s2/provenance').json()['run'] is None
    assert asked == []
    client.get('/scenarios/s1/provenance')
    assert 'pipeline_run_results' in asked


@requires_db
def test_when_several_runs_are_linked_the_most_recent_is_served(conn, client):
    old = _insert_run(conn, started_at='2026-09-01T10:00:00+00', finished_at='2026-09-01T11:00:00+00', git_revision='1' * 40)
    new = _insert_run(conn, started_at='2026-10-05T10:00:00+00', finished_at='2026-10-05T11:00:00+00', git_revision='2' * 40)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO pipeline_run_results VALUES (%s, 's1', 'run-s1'), (%s, 's1', 'run-s1')", (old, new))
    assert client.get('/scenarios/s1/provenance').json()['run']['git_revision'] == '2' * 40


@requires_db
def test_the_provenance_without_a_manifest_is_otherwise_unchanged(conn, client):
    body = client.get('/scenarios/s1/provenance').json()
    assert body['run'] is None and body['stress_run_id'] == 'run-s1' and body['scene_fingerprint'] == 'fp'
    assert set(body) == set(ProvenanceResponse.model_fields)


@requires_db
def test_all_of_it_works_as_the_read_only_role_which_cannot_read_the_raw_table(monkeypatch, mf_db):
    _, c = mf_db
    with c.cursor() as cur:
        cur.execute('DELETE FROM pipeline_run_results; DELETE FROM pipeline_runs')
    run = _insert_run(c, command_line=f'secret {SECRET}')
    with c.cursor() as cur:
        cur.execute("INSERT INTO pipeline_run_results VALUES (%s, 's1', 'run-s1')", (run,))
    try:
        with _client(monkeypatch, c, role=True) as client:
            assert [r['pipeline_run_id'] for r in client.get('/pipeline-runs').json()] == [run]
            assert client.get('/scenarios/s1/provenance').json()['run']['pipeline_run_id'] == run
            with c.cursor() as cur:
                for stmt in ('SELECT count(*) FROM pipeline_runs', 'SELECT command_line FROM pipeline_runs_public',
                             "INSERT INTO pipeline_run_results VALUES (gen_random_uuid(), 'x', 'y')", 'DELETE FROM pipeline_runs'):
                    with pytest.raises((psycopg2.errors.InsufficientPrivilege, psycopg2.errors.UndefinedColumn)):
                        cur.execute(stmt)
                    c.rollback() if not c.autocommit else None
    finally:
        with c.cursor() as cur:
            cur.execute('RESET ROLE')


# ── a role that cannot read the manifest tables ─────────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope='module')
def roles(mf_db):
    """
    Three login roles in the throw-away database, each with USAGE on the schema and SELECT on scenario_scores only, plus: nothing
    (`deny`), the view (`view`), or the link table (`link`). Each has a password made for the run; the API connects as the role itself,
    as it does in production, not through SET ROLE (a SET undone by a rolled-back statement would hide the very thing tested).
    """
    name, owner = mf_db
    made, out = [], {}
    grants = {'deny': '', 'view': 'GRANT SELECT ON pipeline_runs_public TO {r}', 'link': 'GRANT SELECT ON pipeline_run_results TO {r}'}
    base = _server()
    with owner.cursor() as cur:
        for key, extra in grants.items():
            role = f'av_mf_{key}_{uuid.uuid4().hex[:6]}'
            password = secrets.token_hex(16)
            cur.execute(f"CREATE ROLE {role} LOGIN NOSUPERUSER PASSWORD %s", (password,))
            made.append(role)
            cur.execute(f'GRANT USAGE ON SCHEMA public TO {role}')
            cur.execute(f'GRANT SELECT ON scenario_scores TO {role}')
            if extra:
                cur.execute(extra.format(r=role))
            out[key] = {**base, 'dbname': name, 'user': role, 'password': password}
    yield out
    for role in made:
        with owner.cursor() as cur:
            cur.execute(f'REASSIGN OWNED BY {role} TO CURRENT_USER')
            cur.execute(f'DROP OWNED BY {role}')
            cur.execute(f'DROP ROLE IF EXISTS {role}')


def _as(monkeypatch, dsn):
    """An app whose connection is a real login as this role, in a transaction as the pool's connections are (not autocommit)."""
    c = psycopg2.connect(**dsn)
    monkeypatch.setattr(api_main, 'init_pool', lambda: None)
    monkeypatch.setattr(api_main, 'close_pool', lambda: None)
    app = api_main.create_app()
    app.dependency_overrides[db_pool.get_db] = lambda: c
    return c, TestClient(app, raise_server_exceptions=False)


def _seed_manifest(conn):
    run = _insert_run(conn)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO pipeline_run_results VALUES (%s, 's1', 'run-s1')", (run,))
    return run


@requires_db
def test_a_role_with_no_select_on_the_manifest_tables_gets_the_provenance_without_a_manifest_not_a_500(monkeypatch, conn, roles, caplog):
    _seed_manifest(conn)
    c, client = _as(monkeypatch, roles['deny'])
    try:
        with caplog.at_level(logging.WARNING, logger='src.api.routes'):
            r = client.get('/scenarios/s1/provenance')
        assert r.status_code == 200, r.text
        body = r.json()
        assert body['run'] is None and body['stress_run_id'] == 'run-s1' and body['scene_fingerprint'] == 'fp'
        warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING and rec.name == 'src.api.routes']
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert 'cannot be read by the API role' in message and 'DEPLOY.md' in message
        for secret in (roles['deny']['password'], roles['deny']['user'], 'pipeline_run_results', 'SELECT', 'host=', 'postgres'):
            assert secret not in message, secret
    finally:
        c.close()


@requires_db
def test_the_same_role_gets_an_empty_list_not_a_500_and_the_connection_stays_usable(monkeypatch, conn, roles, caplog):
    _seed_manifest(conn)
    c, client = _as(monkeypatch, roles['deny'])
    try:
        with caplog.at_level(logging.WARNING, logger='src.api.routes'):
            r = client.get('/pipeline-runs')
            assert r.status_code == 200 and r.json() == []
            # The failed statement aborted the transaction; the route rolled it back, so the next request on the same connection works.
            for _ in range(3):
                assert client.get('/pipeline-runs').status_code == 200
                again = client.get('/scenarios/s1/provenance')
                assert again.status_code == 200 and again.json()['scenario_id'] == 's1'
                assert client.get('/scenarios/s1').status_code == 200
    finally:
        c.close()


@requires_db
def test_a_role_that_can_read_the_view_but_not_the_link_table_lists_runs_and_serves_no_manifest(monkeypatch, conn, roles, caplog):
    run = _seed_manifest(conn)
    c, client = _as(monkeypatch, roles['view'])
    try:
        assert [r['pipeline_run_id'] for r in client.get('/pipeline-runs').json()] == [run]
        with caplog.at_level(logging.WARNING, logger='src.api.routes'):
            r = client.get('/scenarios/s1/provenance')
        assert r.status_code == 200 and r.json()['run'] is None
        assert len([x for x in caplog.records if x.levelno == logging.WARNING and x.name == 'src.api.routes']) == 1
    finally:
        c.close()


@requires_db
def test_a_role_that_can_read_the_link_table_but_not_the_view_gets_no_manifest_and_no_list(monkeypatch, conn, roles):
    _seed_manifest(conn)
    c, client = _as(monkeypatch, roles['link'])
    try:
        assert client.get('/pipeline-runs').json() == []
        r = client.get('/scenarios/s1/provenance')
        assert r.status_code == 200 and r.json()['run'] is None
    finally:
        c.close()


# ── the contract ────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_the_contract_has_the_route_and_the_manifest_in_the_provenance():
    schema = api_main.create_app().openapi()
    assert set(schema['paths']['/pipeline-runs']) == {'get'}
    props = schema['components']['schemas']['PipelineRun']['properties']
    assert set(props) == set(PipelineRun.model_fields)
    for private in ('command_line', 'host_label', 'notes'):
        assert private not in props
    assert 'run' in schema['components']['schemas']['ProvenanceResponse']['properties']
    assert set(schema['paths']['/pipeline-runs']['get']['parameters'][0]['schema']) >= {'minimum', 'maximum'}


def test_the_routes_read_the_view_and_the_link_table_and_never_the_raw_table():
    text = open(os.path.join(PROJECT, 'src', 'api', 'routes.py')).read()
    block = text[text.index('_RUN_COLUMNS ='):text.index("@router.get('/scenarios/{scenario_id}/provenance'")]
    tables = set(re.findall(r'\b(?:FROM|JOIN)\s+([a-z_]+)', block))
    assert tables == {'pipeline_run_results', 'pipeline_runs_public'}


# ── what the tables themselves refuse ───────────────────────────────────────────────────────────────────────────────────────────


@requires_db
@pytest.mark.parametrize('column,bad', [
    ('git_revision', 'a' * 39), ('git_revision', 'A' * 40), ('git_revision', 'HEAD'), ('git_revision', 'a' * 41),
    ('shard_sha256', 'b' * 63), ('shard_sha256', 'B' * 64), ('shard_sha256', 'sha256'), ('shard_sha256', 'b' * 65),
])
def test_the_table_refuses_a_revision_or_checksum_that_is_not_hex_of_the_right_length(conn, column, bad):
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert_run(conn, **{column: bad})


@requires_db
def test_the_table_refuses_a_running_run_with_a_finish_time_and_a_finished_one_without(conn):
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert_run(conn, status='running', finished_at='2026-10-01T11:30:00+00')
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert_run(conn, status='finished', finished_at=None)


@requires_db
def test_the_table_refuses_a_run_that_finishes_before_it_starts(conn):
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert_run(conn, started_at='2026-10-01T12:00:00+00', finished_at='2026-10-01T11:00:00+00')


@requires_db
def test_a_link_needs_a_run_and_goes_with_it(conn):
    with pytest.raises(psycopg2.errors.ForeignKeyViolation):
        with conn.cursor() as cur:
            cur.execute("INSERT INTO pipeline_run_results VALUES (gen_random_uuid(), 's1', 'run-s1')")
    run = _insert_run(conn)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO pipeline_run_results VALUES (%s, 's1', 'run-s1')", (run,))
        cur.execute('DELETE FROM pipeline_runs WHERE pipeline_run_id = %s', (run,))
        cur.execute('SELECT count(*) FROM pipeline_run_results')
        assert cur.fetchone()[0] == 0
