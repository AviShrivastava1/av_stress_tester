"""
test_notebook_external_db.py — cell 12/13's external-database mode.

The A_MAX=12 re-run writes to a hosted Postgres (Render) instead of the throwaway
Postgres the notebook installs inside Colab. That changes three things this file pins:

  * cell 12 reads ONE Colab secret holding the database URL, parses it with
    psycopg2.extensions.parse_dsn (libpq's own parser, the one psycopg2 connects
    with), exports the PG* variables from it, installs nothing, and never prints the
    credentials;
  * the connection must use TLS (Render rejects sslmode=disable), so PGSSLMODE
    defaults to require and weaker modes are refused rather than passed through;
  * cell 13 refuses to continue, before any pass writes, unless the connection is
    what it should be, and shows whether the database already holds stored results
    and under which acceleration cap they were searched.

The cell-12 tests execute the REAL cell source with google.colab.userdata stubbed and
subprocess.run replaced by a function that fails the test if called. The cell-13 tests
need a DISPOSABLE Postgres/PostGIS database and skip unless AV_CLAIMS_DB=1, the same
requirement as every other DB-gated file here; they DROP the three project tables.

Run:
    ./venv/bin/python -m pytest tests/test_notebook_external_db.py -q
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_notebook_external_db.py -q
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(PROJECT, 'notebooks', 'colab_validation_run.ipynb')

PG_VARS = ('PGHOST', 'PGPORT', 'PGUSER', 'PGPASSWORD', 'PGDATABASE', 'PGSSLMODE')
SECRET = 'AV_STRESS_DATABASE_URL'
# `.invalid` is reserved (RFC 2606): these hosts can never resolve, so no test here can
# reach a real server even if a variable escaped its fixture.
URL = 'postgresql://render_user:s3cr%40t%2Fpw@dpg-example.oregon-postgres.invalid:5432/av_stress'

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)


def _cell_by_content(*required):
    cells = json.load(open(NOTEBOOK))['cells']
    hits = [''.join(c['source']) for c in cells
            if c['cell_type'] == 'code' and all(t in ''.join(c['source']) for t in required)]
    assert len(hits) == 1, f'expected exactly one cell matching {required}, got {len(hits)}'
    return hits[0]


def _db_setup_cell():
    return _cell_by_content('DB_MODE', 'apt-get')


def _schema_cell():
    return _cell_by_content('init_geometry_schema', 'PostGIS_Version()')


def _run(source, namespace):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, '<notebook cell>', 'exec'), namespace)
    return out.getvalue()


@pytest.fixture
def clean_env():
    """
    Every PG* variable starts unset, and ALL of them are restored exactly afterwards.

    Not monkeypatch.delenv: that restores only variables that existed beforehand, so
    anything the cell under test sets on a variable that was unset would leak into the
    tests that run next. An earlier version of this fixture did exactly that, and the
    DB-gated tests below then tried to connect to this file's fake host.
    """
    saved = {var: os.environ.get(var) for var in PG_VARS}
    for var in PG_VARS:
        os.environ.pop(var, None)
    yield
    for var, value in saved.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


@pytest.fixture
def colab_secret(monkeypatch):
    """Stubs google.colab.userdata; returns the dict its get() reads from."""
    secrets = {}

    class SecretNotFoundError(Exception):
        pass

    def get(name):
        if name not in secrets:
            raise SecretNotFoundError(name)
        return secrets[name]

    google = types.ModuleType('google')
    colab = types.ModuleType('google.colab')
    userdata = types.ModuleType('google.colab.userdata')
    userdata.get = get
    colab.userdata = userdata
    google.colab = colab
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    monkeypatch.setitem(sys.modules, 'google.colab.userdata', userdata)
    return secrets


@pytest.fixture
def no_subprocess(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f'external mode ran a subprocess: {args[0] if args else kwargs}')
    monkeypatch.setattr(subprocess, 'run', refuse)


def _external(**extra):
    namespace = {'os': os, 'DB_MODE': 'external', 'DB_URL_SECRET': SECRET,
                 'PG_USER': 'avi', 'PG_PASSWORD': 'avi', 'PG_DB': 'av_stress'}
    namespace.update(extra)
    return namespace


# ── cell 12, external mode ──────────────────────────────────────────────────────

def test_external_mode_exports_the_url_and_installs_nothing(clean_env, colab_secret, no_subprocess):
    colab_secret[SECRET] = URL
    out = _run(_db_setup_cell(), _external())

    assert os.environ['PGHOST'] == 'dpg-example.oregon-postgres.invalid'
    assert os.environ['PGPORT'] == '5432'
    assert os.environ['PGUSER'] == 'render_user'
    assert os.environ['PGPASSWORD'] == 's3cr@t/pw', 'percent-encoding was not decoded'
    assert os.environ['PGDATABASE'] == 'av_stress'
    assert os.environ['PGSSLMODE'] == 'require', 'TLS must default to require'

    assert 's3cr@t/pw' not in out and 's3cr%40t' not in out, 'the password was printed'
    assert 'render_user' not in out, 'the user name was printed'
    assert 'dpg-example.oregon-postgres.invalid' in out and 'av_stress' in out


def test_a_missing_port_leaves_libpqs_default(clean_env, colab_secret, no_subprocess):
    colab_secret[SECRET] = 'postgresql://u:p@db.example.invalid/av_stress'
    _run(_db_setup_cell(), _external())
    assert 'PGPORT' not in os.environ


def test_an_explicit_strong_sslmode_is_kept(clean_env, colab_secret, no_subprocess):
    colab_secret[SECRET] = URL + '?sslmode=verify-full'
    _run(_db_setup_cell(), _external())
    assert os.environ['PGSSLMODE'] == 'verify-full'


@pytest.mark.parametrize('url,message', [
    (URL + '?sslmode=disable', 'sslmode'),
    (URL + '?sslmode=prefer', 'sslmode'),
    ('postgresql://u:p@localhost:5432/av_stress', 'localhost'),
    ('postgresql://u:p@127.0.0.1/av_stress', 'localhost'),
    (URL + '?application_name=x', 'application_name'),
    ('mysql://u:p@db.example.invalid/av_stress', 'parse'),
    ('postgresql:///av_stress', 'host'),
])
def test_external_mode_refuses_a_url_it_cannot_use_safely(clean_env, colab_secret, no_subprocess,
                                                          url, message):
    colab_secret[SECRET] = url
    with pytest.raises(RuntimeError) as refused:
        _run(_db_setup_cell(), _external())
    assert message in str(refused.value)
    assert 'u:p' not in str(refused.value) and ':p@' not in str(refused.value), (
        'the refusal message echoed the credentials'
    )
    # The traceback Colab prints includes any chained exception, and parse_dsn's own
    # error echoes the whole connection string. So nothing may be chained visibly.
    shown_context = refused.value.__cause__ or (
        None if refused.value.__suppress_context__ else refused.value.__context__)
    assert shown_context is None, f'a chained exception would be printed: {shown_context!r}'
    assert not any(var in os.environ for var in PG_VARS), 'a refused URL still exported variables'


def test_a_missing_secret_says_what_to_set(clean_env, colab_secret, no_subprocess):
    with pytest.raises(RuntimeError) as refused:
        _run(_db_setup_cell(), _external())
    assert SECRET in str(refused.value)


def test_an_unknown_mode_is_refused(clean_env, colab_secret, no_subprocess):
    with pytest.raises(RuntimeError, match='DB_MODE'):
        _run(_db_setup_cell(), _external(DB_MODE='render'))


# ── cell 12, local mode still installs and exports localhost ────────────────────

def test_local_mode_installs_postgres_and_exports_localhost(clean_env, monkeypatch):
    calls = []

    def record(cmd, *args, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout='', stderr='')

    monkeypatch.setattr(subprocess, 'run', record)
    real_listdir = os.listdir
    monkeypatch.setattr(os, 'listdir',
                        lambda p: ['16'] if p == '/usr/lib/postgresql' else real_listdir(p))
    monkeypatch.setenv('PGSSLMODE', 'require')   # left over from an earlier external run

    namespace = _external(DB_MODE='colab_local')
    _run(_db_setup_cell(), namespace)

    flat = [' '.join(c) if isinstance(c, (list, tuple)) else c for c in calls]
    assert any('postgresql-16-postgis-3' in c for c in flat), flat
    assert any('CREATE EXTENSION IF NOT EXISTS postgis' in c for c in flat), flat
    assert os.environ['PGHOST'] == 'localhost'
    assert os.environ['PGPASSWORD'] == 'avi'
    assert 'PGSSLMODE' not in os.environ, 'a stale PGSSLMODE would break the local connection'


# ── cell 13, before any pass writes ─────────────────────────────────────────────

def _drop_project_tables():
    from src.scoring import db
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, '
                    'scenario_scores CASCADE')
    connection.commit()
    connection.close()


def _seed_stored_results():
    from psycopg2.extras import Json
    from src.scoring import db
    connection = db.get_connection()
    db.init_schema(connection)
    rows = [('s-5', {'a_max': 5.0}), ('s-12a', {'a_max': 12.0}), ('s-12b', {'a_max': 12.0}),
            ('s-legacy', {'method': 'de'}), ('s-null', None), ('s-untested', 'untested')]
    with connection.cursor() as cur:
        for sid, provenance in rows:
            tested = provenance != 'untested'
            cur.execute(
                """INSERT INTO scenario_scores (scenario_id, fragility_score, stress_tested_at,
                                                search_provenance)
                   VALUES (%s, 1.0, CASE WHEN %s THEN now() END, %s)""",
                (sid, tested, Json(provenance) if tested and provenance is not None else None),
            )
    connection.commit()
    connection.close()


@requires_db
def test_the_schema_cell_reports_stored_results_by_cap(monkeypatch):
    monkeypatch.delenv('PGSSLMODE', raising=False)
    _drop_project_tables()
    _seed_stored_results()

    namespace = {'DB_MODE': 'colab_local'}
    out = _run(_schema_cell(), namespace)

    assert 'PostGIS_Version():' in out
    assert 'scenario_scores rows: 6' in out
    assert 'stored results: 5' in out
    assert 'a_max=5.0: 1' in out
    assert 'a_max=12.0: 2' in out
    assert 'no a_max recorded: 2' in out
    namespace['conn'].close()


@requires_db
def test_external_mode_refuses_a_connection_without_tls(monkeypatch):
    # The local disposable server has ssl off, and nothing sets PGSSLMODE, so the
    # connection succeeds in plain text: exactly what the check must refuse.
    monkeypatch.delenv('PGSSLMODE', raising=False)
    _drop_project_tables()
    with pytest.raises(RuntimeError, match='TLS'):
        _run(_schema_cell(), {'DB_MODE': 'external'})

    from src.scoring import db
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute("SELECT to_regclass('scenario_scores')")
        assert cur.fetchone()[0] is None, 'the schema was written before the TLS check'
    connection.close()
