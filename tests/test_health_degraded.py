"""
GET /health says "degraded" when the database answers but PostGIS does not.

Until now the route reported status "ok" whenever the database answered, and put a null in `postgis` when the PostGIS
version call failed. A reader looking only at `status` (the Corpus page's health panel prints it) saw "ok" for a service
that cannot serve geometry. Now:

  database answers, PostGIS answers          200  {"status": "ok",       "database": "connected", "postgis": "<version>"}
  database answers, PostGIS missing/raising  200  {"status": "degraded", "database": "connected", "postgis": null}
  database not reachable                     as before: 503 with Retry-After when the pool is exhausted, and the 500 the
                                             connection error has always produced when nothing can connect

The status code stays 200 in the degraded case on purpose: Render's health check is pointed at /health (DEPLOY.md), and
Render takes an instance out of rotation on any 4xx or 5xx, which would take down the scores endpoints that still work.

The "missing" case is a real database created from template0, which has no extensions; the "raising" case is a database
that has PostGIS, reached over connections that cannot resolve its functions. The pool holds ONE connection in each, so a
second request proves the connection came back to the pool usable (the failed statement aborts the transaction; the route
rolls it back).

Run:
    AV_ROBUSTNESS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_health_degraded.py -q
"""

import os
import sys
import uuid

import psycopg2
import pytest
from fastapi.testclient import TestClient
from psycopg2.pool import ThreadedConnectionPool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import db_pool, main as api_main, routes

requires_db = pytest.mark.skipif(
    os.environ.get('AV_ROBUSTNESS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

def _base():
    return api_main.settings.dsn_kwargs()


def _client(monkeypatch, pool):
    monkeypatch.setattr(db_pool, '_pool', pool)
    monkeypatch.setattr(api_main, 'init_pool', lambda: None)
    monkeypatch.setattr(api_main, 'close_pool', lambda: None)
    # No context manager: the lifespan would replace the pool this test controls.
    return TestClient(api_main.create_app(), raise_server_exceptions=False)


@pytest.fixture
def pool():
    p = ThreadedConnectionPool(1, 1, **_base())
    yield p
    p.closeall()


@pytest.fixture
def pool_without_postgis():
    """A database with no extensions, made for the test and dropped after it."""
    base = _base()
    name = f'{base["dbname"]}_nopg_{uuid.uuid4().hex[:8]}'
    admin = psycopg2.connect(**base)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{name}" TEMPLATE template0')
    p = ThreadedConnectionPool(1, 1, **{**base, 'dbname': name})
    try:
        yield p, name
    finally:
        p.closeall()
        with admin.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()


# ── PostGIS answers ─────────────────────────────────────────────────────────────


@requires_db
def test_postgis_answers_so_status_is_ok(monkeypatch, pool):
    response = _client(monkeypatch, pool).get('/health')
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {'status', 'database', 'postgis'}
    assert (body['status'], body['database']) == ('ok', 'connected')
    assert isinstance(body['postgis'], str) and body['postgis'][:1].isdigit(), body


# ── PostGIS missing ─────────────────────────────────────────────────────────────


@requires_db
def test_a_database_without_postgis_is_degraded_with_a_200(monkeypatch, pool_without_postgis):
    pool, name = pool_without_postgis
    with pool.getconn() as probe, probe.cursor() as cur:  # premise: the extension really is absent
        cur.execute("SELECT count(*) FROM pg_extension WHERE extname = 'postgis'")
        assert cur.fetchone()[0] == 0, name
    pool.putconn(probe)

    client = _client(monkeypatch, pool)
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'degraded', 'database': 'connected', 'postgis': None}

    # The one pooled connection came back usable: the failed statement was rolled back.
    again = client.get('/health')
    assert again.status_code == 200
    assert again.json() == {'status': 'degraded', 'database': 'connected', 'postgis': None}


@requires_db
def test_the_route_itself_leaves_the_connection_usable(pool_without_postgis):
    # Through the pool the connection is also rolled back when it is returned (psycopg2's putconn does that for a connection
    # that is not idle), so the route's own rollback cannot be seen from outside the pool. Called directly, with no pool
    # clean-up after it, it is: the failed PostGIS statement has aborted the transaction, and the next statement on the
    # same connection works only because the route rolled it back.
    pool, _ = pool_without_postgis
    connection = pool.getconn()
    try:
        answer = routes.health(conn=connection)
        assert (answer.status, answer.database, answer.postgis) == ('degraded', 'connected', None)
        with connection.cursor() as cur:
            cur.execute('SELECT 1')
            assert cur.fetchone()[0] == 1
    finally:
        pool.putconn(connection)


# ── PostGIS raising ─────────────────────────────────────────────────────────────


@requires_db
def test_postgis_that_raises_is_degraded_with_a_200(monkeypatch, pool):
    # The extension is installed, but these connections cannot resolve its functions, so PostGIS_Version() raises.
    connection = pool.getconn()
    with connection.cursor() as cur:
        cur.execute("SELECT count(*) FROM pg_extension WHERE extname = 'postgis'")
        assert cur.fetchone()[0] == 1  # premise: it is installed
        cur.execute('SET search_path = pg_catalog')
    connection.commit()
    pool.putconn(connection)

    client = _client(monkeypatch, pool)
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'degraded', 'database': 'connected', 'postgis': None}
    assert client.get('/health').status_code == 200  # the connection is usable for the next request


# ── database not reachable: unchanged ───────────────────────────────────────────


@requires_db
def test_pool_exhaustion_is_still_a_503_with_retry_after(monkeypatch, pool):
    held = pool.getconn()
    try:
        response = _client(monkeypatch, pool).get('/health')
    finally:
        pool.putconn(held)
    assert response.status_code == 503
    assert response.json() == {'detail': 'All database connections are busy. Retry shortly.'}
    assert response.headers['Retry-After'] == '1'


def test_a_database_nothing_can_connect_to_is_still_the_500_it_was(monkeypatch):
    # No pool yet and nothing listening: the first request cannot connect. This is today's behaviour and is not changed;
    # what matters here is that it is not a 200 that says "ok" or "degraded".
    monkeypatch.setattr(db_pool, '_pool', None)
    monkeypatch.setattr(api_main.settings.__class__, 'dsn_kwargs', lambda self: {**_base(), 'host': '127.0.0.1', 'port': 1})
    monkeypatch.setattr(api_main, 'init_pool', db_pool.init_pool)
    monkeypatch.setattr(api_main, 'close_pool', lambda: None)
    response = TestClient(api_main.create_app(), raise_server_exceptions=False).get('/health')
    assert response.status_code == 500
    assert response.text == 'Internal Server Error'
