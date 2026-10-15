"""
deploy/search_requests.sql: the table, the function and the role that queued search requests use.

A request is a scenario id and a seed, filed through the API and run later, offline, by the owner. This file tests the
database side on a throw-away database made for the test and dropped after it (never the database the PG* variables name,
which is only used to reach the server):

  * the script is re-runnable, and no schema initialiser in the repository creates the table;
  * the role av_req_insert holds no privilege on any table, its schema or anything else, and can call exactly one function;
    every direct read or write of the table, and any read of the stored-result tables, is refused;
  * the read-only API role, if it exists with the default privilege DEPLOY.md gives it, cannot read the table;
  * the function: a new request is filed; the same scenario and seed returns the first request, even from another
    requester and even when every limit is reached; each limit refuses, writes nothing, and says when to retry; the limits
    count what they say they count; two filings at once cannot both pass the same count;
  * the owner's pending-requests file runs, lists the waiting requests oldest first, and cannot write.

Run:
    AV_ROBUSTNESS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_search_requests_schema.py -q
"""

import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import uuid

import psycopg2
import psycopg2.errors
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.config import settings
from src.scoring import db

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(PROJECT, 'deploy', 'search_requests.sql')
PENDING = os.path.join(PROJECT, 'deploy', 'pending_search_requests.sql')

requires_db = pytest.mark.skipif(
    os.environ.get('AV_ROBUSTNESS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

ROLE = 'av_req_insert'
READ_ROLE = 'av_api_ro'
#: The throw-away password the fixture gives the request role when it made the role, so that a connection AS the role works on a
#: server that asks for passwords (CI's does; a laptop that trusts local connections does not). None when the role was already
#: there: its password is not ours to change, and then only a trusting server can be used.
ROLE_PASSWORD = None
HASH_A = 'a' * 64
HASH_B = 'b' * 64
HASH_C = 'c' * 64


def _server():
    return settings.dsn_kwargs()


@pytest.fixture(scope='module')
def scratch():
    """A database from template0 with the real score schema and the real script applied; its name."""
    base = _server()
    name = f'{base["dbname"]}_req_{uuid.uuid4().hex[:8]}'
    admin = psycopg2.connect(**base)
    admin.autocommit = True
    global ROLE_PASSWORD
    made_roles = []
    ROLE_PASSWORD = None
    with admin.cursor() as cur:
        for role in (ROLE, READ_ROLE):
            cur.execute('SELECT 1 FROM pg_roles WHERE rolname = %s', (role,))
            if cur.fetchone() is None:
                made_roles.append(role)
        cur.execute(f'CREATE DATABASE "{name}" TEMPLATE template0')
    owner = psycopg2.connect(**{**base, 'dbname': name})
    owner.autocommit = True
    try:
        db.init_schema(owner)
        with owner.cursor() as cur:
            cur.execute("INSERT INTO scenario_scores (scenario_id, shard, fragility_score) VALUES ('s1', 'x', 1.0), ('s2', 'x', 2.0)")
            # The read-only API role as DEPLOY.md step 2 makes it, including the default privilege on later tables.
            if READ_ROLE in made_roles:
                cur.execute(f'CREATE ROLE {READ_ROLE} LOGIN NOSUPERUSER')
            cur.execute(f'GRANT USAGE ON SCHEMA public TO {READ_ROLE}')
            cur.execute(f'GRANT SELECT ON scenario_scores TO {READ_ROLE}')
            cur.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {READ_ROLE}')
            cur.execute(open(SCRIPT).read())
            if ROLE in made_roles:  # made for this test: give it a password made for this test
                ROLE_PASSWORD = secrets.token_hex(16)
                cur.execute(f"ALTER ROLE {ROLE} PASSWORD %s", (ROLE_PASSWORD,))
        yield name
    finally:
        ROLE_PASSWORD = None
        owner.close()
        with admin.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            for role in made_roles:
                cur.execute(f'DROP ROLE IF EXISTS {role}')
        admin.close()


@pytest.fixture
def owner(scratch):
    conn = psycopg2.connect(**{**_server(), 'dbname': scratch})
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute('DELETE FROM search_requests')
    yield conn
    conn.close()


def _as_role(scratch, role=ROLE):
    """A connection that has become the role (the cluster trusts local connections, so this is the same as logging in)."""
    conn = psycopg2.connect(**{**_server(), 'dbname': scratch})
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f'SET ROLE {role}')
    return conn


def _file(conn, scenario='s1', seed=0, who=HASH_A, *, active=3, per_day=10, per_requester_day=None, queue=10, rid=None):
    """file_search_request with the DESIGN limits unless told otherwise; returns (outcome, id, retry_after)."""
    per_requester_day = per_day if per_requester_day is None else per_requester_day
    rid = rid or str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute('SELECT outcome, filed_request_id, retry_after_seconds FROM file_search_request('
                    '%s::uuid, %s, %s, %s, %s, %s, %s, %s)',
                    (rid, scenario, seed, who, active, per_requester_day, per_day, queue))
        outcome, filed, retry = cur.fetchone()
    return outcome, (str(filed) if filed else None), retry


def _count(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM search_requests')
        return cur.fetchone()[0]


# ── the script ──────────────────────────────────────────────────────────────────────────────


@requires_db
def test_the_script_can_be_run_again_and_leaves_one_of_each(scratch, owner):
    with owner.cursor() as cur:
        cur.execute(open(SCRIPT).read())
        cur.execute(open(SCRIPT).read())
        cur.execute("SELECT count(*) FROM pg_proc WHERE proname = 'file_search_request'")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM pg_roles WHERE rolname = %s", (ROLE,))
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM pg_class WHERE relname = 'search_requests' AND relkind = 'r'")
        assert cur.fetchone()[0] == 1


@requires_db
def test_the_function_is_security_definer_with_a_fixed_search_path(owner):
    with owner.cursor() as cur:
        cur.execute("SELECT prosecdef, proconfig FROM pg_proc WHERE proname = 'file_search_request'")
        secdef, config = cur.fetchone()
    assert secdef is True
    assert config == ['search_path=public, pg_catalog']


def test_no_schema_initialiser_in_the_repository_creates_the_table():
    for folder in ('src',):
        for root, _dirs, files in os.walk(os.path.join(PROJECT, folder)):
            for filename in files:
                if filename.endswith('.py'):
                    text = open(os.path.join(root, filename)).read()
                    assert not re.search(r'CREATE\s+TABLE[^;]*search_requests', text, re.I | re.S), filename
                    # Naming the script in a docstring is fine; reading or running it is not.
                    assert not re.search(r'open\([^)]*search_requests\.sql|execute\([^)]*search_requests\.sql', text), filename
    for filename in os.listdir(os.path.join(PROJECT, 'src', 'scoring')):
        if filename.endswith('.py'):
            assert 'search_requests' not in open(os.path.join(PROJECT, 'src', 'scoring', filename)).read(), filename


@requires_db
def test_the_schema_initialisers_do_not_create_it(scratch):
    base = _server()
    name = f'{scratch}_init'
    admin = psycopg2.connect(**base)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{name}" TEMPLATE template0')
    try:
        conn = psycopg2.connect(**{**base, 'dbname': name})
        db.init_schema(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('search_requests') IS NULL, to_regproc('file_search_request') IS NULL")
            assert cur.fetchone() == (True, True)
        conn.close()
    finally:
        with admin.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()


# ── the role ────────────────────────────────────────────────────────────────────────────────


@requires_db
def test_the_role_can_log_in_and_has_no_other_attribute(owner):
    with owner.cursor() as cur:
        cur.execute('SELECT rolcanlogin, rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls '
                    'FROM pg_roles WHERE rolname = %s', (ROLE,))
        assert cur.fetchone() == (True, False, False, False, False, False)


@requires_db
def test_the_role_holds_no_privilege_on_any_table_or_the_schema(owner):
    with owner.cursor() as cur:
        cur.execute("""
            SELECT c.relname, p.priv
              FROM pg_class c
              JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = 'public'
              CROSS JOIN (VALUES ('SELECT'), ('INSERT'), ('UPDATE'), ('DELETE'), ('TRUNCATE'),
                                 ('REFERENCES'), ('TRIGGER')) AS p(priv)
             WHERE c.relkind IN ('r', 'p', 'v', 'm', 'S')
               AND has_table_privilege(%s, c.oid, p.priv)
        """, (ROLE,))
        assert cur.fetchall() == []
        cur.execute("SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = 'public' AND c.relkind = 'r'")
        assert cur.fetchone()[0] >= 2, 'premise: there are tables to hold no privilege on'
        cur.execute("SELECT has_schema_privilege(%s, 'public', 'CREATE'), has_schema_privilege(%s, 'public', 'USAGE')",
                    (ROLE, ROLE))
        assert cur.fetchone() == (False, True)
        cur.execute("SELECT has_database_privilege(%s, current_database(), 'CREATE')", (ROLE,))
        assert cur.fetchone() == (False,)


@requires_db
def test_the_role_can_execute_the_one_function_and_no_other(owner):
    with owner.cursor() as cur:
        cur.execute("""
            SELECT p.proname
              FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace AND n.nspname = 'public'
             WHERE has_function_privilege(%s, p.oid, 'EXECUTE')
        """, (ROLE,))
        assert cur.fetchall() == [('file_search_request',)]


@requires_db
def test_the_role_cannot_read_or_write_the_table_directly(scratch, owner):
    _file(owner)
    conn = _as_role(scratch)
    with conn.cursor() as cur:
        for sql in ("SELECT * FROM search_requests",
                    "SELECT count(*) FROM search_requests",
                    "INSERT INTO search_requests (request_id, scenario_id, seed, requester_hash) "
                    f"VALUES ('{uuid.uuid4()}', 's1', 9, '{HASH_A}')",
                    "UPDATE search_requests SET status = 'done'",
                    "DELETE FROM search_requests",
                    "TRUNCATE search_requests",
                    "DROP TABLE search_requests"):
            with pytest.raises((psycopg2.errors.InsufficientPrivilege, psycopg2.errors.UndefinedTable)):
                cur.execute(sql)
    conn.close()
    assert _count(owner) == 1


@requires_db
def test_the_role_cannot_read_the_stored_results_or_create_anything(scratch):
    conn = _as_role(scratch)
    with conn.cursor() as cur:
        for sql in ("SELECT * FROM scenario_scores",
                    "UPDATE scenario_scores SET n_agents = 1",
                    "CREATE TABLE t (i int)",
                    "CREATE FUNCTION f() RETURNS int LANGUAGE sql AS 'SELECT 1'"):
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute(sql)
    conn.close()


@requires_db
def test_the_role_can_file_a_request_through_the_function(scratch, owner):
    conn = _as_role(scratch)
    outcome, rid, retry = _file(conn, 's1', 3)
    conn.close()
    assert (outcome, retry) == ('filed', None)
    with owner.cursor() as cur:
        cur.execute('SELECT request_id::text, scenario_id, seed, requester_hash, status FROM search_requests')
        assert cur.fetchall() == [(rid, 's1', 3, HASH_A, 'queued')]


@requires_db
def test_the_read_only_api_role_cannot_read_the_table_even_with_the_default_privilege(owner):
    with owner.cursor() as cur:
        cur.execute("SELECT has_table_privilege(%s, 'scenario_scores', 'SELECT')", (READ_ROLE,))
        assert cur.fetchone() == (True,), 'premise: the default privilege and the grant are in place'
        cur.execute("""SELECT priv FROM (VALUES ('SELECT'), ('INSERT'), ('UPDATE'), ('DELETE'), ('TRUNCATE'),
                                          ('REFERENCES'), ('TRIGGER')) AS p(priv)
                        WHERE has_table_privilege(%s, 'search_requests', priv)""", (READ_ROLE,))
        assert cur.fetchall() == []
        cur.execute("SELECT has_function_privilege(%s, 'file_search_request(uuid,text,integer,text,integer,integer,integer,integer)', 'EXECUTE')",
                    (READ_ROLE,))
        assert cur.fetchone() == (False,)


# ── the function ────────────────────────────────────────────────────────────────────────────


@requires_db
def test_a_new_request_is_filed_and_stored_as_queued(owner):
    outcome, rid, retry = _file(owner, 's1', 7, HASH_A)
    assert (outcome, retry) == ('filed', None)
    with owner.cursor() as cur:
        cur.execute('SELECT request_id::text, scenario_id, seed, requester_hash, status, requested_at <= now() '
                    'FROM search_requests')
        assert cur.fetchall() == [(rid, 's1', 7, HASH_A, 'queued', True)]


@requires_db
def test_the_same_scenario_and_seed_returns_the_first_request_and_stores_nothing_more(owner):
    first = _file(owner, 's1', 7, HASH_A)
    again = _file(owner, 's1', 7, HASH_A)
    other = _file(owner, 's1', 7, HASH_B)
    assert first[0] == 'filed'
    assert again == ('existing', first[1], None)
    assert other == ('existing', first[1], None), 'another requester gets the first request, not a second one'
    assert _count(owner) == 1


@requires_db
def test_a_different_seed_or_scenario_is_a_different_request(owner):
    ids = {_file(owner, 's1', 1)[1], _file(owner, 's1', 2)[1], _file(owner, 's2', 1)[1]}
    assert len(ids) == 3 and None not in ids
    assert _count(owner) == 3


@requires_db
def test_an_existing_request_is_returned_even_when_every_limit_is_reached(owner):
    first = _file(owner, 's1', 1, HASH_A, active=1, per_day=1, queue=1)
    assert first[0] == 'filed'
    assert _file(owner, 's2', 1, HASH_A, active=1, per_day=1, queue=1)[0] != 'filed', 'premise: limits reached'
    assert _file(owner, 's1', 1, HASH_A, active=1, per_day=1, queue=1) == ('existing', first[1], None)
    assert _file(owner, 's1', 1, HASH_B, active=1, per_day=1, queue=1) == ('existing', first[1], None)


@requires_db
def test_a_requester_may_have_so_many_queued_or_running_at_once_and_no_more(owner):
    for seed in range(3):
        assert _file(owner, 's1', seed, HASH_A, active=3)[0] == 'filed'
    outcome, rid, retry = _file(owner, 's1', 9, HASH_A, active=3)
    assert (outcome, rid, retry) == ('requester_active', None, 3600)
    assert _count(owner) == 3
    assert _file(owner, 's1', 10, HASH_B, active=3)[0] == 'filed', 'another requester is not held back by it'


@requires_db
def test_running_counts_as_active_and_done_does_not(owner):
    ids = [_file(owner, 's1', seed, HASH_A, active=3)[1] for seed in range(3)]
    with owner.cursor() as cur:
        cur.execute("UPDATE search_requests SET status = 'running' WHERE request_id = %s", (ids[0],))
    assert _file(owner, 's1', 20, HASH_A, active=3)[0] == 'requester_active'
    for status in ('done', 'failed', 'expired', 'cancelled'):
        with owner.cursor() as cur:
            cur.execute("UPDATE search_requests SET status = %s WHERE request_id = %s", (status, ids[0]))
        outcome = _file(owner, 's1', 30, HASH_A, active=3)[0]
        assert outcome == 'filed', status
        with owner.cursor() as cur:
            cur.execute("DELETE FROM search_requests WHERE seed = 30")


@requires_db
def test_a_requester_may_file_so_many_in_a_day_and_the_refusal_says_when_to_retry(owner):
    for seed in range(4):
        assert _file(owner, 's1', seed, HASH_A, active=99, per_day=99, per_requester_day=4)[0] == 'filed'
    with owner.cursor() as cur:  # the oldest was filed 23 hours ago, so it leaves the window in about an hour
        cur.execute("UPDATE search_requests SET requested_at = now() - interval '23 hours' WHERE seed = 0")
    outcome, rid, retry = _file(owner, 's1', 9, HASH_A, active=99, per_day=99, per_requester_day=4)
    assert (outcome, rid) == ('requester_daily', None)
    assert 3500 <= retry <= 3600, retry
    assert _count(owner) == 4
    assert _file(owner, 's1', 10, HASH_B, active=99, per_day=99, per_requester_day=4)[0] == 'filed'


@requires_db
def test_the_requesters_retry_after_is_the_time_until_the_oldest_in_the_window_leaves_it(owner):
    for seed in range(2):
        assert _file(owner, 's1', seed, HASH_A, active=99, per_day=99, per_requester_day=2)[0] == 'filed'
    with owner.cursor() as cur:  # filed 20 hours ago: it leaves the window in 4 hours, not in an hour
        cur.execute("UPDATE search_requests SET requested_at = now() - interval '20 hours' WHERE seed = 0")
    outcome, rid, retry = _file(owner, 's1', 9, HASH_A, active=99, per_day=99, per_requester_day=2)
    assert (outcome, rid) == ('requester_daily', None)
    assert 14300 <= retry <= 14400, retry
    with owner.cursor() as cur:  # the oldest is the one that counts, not the newest
        cur.execute("UPDATE search_requests SET requested_at = now() - interval '2 hours' WHERE seed = 1")
    assert 14300 <= _file(owner, 's1', 9, HASH_A, active=99, per_day=99, per_requester_day=2)[2] <= 14400


@requires_db
def test_a_request_older_than_a_day_no_longer_counts_toward_the_daily_limits(owner):
    for seed in range(4):
        assert _file(owner, 's1', seed, HASH_A, active=99, per_day=4, per_requester_day=4)[0] == 'filed'
    with owner.cursor() as cur:
        cur.execute("UPDATE search_requests SET requested_at = now() - interval '25 hours' WHERE seed = 0")
    assert _file(owner, 's1', 9, HASH_A, active=99, per_day=4, per_requester_day=4)[0] == 'filed'


@requires_db
def test_the_queue_has_a_length_and_the_refusal_says_to_come_back_later(owner):
    for i, who in enumerate((HASH_A, HASH_B, HASH_C)):
        assert _file(owner, 's1', i, who, queue=3)[0] == 'filed'
    outcome, rid, retry = _file(owner, 's2', 0, 'd' * 64, queue=3)
    assert (outcome, rid, retry) == ('queue_full', None, 3600)
    assert _count(owner) == 3
    with owner.cursor() as cur:  # a request that is running or done is not in the queue
        cur.execute("UPDATE search_requests SET status = 'running' WHERE seed = 0")
    assert _file(owner, 's2', 0, 'd' * 64, queue=3)[0] == 'filed'


@requires_db
def test_all_requesters_together_may_file_so_many_in_a_day(owner):
    who = ['a' * 64, 'b' * 64, 'c' * 64, 'd' * 64]
    for seed, w in enumerate(who):
        assert _file(owner, 's1', seed, w, per_day=4, queue=99)[0] == 'filed'
    with owner.cursor() as cur:
        cur.execute("UPDATE search_requests SET status = 'done'")  # the queue is empty; the day's count is not
        cur.execute("UPDATE search_requests SET requested_at = now() - interval '22 hours' WHERE seed = 0")
    outcome, rid, retry = _file(owner, 's2', 0, 'e' * 64, per_day=4, queue=99)
    assert (outcome, rid) == ('global_daily', None)
    assert 7000 <= retry <= 7200, retry
    assert _count(owner) == 4


@requires_db
def test_a_refusal_writes_nothing_and_each_limit_is_named_for_what_it_counts(owner):
    # One requester at its active limit while the day and the queue still have room: it is the requester's limit.
    for seed in range(2):
        _file(owner, 's1', seed, HASH_A, active=2, per_day=50, queue=10)
    assert _file(owner, 's1', 5, HASH_A, active=2, per_day=50, queue=10)[0] == 'requester_active'
    # Everything full: the requester's own limit is reported first, so a capped requester is told about itself.
    assert _file(owner, 's1', 6, HASH_A, active=2, per_day=2, queue=2)[0] == 'requester_active'
    # A requester with room meets the shared limits.
    assert _file(owner, 's1', 7, HASH_B, active=2, per_day=2, queue=50)[0] == 'global_daily'
    assert _file(owner, 's1', 7, HASH_B, active=2, per_day=50, queue=2)[0] == 'queue_full'
    assert _count(owner) == 2


@requires_db
def test_two_filings_at_once_cannot_both_pass_the_same_count(scratch, owner):
    results = []
    lock = threading.Lock()

    def worker(i):
        conn = psycopg2.connect(**{**_server(), 'dbname': scratch})
        conn.autocommit = True
        outcome = _file(conn, 's1', i, f'{i:x}' * 64 if i < 16 else HASH_A, active=99, per_day=99, queue=3)[0]
        with lock:
            results.append(outcome)
        conn.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ['filed'] * 3 + ['queue_full'] * 5, results
    assert _count(owner) == 3


@requires_db
def test_the_same_request_filed_twice_at_once_is_stored_once(scratch, owner):
    results = []
    lock = threading.Lock()

    def worker():
        conn = psycopg2.connect(**{**_server(), 'dbname': scratch})
        conn.autocommit = True
        result = _file(conn, 's1', 4, HASH_A)
        with lock:
            results.append(result)
        conn.close()

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(r[0] for r in results) == ['existing'] * 5 + ['filed']
    assert len({r[1] for r in results}) == 1
    assert _count(owner) == 1


@requires_db
def test_the_table_refuses_what_the_api_would_never_send(owner):
    bad = [
        ("'s1', -1, '%s'" % HASH_A),
        ("'s1', 2147483648, '%s'" % HASH_A),
        ("'s1', 1, 'not-a-hash'"),
        ("'s1', 1, '%s'" % HASH_A.upper().replace('A', 'F')),
        ("'s1', 1, '%s'" % ('a' * 63)),
    ]
    for values in bad:
        # 2147483648 does not fit the integer column at all; the others fail the CHECK.
        with pytest.raises((psycopg2.errors.CheckViolation, psycopg2.errors.NumericValueOutOfRange)):
            with owner.cursor() as cur:
                cur.execute(f"INSERT INTO search_requests (request_id, scenario_id, seed, requester_hash) "
                            f"VALUES ('{uuid.uuid4()}', {values})")
    with pytest.raises(psycopg2.errors.CheckViolation):
        with owner.cursor() as cur:
            cur.execute(f"INSERT INTO search_requests (request_id, scenario_id, seed, requester_hash, status) "
                        f"VALUES ('{uuid.uuid4()}', 's1', 1, '{HASH_A}', 'pending')")
    assert _count(owner) == 0


@requires_db
def test_the_table_itself_refuses_a_second_request_for_the_same_scenario_and_seed(owner):
    insert = ("INSERT INTO search_requests (request_id, scenario_id, seed, requester_hash) "
              "VALUES ('%s', 's1', 1, '" + HASH_A + "')")
    with owner.cursor() as cur:
        cur.execute(insert % uuid.uuid4())
        with pytest.raises(psycopg2.errors.UniqueViolation):
            cur.execute(insert % uuid.uuid4())
    assert _count(owner) == 1


# ── the owner's pending-requests file ───────────────────────────────────────────────────────


def _psql():
    found = shutil.which('psql') or os.path.join(os.environ.get('AV_PG_BIN', ''), 'psql')
    return found if found and os.path.exists(found) else None


@requires_db
@pytest.mark.skipif(_psql() is None, reason='psql is not installed')
def test_the_pending_file_lists_the_waiting_requests_oldest_first_and_no_others(scratch, owner):
    ids = {}
    for seed in range(5):
        ids[seed] = _file(owner, 's1', seed, HASH_A, active=99, per_day=99, queue=99)[1]
    with owner.cursor() as cur:
        cur.execute("UPDATE search_requests SET requested_at = now() - interval '3 hours' WHERE seed = 2")
        cur.execute("UPDATE search_requests SET requested_at = now() - interval '2 hours' WHERE seed = 0")
        cur.execute("UPDATE search_requests SET requested_at = now() - interval '1 hour' WHERE seed = 3")
        cur.execute("UPDATE search_requests SET status = 'running' WHERE seed = 3")
        cur.execute("UPDATE search_requests SET status = 'done' WHERE seed = 1")
        cur.execute("UPDATE search_requests SET status = 'failed' WHERE seed = 4")
    base = _server()
    env = {**os.environ, 'PGHOST': str(base['host']), 'PGPORT': str(base['port']), 'PGUSER': base['user'],
           'PGDATABASE': scratch}
    run = subprocess.run([_psql(), '-X', '-q', '-A', '-t', '-F', '|', '-f', PENDING], capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr
    lines = [line for line in run.stdout.splitlines() if line.strip()]
    # Queued and running are waiting; done and failed are not. Oldest first.
    assert [line.split('|')[:4] for line in lines] == [[ids[2], 's1', '2', 'queued'], [ids[0], 's1', '0', 'queued'],
                                                         [ids[3], 's1', '3', 'running']]


def test_the_pending_file_is_read_only_text():
    sql = open(PENDING).read()
    statements = re.sub(r'--[^\n]*', '', sql)
    assert 'BEGIN READ ONLY;' in statements
    assert not re.search(r'\b(INSERT|UPDATE|DELETE|TRUNCATE|DROP|ALTER|CREATE|GRANT|REVOKE|COPY)\b', statements, re.I)
    assert statements.index('BEGIN READ ONLY;') < statements.index('SELECT')


def test_the_script_says_it_is_run_by_hand_and_sets_no_password():
    text = open(SCRIPT).read()
    assert 'RUN BY HAND' in text
    assert not re.search(r"PASSWORD\s+'", text, re.I), 'a password never appears in the script'
    assert '\\password av_req_insert' in text
