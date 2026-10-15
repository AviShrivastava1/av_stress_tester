"""
POST /search-requests: queued search requests, OFF unless API_SEARCH_REQUESTS is exactly "1".

A request is a scenario id and a seed, filed now and run later, offline, by the owner. Pinned here:

  * OFF (the default): the route is not in the app and not in the published openapi.json, a POST gets the 404 of an unknown
    path, the CORS methods are GET only, the request pool is never opened, and only the value "1" turns it on;
  * ON: the route, its contract, the CORS methods, and a start-up that refuses a half-configured switch;
  * the body is one scenario id and one bounded integer, validated; anything else is refused and nothing is stored;
  * a scenario that is not in scenario_scores is a 404 and nothing is stored;
  * the same scenario and seed returns the existing request, even for a requester at a limit;
  * every limit, with the real constants: per-requester concurrent and daily, global daily, queue length: its status code,
    its Retry-After, and that nothing is stored;
  * the requester is an HMAC of its address under the salt; the address and the salt are in no stored row; behind a proxy
    only the entry the proxy added is believed;
  * the request path uses its own role and pool, and nothing in the read path mentions the request table.

The database tests run on a throw-away database (the fixture of test_search_requests_schema.py) and, through the real
deploy/search_requests.sql, as the real role av_req_insert.

Run:
    AV_ROBUSTNESS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_search_requests_api.py -q
"""

import json
import os
import re
import sys
import uuid

import psycopg2
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import config, db_pool, main as api_main, request_pool, search_requests
from src.api.config import settings
from tests.test_search_requests_schema import ROLE, requires_db, scratch  # noqa: F401  (scratch is a fixture)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALT = 'test-salt-' + uuid.uuid4().hex  # made for this run; never printed, never stored


def _enable(monkeypatch, *, hops=0, salt=SALT, user=ROLE):
    monkeypatch.setattr(settings, 'search_requests_enabled', True)
    monkeypatch.setattr(settings, 'request_pg_user', user)
    monkeypatch.setattr(settings, 'request_pg_password', '')
    monkeypatch.setattr(settings, 'request_salt', salt)
    monkeypatch.setattr(settings, 'request_proxy_hops', hops)


def _raw_request(headers=None, peer=('10.0.0.1', 5000)):
    scope = {'type': 'http', 'method': 'POST', 'path': '/search-requests', 'query_string': b'',
             'headers': [(k.lower().encode(), v.encode()) for k, v in (headers or [])], 'client': peer}
    return Request(scope)


# ── off: the default ────────────────────────────────────────────────────────────────────────


def test_it_is_off_unless_the_setting_is_exactly_1(monkeypatch):
    for value, expected in ((None, False), ('', False), ('0', False), ('true', False), ('yes', False), (' 1', False),
                            ('1 ', False), ('01', False), ('1', True)):
        if value is None:
            monkeypatch.delenv('API_SEARCH_REQUESTS', raising=False)
        else:
            monkeypatch.setenv('API_SEARCH_REQUESTS', value)
        assert config.Settings().search_requests_enabled is expected, value


def test_every_default_is_the_safe_one(monkeypatch):
    for name in ('API_SEARCH_REQUESTS', 'API_REQUEST_PGUSER', 'API_REQUEST_PGPASSWORD', 'API_REQUEST_POOL_MAX',
                 'API_REQUEST_SALT', 'API_REQUEST_PROXY_HOPS'):
        monkeypatch.delenv(name, raising=False)
    s = config.Settings()
    assert s.search_requests_enabled is False
    assert (s.request_pg_user, s.request_pg_password, s.request_salt) == ('', '', ''), 'no default user, password or salt'
    assert s.request_pool_max == 2, 'a small pool of its own'
    assert s.request_proxy_hops == 0, 'the peer of the connection, not a forwarded header nobody has checked'


def test_the_running_settings_are_off_in_this_environment():
    assert os.environ.get('API_SEARCH_REQUESTS') is None
    assert settings.search_requests_enabled is False


def test_with_it_off_the_route_is_not_in_the_app_or_in_openapi(monkeypatch):
    monkeypatch.setattr(settings, 'search_requests_enabled', False)
    app = api_main.create_app()
    schema = app.openapi()
    assert '/search-requests' not in schema['paths']
    assert not [name for name in schema['components']['schemas'] if 'SearchRequest' in name]
    assert 'search-requests' not in json.dumps(schema).lower().replace('search requests', '')
    assert 'search request' not in schema['info']['description'].lower()
    assert 'POST' not in {m for route in app.routes for m in getattr(route, 'methods', set()) or set()}
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post('/search-requests', json={'scenario_id': 's1', 'seed': 1}).status_code == 404
    assert client.get('/search-requests').status_code == 404


def test_the_committed_openapi_does_not_have_the_route():
    text = open(os.path.join(PROJECT, 'frontend', 'openapi.json')).read()
    assert 'search-requests' not in text and 'SearchRequest' not in text


def test_with_it_off_the_cors_methods_are_get_only(monkeypatch):
    monkeypatch.setattr(settings, 'search_requests_enabled', False)
    client = TestClient(api_main.create_app(), raise_server_exceptions=False)
    origin = settings.cors_origins[0]
    post = client.options('/scenarios', headers={'Origin': origin, 'Access-Control-Request-Method': 'POST'})
    assert post.status_code == 400
    get = client.options('/scenarios', headers={'Origin': origin, 'Access-Control-Request-Method': 'GET'})
    assert get.status_code == 200


def test_with_it_off_the_request_pool_is_never_opened(monkeypatch):
    monkeypatch.setattr(settings, 'search_requests_enabled', False)
    opened = []
    monkeypatch.setattr(api_main, 'init_pool', lambda: None)
    monkeypatch.setattr(api_main, 'close_pool', lambda: None)
    monkeypatch.setattr(api_main, 'init_request_pool', lambda: opened.append(1))
    with TestClient(api_main.create_app()):
        pass
    assert opened == []
    assert request_pool._pool is None


# ── on: the contract ────────────────────────────────────────────────────────────────────────


def test_with_it_on_the_route_is_published_with_one_scenario_id_and_one_seed(monkeypatch):
    _enable(monkeypatch)
    schema = api_main.create_app().openapi()
    operation = schema['paths']['/search-requests']['post']
    assert set(schema['paths']['/search-requests']) == {'post'}
    assert set(operation['responses']) == {'200', '201', '404', '422', '429', '503'}
    body = schema['components']['schemas']['SearchRequestBody']
    assert set(body['properties']) == {'scenario_id', 'seed'}
    assert body['additionalProperties'] is False
    assert body['required'] == ['scenario_id', 'seed']
    assert body['properties']['seed']['minimum'] == 0 and body['properties']['seed']['maximum'] == 2**31 - 1
    assert body['properties']['scenario_id']['minLength'] == 1
    assert 'Nothing runs on request' in schema['info']['description']


def test_with_it_on_the_cors_methods_allow_post(monkeypatch):
    _enable(monkeypatch)
    client = TestClient(api_main.create_app(), raise_server_exceptions=False)
    r = client.options('/search-requests', headers={'Origin': settings.cors_origins[0],
                                                    'Access-Control-Request-Method': 'POST',
                                                    'Access-Control-Request-Headers': 'content-type'})
    assert r.status_code == 200
    assert 'POST' in r.headers['access-control-allow-methods']
    for method in ('PUT', 'DELETE', 'PATCH'):
        assert client.options('/search-requests', headers={'Origin': settings.cors_origins[0],
                                                           'Access-Control-Request-Method': method}).status_code == 400


def test_with_it_on_but_not_configured_the_app_does_not_start(monkeypatch):
    _enable(monkeypatch, user='')
    with pytest.raises(RuntimeError, match='API_REQUEST_PGUSER'):
        api_main.create_app()
    _enable(monkeypatch, salt='')
    with pytest.raises(RuntimeError, match='API_REQUEST_SALT'):
        api_main.create_app()
    _enable(monkeypatch, hops=-1)
    with pytest.raises(RuntimeError, match='PROXY_HOPS'):
        api_main.create_app()


def test_the_limits_are_the_design_numbers():
    assert (search_requests.MAX_ACTIVE_PER_REQUESTER, search_requests.MAX_PER_REQUESTER_PER_DAY,
            search_requests.MAX_PER_DAY, search_requests.MAX_QUEUE, search_requests.MAX_SEED) == (3, 10, 50, 10, 2**31 - 1)


# ── the requester ───────────────────────────────────────────────────────────────────────────


def test_the_address_is_the_peer_when_there_is_no_proxy(monkeypatch):
    monkeypatch.setattr(settings, 'request_proxy_hops', 0)
    assert search_requests.client_address(_raw_request([('X-Forwarded-For', '1.1.1.1')], ('9.9.9.9', 1))) == '9.9.9.9'


def test_behind_a_proxy_only_the_entry_it_added_is_believed(monkeypatch):
    monkeypatch.setattr(settings, 'request_proxy_hops', 1)
    # The client wrote the first two entries; the proxy appended the third, which is the address it saw.
    r = _raw_request([('X-Forwarded-For', '6.6.6.6, 7.7.7.7, 203.0.113.9')])
    assert search_requests.client_address(r) == '203.0.113.9'
    monkeypatch.setattr(settings, 'request_proxy_hops', 2)
    assert search_requests.client_address(r) == '7.7.7.7'


def test_forwarded_for_in_several_header_lines_is_read_as_one_list(monkeypatch):
    monkeypatch.setattr(settings, 'request_proxy_hops', 1)
    r = _raw_request([('X-Forwarded-For', '6.6.6.6'), ('X-Forwarded-For', '203.0.113.9')])
    assert search_requests.client_address(r) == '203.0.113.9'


def test_a_chain_shorter_than_the_proxies_configured_falls_back_to_the_peer(monkeypatch):
    monkeypatch.setattr(settings, 'request_proxy_hops', 2)
    assert search_requests.client_address(_raw_request([('X-Forwarded-For', '203.0.113.9')], ('10.1.1.1', 1))) == '10.1.1.1'
    assert search_requests.client_address(_raw_request([], ('10.1.1.1', 1))) == '10.1.1.1'


def test_the_hash_depends_on_the_salt_and_the_address_and_is_not_the_address(monkeypatch):
    monkeypatch.setattr(settings, 'request_salt', 'salt-one')
    a = search_requests.requester_hash('203.0.113.9')
    assert re.fullmatch(r'[0-9a-f]{64}', a)
    assert a == search_requests.requester_hash('203.0.113.9')
    assert a != search_requests.requester_hash('203.0.113.10')
    monkeypatch.setattr(settings, 'request_salt', 'salt-two')
    assert a != search_requests.requester_hash('203.0.113.9')
    import hashlib
    import hmac
    assert search_requests.requester_hash('203.0.113.9') == hmac.new(b'salt-two', b'203.0.113.9', hashlib.sha256).hexdigest()


# ── the read path does not know the table ───────────────────────────────────────────────────


def test_nothing_in_the_read_path_mentions_the_request_table_or_function():
    api = os.path.join(PROJECT, 'src', 'api')
    for filename in sorted(os.listdir(api)):
        # config.py names the module in a comment and holds the settings; the other three are the request path itself.
        if not filename.endswith('.py') or filename in ('search_requests.py', 'request_pool.py', 'config.py'):
            continue
        text = open(os.path.join(api, filename)).read()
        if filename == 'main.py':  # wires the router and the pool; names neither the table nor the function
            assert 'file_search_request' not in text
            continue
        assert 'search_requests' not in text, filename
        assert 'file_search_request' not in text, filename


def test_the_read_connection_in_the_request_path_only_reads_the_scenario_table():
    text = open(os.path.join(PROJECT, 'src', 'api', 'search_requests.py')).read()
    executed = re.findall(r"cur\.execute\(\s*'([^']*)'", text)
    reads = [sql for sql in executed if sql.startswith('SELECT 1')]
    assert reads == ['SELECT 1 FROM scenario_scores WHERE scenario_id = %s']
    assert 'FROM search_requests' not in text and 'INSERT' not in text, 'the write goes through the function only'


# ── on a database ───────────────────────────────────────────────────────────────────────────


@pytest.fixture
def served(monkeypatch, scratch):  # noqa: F811
    """The app, ON, serving the throw-away database through the real role; (client, owner connection, name)."""
    _enable(monkeypatch)
    monkeypatch.setattr(settings, 'pg_database', scratch)
    base = settings.dsn_kwargs()
    owner = psycopg2.connect(**{**base})
    owner.autocommit = True
    with owner.cursor() as cur:
        cur.execute('DELETE FROM search_requests')
    with TestClient(api_main.create_app(), raise_server_exceptions=False) as client:
        yield client, owner
    owner.close()
    assert db_pool._pool is None and request_pool._pool is None


def _post(client, body, headers=None):
    return client.post('/search-requests', json=body, headers=headers or {})


def _rows(owner):
    with owner.cursor() as cur:
        cur.execute('SELECT request_id::text, scenario_id, seed, requester_hash, status FROM search_requests '
                    'ORDER BY requested_at, request_id')
        return cur.fetchall()


def _settle(owner):
    with owner.cursor() as cur:
        cur.execute("UPDATE search_requests SET status = 'done'")


@requires_db
def test_a_request_is_filed_as_201_and_stored_with_a_hash_not_the_address(served):
    client, owner = served
    r = _post(client, {'scenario_id': 's1', 'seed': 5})
    assert r.status_code == 201, r.text
    body = r.json()
    assert set(body) == {'request_id', 'scenario_id', 'seed', 'state'}
    assert (body['scenario_id'], body['seed'], body['state']) == ('s1', 5, 'filed')
    uuid.UUID(body['request_id'])
    [row] = _rows(owner)
    assert row[:3] == (body['request_id'], 's1', 5) and row[4] == 'queued'
    assert row[3] == search_requests.requester_hash('testclient')
    stored = json.dumps(row)
    assert 'testclient' not in stored and SALT not in stored


@requires_db
def test_the_same_scenario_and_seed_returns_the_existing_request(served):
    client, owner = served
    first = _post(client, {'scenario_id': 's1', 'seed': 5}).json()
    again = _post(client, {'scenario_id': 's1', 'seed': 5})
    assert again.status_code == 200
    assert again.json() == {**first, 'state': 'existing'}
    other_address = _post(client, {'scenario_id': 's1', 'seed': 5}, {'X-Forwarded-For': '203.0.113.9'})
    assert other_address.json()['request_id'] == first['request_id']
    assert len(_rows(owner)) == 1
    assert _post(client, {'scenario_id': 's1', 'seed': 6}).json()['request_id'] != first['request_id']


@requires_db
def test_an_unknown_scenario_is_a_404_and_stores_nothing(served):
    client, owner = served
    r = _post(client, {'scenario_id': 'nope', 'seed': 1})
    assert r.status_code == 404
    assert r.json() == {'detail': 'Unknown scenario: nope'}
    assert _rows(owner) == []


@requires_db
@pytest.mark.parametrize('body', [
    {'scenario_id': 's1'},
    {'seed': 1},
    {},
    {'scenario_id': 's1', 'seed': -1},
    {'scenario_id': 's1', 'seed': 2**31},
    {'scenario_id': 's1', 'seed': 1.5},
    {'scenario_id': 's1', 'seed': 1.0},
    {'scenario_id': 's1', 'seed': '5'},
    {'scenario_id': 's1', 'seed': True},
    {'scenario_id': 's1', 'seed': None},
    {'scenario_id': '', 'seed': 1},
    {'scenario_id': 's' * 4097, 'seed': 1},
    {'scenario_id': 5, 'seed': 1},
    {'scenario_id': None, 'seed': 1},
    {'scenario_id': 's1', 'seed': 1, 'popsize': 100},
    {'scenario_id': 's1', 'seed': 1, 'maxiter': 1},
    {'scenario_id': 's1', 'seed': 1, 'tol': 0.1},
    {'scenario_id': 's1', 'seed': 1, 'status': 'done'},
    {'scenario_id': 's1\x00', 'seed': 1},
    {'scenario_id': 's1\ud800', 'seed': 1},
    {'scenario_id': 's\n1', 'seed': 1},
], ids=lambda b: json.dumps(b, ensure_ascii=True)[:60])
def test_a_body_that_is_not_one_scenario_id_and_one_bounded_integer_is_refused(served, body):
    client, owner = served
    r = client.post('/search-requests', content=json.dumps(body, ensure_ascii=True), headers={'content-type': 'application/json'})
    assert r.status_code == 422, (r.status_code, r.text)
    assert set(r.json()) == {'detail'}
    assert _rows(owner) == []


@requires_db
def test_a_refusal_never_echoes_what_the_client_sent_and_survives_text_it_cannot_encode(served):
    client, owner = served
    # A key with an unpaired surrogate (pydantic cannot read the body at all) and a value that is secret-looking: neither comes
    # back, and the answer is a 422 where FastAPI's own would have failed to encode.
    body = json.dumps({'scenario_id': 's1', 'seed': 'hunter2-not-a-number', 'x\ud800key': 1}, ensure_ascii=True)
    r = client.post('/search-requests', content=body, headers={'content-type': 'application/json'})
    assert r.status_code == 422
    assert 'hunter2' not in r.text
    for entry in r.json()['detail']:
        assert set(entry) == {'loc', 'msg', 'type'}
    assert 'ud800' not in r.text and 'key' not in r.text


@requires_db
def test_the_seed_bounds_are_accepted(served):
    client, owner = served
    assert _post(client, {'scenario_id': 's1', 'seed': 0}).status_code == 201
    assert _post(client, {'scenario_id': 's1', 'seed': 2**31 - 1}).status_code == 201
    assert len(_rows(owner)) == 2


@requires_db
def test_a_body_that_is_not_json_or_not_an_object_is_refused(served):
    client, owner = served
    for content, ctype in (('not json', 'application/json'), ('[1,2]', 'application/json'), ('"s1"', 'application/json'),
                           ('scenario_id=s1&seed=1', 'application/x-www-form-urlencoded'), ('', 'application/json')):
        r = client.post('/search-requests', content=content, headers={'content-type': ctype})
        assert r.status_code == 422, (content, r.status_code)
    assert _rows(owner) == []


@requires_db
def test_only_post_is_answered(served):
    client, owner = served
    assert client.get('/search-requests').status_code == 405
    assert client.put('/search-requests', json={'scenario_id': 's1', 'seed': 1}).status_code == 405
    assert client.delete('/search-requests').status_code == 405
    assert _rows(owner) == []


@requires_db
def test_a_requester_is_held_to_three_waiting_and_the_refusal_is_a_429_with_retry_after(served, monkeypatch):
    client, owner = served
    for seed in range(3):
        assert _post(client, {'scenario_id': 's1', 'seed': seed}).status_code == 201
    r = _post(client, {'scenario_id': 's1', 'seed': 9})
    assert r.status_code == 429
    assert r.headers['retry-after'] == '3600'
    assert 'waiting' in r.json()['detail']
    assert len(_rows(owner)) == 3
    # The same request is not a new one, so it is answered rather than refused.
    assert _post(client, {'scenario_id': 's1', 'seed': 1}).status_code == 200
    # Another visitor, behind the proxy, is not held back by it.
    monkeypatch.setattr(settings, 'request_proxy_hops', 1)
    assert _post(client, {'scenario_id': 's2', 'seed': 0}, {'X-Forwarded-For': '203.0.113.9'}).status_code == 201


@requires_db
def test_a_requester_may_file_ten_in_a_day_and_the_eleventh_is_a_429(served):
    client, owner = served
    for batch in range(4):
        for i in range(3):
            seed = batch * 3 + i
            if seed == 10:
                break
            assert _post(client, {'scenario_id': 's1', 'seed': seed}).status_code == 201, seed
        _settle(owner)  # run offline: they no longer count as waiting
    r = _post(client, {'scenario_id': 's1', 'seed': 10})
    assert r.status_code == 429, r.text
    assert 1 <= int(r.headers['retry-after']) <= 86400
    assert 'in a day' in r.json()['detail']
    assert len(_rows(owner)) == 10


@requires_db
def test_the_queue_holds_ten_and_the_eleventh_visitor_gets_a_503_with_retry_after(served, monkeypatch):
    client, owner = served
    monkeypatch.setattr(settings, 'request_proxy_hops', 1)
    for i in range(10):
        r = _post(client, {'scenario_id': 's1', 'seed': i}, {'X-Forwarded-For': f'203.0.113.{i}'})
        assert r.status_code == 201, (i, r.text)
    r = _post(client, {'scenario_id': 's2', 'seed': 0}, {'X-Forwarded-For': '203.0.113.99'})
    assert r.status_code == 503
    assert r.headers['retry-after'] == '3600'
    assert 'queue' in r.json()['detail']
    assert len(_rows(owner)) == 10
    # One that has run leaves the queue.
    with owner.cursor() as cur:
        cur.execute("UPDATE search_requests SET status = 'done' WHERE seed = 0")
    assert _post(client, {'scenario_id': 's2', 'seed': 0}, {'X-Forwarded-For': '203.0.113.99'}).status_code == 201


@requires_db
def test_everyone_together_may_file_fifty_in_a_day_and_the_fifty_first_is_a_503(served, monkeypatch):
    client, owner = served
    monkeypatch.setattr(settings, 'request_proxy_hops', 1)
    for i in range(50):
        r = _post(client, {'scenario_id': 's1' if i % 2 else 's2', 'seed': i}, {'X-Forwarded-For': f'198.51.100.{i}'})
        assert r.status_code == 201, (i, r.text)
        if i % 5 == 4:
            _settle(owner)  # keep the queue short so that only the day's count is the limit
    r = _post(client, {'scenario_id': 's1', 'seed': 999}, {'X-Forwarded-For': '198.51.100.250'})
    assert r.status_code == 503, r.text
    assert 1 <= int(r.headers['retry-after']) <= 86400
    assert 'in a day' in r.json()['detail']
    assert len(_rows(owner)) == 50


@requires_db
def test_a_refusal_leaves_the_connection_usable(served):
    client, owner = served
    for seed in range(3):
        _post(client, {'scenario_id': 's1', 'seed': seed})
    for _ in range(6):  # more refusals than the pool has connections
        assert _post(client, {'scenario_id': 's1', 'seed': 9}).status_code == 429
    assert _post(client, {'scenario_id': 's1', 'seed': 1}).status_code == 200


@requires_db
def test_the_request_pool_connects_as_the_request_role_and_the_read_pool_does_not(served):
    client, owner = served
    conn = request_pool._pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT current_user')
            assert cur.fetchone() == (ROLE,)
    finally:
        request_pool._pool.putconn(conn)
    conn = db_pool._pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT current_user')
            assert cur.fetchone() != (ROLE,)
    finally:
        db_pool._pool.putconn(conn)
    assert request_pool._pool.maxconn == 2


@requires_db
def test_a_full_request_pool_is_a_503_with_retry_after(served, monkeypatch):
    client, owner = served
    held = [request_pool._pool.getconn() for _ in range(request_pool._pool.maxconn)]
    try:
        monkeypatch.setattr(settings, 'pool_wait_ms', 20)
        r = _post(client, {'scenario_id': 's1', 'seed': 1})
        assert r.status_code == 503
        assert r.headers['retry-after'] == '1'
    finally:
        for c in held:
            request_pool._pool.putconn(c)
    assert _rows(owner) == []
    assert _post(client, {'scenario_id': 's1', 'seed': 1}).status_code == 201


@requires_db
def test_the_stored_results_are_served_as_before_with_the_switch_on(served):
    client, owner = served
    r = client.get('/scenarios/s1')
    assert r.status_code == 200 and r.json()['scenario_id'] == 's1'
    assert client.get('/scenarios/nope').status_code == 404
    assert client.get('/health').status_code == 200
    with owner.cursor() as cur:
        cur.execute('SELECT count(*) FROM scenario_scores')
        assert cur.fetchone()[0] == 2


# ── DEPLOY.md ───────────────────────────────────────────────────────────────────────────────


def test_deploy_md_names_every_variable_the_request_path_reads_and_says_it_is_off():
    deploy = open(os.path.join(PROJECT, 'DEPLOY.md')).read()
    config_text = open(os.path.join(PROJECT, 'src', 'api', 'config.py')).read()
    names = set(re.findall(r"environ\.get\('(API_(?:SEARCH_REQUESTS|REQUEST_[A-Z_]+))'", config_text))
    assert names == {'API_SEARCH_REQUESTS', 'API_REQUEST_PGUSER', 'API_REQUEST_PGPASSWORD', 'API_REQUEST_POOL_MAX',
                     'API_REQUEST_SALT', 'API_REQUEST_PROXY_HOPS'}
    for name in names:
        assert name in deploy, name
    section = deploy[deploy.index('## Queued search requests'):deploy.index('## Replacing the data in a live database')]
    assert 'off' in section.split('\n', 1)[0].lower()
    assert 'deploy/search_requests.sql' in section and 'deploy/pending_search_requests.sql' in section
    assert 'no privilege on any table' in section
    assert 'no longer strictly true' in section, 'it says which shipped sentences stop being true when it is on'
