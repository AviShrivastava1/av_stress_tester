"""
search_requests.py — queued search requests: the one write this API can make, OFF unless the environment says so.

What it does. POST /search-requests files a request: a scenario id and a seed. The request is run LATER, OFFLINE, by the
owner (deploy/pending_search_requests.sql lists what is waiting). Nothing runs on request: this module computes nothing,
reads no tfrecord, and never touches the stored results.

When it exists. main.create_app registers this router only when API_SEARCH_REQUESTS is exactly "1" (config.py). With it
unset the route is not in the app and not in openapi.json, and the request pool is never opened.

What it is made of, each kept apart from the read path in routes.py (which never mentions the request table):
  * its own pool and its own role (request_pool.py, deploy/search_requests.sql). The role holds no privilege on any table;
    it can call file_search_request() and nothing else;
  * that function, which counts and inserts in one transaction under one lock, so the limits below cannot be raced;
  * one read, through the ordinary read-only pool: does the scenario exist in scenario_scores?

The limits are DESIGN-21.md section 4, as constants here, passed to the function on every call. A requester is the HMAC of
its address under a secret salt (config.py request_salt); the address is never stored or logged.
"""

import hashlib
import hmac
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

from src.api.config import settings
from src.api.db_pool import dict_cursor, get_db
from src.api.request_pool import get_request_db
from src.api.routes import _MAX_SCENARIO_ID_LEN, _reject_unstorable_text

# ── the limits (DESIGN-21.md, section 4) ────────────────────────────────────────────────────────

#: Queued or running requests one requester may have at once.
MAX_ACTIVE_PER_REQUESTER = 3
#: Requests one requester may file in any 24 hours.
MAX_PER_REQUESTER_PER_DAY = 10
#: Requests all requesters together may file in any 24 hours.
MAX_PER_DAY = 50
#: Requests waiting at once; past it the answer is 503 with Retry-After.
MAX_QUEUE = 10
#: The largest seed: the database's integer column, and numpy's.
MAX_SEED = 2**31 - 1


class SearchRequestBody(BaseModel):
    """The whole request: a scenario id and a seed. Any other field is refused."""

    model_config = ConfigDict(extra='forbid')

    scenario_id: str = Field(min_length=1, max_length=_MAX_SCENARIO_ID_LEN,
                             description='A scenario that is in the stored results.')
    seed: int = Field(strict=True, ge=0, le=MAX_SEED,
                      description=f'The search seed: a whole number from 0 to {MAX_SEED}.')


class SearchRequestResponse(BaseModel):
    """
    A request that is filed, and waiting to be run offline. Nothing has run.

    `state` is "filed" when this call filed it and "existing" when the same scenario and seed had been
    filed before: the first request is returned and no second one is made.
    """

    request_id: str
    scenario_id: str
    seed: int
    state: Literal['filed', 'existing']


def _printable(value):
    """Text that can always be encoded: anything a client sent that cannot is shown as an escape."""
    return value.encode('utf-8', 'backslashreplace').decode('utf-8') if isinstance(value, str) else value


class _RefusingRoute(APIRoute):
    """
    A route whose 422 never echoes what the client sent.

    FastAPI's own 422 body repeats the offending input, and a JSON body can hold an unpaired surrogate (\\ud800), which
    Python decodes but cannot encode into the response: the refusal itself then fails as a 500. This route answers with the
    location, the message and the type of each error only, with every string made encodable. Other routes are not touched.
    """

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def refusing(request: Request):
            try:
                return await handler(request)
            except RequestValidationError as exc:
                errors = [{'loc': [_printable(part) for part in e['loc']], 'msg': _printable(e['msg']), 'type': e['type']}
                          for e in exc.errors()]
                return JSONResponse(status_code=422, content={'detail': jsonable_encoder(errors)})

        return refusing


request_router = APIRouter(route_class=_RefusingRoute)


def validate_settings() -> None:
    """Refuse to start with the switch on and the request path half configured."""
    missing = [name for name, value in (('API_REQUEST_PGUSER', settings.request_pg_user),
                                        ('API_REQUEST_SALT', settings.request_salt)) if not value]
    if missing:
        raise RuntimeError(f'API_SEARCH_REQUESTS=1 needs {" and ".join(missing)} to be set.')
    if settings.request_proxy_hops < 0:
        raise RuntimeError('API_REQUEST_PROXY_HOPS must be 0 or more.')


def client_address(request: Request) -> str:
    """
    The requester's address: the peer of the connection, or, behind N reverse proxies, the entry the Nth proxy added.

    X-Forwarded-For is appended to by every proxy and can be written by the client, so only entries counted from the RIGHT,
    which the proxies add, are used, and never the leftmost. If the header has fewer entries than the proxies configured,
    the chain is not what was configured and the peer is used, which is the proxy: every requester then shares one bucket,
    which is the strict way to be wrong.
    """
    peer = request.client.host if request.client else 'unknown'
    hops = settings.request_proxy_hops
    if hops <= 0:
        return peer
    entries = [e.strip() for line in request.headers.getlist('x-forwarded-for') for e in line.split(',') if e.strip()]
    return entries[-hops] if len(entries) >= hops else peer


def requester_hash(address: str) -> str:
    """HMAC-SHA256 of the address under the secret salt, hex. The only form of an address that is ever stored."""
    return hmac.new(settings.request_salt.encode(), address.encode(), hashlib.sha256).hexdigest()


# What each refusal of the function becomes: (status, detail). Retry-After comes from the function.
_REFUSALS = {
    'requester_active': (429, 'You already have the most search requests waiting that one visitor may have. '
                              'Try again after one has been run.'),
    'requester_daily': (429, 'You have filed the most search requests that one visitor may file in a day.'),
    'queue_full': (503, 'The queue of search requests is full. Try again later.'),
    'global_daily': (503, 'The most search requests that can be filed in a day have been filed. Try again later.'),
}


@request_router.post('/search-requests', response_model=SearchRequestResponse, status_code=201,
                     tags=['search requests'],
                     responses={200: {'description': 'The same scenario and seed had been filed before; the first request.'},
                                404: {'description': 'The scenario is not in the stored results.'},
                                429: {'description': 'This visitor has reached a limit. Retry-After says when to try again.'},
                                503: {'description': 'The queue or the day is full. Retry-After says when to try again.'}})
def file_search_request(body: SearchRequestBody, request: Request, response: Response,
                        read_conn=Depends(get_db), conn=Depends(get_request_db)):
    """
    File a request to search one scenario with one seed, to be run later, offline.

    Nothing runs now. A scenario that is not in the stored results is a 404. The same scenario and seed is one request:
    filing it again returns the first. One visitor may have 3 requests waiting and file 10 in a day; everyone together may
    file 50 in a day and have 10 waiting. Past a limit the answer is 429 (this visitor) or 503 (everyone) with Retry-After.
    """
    _reject_unstorable_text(body.scenario_id, 'scenario_id')
    with dict_cursor(read_conn) as cur:
        cur.execute('SELECT 1 FROM scenario_scores WHERE scenario_id = %s', (body.scenario_id,))
        known = cur.fetchone() is not None
    if not known:
        raise HTTPException(status_code=404, detail=f'Unknown scenario: {body.scenario_id}')

    with conn.cursor() as cur:
        cur.execute(
            'SELECT outcome, filed_request_id, retry_after_seconds FROM file_search_request('
            '%s::uuid, %s, %s, %s, %s, %s, %s, %s)',
            (str(uuid.uuid4()), body.scenario_id, body.seed, requester_hash(client_address(request)),
             MAX_ACTIVE_PER_REQUESTER, MAX_PER_REQUESTER_PER_DAY, MAX_PER_DAY, MAX_QUEUE))
        outcome, request_id, retry_after = cur.fetchone()
    if outcome in _REFUSALS:
        conn.rollback()
        status, detail = _REFUSALS[outcome]
        raise HTTPException(status_code=status, detail=detail, headers={'Retry-After': str(retry_after)})
    conn.commit()
    if outcome == 'existing':
        response.status_code = 200
    return SearchRequestResponse(request_id=str(request_id), scenario_id=body.scenario_id, seed=body.seed,
                                 state='filed' if outcome == 'filed' else 'existing')
