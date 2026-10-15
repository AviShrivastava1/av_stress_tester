"""
request_pool.py — the connection pool of the queued-search-request path.

Its own pool, small, and its own role: the one in deploy/search_requests.sql, which holds no privilege on any table and can
call one function. It is opened only when the request path is switched on (main.py), so with the switch off no connection
is ever made as that role, and the read pool in db_pool.py is not touched or shared.
"""

import psycopg2.pool

from src.api.config import settings
from src.api.db_pool import _getconn_or_503

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def init_request_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Create the pool. Idempotent."""
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(1, settings.request_pool_max, **settings.request_dsn_kwargs())
    return _pool


def close_request_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


def get_request_db():
    """FastAPI dependency: a connection of the request pool, returned afterwards; a full pool is a 503 with Retry-After."""
    pool = _pool if _pool is not None else init_request_pool()
    conn = _getconn_or_503(pool)  # outside the try, as in db_pool.get_db
    try:
        yield conn
    finally:
        pool.putconn(conn)
