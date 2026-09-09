"""
db_pool.py — Phase 6.

A module-level psycopg2 connection pool, plus the FastAPI dependency that hands
handlers a connection and always gives it back.

Why a pool at all
-----------------
Opening a PostgreSQL connection is not free: a TCP handshake plus authentication
plus backend process startup costs on the order of tens of milliseconds. The
queries this API runs are single-digit milliseconds. Connecting per request would
mean spending ~90% of each request's wall clock on setup — the connection cost
would dominate the thing it exists to serve. A pool pays that cost once and hands
out live connections.

Why synchronous psycopg2 rather than asyncpg
--------------------------------------------
FastAPI runs plain `def` handlers (as opposed to `async def`) in a worker
threadpool. A blocking database call inside one of those therefore blocks a worker
thread, NOT the event loop, so the server keeps accepting and serving other
requests. That is the whole reason this is safe.

The honest trade-off: async wins once you have many concurrent IO-bound requests
and the threadpool itself becomes the bottleneck — each thread costs real memory
and context-switching, while async coroutines are far cheaper to multiplex. At this
scale (a dashboard with a handful of users) that crossover is nowhere in sight, and
async would buy nothing measurable while adding a second database dialect to the
project. Phase 5 already speaks psycopg2; keeping one dialect means one set of
idioms, one adapter behaviour, one thing to get right.

`ThreadedConnectionPool` and not `SimpleConnectionPool` precisely BECAUSE those
handlers run on multiple threads — SimpleConnectionPool is not thread-safe, and
using it here would be a race waiting to happen.

Why putconn must live in a `finally`
------------------------------------
If a handler raises and the connection is never returned, that connection is gone
for the life of the process. With a ceiling of 8, it takes only eight unhandled
exceptions to leave the API permanently unable to serve any request — and the
failure appears long after the bug that caused it, which makes it miserable to
diagnose. The `finally` is what turns "an error" into "just an error".
"""

from contextlib import contextmanager

import psycopg2.pool
from psycopg2.extras import RealDictCursor

from src.api.config import settings


# Module-level, created at app startup by init_pool() — see the lifespan handler in
# main.py for why it is not created at import time.
_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def init_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """
    Create the pool. Idempotent: calling it twice returns the existing pool rather
    than leaking the first one.
    """
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            settings.pool_min,
            settings.pool_max,
            **settings.dsn_kwargs(),
        )
    return _pool


def close_pool() -> None:
    """Close every connection in the pool. Called on shutdown."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """The live pool, initializing it on first use if startup has not run yet."""
    return _pool if _pool is not None else init_pool()


@contextmanager
def connection():
    """
    Borrow a connection for the duration of a block, and always return it.

    Used by scripts and tests; request handlers get the same guarantee through
    get_db() below.
    """
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


def get_db():
    """
    FastAPI dependency. Yields a pooled connection and returns it afterwards.

    FastAPI runs the code after `yield` even when the handler raised, which is what
    makes this equivalent to the try/finally above.
    """
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


def dict_cursor(conn):
    """
    A cursor returning dict-like rows instead of tuples.

    Worth the tiny overhead: `row['fragility_score']` survives someone adding a
    column to the SELECT, whereas `row[5]` silently starts returning the wrong
    field. In a safety-auditing tool, silently-wrong is the worst failure mode.
    """
    return conn.cursor(cursor_factory=RealDictCursor)
