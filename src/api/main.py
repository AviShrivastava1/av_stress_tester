"""
main.py — Phase 6.

The FastAPI application: `uvicorn src.api.main:app --reload`.

An app FACTORY plus a module-level instance. The factory is what lets a test build
a fresh app without importing global state, and the module-level `app` is what
uvicorn's `module:attribute` syntax needs.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import settings
from src.api.db_pool import close_pool, init_pool
from src.api.request_pool import close_request_pool, init_request_pool
from src.api.routes import router
from src.api import search_requests


API_DESCRIPTION = """
Read-only access to the AV scenario stress-tester's results.

Scenarios are ranked by **fragility score** (higher = closer to catastrophe), and
each one can be inspected: its danger metrics, the minimum perturbation that causes
a collision, and the agent trajectories to draw it.

**Coordinates are local planar metres, not longitude/latitude.** Waymo Open Motion
Dataset scenarios use a local metric frame; treating these values as lon/lat will
place every scenario off the coast of West Africa and make every distance
meaningless.

Nothing here starts a computation — the optimizer runs offline and this API reads
what it wrote.
"""


# Appended to the description only when the request path is switched on, so with it off the published description is unchanged.
SEARCH_REQUESTS_NOTE = """

**One write exists in this deployment, and it is switched on:** `POST /search-requests` files a request to search one scenario
with one seed. The request is run later, offline, by the owner. Nothing runs on request, and the stored results above are
never written by it.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Open the connection pool on startup, close it on shutdown.

    Why the pool belongs HERE and not elsewhere:

      * Not per request — that pays a TCP handshake plus auth on every call, tens
        of milliseconds added to queries that take single-digit milliseconds.

      * Not at import time — importing a module must not open network sockets. It
        would mean any process that merely imports this file (a test collector, a
        linter, `python -c "from src.api.main import app"`) tries to reach the
        database, so an import fails on a machine where Postgres simply is not
        running. Startup is the correct moment: it happens once, and only when the
        app is actually being served.

    NOTE for tests: TestClient must be used as a context manager
    (`with TestClient(app) as client:`) for this handler to run at all. Without the
    `with`, lifespan never fires and the pool is never initialized.
    """
    init_pool()
    if settings.search_requests_enabled:
        init_request_pool()
    try:
        yield
    finally:
        close_pool()
        close_request_pool()


def create_app() -> FastAPI:
    """Build the application."""
    # The one write path (src/api/search_requests.py) exists only when API_SEARCH_REQUESTS=1. Read here, at build time, so
    # that with it off the route is not in the app and not in openapi.json.
    requests_on = settings.search_requests_enabled
    if requests_on:
        search_requests.validate_settings()
    app = FastAPI(
        title="AV Scenario Stress-Tester API",
        description=API_DESCRIPTION + (SEARCH_REQUESTS_NOTE if requests_on else ""),
        version="0.6.0",
        lifespan=lifespan,
    )

    # Explicit origins, never "*". A GET-only API still hands data to whatever
    # origin it allows — read-only limits what an attacker can change, not what
    # they can read. allow_credentials stays False because this API has no
    # session: there are no cookies worth forwarding, and pairing credentials with
    # a loose origin list is how CORS mistakes become data leaks.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"] if requests_on else ["GET"],
        allow_credentials=False,
        allow_headers=["*"],
    )

    app.include_router(router)
    if requests_on:
        app.include_router(search_requests.request_router)
    return app


app = create_app()
