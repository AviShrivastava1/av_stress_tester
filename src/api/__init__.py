"""
Phase 6 — the read-only HTTP API.

Serves the Phase 5 results (and the Phase 6 PostGIS geometry) over HTTP so a
frontend has a typed contract to build against.

Why this API is READ-ONLY, deliberately
---------------------------------------
There is no POST, no PUT, no DELETE, and in particular there is NO endpoint that
runs the optimizer. That is a design decision, not an omission.

Phase 4 takes minutes per scenario. Work of that shape has no business behind an
HTTP request: the client would sit on an open connection waiting for a timeout, and
the server would burn a worker thread on CPU-bound numerical search while every
other request queued behind it. Doing it properly means a job queue, a job table,
status polling, and a retry story — real machinery, and the wrong complexity to
take on here.

So the split is: heavy compute runs offline in Colab, writes its results to
PostgreSQL, and this layer only reads what that compute already produced. The API
is a window onto a finished batch, not a way to start one.

Two consequences worth stating:

  * This package MUST NOT import anything from src/data/. That package needs the
    Waymo protobufs, which realistically only install in Colab, and an API that
    can't start on an arbitrary machine is not deployable. Trajectory geometry
    reaches the API through PostGIS (see src/scoring/export_geometry.py), never
    through the shard.

  * No credentials live in code. Configuration comes from the standard PG*
    environment variables, the same ones Phase 5 uses, so pointing this at a hosted
    database later is an environment change with zero code change.

Modules:
    config    — environment-driven settings, read once
    db_pool   — a threaded psycopg2 connection pool + FastAPI dependency
    models    — Pydantic response models; the typed contract with the frontend
    routes    — the six GET endpoints
    main      — app factory, lifespan, CORS
"""
