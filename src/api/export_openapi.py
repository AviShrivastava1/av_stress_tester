"""
export_openapi.py — Phase 7.

Write the API's OpenAPI schema to frontend/openapi.json:

    ./venv/bin/python -m src.api.export_openapi

The frontend generates its TypeScript types from that committed snapshot rather than
from a live server. A build that fetched /openapi.json from the deployed API would
depend on the backend being awake (free-tier hosts cold-start) and would produce
different types depending on which backend happened to answer — a non-reproducible
build. The snapshot makes the contract a file in the diff instead.

No database is needed: app.openapi() is computed from the route and response-model
declarations alone. The lifespan handler that opens the pool never runs here.

tests/test_openapi_snapshot.py fails when the snapshot and the app disagree, so a
backend model change cannot land without the regenerated contract beside it.
"""

import json
import os

from src.api.main import app


SNAPSHOT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'frontend', 'openapi.json',
)


def render_schema() -> str:
    """The schema exactly as it is written to disk — one definition of the format."""
    return json.dumps(app.openapi(), indent=2, ensure_ascii=False) + '\n'


def main() -> None:
    with open(SNAPSHOT_PATH, 'w', encoding='utf-8') as f:
        f.write(render_schema())
    print(f'wrote {SNAPSHOT_PATH}')


if __name__ == '__main__':
    main()
