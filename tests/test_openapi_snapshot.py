"""
test_openapi_snapshot.py — Phase 7.

frontend/openapi.json is the contract the frontend's TypeScript types are generated
from. It is a committed snapshot, not fetched from a live server (see
src/api/export_openapi.py for why), which means it can go stale: a Pydantic model
changes, the API serves the new shape, and the frontend keeps compiling against the
old one. This test is what turns that silent drift into a red suite.

Compared byte-for-byte against the exporter's own rendering, not as parsed JSON:
the snapshot is only ever written by that function, so any byte difference means
either the schema changed or someone hand-edited the file — and a hand-edited
contract is its own defect.

No database needed.

Run:
    ./venv/bin/python -m pytest tests/test_openapi_snapshot.py -q
Fix a failure with:
    ./venv/bin/python -m src.api.export_openapi
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.export_openapi import SNAPSHOT_PATH, render_schema


def test_snapshot_matches_app_schema():
    with open(SNAPSHOT_PATH, encoding='utf-8') as f:
        committed = f.read()
    assert committed == render_schema(), (
        'frontend/openapi.json is stale. Regenerate it with '
        '`./venv/bin/python -m src.api.export_openapi` and commit it with the '
        'backend change that caused the drift.'
    )
