"""
test_deploy_contract.py — what the Render API service and the Vercel frontend are built
from, pinned to what the code actually needs.

The deployed API installs requirements-api.txt, not requirements.txt: the latter pulls in
TensorFlow, the Waymo package and torch for the shard-reading and optimisation code, none
of which src/api imports. What this file pins:

  * every third-party import in src/api is provided by requirements-api.txt, so a new
    import cannot reach the server only to fail at startup;
  * requirements-api.txt pins exact versions, the ones this suite runs against, and
    carries none of the heavy shard/optimiser packages;
  * .python-version names the Python the suite runs on (Render reads it; without it,
    Render's default is 3.14.3, which this code has never run on);
  * DEPLOY.md grants the read-only role SELECT on exactly the tables the API reads,
    plus default privileges for tables created later, and its post-deploy checks carry
    the run's real numbers;
  * frontend/vercel.json rewrites every path to index.html, so a direct link to
    /scenarios/<id> or /stats loads.

No database, no network.

Run:
    ./venv/bin/python -m pytest tests/test_deploy_contract.py -q
"""

import ast
import importlib.metadata
import json
import os
import re
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(PROJECT, 'src', 'api')
REQUIREMENTS_API = os.path.join(PROJECT, 'requirements-api.txt')
DEPLOY = os.path.join(PROJECT, 'DEPLOY.md')

# Import name -> the distribution that provides it, where the two differ.
DISTRIBUTION = {'psycopg2': 'psycopg2-binary'}


def _requirements():
    """{distribution name (lowercase, no extras): pinned version}."""
    pins = {}
    for line in open(REQUIREMENTS_API):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)(\[[^\]]+\])?==([0-9][0-9A-Za-z.]*)', line)
        assert match, f'requirements-api.txt: not an exact pin: {line!r}'
        pins[match.group(1).lower()] = match.group(3)
    return pins


def _api_imports():
    """Top-level module names imported anywhere in src/api."""
    names = set()
    for filename in os.listdir(API_DIR):
        if not filename.endswith('.py'):
            continue
        tree = ast.parse(open(os.path.join(API_DIR, filename)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {alias.name.split('.')[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split('.')[0])
    return names


def test_every_third_party_import_in_the_api_is_in_requirements_api():
    third_party = {n for n in _api_imports() if n not in sys.stdlib_module_names and n != 'src'}
    assert third_party, 'premise: src/api imports third-party packages'
    pins = _requirements()
    missing = sorted(n for n in third_party if DISTRIBUTION.get(n, n).lower() not in pins)
    assert not missing, f'src/api imports packages requirements-api.txt does not install: {missing}'


def test_requirements_api_pins_the_versions_this_suite_runs_against():
    for name, pinned in _requirements().items():
        installed = importlib.metadata.version(name)
        assert installed == pinned, f'{name}: pinned {pinned}, the suite runs {installed}'


def test_requirements_api_carries_none_of_the_shard_or_optimiser_packages():
    heavy = {'tensorflow', 'torch', 'waymo-open-dataset-tf-2-11-0', 'scipy', 'shapely', 'numpy'}
    assert not heavy & set(_requirements()), 'the API server must not install these'


def test_python_version_names_the_python_this_suite_runs_on():
    pinned = open(os.path.join(PROJECT, '.python-version')).read().strip()
    assert re.fullmatch(r'\d+\.\d+\.\d+', pinned), pinned
    running = '.'.join(str(p) for p in sys.version_info[:2])
    assert pinned.rsplit('.', 1)[0] == running, f'.python-version {pinned}, suite runs {running}'


def test_requirements_txt_no_longer_points_at_the_unused_dockerfile():
    text = open(os.path.join(PROJECT, 'requirements.txt')).read()
    assert 'Dockerfile' not in text
    assert 'requirements-api.txt' in text, 'it should say where the deployed API gets its packages'


# ── DEPLOY.md ────────────────────────────────────────────────────────────────────

def _deploy():
    return open(DEPLOY).read()


def _tables_the_api_reads():
    """The schema's tables (every CREATE TABLE in src/scoring) named after FROM or JOIN
    anywhere in src/api. Derived, so a new table the API starts reading fails the grant
    test below until DEPLOY.md grants it too."""
    schema = ''.join(open(os.path.join(PROJECT, 'src', 'scoring', f)).read()
                     for f in ('db.py', 'export_geometry.py'))
    tables = set(re.findall(r'CREATE TABLE IF NOT EXISTS ([a-z_]+)', schema))
    source = ''.join(open(os.path.join(API_DIR, f)).read()
                     for f in os.listdir(API_DIR) if f.endswith('.py'))
    read = {t for t in re.findall(r'\b(?:FROM|JOIN)\s+([a-z_]+)', source) if t in tables}
    assert read, f'premise: the API reads some of {sorted(tables)}'
    return read


def test_the_read_only_role_is_granted_select_on_exactly_the_tables_the_api_reads():
    sql = _deploy()
    grant = re.search(r'GRANT SELECT ON ([a-z_, ]+) TO av_api_ro;', sql)
    assert grant, 'DEPLOY.md has no GRANT SELECT ... TO av_api_ro'
    granted = {t.strip() for t in grant.group(1).split(',')}
    assert granted == _tables_the_api_reads()


def test_the_read_only_role_also_sees_tables_created_later():
    deploy = _deploy()
    assert ('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO av_api_ro;'
            in deploy)
    assert 'default_transaction_read_only = on' in deploy
    assert re.search(r'\\password av_api_ro', deploy), 'the password must be set at a silent prompt'


def test_deploy_md_starts_the_api_the_way_render_expects():
    assert 'uvicorn src.api.main:app --host 0.0.0.0 --port $PORT' in _deploy()
    assert 'pip install -r requirements-api.txt' in _deploy()


def test_deploy_md_checks_carry_the_runs_real_numbers():
    flat = ' '.join(_deploy().split())
    for expected in ('total_scenarios: 100', 'stress_tested: 18', 'collisions_found: 18',
                     'replay_infeasible: 2', 'no_collision_found: 0', 'with_geometry: 20'):
        assert expected in flat, f'DEPLOY.md checks are missing {expected!r}'


def test_deploy_md_contains_no_credential():
    deploy = _deploy()
    assert not re.search(r'postgres(?:ql)?://[^\s/:@]+:[^\s@]+@', deploy), 'a URL with a password'
    assert not re.search(r"PASSWORD\s+'", deploy), 'a password literal in SQL'


# ── the frontend ─────────────────────────────────────────────────────────────────

def test_vercel_rewrites_every_path_to_the_spa_entry():
    config = json.load(open(os.path.join(PROJECT, 'frontend', 'vercel.json')))
    assert config['rewrites'] == [{'source': '/(.*)', 'destination': '/index.html'}]


def test_the_vite_config_runs_the_api_url_guard_on_every_build():
    """The guard itself is unit-tested in frontend/src/build; this pins that the config
    calls it with the mode and the VITE_ env, so a production build cannot skip it."""
    config = open(os.path.join(PROJECT, 'frontend', 'vite.config.ts')).read()
    assert "import { requireApiBaseUrl } from './src/build/requireApiBaseUrl';" in config
    assert "requireApiBaseUrl(mode, loadEnv(mode, '.', 'VITE_'));" in config
