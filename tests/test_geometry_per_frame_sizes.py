"""
test_geometry_per_frame_sizes.py — scenario_agents records each agent's box size at every
exported frame, in two arrays aligned with the path.

The exact collision check, the DE archive's signed gap and PET all build an agent's box from
the length and width recorded at THAT frame. The export stored one scalar per agent, read at
its first valid frame, so the stored geometry cannot reproduce what the verifier used. The
two arrays carry the per-frame sizes: element i is the size at vertex i of `path`, and at
headings[i], because all three are built from the same list of valid frames. The scalar
columns are unchanged.

A size that PerturbationSpace._valid_dimension would refuse (not finite, or not above zero)
is stored as NULL for that element. NaN and infinity cannot go into the API's JSON, and
zero or a negative number is not a box. The array keeps its length, so alignment survives.

Expected values are built here from explicit per-frame size tables passed through
float(np.float32(...)); nothing is read back from a run.

Needs a DISPOSABLE Postgres/PostGIS database and skips unless AV_CLAIMS_DB=1, the same
requirement as every other DB-gated file. The dump/restore test also needs a pg_dump whose
major version equals the server's; it looks in AV_PG_BIN, then on PATH, and skips otherwise.

Run:
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_geometry_per_frame_sizes.py -q
"""

import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring import db

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SID = 'per_frame_sizes'
T = 10

# Valid frames per agent, and the size each agent had at each of them. Different for every
# frame and for the two agents, so an array shifted by one place or taken from the wrong
# agent cannot match.
VALID = {0: [2, 3, 4, 5, 6, 7, 8, 9], 1: [0, 1, 4, 5, 9]}


def _length(agent, t):
    return 4.0 + 0.37 * agent + 0.013 * t


def _width(agent, t):
    return 1.8 + 0.21 * agent + 0.007 * t


def _f32(x):
    return float(np.float32(x))


def _scene(length=_length, width=_width):
    """Two agents, T frames. Invalid frames hold a junk size (99.0) that must never be stored."""
    states = np.zeros((2, T, 7), dtype=np.float32)
    validity = np.zeros((2, T), dtype=bool)
    for agent, frames in VALID.items():
        for t in range(T):
            states[agent, t, 0] = 10.0 * agent + t          # x
            states[agent, t, 1] = 3.0 * agent               # y
            states[agent, t, 4] = 0.01 * t                  # heading
            if t in frames:
                validity[agent, t] = True
                states[agent, t, 5] = length(agent, t)
                states[agent, t, 6] = width(agent, t)
            else:
                states[agent, t, 5] = 99.0
                states[agent, t, 6] = 99.0
    return states, validity, np.array([1, 2])


@pytest.fixture
def conn():
    from src.scoring.export_geometry import init_geometry_schema
    c = db.get_connection()
    with c.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, scenario_scores CASCADE')
    c.commit()
    db.init_schema(c)
    init_geometry_schema(c)
    db.upsert_scores(c, [dict(scenario_id=SID, shard='x', n_agents=2, min_ttc=9.0,
                              min_pet=9.0, fragility_score=1.0)])
    yield c
    c.rollback()
    c.close()


def _export(conn, states, validity, types):
    from src.scoring.export_geometry import export_scenario_agents
    return export_scenario_agents(conn, SID, states, validity, types, 0)


def _rows(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT agent_idx, length_m, width_m, lengths_m, widths_m, n_points, headings,
                   ST_AsText(path)
              FROM scenario_agents WHERE scenario_id = %s ORDER BY agent_idx
        """, (SID,))
        return {r[0]: dict(length_m=r[1], width_m=r[2], lengths_m=r[3], widths_m=r[4],
                           n_points=r[5], headings=r[6], wkt=r[7]) for r in cur.fetchall()}


def _measures(wkt):
    """The M ordinate of every vertex of a 'LINESTRING M (x y m, ...)'."""
    body = re.search(r'\((.*)\)', wkt).group(1)
    return [int(float(v.split()[2])) for v in body.split(',')]


# ── 1. alignment ─────────────────────────────────────────────────────────────────────


@requires_db
def test_the_arrays_have_one_element_per_vertex_and_each_is_that_vertexs_size(conn):
    _export(conn, *_scene())
    rows = _rows(conn)

    assert set(rows) == {0, 1}
    for agent, row in rows.items():
        frames = _measures(row['wkt'])
        assert frames == VALID[agent]                    # premise: M is the frame index
        assert len(row['lengths_m']) == len(row['widths_m']) == len(row['headings']) == len(frames)
        assert row['n_points'] == len(frames)
        assert row['lengths_m'] == [_f32(_length(agent, t)) for t in frames]
        assert row['widths_m'] == [_f32(_width(agent, t)) for t in frames]


@requires_db
def test_sizes_on_frames_the_agent_was_not_observed_are_not_stored(conn):
    _export(conn, *_scene())
    for row in _rows(conn).values():
        assert 99.0 not in row['lengths_m'] and 99.0 not in row['widths_m']


@requires_db
def test_the_scalar_columns_are_unchanged_the_size_at_the_first_valid_frame(conn):
    # Passes on main on purpose: the scalars are not touched by this change. Reads only the
    # scalar columns, so it does not depend on the new ones existing.
    _export(conn, *_scene())
    with conn.cursor() as cur:
        cur.execute('SELECT agent_idx, length_m, width_m FROM scenario_agents '
                    'WHERE scenario_id = %s ORDER BY agent_idx', (SID,))
        scalars = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    for agent, frames in VALID.items():
        assert scalars[agent] == (_f32(_length(agent, frames[0])), _f32(_width(agent, frames[0])))
        # not simply the last frame's size, nor the mean: this is the first valid frame's
        assert scalars[agent][0] != _f32(_length(agent, frames[-1]))


@requires_db
def test_a_re_export_replaces_the_arrays(conn):
    _export(conn, *_scene())
    _export(conn, *_scene(length=lambda a, t: 5.5 + 0.1 * t, width=lambda a, t: 2.2 + 0.02 * t))
    rows = _rows(conn)
    for agent, frames in VALID.items():
        assert rows[agent]['lengths_m'] == [_f32(5.5 + 0.1 * t) for t in frames]
        assert rows[agent]['widths_m'] == [_f32(2.2 + 0.02 * t) for t in frames]


# ── 2. a size the verifier's own dimension rule refuses ──────────────────────────────

BAD = [float('nan'), float('inf'), float('-inf'), 0.0, -2.0]
BAD_IDS = ['nan', 'inf', '-inf', 'zero', 'negative']


@requires_db
@pytest.mark.parametrize('field', ['length', 'width'])
@pytest.mark.parametrize('bad', BAD, ids=BAD_IDS)
def test_an_unusable_size_at_one_frame_is_null_at_that_element_and_nothing_else_moves(conn, field, bad):
    bad_frame = 5

    def length(a, t):
        return bad if (field == 'length' and a == 0 and t == bad_frame) else _length(a, t)

    def width(a, t):
        return bad if (field == 'width' and a == 0 and t == bad_frame) else _width(a, t)

    _export(conn, *_scene(length, width))
    row = _rows(conn)[0]
    frames = VALID[0]
    idx = frames.index(bad_frame)

    bad_array, good_array = ((row['lengths_m'], row['widths_m']) if field == 'length'
                             else (row['widths_m'], row['lengths_m']))
    good_fn = _width if field == 'length' else _length
    bad_fn = _length if field == 'length' else _width

    assert len(bad_array) == len(good_array) == len(frames)           # alignment survives
    assert bad_array[idx] is None
    assert [v for i, v in enumerate(bad_array) if i != idx] == \
           [_f32(bad_fn(0, t)) for t in frames if t != bad_frame]
    assert good_array == [_f32(good_fn(0, t)) for t in frames]
    # What the API will serialise must be valid JSON: NaN and Infinity are not.
    json.dumps({'lengths_m': row['lengths_m'], 'widths_m': row['widths_m']}, allow_nan=False)


@requires_db
def test_an_agent_with_no_usable_size_gets_an_all_null_array_of_the_right_length(conn):
    _export(conn, *_scene(length=lambda a, t: float('nan') if a == 1 else _length(a, t)))
    row = _rows(conn)[1]
    assert row['lengths_m'] == [None] * len(VALID[1])
    assert row['widths_m'] == [_f32(_width(1, t)) for t in VALID[1]]


# ── 3. the schema ────────────────────────────────────────────────────────────────────

LEGACY_AGENTS_DDL = """
    CREATE TABLE scenario_agents (
        scenario_id TEXT NOT NULL REFERENCES scenario_scores(scenario_id) ON DELETE CASCADE,
        agent_idx   INTEGER NOT NULL,
        agent_type  INTEGER,
        is_sdc      BOOLEAN NOT NULL DEFAULT FALSE,
        length_m    DOUBLE PRECISION,
        width_m     DOUBLE PRECISION,
        n_points    INTEGER NOT NULL,
        headings    DOUBLE PRECISION[],
        path        geometry(LINESTRINGM, 0),
        PRIMARY KEY (scenario_id, agent_idx)
    )
"""


@pytest.fixture
def legacy_conn():
    """scenario_agents exactly as the first schema created it (no fingerprint, no sdc_idx,
    no per-frame arrays), holding one row written then."""
    c = db.get_connection()
    with c.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, scenario_scores CASCADE')
    c.commit()
    db.init_schema(c)
    db.upsert_scores(c, [dict(scenario_id=SID, shard='x', n_agents=2, min_ttc=9.0,
                              min_pet=9.0, fragility_score=1.0)])
    with c.cursor() as cur:
        cur.execute(LEGACY_AGENTS_DDL)
        cur.execute("""
            INSERT INTO scenario_agents
                (scenario_id, agent_idx, agent_type, is_sdc, length_m, width_m, n_points,
                 headings, path)
            VALUES (%s, 0, 1, TRUE, 4.5, 2.0, 2, %s, ST_GeomFromText('LINESTRING M (0 0 0, 1 0 1)', 0))
        """, (SID, [0.0, 0.0]))
    c.commit()
    yield c
    c.rollback()
    c.close()


def _array_columns(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT attname, format_type(atttypid, atttypmod), attnotnull
              FROM pg_attribute
             WHERE attrelid = to_regclass('scenario_agents') AND attnum > 0 AND NOT attisdropped
               AND attname IN ('lengths_m', 'widths_m') ORDER BY attname
        """)
        return cur.fetchall()


@requires_db
def test_a_legacy_table_without_the_columns_is_migrated_by_the_schema_init(legacy_conn):
    from src.scoring.export_geometry import init_geometry_schema
    assert _array_columns(legacy_conn) == []                              # premise

    init_geometry_schema(legacy_conn)
    init_geometry_schema(legacy_conn)                                     # and again: idempotent

    assert _array_columns(legacy_conn) == [('lengths_m', 'double precision[]', False),
                                           ('widths_m', 'double precision[]', False)]


@requires_db
def test_a_row_that_existed_before_the_columns_reads_null_and_keeps_its_scalars(legacy_conn):
    from src.scoring.export_geometry import init_geometry_schema
    init_geometry_schema(legacy_conn)
    row = _rows(legacy_conn)[0]
    assert row['lengths_m'] is None and row['widths_m'] is None
    assert (row['length_m'], row['width_m']) == (4.5, 2.0)


@requires_db
def test_exporting_over_a_migrated_legacy_row_fills_the_arrays(legacy_conn):
    from src.scoring.export_geometry import init_geometry_schema
    init_geometry_schema(legacy_conn)
    _export(legacy_conn, *_scene())
    row = _rows(legacy_conn)[0]
    assert row['lengths_m'] == [_f32(_length(0, t)) for t in VALID[0]]


# ── 4. the read-only role ────────────────────────────────────────────────────────────

ROLE = 'av_a1_ro_test'


def _deploy_grants():
    """DEPLOY.md's grant statements for the read-only role, with the role renamed so a test
    never touches a real av_api_ro. Taken from the document, not retyped."""
    text = open(os.path.join(PROJECT, 'DEPLOY.md')).read()
    grant = re.search(r'GRANT SELECT ON [a-z_, ]+ TO av_api_ro;', text).group(0)
    default = ('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO av_api_ro;')
    assert default in text
    return [s.replace('av_api_ro', ROLE) for s in (grant, default)]


@pytest.fixture
def role_conn(conn):
    with conn.cursor() as cur:
        if _role_exists(conn):
            cur.execute(f'DROP OWNED BY {ROLE}')
        cur.execute(f'DROP ROLE IF EXISTS {ROLE}')
        cur.execute(f'CREATE ROLE {ROLE} NOSUPERUSER NOCREATEDB NOCREATEROLE')
        cur.execute(f'GRANT USAGE ON SCHEMA public TO {ROLE}')
        for statement in _deploy_grants():
            cur.execute(statement)
    conn.commit()
    yield conn
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(f'DROP OWNED BY {ROLE}')
        cur.execute(f'DROP ROLE {ROLE}')
    conn.commit()


def _role_exists(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT 1 FROM pg_roles WHERE rolname = %s', (ROLE,))
        return cur.fetchone() is not None


def _role_can_read_the_arrays(conn):
    with conn.cursor() as cur:
        for column in ('lengths_m', 'widths_m'):
            cur.execute("SELECT has_column_privilege(%s, 'scenario_agents', %s, 'SELECT')", (ROLE, column))
            assert cur.fetchone()[0] is True, column
        cur.execute(f'SET ROLE {ROLE}')
        try:
            cur.execute('SELECT lengths_m, widths_m FROM scenario_agents WHERE scenario_id = %s ORDER BY agent_idx',
                        (SID,))
            return cur.fetchall()
        finally:
            cur.execute('RESET ROLE')
            conn.commit()


@requires_db
def test_the_deploy_grant_covers_the_new_columns_of_a_table_that_already_existed(role_conn):
    # The GRANT is table-level, so a column added afterwards is covered with no new grant.
    with role_conn.cursor() as cur:
        cur.execute('ALTER TABLE scenario_agents DROP COLUMN lengths_m, DROP COLUMN widths_m')
    role_conn.commit()
    from src.scoring.export_geometry import init_geometry_schema
    init_geometry_schema(role_conn)
    _export(role_conn, *_scene())
    assert len(_role_can_read_the_arrays(role_conn)) == 2


@requires_db
def test_the_deploy_grants_cover_the_tables_when_they_are_recreated_by_the_restoring_role(role_conn):
    # What a restore does: drop the tables and create them again as the connected user,
    # carrying no grants of their own (--no-privileges). Only the default privileges remain.
    from src.scoring.export_geometry import init_geometry_schema
    _export(role_conn, *_scene())
    with role_conn.cursor() as cur:
        cur.execute('DROP TABLE perturbed_paths, scenario_agents')
    role_conn.commit()
    init_geometry_schema(role_conn)
    _export(role_conn, *_scene())
    assert len(_role_can_read_the_arrays(role_conn)) == 2


def _pg_bin(server_major):
    """A directory holding pg_dump and pg_restore of the server's major version, or None."""
    candidates = []
    if os.environ.get('AV_PG_BIN'):
        candidates.append(os.environ['AV_PG_BIN'])
    found = shutil.which('pg_dump')
    if found:
        candidates.append(os.path.dirname(found))
    for directory in candidates:
        try:
            out = subprocess.run([os.path.join(directory, 'pg_dump'), '--version'],
                                 capture_output=True, text=True, check=True).stdout
            major = int(re.search(r'(\d+)(?:\.\d+)?\s*(?:\(|$)', out.strip()).group(1))
        except (OSError, subprocess.CalledProcessError, AttributeError, ValueError):
            continue
        if major == server_major and os.path.exists(os.path.join(directory, 'pg_restore')):
            return directory
    return None


@requires_db
def test_a_real_dump_restored_with_the_replacement_flags_leaves_the_role_able_to_read_the_arrays(role_conn, tmp_path):
    with role_conn.cursor() as cur:
        cur.execute('SHOW server_version_num')
        server_major = int(cur.fetchone()[0]) // 10000
    directory = _pg_bin(server_major)
    if directory is None:
        pytest.skip(f'no pg_dump {server_major} (set AV_PG_BIN)')

    _export(role_conn, *_scene(length=lambda a, t: float('nan') if (a == 0 and t == 5) else _length(a, t)))
    before = _rows(role_conn)
    role_conn.commit()                  # hold no lock while the restore drops the tables
    params = role_conn.get_dsn_parameters()
    env = dict(os.environ)
    target = ['-p', params['port'], '-U', params['user'], '-d', params['dbname']]
    if params.get('host'):
        target = ['-h', params['host'], *target]
    dump = str(tmp_path / 'tables.dump')
    tables = [arg for t in ('scenario_scores', 'scenario_agents', 'perturbed_paths') for arg in ('-t', t)]
    subprocess.run([os.path.join(directory, 'pg_dump'), '--format=custom', '--no-owner', '--no-privileges',
                    *tables, '--file', dump, *target], check=True, env=env)
    # The same flags as the replacement: the whole restore is one transaction.
    subprocess.run([os.path.join(directory, 'pg_restore'), '--clean', '--if-exists', '--single-transaction',
                    '--no-owner', '--no-privileges', *target, dump], check=True, env=env)

    role_conn.rollback()
    assert _rows(role_conn) == before                      # NULL elements and all
    assert len(_role_can_read_the_arrays(role_conn)) == 2
