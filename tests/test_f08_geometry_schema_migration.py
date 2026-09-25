"""
test_f08_geometry_schema_migration.py — fix F08 (independent review, post-F02/F03/F04/F05).

update_stress_results checks `to_regclass('perturbed_paths') IS NOT NULL` once, up
front, before the per-scenario loop, and skips the perturbed_paths DELETE entirely
when that is false — Pass 2 legitimately runs before Pass 3's geometry tables exist
at all, and that state must not crash. But table existence is not schema
compatibility: the DELETE's WHERE clause references pp.stress_run_id and
pp.scene_fingerprint, both added to perturbed_paths via separate
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statements inside the idempotent
GEOMETRY_SCHEMA_SQL, after the CREATE TABLE. A perturbed_paths table created by an
init_geometry_schema() that ran once, before those ALTERs existed, and was never
re-run since is a real, reachable state — Postgres resolves column references at
parse time, so pp.scene_fingerprint on that table raises UndefinedColumn, caught by
the per-scenario SAVEPOINT and landing in `.failures` as a cryptic SQL error for
EVERY scenario that would otherwise qualify, over a table the write does not need
yet.

This file holds four properties: the stale-schema state degrades safely rather than
crashing, that state is reported distinctly (not conflated with "the table does not
exist", which is the ordinary, unremarkable pre-Pass-3 state), the fully-migrated
path is unaffected, and — fix G07, independent review, 2026-09-25 — the column check
itself resolves the SAME relation to_regclass and the DELETE resolve, rather than
matching column names across every same-named table on the whole search path. The
original check (`information_schema.columns WHERE table_name = 'perturbed_paths'`)
had no schema in it at all: a second schema's own perturbed_paths table, sharing no
relationship with the one this connection actually writes to, could supply columns
the query counted as if they belonged to the table being checked. Fixed by resolving
to_regclass('perturbed_paths') once and querying pg_attribute keyed on that exact
OID.

Needs a DISPOSABLE Postgres/PostGIS database and skips unless AV_CLAIMS_DB=1, same
requirement as the other DB-gated files in this project.

Run:
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_f08_geometry_schema_migration.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring import db

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

SID = 'f08_geometry_schema'

# search_ran=True (delta/target_idx/method present) and no scene_fingerprint key, so
# scene_matches is true by the NULL-permissive carve-out (fix F02) regardless of what
# this fixture seeds — this file is about the geometry TABLE'S schema, not the scene
# identity guard, and must not accidentally also exercise that one.
_RESULT = {'status': 'ok', 'outcome': 'collision_found', 'collision': True,
           'min_perturbation': 0.5, 'delta': [-1.0, 0.0, 0.0, 0.0],
           'collision_timestep': 9, 'target_idx': 1, 'method': 'de'}


def _seed(conn, sid=SID):
    db.upsert_scores(conn, [dict(scenario_id=sid, shard='x', n_agents=2, min_ttc=9.0,
                                 min_pet=9.0, fragility_score=1.0)])


def _ped(shift=0.0):
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 5:7] = [0.6, 0.6]
    states[1, :, 0] = 5.0 - 2.0 * np.arange(10) * 0.1 + shift
    states[1, :, 2] = -2.0
    states[1, :, 4] = np.pi
    return states, np.ones((2, 10), dtype=bool), np.array([1, 2])


@pytest.fixture
def bare_conn():
    """scenario_scores only — the state before Pass 3 has ever run at all. The
    ordinary case geometry_exists's original table-existence check already handled
    correctly; kept here as the contrast case F08's fix must not regress."""
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, '
                    'scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    yield connection
    connection.rollback()
    connection.close()


@pytest.fixture
def stale_geom_conn():
    """
    scenario_scores plus a perturbed_paths table shaped as it was BEFORE the
    stress_run_id/scene_fingerprint ALTERs existed — a real init_geometry_schema()
    that ran once, long ago, and has not been re-run since this project's later
    batches added those columns. Deliberately hand-rolled rather than calling
    init_geometry_schema() and dropping columns afterward: this is what the table
    actually looked like at that point in the project's history, not a mutation of
    today's schema.
    """
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, '
                    'scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    with connection.cursor() as cur:
        cur.execute("""
            CREATE TABLE perturbed_paths (
                scenario_id TEXT PRIMARY KEY
                            REFERENCES scenario_scores(scenario_id) ON DELETE CASCADE,
                target_idx  INTEGER NOT NULL,
                n_points    INTEGER NOT NULL,
                exported_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
    connection.commit()
    yield connection
    connection.rollback()
    connection.close()


@pytest.fixture
def migrated_conn():
    """scenario_scores plus a fully current perturbed_paths — the ordinary path this
    fix must leave unchanged."""
    from src.scoring.export_geometry import init_geometry_schema
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, '
                    'scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    init_geometry_schema(connection)
    yield connection
    connection.rollback()
    connection.close()


@pytest.fixture
def cross_schema_shadow_conn():
    """
    THE G07 REPRO. public.perturbed_paths carries ONLY stress_run_id; a second
    schema's OWN perturbed_paths — same name, unrelated table, never touched by
    this connection's actual writes — carries ONLY scene_fingerprint. Neither
    table alone has both columns, but information_schema.columns has no schema
    in its WHERE clause and counts 2 across the two of them, which is exactly the
    state that made update_stress_results report this schema as complete while
    to_regclass('perturbed_paths') (and the DELETE) resolve to public's table
    alone, which does not have scene_fingerprint at all.

    The shadow schema is not on the search path (CREATE SCHEMA never adds
    itself), so this is not a search_path manipulation — just a second,
    ordinary schema that happens to hold a same-named table, the way a
    migration-shadow or a differently-privileged app schema legitimately might.
    """
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP SCHEMA IF EXISTS audit_shadow CASCADE')
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, '
                    'scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    with connection.cursor() as cur:
        cur.execute("""
            CREATE TABLE perturbed_paths (
                scenario_id TEXT PRIMARY KEY
                            REFERENCES scenario_scores(scenario_id) ON DELETE CASCADE,
                target_idx  INTEGER NOT NULL,
                n_points    INTEGER NOT NULL,
                exported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                stress_run_id TEXT
            )
        """)
        cur.execute('CREATE SCHEMA audit_shadow')
        cur.execute("""
            CREATE TABLE audit_shadow.perturbed_paths (
                scenario_id TEXT PRIMARY KEY,
                scene_fingerprint TEXT
            )
        """)
    connection.commit()
    yield connection
    connection.rollback()
    with connection.cursor() as cur:
        cur.execute('DROP SCHEMA IF EXISTS audit_shadow CASCADE')
    connection.commit()
    connection.close()


@requires_db
def test_F08_a_stale_perturbed_paths_schema_does_not_crash_the_update(stale_geom_conn):
    """
    THE FINDING'S OWN REPRO. Pre-fix, `pp.scene_fingerprint` in the DELETE's WHERE
    raised UndefinedColumn on exactly this table shape, caught by the per-scenario
    savepoint — so the whole scenario write, scoring included, landed in .failures
    over a table the write does not even touch for anything else.
    """
    _seed(stale_geom_conn)

    report = db.update_stress_results(stale_geom_conn, {SID: dict(_RESULT)})

    assert report.failures == [], (
        f'the write failed over a table it does not need for scoring: {report.failures}'
    )
    assert int(report) == 1, 'the scenario itself must still be scored'
    assert db.fetch_scenario(stale_geom_conn, SID)['min_perturbation'] == 0.5, (
        'the result must actually be persisted, not merely absent from .failures'
    )
    assert report.geometry_schema_incomplete is True, (
        'a perturbed_paths table missing the columns this call needs must be '
        'reported, not silently limped past'
    )


@requires_db
def test_G07_a_same_named_table_in_another_schema_does_not_launder_completeness(
        cross_schema_shadow_conn):
    """
    THE G07 REPRO. public.perturbed_paths has stress_run_id alone; a second
    schema's own perturbed_paths has scene_fingerprint alone. Pre-fix,
    information_schema.columns summed both schemas' matching columns (2) and
    reported geometry_schema_incomplete=False, but update_stress_results
    resolves and touches public.perturbed_paths ALONE — which does not have
    scene_fingerprint — so the DELETE raised UndefinedColumn anyway, over a
    schema check that had just claimed everything was fine.
    """
    _seed(cross_schema_shadow_conn)

    report = db.update_stress_results(cross_schema_shadow_conn, {SID: dict(_RESULT)})

    assert report.failures == [], (
        f'the write failed over a table it does not need for scoring: {report.failures}'
    )
    assert int(report) == 1, 'the scenario itself must still be scored'
    assert db.fetch_scenario(cross_schema_shadow_conn, SID)['min_perturbation'] == 0.5, (
        'the result must actually be persisted, not merely absent from .failures'
    )
    assert report.geometry_schema_incomplete is True, (
        "public.perturbed_paths is missing scene_fingerprint — a second schema's "
        'unrelated table having it must not launder this into "complete"'
    )


@requires_db
def test_F08_a_missing_table_is_not_reported_as_incomplete_schema(bare_conn):
    """
    THE CARVE-OUT THIS FIX MUST NOT BREAK IN THE OTHER DIRECTION. "The table does
    not exist yet" is the ordinary, unremarkable pre-Pass-3 state — not a migration
    problem — and must stay indistinguishable from success on this flag, exactly as
    it already is on geometry_exists.
    """
    _seed(bare_conn)

    report = db.update_stress_results(bare_conn, {SID: dict(_RESULT)})

    assert report.failures == []
    assert int(report) == 1
    assert report.geometry_schema_incomplete is False, (
        'a database where Pass 3 has simply never run was reported as if its schema '
        'were stale — these are different states with different remedies'
    )


@requires_db
def test_F08_a_fully_migrated_schema_still_invalidates_and_reports_nothing(migrated_conn):
    """
    THE ORDINARY PATH MUST NOT BECOME A TAX. A fully current perturbed_paths must
    keep invalidating stale geometry exactly as it did before this fix (audit B14),
    and must not report geometry_schema_incomplete — the column-compatibility check
    must not itself become a false positive on the common case.
    """
    from src.scoring.export_geometry import export_scenario_agents, export_perturbed_path

    states, validity, types = _ped()
    _seed(migrated_conn)
    first = dict(_RESULT, delta=[-1.0, 0.0, 0.0, 0.0])
    db.update_stress_results(migrated_conn, {SID: first})
    export_scenario_agents(migrated_conn, SID, states, validity, types, 0)
    export_perturbed_path(migrated_conn, SID, states, validity, 1,
                          delta=first['delta'], method='de')

    with migrated_conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM perturbed_paths WHERE scenario_id = %s', (SID,))
        assert cur.fetchone()[0] == 1, 'fixture regressed: nothing exported'

    second = dict(_RESULT, delta=[-2.0, 0.0, 0.0, 0.0], min_perturbation=1.0)
    report = db.update_stress_results(migrated_conn, {SID: second})

    assert report.failures == []
    assert report.geometry_schema_incomplete is False
    with migrated_conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM perturbed_paths WHERE scenario_id = %s', (SID,))
        assert cur.fetchone()[0] == 0, (
            'a genuinely new result must still invalidate the old geometry (audit B14) '
            '— the schema-compatibility check must not weaken this'
        )
