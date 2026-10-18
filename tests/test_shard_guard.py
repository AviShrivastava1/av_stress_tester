"""
The loader guard (#18): upsert_scores (src/scoring/db.py) writes scenario_scores conflicting on scenario_id alone, and used to overwrite
`shard`, so a scenario id that two shards carry replaced the first shard's row with the second's. Now:

  * an id already stored under a DIFFERENT non-NULL shard is refused: ShardConflictError, nothing written for the call (not the other
    records either), the stored row byte-identical (its scored_at included);
  * the same shard again is a re-score, as it was;
  * a NULL shard on the stored row is not a conflict: the row takes the offered shard (it recorded none);
  * a record that offers no shard is written as it was (the shard column becomes NULL): not a conflict and not changed by this guard;
  * the same condition is inside the statement, for a row another writer stored between the check and the statement: it is left as it
    was and the call is refused too.

It touches neither the shards table nor any route.

Needs a DISPOSABLE Postgres database and skips unless AV_CLAIMS_DB=1, like every other DB-gated file here.

Run:
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_shard_guard.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)


def _rec(sid, shard, **over):
    return {'scenario_id': sid, 'shard': shard, 'n_agents': 3, 'min_ttc': 9.0, 'min_pet': 9.0, 'fragility_score': 1.0,
            'scene_fingerprint': 'fp', 'sdc_idx': 0, **over}


@pytest.fixture
def dconn():
    from src.scoring import db
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    yield connection
    connection.rollback()
    connection.close()


def _row(conn, sid):
    """The whole stored row as text, so that 'byte-identical' means every column, scored_at included."""
    with conn.cursor() as cur:
        cur.execute('SELECT t::text FROM scenario_scores t WHERE scenario_id = %s', (sid,))
        found = cur.fetchone()
        return None if found is None else found[0]


def _count(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM scenario_scores')
        return cur.fetchone()[0]


@requires_db
def test_an_id_stored_under_another_shard_is_refused_and_its_row_is_byte_identical(dconn):
    from src.scoring import db
    db.upsert_scores(dconn, [_rec('s', 'shard-a')])
    before = _row(dconn, 's')
    with pytest.raises(db.ShardConflictError) as e:
        db.upsert_scores(dconn, [_rec('s', 'shard-b', n_agents=99, min_ttc=1.0, scene_fingerprint='other', sdc_idx=5)])
    assert e.value.conflicts == [('s', 'shard-a', 'shard-b')]
    assert "'s'" in str(e.value) and "'shard-a'" in str(e.value) and "'shard-b'" in str(e.value) and 'Nothing was written' in str(e.value)
    assert _row(dconn, 's') == before


@requires_db
def test_the_whole_call_is_refused_not_just_the_conflicting_record(dconn):
    from src.scoring import db
    db.upsert_scores(dconn, [_rec('s', 'shard-a'), _rec('t', 'shard-a')])
    before = (_row(dconn, 's'), _row(dconn, 't'))
    with pytest.raises(db.ShardConflictError) as e:
        db.upsert_scores(dconn, [_rec('new', 'shard-b'), _rec('t', 'shard-a', n_agents=50), _rec('s', 'shard-b')])
    assert e.value.conflicts == [('s', 'shard-a', 'shard-b')]
    assert (_row(dconn, 's'), _row(dconn, 't')) == before and _row(dconn, 'new') is None and _count(dconn) == 2


@requires_db
def test_on_an_autocommit_connection_too_nothing_of_a_refused_call_is_written(dconn):
    # there a rollback cannot take a write back, so the refusal has to come before any write
    from src.scoring import db
    db.upsert_scores(dconn, [_rec('s', 'shard-a')])
    before = _row(dconn, 's')
    dconn.commit()
    dconn.autocommit = True
    with pytest.raises(db.ShardConflictError):
        db.upsert_scores(dconn, [_rec('new', 'shard-b'), _rec('s', 'shard-b')])
    assert _row(dconn, 's') == before and _row(dconn, 'new') is None and _count(dconn) == 1


@requires_db
def test_every_conflict_is_named_and_a_long_list_is_shortened_in_the_message(dconn):
    from src.scoring import db
    db.upsert_scores(dconn, [_rec(f's{i}', 'a') for i in range(5)])
    with pytest.raises(db.ShardConflictError) as e:
        db.upsert_scores(dconn, [_rec(f's{i}', 'b') for i in range(5)])
    assert [c[0] for c in e.value.conflicts] == [f's{i}' for i in range(5)]
    assert '5 scenario id(s)' in str(e.value) and '(and 2 more)' in str(e.value) and "'s3'" not in str(e.value)


@requires_db
def test_the_same_shard_again_is_a_rescore_as_before(dconn):
    from src.scoring import db
    db.upsert_scores(dconn, [_rec('s', 'shard-a')])
    n = db.upsert_scores(dconn, [_rec('s', 'shard-a', n_agents=7, min_ttc=2.5)])
    assert n == 1 and not n.rescoped
    row = db.fetch_scenario(dconn, 's')
    assert row['n_agents'] == 7 and row['min_ttc'] == 2.5 and row['shard'] == 'shard-a'


@requires_db
def test_a_stored_row_with_no_shard_is_not_a_conflict_and_takes_the_offered_shard(dconn):
    from src.scoring import db
    db.upsert_scores(dconn, [_rec('s', None)])
    assert db.fetch_scenario(dconn, 's')['shard'] is None
    db.upsert_scores(dconn, [_rec('s', 'shard-b', n_agents=4)])
    row = db.fetch_scenario(dconn, 's')
    assert row['shard'] == 'shard-b' and row['n_agents'] == 4


@requires_db
def test_a_record_that_offers_no_shard_is_written_as_before(dconn):
    # unchanged by the guard: no shard offered is no claim about the shard, so it is not refused (and, as before, it leaves the column NULL)
    from src.scoring import db
    db.upsert_scores(dconn, [_rec('s', 'shard-a')])
    assert db.upsert_scores(dconn, [_rec('s', None, n_agents=8)]) == 1
    row = db.fetch_scenario(dconn, 's')
    assert row['shard'] is None and row['n_agents'] == 8


@requires_db
def test_a_new_id_is_written(dconn):
    from src.scoring import db
    assert db.upsert_scores(dconn, [_rec('a', 'shard-a'), _rec('b', 'shard-b')]) == 2 and _count(dconn) == 2


@requires_db
def test_a_stored_result_is_not_touched_by_a_refused_write(dconn):
    from src.scoring import db
    db.upsert_scores(dconn, [_rec('s', 'shard-a')])
    db.update_stress_results(dconn, {'s': {'status': 'ok', 'outcome': 'collision_found', 'collision': True, 'min_perturbation': 0.5,
                                           'delta': [-1.0, 0.0, 0.0, 0.0], 'collision_timestep': 9, 'target_idx': 1, 'method': 'de',
                                           'scene_fingerprint': 'fp', 'sdc_idx': 0}})
    before = _row(dconn, 's')
    assert db.fetch_scenario(dconn, 's')['min_perturbation'] == 0.5
    with pytest.raises(db.ShardConflictError):
        db.upsert_scores(dconn, [_rec('s', 'shard-b', scene_fingerprint='a different scene')])
    assert _row(dconn, 's') == before


@requires_db
def test_a_row_another_writer_stored_after_the_check_is_left_alone_by_the_statement_and_the_call_is_refused(dconn, monkeypatch):
    # the check is made to see nothing, as it would if the other shard's row were stored just after it: the statement's own condition holds
    from src.scoring import db
    db.upsert_scores(dconn, [_rec('s', 'shard-a')])
    before = _row(dconn, 's')
    monkeypatch.setattr(db, '_shard_conflicts', lambda conn, records: [])
    with pytest.raises(db.ShardConflictError) as e:
        db.upsert_scores(dconn, [_rec('new', 'shard-b'), _rec('s', 'shard-b', n_agents=77)])
    assert e.value.conflicts == [('s', 'shard-a', 'shard-b')]
    assert _row(dconn, 's') == before and _row(dconn, 'new') is None          # and the call's other write is taken back


def _idle(conn):
    import psycopg2.extensions
    return conn.info.transaction_status == psycopg2.extensions.TRANSACTION_STATUS_IDLE


@requires_db
def test_a_refused_call_leaves_the_connection_with_no_open_transaction_so_it_holds_no_lock(dconn):
    # a connection left idle in a transaction holds its locks: the next DROP TABLE or ALTER of anyone else waits for it
    from src.scoring import db
    db.upsert_scores(dconn, [_rec('s', 'shard-a')])
    with pytest.raises(db.ShardConflictError):
        db.upsert_scores(dconn, [_rec('s', 'shard-b')])
    assert _idle(dconn)


@requires_db
def test_a_row_the_statement_left_also_leaves_no_open_transaction(dconn, monkeypatch):
    from src.scoring import db
    db.upsert_scores(dconn, [_rec('s', 'shard-a')])
    monkeypatch.setattr(db, '_shard_conflicts', lambda conn, records: [])
    with pytest.raises(db.ShardConflictError):
        db.upsert_scores(dconn, [_rec('s', 'shard-b')])
    assert _idle(dconn)


@requires_db
def test_a_check_that_fails_leaves_no_open_transaction_and_the_error_is_its_own(dconn, monkeypatch):
    from src.scoring import db

    def broken(conn, records):
        with conn.cursor() as cur:
            cur.execute('SELECT 1')
        raise RuntimeError('the check failed')
    monkeypatch.setattr(db, '_shard_conflicts', broken)
    with pytest.raises(RuntimeError, match='the check failed'):
        db.upsert_scores(dconn, [_rec('s', 'shard-a')])
    assert _idle(dconn) and _count(dconn) == 0


def test_the_shards_table_and_the_routes_are_not_part_of_it():
    import inspect
    from src.scoring import db
    source = inspect.getsource(db._shard_conflicts) + inspect.getsource(db.ShardConflictError)
    assert 'FROM shards' not in source and 'INSERT INTO shards' not in source
