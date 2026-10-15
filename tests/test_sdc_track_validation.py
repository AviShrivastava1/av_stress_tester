"""
An SDC index that cannot be scored on must not be scored as a scenario with no risk.

The SDC-restricted engines answer "no interaction" (999 / 999, fragility 0) for an index
outside the scene and for an SDC that was never observed, and a fractional, Boolean, None or
string index fell through to an unnamed numpy error. validate_sdc_track now refuses each with
a named SdcTrackError before any danger value is computed, and Pass 1 records the scenario
as not scored, with that reason, through the mechanism it already had for a scenario it
cannot score: an entry in score_shard's `errors` and no record, so no row is written.

The accepted side is pinned with the 18 SDC tracks the repository has as production-shaped
records (frontend/src/test/realCollisionFrames.json: the agent index and the frames each was
observed at), and with scenes whose SDC is observed but never interacts, which must still score.
"""
import json
import os
import struct
import sys
import types as _types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring import batch_scorer
from src.scoring.batch_scorer import SdcTrackError, _score_one, validate_sdc_track

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(PROJECT, 'frontend', 'src', 'test', 'realCollisionFrames.json')

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)


def _scene(n=3, T=10):
    """Agent 0 drives at agent 1 and would reach it; agent 2 is far away."""
    states = np.zeros((n, T, 7), dtype=np.float32)
    validity = np.ones((n, T), dtype=bool)
    t = np.arange(T) * 0.1
    states[:, :, 5:7] = [4.5, 2.0]
    states[0, :, 0] = -20.0 + 10.0 * t
    states[0, :, 2] = 10.0
    states[1, :, 0] = 30.0
    if n > 2:
        states[2, :, 1] = 50.0
    return states, validity


# ── the validator ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize('index', [0, 1, 2, np.int64(1), np.int32(2), np.uint8(0)])
def test_a_usable_index_is_returned_as_a_plain_int(index):
    _, validity = _scene()
    result = validate_sdc_track(index, validity, scenario_id='s')
    assert result == int(index) and type(result) is int


def _unobserved():
    _, validity = _scene()
    validity[0, :] = False
    return validity


BAD = [
    # name,                     index,           validity,                 words that must be in the message
    ('past the last agent',      3,               lambda: _scene()[1],      ['out of range', '3 agents', '0..2']),
    ('far past the last agent',  70,              lambda: _scene()[1],      ['out of range', '70']),
    ('negative',                 -1,              lambda: _scene()[1],      ['-1', 'negative']),
    ('never observed',           0,               _unobserved,              ['no observed frame']),
    ('fractional',               0.5,             lambda: _scene()[1],      ['float', 'whole-number']),
    ('a whole float',            1.0,             lambda: _scene()[1],      ['float', 'whole-number']),
    ('True',                     True,            lambda: _scene()[1],      ['Boolean', 'True']),
    ('False',                    False,           lambda: _scene()[1],      ['Boolean', 'False']),
    ('numpy True',               np.bool_(True),  lambda: _scene()[1],      ['Boolean']),
    ('None',                     None,            lambda: _scene()[1],      ['NoneType']),
    ('a string',                 '0',             lambda: _scene()[1],      ['str']),
]


@pytest.mark.parametrize('name,index,validity,words', BAD, ids=[b[0] for b in BAD])
def test_each_bad_index_is_refused_by_name(name, index, validity, words):
    with pytest.raises(SdcTrackError) as raised:
        validate_sdc_track(index, validity(), scenario_id='scene_77')
    message = str(raised.value)
    assert message.startswith('cannot score scene_77: SDC index ')
    for word in words:
        assert word in message, message
    assert raised.value.scenario_id == 'scene_77'
    assert isinstance(raised.value, ValueError)
    assert len(message) < 120          # a reason, not the arrays


def test_an_observed_sdc_that_never_interacts_is_not_refused():
    states, validity = _scene(n=2)
    states[1, :, 1] = 500.0            # far from agent 0 the whole time
    record = _score_one(states, validity, 'quiet', 0, 'shard')
    assert (record['min_ttc'], record['min_pet'], record['fragility_score']) == (999.0, 999.0, 0.0)
    assert record['sdc_idx'] == 0


# ── the accepted side: the repository's production-shaped SDC tracks ────────────────


def _real_sdc_tracks():
    with open(REAL, encoding='utf-8') as f:
        scenarios = json.load(f)['scenarios']
    return [(e['scenario_id'], e['sdc']) for e in scenarios]


def test_the_real_sdc_tracks_are_all_there():
    tracks = _real_sdc_tracks()
    assert len(tracks) == 18 and all(len(t['timesteps']) >= 2 for _, t in tracks)


@pytest.mark.parametrize('sid,track', [pytest.param(s, t, id=s) for s, t in _real_sdc_tracks()])
def test_every_production_shaped_sdc_still_validates(sid, track):
    index = track['agent_idx']
    validity = np.zeros((index + 3, 91), dtype=bool)          # the SDC is not agent 0 in these scenes
    validity[index, [int(m) for m in track['timesteps']]] = True
    assert validate_sdc_track(index, validity, scenario_id=sid) == index


# ── _score_one: refused before anything is computed ─────────────────────────────────


@pytest.mark.parametrize('name,index,validity,words', BAD, ids=[b[0] for b in BAD])
def test_score_one_produces_no_record_for_a_bad_index(monkeypatch, name, index, validity, words):
    def must_not_run(*args, **kwargs):
        raise AssertionError('the danger engines ran for an SDC index that was refused')
    monkeypatch.setattr(batch_scorer, 'score_scenario', must_not_run)
    states, _ = _scene()
    with pytest.raises(SdcTrackError):
        _score_one(states, validity(), 'scene_77', index, 'shard')


def test_score_one_is_unchanged_for_a_usable_index():
    states, validity = _scene()
    record = _score_one(states, validity, 'ok', 0, 'shard')
    assert record['scenario_id'] == 'ok' and record['sdc_idx'] == 0 and record['n_agents'] == 3
    assert record['min_ttc'] < 999 and record['fragility_score'] > 0


# ── Pass 1: the scenario is recorded as not scored, with the reason ────────────────

SCENES = {}


def _write_shard(path, payloads):
    from src.data.loader import _masked_crc32c
    blob = b''
    for payload in payloads:
        header = struct.pack('<Q', len(payload))
        blob += (header + struct.pack('<I', _masked_crc32c(header))
                 + payload + struct.pack('<I', _masked_crc32c(payload)))
    path.write_bytes(blob)
    return str(path)


@pytest.fixture
def stub_parser(monkeypatch):
    module = _types.ModuleType('src.data.parser')

    class ScenarioParser:
        def __init__(self, raw):
            self.sid = raw.decode()

        def get_scenario_id(self):
            return self.sid

        def get_agent_states(self):
            return SCENES[self.sid][0]

        def get_agent_validity(self):
            return SCENES[self.sid][1]

        def get_agent_types(self):
            return np.ones(SCENES[self.sid][0].shape[0], dtype=int)

        def get_sdc_index(self):
            return SCENES[self.sid][2]

    module.ScenarioParser = ScenarioParser
    monkeypatch.setitem(sys.modules, 'src.data.parser', module)
    SCENES.clear()
    yield
    SCENES.clear()


def _pass_one(tmp_path, scenes):
    for sid, (index, validity) in scenes.items():
        SCENES[sid] = (_scene()[0], validity if validity is not None else _scene()[1], index)
    path = _write_shard(tmp_path / 'shard.tfrecord', [sid.encode() for sid in scenes])
    return batch_scorer.score_shard(path, verbose=False)


def test_pass_one_records_each_refused_scenario_as_not_scored_and_scores_the_rest(tmp_path, stub_parser):
    records, errors = _pass_one(tmp_path, {
        'good_first': (0, None),
        'past_the_end': (9, None),
        'negative': (-1, None),
        'unobserved': (0, _unobserved()),
        'fractional': (0.5, None),
        'good_last': (1, None),
    })
    assert [r['scenario_id'] for r in records] == ['good_first', 'good_last']
    assert all(r['sdc_idx'] in (0, 1) for r in records)
    refused = {e['scenario_id']: e for e in errors}
    assert set(refused) == {'past_the_end', 'negative', 'unobserved', 'fractional'}
    assert refused['past_the_end']['error'].startswith('SdcTrackError: cannot score past_the_end: SDC index 9 is out of range')
    assert 'negative' in refused['negative']['error']
    assert 'no observed frame' in refused['unobserved']['error']
    assert 'whole-number' in refused['fractional']['error']       # was an unnamed IndexError
    assert all('kind' not in e for e in errors)                   # not the shard_truncated kind
    assert [e['index'] for e in errors] == [1, 2, 3, 4]


@requires_db
def test_no_row_is_written_for_a_refused_scenario(tmp_path, stub_parser):
    from src.scoring import db
    records, errors = _pass_one(tmp_path, {'kept': (0, None), 'refused': (9, None)})
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, scenario_scores CASCADE')
        conn.commit()
        db.init_schema(conn)
        db.upsert_scores(conn, records)
        with conn.cursor() as cur:
            cur.execute('SELECT scenario_id FROM scenario_scores ORDER BY 1')
            assert [r[0] for r in cur.fetchall()] == ['kept']
    finally:
        conn.rollback()
        conn.close()
    assert [e['scenario_id'] for e in errors] == ['refused']
