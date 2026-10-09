"""
test_cursor_decode.py — what src.api.routes._decode_cursor accepts and refuses.

Pure unit tests: no database, no network. The cursor is what GET /scenarios hands back as
next_cursor and what a client must send back verbatim. Anything else is a bad REQUEST (422),
never a 500 and never a silently different page:

  * a non-finite score (NaN, +-inf, 1e999) makes the pagination predicate true for every row,
    so the page restarts from the top and a client that follows it can loop
  * a score that is not a number (a string, a bool, null, a list) or an id that is not a string
    was coerced by float()/str() into a position nobody minted
  * a payload with trailing junk, extra keys or a different spelling of the same number decodes
    to a usable position but is not what the server sent, so "pass it back verbatim" was not
    enforced
  * deeply nested JSON raises RecursionError and an enormous integer literal raises
    OverflowError; neither is a ValueError, so a narrow `except ValueError` would let them out
    as a 500

A cursor whose finite score and id are well formed but that no page produced is a valid
POSITION and is accepted: the cursor is unsigned, so it cannot be told from a real one.
"""

import base64
import json
import math
import os
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import routes
from src.api.routes import _MAX_CURSOR_LEN, _MAX_SCENARIO_ID_LEN, _decode_cursor, _encode_cursor

MALFORMED = 'Malformed cursor. Pass back a next_cursor value verbatim.'


def enc(raw: str) -> str:
    return base64.urlsafe_b64encode(raw.encode()).decode()


def refused(cursor):
    with pytest.raises(HTTPException) as caught:
        _decode_cursor(cursor)
    assert caught.value.status_code == 422, caught.value.status_code
    return caught.value.detail


# ── what is accepted ────────────────────────────────────────────────────────────

@pytest.mark.parametrize('score', [0.0, -0.0, 1.0, 2.5, 1.347500000000001, 5e-324, 1e308, -1e308,
                                   1.7976931348623157e308])
def test_a_cursor_the_server_mints_decodes_to_the_same_bits(score):
    f, s = _decode_cursor(_encode_cursor(score, 'scene_1'))
    assert repr(f) == repr(score) and s == 'scene_1'


@pytest.mark.parametrize('scenario_id', ['', 'a', 'ünïcødé', 'a b+c=d&e', '😀', 'x' * 200])
def test_any_storable_id_round_trips(scenario_id):
    assert _decode_cursor(_encode_cursor(1.5, scenario_id)) == (1.5, scenario_id)


def test_the_longest_cursor_the_server_can_mint_is_accepted():
    # the id bound the code enforces on the way back in is 4096 characters; the worst case is an
    # id of astral characters, which json.dumps writes as 12 bytes each (a surrogate pair)
    worst = _encode_cursor(-1.7976931348623157e308, '😀' * _MAX_SCENARIO_ID_LEN)
    assert len(worst) <= _MAX_CURSOR_LEN
    assert _decode_cursor(worst)[1] == '😀' * _MAX_SCENARIO_ID_LEN


def test_the_cap_is_well_above_the_longest_cursor_the_server_can_mint():
    worst = _encode_cursor(-1.7976931348623157e308, '😀' * _MAX_SCENARIO_ID_LEN)
    assert _MAX_CURSOR_LEN >= 2 * len(worst) - 8


def test_a_finite_position_no_page_produced_is_accepted():
    assert _decode_cursor(_encode_cursor(3.0, 'zzz')) == (3.0, 'zzz')


# ── what is refused ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('payload', [
    '{"f":NaN,"s":"a"}', '{"f":Infinity,"s":"a"}', '{"f":-Infinity,"s":"a"}', '{"f":1e999,"s":"a"}',
    '{"f":-1e999,"s":"a"}',
])
def test_a_non_finite_score_is_refused(payload):
    assert refused(enc(payload)) == MALFORMED


@pytest.mark.parametrize('payload', [
    '{"f":"nan","s":"a"}', '{"f":"2.0","s":"a"}', '{"f":"abc","s":"a"}', '{"f":true,"s":"a"}',
    '{"f":false,"s":"a"}', '{"f":null,"s":"a"}', '{"f":[1],"s":"a"}', '{"f":{"x":1},"s":"a"}',
])
def test_a_score_that_is_not_a_number_is_refused(payload):
    assert refused(enc(payload)) == MALFORMED


@pytest.mark.parametrize('payload', [
    '{"f":2.0,"s":5}', '{"f":2.0,"s":null}', '{"f":2.0,"s":[1]}', '{"f":2.0,"s":true}',
    '{"f":2.0,"s":{"a":1}}',
])
def test_an_id_that_is_not_a_string_is_refused(payload):
    assert refused(enc(payload)) == MALFORMED


@pytest.mark.parametrize('payload', ['{"s":"b"}', '{"f":2.0}', '{}', '[1,2]', '5', 'null', '"text"', 'true', 'hello', ''])
def test_a_payload_that_is_not_an_object_with_f_and_s_is_refused(payload):
    assert refused(enc(payload)) == MALFORMED


@pytest.mark.parametrize('cursor', ['', 'not-a-real-cursor!!', 'é', '=', '====', 'A'])
def test_something_that_is_not_base64_is_refused(cursor):
    assert refused(cursor) == MALFORMED


def test_a_truncated_cursor_is_refused():
    good = _encode_cursor(2.0, 'b')
    for n in range(len(good)):
        assert refused(good[:n]) == MALFORMED, n


def test_a_cursor_that_is_not_in_the_form_the_server_writes_is_refused():
    f_s = lambda s: enc(s)
    assert _decode_cursor(_encode_cursor(2.0, 'b')) == (2.0, 'b')
    for noncanonical in [
        _encode_cursor(2.0, 'b') + 'zz',                    # trailing junk after the padding
        enc('{"f":2.0,"s":"b","x":1}'),                     # an extra key
        enc('{"s":"b","f":2.0}'),                           # the keys in the other order
        enc('{"f": 2.0, "s": "b"}'),                        # whitespace
        enc('{"f":2,"s":"b"}'),                             # "2" for "2.0"
        enc('{"f":2.00,"s":"b"}'),                          # another spelling of the same number
        enc('{"f":2.0,"s":"\\u0062"}'),                     # an escaped "b"
        _encode_cursor(2.0, 'b').rstrip('='),               # padding removed
        _encode_cursor(2.0, 'b') + '=',                     # one padding character too many
        _encode_cursor(2.0, 'b') + '==',                    # two too many
    ]:
        assert refused(noncanonical) == MALFORMED, base64.urlsafe_b64decode(noncanonical + '==')


def test_an_id_with_a_control_character_or_a_lone_surrogate_keeps_its_own_message():
    assert 'control characters' in refused(enc('{"f":1.0,"s":"\\u0000"}'))
    assert 'unpaired surrogate' in refused(enc('{"f":1.0,"s":"\\ud800"}'))
    assert 'too long' in refused(_encode_cursor(1.0, 'x' * (_MAX_SCENARIO_ID_LEN + 1)))


# ── what must never escape as a 500 ─────────────────────────────────────────────

def test_deeply_nested_json_is_refused_not_raised():
    # json.loads raises RecursionError here, which is not a ValueError
    assert refused(enc('[' * 3000 + ']' * 3000)) == MALFORMED
    assert refused(enc('{"f":' * 1000 + '1' + '}' * 1000)) == MALFORMED


@pytest.mark.parametrize('sign', ['', '-'])
def test_an_integer_too_large_for_a_float_is_refused_not_raised(sign):
    # float(int) raises OverflowError, which is not a ValueError
    assert refused(enc('{"f":' + sign + '1' * 400 + ',"s":"a"}')) == MALFORMED


def test_an_integer_literal_past_the_interpreters_digit_limit_is_refused():
    assert refused(enc('{"f":' + '1' * 5000 + ',"s":"a"}')) == MALFORMED


def test_a_cursor_longer_than_anything_the_server_mints_is_refused_before_it_is_parsed(monkeypatch):
    calls = []

    def parsed(*args, **kwargs):
        calls.append(args)
        raise ValueError('recorded')
    monkeypatch.setattr(routes.json, 'loads', parsed)
    monkeypatch.setattr(routes.base64, 'urlsafe_b64decode', parsed)
    assert refused('A' * (_MAX_CURSOR_LEN + 1)) == MALFORMED
    assert refused('A' * 1_000_000) == MALFORMED
    assert calls == [], 'an over-length cursor reached the decoder'
    refused('A' * _MAX_CURSOR_LEN)   # at the cap it is decoded (and refused for being junk)
    assert len(calls) == 1


def test_every_refusal_is_a_422_with_one_of_three_messages():
    cursors = [enc('{"f":NaN,"s":"a"}'), enc('[1]'), 'x' * 10, enc('{"f":2.0,"s":"\\u0000"}')]
    for cursor in cursors:
        detail = refused(cursor)
        assert detail == MALFORMED or detail.startswith('Cursor scenario id'), detail
    assert math.isfinite(_decode_cursor(_encode_cursor(0.1, 'a'))[0])
