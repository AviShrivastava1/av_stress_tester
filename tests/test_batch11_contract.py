"""
test_batch11_contract.py — Batch 11 contract tests (audit A02, A12, A05).

The audit's own repros live in tests/test_audit3_regressions.py, reconstructed and
labelled as such. THIS file holds the properties: the things that must be true of any
correct fix, including the ones the repro cannot see.

The fingerprint is the whole batch. If it is not stable, every export in the corpus
starts refusing; if it is not sensitive, none of the three findings is actually fixed.
Both directions are asserted here, and the stability half matters more, because a
fingerprint that fails closed on valid input is worse than the defect it replaces —
the argument R10 and R11 both turned on.

Mostly pure numpy. The last section needs a DISPOSABLE Postgres/PostGIS database and
skips unless AV_CLAIMS_DB=1 — those three tests each exist to fail one specific
mutation that nothing else in the suite catches, which is why they are here rather
than folded into the repro file.

Run:
    ./venv/bin/python -m pytest tests/test_batch11_contract.py -q
    AV_CLAIMS_DB=1 PGDATABASE=<disposable> ./venv/bin/python -m pytest tests/test_batch11_contract.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring.db import compute_scene_fingerprint, compute_stress_run_id


def _scene(n=3, t=10):
    states = np.zeros((n, t, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[1, :, 0] = np.arange(t) * 1.5
    states[2, :, 1] = 7.25
    states[1, :, 4] = 0.75
    return states, np.ones((n, t), dtype=bool), np.array([1, 2, 3])


# ── stability: the half that must not fail closed ───────────────────────────────

def test_the_fingerprint_is_stable_across_repeated_calls():
    """Trivially true, and asserted anyway: everything below assumes it."""
    scene = _scene()
    first = compute_scene_fingerprint(*scene)
    assert all(compute_scene_fingerprint(*scene) == first for _ in range(20))


def test_the_fingerprint_ignores_memory_layout():
    """
    A view, a slice and a transposed-then-restored array all REPRESENT the same scene.
    ascontiguousarray is what makes the hash a function of the value rather than of how
    numpy happens to be storing it, and without it an export could refuse purely
    because the caller sliced its arrays differently.
    """
    states, validity, types = _scene()
    expected = compute_scene_fingerprint(states, validity, types)

    padded = np.zeros((states.shape[0], states.shape[1], 9), dtype=np.float32)
    padded[:, :, :7] = states
    assert compute_scene_fingerprint(padded[:, :, :7], validity, types) == expected, (
        'a non-contiguous view of the same values hashed differently'
    )

    round_tripped = np.ascontiguousarray(states.transpose(2, 1, 0).transpose(2, 1, 0))
    assert compute_scene_fingerprint(round_tripped, validity, types) == expected


def test_the_fingerprint_ignores_an_input_arrays_byte_order():
    """
    A caller may hand in a big-endian array; it must hash as the values it holds.

    WHAT THIS TEST CANNOT PROVE, stated because the obvious reading of it is wrong.
    It does NOT establish that spelling the dtype '<f4' rather than 'f4' is
    load-bearing. Measured on this machine:

        np.ascontiguousarray(big_endian, dtype='f4').tobytes()
          == np.ascontiguousarray(big_endian, dtype='<f4').tobytes()   -> True

    because 'f4' means NATIVE order and this host is little-endian, so both spellings
    convert identically here and no test runnable on this hardware can separate them.
    The '<' matters only when the fingerprint is computed on a big-endian host and
    verified on a little-endian one — the real risk being this project's documented
    Colab-3.8 / local-3.11 split — and that is an argument for the explicit spelling,
    not something this file demonstrates.

    Recorded rather than papered over: a mutation of '<f4' to 'f4' does not fail any
    test here, and reporting it as covered would be the same error as a guard that
    cannot be shown to fail.
    """
    states, validity, types = _scene()
    expected = compute_scene_fingerprint(states, validity, types)
    swapped = states.astype('>f4')
    assert swapped.dtype.byteorder == '>', 'fixture regressed: not big-endian'
    assert np.array_equal(swapped, states), 'fixture regressed: values changed'
    assert compute_scene_fingerprint(swapped, validity, types) == expected, (
        'the same scene hashed differently purely because of byte order'
    )


def test_the_fingerprint_ignores_an_equivalent_integer_width():
    """
    types arrives as whatever the parser produced — int64 on this machine, int32
    elsewhere. Casting to a fixed width before hashing is what stops that being a
    difference. The cast is lossless for the values involved (0..4).
    """
    states, validity, _ = _scene()
    expected = compute_scene_fingerprint(states, validity, np.array([1, 2, 3]))
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        assert compute_scene_fingerprint(
            states, validity, np.array([1, 2, 3], dtype=dtype)) == expected, dtype


# ── sensitivity: the half that makes the fix a fix ──────────────────────────────

@pytest.mark.parametrize('which', ['states', 'validity', 'types'])
def test_the_fingerprint_notices_each_array_independently(which):
    """
    All three are hashed, and a test that changed only `states` would pass against an
    implementation that ignored the other two entirely.
    """
    states, validity, types = _scene()
    before = compute_scene_fingerprint(states, validity, types)

    if which == 'states':
        states = states.copy()
        states[1, 5, 0] += 0.5
    elif which == 'validity':
        validity = validity.copy()
        validity[2, 3] = False
    else:
        types = types.copy()
        types[2] = 1

    assert compute_scene_fingerprint(states, validity, types) != before, (
        f'a change to {which} did not change the fingerprint'
    )


def test_the_fingerprint_notices_the_audits_own_displacement():
    """The repro's 10 m shift, at the level the fingerprint works on."""
    states, validity, types = _scene()
    moved = states.copy()
    moved[1, :, 0] += 10.0
    assert compute_scene_fingerprint(moved, validity, types) != \
        compute_scene_fingerprint(states, validity, types)


def test_the_fingerprint_notices_the_smallest_representable_change():
    """
    Not a rounding tolerance. A fingerprint answers "is this the same scene", and one
    float32 ulp is a different scene — the threshold question ('is this difference big
    enough to matter?') belongs to whoever reads the refusal, not to the identity.
    """
    states, validity, types = _scene()
    nudged = states.copy()
    nudged[0, 0, 0] = np.nextafter(np.float32(0.0), np.float32(1.0))
    assert nudged[0, 0, 0] != states[0, 0, 0]
    assert compute_scene_fingerprint(nudged, validity, types) != \
        compute_scene_fingerprint(states, validity, types)


def test_a_reshape_cannot_collide_with_a_different_scene():
    """
    THE REASON SHAPES ARE HASHED ALONGSIDE THE BYTES.

    A (3, 10, 7) array and a (10, 3, 7) array built from the same buffer have the same
    tobytes() and are different scenes. Hashing bytes alone would call them identical.
    """
    states, validity, types = _scene(n=3, t=10)
    reshaped = states.reshape(10, 3, 7)
    assert reshaped.tobytes() == states.tobytes(), 'fixture regressed'
    v2 = np.ones((10, 3), dtype=bool)
    assert compute_scene_fingerprint(reshaped, v2, types) != \
        compute_scene_fingerprint(states, validity, types)


def test_an_agent_count_change_is_visible():
    """
    B15 already removes stale agent rows on re-export. This is the other half: a scene
    that GAINED or LOST an agent is a different scene before any row is written.
    """
    states, validity, types = _scene(n=3)
    fewer = compute_scene_fingerprint(states[:2], validity[:2], types[:2])
    assert fewer != compute_scene_fingerprint(states, validity, types)


# ── the two identities are independent ──────────────────────────────────────────

def test_the_scene_and_the_run_are_different_questions():
    """
    compute_stress_run_id identifies a PERTURBATION APPLIED TO a scene;
    compute_scene_fingerprint identifies the scene. The whole batch exists because the
    first was being used to answer the second, so they must not be derivable from one
    another — a run id that moved with the scene would make the fingerprint redundant,
    and one that did not would leave A02 open exactly as it was.
    """
    states, validity, types = _scene()
    moved = states.copy()
    moved[1, :, 0] += 10.0

    # The scene changed; the run id did not, because it never depended on the scene.
    run_a = compute_stress_run_id('sid', 1, [-1.0, 0.0, 0.0, 0.0], 'de')
    run_b = compute_stress_run_id('sid', 1, [-1.0, 0.0, 0.0, 0.0], 'de')
    assert run_a == run_b, 'fixture regressed'
    assert compute_scene_fingerprint(moved, validity, types) != \
        compute_scene_fingerprint(states, validity, types), (
        'the fingerprint is as blind to the scene as the run id was'
    )

    # And the converse: a different delta moves the run id while the scene stands still.
    assert compute_stress_run_id('sid', 1, [-2.0, 0.0, 0.0, 0.0], 'de') != run_a


def test_the_fingerprint_is_the_shape_the_run_id_is():
    """
    16 lowercase hex, matching compute_stress_run_id. Both are stored in TEXT columns
    compared with IS NOT DISTINCT FROM, and a length or case difference between them
    would be a needless second convention in the same table.
    """
    fingerprint = compute_scene_fingerprint(*_scene())
    assert len(fingerprint) == 16
    assert fingerprint == fingerprint.lower()
    assert all(c in '0123456789abcdef' for c in fingerprint)
    assert len(fingerprint) == len(compute_stress_run_id('s', 0, [0.0], 'de'))


def test_the_version_prefix_is_actually_in_the_hash():
    """
    The prefix exists so the algorithm can change without a new value colliding with an
    old one. That claim is only true if the prefix is hashed rather than decorative —
    which is exactly the kind of thing that reads as obviously-present and is not.
    """
    import hashlib
    states, validity, types = _scene()
    parts = []
    for array, dtype in ((states, '<f4'), (validity, 'bool'), (types, '<i4')):
        canonical = np.ascontiguousarray(np.asarray(array), dtype=dtype)
        parts.append(repr(canonical.shape).encode('ascii'))
        parts.append(canonical.tobytes())

    without_prefix = hashlib.sha256(b'|'.join(parts)).hexdigest()[:16]
    assert compute_scene_fingerprint(states, validity, types) != without_prefix, (
        'the version prefix is not part of the hash, so a future scenefp2 could '
        'silently collide with a scenefp1 value'
    )


# ── DB-backed: three guards nothing else would catch ────────────────────────────

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

_RESULT = {'status': 'ok', 'outcome': 'collision_found', 'collision': True,
           'min_perturbation': 0.5, 'delta': [-1.0, 0.0, 0.0, 0.0],
           'collision_timestep': 9, 'target_idx': 1, 'method': 'de'}


def _ped(shift=0.0):
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 5:7] = [0.6, 0.6]
    states[1, :, 0] = 5.0 - 2.0 * np.arange(10) * 0.1 + shift
    states[1, :, 2] = -2.0
    states[1, :, 4] = np.pi
    return states, np.ones((2, 10), dtype=bool), np.array([1, 2])


@pytest.fixture
def bconn():
    from src.scoring import db
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


@requires_db
def test_a_changed_scene_invalidates_geometry_even_at_an_unchanged_delta(bconn):
    """
    THE REASON THE FINGERPRINT IS IN A05'S DELETE PREDICATE, AND THE ONLY TEST THAT
    FAILS IF IT IS REMOVED.

    A05 makes the perturbed_paths delete conditional so a bit-for-bit identical retry
    stops destroying working geometry. Conditioning on stress_run_id alone looks
    sufficient — until a re-parsed scene yields the SAME optimal delta, which produces
    the SAME run id by construction. The old-scene geometry then survives a result
    computed against the new scene, and the read-side join pairs them happily: A05
    fixed by reopening A02 one table over.

    test_A05_a_genuinely_new_result_still_invalidates_geometry cannot see this, because
    it changes the delta and so changes the run id too.
    """
    from src.scoring import db
    from src.scoring.export_geometry import export_scenario_agents, export_perturbed_path

    states, validity, types = _ped()
    fp_old = compute_scene_fingerprint(states, validity, types)
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp_old)])
    db.update_stress_results(bconn, {'s': dict(_RESULT)})
    export_scenario_agents(bconn, 's', states, validity, types, 0)
    export_perturbed_path(bconn, 's', states, validity, 1, delta=_RESULT['delta'],
                          method='de', scene_fingerprint=fp_old)

    with bconn.cursor() as cur:
        cur.execute("SELECT count(*) FROM perturbed_paths WHERE scenario_id='s'")
        assert cur.fetchone()[0] == 1, 'fixture regressed: nothing exported'

    # Pass 1 re-runs over a re-parsed scene. Pass 2 then reproduces the IDENTICAL
    # delta, so the run id does not move.
    moved, mv, mt = _ped(shift=10.0)
    fp_new = compute_scene_fingerprint(moved, mv, mt)
    assert fp_new != fp_old, 'fixture regressed'
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp_new)])
    before = db.fetch_scenario(bconn, 's')['stress_run_id']
    db.update_stress_results(bconn, {'s': dict(_RESULT)})
    assert db.fetch_scenario(bconn, 's')['stress_run_id'] == before, (
        'fixture regressed: the run id moved, so this would pass on run id alone'
    )

    with bconn.cursor() as cur:
        cur.execute("SELECT count(*) FROM perturbed_paths WHERE scenario_id='s'")
        assert cur.fetchone()[0] == 0, (
            'geometry from the OLD scene survived a result computed against the NEW '
            'one, at an unchanged run id — the delete is conditioned on the run alone'
        )


@requires_db
def test_the_read_side_will_not_pair_a_result_with_another_scenes_path(bconn):
    """
    THE REASON routes.py's JOIN GAINS THE FINGERPRINT.

    The write-side guards refuse to CREATE a mismatched pair. This is the second line
    of defence for a pair that exists already — a row written out of band, or by a
    partially-completed export — and it is the same argument B14 made for adding
    stress_run_id to that join in the first place. An absent path is the right failure
    mode; drawing an old scene's trajectory beside a current result and labelling it
    evidence is worse than drawing nothing.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.scoring import db
    from src.scoring.export_geometry import export_scenario_agents, export_perturbed_path

    states, validity, types = _ped()
    fp = compute_scene_fingerprint(states, validity, types)
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp)])
    db.update_stress_results(bconn, {'s': dict(_RESULT)})
    export_scenario_agents(bconn, 's', states, validity, types, 0)
    export_perturbed_path(bconn, 's', states, validity, 1, delta=_RESULT['delta'],
                          method='de', scene_fingerprint=fp)

    with TestClient(app) as client:
        assert client.get('/scenarios/s/perturbed').json()['perturbed'] is not None, (
            'fixture regressed: a matching pair must be served'
        )

    # Out of band: the path now claims a scene the score row does not.
    with bconn.cursor() as cur:
        cur.execute("UPDATE perturbed_paths SET scene_fingerprint = 'deadbeefdeadbeef' "
                    "WHERE scenario_id = 's'")
    bconn.commit()

    with TestClient(app) as client:
        data = client.get('/scenarios/s/perturbed').json()
    assert data['perturbed'] is None, (
        'a path from a different scene was served beside this result'
    )
    assert data['min_perturbation'] is not None, (
        'the score fields must survive — only the picture is withheld'
    )


@requires_db
def test_a_rescore_without_a_fingerprint_does_not_erase_the_recorded_one(bconn):
    """
    THE REASON upsert_scores COALESCEs RATHER THAN OVERWRITING.

    Pass 1 owns this column, but not every caller of upsert_scores computes it — the
    audit fixtures do not, and neither does any hand-built record. A bare
    `scene_fingerprint = EXCLUDED.scene_fingerprint` would let such a caller NULL out a
    fingerprint that had been recorded, which silently returns the row to the legacy
    carve-out and leaves every later export unguarded. A fix that can be undone by an
    unrelated write is not a fix.
    """
    from src.scoring import db

    states, validity, types = _ped()
    fp = compute_scene_fingerprint(states, validity, types)
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp)])

    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=8.0,
                                  min_pet=8.0, fragility_score=2.0)])

    with bconn.cursor() as cur:
        cur.execute("SELECT scene_fingerprint, fragility_score FROM scenario_scores "
                    "WHERE scenario_id = 's'")
        stored, fragility = cur.fetchone()
    assert fragility == 2.0, 'the re-score did not land'
    assert stored == fp, (
        'a re-score with no fingerprint erased the recorded one, returning the row to '
        'the legacy carve-out'
    )


@requires_db
def test_a_stale_baseline_is_not_served_after_the_cross_function_window(bconn):
    """
    THE COMPOSITION GAP BETWEEN THE TWO EXPORTS, CLOSED RATHER THAN DOCUMENTED.

    export_scenario_agents commits and releases its FOR UPDATE lock before
    export_perturbed_path opens its own transaction, so a concurrent Pass 1 can move
    scenario_scores.scene_fingerprint in between. The perturbed half self-guards and
    refuses — no mismatched PAIR is ever published — but the baseline was already
    committed, and get_trajectories reads scenario_agents with no other identity check
    of any kind. Before scenario_agents carried a fingerprint, that stale scene was
    SERVED as current geometry.

    This was briefly accepted as a narrow residual and should not have been: R03's
    precedent is that a window you can close to zero beats one you can argue is small.
    Two things had to be true, and this fails if either is removed — the column on
    scenario_agents, or the predicate in get_trajectories.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.scoring import db
    from src.scoring.export_geometry import (
        export_scenario_agents, export_perturbed_path, SceneChangedError,
    )

    old_states, ov, ot = _ped()
    fp_old = compute_scene_fingerprint(old_states, ov, ot)
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp_old)])
    db.update_stress_results(bconn, {'s': dict(_RESULT)})
    export_scenario_agents(bconn, 's', old_states, ov, ot, 0)

    with TestClient(app) as client:
        assert client.get('/scenarios/s/trajectories').json()['agents'], (
            'fixture regressed: the baseline must be served before the window opens'
        )

    # The window: a concurrent Pass 1 lands between the two exports.
    new_states, nv, nt = _ped(shift=10.0)
    fp_new = compute_scene_fingerprint(new_states, nv, nt)
    assert fp_new != fp_old, 'fixture regressed'
    other = db.get_connection()
    db.upsert_scores(other, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp_new)])
    other.close()

    # The perturbed half refuses on its own, as it always did.
    with pytest.raises(SceneChangedError):
        export_perturbed_path(bconn, 's', old_states, ov, 1, delta=_RESULT['delta'],
                              method='de', scene_fingerprint=fp_old)
    bconn.rollback()

    # The baseline is still committed in the table — and must no longer be served.
    with bconn.cursor() as cur:
        cur.execute("SELECT count(*) FROM scenario_agents WHERE scenario_id = 's'")
        assert cur.fetchone()[0] == 2, (
            'fixture regressed: the stale rows should still be present, just inert'
        )

    with TestClient(app) as client:
        data = client.get('/scenarios/s/trajectories').json()
    assert data['agents'] == [], (
        'geometry built from a scene that no longer matches the stored row was served '
        'as current — the baseline side has no identity check'
    )


@requires_db
def test_an_explicit_run_id_does_not_vouch_for_the_scene(bconn):
    """
    A12 THROUGH A DIFFERENT DOOR, and nothing exercises it — which is why it is here.

    An earlier version of the refusal was gated on
    `derived = stress_run_id is not None or delta is not None`. A caller passing
    stress_run_id= explicitly — a pattern this function's own docstring invites —
    made `derived` true, skipped the refusal, and then the write fell back to
    `stored_scene`, read from the same row moments earlier. The scene half of the
    WHERE compared the row against ITSELF: the vacuous self-comparison the whole A12
    finding is about, reappearing with the run half still perfectly sound.

    A census found zero callers in that state (`stress_run_id=` appears nowhere in this
    repository outside the signature), so this is dead in practice and the contract was
    still wrong — the treatment B16 and R05 got. Asserted here because a guard nothing
    exercises is a guard that rots.
    """
    from src.scoring import db
    from src.scoring.export_geometry import (
        export_scenario_agents, export_perturbed_path, SceneChangedError,
    )

    states, validity, types = _ped()
    fp = compute_scene_fingerprint(states, validity, types)
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp)])
    db.update_stress_results(bconn, {'s': dict(_RESULT)})
    export_scenario_agents(bconn, 's', states, validity, types, 0)
    run_id = db.fetch_scenario(bconn, 's')['stress_run_id']

    # The CORRECT run id, supplied explicitly, with geometry from a scene 99 m away.
    # The run half of the check cannot see this: the id really is the current one.
    wrong, wv, _ = _ped(shift=99.0)
    with pytest.raises(SceneChangedError):
        export_perturbed_path(bconn, 's', wrong, wv, 1, stress_run_id=run_id)
    bconn.rollback()

    with bconn.cursor() as cur:
        cur.execute("SELECT count(*) FROM perturbed_paths WHERE scenario_id = 's'")
        assert cur.fetchone()[0] == 0, (
            'an explicitly-supplied run id published geometry from another scene'
        )

    # And the same call WITH a scene named is still allowed, so this is a demand for
    # provenance rather than a ban on the parameter.
    assert export_perturbed_path(bconn, 's', states, validity, 1,
                                 stress_run_id=run_id, scene_fingerprint=fp) is True


# ── fix F03: get_perturbed's OWN version of the cross-function-window gap ───────
#
# test_a_stale_baseline_is_not_served_after_the_cross_function_window, above, closed
# this for get_trajectories. get_perturbed had the identical gap in two places — its
# own baseline read of scenario_agents, and the LABEL RESOLUTION fallback's separate
# read of the same table — neither guarded by scene_fingerprint, even though the
# column and the pattern already existed one function away. A run-id-matched
# perturbed_paths row (get_perturbed's OWN pre-existing guard) proves nothing about
# scenario_agents's own fingerprint: export_scenario_agents and export_perturbed_path
# are independent calls, each checked against scenario_scores at its OWN export time,
# so the two tables can legitimately disagree about which scene they last verified
# against — exactly the race the test above measures for get_trajectories.

def _veh_challenger(shift=0.0):
    """
    Same shape as _ped(), but agent_idx 1 (the challenger _RESULT's target_idx names)
    is typed as a VEHICLE rather than a pedestrian. Used only to make a stale-label
    test decisive rather than coincidentally correct: _ped(shift=...) alone changes
    position but not agent type, so a test built on it could pass even unguarded, the
    same "fully-valid data hides a real gap" trap B13's own docstring warns about.
    """
    states, validity, _ = _ped(shift=shift)
    return states, validity, np.array([1, 1])


@requires_db
def test_F03_a_stale_baseline_and_perturbed_dimensions_are_not_served_after_the_cross_function_window(bconn):
    """
    THE FINDING'S OWN REPRODUCTION, get_perturbed's side. Both scenario_agents reads
    (the baseline, and the perturbed track's dimensions, which are read off the SAME
    base_row — see the comment at routes.py's `is_sdc=False` perturbed-track
    construction) must stop being served once the scene has moved on, while the
    perturbed track's PATH — sourced from perturbed_paths, independently re-exported
    against the current scene below — keeps rendering. That split (path present,
    dimensions absent) is the same degraded shape
    test_only_one_geometry_table_present_still_degrades already established as
    correct for a missing scenario_agents table; a stale one must degrade the same
    way, not serve wrong numbers silently.

    THE RACE MUST LEAVE perturbed_paths CURRENT, DELIBERATELY, or this test would not
    isolate anything: if perturbed_paths were ALSO left stale, its own pre-existing
    guard would already return pert_row=None, and execution would never reach the
    scenario_agents queries this fix touches at all.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.scoring import db
    from src.scoring.export_geometry import export_scenario_agents, export_perturbed_path

    states_a, va, ta = _ped()
    fp_a = compute_scene_fingerprint(states_a, va, ta)
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp_a)])
    db.update_stress_results(bconn, {'s': dict(_RESULT)})
    export_scenario_agents(bconn, 's', states_a, va, ta, 0)
    export_perturbed_path(bconn, 's', states_a, va, 1, delta=_RESULT['delta'],
                          method=_RESULT['method'], scene_fingerprint=fp_a)

    with TestClient(app) as client:
        before = client.get('/scenarios/s/perturbed').json()
    assert before['baseline'] is not None, 'fixture regressed: baseline must serve pre-race'
    assert before['baseline']['length_m'] is not None
    assert before['perturbed']['length_m'] is not None, (
        'fixture regressed: perturbed dimensions must serve pre-race'
    )

    # The window: a concurrent Pass 1 lands between the two exports, same shape as
    # the get_trajectories test above.
    states_b, vb, tb = _ped(shift=10.0)
    fp_b = compute_scene_fingerprint(states_b, vb, tb)
    assert fp_b != fp_a, 'fixture regressed'
    other = db.get_connection()
    db.upsert_scores(other, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp_b)])
    other.close()

    # perturbed_paths re-exported AGAINST THE NEW SCENE, so its own guard still
    # passes and pert_row stays alive — the baseline read is what this test is about.
    export_perturbed_path(bconn, 's', states_b, vb, 1, delta=_RESULT['delta'],
                          method=_RESULT['method'], scene_fingerprint=fp_b)

    with TestClient(app) as client:
        after = client.get('/scenarios/s/perturbed').json()

    assert after['perturbed'] is not None and after['perturbed']['path'], (
        'fixture regressed: the re-exported perturbed path must still serve'
    )
    assert after['baseline'] is None, (
        'a baseline built from a scene that no longer matches the stored row was '
        'served as current — the scenario_agents read has no identity check'
    )
    # THE SECOND-ORDER CONSEQUENCE (what the finding asked to check specifically):
    # base_row feeds the PERTURBED track's dimensions too, not only the baseline's.
    assert after['perturbed']['agent_type'] is None, (
        'the perturbed track kept a stale agent_type from the old scene'
    )
    assert after['perturbed']['length_m'] is None, (
        'the perturbed track kept a stale length_m from the old scene'
    )
    assert after['perturbed']['width_m'] is None, (
        'the perturbed track kept a stale width_m from the old scene'
    )


@requires_db
def test_F03_the_label_resolution_fallback_does_not_use_a_stale_agent_type(bconn):
    """
    THE SECOND, INDEPENDENT INSTANCE, isolated from the baseline query above: no
    perturbed_paths row is ever exported, so pert_row is None throughout and the
    baseline query never runs at all — the ONLY route into scenario_agents here is
    the LABEL RESOLUTION fallback (audit B13), which _resolve_delta_labels reads
    agent_type from when search_provenance carries no parameterization, exactly
    _RESULT's own shape.

    _veh_challenger, not _ped shifted, so a stale read would be DEMONSTRABLY wrong
    (vehicle labels for a delta the current scene's own agent type says is a
    pedestrian's) rather than coincidentally right — B13's own docstring warns
    against a test that cannot tell the difference.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.scoring import db
    from src.api.models import LINEAR_DELTA_LABELS
    from src.scoring.export_geometry import export_scenario_agents

    states_a, va, ta = _ped()   # agent_idx 1 is a pedestrian (type 2)
    fp_a = compute_scene_fingerprint(states_a, va, ta)
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp_a)])
    db.update_stress_results(bconn, {'s': dict(_RESULT)})   # no search_provenance
    export_scenario_agents(bconn, 's', states_a, va, ta, 0)

    with TestClient(app) as client:
        before = client.get('/scenarios/s/perturbed').json()
    assert before['delta_labels'] == LINEAR_DELTA_LABELS, (
        f'fixture regressed: pre-race labels must infer linear from the pedestrian '
        f'type: {before["delta_labels"]}'
    )

    # The race: Pass 1 moves to a scene whose agent_idx 1 is a VEHICLE instead.
    states_b, vb, tb = _veh_challenger(shift=10.0)
    fp_b = compute_scene_fingerprint(states_b, vb, tb)
    assert fp_b != fp_a, 'fixture regressed'
    other = db.get_connection()
    db.upsert_scores(other, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp_b)])
    other.close()

    with TestClient(app) as client:
        after = client.get('/scenarios/s/perturbed').json()
    assert after['delta_labels'] is None, (
        f"the fallback inferred labels from scenario_agents's stale, pre-race agent "
        f"type instead of refusing to guess: {after['delta_labels']}"
    )


@requires_db
def test_F03_a_genuinely_matching_scene_still_serves_baseline_and_labels(bconn):
    """
    THE HOLE THIS FIX MUST NOT OPEN, both instances at once: the ordinary case —
    scenario_agents genuinely describes the scene scenario_scores currently records
    — is what the rest of this project's test suite exercises constantly, and it
    must keep working exactly as before.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.scoring import db
    from src.api.models import LINEAR_DELTA_LABELS
    from src.scoring.export_geometry import export_scenario_agents, export_perturbed_path

    states, validity, types = _ped()
    fp = compute_scene_fingerprint(states, validity, types)
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0,
                                  scene_fingerprint=fp)])
    db.update_stress_results(bconn, {'s': dict(_RESULT)})
    export_scenario_agents(bconn, 's', states, validity, types, 0)
    export_perturbed_path(bconn, 's', states, validity, 1, delta=_RESULT['delta'],
                          method=_RESULT['method'], scene_fingerprint=fp)

    with TestClient(app) as client:
        data = client.get('/scenarios/s/perturbed').json()

    assert data['baseline'] is not None
    assert data['baseline']['length_m'] == pytest.approx(0.6)
    assert data['perturbed'] is not None
    assert data['perturbed']['length_m'] == pytest.approx(0.6)
    assert data['perturbed']['agent_type'] == 2
    assert data['delta_labels'] == LINEAR_DELTA_LABELS


@requires_db
def test_F03_a_legacy_row_with_no_recorded_fingerprint_still_serves(bconn):
    """
    THE MIRROR: a scenario Pass 1 never fingerprinted (or exported before Batch 11's
    columns existed) must not become permanently unable to serve a baseline. Same
    NULL-means-not-recorded carve-out A12 established for the export path and F02
    extended to update_stress_results; this is the read-side third instance.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.scoring import db
    from src.scoring.export_geometry import export_scenario_agents, export_perturbed_path

    states, validity, types = _ped()
    # No scene_fingerprint= at all — the pre-Batch-11 shape.
    db.upsert_scores(bconn, [dict(scenario_id='s', shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0)])
    before = db.fetch_scenario(bconn, 's')
    assert before['scene_fingerprint'] is None, 'fixture regressed'

    db.update_stress_results(bconn, {'s': dict(_RESULT)})
    export_scenario_agents(bconn, 's', states, validity, types, 0)
    with bconn.cursor() as cur:
        cur.execute("SELECT scene_fingerprint FROM scenario_agents "
                    "WHERE scenario_id = 's' AND agent_idx = 1")
        assert cur.fetchone()[0] is None, 'fixture regressed: must stamp NULL too'
    export_perturbed_path(bconn, 's', states, validity, 1, delta=_RESULT['delta'],
                          method=_RESULT['method'])

    with TestClient(app) as client:
        data = client.get('/scenarios/s/perturbed').json()
    assert data['baseline'] is not None, (
        'a legacy row with no recorded scene identity was refused a baseline — the '
        'carve-out closed'
    )
    assert data['baseline']['length_m'] == pytest.approx(0.6)
