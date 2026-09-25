"""
test_audit3_regressions.py — third audit, finding A01.

PROVENANCE, STATED PLAINLY: reconstructed from the audit's description, NOT copied
from its own code. No third-audit bundle exists — searched exhaustively on 2026-09-15
across ~/Downloads, ~/Desktop, ~/Documents and /tmp: the only audit archive on disk is
av_stress_tester_audit_bundle.zip (2026-09-10), which is the FIRST, B-numbered audit,
and nothing anywhere contains a `test_A0*` name. av_stress_tester-main.zip is a source
snapshot of commit 2b71cb7 going OUT to the auditor, not a bundle coming back.

Same standard as the second audit's reconstructions: each test was run against the
UNFIXED code first and shown to reproduce the audit's SPECIFIC claim.

    A01  stationary cyclist at (0, 1.8), 2.0 x 0.6, heading 0, beside a 4.5 x 2.0 SDC
         at the origin; 10 valid frames; linear model (type 3)
           -> delta [0, 1e-9, 0, 0] rotated the footprint to arctan2(1e-9, 0) = pi/2
              EXACTLY, colliding at frame 0 with the centre unmoved, at weighted norm
              5.000000e-10. Identical at 1e-12, 1e-30 and 1e-45 — atan2(+x, 0) is
              pi/2 for every positive x, so more precision makes this worse, not
              better.
           -> at dvy0 <= 1e-30 the weighted norm was EXACTLY 0.0, because the float32
              square underflows while atan2 does not: a verified exact-SAT collision
              reported as caused by a perturbation of magnitude zero.
           -> a real DE run (popsize=4, maxiter=20, seed=1) found it unprompted at
              min_perturbation = 0.003215756034478545, matching the audit's
              0.003215756. Reachable by the ordinary pipeline:
              pick_nearest_challenger has no speed filter and selects this agent.

    A01b THE DEFECT THE AUDIT DID NOT REPORT, found while reproducing A01 and worse
         because it needs no perturbation at all: for a stationary agent vx = vy = 0
         in the LOGGED data too, so arctan2(0, 0) = 0 fired on the ZERO-delta replay.
         A parked agent logged facing 0.7 rad replayed facing 0.0000 — a 40-degree
         footprint rotation applied by a delta of exactly zero — while
         baseline_replay_error reported 0.0000 m, because _measure_baseline_replay
         compares states[..., :2] and heading is column 4.

THE DE FIGURE ABOVE IS NOT ASSERTED ANYWHERE. It matched the audit to every printed
digit on this machine, and it is still a constant obtained by running the code, which
Batch 5's review established twice is a portability hazard. It is recorded here as
evidence and derived at run time where a test needs it.

Run:
    ./venv/bin/python -m pytest tests/test_audit3_regressions.py -q
"""

import contextlib
import io
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.danger.collision_detector import check_collision_trajectory
from src.optimization.perturbation_space import PerturbationSpace


def _stationary_cyclist(heading=0.0, lateral=1.8):
    """The audit's fixture: a parked cyclist beside a stationary SDC."""
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]          # SDC at the origin
    states[1, :, 1] = lateral
    states[1, :, 5:7] = [2.0, 0.6]
    states[1, :, 4] = heading
    #                        3 = cyclist -> linear model, heading derived not carried
    return states, np.ones((2, 10), dtype=bool), np.array([1, 3])


@pytest.mark.parametrize('dvy0', [1e-9, 1e-12, 1e-30, 1e-45])
def test_A01_a_vanishing_delta_cannot_rotate_a_stationary_footprint(dvy0):
    """
    The audit's claim, parametrized over the magnitudes that prove it is not a
    precision artifact: atan2(+x, 0) is pi/2 for every positive x, so shrinking the
    delta makes the exploit cheaper without making the rotation smaller.

    Asserted as the INVARIANT rather than the mechanism — a delta this small must not
    produce a verified collision, however the implementation chooses to achieve that.
    """
    states, validity, types = _stationary_cyclist()
    space = PerturbationSpace(states, validity, types, 0, 1)
    assert not space.is_vehicle, 'fixture regressed: this must be a linear-model agent'

    delta = np.array([0.0, dvy0, 0.0, 0.0])
    perturbed = space.apply(delta)
    collided, frame = check_collision_trajectory(perturbed, validity, 0, 1)

    assert not collided, (
        f'dvy0={dvy0:g} produced a verified collision at frame {frame} for weighted '
        f'norm {space.weighted_norm(delta)!r}; heading went '
        f'{float(states[1, 0, 4]):.6f} -> {float(perturbed[1, 0, 4]):.6f} with the '
        f'centre unmoved'
    )


def test_A01_no_collision_is_ever_reported_at_zero_weighted_norm():
    """
    The sharpest form of the finding. At dvy0 <= 1e-30 the float32 square underflows
    and weighted_norm returns EXACTLY 0.0 while atan2 still returns pi/2 — so the
    project's headline number, the smallest perturbation that causes a collision,
    evaluated to zero for a real exact-SAT collision.

    Stated as a property of the whole search space, not of one delta: no delta whose
    weighted norm is zero may collide, because such a delta has by definition changed
    nothing this project can charge for.
    """
    states, validity, types = _stationary_cyclist()
    space = PerturbationSpace(states, validity, types, 0, 1)

    zero_norm_deltas = [
        np.zeros(4),
        np.array([0.0, 1e-30, 0.0, 0.0]),
        np.array([0.0, -1e-30, 0.0, 0.0]),
        np.array([1e-30, 1e-30, 1e-30, 1e-30]),
    ]
    # THE LOOP BELOW SKIPS ANY DELTA WHOSE NORM IS NOT ZERO, SO SOMETHING OUTSIDE IT
    # HAS TO CONFIRM THE INTERESTING CASE EXISTS. Without this the test would pass
    # having asserted nothing if float32 underflow behaved differently anywhere. The
    # standing lesson from Batch 4's `finite > 100` and Batch 6's `checked > 10000`:
    # a conditional `continue` is only a real test when its condition is shown to fire.
    assert any(space.weighted_norm(d) == 0.0 for d in zero_norm_deltas), (
        'no candidate delta produced a weighted norm of exactly 0.0 on this machine, '
        'so the property this test exists to check was never exercised'
    )
    for delta in zero_norm_deltas:
        norm = space.weighted_norm(delta)
        if norm != 0.0:
            continue                      # not a zero-norm delta on this machine
        collided, frame = check_collision_trajectory(
            space.apply(delta), validity, 0, 1)
        assert not collided, (
            f'delta {delta.tolist()} has weighted norm exactly {norm!r} and still '
            f'collides at frame {frame}: a collision caused by nothing'
        )


@pytest.mark.parametrize('seed', range(8))
def test_A01_a_real_search_must_pay_for_the_rotation_it_buys(seed):
    """
    Reachability, which is what separates this from a constructed curiosity: DE found
    the exploit on its own, at norm 0.003215756034478545 pre-fix.

    AN EARLIER VERSION OF THIS TEST ASSERTED THE WRONG THING, and it is recorded
    rather than quietly replaced because the mistake is the instructive part. It read
    `moved > 1e-6` — "post-fix, a collision must involve displacement" — and passed.
    It passed on seed 1 by luck. Sweeping seeds 0-11 showed seeds 6, 7 and 8 collide
    at frame 0 with EXACTLY zero displacement, and correctly so: apply() perturbs the
    initial VELOCITY, not the initial position, so at frame 0 the agent sits exactly
    where it was logged while carrying enough speed for its heading to be derived.
    The fix never promised rotation would be impossible. It promised rotation would be
    PAID FOR, and a test asserting the stronger thing was asserting something false.

    So this asserts the guarantee the design actually makes, with the bound DERIVED
    from the space's own weights rather than hardcoded — DE constants are not
    bit-portable (Batch 5) and a ratio against the pre-fix answer swings 6x to 71x
    across seeds, which would be just as fragile.

    THE BOUND. Rotation requires speed >= floor at the colliding frame. Speed can be
    bought with initial velocity (weight w_v) or with an acceleration bias acting over
    the horizon T (weight w_a), and combining them is cheaper than either alone
    because the norm adds in quadrature while speed adds linearly. Minimising
    ||W d|| subject to  dv + a*T = floor  is the distance from the origin to a line:

        min_norm = floor / sqrt((1/w_v)^2 + (T/w_a)^2)

    which is 0.5 / sqrt(4 + 0.81) = 0.227980 here. Measured: DE's cheapest answer over
    12 seeds was 0.228815, within 0.4% of that bound — it finds the optimum almost
    exactly. Pre-fix the most expensive answer over the same seeds was 0.042078, which
    is 0.185x the bound, so the separation is not marginal.
    """
    from src.optimization.scipy_optimizer import optimize_scenario

    states, validity, types = _stationary_cyclist()
    space = PerturbationSpace(states, validity, types, 0, 1)
    result = optimize_scenario(space, popsize=4, maxiter=20, seed=seed)

    assert result['collision'], (
        f'seed {seed}: the search found nothing at all, so this run is evidence about '
        'the budget rather than about the fix. Measured 12/12 seeds colliding at this '
        'budget post-fix; if that has changed, raise the budget rather than passing.'
    )

    w_v, w_a = float(space.weights[1]), float(space.weights[3])
    horizon = (states.shape[1] - 1) * float(space.dt)
    from src.physics.linear_model import V_HEADING_MIN
    min_norm = V_HEADING_MIN / np.sqrt((1.0 / w_v) ** 2 + (horizon / w_a) ** 2)

    assert result['min_perturbation'] >= 0.98 * min_norm, (
        f"seed {seed}: DE collided for weighted norm {result['min_perturbation']}, "
        f'below the {min_norm:.6f} that reaching the heading floor costs — so it '
        f'bought a rotation it did not pay for'
    )

    # ...and that the same search WITH THE FLOOR OFF still reproduces the exploit,
    # on this machine, so the comparison is against measured pre-fix behaviour rather
    # than against a remembered number.
    unguarded = PerturbationSpace(states, validity, types, 0, 1,
                                  heading_speed_floor=None)
    before = optimize_scenario(unguarded, popsize=4, maxiter=20, seed=seed)
    assert before['collision'] and before['min_perturbation'] < min_norm, (
        f'seed {seed}: with the floor off the pre-A01 exploit must still be findable '
        f'and must cost less than {min_norm:.6f}; got {before}'
    )


@pytest.mark.parametrize('logged_heading', [0.0, 0.7, np.pi / 2, 2.5, -1.2])
def test_A01b_the_zero_delta_replay_preserves_the_logged_heading(logged_heading):
    """
    The unreported half: the identity perturbation must reproduce the logged track,
    and heading is part of the logged track.

    Pre-fix every non-zero heading here came back as 0.0000 with
    baseline_replay_error reporting 0.0000 m, because the gate compares positions
    only. That is a Batch 1 / audit B03 violation in a column B03 never inspected.

    Laterally separated so the rotated footprint does not overlap the SDC — otherwise
    the fixture is refused by the replay-fidelity gate before it can be measured,
    which is itself correct behaviour and is pinned separately in
    tests/test_batch9_contract.py.
    """
    states, validity, types = _stationary_cyclist(heading=logged_heading, lateral=6.0)
    space = PerturbationSpace(states, validity, types, 0, 1)

    replayed = space.apply(np.zeros(4, dtype=np.float32))
    for frame in range(10):
        assert replayed[1, frame, 4] == pytest.approx(logged_heading, abs=1e-6), (
            f'frame {frame}: logged heading {logged_heading} replayed as '
            f'{float(replayed[1, frame, 4])} under a ZERO perturbation'
        )
    assert space.baseline_replay_error == pytest.approx(0.0, abs=1e-9)


# ── A02 / A12 / A05: scene identity ─────────────────────────────────────────────
#
# One gap, three exploitations. compute_stress_run_id identifies a PERTURBATION
# APPLIED TO A SCENE; nothing identifies the scene. Every guard this project has
# built — B14's join, R01's derived id, R03's single-read snapshot — compares
# identifiers that were never functions of scene content.
#
# Reproduced against the UNFIXED tree before any of this was designed:
#
#   A02(1) persist a pedestrian collision (delta [-1,0,0,0], frame 9), export, then
#          re-export the SAME scenario_id with the target shifted 10 m
#            -> "1 exported, stale=0, errors=0". Baseline frame 9 moved 3.2000 ->
#               13.2000, perturbed 2.3000 -> 12.3000, run id identical both times
#               (a0ee7366db2b7331), scenario_scores untouched. The API then serves a
#               frame-9 collision claim beside geometry 10 m away, every guard green.
#
#   A02(2) export a NEW scene + NEW delta, then attempt an export with the OLD pair
#            -> R01 correctly refused the perturbed path (attempted 626b25afde783cee
#               against current eb210794893f020a) while agents_written=2 had ALREADY
#               committed the old baseline: 3.7000 -> 3.2000. export_scenario_agents
#               calls conn.commit() in its own body and never reads scenario_scores,
#               and export_shard_geometry calls it FIRST — so R01's guard, which lives
#               inside export_perturbed_path, runs one commit too late and the
#               StaleExportError handler's rollback has nothing left to undo.
#
#   A12    the same wrong content (scene shifted 99 m) through both signatures
#            -> 8-arg REFUSED (059e486ac468f9f0 vs 1dc547834b85949c);
#               5-arg legacy PUBLISHED, path frame 9 x=102.2000, stamped with the
#               stored id so the B14/R01 read-side join PASSES. A live bypass.
#
#   A05    a bit-for-bit identical retry (same delta, method, target)
#            -> perturbed_paths rows 1 -> 0, stored run id unchanged
#               (6854242eaa4550ec). Working geometry destroyed for nothing.

requires_db = pytest.mark.skipif(
    os.environ.get('AV_CLAIMS_DB') != '1',
    reason='Requires an explicitly-nominated disposable database',
)

A02_SID = 'syn_scene_identity'
_RESULT = {'status': 'ok', 'outcome': 'collision_found', 'collision': True,
           'min_perturbation': 0.5, 'delta': [-1.0, 0.0, 0.0, 0.0],
           'collision_timestep': 9, 'target_idx': 1, 'method': 'de'}


def _pedestrian_scene(shift=0.0):
    """A vehicle SDC at the origin and a pedestrian walking in, optionally displaced."""
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 5:7] = [0.6, 0.6]
    states[1, :, 0] = 5.0 - 2.0 * np.arange(10) * 0.1 + shift
    states[1, :, 2] = -2.0
    states[1, :, 4] = np.pi
    return states, np.ones((2, 10), dtype=bool), np.array([1, 2])


def _fingerprint(states, validity, types):
    """
    The scene fingerprint, resolved only if it exists.

    RETURNS None AGAINST THE UNFIXED TREE, DELIBERATELY. These tests must be able to
    run before the fix or they prove nothing about it, and importing a function that
    does not exist yet fails with an ImportError — which demonstrates nothing about
    scene identity. Batch 7 made exactly this mistake with test_R01 (a TypeError about
    a new keyword, reported as reproducing a defect it never reached) and Batch 10
    made the same allowance for _reject_unstorable_text.

    Pre-fix the seeded row therefore carries no fingerprint and the export proceeds —
    which is the defect, and the assertions below are written against the OBSERVABLE
    damage (geometry that no longer matches the result it sits beside) rather than
    against the presence of a column.
    """
    from src.scoring import db
    fn = getattr(db, 'compute_scene_fingerprint', None)
    return None if fn is None else fn(states, validity, types)


def _export_perturbed(conn, sid, states, validity, target, **kw):
    """
    export_perturbed_path, with keyword arguments it does not yet accept dropped.

    Same reason _fingerprint resolves defensively, and the mistake was made here first:
    the initial version of these tests passed scene_fingerprint= unconditionally and
    six of them failed pre-fix with `TypeError: unexpected keyword argument`, which
    demonstrates nothing about scene identity. A test that cannot reach the code it
    names is not testing it, however red it is.
    """
    import inspect
    from src.scoring.export_geometry import export_perturbed_path
    accepted = inspect.signature(export_perturbed_path).parameters
    return export_perturbed_path(conn, sid, states, validity, target,
                                 **{k: v for k, v in kw.items() if k in accepted})


@pytest.fixture
def sconn():
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


def _seed_scene(conn, states, validity, types, sid=A02_SID, result=None):
    """Pass 1 + Pass 2, through the real write paths."""
    from src.scoring import db
    db.upsert_scores(conn, [dict(scenario_id=sid, shard='synthetic', n_agents=2,
                                 min_ttc=9.0, min_pet=9.0, fragility_score=1.0,
                                 scene_fingerprint=_fingerprint(states, validity, types))])
    db.update_stress_results(conn, {sid: dict(result or _RESULT)})


def _frame9_x(conn, table, sid=A02_SID, where=''):
    with conn.cursor() as cur:
        cur.execute(f"SELECT ST_X(ST_PointN(path, 10)) FROM {table} "
                    f"WHERE scenario_id = %s {where}", (sid,))
        row = cur.fetchone()
    return None if row is None else row[0]


@requires_db
def test_A02_a_reparsed_scene_cannot_overwrite_geometry_under_a_stable_id(sconn):
    """
    THE AUDIT'S REPRO 1, at the function the damage happens in.

    scenario_id is a stable natural key; the arrays behind it are not. A parser fix, a
    reinterpreted protobuf, or simply a different shard carrying the same id (
    upsert_scores conflicts on scenario_id alone and overwrites `shard`) produces a
    genuinely different scene under an unchanged key — and export_scenario_agents
    upserts it without reading scenario_scores at all.
    """
    from src.scoring.export_geometry import export_scenario_agents

    states, validity, types = _pedestrian_scene()
    _seed_scene(sconn, states, validity, types)
    export_scenario_agents(sconn, A02_SID, states, validity, types, 0)
    before = _frame9_x(sconn, 'scenario_agents', where='AND agent_idx = 1')

    moved, v2, t2 = _pedestrian_scene(shift=10.0)
    try:
        export_scenario_agents(sconn, A02_SID, moved, v2, t2, 0)
    except Exception as exc:
        assert type(exc).__name__ == 'SceneChangedError', f'wrong refusal: {exc!r}'
        sconn.rollback()

    after = _frame9_x(sconn, 'scenario_agents', where='AND agent_idx = 1')
    assert after == before, (
        f'geometry for {A02_SID} moved from {before} to {after} while the stored '
        f'result still describes the original scene — a re-parse overwrote a scene '
        f'under a stable id'
    )


@requires_db
def test_A02_baseline_geometry_is_not_committed_before_the_guard_runs(sconn):
    """
    THE AUDIT'S REPRO 2, and the ordering is the finding.

    R01's stale-export guard works — it refuses the perturbed path. But it lives
    inside export_perturbed_path, and export_shard_geometry commits the baseline
    first, so the refusal arrives after the damage. Measured pre-fix: perturbed
    correctly untouched at 1.9000 while the baseline moved 3.7000 -> 3.2000, leaving
    one committed database holding a baseline from one scene and a path from another.
    """
    from src.scoring.export_geometry import (
        export_scenario_agents, export_perturbed_path, StaleExportError,
    )

    new_states, nv, nt = _pedestrian_scene(shift=0.5)
    _seed_scene(sconn, new_states, nv, nt,
                result=dict(_RESULT, delta=[-2.0, 0.0, 0.0, 0.0], min_perturbation=1.0))
    export_scenario_agents(sconn, A02_SID, new_states, nv, nt, 0)
    _export_perturbed(sconn, A02_SID, new_states, nv, 1,
                      delta=[-2.0, 0.0, 0.0, 0.0], method='de',
                      scene_fingerprint=_fingerprint(new_states, nv, nt))
    base_before = _frame9_x(sconn, 'scenario_agents', where='AND agent_idx = 1')
    pert_before = _frame9_x(sconn, 'perturbed_paths')

    old_states, ov, ot = _pedestrian_scene(shift=0.0)
    try:
        export_scenario_agents(sconn, A02_SID, old_states, ov, ot, 0)
    except Exception as exc:
        assert type(exc).__name__ == 'SceneChangedError', f'wrong refusal: {exc!r}'
        sconn.rollback()
    try:
        _export_perturbed(sconn, A02_SID, old_states, ov, 1,
                          delta=_RESULT['delta'], method='de',
                          scene_fingerprint=_fingerprint(old_states, ov, ot))
    except (StaleExportError, Exception):
        sconn.rollback()

    assert _frame9_x(sconn, 'scenario_agents', where='AND agent_idx = 1') == base_before, (
        'the baseline was overwritten by an export whose perturbed half was refused — '
        'the guard ran one commit too late'
    )
    assert _frame9_x(sconn, 'perturbed_paths') == pert_before, 'the path moved too'


@requires_db
def test_A12_the_legacy_signature_cannot_publish_against_a_fingerprinted_scene(sconn):
    """
    Batch 7 kept the 5-argument signature for backward compatibility and warned in the
    docstring that the stale-export protection is "vacuous without delta/method". The
    audit shows that warning is a live bypass, not a caveat: the run id is read from
    scenario_scores and stamped onto whatever trajectory was handed in, so the WHERE
    clause compares a value against itself and always passes.

    Measured pre-fix with a scene displaced 99 m: the 8-arg call REFUSED and the 5-arg
    call PUBLISHED, stamped with the stored id, so every downstream join agreed.
    """
    from src.scoring.export_geometry import export_scenario_agents, export_perturbed_path

    states, validity, types = _pedestrian_scene()
    _seed_scene(sconn, states, validity, types)
    export_scenario_agents(sconn, A02_SID, states, validity, types, 0)

    wrong, wv, _ = _pedestrian_scene(shift=99.0)
    published = None
    try:
        published = export_perturbed_path(sconn, A02_SID, wrong, wv, 1)
    except Exception as exc:
        assert type(exc).__name__ in ('SceneChangedError', 'StaleExportError'), (
            f'wrong refusal: {exc!r}'
        )
        sconn.rollback()
        published = False

    assert not published, (
        'the 5-argument legacy call published geometry built from a scene 99 m away '
        'from the one the stored result describes'
    )
    assert _frame9_x(sconn, 'perturbed_paths') is None, (
        'a perturbed path from the wrong scene reached the database'
    )


@requires_db
def test_A12_a_row_without_a_fingerprint_still_exports(sconn):
    """
    THE CARVE-OUT, AND THE FIX IS WORTHLESS WITHOUT IT.

    Rows written before this column existed have scene_fingerprint NULL, which means
    "not recorded" and is not evidence of a mismatch — the same rule B14 applies to
    stress_run_id and routes.py applies with IS NOT DISTINCT FROM. A fix that refused
    these would be a fail-closed regression on every legacy row, which this project
    has repeatedly said is worse than the defect it replaces.

    This is also what keeps the audit's own B13/B14 fixtures passing unmodified: they
    seed scenario_scores by hand and carry no fingerprint.
    """
    from src.scoring import db
    from src.scoring.export_geometry import export_scenario_agents, export_perturbed_path

    states, validity, types = _pedestrian_scene()
    db.upsert_scores(sconn, [dict(scenario_id=A02_SID, shard='synthetic', n_agents=2,
                                  min_ttc=9.0, min_pet=9.0, fragility_score=1.0)])
    db.update_stress_results(sconn, {A02_SID: dict(_RESULT)})
    # Tolerant of the column not existing yet, for the reason _fingerprint and
    # _export_perturbed are: this test asserts a behaviour that ALREADY HOLDS today
    # and must keep holding, so it has to be runnable on both sides of the fix. An
    # UndefinedColumn error here would be an infrastructure failure wearing the
    # costume of a finding.
    with sconn.cursor() as cur:
        cur.execute("SELECT to_regclass('scenario_scores') IS NOT NULL")
        cur.execute("""SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'scenario_scores'
                          AND column_name = 'scene_fingerprint'""")
        has_column = cur.fetchone() is not None
        if has_column:
            cur.execute("SELECT scene_fingerprint FROM scenario_scores "
                        "WHERE scenario_id = %s", (A02_SID,))
            assert cur.fetchone()[0] is None, (
                'fixture regressed: this row must carry no fingerprint'
            )

    export_scenario_agents(sconn, A02_SID, states, validity, types, 0)
    assert export_perturbed_path(sconn, A02_SID, states, validity, 1) is True, (
        'a legacy row with no recorded fingerprint was refused — the carve-out closed'
    )


@requires_db
def test_A05_an_identical_retry_keeps_its_geometry(sconn):
    """
    update_stress_results deletes perturbed_paths whenever a pass produced a result,
    without ever comparing the new identity against the stored one. A bit-for-bit
    identical retry — same delta, same method, same target, therefore the same
    stress_run_id by construction — destroys working geometry for nothing.
    """
    from src.scoring import db
    from src.scoring.export_geometry import export_scenario_agents, export_perturbed_path

    states, validity, types = _pedestrian_scene()
    _seed_scene(sconn, states, validity, types)
    export_scenario_agents(sconn, A02_SID, states, validity, types, 0)
    _export_perturbed(sconn, A02_SID, states, validity, 1,
                      delta=_RESULT['delta'], method='de',
                      scene_fingerprint=_fingerprint(states, validity, types))
    before = _frame9_x(sconn, 'perturbed_paths')
    assert before is not None, 'fixture regressed: nothing was exported'

    db.update_stress_results(sconn, {A02_SID: dict(_RESULT)})

    assert _frame9_x(sconn, 'perturbed_paths') == before, (
        'a bit-for-bit identical retry destroyed the exported geometry'
    )


@requires_db
def test_A05_a_genuinely_new_result_still_invalidates_geometry(sconn):
    """
    THE HOLE THE A05 FIX MUST NOT OPEN.

    B14's invalidation exists because a path exported from the PREVIOUS delta must not
    survive a new result. Making the delete conditional is only correct if the
    condition still fires whenever the identity actually changed.
    """
    from src.scoring import db
    from src.scoring.export_geometry import export_scenario_agents, export_perturbed_path

    states, validity, types = _pedestrian_scene()
    _seed_scene(sconn, states, validity, types)
    export_scenario_agents(sconn, A02_SID, states, validity, types, 0)
    _export_perturbed(sconn, A02_SID, states, validity, 1,
                      delta=_RESULT['delta'], method='de',
                      scene_fingerprint=_fingerprint(states, validity, types))
    assert _frame9_x(sconn, 'perturbed_paths') is not None, 'fixture regressed'

    db.update_stress_results(sconn, {A02_SID: dict(_RESULT, delta=[-2.0, 0.0, 0.0, 0.0],
                                                   min_perturbation=1.0)})

    assert _frame9_x(sconn, 'perturbed_paths') is None, (
        'a NEW delta left the old geometry in place — the A05 fix reopened B14'
    )


# ── A03 / A06 / A07: what survives persistence when things go wrong ─────────────
#
# A looser family than A02/A05/A12 were, and the file says so rather than inventing a
# tighter story: three findings about what update_stress_results and _stress_one keep
# when a result is malformed, generic, or superseded internally.
#
# Reproduced against the UNFIXED tree before any of it was designed:
#
#   A03  json.dumps(float('inf')) emits the non-standard token Infinity, which
#        PostgreSQL's JSONB parser correctly rejects:
#          psycopg2.errors.InvalidTextRepresentation: invalid input syntax for type json
#        AND — the part that makes it more than one lost row — update_stress_results
#        commits ONCE after its loop, so three results in one call all came back
#        last_attempt_outcome=None.
#
#        THAT THE ALREADY-WRITTEN ROW WAS DISCARDED, RATHER THAN NEVER REACHED, IS A
#        SEPARATE MEASUREMENT AND NOT SOMETHING THE TEST BELOW CAN SHOW. A statement
#        error aborts the transaction, so the test MUST roll back before it can query
#        anything — and a rollback reverts to the last commit, which makes "its UPDATE
#        ran and was undone" and "its UPDATE never ran" observationally identical.
#        Replicating the pre-fix loop by hand and observing INSIDE the transaction:
#
#            good_a UPDATE rowcount = 1            <- it really executed
#            good_a IN-TRANSACTION value = 'no_challenger'
#            bad UPDATE raised InvalidTextRepresentation
#            good_a AFTER rollback = None          <- the work was discarded
#
#        Recorded here rather than asserted there, because a docstring on a test
#        should claim what that test demonstrates.
#
#   A06  a target with zero length/width raises out of _stress_one
#        ("ValueError: agent 1 has unusable length=0.0 at its first valid frame 0"),
#        stress_test_scenarios turns it into {'status':'error','error': <fused str>},
#        and the row persists as last_attempt_outcome='error' with
#        last_attempt_diagnostics=None. The outcome is recorded; the reason is not.
#
#   A07  _stress_one(..., use_autograd=True, de_kwargs={'popsize':4,'maxiter':0,
#        'seed':0}) on two vehicles 2.1 m apart: refinement wins, `result = refined`
#        replaces the dict, and the provenance built afterwards reads DE's keys off a
#        dict that no longer has them — de_n_iter=None, de_n_eval=None.
#
# THE AUDIT'S OWN A03 TRIGGER DOES NOT FIRE, and this is recorded because the next
# reader will otherwise re-derive it. _measure_baseline_replay does set
# error = float('inf') when the rollout is non-finite (confirmed at drift >= 1e22 with
# the collision check neutralised) — but it then calls check_collision_trajectory on
# that same non-finite trajectory two lines later, and Shapely raises GEOSException
# before the function can return. The condition that sets inf is the condition that
# breaks the collision check, so inf never escapes as a ReplayFidelityError. A blown-up
# rollout lands in status='error' instead, which is A06's defect — and is the trigger
# test_A06_a_blown_up_rollout_is_diagnosable pins.

def _ped_error_scene():
    """A target with zero length/width — B02's fail-loud path, A06's own fixture."""
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[0, :, 5:7] = [4.5, 2.0]
    states[1, :, 0] = 8.0
    states[1, :, 5:7] = [0.0, 0.0]
    return states, np.ones((2, 10), dtype=bool), np.array([1, 1])


def _blown_rollout_scene(vx=1e22):
    """
    Pickable AND explosive: the challenger sits next to the SDC so
    pick_nearest_challenger selects it, then moves at a speed that overflows the
    rollout. An earlier version put it far away and got no_challenger instead — the
    scene never reached the code it was written to exercise.
    """
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[1, :, 0] = 8.0 + np.arange(10) * 0.1 * vx
    states[1, :, 2] = vx
    return states, np.ones((2, 10), dtype=bool), np.array([1, 1])


def _write_shard(path, payloads):
    import struct
    from src.data.loader import _masked_crc32c
    blob = b''
    for payload in payloads:
        header = struct.pack('<Q', len(payload))
        blob += (header + struct.pack('<I', _masked_crc32c(header))
                 + payload + struct.pack('<I', _masked_crc32c(payload)))
    path.write_bytes(blob)
    return str(path)


def _install_parser(scenes, monkeypatch):
    """
    Substitute src.data.parser, the technique Batch 2's B05 substitutes established.

    monkeypatch.setitem, NOT A BARE ASSIGNMENT (fix F06). The two callers below used
    to do `sys.modules['src.data.parser'] = module` directly and never undo it —
    Python caches modules by name, so the substitution outlived both tests for the
    rest of the process. Any later test in the same invocation that imports
    src.data.parser, directly or through batch_scorer's own lazy import, got this
    stub back instead of the real module. monkeypatch is a required parameter, not
    optional, so a future third caller cannot reintroduce the leak by forgetting to
    pass one — omitting it is a TypeError at the call site, not a silent leak.
    """
    import types as _types
    module = _types.ModuleType('src.data.parser')
    # Marker for test_F06_the_parser_stub_does_not_leak_past_the_A06_tests_above,
    # which needs to recognize ONE OF THIS FUNCTION'S OWN STUBS specifically —
    # importing the real src.data.parser to compare against is not an option here,
    # since it raises ModuleNotFoundError without the waymo package installed.
    module._is_test_audit3_parser_stub = True

    class ScenarioParser:
        def __init__(self, raw):
            self.sid = raw.decode()

        def get_scenario_id(self):
            return self.sid

        def get_agent_states(self):
            return scenes[self.sid][0]

        def get_agent_validity(self):
            return scenes[self.sid][1]

        def get_agent_types(self):
            return scenes[self.sid][2]

        def get_sdc_index(self):
            return 0

    module.ScenarioParser = ScenarioParser
    monkeypatch.setitem(sys.modules, 'src.data.parser', module)
    return module


_INF_REFUSAL = {'status': 'replay_infeasible', 'outcome': 'replay_infeasible',
                'target_idx': 1, 'baseline_replay_error': float('inf'),
                'baseline_replay_collides': False, 'reason': 'drift',
                'challengers_total': 2, 'challengers_searched': 0}
_NO_CHALLENGER = {'status': 'no_challenger', 'outcome': 'no_challenger',
                  'challengers_total': 0, 'challengers_searched': 0}


@pytest.fixture
def pconn():
    from src.scoring import db
    connection = db.get_connection()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, '
                    'scenario_scores CASCADE')
    connection.commit()
    db.init_schema(connection)
    yield connection
    connection.rollback()
    connection.close()


@requires_db
def test_A03_a_non_finite_diagnostic_does_not_kill_the_batch(pconn):
    """
    THE BLAST RADIUS: A FAILURE ANYWHERE IN THE CALL DISCARDS EVERY ROW'S OUTCOME
    FROM THAT SAME CALL, VALID OR NOT, REGARDLESS OF PROCESSING ORDER.

    That is exactly what this test demonstrates, and deliberately no more. The
    `except: rollback()` below is not cleanup — it is structurally required, because a
    statement error aborts the transaction and every later query, including the
    fetch_scenario calls the assertions depend on, would raise
    InFailedSqlTransaction without it. So the state observed here is POST-rollback,
    which cannot separate "good_a's UPDATE ran and was undone" from "good_a was never
    reached": both leave the row at its seeded value. The ordering claim is true and
    was measured separately — see the file header — but not by this.

    What this DOES discriminate is the thing that matters: post-fix each row persists
    its own outcome through its own savepoint regardless of its neighbours, and
    pre-fix none of them do. R02's rule — one bad record must never kill the batch —
    applied to the stage that never got it. Pass 1 and Pass 2 got it in Batch 5; Pass 3
    has had it from the start; persistence, the one loop whose failure discards work
    that has already been COMPUTED, had none.
    """
    from src.scoring import db

    db.upsert_scores(pconn, [dict(scenario_id=s, shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0)
                             for s in ('good_a', 'bad', 'good_b')])

    results = {'good_a': dict(_NO_CHALLENGER),
               'bad': dict(_INF_REFUSAL),
               'good_b': dict(_NO_CHALLENGER)}
    assert list(results) == ['good_a', 'bad', 'good_b'], 'ordering is the fixture'

    try:
        db.update_stress_results(pconn, results)
    except Exception:
        pconn.rollback()

    assert db.fetch_scenario(pconn, 'good_a')['last_attempt_outcome'] == 'no_challenger', (
        'a valid result that had already been written was rolled back by an unrelated '
        "row's serialization failure"
    )
    assert db.fetch_scenario(pconn, 'good_b')['last_attempt_outcome'] == 'no_challenger', (
        'a valid result after the bad one was never written'
    )


@requires_db
def test_A03_a_non_finite_value_is_recorded_as_such_not_dropped(pconn):
    """
    NULL, PLUS A RECORD OF WHY — because null already means something else.

    Project doctrine is that JSON has no infinity literal, so the value cannot persist
    as a number (Block 6 Concept 24, the same rule min_perturbation follows). But a
    bare null is indistinguishable from "this attempt recorded no such diagnostic",
    which _attempt_diagnostics' own docstring is careful about. So the field goes null
    AND one extra key names what was dropped and which non-finite value it held.
    """
    from src.scoring import db

    db.upsert_scores(pconn, [dict(scenario_id='bad', shard='x', n_agents=2,
                                  min_ttc=9.0, min_pet=9.0, fragility_score=1.0)])
    try:
        db.update_stress_results(pconn, {'bad': dict(_INF_REFUSAL)})
    except Exception:
        pconn.rollback()

    row = db.fetch_scenario(pconn, 'bad')
    assert row['last_attempt_outcome'] == 'replay_infeasible', 'the row was not written'
    diagnostics = row['last_attempt_diagnostics']
    assert diagnostics is not None, 'the refusal persisted no diagnostics at all'
    assert diagnostics['reason'] == 'drift', 'the finite fields must survive intact'
    assert diagnostics['baseline_replay_error'] is None, (
        'a non-finite value was stored as a number'
    )
    assert diagnostics.get('nonfinite_fields') == {'baseline_replay_error': 'inf'}, (
        'the null is indistinguishable from "not recorded": '
        f'{diagnostics.get("nonfinite_fields")!r}'
    )


@requires_db
def test_A06_a_generic_exception_keeps_its_type_and_message(pconn, tmp_path, monkeypatch):
    """
    _ATTEMPT_DIAGNOSTIC_FIELDS was built around what ReplayFidelityError carries, so
    the 'error' outcome — the catch-all for everything unexpected — had nothing in the
    allowlist and was silently dropped the moment the in-memory dict went out of scope.
    The row said 'error' and could not say why.

    TYPE AND MESSAGE SEPARATELY, because today they are neither separate nor absent:
    stress_test_scenarios builds f'{type(e).__name__}: {e}', fusing both into one
    string that the notebook then takes apart again with .split(':')[0].
    """
    from src.scoring import db
    from src.scoring.batch_scorer import stress_test_scenarios

    _install_parser({'errscene': _ped_error_scene()}, monkeypatch)
    shard = _write_shard(tmp_path / 'err.tfrecord', [b'errscene'])
    results = stress_test_scenarios(shard, ['errscene'], verbose=False)
    assert results['errscene']['status'] == 'error', 'fixture regressed'

    db.upsert_scores(pconn, [dict(scenario_id='errscene', shard='x', n_agents=2,
                                  min_ttc=9.0, min_pet=9.0, fragility_score=1.0)])
    db.update_stress_results(pconn, results)

    row = db.fetch_scenario(pconn, 'errscene')
    assert row['last_attempt_outcome'] == 'error'
    diagnostics = row['last_attempt_diagnostics']
    assert diagnostics is not None, (
        'the outcome was recorded and the reason was not — the row says "error" and '
        'cannot say why'
    )
    assert diagnostics['error_type'] == 'ValueError', diagnostics
    assert 'unusable length' in diagnostics['error_message'], diagnostics
    assert ':' not in diagnostics['error_type'], (
        'the type still carries the fused message'
    )


@requires_db
def test_A06_a_blown_up_rollout_is_diagnosable(pconn, tmp_path, monkeypatch):
    """
    THE TRIGGER FOUND WHILE DISPROVING A03'S, AND IT BELONGS HERE.

    _measure_baseline_replay sets error = float('inf') for a non-finite rollout, but
    calls check_collision_trajectory on that same trajectory before returning, and
    Shapely raises GEOSException first. So the audit's A03 narrative never fires: a
    blown-up rollout arrives as status='error', whose message A06 is about.

    GEOSException subclasses Exception (MRO checked), so R02's guarantee holds and the
    batch survives — asserted here too, since "one bad record must never kill the
    batch" is the property that makes this merely undiagnosable rather than fatal.
    """
    from src.scoring import db
    from src.scoring.batch_scorer import stress_test_scenarios

    fine = np.zeros((2, 10, 7), dtype=np.float32)
    fine[:, :, 5:7] = [4.5, 2.0]
    fine[1, :, 0] = 8.0
    _install_parser({'blown': _blown_rollout_scene(),
                     'fine': (fine, np.ones((2, 10), dtype=bool), np.array([1, 1]))},
                    monkeypatch)
    shard = _write_shard(tmp_path / 'blown.tfrecord', [b'blown', b'fine'])

    results = stress_test_scenarios(shard, ['blown', 'fine'], verbose=False,
                                    de_kwargs={'popsize': 4, 'maxiter': 1, 'seed': 0})
    assert results['blown']['status'] == 'error', 'fixture regressed'
    assert results['fine']['status'] != 'error', (
        'the pathological scenario took its neighbour down — R02 is broken'
    )

    db.upsert_scores(pconn, [dict(scenario_id=s, shard='x', n_agents=2, min_ttc=9.0,
                                  min_pet=9.0, fragility_score=1.0)
                             for s in ('blown', 'fine')])
    db.update_stress_results(pconn, results)

    diagnostics = db.fetch_scenario(pconn, 'blown')['last_attempt_diagnostics']
    assert diagnostics is not None, 'a blown-up rollout persisted no reason'
    assert diagnostics['error_type'] == 'GEOSException', diagnostics


# ── fix F06: _install_parser's sys.modules substitution used to outlive its test ──
#
# sys.modules['src.data.parser'] = module, with no teardown, called directly from
# the two test_A06_* tests above. Python caches modules by name, so the stub sat in
# sys.modules for the rest of the process — any LATER test in the same invocation
# that imported src.data.parser, directly or through batch_scorer's own lazy
# import, got the stub back instead of the real module.
#
# Two checks, neither alone sufficient. (1) is fast and needs no waymo package:
# it confirms the teardown mechanism itself fires, self-contained AND (matching how
# the leak itself was originally found) relying on the two real tests above having
# already run in file order. (2) is the actual real-world claim — the two tests the
# leak was shown to break live in a DIFFERENT file (test_audit_core.py) and need the
# real waymo_open_dataset package, which is not installed in this dev environment
# (the same gap visible in every full-suite run of this project). See that test's
# own docstring for exactly what it can and cannot show here.

# Set in the CHILD's environment by test_F06_the_two_named_victims_...'s own
# subprocess call, below, so that test can recognize "I am the subprocess I myself
# spawned" and refuse to spawn a further one — the guard that survives even if that
# test's --deselect flag is ever edited out. See its docstring for why this exists.
_F06_SUBPROCESS_GUARD_ENV = 'AV_F06_SUBPROCESS_ALREADY_RUNNING'


def _pytest_outcomes(output, node_ids):
    """
    Map each of `node_ids` to 'PASSED', 'FAILED', or None (neither line found),
    reading pytest's own short-test-summary lines out of combined stdout+stderr
    from a subprocess run made with `-rpf` (report passed+failed).

    A SEPARATE, UNIT-TESTABLE FUNCTION rather than inlined into
    test_F06_the_two_named_victims_fail_only_on_missing_waymo_not_the_leak
    itself (independent review, 2026-09-25) — that test's own
    waymo-open-dataset-installed branch cannot be exercised end to end on this
    machine (the same manylinux/TensorFlow constraint documented since Block
    1), so the one thing actually checkable here is that the MATCHING LOGIC is
    correct — see the standalone tests just below, run against hand-built
    strings shaped like a real run's output, not a live subprocess.

    Matches on 'PASSED <node_id>' / 'FAILED <node_id>' as literal substrings.
    Deliberately NOT a per-line regex anchored at line start: pytest's `-rpf`
    summary lines are consistently `f'{OUTCOME} {nodeid}'` with nothing before
    the outcome word on that line, so a substring match is exactly as precise
    here and needs no assumption about surrounding whitespace or platform line
    endings.
    """
    outcomes = {}
    for node_id in node_ids:
        if f'PASSED {node_id}' in output:
            outcomes[node_id] = 'PASSED'
        elif f'FAILED {node_id}' in output:
            outcomes[node_id] = 'FAILED'
        else:
            outcomes[node_id] = None
    return outcomes


def test__pytest_outcomes_reads_passed_lines_from_a_real_shaped_report():
    """
    Hand-built, shaped like the actual `-q -rpf` short-test-summary section a
    genuine waymo-open-dataset-installed, leak-free run would produce (per this
    file's own test_F06 docstring: 'both tests exercise the REAL ScenarioParser
    and either pass on their own merits...'). Includes OTHER tests' PASSED
    lines too, matching a real ~40-test-file run, to prove this does not just
    detect "the word PASSED appeared somewhere".
    """
    fake_output = (
        '....................................                              [100%]\n'
        '=========================== short test summary info ===========================\n'
        'PASSED tests/test_audit3_regressions.py::test_A07_de_provenance_survives_an_accepted_refinement\n'
        'PASSED tests/test_audit_core.py::test_B05_corrupt_next_record_does_not_overwrite_previous_success\n'
        'PASSED tests/test_audit_core.py::test_control_parser_preserves_valid_states_and_flags\n'
        '38 passed in 12.34s\n'
    )
    outcomes = _pytest_outcomes(fake_output, (
        'tests/test_audit_core.py::test_B05_corrupt_next_record_does_not_overwrite_previous_success',
        'tests/test_audit_core.py::test_control_parser_preserves_valid_states_and_flags',
    ))
    assert outcomes == {
        'tests/test_audit_core.py::test_B05_corrupt_next_record_does_not_overwrite_previous_success': 'PASSED',
        'tests/test_audit_core.py::test_control_parser_preserves_valid_states_and_flags': 'PASSED',
    }


def test__pytest_outcomes_reads_failed_lines_from_a_real_shaped_report():
    """Hand-built, shaped like this machine's own actual output (waymo absent)."""
    fake_output = (
        'FF..................................                              [100%]\n'
        '=========================== short test summary info ===========================\n'
        'FAILED tests/test_audit_core.py::test_B05_corrupt_next_record_does_not_overwrite_previous_success\n'
        'FAILED tests/test_audit_core.py::test_control_parser_preserves_valid_states_and_flags\n'
        '2 failed, 36 passed in 5.67s\n'
    )
    outcomes = _pytest_outcomes(fake_output, (
        'tests/test_audit_core.py::test_B05_corrupt_next_record_does_not_overwrite_previous_success',
        'tests/test_audit_core.py::test_control_parser_preserves_valid_states_and_flags',
    ))
    assert outcomes == {
        'tests/test_audit_core.py::test_B05_corrupt_next_record_does_not_overwrite_previous_success': 'FAILED',
        'tests/test_audit_core.py::test_control_parser_preserves_valid_states_and_flags': 'FAILED',
    }


def test__pytest_outcomes_distinguishes_per_node_id_not_just_any_match():
    """
    THE CASE THAT WOULD CATCH A SLOPPY IMPLEMENTATION: one of the two named
    tests passed, the other failed — each of THOSE outcomes must be attributed
    to the RIGHT node id, not to "PASSED appeared somewhere so both must have".
    """
    fake_output = (
        '=========================== short test summary info ===========================\n'
        'FAILED tests/test_audit_core.py::test_B05_corrupt_next_record_does_not_overwrite_previous_success\n'
        'PASSED tests/test_audit_core.py::test_control_parser_preserves_valid_states_and_flags\n'
        '1 failed, 1 passed in 1.0s\n'
    )
    outcomes = _pytest_outcomes(fake_output, (
        'tests/test_audit_core.py::test_B05_corrupt_next_record_does_not_overwrite_previous_success',
        'tests/test_audit_core.py::test_control_parser_preserves_valid_states_and_flags',
    ))
    assert outcomes['tests/test_audit_core.py::test_B05_corrupt_next_record_does_not_overwrite_previous_success'] == 'FAILED'
    assert outcomes['tests/test_audit_core.py::test_control_parser_preserves_valid_states_and_flags'] == 'PASSED'


def test__pytest_outcomes_returns_none_when_neither_line_appears():
    """
    THE DEFENSIVE CASE: a collection error, a timeout, or any other shape of
    subprocess failure that never reaches either named test at all must not be
    silently misread as a pass or a fail for either of them.
    """
    fake_output = (
        'ERROR collecting tests/test_audit3_regressions.py\n'
        '1 error in 0.5s\n'
    )
    outcomes = _pytest_outcomes(fake_output, (
        'tests/test_audit_core.py::test_B05_corrupt_next_record_does_not_overwrite_previous_success',
        'tests/test_audit_core.py::test_control_parser_preserves_valid_states_and_flags',
    ))
    assert outcomes == {
        'tests/test_audit_core.py::test_B05_corrupt_next_record_does_not_overwrite_previous_success': None,
        'tests/test_audit_core.py::test_control_parser_preserves_valid_states_and_flags': None,
    }


def test_F06_install_parser_reverts_when_its_monkeypatch_context_exits():
    """
    SELF-CONTAINED, ORDER-INDEPENDENT. pytest.MonkeyPatch is the same class the
    `monkeypatch` fixture wraps, used here directly as a context manager so the
    revert is observed within one test body rather than inferred from what some
    other test leaves behind.
    """
    before = sys.modules.get('src.data.parser')
    with pytest.MonkeyPatch.context() as mp:
        stub = _install_parser({'x': _ped_error_scene()}, mp)
        assert sys.modules['src.data.parser'] is stub, 'fixture regressed: not installed'
    assert sys.modules.get('src.data.parser') is before, (
        'the substitution outlived the monkeypatch context that installed it — '
        f'sys.modules["src.data.parser"] is still {sys.modules.get("src.data.parser")!r}'
    )


def test_F06_the_parser_stub_does_not_leak_past_the_A06_tests_above():
    """
    ORDER-DEPENDENT, DELIBERATELY — matching how the leak itself was found. Relies
    on default pytest execution order: test_A06_a_generic_exception_keeps_its_type_
    and_message and test_A06_a_blown_up_rollout_is_diagnosable, both above, have
    already run by the time this test does, each installing its own stub via
    _install_parser(..., monkeypatch). If either failed to revert, sys.modules would
    still hold one of THOSE specific stubs.

    Compared against a marker attribute, not against "the real module" — importing
    the real src.data.parser here to compare identity against would itself raise
    ModuleNotFoundError without the waymo package, which would make this test fail
    for an unrelated reason on every machine in this project's Colab/local split.
    """
    current = sys.modules.get('src.data.parser')
    assert not getattr(current, '_is_test_audit3_parser_stub', False), (
        f'a parser stub installed by an earlier test_A06_* test leaked past its own '
        f'test: sys.modules["src.data.parser"] is still {current!r}'
    )


@requires_db
def test_F06_the_two_named_victims_fail_only_on_missing_waymo_not_the_leak(tmp_path):
    """
    THE REAL-WORLD CLAIM, subprocess-based — the same technique
    test_audit2_regressions.py's _pytest_run helper uses for the identical shape of
    problem ("the defect is at MODULE IMPORT... os.environ cannot be un-read"; here
    it is sys.modules that cannot be un-imported). @requires_db because the two
    test_A06_* tests below are themselves @requires_db and would simply skip without
    a reachable disposable database in the subprocess's environment — proving
    nothing about the leak either way.

    File order on the command line puts test_audit3_regressions.py (and both
    A06 tests) first, then the two named victims in test_audit_core.py.

    WHAT THIS TEST CAN AND CANNOT SHOW ON THIS MACHINE, STATED PLAINLY RATHER THAN
    ASSUMED. Both target tests import waymo_open_dataset.protos.scenario_pb2
    directly and unconditionally, on the line immediately after touching
    src.data.parser. This was verified by running this exact subprocess invocation
    twice, once against the fixed _install_parser and once against the pre-fix bare
    `sys.modules[...] = module`. The two assertions below — exception TYPE and
    MESSAGE identical, no UnicodeDecodeError or KeyError anywhere — held in both
    runs. The raw traceback SHAPE was not identical, and that is worth recording
    rather than glossing over: pre-fix (leak present), `import src.data.parser`
    retrieves the cached stub silently and the failure surfaces one line later, on
    the SEPARATE, unconditional `from waymo_open_dataset...` import — a single-frame
    traceback. Post-fix, `import src.data.parser` has nothing cached and genuinely
    attempts the real module, which fails one frame deeper, inside
    src/data/parser.py's own top-level import — an extra frame, at a different line
    number. Both collapse to the same exception type and message either way, which
    is exactly what the two assertions below check — not a full-output comparison,
    deliberately, since a full-output comparison would be fragile against exactly
    this harmless frame-depth difference. So this test does NOT currently
    discriminate the fix from its absence on the ultimate pass/fail outcome — it is
    not a no-op, it guards against a DIFFERENT regression (something else making
    these two tests fail some other way, or the leak actually reaching the stub),
    and it is the closest thing to the real claim that is checkable without the
    waymo package.

    G05 (independent review, 2026-09-25): the assertion below used to be
    unconditional — `output.count(...) == 2` with no branch for an environment
    where waymo-open-dataset IS installed, where both named tests would run for
    real, pass, and make that count 0 — failing this test precisely when the
    parser fix it exists to validate is working correctly. Branched now on
    `importlib.util.find_spec('waymo_open_dataset')`, checked, not imported
    (confirmed live: leaves sys.modules untouched, which matters in a file
    already about what leaks into it). THE WAYMO-INSTALLED BRANCH REMAINS
    UNVERIFIABLE END TO END ON THIS MACHINE — same manylinux/TensorFlow
    constraint this project has documented since Block 1, not something to work
    around here. What IS checkable, and is checked, in
    test__pytest_outcomes_reads_passed_lines_from_a_real_shaped_report and its
    three siblings just above this test: the MATCHING LOGIC that branch depends
    on (_pytest_outcomes, reading 'PASSED <nodeid>'/'FAILED <nodeid>' lines out
    of a `-rpf` report) is correct, exercised against a hand-built string shaped
    like what a genuine waymo-installed passing run would produce. That proves
    the parsing code is not broken; it does not prove the real package actually
    behaves as this docstring predicts — those are different claims, and only
    the first is reachable here.

    PER-TEST OUTCOME, NOT A BARE proc.returncode (also independent review,
    2026-09-25, raised alongside the branching above). The command line below
    runs the WHOLE of test_audit3_regressions.py (minus this test) ahead of the
    two named victims — dozens of unrelated tests in the same subprocess. A
    bare `proc.returncode == 0` check would be wrong in both directions: it
    would fail this test over some OTHER, unrelated test breaking elsewhere in
    that file, and it still could not say WHICH of the two named tests actually
    passed or failed, only that something in the whole run did. `-rpf` plus
    _pytest_outcomes checks the two node IDs that actually matter, by name,
    independent of anything else the subprocess ran.

    A GUARD AGAINST THIS TEST RECURSING INTO ITSELF, AND IT IS NOT DECORATIVE. This
    test lives inside test_audit3_regressions.py, and the subprocess command below
    names that whole file — which, without the `--deselect` below, would re-collect
    and re-run THIS VERY TEST, spawning another subprocess that does the same thing
    again, unbounded. Hit for real during development: leaving this unguarded
    produced roughly 150 stray pytest processes in under two minutes before being
    killed by hand. `--deselect` is the primary defence; the environment-variable
    check right below is the one that survives if that flag is ever edited out, and
    it propagates to any depth because it is set in the CHILD's environment too.

    What a run WITH waymo-open-dataset installed would show instead:
      pre-fix  (leak present): test_control_parser_preserves_valid_states_and_flags
               would get the LEAKED STUB's ScenarioParser back from
               `from src.data.parser import ScenarioParser`, whose __init__ does
               `self.sid = raw.decode()` on arbitrary serialized-protobuf bytes —
               very likely UnicodeDecodeError, or (if those particular bytes happen
               to decode) KeyError inside get_agent_states(), since
               'parser_control' was never registered with any _install_parser call.
               test_B05_corrupt_next_record_does_not_overwrite_previous_success
               would hit the same failure inside stress_test_scenarios's own lazy
               `from src.data.parser import ScenarioParser`; its per-record
               `except Exception` (R02's isolation) would catch it and never
               populate results['A'], so the test's own
               `assert result['A']['status'] == 'ok'` would fail with
               KeyError: 'A' — not the fixture-regression message it is meant to
               report.
      post-fix (leak closed): both tests exercise the REAL ScenarioParser and
               either pass on their own merits or fail on whatever they were
               actually written to catch — not an artifact of an earlier, unrelated
               test.
    """
    import importlib.util
    import subprocess

    if os.environ.get(_F06_SUBPROCESS_GUARD_ENV) == '1':
        pytest.skip('running inside the subprocess this test itself spawned')

    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env[_F06_SUBPROCESS_GUARD_ENV] = '1'
    this_test_id = (
        'tests/test_audit3_regressions.py::'
        'test_F06_the_two_named_victims_fail_only_on_missing_waymo_not_the_leak'
    )
    victim_ids = (
        'tests/test_audit_core.py::test_B05_corrupt_next_record_does_not_overwrite_previous_success',
        'tests/test_audit_core.py::test_control_parser_preserves_valid_states_and_flags',
    )
    proc = subprocess.run(
        [sys.executable, '-m', 'pytest',
         'tests/test_audit3_regressions.py',
         '--deselect', this_test_id,
         *victim_ids,
         '-q', '-rpf', '-p', 'no:cacheprovider'],
        cwd=project, env=env, capture_output=True, text=True, timeout=300,
    )
    output = proc.stdout + proc.stderr
    outcomes = _pytest_outcomes(output, victim_ids)

    # THE LEAK REACHING AN UNRELATED TEST, CHECKED REGARDLESS OF WHICH BRANCH
    # BELOW APPLIES — this is the actual regression this whole test exists to
    # catch, and it is exactly as meaningful whether or not waymo is installed
    # here.
    assert 'UnicodeDecodeError' not in output, (
        f'the leak reached a test in another file:\n{output[-3000:]}'
    )
    assert 'KeyError' not in output, (
        f'the leak reached a test in another file:\n{output[-3000:]}'
    )

    if importlib.util.find_spec('waymo_open_dataset') is not None:
        # WAYMO IS INSTALLED HERE (fix G05). Neither named test's missing-
        # dependency failure applies — per this docstring's own "what a run
        # WITH waymo-open-dataset installed would show instead", post-fix both
        # exercise the real ScenarioParser and must pass on their own merits.
        for node_id in victim_ids:
            assert outcomes[node_id] == 'PASSED', (
                f'{node_id} did not pass with waymo-open-dataset installed '
                f'(outcome={outcomes[node_id]!r}):\n{output[-3000:]}'
            )
    else:
        # WAYMO IS ABSENT HERE — this machine, and every machine in this
        # project's documented local/Colab split. Both named victims must fail,
        # and specifically for the missing-dependency reason, not some other
        # way (which is what the leak reaching the stub would look like).
        for node_id in victim_ids:
            assert outcomes[node_id] == 'FAILED', (
                f'{node_id} did not fail the expected way '
                f'(outcome={outcomes[node_id]!r}):\n{output[-3000:]}'
            )
        assert output.count("ModuleNotFoundError: No module named 'waymo_open_dataset'") == 2, (
            f'expected exactly the standard missing-waymo failure for both named tests:\n'
            f'{output[-3000:]}'
        )


def test_A07_de_provenance_survives_an_accepted_refinement():
    """
    `result = refined` replaces the whole dict, and the provenance built afterwards
    reads DE's keys off it — so accepting a refinement erased the record of the search
    that produced its warm start. maxiter=0 makes DE's own numbers small and definite
    (one evaluation of the initial population) rather than incidental.

    candidate_source and archive_covered_all_evaluations have existed in
    optimize_scenario's return since Batch 6 and were NEVER persisted, so those two are
    new coverage rather than something being restored.
    """
    from src.scoring.batch_scorer import _stress_one

    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[1, :, 1] = 2.1
    result = _stress_one(states, np.ones((2, 10), dtype=bool), np.array([1, 1]), 0,
                         use_autograd=True,
                         de_kwargs={'popsize': 4, 'maxiter': 0, 'seed': 0})

    assert result['method'] == 'de+autograd', 'fixture regressed: refinement must win'
    provenance = result['search_provenance']

    assert provenance['de_n_eval'] is not None, (
        "DE's evaluation count was erased by the refinement that used its answer"
    )
    assert provenance['de_n_iter'] is not None, "DE's iteration count was erased"
    assert provenance['de_candidate_source'] is not None, (
        "Batch 6's two-stage selection outcome is not recorded"
    )
    assert provenance['de_archive_covered_all_evaluations'] is not None
    assert provenance['selected_stage'] == 'de+autograd', (
        'which stage actually won is not stated'
    )


# ── A04 / A13 / A14: the notebook's own defects ─────────────────────────────────
#
# Three independent findings that happen to share one file. Stated that way rather than
# given a shared invariant: A04 is a crash, A13 is a measurement that answers the wrong
# question, A14 is a namespace collision.
#
# Reproduced against the UNFIXED tree, through the real pipeline rather than injected
# dicts wherever the finding is about end-to-end behaviour:
#
#   A04  SDC with ten valid frames, challenger valid ONLY at frame 0. B15 skips an
#        agent with fewer than two valid timesteps, so it is never exported — but R06
#        persists target_idx on scenario_scores regardless. Measured:
#          export_scenario_agents: written=1 skipped=1   exported agent set: [0]
#          scenario_scores.target_idx = 1
#          /perturbed  target_idx=1  perturbed=None  delta=[-0.7056..., ...]
#        and the round-trip cell's guard, which fires only on target_idx IS None, does
#        not fire — so its bare next(...) raises StopIteration. Confirmed under BOTH
#        triggers, the second through a real DE search rather than assumed:
#          collision     outcome=collision_found     t_hit=0     -> StopIteration
#          no-collision  outcome=no_collision_found  t_hit=None  -> StopIteration
#
#        NOTE delta is PRESENT in both. update_stress_results stores it whenever a
#        search ran, so the API cell's "(null above is EXPECTED — no collision was
#        found)" is wrong about which fields are absent: only collision_timestep and
#        min_perturbation are.
#
#   A13  on B06's canonical fixture, visits_a=[(0,1),(5,6)] visits_b=[(3,3)]:
#          corrected (min over pairs) = +0.2000   winners = [(0,0),(1,0)]
#          merged-span (pre-B06)      = -0.3000
#          sign flip: True   magnitude: 0.5000 s
#          n_multi_visit_decisive += 0
#        Both pairings tie, so pair one is AMONG the winners and the counter reports
#        that separate visits never decided anything — on the exact fixture it exists
#        to catch.
#
#   A14  cell binding rho for the all-pairs-vs-SDC ranking correlation, and the cell
#        binding rho for the old-vs-new TTC correlation, are different experiments
#        sharing one bare global. The summary reads rho under the FIRST experiment's
#        label and gets whichever ran last.

def _notebook_cells():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'notebooks', 'colab_validation_run.ipynb')
    return json.load(open(path))['cells']


def _cell_by_content(*required):
    """
    Located by CONTENT, never by index — the rule this project adopted after the
    fourth index-drift incident. Cells 24-31 and 39-40 of this notebook carry no `id`
    at all, so even id-based location would be partial.
    """
    hits = [''.join(c['source']) for c in _notebook_cells()
            if c['cell_type'] == 'code'
            and all(token in ''.join(c['source']) for token in required)]
    assert len(hits) == 1, f'expected exactly one cell matching {required}, got {len(hits)}'
    return hits[0]


def _one_frame_challenger(lateral):
    """SDC fully observed; challenger observed ONLY at its first frame."""
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[1, :, 1] = lateral
    validity = np.ones((2, 10), dtype=bool)
    validity[1, 1:] = False
    return states, validity, np.array([1, 1])


def _fully_observed(lateral):
    states = np.zeros((2, 10, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[1, :, 1] = lateral
    return states, np.ones((2, 10), dtype=bool), np.array([1, 1])


def _run_pipeline(conn, sid, scene, de_kwargs=None):
    """Pass 2 + Pass 3 through the REAL functions, then the REAL API."""
    from src.scoring import db
    from src.scoring.export_geometry import init_geometry_schema, export_scenario_agents
    from src.scoring.batch_scorer import _stress_one

    states, validity, types = scene
    with conn.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS perturbed_paths, scenario_agents, '
                    'scenario_scores CASCADE')
    conn.commit()
    db.init_schema(conn)
    init_geometry_schema(conn)

    result = _stress_one(states, validity, types, 0,
                         de_kwargs=de_kwargs or {'popsize': 4, 'maxiter': 3, 'seed': 0})
    db.upsert_scores(conn, [dict(
        scenario_id=sid, shard='synthetic', n_agents=int(states.shape[0]),
        min_ttc=9.0, min_pet=9.0, fragility_score=1.0,
        scene_fingerprint=db.compute_scene_fingerprint(states, validity, types))])
    db.update_stress_results(conn, {sid: result})
    export_scenario_agents(conn, sid, states, validity, types, 0)
    return result


def _real_httpx(client, sid):
    """
    An httpx-shaped shim over the REAL TestClient, so the notebook cell sees exactly
    what the API returns rather than a hand-written dict. The finding is about a state
    the pipeline produces; a fake response could assert that state into existence.
    """
    import types as _types

    def get(url, params=None):
        path = url.split('http://x')[-1] if url.startswith('http://x') else url
        return client.get(path, params=params)
    return _types.SimpleNamespace(get=get)


def _real_dump_points_m(conn):
    """cell 13's helper, against the real database."""
    def dump(connection, table, scenario_id, agent_idx=None):
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT n_points, ARRAY(SELECT ST_M(dp.geom)
                                         FROM ST_DumpPoints(path) dp ORDER BY dp.path)
                  FROM {table} WHERE scenario_id = %s AND agent_idx = %s
            """, (scenario_id, agent_idx))
            row = cur.fetchone()
        assert row is not None, f'no {table} row for agent {agent_idx}'
        return None, row[0], None, row[1]
    return dump


def _exec_api_cells(conn, client, sid, result):
    """Run the notebook's two API cells against real responses."""
    import types as _types

    env = {
        'httpx': _real_httpx(client, sid), 'base': 'http://x', 'np': np,
        'stress_results': {sid: result}, 'ids_to_test': [sid],
        'dump_points_m': _real_dump_points_m(conn), 'conn': conn,
        'server': _types.SimpleNamespace(should_exit=False),
    }
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(_cell_by_content('stress_tested_sid', 'GET /health'), 'api', 'exec'), env)
        exec(compile(_cell_by_content('http_timesteps'), 'roundtrip', 'exec'), env)
    return env, out.getvalue()


@requires_db
@pytest.mark.parametrize('label,lateral', [('collision', 2.1), ('no_collision', 40.0)])
def test_A04_a_challenger_without_geometry_does_not_crash_the_round_trip(
        pconn, label, lateral):
    """
    R09 closed the door where target_idx was None. This is a DIFFERENT door that R09's
    fixture could not produce: R06 makes target_idx persist even when B15 declined to
    export that agent, so the guard does not fire and the bare next(...) has no default.

    BOTH TRIGGERS, because the audit states both and the collision case runs a real DE
    search rather than being asserted into existence.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app

    result = _run_pipeline(pconn, 'a04', _one_frame_challenger(lateral))
    expected = 'collision_found' if label == 'collision' else 'no_collision_found'
    assert result['outcome'] == expected, f"fixture regressed: {result['outcome']}"

    with TestClient(app) as client:
        traj = client.get('/scenarios/a04/trajectories').json()
        pert = client.get('/scenarios/a04/perturbed').json()
    assert pert['target_idx'] == 1, 'fixture regressed: R06 must persist the target'
    assert all(a['agent_idx'] != 1 for a in traj['agents']), (
        'fixture regressed: the target must NOT be among the exported agents'
    )

    with TestClient(app) as client:
        env, printed = _exec_api_cells(pconn, client, 'a04', result)

    assert 'CONFIRMED: HTTP timesteps == Postgres M values' in printed, printed
    assert env['target_idx_api'] in {a['agent_idx'] for a in env['traj']['agents']}, (
        'the round trip selected an agent that is not in the exported set'
    )


@requires_db
def test_A04_the_round_trip_still_prefers_the_real_challenger(pconn):
    """
    The fix must not trade a crash for a wrong agent. When the target IS exported, the
    round trip must still close on it rather than falling back to agent 0 — otherwise
    the assertion silently stops describing the challenger the result is about.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.scoring.export_geometry import export_perturbed_path
    from src.scoring import db

    scene = _fully_observed(2.1)
    result = _run_pipeline(pconn, 'a04ok', scene)
    assert result.get('collision'), 'fixture regressed: need a collision'
    states, validity, types = scene
    from src.optimization.perturbation_space import PerturbationSpace
    space = PerturbationSpace(states, validity, types, 0, int(result['target_idx']))
    export_perturbed_path(pconn, 'a04ok',
                          space.apply(np.asarray(result['delta'], dtype=np.float32)),
                          validity, int(result['target_idx']),
                          delta=result['delta'], method=result.get('method'),
                          scene_fingerprint=db.compute_scene_fingerprint(*scene))

    with TestClient(app) as client:
        env, printed = _exec_api_cells(pconn, client, 'a04ok', result)

    assert env['target_idx_api'] == result['target_idx'], (
        'the round trip fell back to another agent although the target WAS exported'
    )
    assert 'CONFIRMED: HTTP timesteps == Postgres M values' in printed


@requires_db
def test_A04_a_completed_search_still_reports_its_delta(pconn):
    """
    The API cell's smaller error, independent of the crash. It prints
    "(null above is EXPECTED — no collision was found for this scenario)" beneath a
    line showing BOTH delta and collision_timestep — but update_stress_results stores
    the delta whenever a search ran, so a completed unsuccessful search exports a
    candidate delta. Only collision_timestep and min_perturbation are absent.

    Asserted against the pipeline rather than against the prose, so the claim is
    checked where it is made false rather than where it is written.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app

    result = _run_pipeline(pconn, 'a04nc', _fully_observed(40.0))
    assert result['outcome'] == 'no_collision_found', result['outcome']

    with TestClient(app) as client:
        pert = client.get('/scenarios/a04nc/perturbed').json()

    assert pert['delta'] is not None, (
        'fixture regressed: a completed search must store its candidate delta'
    )
    assert pert['collision_timestep'] is None
    assert pert['min_perturbation'] is None

    cell = _cell_by_content('stress_tested_sid', 'GET /health')
    assert 'null above is EXPECTED — no collision was found' not in cell, (
        'the cell still claims the delta is null for a completed search; measured, it '
        f'is {pert["delta"]!r}'
    )


def _b06_tie_scene():
    """
    A REAL scene whose occupancy visits reproduce B06's canonical fixture exactly —
    including the tie that makes the old counter blind.

    Agent 0 nudges along x and is observed with a validity gap, so it enters the shared
    zone twice. Agent 1 is observed at two frames, only ONE of which is inside the zone
    (the other is 20 m away), which is what makes its visit list a single (3, 3) span
    rather than (3, 4) — and that single span is what makes both pairings tie.

    Verified to produce, through the real pet_engine:
        visits_a = [(0, 1), (5, 6)]   visits_b = [(3, 3)]
        pair (0,0) -> +0.2   pair (1,0) -> +0.2   winners = both, so (0,0) IS a winner
        merged-span (pre-B06) = -0.3
    """
    T = 7
    states = np.zeros((2, T, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[0, :, 0] = np.linspace(-1, 1, T)
    states[1, :, 0] = 0.0
    states[1, :, 1] = [20.0, 20.0, 20.0, 0.0, 20.0, 20.0, 20.0]
    validity = np.zeros((2, T), dtype=bool)
    validity[0] = [1, 1, 0, 0, 0, 1, 1]
    validity[1] = [0, 0, 0, 1, 0, 0, 1]
    return states, validity


def test_A13_the_before_after_measure_sees_a_sign_flip():
    """
    n_multi_visit_decisive increments only when the winning visit pair is not (0,0) —
    "did we need to look past pair one", which is not "did separate visits change the
    answer". On B06's own fixture both pairings TIE, so pair one is among the winners
    and the counter reports zero for a change of half a second that flips sign.

    THE CELL IS EXECUTED, NOT GREPPED. An earlier version of this test asserted that
    the string "merged" appeared in the cell, and it PASSED against the unfixed code —
    because the cell's comments already discuss merged spans at length. A substring
    check cannot tell prose from computation, which is the same "a test that cannot
    reach what it names" failure this project keeps finding.
    """
    from src.danger.pet_engine import get_path_polygon, _occupancy_visits, DT

    states, validity = _b06_tie_scene()
    path_a = get_path_polygon(states, 0, validity)
    path_b = get_path_polygon(states, 1, validity)
    zone = path_a.intersection(path_b)
    visits_a = _occupancy_visits(states, validity, 0, zone)
    visits_b = _occupancy_visits(states, validity, 1, zone)

    scored = [(max(eb - xa, ea - xb) * DT, i_a, i_b)
              for i_a, (ea, xa) in enumerate(visits_a)
              for i_b, (eb, xb) in enumerate(visits_b)]
    corrected = min(v for v, _, _ in scored)
    winners = [(i_a, i_b) for v, i_a, i_b in scored if v == corrected]
    merged = max(visits_b[0][0] - visits_a[-1][1],
                 visits_a[0][0] - visits_b[-1][1]) * DT

    assert len(visits_a) > 1, f'fixture regressed: visits_a={visits_a}'
    assert (merged < 0) != (corrected < 0), (
        f'fixture regressed: no sign flip ({merged} -> {corrected})'
    )
    assert (0, 0) in winners, (
        'fixture regressed: pair one must be AMONG the winners, or the old counter '
        'would already catch this and there would be nothing to fix'
    )

    env = {'diag_cache': [{'states': states, 'validity': validity, 'sdc_idx': 0,
                           'scenario_id': 'b06_tie'}],
           'np': np}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(_cell_by_content('audit B06 — did separate visits'),
                     'pet_cell', 'exec'), env)
    printed = out.getvalue()

    assert 'never decides the result' not in printed, (
        f'the cell reports that multi-visit never decided anything, on a fixture where '
        f'the pre-B06 answer was {merged:+.2f} s and the corrected answer is '
        f'{corrected:+.2f} s:\n{printed}'
    )
    assert f'{merged:+.2f}' in printed or f'{merged:.2f}' in printed, (
        f'the cell never reports the pre-B06 merged-span answer ({merged:+.4f}), so it '
        f'cannot be comparing against it:\n{printed}'
    )


def test_A14_the_summary_reads_a_name_no_later_cell_rebinds():
    """
    The ranking-correlation cell and the TTC-correlation cell are different
    experiments; both bound a bare `rho`; the summary reads `rho` under the FIRST
    experiment's label and gets whichever ran last.

    Asserted STRUCTURALLY rather than by executing three heavyweight cells: the defect
    is that a name the summary depends on is bound in more than one place, which is a
    property of the notebook's namespace and not of any particular run's numbers.
    """
    import ast as _ast

    cells = [(i, ''.join(c['source'])) for i, c in enumerate(_notebook_cells())
             if c['cell_type'] == 'code']
    summary = next(src for _, src in cells if 'COLAB VALIDATION RUN — SUMMARY' in src)

    # the name the summary prints beside the ranking-experiment label
    ranking_line = next(l for l in summary.splitlines() if 'Spearman rho=' in l)
    name = ranking_line.split('{')[1].split(':')[0].split('}')[0].strip()

    binders = []
    for index, src in cells:
        try:
            tree = _ast.parse(src)
        except SyntaxError:
            continue
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Assign):
                for target in node.targets:
                    for leaf in _ast.walk(target):
                        if isinstance(leaf, _ast.Name) and leaf.id == name:
                            binders.append(index)
    binders = sorted(set(binders))

    assert len(binders) == 1, (
        f'the summary reads {name!r}, which is bound by {len(binders)} different cells '
        f'{binders} — it prints whichever ran last under one experiment\'s label'
    )
