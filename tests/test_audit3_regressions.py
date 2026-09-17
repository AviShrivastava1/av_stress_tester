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
