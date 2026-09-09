"""
test_api.py — Phase 6 end-to-end test.

Runs the WHOLE pipeline against a real PostgreSQL with PostGIS: score, persist,
rank, stress-test, export geometry, then exercise every API endpoint through
FastAPI's TestClient.

No WOMD data required. The scenarios are built directly as numpy arrays, which is
what makes this runnable on a laptop with no Waymo package and no 1.5 GB shard —
_score_one and _stress_one were deliberately split out of the shard-reading loop
in Phase 5 for exactly this reason.

Run:
    PGDATABASE=av_stress PGUSER=$(whoami) python tests/test_api.py

WARNING: this clears scenario_scores (and, by ON DELETE CASCADE, the geometry
tables) so the corpus-level counts are deterministic. Point it at a test database.
"""

import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from src.scoring import db
from src.scoring.batch_scorer import _score_one, _stress_one
from src.scoring.ranker import rank_scenarios
from src.scoring.export_geometry import (
    init_geometry_schema, export_scenario_agents, export_perturbed_path,
)
from src.optimization.perturbation_space import PerturbationSpace
from src.api.main import app


T = 91           # WOMD scenario length
DT = 0.1         # seconds per timestep
SPEED = 10.0     # m/s for both agents
CAR_L, CAR_W = 4.5, 2.0

# Four scenarios, not three. The first three vary the challenger's head start, so
# fragility is distinct and the ranking is meaningful. The FOURTH duplicates the
# third's geometry exactly under a different id, which makes two rows TIE on
# fragility_score.
#
# That tie is the entire point of this scenario existing. Keyset pagination's
# hardest case is a cursor that lands between two rows with equal sort keys, and
# with only distinct scores no test could ever reach that case — a pagination bug
# in the tie branch would pass every assertion and ship. Paging at limit=2 across
# four rows puts the page boundary exactly between syn_medium_a and syn_medium_b.
SCENARIOS = [
    ('syn_close',    -28.0),
    ('syn_medium_a', -33.0),
    ('syn_medium_b', -33.0),   # identical geometry to syn_medium_a -> exact tie
    ('syn_far',      -60.0),
]


def build_scenario(challenger_y0):
    """
    A 2-vehicle crossing: the SDC drives east along y=0, a challenger drives north
    through x=0. `challenger_y0` sets how close the near-miss is — the later the
    challenger arrives at the intersection, the safer the scenario.

    Returns (states, validity, types) with states as (N, T, 7):
        [x, y, vx, vy, heading, length, width]
    """
    t = np.arange(T) * DT
    states = np.zeros((2, T, 7), dtype=np.float32)

    # agent 0 — the SDC, heading east (heading 0) from x=-20
    states[0, :, 0] = -20.0 + SPEED * t
    states[0, :, 1] = 0.0
    states[0, :, 2] = SPEED
    states[0, :, 3] = 0.0
    states[0, :, 4] = 0.0

    # agent 1 — the challenger, heading north (heading pi/2) through x=0
    states[1, :, 0] = 0.0
    states[1, :, 1] = challenger_y0 + SPEED * t
    states[1, :, 2] = 0.0
    states[1, :, 3] = SPEED
    states[1, :, 4] = np.pi / 2

    states[:, :, 5] = CAR_L
    states[:, :, 6] = CAR_W

    validity = np.ones((2, T), dtype=bool)
    types = np.array([1, 1])          # both vehicles
    return states, validity, types


def check(label, condition, detail=''):
    """Assert with a readable label, so a failure names what broke."""
    if not condition:
        raise AssertionError(f"{label} FAILED {detail}")
    print(f"  ok  {label}")


def main():
    print("Phase 6 end-to-end test\n" + "=" * 60)

    conn = db.get_connection()
    db.init_schema(conn)
    init_geometry_schema(conn)

    # Deterministic corpus counts need a known-empty table. The FK on the geometry
    # tables is ON DELETE CASCADE, so this clears them too.
    with conn.cursor() as cur:
        cur.execute("DELETE FROM scenario_scores")
    conn.commit()
    print("Cleared scenario_scores (cascades to geometry tables)")

    # ── PASS 1: score every scenario ────────────────────────────────────────────
    built = {}
    records = []
    for sid, y0 in SCENARIOS:
        states, validity, types = build_scenario(y0)
        built[sid] = (states, validity, types)
        rec = _score_one(states, validity, sid, shard_name='synthetic')
        records.append(rec)
        print(f"  scored {sid}: ttc={rec['min_ttc']:.2f} pet={rec['min_pet']:.2f} "
              f"fragility={rec['fragility_score']:.4f}")

    db.upsert_scores(conn, records)
    ranked = rank_scenarios(records)
    expected_order = [r['scenario_id'] for r in ranked]
    print(f"\nRanked: {expected_order}")

    check("ranked order is close > medium_a > medium_b > far",
          expected_order == ['syn_close', 'syn_medium_a', 'syn_medium_b', 'syn_far'],
          f"got {expected_order}")

    tie_a, tie_b = records[1]['fragility_score'], records[2]['fragility_score']
    check("syn_medium_a and syn_medium_b tie exactly on fragility_score",
          tie_a == tie_b, f"{tie_a!r} != {tie_b!r}")

    # ── PASS 2: stress-test the most fragile scenario ───────────────────────────
    top_sid = expected_order[0]
    states, validity, types = built[top_sid]
    print(f"\nStress-testing {top_sid} (this runs the Phase 4 optimizer) ...")
    result = _stress_one(
        states, validity, types, sdc_idx=0,
        de_kwargs=dict(popsize=10, maxiter=60, tol=1e-2, seed=1),
    )
    print(f"  status={result.get('status')} collision={result.get('collision')} "
          f"||delta||={result.get('min_perturbation')} "
          f"t_hit={result.get('collision_timestep')}")
    check("stress test found a collision", bool(result.get('collision')),
          f"result={result}")

    db.update_stress_results(conn, {top_sid: result})

    # ── PASS 3: export geometry ─────────────────────────────────────────────────
    for sid, _ in SCENARIOS:
        s, v, ty = built[sid]
        written, skipped = export_scenario_agents(conn, sid, s, v, ty, sdc_idx=0)
        print(f"  geometry {sid}: {written} agents, {skipped} skipped")

    space = PerturbationSpace(states, validity, types, 0, int(result['target_idx']))
    perturbed_states = space.apply(np.asarray(result['delta'], dtype=np.float32))
    export_perturbed_path(conn, top_sid, perturbed_states, validity,
                          int(result['target_idx']))
    print(f"  perturbed path exported for {top_sid}")

    conn.close()

    # ── API ─────────────────────────────────────────────────────────────────────
    # TestClient MUST be a context manager, or the lifespan handler never runs and
    # the connection pool is never initialized.
    print("\nAPI assertions\n" + "-" * 60)
    with TestClient(app) as client:

        # /health — must do a real round-trip, not just return "ok"
        r = client.get('/health')
        check("/health returns 200", r.status_code == 200, r.text)
        check("/health reports a PostGIS version",
              r.json()['postgis'] is not None, r.text)

        # /stats
        r = client.get('/stats')
        s = r.json()
        check("/stats total_scenarios == 4", s['total_scenarios'] == 4, str(s))
        check("/stats stress_tested == 1", s['stress_tested'] == 1, str(s))
        check("/stats with_geometry == 4", s['with_geometry'] == 4, str(s))
        check("/stats collisions_found == 1", s['collisions_found'] == 1, str(s))

        # /scenarios — order must match ranker.rank_scenarios exactly
        r = client.get('/scenarios', params={'limit': 10})
        page = r.json()
        api_order = [i['scenario_id'] for i in page['items']]
        check("/scenarios order identical to ranker.rank_scenarios",
              api_order == expected_order, f"{api_order} != {expected_order}")
        check("/scenarios last page has next_cursor=None",
              page['next_cursor'] is None, str(page['next_cursor']))

        top = page['items'][0]
        check("top scenario stress_tested=True", top['stress_tested'] is True, str(top))
        check("top scenario robustly_safe=False", top['robustly_safe'] is False, str(top))
        check("other scenarios stress_tested=False",
              all(i['stress_tested'] is False for i in page['items'][1:]),
              str(page['items'][1:]))

        # ── pagination across the tie boundary ──────────────────────────────────
        # limit=2 over four rows puts the cursor exactly between syn_medium_a and
        # syn_medium_b, which tie. Under the row-comparison predicate
        # `(fragility_score, scenario_id) < (f0, s0)`, page 2 would come back as
        # [syn_far] alone: the tie branch compares scenario_id in the WRONG
        # direction for a mixed-direction ORDER BY, so syn_medium_b is silently
        # dropped. This assertion is what catches that.
        r1 = client.get('/scenarios', params={'limit': 2})
        p1 = r1.json()
        check("page 1 has 2 items", len(p1['items']) == 2, str(p1))
        check("page 1 has a next_cursor", p1['next_cursor'] is not None, str(p1))

        r2 = client.get('/scenarios',
                        params={'limit': 2, 'cursor': p1['next_cursor']})
        p2 = r2.json()
        combined = ([i['scenario_id'] for i in p1['items']]
                    + [i['scenario_id'] for i in p2['items']])
        check("two pages of limit=2 reconstruct the full order across the tie",
              combined == expected_order, f"{combined} != {expected_order}")
        check("no duplicates across pages",
              len(combined) == len(set(combined)), str(combined))
        check("final page has next_cursor=None",
              p2['next_cursor'] is None, str(p2['next_cursor']))

        # ── cursor float precision ──────────────────────────────────────────────
        # The tie branch of the predicate is `fragility_score = %s`, an exact
        # IEEE-754 equality. If anything ever rounds or reformats the float on its
        # way into the cursor, tied rows silently vanish (rounded down) or repeat
        # (rounded up). The tie test above would catch gross breakage, but it fails
        # with a confusing "missing scenario" message — this names the real cause.
        decoded = json.loads(base64.urlsafe_b64decode(
            p1['next_cursor'].encode('ascii')))
        last_of_page1 = p1['items'][-1]
        check("cursor float round-trips bit-exactly",
              decoded['f'] == last_of_page1['fragility_score'],
              f"{decoded['f']!r} != {last_of_page1['fragility_score']!r}")
        check("cursor scenario_id round-trips",
              decoded['s'] == last_of_page1['scenario_id'], str(decoded))

        # ── error handling ──────────────────────────────────────────────────────
        r = client.get('/scenarios', params={'cursor': 'not-a-real-cursor!!'})
        check("malformed cursor returns 422", r.status_code == 422,
              f"got {r.status_code}: {r.text}")

        r = client.get('/scenarios/does_not_exist')
        check("unknown scenario detail returns 404", r.status_code == 404,
              str(r.status_code))
        r = client.get('/scenarios/does_not_exist/trajectories')
        check("unknown scenario trajectories returns 404", r.status_code == 404,
              str(r.status_code))

        # ── filtering ───────────────────────────────────────────────────────────
        r = client.get('/scenarios', params={'stress_tested_only': True})
        items = r.json()['items']
        check("stress_tested_only returns exactly the tested scenario",
              [i['scenario_id'] for i in items] == [top_sid], str(items))

        # ── trajectories ────────────────────────────────────────────────────────
        r = client.get(f'/scenarios/{top_sid}/trajectories')
        traj = r.json()
        check("trajectories returns 2 agents", len(traj['agents']) == 2, str(traj))
        a0, a1 = traj['agents']
        check("agent 0 is the SDC", a0['is_sdc'] is True, str(a0['is_sdc']))
        check("agent 1 is not the SDC", a1['is_sdc'] is False, str(a1['is_sdc']))
        check("agent 0 has 91 path points", len(a0['path']) == T, str(len(a0['path'])))
        check("timesteps survive the GeoJSON round-trip (M ordinate)",
              a0['timesteps'][0] == 0.0 and a0['timesteps'][-1] == 90.0,
              f"{a0['timesteps'][:2]} .. {a0['timesteps'][-2:]}")
        check("headings align with path length",
              len(a0['headings']) == len(a0['path']),
              f"{len(a0['headings'])} != {len(a0['path'])}")

        # geometry not yet exported is NOT an error, but here everything is
        # exported, so all four scenarios have agents
        r = client.get('/scenarios/syn_far/trajectories')
        check("syn_far also has geometry", len(r.json()['agents']) == 2, r.text)

        # ── perturbed ───────────────────────────────────────────────────────────
        r = client.get(f'/scenarios/{top_sid}/perturbed')
        p = r.json()
        check("/perturbed returns 200", r.status_code == 200, r.text)
        check("/perturbed has a baseline path", p['baseline'] is not None, r.text)
        check("/perturbed has a perturbed path", p['perturbed'] is not None, r.text)
        check("/perturbed delta has 4 dimensions", len(p['delta']) == 4, str(p['delta']))
        check("/perturbed delta_labels has 4 entries",
              len(p['delta_labels']) == 4, str(p['delta_labels']))
        check("/perturbed collision_timestep is not null",
              p['collision_timestep'] is not None, str(p['collision_timestep']))

        base_xy = np.asarray(p['baseline']['path'], dtype=float)
        pert_xy = np.asarray(p['perturbed']['path'], dtype=float)
        max_shift = float(np.abs(base_xy - pert_xy).max())
        check("perturbed path measurably differs from baseline",
              max_shift > 0.5, f"max shift only {max_shift:.4f} m")
        print(f"      (max deviation {max_shift:.2f} m)")

        # a scenario that was never stress-tested: 200 with nulls, not 404
        r = client.get('/scenarios/syn_far/perturbed')
        p = r.json()
        check("untested scenario /perturbed returns 200", r.status_code == 200, r.text)
        check("untested scenario has perturbed=None", p['perturbed'] is None, r.text)
        check("untested scenario has delta=None", p['delta'] is None, r.text)

        # ── the geometry-consistency guard ──────────────────────────────────────
        # Corrupt one agent's headings so it disagrees with the coordinate count,
        # and assert the endpoint REFUSES instead of padding or truncating.
        #
        # This exists because the tempting fallback — substituting the vertex index
        # when a measure is missing — would pass every other assertion in this
        # file. Synthetic scenarios have all 91 timesteps valid, so index equals
        # timestep and nothing looks wrong. Real WOMD scenarios with a validity gap
        # are where the two diverge, and there the fallback would attach frames to
        # the wrong moments in time while still returning a perfectly plausible
        # chart. Guessing is how that bug gets reintroduced; this test is what
        # stops it.
        conn2 = db.get_connection()
        with conn2.cursor() as cur:
            cur.execute("UPDATE scenario_agents SET headings = headings[1:5] "
                        "WHERE scenario_id = 'syn_far' AND agent_idx = 1")
        conn2.commit()

        r = client.get('/scenarios/syn_far/trajectories')
        check("geometry length mismatch is refused, not papered over",
              r.status_code == 500, f"got {r.status_code}: {r.text[:200]}")
        check("the refusal names the mismatch",
              'Refusing to pad or truncate' in r.text, r.text[:200])

        # restore, so the database is left consistent for manual poking
        s_far, v_far, t_far = built['syn_far']
        export_scenario_agents(conn2, 'syn_far', s_far, v_far, t_far, sdc_idx=0)
        conn2.close()
        r = client.get('/scenarios/syn_far/trajectories')
        check("geometry restored after the guard test", r.status_code == 200, r.text[:200])

    print("\n" + "=" * 60)
    print("ALL PHASE 6 TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    main()
