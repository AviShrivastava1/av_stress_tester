"""
test_pet_components.py — PET is paired within one connected component of the conflict zone.

The conflict zone is where two agents' swept paths overlap. Shapely returns it as ONE
geometry even when it holds pieces in different places (a MultiPolygon, or a
GeometryCollection mixing polygons and lines). compute_pet_pair used to treat that whole
geometry as a single place: it collected each agent's visits to ANY piece and paired every
A-visit with every B-visit. Two agents that cross each other in two separate places, at
two separate times, then looked simultaneous: A's early visit to one place was paired with
B's early visit to the OTHER place, and PET came out negative although they were never
near each other when it mattered.

compute_pet_pair now splits the zone into connected components (pieces that touch or
overlap are one component) and pairs visits only within a component.

WHAT A NEGATIVE PET MEANS, stated narrowly: both agents' boxes intersected the same
connected component of the zone during a common frame. It does not mean their boxes
overlapped each other; the zone can be larger than either box.

No expected value here is a number copied from a run. Frame gaps are derived from the
frame constants the test itself sets; the single-component comparison runs against a
labelled copy of the previous implementation.

Pure numpy/shapely, no database. Run:
    ./venv/bin/python -m pytest tests/test_pet_components.py -q
"""

import os
import sys

import numpy as np
import pytest
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.danger.collision_detector import get_corners
from src.danger.pet_engine import (
    DT,
    PET_INFINITY,
    _occupancy_visits,
    compute_pet_pair,
    get_path_polygon,
)


# ══════════════════════════════════════════════════════════════════════════════════
# REFERENCE ORACLE — NOT THE FUNCTION UNDER TEST.
# A verbatim copy of compute_pet_pair as it stood at commit ada514f, before the zone was
# split into components (its docstring omitted). It calls the module's own
# get_path_polygon and _occupancy_visits, which this change does not touch. It exists so
# the new function can be compared with the old one on cases where the answer must not
# move, and so the tests can show what the old answer was.
# ══════════════════════════════════════════════════════════════════════════════════
def _previous_compute_pet_pair(states, validity, agent_a, agent_b, path_a=None, path_b=None):
    if path_a is None:
        path_a = get_path_polygon(states, agent_a, validity)
    if path_b is None:
        path_b = get_path_polygon(states, agent_b, validity)

    if path_a is None or path_b is None:
        return PET_INFINITY

    conflict_zone = path_a.intersection(path_b)

    if conflict_zone.is_empty:
        return PET_INFINITY  # paths never cross spatially

    visits_a = _occupancy_visits(states, validity, agent_a, conflict_zone)
    visits_b = _occupancy_visits(states, validity, agent_b, conflict_zone)

    if not visits_a or not visits_b:
        return PET_INFINITY

    pet = min(
        max(enter_b - exit_a, enter_a - exit_b)
        for enter_a, exit_a in visits_a
        for enter_b, exit_b in visits_b
    ) * DT
    return float(pet)


# ── helpers ──────────────────────────────────────────────────────────────────────────


def _box(x, y, size=2.0):
    return Polygon(get_corners(x, y, 0.0, size, size))


def _scene(T, plan):
    """
    Two 2x2 agents. `plan[agent]` maps frame -> x position; every other frame is invalid
    for that agent. y is 0 throughout.
    """
    states = np.zeros((2, T, 7), dtype=np.float32)
    states[:, :, 5:7] = [2.0, 2.0]
    validity = np.zeros((2, T), dtype=bool)
    for agent, frames in plan.items():
        for t, x in frames.items():
            validity[agent, t] = True
            states[agent, t, 0] = x
    return states, validity


def _frames(frame_range, x):
    return {t: x for t in frame_range}


# ── 1. the disconnected-zones case, with hand-passed swept paths ─────────────────────

EARLY = range(0, 5)        # frames 0..4
LATE = range(84, 89)       # frames 84..88


def test_pet_never_pairs_visits_from_disconnected_conflict_zones():
    """
    Agent A is at the left component early and the right component late; agent B does
    the reverse. Treating the union as one zone pairs A's early left visit with B's early
    right visit (same frames, different places) and returns about -0.4 s. Inside either
    component the agents are an entire late-minus-early apart.
    """
    states, validity = _scene(89, {
        0: {**_frames(EARLY, -10.0), **_frames(LATE, 10.0)},
        1: {**_frames(EARLY, 10.0), **_frames(LATE, -10.0)},
    })
    two_zones = MultiPolygon([_box(-10.0, 0.0), _box(10.0, 0.0)])

    pet = compute_pet_pair(states, validity, 0, 1, path_a=two_zones, path_b=two_zones)

    # Derived from the frames above: inside one component the visits are [0, 4] and [84, 88].
    assert pet == pytest.approx((LATE[0] - EARLY[-1]) * DT)


# ── 2. the same bug, from real moving agents and their real swept paths ─────────────


def _two_crossings():
    """
    Agent A follows y = 8 cos(pi x / 20) from x = -20 to +20; agent B drives straight along
    y = 0 from x = +20 to -20, both at 40 m / 9 s. The curves meet at x = -10 and x = +10.
    A is at x = -10 at t = 2.25 s and at x = +10 at t = 6.75 s; B is at x = +10 at 2.25 s and
    at x = -10 at 6.75 s. In each place the two agents are 4.5 s apart.
    """
    T = 91
    t = np.arange(T) * DT
    v = 40.0 / 9.0
    x_a = -20.0 + v * t
    y_a = 8.0 * np.cos(np.pi * x_a / 20.0)
    heading_a = np.arctan2(-8.0 * np.pi / 20.0 * np.sin(np.pi * x_a / 20.0), 1.0)
    states = np.zeros((2, T, 7), dtype=np.float32)
    states[:, :, 5:7] = [4.5, 2.0]
    states[0, :, 0], states[0, :, 1], states[0, :, 4] = x_a, y_a, heading_a
    states[1, :, 0], states[1, :, 1], states[1, :, 4] = 20.0 - v * t, 0.0, np.pi
    validity = np.ones((2, T), dtype=bool)
    return states, validity, 2.25, 6.75


def test_two_separate_crossings_by_moving_agents_are_not_simultaneous():
    states, validity, first_a, second_a = _two_crossings()
    zone = get_path_polygon(states, 0, validity).intersection(get_path_polygon(states, 1, validity))
    # The premise: the swept paths really do overlap in two separate places.
    assert zone.geom_type == 'MultiPolygon' and len(zone.geoms) == 2

    # What the previous implementation said, from the labelled copy: a negative PET.
    assert _previous_compute_pet_pair(states, validity, 0, 1) < 0.0

    pet = compute_pet_pair(states, validity, 0, 1)

    # In each place the agents are (second_a - first_a) apart at the crossing points; PET is
    # that gap minus the time each box spends in the component, so it is positive and
    # cannot exceed it.
    assert 0.0 < pet < second_a - first_a


# ── 3. cases with 0 or 1 component are unchanged; 2 or more are not ─────────────────

SEED = 20261006
N_PAIRS = 1500
MIN_ZERO_COMPONENT = 300       # stated in advance: what the sweep must contain
MIN_ONE_COMPONENT = 150
MIN_TWO_OR_MORE = 30
MIN_CHANGED_TWO_OR_MORE = 5


def _atomic_parts(geometry):
    if geometry.is_empty:
        return []
    children = getattr(geometry, 'geoms', None)
    if children is None:
        return [geometry]
    return [piece for child in children for piece in _atomic_parts(child)]


def _count_components(zone):
    """Independent of the code under test: the number of pieces of the UNION of the parts."""
    parts = _atomic_parts(zone)
    if not parts:
        return 0
    merged = unary_union(parts)
    return len(_atomic_parts(merged))


@pytest.fixture(scope='module')
def sweep():
    """Seeded pairs of winding agents, so conflict zones often have several pieces."""
    rng = np.random.default_rng(SEED)
    cases = []
    for _ in range(N_PAIRS):
        T = 91
        states = np.zeros((2, T, 7), dtype=np.float32)
        states[:, :, 5:7] = [4.5, 2.0]
        for agent in range(2):
            pos = rng.uniform(-30, 30, 2)
            heading = rng.uniform(-np.pi, np.pi)
            speed = rng.uniform(2, 9)
            for t in range(T):
                heading += rng.normal(0, 0.12)
                pos = pos + speed * DT * np.array([np.cos(heading), np.sin(heading)])
                states[agent, t, :2] = pos
                states[agent, t, 4] = heading
        validity = np.ones((2, T), dtype=bool)
        zone = get_path_polygon(states, 0, validity).intersection(get_path_polygon(states, 1, validity))
        cases.append({
            'states': states, 'validity': validity,
            'components': _count_components(zone),
            'previous': _previous_compute_pet_pair(states, validity, 0, 1),
            'new': compute_pet_pair(states, validity, 0, 1),
            'new_reversed': compute_pet_pair(states, validity, 1, 0),
        })
    return cases


def test_zero_and_one_component_zones_give_exactly_the_previous_answer(sweep):
    # Passes on main on purpose (new and oracle are the same code there); on the new code it
    # is the guard that the split changes nothing where there is nothing to split.
    unsplit = [c for c in sweep if c['components'] <= 1]
    assert unsplit
    differing = [(c['components'], c['previous'], c['new']) for c in unsplit if c['new'] != c['previous']]
    assert not differing, differing[:5]


def test_the_sweep_reaches_every_class_and_two_or_more_components_do_change_results(sweep):
    zero = sum(1 for c in sweep if c['components'] == 0)
    one = sum(1 for c in sweep if c['components'] == 1)
    many = [c for c in sweep if c['components'] >= 2]
    changed = [c for c in many if c['new'] != c['previous']]
    print(f'\nsweep: {len(sweep)} pairs: {zero} with 0 components, {one} with 1, {len(many)} with 2 or more; '
          f'{len(changed)} of the {len(many)} changed')

    assert zero >= MIN_ZERO_COMPONENT and one >= MIN_ONE_COMPONENT and len(many) >= MIN_TWO_OR_MORE, (zero, one, len(many))
    assert len(changed) >= MIN_CHANGED_TWO_OR_MORE, len(changed)


# ── 4. symmetry ──────────────────────────────────────────────────────────────────────


def test_pet_is_the_same_whichever_agent_is_passed_first(sweep):
    # Passes on main on purpose: the inner max has always made it symmetric. It guards the
    # new code, whose component loop must not reintroduce an a/b asymmetry.
    asymmetric = [(c['new'], c['new_reversed']) for c in sweep if c['new'] != c['new_reversed']]
    assert not asymmetric, asymmetric[:5]


# ── 5. which pieces form one component ───────────────────────────────────────────────


def _components(zone):
    from src.danger.pet_engine import _conflict_components
    return _conflict_components(zone)


def _rect(x0, y0, x1, y1):
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


@pytest.mark.parametrize('label, parts, expected', [
    ('edge-touching polygons', [_rect(0, 0, 2, 2), _rect(2, 0, 4, 2)], 1),
    ('corner-touching polygons', [_rect(0, 0, 2, 2), _rect(2, 2, 4, 4)], 1),
    ('overlapping polygons', [_rect(0, 0, 3, 3), _rect(2, 2, 5, 5)], 1),
    ('a chain of three touching polygons', [_rect(0, 0, 2, 2), _rect(2, 0, 4, 2), _rect(4, 0, 6, 2)], 1),
    ('two separate polygons', [_rect(0, 0, 2, 2), _rect(10, 0, 12, 2)], 2),
    ('two touching and one apart', [_rect(0, 0, 2, 2), _rect(2, 0, 4, 2), _rect(20, 0, 22, 2)], 2),
    ('a line touching a polygon', [_rect(0, 0, 2, 2), LineString([(2, 1), (5, 1)])], 1),
    ('a line apart from a polygon', [_rect(0, 0, 2, 2), LineString([(8, 1), (11, 1)])], 2),
])
def test_pieces_that_touch_are_one_component(label, parts, expected):
    zone = GeometryCollection(parts)
    components = _components(zone)

    assert len(components) == expected, label
    assert all(not c.is_empty for c in components)
    # Grouping only regroups: the union of the components is the union of the parts.
    assert unary_union(components).equals(unary_union(parts))


def test_a_multipolygon_splits_into_its_polygons_and_a_single_piece_stays_whole():
    assert len(_components(MultiPolygon([_rect(0, 0, 2, 2), _rect(9, 9, 11, 11)]))) == 2
    single = _rect(0, 0, 2, 2)
    assert [c.equals(single) for c in _components(single)] == [True]
    assert _components(GeometryCollection()) == []


# ── 6. empty zones, absent visitors, slivers ─────────────────────────────────────────


def test_an_empty_zone_is_pet_infinity():
    # Passes on main on purpose.
    states, validity = _scene(10, {0: _frames(range(10), -50.0), 1: _frames(range(10), 50.0)})
    assert compute_pet_pair(states, validity, 0, 1, path_a=_box(-50.0, 0.0), path_b=_box(50.0, 0.0)) == PET_INFINITY


def test_a_zone_one_agent_never_occupies_is_pet_infinity():
    # Passes on main on purpose. The swept paths overlap, but agent B's boxes never reach it.
    states, validity = _scene(10, {0: _frames(range(10), 0.0), 1: _frames(range(10), 300.0)})
    zone_polygon = _box(0.0, 0.0, size=6.0)
    assert compute_pet_pair(states, validity, 0, 1, path_a=zone_polygon, path_b=zone_polygon) == PET_INFINITY


def _sliver(x):
    """A strip 1 mm wide that no 2 m box in these scenes touches."""
    return Polygon([(x, -1.0), (x + 0.001, -1.0), (x + 0.001, 1.0), (x, 1.0)])


def test_a_sliver_neither_agent_touches_is_skipped_beside_a_real_component():
    # Passes on main on purpose: its visits from the union are the same. It is what makes the
    # "skip a component nobody visits" mutant crash (min over an empty sequence).
    states, validity = _scene(30, {0: _frames(range(0, 5), 0.0), 1: _frames(range(20, 25), 0.0)})
    real = _box(0.0, 0.0)
    with_sliver = MultiPolygon([real, _sliver(500.0)])

    alone = compute_pet_pair(states, validity, 0, 1, path_a=real, path_b=real)
    beside = compute_pet_pair(states, validity, 0, 1, path_a=with_sliver, path_b=with_sliver)

    assert beside == alone == pytest.approx((20 - 4) * DT)


def test_a_zone_made_only_of_slivers_is_pet_infinity_and_does_not_crash():
    states, validity = _scene(10, {0: _frames(range(10), 0.0), 1: _frames(range(10), 0.0)})
    slivers = MultiPolygon([_sliver(500.0), _sliver(900.0)])
    assert compute_pet_pair(states, validity, 0, 1, path_a=slivers, path_b=slivers) == PET_INFINITY


def test_a_component_only_one_agent_visits_is_skipped_not_paired_across():
    """
    A visits the left component early [0, 4] and the right component late [84, 88]. B visits
    only the left component, late [84, 88]. The old pairing matched A's late RIGHT visit with
    B's late LEFT visit (same frames, different places) and returned a negative PET. Only the
    left component has both agents; there they are 84 - 4 frames apart.
    """
    states, validity = _scene(89, {
        0: {**_frames(EARLY, -10.0), **_frames(LATE, 10.0)},
        1: _frames(LATE, -10.0),
    })
    two_zones = MultiPolygon([_box(-10.0, 0.0), _box(10.0, 0.0)])

    pet = compute_pet_pair(states, validity, 0, 1, path_a=two_zones, path_b=two_zones)

    assert pet == pytest.approx((LATE[0] - EARLY[-1]) * DT)


@pytest.mark.parametrize('flip', [1.0, -1.0], ids=['left_is_first', 'right_is_first'])
def test_the_smallest_gap_over_all_components_is_the_answer_wherever_it_sits(flip):
    # Passes on main on purpose (cross-place pairs are far apart in time here). It exists so
    # that a loop returning the LAST component's value, or the largest, cannot pass: the small
    # gap is in the second component for one orientation and in the first for the other.
    states, validity = _scene(89, {
        0: {**_frames(EARLY, flip * -10.0), **_frames(range(40, 45), flip * 10.0)},
        1: {**_frames(LATE, flip * -10.0), **_frames(range(50, 55), flip * 10.0)},
    })
    two_zones = MultiPolygon([_box(-10.0, 0.0), _box(10.0, 0.0)])

    pet = compute_pet_pair(states, validity, 0, 1, path_a=two_zones, path_b=two_zones)

    # Component with A at [40, 44] and B at [50, 54]: 50 - 44 frames. The other is 84 - 4.
    assert pet == pytest.approx((50 - 44) * DT)


# ── 7. the earlier visit-gap behaviour (audit B06) still holds ───────────────────────


def test_a_visit_gap_inside_one_component_is_not_bridged():
    # Passes on main on purpose. Agent A is in the zone at frames 0-1 and 5-6, B only at
    # frame 3: no frame has both, so no negative PET. Both visit pairs are 3 - 1 = 2 and
    # 5 - 3 = 2 frames apart; the old merged-span answer would have been negative.
    plan_a = {**_frames([0, 1, 5, 6], 0.0), **_frames([2, 3, 4], 300.0)}
    states, validity = _scene(10, {0: plan_a, 1: {3: 0.0}})
    zone = _box(0.0, 0.0, size=6.0)

    pet = compute_pet_pair(states, validity, 0, 1, path_a=zone, path_b=zone)

    assert pet == pytest.approx(2 * DT)
