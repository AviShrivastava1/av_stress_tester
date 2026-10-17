"""
sensitivity.py — the offline runner of a sensitivity slice (#15): the weighted norm and the collision verdict over a grid of two
components of a stored result's perturbation, the other two held at the stored values.

A stored result is one point in a four-dimensional space of perturbations. A slice looks at a two-dimensional cut through that space,
through the stored point, on the same scene and against the same challenger: for every pair of values of the two chosen components it
records whether the replay collides (the exact check, the same one the search used) and the weighted norm of the whole perturbation.
It is a picture of a cut, not of the space: collisions can exist between the grid's cells and in the components that are held, and a
cell without a collision shows only that this perturbation did not collide.

Rules, each pinned by tests/test_sensitivity.py:

  * Run by hand, like the other offline passes. It writes only `sensitivity_slices` (deploy/rerun_class_additive.sql). It never touches
    scenario_scores, scenario_agents or perturbed_paths, never calls update_stress_results and never changes what a result's identity
    is. The slice carries the `stress_run_id` it is drawn around; the API shows it only while that is still the stored result's.
  * Everything that can be checked without evaluating anything is checked first and refuses with a clear error: the axes (two different
    components, 0 to 3), the grid sizes (2 to 101 per axis, as the table enforces), the ranges (finite, low below high, inside the
    search's bounds, containing the stored value), and the scene (search_runs.load_stored_search: the scene and SDC index must be the
    ones the stored result was searched against, and it must record its seed and budget). The space this runner builds must be the stored
    result's own: the same kinematic model as its `delta_parameterization`, and the same weighted norm at the stored point as its
    `min_perturbation`. The stored point is evaluated before the grid and must collide, as the stored result does; if it does not, the slice
    would not describe the stored result and nothing is written.
  * The grid contains the stored point exactly: the grid value nearest to the stored component is replaced by the stored value (a shift
    of less than half a step), so the stored point is a cell and not an interpolation.
  * A slice of the same result and the same two axes that is already recorded is left as it is and reported as a conflict, before any
    evaluation (the table's unique key). To replace a slice, delete the row by hand.
  * Which model's four components these are (`delta_parameterization`) is read from the stored provenance and recorded; a stored
    result that does not record it is refused, because the table requires it.
"""

import math

import numpy as np

from src.scoring.manifest import ManifestTablesMissing
from src.scoring.search_runs import TraceRerunRefused, load_stored_search

MIN_POINTS = 2
MAX_POINTS = 101
COMPONENTS = 4
#: How far the weighted norm of the stored perturbation, recomputed here, may be from the stored `min_perturbation` (relative; both are the same
#: arithmetic on the same float32 values, so a larger difference means different weights, bounds or code)
NORM_TOLERANCE = 1e-6


class SliceRefused(TraceRerunRefused):
    """The slice cannot be drawn as asked, or cannot be drawn around the stored result. Nothing is written."""


def _require_table(cur):
    cur.execute('SELECT to_regclass(%s) IS NOT NULL', ('sensitivity_slices',))
    if not cur.fetchone()[0]:
        raise ManifestTablesMissing('sensitivity_slices is not in this database; run deploy/rerun_class_additive.sql by hand (on a copy first).')


def check_request(axis_x, axis_y, nx, ny, x_range, y_range):
    """The form of a request, before anything is read or evaluated. Raises SliceRefused."""
    for name, axis in (('axis_x', axis_x), ('axis_y', axis_y)):
        if isinstance(axis, bool) or not isinstance(axis, int) or not 0 <= axis < COMPONENTS:
            raise SliceRefused(f'{name} is a component number from 0 to {COMPONENTS - 1}, not {axis!r}')
    if axis_x == axis_y:
        raise SliceRefused('the two axes must be two different components')
    for name, n in (('nx', nx), ('ny', ny)):
        if not isinstance(n, int) or not MIN_POINTS <= n <= MAX_POINTS:        # a boolean is 0 or 1, below the minimum, so it is refused here too
            raise SliceRefused(f'{name} is a number of grid points from {MIN_POINTS} to {MAX_POINTS}, not {n!r}')
    for name, rng in (('x_range', x_range), ('y_range', y_range)):
        if rng is None:
            continue
        ok = isinstance(rng, (tuple, list)) and len(rng) == 2 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in rng)
        if not ok or not rng[0] < rng[1]:
            raise SliceRefused(f'{name} is (low, high) with finite numbers and low below high, not {rng!r}')


def grid_through(stored_value, low, high, n):
    """n values from low to high that include stored_value exactly: an even grid whose nearest value is replaced by the stored one."""
    if not low <= stored_value <= high:
        raise SliceRefused(f'the range ({low}, {high}) does not contain the stored value {stored_value}')
    values = np.linspace(low, high, n)
    values[int(np.argmin(np.abs(values - stored_value)))] = stored_value
    return [float(v) for v in values]


def record_sensitivity_slice(conn, scenario_id, states, validity, types, sdc_idx, *, axis_x, axis_y, nx=21, ny=21,
                             x_range=None, y_range=None, pipeline_run_id=None) -> dict:
    """
    Draw and record one slice of the stored result of `scenario_id`: components `axis_x` (horizontal) and `axis_y` (vertical) over
    `nx` by `ny` values, the other two held at the stored values. `x_range`, `y_range` default to the search's bounds for the
    component. Returns {'status': 'recorded' or 'conflict', 'slice_id', ...} and commits. See the module docstring for what it refuses.
    """
    from src.optimization.perturbation_space import HeadingBlendSingularityError, PerturbationSpace, ReplayFidelityError
    from src.danger.collision_detector import check_collision_trajectory

    check_request(axis_x, axis_y, nx, ny, x_range, y_range)
    stored, _, _ = load_stored_search(conn, scenario_id, states, validity, types, sdc_idx)
    prov = stored['search_provenance']
    parameterization = prov.get('delta_parameterization')
    if parameterization not in ('bicycle', 'linear'):
        raise SliceRefused(f'{scenario_id}: the stored result does not record which model its perturbation is in')
    if stored['target_idx'] is None:
        raise SliceRefused(f'{scenario_id}: the stored result does not record which challenger it was searched against')
    held = [float(v) for v in stored['delta']]
    if len(held) != COMPONENTS:
        raise SliceRefused(f'{scenario_id}: the stored perturbation has {len(held)} components, not {COMPONENTS}')

    with conn.cursor() as cur:
        _require_table(cur)
        cur.execute('SELECT min_perturbation FROM scenario_scores WHERE scenario_id = %s', (scenario_id,))
        recorded_norm = cur.fetchone()[0]
        cur.execute('SELECT slice_id FROM sensitivity_slices WHERE scenario_id = %s AND stress_run_id = %s AND axis_x = %s AND axis_y = %s',
                    (scenario_id, stored['stress_run_id'], axis_x, axis_y))
        existing = cur.fetchone()
    if existing is not None:
        return {'status': 'conflict', 'slice_id': existing[0], 'reason': 'a slice of this result and these axes is already recorded'}

    try:
        space = PerturbationSpace(states, validity, types, sdc_idx, int(stored['target_idx']))
    except (ReplayFidelityError, HeadingBlendSingularityError) as e:
        raise SliceRefused(f'{scenario_id}: the replay of the challenger is not faithful enough to draw a slice around: {e}') from None
    # The space this runner builds must be the stored result's own: the same kinematic model (so the four components mean what the stored
    # `delta_parameterization` says) and, below, the same weighted norm at the stored point. Otherwise the slice would put one model's labels
    # on another's numbers.
    space_model = 'bicycle' if space.is_vehicle else 'linear'
    if space_model != parameterization:
        raise SliceRefused(f'{scenario_id}: the stored result is recorded as the {parameterization} model but its challenger is replayed with the '
                           f'{space_model} model here, so a slice would put one model\'s labels on the other\'s numbers')
    bounds = np.asarray(space.bounds, dtype=np.float64)
    ranges = []
    for axis, rng in ((axis_x, x_range), (axis_y, y_range)):
        low, high = (float(bounds[axis][0]), float(bounds[axis][1])) if rng is None else (float(rng[0]), float(rng[1]))
        if low < float(bounds[axis][0]) or high > float(bounds[axis][1]):
            raise SliceRefused(f'component {axis}: the range ({low}, {high}) is outside the search bounds ({bounds[axis][0]}, {bounds[axis][1]})')
        ranges.append((low, high))
    xs = grid_through(held[axis_x], ranges[0][0], ranges[0][1], nx)
    ys = grid_through(held[axis_y], ranges[1][0], ranges[1][1], ny)

    def verdict(delta):
        delta = np.asarray(delta, dtype=np.float32)
        collided, _ = check_collision_trajectory(space.apply(delta), validity, sdc_idx, int(stored['target_idx']))
        return bool(collided), float(space.weighted_norm(delta))

    stored_collides, stored_norm = verdict(held)
    if not stored_collides:
        raise SliceRefused(f'{scenario_id}: the stored perturbation does not collide in this code, so a slice would not describe the stored result')
    if recorded_norm is None or not math.isfinite(recorded_norm):
        raise SliceRefused(f'{scenario_id}: the stored result does not record its weighted norm, so the slice cannot be checked against it')
    if abs(stored_norm - recorded_norm) > NORM_TOLERANCE * max(1.0, abs(recorded_norm)):
        raise SliceRefused(f'{scenario_id}: the weighted norm of the stored perturbation is {stored_norm:.9g} here and {recorded_norm:.9g} as stored, '
                           'so the weights or bounds are not the ones it was searched with and a slice would not describe the stored result')

    norms, collides = [], []
    for y in ys:
        for x in xs:
            point = list(held)
            point[axis_x], point[axis_y] = x, y
            hit, norm = verdict(point)
            collides.append(hit)
            norms.append(norm)

    from src.scoring.db import compute_scene_fingerprint
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO sensitivity_slices (scenario_id, stress_run_id, scene_fingerprint, sdc_idx, axis_x, axis_y, delta_parameterization,
                                               x_values, y_values, weighted_norm, collides, held_delta, pipeline_run_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (scenario_id, stress_run_id, axis_x, axis_y) DO NOTHING
               RETURNING slice_id""",
            (scenario_id, stored['stress_run_id'], compute_scene_fingerprint(states, validity, types), int(sdc_idx), axis_x, axis_y,
             parameterization, xs, ys, norms, collides, held, pipeline_run_id))
        row = cur.fetchone()
    conn.commit()
    if row is None:
        return {'status': 'conflict', 'slice_id': None, 'reason': 'a slice of this result and these axes was recorded meanwhile'}
    return {'status': 'recorded', 'slice_id': row[0], 'nx': nx, 'ny': ny, 'cells': nx * ny, 'collisions': int(sum(collides)),
            'stored_point_collides': stored_collides}
