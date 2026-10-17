"""
sensitivity.py — what the API serves of a recorded sensitivity slice (GET /scenarios/{id}/slice). Pure Python: the deployed API has no numpy.

A row of `sensitivity_slices` is a two-dimensional cut through the four-dimensional space of perturbations, through a stored result
(src/scoring/sensitivity.py draws it, by hand). It is served field by field through the allow-list below: each field has a type, a
value of another type is null, and a grid whose arrays do not fit together is not served as a grid at all. The API does not decide
what a cell means: it passes on the verdicts the offline pass recorded.

Not here, on purpose: the pipeline run id, the creation time, the scene fingerprint and everything else the table holds.
"""

import math

PARAMETERIZATIONS = ('bicycle', 'linear')
COMPONENTS = 4
MIN_POINTS = 2
MAX_POINTS = 101
#: the most slices one response lists: every ordered pair of two different components out of four (4 * 3 = 12), which is also the most the table's
#: unique key (scenario, stored result, axis_x, axis_y) allows around one stored result. With this value `truncated` cannot become true unless that
#: key is ever relaxed; it is kept as the guard, not as a state to expect.
MAX_SLICES = 12


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def select_parameterization(v):
    return v if isinstance(v, str) and v in PARAMETERIZATIONS else None


def _axis(v):
    return v if _is_int(v) and 0 <= v < COMPONENTS else None


def _values(v):
    """An axis: MIN_POINTS to MAX_POINTS finite numbers, strictly increasing; otherwise null."""
    if not isinstance(v, (list, tuple)) or not MIN_POINTS <= len(v) <= MAX_POINTS or not all(_finite(x) for x in v):
        return None
    if not all(a < b for a, b in zip(v, v[1:])):
        return None
    return [float(x) for x in v]


def _norms(v, cells):
    """The weighted norms, one per cell: finite and not negative, else null for that cell; null for the lot if the count is wrong."""
    if cells is None or not isinstance(v, (list, tuple)) or len(v) != cells:
        return None
    return [float(x) if _finite(x) and x >= 0 else None for x in v]


def _verdicts(v, cells):
    if cells is None or not isinstance(v, (list, tuple)) or len(v) != cells:
        return None
    return [x if isinstance(x, bool) else None for x in v]


def _held(v):
    if not isinstance(v, (list, tuple)) or len(v) != COMPONENTS or not all(_finite(x) for x in v):
        return None
    return [float(x) for x in v]


def _index_of(values, target):
    if values is None or target is None:
        return None
    for i, x in enumerate(values):
        if x == target:
            return i
    return None


def select_slice(row) -> dict:
    """One row of sensitivity_slices (a mapping with the columns the route selects) as the API serves it."""
    axis_x, axis_y = _axis(row.get('axis_x')), _axis(row.get('axis_y'))
    if axis_x is not None and axis_x == axis_y:
        axis_x = axis_y = None                       # one component against itself is not a slice
    x_values, y_values = _values(row.get('x_values')), _values(row.get('y_values'))
    cells = len(x_values) * len(y_values) if x_values is not None and y_values is not None else None
    held = _held(row.get('held_delta'))
    stored_x = _index_of(x_values, held[axis_x]) if held is not None and axis_x is not None else None
    stored_y = _index_of(y_values, held[axis_y]) if held is not None and axis_y is not None else None
    slice_id = row.get('slice_id')
    return {
        'slice_id': slice_id if _is_int(slice_id) and slice_id >= 0 else None,
        'axis_x': axis_x,
        'axis_y': axis_y,
        'delta_parameterization': select_parameterization(row.get('delta_parameterization')),
        'x_values': x_values,
        'y_values': y_values,
        'weighted_norm': _norms(row.get('weighted_norm'), cells),
        'collides': _verdicts(row.get('collides'), cells),
        'held_delta': held,
        'stored_x_index': stored_x,
        'stored_y_index': stored_y,
    }
