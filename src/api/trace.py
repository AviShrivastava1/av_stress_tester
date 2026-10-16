"""
trace.py — what the API serves of a recorded convergence trace (GET /scenarios/{id}/trace). Pure Python: the deployed API has no numpy.

A trace is served only for the stored result it re-executed, and only when the re-run ended where the stored search ended.

TRACE_DELTA_TOLERANCE = 1e-6 is the largest absolute difference, in any of the four components of the perturbation, between the stored
delta and the re-run's. Why that number: the components are float32 values (spacing about 1.2e-7 per unit), and a re-run on the same
machine and library versions with the same seed and budget returns the same delta exactly, so a difference of zero is the normal case.
Anything beyond a few float32 steps means Differential Evolution took a different path (another library build, another floating-point
reduction order) and ended somewhere else; its curve then describes a different search, however alike the final numbers look. The
bound is a gate and not a measurement: the response carries the difference itself, so a reader can see how close it was.
"""

import math

TRACE_DELTA_TOLERANCE = 1e-6

#: Differential Evolution runs at most `maxiter` generations (200 by default); a trace longer than this is not one this pipeline wrote.
MAX_POINTS = 2000


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def select_points(rows) -> list[dict]:
    """
    The points of a trace in generation order, each as {generation, n_eval_cumulative, best_weighted_norm}. A point whose generation or
    evaluation count is not a non-negative integer is dropped; a best norm that is not a finite number is null (no collision found yet).
    """
    points = []
    for generation, n_eval, best in rows:
        generation, n_eval = _count(generation), _count(n_eval)
        if generation is None or n_eval is None:
            continue
        ok = isinstance(best, (int, float)) and not isinstance(best, bool) and math.isfinite(best)
        points.append({'generation': generation, 'n_eval_cumulative': n_eval, 'best_weighted_norm': float(best) if ok else None})
    points.sort(key=lambda p: p['generation'])
    return points[:MAX_POINTS]
