"""
other_searches.py — what the API serves of the other searches of a scenario (GET /scenarios/{id}/other-searches). Pure Python: the
deployed API has no numpy.

A row of `search_runs` of kind 'seed', 'budget' or 'challenger' is a separate search of a scenario that is not the stored result (the
offline runners are src/scoring/other_searches.py; the rules they and this follow are in SCHEMA-PLAN.md, "Ids that can coincide"). It is
served field by field through the allow-list below: each field has a type, a value of another type is null, and a row that is not one
of the three kinds is not served at all. Nothing here ranks the searches, and nothing marks one as the smallest.

Not here, on purpose: the search provenance of a row (the JSON blob), the pipeline run id, and everything else the table holds.
"""

import math
import re

KINDS = ('seed', 'budget', 'challenger')
OUTCOMES = ('collision_found', 'no_collision_found')
PARAMETERIZATIONS = ('bicycle', 'linear')
#: the most rows one response lists; when more matched, the response says so (`truncated`) and lists the first of them in the fixed order
MAX_SEARCHES = 200
#: the most components a served delta may have (the models in use have four)
MAX_COMPONENTS = 8

_ID = re.compile(r'[0-9a-f]{16}')


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _count(v, low=0):
    return v if _is_int(v) and v >= low else None


def _number(v, low=0.0):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= low else None


def select_parameterization(v):
    return v if isinstance(v, str) and v in PARAMETERIZATIONS else None


def _delta(v):
    if not isinstance(v, (list, tuple)) or not 1 <= len(v) <= MAX_COMPONENTS:
        return None
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v):
        return None
    return [float(x) for x in v]


def select_search(row) -> dict | None:
    """One row of search_runs (a mapping with the columns the route selects) as the API serves it, or None if its kind is not served."""
    kind = row.get('kind')
    if not isinstance(kind, str) or kind not in KINDS:
        return None
    search_run_id = row.get('search_run_id')
    outcome = row.get('outcome')
    return {
        'search_run_id': search_run_id if isinstance(search_run_id, str) and _ID.fullmatch(search_run_id) else None,
        'kind': kind,
        'challenger_rank': _count(row.get('challenger_rank'), 1),
        'target_idx': _count(row.get('target_idx')),
        'seed': _count(row.get('seed')),
        'popsize': _count(row.get('popsize'), 1),
        'maxiter': _count(row.get('maxiter'), 1),
        'tol': _number(row.get('tol')),
        'outcome': outcome if isinstance(outcome, str) and outcome in OUTCOMES else None,
        'min_perturbation': _number(row.get('min_perturbation')),
        'collision_timestep': _count(row.get('collision_timestep')),
        'delta': _delta(row.get('delta')),
        'delta_parameterization': select_parameterization(row.get('delta_parameterization')),
        'n_iter': _count(row.get('n_iter')),
        'n_eval': _count(row.get('n_eval')),
    }
