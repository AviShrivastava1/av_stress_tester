"""
provenance.py — which parts of a stored result's search provenance the API serves, and how.

`scenario_scores.search_provenance` is a JSONB blob written by the offline search (src/scoring/batch_scorer.py). It is NOT served
whole: a column added to it tomorrow must not become public by being there. This module is the allow-list: each served key, its
type, and nothing else. A stored value of the wrong type (a JSON string where a number belongs, an infinity that survived into
the blob, a boolean for a count) is served as null, never repaired and never a 500, because "not recorded" is an honest answer
and a guess is not.

Not here, on purpose: baseline_offset_at_collision and baseline_replay_error (already served on GET /scenarios/{id}),
and everything the blob holds that is not listed below.
"""

import math

#: kind -> predicate. bool is excluded from the numeric kinds: a JSON true is not a count.
def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_str(v):
    return isinstance(v, str)


def _is_bool(v):
    return isinstance(v, bool)


def _is_bounds(v):
    return (isinstance(v, list) and all(isinstance(p, list) and len(p) == 2 and all(_is_number(x) for x in p) for p in v))


KINDS = {'str': _is_str, 'int': _is_int, 'number': _is_number, 'bool': _is_bool, 'bounds': _is_bounds}

#: The served keys, in the order the response lists them, with the kind each must have.
SERVED = [
    # which code produced the result, and which model's units its four components carry
    ('replay_model_version', 'str'),
    ('collision_geometry_version', 'str'),
    ('delta_parameterization', 'str'),
    ('challenger_selection', 'str'),
    ('selected_stage', 'str'),
    # the search's budget, seed and what it did with them
    ('de_popsize', 'int'),
    ('de_maxiter', 'int'),
    ('de_tol', 'number'),
    ('de_seed', 'int'),
    ('de_n_iter', 'int'),
    ('de_n_eval', 'int'),
    ('de_candidate_source', 'str'),
    ('de_archive_covered_all_evaluations', 'bool'),
    ('refine_n_iters', 'int'),
    # the bounds the search was confined to, one [low, high] per component of the perturbation
    ('bounds', 'bounds'),
    # the gates the result was admitted under
    ('max_baseline_drift', 'number'),
    ('max_speed_step', 'number'),
    ('speed_step_hold_s', 'number'),
    ('held_speed_excess', 'number'),
    ('a_max', 'number'),
    ('heading_speed_floor', 'number'),
    ('heading_transition_width', 'number'),
    # facts about the challenger the search was run on
    ('target_min_speed', 'number'),
    ('frames_observed', 'int'),
    ('frames_below_heading_floor', 'int'),
    ('frames_in_heading_transition_band', 'int'),
    ('target_has_interior_gap', 'bool'),
]

KEYS = [key for key, _ in SERVED]


def select_search_record(provenance) -> dict | None:
    """
    The served keys of a stored search_provenance, each as stored or null. None when there is no provenance at all (not a
    mapping), so that "nothing stored" and "stored, with some keys missing" are different answers.
    """
    if not isinstance(provenance, dict):
        return None
    return {key: (provenance[key] if key in provenance and KINDS[kind](provenance[key]) else None) for key, kind in SERVED}
