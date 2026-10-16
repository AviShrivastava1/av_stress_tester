"""
search_runs.py — the offline recorder of searches that are not the stored result (the tables search_runs and search_traces of
deploy/rerun_class_additive.sql). This file holds the first of them: the convergence trace.

A stored result was searched once and nothing was kept of how the search went. The trace is the answer to "how fast did it close in?":
the smallest weighted norm of an exactly verified collision after each generation of Differential Evolution. To get it, the search is
run again, with the stored result's own seed and budget, and a callback that only records (optimize_scenario(..., trace=[]); the
search is bit for bit the same with and without it, pinned by tests/test_convergence_trace.py).

WHAT THE TRACE IS, AND WHEN IT MAY BE SHOWN.
  * It belongs to the re-run, not to the stored result. The row is kind 'trace_rerun' and names the stored result it re-executes
    (`reruns_stress_run_id`) and how far the re-run's final delta is from the stored delta (`max_abs_delta_difference`, the largest
    absolute difference of any of the four components). The API serves the trace only when that difference is below
    src/api/trace.py TRACE_DELTA_TOLERANCE: a re-run that ended somewhere else did not follow the stored search, and its curve says
    nothing about it.
  * It is the Differential Evolution stage only. When the stored result was refined by the gradient stage afterwards ('de+autograd'),
    the stored norm can be below the trace's last point; the re-run runs the same refinement, so the deltas still compare.
  * It writes only search_runs and search_traces. It never touches scenario_scores, scenario_agents or perturbed_paths, never calls
    update_stress_results, and the stored result is not changed by anything here. Promoting a recorded search to the stored result
    is not done here, and no API route does it.
  * It does not create the tables. A missing table is a clear error.
"""

import json
import math

import numpy as np

from src.scoring.manifest import ManifestTablesMissing

_TABLES = ('search_runs', 'search_traces')


class TraceRerunRefused(RuntimeError):
    """The stored result cannot be re-run as it was stored, so no trace is recorded. The message says why."""


def _require_tables(cur):
    for table in _TABLES:
        cur.execute('SELECT to_regclass(%s) IS NOT NULL', (table,))
        if not cur.fetchone()[0]:
            raise ManifestTablesMissing(f'{table} is not in this database; run deploy/rerun_class_additive.sql by hand (on a copy first).')


def max_abs_delta_difference(stored, rerun) -> float:
    """The largest absolute difference between two deltas, component by component, in float64. inf when they differ in length."""
    a = np.asarray(stored, dtype=np.float64)
    b = np.asarray(rerun, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        return math.inf
    return float(np.max(np.abs(a - b)))


def _stored_result(cur, scenario_id):
    cur.execute(
        """SELECT scenario_id, stress_run_id, scene_fingerprint, sdc_idx, target_idx, stress_method, delta, search_provenance
             FROM scenario_scores WHERE scenario_id = %s""", (scenario_id,))
    row = cur.fetchone()
    if row is None:
        raise TraceRerunRefused(f'{scenario_id}: no such scenario')
    cols = ('scenario_id', 'stress_run_id', 'scene_fingerprint', 'sdc_idx', 'target_idx', 'stress_method', 'delta', 'search_provenance')
    return dict(zip(cols, row))


def record_trace_rerun(conn, scenario_id, states, validity, types, sdc_idx, *, pipeline_run_id=None) -> dict:
    """
    Re-run the stored search of `scenario_id` with a trace and record it. `states`, `validity`, `types` and `sdc_idx` are the scene as
    parsed from the shard now. Returns {'search_run_id', 'max_abs_delta_difference', 'generations', 'matches_stored', 'inserted', 'will_be_served'} and commits. `inserted` is False when a row
    with this id already exists, of any kind (the primary key); then nothing of this re-run is written.

    Refuses (TraceRerunRefused, nothing written) when there is no stored search, when the stored scene fingerprint or SDC index is not
    the one given (the scene is not the one searched), when the stored provenance lacks the seed and budget, or when the re-run itself
    does not produce a searched result. It does not refuse a re-run that ended somewhere else: that row is recorded with its
    difference, and the API will not show its trace.
    """
    from src.api.trace import TRACE_DELTA_TOLERANCE
    from src.scoring.batch_scorer import _stress_one
    from src.scoring.db import _json_safe, compute_scene_fingerprint, compute_stress_run_id

    with conn.cursor() as cur:
        _require_tables(cur)
        stored = _stored_result(cur, scenario_id)

    if stored['stress_run_id'] is None or stored['delta'] is None or not isinstance(stored['search_provenance'], dict):
        raise TraceRerunRefused(f'{scenario_id}: no stored search to re-run')
    if stored['scene_fingerprint'] is not None and stored['scene_fingerprint'] != compute_scene_fingerprint(states, validity, types):
        raise TraceRerunRefused(f'{scenario_id}: the scene given is not the scene the stored result was searched against')
    if stored['sdc_idx'] is not None and int(stored['sdc_idx']) != int(sdc_idx):
        raise TraceRerunRefused(f'{scenario_id}: the SDC index given is not the one the stored result was searched with')
    prov = stored['search_provenance']
    try:
        de_kwargs = {'popsize': int(prov['de_popsize']), 'maxiter': int(prov['de_maxiter']),
                     'tol': float(prov['de_tol']), 'seed': int(prov['de_seed'])}
    except (KeyError, TypeError, ValueError):
        raise TraceRerunRefused(f'{scenario_id}: the stored provenance does not hold the seed and the budget') from None
    # The gradient stage ran if and only if it recorded its iteration count (refine_n_iters is None otherwise).
    use_autograd = prov.get('refine_n_iters') is not None

    trace = []
    result = _stress_one(states, validity, types, sdc_idx, use_autograd=use_autograd, de_kwargs={**de_kwargs, 'trace': trace})
    if result.get('status') != 'ok' or result.get('delta') is None:
        raise TraceRerunRefused(f'{scenario_id}: the re-run did not run a search ({result.get("status")})')
    if int(result['target_idx']) != int(stored['target_idx'] if stored['target_idx'] is not None else result['target_idx']):
        raise TraceRerunRefused(f'{scenario_id}: the re-run searched a different challenger than the stored result')

    difference = max_abs_delta_difference(stored['delta'], result['delta'])
    search_run_id = compute_stress_run_id(scenario_id, result['target_idx'], result['delta'], result['method'], result['search_provenance'])
    rerun_prov = result['search_provenance']
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO search_runs (search_run_id, scenario_id, scene_fingerprint, sdc_idx, target_idx, delta_parameterization,
                                        kind, seed, popsize, maxiter, tol, outcome, min_perturbation, collision_timestep, delta,
                                        n_iter, n_eval, search_provenance, reruns_stress_run_id, max_abs_delta_difference,
                                        pipeline_run_id)
               VALUES (%s, %s, %s, %s, %s, %s, 'trace_rerun', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
               ON CONFLICT (search_run_id) DO NOTHING""",
            (search_run_id, scenario_id, compute_scene_fingerprint(states, validity, types), int(sdc_idx), int(result['target_idx']),
             rerun_prov.get('delta_parameterization'), de_kwargs['seed'], de_kwargs['popsize'], de_kwargs['maxiter'], de_kwargs['tol'],
             result['outcome'],
             float(result['min_perturbation']) if result.get('collision') and math.isfinite(result['min_perturbation']) else None,
             int(result['collision_timestep']) if result.get('collision') else None,
             [float(x) for x in result['delta']], rerun_prov.get('de_n_iter'), rerun_prov.get('de_n_eval'),
             json.dumps(_json_safe(rerun_prov), allow_nan=False), stored['stress_run_id'],
             difference if math.isfinite(difference) else None, pipeline_run_id))
        inserted = cur.rowcount == 1
        if inserted:
            cur.executemany(
                """INSERT INTO search_traces (search_run_id, generation, n_eval_cumulative, best_weighted_norm) VALUES (%s, %s, %s, %s)""",
                [(search_run_id, p['generation'], p['n_eval_cumulative'], p['best_weighted_norm']) for p in trace])
    conn.commit()
    return {'search_run_id': search_run_id, 'max_abs_delta_difference': difference, 'generations': len(trace),
            'matches_stored': search_run_id == stored['stress_run_id'], 'inserted': inserted,
            # What the database holds now, not what this call computed: True only if a trace re-run row with this id names the stored
            # result, is under the tolerance and has trace points, i.e. what GET /scenarios/{id}/trace would pick (src/api/trace.py).
            # A row of another kind that already has the id (search_run_id is the primary key) keeps it, nothing is inserted and this is False.
            'will_be_served': _is_servable(conn, scenario_id, search_run_id, stored['stress_run_id'], TRACE_DELTA_TOLERANCE)}


def _is_servable(conn, scenario_id, search_run_id, stored_stress_run_id, tolerance) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT EXISTS (
                   SELECT 1 FROM search_runs r
                   WHERE r.search_run_id = %s AND r.scenario_id = %s AND r.kind = 'trace_rerun' AND r.reruns_stress_run_id = %s
                     AND r.max_abs_delta_difference >= 0 AND r.max_abs_delta_difference < %s
                     AND EXISTS (SELECT 1 FROM search_traces t WHERE t.search_run_id = r.search_run_id))""",
            (search_run_id, scenario_id, stored_stress_run_id, tolerance))
        return bool(cur.fetchone()[0])
