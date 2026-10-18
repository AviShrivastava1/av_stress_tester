"""
other_searches.py — the offline runners of the other searches of a stored result: other seeds (#16), other budgets (#17) and other
challengers (#1).

A stored result is the outcome of one search, with one seed and one budget, against one challenger. These runners search the same
scenario again: against the same challenger with a different seed (`run_seed_searches`) or a different budget (`run_budget_searches`),
or, with the stored seed and budget, against the k nearest challengers (`run_challenger_searches`). Each search is recorded as a row of
`search_runs` (kind 'seed', 'budget' or 'challenger') so that the site can list them under "Other searches of this scenario". They
exist to show how much the answer depends on the seed, the budget and the challenger chosen. A row says what that search found; it is
not a claim about the smallest perturbation there is, none of them is the stored result, and the figure of a search against another
challenger is a perturbation of a different agent, not comparable with the stored one.

Rules, each pinned by tests/test_other_searches.py:

  * Run by hand, like the other offline passes. They write only `search_runs`. They never touch scenario_scores, scenario_agents or
    perturbed_paths, never call update_stress_results and never change what a result's identity is. Nothing here promotes a search to
    the stored result; that is Pass 2 and Pass 3, by hand.
  * They refuse a scene or an SDC index that is not the one the stored result was searched against, a scenario with no stored search,
    and a stored search that does not record its seed and budget (search_runs.load_stored_search: the same checks as the trace re-run).
  * A configuration is SKIPPED, not recorded, when it is the stored search itself: before running, if its seed and budget are the stored
    ones; after running, if its `search_run_id` (the content hash of the search, db.compute_stress_run_id) equals the stored
    `stress_run_id`. It is the stored result, not an alternative to it.
  * `search_run_id` is the primary key. If a row with the id is already there, of any kind, nothing is inserted and nothing is changed;
    the configuration is reported as a conflict with the kind that holds the id.
  * The gradient stage is run when the stored search ran it, so an alternative is the same pipeline with another seed or budget.
  * No convergence trace is written for these rows (the trace route serves traces of re-runs of the stored search only).
  * Other challengers (`run_challenger_searches`): the challengers are ranked by the ranking the pipeline picks its one from
    (perturbation_space.rank_challengers: the smallest distance between centers over the timesteps both are valid; rank 1 is the
    nearest), each of the first k is searched with the stored seed and budget, and each is recorded with its `challenger_rank` and
    `target_idx` and its own model (a pedestrian is in the linear model). Rank 1 searched with the stored configuration IS the stored
    result when the nearest challenger is the stored target: it is then reported as skipped before any search is run. Otherwise it
    is searched, and any search whose `search_run_id` is the stored `stress_run_id` is skipped after it ran, as the others are. A challenger whose search does not run (its replay is not
    faithful enough, its heading blend is singular) is reported under `not_run` and nothing is recorded for it; the others go on.
    k is refused before anything runs unless it is an integer from 1 to the number of challengers that can be ranked. No norm of one
    challenger is compared with another's or with the stored one here, and nothing is promoted.

Usage, with the scene arrays of the scenario as parsed from the shard:

    report = run_seed_searches(conn, scenario_id, states, validity, types, sdc_idx, seeds=[1, 2, 3, 4])
    report = run_budget_searches(conn, scenario_id, states, validity, types, sdc_idx,
                                 budgets=[{'popsize': 10, 'maxiter': 100}, {'popsize': 30, 'maxiter': 400, 'tol': 1e-4}])
    report['recorded'], report['skipped'], report['conflicts']        # lists of per-configuration reports

Each per-configuration report has the configuration, its status ('recorded', 'skipped' or 'conflict'), and, when a search was run, its
`search_run_id` and outcome.
"""

import json
import math

from src.scoring.search_runs import TraceRerunRefused, load_stored_search

#: A call runs at most this many configurations: each is a full search.
MAX_CONFIGURATIONS = 200


def _seeds(seeds):
    out = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f'a seed is a non-negative integer, not {seed!r}')
        if seed not in out:
            out.append(seed)
    return out


def _budgets(budgets, stored):
    out = []
    for budget in budgets:
        unknown = set(budget) - {'popsize', 'maxiter', 'tol'}
        if unknown:
            raise ValueError(f'a budget has popsize, maxiter and tol, not {sorted(unknown)}')
        popsize = budget.get('popsize', stored['popsize'])
        maxiter = budget.get('maxiter', stored['maxiter'])
        tol = budget.get('tol', stored['tol'])
        for name, value, low in (('popsize', popsize, 2), ('maxiter', maxiter, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f'{name} is an integer of at least {low}, not {value!r}')
        if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not math.isfinite(tol) or tol < 0:
            raise ValueError(f'tol is a non-negative number, not {tol!r}')
        config = {'popsize': popsize, 'maxiter': maxiter, 'tol': float(tol)}
        if config not in out:
            out.append(config)
    return out


def _run(conn, kind, scenario_id, states, validity, types, sdc_idx, make_configs, *, pipeline_run_id):
    from src.scoring.batch_scorer import _stress_one
    from src.scoring.db import _json_safe, compute_scene_fingerprint, compute_stress_run_id

    stored, stored_kwargs, use_autograd = load_stored_search(conn, scenario_id, states, validity, types, sdc_idx)
    configs = make_configs(stored_kwargs)
    if len(configs) > MAX_CONFIGURATIONS:
        raise ValueError(f'at most {MAX_CONFIGURATIONS} configurations in one call, not {len(configs)}')
    fingerprint = compute_scene_fingerprint(states, validity, types)
    report = {'kind': kind, 'scenario_id': scenario_id, 'recorded': [], 'skipped': [], 'conflicts': []}
    nearest = None
    if kind == 'challenger':
        from src.optimization.perturbation_space import rank_challengers
        report['not_run'] = []
        ranked = rank_challengers(states, validity, sdc_idx)
        nearest = ranked[0] if ranked else None

    for config in configs:
        rank = config.get('challenger_rank')                  # only the challenger runner has one
        de = {k: config[k] for k in ('seed', 'popsize', 'maxiter', 'tol')}
        where = {**de, **({} if rank is None else {'challenger_rank': rank})}
        if where == stored_kwargs:                              # (a challenger's `where` carries its rank, so it is never the stored configuration)
            report['skipped'].append({**where, 'status': 'skipped', 'reason': 'same configuration as the stored result'})
            continue
        if rank == 1 and nearest is not None and stored['target_idx'] is not None and int(stored['target_idx']) == int(nearest):
            # The nearest challenger, searched with the stored seed and budget, is the stored result: it is not searched again. (If the
            # nearest is not the stored target, or the stored result does not say its target, it is searched and its id is compared below.)
            report['skipped'].append({**where, 'status': 'skipped', 'target_idx': int(nearest), 'search_run_id': stored['stress_run_id'],
                                      'reason': 'the nearest challenger, searched with the stored seed and budget, is the stored result; not searched again'})
            continue
        result = _stress_one(states, validity, types, sdc_idx, use_autograd=use_autograd, de_kwargs=dict(de),
                             **({} if rank is None else {'challenger_rank': rank}))
        if rank is not None and (result.get('status') != 'ok' or result.get('delta') is None):
            report['not_run'].append({**where, 'status': 'not_run', 'target_idx': result.get('target_idx'),
                                      'reason': f'the search did not run ({result.get("status")})'})
            continue
        if result.get('status') != 'ok' or result.get('delta') is None:
            raise TraceRerunRefused(f'{scenario_id}: a search with {where} did not run ({result.get("status")}); nothing after it was run')
        if rank is None and stored['target_idx'] is not None and int(result['target_idx']) != int(stored['target_idx']):
            raise TraceRerunRefused(f'{scenario_id}: a search with {where} chose another challenger than the stored result; nothing was recorded for it')
        prov = result['search_provenance']
        search_run_id = compute_stress_run_id(scenario_id, result['target_idx'], result['delta'], result['method'], prov)
        if search_run_id == stored['stress_run_id']:
            report['skipped'].append({**where, 'status': 'skipped', 'reason': 'same as the stored result', 'search_run_id': search_run_id,
                                      **({} if rank is None else {'target_idx': int(result['target_idx'])})})
            continue
        collided = bool(result.get('collision'))
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO search_runs (search_run_id, scenario_id, scene_fingerprint, sdc_idx, target_idx, delta_parameterization,
                                            kind, challenger_rank, seed, popsize, maxiter, tol, outcome, min_perturbation, collision_timestep,
                                            delta, n_iter, n_eval, search_provenance, pipeline_run_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                   ON CONFLICT (search_run_id) DO NOTHING""",
                (search_run_id, scenario_id, fingerprint, int(sdc_idx), int(result['target_idx']), prov.get('delta_parameterization'),
                 kind, rank, where['seed'], where['popsize'], where['maxiter'], where['tol'], result['outcome'],
                 float(result['min_perturbation']) if collided and math.isfinite(result['min_perturbation']) else None,
                 int(result['collision_timestep']) if collided else None,
                 [float(x) for x in result['delta']], prov.get('de_n_iter'), prov.get('de_n_eval'),
                 json.dumps(_json_safe(prov), allow_nan=False), pipeline_run_id))
            inserted = cur.rowcount == 1
            held_by = None
            if not inserted:
                cur.execute('SELECT kind FROM search_runs WHERE search_run_id = %s', (search_run_id,))
                held_by = cur.fetchone()[0]
        conn.commit()
        entry = {**where, 'search_run_id': search_run_id, 'outcome': result['outcome'],
                 **({} if rank is None else {'target_idx': int(result['target_idx'])})}
        if inserted:
            report['recorded'].append({**entry, 'status': 'recorded'})
        else:
            report['conflicts'].append({**entry, 'status': 'conflict', 'held_by_kind': held_by})
    return report


def run_seed_searches(conn, scenario_id, states, validity, types, sdc_idx, *, seeds, pipeline_run_id=None) -> dict:
    """Search the scenario again with each of `seeds` and the stored budget; record each as kind 'seed'. See the module docstring."""
    seeds = _seeds(seeds)
    return _run(conn, 'seed', scenario_id, states, validity, types, sdc_idx,
                lambda stored: [{**{k: stored[k] for k in ('popsize', 'maxiter', 'tol')}, 'seed': s} for s in seeds],
                pipeline_run_id=pipeline_run_id)


def run_budget_searches(conn, scenario_id, states, validity, types, sdc_idx, *, budgets, pipeline_run_id=None) -> dict:
    """
    Search the scenario again with each budget in `budgets` (dicts with popsize, maxiter and tol; one left out is the stored one) and
    the stored seed; record each as kind 'budget'. See the module docstring.
    """
    budgets = list(budgets)
    return _run(conn, 'budget', scenario_id, states, validity, types, sdc_idx,
                lambda stored: [{**b, 'seed': stored['seed']} for b in _budgets(budgets, stored)],
                pipeline_run_id=pipeline_run_id)


def run_challenger_searches(conn, scenario_id, states, validity, types, sdc_idx, *, k, pipeline_run_id=None) -> dict:
    """
    Search the scenario against each of the k nearest challengers with the stored seed and budget; record each as kind 'challenger' with
    its rank (1 = nearest). `k` is an integer from 1 to the number of challengers that can be ranked, else ValueError, before anything is
    run. See the module docstring.
    """
    from src.optimization.perturbation_space import rank_challengers

    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError(f'k is a number of challengers, an integer from 1, not {k!r}')
    available = len(rank_challengers(states, validity, sdc_idx))
    if not 1 <= k <= available:
        raise ValueError(f'k is from 1 to {available}, the number of challengers that can be ranked in this scene, not {k}')
    return _run(conn, 'challenger', scenario_id, states, validity, types, sdc_idx,
                lambda stored: [{**stored, 'challenger_rank': r} for r in range(1, k + 1)],
                pipeline_run_id=pipeline_run_id)
