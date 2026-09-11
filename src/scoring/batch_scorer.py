"""
batch_scorer.py — Phase 5.

Two batch passes over a WOMD shard:

  PASS 1 (cheap, every scenario):   score_shard()
      ShardLoader -> ScenarioParser -> danger.score_scenario
      Computes min TTC, min PET, and the fragility score for every scenario in
      the shard. Each scenario is isolated in its own try/except: one corrupt
      record must never kill a multi-hour batch run.

  PASS 2 (expensive, explicit subset): stress_test_scenarios()
      Re-reads the shard, and for ONLY the requested scenario IDs (typically the
      top-N from ranker.py) runs the Phase 4 optimizer to find the minimum
      collision-causing perturbation.

Why two passes instead of one? Phase 4 costs thousands of simulations per
scenario; TTC/PET cost milliseconds. Running the optimizer on everything would
make the pipeline ~1000x more expensive for no benefit — scenarios with huge
TTC/PET margins are not where the fragile cases live. The danger score is the
cheap filter that earns Phase 4 its tractability.

Note: the waymo protobuf import lives inside score_shard / stress_test_scenarios
(not module level) so this module imports cleanly on machines without the Waymo
package — only the shard-reading paths need it (i.e., Colab).
"""

import time
import numpy as np

from src.danger.danger_score import score_scenario


def _score_one(states, validity, scenario_id, sdc_index,
               shard_name=None, pet_max_pairs=50):
    """
    Score a single parsed scenario. Pure function of arrays -> record dict.
    Split out from score_shard so it is testable without the Waymo package.

    sdc_index is required — score_scenario now ranks on SDC-restricted TTC/PET
    (see its docstring), and a missing sdc_index should fail loudly here rather
    than silently reverting to the all-pairs behaviour.
    """
    t0 = time.time()
    record = score_scenario(states, validity, scenario_id, sdc_index,
                            min_perturbation=None, pet_max_pairs=pet_max_pairs)
    record['n_agents'] = int(states.shape[0])
    record['shard'] = shard_name
    record['score_seconds'] = round(time.time() - t0, 3)
    return record


def score_shard(
    shard_path: str,
    max_scenarios: int = None,
    pet_max_pairs: int = 50,
    progress_every: int = 25,
    verbose: bool = True,
):
    """
    PASS 1 — compute the danger profile for every scenario in a shard.

    Args:
        shard_path:     path to the .tfrecord shard
        max_scenarios:  stop after this many (None = whole shard)
        pet_max_pairs:  cap on agent pairs for the PET engine
        progress_every: print progress every K scenarios
        verbose:        print progress / error lines

    Returns:
        (records, errors)
        records: list of dicts — scenario_id, min_ttc, min_pet, fragility_score,
                 min_perturbation (None), n_agents, shard, score_seconds
        errors:  list of dicts — index, scenario_id (if known), error message.
                 Errors are RECORDED, not raised: one bad record never kills
                 the batch.
    """
    # Lazy imports: only the shard-reading path needs the Waymo protobufs.
    from src.data.loader import ShardLoader
    from src.data.parser import ScenarioParser

    shard_name = str(shard_path).split('/')[-1]
    records, errors = [], []
    t_start = time.time()

    for i, raw in enumerate(ShardLoader(shard_path)):
        if max_scenarios is not None and len(records) >= max_scenarios:
            break
        scenario_id = None
        try:
            parser = ScenarioParser(raw)
            scenario_id = parser.get_scenario_id()
            states = parser.get_agent_states()
            validity = parser.get_agent_validity()
            sdc_index = parser.get_sdc_index()
            records.append(_score_one(states, validity, scenario_id, sdc_index,
                                      shard_name, pet_max_pairs))
        except Exception as e:  # noqa: BLE001 — deliberate: isolate per scenario
            errors.append({'index': i, 'scenario_id': scenario_id,
                           'error': f'{type(e).__name__}: {e}'})
            if verbose:
                print(f"  [skip] record {i} ({scenario_id}): {e}")
            continue

        if verbose and progress_every and len(records) % progress_every == 0:
            elapsed = time.time() - t_start
            print(f"  scored {len(records)} scenarios "
                  f"({elapsed:.1f}s, {elapsed/len(records):.2f}s each)")

    if verbose:
        print(f"Done: {len(records)} scored, {len(errors)} skipped, "
              f"{time.time() - t_start:.1f}s total")
    return records, errors


# ── PASS 2: explicit Phase 4 stress-test over a chosen subset ────────────────

def _stress_one(states, validity, types, sdc_idx,
                use_autograd=False, de_kwargs=None):
    """
    Run the Phase 4 pipeline on one parsed scenario:
    pick the nearest challenger, build the perturbation space, DE search,
    optional autograd refinement (vehicles only), keep the better verified result.
    Testable without the Waymo package.
    """
    from src.optimization.perturbation_space import (
        PerturbationSpace, ReplayFidelityError, pick_nearest_challenger,
    )
    from src.optimization.scipy_optimizer import optimize_scenario

    tgt = pick_nearest_challenger(states, validity, sdc_idx)
    if tgt < 0:
        return {'status': 'no_challenger'}

    # Wrapped narrowly, around this one call and nothing else. A refused replay is a
    # known, expected, measurable outcome of a working pipeline, so it gets its own
    # status — the same reason 'no_challenger' is a status rather than an exception.
    # Left unwrapped it would fall through to stress_test_scenarios' generic handler
    # and arrive as status='error' with a stringified message, throwing away the
    # structured reason/error that the real-shard run needs to break the refusals
    # down. Genuine failures elsewhere in this function still reach that handler.
    try:
        space = PerturbationSpace(states, validity, types, sdc_idx, tgt)
    except ReplayFidelityError as e:
        return {'status': 'replay_infeasible',
                'target_idx': int(tgt),
                'baseline_replay_error': e.baseline_replay_error,
                'reason': e.reason}

    result = optimize_scenario(space, **(de_kwargs or {}))
    result['target_idx'] = int(tgt)
    result['method'] = 'de'

    if use_autograd and space.is_vehicle:
        from src.optimization.autograd_optimizer import refine_scenario
        refined = refine_scenario(space, delta_init=result['delta'])
        # keep whichever exact-verified collision has the smaller norm
        if refined['collision'] and (
            not result['collision']
            or refined['min_perturbation'] < result['min_perturbation']
        ):
            refined['target_idx'] = int(tgt)
            refined['method'] = 'de+autograd'
            result = refined

    result['status'] = 'ok'
    return result


def stress_test_scenarios(
    shard_path: str,
    scenario_ids,
    use_autograd: bool = False,
    de_kwargs: dict = None,
    verbose: bool = True,
):
    """
    PASS 2 — run the Phase 4 optimizer on ONLY the given scenario IDs
    (typically ranker.top_n output). Re-reads the shard and parses records
    until all requested IDs have been found.

    Returns:
        dict scenario_id -> phase 4 result dict
        (keys: collision, min_perturbation, delta, collision_timestep,
         target_idx, method, status — or status='no_challenger'/'error')
    """
    from src.data.loader import ShardLoader
    from src.data.parser import ScenarioParser

    wanted = set(scenario_ids)
    results = {}
    t_start = time.time()

    for raw in ShardLoader(shard_path):
        if not wanted:
            break
        try:
            parser = ScenarioParser(raw)
            sid = parser.get_scenario_id()
            if sid not in wanted:
                continue
            wanted.discard(sid)
            if verbose:
                print(f"  stress-testing {sid} ...")
            states = parser.get_agent_states()
            validity = parser.get_agent_validity()
            types = parser.get_agent_types()
            sdc_idx = parser.get_sdc_index()
            results[sid] = _stress_one(states, validity, types, sdc_idx,
                                       use_autograd, de_kwargs)
            if verbose:
                r = results[sid]
                if r.get('collision'):
                    print(f"    collision at ||delta||={r['min_perturbation']:.4f} "
                          f"(t={r['collision_timestep']}, {r['method']})")
                else:
                    print(f"    robustly safe within bounds ({r.get('status')})")
        except Exception as e:  # noqa: BLE001
            sid = sid if 'sid' in dir() else None
            results[str(sid)] = {'status': 'error',
                                 'error': f'{type(e).__name__}: {e}'}
            if verbose:
                print(f"  [error] {sid}: {e}")

    if verbose:
        missing = wanted
        if missing:
            print(f"  warning: {len(missing)} requested IDs not found in shard")
        print(f"Stress pass done: {len(results)} scenarios, "
              f"{time.time() - t_start:.1f}s")
    return results