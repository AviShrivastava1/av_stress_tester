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
# MODULE LEVEL, unlike the other src.optimization imports in this file, which are
# deferred into the functions that need them. selection.py imports nothing at all and
# src/optimization/__init__.py is docstring-only, so this costs no scipy, no shapely
# and no torch — batch_scorer stays importable on a machine with none of them.
#
# It is also what makes the guarantee checkable: a deferred import inside _stress_one
# would leave no module attribute for test_all_three_selection_sites_are_one_object to
# bind to, and "they are the same object" would go back to being an argument rather
# than an assertion.
from src.optimization.selection import keeps_challenger


def _score_one(states, validity, scenario_id, sdc_index,
               shard_name=None, pet_max_pairs=50, types=None):
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

    # WHICH SCENE this row describes (audit A02). Pass 1 is where it belongs: the
    # fingerprint is a property of the parsed INPUT, not of any search, so it is
    # recorded by the pass that reads the input and is never touched again.
    #
    # `types` is optional, and its absence is not an error — a caller that does not
    # have the agent types records no fingerprint, and NULL means "not recorded", the
    # same carve-out B14 makes for stress_run_id. Adding it as a required argument
    # would turn every pre-existing caller into a TypeError for a column that is
    # allowed to be absent.
    if types is not None:
        from src.scoring.db import compute_scene_fingerprint
        record['scene_fingerprint'] = compute_scene_fingerprint(states, validity, types)

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

    # ── why this is a while loop and not `for raw in ShardLoader(...)` ──────────
    #
    # THE FETCH IS OUTSIDE THE BODY'S TRY (audit R02). A `for` statement calls
    # next() on the iterator as part of its own protocol, BEFORE the body runs — so
    # an exception raised inside ShardLoader.__iter__ is structurally outside the
    # per-record handler below and propagates straight out of this function,
    # discarding every record already scored. Batch 5 (audit B11) made the loader
    # correctly refuse truncated framing instead of reading it as clean EOF, and that
    # fix turned a silently-short batch into a lost one: Block 5 Concept 19's rule
    # that "one bad record must never kill the batch" was defeated by a fix to an
    # unrelated finding.
    #
    # THE COMPLETION CHECK RUNS BEFORE THE FETCH, not after. Asking for N scenarios
    # used to read the N+1'th record before noticing it was done — so a bounded run
    # touched shard data it was never asked for, and could die on corruption beyond
    # its own request.
    reader = iter(ShardLoader(shard_path))
    i = -1
    while True:
        if max_scenarios is not None and len(records) >= max_scenarios:
            break
        i += 1
        try:
            raw = next(reader)
        except StopIteration:
            # Clean end of file, exactly on a record boundary. Normal completion, and
            # caught BEFORE the clause below because StopIteration subclasses
            # Exception — sharing a handler would log every successful run as a fault.
            break
        except Exception as e:  # noqa: BLE001
            # The reader itself failed. Terminal by nature: the iterator is dead and
            # no later record is reachable, so the response is the same whatever the
            # type. Recorded with a `kind` rather than a bare flag so a consumer can
            # tell "the shard ends mid-record, an unknown number of scenarios were
            # never seen" from "record 7 was garbage and we carried on" — those mean
            # different things about whether this run is COMPLETE, and an
            # undifferentiated list asserts the weaker one for both.
            errors.append({'index': i, 'scenario_id': None,
                           'error': f'{type(e).__name__}: {e}',
                           'kind': 'shard_truncated'})
            if verbose:
                print(f"  [fatal] shard unreadable at record {i}: {e}")
            break

        scenario_id = None
        try:
            parser = ScenarioParser(raw)
            scenario_id = parser.get_scenario_id()
            states = parser.get_agent_states()
            validity = parser.get_agent_validity()
            sdc_index = parser.get_sdc_index()
            # types is read here ONLY for the scene fingerprint — Pass 1's scoring does
            # not use it. Parsed in the same breath as the arrays it describes so the
            # fingerprint covers the scene as this pass actually saw it.
            types = parser.get_agent_types()
            records.append(_score_one(states, validity, scenario_id, sdc_index,
                                      shard_name, pet_max_pairs, types=types))
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
    from src.scoring.db import (
        OUTCOME_COLLISION_FOUND, OUTCOME_NO_CHALLENGER,
        OUTCOME_NO_COLLISION_FOUND, OUTCOME_REPLAY_INFEASIBLE,
    )

    # How many challengers COULD have been searched, against how many were. Recorded
    # because the honest description of this pass is "one heuristically-chosen
    # challenger", and a reader of the result should not have to know that to
    # interpret it (audit B04).
    challengers_total = int(sum(1 for j in range(states.shape[0]) if j != sdc_idx))

    tgt = pick_nearest_challenger(states, validity, sdc_idx)
    if tgt < 0:
        return {'status': 'no_challenger',
                'outcome': OUTCOME_NO_CHALLENGER,
                'challengers_total': challengers_total,
                'challengers_searched': 0}

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
        # NOT a search that found nothing — no search ran at all. Kept as its own
        # outcome all the way to the API so it can never be read as "came back clean".
        return {'status': 'replay_infeasible',
                'outcome': OUTCOME_REPLAY_INFEASIBLE,
                'target_idx': int(tgt),
                'baseline_replay_error': e.baseline_replay_error,
                # EVERY field the exception carries, not a selection of them (audit
                # R07). baseline_replay_collides was dropped here, so no column could
                # persist it however the schema was written — the loss happened before
                # the database was involved.
                'baseline_replay_collides': e.baseline_replay_collides,
                'reason': e.reason,
                'challengers_total': challengers_total,
                'challengers_searched': 0}

    de_kwargs = dict(de_kwargs or {})
    result = optimize_scenario(space, **de_kwargs)
    result['target_idx'] = int(tgt)
    result['method'] = 'de'

    # DE'S OWN NUMBERS, CAPTURED BEFORE ANYTHING CAN OVERWRITE THEM (audit A07).
    #
    # `result = refined` below replaces the WHOLE dict, and search_provenance is built
    # after that — so it was reading DE's keys off a dict that no longer had DE's
    # shape, and an accepted refinement erased the record of the search that produced
    # its own warm start. Measured with popsize=4, maxiter=0: de_n_iter and de_n_eval
    # both came back None.
    #
    # CAPTURED, NOT READ LATER. Patching the read would leave the same trap for the
    # next stage anybody adds; a local bound here cannot be reassigned by a later
    # branch, whatever that branch decides to do with `result`.
    de_search = {
        'de_n_iter': result.get('n_iter'),
        'de_n_eval': result.get('n_eval'),
        # Present in optimize_scenario's return since Batch 6 and NEVER persisted, so
        # these two are new coverage rather than something being restored: which of the
        # two-stage candidates won, and whether the archive saw every evaluation.
        'de_candidate_source': result.get('candidate_source'),
        'de_archive_covered_all_evaluations': result.get('archive_covered_all_evaluations'),
    }
    refine_n_iters = None

    if use_autograd and space.is_vehicle:
        from src.optimization.autograd_optimizer import refine_scenario
        refined = refine_scenario(space, delta_init=result['delta'])
        refine_n_iters = refined.get('n_iters')      # note: n_iters, not DE's n_iter
        # Keep whichever exact-verified collision has the smaller norm — THE SAME
        # FUNCTION OBJECT the DE archive and the refiner's own iterate loop call, not
        # a transcription of it (Batch 10). This was an inline expression held equal
        # to theirs by test until src/scoring/ came back into scope.
        if keeps_challenger(refined['collision'], refined['min_perturbation'],
                            result['collision'], result['min_perturbation']):
            refined['target_idx'] = int(tgt)
            refined['method'] = 'de+autograd'
            result = refined

    result['status'] = 'ok'
    result['outcome'] = (OUTCOME_COLLISION_FOUND if result.get('collision')
                         else OUTCOME_NO_COLLISION_FOUND)
    result['challengers_total'] = challengers_total
    result['challengers_searched'] = 1

    # What the claim rests on, recorded alongside the claim. Without this, a
    # no_collision_found row is indistinguishable from a thorough search, when in
    # fact it is one challenger under a finite stochastic budget.
    result['search_provenance'] = {
        # Which kinematic model produced this delta, and therefore which units its four
        # components carry (audit B13). Recorded as the DIRECT fact — the model actually
        # dispatched to — rather than left to be inferred downstream from agent type,
        # because the agent type is only available once geometry has been exported and
        # the label has to be right before that.
        #
        # It lives in search_provenance because it IS search provenance: "how was the
        # stored result produced" is the same concept as the bounds and the DE budget
        # beside it, and Batch 2 established that fields of one concept travel together
        # rather than being split across a separate column.
        'delta_parameterization': 'bicycle' if space.is_vehicle else 'linear',
        'challenger_selection': 'nearest_by_min_center_distance',
        'bounds': [[float(lo), float(hi)] for lo, hi in space.bounds],
        'de_popsize': de_kwargs.get('popsize', 15),
        'de_maxiter': de_kwargs.get('maxiter', 200),
        'de_tol': de_kwargs.get('tol', 1e-3),
        'de_seed': de_kwargs.get('seed', 0),
        # Unpacked from the capture taken before any reassignment, not read off
        # `result` — see de_search above.
        **de_search,
        'refine_n_iters': refine_n_iters,
        # Which stage's answer every other field describes. Derivable from
        # stress_method today, and recorded anyway for the reason
        # baseline_replay_collides is: a reader should not have to learn that a
        # derivation exists, nor notice when it stops being two-way.
        'selected_stage': result['method'],
        'baseline_replay_error': float(space.baseline_replay_error),
        'target_has_interior_gap': bool(space.has_interior_gap),
        # How near-stationary this challenger actually was (audit A01). Recorded on
        # EVERY searched scenario, exactly like target_has_interior_gap beside it and
        # baseline_replay_error above it, and for the same reason both of those are:
        # V_HEADING_MIN = 0.5 m/s is a proposed default, and the only way it becomes a
        # defended one is a query over a real shard. A threshold whose effect is only
        # visible where it already fired cannot tell anyone how often it fires.
        #
        # This is the one place Batch 9 surfaces something outside
        # src/optimization/ and src/physics/, and it is surfaced because the
        # measurement has no route to the validation pass otherwise — the notebook
        # reads search_provenance, not PerturbationSpace instances.
        #
        # None rather than inf when the challenger has no observed frame, because
        # JSON has no infinity literal and this dict is written to JSONB. That is
        # Phase 5 doctrine — Block 6 Concept 24, "Why NULL Instead of Infinity", the
        # same rule db.update_stress_results applies to min_perturbation — and not an
        # audit finding; B04 is the robustly_safe defect, which is a different thing
        # that merely happens to be discussed alongside it in api/models.py.
        'target_min_speed': (float(space.target_min_speed)
                             if np.isfinite(space.target_min_speed) else None),
        'frames_below_heading_floor': int(space.frames_below_heading_floor),
        # A LOGGED frame inside [floor, floor+width) no longer replays byte-exact
        # under delta=0 (independent review, post-A01: the fix for the boundary
        # cliff at v=heading_speed_floor trades a bounded, measured amount of this
        # for removing a discontinuity an optimiser could cross for free). Recorded
        # for the same reason frames_below_heading_floor is: HEADING_TRANSITION_WIDTH
        # is a proposal until a real shard says how often this actually matters.
        'frames_in_heading_transition_band': int(space.frames_in_heading_transition_band),
        'frames_observed': int(space.frames_observed),
        'heading_speed_floor': (None if space.heading_speed_floor is None
                                else float(space.heading_speed_floor)),
        'heading_transition_width': (None if not space.heading_transition_width
                                     else float(space.heading_transition_width)),
    }
    return result


_MAX_ERROR_MESSAGE_LEN = 2000


def _bounded(message: str) -> str:
    """An exception message, truncated visibly rather than silently (audit A06)."""
    if len(message) <= _MAX_ERROR_MESSAGE_LEN:
        return message
    return (message[:_MAX_ERROR_MESSAGE_LEN]
            + f'... [truncated, {len(message)} chars total]')


def _describe_outcome(r) -> str:
    """
    One console line per result, saying what the pass actually concluded.

    THIS USED TO BE A TWO-BRANCH `if r.get('collision') ... else`, and the else printed
    "robustly safe within bounds" for EVERY other outcome (audit R08 / A09, raised by
    two separate audits at two different sets of lines). That included
    replay_infeasible — a scenario whose own zero-perturbation replay was not faithful
    enough to measure against, so NO SEARCH RAN AT ALL. There was no budget, no bounds
    and no result, and the console announced it as a safety conclusion.

    The API and models.py stopped making that claim in Batch 2, when B04 replaced the
    inferred robustly_safe with an explicit search_certifies_infeasibility that no code
    path sets. The console was never brought along. So this is not a wording change: it
    is the console finally saying the same thing the rest of the system says.

    EVERY OUTCOME IS NAMED EXPLICITLY, with no catch-all that could absorb a new one
    silently — a `else: pass` here is how the original defect got in. A status this
    does not know about is reported AS unrecognized rather than described.

    Phrased in the OUTCOME vocabulary (src/scoring/db.py's OUTCOME_* constants), so a
    console line, an API field and a database column all read the same.
    """
    from src.scoring.db import (
        OUTCOME_COLLISION_FOUND, OUTCOME_ERROR, OUTCOME_NO_CHALLENGER,
        OUTCOME_NO_COLLISION_FOUND, OUTCOME_REPLAY_INFEASIBLE,
    )

    outcome = r.get('outcome')

    if outcome == OUTCOME_COLLISION_FOUND:
        return (f"{OUTCOME_COLLISION_FOUND}: collision at "
                f"||delta||={r['min_perturbation']:.4f} "
                f"(t={r['collision_timestep']}, {r['method']})")

    if outcome == OUTCOME_NO_COLLISION_FOUND:
        # DELIBERATELY NOT "safe". A completed search that found nothing is a fact
        # about THIS search: one heuristically-chosen challenger, a finite stochastic
        # budget, and a bounded box. The counts are printed because they are what
        # makes that readable rather than something the reader has to know.
        return (f"{OUTCOME_NO_COLLISION_FOUND}: no collision within this search's "
                f"budget and bounds "
                f"({r.get('challengers_searched')}/{r.get('challengers_total')} "
                f"challengers)")

    if outcome == OUTCOME_REPLAY_INFEASIBLE:
        return (f"{OUTCOME_REPLAY_INFEASIBLE}: no search ran — "
                f"reason={r.get('reason')}, "
                f"baseline drift={r.get('baseline_replay_error')}, "
                f"baseline collides={r.get('baseline_replay_collides')}")

    if outcome == OUTCOME_NO_CHALLENGER:
        return (f"{OUTCOME_NO_CHALLENGER}: no search ran — nothing to perturb "
                f"({r.get('challengers_total')} candidate agents)")

    if outcome == OUTCOME_ERROR:
        return f"{OUTCOME_ERROR}: {r.get('error')}"

    return f"unrecognized outcome {outcome!r} (status={r.get('status')!r})"


class StressResults(dict):
    """
    dict scenario_id -> result, plus `.errors` for failures that have no scenario id.

    A plain dict cannot express "this record failed before we learned which scenario
    it was" (audit B05). The old code wrote such failures under whatever `sid` was
    left over from the previous iteration, destroying a result that had already
    succeeded.

    A dict SUBCLASS rather than a (results, errors) tuple, which would have been
    symmetric with score_shard: every existing caller indexes the return directly —
    export_shard_geometry, db.update_stress_results, the validation notebook, and the
    audit fixture — and all of them keep working unchanged this way.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.errors = []


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
        StressResults — a dict scenario_id -> phase 4 result dict
        (keys: collision, min_perturbation, delta, collision_timestep, target_idx,
         method, status, outcome — or status='no_challenger'/'replay_infeasible'/
         'error'), carrying `.errors` for records that failed before yielding an id.
    """
    from src.data.loader import ShardLoader
    from src.data.parser import ScenarioParser

    wanted = set(scenario_ids)
    results = StressResults()
    t_start = time.time()

    # Same shape as score_shard's loop, and for the same two reasons (audit R02):
    # the fetch must sit inside a handler, and the completion check must run before
    # it. The second matters more here than there — this function is USUALLY called
    # with a handful of ids, so reading one record past the last one found is the
    # common case, not the edge case. stress_test_scenarios(shard, ['A']) with A as
    # record 0 used to read record 1 anyway, and died if it was corrupt.
    reader = iter(ShardLoader(shard_path))
    record_index = -1
    while True:
        if not wanted:
            break
        record_index += 1
        try:
            raw = next(reader)
        except StopIteration:
            break                      # clean EOF — normal completion, not an error
        except Exception as e:  # noqa: BLE001
            results.errors.append({'record_index': record_index,
                                   'scenario_id': None,
                                   'error': f'{type(e).__name__}: {e}',
                                   'kind': 'shard_truncated'})
            if verbose:
                print(f"  [fatal] shard unreadable at record {record_index}: {e}")
            break

        # Reset every iteration (audit B05). Without this, `sid` survives from the
        # previous pass, and a record that fails to parse BEFORE the assignment below
        # lands in the except handler still holding the last scenario's id.
        sid = None
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
                print(f"    {_describe_outcome(results[sid])}")
        except Exception as e:  # noqa: BLE001
            detail = f'{type(e).__name__}: {e}'
            # TYPE AND MESSAGE AS THEIR OWN FIELDS (audit A06). `detail` fuses both
            # into one string, and the fused form is KEPT because two consumers read
            # it: _describe_outcome above, and the validation notebook, which recovers
            # the type with e['error'].split(':')[0] — string surgery that is both the
            # argument for separate fields and the reason the fused key cannot go.
            #
            # THE MESSAGE IS BOUNDED. str(e) is unbounded in principle: an exception
            # carrying a numpy repr can be megabytes, and this lands in a JSONB column
            # beside four small fixed-shape fields. Observed messages on the two
            # reachable triggers are 93 and 102 characters, so 2000 is ~20x headroom
            # and truncation should never fire in practice — a resource guard, not a
            # format assumption, the same reasoning as _MAX_SCENARIO_ID_LEN in
            # api/routes.py. Truncation is MARKED so a reader can tell a message ends
            # because it ended, not because it was cut.
            if sid is None:
                # The record never yielded an id, so there is no scenario to attribute
                # this to. It goes in the separate error list against its record
                # index — writing it into `results` under any key would either invent
                # a scenario or overwrite a real one.
                results.errors.append({'record_index': record_index,
                                       'scenario_id': None, 'error': detail})
                if verbose:
                    print(f"  [error] record {record_index} (unidentified): {e}")
            else:
                results[sid] = {'status': 'error', 'outcome': 'error',
                                'error': detail,
                                'error_type': type(e).__name__,
                                'error_message': _bounded(str(e))}
                if verbose:
                    print(f"  [error] {sid}: {e}")

    if verbose:
        missing = wanted
        if missing:
            print(f"  warning: {len(missing)} requested IDs not found in shard")
        if results.errors:
            print(f"  {len(results.errors)} record(s) failed before yielding an id")
        print(f"Stress pass done: {len(results)} scenarios, "
              f"{time.time() - t_start:.1f}s")
    return results