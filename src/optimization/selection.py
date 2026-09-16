"""
selection.py — the one rule for "is this candidate better than the one we are holding".

WHY THIS IS ITS OWN MODULE
--------------------------
Three places in this project make that comparison: the DE archive in
scipy_optimizer._DEObjective, refine_scenario's iterate loop in autograd_optimizer,
and batch_scorer._stress_one choosing between DE's answer and the refiner's. Batch 6
made the first two share one function and left the third as an inline expression,
held equal by test (test_the_de_selection_rule_matches_stress_one) rather than by
sharing the code, because src/scoring/ was out of scope for that batch. The docstring
said "when a third real importer appears this earns its own module". It has.

WHY HERE AND NOT SOMEWHERE NEUTRAL
----------------------------------
src/scoring/batch_scorer.py already imports PerturbationSpace, pick_nearest_challenger,
optimize_scenario and refine_scenario from src.optimization, so src/scoring ->
src/optimization is an existing dependency edge and this adds none. A neutral
src/selection.py would be the first top-level module in src/ — there are none — and
would invert the ownership: WHICH CANDIDATE WINS belongs to the search. _stress_one is
a consumer applying the search's rule to the search's two answers, not a co-owner of it.

This module deliberately imports NOTHING. That is what lets batch_scorer import it at
module level without dragging scipy, shapely or torch into a module that must stay
importable on a machine with none of them.
"""


def keeps_challenger(challenger_collides, challenger_norm,
                     incumbent_collides, incumbent_norm) -> bool:
    """
    THE ONE definition of "this candidate is better than the one we are holding":
    feasibility first, then smaller weighted norm. Ties go to the incumbent.

    All three sites import this function, so they are the same object rather than
    three copies that agree today. The drift this prevents is silent: an internal rule
    that disagreed with _stress_one's would make a search discard work it had
    correctly done, and report a larger minimum perturbation than it actually found.

    Args:
        challenger_collides: does the candidate under consideration EXACT-verify?
        challenger_norm:     its space.weighted_norm.
        incumbent_collides:  does the currently-held answer exact-verify?
        incumbent_norm:      its space.weighted_norm.

    Returns:
        True iff the challenger should replace the incumbent.
    """
    # STRICT `<`, so an exact tie in weighted norm KEEPS THE INCUMBENT, always and
    # deterministically. Nothing observable rides on it today — tied norms report the
    # same min_perturbation whichever delta is kept — but which DELTA survives a tie
    # is a real choice, and this project has been caught before treating a tiebreak as
    # cosmetic (the scenario_id tiebreak in Block 5 read as decorative until keyset
    # pagination made it load-bearing). Written down rather than left to be inferred
    # from an operator.
    return bool(challenger_collides) and (
        not incumbent_collides or challenger_norm < incumbent_norm
    )
