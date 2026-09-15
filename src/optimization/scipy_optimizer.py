"""
scipy_optimizer.py — Phase 4, Concepts 13 & 14.

Gradient-free global search for the minimum perturbation that causes a collision
between the SDC and a chosen challenger, using Differential Evolution.

Problem (Concept 13):
    minimize    ||delta||_w
    subject to  collision(SDC, challenger) under (sigma (+) delta)
                delta in box bounds

We do NOT optimize the boolean collision flag directly (non-differentiable,
non-convex). Instead we use a continuous margin g(delta) = the minimum signed gap
between the two agents over the trajectory, and a penalty objective:

    J(delta) = ||delta||_w^2  +  LAMBDA * relu( g(delta) )

    g > 0  -> a gap remains (safe);  relu(g) penalizes it, pushing toward collision
    g <= 0 -> overlap (collision);   penalty is zero, so J collapses to ||delta||^2

LAMBDA is large so the search first achieves a collision, then shrinks delta.
The continuous margin is only a SEARCH SIGNAL — every reported collision is
re-verified with the EXACT Shapely SAT check from Phase 3.

WHY THE PENALTY OBJECTIVE ALONE CANNOT BE TRUSTED TO PICK THE ANSWER (audit R04)
-------------------------------------------------------------------------------
`relu(g)` is CONTINUOUS at g = 0, deliberately — that continuity is the whole
reason DE gets a "getting warmer" signal instead of a flat boolean. But it also
means J has no discontinuity at feasibility, so J does not rank feasible above
infeasible. It ranks CLOSER above FURTHER. An infeasible point beats the best
feasible one whenever

    0 < g < (norm_feasible^2 - norm_infeasible^2) / LAMBDA

Raising LAMBDA shrinks that window. Nothing closes it. This is a property of the
penalty formulation, not a tuning failure and not DE's stochasticity.

Measured on the two-agent fixture with the challenger at y = 2.6, popsize=6,
maxiter=25, seed=5: of 624 evaluations, 3 exact-verified as colliding, the best of
them at weighted norm 1.687680960 (J = 2.848267). `optimize_scenario` returned a
NON-colliding point at norm 1.652127862 whose margin was +1.4506e-05 m — 14.5
micrometres of clearance, charged 0.0145 by the penalty against a norm^2 advantage
of 0.119, for J = 2.744032. It reported collision=False, min_perturbation=inf,
having already evaluated three collisions.

So the objective's ranking is kept as the SEARCH signal and a separate archive
keeps the best exact-verified collision; see _DEObjective and keeps_challenger.
"""

import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import differential_evolution

from src.danger.collision_detector import get_corners, check_collision_trajectory


# penalty weight: must dominate ||delta||^2 so feasibility (a real collision) is
# found before the norm is minimized. Tunable; 1e3 works for these unit scales.
LAMBDA = 1.0e3


def keeps_challenger(challenger_collides, challenger_norm,
                     incumbent_collides, incumbent_norm) -> bool:
    """
    THE ONE definition of "this candidate is better than the one we are holding":
    feasibility first, then smaller weighted norm. Ties go to the incumbent.

    Three places in this project make that comparison — the DE archive below,
    refine_scenario's iterate loop, and batch_scorer._stress_one choosing between
    DE's answer and the refiner's. Written out three times they can drift, and a
    drift is silent: an internal rule that disagreed with _stress_one's would make
    a search discard work it had correctly done. Two of the three import this
    function. The third, _stress_one, still carries the expression inline because
    src/scoring/ was out of scope for the batch that added this, and is held equal
    to it by test (test_the_de_selection_rule_matches_stress_one) rather than by
    sharing the code. When a third real importer appears this earns its own module.

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


def _signed_gap(states, validity, a, b) -> float:
    """
    Continuous collision margin between agents a and b over the full trajectory.

        gap(t) = +distance(OBB_a, OBB_b)         if the boxes are disjoint
               = -sqrt(overlap_area)              if they intersect (penetration proxy)
        g      = min over valid t of gap(t)

    g > 0 -> no collision, g <= 0 -> collision. Continuous in the agents' poses,
    so the optimizer gets a smooth "getting warmer" signal even before any overlap.

    THE SIGN OF g IS NOT A SURROGATE FOR THE EXACT CHECK — IT IS THE EXACT CHECK.
    This loop and check_collision_trajectory walk the same frames under the same
    validity gate, build boxes with the same get_corners, and ask the same Shapely
    `intersects` predicate. The intersecting branch returns -sqrt(area + 1e-12),
    which is at most -1e-6; the disjoint branch returns a strictly positive
    distance. So `g < 0` holds exactly when check_collision_trajectory returns
    True, and g is never exactly 0. PerturbationSpace.apply casts its argument to
    float32 on entry, so the g computed here on DE's float64 vector is also the g
    of that vector's float32 cast — the round-trip in optimize_scenario cannot
    change the verdict.

    Measured over 4200 random deltas across six scenes: 0 sign disagreements with
    check_collision_trajectory, 0 differences between g(float64 d) and
    g(float32 d), 0 values of g exactly equal to 0.

    This is what makes the R04 archive free: the objective already knows whether
    each candidate collides. It used to throw that away, because relu maps every
    negative g to zero and the sign never left the function.

    NOTE the contrast with autograd_optimizer._smooth_margin, whose own docstring
    argues AGAINST gating on a margin. That one is a softmin over circle-covered
    approximations — two lossy layers, measured at 100/100 against exact SAT's
    10/100. This one is exact polygon geometry. Different object, opposite
    conclusion.
    """
    T = states.shape[1]
    worst = np.inf
    for t in range(T):
        if not (validity[a, t] and validity[b, t]):
            continue
        pa = Polygon(get_corners(states[a, t, 0], states[a, t, 1], states[a, t, 4],
                                 states[a, t, 5], states[a, t, 6]))
        pb = Polygon(get_corners(states[b, t, 0], states[b, t, 1], states[b, t, 4],
                                 states[b, t, 5], states[b, t, 6]))
        if pa.intersects(pb):
            gap = -np.sqrt(pa.intersection(pb).area + 1e-12)
        else:
            gap = pa.distance(pb)
        if gap < worst:
            worst = gap
    return float(worst)


class _DEObjective:
    """
    The DE penalty objective, as a MODULE-LEVEL callable instead of a closure.

    J(delta) = ||delta||_w^2 + lam * relu(g(delta))

    This used to be a nested `def objective(delta)` inside optimize_scenario, closing
    over `space`, `lam`, `sdc`, `tgt` and `validity`. Nested functions are not
    picklable — pickle stores a function by qualified name, and
    `optimize_scenario.<locals>.objective` cannot be looked up in a fresh interpreter.
    So `workers=-1` or any `workers > 1`, which scipy services with a process pool,
    died at serialization with an AttributeError before evaluating anything (audit
    B10). The documented `workers` parameter was unusable for every value it exists
    to support.

    A class with the state as attributes pickles by reference to the class plus a
    dict of instance attributes, all of which are picklable here — PerturbationSpace
    holds plain numpy arrays and scalars, verified to round-trip. functools.partial
    over a module-level function would work equally well; a named class was chosen
    so the object has a meaningful repr in a worker traceback, where the failure
    would otherwise surface far from this file.

    BEHAVIOURALLY IDENTICAL AT workers=1. The arithmetic below is the closure's,
    unchanged and in the same order, so the default path returns bit-identical
    results — which is asserted against pre-refactor captured values rather than
    assumed, since no test in this project exercises workers > 1 at all.

    IT ALSO CARRIES THE ARCHIVE (audit R04): the smallest-norm delta seen so far
    whose `g` was negative, i.e. which EXACT-collides. See _signed_gap's docstring
    for why the sign is a certificate rather than a guess. The returned value is
    untouched, so DE's trajectory is bit-identical to a run without the archive —
    the archive only remembers, it never steers.

    Cost is a float compare and, on improvement only, a 4-element copy. Measured
    across 3 runs at popsize=15/maxiter=200: 1.205s -> 1.205s (3360 evaluations,
    28 archive updates) and 1.840s -> 1.834s (5760 evaluations, 20 updates).
    Within noise, because no extra geometry is computed: g and norm are already in
    hand when the comparison happens.

    WHAT THE ARCHIVE DOES NOT SURVIVE: workers != 1. scipy services those with a
    process pool, which pickles this object to each child; the children update
    their own copies and the parent's stays empty. That is stated in
    optimize_scenario's return value rather than left to be discovered — see
    `archive_covered_all_evaluations`. Making it survive would need a
    Manager-backed structure and an IPC round trip per evaluation, on a path
    nothing in this project has ever called and that cannot be exercised in this
    environment at all (Batch 5: the forkserver is sandboxed). A fix that cannot
    be shown to work is not a fix, so the guarantee is scoped to workers=1 and
    said out loud instead.
    """

    def __init__(self, space, lam: float):
        self.space = space
        self.lam = lam
        self.sdc = space.sdc_idx
        self.tgt = space.target_idx
        self.validity = space.validity
        # the archive: best exact-verified colliding candidate seen (audit R04)
        self.best_delta = None
        self.best_norm = float('inf')

    def __call__(self, delta):
        pert = self.space.apply(delta)
        g = _signed_gap(pert, self.validity, self.sdc, self.tgt)
        norm = self.space.weighted_norm(delta)
        # `g < 0` IS the exact collision check (see _signed_gap), so this records a
        # verified feasible point, not a promising-looking one.
        if keeps_challenger(g < 0.0, norm,
                            self.best_delta is not None, self.best_norm):
            self.best_delta = np.asarray(delta, np.float32).copy()
            self.best_norm = norm
        return norm * norm + self.lam * max(0.0, g)


def optimize_scenario(
    space,
    lam: float = LAMBDA,
    popsize: int = 15,
    maxiter: int = 200,
    tol: float = 1e-3,
    seed: int = 0,
    workers: int = 1,
    verbose: bool = False,
) -> dict:
    """
    Run Differential Evolution over a PerturbationSpace.

    Args:
        space:    a PerturbationSpace instance (defines apply, bounds, norm).
        lam:      penalty weight for the relu(margin) term.
        popsize:  DE population multiplier (population = popsize * dim).
        maxiter:  max generations.
        tol:      convergence tolerance on the population spread.
        seed:     RNG seed — makes the audit reproducible.
        workers:  parallel workers (-1 uses all cores). Each eval is one sim.
        verbose:  print DE progress.

    Returns:
        dict with:
            collision         : bool  (EXACT-verified, not the surrogate)
            min_perturbation  : weighted norm of the best colliding delta (or inf)
            delta             : the best delta found (np.ndarray, dim,)
            collision_timestep: first colliding timestep (-1 if none)
            margin            : surrogate margin at the best delta
            n_iter, n_eval    : DE iteration / evaluation counts
            candidate_source  : 'de_winner' or 'archive' — which candidate won
            archive_covered_all_evaluations
                              : False when workers != 1, where the archive cannot
                                see what child processes evaluated. The search
                                still runs; it just carries the pre-R04 guarantee

    EVERY FIELD DESCRIBES THE RETURNED DELTA, including when the archive wins —
    the same rule Batch 5 applied to refine_scenario, for the same reason.
    """
    objective = _DEObjective(space, lam)

    bounds = [tuple(b) for b in space.bounds]

    result = differential_evolution(
        objective, bounds,
        popsize=popsize, mutation=(0.5, 1.0), recombination=0.7,
        maxiter=maxiter, tol=tol, seed=seed, workers=workers,
        polish=False, init='latinhypercube', disp=verbose,
    )

    sdc, tgt = space.sdc_idx, space.target_idx
    validity = space.validity

    # DE's own winner — the population member with the lowest penalty objective.
    delta_star = result.x.astype(np.float32)
    pert = space.apply(delta_star)
    # EXACT verification — the surrogate never gets the final word.
    collided, t_hit = check_collision_trajectory(pert, validity, sdc, tgt)
    de_norm = space.weighted_norm(delta_star)
    source = 'de_winner'

    # ── the archive gets to CHALLENGE, and exact geometry decides (audit R04) ──
    #
    # Two comparisons, not one, and the split is the point. The first uses the
    # archive's certificate to decide whether a re-verification is even worth
    # doing, so the common case costs nothing. The second re-runs the EXACT check
    # on the challenger and re-decides on that. So if `g < 0` were ever wrong, the
    # worst outcome is a wasted verification: a mis-certified candidate loses the
    # second comparison and DE's winner is kept. The certificate proposes; Shapely
    # disposes. That is the same discipline as "the surrogate never gets the final
    # word" above, applied to the thing doing the proposing.
    if objective.best_delta is not None and keeps_challenger(
            True, objective.best_norm, collided, de_norm):
        cand = objective.best_delta
        cand_pert = space.apply(cand)
        cand_hit, cand_t = check_collision_trajectory(cand_pert, validity, sdc, tgt)
        if keeps_challenger(cand_hit, space.weighted_norm(cand), collided, de_norm):
            delta_star, pert, collided, t_hit = cand, cand_pert, cand_hit, cand_t
            source = 'archive'

    margin = _signed_gap(pert, validity, sdc, tgt)

    return {
        'collision':          bool(collided),
        'min_perturbation':   space.weighted_norm(delta_star) if collided else float('inf'),
        'delta':              delta_star,
        'collision_timestep': int(t_hit),
        'margin':             margin,
        'n_iter':             int(result.nit),
        'n_eval':             int(result.nfev),
        'candidate_source':   source,
        'archive_covered_all_evaluations': bool(workers == 1),
    }