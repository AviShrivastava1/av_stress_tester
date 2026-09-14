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
"""

import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import differential_evolution

from src.danger.collision_detector import get_corners, check_collision_trajectory


# penalty weight: must dominate ||delta||^2 so feasibility (a real collision) is
# found before the norm is minimized. Tunable; 1e3 works for these unit scales.
LAMBDA = 1.0e3


def _signed_gap(states, validity, a, b) -> float:
    """
    Continuous collision margin between agents a and b over the full trajectory.

        gap(t) = +distance(OBB_a, OBB_b)         if the boxes are disjoint
               = -sqrt(overlap_area)              if they intersect (penetration proxy)
        g      = min over valid t of gap(t)

    g > 0 -> no collision, g <= 0 -> collision. Continuous in the agents' poses,
    so the optimizer gets a smooth "getting warmer" signal even before any overlap.
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
    """

    def __init__(self, space, lam: float):
        self.space = space
        self.lam = lam
        self.sdc = space.sdc_idx
        self.tgt = space.target_idx
        self.validity = space.validity

    def __call__(self, delta):
        pert = self.space.apply(delta)
        g = _signed_gap(pert, self.validity, self.sdc, self.tgt)
        norm = self.space.weighted_norm(delta)
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

    delta_star = result.x.astype(np.float32)
    pert = space.apply(delta_star)

    # EXACT verification — the surrogate never gets the final word.
    collided, t_hit = check_collision_trajectory(pert, validity, sdc, tgt)
    margin = _signed_gap(pert, validity, sdc, tgt)

    return {
        'collision':          bool(collided),
        'min_perturbation':   space.weighted_norm(delta_star) if collided else float('inf'),
        'delta':              delta_star,
        'collision_timestep': int(t_hit),
        'margin':             margin,
        'n_iter':             int(result.nit),
        'n_eval':             int(result.nfev),
    }