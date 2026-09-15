"""
autograd_optimizer.py — Phase 4, Concepts 15 & 16.

Gradient-based local refinement of the perturbation. Where Differential Evolution
(scipy_optimizer) is a robust but blind global search, this solver makes the entire
pipeline from delta -> loss differentiable and uses reverse-mode autograd to get
the full gradient in a single backward pass, then descends with Adam.

Two ingredients make this possible:
  * a differentiable re-implementation of the bicycle rollout in torch
    (Concept 15) — mirrors bicycle_step exactly, all ops differentiable;
  * a smooth, multi-circle collision margin with a log-sum-exp softmin
    (Concept 16) — replaces the non-differentiable Shapely SAT check during search.

The smooth margin is ONLY a gradient signal. The final delta is always re-checked
with the exact Shapely SAT detector (via PerturbationSpace.apply + Phase 3), so
differentiability never compromises correctness.

Intended use: warm-start from the DE solution (the hybrid strategy) and refine to
the true minimum-norm perturbation. Vehicle challengers only; for pedestrian /
cyclist challengers use scipy_optimizer (it handles every agent type).
"""

import numpy as np
import torch

from src.physics.bicycle_model import get_wheelbase, DELTA_MAX, A_MAX, V_MAX
from src.danger.collision_detector import check_collision_trajectory
from src.optimization.scipy_optimizer import keeps_challenger


def _softmin(x: torch.Tensor, beta: float) -> torch.Tensor:
    """Smooth minimum via log-sum-exp: -1/beta * logsumexp(-beta * x)."""
    return -torch.logsumexp(-beta * x, dim=0) / beta


def _circle_centers(x, y, theta, length, width, k: int):
    """
    Place k covering circles along an agent's length (rear->front).
    Returns (k, 2) centers tensor and a scalar radius. Differentiable in x,y,theta.
    """
    r = width / 2.0
    half = torch.clamp(length / 2.0 - r, min=0.0)
    if k == 1:
        offsets = torch.zeros(1, dtype=x.dtype)
    else:
        offsets = torch.linspace(-1.0, 1.0, k, dtype=x.dtype) * half
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    cx = x + offsets * cos_t
    cy = y + offsets * sin_t
    return torch.stack([cx, cy], dim=1), r  # (k,2), scalar


class _DiffBicycleRollout:
    """Differentiable bicycle rollout for the challenger, built from a PerturbationSpace."""

    def __init__(self, space, dtype=torch.float64):
        if not space.is_vehicle:
            raise NotImplementedError(
                "autograd_optimizer supports vehicle challengers only; "
                "use scipy_optimizer for pedestrian/cyclist targets."
            )
        self.space = space
        self.dt = float(space.dt)
        # space.target_len is read at the challenger's first VALID frame, so this can
        # no longer pick up a zero-filled frame 0 and produce a zero wheelbase.
        self.L = float(get_wheelbase(space.target_len))
        if not self.L > 0:
            raise ValueError(f"non-positive wheelbase {self.L} for the challenger")
        self.dtype = dtype
        self.t0 = int(space.t0)

        # Baseline controls sliced to start at the challenger's first valid frame, so
        # rollout row k corresponds to GLOBAL frame t0 + k (audit B01). The numpy
        # path slices identically; if these two ever disagree, DE and the refiner are
        # silently optimizing different problems.
        self.base_controls = torch.tensor(space.base_controls[self.t0:], dtype=dtype)
        self.base_init     = torch.tensor(space.base_init, dtype=dtype)       # (4,)

    def rollout(self, delta: torch.Tensor) -> torch.Tensor:
        """
        delta = [dv0, dtheta0, da_bias, ddelta_bias].
        Returns trajectory (T - t0, 4) = [x, y, theta, v], differentiable in delta.
        Row k is global frame t0 + k.
        """
        dv0, dtheta0, da_bias, ddelta_bias = delta[0], delta[1], delta[2], delta[3]

        steer = self.base_controls[:, 0] + ddelta_bias
        accel = self.base_controls[:, 1] + da_bias

        x = self.base_init[0]
        y = self.base_init[1]
        theta = self.base_init[2] + dtheta0
        # clamp the initial speed exactly as PerturbationSpace.apply does (audit B08)
        v = torch.clamp(self.base_init[3] + dv0, 0.0, V_MAX)

        traj = [torch.stack([x, y, theta, v])]
        for t in range(steer.shape[0]):
            d = torch.clamp(steer[t], -DELTA_MAX, DELTA_MAX)
            a = torch.clamp(accel[t], -A_MAX, A_MAX)
            # Mirrors bicycle_step exactly, including the trapezoidal position update
            # (audit B03): heading and speed first, then position from the midpoints.
            # This ordering is load-bearing — the numpy and torch rollouts must agree
            # step for step, not merely approximately.
            theta_next = theta + (v / self.L) * torch.tan(d) * self.dt
            v_next = torch.clamp(v + a * self.dt, 0.0, V_MAX)
            v_mid = 0.5 * (v + v_next)
            theta_mid = 0.5 * (theta + theta_next)
            x = x + v_mid * torch.cos(theta_mid) * self.dt
            y = y + v_mid * torch.sin(theta_mid) * self.dt
            theta, v = theta_next, v_next
            traj.append(torch.stack([x, y, theta, v]))
        return torch.stack(traj)  # (T - t0, 4)


def _smooth_margin(space, traj, beta, n_circles, dtype):
    """
    Smooth collision margin between the rolled-out challenger trajectory and the
    fixed SDC trajectory. Uses multi-circle covering + softmin over circle pairs
    and over time. > 0 = clearance, <= 0 = overlap.
    """
    states, validity = space.states0, space.validity
    sdc, tgt = space.sdc_idx, space.target_idx
    t0 = int(space.t0)

    # Dimensions come from each agent's own first valid frame, via the space, rather
    # than from frame 0 — which for a late-appearing agent is zero-fill (audit B02).
    sdc_len = float(space.sdc_len); sdc_wid = float(space.sdc_wid)
    tgt_len = float(space.target_len);  tgt_wid = float(space.target_wid)
    tgt_len_t = torch.tensor(tgt_len, dtype=dtype)
    tgt_wid_t = torch.tensor(tgt_wid, dtype=dtype)

    per_t_min = []
    # traj row k is GLOBAL frame t0 + k (audit B01). This loop used to index the
    # logged SDC state with the rollout row index, which is only the same thing when
    # t0 == 0 — once the rollout starts at t0 the two diverge and the margin would be
    # compared against the wrong SDC frame entirely.
    for k in range(traj.shape[0]):
        t = t0 + k
        if t >= states.shape[1]:
            break
        if not (validity[sdc, t] and validity[tgt, t]):
            continue
        # SDC circles (fixed, no grad)
        sx = torch.tensor(states[sdc, t, 0], dtype=dtype)
        sy = torch.tensor(states[sdc, t, 1], dtype=dtype)
        sth = torch.tensor(states[sdc, t, 4], dtype=dtype)
        sc, sr = _circle_centers(sx, sy, sth,
                                 torch.tensor(sdc_len, dtype=dtype),
                                 torch.tensor(sdc_wid, dtype=dtype), n_circles)
        # challenger circles (differentiable)
        tx, ty, tth = traj[k, 0], traj[k, 1], traj[k, 2]
        tc, tr = _circle_centers(tx, ty, tth, tgt_len_t, tgt_wid_t, n_circles)

        # pairwise center distances minus radius sums -> (k*k,)
        diff = sc.unsqueeze(1) - tc.unsqueeze(0)          # (k,k,2)
        dist = torch.sqrt((diff ** 2).sum(-1) + 1e-9)     # (k,k)
        gap = (dist - (sr + tr)).reshape(-1)              # (k*k,)
        per_t_min.append(_softmin(gap, beta))

    if not per_t_min:
        return torch.tensor(float('inf'), dtype=dtype)
    return _softmin(torch.stack(per_t_min), beta)


def _reject_out_of_bounds(space, delta_init) -> np.ndarray:
    """
    A warm start outside the declared search space is a CALLER ERROR, and is
    refused rather than quietly projected (audit R05).

    refine_scenario used to exact-verify delta_init and store it as the incumbent
    BEFORE anything looked at space.bounds, while every Adam iterate after it was
    clamped every step. So a warm start that won outright was the one candidate
    that never had to be legal. The audit's repro: bounds [[0,0], [-0.01, 0.01],
    [0,0], [0,0]] with delta_init [0, 0.15, 0, 0] — a heading offset 15x its
    allowed maximum — came back as collision=True with min_perturbation 15.0,
    which under Batch 5's B20 weighting reads as "fifteen times the entire
    permitted range", presented as a verified minimum.

    REJECT, NOT CLAMP, and the distinction is not stylistic. Iterates are clamped
    because the optimizer GENERATES them and clamping is how a box constraint is
    enforced during descent. delta_init is INPUT. Validating input and projecting
    an iterate are different operations that happen to share an expression, and
    collapsing them is what produced the defect. Clamping would also return a
    delta the caller never supplied, and if the clamped version does not collide
    the caller gets collision=False with nothing to explain why.

    The usability argument for clamping — an external caller that reasonably does
    not know the bounds — does not apply here: space.bounds is a public attribute
    of the object the caller must already build to call this function at all.
    There is no state in which a caller holds a delta but cannot see the box.
    This is the same fail-loud instinct as PerturbationSpace's dimension checks
    and _make_track's no-fallback rule.

    Checked AFTER delta_init has defaulted, because the check is on the value
    actually used: a bounds box that excludes the origin makes the zero default
    illegal, and that should be loud too.

    Comparison is exact, on the float32 cast against the float32 bounds — the
    precision this function actually computes in.

    NOTHING IN THE PIPELINE TRIGGERS THIS. _stress_one and the validation
    notebook both warm-start from DE, whose output scipy keeps inside the bounds;
    measured across 12036 evaluated deltas over six configurations, zero lay
    outside in float64 and zero after the float32 cast (the cast cannot push a
    point out, because space.bounds is float32 so the endpoints are exactly
    representable and round-to-nearest stays inside). Audit R04 changes WHICH
    delta DE returns, and the archived candidates were included in that
    measurement. So this is audit B16's category: dead in practice, wrong in the
    contract, and the first hand-supplied warm start pays for it.
    """
    candidate = np.asarray(delta_init, np.float32)
    low = np.asarray(space.bounds[:, 0], np.float32)
    high = np.asarray(space.bounds[:, 1], np.float32)
    offending = np.nonzero((candidate < low) | (candidate > high))[0]
    if offending.size:
        detail = '; '.join(
            f'component {int(i)} = {float(candidate[i])!r} outside '
            f'[{float(low[i])!r}, {float(high[i])!r}]'
            for i in offending
        )
        raise ValueError(
            'delta_init lies outside the perturbation space bounds and is '
            f'refused rather than clamped (audit R05): {detail}'
        )
    return candidate


def refine_scenario(
    space,
    delta_init=None,
    lam: float = 50.0,
    lr: float = 0.02,
    n_iters: int = 300,
    beta: float = 6.0,
    n_circles: int = 3,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """
    Gradient-descent refinement of the perturbation for a VEHICLE challenger.

    Args:
        space:      PerturbationSpace (vehicle target).
        delta_init: warm start (e.g. the DE solution). Defaults to zeros.
        lam:        penalty weight on relu(smooth_margin).
        lr:         Adam learning rate.
        n_iters:    gradient steps.
        beta:       softmin temperature (higher = closer to true min, sharper grad).
        n_circles:  circles per agent for the covering margin.
        seed:       torch RNG seed.
        verbose:    print loss every 50 steps.

    Returns:
        dict with collision (EXACT-verified), min_perturbation, delta,
        collision_timestep, smooth_margin, n_iters.

    Raises:
        ValueError: delta_init lies outside space.bounds (audit R05). See
                    _reject_out_of_bounds for why this is refused, not clamped.
    """
    torch.manual_seed(seed)
    dtype = torch.float64

    roll = _DiffBicycleRollout(space, dtype=dtype)

    if delta_init is None:
        delta_init = np.zeros(space.dim, dtype=np.float64)
    # Before the warm start becomes a candidate — which is exactly the ordering
    # bug (audit R05).
    warm = _reject_out_of_bounds(space, delta_init)
    delta = torch.tensor(np.asarray(delta_init, np.float64), dtype=dtype, requires_grad=True)

    low = torch.tensor(space.bounds[:, 0], dtype=dtype)
    high = torch.tensor(space.bounds[:, 1], dtype=dtype)
    weights = torch.tensor(space.weights, dtype=dtype)

    opt = torch.optim.Adam([delta], lr=lr)

    # ── keep the best EXACT-VERIFIED candidate, not the last iterate (audit B09) ──
    #
    # Adam minimizes wnorm2 + lam * relu(smooth_margin). The smooth margin is a
    # SURROGATE, so the loop can walk out of the exact-collision region while its own
    # objective still reports progress — and the function used to verify only wherever
    # the loop happened to stop. On the audit's scene it was handed a warm start that
    # exact-verifies as colliding at norm 0.750 and returned collision=False, having
    # discarded a feasible answer it was given for free.
    #
    # The warm start is a candidate, which is what makes "never worse than what DE
    # handed in" true by construction rather than by luck.
    #
    # "Better" = exact-verified colliding, smallest space.weighted_norm. That is the
    # SAME predicate batch_scorer._stress_one uses to choose between DE's result and
    # this one:
    #     if refined['collision'] and (not result['collision']
    #                                  or refined['min_perturbation'] < ...)
    # so the two cannot disagree about the same comparison. It is no longer written
    # out here at all: the comparison below calls scipy_optimizer.keeps_challenger,
    # which the DE archive also calls, so two of the three sites share one
    # definition instead of three sites agreeing by inspection (audit R04).
    def _verified(candidate):
        """(collides, timestep, weighted_norm) under the EXACT Shapely SAT check."""
        hit, t = check_collision_trajectory(space.apply(candidate), space.validity,
                                            space.sdc_idx, space.target_idx)
        return bool(hit), int(t), space.weighted_norm(candidate)

    best_delta, best_t, best_norm = None, -1, float('inf')

    warm_hit, warm_t, warm_norm = _verified(warm)
    if keeps_challenger(warm_hit, warm_norm, best_delta is not None, best_norm):
        best_delta, best_t, best_norm = warm.copy(), warm_t, warm_norm

    for it in range(n_iters):
        opt.zero_grad()
        traj = roll.rollout(delta)
        g = _smooth_margin(space, traj, beta, n_circles, dtype)
        wnorm2 = ((delta * weights) ** 2).sum()
        loss = wnorm2 + lam * torch.relu(g)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta], max_norm=10.0)  # tame long-horizon grads
        opt.step()
        with torch.no_grad():
            delta.clamp_(low, high)  # stay inside the box bounds

        # EXACT-verify this iterate. Unconditional, and both halves of that were
        # measured on the audit's scene rather than assumed:
        #
        #   * Cost. One apply+verify against one Adam iteration is a few percent —
        #     the torch rollout, the softmin and the backward pass dominate.
        #
        #   * Gating on the surrogate would not be safe. Filtering "only verify when
        #     smooth_margin <= 0" looks free, and here it is worthless: the softmin
        #     reports colliding at 100/100 iterates while exact SAT agrees at 10/100,
        #     so it admits everything. Worse, the bias runs the OTHER way when the
        #     circle covering dominates instead of the softmin — three circles
        #     under-cover a box's corners (Block 4 Concept 16), and isolating that
        #     effect on this same scene gives circle-min +0.0206 against truth,
        #     flagging 3/100 where exact flags 10/100. In that regime a
        #     smooth_margin > 0 filter would SKIP real collisions. A filter that is
        #     useless in one regime and unsafe in the other is not a filter.
        candidate = delta.detach().numpy().astype(np.float32)
        hit, t_cand, norm_cand = _verified(candidate)
        if keeps_challenger(hit, norm_cand, best_delta is not None, best_norm):
            best_delta, best_t, best_norm = candidate.copy(), t_cand, norm_cand

        if verbose and it % 50 == 0:
            print(f"[{it:4d}] loss={loss.item():.4f} margin={g.item():+.3f} "
                  f"||d||={space.weighted_norm(delta.detach().numpy()):.4f}")

    # Report the best verified candidate if one was ever seen; otherwise the final
    # iterate, which is what this function always used to return.
    if best_delta is not None:
        delta_np, collided, t_hit = best_delta, True, best_t
    else:
        delta_np = delta.detach().numpy().astype(np.float32)
        collided, t_hit = check_collision_trajectory(
            space.apply(delta_np), space.validity, space.sdc_idx, space.target_idx)

    # EVERY field below describes delta_np, the delta actually being returned — not
    # the final iterate. A smooth_margin or collision_timestep left over from a
    # different candidate would be the same juxtaposition defect Batch 2 removed from
    # scenario_scores: fields of one result that silently describe two.
    final_margin = _smooth_margin(space, roll.rollout(
        torch.tensor(delta_np, dtype=dtype)), beta, n_circles, dtype).item()

    return {
        'collision':          bool(collided),
        'min_perturbation':   space.weighted_norm(delta_np) if collided else float('inf'),
        'delta':              delta_np,
        'collision_timestep': int(t_hit),
        'smooth_margin':      float(final_margin),
        'n_iters':            n_iters,
    }