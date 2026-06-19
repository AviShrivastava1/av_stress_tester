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
        self.L = float(get_wheelbase(space.target_len))
        self.dtype = dtype

        # baseline controls and initial state as fixed tensors
        self.base_controls = torch.tensor(space.base_controls, dtype=dtype)  # (T-1,2)
        self.base_init     = torch.tensor(space.base_init, dtype=dtype)       # (4,)

    def rollout(self, delta: torch.Tensor) -> torch.Tensor:
        """
        delta = [dv0, dtheta0, da_bias, ddelta_bias].
        Returns trajectory (T, 4) = [x, y, theta, v], differentiable in delta.
        """
        dv0, dtheta0, da_bias, ddelta_bias = delta[0], delta[1], delta[2], delta[3]

        steer = self.base_controls[:, 0] + ddelta_bias
        accel = self.base_controls[:, 1] + da_bias

        x = self.base_init[0]
        y = self.base_init[1]
        theta = self.base_init[2] + dtheta0
        v = self.base_init[3] + dv0

        traj = [torch.stack([x, y, theta, v])]
        for t in range(steer.shape[0]):
            d = torch.clamp(steer[t], -DELTA_MAX, DELTA_MAX)
            a = torch.clamp(accel[t], -A_MAX, A_MAX)
            x = x + v * torch.cos(theta) * self.dt
            y = y + v * torch.sin(theta) * self.dt
            theta = theta + (v / self.L) * torch.tan(d) * self.dt
            v = torch.clamp(v + a * self.dt, 0.0, V_MAX)
            traj.append(torch.stack([x, y, theta, v]))
        return torch.stack(traj)  # (T,4)


def _smooth_margin(space, traj, beta, n_circles, dtype):
    """
    Smooth collision margin between the rolled-out challenger trajectory and the
    fixed SDC trajectory. Uses multi-circle covering + softmin over circle pairs
    and over time. > 0 = clearance, <= 0 = overlap.
    """
    states, validity = space.states0, space.validity
    sdc, tgt = space.sdc_idx, space.target_idx
    T = traj.shape[0]

    sdc_len = float(states[sdc, 0, 5]); sdc_wid = float(states[sdc, 0, 6])
    tgt_len = float(space.target_len);  tgt_wid = float(states[tgt, 0, 6])
    tgt_len_t = torch.tensor(tgt_len, dtype=dtype)
    tgt_wid_t = torch.tensor(tgt_wid, dtype=dtype)

    per_t_min = []
    for t in range(T):
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
        tx, ty, tth = traj[t, 0], traj[t, 1], traj[t, 2]
        tc, tr = _circle_centers(tx, ty, tth, tgt_len_t, tgt_wid_t, n_circles)

        # pairwise center distances minus radius sums -> (k*k,)
        diff = sc.unsqueeze(1) - tc.unsqueeze(0)          # (k,k,2)
        dist = torch.sqrt((diff ** 2).sum(-1) + 1e-9)     # (k,k)
        gap = (dist - (sr + tr)).reshape(-1)              # (k*k,)
        per_t_min.append(_softmin(gap, beta))

    if not per_t_min:
        return torch.tensor(float('inf'), dtype=dtype)
    return _softmin(torch.stack(per_t_min), beta)


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
    """
    torch.manual_seed(seed)
    dtype = torch.float64

    roll = _DiffBicycleRollout(space, dtype=dtype)

    if delta_init is None:
        delta_init = np.zeros(space.dim, dtype=np.float64)
    delta = torch.tensor(np.asarray(delta_init, np.float64), dtype=dtype, requires_grad=True)

    low = torch.tensor(space.bounds[:, 0], dtype=dtype)
    high = torch.tensor(space.bounds[:, 1], dtype=dtype)
    weights = torch.tensor(space.weights, dtype=dtype)

    opt = torch.optim.Adam([delta], lr=lr)

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
        if verbose and it % 50 == 0:
            print(f"[{it:4d}] loss={loss.item():.4f} margin={g.item():+.3f} "
                  f"||d||={space.weighted_norm(delta.detach().numpy()):.4f}")

    delta_np = delta.detach().numpy().astype(np.float32)

    # EXACT verification using the numpy pipeline (same equations as the rollout).
    pert = space.apply(delta_np)
    collided, t_hit = check_collision_trajectory(pert, space.validity,
                                                 space.sdc_idx, space.target_idx)
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