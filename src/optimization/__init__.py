
"""
Phase 4 — The Optimization Engine.
 
Finds the minimum perturbation to a real WOMD scenario that causes a collision
between the self-driving car (SDC) and a chosen challenger agent.
 
Modules:
    perturbation_space  — defines what we are allowed to perturb, the bounds,
                          how a perturbation is applied (re-simulated through the
                          Phase 2 physics engine), and the weighted norm.
    scipy_optimizer     — gradient-free global search via Differential Evolution.
    autograd_optimizer  — gradient-based local refinement via a differentiable
                          PyTorch rollout + smooth (multi-circle) collision margin.
"""
 