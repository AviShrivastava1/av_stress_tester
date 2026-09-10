import numpy as np
from src.danger.ttc_engine import (
    compute_min_ttc_scenario, compute_min_ttc_sdc, TTC_INFINITY,
)
from src.danger.pet_engine import (
    compute_min_pet_scenario, compute_min_pet_sdc, PET_INFINITY,
)


TTC_FLOOR = 0.01    # minimum TTC to avoid division by zero
PET_FLOOR = 0.01    # minimum PET to avoid division by zero

# weights for combining signals into fragility score
W_TTC         = 0.4
W_PET         = 0.3
W_PERTURBATION = 0.3


def compute_danger_score(
    min_ttc: float,
    min_pet: float,
    min_perturbation: float = None
) -> float:
    """
    Combine TTC, PET, and perturbation magnitude into a single fragility score.
    Higher score = more fragile = closer to catastrophe.

    Uses reciprocals because smaller TTC/PET/perturbation = more dangerous.

    Args:
        min_ttc:          minimum TTC across scenario (seconds)
        min_pet:          minimum PET across scenario (seconds)
        min_perturbation: minimum perturbation magnitude from Phase 4
                          (None if Phase 4 not yet run)

    Returns:
        fragility_score: float, higher is more dangerous
    """
    # clamp to floor to avoid division by zero
    ttc_safe = max(min_ttc, TTC_FLOOR)
    pet_safe = max(min_pet, PET_FLOOR)

    # reciprocals — smaller value = larger contribution = more dangerous
    ttc_signal = 1.0 / ttc_safe if min_ttc < TTC_INFINITY else 0.0
    pet_signal = 1.0 / pet_safe if min_pet < PET_INFINITY else 0.0

    if min_perturbation is not None:
        pert_safe = max(min_perturbation, 1e-6)
        pert_signal = 1.0 / pert_safe
        score = W_TTC * ttc_signal + W_PET * pet_signal + W_PERTURBATION * pert_signal
    else:
        # Phase 4 not run yet — distribute weight between TTC and PET
        w_ttc_adj = W_TTC / (W_TTC + W_PET)
        w_pet_adj = W_PET / (W_TTC + W_PET)
        score = w_ttc_adj * ttc_signal + w_pet_adj * pet_signal

    return float(score)


def score_scenario(
    states: np.ndarray,
    validity: np.ndarray,
    scenario_id: str,
    sdc_index: int,
    min_perturbation: float = None,
    pet_max_pairs: int = 50
) -> dict:
    """
    Compute the full danger profile for one scenario.

    `sdc_index` is required — no default. The fragility score is now computed from
    TTC/PET restricted to pairs involving the self-driving car, because the
    all-pairs versions saturate on real WOMD data (a dense scene's scene-wide
    minimum TTC is driven to zero by any two agents passing close, regardless of
    whether the SDC was ever at risk — 100/100 scenarios on the validation shard).
    A forgotten `sdc_index` must be a loud TypeError, not a silent fall back to
    the all-pairs behaviour that this rework exists to stop ranking on.

    The all-pairs values are still computed and returned as `min_ttc_all_pairs` /
    `min_pet_all_pairs` — a cheap "how crowded was this scene" diagnostic — but
    they never feed `compute_danger_score`.

    Args:
        states:           shape (N, T, 7)
        validity:         shape (N, T)
        scenario_id:      string ID from ScenarioParser
        sdc_index:        index of the self-driving car (from get_sdc_index())
        min_perturbation: from Phase 4 optimizer (optional)
        pet_max_pairs:    max agent pairs to check for the all-pairs PET diagnostic

    Returns:
        dict with scenario_id, min_ttc, min_pet (both SDC-restricted),
        fragility_score, min_perturbation, and the min_ttc_all_pairs /
        min_pet_all_pairs diagnostics.
    """
    min_ttc_sdc = compute_min_ttc_sdc(states, validity, sdc_index)
    min_pet_sdc = compute_min_pet_sdc(states, validity, sdc_index)

    min_ttc_all = compute_min_ttc_scenario(states, validity)
    min_pet_all = compute_min_pet_scenario(states, validity, max_pairs=pet_max_pairs)

    fragility = compute_danger_score(min_ttc_sdc, min_pet_sdc, min_perturbation)

    return {
        'scenario_id':      scenario_id,
        'min_ttc':          min_ttc_sdc,
        'min_pet':          min_pet_sdc,
        'min_ttc_all_pairs': min_ttc_all,
        'min_pet_all_pairs': min_pet_all,
        'fragility_score':  fragility,
        'min_perturbation': min_perturbation
    }