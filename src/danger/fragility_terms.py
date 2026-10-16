"""
fragility_terms.py — the two terms of the stored fragility score, from the stored TTC and PET.

No numpy, so the deployed API (which installs neither numpy nor scipy) can import it. It does NOT replace the scoring code:
src/danger/danger_score.py compute_danger_score is what scored the scenarios, and this module restates its arithmetic for the
case the pipeline always uses (no perturbation term) so the API can show how a stored score is made up. Nothing here is
imported by the scoring path, and tests/test_fragility_terms.py pins every constant and the whole formula against
compute_danger_score so that the two cannot drift apart without a test failing.

The formula, from danger_score.py with min_perturbation=None (which is what batch_scorer.py passes: the score ranks the
recorded scene only):

    ttc_signal = 1 / max(min_ttc, TTC_FLOOR)   if min_ttc < 999.0   else 0
    pet_signal = 1 / max(min_pet, PET_FLOOR)   if min_pet < 999.0   else 0
    fragility  = (W_TTC / (W_TTC + W_PET)) * ttc_signal + (W_PET / (W_TTC + W_PET)) * pet_signal
               = (0.4 / 0.7) * ttc_signal + (0.3 / 0.7) * pet_signal

min_ttc and min_pet are the SDC-restricted values. 999.0 is the sentinel for "no closing pair" / "no shared conflict zone",
which contributes nothing. The floor means a TTC or PET at or below 0.01 s, a PET of exactly zero and any overlap (a negative
PET) included, contributes the same 100 times its weight: a score of exactly 100 is both terms at the floor.
"""

TTC_FLOOR = 0.01
PET_FLOOR = 0.01
W_TTC = 0.4
W_PET = 0.3
#: The sentinel the TTC and PET engines store for "none" (ttc_engine.TTC_INFINITY, pet_engine.PET_INFINITY).
NO_CONFLICT = 999.0

#: How closely the recomputed terms must add up to the stored score to be called the same number. The stored score is a double
#: computed by this arithmetic from the same stored inputs, so they agree to the last bit; this is only roundoff headroom.
TOLERANCE = 1e-9


def signal(value: float, floor: float) -> float:
    """1 / max(value, floor), or 0 for the "none" sentinel."""
    return 1.0 / max(value, floor) if value < NO_CONFLICT else 0.0


def breakdown(min_ttc: float | None, min_pet: float | None, stored_score: float) -> dict | None:
    """
    The two terms of the score for these stored TTC and PET values, and whether they add up to the stored score.

    None when either value is not stored (nothing to decompose). Otherwise a dict with the weights, the signals, the terms,
    their sum, whether each input is the "none" sentinel or sits at the floor, and `reproduces_stored_score`.
    """
    if min_ttc is None or min_pet is None:
        return None
    w_total = W_TTC + W_PET
    ttc_weight = W_TTC / w_total
    pet_weight = W_PET / w_total
    ttc_signal = signal(min_ttc, TTC_FLOOR)
    pet_signal = signal(min_pet, PET_FLOOR)
    ttc_term = ttc_weight * ttc_signal
    pet_term = pet_weight * pet_signal
    recomputed = ttc_weight * ttc_signal + pet_weight * pet_signal
    return {
        'ttc_weight': ttc_weight,
        'pet_weight': pet_weight,
        'ttc_signal': ttc_signal,
        'pet_signal': pet_signal,
        'ttc_term': ttc_term,
        'pet_term': pet_term,
        'recomputed_score': recomputed,
        'ttc_floor_s': TTC_FLOOR,
        'pet_floor_s': PET_FLOOR,
        'ttc_none': min_ttc >= NO_CONFLICT,
        'pet_none': min_pet >= NO_CONFLICT,
        'ttc_at_floor': min_ttc < NO_CONFLICT and min_ttc <= TTC_FLOOR,
        'pet_at_floor': min_pet < NO_CONFLICT and min_pet <= PET_FLOOR,
        'reproduces_stored_score': abs(recomputed - stored_score) <= TOLERANCE * max(1.0, abs(stored_score)),
    }
