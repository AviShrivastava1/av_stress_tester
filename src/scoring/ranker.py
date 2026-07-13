"""
ranker.py — Phase 5.

Orders scored scenarios by fragility and selects the top-N for the expensive
Phase 4 stress-test pass. Deliberately small: ranking is simple, but two details
matter for a safety-auditing tool and are easy to get wrong.

  1. Determinism. Python's sort is stable, but two scenarios can tie on
     fragility_score. We tiebreak on scenario_id so the SAME input always
     produces the SAME ranking — reproducibility again, same reason as the DE
     seed. A top-50 list that reshuffles between runs is an audit nightmare.

  2. The ranking is DESCENDING on fragility: rank 1 = most fragile = closest
     to catastrophe. That is the ordering the dashboard, the database index,
     and the stress-test pass all assume.
"""

import numpy as np


def rank_scenarios(records):
    """
    Sort by fragility (desc), tiebreak by scenario_id (asc), and attach a
    1-based 'rank' to each record. Returns a NEW sorted list; input not mutated.
    """
    ranked = sorted(
        records,
        key=lambda r: (-r['fragility_score'], str(r['scenario_id'])),
    )
    out = []
    for i, r in enumerate(ranked):
        r = dict(r)          # copy — don't mutate caller's records
        r['rank'] = i + 1
        out.append(r)
    return out


def top_n(records, n):
    """Rank and return the n most fragile scenario records."""
    return rank_scenarios(records)[:n]


def top_n_ids(records, n):
    """Convenience: just the scenario IDs of the top-N (input to PASS 2)."""
    return [r['scenario_id'] for r in top_n(records, n)]


def summarize(records):
    """
    Small sanity-check summary of a scoring run. Useful to print after PASS 1
    and to store alongside the batch in the DB/logs.
    """
    if not records:
        return {'count': 0}
    frag = np.array([r['fragility_score'] for r in records], dtype=float)
    ttc = np.array([r['min_ttc'] for r in records], dtype=float)
    pet = np.array([r['min_pet'] for r in records], dtype=float)
    return {
        'count':            len(records),
        'fragility_mean':   float(frag.mean()),
        'fragility_max':    float(frag.max()),
        'fragility_p95':    float(np.percentile(frag, 95)),
        'n_finite_ttc':     int((ttc < 999.0).sum()),
        'n_finite_pet':     int((pet < 999.0).sum()),
    }