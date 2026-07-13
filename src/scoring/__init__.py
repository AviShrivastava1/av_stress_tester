"""
Phase 5 — Scenario Scoring at Scale.

Turns the single-scenario tools from Phases 3-4 into a batch pipeline:

    batch_scorer — iterate a WOMD shard, compute the danger profile (TTC, PET,
                   fragility score) for every scenario, with per-scenario error
                   isolation; plus a second, explicit Phase 4 stress-test pass
                   over a chosen subset (typically the top-N most dangerous).
    ranker       — order scored scenarios by fragility and select the top-N.
    db           — persist results to PostgreSQL (idempotent upserts, ranked reads).

Design: the cheap signals (TTC/PET) run on EVERY scenario; the expensive
optimizer (Phase 4) runs only on the top-N ranked by those signals. That is the
decision that keeps the system tractable at 100k+ scenarios.
"""
