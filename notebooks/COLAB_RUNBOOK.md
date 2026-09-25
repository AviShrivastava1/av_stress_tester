# Colab session runbook

The one piece of outstanding work that cannot be done on the dev machine: it needs a real
WOMD shard and the `waymo-open-dataset` package, which is manylinux-only.

`src/` and `tests/` are settled through the fifth independent audit (`b827969`): every
finding from all five rounds — the original three post-launch audits (B01–B20, B11's
`invalid_crc` case the one deliberate exception, `xfail(strict=True)`; R01–R12; A01–A14)
plus the fourth and fifth independent audits (F01–F09, G01–G08) — is closed. Nothing here
changes code. This runbook's own narrative was silent on F/G until this pass, even though
individual notebook cells had already been kept current piecemeal — cell 44's write
confirmation, for instance, already carried F07/F08/G03/G06 verbatim in its own comments
before this pass touched anything. This session **measures**, and five decisions are
waiting on what it measures.

---

## Ground rules

**Section numbers are the stable reference; cell indices are a convenience.** Every index
in this document was recounted against `len(nb['cells'])` on 2026-09-25 at 56 cells, but
inserting a cell renumbers everything after it — which has now broken a cell reference
three times in this project (Batch 4 shifted the audit's B19 fixture, Batch 2 hit a
name collision on `n_exact_match`, and this runbook's own first draft pointed at the
summary cell instead of geometry verification). If an index and a section heading
disagree, **the heading is right**. Recount with:

```python
import json
nb = json.load(open('notebooks/colab_validation_run.ipynb'))
for i, c in enumerate(nb['cells']):
    s = ''.join(c['source']).strip()
    if c['cell_type'] == 'markdown' and s.startswith('## '):
        print(i, s.splitlines()[0])
```


**Capture raw printed output for every cell.** Not a summary, not "ran successfully" — the
actual text. Every batch in this project has been reviewed against raw output, and a
session that reports conclusions without them cannot be checked by anyone, including the
person who ran it.

**A failure is a finding, not an obstacle.** Steps 2 and 3 have explicit stop conditions.
If one trips, stop and report rather than working around it — that is the whole reason
those steps run first.

**Nothing in this session is expected to change code.** If something here demands a code
change, that is a new batch, planned and reviewed like every other one.

---

## Step 1 — Environment

| | |
|---|---|
| Repo | `main` at `b827969` or later — the fifth-audit commit. Confirm with `git log --oneline -1` and record it. |
| Shard | one real `.tfrecord` on Drive, path in cell 6 `SHARD_PATH` |
| Waymo package | `waymo-open-dataset-tf-2-11-0`, `--no-deps` (cell 4) |
| Postgres/PostGIS | cells 9–11 |
| `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` | cell 2, **before** any protobuf import |

Cells 1–13, in order. Cell 13 is a one-scenario smoke test; if it fails, nothing later is
worth running.

**Config (cell 6), the values this runbook assumes:**

```
MAX_SCENARIOS = 100    Pass 1 batch size
TOP_N         = 5      Pass 2 stress-test set — UNCHANGED, see step 6
DIAG_N        = 50     7c/7d sample
B09_N         = 25     10b sample — deliberately not TOP_N, see step 5
DE_KWARGS     = popsize=15, maxiter=200, tol=1e-3, seed=1
```

---

## Step 2 — The two Linux-gated audit tests, FIRST

```bash
cd /content/av_stress_tester
AV_AUDIT_PROJECT=$PWD python -m pytest tests/test_audit_core.py -v
```

These have never run anywhere. Both fail on the dev machine for one reason only: they
import `waymo_open_dataset.protos.scenario_pb2` inside the test body, and the package is
manylinux-only.

- `test_B05_corrupt_next_record_does_not_overwrite_previous_success` — Batch 2's B05 fix
  was verified against **substitute** tests that stub `src.data.parser` in `sys.modules`.
  This is the first time the audit's own fixture runs.
- `test_control_parser_preserves_valid_states_and_flags` — builds a real `Scenario`
  protobuf and checks `ScenarioParser` preserves states and validity flags.

**Expected: `20 passed, 0 failed, 1 xfailed`.** The arithmetic, from the dev machine's
current `2 failed, 18 passed, 1 xfailed`:

```
  21 test items total
   2 failed   - 2 now passing  =  0 failed
  18 passed   + 2 now passing  = 20 passed
   1 xfailed  unchanged         =  1 xfailed
  ------------------------------------------
                          total = 21  (unchanged)
```

The xfail is `test_B11_loader_rejects_corrupt_tfrecord_framing[invalid_crc]`, marked
`strict=True` citing Block 1 Concept 2: CRC verification is opt-in by design. It is
platform-independent and **must stay xfailed**. An XPASS here means someone changed
`verify_crc`'s default, and strict mode will say so loudly.

**STOP CONDITION.** If B05 fails, Batch 2's B05 fix does not work against the real fixture
and the substitutes were not equivalent. That is a genuine finding and it invalidates a
closed batch. Capture the full traceback and stop — do not continue to Pass 1.

**Decision this feeds:** whether the audit is actually closed. It is currently reported as
19 of 20 fixed, and that claim rests on this step.

---

## Step 3 — Pass 1 across the shard

Cells 14–16. `score_shard` over `MAX_SCENARIOS`.

Every danger number in the database is stale: Batch 4 rewrote both TTC (quadratic root,
audit B07) and PET (visit pairing, audit B06), and Batch 5's B09/B20 changed Phase 4
outputs on top of that. Pass 1's ranking also selects which scenarios Pass 2 stress-tests,
so **re-running it can change which scenarios would ever have been candidates** — this is
not a refresh, it is the first ranking these engines have ever produced.

**Success:** `score_shard` completes over `MAX_SCENARIOS` with no exceptions; cell 17–18
diagnostics print; `records` is populated.

**Capture:** the timing probe from cell 15, and cell 18's full diagnostic block.

**STOP CONDITION.** Any unhandled exception. The engines are new; a crash here is a real
defect, not a data quirk.

Then cells 19–20 to build `diag_cache` (`DIAG_N = 50`), which 7c/7d/7e/7f all consume.

---

## Step 4 — The dormant cells, in dependency order

None of these have ever executed. All depend on `diag_cache` from step 3.

| Cell | Section | First written | Measures |
|---|---|---|---|
| 21–23 | 7c | Phase 3 rework | rank correlation, all-pairs vs SDC-restricted |
| 24–25 | **7d** | **Batch 4** | PET sign semantics + B06 visit separation |
| 26–27 | **7e** | **Batch 4** | TTC discrimination rate after B07 |
| 28–29 | **7f** | **Batch 4** | TTC before/after on identical inputs |

**7d** is the one with a known history: until Batch 4 this cell verified negative PETs
against *merged* occupancy spans, which meant it would have **certified the exact defect
B06 fixes** — fed the audit's B06 scene it reported no regression. It now requires an
overlapping *visit pair*. `n_multi_visit_pairs` says whether B06's condition ever arose on
real data.

**Whether it ever CHANGED the answer is measured directly (audit A13).** The cell computes
the pre-B06 merged-span PET on the same visits and compares it against the corrected
engine's value, reporting `n_multi_visit_changed`, `n_multi_visit_signflip` and the largest
change in seconds. It used to infer this from `n_multi_visit_decisive` — "did the minimum
come from a pair other than the first" — which is a proxy that fails on B06's own canonical
fixture: with `visits_a=[(0,1),(5,6)]` and `visits_b=[(3,3)]` both pairings score `+0.2`, so
the first pair is *among* the winners and the proxy reported zero, while the merged-span
answer for that fixture is `-0.3`. A half-second change that flips sign, reported as "never
decided anything". That counter survives as `n_multi_visit_later_pair`, printed separately,
because how often the minimum comes from a later pair is a real but different fact.

Compare against Block 3 v3's recorded numbers, printed inline as `[v3 measured X]`:
100/100 all-pairs saturation, 70/100 SDC-restricted, 56% negative PET pairs.

**Success:** all four run without assertion failures. `regressions` empty in 7d.

**Decisions these feed:**
- **7e's `D < 0` count** is the prerequisite for the deferred TTC saturation fallback. That
  design was deliberately deferred out of Batch 4 because it cannot be specified against a
  pre-B07 saturation rate. If that count is small, the fallback may not be worth designing
  at all.
- **7f's Spearman rho** says whether B07 was a correctness footnote or a change to which
  scenarios Phase 4 ever saw.
- **7d's `n_multi_visit_changed` / `n_multi_visit_signflip`** say the same for B06. Not
  `n_multi_visit_later_pair`, which is printed beside them and answers a different
  question — see the A13 note above.

---

## Step 5 — The new cells

### 7g — baseline replay drift sweep (cells 30–31)

Full shard, `max_baseline_drift=None`. One additional sequential Drive pass; ~4 s of
compute for ~1000 scenarios.

**Success:** completes; `scenarios measured` is close to the shard's scenario count; the
skip counters are small and explained. `never_valid` **must be 0** — it is structurally
unreachable, because `pick_nearest_challenger` only returns an agent sharing a valid frame
with the SDC, so `first_valid_index` cannot raise on it. If it is nonzero, something
upstream changed and the sweep's assumptions need re-checking.

**Also watch:** the assertion that every refusal has `reason == 'collision'`. With
`max_baseline_drift=None` the soft gate is disabled, so a `'drift'` refusal would mean the
kwarg does not do what Batch 1 documents — and would invalidate the whole sweep.

**Decision this feeds — the `max_baseline_drift` default.** Three readings, printed by the
cell:

- **bimodal with a clean valley** → put the threshold in the valley; 0.5 m is defensible
  only if that is where the valley is
- **continuous** → any threshold is arbitrary; record drift per scenario instead of gating
  on it
- **nearly everything under 0.5 m** → the gate is a tripwire, not a filter, and its value
  is catching the pathological case

The by-agent-type breakdown is a separate question worth reading on its own: vehicles use
the bicycle model, pedestrians and cyclists the linear one, and Block 2 Concept 6's
fidelity argument was derived from vehicle behaviour only. If linear-model challengers
drift differently, that argument has a gap nobody has had the data to see.

The by-interior-gap breakdown tests Batch 1's own prediction that drift concentrates there.
**A refutation is the more interesting result** and should be reported as such, not buried.

**This cell's own `except` clauses were incomplete for a sixth outcome.**
`HeadingBlendSingularityError` (independent review, 2026-09-24) is a `ValueError`
subclass — a sibling of `ReplayFidelityError`, not a child of it — and this cell's loop
used to fall through to the generic `except ValueError` for it. That clause buckets by
matching `'unusable'` in the exception's message, which a singularity refusal never
contains, so it landed in `never_valid` — the bucket this cell's own success condition
above calls **structurally unreachable** and treats a nonzero count as evidence something
broke upstream. It didn't; a real, new outcome from the fourth/fifth audits just wasn't
given its own clause. Fixed: an explicit `except HeadingBlendSingularityError` now sits
above the generic clause, with its own counter and its own row list, the same two-clause
discipline `_stress_one` already uses in `batch_scorer.py` for the identical reason. See
7g-ii, immediately below, for what this refusal measures once it's counted correctly.

**A second bug in the same cell, caught reviewing 7g-ii's new fields:** the three new
per-scenario fields 7g-ii reads (`frames_below_heading_floor`,
`frames_in_heading_transition_band`, `baseline_heading_blend_min_magnitude`) were
originally read off `space` *after* the `try`/`except` block — but the hard-gate
`except ReplayFidelityError` branch has no `continue`, and `PerturbationSpace.__init__`
raises before `space` is ever assigned on that path. On the shard's first scenario, a
hard-gate refusal there would `NameError`; on any later one, `space` still held
whatever object survived the *previous* successful construction, so the row silently
got a previous scenario's values attached to it — reproduced live, both ways, against
the audit's own hard-gate fixture (`tests/test_replay_contract.py::_colliding_baseline_scene`).
Fixed: the three fields are now captured as locals inside the `try`, right beside
`err`/`collides`, and set to `None` in the `ReplayFidelityError` branch — the same
treatment `has_interior_gap` already gets, for the same reason.

### 7g-ii — heading floor / transition / singularity calibration (cells 32–33)

`V_HEADING_MIN` (`linear_model.py`) and `HEADING_TRANSITION_WIDTH` /
`HEADING_BLEND_SINGULARITY_MARGIN` (`perturbation_space.py`) are each marked in their own
source comments as calibrated, not validated against real data, and none of the three
appear anywhere above this line in this runbook. **No second Drive pass** — 7g's own loop
above now already collects the fields this cell reads, on every scenario it visits, full
shard.

**What this cell reports, split by challenger agent type throughout, not pooled**
(independent review, 2026-09-26 — vehicles use the bicycle model and never call
`_linear_heading`, so `baseline_heading_blend_min_magnitude` is exactly `inf` and
`frames_in_heading_transition_band` is exactly `0` for every vehicle row, by
construction; WOMD is vehicle-heavy, so pooling would dilute the exact question this
cell exists to answer with structural zeros/infs. Same pattern 7g's own "drift by
challenger agent type" section already uses):
- `frames_below_heading_floor` / `frames_in_heading_transition_band` percentiles, per
  agent type, over every scenario that both survived construction *and* did not trip
  the hard replay-fidelity gate — full shard, not the `TOP_N = 5` Pass-2 sample.
  (`stress_results`'s own `search_provenance` carries these two per Pass-2 scenario
  already; reading that instead would be strictly worse, N=5 against N≈shard size, so
  it is not read separately here.)
- The singularity guard's fire rate against the attempted population (constructions
  actually reached — excludes `no_challenger`, where nothing was ever attempted).
- `baseline_heading_blend_min_magnitude` percentiles per agent type, for the same
  surviving-and-not-hard-gated population — how close real scenarios sit to the 60°
  margin, not just whether they cross it.

**A hard-gate collision refusal (`ReplayFidelityError('collision', ...)`) carries none
of these three fields, and 7g's cell reflects that explicitly rather than silently
reusing a previous scenario's values.** `PerturbationSpace.__init__` raises before
returning the object, so nothing on `self` is recoverable there — the same limitation
`has_interior_gap` already has, described above — and the exception itself doesn't carry
these three the way it carries `baseline_replay_error`/`baseline_replay_collides`
(confirmed against its `__init__`). 7g's `except ReplayFidelityError` branch sets all
three to `None` for that row; this cell filters those rows out (`measured_rows`) before
computing any of the percentiles above, and prints the excluded count.

**Success:** completes without exception, and 7g's own tripwire count
(`skipped['heading_blend_singularity']`) agrees with this cell's per-scenario detail
(`len(singularity_rows)`) — a mismatch means the two cells drifted apart.

**Decision this feeds — the fifth item in "what to bring back."** Three readings, the
same shape as 7g's own `max_baseline_drift` decision:
- refusals rare, surviving population sits well clear of the margin → `V_HEADING_MIN`,
  `HEADING_TRANSITION_WIDTH`, and `HEADING_BLEND_SINGULARITY_MARGIN` stand as defended
  defaults, not just calibrated ones
- meaningful mass sits close to the margin → the margin, or the floor/width it sits on,
  needs revisiting
- refusals are common → the antipodal-heading defect is not a corner case on real data,
  and the structural redesign G02 deferred (constrain DE's search domain, or rebuild the
  blend against a singularity-free anchor) needs reconsidering, not just re-margined

### 10b — B09 before/after (cells 41–42)

Runs after Pass 2 diagnostics because it needs `ranked`. `B09_N = 25`, ~8 minutes.

**Read the self-test line first.** The cell reconstructs the pre-B09 behaviour (no flag
recovers it), and validates that reconstruction against a fixture where an *iterate* wins.
If it prints `FAIL`, the cell raises and every number after it would have been meaningless.

**Then read `fidelity guard informative on: M/N`.** The per-scenario guard proves nothing
when the warm start wins, which is most of the time. If `M = 0`, the reconstruction rests
on the self-test plus code inspection — the cell says so itself.

**Then read the sample size, which prints directly above the conclusion.** A zero with
`n < 20` is reported as *"no evidence it fires"*, never as *"evidence it does not"*.

**Decision this feeds:** whether B09 needs a Block 4 doctrine note. If it fires on real
data, Block 4 describes a refinement stage that returns its endpoint — materially wrong,
and a new textbook correction (Block 2's and Block 6's are already fixed; this would be
the first one still open).

**This cell's own `except` clause had the same gap 7g's had.** `HeadingBlendSingularityError`
is a `ValueError` subclass, and this cell's single `except (ReplayFidelityError, ValueError)`
folded a singularity refusal into `infeasible` — a real, new outcome silently miscounted as
an old one. Fixed with its own explicit `except HeadingBlendSingularityError` clause and its
own `heading_blend_singularity` counter, printed alongside `infeasible`.

### G01 + G04 — a decision, not a cell

Both add an atomic identity check against a *previous* write: `upsert_scores`'s
`UpsertReport.rescoped` (G01) and `export_scenario_agents`/`get_trajectories`'s `sdc_idx`
guard (G04). Both need a scenario to be written once, then rescored under a genuinely
different scene fingerprint or SDC index, before either has anything to catch. This
session runs Pass 1 once, over a shard being read for the first time — there is no
earlier write for anything here to diverge from. Section 8's idempotency check (cell 36)
already calls `upsert_scores` twice with *identical* records; it now also asserts
`.rescoped` is empty both times — a cheap negative-control tripwire, not a validation of
the rescope path itself. Confirming the guards actually fire needs a second pass over the
same shard after a code or data change that alters an already-scored scenario's identity
— different infrastructure than a single real-shard run provides.

### F04 — a decision, not a cell

Guards `export_perturbed_path`'s 5-argument, no-`delta` calling convention against
silently borrowing a stale run id. `export_shard_geometry` — the only caller this
notebook ever exercises (section 12) — always supplies `delta` and `method`, so the
vacuous path F04 closes is never the one taken here. A real-shard run cannot exercise
this without a second, narrower caller this pipeline doesn't have.

### F05 + G08 — a decision, not a cell

Both replace a *recomputed* value with a *reused* one at export time: F05 reuses
`export_scenario_agents`'s already-verified `scene_fingerprint` instead of recomputing
it; G08 reuses the delta's own `search_provenance` (`heading_speed_floor`,
`heading_transition_width`) instead of today's ambient module defaults. Both code paths
run on every export this session makes — unlike G01/G04/F04, nothing here is skipped.
But neither fix is *observable* without a condition this run can't produce: F05's
reuse-vs-recompute distinction only shows up on a legacy row with no fingerprint
recorded (every row here gets one, fresh, from Pass 1); G08's provenance-vs-ambient
distinction only shows up if the module constants changed between search and export
(nothing changes them mid-process). A before/after here would report identical numbers
either way, for reasons that have nothing to do with whether the fix works.

### G07 — a decision, not a cell

Fixes a cross-schema column-count false positive in the `perturbed_paths` migration
check, reachable only when an unrelated schema on the search path has a same-named
table. A normally-configured Colab Postgres instance doesn't have that; producing it
would mean deliberately polluting the schema, not running a real shard. Already covered
by its own reconstructed fixture in `tests/test_f08_geometry_schema_migration.py` — out
of scope here.

### B20 — a decision, not a cell

No before/after cell, deliberately. Batch 5 established by repo-wide search that **nothing
in `src/` or `tests/` has ever passed custom `bounds`** except the audit's own B20 fixture.
Every production call uses symmetric defaults, where `max(|low|, |high|) == |high|` and the
new weights are identical term for term — verified elementwise for both default sets.

A real-data before/after would measure exactly zero differences. A cell whose output is
guaranteed to be "no change" is ceremony, not evidence. B20's value is that a *future*
one-sided bound will be priced correctly. Recorded here so the absence is legible rather
than looking forgotten.

---

## Step 6 — Pass 2, and the rest of the pipeline

Sections 8 through 10 (rank + persist, `stress_test_scenarios`, diagnostics) at cells
35–40, then 10b, then sections 11 through 15 at cells 43–55 (update, Pass 3 export,
geometry verification, API round trip, summary).

**`TOP_N` stays at 5.** Phase 5's architecture is a cheap filter feeding an expensive pass
over a small selected set; running Pass 2 at a size chosen to make a measurement look
better would misrepresent how the pipeline works. 10b gets its own `B09_N` instead —
that decoupling is the point.

**Geometry verification (section 13, cell 49)** carries Batch 5's B19 fix: it now asserts *coverage
before correctness*. The old version passed vacuously at `0 == 0` when every agent was
missing from the database — the cell whose job is catching missing geometry was blind to
geometry being missing in full. It now tracks expected-exportable agents and fails if any
is absent.

**Note on the audit's B19 fixture:** it used to locate this cell by index, and Batch 4's
four inserted cells silently broke it — the test began exec'ing a markdown cell and dying
with `SyntaxError` instead of reaching its assertion, staying red either way so the count
never moved. Batch 5 switched it to content lookup. A later session's two new cells (7g,
10b) shifted indices again with the fixture unaffected, and this pass's own two new cells
(7g-ii) do the same — which is the fix doing its job, again.

**Success:** section 13's code cell prints `CONFIRMED: all N exportable agents present ...`; the API round
trip closes the loop between HTTP timesteps and the M values read directly from PostGIS.

**If no scenario in the Pass 2 sample produced a verified collision** — an ordinary outcome
at `TOP_N = 5`, not a failure — section 14 no longer crashes on it (audit R09). It falls
back to any scenario with exported geometry, prints every outcome it did see, and states
explicitly that collision-specific checks were **skipped, not passed**. The round-trip
assertion still runs, on a non-colliding agent; it never needed a delta. Read that notice
if it appears: it means the `/perturbed` delta and `collision_timestep` printed as null
legitimately, and it is also the signal that `B09_N`'s sample in section 10b may contain
few or no usable comparisons.

---

## What to bring back

1. `git log --oneline -1` — the exact commit this ran against.
2. Raw output for every cell in steps 2–6. Not summaries.
3. For each of the five decisions: the number, and which reading it supports.
   - `max_baseline_drift` default — from 7g
   - `V_HEADING_MIN` / `HEADING_TRANSITION_WIDTH` / `HEADING_BLEND_SINGULARITY_MARGIN`
     calibration — from 7g-ii
   - B09 doctrine note — from 10b
   - TTC saturation fallback, worth designing or not — from 7e
   - whether the audit is genuinely closed — from step 2
4. Anything that failed, with its full traceback.

## What this session does not do

**No code changes.** If something here demands one, it is a new batch.

**Doctrine corrections already made.** Block 6 Concept 24's `stress_tested_at`/`robustly_safe`
passage was corrected in Block 6 v2, citing audit B04. Block 2 §6/§7's common-mode-cancellation
claim was corrected in Block 2 v3, citing audit B03 — not Block 4 §17, which is Perturbation
Space Design and never carried this claim. Neither correction needed a shard or a Linux box,
and neither is open now. One related item remains: if 10b shows B09 firing on real data,
Block 4's description of the refinement stage is wrong in a new way and needs its own note —
and that's its own pass, not this one.
