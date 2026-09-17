# Colab session runbook

The one piece of outstanding work that cannot be done on the dev machine: it needs a real
WOMD shard and the `waymo-open-dataset` package, which is manylinux-only.

`src/` and `tests/` are settled through Batch 5 (`8e6cfb7`). Nothing here changes code.
This session **measures**, and four decisions are waiting on what it measures.

---

## Ground rules

**Section numbers are the stable reference; cell indices are a convenience.** Every index
in this document was recounted against `len(nb['cells'])` on 2026-09-15 at 54 cells, but
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
| Repo | `main` at `8e6cfb7` or later. Confirm with `git log --oneline -1` and record it. |
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
DE_KWARGS     = popsize=10, maxiter=60, tol=1e-2, seed=1
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

## Step 5 — The two new cells

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

### 10b — B09 before/after (cells 39–40)

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
and a **third** outstanding textbook correction.

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
33–38, then 10b, then sections 11 through 15 at cells 41–53 (update, Pass 3 export,
geometry verification, API round trip, summary).

**`TOP_N` stays at 5.** Phase 5's architecture is a cheap filter feeding an expensive pass
over a small selected set; running Pass 2 at a size chosen to make a measurement look
better would misrepresent how the pipeline works. 10b gets its own `B09_N` instead —
that decoupling is the point.

**Geometry verification (section 13, cell 47)** carries Batch 5's B19 fix: it now asserts *coverage
before correctness*. The old version passed vacuously at `0 == 0` when every agent was
missing from the database — the cell whose job is catching missing geometry was blind to
geometry being missing in full. It now tracks expected-exportable agents and fails if any
is absent.

**Note on the audit's B19 fixture:** it used to locate this cell by index, and Batch 4's
four inserted cells silently broke it — the test began exec'ing a markdown cell and dying
with `SyntaxError` instead of reaching its assertion, staying red either way so the count
never moved. Batch 5 switched it to content lookup. **This session's two new cells shift
indices again, and the fixture is unaffected** — which is the fix doing its job.

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
3. For each of the four decisions: the number, and which reading it supports.
   - `max_baseline_drift` default — from 7g
   - B09 doctrine note — from 10b
   - TTC saturation fallback, worth designing or not — from 7e
   - whether the audit is genuinely closed — from step 2
4. Anything that failed, with its full traceback.

## What this session does not do

**No code changes.** If something here demands one, it is a new batch.

**No doctrine corrections.** Block 6 Concept 24's `stress_tested_at` prose and Block 2 §6 /
Block 4 §17's common-mode cancellation claim have now been deferred through five batches
and this session makes six. They are known-false statements in the project's own teaching
material, they need no shard and no Linux box, and 10b may add a third to the pile. They
want their own small pass, and the fact that they have never fit anyone's scope is the
argument for giving them one rather than the reason to keep deferring.
