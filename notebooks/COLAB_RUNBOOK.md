# Colab session runbook

The one piece of outstanding work that cannot be done on the dev machine: it needs a real
WOMD shard and the `waymo-open-dataset` package, which is manylinux-only.

`src/` and `tests/` are settled through the fifth independent audit (`b827969`): every
finding from all five rounds — the original three post-launch audits (B01–B20, B11's
`invalid_crc` case the one deliberate exception, `xfail(strict=True)`; R01–R12; A01–A14)
plus the fourth and fifth independent audits (F01–F09, G01–G08) — is closed. Nothing here
changes code. This runbook's own narrative was silent on F/G until this pass, even though
individual notebook cells had already been kept current piecemeal — cell 54's write
confirmation, for instance, already carried F07/F08/G03/G06 verbatim in its own comments
before this pass touched anything. This session **measures**, and five decisions are
waiting on what it measures.

---

## Ground rules

**Section numbers are the stable reference; cell indices are a convenience.** Every index
in this document was recounted against `len(nb['cells'])` on 2026-09-28 at 66 cells, but
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

**Colab's own limits.** Google's FAQ gives a maximum runtime of "at most 12 hours,
depending on availability and your usage patterns", says runtimes "time out if you are
idle", and offers no background execution on the free tier. **The idle timeout is not
documented**: no duration is given, and nothing says whether a running cell counts as
activity. The re-run at `62ab1a6` (10b skipped, `TOP_N = 20`, the drift-gate batch's search)
is the measured reference. Its own printed timings: shard cache build 264.4 s
(`[CACHE-TIMING] shard cache build`), Pass 1 151.0 s (`Pass 1 done`), Pass 2 1,521.1 s for 20
scenarios, 18 searched and 2 refused for drift (`Pass 2 done`), Pass 3 59.4 s (`elapsed` in the
Pass 3 summary). Derived from those, not printed: the four add up to 1,995.9 s, about 33
minutes, not counting the setup before the cache build or the dump after Pass 3; Pass 2 is
about 25 minutes of it; and if the two refusals cost nothing, a search took about 84.5 s
(1,521.1 / 18), where the notebook prints 76.1 s per scenario over all 20. Pass 2 is the
longest single step, which is what the Pass 2 checkpoint below is for. The run is still far
inside the lifetime cap, so the realistic risks are an idle disconnect and a lost
connection. So:
keep the Colab tab open and in front; keep the Mac awake (`caffeinate -dims` in a
terminal, left running until the dump is done); and run in blocks — cells 1–13, read the
pre-write report, then the rest.

**Clear all outputs before committing the notebook** (Colab: *Edit → Clear all outputs*,
then save). In external-database mode, cell outputs contain the database host and
everything a later cell prints about the data, and this repository is public.
`tests/test_notebook_committed_clean.py` fails if the committed notebook carries any
outputs or execution counts, so this is enforced, not just remembered.

---

## Step 1 — Environment

| | |
|---|---|
| Repo | `main` at `51a02c9` or later — the fifth-audit commit (`b827969`) plus this runbook/notebook's own reconciliation with it. Confirm with `git log --oneline -1` and record it. |
| Shard | one real `.tfrecord` on Drive, path in cell 6 `SHARD_PATH` |
| Waymo package | `waymo-open-dataset-tf-2-11-0`, `--no-deps` (cell 4) |
| Postgres/PostGIS | cells 11–13; `DB_MODE` in cell 6 picks a throwaway local database or a hosted one (see *External database* below) |
| `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` | cell 2, **before** any protobuf import |

Cells 1–15, in order. Cell 15 is a one-scenario smoke test; if it fails, nothing later is
worth running. **Section 3b's freshness guard (cells 8–9) is new** (independent review,
2026-09-27) — cell 9 runs `git log --oneline -1` automatically and hard-asserts
`HeadingBlendSingularityError` is importable from the just-cloned `src/`, so a stale
clone or a hand-edited `DE_KWARGS` fails loudly here instead of producing 20 minutes of
output from a different codebase. See cell 8's own markdown for exactly what it does and
does not catch.

**Config (cell 6), the values this runbook assumes:**

```
MAX_SCENARIOS = 100    Pass 1 batch size
TOP_N         = 20     Pass 2 stress-test set — 20 for the site's dataset, see step 6
DIAG_N        = 50     7c/7d sample
RUN_B09       = False  10b skipped by default: its result is recorded, see step 5
B09_N         = 25     10b sample — deliberately not TOP_N, see step 5
DE_KWARGS     = popsize=15, maxiter=200, tol=1e-3, seed=1
DB_MODE       = "colab_local"   or "external" for the A_MAX=12 re-run, see below
DB_URL_SECRET = "AV_STRESS_DATABASE_URL"
```

**External database — for the `A_MAX=12` re-run.** Every earlier run wrote to a Postgres
installed inside Colab, which disappears with the session; no data from those runs
survives. The re-run writes to a hosted Render Postgres instead, so it produces the first
dataset that persists. Render's documentation (checked 2026-09-29) says:

- External connections must use TLS: Render rejects `sslmode=disable`. Section 4 sets
  `PGSSLMODE=require` unless the URL asks for `verify-ca` or `verify-full`, and refuses
  any weaker mode.
- PostGIS is enabled by the database user with `CREATE EXTENSION` on Postgres 13 and
  later, which `init_geometry_schema` already runs. Cell 13 checks the server version and
  `PostGIS_Version()` before anything is written.
- **A free database expires 30 days after creation, then Render deletes it after a 14-day
  grace period, and free databases have no backups of any kind.** Whether to use a paid
  plan is decided before deploying, not here. Either way, take the dump below.

Before the session:

1. Create the Render Postgres: version 13 or later, in a **US** region (Colab VMs are
   usually US-hosted, so round trips stay short; not verified). Leave external access open
   to all IPs for the run, because Colab's outgoing address changes between sessions.
2. Copy its **external** database URL (`postgresql://user:password@host:5432/dbname`)
   into a Colab secret named `AV_STRESS_DATABASE_URL` (key icon in the left sidebar), and
   enable notebook access for it. Never paste the URL into a cell.
3. Set `DB_MODE = "external"` in cell 6, in Colab only. Don't commit that change: the
   committed default stays `"colab_local"`.

**Check cell 12's output before running anything after it.** It must print
`External database: host=<the Render host>, …`. If it prints `Postgres running, role=avi`,
stop: the run is about to write to Colab's throwaway database, which is what happened at
`19fddb4`. Set `DB_MODE = "external"` in cell 6 and restart from cell 1.

In the session, cell 12 prints only the host, database name and sslmode, and cell 13
prints how many stored results the database already holds, grouped by the `A_MAX` each
search recorded. On a fresh database that is zero. A resumed run is allowed (every pass
upserts), but results from different caps must not end up side by side unnoticed.

**Immediately after the run, dump the database**, because a free database keeps no
backups. Do it from the dev machine, not Colab: in external mode Colab has no Postgres
client installed. `pg_dump` refuses a server whose major version is newer than its own
(this project's Homebrew default `pg_dump` 16 already refused a Postgres 17 server once),
so check the server first and use a matching client, e.g. Homebrew's `postgresql@17`.
Read the URL at a silent prompt, so the password never lands in shell history (typing
`export AV_STRESS_DATABASE_URL=postgresql://…` would put it in `~/.zsh_history`); paste
the URL when the prompt waits:

```
read -rs AV_STRESS_DATABASE_URL && export AV_STRESS_DATABASE_URL
psql "$AV_STRESS_DATABASE_URL" -Atc "SHOW server_version"
/opt/homebrew/opt/postgresql@17/bin/pg_dump --no-owner --no-privileges -Fc \
    -f av_stress_amax12.dump "$AV_STRESS_DATABASE_URL"
```

Never write the URL into a file in this repository. Keep a copy of the dump on Drive as
well. To use it locally, restore into a **new** database, not `av_stress` (that one holds
the four synthetic rows; restoring on top would fail on existing tables or mix synthetic
and real rows). Local PostGIS is already installed, so the extension restores with it:

```
createdb av_stress_amax12
pg_restore --no-owner -d av_stress_amax12 av_stress_amax12.dump
```

Point the local API at it with `PGDATABASE=av_stress_amax12`; that is the real-scale data
the frontend is checked against. Afterwards, external access can be restricted or turned
off: a Render-hosted API connects through the internal URL.

**The re-run at `62ab1a6` dumped inside Colab, and only the three tables.** It used the
runtime's own throwaway Postgres (`DB_MODE = 'colab_local'`; server, `pg_dump` and
`pg_restore` all 16.15, PostGIS 3.4, as its version gate printed), so the dump was made
before anything left Colab. Its cell ran
`pg_dump --format=custom --no-owner --no-privileges -t scenario_scores -t scenario_agents -t perturbed_paths`:
`--exclude-extension` exists only from pg_dump 17, so the extension is kept out by naming the
tables, and the cell then asserted that `pg_restore -l` listed exactly 14 entries, none of
them an EXTENSION. It also refused to run unless `pg_dump`'s major version equalled the
server's, `pg_restore`'s equalled `pg_dump`'s, and `pg_dump` was no newer than 17, because
the `pg_restore` that loads the dump is 17. Replacing the data in the hosted database from
such a dump is described in DEPLOY.md, "Replacing the data in a live database".

That re-run was started from scratch cells that are not committed (they sit beside the
notebook's cells and are not part of it): a fresh clone of `main` whose `HEAD` was asserted
to be the pinned commit, equal to `origin/main`, with a clean tree; a checkpoint folder named
for that commit, with any existing folder or old `av_stress_*.pkl` checkpoint moved aside and
never deleted; and a check after each pass that its checkpoint was computed, not resumed
(`Pass 1 computed 100, resumed 0`, `Pass 2 computed 20 stress results, resumed 0`).

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

### 5b — shard cache (cells 16–17)

New this pass. Six or more cells each called `for raw in ShardLoader(SHARD_PATH):` and
re-parsed the shard through the pure-Python protobuf implementation this notebook
forces — roughly 0.5s/scenario, so 4-5 minutes per full pass, and the cells hunting for
a specific 20 or so IDs still read most of the shard to find them. This cell pays that
cost once, in shard order, right after the smoke test (which is where
`ShardLoader`/`ScenarioParser` get imported) and before the timing probe — the probe
still runs against 3 freshly-parsed scenarios, unconverted on purpose, since it exists
to measure raw parse cost for the `MAX_SCENARIOS` projection below, and converting it
would make it measure the cache instead.

**Read-only by design.** Every cached array gets `flags.writeable = False` right after
the build. A consumer that mutates one in place fails loudly with a `ValueError` right
there — a stop-and-report finding about that consumer, not something to route around
with a defensive `.copy()` in this cell or anywhere downstream.

**Parity guard**, in the same cell: re-parses the first, middle, and last scenario
directly and asserts `np.array_equal` against the cache for states, validity, and
types, plus asserts the record count read matches the cache length. This constructs a
`ScenarioParser` — the expensive `ParseFromString` plus the per-track Python loop in
`get_agent_states` — for only those 3 records; `ShardLoader` itself does raw
struct-level byte reads, not protobuf parsing (confirmed by reading
`src/data/loader.py`), so this pass should be bounded mostly by file I/O for the
~493 records it skips. **Its cost is measured, not estimated** — the `[CACHE-TIMING]`
line prints the real number; do not assume it costs anywhere near a second full build
pass just because it walks to the end of the shard.

**Ten loop instances across nine cells convert** to read from `shard_cache` instead of
re-parsing: the validity-gap scan in Pass 1 diagnostics, 7b (now a one-line slice,
`diag_cache = shard_cache[:DIAG_N]`, since its dict keys are identical), 7g, 7g-iii,
7g-iv, 7g-v (two passes), the Pass 2 diagnostics re-parse, 10b, and geometry
verification. Each conversion replaces only the loop header (`for raw in
ShardLoader(SHARD_PATH):` → `for c in shard_cache:`) and the `p.get_*()` calls
(`p.get_agent_states()` → `c['states']`, etc.) — nothing else in any of those cells
changed. 7c/7d/7e/7f, and the cells above that already consumed `diag_cache`, need no
change at all: they already read `d['states']`/`d['validity']`/`d['sdc_idx']` from a
dict, and now that dict is `shard_cache`'s (or its slice's) read-only version instead
of a freshly-parsed one.

**What does not convert, and why:** the smoke test (cell 15) — it's where the imports
this cache depends on come from, and it runs before the cache exists. The timing probe
(cell 19) — deliberately left reading raw, uncached scenarios; see above. `score_shard`,
`stress_test_scenarios`, and `export_shard_geometry` (cells 20, and their own Pass
2/Pass 3 calls) — these are `src/`-side functions that open their own `ShardLoader`
internally; changing them is a `src/` batch, not this one. Measured costs for these
three: `score_shard` 164s, `stress_test_scenarios` 255s, `export_shard_geometry` not
independently measured (structurally the same targeted re-read as Pass 2, over a
similarly small scenario-id set — expect a comparable or smaller order of magnitude,
unconfirmed). None of the three shrink from this cache; the saving is only in the ten
notebook-level loops.

**Verified** against constructed multi-agent fixtures (interior gap, non-vehicle
target, hard-gated row included) run through the real `PerturbationSpace`/danger
engines: every one of the ten converted loops produces byte-identical output to its
original, and 7c/7d/7e/7f produce byte-identical output whether `diag_cache` was built
by the original loop or is a `shard_cache` slice — confirming the actual consumers
(`compute_min_ttc_scenario`, `compute_pet_pair`, `score_scenario`, `PerturbationSpace`,
etc.) don't just avoid crashing on a read-only array, they produce the same numbers.
`PerturbationSpace.states0 = states.astype(np.float32)` copies regardless of the input
array's dtype (numpy's default is `copy=True`), so it's independently writable while
the source cache entry stays untouched and read-only — demonstrated, not just grepped.

**A resume checkpoint, described here, not as a notebook cell:** the runtime can
restart mid-session, and re-running 7g/7g-iii/7g-iv from scratch afterward is the exact
cost this cache exists to avoid paying twice. Paste this into a scratch cell (not
committed — a cell in the notebook *is* committed, so this stays out of the file) after
7g-iv or 7g-v finishes:

```python
import pickle
from src.physics.bicycle_model import A_MAX
with open('/content/drive/MyDrive/av_stress_checkpoint.pkl', 'wb') as f:
    pickle.dump({'rows': rows, 'decompositions': decompositions,
                'mechanism': mechanism, 'a_max': A_MAX}, f)
```

**Delete any checkpoint written before `ccab66d`** (the `A_MAX` 5.0 → 12.0 change): it was
computed under the old cap and carries no `a_max` key. The load snippet below refuses it,
and refuses any checkpoint whose recorded cap differs from the current one, before binding
anything:

```python
import pickle
from src.physics.bicycle_model import A_MAX
with open('/content/drive/MyDrive/av_stress_checkpoint.pkl', 'rb') as f:
    checkpoint = pickle.load(f)
assert checkpoint.get('a_max') == A_MAX, (
    f"checkpoint was computed under A_MAX={checkpoint.get('a_max')!r}, this code uses "
    f"{A_MAX}: re-run 7g onward instead of resuming"
)
rows, decompositions, mechanism = (checkpoint['rows'], checkpoint['decompositions'],
                                   checkpoint['mechanism'])
```

Written to Drive, not `/content` — `/content` is wiped on exactly the restart this
checkpoint exists for. To resume: re-run cells 1–17 (setup through the shard cache —
`shard_cache` itself isn't in the pickle, and rebuilding it is exactly the cost this
whole cache exists to pay once per session, unavoidable after a genuine restart), then
run the load snippet above in a scratch cell instead of
re-running 7g/7g-iii/7g-iv from scratch. No DB connection is needed for this path —
7g/7g-iii/7g-iv/7g-v never touch Postgres.

**Pass checkpoints — surviving a disconnect after Pass 1 or Pass 2.** Pass 1 (section 6,
`score_shard`) and Pass 2 (section 9) are the expensive computations; their database
writes are idempotent, so after a restart the only thing worth saving is their output. In
a scratch cell (not committed, like the snippet above), right after each pass:

```python
import pickle
with open('/content/drive/MyDrive/av_stress_pass1.pkl', 'wb') as f:
    pickle.dump({'meta': run_metadata(), 'records': records, 'errors': errors}, f)
```

```python
import pickle
from src.scoring.ranker import rank_scenarios
with open('/content/drive/MyDrive/av_stress_pass2.pkl', 'wb') as f:
    pickle.dump({'meta': run_metadata(),
                 'pass1_ranking': [(r['scenario_id'], r['fragility_score'])
                                   for r in rank_scenarios(records)],
                 'ids_to_test': ids_to_test, 'stress_results': stress_results}, f)
```

`run_metadata()` (cell 9) records the commit, `SHARD_PATH`, `MAX_SCENARIOS`, `TOP_N`,
`DE_KWARGS` and `A_MAX`. To resume after a restart: re-run cells 1–17, then load Pass 1
**instead of** running section 6, and run sections 7 onward as normal (section 8's upsert
is idempotent):

```python
import pickle
with open('/content/drive/MyDrive/av_stress_pass1.pkl', 'rb') as f:
    pass1 = pickle.load(f)
assert pass1.get('meta') == run_metadata(), (
    f"Pass 1 checkpoint was made under {pass1.get('meta')!r}, this session is "
    f"{run_metadata()!r}: run Pass 1 instead of resuming"
)
records, errors = pass1['records'], pass1['errors']
```

and, if Pass 2 also finished, load it **instead of** running section 9. It refuses unless
Pass 1 is already in place and ranks exactly as it did when Pass 2 ran, so `stress_results`
can never be paired with a different ranking:

```python
import pickle
from src.scoring.ranker import rank_scenarios, top_n_ids
assert 'records' in globals(), "load (or run) Pass 1 before loading Pass 2"
with open('/content/drive/MyDrive/av_stress_pass2.pkl', 'rb') as f:
    pass2 = pickle.load(f)
assert pass2.get('meta') == run_metadata(), (
    f"Pass 2 checkpoint was made under {pass2.get('meta')!r}, this session is "
    f"{run_metadata()!r}: run Pass 2 instead of resuming"
)
assert pass2.get('pass1_ranking') == [(r['scenario_id'], r['fragility_score'])
                                      for r in rank_scenarios(records)], (
    "Pass 2 checkpoint was made against a different Pass 1 ranking: run Pass 2 instead"
)
assert pass2['ids_to_test'] == top_n_ids(records, TOP_N)
ids_to_test, stress_results = pass2['ids_to_test'], pass2['stress_results']
```

The write passes (sections 8, 11, 12) each call `refresh_connection()` first, so a
connection that went idle during the diagnostics is replaced, with the same TLS check
section 4 applies, before anything is written.

**Diagnostic-only path.** For `A_MAX`-related work that doesn't need Pass 1/2/3, the
minimal code cells are **2, 4, 6, 7, 9, 10, 15, 17, then 35, 39, 41, 43** — not
36–37 (7g-ii, a different calibration question, not on this path). This skips the
Postgres install (cells 11–13) entirely: by content, the cache cell (17) needs only
`np`/`ShardLoader`/`ScenarioParser`/`SHARD_PATH`, from the smoke test (15) and the
config cell (6); 7g (35) needs only `shard_cache`; 7g-iii (39) needs `rows` and
`TYPE_NAMES` from 7g; 7g-iv (41) needs `decompositions`; 7g-v (43) needs `mechanism`.
Nothing on this path touches the database. It also skips Pass 1, 7b–7f, Pass 2, 10b,
and cells 53–65 — everything those four need (`rows`, `decompositions`, `mechanism`)
comes from each other, not from `records`/`ranked`/`stress_results`.

---

## Step 3 — Pass 1 across the shard

Cells 18–20. `score_shard` over `MAX_SCENARIOS`.

Every danger number in the database is stale: Batch 4 rewrote both TTC (quadratic root,
audit B07) and PET (visit pairing, audit B06), and Batch 5's B09/B20 changed Phase 4
outputs on top of that. Pass 1's ranking also selects which scenarios Pass 2 stress-tests,
so **re-running it can change which scenarios would ever have been candidates** — this is
not a refresh, it is the first ranking these engines have ever produced.

**Success:** `score_shard` completes over `MAX_SCENARIOS` with no exceptions; cell 21–22
diagnostics print; `records` is populated.

**Capture:** the timing probe from cell 19, and cell 22's full diagnostic block.

**STOP CONDITION.** Any unhandled exception. The engines are new; a crash here is a real
defect, not a data quirk.

Then cells 23–24 to build `diag_cache` (`DIAG_N = 50`), which 7c/7d/7e/7f all consume —
cell 24 is now a slice of cell 17's shard cache, not a re-parse (see "Shard cache" below).

---

## Step 4 — The dormant cells, in dependency order

None of these have ever executed. All depend on `diag_cache` from step 3.

| Cell | Section | First written | Measures |
|---|---|---|---|
| 25–27 | 7c | Phase 3 rework | rank correlation, all-pairs vs SDC-restricted |
| 28–29 | **7d** | **Batch 4** | PET sign semantics + B06 visit separation |
| 30–31 | **7e** | **Batch 4** | TTC discrimination rate after B07 |
| 32–33 | **7f** | **Batch 4** | TTC before/after on identical inputs |

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

### 7g — baseline replay drift sweep (cells 34–35)

Full shard, both soft gates off (`max_baseline_drift=None, max_speed_step=None`): this cell
measures, it does not filter. One additional sequential Drive pass; ~4 s of compute for
~1000 scenarios.

**Success:** completes; `scenarios measured` is close to the shard's scenario count; the
skip counters are small and explained. `never_valid` **must be 0** — it is structurally
unreachable, because `pick_nearest_challenger` only returns an agent sharing a valid frame
with the SDC, so `first_valid_index` cannot raise on it. If it is nonzero, something
upstream changed and the sweep's assumptions need re-checking.

**Also watch:** the assertion that every refusal has `reason == 'collision'`. With both
soft gates off, a `'drift'` or `'speed_step'` refusal would mean a kwarg does not do what
`perturbation_space.py` documents — and would invalidate the whole sweep.

**Decided (drift-gate batch).** The `19fddb4` run's 7g showed a continuous distribution with
no valley, so any fixed threshold is a judgement. `perturbation_space.py` now checks, after
the hard gate: a **speed-step refusal** (a vehicle challenger whose logged speed holds more
than 1.0 m/s it cannot replay under `A_MAX`, for 1 s), then a **2 m drift backstop** (about
one vehicle width). Drift is recorded for every scenario; the reasoning for both values is
in the constants' comments. The cell now prints where both gates fall.

**Pass criteria for 7g** (predicted by running this cell on the same shard locally, with the
batch's code; the shard was parsed by the not-yet-reviewed descriptor-pool loader, checked
against the `19fddb4` run's recorded 7g output, which it reproduced exactly):

- the drift distribution unchanged from the `19fddb4` run: 496 measured, hard gate 7/496,
  all-measured p50 0.431, p90 1.856, p99 14.044, max 25.582 m, 56.2% at or under 0.5 m
- `speed-step refusals: 12 of 453 vehicle challengers`, `of which also drift past the
  backstop: 12`, led by `ef85eea7` (held 7.181 m/s), `bf87cb57` (4.539) and `476f5ac1`
  (3.594)
- `drift refusals that are not speed steps: 27 (of which gapped: 6)`

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

### 7g-ii — heading floor / transition / singularity calibration (cells 36–37)

`V_HEADING_MIN` (`linear_model.py`) and `HEADING_TRANSITION_WIDTH` /
`HEADING_BLEND_SINGULARITY_MARGIN` (`perturbation_space.py`) are each marked in their own
source comments as calibrated, not validated against real data, and none of the three
appear anywhere above this line in this runbook. **No second Drive pass** — 7g's own loop
above now already collects the fields this cell reads, on every scenario it visits, full
shard.

**Corrected (independent review, 2026-09-27): only ONE of the three fields is
genuinely vehicle-inapplicable.** This section previously claimed
`frames_below_heading_floor` and `frames_in_heading_transition_band` were "exactly 0
for every vehicle row, by construction." Real data on the 496-scenario shard showed
vehicle rows with p75=69 and p90=91 frames below the floor, and that claim was false.
Both counters are computed from **logged speed alone**
(`perturbation_space.py`'s own comment: "RECORDED ON EVERY SCENARIO, INCLUDING FOR
VEHICLES"), with no dependence on `is_vehicle` — a slow or parked car counts exactly
like a slow pedestrian does. Reproduced live against a real, stationary `VEHICLE`
fixture: `frames_below_heading_floor` came back `91`, not `0`. Only
`baseline_heading_blend_min_magnitude` is genuinely vehicle-inapplicable — it comes
from `_last_heading_blend_min_magnitude`, touched exclusively by `_linear_heading`,
which vehicles never call — and that one claim stays true.

**What this cell reports, split by challenger agent type throughout, not pooled**
(still the right design — real speed distributions differ by agent type in WOMD, not
because the floor/band counters structurally don't apply to vehicles; pooling would
still average away that difference, for a different reason than given last round.
Same pattern 7g's own "drift by challenger agent type" section already uses):
- `frames_below_heading_floor` / `frames_in_heading_transition_band` percentiles, per
  agent type, over every scenario that both survived construction *and* did not trip
  the hard replay-fidelity gate — full shard, not the `TOP_N`-scenario Pass-2 sample.
  (`stress_results`'s own `search_provenance` carries these two per Pass-2 scenario
  already; reading that instead would be strictly worse, N=`TOP_N` against N≈shard size,
  so it is not read separately here.)
- The singularity guard's fire rate against the attempted population (constructions
  actually reached — excludes `no_challenger`, where nothing was ever attempted, and
  counted over `rows`, not `measured_rows`: a hard-gate refusal still means
  construction was attempted. In the first real run (496 scenarios), 0 of 39 non-vehicle
  challengers attempted fired the guard — 36 of those were measured directly, the other
  3 were hard-gate-excluded, and counting from `measured_rows` alone would have missed
  them), with a sample-size sentence next to it (independent review, 2026-09-27 — same
  discipline 10b's before/after already uses): the guard tests a deliberately rare
  event, so a zero-fire result is "no evidence it fires", never "evidence it does not",
  regardless of how large the non-vehicle population attempted turns out to be.
- `baseline_heading_blend_min_magnitude` percentiles per agent type, for the same
  surviving-and-not-hard-gated population — how close real scenarios sit to the 60°
  margin, not just whether they cross it. **Not a bare percentile line** (independent
  review, 2026-09-27): `percentiles()` (7g's own helper, untouched) calls
  `np.percentile` on the raw array, which is `nan` at every quantile with a
  `RuntimeWarning` on an all-inf VEHICLE group — reproduced live. A local helper here
  prints "all inf (guard cannot apply to the bicycle model)" for those, and for every
  non-vehicle group also prints the **minimum at full precision** and a count of rows
  below `0.999` — the real run printed `1.000` at every percentile for every
  pedestrian/cyclist row, which cannot distinguish an exact `1.0` from `0.9998`.

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

### 7g-iii — drift-outlier decomposition (cells 38–39)

Diagnosis only — does not change `max_baseline_drift` or the integrator. 7g's sweep
found a continuous drift distribution (p50=0.77 m, p90=4.45 m, p99=17.5 m, max=32.7 m,
only 40.1% at/under the 0.5 m default) and a worst-10 that is all vehicles, no interior
gap, no hard-gate collision, at 14-33 m. This cell asks *why*, over a targeted revisit
of the worst `K=10` by drift plus `K=10` nearest the median (0.77 m) — 20 scenarios, one
more targeted `ShardLoader` pass, not a second full read.

**The shape follows one fact:** `extract_state_from_womd` (`bicycle_model.py`) builds
the bicycle model's speed as `v = sqrt(vx**2 + vy**2)` — a scalar, from velocity alone,
that never sees position — and `bicycle_step` clamps it to `[0, V_MAX]`, integrating
position only along the heading direction. Every measurement below tests a different
way that can go wrong; see the notebook's own 7g-iii markdown cell for the full
breakdown, reproduced live against constructed fixtures before this cell was written.

**Candidates C1-C5:** independently-measured channels disagreeing (C1); recovered
controls hitting `DELTA_MAX`/`A_MAX` (C2); perception noise no smooth control sequence
reproduces (C3); a short or late-starting track (C4); and **reverse motion (C5)** — a
car backing up has a positive scalar speed but a logged velocity *vector* opposite its
heading, and the bicycle model can't represent reversing at all, so the replay drives
it forward at roughly `2v` error growth. C5 fits the worst-10's profile better than
C1-C4 and needed its own measurement, since it's invisible to a plain speed-magnitude
check by construction.

**Measurements M1-M6:** M1 (per-frame error profile — jump frame reported as the real
global frame `ts[k+1]`, not a bare array index; a concentrated jump is *not* C4-only —
a single bad frame under C1 or C3 looks identical, and M6's track shape is what actually
separates C4) and its cheap M1b addition (longitudinal/lateral decomposition of the
offset vector at the worst frame — one dot product); M2 (speed *magnitude* consistency —
sign-blind, so C5 passes it cleanly, why M4 exists); M3 (heading consistency, gated on
*implied* speed, `V_HEADING_MIN` reused from `linear_model.py` rather than a new
constant); M4 (direction consistency — logged velocity vector vs. logged heading, gated
on *logged* speed — gating on implied speed would exclude exactly the disagreeing frames
this measurement exists to catch); M5 (**unexplained fraction from a predicted OFFSET
VECTOR — corrected twice, independent review 2026-09-29** — round one compared a
scalar cumsum of the signed residual against `m1b_longitudinal` at the max-offset
frame, fixing an earlier bug that compared the *final* cumsum against the *max* offset
instead, wrong whenever drift is non-monotone; that scalar comparison was *itself*
still wrong on a turning track — it sums each interval's residual as if every interval
pointed the same way, which a closed-form function of the heading change contradicts,
so a caveat was the wrong response. **The fix accumulates the residual as a vector, one
per interval along that interval's own heading, and compares the resulting predicted
offset vector against the actual offset vector, both at the identical frame** — `norm
(offset_vec - pred_vec) / norm(offset_vec)`. Verified on five fixtures built from
`bicycle_step` itself (so the fixture carries no discretization mismatch against the
replay's own integrator): straight-with-bias and a mid-track peak both landed at 0%
unexplained; a 100° turn with the same bias and a 100° turn while reversing both
landed at 1-3% (fixture-dependent); a 100° turn with a genuine 8° heading error
landed at 99.9%. The scalar
comparison is kept as context only — it is exact solely on a straight track. **These
numbers are a floor, not a benchmark** (independent review, 2026-09-30): the fixtures'
"true" reference path came from the replay's own integrator, with no noise and no
independent measurement error for the speed channel to compete against. Real logged
positions come from perception, not from replaying `bicycle_step` — a genuinely
well-explained real scenario can still carry some nonzero unexplained fraction from
ordinary sensor noise, and that real-data floor is unknown; nothing here measures it.
A low fraction is consistent with the speed channel explaining the drift, not a score
to match against 0% or 1-3%); M6 (track shape — `t0`, `frames_observed/T` — the only
measurement that actually separates C4).

**What stays unresolved:** C1 and C3 are not separable by anything here — both show as
"channels disagree, nothing clipped, not reversing, not short" — reported as one
combined finding, never a forced pick.

**Two state details:** hard-asserts `rows` has 7g's own shape first — not just that a
variable named `rows` exists (corrected, independent review 2026-09-29: 7f used to define
its *own* `rows`, a list of `{'scenario_id', 'old', 'new'}` dicts, so an existence-only
check passed after a restart-and-partial-rerun that hit 7f but skipped 7g, and this cell
then `KeyError`ed on `r['collides']` confusingly deep in the loop instead of failing here
with a clear message. 7f's variable is now `ttc_rows`, so `rows` is 7g's alone; the shape
check stays as a second line of defence for a restart that skips 7g. The content check
itself needs `'rows' in globals()` as its own short-circuiting first clause —
`rows and {...} <= rows[0].keys()` alone raises a bare
`NameError` when `rows` doesn't exist at all, which is the exact case the guard exists
for); the decomposable population explicitly excludes hard-gate rows (`r['collides']`)
for *both* groups, since `PerturbationSpace.__init__` raises there and discards `self`
— the worst-10 happen to satisfy this already, the median-nearest group is not
guaranteed to on a different shard.

**Descriptive, not inferential**, stated as such in the cell's own output: a hand-picked
10-vs-10 contrast shows whether a pattern is visible, and claims no statistical
significance.

**Success:** completes for all 20 scenarios; every scenario prints all six
measurements; no hypothesis is declared confirmed.

### 7g-iv — clipped-acceleration mechanism check (cells 40–41)

Diagnosis only, same 20 scenarios as 7g-iii, no fix proposed. 7g-iii ruled out reverse
motion for the worst-10 and found the speed channel alone does not explain the drift
(M5 unexplained 88-103%), while the offset is longitudinal and steady. The one signal
that separates the worst-10 from the middle-10 is `clip_a` — 3.3-30.0% (median 15.0%)
vs. 0-6.7% (median 2.2%). This cell tests whether `invert_bicycle` clipping acceleration
to `A_MAX` is the mechanism, by re-integrating the clipped acceleration and checking
whether the resulting speed gap accounts for the position offset.

**Six changes from independent review, 2026-09-30, folded in before this cell was
written, each verified against constructed fixtures run through the real
`PerturbationSpace`:**

1. **Closure, made the primary number.** Comparing speeds alone never checks that the
   gap accounts for the 14-33 m offset. Reuses 7g-iii's M5 vector-accumulation with
   `replay_speed` as the driver term in place of the logged speed, compared against the
   actual offset vector at the worst frame. A hard-stop fixture (real -8 m/s² decel,
   beyond the cap in force when this was written, `A_MAX = 5.0`; the current 12.0 would not
   clip it) landed at 4.4% unexplained; a noise-only fixture (no systematic
   clipping direction) landed at 0.1% — both in the low range 7g-iii's own fixtures
   established, confirming the closure metric reads a clipping-explained case as
   low when it should.
2. **Proximity test replaced, not kept as a hard check.** "First departure within one
   transition of a clipped one" is close to chance at real clip rates (roughly half of
   all 3-transition windows contain one at a 20% clip rate). Replaced with comparing the
   first frame `|v_sim - logged| > 0.5` against the first frame `|replay - logged| >
   0.5` directly; the base rate is still printed, labeled as context, not evidence.
3. **Sign and net effect reported per scenario.** Two different quantities, both
   printed: the sign split of individually clipped transitions (`a_raw < -A_MAX` for a
   hard stop — negative), and the net `Σ(clip(a_raw) - a_raw)·dt` (positive for that
   same hard stop, since clipping applies *less* braking than the raw signal called
   for — the direction that produces a replay-ahead offset). The net is checked against
   the replay's own ahead/behind direction — a disagreement is evidence against clipping
   being that scenario's cause, not for it.
4. **Clip fraction over valid transitions, not all `T-1`.** 7g-iii's own `clip_a`
   divides by every row, including zero-padded rows before `t0` or across a gap,
   understating the rate for a late-starting track. Both are now printed, labeled.
5. **Division guarded by a tolerance, not just `> 0`.** A no-clip fixture with a
   near-zero offset printed 18,481,207% unexplained from float dust before this fix;
   `OFFSET_TOL = 1e-3` m now prints `n/a` below that floor instead.
6. **Direct cross-check against `space.base_controls`.** Reconstructs `clip(a_raw)` per
   GLOBAL adjacent frame — the inverter's own unit — and asserts it matches
   `base_controls` on every gap-free transition.

**Two more fixes, independent review, 2026-09-30 round 2**, both against a constructed
gapped fixture (mild brake, zero real clipping, frames 10-14 invalid) that exposed each:

7. **The sign check gave a false DISAGREES when nothing was clipped.** With
   `net_clip_speed` exactly 0 and a noise-level `mean_speed_diff`, the check read one as
   "BEHIND/neutral" and the other's sign at random, printing a spurious disagreement —
   on real data the middle group would collect meaningless DISAGREES lines exactly like
   this. Fixed by gating the comparison on materiality: it only runs when both
   `|net_clip_speed|` and `|mean_speed_diff|` exceed `DEPARTURE_THRESHOLD` (0.5 m/s);
   otherwise it prints `n/a (clip effect or speed gap below 0.5 m/s -- not material)`.
8. **`v_sim` was built on `ts`'s gap-bridged `dt_actual`, not global adjacent frames.**
   Change 6 above applied the global-frame reconstruction only to the `base_controls`
   cross-check, not to `v_sim` itself. On the gapped fixture this missed the actual
   mechanism: `TrajectoryInverter.invert` writes a ZERO control across the gap (the
   inverter never attempts to recover one there), so the real replay HOLDS speed while
   the logged car keeps braking — a `dt_actual`-bridged reconstruction cannot represent
   that, and read a 0.80 m/s build-check error where there should have been none. Fixed
   by rebuilding `v_sim` frame-by-frame from `t0` over the same global-adjacent-frame
   `clip(a_raw)` change 6 already reconstructs (zeroed wherever either endpoint is
   invalid) — the build check is now exact to float precision on every scenario, gap or
   not, and the departure-frame comparison and sign check both become valid on a gapped
   scenario too. The clip counts and `net_clip_speed` moved to this same global-frame
   basis for the same reason and are unaffected on every gap-free fixture (identical to
   round 1's numbers there, since a gap-free `ts` and the global frame range coincide).

**Gap caveat, current state:** `TrajectoryInverter.invert` writes zero across any
transition touching an invalid frame — it never bridges a gap with a larger `dt`. After
round 2, `v_sim`, the clip counts, `net_clip_speed`, and the sign check are all built on
that same global-adjacent-frame unit, so they are exact regardless of a gap. Only the
**closure** still bridges a gap, because `implied_longitudinal` is position-derived
(`dt_actual = diff(ts) * DT`, the same construction 7g-iii's M2-M5 use) rather than
accel-derived — the gapped fixture's closure still printed 11.3% with zero real
clipping, which is that approximation's own floor, not a residual. The CLOSURE print
line carries a `[GAP ...]` marker whenever `has_interior_gap` is true so this isn't
misread. Separately, the same gapped fixture shows a real 0.52 m/s replay-ahead speed
departure with `net_clip_speed` at zero (no clipped transitions at all) — that is
`TrajectoryInverter.invert`'s own zero-control-across-the-gap behavior, not clipping,
and the reading guide now says so explicitly: on real data, a gapped scenario with a
speed departure and near-zero `net_clip_speed` points to gap-zeroing, not clipping.

**Also verified:** a non-vehicle (pedestrian) target and a single-valid-frame track
both hit the early-exit branch cleanly with no missing-field error; the
`max_offset_frame` recomputed here is asserted equal to 7g-iii's own stored value on
every scenario, as a cheap proof the two cells are looking at the same computation.

**Descriptive, not inferential** — same 20 scenarios, no statistical claim. A low
closure number closes the chain for *this* mechanism on a given scenario; it does not
rule out a compounding cause on a scenario where the numbers don't cleanly close.

**Success:** completes for all 20 (vehicle-only fields are absent for a non-vehicle
target or a track under 2 valid frames), the `base_controls` cross-check assertion
holds on every gap-free transition it checks, and no fix to the inverter is proposed.

### 7g-v — shard-wide `A_MAX` calibration (cells 42–43)

Diagnosis only, no `A_MAX` or inverter change proposed. 7g-iv's real output closed 8
of the worst 10 to 0.2-7% unexplained. This cell asks what the clipped transitions look
like and what `A_MAX` would remove the tail — still no `PerturbationSpace` anywhere,
one full-shard pass shared between items 1 and 3, a targeted 20-scenario pass for item 2.

**One real fix, independent review, 2026-09-30 round 3, found before this cell was
finished:** the counterfactual sweep's driver term must be the **mid-interval** speed
`0.5*(v_sim[k]+v_sim[k+1])`, matching `bicycle_step`'s own `v_mid`, not the
**start-of-interval** `v_sim[k]` 7g-iv's own closure uses. The start-of-interval form
leaves a discretization floor of `-0.5*dt*(v[k+1]-v[k])` even with zero clipping —
confirmed on a constructed clean (never-clipping) track, which the buggy form still
read as a nonzero predicted offset. Negligible against 7g-iv's 14-33 m real offsets (a
few percent at most — those numbers stand), but dominant against the sub-meter offsets
this cell predicts for clean tracks, where it would fail every clean scenario for a
reason unrelated to `A_MAX`. Verified against four constructed tracks (a clean brake,
two hard stops at different severities, and a noisy constant speed) at
`A ∈ {5,8,10,12,15,∞}`: the start-of-interval form leaves a nonzero floor at `A=∞` on
every one of them; the mid-interval form goes to (numerically) zero there, as it should
with no clipping in effect.

**Two roles for the driver term, kept deliberately separate:** the shard-wide sweep
(items 1/3's headline numbers) uses the corrected MID-interval driver throughout. The
**worst-10 identity check** deliberately reproduces 7g-iv's own START-of-interval
formula instead, at the real `A_MAX=5` — this check validates that this cell's
independent, `PerturbationSpace`-free reconstruction reproduces an already-computed
number byte-for-byte (asserted to `1e-6`, rebuilding `offset_vec` from
`decompositions`' `m1b_longitudinal`/`m1b_lateral` and the heading at
`max_offset_frame`), not the mid-interval fix. Verified on constructed fixtures run
through the real `PerturbationSpace`/7g-iii/7g-iv machinery: the identity check landed
at 0.00 and 1.99e-08 difference against the two clipped scenarios tested. Separately,
the mid-interval **max-norm proxy** compared against each of 20 scenarios' real
`max_offset` (not asserted, reported for both groups) matched to three decimal places
on every non-gapped fixture tested — expected, since mid-interval is now the same
discretization the real replay's own position update uses.

**Item 3's primary metric is the count/fraction above 5 m and 2 m, plus p90/p99, with
an `A=∞` reference and a clip-attributable part `max‖pred(A)-pred(∞)‖`** — not the
`<=0.5` m fraction kept as context, since only ~7% of the shard drifts over 5 m and
that fraction is set by the noise floor, not `A_MAX`. **The refutation criterion is
worded accordingly:** the mechanism is undercut if the count above 5 m does not fall
materially from `A=5` to `A=15` — a flat `<=0.5` m fraction is not that signal.
Scenarios with an interior gap are marked, counted, and excluded from item 3's headline
fractions (their `implied_longitudinal` still bridges the gap against a gap-free
global-frame `v_sim`, the same asymmetry as 7g-iv's own gap caveat) — verified against
a constructed gapped fixture, correctly flagged and excluded without a crash.

**Item 1 additions:** per-agent fraction with `>=1` and with `>=3` transitions above
each threshold, alongside the per-transition fraction (a per-transition rate
understates the risk of one bad frame carrying through an entire open-loop replay); a
shard-wide glitch-vs-braking split at 12 m/s² (fraction of such transitions within 2
frames of an opposite-sign one above the same threshold) — item 2 can only answer this
for 20 scenarios, this answers it shard-wide. Verified on a constructed multi-agent
scenario with one deliberate glitch pair and one deliberate sustained-brake run: the
split correctly separated them.

**Item 2** prints every clipped transition's frame and value, grouped into maximal
same-sign runs vs. flagged spike-pairs, plus the single largest transition —
verified on a constructed noisy hard-stop fixture, producing a mix of runs and
spike-pairs that matches the fixture's own construction by inspection.

**Item 4** states plainly this measures only the speed-driven part, and names
`8fd0…`/`ef85…` (which didn't close in 7g-iv) without attributing a cause.

**Descriptive, not inferential** — items 1 and 3 are shard-wide, not a sample, but
remain a counterfactual built from logged data, not a claim about what would actually
happen if `A_MAX` changed in `src/`.

**Success:** item 1 reports all percentiles/threshold fractions over a nonzero sample;
item 2 prints anatomy for every clipped transition with no crash on a zero-clip
scenario; item 3 reports the 5 m/2 m stats at all five `A_MAX` values plus `A=∞`, gapped
scenarios marked and excluded; the identity check passes or reports its actual
discrepancy; item 4 names the two unclosed scenarios without claiming their cause.

### 10b — B09 before/after (cells 51–52)

**Skipped by default: `RUN_B09 = False`.** Measured already: **B09 fired on all 6
evaluable scenarios, and the warm start won every time.** Where that comes from: the first
real-shard run (496 scenarios; its commit wasn't recorded, because it predates the
freshness guard), run under `A_MAX = 5`. 6 of the 25 sampled scenarios were evaluable; the
0.5 m drift gate refused 18. That output is not in this repository, and the result has
not been re-derived here. It is not from the last full run (cache commit `32a2853`), whose
10b was interrupted before it finished. Re-measuring under `A_MAX = 12` would likely
evaluate more scenarios, since fewer would be refused for drift. The cap matters because
the result was measured under the old physics: the conclusion, that B09 fires on real
data, doesn't depend on it; the sample does. With the flag off the cell prints one line
saying it was skipped and where this result is, and runs nothing — not even its imports.
Set `RUN_B09 = True` only to re-measure. The rest of this section describes that run.

Runs after Pass 2 diagnostics because it needs `ranked`. `B09_N = 25`, ~8 minutes.

**Read the self-test line first.** The cell reconstructs the pre-B09 behaviour (no flag
recovers it), and validates that reconstruction against a fixture where an *iterate* wins.
If it prints `FAIL`, the cell raises and every number after it would have been meaningless.

**Then read `fidelity guard informative on: M/N`.** The per-scenario guard proves nothing
when the warm start wins, which is most of the time. If `M = 0`, the reconstruction rests
on the self-test plus code inspection — the cell says so itself.

**Then read the sample size, which prints directly above the conclusion.** A zero with
`n < 20` is reported as *"no evidence it fires"*, never as *"evidence it does not"*.

**Decision this fed, now made:** whether B09 needs a Block 4 doctrine note. It does.
Block 4 describes a refinement stage that returns its endpoint. B09 firing means the stage
returned a better verified candidate than its endpoint instead, and on every one of the 6
evaluable scenarios that candidate was the warm start. So Block 4's description is
materially wrong. The note is already drafted in the review session's doctrine corrections;
Avi decided compiling those into the PDFs isn't needed.

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
earlier write for anything here to diverge from. Section 8's idempotency check (cell 46)
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
45–50, then 10b, then sections 11 through 15 at cells 53–65 (update, Pass 3 export,
geometry verification, API round trip, summary).

**`TOP_N = 20`, for the deployed site, not for any measurement.** This paragraph used to
say `TOP_N` stays at 5: Phase 5's architecture is a cheap filter feeding an expensive pass
over a small selected set, and running Pass 2 at a size chosen to make a measurement look
better would misrepresent how the pipeline works. That still holds at 20. 20 of 100 is
still a small, selected set, so the cheap-filter-then-expensive-pass architecture is
unchanged. The reason for 20 is the dataset the deployed site serves, 20 scenarios with a
perturbation and playback instead of 5, not any measurement in this run. The S1–S8 drift
checks don't depend on `TOP_N` either: 7g–7g-v read the whole shard through the cache.
10b keeps its own `B09_N`, so its sample never depends on Pass 2's size. **The tie
caveat:** at least a quarter and fewer than half of the 100 scenarios tie at the maximum
fragility score, 100 (the re-run's Pass 1 fragility percentiles, as the executed notebook
printed them: p50 57.1429, p75 100.0000; 28 of the 100 stored scores are at the maximum,
counted from the stored scores). 100 is Pass 1's ceiling, TTC and PET both at their
0.01 s floor. `rank_scenarios` breaks ties by scenario ID, and at least 25 scenarios tie at
the maximum, so all 20 selected fall inside the tie: the 20 are the first 20 by scenario ID
among the most fragile, not the 20 most fragile. The site's description of its data should
eventually say the same.

**Geometry verification (section 13, cell 59)** carries Batch 5's B19 fix: it now asserts *coverage
before correctness*. The old version passed vacuously at `0 == 0` when every agent was
missing from the database — the cell whose job is catching missing geometry was blind to
geometry being missing in full. It now tracks expected-exportable agents and fails if any
is absent.

**Note on the audit's B19 fixture:** it used to locate this cell by index, and Batch 4's
four inserted cells silently broke it — the test began exec'ing a markdown cell and dying
with `SyntaxError` instead of reaching its assertion, staying red either way so the count
never moved. Batch 5 switched it to content lookup. A later session's two new cells (7g,
10b) shifted indices again with the fixture unaffected, a subsequent pass's two new cells
(7g-ii) did the same, a later pass's two new cells (the freshness guard, section 3b) did
it a third time, a subsequent pass's two new cells (7g-iii) did it a fourth, a later
pass's two new cells (7g-iv) did it a fifth, a subsequent pass's two new cells (7g-v)
did it a sixth, and this pass's own two new cells (the shard cache, section 5b) do it
a seventh — which is the fix doing its job, again.

**Success:** section 13's code cell prints `CONFIRMED: all N exportable agents present ...`; the API round
trip closes the loop between HTTP timesteps and the M values read directly from PostGIS.

**If no scenario in the Pass 2 sample produced a verified collision** — an ordinary
outcome, not a failure — section 14 no longer crashes on it (audit R09). It falls
back to any scenario with exported geometry, prints every outcome it did see, and states
explicitly that collision-specific checks were **skipped, not passed**. The round-trip
assertion still runs, on a non-colliding agent; it never needed a delta. Read that notice
if it appears: it means the `/perturbed` delta and `collision_timestep` printed as null
legitimately, and it is also the signal that `B09_N`'s sample in section 10b may contain
few or no usable comparisons.

**Pass criteria (checked against the re-run at `62ab1a6`).** The search is deterministic for a
fixed commit; across commits it is not bit-identical (second bullet). Each criterion says where
it was checked: the executed notebook's printed output, or the stored results.

- **Pass 2: 18 collisions, 2 refusals, 0 errors**, and **0 speed-step refusals.** Checked: the
  executed notebook's `PASS 2 AGGREGATES` printed `collision found: 18`, `replay_infeasible: 2`,
  `error: 0` and `replay_infeasible by reason: drift=2`, with no speed-step reason; the stored
  scores hold 18 `collision_found` and 2 `replay_infeasible`.
- **The 12 collisions of the `19fddb4` run come back at the same target agent and the same
  collision frame, with norms that differ slightly.** The 12 are derived, not recorded: the 18
  collisions in the hosted database before this re-run, minus the six listed below. Checked
  against those earlier stored results: all 12 are present, all with the same target and the
  same collision frame; their norms differ from the earlier ones by at most 2.3e-4 relative
  (median 5.5e-5), and over all 17 scenarios that have a collision in both data sets, by at
  most 6.6e-4. For example `1492befc`: 0.02243394 before, 0.02243386 now, t=55 in both. The
  earlier results carry no `collision_geometry_version` and these carry
  `oriented-box-float64-v1`; what else differs between the two runs' code was not isolated
  here, so the cause of the small differences is not claimed.
- **The set of 18 differs by one scenario.** The earlier 18 contained `c302c905`; these contain
  `c549c69c` in its place. `c302c905`'s `min_pet` changed from -1.3 to +1.3 between the two
  data sets, which lowers its fragility score from 100 to 57.47 and takes it out of the top 20
  (rank 32 in this re-run's Pass 1 ranking); `c549c69c` is rank 20.
- **Six scenarios with a stored result have a baseline drift above the old 0.5 m gate**, which
  the 2 m backstop admits (stored `baseline_replay_error`, `min_perturbation`,
  `collision_timestep`, `baseline_offset_at_collision`):

  | scenario | drift (m) | norm | t | offset at collision (m) |
  |---|---|---|---|---|
  | `8ec2910b` | 1.332 | 0.4034 | 39 | 0.248 |
  | `c549c69c` | 0.922 | 0.0089 | 90 | 0.922 |
  | `a6bf1ade` | 0.763 | 0.0101 | 82 | 0.681 |
  | `38c703d6` | 0.676 | 0.0153 | 90 | 0.672 |
  | `b1e5a345` | 0.576 | 0.0204 | 54 | 0.326 |
  | `19043d68` | 0.525 | 0.0197 | 90 | 0.525 |

  The earlier data's six were the same except that `c302c905` (drift 0.631 m, norm 0.1109,
  t=59, offset 0.577 m) stood where `c549c69c` stands now.
- **2 drift refusals**: `58d5f1b9` at 2.562 m and `504dd390` at 3.464 m (the printed
  `baseline drift=` values; `replay_infeasible by reason: drift=2`). The same two scenarios
  were refused in the earlier data.
- **Pass 3: `replay_refused: 0`**, and the other four refusal buckets 0 on a fresh database.
  Checked: the Pass 3 summary has `replay_refused`, `perturbed_stale`, `scene_changed`,
  `sdc_changed`, `a_max_changed` and `errors` all empty, and `exported: 20`,
  `agents_written: 1658`, `agents_skipped: 21`, `perturbed_written: 18`.

Several of these collisions have norms of 0.01 to 0.02 beside offsets of 0.3 to 0.9 m at the
collision frame. That is why `baseline_offset_at_collision` is recorded: it is the number to
show beside the norm, and the scenario detail page now does (the replay offset and
whole-track drift are served and shown since `817d1b1` and `e074bfd`).

---

## What to bring back

1. `git log --oneline -1` — the exact commit this ran against. Cell 9's freshness guard
   now prints this automatically, so it lands in the saved output even if nobody thinks
   to run it by hand.
2. Raw output for every cell in steps 2–6. Not summaries.
3. For each of the five decisions: the number, and which reading it supports.
   - drift gate — **already decided** (drift-gate batch): bring back 7g's printed gate
     counts and Pass 2's per-scenario outcomes, checked against the pass criteria in
     steps 5 (7g) and 6
   - `V_HEADING_MIN` / `HEADING_TRANSITION_WIDTH` / `HEADING_BLEND_SINGULARITY_MARGIN`
     calibration — from 7g-ii
   - B09 doctrine note — **already decided, nothing to bring back**: a Block 4 note is
     needed (step 5, 10b, says why and where the result comes from), and it is already
     drafted in the review session's doctrine corrections. 10b is skipped by default.
   - TTC saturation fallback, worth designing or not — from 7e
   - whether the audit is genuinely closed — from step 2
4. Anything that failed, with its full traceback.

## What this session does not do

**No code changes.** If something here demands one, it is a new batch.

**Doctrine corrections already made.** Block 6 Concept 24's `stress_tested_at`/`robustly_safe`
passage was corrected in Block 6 v2, citing audit B04. Block 2 §6/§7's common-mode-cancellation
claim was corrected in Block 2 v3, citing audit B03 — not Block 4 §17, which is Perturbation
Space Design and never carried this claim. Neither correction needed a shard or a Linux box,
and neither is open now. One related item: B09 fires on real data (step 5, 10b), so Block
4's description of the refinement stage as returning its endpoint is materially wrong. Its
note is already drafted in the review session's doctrine corrections, so nothing is left
to do for it here.
