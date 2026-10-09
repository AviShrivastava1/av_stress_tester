# av_stress_tester

How small a change to a recorded driving scene is enough to cause a collision? This
project takes real scenarios from the Waymo Open Motion Dataset, replays one nearby agent
under a kinematic model, and searches for the smallest change to that agent's motion that
makes its oriented bounding box intersect the self-driving car's. The results are stored
in PostgreSQL/PostGIS, served by a read-only API, and replayed frame by frame in a web app.

Deployed site: https://av-stress-tester.vercel.app/ (it shows stored results only).

Deploying the API (Render) and the frontend (Vercel): see [DEPLOY.md](DEPLOY.md).
How a scene becomes a stored result is also described on the site's Method page.

## What the stored run contains

Counts are from the results database. The hosted API's `/stats` reports the same 100
scored, 18 stress-tested (all collisions), 2 replay-infeasible and 20 with geometry, as of
2026-10-09. One shard went through the pipeline:
`uncompressed_scenario_training_training.tfrecord-00000-of-01000`.

| Stage | Count |
|---|---|
| Scenarios scored (TTC, PET, fragility) | 100 |
| Scenarios with a search attempt | 20 |
| of which a collision was found | 18 (all by Differential Evolution; none used the optional gradient refinement) |
| of which the search was refused: the zero-perturbation replay drifted too far from the log | 2 |
| Scenarios never searched | 80 |
| Scenarios with exported geometry | 20 |
| Scenarios with a stored perturbed path | 18 |

The list in the web app shows all 100 scored scenarios. The 20 with geometry can be
replayed; the 18 with a perturbed path show the logged and perturbed motion side by side.

The 20 searched scenarios are not "the 20 most fragile". 28 of the 100 scenarios are tied
at the top fragility score, and the search ran on the first 20 of those in scenario-ID
order (`ranker.rank_scenarios` sorts by score, then scenario ID; the run took
`TOP_N = 20` from that order). 8 tied scenarios were not searched.

17 of the 18 collision targets are vehicles; one is a pedestrian.

## How it works

1. **Parse.** Scenarios are read from Waymo Open Motion Dataset TFRecords
   (`src/data/parser.py`, `ScenarioParser`). Parsing needs the Waymo package, which only
   installs on Linux, so the offline passes run in Colab (`notebooks/COLAB_RUNBOOK.md`).
2. **Rank.** For each scenario, the minimum time-to-collision (`src/danger/ttc_engine.py`,
   `compute_min_ttc_sdc`) and the minimum post-encroachment time
   (`src/danger/pet_engine.py`, `compute_min_pet_sdc`, computed per connected conflict
   zone) are taken over pairs that involve the self-driving car (SDC). Both are combined
   into a fragility score (`src/danger/danger_score.py`, `compute_danger_score`).
   The score is a ranking, not a collision probability.
3. **Replay.** One challenger agent per scenario, the one nearest the SDC
   (`challenger_selection: nearest_by_min_center_distance` in the stored provenance), is
   replayed from its logged motion: the bicycle model for vehicles, the linear model for
   pedestrians and cyclists (`src/physics/simulator.py`, `ForwardSimulator`;
   `src/optimization/perturbation_space.py`, `PerturbationSpace`). A scenario is refused
   instead of searched if the zero-perturbation replay already collides, follows a logged
   speed step the model cannot, drifts from the log by more than `max_baseline_drift`
   (2.0 m in the stored provenance), or sits next to the heading-blend singularity
   described under Known limitations.
4. **Search.** A four-dimensional, bounded perturbation (initial speed or velocity,
   initial heading for vehicles, and two constant control biases) is searched with
   Differential Evolution (`src/optimization/scipy_optimizer.py`, `optimize_scenario`),
   with optional gradient refinement (`src/optimization/autograd_optimizer.py`).
   The cost is a weighted norm of the perturbation.
5. **Verify.** A candidate counts as a collision only if the oriented bounding boxes of
   the two agents intersect, using float64 corners and an exact Shapely intersection
   (`src/danger/collision_detector.py`; geometry version `oriented-box-float64-v1`).
   Touching counts.
6. **Publish.** Results and geometry are written to PostgreSQL/PostGIS
   (`src/scoring/db.py`, `src/scoring/export_geometry.py`). The API reads them; it never
   starts a search.

### Fragility score

The stored fragility score does not include the perturbation. It is computed in the
scoring pass that runs before any search (Pass 1 in `notebooks/COLAB_RUNBOOK.md`; Pass 2 is
the search and Pass 3 the geometry export), which calls
`score_scenario(..., min_perturbation=None)` (`src/scoring/batch_scorer.py`), and
`compute_danger_score` then reweights TTC and PET to 0.4/0.7 and 0.3/0.7. Both terms are
clamped at 0.01 s, so a scenario whose minimum TTC and minimum PET are both at or below
0.01 s scores 100.00000000000001. 28 of the 100 scenarios do, including all 18 where a
collision was found; the list orders them by scenario ID only.

### Identity and sizes

- Each stored result has a `stress_run_id`: a content hash of the scenario, target, delta,
  method and search provenance (version 2 when provenance is present;
  `compute_stress_run_id` in `src/scoring/db.py`). The exported perturbed path carries the
  same id and the scene fingerprint of the scenario it was built from. The API serves
  geometry when the ids and fingerprints match; rows exported before these ids existed
  (NULL on both sides) are served as before (`IS NOT DISTINCT FROM` in `get_trajectories`
  and `get_perturbed`, `src/api/routes.py`).
- Each agent's box is stored at its size in every frame (`scenario_agents.lengths_m` and
  `widths_m`), which is the size the collision check used, and the browser draws that
  size. Data exported before this was stored is drawn at one size per agent, and the scene
  view says so.

### Backup and restore

`DEPLOY.md` describes how the hosted data is replaced: a custom-format `pg_dump` of the
three result tables, a `pg_restore` of at least that version, and a fingerprint query that
must give the same answer before and after, checked on a scratch database first.
`tests/test_geometry_per_frame_sizes.py` runs a dump and restore when a `pg_dump` of the
server's major version is available.

## Stack

- Offline passes: Python 3.11, NumPy, SciPy, Shapely, PyTorch (optional refinement), run in
  Colab against a Waymo shard.
- Database: PostgreSQL 17 with PostGIS.
- API: FastAPI, read-only, on Render (`requirements-api.txt` pins what it installs).
- Frontend: React 19, TypeScript, Vite, on Vercel.

## API

A read-only HTTP layer over the scored results. Nothing here starts a
computation: the optimizer runs offline and this reads what it wrote.

```bash
PGDATABASE=av_stress PGUSER=$(whoami) uvicorn src.api.main:app --port 8000
```

Interactive docs at `/docs`.

| Endpoint | Returns |
|---|---|
| `GET /health` | Liveness, including a real database round-trip and the PostGIS version |
| `GET /stats` | Corpus counts — total, stress-tested, collisions found, searches that found nothing, replays refused, and the fragility range |
| `GET /scenarios` | Ranked by fragility, most fragile first. Keyset-paginated via `limit`, `cursor`, `stress_tested_only` |
| `GET /scenarios/{id}` | One scenario, including its raw 4-D perturbation vector |
| `GET /scenarios/{id}/trajectories` | Every agent's logged path, with per-vertex timesteps and headings |
| `GET /scenarios/{id}/perturbed` | The challenger's logged path beside its minimally-perturbed one |

**The API needs no Waymo package and no `.tfrecord` shard.** It reads only
PostgreSQL/PostGIS, so it starts anywhere — trajectory geometry is exported ahead
of time by `src/scoring/export_geometry.py`, which is the only module that touches
a shard. That split is deliberate: the Waymo dependency is Colab-only, and an API
that could not start without it would not be deployable.

**Coordinates are local planar metres, not longitude/latitude.** WOMD scenarios use
a local metric frame. Treating these values as lon/lat puts every scenario off the
coast of West Africa and makes every distance meaningless.

A cursor must be passed back exactly as the API returned it; anything else is a 422.

## Frontend

React + TypeScript + Vite, in `frontend/`. Needs Node 22 (`frontend/.nvmrc`) and
the API above running on port 8000 (override with `VITE_API_BASE_URL`).

```bash
cd frontend && npm install && npm run dev
```

Its TypeScript types are generated from `frontend/openapi.json`, a committed
snapshot of the API's schema. After changing a response model, or the docstring of an
endpoint, which is published as its description, regenerate it and commit it alongside the
backend change — `tests/test_openapi_snapshot.py` fails until you do:

```bash
./venv/bin/python -m src.api.export_openapi
```

## Running it and the tests

The repository holds no result data, so a fresh clone shows an empty list until a
database has been filled; the offline passes that do it are in
`notebooks/COLAB_RUNBOOK.md`. The tests build their own fixtures.

Backend (Python 3.11, a disposable PostgreSQL with PostGIS):

```bash
python3.11 -m venv venv
# requirements.txt lists tensorflow and the Waymo package, which do not install on
# Python 3.11; drop those two lines for tests, as CI does
grep -v -x -F -e tensorflow -e waymo-open-dataset-tf-2-11-0 requirements.txt > /tmp/requirements-dev.txt
./venv/bin/python -m pip install -r /tmp/requirements-dev.txt -r requirements-api.txt pytest

createdb av_test && psql av_test -c "create extension postgis"
export PGDATABASE=av_test AV_CLAIMS_DB=1 AV_AUDIT_DB=1 AV_ROBUSTNESS_DB=1
./venv/bin/python -m pytest tests -q -rf \
  --deselect tests/test_audit_core.py::test_B05_corrupt_next_record_does_not_overwrite_previous_success \
  --deselect tests/test_audit_core.py::test_control_parser_preserves_valid_states_and_flags
```

On the author's machine this gives 869 passed, 1 skipped (the dump and restore test, which
needs a `pg_dump` of the server's version), 2 deselected (they need the Waymo package) and
2 xfailed. The same command runs in CI (`.github/workflows/ci.yml`).

Frontend:

```bash
cd frontend
npm ci
npm run typecheck
npx vitest run          # 473 tests in 35 files on the author's machine
VITE_API_BASE_URL=https://api.example.invalid npm run build
```

## Known limitations

- **One shard, 100 scenarios, 20 searched.** The searched scenarios are the first 20 of a
  28-way tie, in scenario-ID order. Nothing here says how common collisions are in the
  dataset.
- **Min perturbation is the smallest the search found, not a proven minimum.** The search
  budget is stored with each result (`search_provenance`: `de_popsize` 15, `de_maxiter`
  200 in this run).
- **One challenger per run**, chosen by distance to the SDC, not every possible
  interaction.
- **A collision is an oriented-box intersection at the stored coordinate precision**, not
  a forecast of real-world impact. "No collision found" would describe the search, not
  safety.
- **The fragility score ranks; it does not measure.** It excludes the perturbation, and the
  top of the ranking is a 28-way tie.
- **Pedestrian and cyclist replay is less well measured than vehicle replay.** The linear
  model's acceleration limit "has never been measured"
  (`perturbation_space.py`, comment above `SPEED_STEP_REFUSE_MPS`), and the heading
  constants are defaults that have not been validated on real data.
- **Heading ramp for pedestrians and cyclists (open).** Below 0.5 m/s the linear model
  keeps a challenger's logged heading; above it, the heading blends toward the velocity
  direction over a 0.05 m/s band. A small acceleration bias can move a challenger's speed
  across that band and rotate its box a long way while its centre barely moves. The
  baseline-only check at construction does not prevent this. The code says so:
  `_linear_heading` in `src/optimization/perturbation_space.py` documents "NOT fixed —
  above the floor the 1/|v| sensitivity remains", and the comment on
  `HEADING_BLEND_SINGULARITY_MARGIN` has a paragraph headed "WHAT THIS DOES NOT CLOSE".
  `tests/test_heading_candidate_artifact.py` pins a synthetic reproduction (a cyclist
  fitted to numbers reported in review, not a scene from the dataset) and carries a strict
  xfail for the behaviour a fix would give.
  None of the 18 stored collisions depends on it. 17 targets are vehicles, which use the
  bicycle model and carry heading as state, so they never reach that code. The one other
  target, a pedestrian (`91a2a6ad0c50ce55`, collision frame 15), has a perturbed heading
  equal to its logged heading at the collision frames.
- **The site shows stored results.** It runs no search and computes nothing on request.

## License and data

The code is released under the [MIT License](LICENSE).

<!-- TODO(Avi): confirm against the Waymo Open Dataset license. -->
Raw Waymo Open Motion Dataset TFRecords are not in this repository (`/data/` and
`*.tfrecord` are git-ignored). The site shows trajectories derived from the dataset for
20 scenarios, with the attribution Waymo's license requires on every page. The data
remains under Waymo's terms.

<!-- TODO(Avi): DEPLOY.md quotes Vercel's Hobby plan as "non-commercial, personal use only".
Decide whether this README should say anything about that. It is deliberately not repeated here. -->

## Author

[github.com/AviShrivastava1](https://github.com/AviShrivastava1)
