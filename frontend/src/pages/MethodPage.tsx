/*
 * How a recorded scene becomes a stored, replayable result, and what the output does not mean.
 * Static: it makes no API request, and it claims nothing the site does not do (nothing is
 * computed live, nothing is certified). What backs each sentence:
 *   01  src/data/parser.py (ScenarioParser), src/data/validity.py
 *   02  src/danger/ttc_engine.py, src/danger/pet_engine.py (_conflict_components),
 *       src/danger/danger_score.py, src/scoring/ranker.py (ties broken by scenario ID)
 *   03  src/optimization/perturbation_space.py, scipy_optimizer.py (differential_evolution),
 *       autograd_optimizer.py (torch; refine_scenario, off unless _stress_one(use_autograd=True))
 *   04  perturbation_space.ReplayFidelityError, src/danger/collision_detector.py (float64 corners),
 *       src/scoring/db.py (compute_scene_fingerprint, compute_stress_run_id)
 *   05  src/scoring/export_geometry.py, src/api/ (no write route), scene/geometry.ts (observationAt)
 * Principles: outcomes.ts and batch_scorer (distinct failure statuses), requirements-api.txt
 * (no TensorFlow or shard at request time), the scene_fingerprint / search_provenance columns.
 *
 * The fifth limit, on pedestrian and cyclist replay, sentence by sentence:
 *   "less well measured than vehicle replay"
 *       perturbation_space.py, the comment above SPEED_STEP_REFUSE_MPS: the linear model's A_MAX (5.0)
 *       "has never been measured" and its replay "carries ~0.16 m/s of speed error even without a step";
 *       the vehicle side was measured (the comments above BASELINE_DRIFT_REFUSE_M and SPEED_STEP_REFUSE_MPS)
 *   "the recorded heading when the agent is slow and the heading implied by its motion when it is faster,
 *    blending between them over a narrow speed band"
 *       perturbation_space.py _linear_heading: the logged heading below heading_speed_floor (V_HEADING_MIN),
 *       the velocity-derived heading from the floor plus HEADING_TRANSITION_WIDTH upward, a smoothstep blend in between
 *   "near that band a very small change can turn a box a long way while its centre barely moves"
 *       _linear_heading's docstring ("NOT fixed — above the floor the 1/|v| sensitivity remains"), and
 *       tests/test_heading_candidate_artifact.py (a weighted norm of 0.001 moved the heading 0.67 rad, the centre 4 cm)
 *   "Vehicles use the bicycle model, which carries heading as part of its state, and are not affected by this"
 *       src/physics/simulator.py ForwardSimulator.step (TYPE_VEHICLE -> bicycle_step); _linear_heading's
 *       docstring ("Vehicles never reach this function")
 * Nothing here is a count: the page carries no data numbers, which go stale.
 */
const PIPELINE = [
  {
    number: '01',
    title: 'Ingest recorded motion',
    body: 'Parse masked agent trajectories from WOMD and preserve the local metric coordinate frame.',
    meta: 'WOMD · NumPy · validity masks',
  },
  {
    number: '02',
    title: 'Rank scenes by fragility',
    body: 'Measure SDC-restricted TTC and component-aware PET, then rank the corpus by a fragility score.',
    meta: 'TTC · PET · deterministic ranking',
  },
  {
    number: '03',
    title: 'Search a constrained replay',
    body:
      'Perturb one selected challenger in a four-dimensional, model-specific space using Differential ' +
      'Evolution and optional autograd refinement.',
    meta: 'SciPy · optional PyTorch · kinematic models',
  },
  {
    number: '04',
    title: 'Verify before publishing',
    body:
      'Refuse unfaithful baselines, check contact with oriented boxes, and bind every result to its scene ' +
      'and replay provenance.',
    meta: 'Replay gates · float64 geometry · hashes',
  },
  {
    number: '05',
    title: 'Serve a stored replay',
    body:
      'Export geometry once, keep the production API read-only, and let the browser compare logged and ' +
      'perturbed paths frame by frame.',
    meta: 'PostGIS · FastAPI · React',
  },
] as const;

const PRINCIPLES = [
  {
    title: 'Failure is a first-class result',
    body:
      'Replay drift, speed discontinuities, heading singularities, stale exports, and ordinary search ' +
      'misses stay distinct from one another.',
  },
  {
    title: 'The online path stays lightweight',
    body:
      'Waymo parsing and optimization run offline. The deployed API reads Postgres/PostGIS and never needs ' +
      'a shard or TensorFlow at request time.',
  },
  {
    title: 'Claims carry their provenance',
    body:
      'Scene fingerprints, SDC identity, replay settings, search budgets, and run IDs travel with stored ' +
      'results and exported geometry.',
  },
] as const;

const LIMITS = [
  'The minimum perturbation is the best result found within a finite search budget, not a proven global minimum.',
  'Each run searches one heuristically selected challenger, not every possible agent interaction.',
  'A collision is an oriented-box intersection at stored coordinate precision, not a forecast of real-world impact.',
  'The browser draws each box at the size recorded for that frame, which is the size the collision check used. ' +
    'Data exported before per-frame sizes were stored is drawn at one size per agent, and the note under the ' +
    'scene says so.',
  'Pedestrian and cyclist replay is less well measured than vehicle replay. For these agents the model uses the ' +
    'recorded heading when the agent is slow and the heading implied by its motion when it is faster, blending ' +
    'between them over a narrow speed band; near that band a very small change can turn a box a long way while ' +
    'its centre barely moves. Vehicles use the bicycle model, which carries heading as part of its state, and are ' +
    'not affected by this.',
] as const;

export function MethodPage() {
  return (
    <section className="method-page">
      <header className="method-hero">
        <div>
          <p className="eyebrow">SYSTEM DESIGN</p>
          <h1>From a recorded scene to a stored, replayable result.</h1>
        </div>
        <p className="method-lede">
          This is not a collision-probability model. It asks a narrower question: under a bounded kinematic
          replay, how small a change to one agent's motion produces geometric contact?
        </p>
      </header>

      <div className="pipeline-list" aria-label="System pipeline">
        {PIPELINE.map((step) => (
          <article className="pipeline-step" key={step.number}>
            <span className="pipeline-number" aria-hidden="true">{step.number}</span>
            <div>
              <h2>{step.title}</h2>
              <p>{step.body}</p>
              <span className="pipeline-meta">{step.meta}</span>
            </div>
          </article>
        ))}
      </div>

      <section className="method-section">
        <p className="eyebrow">ENGINEERING PRINCIPLES</p>
        <h2>Built to expose assumptions, not hide them.</h2>
        <div className="principle-grid">
          {PRINCIPLES.map((principle) => (
            <article key={principle.title}>
              <h3>{principle.title}</h3>
              <p>{principle.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="limits-panel">
        <div>
          <p className="eyebrow">INTERPRETATION</p>
          <h2>What the output does—and does not—mean</h2>
        </div>
        <ul role="list">
          {LIMITS.map((limit) => (
            <li key={limit}>{limit}</li>
          ))}
        </ul>
      </section>
    </section>
  );
}
