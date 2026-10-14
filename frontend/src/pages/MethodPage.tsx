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
 *
 * The words at the top. What backs each one:
 *   eyebrow "SYSTEM / PROVENANCE"
 *       the h1's subject: the run and scene a stored replay is tied to (scenario_scores.stress_run_id, scene_fingerprint,
 *       search_provenance; src/scoring/db.py)
 *   h1 "A stored replay is shown only with the run it came from."
 *       src/api/routes.py get_perturbed reads the stored perturbed path only where perturbed_paths.stress_run_id and
 *       perturbed_paths.scene_fingerprint match the scenario_scores row's (`IS NOT DISTINCT FROM`), and reads the logged
 *       track only where scenario_agents.scene_fingerprint matches; a path from another run or scene is not served
 *       (tests test_a_mismatched_run_id_hides_the_perturbed_path and the scene-fingerprint audit A02 tests). The ids are
 *       src/scoring/db.py compute_stress_run_id (a truncated sha256 over the scenario, challenger, delta, method and
 *       settings) and compute_scene_fingerprint (a truncated sha256 over states, validity and types). A legacy row whose id
 *       is NULL matches only a path whose id is NULL, so the claim is about stored replays, not every scored scenario:
 *       a scenario nobody searched, or whose attempt was refused, has no stored replay, and the page says nothing about it
 *   line "This is not a collision-probability model."
 *       README.md ("not a collision probability"); the list's metric definitions say the same of fragility, which ranks
 *       recorded scenes (src/danger/danger_score.py), while min_perturbation is the result of a bounded search
 *       (src/optimization/scipy_optimizer.py optimize_scenario) for geometric contact
 *   METHOD_SIGNATURE
 *       Optimizer "SciPy Differential Evolution"  scipy_optimizer.py, `differential_evolution` in optimize_scenario
 *       Models "bicycle / linear"                 src/physics/bicycle_model.py, linear_model.py, chosen by simulator.py
 *       Geometry "float64 oriented boxes"         collision_detector.get_corners (np.float64), COLLISION_GEOMETRY_VERSION
 *       Identity "truncated sha256 digests"       db.compute_scene_fingerprint and db.compute_stress_run_id keep the first 16 hex
 *                                                 characters (64 bits) of a sha256, as their docstrings say
 *       Serving "PostGIS → FastAPI → React"       src/scoring/export_geometry.py, src/api/routes.py, frontend/
 * The limits and the pipeline steps are as before. The page carries no data numbers (a test checks), because counts go stale.
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
 *
 * The sixth limit, on the search window, sentence by sentence:
 *   "The search covers only the recorded frames."
 *       the challenger is re-simulated from its first observed frame to the end of the recording, and nothing past that
 *       is simulated: perturbation_space.py PerturbationSpace.apply, `controls = self.base_controls[self.t0:].copy()`
 *       (line 649) and `traj = self.simulator.simulate(init, controls, self.dt)  # (T-t0, state_dim)` (line 675), where
 *       simulator.py `T = len(controls) + 1` (line 86) makes the rollout end at the last frame, and _write_back's
 *       `n = min(self.T - t0, traj.shape[0])` (line 774) writes frames up to T - 1. The recording is T frames
 *       (src/data/parser.py, `T = 91` at line 32). The contact check reads the same frames only:
 *       collision_detector.py check_collision_trajectory, `for t in range(T):` (line 115), and the search's own margin,
 *       scipy_optimizer.py `for t in range(T):` (line 105), both skipping frames where either agent was not observed
 *       (collision_detector.py lines 117-118, scipy_optimizer.py line 106)
 *   "A collision that would happen after the last recorded frame is not looked for."
 *       there is no frame after the last one in those arrays, and no special case for the last frame, so a stored
 *       collision at the last frame is a contact found there and says nothing about later frames (the detail page says
 *       so under the note for the collision frame, scene/sceneModel.ts searchWindowNote)
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

const METHOD_SIGNATURE = [
  ['Optimizer', 'SciPy Differential Evolution'],
  ['Models', 'bicycle / linear'],
  ['Geometry', 'float64 oriented boxes'],
  ['Identity', 'truncated sha256 digests'],
  ['Serving', 'PostGIS → FastAPI → React'],
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
  'The search covers only the recorded frames. A collision that would happen after the last recorded frame is not looked for.',
] as const;

export function MethodPage() {
  return (
    <section className="method-page">
      <header className="method-hero">
        <div>
          <p className="eyebrow">SYSTEM / PROVENANCE</p>
          <h1>A stored replay is shown only with the run it came from.</h1>
          <p className="method-lede">This is not a collision-probability model.</p>
        </div>
        <dl className="method-signature" aria-label="System components">
          {METHOD_SIGNATURE.map(([term, value]) => (
            <div key={term}><dt>{term}</dt><dd>{value}</dd></div>
          ))}
        </dl>
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
