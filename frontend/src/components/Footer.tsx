/**
 * Required verbatim in any Derivative IP by the Waymo Dataset License Agreement for
 * Non-Commercial Use (license text as published March 2025). Rendered on every page.
 */
export const WAYMO_ATTRIBUTION =
  'This software was made using the Waymo Open Dataset, provided by Waymo LLC under ' +
  'the Waymo Dataset License Agreement for Non-Commercial Use, available at ' +
  'waymo.com/open/terms, and your access and use of such work are governed by the ' +
  'terms and conditions therein.';

/*
 * The two lines above the attribution:
 *   "AV Scenario Stress-Tester"   the project's name, as in frontend/index.html <title> and the header's
 *                                 home-link label (App.tsx)
 *   "Independent engineering project"
 *                                 an authorship statement by the project's author, not derived from code;
 *                                 Avi confirms it
 *   "scenario stress-testing on Waymo Open Motion data"
 *                                 the scenes are Waymo Open Motion Dataset scenarios (src/data/parser.py
 *                                 ScenarioParser) and "stress-testing" is src/scoring/batch_scorer.py
 *                                 stress_test_scenarios
 */
export const PROJECT_NAME = 'AV Scenario Stress-Tester';
export const PROJECT_DESCRIPTION =
  'Independent engineering project · scenario stress-testing on Waymo Open Motion data';

export function Footer() {
  return (
    <footer className="site-footer">
      <p className="footer-project">
        <strong>{PROJECT_NAME}</strong>
        <span>{PROJECT_DESCRIPTION}</span>
      </p>
      <p>{WAYMO_ATTRIBUTION}</p>
    </footer>
  );
}
