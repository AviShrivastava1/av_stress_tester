/**
 * Required verbatim in any Derivative IP by the Waymo Dataset License Agreement for
 * Non-Commercial Use (license text as published March 2025). Rendered on every page.
 */
export const WAYMO_ATTRIBUTION =
  'This software was made using the Waymo Open Dataset, provided by Waymo LLC under ' +
  'the Waymo Dataset License Agreement for Non-Commercial Use, available at ' +
  'waymo.com/open/terms, and your access and use of such work are governed by the ' +
  'terms and conditions therein.';

export function Footer() {
  return (
    <footer className="site-footer">
      <p>{WAYMO_ATTRIBUTION}</p>
    </footer>
  );
}
