/*
 * The style of an exported scene frame. The SVG the page draws is styled by classes in styles.css, which a file taken out of the page does not
 * have, so the rules for the scene are carried here, verbatim, with the colours they use. exportStyle.test.ts reads styles.css and fails if any of
 * these lines is not in it, so the export cannot drift from the page. The page's `.scene` rule (size and display) is left out: the export sets the size itself.
 */
export const EXPORT_ROOT_VARIABLES =
  '--role-sdc:#3f83ff; --role-challenger:#e2ad44; --role-perturbed:#ff5c68; --kind-vehicle:#86928c; --kind-pedestrian:#4ecb9f; --kind-cyclist:#b78df0; --kind-unknown:#89928d;';

export const EXPORT_SCENE_RULES: readonly string[] = [
  '.scene .path { fill:none; stroke-width:1.5px; stroke-linejoin:round; stroke-linecap:round; }',
  '.scene .track.role-other .path { stroke-width:.8px; opacity:.58; }',
  '.scene .kind-vehicle { stroke:var(--kind-vehicle); fill:var(--kind-vehicle); }',
  '.scene .kind-pedestrian { stroke:var(--kind-pedestrian); fill:var(--kind-pedestrian); }',
  '.scene .kind-cyclist { stroke:var(--kind-cyclist); fill:var(--kind-cyclist); }',
  '.scene .kind-unknown { stroke:var(--kind-unknown); fill:var(--kind-unknown); }',
  '.scene .role-sdc { stroke:var(--role-sdc); fill:var(--role-sdc); }',
  '.scene .role-challenger_logged { stroke:var(--role-challenger); fill:var(--role-challenger); }',
  '.scene .role-challenger_perturbed { stroke:var(--role-perturbed); fill:var(--role-perturbed); }',
  '.scene .track.role-sdc .path,.scene .track.role-challenger_logged .path,.scene .track.role-challenger_perturbed .path { stroke-width:2.5px; }',
  '.scene .track.role-challenger_perturbed .path { stroke-dasharray:6 4; }',
  '.scene .track .path { fill:none; }',
  '.scene .footprint polygon { fill-opacity:.3; stroke-width:1.5px; }',
  '.scene .footprint .heading { stroke-width:1.5px; }',
  '.scene .footprint.ghost polygon { fill:none; stroke-dasharray:3 2; }',
  '.scene .footprint.colliding polygon { fill-opacity:.7; stroke-width:2.8px; }',
  '.scene .footprint.colliding.role-sdc polygon { filter:drop-shadow(0 0 5px var(--role-sdc)); }',
  '.scene .footprint.colliding.role-challenger_perturbed polygon { filter:drop-shadow(0 0 5px var(--role-perturbed)); }',
  '.scene .scale-bar line { stroke:#a7b0aa; stroke-width:2px; }',
  '.scene .scale-bar text { fill:#a7b0aa; }',
];

/** Everything an exported frame is styled with, as the text of one <style> element. */
export const EXPORT_STYLE = `:root { ${EXPORT_ROOT_VARIABLES} }\n${EXPORT_SCENE_RULES.join('\n')}`;
