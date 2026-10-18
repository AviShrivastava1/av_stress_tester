import { describe, expect, it } from 'vitest';
import { readFrontendFile } from '../../test/readFrontendFile';
import { EXPORT_ROOT_VARIABLES, EXPORT_SCENE_RULES, EXPORT_STYLE } from './exportStyle';

/*
 * The style an exported frame carries (exportStyle.ts) is a copy of the page's scene rules. Every line of it must be in styles.css, word for word,
 * so that the export cannot drift from what the page draws; and every class SceneView draws has a rule in it.
 */

const css = readFrontendFile('src/styles.css');

describe('the export style', () => {
  it('has every rule verbatim in styles.css', () => {
    expect(EXPORT_SCENE_RULES.length).toBe(20);
    for (const rule of EXPORT_SCENE_RULES) expect(css, rule).toContain(rule);
  });

  it('has every colour variable it uses, with the page’s value', () => {
    for (const declaration of EXPORT_ROOT_VARIABLES.split(';').map((d) => d.trim()).filter(Boolean)) expect(css, declaration).toContain(`${declaration};`);
    const used = new Set([...EXPORT_SCENE_RULES.join('\n').matchAll(/var\((--[a-z-]+)\)/g)].map((m) => m[1]));
    for (const name of used) expect(EXPORT_ROOT_VARIABLES, name).toContain(`${name}:`);
  });

  it('starts with the colour variables, then the rules, so that a file taken out of the page has the colours', () => {
    expect(EXPORT_STYLE.startsWith(`:root { ${EXPORT_ROOT_VARIABLES} }\n.scene .path {`)).toBe(true);
    expect(EXPORT_STYLE.split('\n')).toHaveLength(1 + EXPORT_SCENE_RULES.length);
    expect(EXPORT_STYLE).toContain('--role-perturbed:#ff5c68;');
  });

  it('has a rule for each role and kind the scene draws, and leaves out the page’s sizing rule', () => {
    for (const cls of ['.path', '.kind-vehicle', '.kind-pedestrian', '.kind-cyclist', '.kind-unknown', '.role-sdc', '.role-challenger_logged', '.role-challenger_perturbed', '.footprint', '.ghost', '.colliding', '.scale-bar']) {
      expect(EXPORT_STYLE, cls).toContain(cls);
    }
    expect(EXPORT_STYLE).not.toContain('min-height');
    expect(EXPORT_STYLE).not.toContain('68vh');
  });
});
