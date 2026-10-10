import { describe, expect, it } from 'vitest';
import { cssRules } from './cssRules';
import { readFrontendFile } from './readFrontendFile';

/*
 * The stylesheet parser the style tests stand on. It must not lose a rule: a rule it drops is a rule no style test can
 * see. Only the contents of @keyframes are skipped.
 */

describe('cssRules on small sheets', () => {
  it('returns top-level rules with no condition, comments removed', () => {
    const rules = cssRules('/* a { color:red } */ .a { color:red; } .b, .c { margin:0 4px; }');
    expect(rules.map((r) => [r.selector, r.declarations, r.media, r.conditions])).toEqual([
      ['.a', 'color:red;', null, []],
      ['.b, .c', 'margin:04px;', null, []],
    ]);
    expect(rules[1]!.text).toBe('margin:0 4px;');
  });

  it('returns a rule inside @media with its condition, and the rules after the block without one', () => {
    const rules = cssRules('@media(max-width:560px) { .a { x:1; } .b { x:2; } } .c { x:3; }');
    expect(rules.map((r) => [r.selector, r.media, r.conditions])).toEqual([
      ['.a', '(max-width:560px)', ['@media(max-width:560px)']],
      ['.b', '(max-width:560px)', ['@media(max-width:560px)']],
      ['.c', null, []],
    ]);
  });

  it('keeps the rules inside @supports, @layer and @container, with their conditions', () => {
    const rules = cssRules(
      '@supports (display:grid) { .g { display:grid; } } @layer base { .l { color:red; } } @container (min-width:400px) { .k { x:1; } }',
    );
    expect(rules.map((r) => [r.selector, r.conditions])).toEqual([
      ['.g', ['@supports (display:grid)']],
      ['.l', ['@layer base']],
      ['.k', ['@container (min-width:400px)']],
    ]);
    expect(rules.map((r) => r.media)).toEqual([null, null, null]); // `media` is only for @media
  });

  it('nests conditions: a rule in @supports inside @media has both', () => {
    const [rule] = cssRules('@media(max-width:560px) { @supports (display:grid) { .g { x:1; } } .h { x:2; } }');
    expect(rule).toMatchObject({ selector: '.g', media: '(max-width:560px)', conditions: ['@media(max-width:560px)', '@supports (display:grid)'] });
    expect(cssRules('@media(max-width:560px) { @supports (display:grid) { .g { x:1; } } .h { x:2; } }')[1]).toMatchObject({
      selector: '.h',
      conditions: ['@media(max-width:560px)'],
    });
  });

  it('skips the contents of @keyframes, and nothing after them', () => {
    const rules = cssRules('@keyframes spin { from { x:0; } to { x:1; } } .after { y:2; }');
    expect(rules.map((r) => r.selector)).toEqual(['.after']);
  });

  it('returns a block of declarations such as @font-face as a rule of its own', () => {
    const rules = cssRules("@font-face { font-family:X; src:url(x.woff2); } .a { x:1; }");
    expect(rules.map((r) => [r.selector, r.declarations])).toEqual([
      ['@font-face', 'font-family:X;src:url(x.woff2);'],
      ['.a', 'x:1;'],
    ]);
  });

  it('does not let a statement at-rule (@import, @charset, @layer a, b;) become part of the next selector', () => {
    const rules = cssRules('@charset "utf-8"; @import url(x.css); @layer a, b; .a { x:1; }');
    expect(rules.map((r) => r.selector)).toEqual(['.a']);
  });
});

describe('cssRules on the real stylesheet', () => {
  const css = readFrontendFile('src/styles.css');
  const text = css.replace(/\/\*[\s\S]*?\*\//g, '');

  it('uses only the at-rules the parser handles: @media and @keyframes', () => {
    const kinds = [...new Set([...text.matchAll(/@([a-zA-Z-]+)/g)].map((m) => m[1]))].sort();
    expect(kinds).toEqual(['keyframes', 'media']);
  });

  it('misses no rule that is in the stylesheet text; only the contents of @keyframes are skipped', () => {
    // Independent of the parser: drop each @keyframes block (one level of inner blocks), then every `selector { … }`
    // that is left, with no brace inside it, is a rule.
    const withoutKeyframes = text.replace(/@keyframes[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}/g, '');
    const expected = [...withoutKeyframes.matchAll(/([^{}]+)\{([^{}]*)\}/g)]
      .filter((m) => !m[1]!.trim().startsWith('@'))
      .map((m) => `${m[1]!.trim().replace(/\s+/g, ' ')} {${m[2]!.replace(/\s+/g, '')}}`);
    const actual = cssRules(css).map((r) => `${r.selector} {${r.declarations}}`);
    expect(expected.length).toBeGreaterThan(250); // there is a real stylesheet here
    expect(actual.sort()).toEqual(expected.sort());
  });

  it('sees the rules inside every @media block, each with its condition', () => {
    const inMedia = cssRules(css).filter((r) => r.media !== null);
    expect(inMedia.length).toBeGreaterThan(40);
    expect(new Set(inMedia.map((r) => r.media))).toEqual(
      new Set(['(max-width:1100px)', '(max-width:1000px)', '(max-width:820px)', '(max-width:560px)', '(prefers-reduced-motion:reduce)']),
    );
  });
});
