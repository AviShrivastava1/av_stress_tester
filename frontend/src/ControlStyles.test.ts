import { describe, expect, it } from 'vitest';
import { cssRules, type CssRule } from './test/cssRules';
import { readFrontendFile } from './test/readFrontendFile';

/*
 * The replay and scene controls, and the detail page's small text, checked as text (jsdom does not apply the
 * stylesheet):
 *
 *   pressed state     the framing control (.segmented) shows which item is pressed by more than colour: an underline
 *                     that changes colour, and a heavier weight
 *   control size      every control rule that sets a size gives a target of at least 24 px each way; the framing items
 *                     are at least 30 px tall; the playback buttons are at least 40 px each way on a phone
 *   contrast          the colour pairs the detail page sets (back link, panel kicker, toolbar, framing items, the play
 *                     button, the playback-rate border) are computed from the tokens in :root
 *   focus             nothing removes the focus ring from a control, except the two selectors that draw their own
 */

const css = readFrontendFile('src/styles.css');
const rules = cssRules(css);
const root = rules.find((r) => r.selector === ':root')!;
const tokens = Object.fromEntries([...root.declarations.matchAll(/(--[\w-]+):([^;]+);/g)].map((m) => [m[1]!, m[2]!]));

const parts = (selector: string) => selector.split(',').map((part) => part.trim());
/** The rule with exactly this selector (a selector list matches when one of its parts is it), outside any media block unless `media` is given. */
const rule = (selector: string, media: string | null = null): CssRule | undefined =>
  rules.find((r) => r.media === media && parts(r.selector).includes(selector));
/** The value of a property in a rule's declarations, or undefined. */
const prop = (r: CssRule | undefined, name: string) => new RegExp(`(?:^|;)\\s*${name}:\\s*([^;]+)`).exec(r?.text ?? '')?.[1]?.trim();

/** A colour as `#rrggbb`: a hex value, or a var(--token) that resolves to one. */
function hex(value: string): string {
  const v = value.trim();
  const ref = /^var\((--[\w-]+)\)$/.exec(v);
  if (ref) return hex(tokens[ref[1]!] ?? '');
  if (!/^#[0-9a-f]{6}$/i.test(v)) throw new Error(`not a plain colour: ${value}`);
  return v.toLowerCase();
}
const luminance = (color: string) => {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16) / 255).map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!;
};
/** WCAG contrast ratio of two colours. */
const contrast = (a: string, b: string) => {
  const [hi, lo] = [luminance(hex(a)), luminance(hex(b))].sort((x, y) => y - x);
  return (hi! + 0.05) / (lo! + 0.05);
};
/** A length in px as a number, or undefined (so a missing or non-px value fails a >= check). */
const px = (value: string | undefined) => (value !== undefined && /^\d+(?:\.\d+)?px$/.test(value) ? Number.parseFloat(value) : undefined);
/** The left plus right padding of a `padding` shorthand (1 to 4 values), in px. */
function horizontalPadding(r: CssRule | undefined): number {
  const [top, right = top, bottom = top, left = right] = prop(r, 'padding')?.split(/\s+/) ?? [];
  void bottom;
  return (px(right) ?? 0) + (px(left) ?? 0);
}

describe('the contrast checker', () => {
  it('computes known ratios and sees a failing pair', () => {
    expect(contrast('#000000', '#ffffff')).toBeCloseTo(21, 0);
    expect(contrast('#777777', '#ffffff')).toBeLessThan(4.5);
    expect(contrast('var(--text)', 'var(--bg)')).toBeGreaterThan(15);
  });
});

describe('the framing control (.segmented)', () => {
  const base = () => rule('.segmented button')!;
  const pressed = () => rule(".segmented button[aria-pressed='true']")!;

  it('has rules for the item and its pressed state', () => {
    expect(base()).toBeDefined();
    expect(pressed()).toBeDefined();
  });

  it('shows the pressed item by more than colour: an underline that changes colour, and a heavier weight', () => {
    expect(prop(base(), 'border-bottom')).toMatch(/^1px/); // a 1 px underline is there for every item
    expect(prop(pressed(), 'border-bottom-color')).toBeDefined(); // and the pressed item's is a different colour
    expect(Number(prop(pressed(), 'font-weight'))).toBeGreaterThan(Number(prop(base(), 'font-weight') ?? 400));
  });

  it('is not a filled pill', () => {
    expect(prop(rule('.segmented'), 'background')).toBeUndefined();
    expect(prop(pressed(), 'background')).toBeUndefined();
  });

  it('gives each item a target of at least 24 px wide and 30 px tall', () => {
    expect(px(prop(base(), 'min-height'))).toBeGreaterThanOrEqual(30);
    expect(px(prop(base(), 'min-width'))).toBeGreaterThanOrEqual(24);
    expect(horizontalPadding(base())).toBeGreaterThanOrEqual(16);
  });
});

describe('control sizes', () => {
  const CONTROLS = ['.segmented button', '.playback-row button', '.rate select'];

  it.each(CONTROLS)('%s is at least 24 px each way', (selector) => {
    const r = rule(selector);
    expect(r, `a rule for ${selector}`).toBeDefined();
    expect(px(prop(r, 'min-height')), `${selector} min-height`).toBeGreaterThanOrEqual(24);
    const wide = px(prop(r, 'min-width')) ?? 0;
    expect(Math.max(wide, horizontalPadding(r)), `${selector} width`).toBeGreaterThanOrEqual(24);
  });

  it('the playback buttons are 36 by 34 px, and at least 40 px each way on a phone', () => {
    const base = rule('.playback-row button')!;
    expect(px(prop(base, 'min-width'))).toBeGreaterThanOrEqual(36);
    expect(px(prop(base, 'min-height'))).toBeGreaterThanOrEqual(34);
    const phone = rule('.playback-row button', '(max-width:560px)');
    expect(phone, 'a rule for the playback buttons at <=560 px').toBeDefined();
    expect(px(prop(phone, 'min-width'))).toBeGreaterThanOrEqual(40);
    expect(px(prop(phone, 'min-height'))).toBeGreaterThanOrEqual(40);
  });

  it('the checker sees a button that is too small', () => {
    const small = cssRules('.playback-row button { min-width:20px; min-height:20px; } @media(max-width:560px) { .playback-row button { min-width:36px; min-height:36px; } }');
    expect(px(prop(small.find((r) => r.media === null), 'min-width'))).toBeLessThan(24);
    expect(px(prop(small.find((r) => r.media !== null), 'min-height'))).toBeLessThan(40);
  });
});

describe('contrast of the colours the detail page sets', () => {
  const colour = (selector: string, name = 'color') => {
    const value = prop(rule(selector), name);
    if (value === undefined) throw new Error(`${selector} sets no ${name}`);
    return value;
  };
  const PAIRS: Array<[string, () => string, () => string]> = [
    ['the back link', () => colour('.back-link'), () => 'var(--bg)'],
    ['the back link on hover', () => colour('.back-link:hover'), () => 'var(--bg)'],
    ['the panel kicker', () => colour('.panel-kicker'), () => 'var(--bg)'],
    ['the toolbar text', () => colour('.scene-toolbar'), () => 'var(--bg)'],
    ['an unpressed framing item', () => colour('.segmented button'), () => 'var(--bg)'],
    ['the pressed framing item', () => colour(".segmented button[aria-pressed='true']"), () => 'var(--bg)'],
    ['the play button', () => colour('.playback-row .play'), () => colour('.playback-row .play', 'background')],
    ['the play button on hover', () => colour('.playback-row .play:hover:not(:disabled)'), () => colour('.playback-row .play:hover:not(:disabled)', 'background')],
  ];

  it.each(PAIRS)('%s is at least 4.5:1', (_name, fg, bg) => {
    expect(contrast(fg(), bg())).toBeGreaterThanOrEqual(4.5);
  });

  it('the playback-rate border is at least 3:1 on the page background', () => {
    const border = /^1px solid (.+)$/.exec(prop(rule('.rate select'), 'border') ?? '')?.[1];
    expect(border, 'a 1px solid border colour').toBeDefined();
    expect(contrast(border!, 'var(--bg)')).toBeGreaterThanOrEqual(3);
  });
});

describe('the focus ring', () => {
  /** Any way to switch the outline off: `outline:none|0`, `outline-style:none`, `outline-width:0`, with or without !important. */
  const REMOVES = /(?:^|;)(?:outline(?:-style)?:(?:none|0|0px)|outline-width:0(?:px)?)(?:!important)?(?:;|$)/;
  /** The only two selectors that draw their own focus: the main landmark (a skip-link target) and the lookup field, whose container shows it. */
  const OWN_FOCUS = new Set(['main:focus', '.scenario-lookup input:focus']);

  it('is drawn for every focusable element by one global rule', () => {
    const global = rule(':focus-visible');
    expect(global, 'a global :focus-visible rule').toBeDefined();
    expect(global!.declarations).toMatch(/outline:2pxsolidvar\(--accent\)/);
  });

  it('is removed by no rule but the two that draw their own', () => {
    const removing = rules.filter((r) => REMOVES.test(r.declarations)).flatMap((r) => parts(r.selector));
    expect(removing.filter((selector) => !OWN_FOCUS.has(selector))).toEqual([]);
  });

  it.each([
    'outline:none;',
    'outline:0;',
    'outline:none !important;',
    'outline:0 !important;',
    'outline-width:0;',
    'outline-width:0px !important;',
    'outline-style:none;',
    'color:red; outline-style:none !important',
  ])('the checker finds a rule that removes it from a control: %s', (declaration) => {
    const bad = cssRules(`.segmented button { ${declaration} }`);
    expect(bad.filter((r) => REMOVES.test(r.declarations))).toHaveLength(1);
  });

  it('the checker leaves a rule that draws or moves the outline alone', () => {
    const fine = cssRules('.a { outline:2px solid red; } .b { outline-offset:0; } .c { outline-width:2px; } .d { outline-color:transparent; }');
    expect(fine.filter((r) => REMOVES.test(r.declarations))).toHaveLength(0);
  });
});

describe('the glow on a colliding pair', () => {
  /*
   * The glow is a drop-shadow on the footprint polygon. `currentColor` there is the polygon's `color`, which is the page
   * text colour (nothing sets it from the role), so the glow would be dark on the dark scene frame and invisible. Each
   * role that can collide (the SDC and the perturbed challenger) gets a glow in its own role colour.
   */
  const glows = () => rules.filter((r) => parts(r.selector).some((part) => /\.footprint\.colliding/.test(part)) && /filter:/.test(r.declarations));
  const ROLES: Array<[string, string]> = [['role-sdc', '--role-sdc'], ['role-challenger_perturbed', '--role-perturbed']];

  it('is never currentColor, unless the same rule sets color from the role', () => {
    expect(glows().length).toBeGreaterThanOrEqual(2);
    for (const r of glows()) {
      const usesCurrent = /drop-shadow\([^)]*currentcolor/i.test(r.declarations);
      const colourFromRole = /(?:^|;)color:var\(--role-/.test(r.declarations);
      expect(usesCurrent && !colourFromRole, `${r.selector} glows in the text colour`).toBe(false);
    }
  });

  it.each(ROLES)('a colliding %s footprint glows in its own role colour', (roleClass, token) => {
    const rule = glows().find((r) => parts(r.selector).some((part) => part.includes(`.${roleClass}`)));
    expect(rule, `a glow rule for .${roleClass}`).toBeDefined();
    expect(rule!.text).toContain(`drop-shadow(0 0 5px var(${token}))`);
  });

  it('the checker sees a glow in the text colour', () => {
    const bad = cssRules('.scene .footprint.colliding polygon { filter:drop-shadow(0 0 5px currentColor); }');
    const r = bad[0]!;
    expect(/drop-shadow\([^)]*currentcolor/i.test(r.declarations) && !/(?:^|;)color:var\(--role-/.test(r.declarations)).toBe(true);
  });
});
