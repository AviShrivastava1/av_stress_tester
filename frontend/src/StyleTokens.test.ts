import { describe, expect, it } from 'vitest';
import { cssRules } from './test/cssRules';
import { readFrontendFile } from './test/readFrontendFile';

/*
 * The stylesheet's palette and the text sizes it must not shrink below, checked as text (jsdom does not apply it).
 *
 *   light only         the palette is one light scheme: color-scheme is `light` and there is no prefers-color-scheme
 *                      block, so the browser's own controls and the page agree
 *   theme-color        the browser chrome colour in index.html is the page background, --bg
 *   defined tokens     every var(--x) in the stylesheet has a definition in :root
 *   scene tokens       the role and agent-kind colours are the ones the dark scene frame was designed for
 *   11 px floor        text that carries a caveat, a denial or the Waymo licence attribution is never set below 11 px;
 *                      the hero note, when it arrives, never below 12 px
 */

const css = readFrontendFile('src/styles.css');
const html = readFrontendFile('index.html');
const rules = cssRules(css);
const root = rules.find((r) => r.selector === ':root')!;
const token = (name: string) => new RegExp(`${name}:([^;]+);`).exec(root.declarations)?.[1];

describe('the palette is light only', () => {
  it('declares color-scheme: light, and no dark', () => {
    expect(root.declarations).toContain('color-scheme:light;');
    expect(css).not.toMatch(/color-scheme:\s*[^;]*dark/);
  });

  it('has no prefers-color-scheme block', () => {
    expect(css).not.toMatch(/prefers-color-scheme/);
  });
});

describe('theme-color', () => {
  it('is the page background, --bg', () => {
    const themeColor = /<meta\s+name="theme-color"\s+content="([^"]+)"\s*\/?>/.exec(html)?.[1];
    expect(token('--bg')).toBe('#f2f3ee');
    expect(themeColor).toBe(token('--bg'));
  });

  it('is declared exactly once', () => {
    expect(html.match(/name="theme-color"/g)).toHaveLength(1);
  });
});

describe('the tokens', () => {
  it('every var(--x) used in the stylesheet is defined in :root', () => {
    const used = new Set([...css.matchAll(/var\((--[\w-]+)/g)].map((m) => m[1]!));
    const defined = new Set([...root.declarations.matchAll(/(--[\w-]+):/g)].map((m) => m[1]!));
    expect(used.size).toBeGreaterThan(10);
    expect([...used].filter((name) => !defined.has(name))).toEqual([]);
  });

  it('keeps the scene colours the dark scene frame was designed for', () => {
    expect(token('--role-sdc')).toBe('#3f83ff');
    expect(token('--role-challenger')).toBe('#e2ad44');
    expect(token('--role-perturbed')).toBe('#ff5c68');
    expect(token('--kind-vehicle')).toBe('#86928c');
    expect(token('--kind-pedestrian')).toBe('#4ecb9f');
    expect(token('--kind-cyclist')).toBe('#b78df0');
    expect(token('--kind-unknown')).toBe('#89928d');
    const frame = rules.find((r) => r.selector === '.scene-frame')!;
    expect(frame.declarations).toContain('background-color:#0b100e');
  });
});

/**
 * Text that carries a caveat, a denial or the Waymo licence attribution, by the class that styles it. The list is an
 * audit of the copy in the .tsx files (their backing comments), not a pattern: a class is here because a sentence on
 * a page is set in it. Each must exist in styles.css, and none may set a size below the floor.
 */
const CAVEAT_CLASSES: Record<string, string> = {
  'site-footer': 'the Waymo licence attribution (components/Footer.tsx)',
  caption: 'the list caption (ties, "not a proven minimum"); the Corpus captions and the non-partition caveat',
  'metric-guide': 'the list metric guide ("not a collision probability", "not a proven global minimum")',
  note: 'the detail scene notes (box sizes, positions at frame); the Corpus "not guaranteed to add up" note',
  'replay-note': 'the detail baseline-offset note',
  'stat-def': 'the Corpus definitions, including "not a safety certificate"',
  'limits-panel': 'the Method page LIMITS',
  'method-lede': 'the Method page "not a collision-probability model"',
  status: 'the detail "No geometry to draw…" sentence; loading and error messages',
  'status-error': 'inline error text',
  'keyboard-hint': 'the instruction sentence under the replay',
  'hero-note': 'the list hero note: "not proven", searches ran offline, what a tie means',
  'trace-label': 'the list hero "ILLUSTRATION · not measured data"',
};
const FLOOR = 11;
/** The hero note, the first thing under the hero claim, is held to 12 px. */
const FLOOR_BY_CLASS: Record<string, number> = { 'hero-note': 12 };

/**
 * The only selectors allowed below the floor: short labels, eyebrows and numbers, not sentences. A new selector below
 * the floor that is not named here fails, so a new class is classified on purpose instead of by a pattern.
 */
const SHORT_LABELS = new Set([
  '.eyebrow', '.facts dd.mono', '.hero-replay > span:first-child', '.legend', '.panel-kicker', '.pipeline-meta',
  '.method-signature dt', '.pipeline-number', '.rate select', '.system-signature dt', '.readout', '.result-overview > div > span',
  '.scene-panel-heading > span', '.tag',
]);

/** The font size, in px, a rule sets by `font-size` or the `font` shorthand; null if it sets none. */
const px = (text: string): number | null => {
  const size = /(?:^|;)\s*font-size:\s*(\d+(?:\.\d+)?)px/.exec(text) ?? /(?:^|;)\s*font:[^;]*?(\d+(?:\.\d+)?)px/.exec(text);
  return size ? Number(size[1]) : null;
};
const parts = (selector: string) => selector.split(',').map((part) => part.trim()).filter((part) => !part.includes(':not('));
/** True when the selector names the class (also in a compound such as `dd.replay-note`), not a longer class such as `.footnote`. */
const hasClass = (part: string, name: string) => new RegExp(`\\.${name}(?![\\w-])`).test(part);

function sizeViolations(all: ReturnType<typeof cssRules>): string[] {
  const out: string[] = [];
  for (const r of all) {
    const size = px(r.text);
    if (size === null) continue;
    for (const part of parts(r.selector)) {
      for (const name of Object.keys(CAVEAT_CLASSES)) {
        const floor = FLOOR_BY_CLASS[name] ?? FLOOR;
        if (hasClass(part, name) && size < floor) out.push(`${part} { ${size}px } carries ${name} text (floor ${floor}px)`);
      }
      if (size < FLOOR && !SHORT_LABELS.has(part)) out.push(`${part} { ${size}px } is under ${FLOOR}px and is not a listed short label`);
    }
  }
  return [...new Set(out)];
}

describe('the text-size floor', () => {
  it('every class that styles a caveat, denial or licence sentence is in the stylesheet, by its real name', () => {
    for (const name of Object.keys(CAVEAT_CLASSES)) {
      expect(rules.some((r) => parts(r.selector).some((part) => hasClass(part, name))), `.${name} has no rule`).toBe(true);
    }
  });

  it('no caveat, denial or licence text is set below 11 px (hero note: 12 px), in any media block', () => {
    expect(rules.filter((r) => px(r.text) !== null).length).toBeGreaterThan(40); // the checker reads real rules
    expect(sizeViolations(rules)).toEqual([]);
  });

  it('nothing else is under 11 px unless it is a listed short label: no unclassified small text', () => {
    const small = rules.flatMap((r) => (px(r.text) !== null && px(r.text)! < FLOOR ? parts(r.selector) : []));
    expect(small.length).toBeGreaterThan(5);
    expect(small.filter((part) => !SHORT_LABELS.has(part))).toEqual([]);
  });

  it('font sizes are in px, so the floor means what it says', () => {
    expect(css).not.toMatch(/font-size:\s*[\d.]+(?:r?em|%)/);
    expect(css).not.toMatch(/(?:^|[;{\s])font:[^;}]*\b[\d.]+(?:r?em|%)\//);
  });

  it('the footer, which carries the Waymo licence attribution, is not smaller than it was (12 px)', () => {
    const sizes = rules.filter((r) => parts(r.selector).some((part) => hasClass(part, 'site-footer')) && px(r.text) !== null);
    expect(sizes.length).toBeGreaterThan(0);
    for (const r of sizes) expect(px(r.text)!, r.selector).toBeGreaterThanOrEqual(12);
  });

  it('the checker itself: finds a class by its real name, an unlisted small class, and spares short labels', () => {
    const sample = cssRules(
      '.limits-panel ul { font-size:10px; } .scene-note { font-size:9px; } .keyboard-hint { font-size:10px; } .hero-note { font-size:11px; } ' +
        '.eyebrow { font:720 10px/1.5 mono; } .legend { font-size:10px; } .stat[data-stat="x"] > dd:not(.stat-def) { font-size:9px; } .footnote { font-size:12px; }',
    );
    expect(sizeViolations(sample)).toEqual([
      '.limits-panel ul { 10px } carries limits-panel text (floor 11px)',
      '.limits-panel ul { 10px } is under 11px and is not a listed short label',
      '.scene-note { 9px } is under 11px and is not a listed short label',
      '.keyboard-hint { 10px } carries keyboard-hint text (floor 11px)',
      '.keyboard-hint { 10px } is under 11px and is not a listed short label',
      '.hero-note { 11px } carries hero-note text (floor 12px)',
    ]);
  });
});

/*
 * The header: pinned to the top while the page scrolls, with the current page underlined, and the title hidden only where
 * the header has no room for it (below 560 px; the home link's aria-label carries the name then, see App.tsx).
 */
function headerProblems(sheet: string): string[] {
  const all = cssRules(sheet);
  const problems: string[] = [];

  // the title: hidden only inside the max-width:560px block, and hidden there
  const titleRules = all.filter((r) => parts(r.selector).some((part) => hasClass(part, 'site-title')));
  const titleHidden = titleRules.filter((r) => /(?:^|;)display:none(?:!important)?(?:;|$)/.test(r.declarations));
  if (titleHidden.length === 0) problems.push('.site-title is never hidden: the narrow header has no room for it');
  for (const r of titleHidden) {
    if (r.media !== '(max-width:560px)') problems.push(`.site-title is hidden outside the 560px block: ${r.selector} in ${r.media ?? 'no media block'}`);
  }

  // the header stays at the top while the page scrolls: sticky at the top level, and no rule in any block says otherwise
  const headerRules = all.filter((r) => parts(r.selector).some((part) => /^(?:header)?\.site-header$/.test(part)));
  if (!headerRules.some((r) => r.media === null && /(?:^|;)position:sticky(?:!important)?(?:;|$)/.test(r.declarations))) {
    problems.push('.site-header is not position:sticky');
  }
  for (const r of headerRules) {
    const position = /(?:^|;)position:([^;!]+)/.exec(r.declarations)?.[1];
    if (position !== undefined && position !== 'sticky') problems.push(`.site-header sets position:${position} in ${r.media ?? 'no media block'}`);
  }

  // the current page is underlined: the ::after of the active link has a visible background, and it is the last word
  const active = all.filter((r) => r.selector === '.site-nav a.active::after');
  const background = /(?:^|;)background(?:-color)?:([^;]+)/.exec(active.at(-1)?.declarations ?? '')?.[1];
  if (background === undefined || /^(?:transparent|none|initial|inherit|unset)$/.test(background)) {
    problems.push('.site-nav a.active::after has no visible background: the current page is not underlined');
  }
  return problems;
}

describe('the header', () => {
  it('is sticky, underlines the current page, and hides its title only below 560 px', () => {
    expect(rules.some((r) => r.media === '(max-width:560px)')).toBe(true); // the parser sees media blocks
    expect(headerProblems(css)).toEqual([]);
  });

  it('the checker finds the title hidden outside the 560px block, in a copy of the stylesheet', () => {
    const broken = css.replace('.site-title { font-size:13px;', '.site-title { display:none; }\n.site-title { font-size:13px;');
    expect(broken).not.toBe(css);
    expect(headerProblems(broken)).toEqual(['.site-title is hidden outside the 560px block: .site-title in no media block']);
  });

  it('the checker finds the title hidden with !important outside the 560px block, in a copy of the stylesheet', () => {
    const broken = css.replace('.site-title { font-size:13px;', '.site-title { display:none !important; }\n.site-title { font-size:13px;');
    expect(broken).not.toBe(css);
    expect(headerProblems(broken)).toEqual(['.site-title is hidden outside the 560px block: .site-title in no media block']);
  });

  it('accepts !important on the rule that hides the title inside the 560px block', () => {
    const important = css.replace('.site-title { display:none; }', '.site-title { display:none !important; }');
    expect(important).not.toBe(css);
    expect(headerProblems(important)).toEqual([]);
  });

  it('the checker finds the header made non-sticky inside the 560px block, in a copy of the stylesheet', () => {
    const broken = css.replace('.site-header { min-height:68px; }', '.site-header { min-height:68px; } .site-header { position:static; }');
    expect(broken).not.toBe(css);
    expect(headerProblems(broken)).toEqual(['.site-header sets position:static in (max-width:560px)']);
  });

  it('the checker finds the header no longer sticky, in a copy of the stylesheet', () => {
    const broken = css.replace('.site-header { position:sticky; top:0;', '.site-header { top:0;');
    expect(broken).not.toBe(css);
    expect(headerProblems(broken)).toEqual(['.site-header is not position:sticky']);
  });

  it('the checker finds the current page no longer underlined, in a copy of the stylesheet', () => {
    const broken = css.replace('.site-nav a.active::after { background:var(--text); }', '.site-nav a.active::after { background:transparent; }');
    expect(broken).not.toBe(css);
    expect(headerProblems(broken)).toEqual(['.site-nav a.active::after has no visible background: the current page is not underlined']);
  });
});

describe('the hero note', () => {
  it('has a rule, and it sets at least 12 px', () => {
    const sizes = rules.filter((r) => parts(r.selector).some((part) => hasClass(part, 'hero-note')) && px(r.text) !== null);
    expect(sizes.length).toBeGreaterThan(0);
    for (const r of sizes) expect(px(r.text)!, r.selector).toBeGreaterThanOrEqual(12);
  });
});

/*
 * The hero's words are never hidden at any width: the eyebrow and the lede say "counterfactual", the note says "not
 * proven" and that the search ran offline, and the illustration label says the drawing is not data. A stylesheet that
 * hid one of them on a phone would take the words away where most people read them.
 */
const HERO_WORDS = ['hero-eyebrow', 'hero-lede', 'hero-note', 'hero-actions', 'trace-label'];
const hidesHeroWords = (all: ReturnType<typeof cssRules>) =>
  all
    .filter((r) => parts(r.selector).some((part) => HERO_WORDS.some((name) => hasClass(part, name)) || /^\.hero-copy(?: [>+~] |\s)*$/.test(part)))
    .filter((r) => /(?:^|;)(?:display:none|visibility:hidden|opacity:0|font-size:0)(?:!important)?(?:;|$)/.test(r.declarations))
    .map((r) => `${r.selector} in ${r.media ?? 'no media block'}`);

describe('the hero words stay visible', () => {
  it('no rule, in any media block, hides the hero eyebrow, lede, note, actions or illustration label', () => {
    expect(rules.some((r) => parts(r.selector).some((part) => hasClass(part, 'hero-note')))).toBe(true); // the rules exist
    expect(hidesHeroWords(rules)).toEqual([]);
  });

  it('the checker finds a rule that hides the lede on a phone', () => {
    const broken = css + '\n@media(max-width:560px) { .hero-lede { display:none; } }';
    expect(hidesHeroWords(cssRules(broken))).toEqual(['.hero-lede in (max-width:560px)']);
  });
});

describe('generated content', () => {
  it('puts no text in front of or behind a link or a button: generated text is read as part of the name', () => {
    const generated = rules.filter(
      (r) =>
        parts(r.selector).some((part) => /(?:\ba|\bbutton|\.button-\w+|\.text-link)(?:\.[\w-]+)*(?:\[[^\]]*\])*::?(?:before|after)$/.test(part)) &&
        /(?:^|;)content:(?!(?:''|"")(?:;|$))/.test(r.declarations),
    );
    expect(rules.some((r) => /::(?:before|after)/.test(r.selector))).toBe(true); // there are pseudo-elements to look at
    expect(generated.map((r) => r.selector)).toEqual([]);
  });
});

describe('the illustration under reduced motion', () => {
  const reduced = rules.filter((r) => r.media === '(prefers-reduced-motion:reduce)');

  it('stops every animation, and the scroll animation', () => {
    expect(reduced.some((r) => r.selector.startsWith('*') && /animation:none!important/.test(r.declarations))).toBe(true);
    expect(reduced.some((r) => r.selector === 'html' && /scroll-behavior:auto/.test(r.declarations))).toBe(true);
  });

  it('keeps the contact ring: nothing in the reduced-motion block hides it, so CONTACT still points at a mark', () => {
    expect(reduced.filter((r) => /trace-contact/.test(r.selector) && /display:none|visibility:hidden|opacity:0/.test(r.declarations))).toEqual([]);
    expect(rules.some((r) => hasClass(r.selector, 'trace-contact-pulse'))).toBe(true);
  });

  it('draws the paths in full when the animation is off: the draw-on starts from the finished state', () => {
    // with `animation:none` the paths fall back to their own stroke-dashoffset, which must not be a hidden state
    const paths = rules.filter((r) => /\.trace-path/.test(r.selector) && /stroke-dashoffset/.test(r.declarations));
    expect(paths.map((r) => r.selector)).toEqual([]);
  });
});

/*
 * The Method page's limits and its denial are never hidden at any width: the five limits are the page's statement of
 * what the output does not mean, and the line under the heading says it is not a collision-probability model. A
 * stylesheet that collapsed or hid them on a phone would turn the page back into the claim without its limits.
 */
const LIMIT_CLASSES = ['limits-panel', 'method-lede'];
const hidesLimits = (all: ReturnType<typeof cssRules>) =>
  all
    .filter((r) => parts(r.selector).some((part) => LIMIT_CLASSES.some((name) => hasClass(part, name))))
    .filter((r) => /(?:^|;)(?:display:none|visibility:hidden|opacity:0|font-size:0|height:0|max-height:0|clip-path:inset\(50%\))(?:!important)?(?:;|$)/.test(r.declarations))
    .map((r) => `${r.selector} in ${r.media ?? 'no media block'}`);

describe('the limits stay visible', () => {
  it('no rule, in any media block, hides the limits panel, its heading or its list, or the denial line', () => {
    expect(rules.some((r) => parts(r.selector).some((part) => hasClass(part, 'limits-panel')))).toBe(true); // the rules exist
    expect(hidesLimits(rules)).toEqual([]);
  });

  it('the checker finds a rule that hides the limits list on a phone, and one that clips the heading', () => {
    expect(hidesLimits(cssRules(css + '\n@media(max-width:560px) { .limits-panel ul { display:none; } }'))).toEqual(['.limits-panel ul in (max-width:560px)']);
    expect(hidesLimits(cssRules(css + '\n.limits-panel h2 { height:0; }'))).toEqual(['.limits-panel h2 in no media block']);
  });
});

/*
 * No text that carries a caveat, a denial or the licence attribution is hidden at any width, in any rule: the Corpus
 * page's "these counts are not a partition" (.caption), its definitions (.stat-def) and notes (.note), the footer, the
 * Method limits, the list's note. The one exception is the keyboard hint, which is an instruction for people with a
 * keyboard and is dropped on a phone.
 */
const MAY_BE_HIDDEN = ['keyboard-hint'];
const hidesCaveats = (all: ReturnType<typeof cssRules>) =>
  all
    .filter((r) =>
      parts(r.selector).some((part) => Object.keys(CAVEAT_CLASSES).some((name) => !MAY_BE_HIDDEN.includes(name) && hasClass(part, name))),
    )
    .filter((r) => /(?:^|;)(?:display:none|visibility:hidden|opacity:0|font-size:0|height:0|max-height:0|clip-path:inset\(50%\))(?:!important)?(?:;|$)/.test(r.declarations))
    .map((r) => `${r.selector} in ${r.media ?? 'no media block'}`);

describe('caveat text is never hidden', () => {
  it('no rule, in any media block, hides a caveat class, bar the keyboard hint', () => {
    expect(hidesCaveats(rules)).toEqual([]);
  });

  it('the checker finds a rule that hides the Corpus caption on a phone, and ignores the keyboard hint', () => {
    expect(hidesCaveats(cssRules(css + '\n@media(max-width:560px) { .caption { display:none; } }'))).toEqual(['.caption in (max-width:560px)']);
    expect(hidesCaveats(cssRules('.keyboard-hint { display:none; } .stat .stat-def { display:none; }'))).toEqual(['.stat .stat-def in no media block']);
  });
});

/*
 * Links that look like buttons are square, like the buttons: one rule gives both the same border, radius, colours and
 * padding, and nothing rounds a .button-link. (The only ones are the "Back to scenarios" links on the pages for an
 * unknown address and for an error.)
 */
describe('button links match the buttons', () => {
  it('one rule styles button and .button-link together, with a square corner', () => {
    const shared = rules.find((r) => !r.media && parts(r.selector).includes('button') && parts(r.selector).includes('.button-link') && /border-radius:0/.test(r.declarations));
    expect(shared, 'a rule for `button,.button-link` with border-radius:0').toBeDefined();
    expect(shared!.declarations).toMatch(/border:1pxsolidvar\(--border-strong\)/);
    expect(shared!.declarations).toMatch(/background:transparent/);
  });

  it('no rule gives a .button-link, or a button, a rounded corner', () => {
    const rounded = rules
      .filter((r) => parts(r.selector).some((part) => /(^|[\s,])(?:button|\.button-link)(?![\w-])/.test(part) && !/\.(?:segmented|scenario-lookup|playback-row|link-button)/.test(part)))
      .filter((r) => /border-radius:(?!0(?:;|px;))/.test(r.declarations))
      .map((r) => r.selector);
    expect(rounded).toEqual([]);
  });

  it('hovering a .button-link does what hovering a button does', () => {
    const hover = rules.find((r) => parts(r.selector).includes('.button-link:hover'));
    expect(hover, 'a hover rule for .button-link').toBeDefined();
    expect(hover!.declarations).toMatch(/border-color:var\(--accent\)/);
    expect(hover!.declarations).toMatch(/(?:^|;)color:var\(--accent\)/);
  });
});

describe('the PostGIS value on the Corpus page', () => {
  it('is right-aligned, like the other values', () => {
    const rule = rules.find((r) => r.selector === ".stat[data-stat='postgis'] > dd:not(.stat-def)");
    expect(rule).toBeDefined();
    expect(rule!.declarations).toMatch(/text-align:right/);
  });
});
