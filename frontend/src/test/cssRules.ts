export interface CssRule {
  selector: string;
  /** The declarations with all whitespace removed (`display:none;color:red;`). */
  declarations: string;
  /** The declarations with whitespace squeezed to single spaces, so shorthand tokens stay apart (`font:720 10px/1.5 mono;`). */
  text: string;
  /** The enclosing @media condition with whitespace removed (`(max-width:560px)`), or null if the rule is in no @media block. */
  media: string | null;
  /** Every enclosing conditional at-rule, outermost first, as written with whitespace squeezed (`@media(max-width:560px)`, `@supports (display:grid)`, `@layer base`). */
  conditions: string[];
}

/** At-rules whose block holds more rules: their rules are returned, tagged with the at-rule as a condition. */
const GROUP_AT_RULE = /^@(?:media|supports|layer|container|scope|starting-style)(?![\w-])/;
/** At-rules whose block holds keyframe selectors (`from`, `50%`), not rules about the page: the block is skipped. */
const KEYFRAMES_AT_RULE = /^@(?:-[a-z]+-)?keyframes(?![\w-])/;

/**
 * The rules of a stylesheet, as text: `selector { declarations }` with whitespace squeezed. A rule inside a conditional
 * at-rule (@media, @supports, @layer, @container, ...) is returned on its own with the at-rules around it in `conditions`
 * (and the @media ones in `media`). The contents of @keyframes are skipped, and nothing else is: a block of declarations
 * such as @font-face comes back as a rule whose selector is the at-rule, and a statement such as `@import url(x);` or
 * `@layer a, b;` is dropped without touching the rule after it. Comments are dropped. jsdom does not apply the
 * project's stylesheet, so the tests that check styles check them this way. Rules nested inside rules (CSS nesting) are
 * not supported; the real-stylesheet test in cssRules.test.ts would notice one.
 */
export function cssRules(text: string): CssRule[] {
  const src = text.replace(/\/\*[\s\S]*?\*\//g, '');
  const rules: CssRule[] = [];
  const open: string[] = []; // the enclosing group at-rules, outermost first
  let prelude = '';
  let i = 0;
  while (i < src.length) {
    const ch = src[i]!;
    if (ch === '{') {
      const selector = prelude.trim().replace(/\s+/g, ' ');
      if (GROUP_AT_RULE.test(selector)) {
        open.push(selector);
        i += 1;
      } else if (KEYFRAMES_AT_RULE.test(selector)) {
        let depth = 1;
        i += 1;
        while (i < src.length && depth > 0) {
          if (src[i] === '{') depth += 1;
          else if (src[i] === '}') depth -= 1;
          i += 1;
        }
      } else {
        const end = src.indexOf('}', i + 1);
        const body = src.slice(i + 1, end < 0 ? src.length : end);
        const medias = open.filter((c) => c.startsWith('@media')).map((c) => c.slice('@media'.length).replace(/\s+/g, ''));
        rules.push({
          selector,
          declarations: body.replace(/\s+/g, ''),
          text: body.replace(/\s+/g, ' ').trim(),
          media: medias.length > 0 ? medias.join(' and ') : null,
          conditions: [...open],
        });
        i = end < 0 ? src.length : end + 1;
      }
      prelude = '';
    } else if (ch === '}') {
      open.pop();
      prelude = '';
      i += 1;
    } else if (ch === ';' && prelude.trim().startsWith('@')) {
      prelude = ''; // a statement at-rule: @import, @charset, @layer a, b;
      i += 1;
    } else {
      prelude += ch;
      i += 1;
    }
  }
  return rules;
}
