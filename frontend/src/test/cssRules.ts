export interface CssRule {
  selector: string;
  /** The declarations with all whitespace removed (`display:none;color:red;`). */
  declarations: string;
  /** The declarations with whitespace squeezed to single spaces, so shorthand tokens stay apart (`font:720 10px/1.5 mono;`). */
  text: string;
}

/**
 * The rules of a stylesheet, as text: `selector { declarations }` with whitespace squeezed (`declarations` has it removed; `text` keeps single spaces). A rule inside an @media block is returned on its own, without the media condition. Comments are dropped.
 * jsdom does not apply the project's stylesheet, so the tests that check styles check them this way.
 */
export function cssRules(text: string): CssRule[] {
  const stripped = text.replace(/\/\*[\s\S]*?\*\//g, '');
  // A rule is `selector { declarations }` with no braces inside, so a rule inside @media matches on its own.
  return [...stripped.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({
    selector: m[1]!.trim().replace(/\s+/g, ' '),
    declarations: m[2]!.replace(/\s+/g, ''),
    text: m[2]!.replace(/\s+/g, ' ').trim(),
  }));
}
