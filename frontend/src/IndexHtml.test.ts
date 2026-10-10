import { describe, expect, it } from 'vitest';
import { readFrontendFile } from './test/readFrontendFile';

/*
 * index.html: the title and the description a search result or a link preview shows. The description is copy, so it
 * is held to the same words as the pages (copyLint.test.tsx): none of live, safe, safety, end-to-end, adversarial,
 * transparent, auditable.
 */

const html = readFrontendFile('index.html');
const DESCRIPTION =
  'Replay recorded driving scenes and inspect the smallest model-constrained change a finite search found that produced contact.';
const metas = [...html.matchAll(/<meta\s+name="([^"]+)"\s+content="([^"]*)"\s*\/?>/g)].map((m) => ({ name: m[1]!, content: m[2]! }));

/** The description as the page serves it. */
const served = metas.find((m) => m.name === 'description')?.content ?? '';

describe('index.html', () => {
  it('has exactly one description, and it is the one the project can back', () => {
    const descriptions = metas.filter((m) => m.name === 'description');
    expect(descriptions).toEqual([{ name: 'description', content: DESCRIPTION }]);
  });

  it('says "found" and "finite search": the smallest change found, not a proven minimum', () => {
    expect(served).toMatch(/smallest[^.]*\bfound\b/);
    expect(served).toMatch(/finite search/);
    // the unqualified wording, which reads as a proven minimum
    expect(html).not.toMatch(/smallest model-constrained changes that produced contact/);
  });

  it('uses none of the words the project cannot back', () => {
    expect(served.length).toBeGreaterThan(40); // there is a description to lint
    const copy = [served, html.match(/<title>([^<]*)<\/title>/)![1]!].join(' ');
    expect(copy).not.toMatch(/\b(live|safe|safety|end-to-end|adversarial|transparent|auditable)\b/i);
  });

  it('keeps its title', () => {
    expect(html).toContain('<title>AV Scenario Stress-Tester</title>');
  });
});
