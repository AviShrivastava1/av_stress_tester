import { screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { ScenarioDetail } from '../api/client';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { makeDetail, makePage, makePerturbed, makeRow, makeTrajectories } from '../test/fixtures';
import { renderApp } from '../test/renderApp';

/*
 * The detail page's replay-offset block: two distances, in metres, taken from the
 * stored search result, shown under "Min perturbation".
 *
 *   Baseline offset at collision   how far the zero-perturbation replay of the
 *                                  challenger is from its logged position at the
 *                                  collision frame
 *   Replay drift, whole track (max)  the largest such distance over the whole track
 *
 * They qualify the result; they are not an error bar, a margin or a safety claim, and
 * they are not comparable with Min perturbation (a weighted norm over mixed units), so
 * nothing here may divide or compare one by the other.
 */

const OFFSET = 'Baseline offset at collision';
const DRIFT = 'Replay drift, whole track (max)';
const ok = (body: unknown): Reply => ({ status: 200, body });

function open(overrides: Partial<ScenarioDetail>) {
  stubFetch((req: RecordedRequest) => {
    if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
    if (req.path.endsWith('/perturbed')) return ok(makePerturbed());
    return ok(makeDetail(overrides));
  });
  renderApp('/scenarios/synthetic-0001');
}

const FOUND: Partial<ScenarioDetail> = {
  stress_tested: true, stress_attempted: true, stress_outcome: 'collision_found',
  collision_timestep: 55, min_perturbation: 0.022,
};
const NONE_FOUND: Partial<ScenarioDetail> = {
  stress_tested: true, stress_attempted: true, stress_outcome: 'no_collision_found',
  no_collision_found: true, collision_timestep: null, min_perturbation: null,
};
// A row written before stress_outcome existed: a stored result, outcome unknown.
const LEGACY: Partial<ScenarioDetail> = {
  stress_tested: true, stress_attempted: true, stress_outcome: null,
  collision_timestep: null, min_perturbation: null,
};

const block = () => screen.findByTestId('replay-offset');
const valueOf = (container: HTMLElement, label: string) =>
  within(container).getByText(label).nextElementSibling as HTMLElement;

describe('replay offset: values', () => {
  it('shows both distances in metres, the offset first', async () => {
    open({ ...FOUND, baseline_offset_at_collision: 0.4076, baseline_replay_error: 0.428 });
    const container = await block();
    expect(valueOf(container, OFFSET)).toHaveTextContent('0.408 m');
    expect(valueOf(container, DRIFT)).toHaveTextContent('0.428 m');
    const [offset, drift] = [within(container).getByText(OFFSET), within(container).getByText(DRIFT)];
    expect(offset.compareDocumentPosition(drift) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('keeps the exact stored value as a tooltip', async () => {
    open({ ...FOUND, baseline_offset_at_collision: 0.6810235381126404, baseline_replay_error: 1.3323498964309692 });
    const container = await block();
    expect(within(valueOf(container, OFFSET)).getByText('0.681 m')).toHaveAttribute('title', '0.6810235381126404');
    expect(within(valueOf(container, DRIFT)).getByText('1.332 m')).toHaveAttribute('title', '1.3323498964309692');
  });

  it('shows an exact zero as zero', async () => {
    open({ ...FOUND, baseline_offset_at_collision: 0, baseline_replay_error: 0 });
    const container = await block();
    expect(valueOf(container, OFFSET)).toHaveTextContent('0.000 m');
    expect(valueOf(container, DRIFT)).toHaveTextContent('0.000 m');
  });

  it('keeps a tiny nonzero distance nonzero', async () => {
    open({ ...FOUND, baseline_offset_at_collision: 3.77e-7, baseline_replay_error: 3.77e-7 });
    const container = await block();
    expect(valueOf(container, OFFSET)).toHaveTextContent('3.77e-7 m');
  });
});

describe('replay offset: the three states', () => {
  it('shows nothing for a scenario with no stored result', async () => {
    open({ stress_tested: false, stress_attempted: false, stress_outcome: null,
           baseline_offset_at_collision: null, baseline_replay_error: null });
    await screen.findByText('Min perturbation');
    expect(screen.queryByTestId('replay-offset')).not.toBeInTheDocument();
    expect(screen.queryByText(OFFSET)).not.toBeInTheDocument();
  });

  it('shows nothing, and never says "Not recorded", for a scenario whose only attempt was refused', async () => {
    open({ stress_tested: false, stress_attempted: true, stress_outcome: null,
           last_attempt_outcome: 'replay_infeasible', collision_timestep: null, min_perturbation: null,
           baseline_offset_at_collision: null, baseline_replay_error: null });
    await screen.findByText('Min perturbation');
    expect(screen.queryByTestId('replay-offset')).not.toBeInTheDocument();
    expect(screen.queryByText(/Not recorded for this result/)).not.toBeInTheDocument();
    expect(screen.queryByText(/no collision found/i)).not.toBeInTheDocument();
  });

  it('says "Not recorded for this result", not "None", for a result with an unknown outcome and a perturbation but no figures', async () => {
    open({ stress_tested: true, stress_attempted: true, stress_outcome: null,
           collision_timestep: null, min_perturbation: 0.022,
           baseline_offset_at_collision: null, baseline_replay_error: null });
    const container = await block();
    for (const label of [OFFSET, DRIFT]) {
      expect(valueOf(container, label)).toHaveTextContent('Not recorded for this result');
    }
    expect(container).not.toHaveTextContent('None, no collision found');
  });

  it('says "None, no collision found" for the offset of a completed search that found none', async () => {
    open({ ...NONE_FOUND, baseline_offset_at_collision: null, baseline_replay_error: 0.31 });
    const container = await block();
    expect(valueOf(container, OFFSET)).toHaveTextContent('None, no collision found');
    expect(valueOf(container, OFFSET)).not.toHaveTextContent('Not recorded');
    expect(valueOf(container, DRIFT)).toHaveTextContent('0.310 m');
  });

  it('says "Not recorded for this result" for a missing drift on a no-collision result', async () => {
    open({ ...NONE_FOUND, baseline_offset_at_collision: null, baseline_replay_error: null });
    const container = await block();
    expect(valueOf(container, OFFSET)).toHaveTextContent('None, no collision found');
    expect(valueOf(container, DRIFT)).toHaveTextContent('Not recorded for this result');
  });

  it('says "Not recorded for this result" when a found collision has no stored figures', async () => {
    open({ ...FOUND, baseline_offset_at_collision: null, baseline_replay_error: null });
    const container = await block();
    for (const label of [OFFSET, DRIFT]) {
      expect(valueOf(container, label)).toHaveTextContent('Not recorded for this result');
      expect(valueOf(container, label)).not.toHaveTextContent('no collision found');
    }
  });

  it('says "Not recorded for this result" for a stored result of unknown outcome', async () => {
    open({ ...LEGACY, baseline_offset_at_collision: null, baseline_replay_error: null });
    const container = await block();
    for (const label of [OFFSET, DRIFT]) {
      expect(valueOf(container, label)).toHaveTextContent('Not recorded for this result');
    }
  });

  it('shows a stored offset and says "Not recorded for this result" only for a missing drift', async () => {
    open({ ...FOUND, baseline_offset_at_collision: 0.4076, baseline_replay_error: null });
    const container = await block();
    expect(valueOf(container, OFFSET)).toHaveTextContent('0.408 m');
    expect(valueOf(container, OFFSET)).not.toHaveTextContent('Not recorded');
    expect(valueOf(container, DRIFT)).toHaveTextContent('Not recorded for this result');
    expect(valueOf(container, DRIFT)).not.toHaveTextContent(/\bm\b/);
  });

  it('shows a stored drift and says "Not recorded for this result" only for a missing offset', async () => {
    open({ ...FOUND, baseline_offset_at_collision: null, baseline_replay_error: 0.428 });
    const container = await block();
    expect(valueOf(container, DRIFT)).toHaveTextContent('0.428 m');
    expect(valueOf(container, DRIFT)).not.toHaveTextContent('Not recorded');
    expect(valueOf(container, OFFSET)).toHaveTextContent('Not recorded for this result');
    expect(valueOf(container, OFFSET)).not.toHaveTextContent(/\bm\b/);
  });

  it('never turns a missing figure into a zero', async () => {
    open({ ...FOUND, baseline_offset_at_collision: null, baseline_replay_error: null });
    const container = await block();
    expect(container.textContent).not.toMatch(/\b0(\.0+)?\s*m\b/);
  });
});

describe('replay offset: wording', () => {
  const full = { ...FOUND, baseline_offset_at_collision: 0.4076, baseline_replay_error: 0.428 };

  it('describes the offset as a distance at the collision frame, the drift as the whole-track maximum', async () => {
    open(full);
    const text = (await block()).textContent ?? '';
    expect(text).toMatch(/at the collision frame/i);
    expect(text).toMatch(/zero-perturbation replay/i);
    expect(text).toMatch(/logged position/i);
    expect(text).toMatch(/whole track/i);
    expect(text).toMatch(/metres/i);
  });

  it('says neither figure is comparable with Min perturbation', async () => {
    open(full);
    const text = (await block()).textContent ?? '';
    expect(text).toMatch(/weighted norm/i);
    expect(text).toMatch(/mixed units/i);
    expect(text).toMatch(/Min perturbation/);
  });

  it.each([
    'error bar', 'margin', 'uncertainty', 'safe', 'safety', 'caused', 'cause', 'causes',
  ])('does not use the word "%s"', async (word) => {
    open(full);
    const text = (await block()).textContent ?? '';
    expect(text).not.toMatch(new RegExp(`\\b${word}\\b`, 'i'));
  });

  // textContent skips tooltips and accessible names, which are read out and shown on hover.
  function namesAndTooltips(container: HTMLElement): string[] {
    const texts: string[] = [];
    for (const el of [container, ...Array.from(container.querySelectorAll<HTMLElement>('*'))]) {
      for (const attr of ['title', 'aria-label', 'aria-description', 'alt']) {
        const value = el.getAttribute(attr);
        if (value) texts.push(value);
      }
      for (const id of (el.getAttribute('aria-labelledby') ?? '').split(/\s+/).filter(Boolean)) {
        const text = document.getElementById(id)?.textContent;
        if (text) texts.push(text);
      }
    }
    return texts;
  }

  it.each([
    ['both figures stored', { baseline_offset_at_collision: 0.4076, baseline_replay_error: 0.428 }],
    ['neither stored', { baseline_offset_at_collision: null, baseline_replay_error: null }],
  ])('uses none of the banned words in any tooltip or accessible name (%s)', async (_name, figures) => {
    open({ ...FOUND, ...figures });
    const texts = namesAndTooltips(await block());
    for (const word of ['error bar', 'margin', 'uncertainty', 'safe', 'safety', 'cause', 'caused', 'causes']) {
      for (const text of texts) {
        expect(text, `"${word}" in a tooltip or name`).not.toMatch(new RegExp(`\\b${word}\\b`, 'i'));
      }
    }
  });

  it('puts the exact stored value in the tooltip of each figure, so the check above has something to read', async () => {
    open({ ...FOUND, baseline_offset_at_collision: 0.4076, baseline_replay_error: 0.428 });
    const texts = namesAndTooltips(await block());
    expect(texts).toContain('0.4076');
    expect(texts).toContain('0.428');
  });

  it('does not relate the figures to Min perturbation numerically', async () => {
    // 0.4 / 0.004 = 100; 0.428 / 0.004 = 107. Neither may appear, nor a ratio mark.
    open({ ...FOUND, min_perturbation: 0.004, baseline_offset_at_collision: 0.4, baseline_replay_error: 0.428 });
    const text = (await block()).textContent ?? '';
    expect(text).not.toMatch(/\b100\b|\b107\b|×|%|\btimes\b|\bratio\b/i);
  });
});

describe('replay offset: no derived numbers', () => {
  // Added after a mutant that appended "(100x)" to the offset got past the ratio test above,
  // whose word boundaries do not match a number followed by a letter. Whatever a ratio looks
  // like, it is a number that is not one of the two stored figures, so the block may contain
  // exactly those numbers and no others.
  it('contains no number other than the two stored figures', async () => {
    open({ ...FOUND, min_perturbation: 0.004, baseline_offset_at_collision: 0.4, baseline_replay_error: 0.428 });
    const text = (await block()).textContent ?? '';
    expect(text.match(/\d+(?:\.\d+)?(?:e-?\d+)?/gi)).toEqual(['0.400', '0.428']);
  });

  it('contains no number at all when neither figure is stored', async () => {
    open({ ...FOUND, min_perturbation: 0.004, baseline_offset_at_collision: null, baseline_replay_error: null });
    const text = (await block()).textContent ?? '';
    expect(text.match(/\d/g)).toBeNull();
  });
});

describe('replay offset: elsewhere', () => {
  it('does not appear on the list page', async () => {
    stubFetch(() => ok(makePage([makeRow({ stress_tested: true, min_perturbation: 0.022 })], null)));
    renderApp('/');
    await screen.findByRole('link', { name: 'synthetic-0001' });
    expect(screen.queryByText(OFFSET)).not.toBeInTheDocument();
    expect(screen.queryByText(DRIFT)).not.toBeInTheDocument();
  });
});
