import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  formatWait,
  MAX_SEED,
  parseSeed,
  searchRequestsEnabled,
} from '../api/searchRequests';
import { makeDetail, makePage, makePerturbed, makeTrajectories } from '../test/fixtures';
import { readFrontendFile } from '../test/readFrontendFile';
import { renderApp } from '../test/renderApp';

/*
 * The request form (pages/detail/RequestSearchPanel.tsx, api/searchRequests.ts): a scenario id and a seed, filed through
 * POST /search-requests and run, if at all, offline by the owner at no stated time. OFF unless the site is built with VITE_SEARCH_REQUESTS=true.
 *
 * Off (the default, and the production build): nothing renders, nothing is requested, and the sentences the site already
 * says about stored results, offline searches and a read-only API are still on the pages they are on. On: the form says
 * plainly that the request is filed, that the owner may run it offline with no time promised and that nothing runs on request, sends exactly one scenario id and
 * one seed, and says in words what each answer of the API means.
 */

const ID = 'syn-0001';

interface Sent {
  method: string;
  path: string;
  contentType: string | null;
  body: unknown;
}

/** A fetch stub that sees the method and the body, which the shared stub does not: GETs answer a scenario, a POST is `answer`. */
function serve(answer: () => Response | Promise<Response> = () => json(201, filed())) {
  const sent: Sent[] = [];
  vi.stubGlobal('fetch', async (input: Request) => {
    const url = new URL(input.url);
    const text = input.method === 'GET' ? '' : await input.clone().text();
    sent.push({
      method: input.method,
      path: url.pathname,
      contentType: input.headers.get('content-type'),
      body: text === '' ? undefined : JSON.parse(text),
    });
    if (url.pathname === '/search-requests') return answer();
    if (url.pathname.endsWith('/perturbed')) return json(200, makePerturbed({ scenario_id: ID }));
    if (url.pathname.endsWith('/trajectories')) return json(200, makeTrajectories([], ID));
    if (url.pathname.startsWith('/scenarios/')) return json(200, makeDetail({ scenario_id: ID }));
    return json(200, makePage([], null));
  });
  return sent;
}

const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json', ...headers } });
const filed = (over: Record<string, unknown> = {}) => ({
  request_id: '0b0f4b3e-6a61-4e55-8d3b-8c3d6d2a9c11', scenario_id: ID, seed: 7, state: 'filed', ...over,
});
const posts = (sent: Sent[]) => sent.filter((s) => s.method !== 'GET');

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

function enable() {
  vi.stubEnv('VITE_SEARCH_REQUESTS', 'true');
}

async function open() {
  renderApp(`/scenarios/${ID}`);
  await screen.findByRole('heading', { name: 'Run details' });
}

const panel = () => screen.getByRole('region', { name: 'Request a search' });
const inPanel = () => within(panel());
const inPanelAsync = async (role: 'status' | 'alert') =>
  within(await screen.findByRole('region', { name: 'Request a search' })).findByRole(role);
const seedField = () => screen.getByRole('textbox', { name: 'Seed' });
const fileButton = () => screen.getByRole('button', { name: /File request|Filing/ });

async function fileSeed(text: string) {
  const user = userEvent.setup();
  await user.clear(seedField());
  await user.type(seedField(), text);
  await user.click(fileButton());
}

// ── off ─────────────────────────────────────────────────────────────────────────────────────

describe('the switch', () => {
  it('is off when the variable is not set, which is what every build without it gets', () => {
    expect(import.meta.env.VITE_SEARCH_REQUESTS).toBeUndefined();
    expect(searchRequestsEnabled()).toBe(false);
  });

  it.each(['', '1', 'TRUE', 'True', 'yes', 'on', ' true', 'true ', 'false', '0'])('is off for %j: only exactly "true" turns it on', (value) => {
    vi.stubEnv('VITE_SEARCH_REQUESTS', value);
    expect(searchRequestsEnabled()).toBe(false);
  });

  it('is on for exactly "true"', () => {
    vi.stubEnv('VITE_SEARCH_REQUESTS', 'true');
    expect(searchRequestsEnabled()).toBe(true);
  });

  it('is not set by any committed env file or by the deploy configuration', () => {
    for (const file of ['.env.example', 'vercel.json', 'package.json', 'vite.config.ts']) {
      const text = readFrontendFile(file);
      const assignments = text.split('\n').filter((line) => /VITE_SEARCH_REQUESTS/.test(line) && !/^\s*#/.test(line));
      expect(assignments, file).toEqual([]);
    }
    expect(readFrontendFile('.env.example')).toMatch(/^# VITE_SEARCH_REQUESTS=true$/m); // documented, commented out
  });
});

describe('with the switch off', () => {
  it('renders no form, no heading and no text about requests on a scenario page', async () => {
    const sent = serve();
    await open();
    expect(screen.queryByRole('heading', { name: 'Request a search' })).not.toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: 'Seed' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /File request/ })).not.toBeInTheDocument();
    expect(document.querySelector('.search-request')).toBeNull();
    expect(document.body.textContent).not.toMatch(/Request a search|File request|filed|seed/i);
    expect(sent.some((s) => s.path === '/search-requests')).toBe(false);
    expect(posts(sent)).toEqual([]);
  });

  it('makes only GET requests anywhere in the app', async () => {
    const sent = serve();
    await open();
    await userEvent.click(screen.getByRole('link', { name: /All scenarios/ }));
    await screen.findByRole('heading', { level: 1 });
    expect(sent.every((s) => s.method === 'GET')).toBe(true);
  });

  it('leaves the four statements about stored results, offline searches and a read-only API true on their pages', async () => {
    serve();
    renderApp('/');
    expect(await screen.findByText(/Searches ran offline; this site shows stored results\./)).toBeInTheDocument();
    expect(document.body.textContent).toContain('PostGIS, read-only API');
    vi.unstubAllGlobals();
    serve();
    renderApp('/method');
    expect(await screen.findByText(/Export geometry once, keep the production API read-only/)).toBeInTheDocument();
  });
});

// ── on ──────────────────────────────────────────────────────────────────────────────────────

describe('with the switch on', () => {
  it('shows the form, and says plainly that the request is filed, the owner may run it with no time promised, and nothing runs on request', async () => {
    enable();
    const sent = serve();
    await open();
    const panel = screen.getByRole('region', { name: 'Request a search' });
    expect(within(panel).getByRole('heading', { level: 2, name: 'Request a search' })).toBeInTheDocument();
    expect(panel).toHaveTextContent(
      'Ask for this scenario to be searched with a seed of your choice. The request is filed, and the owner may run it offline; ' +
        'no time is promised. Nothing runs when you press the button. The results on this page do not change when it is run.',
    );
    expect(seedField()).toHaveValue('');
    expect(fileButton()).toBeDisabled();
    expect(posts(sent)).toEqual([]); // opening the page files nothing
  });

  it('sends exactly the scenario of the page and the seed typed, as one JSON body, and nothing else', async () => {
    enable();
    const sent = serve();
    await open();
    await fileSeed('7');
    await within(await screen.findByRole('region', { name: 'Request a search' })).findByText(/Nothing has run/);
    expect(posts(sent)).toEqual([{ method: 'POST', path: '/search-requests', contentType: 'application/json', body: { scenario_id: ID, seed: 7 } }]);
  });

  it('says it is filed, names the request, and says nothing has run', async () => {
    enable();
    serve();
    await open();
    await fileSeed('7');
    const status = await within(await screen.findByRole('region', { name: 'Request a search' })).findByRole('status');
    expect(status).toHaveTextContent(
      'Filed: request 0b0f4b3e-6a61-4e55-8d3b-8c3d6d2a9c11 for seed 7. It is queued: the owner may run it offline, at no stated time, ' +
        'and this page does not update. Nothing has run.',
    );
  });

  it('says it was already filed when the API returns the first request for the same scenario and seed', async () => {
    enable();
    serve(() => json(200, filed({ state: 'existing', seed: 7 })));
    await open();
    await fileSeed('7');
    expect(await inPanelAsync('status')).toHaveTextContent(
      'Already filed: request 0b0f4b3e-6a61-4e55-8d3b-8c3d6d2a9c11 for seed 7. It is queued: the owner may run it offline, at no stated time, ' +
        'and this page does not update. Nothing has run.',
    );
  });

  it('accepts the smallest and the largest seed', async () => {
    enable();
    const sent = serve();
    await open();
    await fileSeed('0');
    await inPanelAsync('status');
    await fileSeed(String(MAX_SEED));
    await waitFor(() => expect(posts(sent)).toHaveLength(2));
    expect(posts(sent).map((s) => (s.body as { seed: number }).seed)).toEqual([0, MAX_SEED]);
  });

  it.each(['-1', '1.5', 'abc', '2147483648', '1e3', '0x10', '+5', '99999999999', '١٢٣'])('refuses the seed %j in the page, and sends nothing', async (text) => {
    enable();
    const sent = serve();
    await open();
    await fileSeed(text);
    expect(await inPanelAsync('alert')).toHaveTextContent(`The seed must be a whole number from 0 to ${MAX_SEED}.`);
    expect(seedField()).toHaveAttribute('aria-invalid', 'true');
    expect(posts(sent)).toEqual([]);
  });

  it('takes the invalid mark and its message away when the seed is changed', async () => {
    enable();
    serve();
    await open();
    await fileSeed('abc');
    await inPanelAsync('alert');
    expect(seedField()).toHaveAttribute('aria-invalid', 'true');
    await userEvent.type(seedField(), '1');
    expect(seedField()).toHaveAttribute('aria-invalid', 'false');
    expect(inPanel().queryByRole('alert')).not.toBeInTheDocument();
    expect(seedField()).not.toHaveAttribute('aria-describedby');
  });

  it('clears the message when the seed is changed', async () => {
    enable();
    serve();
    await open();
    await fileSeed('7');
    await inPanelAsync('status');
    await userEvent.type(seedField(), '1');
    expect(inPanel().queryByRole('status')).not.toBeInTheDocument();
    expect(seedField()).toHaveValue('71');
  });

  it('does not file twice when the button is pressed twice while the first is on its way', async () => {
    enable();
    let release: (r: Response) => void = () => undefined;
    const sent = serve(() => new Promise<Response>((resolve) => { release = resolve; }));
    await open();
    const user = userEvent.setup();
    await user.type(seedField(), '7');
    await user.click(fileButton());
    await screen.findByRole('button', { name: 'Filing…' });
    expect(screen.getByRole('button', { name: 'Filing…' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Filing…' }));
    await user.type(seedField(), '{Enter}');
    // A form can be submitted without the button (a script, an assistive technology): the page itself holds the second off.
    fireEvent.submit(document.querySelector('form.search-request-form')!);
    expect(posts(sent)).toHaveLength(1);
    release(json(201, filed()));
    await within(await screen.findByRole('region', { name: 'Request a search' })).findByText(/Nothing has run/);
    expect(posts(sent)).toHaveLength(1);
  });

  it.each([
    ['404', 404, { detail: 'Unknown scenario: syn-0001' }, {}, 'This scenario is not in the stored results, so nothing was filed.'],
    ['422', 422, { detail: [] }, {}, `The request was not accepted, so nothing was filed. The seed must be a whole number from 0 to ${MAX_SEED}.`],
    ['429 with an hour', 429, { detail: 'x' }, { 'Retry-After': '3600' }, 'You have reached a limit on requests, so nothing was filed. Try again in about an hour.'],
    ['429 with minutes', 429, { detail: 'x' }, { 'Retry-After': '600' }, 'You have reached a limit on requests, so nothing was filed. Try again in about 10 minutes.'],
    ['429 with hours', 429, { detail: 'x' }, { 'Retry-After': '28800' }, 'You have reached a limit on requests, so nothing was filed. Try again in about 8 hours.'],
    ['429 with no time', 429, { detail: 'x' }, {}, 'You have reached a limit on requests, so nothing was filed. Try again later.'],
    ['503 with seconds', 503, { detail: 'x' }, { 'Retry-After': '1' }, 'Requests cannot be filed right now, so nothing was filed. Try again in about a minute.'],
    ['503 with an hour', 503, { detail: 'x' }, { 'Retry-After': '3600' }, 'Requests cannot be filed right now, so nothing was filed. Try again in about an hour.'],
  ])('says in words what a %s means, and that nothing was filed', async (_name, status, body, headers, expected) => {
    enable();
    serve(() => json(status, body, headers));
    await open();
    await fileSeed('7');
    expect(await inPanelAsync('alert')).toHaveTextContent(expected);
    expect(inPanel().queryByRole('status')).not.toBeInTheDocument();
  });

  it('does not claim nothing was filed after an error that may have come after the request was stored', async () => {
    enable();
    serve(() => json(500, { detail: 'boom' }));
    await open();
    await fileSeed('7');
    const text = (await inPanelAsync('alert')).textContent ?? '';
    expect(text).toMatch(/500: boom/);
    expect(text).toMatch(/Whether the request was filed is not known\. Pressing the button again is harmless: the same scenario and seed is one request\.$/);
    expect(text).not.toMatch(/nothing was filed/i);
  });

  it('says the same after a network failure', async () => {
    enable();
    vi.stubGlobal('fetch', async (input: Request) => {
      const path = new URL(input.url).pathname;
      if (path === '/search-requests') throw new TypeError('network down');
      if (path.endsWith('/perturbed')) return json(200, makePerturbed({ scenario_id: ID }));
      if (path.endsWith('/trajectories')) return json(200, makeTrajectories([], ID));
      return json(200, makeDetail({ scenario_id: ID }));
    });
    await open();
    await fileSeed('8');
    const text = (await inPanelAsync('alert')).textContent ?? '';
    expect(text).toMatch(/^Could not reach the API at .*\. Whether the request was filed is not known\./);
    expect(text).not.toMatch(/nothing was filed/i);
  });

  it.each([
    ['a state it does not know', { request_id: 'r', scenario_id: ID, seed: 7, state: 'queued' }],
    ['no state', { request_id: 'r', scenario_id: ID, seed: 7 }],
    ['a seed that is not a whole number', { request_id: 'r', scenario_id: ID, seed: 7.5, state: 'filed' }],
    ['no request id', { scenario_id: ID, seed: 7, state: 'filed' }],
  ])('does not take %s for a filed request', async (_name, body) => {
    enable();
    serve(() => json(201, body));
    await open();
    await fileSeed('7');
    const text = (await inPanelAsync('alert')).textContent ?? '';
    expect(text).toMatch(/does not understand\. Whether the request was filed is not known\./);
    expect(inPanel().queryByRole('status')).not.toBeInTheDocument();
  });

  it('says the same when the API answers with something that is not a filed request', async () => {
    enable();
    serve(() => json(201, { ok: true }));
    await open();
    await fileSeed('7');
    const text = (await inPanelAsync('alert')).textContent ?? '';
    expect(text).toMatch(/^The API answered with something this page does not understand\. Whether the request was filed is not known\./);
    expect(text).not.toMatch(/nothing was filed/i);
  });

  it('lets a later press that works replace a refusal', async () => {
    enable();
    let n = 0;
    serve(() => (++n === 1 ? json(503, { detail: 'x' }, { 'Retry-After': '1' }) : json(201, filed())));
    await open();
    await fileSeed('7');
    await inPanelAsync('alert');
    await userEvent.click(fileButton());
    await inPanelAsync('status');
    expect(inPanel().queryByRole('alert')).not.toBeInTheDocument();
  });

  it('is on the scenario page only, and files for the scenario on that page', async () => {
    enable();
    serve();
    renderApp('/walkthrough');
    await screen.findByRole('heading', { level: 1, name: 'Three stored results' });
    expect(screen.queryByRole('heading', { name: 'Request a search' })).not.toBeInTheDocument();
    expect(document.querySelector('.search-request')).toBeNull();
  });
});

// ── no promise of a run or a time ─────────────────────────────────────────────────────────

/*
 * Nothing in the repository runs a request: the owner reads what is waiting and decides. So the form must never say or imply
 * that a run will happen, or when. The panel's own paragraph and the line shown after a request is filed (new or existing) are
 * checked against phrases that promise a run, a time or progress, and against the words that make the opposite claim, which
 * must be there. The retry waits in refusals ("Try again in about an hour") are about the visitor trying again, not about a
 * run, and are not part of this.
 */
const PROMISES: [string, RegExp][] = [
  ['will be run / will run / will be searched', /\bwill\b/i],
  ['waiting to be run', /waiting to be run/i],
  ['is being run, or is run (as a fact rather than a condition: "when it is run" is allowed)', /\bis being (run|processed|searched)\b|(?<!\b(when|if|once) )\bit is (run|processed|searched)\b/i],
  ['soon, shortly, promptly, quickly', /\b(soon|shortly|promptly|quickly|asap|imminently)\b/i],
  ['later, tomorrow, today, tonight, next', /\b(later|tomorrow|today|tonight|next (day|week))\b/i],
  ['a duration or a deadline', /\b(within|by)\b.*\b(hour|hours|day|days|minute|minutes|week|weeks)\b|\b\d+\s*(hour|hours|day|days|minute|minutes|week|weeks)\b/i],
  ['a place in a queue or progress', /\b(position|ahead of|in progress|processing|working on|estimated|eta)\b/i],
  ['you will be told / notified / emailed', /\b(notify|notified|email|emailed|told|results (will )?appear)\b/i],
];

describe('the form does not promise a run or a time', () => {
  function violations(text: string) {
    return PROMISES.filter(([, pattern]) => pattern.test(text)).map(([name]) => name);
  }

  it('in its paragraph', async () => {
    enable();
    serve();
    await open();
    const text = panel().querySelector('p.note')!.textContent!;
    expect(violations(text)).toEqual([]);
    expect(text).toMatch(/\bmay run it offline\b/);
    expect(text).toMatch(/no time is promised/);
    expect(text).toMatch(/Nothing runs when you press the button\./);
  });

  it.each([
    ['a request that is filed', 201, 'filed'],
    ['a request that was filed before', 200, 'existing'],
  ])('in the line shown for %s', async (_name, status, state) => {
    enable();
    serve(() => json(status, filed({ state })));
    await open();
    await fileSeed('7');
    const text = (await inPanelAsync('status')).textContent!;
    expect(violations(text)).toEqual([]);
    expect(text).toMatch(/It is queued: the owner may run it offline, at no stated time, and this page does not update\./);
    expect(text).toMatch(/Nothing has run\.$/);
  });

  // One sentence per pattern, each flagged by that pattern alone, so that weakening any one pattern fails here.
  it.each(PROMISES.map(([name], index) => [name, [
    'The owner will run it.',
    'It is waiting to be run offline.',
    'It is being run.',
    'Expect it soon.',
    'The owner may run it later.',
    'Expect it in 2 days.',
    'You are in position 3.',
    'You are notified by email.',
  ][index]!]))('flags %s', (name, sentence) => {
    expect(violations(sentence), sentence).toEqual([name]);
  });

  it('flags "it is run" as a fact, whichever way round it is said', () => {
    for (const bad of ['It is run every night.', 'Once queued it is run in order.', 'It is processed by the owner.', 'Done within a day.', 'Done by the end of the week.']) {
      expect(violations(bad), bad).not.toEqual([]);
    }
  });

  it('and passes the wording the form uses, and a condition ("when it is run")', () => {
    expect(violations('It is queued: the owner may run it offline, at no stated time, and this page does not update.')).toEqual([]);
    expect(violations('The results on this page do not change when it is run.')).toEqual([]);
    expect(violations('The results on this page do not change if it is run.')).toEqual([]);
  });
});

// ── the helpers ─────────────────────────────────────────────────────────────────────────────

describe('parseSeed', () => {
  it('reads a whole number from 0 to the largest and nothing else', () => {
    expect([parseSeed('0'), parseSeed('7'), parseSeed('2147483647'), parseSeed('0007')]).toEqual([0, 7, MAX_SEED, 7]);
    for (const bad of ['', ' ', '-0', '-1', '1.0', '2147483648', '12345678901', 'NaN', 'Infinity', '1_000', '٣']) {
      expect(parseSeed(bad), bad).toBeNull();
    }
  });
});

describe('formatWait', () => {
  it('puts seconds into words', () => {
    expect(formatWait(null)).toBe('later');
    expect(formatWait(1)).toBe('about a minute');
    expect(formatWait(90)).toBe('about a minute');
    expect(formatWait(91)).toBe('about 2 minutes');
    expect(formatWait(600)).toBe('about 10 minutes');
    expect(formatWait(3600)).toBe('about an hour');
    expect(formatWait(3239)).toBe('about 54 minutes');
    expect(formatWait(3240)).toBe('about an hour');
    expect(formatWait(5399)).toBe('about an hour');
    expect(formatWait(5400)).toBe('about 2 hours');
    expect(formatWait(86400)).toBe('about 24 hours');
  });
});
