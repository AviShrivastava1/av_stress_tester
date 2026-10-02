import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createMemoryRouter, RouterProvider } from 'react-router';
import { describe, expect, it } from 'vitest';
import { routes } from '../App';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { makeDetail, makePage, makePerturbed, makeRow, makeTrajectories } from '../test/fixtures';

const ok = (body: unknown): Reply => ({ status: 200, body });

function server(req: RecordedRequest): Reply {
  if (req.path === '/scenarios') return ok(makePage([makeRow()], null));
  if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
  if (req.path.endsWith('/perturbed')) return ok(makePerturbed());
  return ok(makeDetail());
}

/** The real routes in a memory router, returned so a test can read where it ended up. */
function renderAt(url: string) {
  stubFetch(server);
  const router = createMemoryRouter(routes, { initialEntries: [url] });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  return router;
}

const openBox = () => screen.findByRole('searchbox', { name: 'Open by scenario ID' });
const openButton = () => screen.getByRole('button', { name: 'Open' });

describe('open by scenario ID', () => {
  it('keeps the button disabled for an empty or whitespace-only ID', async () => {
    renderAt('/');
    const box = await openBox();
    expect(openButton()).toBeDisabled();
    await userEvent.type(box, '   ');
    expect(openButton()).toBeDisabled();
    await userEvent.type(box, 'x');
    expect(openButton()).toBeEnabled();
  });

  // Enter on a disabled form never reaches the handler, so the handler's own guard is
  // exercised by submitting the form directly.
  it('does not navigate when a whitespace-only form is submitted anyway', async () => {
    const router = renderAt('/');
    const box = await openBox();
    await userEvent.type(box, '   ');
    fireEvent.submit(box.closest('form')!);
    expect(router.state.location.pathname).toBe('/');
    fireEvent.submit(box.closest('form')!);
    expect(router.state.location.pathname).toBe('/');
  });

  it('cancels the browser\'s own form submission', async () => {
    renderAt('/');
    const box = await openBox();
    await userEvent.type(box, 'abc123');
    // fireEvent returns false when a handler called preventDefault().
    expect(fireEvent.submit(box.closest('form')!)).toBe(false);
  });

  it('trims the ID before opening it', async () => {
    const router = renderAt('/');
    await userEvent.type(await openBox(), '   abc123  ');
    await userEvent.click(openButton());
    expect(router.state.location.pathname).toBe('/scenarios/abc123');
    expect(await screen.findByRole('heading', { name: 'abc123' })).toBeInTheDocument();
  });

  it.each([
    ['a space', 'two words', '/scenarios/two%20words'],
    ['a slash', 'a/b', '/scenarios/a%2Fb'],
    ['a question mark', 'a?b', '/scenarios/a%3Fb'],
    ['a hash', 'a#b', '/scenarios/a%23b'],
    ['a percent sign', 'a%b', '/scenarios/a%25b'],
  ])('encodes an ID containing %s', async (_name, id, path) => {
    const router = renderAt('/');
    await userEvent.type(await openBox(), id);
    await userEvent.click(openButton());
    expect(router.state.location.pathname).toBe(path);
    // The page reads the ID back out of the URL intact.
    expect(await screen.findByRole('heading', { name: id })).toBeInTheDocument();
  });

  it('carries the list\'s search string, so the detail page links back to it', async () => {
    const router = renderAt('/?tested=1&limit=5');
    await userEvent.type(await openBox(), 'abc123');
    await userEvent.click(openButton());
    expect(router.state.location.pathname).toBe('/scenarios/abc123');
    expect(router.state.location.state).toEqual({ listSearch: '?tested=1&limit=5' });
    await userEvent.click(await screen.findByRole('link', { name: /All scenarios/ }));
    expect(router.state.location.pathname).toBe('/');
    expect(router.state.location.search).toBe('?tested=1&limit=5');
  });

  it('carries an empty search string from an unfiltered list', async () => {
    const router = renderAt('/');
    await userEvent.type(await openBox(), 'abc123');
    await userEvent.click(openButton());
    expect(router.state.location.state).toEqual({ listSearch: '' });
  });
});
