import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RouterProvider, createMemoryRouter } from 'react-router';
import { beforeEach, describe, expect, it } from 'vitest';
import { routes } from '../App';
import { makeDetail, makePage, makePerturbed, makeRow, makeTrajectories } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';

/*
 * Each route sets its own document title, "<page> · AV Scenario Stress-Tester", so a tab, a history entry or a screen
 * reader's page announcement says where the visitor is: Scenarios, Method, Corpus, the scenario id on a scenario page,
 * "Page not found" on an unknown address (and "Something went wrong" on the error page, so it never keeps the title of the
 * page before it). The title changes when the route changes, so there is no stale one. The only data in a title is the
 * scenario id, which comes from the address; no count or other number is written into any of them.
 */

const SUFFIX = 'AV Scenario Stress-Tester';
const ok = (body: unknown): Reply => ({ status: 200, body });
const title = (page: string) => `${page} · ${SUFFIX}`;

function serve(req: RecordedRequest): Reply {
  if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([]));
  if (req.path.endsWith('/perturbed')) return ok(makePerturbed());
  if (req.path === '/scenarios') return ok(makePage([makeRow({ scenario_id: 'synthetic-0001' })], null));
  if (req.path.startsWith('/scenarios/')) return ok(makeDetail());
  return { status: 500, body: { detail: 'not under test' } }; // /stats and /health: the page shows an error; the title is set regardless
}

beforeEach(() => {
  document.title = 'stale title from before';
  stubFetch(serve);
});

describe('the title of each route, on arrival', () => {
  it.each([
    ['the scenario list', '/', title('Scenarios')],
    ['the Method page', '/method', title('Method')],
    ['the Corpus page', '/stats', title('Corpus')],
    ['the Walkthrough page', '/walkthrough', title('Walkthrough')],
    ['a scenario page, by its id', '/scenarios/synthetic-0001', title('synthetic-0001')],
    ['an unknown address', '/no/such/page', title('Page not found')],
  ])('is set for %s', async (_name, url, expected) => {
    renderApp(url);
    await waitFor(() => expect(document.title).toBe(expected));
  });

  it('uses whatever scenario id the address carries', async () => {
    renderApp('/scenarios/abcdef0123456789');
    await waitFor(() => expect(document.title).toBe(title('abcdef0123456789')));
  });

  it('is set on the error page too, so it does not keep the title of the page before it', async () => {
    function Boom(): never {
      throw new Error('a page that cannot be displayed');
    }
    const router = createMemoryRouter(
      [{ element: routes[0]!.element, errorElement: routes[0]!.errorElement, children: [{ index: true, element: <Boom /> }] }],
      { initialEntries: ['/'] },
    );
    const quiet = console.error;
    console.error = () => undefined; // React logs the thrown render error
    try {
      render(
        <QueryClientProvider client={new QueryClient()}>
          <RouterProvider router={router} />
        </QueryClientProvider>,
      );
      await screen.findByText('This page could not be displayed');
      expect(document.title).toBe(title('Something went wrong'));
    } finally {
      console.error = quiet;
    }
  });

  it('has the form "<page> · AV Scenario Stress-Tester" and no number in a fixed title', async () => {
    for (const [url, page] of [['/', 'Scenarios'], ['/method', 'Method'], ['/stats', 'Corpus'], ['/no/such/page', 'Page not found']] as const) {
      document.title = 'x';
      const { unmount } = renderApp(url);
      await waitFor(() => expect(document.title).toBe(title(page)));
      expect(document.title).toMatch(/^[^·]+ · AV Scenario Stress-Tester$/);
      expect(document.title).not.toMatch(/\d/);
      unmount();
    }
  });
});

describe('the title follows navigation', () => {
  it('changes from Scenarios to Method to Corpus and back, with no stale title in between', async () => {
    const user = userEvent.setup();
    renderApp('/');
    await waitFor(() => expect(document.title).toBe(title('Scenarios')));
    const nav = screen.getByRole('navigation', { name: 'Main' });

    await user.click(within(nav).getByRole('link', { name: 'Method' }));
    await screen.findByRole('heading', { level: 1, name: /stored replay is shown only/ });
    expect(document.title).toBe(title('Method'));

    await user.click(within(nav).getByRole('link', { name: 'Corpus' }));
    await screen.findByRole('heading', { level: 1, name: 'Corpus' });
    expect(document.title).toBe(title('Corpus'));

    await user.click(within(nav).getByRole('link', { name: 'Scenarios' }));
    await screen.findByText('synthetic-0001');
    expect(document.title).toBe(title('Scenarios'));
  });

  it('changes from the list to a scenario page, and from a scenario page back to the list', async () => {
    const user = userEvent.setup();
    renderApp('/');
    await user.click(await screen.findByRole('link', { name: 'synthetic-0001' }));
    await waitFor(() => expect(document.title).toBe(title('synthetic-0001')));
    await user.click(within(screen.getByRole('navigation', { name: 'Main' })).getByRole('link', { name: 'Scenarios' }));
    await screen.findByText('synthetic-0001');
    expect(document.title).toBe(title('Scenarios'));
  });

  it('changes from one scenario to another when the address changes', async () => {
    const router = createMemoryRouter(routes, { initialEntries: ['/scenarios/synthetic-0001'] });
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    );
    await waitFor(() => expect(document.title).toBe(title('synthetic-0001')));
    await router.navigate('/scenarios/synthetic-0002');
    await waitFor(() => expect(document.title).toBe(title('synthetic-0002')));
  });

  it('changes from an unknown address to the list through the "Back to scenarios" link', async () => {
    const user = userEvent.setup();
    renderApp('/no/such/page');
    await waitFor(() => expect(document.title).toBe(title('Page not found')));
    await user.click(screen.getByRole('link', { name: 'Back to scenarios' }));
    await screen.findByText('synthetic-0001');
    expect(document.title).toBe(title('Scenarios'));
  });
});
