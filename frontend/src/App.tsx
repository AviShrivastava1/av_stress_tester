import type { ReactNode } from 'react';
import { Link, NavLink, Outlet, ScrollRestoration, createBrowserRouter, useParams, type RouteObject } from 'react-router';
import { Footer } from './components/Footer';
import { ScenarioDetailPage } from './pages/ScenarioDetailPage';
import { MethodPage } from './pages/MethodPage';
import { ScenarioListPage } from './pages/ScenarioListPage';
import { StatsPage } from './pages/StatsPage';
import { useDocumentTitle } from './useDocumentTitle';

/*
 * The wordmark, and what backs each part:
 *   "AV/"                      the project's initials (the brand mark), set in mono
 *   "Scenario Stress-Tester"   the project's name as in PROJECT_NAME (components/Footer.tsx) and the <title> in index.html;
 *                              it is hidden below 560 px, so the home link's aria-label carries the whole name:
 *                              "AV Scenario Stress-Tester home"
 * There is no tagline. The one that said "Adversarial scenario search" is gone, and copyLint.test.tsx now bans
 * "adversarial" on every page, with no exemption.
 */
function Layout({ children }: { children?: ReactNode }) {
  return (
    <div className="page">
      <a className="skip-link" href="#main-content">Skip to content</a>
      <header className="site-header">
        <Link to="/" className="brand" aria-label="AV Scenario Stress-Tester home">
          <span className="brand-mark" aria-hidden="true">AV/</span>
          <span className="site-title">Scenario Stress-Tester</span>
        </Link>
        <nav className="site-nav" aria-label="Main">
          <NavLink to="/" end>
            Scenarios
          </NavLink>
          <NavLink to="/method">Method</NavLink>
          <NavLink to="/stats">Corpus</NavLink>
        </nav>
      </header>
      <main id="main-content" tabIndex={-1}>
        {children ?? <Outlet />}
      </main>
      <Footer />
      <ScrollRestoration />
    </div>
  );
}

/*
 * The words each route puts in the document title (see useDocumentTitle for the form), and what backs them:
 *   "Scenarios"        the list page and the nav link of that name
 *   "Method"           the Method page and the nav link of that name
 *   "Corpus"           the Corpus page (the /stats route) and the nav link of that name; its h1 is "Corpus"
 *   the scenario id    the scenario page's own h1 shows the same id, taken from the address (:scenarioId)
 *   "Page not found"   the h1 of NotFoundPage below
 *   "Something went wrong"   the eyebrow of RouteErrorPage below, so the error page does not keep the title of the page before
 */
function PageTitle({ page }: { page: string }) {
  useDocumentTitle(page);
  return null;
}

function ScenarioTitle() {
  const { scenarioId = '' } = useParams();
  useDocumentTitle(scenarioId);
  return null;
}

function NotFoundPage() {
  return <section className="empty-page">
    <PageTitle page="Page not found" />
    <span className="eyebrow">404 · UNKNOWN ADDRESS</span>
    <h1>Page not found</h1>
    <p>This address does not match a page. Browse the scenarios to get back on track.</p>
    <Link className="button-link" to="/">Back to scenarios</Link>
  </section>;
}

function RouteErrorPage() {
  return <Layout><section className="empty-page">
    <PageTitle page="Something went wrong" />
    <span className="eyebrow">SOMETHING WENT WRONG</span>
    <h1>This page could not be displayed</h1>
    <p role="alert">Please reload the page or return to the scenario explorer.</p>
    <a className="button-link" href="/">Back to scenarios</a>
  </section></Layout>;
}

/** The route table, shared by the browser router and the tests' memory router. */
export const routes: RouteObject[] = [
  {
    element: <Layout />,
    errorElement: <RouteErrorPage />,
    children: [
      { index: true, element: <><PageTitle page="Scenarios" /><ScenarioListPage /></> },
      { path: 'scenarios/:scenarioId', element: <><ScenarioTitle /><ScenarioDetailPage /></> },
      { path: 'method', element: <><PageTitle page="Method" /><MethodPage /></> },
      { path: 'stats', element: <><PageTitle page="Corpus" /><StatsPage /></> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
];

export const router = createBrowserRouter(routes);
