import type { ReactNode } from 'react';
import { Link, NavLink, Outlet, ScrollRestoration, createBrowserRouter, type RouteObject } from 'react-router';
import { Footer } from './components/Footer';
import { ScenarioDetailPage } from './pages/ScenarioDetailPage';
import { MethodPage } from './pages/MethodPage';
import { ScenarioListPage } from './pages/ScenarioListPage';
import { StatsPage } from './pages/StatsPage';

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

function NotFoundPage() {
  return <section className="empty-page">
    <span className="eyebrow">404 · UNKNOWN ADDRESS</span>
    <h1>Page not found</h1>
    <p>This address does not match a page. Browse the scenarios to get back on track.</p>
    <Link className="button-link" to="/">Back to scenarios</Link>
  </section>;
}

function RouteErrorPage() {
  return <Layout><section className="empty-page">
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
      { index: true, element: <ScenarioListPage /> },
      { path: 'scenarios/:scenarioId', element: <ScenarioDetailPage /> },
      { path: 'method', element: <MethodPage /> },
      { path: 'stats', element: <StatsPage /> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
];

export const router = createBrowserRouter(routes);
