import type { ReactNode } from 'react';
import { Link, NavLink, Outlet, createBrowserRouter, type RouteObject } from 'react-router';
import { Footer } from './components/Footer';
import { ScenarioDetailPage } from './pages/ScenarioDetailPage';
import { ScenarioListPage } from './pages/ScenarioListPage';
import { StatsPage } from './pages/StatsPage';

function Layout({ children }: { children?: ReactNode }) {
  return (
    <div className="page">
      <header className="site-header">
        <span className="site-title">AV Scenario Stress-Tester</span>
        <nav className="site-nav" aria-label="Main">
          <NavLink to="/" end>
            Scenarios
          </NavLink>
          <NavLink to="/stats">Corpus</NavLink>
        </nav>
      </header>
      <main>
        {children ?? <Outlet />}
      </main>
      <Footer />
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
      { path: 'stats', element: <StatsPage /> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
];

export const router = createBrowserRouter(routes);
