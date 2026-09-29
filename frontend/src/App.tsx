import { NavLink, Outlet, createBrowserRouter, type RouteObject } from 'react-router';
import { Footer } from './components/Footer';
import { ScenarioDetailPage } from './pages/ScenarioDetailPage';
import { ScenarioListPage } from './pages/ScenarioListPage';
import { StatsPage } from './pages/StatsPage';

function Layout() {
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
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}

/** The route table, shared by the browser router and the tests' memory router. */
export const routes: RouteObject[] = [
  {
    element: <Layout />,
    children: [
      { index: true, element: <ScenarioListPage /> },
      { path: 'scenarios/:scenarioId', element: <ScenarioDetailPage /> },
      { path: 'stats', element: <StatsPage /> },
    ],
  },
];

export const router = createBrowserRouter(routes);
