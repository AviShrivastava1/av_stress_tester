import { Outlet, createBrowserRouter } from 'react-router';
import { Footer } from './components/Footer';
import { ScenarioListPage } from './pages/ScenarioListPage';

function Layout() {
  return (
    <div className="page">
      <header className="site-header">
        <span className="site-title">AV Scenario Stress-Tester</span>
      </header>
      <main>
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}

export const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [{ index: true, element: <ScenarioListPage /> }],
  },
]);
