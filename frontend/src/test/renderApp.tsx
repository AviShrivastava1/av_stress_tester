import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render } from '@testing-library/react';
import { RouterProvider, createMemoryRouter } from 'react-router';
import { ScenarioDetailPage } from '../pages/ScenarioDetailPage';
import { ScenarioListPage } from '../pages/ScenarioListPage';

/** Both pages, on the same paths the real router uses, starting at `initialUrl`. */
export function renderApp(initialUrl = '/') {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const router = createMemoryRouter(
    [
      { path: '/', element: <ScenarioListPage /> },
      { path: '/scenarios/:scenarioId', element: <ScenarioDetailPage /> },
    ],
    { initialEntries: [initialUrl] },
  );
  return render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
}
