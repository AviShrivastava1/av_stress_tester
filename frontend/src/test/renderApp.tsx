import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render } from '@testing-library/react';
import { RouterProvider, createMemoryRouter } from 'react-router';
import { routes } from '../App';

/** The real route table and layout, in a memory router starting at `initialUrl`. */
export function renderApp(initialUrl = '/') {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const router = createMemoryRouter(routes, { initialEntries: [initialUrl] });
  return render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
}
