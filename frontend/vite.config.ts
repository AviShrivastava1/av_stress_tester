import react from '@vitejs/plugin-react';
import { loadEnv } from 'vite';
import { defineConfig } from 'vitest/config';
import { requireApiBaseUrl } from './src/build/requireApiBaseUrl';

export default defineConfig(({ mode }) => {
  // A production build without the API's public URL would ship a site pointed at
  // localhost (see requireApiBaseUrl). loadEnv reads the same VITE_ variables Vite
  // exposes to the bundle: .env files and the build environment (Vercel's settings).
  requireApiBaseUrl(mode, loadEnv(mode, '.', 'VITE_'));

  return {
    plugins: [react()],
    server: {
      // The API's default CORS allow-list (src/api/config.py) names exactly this
      // origin. Failing loudly on a taken port beats silently moving to 5174 and
      // meeting a CORS error that looks like a backend bug.
      port: 5173,
      strictPort: true,
    },
    test: {
      environment: 'jsdom',
      setupFiles: ['./src/test/setup.ts'],
    },
  };
});
