import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

export default defineConfig({
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
});
