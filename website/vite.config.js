import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Relative base so the built site works at
// https://<user>.github.io/<repo>/ without hard-coding the repo name.
export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    // Allow importing ../docs/eval_per_class_metrics.csv from the repo root.
    fs: { allow: ['..'] },
  },
});
