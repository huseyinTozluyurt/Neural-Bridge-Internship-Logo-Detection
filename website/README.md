# Documentation site

React + Vite site that explains the logo detection pipeline in this repository.
It is deployed to GitHub Pages by `.github/workflows/deploy-pages.yml` on every
push to `main` that touches `website/`, `docs/` or `assets/`.

```bash
cd website
npm install
npm run dev       # http://localhost:5173
npm run build     # static site in website/dist
```

Images are not duplicated: `scripts/copy-assets.mjs` copies `../docs/images`
and `../assets` into `public/images` before `dev` and `build`. The per-class
results table reads `../docs/eval_per_class_metrics.csv` directly, so updating
that CSV after a new evaluation updates the site.

One-time setup: in the repository's **Settings → Pages**, set **Source** to
**GitHub Actions**.
