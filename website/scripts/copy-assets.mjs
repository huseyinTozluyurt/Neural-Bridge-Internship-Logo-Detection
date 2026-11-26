// Copies the README images that already live in the repository into the
// site's public/ folder, so the site and the README share one source of truth.
import { cpSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const target = resolve(here, '..', 'public', 'images');
mkdirSync(target, { recursive: true });

const sources = [join(repoRoot, 'docs', 'images'), join(repoRoot, 'assets')];
let copied = 0;
for (const dir of sources) {
  if (!existsSync(dir)) continue;
  for (const name of readdirSync(dir)) {
    if (!/\.(png|jpe?g|gif|webp|svg)$/i.test(name)) continue;
    cpSync(join(dir, name), join(target, name));
    copied += 1;
  }
}
console.log(`copy-assets: ${copied} image(s) -> public/images`);
