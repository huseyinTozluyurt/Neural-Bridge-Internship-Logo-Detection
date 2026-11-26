import { useState } from 'react';

// Mirrors the per-row rules in scripts/dataset_validation.py.
function check(text, nc) {
  const results = [];
  text.split('\n').forEach((line, i) => {
    if (!line.trim()) return;
    const where = `line ${i + 1}`;
    const parts = line.trim().split(/\s+/);
    if (parts.length !== 5) return results.push({ ok: false, where, msg: 'expected 5 values (class x_center y_center width height)' });
    const cls = Number(parts[0]);
    const [x, y, w, h] = parts.slice(1).map(Number);
    if (!Number.isInteger(cls) || [x, y, w, h].some((v) => Number.isNaN(v))) return results.push({ ok: false, where, msg: 'contains a non-numeric value' });
    if (cls < 0 || cls >= nc) return results.push({ ok: false, where, msg: `class id ${cls} is outside 0..${nc - 1}` });
    if (![x, y, w, h].every(Number.isFinite)) return results.push({ ok: false, where, msg: 'coordinates must be finite' });
    if (!(x >= 0 && x <= 1 && y >= 0 && y <= 1 && w > 0 && w <= 1 && h > 0 && h <= 1)) return results.push({ ok: false, where, msg: 'coordinates must be normalized and box width/height positive' });
    const eps = 1e-6;
    if (x - w / 2 < -eps || y - h / 2 < -eps || x + w / 2 > 1 + eps || y + h / 2 > 1 + eps) return results.push({ ok: false, where, msg: 'bounding box extends outside the image' });
    results.push({ ok: true, where, msg: `class ${cls}, centre (${x}, ${y}), size ${w} × ${h}` });
  });
  return results;
}

const SAMPLE = `12 0.512 0.430 0.210 0.118
5 0.95 0.5 0.2 0.3
40 0.5 0.5 0.1 0.1
3 0.5 0.5 0.2`;

export default function LabelChecker() {
  const [text, setText] = useState(SAMPLE);
  const [nc, setNc] = useState(32);
  const results = check(text, nc);
  return (
    <div className="widget">
      <div className="widget-head">
        <strong>Try the label rules</strong>
        <span>Paste rows from a YOLO <code>.txt</code> label file. The checks match <code>dataset_validation.py</code>.</span>
      </div>
      <div className="checker">
        <div>
          <label className="inline">
            nc (classes in data.yaml)
            <input type="number" min="1" value={nc} onChange={(e) => setNc(Math.max(1, +e.target.value || 1))} />
          </label>
          <textarea spellCheck="false" value={text} onChange={(e) => setText(e.target.value)} rows={7} />
        </div>
        <ul className="check-results">
          {results.length === 0 && <li className="muted">An empty label file is a confirmed negative (background) image.</li>}
          {results.map((r) => (
            <li key={r.where} className={r.ok ? 'ok' : 'bad'}>
              <span>{r.ok ? '✓' : '✗'}</span>
              <b>{r.where}</b> {r.msg}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
