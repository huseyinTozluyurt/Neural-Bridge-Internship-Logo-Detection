import { useState } from 'react';

// Defaults are the real totals from the FlickrLogos-32 test-split evaluation.
export default function PRCalculator() {
  const [tp, setTp] = useState(300);
  const [fp, setFp] = useState(50);
  const [fn, setFn] = useState(23);
  const p = tp + fp ? tp / (tp + fp) : 0;
  const r = tp + fn ? tp / (tp + fn) : 0;
  const f1 = p + r ? (2 * p * r) / (p + r) : 0;
  const num = (setter) => (e) => setter(Math.max(0, parseInt(e.target.value || '0', 10)));
  const reset = () => { setTp(300); setFp(50); setFn(23); };

  return (
    <div className="widget">
      <div className="widget-head">
        <strong>Precision / recall calculator</strong>
        <span>Pre-filled with this project&apos;s test-split totals (confidence ≥ 0.25, IoU ≥ 0.5).</span>
      </div>
      <div className="pr-grid">
        <label>True positives<input type="number" min="0" value={tp} onChange={num(setTp)} /></label>
        <label>False positives<input type="number" min="0" value={fp} onChange={num(setFp)} /></label>
        <label>Missed (FN)<input type="number" min="0" value={fn} onChange={num(setFn)} /></label>
        <button type="button" className="ghost" onClick={reset}>Reset</button>
      </div>
      <div className="tiles">
        <Tile label="Precision" value={p} formula="TP ÷ (TP + FP)" />
        <Tile label="Recall" value={r} formula="TP ÷ (TP + FN)" />
        <Tile label="F1-score" value={f1} formula="2PR ÷ (P + R)" />
      </div>
    </div>
  );
}

function Tile({ label, value, formula }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{(value * 100).toFixed(2)}%</div>
      <div className="meter" aria-hidden="true"><span style={{ width: `${value * 100}%` }} /></div>
      <div className="tile-formula">{formula}</div>
    </div>
  );
}
