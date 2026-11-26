import { useState } from 'react';

// Ground-truth box is fixed; the prediction moves and scales.
const GT = { x: 90, y: 60, w: 140, h: 100 };

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - inter;
  return { value: union ? inter / union : 0, inter: { x: x1, y: y1, w: Math.max(0, x2 - x1), h: Math.max(0, y2 - y1) } };
}

export default function IoUPlayground() {
  const [dx, setDx] = useState(25);
  const [dy, setDy] = useState(15);
  const [scale, setScale] = useState(1);
  const pred = { x: GT.x + dx, y: GT.y + dy, w: GT.w * scale, h: GT.h * scale };
  const { value, inter } = iou(GT, pred);
  const strictCount = Array.from({ length: 10 }, (_, i) => 0.5 + i * 0.05).filter((t) => value >= t - 1e-9).length;

  return (
    <div className="widget">
      <div className="widget-head">
        <strong>IoU playground</strong>
        <span>Move the predicted box and watch whether it still counts as a true positive.</span>
      </div>
      <div className="iou-grid">
        <svg viewBox="0 0 340 240" role="img" aria-label={`IoU ${value.toFixed(2)}`}>
          <rect x="0" y="0" width="340" height="240" className="iou-bg" />
          <rect {...rectProps(inter)} className="iou-inter" />
          <rect {...rectProps(GT)} className="iou-gt" />
          <rect {...rectProps(pred)} className="iou-pred" />
          <text x={GT.x} y={GT.y - 6} className="iou-label">ground truth</text>
          <text x={pred.x} y={pred.y + pred.h + 14} className="iou-label pred">prediction</text>
        </svg>
        <div className="iou-controls">
          <label>
            Shift x <output>{dx}px</output>
            <input type="range" min="-120" max="160" value={dx} onChange={(e) => setDx(+e.target.value)} />
          </label>
          <label>
            Shift y <output>{dy}px</output>
            <input type="range" min="-80" max="120" value={dy} onChange={(e) => setDy(+e.target.value)} />
          </label>
          <label>
            Scale <output>{scale.toFixed(2)}×</output>
            <input type="range" min="0.4" max="1.6" step="0.01" value={scale} onChange={(e) => setScale(+e.target.value)} />
          </label>
          <div className="iou-readout">
            <div className="big">{value.toFixed(3)}</div>
            <div>IoU = overlap ÷ union</div>
            <div className={value >= 0.5 ? 'ok' : 'bad'}>
              {value >= 0.5 ? '✓ True positive at IoU 0.5 (counts for AP@0.5)' : '✗ Not a match at IoU 0.5: a false positive plus a missed logo'}
            </div>
            <div>
              Counts as correct at <b>{strictCount}/10</b> of the AP@0.5:0.95 thresholds
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function rectProps(r) {
  return { x: r.x, y: r.y, width: Math.max(0, r.w), height: Math.max(0, r.h), rx: 2 };
}
