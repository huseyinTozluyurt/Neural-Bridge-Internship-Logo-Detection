import { useState } from 'react';
import useReveal from './useReveal.js';

// Transcribed from the YOLOv7 console log (validation after each epoch,
// 232 images / 373 boxes). Epoch 3 has no loss row visible in the capture.
export const LOG = [
  { epoch: 3, p: 0.0899, r: 0.178, map50: 0.0816, map: 0.051, loss: null, time: null },
  { epoch: 4, p: 0.056, r: 0.372, map50: 0.0921, map: 0.0628, loss: 0.1059, time: '7:25' },
  { epoch: 5, p: 0.116, r: 0.358, map50: 0.146, map: 0.105, loss: 0.103, time: '7:25' },
  { epoch: 6, p: 0.216, r: 0.334, map50: 0.165, map: 0.117, loss: 0.101, time: '7:26' },
  { epoch: 7, p: 0.191, r: 0.417, map50: 0.216, map: 0.15, loss: 0.09901, time: '7:25' },
  { epoch: 8, p: 0.377, r: 0.329, map50: 0.253, map: 0.179, loss: 0.09505, time: '7:25' },
  { epoch: 9, p: 0.355, r: 0.395, map50: 0.271, map: 0.194, loss: 0.09181, time: '15:43' },
  { epoch: 10, p: 0.29, r: 0.456, map50: 0.341, map: 0.244, loss: 0.08785, time: '34:29' },
  { epoch: 11, p: 0.287, r: 0.515, map50: 0.422, map: 0.302, loss: 0.08482, time: '34:43' },
  { epoch: 12, p: 0.462, r: 0.528, map50: 0.487, map: 0.348, loss: 0.08105, time: '34:46' },
];

const W = 420;
const H = 230;
const M = { t: 16, r: 104, b: 34, l: 42 };
const x = (e) => M.l + ((e - 3) / 9) * (W - M.l - M.r);

function path(rows, key, y) {
  return rows
    .filter((d) => d[key] != null)
    .map((d, i) => `${i ? 'L' : 'M'}${x(d.epoch).toFixed(1)},${y(d[key]).toFixed(1)}`)
    .join(' ');
}

function Chart({ title, series, yMax, yMin = 0, ticks, fmt, visible }) {
  const [hover, setHover] = useState(null);
  const y = (v) => M.t + (1 - (v - yMin) / (yMax - yMin)) * (H - M.t - M.b);
  const onMove = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    const ep = Math.round(3 + ((px - M.l) / (W - M.l - M.r)) * 9);
    setHover(Math.min(12, Math.max(3, ep)));
  };
  const row = hover != null ? LOG.find((d) => d.epoch === hover) : null;

  return (
    <figure className="chart">
      <figcaption className="chart-title">{title}</figcaption>
      {series.length > 1 && (
        <div className="legend" aria-hidden="true">
          {series.map((s) => (
            <span key={s.key}><i style={{ background: s.color }} />{s.label}</span>
          ))}
        </div>
      )}
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${title}, epochs 3 to 12`}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={M.l - 8} y={y(t) + 4} className="axis" textAnchor="end">{fmt(t)}</text>
          </g>
        ))}
        {LOG.map((d) => (
          <text key={d.epoch} x={x(d.epoch)} y={H - M.b + 18} className="axis" textAnchor="middle">{d.epoch}</text>
        ))}
        <text x={(M.l + W - M.r) / 2} y={H - 2} className="axis" textAnchor="middle">epoch</text>
        {hover != null && <line x1={x(hover)} x2={x(hover)} y1={M.t} y2={H - M.b} className="crosshair" />}
        {series.map((s) => {
          const last = [...LOG].reverse().find((d) => d[s.key] != null);
          return (
            <g key={s.key}>
              <path d={path(LOG, s.key, y)} className={`line ${visible ? 'draw' : ''}`} style={{ stroke: s.color }} pathLength="1" />
              {LOG.filter((d) => d[s.key] != null).map((d) => (
                <circle key={d.epoch} cx={x(d.epoch)} cy={y(d[s.key])} r={hover === d.epoch ? 4.5 : 3}
                  className={`dot ${visible ? 'show' : ''}`} style={{ fill: s.color }} />
              ))}
              <text x={x(last.epoch) + 8} y={y(last[s.key]) + 4} className="direct">{s.label} {fmt(last[s.key])}</text>
            </g>
          );
        })}
      </svg>
      {row && (
        <div className="chart-tip" style={{ left: `${(x(row.epoch) / W) * 100}%` }}>
          <b>Epoch {row.epoch}</b>
          {series.map((s) => (
            <span key={s.key}><i style={{ background: s.color }} />{s.label}: {row[s.key] != null ? fmt(row[s.key]) : 'n/a'}</span>
          ))}
        </div>
      )}
    </figure>
  );
}

export default function TrainingDynamics() {
  const [ref, visible] = useReveal(0.3);
  return (
    <div className="widget" ref={ref}>
      <div className="widget-head">
        <strong>Early training dynamics, transcribed from the console log</strong>
        <span>Hover to read exact values. Data: validation results after each of epochs 3–12 shown in Fig. T1.</span>
      </div>
      <div className="chart-row">
        <Chart
          title="Validation mAP"
          visible={visible}
          yMax={0.5}
          ticks={[0, 0.1, 0.2, 0.3, 0.4, 0.5]}
          fmt={(v) => v.toFixed(2)}
          series={[
            { key: 'map50', label: 'mAP@.5', color: 'var(--series)' },
            { key: 'map', label: 'mAP@.5:.95', color: 'var(--series-2)' },
          ]}
        />
        <Chart
          title="Total training loss (box + obj + cls)"
          visible={visible}
          yMin={0.075}
          yMax={0.11}
          ticks={[0.08, 0.09, 0.1, 0.11]}
          fmt={(v) => v.toFixed(3)}
          series={[{ key: 'loss', label: 'loss', color: 'var(--series)' }]}
        />
      </div>
      <details className="data-table">
        <summary>Show the data table</summary>
        <table className="plain">
          <thead><tr><th>Epoch</th><th>P</th><th>R</th><th>mAP@.5</th><th>mAP@.5:.95</th><th>Total loss</th><th>Epoch time</th></tr></thead>
          <tbody>
            {LOG.map((d) => (
              <tr key={d.epoch}>
                <td>{d.epoch}</td><td>{d.p}</td><td>{d.r}</td><td>{d.map50}</td><td>{d.map}</td>
                <td>{d.loss ?? 'n/a'}</td><td>{d.time ?? 'n/a'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
