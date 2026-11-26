import { useMemo, useState } from 'react';
import csv from '../../../docs/eval_per_class_metrics.csv?raw';

function parse(text) {
  const [header, ...rows] = text.trim().split(/\r?\n/);
  const keys = header.split(',');
  return rows.map((row) => {
    const cells = row.split(',');
    const o = {};
    keys.forEach((k, i) => {
      const v = cells[i];
      o[k] = k === 'class_name' ? v : Number(v);
    });
    return o;
  });
}

const COLUMNS = [
  { key: 'class_name', label: 'Class' },
  { key: 'ground_truth', label: 'GT' },
  { key: 'tp', label: 'TP' },
  { key: 'fp', label: 'FP' },
  { key: 'fn', label: 'FN' },
  { key: 'precision_at_report_conf', label: 'Precision', pct: true },
  { key: 'recall_at_report_conf', label: 'Recall', pct: true },
  { key: 'ap50', label: 'AP@0.5', bar: true },
  { key: 'ap50_95', label: 'AP@0.5:0.95', bar: true },
];

export default function MetricsTable() {
  const data = useMemo(() => parse(csv), []);
  const [sort, setSort] = useState({ key: 'ap50_95', dir: 'asc' });
  const [hover, setHover] = useState(null);

  const sorted = useMemo(() => {
    const out = [...data];
    out.sort((a, b) => {
      const av = a[sort.key];
      const bv = b[sort.key];
      const cmp = typeof av === 'string' ? av.localeCompare(bv) : av - bv;
      return sort.dir === 'asc' ? cmp : -cmp;
    });
    return out;
  }, [data, sort]);

  const toggle = (key) =>
    setSort((s) => (s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: key === 'class_name' ? 'asc' : 'desc' }));

  return (
    <div className="widget">
      <div className="widget-head">
        <strong>Per-class results on the held-out test split</strong>
        <span>
          {data.length} classes, loaded from <code>docs/eval_per_class_metrics.csv</code>. Click a header to sort. Sorted by weakest
          strict AP first so the classes that need more data stand out.
        </span>
      </div>
      <div className="table-wrap">
        <table className="metrics">
          <thead>
            <tr>
              {COLUMNS.map((c) => (
                <th key={c.key} aria-sort={sort.key === c.key ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'}>
                  <button type="button" onClick={() => toggle(c.key)}>
                    {c.label}
                    <span className="sort">{sort.key === c.key ? (sort.dir === 'asc' ? '▲' : '▼') : ''}</span>
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => (
              <tr key={row.class_name}>
                {COLUMNS.map((c) => {
                  const v = row[c.key];
                  if (c.bar) {
                    const tip = `${row.class_name} · ${c.label} ${v.toFixed(3)}`;
                    return (
                      <td
                        key={c.key}
                        className="bar-cell"
                        onMouseEnter={() => setHover(`${row.class_name}-${c.key}`)}
                        onMouseLeave={() => setHover(null)}
                        title={tip}
                      >
                        <span className="bar-row">
                          <span className="bar-track">
                            <span className="bar" style={{ width: `${v * 100}%` }} />
                          </span>
                          <span className="bar-value">{v.toFixed(3)}</span>
                        </span>
                        {hover === `${row.class_name}-${c.key}` && <span className="tip">{tip}</span>}
                      </td>
                    );
                  }
                  if (c.pct) return <td key={c.key}>{(v * 100).toFixed(1)}%</td>;
                  return <td key={c.key}>{v}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
