import { useCallback, useEffect, useMemo, useState } from 'react';
import { SAMPLES, VERDICTS } from './qualitativeData.js';
import { imageSrc } from './Figure.jsx';
import Reveal from './Reveal.jsx';

const FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'correct', label: 'Correct' },
  { key: 'partial', label: 'With misses' },
  { key: 'error', label: 'Failures' },
];

// Operating points marked on the slider.
export const PRESETS = [
  { value: 0.4, label: '0.40', note: 'default in detect_yolo.py and the GUI' },
  { value: 0.5, label: '0.50', note: 'video tracking default' },
  { value: 0.77, label: '0.77', note: 'best F1 on the test split (0.93 at 0.774)' },
];
const ALL_DETS = SAMPLES.flatMap((s) => s.dets);
const TOTAL_TP = ALL_DETS.filter(([, ok]) => ok).length;
const TOTAL_FP = ALL_DETS.length - TOTAL_TP;

const matches = (s, f) => f === 'all' || s.verdict === f || (f === 'error' && s.verdict === 'mixed');

export default function QualitativeGallery() {
  const [filter, setFilter] = useState('all');
  const [conf, setConf] = useState(0.4);
  const keptTP = ALL_DETS.filter(([c, ok]) => ok && c >= conf).length;
  const keptFP = ALL_DETS.filter(([c, ok]) => !ok && c >= conf).length;
  const [open, setOpen] = useState(null);
  const shown = useMemo(() => SAMPLES.filter((s) => matches(s, filter)), [filter]);
  const count = (f) => SAMPLES.filter((s) => matches(s, f)).length;

  const step = useCallback(
    (dir) => {
      setOpen((cur) => {
        const i = shown.findIndex((s) => s.id === cur);
        return shown[(i + dir + shown.length) % shown.length].id;
      });
    },
    [shown],
  );

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') setOpen(null);
      if (e.key === 'ArrowRight') step(1);
      if (e.key === 'ArrowLeft') step(-1);
    };
    window.addEventListener('keydown', onKey);
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = '';
    };
  }, [open, step]);

  const current = SAMPLES.find((s) => s.id === open);

  return (
    <div className="qual">
      <div className="threshold">
        <div className="threshold-head">
          <label htmlFor="conf-slider"><strong>Confidence threshold</strong> <output>{conf.toFixed(2)}</output></label>
          <span className="muted">Detections scoring below the threshold fade out, exactly as <code>--conf</code> would drop them.</span>
        </div>
        <div className="slider-wrap">
          <input id="conf-slider" type="range" min="0.25" max="0.95" step="0.01" value={conf}
            onChange={(e) => setConf(+e.target.value)} aria-describedby="conf-presets" />
          <div className="ticks" aria-hidden="true">
            {PRESETS.map((p) => (
              <span key={p.value} style={{ left: `${((p.value - 0.25) / 0.7) * 100}%` }} />
            ))}
          </div>
        </div>
        <div className="presets" id="conf-presets">
          {PRESETS.map((p) => (
            <button key={p.value} type="button" className={Math.abs(conf - p.value) < 0.005 ? 'chip active' : 'chip'} onClick={() => setConf(p.value)}>
              {p.label} <span>{p.note}</span>
            </button>
          ))}
        </div>
        <div className="tiles two">
          <div className="tile">
            <div className="tile-label">Correct detections kept</div>
            <div className="tile-value">{keptTP} <small>/ {TOTAL_TP}</small></div>
            <div className="meter" aria-hidden="true"><span style={{ width: `${(keptTP / TOTAL_TP) * 100}%` }} /></div>
          </div>
          <div className="tile">
            <div className="tile-label">Misclassifications kept</div>
            <div className="tile-value">{keptFP} <small>/ {TOTAL_FP}</small></div>
            <div className="meter bad" aria-hidden="true"><span style={{ width: `${(keptFP / TOTAL_FP) * 100}%` }} /></div>
          </div>
        </div>
      </div>

      <div className="chips" role="tablist" aria-label="Filter examples">
        {FILTERS.filter((f) => count(f.key) > 0).map((f) => (
          <button
            key={f.key}
            type="button"
            role="tab"
            aria-selected={filter === f.key}
            className={filter === f.key ? 'chip active' : 'chip'}
            onClick={() => setFilter(f.key)}
          >
            {f.label} <span>{count(f.key)}</span>
          </button>
        ))}
      </div>

      <div className="qual-grid">
        {shown.map((s, i) => (
          <Reveal key={s.id} as="article" className={`qual-card v-${s.verdict} ${s.dets.every(([c]) => c < conf) ? 'filtered' : ''}`} delay={(i % 3) * 80}>
            <button type="button" className="qual-thumb" onClick={() => setOpen(s.id)} aria-label={`Enlarge (${s.id}) ${s.title}`}>
              <img src={imageSrc(s.file)} alt={`${s.brand}: ${s.pred}`} loading="lazy" />
              <span className="scan" aria-hidden="true" />
              <span className={`badge v-${s.verdict}`}>{VERDICTS[s.verdict].icon} {VERDICTS[s.verdict].label}</span>
            </button>
            <div className="qual-body">
              <div className="qual-meta">
                <span className="fig-id">({s.id})</span>
                {s.dets.map(([c, ok], j) => (
                  <span key={j} className={`det ${ok ? 'ok' : 'bad'} ${c < conf ? 'off' : ''}`} title={ok ? 'correct' : 'misclassification'}>
                    {c.toFixed(2)}
                  </span>
                ))}
              </div>
              <code className="pred">{s.pred}</code>
              <h4>{s.title}</h4>
              <p>{s.text}</p>
            </div>
          </Reveal>
        ))}
      </div>

      {current && (
        <div className="lightbox" role="dialog" aria-modal="true" aria-label={current.title} onClick={() => setOpen(null)}>
          <div className="lightbox-inner" onClick={(e) => e.stopPropagation()}>
            <img src={imageSrc(current.file)} alt={`${current.brand}: ${current.pred}`} />
            <div className="lightbox-caption">
              <div className="qual-meta">
                <span className="fig-id">Fig. Q1({current.id})</span>
                <span className={`badge inline v-${current.verdict}`}>{VERDICTS[current.verdict].icon} {VERDICTS[current.verdict].label}</span>
                <code>{current.pred}</code>
              </div>
              <h4>{current.title}</h4>
              <p>{current.text}</p>
            </div>
            <div className="lightbox-nav">
              <button type="button" onClick={() => step(-1)} aria-label="Previous">← Prev</button>
              <span>{shown.findIndex((s) => s.id === open) + 1} / {shown.length}</span>
              <button type="button" onClick={() => step(1)} aria-label="Next">Next →</button>
              <button type="button" onClick={() => setOpen(null)} aria-label="Close">Close ✕</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
