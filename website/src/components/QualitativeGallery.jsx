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

const matches = (s, f) => f === 'all' || s.verdict === f || (f === 'error' && s.verdict === 'mixed');

export default function QualitativeGallery() {
  const [filter, setFilter] = useState('all');
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
      <div className="chips" role="tablist" aria-label="Filter examples">
        {FILTERS.map((f) => (
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
          <Reveal key={s.id} as="article" className={`qual-card v-${s.verdict}`} delay={(i % 3) * 80}>
            <button type="button" className="qual-thumb" onClick={() => setOpen(s.id)} aria-label={`Enlarge (${s.id}) ${s.title}`}>
              <img src={imageSrc(s.file)} alt={`${s.brand}: ${s.pred}`} loading="lazy" />
              <span className="scan" aria-hidden="true" />
              <span className={`badge v-${s.verdict}`}>{VERDICTS[s.verdict].icon} {VERDICTS[s.verdict].label}</span>
            </button>
            <div className="qual-body">
              <div className="qual-meta">
                <span className="fig-id">({s.id})</span>
                <code>{s.pred}</code>
              </div>
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
