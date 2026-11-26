import { useEffect } from 'react';

const SELECTOR = [
  '.section > .kicker', '.section > h2', '.section > .lead', '.section > h3', '.section > p',
  '.section .card', '.section .figure', '.section .code', '.section .widget', '.section table.plain',
  '.section .callout', '.section .pipeline li', '.section .threshold', '.section ol.steps li', '.section ol.recipe li',
].join(',');

// Fades and lifts content blocks into view as the reader scrolls, with a
// small stagger between siblings, so the page reads as one continuous flow.
export default function useAutoReveal() {
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches || !('IntersectionObserver' in window)) return undefined;
    const els = [...document.querySelectorAll(SELECTOR)].filter((el) => !el.closest('.reveal') && !el.classList.contains('reveal'));
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add('in');
            io.unobserve(e.target);
          }
        });
      },
      { rootMargin: '0px 0px -8% 0px', threshold: 0.08 },
    );
    els.forEach((el) => {
      const siblings = el.parentElement ? [...el.parentElement.children] : [];
      const idx = Math.max(0, siblings.indexOf(el));
      el.style.setProperty('--stagger', `${Math.min(idx, 6) * 70}ms`);
      el.classList.add('auto-reveal');
      io.observe(el);
    });
    return () => io.disconnect();
  }, []);
}
