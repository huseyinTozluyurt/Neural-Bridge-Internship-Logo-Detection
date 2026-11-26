import { useEffect, useState } from 'react';
import useReveal from './useReveal.js';

// Animates a number such as "94.75%" from 0 when it scrolls into view.
export default function CountUp({ value, duration = 1400 }) {
  const [ref, visible] = useReveal(0.4);
  const match = String(value).match(/^([\d.]+)(.*)$/);
  const target = match ? parseFloat(match[1]) : 0;
  const suffix = match ? match[2] : '';
  const decimals = match && match[1].includes('.') ? match[1].split('.')[1].length : 0;
  const [n, setN] = useState(target); // final value until the animation runs
  useEffect(() => {
    if (!visible) return undefined;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setN(target);
      return undefined;
    }
    let raf;
    const start = performance.now();
    const tick = (t) => {
      const k = Math.min(1, (t - start) / duration);
      const eased = 1 - Math.pow(1 - k, 3);
      setN(target * eased);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    setN(0);
    raf = requestAnimationFrame(tick);
    // Guarantee the final value even if animation frames are throttled.
    const done = setTimeout(() => setN(target), duration + 200);
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(done);
    };
  }, [visible, target, duration]);
  return <span ref={ref}>{match ? n.toFixed(decimals) + suffix : value}</span>;
}
