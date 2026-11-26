import { useState } from 'react';

export default function CodeBlock({ code, lang = 'bash', title }) {
  const [copied, setCopied] = useState(false);
  const text = code.replace(/^\n/, '').replace(/\s+$/, '');
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable: ignore */
    }
  };
  return (
    <div className="code">
      <div className="code-bar">
        <span>{title || lang}</span>
        <button type="button" onClick={copy} aria-label="Copy code">
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre>
        <code>{text}</code>
      </pre>
    </div>
  );
}
