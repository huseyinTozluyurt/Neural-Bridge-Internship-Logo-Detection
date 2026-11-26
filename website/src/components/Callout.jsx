export default function Callout({ tone = 'note', title, children }) {
  return (
    <aside className={`callout ${tone}`}>
      {title && <strong>{title}</strong>}
      <div>{children}</div>
    </aside>
  );
}
