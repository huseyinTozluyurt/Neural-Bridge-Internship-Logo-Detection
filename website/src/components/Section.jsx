export default function Section({ id, kicker, title, lead, children }) {
  return (
    <section id={id} className="section">
      {kicker && <div className="kicker">{kicker}</div>}
      <h2>
        <a href={`#${id}`}>{title}</a>
      </h2>
      {lead && <p className="lead">{lead}</p>}
      {children}
    </section>
  );
}
