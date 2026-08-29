export default function StatTile({ label, value, delta, mono = true, emphasis = false }) {
  return (
    <div className={`stat-tile${emphasis ? ' stat-tile--emphasis' : ''}`}>
      <span className="eyebrow">{label}</span>
      <span className={`stat-tile__value${mono ? ' mono-num' : ''}`}>{value}</span>
      {delta && <span className="stat-tile__delta mono-num">{delta}</span>}
    </div>
  );
}
