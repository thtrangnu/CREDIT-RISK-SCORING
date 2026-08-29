const TIER_CLASS = {
  Thấp: 'badge--low',
  'Trung bình': 'badge--medium',
  Cao: 'badge--high',
};

export default function RiskBadge({ tier }) {
  return <span className={`badge ${TIER_CLASS[tier] ?? 'badge--medium'}`}>{tier}</span>;
}
