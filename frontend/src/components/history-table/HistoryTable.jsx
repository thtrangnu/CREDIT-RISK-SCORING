import RiskBadge from '../ui/RiskBadge';
import './history-table.css';

function formatTimestamp(iso) {
  return new Date(iso).toLocaleString('vi-VN', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export default function HistoryTable({ items, isLoading }) {
  if (isLoading) {
    return (
      <div className="history-table__skeletons">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="skeleton history-table__row-skeleton" />
        ))}
      </div>
    );
  }

  if (!items || items.length === 0) {
    return <p className="empty-state">Chưa có lịch sử chấm điểm nào — sang trang Score để bắt đầu.</p>;
  }

  return (
    <div className="history-table__scroll">
      <table className="history-table">
        <thead>
          <tr>
            <th>SK_ID_CURR</th>
            <th>PD (đã hiệu chỉnh)</th>
            <th>Risk tier</th>
            <th>Model version</th>
            <th>Thời điểm chấm</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item, i) => (
            <tr key={item.id} style={{ '--stagger': i }}>
              <td className="mono-num">#{item.sk_id_curr}</td>
              <td className="mono-num">{(item.pd_score * 100).toFixed(2)}%</td>
              <td>
                <RiskBadge tier={item.risk_tier} />
              </td>
              <td className="font-mono history-table__version">{item.model_version}</td>
              <td className="history-table__time">{formatTimestamp(item.scored_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
