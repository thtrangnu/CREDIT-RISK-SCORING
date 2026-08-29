import './reason-code-list.css';

function formatValue(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toLocaleString('vi-VN') : value.toFixed(3);
  }
  return String(value);
}

export default function ReasonCodeList({ reasons, title = 'Lý do chính (reason codes)' }) {
  if (!reasons || reasons.length === 0) {
    return <p className="empty-state">Chưa có reason codes — chấm điểm 1 applicant trước.</p>;
  }

  return (
    <div className="reason-code-list">
      <h3 className="reason-code-list__title">{title}</h3>
      <ol className="reason-code-list__items">
        {reasons.map((reason, i) => (
          <li key={reason.feature} className="reason-code-list__item" style={{ '--stagger': i }}>
            <span className="reason-code-list__rank mono-num">{String(i + 1).padStart(2, '0')}</span>
            <div className="reason-code-list__body">
              <div className="reason-code-list__label-row">
                <span className="reason-code-list__label">{reason.label}</span>
                {!reason.curated && <span className="badge badge--curated">chưa curate</span>}
              </div>
              <div className="reason-code-list__meta">
                <span className={`reason-code-list__direction ${reason.shap > 0 ? 'is-up' : 'is-down'}`}>
                  {reason.shap > 0 ? '▲' : '▼'} {reason.direction}
                </span>
                <span className="reason-code-list__value mono-num">giá trị: {formatValue(reason.value)}</span>
              </div>
            </div>
            <span className="reason-code-list__shap mono-num" title="SHAP value (margin space)">
              {reason.shap > 0 ? '+' : ''}
              {reason.shap.toFixed(3)}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
