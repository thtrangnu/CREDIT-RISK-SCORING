import './shap-waterfall.css';

const ROW_HEIGHT = 40;
const CHART_WIDTH = 640;
const LABEL_WIDTH = 210;
const BAR_AREA = CHART_WIDTH - LABEL_WIDTH - 70;
const TOP_PAD = 36;
const BOTTOM_PAD = 28;

function buildRows(baseValue, reasons, rawMargin) {
  const reasonsSum = reasons.reduce((acc, r) => acc + r.shap, 0);
  const othersShap = rawMargin - baseValue - reasonsSum;

  let cursor = baseValue;
  const rows = reasons.map((r) => {
    const start = cursor;
    cursor += r.shap;
    return { label: r.label, shap: r.shap, start, end: cursor };
  });
  rows.push({ label: 'Các yếu tố còn lại (gộp)', shap: othersShap, start: cursor, end: cursor + othersShap, isOthers: true });

  return rows;
}

export default function ShapWaterfall({ baseValue, reasons, rawMargin }) {
  if (baseValue == null || rawMargin == null || !reasons?.length) {
    return <p className="empty-state">Chưa có dữ liệu SHAP — chấm điểm 1 applicant trước.</p>;
  }

  const rows = buildRows(baseValue, reasons, rawMargin);
  const allValues = [baseValue, rawMargin, ...rows.flatMap((r) => [r.start, r.end])];
  const domainMin = Math.min(...allValues);
  const domainMax = Math.max(...allValues);
  const pad = (domainMax - domainMin) * 0.08 || 0.5;
  const scale = (v) => ((v - (domainMin - pad)) / (domainMax - domainMin + pad * 2)) * BAR_AREA;

  const height = TOP_PAD + rows.length * ROW_HEIGHT + BOTTOM_PAD;
  const baseX = scale(baseValue);
  const finalX = scale(rawMargin);

  return (
    <div className="shap-waterfall">
      <div className="shap-waterfall__header">
        <span className="eyebrow">Waterfall — margin space (log-odds)</span>
        <div className="shap-waterfall__legend">
          <span className="shap-waterfall__legend-item">
            <i className="shap-waterfall__swatch shap-waterfall__swatch--up" /> tăng rủi ro
          </span>
          <span className="shap-waterfall__legend-item">
            <i className="shap-waterfall__swatch shap-waterfall__swatch--down" /> giảm rủi ro
          </span>
        </div>
      </div>

      <svg viewBox={`0 0 ${CHART_WIDTH} ${height}`} className="shap-waterfall__svg" role="img" aria-label="SHAP waterfall chart">
        <line
          x1={LABEL_WIDTH + baseX}
          x2={LABEL_WIDTH + baseX}
          y1={8}
          y2={height - 8}
          className="shap-waterfall__guide shap-waterfall__guide--base"
        />
        <line
          x1={LABEL_WIDTH + finalX}
          x2={LABEL_WIDTH + finalX}
          y1={8}
          y2={height - 8}
          className="shap-waterfall__guide shap-waterfall__guide--final"
        />
        <text x={LABEL_WIDTH + baseX} y={16} className="shap-waterfall__guide-label" textAnchor="middle">
          Base
        </text>
        <text x={LABEL_WIDTH + finalX} y={16} className="shap-waterfall__guide-label shap-waterfall__guide-label--final" textAnchor="middle">
          Kết quả
        </text>

        {rows.map((row, i) => {
          const y = TOP_PAD + i * ROW_HEIGHT;
          const x1 = scale(Math.min(row.start, row.end));
          const x2 = scale(Math.max(row.start, row.end));
          const isUp = row.shap >= 0;
          const barWidth = x2 - x1;
          // Bar đủ rộng (label giá trị + số ký tự ước lượng) -> đặt nhãn NẰM TRONG bar
          // (tránh đè lên label dòng bên trái khi bar dài gần chạm cột label).
          const labelText = `${row.shap > 0 ? '+' : ''}${row.shap.toFixed(3)}`;
          const estimatedLabelWidth = labelText.length * 6.5;
          const fitsInside = barWidth > estimatedLabelWidth + 16;
          const deltaX = fitsInside
            ? LABEL_WIDTH + (isUp ? x1 : x2) + (isUp ? 10 : -10)
            : isUp
              ? LABEL_WIDTH + x2 + 8
              : LABEL_WIDTH + x1 - 8;
          const deltaAnchor = isUp ? 'start' : 'end';

          return (
            <g key={row.label} className="shap-waterfall__row" style={{ '--stagger': i }}>
              <text x={LABEL_WIDTH - 14} y={y + ROW_HEIGHT / 2 + 4} textAnchor="end" className={`shap-waterfall__label${row.isOthers ? ' is-others' : ''}`}>
                {row.label}
              </text>
              <rect
                x={LABEL_WIDTH + x1}
                y={y + 8}
                width={Math.max(x2 - x1, 2)}
                height={ROW_HEIGHT - 16}
                rx={4}
                className={`shap-waterfall__bar ${isUp ? 'is-up' : 'is-down'}${row.isOthers ? ' is-others' : ''}`}
              />
              <text
                x={deltaX}
                y={y + ROW_HEIGHT / 2 + 4}
                textAnchor={deltaAnchor}
                className={`shap-waterfall__delta mono-num${fitsInside ? ' is-inside' : ''}`}
              >
                {labelText}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
