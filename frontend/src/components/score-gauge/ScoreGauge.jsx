import { useEffect, useState } from 'react';
import './score-gauge.css';

const START_ANGLE = -135;
const SWEEP = 270;
const RADIUS = 120;
const STROKE = 18;
const CENTER = 150;
const BASE_RATE = 0.0807;
const SCALE_MAX = 0.4; // full arc = 5x base rate — làm rõ khác biệt giữa các applicant thay vì gauge gần như rỗng ở scale 0-1

const RISK_COLOR_VAR = {
  Thấp: 'var(--color-risk-low)',
  'Trung bình': 'var(--color-risk-medium)',
  Cao: 'var(--color-risk-high)',
};

function polarPoint(angleDeg, radius) {
  const rad = (angleDeg * Math.PI) / 180;
  return { x: CENTER + radius * Math.cos(rad), y: CENTER + radius * Math.sin(rad) };
}

function arcPath(fromAngle, toAngle, radius) {
  const start = polarPoint(fromAngle, radius);
  const end = polarPoint(toAngle, radius);
  const largeArc = toAngle - fromAngle > 180 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} 1 ${end.x} ${end.y}`;
}

function valueToAngle(value) {
  const clamped = Math.min(Math.max(value, 0), SCALE_MAX);
  return START_ANGLE + (clamped / SCALE_MAX) * SWEEP;
}

export default function ScoreGauge({ pdScore, riskTier, isLoading }) {
  const [animatedValue, setAnimatedValue] = useState(0);

  useEffect(() => {
    if (isLoading || pdScore == null) return undefined;
    let frame;
    const duration = 900;
    const start = performance.now();
    const from = animatedValue;

    function tick(now) {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setAnimatedValue(from + (pdScore - from) * eased);
      if (t < 1) frame = requestAnimationFrame(tick);
    }
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pdScore, isLoading]);

  const color = RISK_COLOR_VAR[riskTier] ?? 'var(--color-accent)';
  const trackPath = arcPath(START_ANGLE, START_ANGLE + SWEEP, RADIUS);
  const valuePath = arcPath(START_ANGLE, valueToAngle(animatedValue), RADIUS);
  const baseRateAngle = valueToAngle(BASE_RATE);
  const baseRateInner = polarPoint(baseRateAngle, RADIUS - STROKE / 2 - 6);
  const baseRateOuter = polarPoint(baseRateAngle, RADIUS + STROKE / 2 + 6);

  return (
    <div className="score-gauge">
      <svg viewBox="0 0 300 260" className="score-gauge__svg" role="img" aria-label={`Xác suất vỡ nợ ${((pdScore ?? 0) * 100).toFixed(1)}%, rủi ro ${riskTier ?? ''}`}>
        <path d={trackPath} className="score-gauge__track" strokeWidth={STROKE} fill="none" />
        <line
          x1={baseRateInner.x}
          y1={baseRateInner.y}
          x2={baseRateOuter.x}
          y2={baseRateOuter.y}
          className="score-gauge__base-tick"
        />
        {!isLoading && pdScore != null && (
          <path
            d={valuePath}
            stroke={color}
            strokeWidth={STROKE}
            strokeLinecap="round"
            fill="none"
            className="score-gauge__value"
          />
        )}
      </svg>

      <div className="score-gauge__readout">
        {isLoading ? (
          <div className="skeleton score-gauge__skeleton" />
        ) : pdScore == null ? (
          <span className="score-gauge__placeholder">—</span>
        ) : (
          <>
            <span className="score-gauge__value-text mono-num" style={{ color }}>
              {(pdScore * 100).toFixed(2)}
              <span className="score-gauge__percent">%</span>
            </span>
            <span className="eyebrow">Xác suất vỡ nợ (đã hiệu chỉnh)</span>
          </>
        )}
      </div>

      <div className="score-gauge__legend">
        <span className="score-gauge__legend-dot" aria-hidden="true" />
        TB quần thể ({(BASE_RATE * 100).toFixed(1)}%)
      </div>
    </div>
  );
}
