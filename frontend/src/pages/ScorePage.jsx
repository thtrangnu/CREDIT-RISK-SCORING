import { useState } from 'react';
import { Link } from 'react-router-dom';
import ApplicantSelector from '../components/applicant-selector/ApplicantSelector';
import ScoreGauge from '../components/score-gauge/ScoreGauge';
import ReasonCodeList from '../components/reason-code-list/ReasonCodeList';
import RiskBadge from '../components/ui/RiskBadge';
import { scoreApplicant } from '../api/client';
import { useScoreContext } from '../context/ScoreContext';
import './score-page.css';

export default function ScorePage() {
  const { applicant, setApplicant, scoreResult, setScoreResult } = useScoreContext();
  const [isScoring, setIsScoring] = useState(false);
  const [error, setError] = useState(null);

  async function handleSelect(row) {
    setApplicant(row);
    setError(null);
    setIsScoring(true);
    try {
      const result = await scoreApplicant(row.sk_id_curr);
      setScoreResult(result);
    } catch (err) {
      setError(err.message);
      setScoreResult(null);
    } finally {
      setIsScoring(false);
    }
  }

  return (
    <main className="score-page">
      <section className="container score-page__hero">
        <span className="eyebrow">Home Credit · Default Risk Scoring</span>
        <h1 className="score-page__title font-display">
          Chấm điểm rủi ro <span className="score-page__title-accent">tức thời</span>, giải trình được.
        </h1>
        <p className="score-page__subtitle">
          Chọn 1 applicant có sẵn trong tập dữ liệu — hệ thống tự kéo lịch sử tín dụng từ 7 bảng, chấm điểm bằng
          LightGBM đã hiệu chỉnh isotonic, và liệt kê top lý do (SHAP) đứng sau con số.
        </p>
        <ApplicantSelector onSelect={handleSelect} selectedId={applicant?.sk_id_curr} disabled={isScoring} />
      </section>

      {error && (
        <div className="container">
          <p className="score-page__error">Lỗi: {error}</p>
        </div>
      )}

      {(applicant || isScoring) && (
        <section className="container score-page__result">
          <div className="card score-page__gauge-card">
            <ScoreGauge pdScore={scoreResult?.pd_score} riskTier={scoreResult?.risk_tier} isLoading={isScoring} />
            {scoreResult && (
              <div className="score-page__gauge-footer">
                <RiskBadge tier={scoreResult.risk_tier} />
                {scoreResult.target_actual !== null && scoreResult.target_actual !== undefined && (
                  <span className="score-page__actual">
                    Thực tế:{' '}
                    <strong className={scoreResult.target_actual === 1 ? 'is-default' : 'is-safe'}>
                      {scoreResult.target_actual === 1 ? 'Đã default' : 'Không default'}
                    </strong>
                  </span>
                )}
              </div>
            )}
          </div>

          <div className="card score-page__applicant-card">
            <span className="eyebrow">Hồ sơ applicant</span>
            {applicant && (
              <dl className="score-page__applicant-grid">
                <div>
                  <dt>SK_ID_CURR</dt>
                  <dd className="mono-num">#{applicant.sk_id_curr}</dd>
                </div>
                <div>
                  <dt>Giới tính</dt>
                  <dd>{applicant.code_gender}</dd>
                </div>
                <div>
                  <dt>Học vấn</dt>
                  <dd>{applicant.name_education_type}</dd>
                </div>
                <div>
                  <dt>Tình trạng hôn nhân</dt>
                  <dd>{applicant.name_family_status}</dd>
                </div>
                <div>
                  <dt>Thu nhập</dt>
                  <dd className="mono-num">{Math.round(applicant.amt_income_total).toLocaleString('vi-VN')}</dd>
                </div>
                <div>
                  <dt>Số tiền vay</dt>
                  <dd className="mono-num">{Math.round(applicant.amt_credit).toLocaleString('vi-VN')}</dd>
                </div>
              </dl>
            )}

            {scoreResult && (
              <>
                <ReasonCodeList reasons={scoreResult.reasons.slice(0, 3)} title="Top 3 lý do" />
                <Link to="/insights" className="btn btn--ghost score-page__insights-link">
                  Xem đầy đủ SHAP waterfall & reason codes →
                </Link>
              </>
            )}
          </div>
        </section>
      )}

      {!applicant && !isScoring && (
        <div className="container">
          <p className="score-page__hint">Gõ 1 SK_ID_CURR ở ô tìm kiếm phía trên (ví dụ 100002) để bắt đầu.</p>
        </div>
      )}
    </main>
  );
}
