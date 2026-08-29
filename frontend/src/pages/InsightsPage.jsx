import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import ShapWaterfall from '../components/shap-waterfall/ShapWaterfall';
import CutoffTable from '../components/cutoff-table/CutoffTable';
import SegmentTable from '../components/segment-table/SegmentTable';
import ReasonCodeList from '../components/reason-code-list/ReasonCodeList';
import StatTile from '../components/ui/StatTile';
import { fetchInsights } from '../api/client';
import { useScoreContext } from '../context/ScoreContext';
import './insights-page.css';

export default function InsightsPage() {
  const { applicant, scoreResult } = useScoreContext();
  const [insights, setInsights] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchInsights().then(setInsights).catch((err) => setError(err.message));
  }, []);

  const metrics = insights?.model_metrics;
  const policy = insights?.policy;
  const fairness = insights?.fairness;
  // Dòng ở ngưỡng duyệt tham chiếu -> con số headline của cả trang.
  const refRow = policy?.table.find(
    (r) => Math.abs(r.approval_rate - policy.reference_approval_rate) < 1e-9,
  );
  const pct1 = (x) => `${(x * 100).toFixed(1)}%`;

  return (
    <main className="insights-page">
      <section className="container insights-page__hero">
        <span className="eyebrow">Insights</span>
        <h1 className="font-display insights-page__title">Model đã học điều gì.</h1>
        <p className="insights-page__subtitle">
          SHAP toàn cục để hiểu model trên toàn quần thể, và soi từng applicant để giải trình quyết định — đúng yêu
          cầu adverse action trong lending.
        </p>
      </section>

      {metrics && (
        <section className="container insights-page__bento">
          {refRow && (
            <StatTile
              emphasis
              label={`Giảm tổn thất tín dụng @ duyệt ${pct1(policy.reference_approval_rate)}`}
              value={`−${pct1(refRow.bad_rate_reduction)}`}
              delta={`bad rate ${pct1(metrics.default_rate)} → ${pct1(refRow.bad_rate_approved)} · chặn ${pct1(refRow.bad_captured)} ca vỡ nợ`}
            />
          )}
          <StatTile label="AUC — baseline → engineered" value={metrics.engineered.auc.toFixed(5)} delta={`+${metrics.auc_delta.toFixed(5)} so với ${metrics.baseline.auc.toFixed(5)}`} />
          <StatTile label="ECE — trước → sau hiệu chỉnh" value={metrics.engineered.ece_calibrated.toFixed(5)} delta={`từ ${metrics.engineered.ece_raw.toFixed(5)}`} />
          <StatTile label="Số feature" value={metrics.n_features} delta={`${metrics.n_train_rows.toLocaleString('vi-VN')} applicant train`} />
          <StatTile label="Default rate quần thể" value={`${(metrics.default_rate * 100).toFixed(2)}%`} />
          {fairness && (
            <StatTile
              label="Adverse impact ratio (giới tính / tuổi)"
              value={`${fairness.adverse_impact_ratio_gender.toFixed(3)} / ${fairness.adverse_impact_ratio_age.toFixed(3)}`}
              delta="ngưỡng 4/5ths = 0.800"
            />
          )}
        </section>
      )}

      {policy && (
        <section className="container insights-page__policy">
          <h2 className="insights-page__section-title font-display">Điểm số dịch sang quyết định.</h2>
          <p className="insights-page__caption">
            AUC nói model xếp hạng tốt cỡ nào, không nói dùng nó thì được gì. Bảng dưới trả lời câu hỏi thật của
            bộ phận risk: ở một tỉ lệ duyệt cho trước, tỉ lệ vỡ nợ trong nhóm được duyệt giảm bao nhiêu.
          </p>
          <div className="card">
            <CutoffTable table={policy.table} referenceRate={policy.reference_approval_rate} />
          </div>
        </section>
      )}

      {fairness && (
        <section className="container insights-page__fairness">
          <h2 className="insights-page__section-title font-display">Ai chịu tác động, và có đồng đều không.</h2>
          <p className="insights-page__caption">
            Giới tính và tuổi là thuộc tính được bảo vệ theo ECOA. Model calibrate rất đều giữa các nhóm (lệch PD so
            với bad rate thật đều dưới 0.2pp, trừ nhóm dưới 25 tuổi lệch +1.9pp) — nhưng tỉ lệ được duyệt thì chênh
            nhau nhiều. “Chênh vì rủi ro chênh thật” không phải biện hộ hợp lệ về pháp lý: 4/5ths rule đo tác động,
            không đo ý định. Số liệu ở tỉ lệ duyệt {pct1(fairness.reference_approval_rate)}.
          </p>
          <div className="card">
            <SegmentTable rows={fairness.by_gender} ratio={fairness.adverse_impact_ratio_gender} label="Theo giới tính" />
            <SegmentTable rows={fairness.by_age_band} ratio={fairness.adverse_impact_ratio_age} label="Theo nhóm tuổi" />
          </div>
        </section>
      )}

      <section className="container insights-page__local">
        <h2 className="insights-page__section-title font-display">
          Giải trình theo applicant {applicant && <span className="mono-num insights-page__applicant-id">#{applicant.sk_id_curr}</span>}
        </h2>

        {!scoreResult ? (
          <div className="card empty-state">
            Chưa chọn applicant nào. <Link to="/">Sang trang Score</Link> để chấm điểm 1 applicant trước.
          </div>
        ) : (
          <div className="insights-page__local-grid">
            <div className="card">
              <ShapWaterfall baseValue={scoreResult.base_value} reasons={scoreResult.reasons} rawMargin={scoreResult.raw_margin} />
            </div>
            <div className="card">
              <ReasonCodeList reasons={scoreResult.reasons} />
            </div>
          </div>
        )}
      </section>

      <section className="container insights-page__global">
        <h2 className="insights-page__section-title font-display">Top feature quan trọng nhất (toàn quần thể)</h2>
        <p className="insights-page__caption">
          Xếp theo mean|SHAP| trên mẫu 5,000 applicant. {insights && `${insights.monotonic_features.length} feature có monotonic constraint (Block 3)`} —
          feature nào trùng cả hai cột được đánh dấu, cho thấy domain reasoning khớp với thứ model thực sự học.
        </p>
        {error && <p className="score-page__error">Lỗi: {error}</p>}
        {insights && (
          <ol className="insights-page__importance-list">
            {insights.global_importance.map((item, i) => {
              const isMonotonic = insights.monotonic_features.includes(item.feature);
              const widthPct = (item.mean_abs_shap / insights.global_importance[0].mean_abs_shap) * 100;
              return (
                <li key={item.feature} className="insights-page__importance-row" style={{ '--stagger': i }}>
                  <span className="mono-num insights-page__importance-rank">{String(i + 1).padStart(2, '0')}</span>
                  <span className="insights-page__importance-label">
                    {item.label}
                    {isMonotonic && (
                      <span className="badge badge--curated insights-page__monotonic-flag" title="Có monotonic constraint ở Block 3">
                        monotonic
                      </span>
                    )}
                  </span>
                  <span className="insights-page__importance-bar-track">
                    <span className="insights-page__importance-bar" style={{ width: `${widthPct}%` }} />
                  </span>
                  <span className="mono-num insights-page__importance-value">{item.mean_abs_shap.toFixed(4)}</span>
                </li>
              );
            })}
          </ol>
        )}
      </section>
    </main>
  );
}
