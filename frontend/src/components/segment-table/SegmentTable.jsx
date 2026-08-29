import './segment-table.css';

const pct = (x) => `${(x * 100).toFixed(1)}%`;
const MIN_GROUP_SIZE = 1000; // dưới mức này số liệu chỉ là nhiễu — xem ml/src/policy.py

/** Phân khúc theo thuộc tính được bảo vệ (giới tính / tuổi).
 *
 *  Hai cột cần đọc kỹ:
 *   - "Lệch calibration" = PD trung bình − bad rate thật. Gần 0 nghĩa là model
 *     dự đoán ĐÚNG mức rủi ro cho nhóm đó (không thiên vị theo nghĩa thống kê).
 *   - "Tỉ lệ được duyệt" chênh nhau = tác động chính sách không đồng đều, kể cả
 *     khi calibration hoàn hảo. Đây là thứ 4/5ths rule đo.
 */
export default function SegmentTable({ rows, ratio, label }) {
  if (!rows?.length) return null;

  const passes = ratio == null ? null : ratio >= 0.8;

  return (
    <div className="segment-table__block">
      <div className="segment-table__head">
        <h3 className="segment-table__title">{label}</h3>
        {ratio != null && (
          <span className={`badge ${passes ? 'badge--low' : 'badge--high'}`}>
            AIR {ratio.toFixed(3)} — {passes ? 'đạt' : 'KHÔNG đạt'} 4/5ths
          </span>
        )}
      </div>

      <div className="segment-table__scroll">
        <table className="segment-table">
          <thead>
            <tr>
              <th scope="col">Nhóm</th>
              <th scope="col">n</th>
              <th scope="col">Bad rate thật</th>
              <th scope="col">PD trung bình</th>
              <th scope="col">Lệch calibration</th>
              <th scope="col">AUC</th>
              <th scope="col">Tỉ lệ được duyệt</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.group} className={r.n < MIN_GROUP_SIZE ? 'is-small' : undefined} style={{ '--stagger': i }}>
                <th scope="row">
                  {r.group}
                  {r.n < MIN_GROUP_SIZE && (
                    <span className="segment-table__note" title={`n < ${MIN_GROUP_SIZE}, không đủ để kết luận`}>
                      mẫu quá nhỏ
                    </span>
                  )}
                </th>
                <td className="mono-num">{r.n.toLocaleString('vi-VN')}</td>
                <td className="mono-num">{pct(r.bad_rate)}</td>
                <td className="mono-num">{pct(r.mean_pd)}</td>
                <td className="mono-num">{r.calibration_gap >= 0 ? '+' : ''}{r.calibration_gap.toFixed(4)}</td>
                <td className="mono-num">{r.auc == null ? '—' : r.auc.toFixed(4)}</td>
                <td className="mono-num">{pct(r.approval_rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
