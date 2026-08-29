import './cutoff-table.css';

const pct = (x) => `${(x * 100).toFixed(1)}%`;

/** Bảng trade-off duyệt/rủi ro. Dòng ở ngưỡng tham chiếu được làm nổi bật vì
 *  đó là con số được trích ra làm headline ở tile phía trên. */
export default function CutoffTable({ table, referenceRate }) {
  if (!table?.length) return null;

  return (
    <div className="cutoff-table__scroll">
      <table className="cutoff-table">
        <caption className="cutoff-table__caption">
          Tính trên OOF đã hiệu chỉnh. “Duyệt X%” = duyệt X% hồ sơ có PD thấp nhất.
        </caption>
        <thead>
          <tr>
            <th scope="col">Tỉ lệ duyệt</th>
            <th scope="col">Ngưỡng PD</th>
            <th scope="col">Bad rate nhóm duyệt</th>
            <th scope="col">Giảm tổn thất</th>
            <th scope="col">Ca vỡ nợ bị chặn</th>
          </tr>
        </thead>
        <tbody>
          {table.map((row, i) => {
            const isRef = Math.abs(row.approval_rate - referenceRate) < 1e-9;
            return (
              <tr
                key={row.approval_rate}
                className={isRef ? 'is-reference' : undefined}
                style={{ '--stagger': i }}
              >
                <th scope="row" className="mono-num">
                  {pct(row.approval_rate)}
                  {isRef && <span className="cutoff-table__ref-tag">tham chiếu</span>}
                </th>
                <td className="mono-num">{row.pd_cutoff.toFixed(4)}</td>
                <td className="mono-num">{pct(row.bad_rate_approved)}</td>
                <td className="mono-num cutoff-table__gain">
                  {row.bad_rate_reduction > 0 ? `−${pct(row.bad_rate_reduction)}` : '—'}
                </td>
                <td className="mono-num">{pct(row.bad_captured)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
