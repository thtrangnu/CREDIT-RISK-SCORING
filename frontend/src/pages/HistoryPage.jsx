import { useEffect, useState } from 'react';
import HistoryTable from '../components/history-table/HistoryTable';
import { fetchHistory } from '../api/client';
import './history-page.css';

export default function HistoryPage() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [filterId, setFilterId] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setIsLoading(true);
    const skIdCurr = filterId.trim() ? Number(filterId.trim()) : undefined;
    fetchHistory({ skIdCurr })
      .then((res) => {
        setItems(res.items);
        setTotal(res.total);
      })
      .catch((err) => setError(err.message))
      .finally(() => setIsLoading(false));
  }, [filterId]);

  return (
    <main className="history-page">
      <section className="container history-page__hero">
        <span className="eyebrow">Audit trail</span>
        <h1 className="font-display history-page__title">Lịch sử chấm điểm.</h1>
        <p className="history-page__subtitle">
          Mỗi lần chấm điểm được ghi lại 1 dòng trong MySQL (model_version, PD, risk tier, reason codes) — phục vụ
          governance và truy vết quyết định.
        </p>
      </section>

      <section className="container history-page__toolbar">
        <input
          type="text"
          className="history-page__filter"
          placeholder="Lọc theo SK_ID_CURR…"
          value={filterId}
          onChange={(e) => setFilterId(e.target.value)}
        />
        <span className="history-page__count mono-num">{total} bản ghi</span>
      </section>

      <section className="container history-page__table-wrap">
        <div className="card">
          {error ? <p className="score-page__error">Lỗi: {error}</p> : <HistoryTable items={items} isLoading={isLoading} />}
        </div>
      </section>
    </main>
  );
}
