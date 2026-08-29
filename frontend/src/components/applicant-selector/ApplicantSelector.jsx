import { useEffect, useRef, useState } from 'react';
import { searchApplicants } from '../../api/client';
import './applicant-selector.css';

const DEBOUNCE_MS = 250;

export default function ApplicantSelector({ onSelect, selectedId, disabled }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const debounceRef = useRef(null);
  const rootRef = useRef(null);

  useEffect(() => {
    window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(async () => {
      setIsLoading(true);
      try {
        const rows = await searchApplicants(query);
        setResults(rows);
      } catch {
        setResults([]);
      } finally {
        setIsLoading(false);
      }
    }, DEBOUNCE_MS);
    return () => window.clearTimeout(debounceRef.current);
  }, [query]);

  useEffect(() => {
    function handleClickOutside(event) {
      if (rootRef.current && !rootRef.current.contains(event.target)) setIsOpen(false);
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="applicant-selector" ref={rootRef}>
      <label className="eyebrow" htmlFor="applicant-search">
        Chọn applicant (SK_ID_CURR)
      </label>
      <input
        id="applicant-search"
        type="text"
        className="applicant-selector__input"
        placeholder={selectedId ? `Đang chọn #${selectedId} — tìm applicant khác…` : 'Tìm theo SK_ID_CURR, ví dụ 100002…'}
        value={query}
        disabled={disabled}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => setIsOpen(true)}
      />

      {isOpen && (
        <ul className="applicant-selector__panel" role="listbox">
          {isLoading && <li className="applicant-selector__hint">Đang tìm…</li>}
          {!isLoading && results.length === 0 && (
            <li className="applicant-selector__hint">Không tìm thấy applicant nào.</li>
          )}
          {!isLoading &&
            results.map((row) => (
              <li key={row.sk_id_curr}>
                <button
                  type="button"
                  className="applicant-selector__option"
                  onClick={() => {
                    onSelect(row);
                    setIsOpen(false);
                    setQuery('');
                  }}
                >
                  <span className="mono-num applicant-selector__id">#{row.sk_id_curr}</span>
                  <span className="applicant-selector__meta">
                    {row.code_gender} · {row.name_education_type} · thu nhập{' '}
                    {Math.round(row.amt_income_total).toLocaleString('vi-VN')}
                  </span>
                  {row.target !== null && (
                    <span className={`applicant-selector__target ${row.target === 1 ? 'is-default' : 'is-safe'}`}>
                      {row.target === 1 ? 'TARGET=1 (default thật)' : 'TARGET=0'}
                    </span>
                  )}
                </button>
              </li>
            ))}
        </ul>
      )}
    </div>
  );
}
