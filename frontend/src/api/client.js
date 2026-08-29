const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export function searchApplicants(query = '') {
  const params = new URLSearchParams();
  if (query) params.set('q', query);
  return request(`/api/applicants?${params.toString()}`);
}

export function scoreApplicant(skIdCurr) {
  return request(`/api/score/${skIdCurr}`, { method: 'POST' });
}

export function fetchHistory({ skIdCurr, limit = 50, offset = 0 } = {}) {
  const params = new URLSearchParams({ limit, offset });
  if (skIdCurr) params.set('sk_id_curr', skIdCurr);
  return request(`/api/history?${params.toString()}`);
}

export function fetchInsights() {
  return request('/api/insights');
}
