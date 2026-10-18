const BASE = ''

export async function fetchApi(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}

export const api = {
  query: (question, top_k = 5) =>
    fetchApi('/api/query', { method: 'POST', body: JSON.stringify({ question, top_k }) }),

  uploadDocument: async (file) => {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${BASE}/api/documents/upload`, { method: 'POST', body: form })
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText)
    return res.json()
  },

  deleteDocument: (docId) =>
    fetchApi(`/api/documents/${docId}`, { method: 'DELETE' }),

  getDocuments: () => fetchApi('/api/documents'),
  clearDocuments: () => fetchApi('/api/documents/clear', { method: 'POST' }),

  getStats: () => fetchApi('/api/documents/stats'),

  getModelInfo: () => fetchApi('/api/model/info'),
  getModelStatus: () => fetchApi('/api/model/status'),

  getConfig: () => fetchApi('/api/config'),
  updateGeneration: (data) =>
    fetchApi('/api/config/generation', { method: 'PATCH', body: JSON.stringify(data) }),

  startTraining: (data) =>
    fetchApi('/api/train/start', { method: 'POST', body: JSON.stringify(data) }),
  getTrainStatus: () => fetchApi('/api/train/status'),
  stopTraining: () => fetchApi('/api/train/stop', { method: 'POST' }),

  getHealth: () => fetchApi('/api/health'),
}
