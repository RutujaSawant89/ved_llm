import { useEffect, useState } from 'react'
import { api } from '../utils/api'

export default function TopBar({ title }) {
  const [docCount, setDocCount] = useState(0)

  useEffect(() => {
    api.getStats()
      .then((s) => {
        if (s?.total_chunks != null) setDocCount(s.total_chunks)
      })
      .catch(() => {})
  }, [])

  return (
    <header className="topbar">
      <span className="topbar-title">{title}</span>
      <div className="topbar-status">
        <span style={{ color: 'var(--green)' }}>RAG Active</span>
        <span style={{ color: 'var(--text-muted)' }}>·</span>
        <span style={{ color: 'var(--text-muted)' }}>{docCount} chunks indexed</span>
      </div>
    </header>
  )
}
