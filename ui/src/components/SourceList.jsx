export default function SourceList({ sources }) {
  if (!sources || sources.length === 0) return null

  return (
    <div className="source-list">
      {sources.map((s, i) => (
        <div key={i} className="source-item">
          <div className="source-item-header">
            {s.document_name}
            {s.page_number != null && ` — Page ${s.page_number}`}
            {` — ${s.relevance}% relevance`}
          </div>
          {s.text && (
            <div className="source-item-preview">{s.text}</div>
          )}
        </div>
      ))}
    </div>
  )
}
