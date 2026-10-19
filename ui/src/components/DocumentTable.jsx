import { useState, useMemo } from 'react'
import { FileText, Trash2, Search, AlertTriangle } from 'lucide-react'
import { formatDate } from '../utils/format'

export default function DocumentTable({ documents = [], onDelete, onClearAll }) {
  const [confirmId, setConfirmId] = useState(null)
  const [confirmClearAll, setConfirmClearAll] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')

  const filteredDocs = useMemo(() => {
    if (!searchQuery.trim()) return documents
    const q = searchQuery.toLowerCase()
    return documents.filter((d) => {
      const name = d.name || d.document_name || d.id || ''
      return name.toLowerCase().includes(q)
    })
  }, [documents, searchQuery])

  if (!documents || documents.length === 0) {
    return (
      <div className="doc-section">
        <div className="doc-section-header">
          <div className="doc-section-left">
            <span className="doc-section-title">Indexed Documents</span>
            <span className="doc-section-count">0</span>
          </div>
        </div>
        <div
          style={{
            color: 'var(--text-muted)',
            fontSize: 'var(--text-sm)',
            padding: '36px 20px',
            textAlign: 'center',
          }}
        >
          No documents have been indexed yet. Drag and drop files above to populate your knowledge base.
        </div>
      </div>
    )
  }

  return (
    <div className="doc-section">
      <div className="doc-section-header">
        <div className="doc-section-left">
          <span className="doc-section-title">Indexed Documents</span>
          <span className="doc-section-count">{documents.length}</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {/* Search filter */}
          <input
            type="text"
            className="doc-filter-input"
            placeholder="Filter documents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />

          {/* Clear all confirmation */}
          {confirmClearAll ? (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontSize: 'var(--text-xs)',
              }}
            >
              <span style={{ color: 'var(--red)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <AlertTriangle size={13} />
                Clear entire database?
              </span>
              <button
                type="button"
                className="doc-confirm-yes"
                style={{
                  background: 'var(--red)',
                  color: '#fff',
                  border: 'none',
                  padding: '4px 8px',
                  borderRadius: 'var(--radius-xs)',
                  cursor: 'pointer',
                  fontSize: '11px',
                }}
                onClick={() => {
                  onClearAll?.()
                  setConfirmClearAll(false)
                }}
              >
                Confirm
              </button>
              <button
                type="button"
                className="doc-confirm-no"
                style={{
                  background: 'var(--bg-3)',
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border)',
                  padding: '4px 8px',
                  borderRadius: 'var(--radius-xs)',
                  cursor: 'pointer',
                  fontSize: '11px',
                }}
                onClick={() => setConfirmClearAll(false)}
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              type="button"
              className="doc-delete-btn"
              onClick={() => setConfirmClearAll(true)}
              title="Delete all documents from ChromaDB knowledge base"
            >
              Clear All
            </button>
          )}
        </div>
      </div>

      <table className="doc-table">
        <thead>
          <tr>
            <th>Document Name</th>
            <th>Chunks</th>
            <th>Ingested</th>
            <th style={{ textAlign: 'right' }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {filteredDocs.map((doc) => {
            const isConfirming = confirmId === doc.id
            const docName = doc.name || doc.document_name || doc.id
            const chunkCount = doc.chunks || doc.chunk_count || 1

            if (isConfirming) {
              return (
                <tr key={doc.id}>
                  <td colSpan={4}>
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '6px 0',
                        fontSize: 'var(--text-xs)',
                      }}
                    >
                      <span>
                        Delete all {chunkCount} chunks of "{docName}"?
                      </span>
                      <div style={{ display: 'flex', gap: '8px' }}>
                        <button
                          type="button"
                          style={{
                            background: 'var(--red)',
                            color: '#fff',
                            border: 'none',
                            padding: '3px 8px',
                            borderRadius: 'var(--radius-xs)',
                            cursor: 'pointer',
                            fontSize: '11px',
                          }}
                          onClick={() => {
                            onDelete(doc.id)
                            setConfirmId(null)
                          }}
                        >
                          Confirm Delete
                        </button>
                        <button
                          type="button"
                          style={{
                            background: 'var(--bg-3)',
                            color: 'var(--text-secondary)',
                            border: '1px solid var(--border)',
                            padding: '3px 8px',
                            borderRadius: 'var(--radius-xs)',
                            cursor: 'pointer',
                            fontSize: '11px',
                          }}
                          onClick={() => setConfirmId(null)}
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  </td>
                </tr>
              )
            }

            return (
              <tr key={doc.id}>
                <td>
                  <div className="doc-name-cell">
                    <FileText size={15} style={{ color: 'var(--text-muted)' }} />
                    <span>{docName}</span>
                  </div>
                </td>
                <td>
                  <span
                    style={{
                      fontFamily: 'var(--mono)',
                      fontSize: 'var(--text-xs)',
                      background: 'var(--bg-3)',
                      padding: '2px 6px',
                      borderRadius: 'var(--radius-xs)',
                    }}
                  >
                    {chunkCount} chunks
                  </span>
                </td>
                <td>
                  <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                    {formatDate(doc.ingested_at || doc.date || new Date().toISOString())}
                  </span>
                </td>
                <td style={{ textAlign: 'right' }}>
                  <button
                    type="button"
                    className="doc-delete-btn"
                    onClick={() => setConfirmId(doc.id)}
                    title={`Delete ${docName}`}
                    aria-label={`Delete ${docName}`}
                  >
                    <Trash2 size={14} />
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
