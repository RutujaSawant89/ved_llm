import { useState } from 'react'
import { formatDate } from '../utils/format'

export default function DocumentTable({ documents, onDelete, onClearAll }) {
  const [confirmId, setConfirmId] = useState(null)
  const [confirmClearAll, setConfirmClearAll] = useState(false)

  if (!documents || documents.length === 0) {
    return (
      <div className="doc-section">
        <div className="doc-section-header">
          <span className="doc-section-title">Documents</span>
          <span className="doc-section-count">0</span>
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: 'var(--text-sm)', padding: '24px 0' }}>
          No documents ingested yet. Drop files above to populate the knowledge base.
        </div>
      </div>
    )
  }

  return (
    <div className="doc-section">
      <div className="doc-section-header" style={{ justifyContent: 'space-between', width: '100%' }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
          <span className="doc-section-title">Documents</span>
          <span className="doc-section-count">{documents.length}</span>
        </div>

        {confirmClearAll ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: 'var(--text-xs)' }}>
            <span style={{ color: 'var(--red)' }}>Clear entire database?</span>
            <button
              type="button"
              className="doc-confirm-yes"
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
            title="Delete all documents from ChromaDB"
          >
            Clear All
          </button>
        )}
      </div>

      <table className="doc-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Chunks</th>
            <th>Ingested</th>
            <th style={{ textAlign: 'right' }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {documents.map((doc) => {
            const isConfirming = confirmId === doc.id

            if (isConfirming) {
              return (
                <tr key={doc.id}>
                  <td colSpan={4}>
                    <div className="doc-confirm">
                      <span>
                        Delete all {doc.chunks || doc.chunk_count || 1} chunks of "{doc.name}"?
                      </span>
                      <button
                        type="button"
                        className="doc-confirm-yes"
                        onClick={() => {
                          onDelete(doc.id)
                          setConfirmId(null)
                        }}
                      >
                        Confirm
                      </button>
                      <button
                        type="button"
                        className="doc-confirm-no"
                        onClick={() => setConfirmId(null)}
                      >
                        Cancel
                      </button>
                    </div>
                  </td>
                </tr>
              )
            }

            return (
              <tr key={doc.id}>
                <td className="doc-name">{doc.name || doc.document_name || doc.id}</td>
                <td>{doc.chunks || doc.chunk_count || 0}</td>
                <td>{formatDate(doc.ingested_at || doc.date || new Date().toISOString())}</td>
                <td style={{ textAlign: 'right' }}>
                  <button
                    type="button"
                    className="doc-delete-btn"
                    onClick={() => setConfirmId(doc.id)}
                  >
                    Delete
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
