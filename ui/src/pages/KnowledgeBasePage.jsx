import { useDocuments } from '../hooks/useDocuments'
import UploadZone from '../components/UploadZone'
import DocumentTable from '../components/DocumentTable'
import { formatNumber } from '../utils/format'
import { Database, Layers, Cpu } from 'lucide-react'

export default function KnowledgeBasePage() {
  const {
    documents,
    stats,
    uploading,
    uploadProgress,
    currentFile,
    uploadFile,
    deleteDocument,
    clearAllDocuments,
  } = useDocuments()

  const totalChunks =
    stats?.total_chunks ??
    documents.reduce((acc, d) => acc + (d.chunks || d.chunk_count || 0), 0)
  const collectionName = stats?.collection_name || 'trustlm_knowledge_base'
  const generatorName = stats?.generator || 'TrustLM'

  return (
    <div className="documents-container">
      <div className="documents-content-width">
        {/* Header */}
        <div className="documents-header">
          <h1 className="documents-title">Document Grounding Library</h1>
          <p className="documents-subtitle">
            Upload research papers, technical manuals, or domain documents.
            TrustLM indexes their contents to provide cited evidence and verified claims.
          </p>
        </div>

        {/* Upload Zone */}
        <UploadZone
          onUpload={uploadFile}
          uploading={uploading}
          uploadProgress={uploadProgress}
          currentFile={currentFile}
        />

        {/* Document Table */}
        <DocumentTable
          documents={documents}
          onDelete={deleteDocument}
          onClearAll={clearAllDocuments}
        />

        {/* Knowledge Base Statistics & Storage Footer */}
        <div className="documents-stats-footer">
          <div className="documents-stats-items">
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Layers size={14} style={{ color: 'var(--text-muted)' }} />
              <strong>{formatNumber(totalChunks)}</strong> indexed chunks
            </span>
            <span style={{ color: 'var(--border)' }}>·</span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Database size={14} style={{ color: 'var(--text-muted)' }} />
              ChromaDB ({collectionName})
            </span>
            <span style={{ color: 'var(--border)' }}>·</span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Cpu size={14} style={{ color: 'var(--text-muted)' }} />
              {generatorName}
            </span>
          </div>
          <div>
            <span>RAG Grounding Enabled</span>
          </div>
        </div>
      </div>
    </div>
  )
}
