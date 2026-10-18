import { useDocuments } from '../hooks/useDocuments'
import UploadZone from '../components/UploadZone'
import DocumentTable from '../components/DocumentTable'
import { formatNumber } from '../utils/format'

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

  const totalChunks = stats?.total_chunks ?? documents.reduce((acc, d) => acc + (d.chunks || 0), 0)
  const collectionName = stats?.collection_name || 'trustlm_knowledge_base'
  const vectorStore = 'ChromaDB'

  return (
    <div className="kb-container">
      <UploadZone
        onUpload={uploadFile}
        uploading={uploading}
        uploadProgress={uploadProgress}
        currentFile={currentFile}
      />

      <DocumentTable
        documents={documents}
        onDelete={deleteDocument}
        onClearAll={clearAllDocuments}
      />

      <div className="kb-footer">
        {formatNumber(totalChunks)} chunks · {collectionName} · {vectorStore}
      </div>
    </div>
  )
}
