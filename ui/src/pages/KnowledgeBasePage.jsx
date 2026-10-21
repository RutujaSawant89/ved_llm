import { useDocuments } from '../hooks/useDocuments'
import UploadZone from '../components/UploadZone'
import DocumentTable from '../components/DocumentTable'

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

  return (
    <div className="documents-container">
      <div className="documents-content-width">
        {/* Header */}
        <div className="documents-header">
          <h1 className="documents-title">Document Grounding Library</h1>
          <p className="documents-subtitle">
            Upload research papers, technical manuals, or domain documents.
            Ved.ai indexes their contents to provide cited evidence and verified claims.
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
      </div>
    </div>
  )
}
