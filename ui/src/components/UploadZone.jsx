import { useRef, useState } from 'react'
import { ArrowUpFromLine, FileText, Loader2 } from 'lucide-react'
import ProgressBar from './ProgressBar'

export default function UploadZone({
  onUpload,
  uploading,
  uploadProgress,
  currentFile,
}) {
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = () => {
    setIsDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)
    if (uploading) return
    const file = e.dataTransfer.files?.[0]
    if (file) onUpload(file)
  }

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      onUpload(file)
      e.target.value = ''
    }
  }

  return (
    <div
      className={`documents-upload-card ${isDragOver ? 'drag-over' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => !uploading && fileInputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          fileInputRef.current?.click()
        }
      }}
      title="Click or drag files here to upload and index"
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: 'none' }}
        accept=".txt,.pdf,.md,.pptx"
      />

      {uploading ? (
        <div className="upload-progress-box">
          <Loader2 size={24} className="spinner" style={{ color: 'var(--text-primary)' }} />
          <div className="documents-upload-heading">
            Indexing {currentFile || 'document'}...
          </div>
          <div className="documents-upload-sub">
            Parsing sections, generating embeddings, and storing in ChromaDB
          </div>
          <div style={{ width: '100%', marginTop: 8 }}>
            <ProgressBar progress={uploadProgress} height={3} />
          </div>
        </div>
      ) : (
        <>
          <div className="documents-upload-icon-wrap">
            <ArrowUpFromLine size={20} />
          </div>
          <div className="documents-upload-heading">
            Drop documents here or click to browse
          </div>
          <div className="documents-upload-sub">
            Supports PDF, TXT, Markdown, and PPTX · Automatically chunked and embedded for RAG
          </div>
        </>
      )}
    </div>
  )
}
