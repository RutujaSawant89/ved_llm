import { useRef, useState } from 'react'
import { ArrowUpFromLine } from 'lucide-react'
import ProgressBar from './ProgressBar'

export default function UploadZone({ onUpload, uploading, uploadProgress, currentFile }) {
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
      className={`upload-zone${isDragOver ? ' drag-over' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => !uploading && fileInputRef.current?.click()}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: 'none' }}
        accept=".txt,.pdf,.md,.pptx"
      />
      {uploading ? (
        <div style={{ width: '80%', textAlign: 'center' }}>
          <div className="upload-zone-text">Uploading {currentFile || 'document'}...</div>
          <ProgressBar progress={uploadProgress} height={2} />
        </div>
      ) : (
        <>
          <ArrowUpFromLine size={20} className="upload-zone-icon" />
          <div className="upload-zone-text">Drop files here or click to browse</div>
          <div className="upload-zone-hint">.txt · .pdf · .md · .pptx</div>
        </>
      )}
    </div>
  )
}
