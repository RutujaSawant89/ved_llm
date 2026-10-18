import { useState, useEffect, useCallback } from 'react'
import { api } from '../utils/api'

export function useDocuments() {
  const [documents, setDocuments] = useState([])
  const [stats, setStats] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [currentFile, setCurrentFile] = useState('')
  const [error, setError] = useState(null)

  const loadDocumentsAndStats = useCallback(async () => {
    try {
      const [docs, statsData] = await Promise.all([
        api.getDocuments().catch(() => []),
        api.getStats().catch(() => null),
      ])
      setDocuments(docs || [])
      setStats(statsData)
    } catch (err) {
      console.error('Failed to load documents from backend:', err)
    }
  }, [])

  useEffect(() => {
    loadDocumentsAndStats()
  }, [loadDocumentsAndStats])

  const uploadFile = useCallback(async (file) => {
    if (!file) return
    setUploading(true)
    setCurrentFile(file.name)
    setUploadProgress(25)
    setError(null)

    try {
      const pTimer = setInterval(() => {
        setUploadProgress((p) => (p < 85 ? p + 15 : p))
      }, 150)

      await api.uploadDocument(file)
      clearInterval(pTimer)
      setUploadProgress(100)

      await loadDocumentsAndStats()
    } catch (err) {
      setError(err.message || 'Upload failed')
      alert(`Upload error: ${err.message || 'Failed to upload document'}`)
    } finally {
      setTimeout(() => {
        setUploading(false)
        setUploadProgress(0)
        setCurrentFile('')
      }, 400)
    }
  }, [loadDocumentsAndStats])

  const deleteDocument = useCallback(async (docId) => {
    try {
      await api.deleteDocument(docId)
      await loadDocumentsAndStats()
    } catch (err) {
      console.error('Failed to delete doc:', err)
      await loadDocumentsAndStats()
    }
  }, [loadDocumentsAndStats])

  const clearAllDocuments = useCallback(async () => {
    try {
      await api.clearDocuments()
      setDocuments([])
      await loadDocumentsAndStats()
    } catch (err) {
      console.error('Failed to clear documents:', err)
    }
  }, [loadDocumentsAndStats])

  return {
    documents,
    stats,
    uploading,
    uploadProgress,
    currentFile,
    error,
    uploadFile,
    deleteDocument,
    clearAllDocuments,
    refresh: loadDocumentsAndStats,
  }
}
