import { useState, useRef, useEffect } from 'react'
import {
  ArrowUp,
  SlidersHorizontal,
  Paperclip,
  Mic,
  Square,
  Loader2,
} from 'lucide-react'
import { api } from '../utils/api'

export default function ChatInput({
  onSend,
  disabled,
  settings,
  onUpdateSettings,
  onDocumentUploaded,
  isCentered = false,
}) {
  const [text, setText] = useState('')
  const [showSettings, setShowSettings] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [speechSupported, setSpeechSupported] = useState(false)

  const textareaRef = useRef(null)
  const fileInputRef = useRef(null)
  const recognitionRef = useRef(null)

  // Initialize Speech Recognition if supported in browser
  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition
    if (SpeechRecognition) {
      setSpeechSupported(true)
      const recognition = new SpeechRecognition()
      recognition.continuous = true
      recognition.interimResults = true
      recognition.lang = 'en-US'

      recognition.onresult = (event) => {
        let currentTranscript = ''
        for (let i = 0; i < event.results.length; i++) {
          currentTranscript += event.results[i][0].transcript
        }
        setText(currentTranscript)
      }

      recognition.onerror = (event) => {
        console.warn('Speech recognition error:', event.error)
        setIsRecording(false)
      }

      recognition.onend = () => {
        setIsRecording(false)
      }

      recognitionRef.current = recognition
    }
  }, [])

  // Auto-resize textarea height
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = Math.min(textarea.scrollHeight, 180) + 'px'
    }
  }, [text])

  const handleSend = () => {
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    if (isRecording) {
      recognitionRef.current?.stop()
      setIsRecording(false)
    }
    onSend(trimmed)
    setText('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const toggleRecording = () => {
    if (!speechSupported) {
      alert(
        'Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.'
      )
      return
    }

    if (isRecording) {
      recognitionRef.current?.stop()
      setIsRecording(false)
    } else {
      try {
        recognitionRef.current?.start()
        setIsRecording(true)
      } catch (err) {
        console.error('Failed to start speech recognition:', err)
        setIsRecording(false)
      }
    }
  }

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setIsUploading(true)
    try {
      await api.uploadDocument(file)
      onDocumentUploaded?.(file.name)
    } catch (err) {
      alert(`Document upload failed: ${err.message || 'Unknown error'}`)
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  return (
    <div className={`chat-composer-container ${isCentered ? 'centered-mode' : ''}`}>
      <div className="chat-composer-box">
        {/* Hidden File Input */}
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept=".pdf,.txt,.md,.pptx"
          onChange={handleFileChange}
        />

        {/* Text Input */}
        <textarea
          ref={textareaRef}
          className="chat-composer-textarea"
          placeholder={
            isRecording
              ? 'Listening... speak now'
              : 'Ask a question about your documents, verify claims, or inspect evidence...'
          }
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={isCentered ? 2 : 1}
          disabled={disabled}
        />

        {/* Action Toolbar */}
        <div className="chat-composer-actions">
          <div className="composer-tools-left">
            {/* Attachment Button */}
            <button
              type="button"
              className="composer-tool-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled || isUploading}
              title="Attach and index document (.pdf, .txt, .md, .pptx)"
              aria-label="Attach document"
            >
              {isUploading ? (
                <Loader2 size={15} className="spinner" />
              ) : (
                <Paperclip size={15} />
              )}
              {isUploading && <span style={{ fontSize: '11px' }}>Indexing...</span>}
            </button>

            {/* Voice Input Button */}
            <button
              type="button"
              className={`composer-tool-btn ${isRecording ? 'recording' : ''}`}
              onClick={toggleRecording}
              disabled={disabled}
              title={
                !speechSupported
                  ? 'Speech recognition unavailable in this browser'
                  : isRecording
                  ? 'Stop voice recording'
                  : 'Start voice dictation'
              }
              aria-label="Voice input"
            >
              {isRecording ? <Square size={14} /> : <Mic size={15} />}
              {isRecording && <span style={{ fontSize: '11px' }}>Listening</span>}
            </button>

            {/* Generation Settings Toggle */}
            <button
              type="button"
              className={`composer-tool-btn ${showSettings ? 'active' : ''}`}
              onClick={() => setShowSettings(!showSettings)}
              title="Generation Settings (Temperature, Top-K, Top-P)"
              aria-label="Generation settings"
            >
              <SlidersHorizontal size={14} />
            </button>
          </div>

          <div className="composer-tools-right">
            <button
              type="button"
              className="composer-send-btn"
              onClick={handleSend}
              disabled={!text.trim() || disabled}
              title="Send question (Enter)"
              aria-label="Send question"
            >
              <ArrowUp size={16} strokeWidth={2.2} />
            </button>
          </div>
        </div>

        {/* Generation Settings Drawer */}
        {showSettings && (
          <div className="composer-settings-drawer">
            <div className="composer-setting-item">
              <div className="composer-setting-label">
                <span>Temperature</span>
                <span>{settings.temperature}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1.5"
                step="0.05"
                value={settings.temperature}
                onChange={(e) =>
                  onUpdateSettings?.({
                    ...settings,
                    temperature: parseFloat(e.target.value),
                  })
                }
                className="composer-slider"
              />
            </div>

            <div className="composer-setting-item">
              <div className="composer-setting-label">
                <span>Top-K Chunks</span>
                <span>{settings.top_k}</span>
              </div>
              <input
                type="range"
                min="1"
                max="10"
                step="1"
                value={settings.top_k}
                onChange={(e) =>
                  onUpdateSettings?.({
                    ...settings,
                    top_k: parseInt(e.target.value, 10),
                  })
                }
                className="composer-slider"
              />
            </div>

            <div className="composer-setting-item">
              <div className="composer-setting-label">
                <span>Top-P Nucleus</span>
                <span>{settings.top_p}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={settings.top_p}
                onChange={(e) =>
                  onUpdateSettings?.({
                    ...settings,
                    top_p: parseFloat(e.target.value),
                  })
                }
                className="composer-slider"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
