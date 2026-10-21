import { useState } from 'react'
import {
  FileSearch,
  Cpu,
  CheckCircle,
  AlertTriangle,
  Quote,
  Sparkles,
} from 'lucide-react'
import MessageList from '../components/MessageList'
import ChatInput from '../components/ChatInput'

export default function ChatPage({
  messages = [],
  loading = false,
  settings,
  setSettings,
  docStats,
  sendMessage,
  regenerateLast,
  refreshStats,
}) {
  const [toast, setToast] = useState(null)

  const handleDocumentUploaded = (fileName) => {
    refreshStats?.()
    setToast(`Document "${fileName}" successfully indexed in knowledge base.`)
    setTimeout(() => setToast(null), 4000)
  }

  const fiveSuggestions = [
    {
      label: 'Key Findings',
      query: 'Summarize the core findings, conclusions, and takeaways in the documents.',
      icon: FileSearch,
    },
    {
      label: 'Methodology',
      query: 'Explain the core architecture, mechanisms, and methodology described.',
      icon: Cpu,
    },
    {
      label: 'Claim Verification',
      query: 'Extract all verifiable factual claims and statistics from the uploaded materials.',
      icon: CheckCircle,
    },
    {
      label: 'Limitations',
      query: 'What constraints, open questions, or limitations are highlighted?',
      icon: AlertTriangle,
    },
    {
      label: 'Source Citations',
      query: 'List the primary citations, document references, and source passages.',
      icon: Quote,
    },
  ]

  const isEmpty = messages.length === 0

  return (
    <div className="chat-container">
      {/* Toast Notification */}
      {toast && (
        <div className="toast-container">
          <div className="toast-item">
            <span>{toast}</span>
          </div>
        </div>
      )}

      {isEmpty ? (
        /* ================= Centered Empty State ================= */
        <div className="chat-scroll-area centered-empty-wrapper">
          <div className="chat-centered-hero">
            {/* Hero Headings */}
            <h1 className="chat-empty-title">Research with context.</h1>
            <p className="chat-empty-subtitle">
              Ask questions, explore your documents, and inspect real evidence with Ved.ai reliability scoring.
            </p>

            {/* Centered Message Composer */}
            <div className="centered-composer-wrap">
              <ChatInput
                onSend={sendMessage}
                disabled={loading}
                settings={settings}
                onUpdateSettings={setSettings}
                onDocumentUploaded={handleDocumentUploaded}
                isCentered={true}
              />
            </div>

            {/* 5 Minimal Suggestion Chips Below Centered Input */}
            <div className="five-suggestions-row">
              {fiveSuggestions.map((s, idx) => {
                const Icon = s.icon
                return (
                  <button
                    key={idx}
                    type="button"
                    className="suggestion-chip"
                    onClick={() => sendMessage(s.query)}
                    title={s.query}
                  >
                    <Icon size={13} className="suggestion-chip-icon" />
                    <span>{s.label}</span>
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      ) : (
        /* ================= Active Conversation View ================= */
        <>
          <MessageList
            messages={messages}
            loading={loading}
            onSuggestionClick={sendMessage}
            docStats={docStats}
            onRegenerate={regenerateLast}
          />

          {/* Docked Floating Composer at bottom */}
          <ChatInput
            onSend={sendMessage}
            disabled={loading}
            settings={settings}
            onUpdateSettings={setSettings}
            onDocumentUploaded={handleDocumentUploaded}
            isCentered={false}
          />
        </>
      )}
    </div>
  )
}
