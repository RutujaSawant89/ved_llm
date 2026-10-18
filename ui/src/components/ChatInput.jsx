import { useState } from 'react'
import { ArrowUp, SlidersHorizontal } from 'lucide-react'

export default function ChatInput({ onSend, disabled, showSettings, onToggleSettings }) {
  const [text, setText] = useState('')

  const handleSend = () => {
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setText('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="chat-input-area">
      <div className="chat-input-wrapper">
        <button
          type="button"
          className="chat-input-settings-btn"
          onClick={onToggleSettings}
          title="Generation Settings"
          style={showSettings ? { color: 'var(--text-primary)' } : {}}
        >
          <SlidersHorizontal size={15} />
        </button>
        <textarea
          className="chat-input"
          placeholder="Message TrustLM..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={disabled}
        />
        <button
          type="button"
          className="chat-send-btn"
          onClick={handleSend}
          disabled={!text.trim() || disabled}
          title="Send"
        >
          <ArrowUp size={16} strokeWidth={2} />
        </button>
      </div>
    </div>
  )
}
