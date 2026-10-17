import { useEffect, useRef } from 'react'
import Message from './Message'

export default function MessageList({ messages, loading, onSuggestionClick }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  if (!messages || messages.length === 0) {
    return (
      <div className="chat-empty">
        <h1>Ved.ai</h1>
        <p>Ask a question about your uploaded documents</p>
        <div className="chat-empty-suggestions">
          <button onClick={() => onSuggestionClick?.("Summarize the key information and findings in my documents.")}>
            Summarize the key information and findings in my documents.
          </button>
          <button onClick={() => onSuggestionClick?.("What are the main requirements and procedures described in the text?")}>
            What are the main requirements and procedures described in the text?
          </button>
          <button onClick={() => onSuggestionClick?.("What evidence or factual data is provided for the core topic?")}>
            What evidence or factual data is provided for the core topic?
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="chat-messages">
      {messages.map((msg, i) => (
        <Message
          key={i}
          role={msg.role}
          content={msg.content}
          sources={msg.sources}
          trust={msg.trust}
          verification={msg.verification}
        />
      ))}
      {loading && (
        <div className="message">
          <div className="message-label">TrustLM</div>
          <div className="message-loading">···</div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  )
}
