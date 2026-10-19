import { useEffect, useRef, useState } from 'react'
import { ArrowDown } from 'lucide-react'
import Message from './Message'
import ThinkingIndicator from './ThinkingIndicator'

export default function MessageList({
  messages = [],
  loading = false,
  onRegenerate,
}) {
  const bottomRef = useRef(null)
  const scrollContainerRef = useRef(null)
  const [showScrollBottom, setShowScrollBottom] = useState(false)

  // Scroll to bottom on new messages or loading state
  useEffect(() => {
    if (!showScrollBottom) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, loading, showScrollBottom])

  const handleScroll = () => {
    const el = scrollContainerRef.current
    if (!el) return
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120
    setShowScrollBottom(!isAtBottom)
  }

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    setShowScrollBottom(false)
  }

  return (
    <div
      className="chat-scroll-area"
      ref={scrollContainerRef}
      onScroll={handleScroll}
    >
      <div className="chat-content-width">
        {messages.map((msg, i) => {
          const isLast = i === messages.length - 1
          return (
            <Message
              key={i}
              role={msg.role}
              content={msg.content}
              sources={msg.sources}
              trust={msg.trust}
              verification={msg.verification}
              generator={msg.generator}
              onRegenerate={isLast ? onRegenerate : undefined}
              canRegenerate={isLast && !loading}
            />
          )
        })}

        {/* Dynamic Multi-Stage Thinking Indicator */}
        {loading && <ThinkingIndicator />}

        <div ref={bottomRef} style={{ height: 28 }} />
      </div>

      {showScrollBottom && (
        <button
          type="button"
          className="scroll-to-bottom-btn"
          onClick={scrollToBottom}
          title="Scroll to latest messages"
          aria-label="Scroll to latest"
        >
          <ArrowDown size={15} />
        </button>
      )}
    </div>
  )
}
