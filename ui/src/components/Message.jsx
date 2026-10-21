import { FileText } from 'lucide-react'
import AnswerAnalysis from './AnswerAnalysis'
import MessageActions from './MessageActions'

function renderFormattedText(text) {
  if (!text) return ''

  // Split by double newlines into blocks
  const blocks = text.split('\n\n')

  return blocks.map((block, bIdx) => {
    const trimmed = block.trim()

    // 1. Heading block (### Heading)
    if (trimmed.startsWith('### ')) {
      return (
        <h4 key={bIdx} className="assistant-subheading">
          {trimmed.replace(/^###\s+/, '')}
        </h4>
      )
    }

    // 2. Bullet list block (- item or * item)
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      const items = trimmed.split('\n').filter((line) => line.trim().startsWith('- ') || line.trim().startsWith('* '))
      return (
        <ul key={bIdx} className="assistant-bullet-list">
          {items.map((item, iIdx) => {
            const cleanItem = item.replace(/^[-*]\s+/, '')
            return (
              <li key={iIdx}>
                {renderInlineMarkdown(cleanItem)}
              </li>
            )
          })}
        </ul>
      )
    }

    // 3. Normal paragraph
    return (
      <p key={bIdx}>
        {renderInlineMarkdown(trimmed)}
      </p>
    )
  })
}

function renderInlineMarkdown(str) {
  // Replace bold **text** and citations (Doc, Page X)
  const parts = []
  let remaining = str
  let key = 0

  // Regex to match bold **text** or citations (filename, Page \d+)
  const regex = /(\*\*([^*]+)\*\*|\(([^)]+\.[a-zA-Z0-9]+(?:,\s*Page\s*\d+)?)\))/g
  let match
  let lastIndex = 0

  while ((match = regex.exec(str)) !== null) {
    // Add text before match
    if (match.index > lastIndex) {
      parts.push(str.substring(lastIndex, match.index))
    }

    if (match[2]) {
      // Bold **text**
      parts.push(<strong key={key++}>{match[2]}</strong>)
    } else if (match[3]) {
      // Citation (doc, page)
      const cleanCitation = match[3].replace(/,\s*Page\s*/i, ' · p.')
      parts.push(
        <span key={key++} className="inline-citation-tag" title={match[3]}>
          <FileText size={10.5} style={{ opacity: 0.7 }} />
          <span>{cleanCitation}</span>
        </span>
      )
    }

    lastIndex = regex.lastIndex
  }

  if (lastIndex < str.length) {
    parts.push(str.substring(lastIndex))
  }

  return parts.length > 0 ? parts : str
}

export default function Message({
  role,
  content,
  sources,
  trust,
  verification,
  generator,
  onRegenerate,
  canRegenerate,
}) {
  const isAssistant = role === 'assistant'

  if (!isAssistant) {
    return (
      <div className="message-row user">
        <div className="user-message-bubble">{content}</div>
      </div>
    )
  }

  return (
    <div className="message-row assistant">
      <div className="assistant-message-container">
        {/* Assistant Identification Header */}
        <div className="assistant-header">
          <div className="assistant-avatar">V</div>
          <span>{generator && !generator.includes('TrustLM') ? generator : 'Ved.ai'}</span>
        </div>

        {/* Structured Message Content */}
        <div className="assistant-body">
          {renderFormattedText(content)}
        </div>

        {/* Answer-Specific TrustLM Reliability & Evidence Panel */}
        <AnswerAnalysis
          trust={trust}
          verification={verification}
          sources={sources}
        />

        {/* Action Row */}
        <MessageActions
          content={content}
          onRegenerate={onRegenerate}
          canRegenerate={canRegenerate}
        />
      </div>
    </div>
  )
}
