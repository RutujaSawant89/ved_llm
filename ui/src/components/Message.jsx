import AnswerAnalysis from './AnswerAnalysis'

export default function Message({ role, content, sources, trust, verification }) {
  const isAssistant = role === 'assistant'

  return (
    <div className="message">
      <div className="message-label">{isAssistant ? 'TrustLM' : 'You'}</div>
      <div className={`message-content${isAssistant ? ' assistant' : ''}`}>
        {content}
      </div>
      {isAssistant && (
        <div style={{ marginTop: '16px' }}>
          <AnswerAnalysis
            trust={trust}
            verification={verification}
            sources={sources}
          />
        </div>
      )}
    </div>
  )
}
