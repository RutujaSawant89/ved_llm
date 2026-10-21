import { useState, useEffect } from 'react'
import { Search, Brain, ShieldCheck, Check, Sparkles } from 'lucide-react'

export default function ThinkingIndicator() {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const startTime = Date.now()
    const timer = setInterval(() => {
      setElapsed(((Date.now() - startTime) / 1000).toFixed(1))
    }, 100)
    return () => clearInterval(timer)
  }, [])

  const currentPhase =
    elapsed < 1.5 ? 0 : elapsed < 3.2 ? 1 : 2

  const phases = [
    {
      title: 'Vector Retrieval',
      desc: 'Searching ChromaDB embeddings for relevant passages',
      icon: Search,
    },
    {
      title: 'Contextual Synthesis',
      desc: 'Formulating grounded answer from retrieved context',
      icon: Brain,
    },
    {
      title: 'Ved.ai Evaluation',
      desc: 'Extracting factual claims & verifying reliability',
      icon: ShieldCheck,
    },
  ]

  return (
    <div className="message-row assistant">
      <div className="assistant-message-container">
        <div className="assistant-header">
          <div className="assistant-avatar">V</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span>Ved.ai Research Engine</span>
            <span style={{ color: 'var(--border-focus)' }}>·</span>
            <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--mono)', fontSize: '11px' }}>
              {elapsed}s elapsed
            </span>
          </div>
        </div>

        {/* Multi-Stage Stepper Card */}
        <div className="thinking-stepper-card">
          <div className="thinking-stepper-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Sparkles size={13} style={{ color: 'var(--text-secondary)' }} />
              <span className="thinking-stepper-title">
                {phases[currentPhase].desc}...
              </span>
            </div>
            <div className="thinking-badge">Processing</div>
          </div>

          <div className="thinking-phases-list">
            {phases.map((p, idx) => {
              const isDone = idx < currentPhase
              const isActive = idx === currentPhase
              const Icon = p.icon

              return (
                <div
                  key={idx}
                  className={`thinking-phase-item ${
                    isDone ? 'done' : isActive ? 'active' : 'pending'
                  }`}
                >
                  <div className="thinking-phase-icon-wrap">
                    {isDone ? (
                      <Check size={12} style={{ color: 'var(--green)' }} />
                    ) : (
                      <Icon size={12} />
                    )}
                  </div>
                  <div className="thinking-phase-content">
                    <span className="thinking-phase-name">{p.title}</span>
                    <span className="thinking-phase-desc">{p.desc}</span>
                  </div>
                  {isActive && <div className="thinking-phase-spinner" />}
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
