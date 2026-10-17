import { useState } from 'react'
import Accordion from './Accordion'

export default function ConfigAccordion({ config, onUpdateGeneration }) {
  const [genValues, setGenValues] = useState({
    temperature: config?.generation?.temperature ?? 0.7,
    top_k: config?.generation?.top_k ?? 40,
    top_p: config?.generation?.top_p ?? 0.9,
    max_new_tokens: config?.generation?.max_new_tokens ?? 128,
  })

  const handleGenChange = (field, val) => {
    const updated = { ...genValues, [field]: val }
    setGenValues(updated)
    onUpdateGeneration?.(updated)
  }

  const sections = [
    { key: 'model', title: 'ModelConfig', data: config?.model },
    { key: 'tokenizer', title: 'TokenizerConfig', data: config?.tokenizer },
    { key: 'training', title: 'TrainingConfig', data: config?.training },
    { key: 'generation', title: 'GenerationConfig', data: config?.generation, editable: true },
    { key: 'verification', title: 'VerificationConfig', data: config?.verification },
    { key: 'trust', title: 'TrustConfig', data: config?.trust },
    { key: 'memory', title: 'MemoryConfig', data: config?.memory },
    { key: 'unlearning', title: 'UnlearningConfig', data: config?.unlearning },
    { key: 'rag', title: 'RAGConfig', data: config?.rag },
  ]

  return (
    <div>
      <div className="section-heading">Configuration</div>
      <div className="section-subheading">Runtime hyperparameters and architecture settings</div>

      <div className="config-sections">
        {sections.map(({ key, title, data, editable }) => {
          if (!data && !editable) return null

          return (
            <Accordion key={key} title={title}>
              <div className="config-grid">
                {editable ? (
                  <>
                    <div className="config-key">temperature</div>
                    <div className="config-val">
                      <input
                        type="number"
                        step="0.05"
                        min="0"
                        max="2"
                        value={genValues.temperature}
                        onChange={(e) => handleGenChange('temperature', parseFloat(e.target.value) || 0)}
                        style={{
                          background: 'var(--bg-2)',
                          border: '1px solid var(--border)',
                          borderRadius: 'var(--radius-sm)',
                          color: 'var(--text-secondary)',
                          fontFamily: 'var(--mono)',
                          fontSize: 'var(--text-xs)',
                          padding: '2px 6px',
                          width: '70px',
                        }}
                      />
                    </div>

                    <div className="config-key">top_k</div>
                    <div className="config-val">
                      <input
                        type="number"
                        min="1"
                        max="100"
                        value={genValues.top_k}
                        onChange={(e) => handleGenChange('top_k', parseInt(e.target.value) || 1)}
                        style={{
                          background: 'var(--bg-2)',
                          border: '1px solid var(--border)',
                          borderRadius: 'var(--radius-sm)',
                          color: 'var(--text-secondary)',
                          fontFamily: 'var(--mono)',
                          fontSize: 'var(--text-xs)',
                          padding: '2px 6px',
                          width: '70px',
                        }}
                      />
                    </div>

                    <div className="config-key">top_p</div>
                    <div className="config-val">
                      <input
                        type="number"
                        step="0.05"
                        min="0"
                        max="1"
                        value={genValues.top_p}
                        onChange={(e) => handleGenChange('top_p', parseFloat(e.target.value) || 0)}
                        style={{
                          background: 'var(--bg-2)',
                          border: '1px solid var(--border)',
                          borderRadius: 'var(--radius-sm)',
                          color: 'var(--text-secondary)',
                          fontFamily: 'var(--mono)',
                          fontSize: 'var(--text-xs)',
                          padding: '2px 6px',
                          width: '70px',
                        }}
                      />
                    </div>

                    <div className="config-key">max_new_tokens</div>
                    <div className="config-val">
                      <input
                        type="number"
                        min="1"
                        max="1024"
                        value={genValues.max_new_tokens}
                        onChange={(e) => handleGenChange('max_new_tokens', parseInt(e.target.value) || 1)}
                        style={{
                          background: 'var(--bg-2)',
                          border: '1px solid var(--border)',
                          borderRadius: 'var(--radius-sm)',
                          color: 'var(--text-secondary)',
                          fontFamily: 'var(--mono)',
                          fontSize: 'var(--text-xs)',
                          padding: '2px 6px',
                          width: '70px',
                        }}
                      />
                    </div>

                    {data && Object.entries(data).map(([k, v]) => {
                      if (['temperature', 'top_k', 'top_p', 'max_new_tokens'].includes(k)) return null
                      return (
                        <div key={k} style={{ display: 'contents' }}>
                          <div className="config-key">{k}</div>
                          <div className="config-val">{typeof v === 'boolean' ? (v ? 'true' : 'false') : String(v)}</div>
                        </div>
                      )
                    })}
                  </>
                ) : (
                  data && Object.entries(data).map(([k, v]) => (
                    <div key={k} style={{ display: 'contents' }}>
                      <div className="config-key">{k}</div>
                      <div className="config-val">
                        {Array.isArray(v)
                          ? `[${v.join(', ')}]`
                          : typeof v === 'boolean'
                          ? (v ? 'true' : 'false')
                          : String(v)}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </Accordion>
          )
        })}
      </div>
    </div>
  )
}
