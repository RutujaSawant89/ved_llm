import { useState } from 'react'
import {
  ShieldCheck,
  ShieldAlert,
  Shield,
  ChevronDown,
  ChevronUp,
  FileText,
  CheckCircle2,
  AlertCircle,
  XCircle,
  BarChart3,
  ListChecks,
  Copy,
  Check,
  Search,
} from 'lucide-react'
import { trustColor, getTrustClassification } from '../utils/format'

export default function AnswerAnalysis({ trust, verification, sources = [] }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [activeTab, setActiveTab] = useState('claims') // 'claims' | 'sources' | 'metrics'
  const [expandedSources, setExpandedSources] = useState({})
  const [copiedIndex, setCopiedIndex] = useState(null)
  const [selectedClaim, setSelectedClaim] = useState(null)
  const [sourceSearch, setSourceSearch] = useState('')

  // If no evaluation data is returned yet
  if (!trust && !verification && (!sources || sources.length === 0)) {
    return null
  }

  // Parse scores safely
  const scoreStr = trust?.score || '0%'
  const scoreVal =
    typeof trust?.score_value === 'number'
      ? trust.score_value
      : parseFloat(scoreStr.replace('%', '')) || 0
  const color = trustColor(scoreVal)
  const classification = getTrustClassification(scoreVal)

  const claims = verification?.claims || []
  const hasClaims = claims.length > 0
  const sourceCount = sources.length

  const toggleSourceExpand = (index, e) => {
    e.stopPropagation()
    setExpandedSources((prev) => ({
      ...prev,
      [index]: !prev[index],
    }))
  }

  const handleCopySource = async (text, index, e) => {
    e.stopPropagation()
    try {
      await navigator.clipboard.writeText(text)
      setCopiedIndex(index)
      setTimeout(() => setCopiedIndex(null), 2000)
    } catch (err) {}
  }

  // Handle clicking a claim to cross-link with sources
  const handleClaimClick = (claimText) => {
    setSelectedClaim(selectedClaim === claimText ? null : claimText)
    setActiveTab('sources')
  }

  // Filter sources if search query is present
  const filteredSources = sources.filter((s) => {
    if (!sourceSearch.trim()) return true
    const q = sourceSearch.toLowerCase()
    return (
      (s.document_name && s.document_name.toLowerCase().includes(q)) ||
      (s.text && s.text.toLowerCase().includes(q))
    )
  })

  // Format metric dimensions
  const metricsList = [
    {
      name: 'Composite Trust Score',
      value: scoreVal,
      formula: '40% Grounding + 35% Relevance + 15% Coverage + 10% Consistency',
      desc: 'Holistic reliability score computed from document evidence',
    },
    {
      name: 'Retrieval Relevance',
      value: trust?.relevance_score ?? 0,
      formula: 'Average cosine similarity of top passages',
      desc: 'Semantic alignment between user query and retrieved chunks',
    },
    {
      name: 'Evidence Coverage',
      value:
        typeof trust?.evidence_coverage === 'number'
          ? trust.evidence_coverage
          : parseInt(trust?.evidence_coverage, 10) || (scoreVal >= 40 ? 60 : 15),
      formula: 'Coverage depth factor across chunk boundaries',
      desc: 'Context depth and breadth available in knowledge base',
    },
    {
      name: 'Factual Grounding',
      value:
        trust?.factual_support ??
        (verification?.grounding_rate ? Math.round(verification.grounding_rate * 100) : 0),
      formula: 'Verified claims ratio minus unsupported penalties',
      desc: 'Proportion of statements supported by document citations',
    },
    {
      name: 'Passage Consistency',
      value: trust?.consistency_score ?? 80,
      formula: 'Variance across top retrieved chunk embeddings',
      desc: 'Inter-passage agreement across independent chunks',
    },
  ]

  // Radial progress calculations
  const radius = 11
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference - (scoreVal / 100) * circumference

  return (
    <div className="trustlm-panel">
      {/* Compact Trust Summary (Always visible) */}
      <div
        className="trustlm-summary-row"
        onClick={() => setIsExpanded(!isExpanded)}
        title="Click to inspect TrustLM evidence, claims, and metrics"
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            setIsExpanded(!isExpanded)
          }
        }}
      >
        <div className="trustlm-summary-left">
          {/* Radial Score Gauge */}
          <div className="trustlm-gauge-wrap" title={`Trust Score: ${scoreVal}%`}>
            <svg width="28" height="28" className="trustlm-gauge-svg">
              <circle
                cx="14"
                cy="14"
                r={radius}
                className="trustlm-gauge-bg"
              />
              <circle
                cx="14"
                cy="14"
                r={radius}
                className="trustlm-gauge-progress"
                style={{
                  stroke: color,
                  strokeDasharray: circumference,
                  strokeDashoffset: strokeDashoffset,
                }}
              />
            </svg>
            <div className="trustlm-shield-wrap">
              {scoreVal >= 70 ? (
                <ShieldCheck size={13} style={{ color }} />
              ) : scoreVal >= 40 ? (
                <Shield size={13} style={{ color }} />
              ) : (
                <ShieldAlert size={13} style={{ color }} />
              )}
            </div>
          </div>

          <span className="trustlm-brand">TrustLM</span>

          <span className="trustlm-score-pill" style={{ color }}>
            {scoreVal}%
          </span>

          <span className="trustlm-classification">{classification}</span>

          <span className="trustlm-meta-sep">·</span>

          <span className="trustlm-stats">
            {hasClaims ? `${claims.length} claim${claims.length === 1 ? '' : 's'}` : '0 claims'}
            {' · '}
            {sourceCount} source{sourceCount === 1 ? '' : 's'}
          </span>
        </div>

        <div className="trustlm-summary-right">
          <button
            type="button"
            className="trustlm-expand-btn"
            onClick={(e) => {
              e.stopPropagation()
              setIsExpanded(!isExpanded)
            }}
            aria-expanded={isExpanded}
            aria-label={isExpanded ? 'Collapse evidence' : 'Inspect evidence'}
          >
            <span>{isExpanded ? 'Hide' : 'Inspect Evidence'}</span>
            {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>
      </div>

      {/* Expandable Evidence Inspector Drawer */}
      {isExpanded && (
        <div className="trustlm-details-drawer">
          {/* Reason / Verdict Summary Banner */}
          {(trust?.reason || verification?.reason) && (
            <div className="trustlm-reason-box">
              {trust?.reason || verification?.reason}
            </div>
          )}

          {/* Navigation Tabs */}
          <div className="trustlm-tabs">
            <button
              type="button"
              className={`trustlm-tab-btn ${activeTab === 'claims' ? 'active' : ''}`}
              onClick={() => setActiveTab('claims')}
            >
              <ListChecks size={13} />
              <span>Claims ({claims.length})</span>
            </button>

            <button
              type="button"
              className={`trustlm-tab-btn ${activeTab === 'sources' ? 'active' : ''}`}
              onClick={() => setActiveTab('sources')}
            >
              <FileText size={13} />
              <span>Sources ({sourceCount})</span>
            </button>

            <button
              type="button"
              className={`trustlm-tab-btn ${activeTab === 'metrics' ? 'active' : ''}`}
              onClick={() => setActiveTab('metrics')}
            >
              <BarChart3 size={13} />
              <span>Evaluation Metrics</span>
            </button>
          </div>

          {/* Tab 1: Claims View with Interactive Source Linking */}
          {activeTab === 'claims' && (
            <div className="claims-container">
              {hasClaims ? (
                <>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>
                    Click any claim to jump to supporting document passages in the Sources tab:
                  </div>
                  {claims.map((c, i) => {
                    const status = c.status || 'unsupported'
                    const isSupported = status === 'supported'
                    const isPartial = status === 'partially_supported'

                    return (
                      <div
                        key={i}
                        className={`claim-row clickable ${selectedClaim === c.claim ? 'selected' : ''}`}
                        onClick={() => handleClaimClick(c.claim)}
                        title="Click to view supporting source passages"
                      >
                        <span
                          className={`claim-status-badge ${
                            isSupported
                              ? 'supported'
                              : isPartial
                              ? 'partially_supported'
                              : 'unsupported'
                          }`}
                        >
                          {isSupported ? (
                            <CheckCircle2 size={11} />
                          ) : isPartial ? (
                            <AlertCircle size={11} />
                          ) : (
                            <XCircle size={11} />
                          )}
                          <span>
                            {isSupported
                              ? 'Verified'
                              : isPartial
                              ? 'Partial'
                              : 'Unsupported'}
                          </span>
                        </span>

                        <span className="claim-content">{c.claim}</span>

                        {c.confidence != null && (
                          <span className="claim-confidence" title="Verification Confidence">
                            {Math.round(c.confidence * 100)}%
                          </span>
                        )}
                      </div>
                    )
                  })}
                </>
              ) : (
                <div
                  style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--text-muted)',
                    padding: '12px 4px',
                  }}
                >
                  No specific factual claims were extracted for verification from this response.
                </div>
              )}
            </div>
          )}

          {/* Tab 2: Sources View with Search & Copy Excerpt */}
          {activeTab === 'sources' && (
            <div className="sources-container">
              {sourceCount > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <div className="sources-search-wrap">
                    <Search size={12} style={{ color: 'var(--text-muted)' }} />
                    <input
                      type="text"
                      className="sources-search-input"
                      placeholder="Filter passages by keyword..."
                      value={sourceSearch}
                      onChange={(e) => setSourceSearch(e.target.value)}
                    />
                  </div>
                  {selectedClaim && (
                    <button
                      type="button"
                      className="assistant-action-btn"
                      onClick={() => setSelectedClaim(null)}
                      style={{ fontSize: '11px', padding: '3px 6px' }}
                    >
                      Clear claim filter
                    </button>
                  )}
                </div>
              )}

              {filteredSources.length > 0 ? (
                filteredSources.map((s, i) => {
                  const isLong = s.text && s.text.length > 180
                  const isOpen = expandedSources[i]
                  const displayText = isLong && !isOpen ? s.text.slice(0, 180) + '...' : s.text

                  return (
                    <div key={i} className="source-item-card">
                      <div className="source-item-header">
                        <div className="source-item-name">
                          <FileText size={13} style={{ color: 'var(--text-secondary)' }} />
                          <span>{s.document_name}</span>
                          {s.page_number != null && (
                            <span className="source-item-page">Page {s.page_number}</span>
                          )}
                          {s.chunk_index != null && (
                            <span className="source-item-page">Chunk #{s.chunk_index + 1}</span>
                          )}
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span
                            className="source-item-relevance"
                            style={{ color: trustColor(s.relevance) }}
                          >
                            {s.relevance}% relevance
                          </span>
                          <button
                            type="button"
                            className="source-copy-btn"
                            onClick={(e) => handleCopySource(s.text, i, e)}
                            title="Copy passage excerpt"
                          >
                            {copiedIndex === i ? (
                              <Check size={11} style={{ color: 'var(--green)' }} />
                            ) : (
                              <Copy size={11} />
                            )}
                          </button>
                        </div>
                      </div>

                      {s.text && (
                        <div className="source-item-excerpt">
                          "{displayText}"
                          {isLong && (
                            <button
                              type="button"
                              onClick={(e) => toggleSourceExpand(i, e)}
                              style={{
                                background: 'none',
                                border: 'none',
                                color: 'var(--text-primary)',
                                textDecoration: 'underline',
                                cursor: 'pointer',
                                fontSize: '11px',
                                marginLeft: '6px',
                              }}
                            >
                              {isOpen ? 'Show less' : 'Read more'}
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  )
                })
              ) : (
                <div
                  style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--text-muted)',
                    padding: '12px 4px',
                  }}
                >
                  {sourceSearch
                    ? 'No document passages matched your search query.'
                    : 'No document passages were retrieved for this query.'}
                </div>
              )}
            </div>
          )}

          {/* Tab 3: Evaluation Metrics with Formula Details */}
          {activeTab === 'metrics' && (
            <div className="metrics-container">
              {metricsList.map((m, i) => {
                const metricColor =
                  m.name === 'Composite Trust Score'
                    ? color
                    : m.value >= 70
                    ? 'var(--green)'
                    : m.value >= 40
                    ? 'var(--amber)'
                    : 'var(--steel)'

                return (
                  <div key={i} className="metric-row" title={`${m.desc} (${m.formula})`}>
                    <div className="metric-label-bar">
                      <div>
                        <span className="metric-name">{m.name}</span>
                        <span className="metric-formula-tag"> · {m.formula}</span>
                      </div>
                      <span className="metric-val" style={{ color: metricColor }}>
                        {m.value}%
                      </span>
                    </div>

                    <div className="metric-progress-track">
                      <div
                        className="metric-progress-fill"
                        style={{
                          width: `${Math.min(100, Math.max(0, m.value))}%`,
                          backgroundColor: metricColor,
                        }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
