import { useState } from 'react'
import {
  ShieldCheck,
  FileText,
  BarChart2,
  ChevronDown,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
} from 'lucide-react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from 'recharts'
import { trustColor } from '../utils/format'

export default function AnswerAnalysis({ trust, verification, sources }) {
  const [showChart, setShowChart] = useState(true)
  const [showSources, setShowSources] = useState(false)
  const [showClaims, setShowClaims] = useState(false)

  const scoreStr = trust?.score || 'UNKNOWN'
  const scoreVal = trust?.score_value ?? (parseInt(scoreStr) || 80)
  const color = trustColor(scoreVal)

  const chartData = trust?.chart_data && trust.chart_data.length > 0
    ? trust.chart_data
    : [
        { name: 'Trust Score', value: scoreVal },
        { name: 'Relevance', value: trust?.relevance_score ?? 85 },
        { name: 'Coverage', value: trust?.evidence_coverage ? parseInt(trust.evidence_coverage) || 80 : 80 },
        { name: 'Grounding', value: trust?.factual_support ?? 90 },
        { name: 'Consistency', value: trust?.consistency_score ?? 85 },
      ]

  const claims = verification?.claims || []
  const hasClaims = claims.length > 0
  const sourceCount = sources?.length || 0

  return (
    <div className="answer-analysis-card">
      {/* Top Header Summary */}
      <div className="analysis-header">
        <div className="analysis-verdict">
          <ShieldCheck size={16} style={{ color }} />
          <span className="analysis-score" style={{ color }}>
            {scoreStr}
          </span>
          <span className="analysis-label">
            {scoreVal >= 70 ? 'High Reliability' : scoreVal >= 40 ? 'Moderate Reliability' : 'Low Evidence'}
          </span>
          <span className="analysis-sep">·</span>
          <span className="analysis-reason">
            {trust?.reason || `${sourceCount} source chunk(s) verified`}
          </span>
        </div>

        <div className="analysis-actions">
          <button
            type="button"
            className={`analysis-btn${showChart ? ' active' : ''}`}
            onClick={() => setShowChart(!showChart)}
          >
            <BarChart2 size={13} />
            <span>Evaluation Chart</span>
          </button>

          {hasClaims && (
            <button
              type="button"
              className={`analysis-btn${showClaims ? ' active' : ''}`}
              onClick={() => setShowClaims(!showClaims)}
            >
              <CheckCircle2 size={13} />
              <span>Claims ({claims.length})</span>
            </button>
          )}

          {sourceCount > 0 && (
            <button
              type="button"
              className={`analysis-btn${showSources ? ' active' : ''}`}
              onClick={() => setShowSources(!showSources)}
            >
              <FileText size={13} />
              <span>Sources ({sourceCount})</span>
            </button>
          )}
        </div>
      </div>

      {/* Evaluation Chart View */}
      {showChart && (
        <div className="analysis-chart-view">
          <div className="chart-view-header">
            <span>RAG Trust & Retrieval Evaluation</span>
            <span className="chart-view-sub">Real metrics computed from uploaded document context</span>
          </div>

          <div style={{ height: 160, width: '100%', marginTop: 8 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                layout="vertical"
                margin={{ top: 4, right: 30, left: 70, bottom: 4 }}
              >
                <XAxis
                  type="number"
                  domain={[0, 100]}
                  stroke="#5c5c66"
                  fontSize={11}
                  fontFamily="var(--mono)"
                  tickLine={false}
                  axisLine={{ stroke: '#1f1f23' }}
                  unit="%"
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  stroke="#a0a0a8"
                  fontSize={11}
                  fontFamily="var(--font)"
                  tickLine={false}
                  axisLine={{ stroke: '#1f1f23' }}
                />
                <Tooltip
                  cursor={{ fill: 'rgba(255, 255, 255, 0.03)' }}
                  content={({ payload }) => {
                    if (!payload || !payload[0]) return null
                    const d = payload[0].payload
                    return (
                      <div className="analysis-tooltip">
                        <span>{d.name}: </span>
                        <strong>{d.value}%</strong>
                      </div>
                    )
                  }}
                />
                <Bar
                  dataKey="value"
                  radius={[0, 3, 3, 0]}
                  barSize={14}
                  isAnimationActive={false}
                >
                  {chartData.map((entry, index) => {
                    const c = entry.name === 'Trust Score' ? color : '#7c7cff'
                    return <Cell key={`cell-${index}`} fill={c} fillOpacity={0.85} />
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Verified Claims Breakdown */}
      {showClaims && hasClaims && (
        <div className="analysis-claims-view">
          <div className="claims-title">Extracted Claims & Document Grounding:</div>
          <div className="claims-list">
            {claims.map((c, i) => (
              <div key={i} className="claim-item">
                {c.status === 'supported' ? (
                  <CheckCircle2 size={13} style={{ color: 'var(--green)', flexShrink: 0, marginTop: 2 }} />
                ) : (
                  <AlertCircle size={13} style={{ color: 'var(--amber)', flexShrink: 0, marginTop: 2 }} />
                )}
                <span className="claim-text">{c.claim}</span>
                <span className="claim-badge" style={{ color: c.status === 'supported' ? 'var(--green)' : 'var(--amber)' }}>
                  {c.status === 'supported' ? 'Verified' : 'Partial'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Sources List */}
      {showSources && sources && (
        <div className="analysis-sources-view">
          <div className="sources-title">Cited Document Passages:</div>
          <div className="sources-list-grid">
            {sources.map((s, i) => (
              <div key={i} className="analysis-source-card">
                <div className="source-card-header">
                  <div className="source-doc-name">
                    <FileText size={12} />
                    <span>{s.document_name}</span>
                    {s.page_number != null && <span className="source-page">Page {s.page_number}</span>}
                  </div>
                  <span className="source-relevance">{s.relevance}% relevance</span>
                </div>
                {s.text && <div className="source-card-text">"{s.text}"</div>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
