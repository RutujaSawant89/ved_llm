import { formatNumber } from '../utils/format'

export default function MetricsLine({ metrics }) {
  if (!metrics) return null

  const {
    step = 0,
    totalSteps = 5000,
    loss = null,
    perplexity = null,
    tokens = null,
  } = metrics

  const formattedTokens = tokens != null
    ? tokens >= 1e6
      ? `${(tokens / 1e6).toFixed(1)}M`
      : `${(tokens / 1e3).toFixed(0)}k`
    : null

  return (
    <div className="metrics-line">
      <span className="label">Step </span>
      <span className="value">{formatNumber(step)}</span>
      <span className="label"> / {formatNumber(totalSteps)}</span>

      {loss != null && (
        <>
          <span className="sep">·</span>
          <span className="label">Loss </span>
          <span className="value">{typeof loss === 'number' ? loss.toFixed(4) : loss}</span>
        </>
      )}

      {perplexity != null && (
        <>
          <span className="sep">·</span>
          <span className="label">Perplexity </span>
          <span className="value">{typeof perplexity === 'number' ? perplexity.toFixed(2) : perplexity}</span>
        </>
      )}

      {formattedTokens != null && (
        <>
          <span className="sep">·</span>
          <span className="value">{formattedTokens}</span>
          <span className="label"> tokens</span>
        </>
      )}
    </div>
  )
}
