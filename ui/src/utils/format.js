export function formatNumber(n) {
  if (n == null) return '0'
  return n.toLocaleString()
}

export function formatDate(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

export function trustColor(score) {
  let num = NaN
  if (typeof score === 'number') {
    num = score
  } else if (typeof score === 'string') {
    num = parseFloat(score.replace('%', ''))
  }
  if (isNaN(num)) return 'var(--text-muted)'
  if (num >= 70) return 'var(--green)'
  if (num >= 40) return 'var(--amber)'
  return 'var(--red)'
}

export function getTrustClassification(score) {
  let num = NaN
  if (typeof score === 'number') {
    num = score
  } else if (typeof score === 'string') {
    num = parseFloat(score.replace('%', ''))
  }
  if (isNaN(num) || num === 0) return 'Low evidence'
  if (num >= 70) return 'High reliability'
  if (num >= 40) return 'Moderate reliability'
  return 'Low reliability'
}
