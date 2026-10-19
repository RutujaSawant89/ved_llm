import { useState, useEffect } from 'react'

export default function SplashScreen({ onComplete }) {
  const [fading, setFading] = useState(false)

  useEffect(() => {
    // Check if already seen in current session
    if (sessionStorage.getItem('ved_splash_seen')) {
      onComplete?.()
      return
    }

    const timer = setTimeout(() => {
      handleDismiss()
    }, 1600)

    const handleKeyDown = (e) => {
      if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') {
        handleDismiss()
      }
    }

    window.addEventListener('keydown', handleKeyDown)

    return () => {
      clearTimeout(timer)
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [onComplete])

  const handleDismiss = () => {
    setFading(true)
    sessionStorage.setItem('ved_splash_seen', 'true')
    setTimeout(() => {
      onComplete?.()
    }, 300)
  }

  return (
    <div
      className={`splash-overlay ${fading ? 'fade-out' : ''}`}
      onClick={handleDismiss}
      title="Click or press Esc to enter"
      role="banner"
    >
      <div className="splash-card">
        <div className="splash-logo-container">
          <div className="splash-logo-glyph">V</div>
          <div className="splash-pulse-ring" />
        </div>

        <div className="splash-typography">
          <h1 className="splash-title">Ved.ai</h1>
          <p className="splash-tagline">Knowledge, grounded.</p>
        </div>

        <div className="splash-progress-track">
          <div className="splash-progress-indicator" />
        </div>

        <span className="splash-hint">Click or press Esc to skip</span>
      </div>
    </div>
  )
}
