import { useState, useEffect } from 'react'
import {
  Copy,
  Check,
  Volume2,
  VolumeX,
  Share2,
  RotateCcw,
  ThumbsUp,
  ThumbsDown,
} from 'lucide-react'

// Global speech state to ensure only one answer speaks at a time
let currentSpeakingUtterance = null

export default function MessageActions({
  content = '',
  onRegenerate,
  canRegenerate = false,
}) {
  const [copied, setCopied] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [feedback, setFeedback] = useState(null) // 'up' | 'down' | null
  const [shareFeedback, setShareFeedback] = useState(false)

  // Clean speech synthesis on unmount
  useEffect(() => {
    return () => {
      if (isSpeaking && window.speechSynthesis) {
        window.speechSynthesis.cancel()
      }
    }
  }, [isSpeaking])

  const handleCopy = async () => {
    if (!content) return
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy to clipboard:', err)
    }
  }

  const handleSpeech = () => {
    if (!window.speechSynthesis) {
      alert('Text-to-speech is not supported in this browser.')
      return
    }

    if (isSpeaking) {
      window.speechSynthesis.cancel()
      setIsSpeaking(false)
      currentSpeakingUtterance = null
    } else {
      window.speechSynthesis.cancel() // Stop any previous speech
      const utterance = new SpeechSynthesisUtterance(content)
      utterance.rate = 1.0
      utterance.pitch = 1.0

      utterance.onend = () => {
        setIsSpeaking(false)
        currentSpeakingUtterance = null
      }

      utterance.onerror = (e) => {
        console.warn('Speech synthesis error:', e)
        setIsSpeaking(false)
        currentSpeakingUtterance = null
      }

      currentSpeakingUtterance = utterance
      setIsSpeaking(true)
      window.speechSynthesis.speak(utterance)
    }
  }

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Ved.ai Research Response',
          text: content,
        })
      } catch (err) {
        if (err.name !== 'AbortError') {
          console.warn('Share failed:', err)
        }
      }
    } else {
      // Fallback to clipboard
      try {
        await navigator.clipboard.writeText(content)
        setShareFeedback(true)
        setTimeout(() => setShareFeedback(false), 2000)
      } catch (err) {
        console.error('Failed to copy share text:', err)
      }
    }
  }

  return (
    <div className="assistant-actions-bar">
      {/* Copy */}
      <button
        type="button"
        className="assistant-action-btn"
        onClick={handleCopy}
        title={copied ? 'Copied to clipboard' : 'Copy answer'}
        aria-label="Copy answer"
      >
        {copied ? <Check size={13} style={{ color: 'var(--green)' }} /> : <Copy size={13} />}
        <span>{copied ? 'Copied' : 'Copy'}</span>
      </button>

      {/* Read Aloud */}
      <button
        type="button"
        className={`assistant-action-btn ${isSpeaking ? 'speaking' : ''}`}
        onClick={handleSpeech}
        title={isSpeaking ? 'Stop speaking' : 'Read answer aloud'}
        aria-label={isSpeaking ? 'Stop read aloud' : 'Read aloud'}
      >
        {isSpeaking ? <VolumeX size={13} /> : <Volume2 size={13} />}
        <span>{isSpeaking ? 'Stop' : 'Read'}</span>
      </button>

      {/* Share */}
      <button
        type="button"
        className="assistant-action-btn"
        onClick={handleShare}
        title="Share answer"
        aria-label="Share answer"
      >
        {shareFeedback ? (
          <Check size={13} style={{ color: 'var(--green)' }} />
        ) : (
          <Share2 size={13} />
        )}
        <span>{shareFeedback ? 'Copied' : 'Share'}</span>
      </button>

      {/* Regenerate */}
      {canRegenerate && onRegenerate && (
        <button
          type="button"
          className="assistant-action-btn"
          onClick={onRegenerate}
          title="Regenerate this answer"
          aria-label="Regenerate answer"
        >
          <RotateCcw size={13} />
          <span>Regenerate</span>
        </button>
      )}

      {/* Thumbs Feedback */}
      <button
        type="button"
        className={`assistant-action-btn ${feedback === 'up' ? 'active' : ''}`}
        onClick={() => setFeedback(feedback === 'up' ? null : 'up')}
        title="Helpful response"
        aria-label="Helpful response"
      >
        <ThumbsUp size={13} />
      </button>

      <button
        type="button"
        className={`assistant-action-btn ${feedback === 'down' ? 'active' : ''}`}
        onClick={() => setFeedback(feedback === 'down' ? null : 'down')}
        title="Unhelpful response"
        aria-label="Unhelpful response"
      >
        <ThumbsDown size={13} />
      </button>
    </div>
  )
}
