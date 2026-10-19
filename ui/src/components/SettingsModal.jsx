import { useState, useEffect } from 'react'
import {
  X,
  User,
  Sliders,
  Database,
  Download,
  LogOut,
  LogIn,
  Check,
  Shield,
  Trash2,
} from 'lucide-react'

const AUTH_KEY = 'ved_ai_auth_user'

const DEFAULT_USER = {
  name: 'Dr. Viraj Sawant',
  email: 'researcher@ved.ai',
  role: 'Principal AI Researcher',
  avatarInitials: 'VS',
}

export default function SettingsModal({
  isOpen,
  onClose,
  settings,
  onUpdateSettings,
  activeConv,
  conversations,
  onClearAllChats,
}) {
  const [activeTab, setActiveTab] = useState('profile')
  const [user, setUser] = useState(() => {
    try {
      const saved = localStorage.getItem(AUTH_KEY)
      if (saved) return JSON.parse(saved)
    } catch (e) {}
    return DEFAULT_USER
  })
  const [isLoggedIn, setIsLoggedIn] = useState(() => {
    return localStorage.getItem('ved_ai_is_logged_in') !== 'false'
  })
  const [exportNotice, setExportNotice] = useState(null)

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isOpen) {
        onClose?.()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  const handleToggleAuth = () => {
    if (isLoggedIn) {
      setIsLoggedIn(false)
      localStorage.setItem('ved_ai_is_logged_in', 'false')
    } else {
      setIsLoggedIn(true)
      localStorage.setItem('ved_ai_is_logged_in', 'true')
    }
  }

  const exportCurrentChat = (format = 'md') => {
    if (!activeConv || !activeConv.messages) return

    let content = ''
    const title = activeConv.title || 'Research Session'

    if (format === 'md') {
      content = `# Ved.ai Research Session: ${title}\n`
      content += `Date: ${new Date(activeConv.created_at || Date.now()).toLocaleString()}\n\n---\n\n`
      activeConv.messages.forEach((m) => {
        const speaker = m.role === 'user' ? '### 👤 User' : '### 🤖 Ved.ai (TrustLM)'
        content += `${speaker}\n${m.content}\n\n`
        if (m.trust) {
          content += `*TrustLM Score: ${m.trust.score || 'N/A'} | Classification: ${m.trust.reason || ''}*\n\n`
        }
      })
    } else {
      content = JSON.stringify(activeConv, null, 2)
    }

    const blob = new Blob([content], {
      type: format === 'md' ? 'text/markdown' : 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${title.toLowerCase().replace(/[^a-z0-9]/g, '_')}.${format}`
    a.click()
    URL.revokeObjectURL(url)

    setExportNotice(`Exported as .${format}`)
    setTimeout(() => setExportNotice(null), 2500)
  }

  if (!isOpen) return null

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-container"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
      >
        {/* Modal Header */}
        <div className="modal-header">
          <h2 id="settings-title" className="modal-title">Settings & Workspace</h2>
          <button
            type="button"
            className="modal-close-btn"
            onClick={onClose}
            aria-label="Close settings"
          >
            <X size={16} />
          </button>
        </div>

        {/* Modal Body */}
        <div className="modal-body">
          {/* Sidebar Tabs */}
          <div className="modal-tabs-sidebar">
            <button
              type="button"
              className={`modal-tab-nav-btn ${activeTab === 'profile' ? 'active' : ''}`}
              onClick={() => setActiveTab('profile')}
            >
              <User size={15} />
              <span>Profile & Account</span>
            </button>
            <button
              type="button"
              className={`modal-tab-nav-btn ${activeTab === 'model' ? 'active' : ''}`}
              onClick={() => setActiveTab('model')}
            >
              <Sliders size={15} />
              <span>Model & RAG</span>
            </button>
            <button
              type="button"
              className={`modal-tab-nav-btn ${activeTab === 'data' ? 'active' : ''}`}
              onClick={() => setActiveTab('data')}
            >
              <Database size={15} />
              <span>Data & Export</span>
            </button>
          </div>

          {/* Tab Content Panes */}
          <div className="modal-tab-content">
            {/* Tab 1: Profile & Account */}
            {activeTab === 'profile' && (
              <div className="settings-pane">
                <div className="profile-card">
                  <div className="profile-avatar-wrap">
                    <div className="profile-avatar">
                      {isLoggedIn ? user.avatarInitials : 'G'}
                    </div>
                  </div>
                  <div className="profile-info">
                    <div className="profile-name">
                      {isLoggedIn ? user.name : 'Guest Researcher'}
                    </div>
                    <div className="profile-email">
                      {isLoggedIn ? user.email : 'guest@local.session'}
                    </div>
                    <div className="profile-badge">
                      <Shield size={11} />
                      <span>{isLoggedIn ? user.role : 'Guest Mode'}</span>
                    </div>
                  </div>
                </div>

                <div className="settings-section">
                  <div className="settings-section-title">Session Authentication</div>
                  <p className="settings-section-desc">
                    {isLoggedIn
                      ? 'You are signed in with persistent local workspace credentials.'
                      : 'You are currently browsing as guest. Sign in to sync research sessions.'}
                  </p>

                  <button
                    type="button"
                    className={`settings-auth-btn ${isLoggedIn ? 'signout' : 'signin'}`}
                    onClick={handleToggleAuth}
                  >
                    {isLoggedIn ? (
                      <>
                        <LogOut size={14} />
                        <span>Sign Out</span>
                      </>
                    ) : (
                      <>
                        <LogIn size={14} />
                        <span>Sign In as Researcher</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {/* Tab 2: Model & RAG Configuration */}
            {activeTab === 'model' && (
              <div className="settings-pane">
                <div className="settings-field">
                  <div className="settings-field-header">
                    <span className="settings-field-name">Sampling Temperature</span>
                    <span className="settings-field-val">{settings.temperature}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1.5"
                    step="0.05"
                    value={settings.temperature}
                    onChange={(e) =>
                      onUpdateSettings?.({
                        ...settings,
                        temperature: parseFloat(e.target.value),
                      })
                    }
                    className="composer-slider"
                  />
                  <span className="settings-field-hint">
                    Lower values provide more deterministic, grounded responses.
                  </span>
                </div>

                <div className="settings-field">
                  <div className="settings-field-header">
                    <span className="settings-field-name">Top-K Retrieved Chunks</span>
                    <span className="settings-field-val">{settings.top_k}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    step="1"
                    value={settings.top_k}
                    onChange={(e) =>
                      onUpdateSettings?.({
                        ...settings,
                        top_k: parseInt(e.target.value, 10),
                      })
                    }
                    className="composer-slider"
                  />
                  <span className="settings-field-hint">
                    Number of context passages extracted from ChromaDB per query.
                  </span>
                </div>

                <div className="settings-field">
                  <div className="settings-field-header">
                    <span className="settings-field-name">Nucleus Sampling (Top-P)</span>
                    <span className="settings-field-val">{settings.top_p}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={settings.top_p}
                    onChange={(e) =>
                      onUpdateSettings?.({
                        ...settings,
                        top_p: parseFloat(e.target.value),
                      })
                    }
                    className="composer-slider"
                  />
                </div>
              </div>
            )}

            {/* Tab 3: Data & Export */}
            {activeTab === 'data' && (
              <div className="settings-pane">
                <div className="settings-section">
                  <div className="settings-section-title">Export Research</div>
                  <p className="settings-section-desc">
                    Download the current conversation including answers, source citations, and TrustLM reliability metrics.
                  </p>

                  <div style={{ display: 'flex', gap: '8px', marginTop: '10px' }}>
                    <button
                      type="button"
                      className="assistant-action-btn"
                      style={{ border: '1px solid var(--border)', padding: '7px 12px' }}
                      onClick={() => exportCurrentChat('md')}
                    >
                      <Download size={13} />
                      <span>Export as Markdown (.md)</span>
                    </button>
                    <button
                      type="button"
                      className="assistant-action-btn"
                      style={{ border: '1px solid var(--border)', padding: '7px 12px' }}
                      onClick={() => exportCurrentChat('json')}
                    >
                      <Download size={13} />
                      <span>Export as JSON</span>
                    </button>
                  </div>
                  {exportNotice && (
                    <div style={{ fontSize: '11px', color: 'var(--green)', marginTop: '8px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <Check size={12} />
                      <span>{exportNotice}</span>
                    </div>
                  )}
                </div>

                <div className="settings-section" style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '16px', marginTop: '16px' }}>
                  <div className="settings-section-title" style={{ color: 'var(--red)' }}>
                    Danger Zone
                  </div>
                  <p className="settings-section-desc">
                    Reset local session storage or delete conversation logs.
                  </p>
                  <button
                    type="button"
                    className="doc-delete-btn"
                    style={{ marginTop: '8px', padding: '6px 12px' }}
                    onClick={() => {
                      if (confirm('Clear all conversation history from local storage?')) {
                        onClearAllChats?.()
                        onClose?.()
                      }
                    }}
                  >
                    <Trash2 size={13} />
                    <span>Clear All Chat Sessions</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
