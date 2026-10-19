import { useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import {
  MessageSquare,
  Folder,
  Plus,
  Trash2,
  PanelLeftClose,
  PanelLeftOpen,
  X,
  Settings,
  User,
} from 'lucide-react'

export default function Sidebar({
  conversations = [],
  activeId,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
  docStats,
  isMobileOpen = false,
  onCloseMobile,
  onOpenSettings,
}) {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const navigate = useNavigate()

  const handleNewChat = () => {
    onNewChat?.()
    navigate('/')
    if (onCloseMobile) onCloseMobile()
  }

  const handleSelectConv = (id) => {
    onSelectConversation?.(id)
    navigate('/')
    if (onCloseMobile) onCloseMobile()
  }

  const totalChunks = docStats?.total_chunks ?? 0
  const isIndexed = totalChunks > 0

  return (
    <>
      {/* Mobile Backdrop */}
      {isMobileOpen && (
        <div
          className="sidebar-backdrop"
          onClick={onCloseMobile}
          aria-label="Close sidebar"
        />
      )}

      <aside
        className={`sidebar ${isCollapsed ? 'collapsed' : 'expanded'} ${
          isMobileOpen ? 'mobile-open' : ''
        }`}
      >
        {/* Brand Header */}
        <div className="sidebar-header">
          {!isCollapsed && (
            <div className="sidebar-brand">
              <div className="sidebar-brand-mark">V</div>
              <div className="sidebar-brand-text">
                <span className="sidebar-brand-title">Ved.ai</span>
                <span className="sidebar-brand-sub">Research Workspace</span>
              </div>
            </div>
          )}

          {isCollapsed && (
            <div className="sidebar-brand-mark" title="Ved.ai">
              V
            </div>
          )}

          {/* Desktop Collapse Toggle */}
          <button
            type="button"
            className="sidebar-collapse-btn"
            onClick={() => setIsCollapsed(!isCollapsed)}
            title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {isCollapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          </button>

          {/* Mobile Close Button */}
          {isMobileOpen && (
            <button
              type="button"
              className="sidebar-collapse-btn mobile-only"
              onClick={onCloseMobile}
              aria-label="Close menu"
            >
              <X size={16} />
            </button>
          )}
        </div>

        {/* New Chat Action */}
        <div className="sidebar-actions">
          <button
            type="button"
            className="sidebar-new-chat-btn"
            onClick={handleNewChat}
            title="Start new research conversation"
          >
            <Plus size={15} />
            {!isCollapsed && <span>New Research</span>}
          </button>
        </div>

        {/* Primary Navigation Links */}
        <div className="sidebar-nav-section">
          {!isCollapsed && <div className="sidebar-section-label">Workspace</div>}
          <ul className="sidebar-nav-list">
            <li>
              <NavLink
                to="/"
                end
                className={({ isActive }) =>
                  `sidebar-nav-item${isActive ? ' active' : ''}`
                }
                title="Chat"
                onClick={onCloseMobile}
              >
                <MessageSquare size={16} />
                {!isCollapsed && <span>Chat</span>}
              </NavLink>
            </li>
            <li>
              <NavLink
                to="/documents"
                className={({ isActive }) =>
                  `sidebar-nav-item${isActive ? ' active' : ''}`
                }
                title="Documents"
                onClick={onCloseMobile}
              >
                <Folder size={16} />
                {!isCollapsed && <span>Documents</span>}
                {!isCollapsed && totalChunks > 0 && (
                  <span className="sidebar-nav-badge">{totalChunks}</span>
                )}
              </NavLink>
            </li>
          </ul>
        </div>

        {/* Research History List */}
        {!isCollapsed && (
          <div className="sidebar-history-section">
            <div className="sidebar-section-label">Recent Sessions</div>
            {conversations && conversations.length > 0 ? (
              <ul className="sidebar-history-list">
                {conversations.map((c) => {
                  const isActive = c.id === activeId
                  return (
                    <li
                      key={c.id}
                      className={`sidebar-history-item${isActive ? ' active' : ''}`}
                      onClick={() => handleSelectConv(c.id)}
                      title={c.title}
                    >
                      <span className="sidebar-history-title">{c.title}</span>
                      <button
                        type="button"
                        className="sidebar-history-delete"
                        onClick={(e) => onDeleteConversation?.(c.id, e)}
                        title="Delete session"
                        aria-label="Delete session"
                      >
                        <Trash2 size={13} />
                      </button>
                    </li>
                  )
                })}
              </ul>
            ) : (
              <div
                style={{
                  fontSize: 'var(--text-xs)',
                  color: 'var(--text-muted)',
                  padding: '8px',
                }}
              >
                No previous sessions
              </div>
            )}
          </div>
        )}

        {/* Bottom Section: Indexing Status & Profile / Settings Button */}
        <div className="sidebar-footer">
          {/* Status pill */}
          <div
            className="sidebar-status-pill"
            title={
              isIndexed
                ? `${totalChunks} document chunks indexed in ChromaDB`
                : 'No documents indexed in knowledge base'
            }
          >
            <span
              className={`sidebar-status-dot ${isIndexed ? '' : 'inactive'}`}
            />
            {!isCollapsed && (
              <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                {isIndexed
                  ? `${totalChunks} chunks indexed`
                  : 'Knowledge base empty'}
              </span>
            )}
          </div>

          {/* User Profile & Settings Trigger */}
          <button
            type="button"
            className="sidebar-profile-btn"
            onClick={onOpenSettings}
            title="User Profile & Settings"
            aria-label="Settings"
          >
            <div className="sidebar-profile-avatar">VS</div>
            {!isCollapsed && (
              <div className="sidebar-profile-info">
                <span className="sidebar-profile-name">Dr. Viraj Sawant</span>
                <span className="sidebar-profile-role">Settings & Auth</span>
              </div>
            )}
            {!isCollapsed && <Settings size={14} className="sidebar-settings-icon" />}
          </button>
        </div>
      </aside>
    </>
  )
}
