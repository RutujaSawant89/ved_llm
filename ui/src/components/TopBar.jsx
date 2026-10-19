import { Menu, RotateCcw } from 'lucide-react'

export default function TopBar({
  title = 'Chat',
  subtitle,
  docStats,
  onOpenMobileSidebar,
  onClearChat,
  hasMessages = false,
}) {
  const totalChunks = docStats?.total_chunks ?? 0
  const isIndexed = totalChunks > 0

  return (
    <header className="topbar">
      <div className="topbar-left">
        <button
          type="button"
          className="topbar-menu-btn"
          onClick={onOpenMobileSidebar}
          aria-label="Open navigation menu"
          title="Open menu"
        >
          <Menu size={18} />
        </button>

        <div className="topbar-title-wrap">
          <span className="topbar-title">{title}</span>
          {subtitle && <span className="topbar-sub">/ {subtitle}</span>}
        </div>
      </div>

      <div className="topbar-right">
        {hasMessages && onClearChat && (
          <button
            type="button"
            className="assistant-action-btn"
            onClick={onClearChat}
            title="Clear current conversation messages"
          >
            <RotateCcw size={13} />
            <span>Clear</span>
          </button>
        )}

        <div
          className="topbar-rag-badge"
          title={
            isIndexed
              ? `RAG Active with ${totalChunks} chunks in knowledge base`
              : 'Knowledge base is empty. Upload documents in the Documents tab.'
          }
        >
          <span
            className={`rag-status-dot ${isIndexed ? 'active' : 'inactive'}`}
          />
          <span>{isIndexed ? 'RAG Active' : 'RAG Standby'}</span>
          <span style={{ color: 'var(--border-focus)' }}>·</span>
          <span>{totalChunks} chunks</span>
        </div>
      </div>
    </header>
  )
}
