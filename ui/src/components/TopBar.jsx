import { Menu, RotateCcw } from 'lucide-react'

export default function TopBar({
  title = 'Chat',
  subtitle,
  docStats,
  onOpenMobileSidebar,
  onClearChat,
  hasMessages = false,
}) {
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
      </div>
    </header>
  )
}
