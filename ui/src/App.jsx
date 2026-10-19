import { useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import TopBar from './components/TopBar'
import ChatPage from './pages/ChatPage'
import KnowledgeBasePage from './pages/KnowledgeBasePage'
import SplashScreen from './components/SplashScreen'
import SettingsModal from './components/SettingsModal'
import { useChat } from './hooks/useChat'

export default function App() {
  const [showSplash, setShowSplash] = useState(() => {
    return !sessionStorage.getItem('ved_splash_seen')
  })
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)

  const chatState = useChat()

  const currentTitle = chatState.activeConv?.title || 'Chat'

  const handleClearAllChats = () => {
    localStorage.removeItem('ved_ai_conversations')
    localStorage.removeItem('ved_ai_active_conv')
    window.location.reload()
  }

  return (
    <>
      {/* Optional Elegant Splash Screen on initial session entry */}
      {showSplash && <SplashScreen onComplete={() => setShowSplash(false)} />}

      <div className="app-layout">
        {/* Sidebar Navigation */}
        <Sidebar
          conversations={chatState.conversations}
          activeId={chatState.activeId}
          onSelectConversation={chatState.selectConversation}
          onNewChat={chatState.newChat}
          onDeleteConversation={chatState.deleteConversation}
          docStats={chatState.docStats}
          isMobileOpen={isMobileSidebarOpen}
          onCloseMobile={() => setIsMobileSidebarOpen(false)}
          onOpenSettings={() => setIsSettingsOpen(true)}
        />

        {/* Main Workspace Surface */}
        <div className="app-main">
          <Routes>
            <Route
              path="/"
              element={
                <>
                  <TopBar
                    title={currentTitle}
                    docStats={chatState.docStats}
                    onOpenMobileSidebar={() => setIsMobileSidebarOpen(true)}
                    onClearChat={chatState.clearCurrentMessages}
                    hasMessages={chatState.messages.length > 0}
                  />
                  <div className="page-content">
                    <ChatPage
                      messages={chatState.messages}
                      loading={chatState.loading}
                      settings={chatState.settings}
                      setSettings={chatState.setSettings}
                      docStats={chatState.docStats}
                      sendMessage={chatState.sendMessage}
                      regenerateLast={chatState.regenerateLast}
                      refreshStats={chatState.refreshStats}
                    />
                  </div>
                </>
              }
            />
            <Route
              path="/documents"
              element={
                <>
                  <TopBar
                    title="Documents & Knowledge Base"
                    docStats={chatState.docStats}
                    onOpenMobileSidebar={() => setIsMobileSidebarOpen(true)}
                  />
                  <div className="page-content">
                    <KnowledgeBasePage />
                  </div>
                </>
              }
            />
            {/* Backward compatibility for /knowledge */}
            <Route path="/knowledge" element={<Navigate to="/documents" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </div>

      {/* Global Settings & Profile Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={chatState.settings}
        onUpdateSettings={chatState.setSettings}
        activeConv={chatState.activeConv}
        conversations={chatState.conversations}
        onClearAllChats={handleClearAllChats}
      />
    </>
  )
}
