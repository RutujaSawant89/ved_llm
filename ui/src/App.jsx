import { Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import TopBar from './components/TopBar'
import ChatPage from './pages/ChatPage'
import KnowledgeBasePage from './pages/KnowledgeBasePage'

export default function App() {
  return (
    <div className="app-layout">
      <Sidebar />
      <div className="app-main">
        <Routes>
          <Route
            path="/"
            element={
              <>
                <TopBar title="Chat" />
                <div className="page-content">
                  <ChatPage />
                </div>
              </>
            }
          />
          <Route
            path="/documents"
            element={
              <>
                <TopBar title="Documents" />
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
  )
}
