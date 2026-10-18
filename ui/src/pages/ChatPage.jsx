import { useState } from 'react'
import { useChat } from '../hooks/useChat'
import MessageList from '../components/MessageList'
import ChatInput from '../components/ChatInput'
import GenSettings from '../components/GenSettings'

export default function ChatPage() {
  const {
    messages,
    loading,
    settings,
    setSettings,
    isModelReady,
    sendMessage,
  } = useChat()

  const [showSettings, setShowSettings] = useState(false)

  return (
    <div className="chat-container">
      <MessageList
        messages={messages}
        loading={loading}
        onSuggestionClick={sendMessage}
        isModelReady={isModelReady}
      />

      <div style={{ marginTop: 'auto' }}>
        {showSettings && (
          <GenSettings
            settings={settings}
            onChange={setSettings}
          />
        )}
        <ChatInput
          onSend={sendMessage}
          disabled={loading}
          showSettings={showSettings}
          onToggleSettings={() => setShowSettings(!showSettings)}
        />
      </div>
    </div>
  )
}
