import { useState, useEffect, useCallback, useRef } from 'react'
import { api } from '../utils/api'

const STORAGE_KEY = 'ved_ai_conversations'
const ACTIVE_CONV_KEY = 'ved_ai_active_conv'

function generateId() {
  return 'conv_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7)
}

function loadConversations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed.map((c) => ({
          ...c,
          title: c.title === 'New Research' ? 'New Chat' : c.title,
          messages: (c.messages || []).map((m) => ({
            ...m,
            generator: m.generator && m.generator.includes('TrustLM') ? 'Ved.ai' : m.generator || 'Ved.ai',
          })),
        }))
      }
    }
  } catch (e) {
    console.error('Failed to load conversations from localStorage:', e)
  }
  const defaultConv = {
    id: generateId(),
    title: 'New Chat',
    created_at: new Date().toISOString(),
    messages: [],
  }
  return [defaultConv]
}

export function useChat() {
  const [conversations, setConversations] = useState(loadConversations)
  const [activeId, setActiveId] = useState(() => {
    const saved = localStorage.getItem(ACTIVE_CONV_KEY)
    const all = loadConversations()
    if (saved && all.some((c) => c.id === saved)) return saved
    return all[0]?.id || generateId()
  })

  const [loading, setLoading] = useState(false)
  const [settings, setSettings] = useState({
    temperature: 0.7,
    top_k: 40,
    top_p: 0.9,
  })
  const [isModelReady, setIsModelReady] = useState(false)
  const [docStats, setDocStats] = useState(null)

  // Sync to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations))
    } catch (e) {
      console.error('Failed to save conversations:', e)
    }
  }, [conversations])

  useEffect(() => {
    try {
      localStorage.setItem(ACTIVE_CONV_KEY, activeId)
    } catch (e) {
      console.error('Failed to save active conversation id:', e)
    }
  }, [activeId])

  // Fetch backend status
  const refreshStats = useCallback(async () => {
    try {
      const stats = await api.getStats().catch(() => null)
      if (stats) setDocStats(stats)
      const modelStatus = await api.getModelStatus().catch(() => null)
      setIsModelReady(Boolean(modelStatus?.usable))
    } catch (e) {
      // Ignore background errors
    }
  }, [])

  useEffect(() => {
    refreshStats()
  }, [refreshStats])

  const activeConv = conversations.find((c) => c.id === activeId) || conversations[0]
  const messages = activeConv?.messages || []

  // Create new conversation
  const newChat = useCallback(() => {
    const newConv = {
      id: generateId(),
      title: 'New Chat',
      created_at: new Date().toISOString(),
      messages: [],
    }
    setConversations((prev) => [newConv, ...prev])
    setActiveId(newConv.id)
  }, [])

  // Switch conversation
  const selectConversation = useCallback((id) => {
    setActiveId(id)
  }, [])

  // Delete conversation
  const deleteConversation = useCallback((id, e) => {
    if (e) e.stopPropagation()
    setConversations((prev) => {
      const filtered = prev.filter((c) => c.id !== id)
      if (filtered.length === 0) {
        const fresh = {
          id: generateId(),
          title: 'New Chat',
          created_at: new Date().toISOString(),
          messages: [],
        }
        setActiveId(fresh.id)
        return [fresh]
      }
      if (id === activeId) {
        setActiveId(filtered[0].id)
      }
      return filtered
    })
  }, [activeId])

  // Clear current messages
  const clearCurrentMessages = useCallback(() => {
    setConversations((prev) =>
      prev.map((c) => (c.id === activeId ? { ...c, messages: [] } : c))
    )
  }, [activeId])

  // Send message
  const sendMessage = useCallback(
    async (question) => {
      if (!question?.trim()) return

      const userMsg = {
        role: 'user',
        content: question.trim(),
        timestamp: new Date().toISOString(),
      }

      // Append user message immediately
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id === activeId) {
            const isFirst = c.messages.length === 0
            const autoTitle = isFirst
              ? question.trim().slice(0, 32) + (question.trim().length > 32 ? '...' : '')
              : c.title
            return {
              ...c,
              title: autoTitle,
              messages: [...c.messages, userMsg],
            }
          }
          return c
        })
      )

      setLoading(true)

      try {
        const res = await api.query(question, settings.top_k)

        const assistantMsg = {
          role: 'assistant',
          content:
            res.answer ||
            (res.sources?.length
              ? 'Retrieved relevant context from knowledge base.'
              : 'No response generated.'),
          sources: res.sources || [],
          trust: res.trust,
          verification: res.verification,
          generator: res.generator && !res.generator.includes('TrustLM') ? res.generator : 'Ved.ai',
          timestamp: new Date().toISOString(),
        }

        setConversations((prev) =>
          prev.map((c) =>
            c.id === activeId
              ? { ...c, messages: [...c.messages, assistantMsg] }
              : c
          )
        );
      } catch (err) {
        const errorMsg = {
          role: 'assistant',
          content: `Error: ${err.message || 'Failed to process query'}`,
          timestamp: new Date().toISOString(),
        }

        setConversations((prev) =>
          prev.map((c) =>
            c.id === activeId
              ? { ...c, messages: [...c.messages, errorMsg] }
              : c
          )
        )
      } finally {
        setLoading(false)
        refreshStats()
      }
    },
    [activeId, settings.top_k, refreshStats]
  )

  // Regenerate last assistant response
  const regenerateLast = useCallback(async () => {
    if (loading || messages.length === 0) return
    // Find the last user message
    let lastUserQuery = ''
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        lastUserQuery = messages[i].content
        break
      }
    }
    if (!lastUserQuery) return

    setLoading(true)
    try {
      const res = await api.query(lastUserQuery, settings.top_k)
      const assistantMsg = {
        role: 'assistant',
        content:
          res.answer ||
          (res.sources?.length
            ? 'Retrieved relevant context from knowledge base.'
            : 'No response generated.'),
        sources: res.sources || [],
        trust: res.trust,
        verification: res.verification,
        generator: res.generator && !res.generator.includes('TrustLM') ? res.generator : 'Ved.ai',
        timestamp: new Date().toISOString(),
      }

      setConversations((prev) =>
        prev.map((c) => {
          if (c.id === activeId) {
            // Replace the last message if it was an assistant message, or append
            const currentMsgs = [...c.messages]
            if (
              currentMsgs.length > 0 &&
              currentMsgs[currentMsgs.length - 1].role === 'assistant'
            ) {
              currentMsgs[currentMsgs.length - 1] = assistantMsg
            } else {
              currentMsgs.push(assistantMsg)
            }
            return { ...c, messages: currentMsgs }
          }
          return c
        })
      )
    } catch (err) {
      console.error('Failed to regenerate answer:', err)
    } finally {
      setLoading(false)
    }
  }, [activeId, loading, messages, settings.top_k])

  return {
    conversations,
    activeId,
    activeConv,
    messages,
    loading,
    settings,
    setSettings,
    isModelReady,
    docStats,
    newChat,
    selectConversation,
    deleteConversation,
    clearCurrentMessages,
    sendMessage,
    regenerateLast,
    refreshStats,
  }
}
