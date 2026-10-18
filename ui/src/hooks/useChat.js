import { useState, useEffect, useCallback } from 'react'
import { api } from '../utils/api'

export function useChat() {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [settings, setSettings] = useState({
    temperature: 0.7,
    top_k: 40,
    top_p: 0.9,
  })
  const [isModelReady, setIsModelReady] = useState(false)

  useEffect(() => {
    api.getModelStatus()
      .then((status) => {
        setIsModelReady(Boolean(status?.usable))
      })
      .catch(() => {
        setIsModelReady(false)
      })
  }, [])

  const sendMessage = useCallback(async (question) => {
    if (!question?.trim()) return

    const userMsg = {
      role: 'user',
      content: question,
    }

    setMessages((prev) => [...prev, userMsg])
    setLoading(true)

    try {
      const res = await api.query(question, settings.top_k)

      const assistantMsg = {
        role: 'assistant',
        content: res.answer || (res.sources?.length ? 'Retrieved relevant context from knowledge base.' : 'No response generated.'),
        sources: res.sources || [],
        trust: res.trust,
        verification: res.verification,
      }

      setMessages((prev) => [...prev, assistantMsg])
    } catch (err) {
      const errorMsg = {
        role: 'assistant',
        content: `Error: ${err.message || 'Failed to process query'}`,
      }
      setMessages((prev) => [...prev, errorMsg])
    } finally {
      setLoading(false)
    }
  }, [settings.top_k])

  return {
    messages,
    loading,
    settings,
    setSettings,
    isModelReady,
    sendMessage,
    setMessages,
  }
}
