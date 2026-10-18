import { useState, useEffect, useCallback } from 'react'
import { api } from '../utils/api'

export function useModel() {
  const [modelInfo, setModelInfo] = useState(null)
  const [modelStatus, setModelStatus] = useState(null)
  const [config, setConfig] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const loadData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [infoRes, statusRes, configRes] = await Promise.allSettled([
        api.getModelInfo(),
        api.getModelStatus(),
        api.getConfig(),
      ])

      if (infoRes.status === 'fulfilled') setModelInfo(infoRes.value?.architecture)
      if (statusRes.status === 'fulfilled') setModelStatus(statusRes.value)
      if (configRes.status === 'fulfilled') setConfig(configRes.value)
    } catch (err) {
      setError(err.message || 'Failed to load model data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadData()
  }, [loadData])

  const updateGeneration = useCallback(async (data) => {
    try {
      const updated = await api.updateGeneration(data)
      setConfig((prev) => ({
        ...prev,
        generation: { ...prev?.generation, ...data },
      }))
      return updated
    } catch (err) {
      console.error('Failed to update generation config:', err)
      throw err
    }
  }, [])

  return {
    modelInfo,
    modelStatus,
    config,
    loading,
    error,
    refresh: loadData,
    updateGeneration,
  }
}
