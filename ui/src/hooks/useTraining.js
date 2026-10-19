import { useState, useEffect, useRef, useCallback } from 'react'
import { api } from '../utils/api'

export function useTraining() {
  const [running, setRunning] = useState(false)
  const [device, setDevice] = useState('CPU')
  const [metrics, setMetrics] = useState(null)
  const [trainLossData, setTrainLossData] = useState([])
  const [valLossData, setValLossData] = useState([])
  const [logs, setLogs] = useState([])
  const pollTimerRef = useRef(null)

  // Fetch device information on initial load
  useEffect(() => {
    api.getHealth()
      .then((h) => {
        if (h?.device) setDevice(h.device.toUpperCase())
      })
      .catch(() => {})

    // Check initial training status
    api.getTrainStatus()
      .then((status) => {
        if (status?.running) {
          setRunning(true)
          if (status.logs) setLogs(status.logs)
        }
      })
      .catch(() => {})
  }, [])

  const addLog = useCallback((line) => {
    setLogs((prev) => [...prev, line])
  }, [])

  const stopTraining = useCallback(async () => {
    try {
      await api.stopTraining()
    } catch (err) {
      console.error(err)
    } finally {
      setRunning(false)
      clearInterval(pollTimerRef.current)
      addLog('[Training] Session stopped by user.')
    }
  }, [addLog])

  const startTraining = useCallback(async (params) => {
    try {
      setLogs([])
      setTrainLossData([])
      setValLossData([])
      setMetrics(null)

      addLog(`[Training] Initializing run: mode=${params.mode}, epochs=${params.epochs}, lr=${params.learning_rate}`)
      const res = await api.startTraining(params)

      if (res?.status === 'already_running') {
        addLog('[Training] A training run is already in progress.')
      }

      setRunning(true)
      addLog(`[Training] Process started (PID ${res?.pid || 'active'}).`)

      // Poll real-time backend telemetry
      clearInterval(pollTimerRef.current)
      pollTimerRef.current = setInterval(async () => {
        try {
          const status = await api.getTrainStatus()
          if (!status) return

          if (status.logs && status.logs.length > 0) {
            setLogs(status.logs)
          }

          if (status.step > 0) {
            const newMetric = {
              step: status.step,
              totalSteps: status.max_steps || (params.epochs * 1000),
              loss: status.train_loss,
              perplexity: status.perplexity,
              tokens: status.tokens_processed,
            }
            setMetrics(newMetric)

            if (status.train_loss > 0) {
              setTrainLossData((prev) => {
                if (prev.length > 0 && prev[prev.length - 1].step === status.step) {
                  return prev
                }
                return [...prev, { step: status.step, loss: status.train_loss }]
              })
            }

            if (status.val_loss > 0) {
              setValLossData((prev) => {
                if (prev.length > 0 && prev[prev.length - 1].step === status.step) {
                  return prev
                }
                return [...prev, { step: status.step, loss: status.val_loss }]
              })
            }
          }

          if (!status.running) {
            clearInterval(pollTimerRef.current)
            setRunning(false)
            addLog('[Training] Process finished. Checkpoint saved.')
          }
        } catch {
          // Keep polling silently during brief hiccups
        }
      }, 800)
    } catch (err) {
      addLog(`[Training Error] ${err.message || 'Failed to start'}`)
      setRunning(false)
    }
  }, [addLog])

  useEffect(() => {
    return () => {
      clearInterval(pollTimerRef.current)
    }
  }, [])

  return {
    running,
    device,
    metrics,
    trainLossData,
    valLossData,
    logs,
    startTraining,
    stopTraining,
  }
}
