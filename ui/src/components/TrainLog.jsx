import { useEffect, useRef } from 'react'

export default function TrainLog({ logs = [] }) {
  const scrollRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs])

  return (
    <div>
      <div className="log-heading">Log</div>
      <div className="train-log" ref={scrollRef}>
        {logs && logs.length > 0 ? (
          logs.map((line, i) => (
            <div key={i}>{line}</div>
          ))
        ) : (
          <div className="train-log-empty">No training session active.</div>
        )}
      </div>
    </div>
  )
}
