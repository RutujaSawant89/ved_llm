export default function ProgressBar({ progress, height = 2, color }) {
  const clamped = Math.min(Math.max(progress || 0, 0), 100)
  return (
    <div className="upload-progress" style={{ height: `${height}px` }}>
      <div
        className="upload-progress-bar"
        style={{
          width: `${clamped}%`,
          ...(color ? { background: color } : {}),
        }}
      />
    </div>
  )
}
