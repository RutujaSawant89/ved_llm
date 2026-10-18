import { useState } from 'react'

export default function TrainControls({ running, onStart, onStop, device = 'CPU' }) {
  const [mode, setMode] = useState('smoke_test')
  const [epochs, setEpochs] = useState(1)
  const [batchSize, setBatchSize] = useState(16)
  const [learningRate, setLearningRate] = useState('3e-4')

  const handleStart = () => {
    onStart({
      mode,
      epochs: parseInt(epochs) || 1,
      batch_size: parseInt(batchSize) || 16,
      learning_rate: parseFloat(learningRate) || 3e-4,
    })
  }

  return (
    <div className="train-controls">
      <div className="train-field">
        <label>Mode</label>
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value)}
          disabled={running}
        >
          <option value="smoke_test">Smoke Test (5 steps)</option>
          <option value="full">Full Training</option>
        </select>
      </div>

      <div className="train-field">
        <label>Epochs</label>
        <input
          type="number"
          min="1"
          max="100"
          value={epochs}
          onChange={(e) => setEpochs(e.target.value)}
          disabled={running || mode === 'smoke_test'}
          style={{ width: '64px' }}
        />
      </div>

      <div className="train-field">
        <label>Batch Size</label>
        <input
          type="number"
          min="1"
          max="256"
          value={batchSize}
          onChange={(e) => setBatchSize(e.target.value)}
          disabled={running || mode === 'smoke_test'}
          style={{ width: '64px' }}
        />
      </div>

      <div className="train-field">
        <label>Learning Rate</label>
        <input
          type="text"
          className="mono"
          value={learningRate}
          onChange={(e) => setLearningRate(e.target.value)}
          disabled={running || mode === 'smoke_test'}
          style={{ width: '80px' }}
        />
      </div>

      <div className="train-field">
        <label>Device</label>
        <div className="train-device">{device}</div>
      </div>

      <div style={{ marginLeft: 'auto' }}>
        {running ? (
          <button type="button" className="btn-stop" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button type="button" className="btn-start" onClick={handleStart}>
            Start Training
          </button>
        )}
      </div>
    </div>
  )
}
