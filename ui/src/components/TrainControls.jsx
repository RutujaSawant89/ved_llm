import { useState } from 'react'
import { Play, Square, Cpu, Sliders } from 'lucide-react'

export default function TrainControls({ running, onStart, onStop, device = 'CPU' }) {
  const [mode, setMode] = useState('smoke_test')
  const [epochs, setEpochs] = useState(5)
  const [steps, setSteps] = useState(1000)
  const [batchSize, setBatchSize] = useState(16)
  const [learningRate, setLearningRate] = useState('3e-4')

  const handleStart = () => {
    onStart({
      mode,
      epochs: parseInt(epochs) || 5,
      steps: mode === 'smoke_test' ? 5 : (parseInt(steps) || 1000),
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
          <option value="full">Full Pretraining</option>
        </select>
      </div>

      <div className="train-field">
        <label>Target Steps</label>
        <input
          type="number"
          min="5"
          max="50000"
          step="50"
          value={mode === 'smoke_test' ? 5 : steps}
          onChange={(e) => setSteps(e.target.value)}
          disabled={running || mode === 'smoke_test'}
          style={{ width: '84px' }}
        />
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
        <div className="train-device">
          <Cpu size={12} />
          <span>{device}</span>
        </div>
      </div>

      <div style={{ marginLeft: 'auto' }}>
        {running ? (
          <button type="button" className="train-btn-stop" onClick={onStop}>
            <Square size={13} fill="currentColor" />
            <span>Stop Run</span>
          </button>
        ) : (
          <button type="button" className="train-btn-start" onClick={handleStart}>
            <Play size={13} fill="currentColor" />
            <span>{mode === 'smoke_test' ? 'Run Smoke Test' : 'Start Pretraining'}</span>
          </button>
        )}
      </div>
    </div>
  )
}
