import { useState, useEffect } from 'react'
import { Cpu, Database, Award, Activity, HardDrive, Terminal } from 'lucide-react'
import { useTraining } from '../hooks/useTraining'
import TrainControls from '../components/TrainControls'
import MetricsLine from '../components/MetricsLine'
import LossChart from '../components/LossChart'
import TrainLog from '../components/TrainLog'
import { api } from '../utils/api'

export default function TrainingPage() {
  const {
    running,
    device,
    metrics,
    trainLossData,
    valLossData,
    logs,
    startTraining,
    stopTraining,
  } = useTraining()

  const [checkpointStatus, setCheckpointStatus] = useState(null)

  useEffect(() => {
    // Check if any checkpoint already exists
    api.getHealth()
      .then((h) => {
        if (h?.checkpoint) setCheckpointStatus(h.checkpoint)
      })
      .catch(() => {})
  }, [running])

  return (
    <div className="train-page-wrapper">
      {/* Overview Header Cards */}
      <div className="train-overview-grid">
        <div className="train-card">
          <div className="train-card-header">
            <Cpu size={14} className="train-card-icon" />
            <span className="train-card-title">Architecture</span>
          </div>
          <div className="train-card-value">TrustLM (Decoder-Only)</div>
          <div className="train-card-meta">6 layers · 256 d_model · 8 heads · ~10.2M params</div>
        </div>

        <div className="train-card">
          <div className="train-card-header">
            <Database size={14} className="train-card-icon" />
            <span className="train-card-title">Training Corpus</span>
          </div>
          <div className="train-card-value">Structured RAG Q&A</div>
          <div className="train-card-meta">data/train.txt · dense 64-token sliding window</div>
        </div>

        <div className="train-card">
          <div className="train-card-header">
            <HardDrive size={14} className="train-card-icon" />
            <span className="train-card-title">Checkpoint Target</span>
          </div>
          <div className="train-card-value">best_checkpoint.pt</div>
          <div className="train-card-meta">
            Saved to checkpoints/ · {running ? 'Training in progress...' : 'Ready for training'}
          </div>
        </div>

        <div className="train-card">
          <div className="train-card-header">
            <Activity size={14} className="train-card-icon" />
            <span className="train-card-title">Compute Execution</span>
          </div>
          <div className="train-card-value">100% Local (No API Keys)</div>
          <div className="train-card-meta">Device: {device} · AdamW optimizer with cosine decay</div>
        </div>
      </div>

      {/* Main Training Controls */}
      <div className="train-section-card">
        <div className="train-section-header">
          <span className="train-section-title">Training Configuration</span>
          {running && (
            <div className="train-live-badge">
              <span className="train-pulse-dot" />
              <span>Training Active</span>
            </div>
          )}
        </div>
        <TrainControls
          running={running}
          onStart={startTraining}
          onStop={stopTraining}
          device={device}
        />
      </div>

      {/* Real-time Metrics Line */}
      {metrics && (
        <div className="train-metrics-wrapper">
          <MetricsLine metrics={metrics} />
        </div>
      )}

      {/* Live Visual Loss Charts */}
      <div className="charts-row">
        <LossChart title="Training Loss (Next-Token Cross-Entropy)" data={trainLossData} dataKey="loss" />
        <LossChart title="Validation Loss (Holdout Set)" data={valLossData} dataKey="loss" />
      </div>

      {/* Terminal Telemetry Log */}
      <div className="train-section-card">
        <div className="train-section-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Terminal size={14} />
            <span className="train-section-title">Telemetry & Stdout Logs</span>
          </div>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            Real-time PyTorch subprocess stream
          </span>
        </div>
        <TrainLog logs={logs} />
      </div>
    </div>
  )
}
