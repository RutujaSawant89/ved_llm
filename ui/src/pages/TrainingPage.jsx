import { useTraining } from '../hooks/useTraining'
import TrainControls from '../components/TrainControls'
import MetricsLine from '../components/MetricsLine'
import LossChart from '../components/LossChart'
import TrainLog from '../components/TrainLog'

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

  return (
    <div className="train-container">
      <TrainControls
        running={running}
        onStart={startTraining}
        onStop={stopTraining}
        device={device}
      />

      <MetricsLine metrics={metrics} />

      <div className="charts-row">
        <LossChart title="Training Loss" data={trainLossData} dataKey="loss" />
        <LossChart title="Validation Loss" data={valLossData} dataKey="loss" />
      </div>

      <TrainLog logs={logs} />
    </div>
  )
}
