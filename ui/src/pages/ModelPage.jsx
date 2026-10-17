import { useModel } from '../hooks/useModel'
import ArchitectureTable from '../components/ArchitectureTable'
import ConfigAccordion from '../components/ConfigAccordion'
import CheckpointLine from '../components/CheckpointLine'

export default function ModelPage() {
  const {
    modelInfo,
    modelStatus,
    config,
    updateGeneration,
  } = useModel()

  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <div className="model-layout">
        <div className="model-arch-column">
          <ArchitectureTable arch={modelInfo} />
        </div>
        <div className="model-config-column">
          <ConfigAccordion
            config={config}
            onUpdateGeneration={updateGeneration}
          />
        </div>
      </div>

      <CheckpointLine status={modelStatus} />
    </div>
  )
}
