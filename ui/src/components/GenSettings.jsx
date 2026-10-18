export default function GenSettings({ settings, onChange }) {
  return (
    <div className="gen-settings">
      <div className="gen-setting">
        <label>Temperature</label>
        <input
          type="range"
          min="0"
          max="2"
          step="0.1"
          value={settings.temperature}
          onChange={(e) => onChange({ ...settings, temperature: parseFloat(e.target.value) })}
        />
        <span className="gen-setting-value">{settings.temperature}</span>
      </div>
      <div className="gen-setting">
        <label>Top-K</label>
        <input
          type="number"
          min="1"
          max="100"
          value={settings.top_k}
          onChange={(e) => onChange({ ...settings, top_k: parseInt(e.target.value) || 1 })}
        />
      </div>
      <div className="gen-setting">
        <label>Top-P</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={settings.top_p}
          onChange={(e) => onChange({ ...settings, top_p: parseFloat(e.target.value) })}
        />
        <span className="gen-setting-value">{settings.top_p}</span>
      </div>
    </div>
  )
}
