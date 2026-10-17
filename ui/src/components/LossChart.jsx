import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts'

export default function LossChart({ title, data = [], dataKey = 'loss' }) {
  const hasData = data && data.length > 0

  return (
    <div className="chart-box">
      <div className="chart-title">{title}</div>
      {hasData ? (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid stroke="#17171b" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="step"
              stroke="#5c5c66"
              fontSize={11}
              fontFamily="var(--mono)"
              tickLine={false}
              axisLine={{ stroke: '#1f1f23' }}
            />
            <YAxis
              stroke="#5c5c66"
              fontSize={11}
              fontFamily="var(--mono)"
              tickLine={false}
              axisLine={{ stroke: '#1f1f23' }}
              domain={['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#16161a',
                border: '1px solid #1f1f23',
                borderRadius: '4px',
                fontSize: '11px',
                fontFamily: 'var(--mono)',
                color: '#ededef',
                padding: '4px 8px',
              }}
              itemStyle={{ color: '#a0a0a8' }}
              labelStyle={{ color: '#5c5c66' }}
            />
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke="#a0a0a8"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div
          style={{
            height: '200px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--text-muted)',
            fontSize: 'var(--text-xs)',
            fontFamily: 'var(--mono)',
          }}
        >
          Awaiting training data
        </div>
      )}
    </div>
  )
}
