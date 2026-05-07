import React from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="panel px-3 py-2 text-xs font-mono" style={{ border: '1px solid #0f2040' }}>
      <div style={{ color: 'var(--muted)' }} className="mb-1">Epoch {label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {Number(p.value).toFixed(4)}
        </div>
      ))}
    </div>
  )
}

export default function LossChart({ history }) {
  if (!history || !history.train_loss?.length) {
    return (
      <div className="panel p-6 flex items-center justify-center h-64">
        <span className="text-sm font-mono" style={{ color: 'var(--muted)' }}>
          No training history — run POST /train first
        </span>
      </div>
    )
  }

  const data = history.train_loss.map((_, i) => ({
    epoch: i + 1,
    'Train Loss':  Number(history.train_loss[i]?.toFixed(4)),
    'Val Loss':    Number(history.val_loss[i]?.toFixed(4)),
    'Phys Loss':   Number(history.train_phys_loss[i]?.toFixed(4)),
  }))

  // Downsample for rendering performance
  const step = Math.max(1, Math.floor(data.length / 80))
  const sampled = data.filter((_, i) => i % step === 0)

  return (
    <div className="panel p-5">
      <div className="text-xs font-mono uppercase tracking-widest mb-4" style={{ color: 'var(--muted)' }}>
        Training Loss Curves
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={sampled} margin={{ top: 4, right: 20, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#0f2040" />
          <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} />
          <Line type="monotone" dataKey="Train Loss" stroke="#00d4ff" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="Val Loss"   stroke="#00ff9d" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="Phys Loss"  stroke="#bf5af2" dot={false} strokeWidth={1.5} strokeDasharray="4 2" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
