import React from 'react'
import { CONDITION_COLORS, CONDITION_ORDER } from '../services/api'

export default function RiskPanel({ distribution = {}, total = 0 }) {
  const data = CONDITION_ORDER.map((c) => ({
    condition: c,
    count: distribution[c] || 0,
    pct: total > 0 ? ((distribution[c] || 0) / total) * 100 : 0,
    color: CONDITION_COLORS[c],
  }))

  return (
    <div className="panel p-5">
      <div className="text-xs font-mono uppercase tracking-widest mb-4" style={{ color: 'var(--muted)' }}>
        Condition Distribution
      </div>
      <div className="space-y-3">
        {data.map(({ condition, count, pct, color }) => (
          <div key={condition}>
            <div className="flex justify-between text-xs font-mono mb-1">
              <span style={{ color }}>{condition}</span>
              <span style={{ color: 'var(--muted)' }}>
                {count} segments ({pct.toFixed(1)}%)
              </span>
            </div>
            <div className="h-2 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.05)' }}>
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${pct}%`,
                  background: color,
                  boxShadow: `0 0 8px ${color}66`,
                }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Total */}
      <div className="mt-4 pt-3 border-t flex justify-between text-xs font-mono" style={{ borderColor: 'var(--border)', color: 'var(--muted)' }}>
        <span>Total Segments</span>
        <span style={{ color: 'var(--accent)' }}>{total.toLocaleString()}</span>
      </div>
    </div>
  )
}
