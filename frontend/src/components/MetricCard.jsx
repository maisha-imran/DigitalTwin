import React from 'react'

export default function MetricCard({ label, value, unit = '', icon: Icon, color = 'var(--accent)', sub = '' }) {
  return (
    <div
      className="panel p-5 flex flex-col gap-2 relative overflow-hidden fade-in-up"
      style={{ borderColor: `${color}22` }}
    >
      {/* Background glow */}
      <div
        className="absolute -top-4 -right-4 w-24 h-24 rounded-full opacity-10 blur-2xl"
        style={{ background: color }}
      />

      <div className="flex items-center justify-between">
        <span className="text-xs font-mono uppercase tracking-widest" style={{ color: 'var(--muted)' }}>
          {label}
        </span>
        {Icon && <Icon size={16} style={{ color }} />}
      </div>

      <div className="flex items-end gap-1 mt-1">
        <span className="text-3xl font-display font-bold" style={{ color }}>
          {value}
        </span>
        {unit && (
          <span className="text-sm font-mono mb-1" style={{ color: 'var(--muted)' }}>
            {unit}
          </span>
        )}
      </div>

      {sub && (
        <div className="text-xs font-mono" style={{ color: 'var(--muted)' }}>
          {sub}
        </div>
      )}

      {/* Bottom accent line */}
      <div
        className="absolute bottom-0 left-0 h-0.5 w-full opacity-40"
        style={{ background: `linear-gradient(90deg, ${color}, transparent)` }}
      />
    </div>
  )
}
