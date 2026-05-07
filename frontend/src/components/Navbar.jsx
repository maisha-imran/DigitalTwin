import React, { useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { Activity, Map, BarChart3, Wrench, Radio } from 'lucide-react'

const NAV = [
  { to: '/dashboard',   icon: Activity,  label: 'Dashboard' },
  { to: '/map',         icon: Map,       label: 'Prediction Map' },
  { to: '/evaluation',  icon: BarChart3, label: 'Evaluation' },
  { to: '/maintenance', icon: Wrench,    label: 'Maintenance' },
]

export default function Navbar() {
  const location = useLocation()
  const [pulse, setPulse] = useState(true)

  return (
    <header
      className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 h-16"
      style={{
        background: 'rgba(3,7,18,0.92)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid #0f2040',
        boxShadow: '0 0 40px rgba(0,212,255,0.06)',
      }}
    >
      {/* Brand */}
      <div className="flex items-center gap-3">
        <div className="relative flex items-center justify-center w-8 h-8">
          <div
            className="absolute w-8 h-8 rounded-full pulse-ring"
            style={{ background: 'rgba(0,212,255,0.2)', border: '1px solid rgba(0,212,255,0.5)' }}
          />
          <Radio size={16} style={{ color: 'var(--accent)', zIndex: 1 }} />
        </div>
        <div>
          <div
            className="text-sm font-display font-bold tracking-widest glow-cyan"
            style={{ color: 'var(--accent)', letterSpacing: '0.15em' }}
          >
            ROAD DIGITAL TWIN
          </div>
          <div className="text-xs font-mono" style={{ color: 'var(--muted)', letterSpacing: '0.1em' }}>
            PI-GNN SYSTEM v1.0
          </div>
        </div>
      </div>

      {/* Nav links */}
      <nav className="flex items-center gap-1">
        {NAV.map(({ to, icon: Icon, label }) => {
          const active = location.pathname === to
          return (
            <NavLink
              key={to}
              to={to}
              className="flex items-center gap-2 px-4 py-2 rounded text-sm font-body transition-all duration-200"
              style={{
                color: active ? 'var(--accent)' : 'var(--muted)',
                background: active ? 'rgba(0,212,255,0.08)' : 'transparent',
                border: active ? '1px solid rgba(0,212,255,0.2)' : '1px solid transparent',
                boxShadow: active ? '0 0 12px rgba(0,212,255,0.15)' : 'none',
                fontWeight: active ? 600 : 400,
              }}
            >
              <Icon size={15} />
              <span className="hidden md:inline">{label}</span>
            </NavLink>
          )
        })}
      </nav>

      {/* Status indicator */}
      <div className="flex items-center gap-2">
        <div className="relative w-2 h-2">
          <div className="absolute inset-0 rounded-full bg-green-400 animate-ping opacity-75" />
          <div className="relative rounded-full w-2 h-2" style={{ background: 'var(--green)' }} />
        </div>
        <span className="text-xs font-mono" style={{ color: 'var(--green)' }}>LIVE</span>
      </div>
    </header>
  )
}
