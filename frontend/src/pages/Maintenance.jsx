import React, { useEffect, useState } from 'react'
import { api, URGENCY_COLORS } from '../services/api'
import MaintenanceTable from '../components/MaintenanceTable'
import MetricCard from '../components/MetricCard'
import { Wrench, DollarSign, AlertTriangle, Download, Loader, TrendingUp } from 'lucide-react'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const URGENCY_ORDER = ['Immediate', 'High', 'Medium', 'Low', 'None']

export default function Maintenance() {
  const [data, setData]   = useState(null)
  const [loading, setLoading] = useState(true)
  const [topN, setTopN]   = useState(50)

  const load = async (n = topN) => {
    setLoading(true)
    try {
      const res = await api.getMaintenance(n)
      setData(res)
    } catch (_) {}
    finally { setLoading(false) }
  }

  useEffect(() => { load() }, [])

  const handleExport = () => {
    if (!data?.roads?.length) return
    const headers = Object.keys(data.roads[0]).join(',')
    const rows = data.roads.map((r) => Object.values(r).join(',')).join('\n')
    const blob = new Blob([headers + '\n' + rows], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'maintenance_plan.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  const roads = data?.roads || []
  const urgencyBreakdown = data?.urgency_breakdown || {}
  const totalCost = data?.total_estimated_cost_usd || 0

  const pieData = URGENCY_ORDER
    .filter((u) => urgencyBreakdown[u])
    .map((u) => ({ name: u, value: urgencyBreakdown[u], color: URGENCY_COLORS[u] }))

  const immediateCnt = urgencyBreakdown['Immediate'] || 0
  const highCnt      = urgencyBreakdown['High'] || 0

  return (
    <div className="p-6 max-w-screen-xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold tracking-widest" style={{ color: 'var(--accent)' }}>
            MAINTENANCE PLANNER
          </h1>
          <p className="text-sm font-mono mt-1" style={{ color: 'var(--muted)' }}>
            Priority ranking · Budget estimation · Export
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <label className="text-xs font-mono" style={{ color: 'var(--muted)' }}>TOP N</label>
            <select
              value={topN}
              onChange={(e) => { setTopN(Number(e.target.value)); load(Number(e.target.value)) }}
              className="px-2 py-1 rounded text-sm font-mono"
              style={{ background: 'var(--panel)', border: '1px solid var(--border)', color: 'var(--accent)' }}
            >
              {[20, 50, 100, 200].map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <button
            onClick={handleExport}
            className="flex items-center gap-2 px-4 py-2 rounded text-sm font-mono font-semibold transition-all"
            style={{
              background: 'rgba(0,255,157,0.1)',
              border: '1px solid rgba(0,255,157,0.3)',
              color: 'var(--green)',
            }}
          >
            <Download size={14} />
            EXPORT CSV
          </button>
        </div>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Total Est. Cost"
          value={`$${(totalCost / 1e6).toFixed(2)}M`}
          icon={DollarSign}
          color="var(--yellow)"
          sub="Top priority segments"
        />
        <MetricCard
          label="Immediate Action"
          value={immediateCnt}
          icon={AlertTriangle}
          color="var(--red)"
          sub="Critical deterioration"
        />
        <MetricCard
          label="High Priority"
          value={highCnt}
          icon={TrendingUp}
          color="var(--orange)"
          sub="Urgent intervention"
        />
        <MetricCard
          label="Ranked Segments"
          value={roads.length}
          icon={Wrench}
          color="var(--accent)"
          sub="In current view"
        />
      </div>

      {/* Pie + budget breakdown */}
      <div className="grid md:grid-cols-3 gap-4">
        {/* Urgency Pie */}
        <div className="panel p-5">
          <div className="text-xs font-mono uppercase tracking-widest mb-4" style={{ color: 'var(--muted)' }}>
            Urgency Distribution
          </div>
          {pieData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={80}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {pieData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ background: '#0a0f1e', border: '1px solid #0f2040', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                />
                <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-48 text-xs font-mono" style={{ color: 'var(--muted)' }}>
              No data — train model first
            </div>
          )}
        </div>

        {/* Budget breakdown */}
        <div className="panel p-5 md:col-span-2">
          <div className="text-xs font-mono uppercase tracking-widest mb-4" style={{ color: 'var(--muted)' }}>
            Budget Summary
          </div>
          <div className="space-y-3">
            {URGENCY_ORDER.map((u) => {
              const count = urgencyBreakdown[u] || 0
              if (!count) return null
              const color = URGENCY_COLORS[u]
              const segCosts = roads
                .filter((r) => r.urgency_predicted === u)
                .reduce((s, r) => s + (r.repair_cost_usd || 0), 0)
              return (
                <div key={u} className="flex items-center justify-between py-2" style={{ borderBottom: '1px solid var(--border)' }}>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 rounded-full" style={{ background: color, boxShadow: `0 0 6px ${color}` }} />
                    <span className="text-sm font-mono" style={{ color }}>{u}</span>
                    <span className="text-xs font-mono" style={{ color: 'var(--muted)' }}>({count} segments)</span>
                  </div>
                  <span className="text-sm font-mono" style={{ color: 'var(--text)' }}>
                    ${segCosts.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                  </span>
                </div>
              )
            })}
            <div className="flex justify-between pt-2 text-sm font-mono font-bold">
              <span style={{ color: 'var(--muted)' }}>TOTAL ESTIMATE</span>
              <span style={{ color: 'var(--yellow)' }}>
                ${totalCost.toLocaleString('en-US', { maximumFractionDigits: 0 })}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Table */}
      {loading ? (
        <div className="flex justify-center py-16">
          <Loader size={24} className="animate-spin" style={{ color: 'var(--accent)' }} />
        </div>
      ) : (
        <div className="panel p-5">
          <div className="text-xs font-mono uppercase tracking-widest mb-4" style={{ color: 'var(--muted)' }}>
            Priority Ranking Table
          </div>
          <MaintenanceTable roads={roads} pageSize={15} />
        </div>
      )}
    </div>
  )
}
