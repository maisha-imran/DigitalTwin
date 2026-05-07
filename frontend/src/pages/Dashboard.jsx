import React, { useEffect, useState } from 'react'
import { api } from '../services/api'
import MetricCard from '../components/MetricCard'
import LossChart from '../components/LossChart'
import RiskPanel from '../components/RiskPanel'
import {
  Activity, Cpu, AlertTriangle, DollarSign, Loader, Play, CheckCircle,
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

const CONDITION_COLORS = {
  Good: '#00ff9d', Fair: '#ffd700', Moderate: '#ff8c00', Poor: '#ff4500', Critical: '#ff2d55',
}

export default function Dashboard() {
  const [metricsData, setMetricsData] = useState(null)
  const [roadsData, setRoadsData]   = useState(null)
  const [training, setTraining]     = useState(false)
  const [trainResult, setTrainResult] = useState(null)
  const [error, setError]           = useState(null)
  const [epochs, setEpochs]         = useState(80)

  const fetchAll = async () => {
    try {
      const [m, r] = await Promise.all([api.getMetrics(), api.getRoads()])
      setMetricsData(m)
      setRoadsData(r)
    } catch (_) {}
  }

  useEffect(() => { fetchAll() }, [])

  const handleTrain = async () => {
    setTraining(true)
    setError(null)
    try {
      const res = await api.train({ epochs, lambda_physics: 0.3 })
      setTrainResult(res)
      await fetchAll()
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setTraining(false)
    }
  }

  const metrics = metricsData?.metrics || {}
  const history = metricsData?.history || {}
  const stats   = roadsData?.stats || {}
  const dist    = roadsData?.condition_distribution || {}
  const total   = roadsData?.total || 0

  const distChartData = Object.entries(dist).map(([k, v]) => ({
    name: k, count: v, color: CONDITION_COLORS[k] || '#888',
  }))

  return (
    <div className="p-6 max-w-screen-xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1
            className="text-2xl font-display font-bold tracking-widest glow-cyan"
            style={{ color: 'var(--accent)' }}
          >
            SYSTEM DASHBOARD
          </h1>
          <p className="text-sm font-mono mt-1" style={{ color: 'var(--muted)' }}>
            Physics-Informed GNN · Road Deterioration Forecasting
          </p>
        </div>

        {/* Train control */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <label className="text-xs font-mono" style={{ color: 'var(--muted)' }}>EPOCHS</label>
            <input
              type="number"
              min={10}
              max={300}
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
              className="w-20 px-2 py-1 rounded text-sm font-mono text-center"
              style={{
                background: 'var(--panel)',
                border: '1px solid var(--border)',
                color: 'var(--accent)',
              }}
            />
          </div>
          <button
            onClick={handleTrain}
            disabled={training}
            className="flex items-center gap-2 px-5 py-2 rounded font-mono text-sm font-semibold transition-all disabled:opacity-50"
            style={{
              background: 'rgba(0,212,255,0.12)',
              border: '1px solid rgba(0,212,255,0.4)',
              color: 'var(--accent)',
              boxShadow: training ? 'none' : '0 0 16px rgba(0,212,255,0.2)',
            }}
          >
            {training ? <Loader size={14} className="animate-spin" /> : <Play size={14} />}
            {training ? 'TRAINING…' : 'TRAIN MODEL'}
          </button>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div
          className="panel px-4 py-3 flex items-center gap-2 text-sm font-mono"
          style={{ borderColor: 'var(--red)', color: 'var(--red)' }}
        >
          <AlertTriangle size={14} />
          {error}
        </div>
      )}

      {/* Train success */}
      {trainResult && (
        <div
          className="panel px-4 py-3 flex items-center gap-2 text-sm font-mono"
          style={{ borderColor: 'var(--green)', color: 'var(--green)' }}
        >
          <CheckCircle size={14} />
          Training complete · MAE: {trainResult.metrics?.mae} · R²: {trainResult.metrics?.r2} · Roads: {trainResult.total_roads}
        </div>
      )}

      {/* Metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Val MAE"
          value={metrics.mae ?? '—'}
          unit="IRI"
          icon={Activity}
          color="var(--accent)"
          sub="Mean Absolute Error"
        />
        <MetricCard
          label="Val RMSE"
          value={metrics.rmse ?? '—'}
          unit="IRI"
          icon={Cpu}
          color="var(--purple)"
          sub="Root Mean Squared Error"
        />
        <MetricCard
          label="R² Score"
          value={metrics.r2 ?? '—'}
          icon={CheckCircle}
          color="var(--green)"
          sub="Coefficient of Determination"
        />
        <MetricCard
          label="Critical Roads"
          value={stats.critical_roads ?? '—'}
          icon={AlertTriangle}
          color="var(--red)"
          sub={`of ${total} total segments`}
        />
      </div>

      {/* Second row: costs + poor roads */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Total Repair Cost"
          value={stats.total_repair_cost_usd ? `$${(stats.total_repair_cost_usd / 1e6).toFixed(2)}M` : '—'}
          icon={DollarSign}
          color="var(--yellow)"
          sub="Estimated budget needed"
        />
        <MetricCard
          label="Poor Roads"
          value={stats.poor_roads ?? '—'}
          icon={AlertTriangle}
          color="var(--orange)"
          sub="Requiring attention"
        />
        <MetricCard
          label="Avg IRI (Current)"
          value={stats.avg_iri_current ?? '—'}
          unit="m/km"
          icon={Activity}
          color="var(--muted)"
          sub="Network average"
        />
        <MetricCard
          label="Avg IRI (Forecast)"
          value={stats.avg_iri_predicted ?? '—'}
          unit="m/km"
          icon={Activity}
          color="var(--orange)"
          sub="Predicted 6-month"
        />
      </div>

      {/* Charts row */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="md:col-span-2">
          <LossChart history={history} />
        </div>
        <RiskPanel distribution={dist} total={total} />
      </div>

      {/* Condition bar chart */}
      {distChartData.length > 0 && (
        <div className="panel p-5">
          <div className="text-xs font-mono uppercase tracking-widest mb-4" style={{ color: 'var(--muted)' }}>
            Road Count by Condition Class
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={distChartData} margin={{ top: 4, right: 20, bottom: 0, left: 0 }}>
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#0a0f1e', border: '1px solid #0f2040', fontFamily: 'JetBrains Mono', fontSize: 11 }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {distChartData.map((entry) => (
                  <Cell key={entry.name} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Model summary */}
      <div className="panel p-5">
        <div className="text-xs font-mono uppercase tracking-widest mb-4" style={{ color: 'var(--muted)' }}>
          Model Architecture
        </div>
        <div className="grid md:grid-cols-2 gap-6 text-sm font-mono">
          <div className="space-y-2">
            {[
              ['Architecture', 'Physics-Informed GAT'],
              ['GAT Layers', '3 × Graph Attention'],
              ['Attention Heads', '4 per layer'],
              ['Hidden Dim', '64'],
              ['Input Features', '9'],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between" style={{ borderBottom: '1px solid var(--border)', paddingBottom: 4 }}>
                <span style={{ color: 'var(--muted)' }}>{k}</span>
                <span style={{ color: 'var(--accent)' }}>{v}</span>
              </div>
            ))}
          </div>
          <div className="space-y-2">
            {[
              ['Loss', 'MSE + λ·Physics'],
              ['Physics Constraints', '4 (Monotonicity, Traffic, Rain, Bounds)'],
              ['Optimizer', 'AdamW + CosineAnneal'],
              ['Output', 'IRI Regression'],
              ['Framework', 'PyTorch Geometric'],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between" style={{ borderBottom: '1px solid var(--border)', paddingBottom: 4 }}>
                <span style={{ color: 'var(--muted)' }}>{k}</span>
                <span style={{ color: 'var(--green)' }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
