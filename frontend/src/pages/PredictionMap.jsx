import React, { useEffect, useState } from 'react'
import { api } from '../services/api'
import RoadMap from '../components/RoadMap'
import RiskPanel from '../components/RiskPanel'
import { Loader, RefreshCw, CloudRain, Car } from 'lucide-react'

export default function PredictionMap() {
  const [roads, setRoads]       = useState([])
  const [stats, setStats]       = useState({})
  const [dist, setDist]         = useState({})
  const [total, setTotal]       = useState(0)
  const [loading, setLoading]   = useState(false)
  const [traffic, setTraffic]   = useState(1.0)
  const [rainfall, setRainfall] = useState(1.0)
  const [error, setError]       = useState(null)

  const fetchRoads = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getRoads(500)
      setRoads(data.roads || [])
      setStats(data.stats || {})
      setDist(data.condition_distribution || {})
      setTotal(data.total || 0)
    } catch (e) {
      setError('Backend unavailable — please start FastAPI and train the model.')
    } finally {
      setLoading(false)
    }
  }

  const handlePredict = async () => {
    setLoading(true)
    setError(null)
    try {
      await api.predict({ traffic_scale: traffic, rainfall_scale: rainfall })
      await fetchRoads()
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchRoads() }, [])

  return (
    <div className="flex h-[calc(100vh-64px)]">
      {/* Sidebar */}
      <div
        className="w-72 flex-shrink-0 overflow-y-auto p-4 space-y-4 border-r"
        style={{ background: 'rgba(10,15,30,0.95)', borderColor: 'var(--border)' }}
      >
        <div>
          <h2 className="text-sm font-display font-bold tracking-widest" style={{ color: 'var(--accent)' }}>
            PREDICTION MAP
          </h2>
          <p className="text-xs font-mono mt-1" style={{ color: 'var(--muted)' }}>
            6-month IRI forecast
          </p>
        </div>

        {/* Environmental sliders */}
        <div className="panel p-4 space-y-4">
          <div className="text-xs font-mono uppercase tracking-widest" style={{ color: 'var(--muted)' }}>
            Environmental Parameters
          </div>

          <div>
            <div className="flex items-center justify-between text-xs font-mono mb-2">
              <div className="flex items-center gap-1" style={{ color: 'var(--text)' }}>
                <Car size={11} /> Traffic Scale
              </div>
              <span style={{ color: 'var(--accent)' }}>{traffic.toFixed(1)}×</span>
            </div>
            <input
              type="range" min={0.5} max={3.0} step={0.1}
              value={traffic}
              onChange={(e) => setTraffic(Number(e.target.value))}
              className="w-full accent-cyan-400"
            />
          </div>

          <div>
            <div className="flex items-center justify-between text-xs font-mono mb-2">
              <div className="flex items-center gap-1" style={{ color: 'var(--text)' }}>
                <CloudRain size={11} /> Rainfall Scale
              </div>
              <span style={{ color: 'var(--accent)' }}>{rainfall.toFixed(1)}×</span>
            </div>
            <input
              type="range" min={0.5} max={3.0} step={0.1}
              value={rainfall}
              onChange={(e) => setRainfall(Number(e.target.value))}
              className="w-full accent-cyan-400"
            />
          </div>

          <button
            onClick={handlePredict}
            disabled={loading}
            className="w-full py-2 rounded text-sm font-mono font-semibold transition-all flex items-center justify-center gap-2"
            style={{
              background: 'rgba(0,212,255,0.12)',
              border: '1px solid rgba(0,212,255,0.3)',
              color: 'var(--accent)',
            }}
          >
            {loading ? <Loader size={13} className="animate-spin" /> : <RefreshCw size={13} />}
            RUN FORECAST
          </button>
        </div>

        <RiskPanel distribution={dist} total={total} />

        {/* Stats */}
        <div className="panel p-4 space-y-2">
          <div className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: 'var(--muted)' }}>
            Network Stats
          </div>
          {[
            ['Avg IRI Now',      `${stats.avg_iri_current ?? '—'} m/km`],
            ['Avg IRI Forecast', `${stats.avg_iri_predicted ?? '—'} m/km`],
            ['Critical',         stats.critical_roads ?? '—'],
            ['Poor',             stats.poor_roads ?? '—'],
            ['Est. Cost',        stats.total_repair_cost_usd ? `$${(stats.total_repair_cost_usd/1e6).toFixed(1)}M` : '—'],
          ].map(([k, v]) => (
            <div key={k} className="flex justify-between text-xs font-mono py-1" style={{ borderBottom: '1px solid var(--border)' }}>
              <span style={{ color: 'var(--muted)' }}>{k}</span>
              <span style={{ color: 'var(--text)' }}>{v}</span>
            </div>
          ))}
        </div>

        {error && (
          <div className="panel p-3 text-xs font-mono" style={{ borderColor: 'var(--red)', color: 'var(--red)' }}>
            {error}
          </div>
        )}
      </div>

      {/* Map */}
      <div className="flex-1 relative">
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center" style={{ background: 'rgba(3,7,18,0.6)' }}>
            <div className="flex flex-col items-center gap-3">
              <Loader size={28} className="animate-spin" style={{ color: 'var(--accent)' }} />
              <span className="text-sm font-mono" style={{ color: 'var(--accent)' }}>Loading road data…</span>
            </div>
          </div>
        )}
        <RoadMap roads={roads} height="100%" />
      </div>
    </div>
  )
}
