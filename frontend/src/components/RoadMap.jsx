import React, { useEffect, useRef, useState } from 'react'
import { CONDITION_COLORS } from '../services/api'

// Leaflet loaded via CDN in index.html
const getL = () => window.L

export default function RoadMap({ roads = [], height = '100%' }) {
  const mapRef = useRef(null)
  const mapInstance = useRef(null)
  const markersRef = useRef([])
  const [selected, setSelected] = useState(null)

  useEffect(() => {
    if (!mapRef.current) return
    const L = getL()
    if (!L) return
    if (mapInstance.current) return

    // Default center: Bengaluru
    const center = roads.length
      ? [
          roads.reduce((s, r) => s + r.lat, 0) / roads.length,
          roads.reduce((s, r) => s + r.lon, 0) / roads.length,
        ]
      : [12.9716, 77.5946]

    const map = L.map(mapRef.current, {
      center,
      zoom: 13,
      zoomControl: true,
      preferCanvas: true,
    })

    // Dark tile layer
    L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
      {
        attribution: '©OpenStreetMap ©CARTO',
        subdomains: 'abcd',
        maxZoom: 19,
      }
    ).addTo(map)

    mapInstance.current = map
  }, [])

  useEffect(() => {
    const L = getL()
    if (!L || !mapInstance.current || !roads.length) return

    const map = mapInstance.current

    // Clear old markers
    markersRef.current.forEach((m) => m.remove())
    markersRef.current = []

    roads.forEach((r) => {
      const color = CONDITION_COLORS[r.condition_predicted] || '#888'
      const iri = Number(r.iri_predicted || r.iri_current || 0)
      const radius = Math.max(4, Math.min(14, 4 + iri * 1.2))

      const circle = L.circleMarker([r.lat, r.lon], {
        radius,
        fillColor: color,
        color: color,
        weight: 1,
        opacity: 0.9,
        fillOpacity: 0.75,
      })

      circle.bindPopup(`
        <div style="font-family:'Exo 2',sans-serif;min-width:200px">
          <div style="color:#00d4ff;font-weight:700;margin-bottom:6px;font-size:12px">
            ${String(r.edge_id).slice(0, 16)}
          </div>
          <table style="font-size:11px;width:100%">
            <tr><td style="color:#64748b">Type</td><td style="color:#e2e8f0;text-transform:capitalize">${r.road_type}</td></tr>
            <tr><td style="color:#64748b">IRI Current</td><td style="color:#e2e8f0">${Number(r.iri_current).toFixed(2)}</td></tr>
            <tr><td style="color:#64748b">IRI Forecast</td><td style="color:${color};font-weight:600">${iri.toFixed(2)}</td></tr>
            <tr><td style="color:#64748b">Condition</td><td style="color:${color}">${r.condition_predicted}</td></tr>
            <tr><td style="color:#64748b">Urgency</td><td style="color:#e2e8f0">${r.urgency_predicted}</td></tr>
            <tr><td style="color:#64748b">Cost Est.</td><td style="color:#e2e8f0">$${Number(r.repair_cost_usd||0).toLocaleString('en-US',{maximumFractionDigits:0})}</td></tr>
          </table>
        </div>
      `)

      circle.on('click', () => setSelected(r))
      circle.addTo(map)
      markersRef.current.push(circle)
    })

    // Fit bounds
    if (roads.length) {
      const lats = roads.map((r) => r.lat)
      const lons = roads.map((r) => r.lon)
      map.fitBounds([
        [Math.min(...lats), Math.min(...lons)],
        [Math.max(...lats), Math.max(...lons)],
      ], { padding: [30, 30] })
    }
  }, [roads])

  return (
    <div style={{ position: 'relative', height }}>
      <div ref={mapRef} style={{ width: '100%', height: '100%', borderRadius: '8px' }} />

      {/* Legend */}
      <div
        className="absolute bottom-4 right-4 panel p-3 z-50"
        style={{ minWidth: 140 }}
      >
        <div className="text-xs font-mono mb-2" style={{ color: 'var(--muted)' }}>
          IRI CONDITION
        </div>
        {Object.entries(CONDITION_COLORS).map(([cond, color]) => (
          <div key={cond} className="flex items-center gap-2 mb-1">
            <div className="w-3 h-3 rounded-full" style={{ background: color, boxShadow: `0 0 6px ${color}` }} />
            <span className="text-xs font-mono" style={{ color }}>{cond}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
