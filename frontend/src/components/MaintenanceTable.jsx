import React, { useState } from 'react'
import { CONDITION_COLORS, URGENCY_COLORS } from '../services/api'
import { ChevronUp, ChevronDown } from 'lucide-react'

const UrgencyBadge = ({ urgency }) => (
  <span
    className="px-2 py-0.5 rounded text-xs font-mono font-semibold"
    style={{
      color: URGENCY_COLORS[urgency] || '#888',
      background: `${URGENCY_COLORS[urgency] || '#888'}18`,
      border: `1px solid ${URGENCY_COLORS[urgency] || '#888'}44`,
    }}
  >
    {urgency}
  </span>
)

export default function MaintenanceTable({ roads = [], pageSize = 15 }) {
  const [page, setPage] = useState(0)
  const [sortKey, setSortKey] = useState('iri_predicted')
  const [sortDir, setSortDir] = useState('desc')

  const sorted = [...roads].sort((a, b) => {
    const v = sortDir === 'asc' ? 1 : -1
    return a[sortKey] > b[sortKey] ? v : -v
  })

  const pages = Math.ceil(sorted.length / pageSize)
  const visible = sorted.slice(page * pageSize, page * pageSize + pageSize)

  const handleSort = (key) => {
    if (key === sortKey) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setSortDir('desc') }
  }

  const SortIcon = ({ k }) =>
    sortKey === k ? (sortDir === 'asc' ? <ChevronUp size={12} /> : <ChevronDown size={12} />) : null

  const cols = [
    { key: 'edge_id',          label: 'Segment ID' },
    { key: 'road_type',        label: 'Type' },
    { key: 'iri_current',      label: 'IRI Now' },
    { key: 'iri_predicted',    label: 'IRI Forecast' },
    { key: 'condition_predicted', label: 'Condition' },
    { key: 'urgency_predicted',   label: 'Urgency' },
    { key: 'repair_cost_usd',     label: 'Cost (USD)' },
  ]

  return (
    <div>
      <div className="overflow-x-auto rounded-lg" style={{ border: '1px solid var(--border)' }}>
        <table className="w-full text-xs font-mono">
          <thead>
            <tr style={{ background: '#0a0f1e', borderBottom: '1px solid var(--border)' }}>
              {cols.map(({ key, label }) => (
                <th
                  key={key}
                  className="px-4 py-3 text-left cursor-pointer select-none hover:text-white transition-colors"
                  style={{ color: 'var(--muted)' }}
                  onClick={() => handleSort(key)}
                >
                  <div className="flex items-center gap-1">
                    {label}
                    <SortIcon k={key} />
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.map((r, i) => (
              <tr
                key={r.edge_id || i}
                className="border-b hover:bg-white/5 transition-colors"
                style={{ borderColor: 'var(--border)' }}
              >
                <td className="px-4 py-2.5" style={{ color: 'var(--accent)' }}>
                  {String(r.edge_id).slice(0, 12)}
                </td>
                <td className="px-4 py-2.5 capitalize" style={{ color: 'var(--text)' }}>
                  {r.road_type}
                </td>
                <td className="px-4 py-2.5" style={{ color: 'var(--muted)' }}>
                  {Number(r.iri_current).toFixed(2)}
                </td>
                <td className="px-4 py-2.5 font-semibold" style={{ color: CONDITION_COLORS[r.condition_predicted] || '#fff' }}>
                  {Number(r.iri_predicted).toFixed(2)}
                </td>
                <td className="px-4 py-2.5">
                  <span style={{ color: CONDITION_COLORS[r.condition_predicted] }}>
                    {r.condition_predicted}
                  </span>
                </td>
                <td className="px-4 py-2.5">
                  <UrgencyBadge urgency={r.urgency_predicted} />
                </td>
                <td className="px-4 py-2.5" style={{ color: 'var(--text)' }}>
                  ${Number(r.repair_cost_usd || 0).toLocaleString('en-US', { maximumFractionDigits: 0 })}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {pages > 1 && (
        <div className="flex items-center justify-between mt-3 text-xs font-mono" style={{ color: 'var(--muted)' }}>
          <span>
            Showing {page * pageSize + 1}–{Math.min((page + 1) * pageSize, sorted.length)} of {sorted.length}
          </span>
          <div className="flex gap-2">
            <button
              disabled={page === 0}
              onClick={() => setPage((p) => p - 1)}
              className="px-3 py-1 rounded disabled:opacity-30 hover:text-white transition"
              style={{ border: '1px solid var(--border)' }}
            >
              Prev
            </button>
            <button
              disabled={page >= pages - 1}
              onClick={() => setPage((p) => p + 1)}
              className="px-3 py-1 rounded disabled:opacity-30 hover:text-white transition"
              style={{ border: '1px solid var(--border)' }}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
