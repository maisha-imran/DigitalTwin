import axios from 'axios'

const BASE = '/api'

export const api = {
  train: (params) =>
    axios.post(`${BASE}/train`, params).then((r) => r.data),

  predict: (params) =>
    axios.post(`${BASE}/predict`, params).then((r) => r.data),

  getMetrics: () =>
    axios.get(`${BASE}/metrics`).then((r) => r.data),

  getRoads: (limit = 500) =>
    axios.get(`${BASE}/roads?limit=${limit}`).then((r) => r.data),

  getMaintenance: (topN = 50) =>
    axios.get(`${BASE}/maintenance?top_n=${topN}`).then((r) => r.data),
}

export const CONDITION_COLORS = {
  Good:     '#00ff9d',
  Fair:     '#ffd700',
  Moderate: '#ff8c00',
  Poor:     '#ff4500',
  Critical: '#ff2d55',
}

export const URGENCY_COLORS = {
  None:      '#00ff9d',
  Low:       '#ffd700',
  Medium:    '#ff8c00',
  High:      '#ff4500',
  Immediate: '#ff2d55',
}

export const CONDITION_ORDER = ['Good', 'Fair', 'Moderate', 'Poor', 'Critical']
