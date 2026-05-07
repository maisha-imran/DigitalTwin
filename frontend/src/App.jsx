import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import PredictionMap from './pages/PredictionMap'
import Evaluation from './pages/Evaluation'
import Maintenance from './pages/Maintenance'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen grid-bg" style={{ background: 'var(--bg)' }}>
        <Navbar />
        <main className="pt-16">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard"   element={<Dashboard />} />
            <Route path="/map"         element={<PredictionMap />} />
            <Route path="/evaluation"  element={<Evaluation />} />
            <Route path="/maintenance" element={<Maintenance />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
