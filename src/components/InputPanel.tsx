import React, { useState } from 'react'
import { Mortar, Region } from '../types'

interface InputPanelProps {
  mortars: Mortar[]
  regions: Region[]
  onSubmit: (data: { desiredStrength: number; threshold: number; mortarId: string; regionId: string }) => void
  loading: boolean
}

const InputPanel: React.FC<InputPanelProps> = ({ mortars, regions, onSubmit, loading }) => {
  const [desiredStrength, setDesiredStrength] = useState<number>(30)
  const [mortarId, setMortarId] = useState<string>(mortars[0]?.id || '')
  const [regionId, setRegionId] = useState<string>(regions[0]?.id || '')
  const [threshold, setThreshold] = useState<number>(5)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (desiredStrength && mortarId && regionId) {
      onSubmit({ desiredStrength, threshold, mortarId, regionId })
    }
  }

  return (
    <section className="border border-gray-600 p-4 bg-white text-black">
      <h2 className="text-xl mb-4">Input Parameters</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="space-y-2">
          <label className="block text-sm">Desired Compressive Strength (MPa)</label>
          <input
            type="number"
            value={desiredStrength}
            onChange={(e) => setDesiredStrength(Number(e.target.value))}
            min="10"
            max="100"
            step="0.1"
            className="basic-input"
            placeholder="Enter desired strength (MPa)"
            required
            disabled={loading}
          />
          <div className="space-y-2">
            <label className="block text-sm">Target Tolerance (± MPa)</label>
            <input
              type="number"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              min="0"
              max="30"
              step="0.1"
              className="basic-input"
              placeholder="Acceptable deviation in MPa"
              required
              disabled={loading}
            />
          </div>
          <div className="text-xs text-gray-700">
            Typical range: 20-60 MPa for structural concrete applications
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="block text-sm">Mortar Selection</label>
            <select
              value={mortarId}
              onChange={(e) => setMortarId(e.target.value)}
              className="basic-input"
              required
              disabled={loading}
            >
              {mortars.map((mortar) => (
                <option key={mortar.id} value={mortar.id}>
                  {mortar.name}
                </option>
              ))}
            </select>
            <div className="text-xs text-gray-700">8 mortar formulations available</div>
          </div>

          <div className="space-y-2">
            <label className="block text-sm">Region Selection</label>
            <select
              value={regionId}
              onChange={(e) => setRegionId(e.target.value)}
              className="basic-input"
              required
              disabled={loading}
            >
              {regions.map((region) => (
                <option key={region.id} value={region.id}>
                  {region.name}
                </option>
              ))}
            </select>
            <div className="text-xs text-gray-700">Regional rock availability database</div>
          </div>
        </div>

        <div className="space-y-2 mt-4">
          <button
            type="submit"
            disabled={loading}
            className="basic-button w-full"
          >
            {loading ? (
              <span className="inline-flex items-center justify-center">
                <svg className="animate-spin h-4 w-4 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Running Inverse ML Model...
              </span>
            ) : (
              'Analyze Optimal Mix Design'
            )}
          </button>
          <div className="text-center text-xs text-gray-700 pt-2 border-t border-gray-400">
            <p>Model will evaluate available rocks against strength requirements</p>
          </div>
        </div>
      </form>
    </section>
  )
}

export default InputPanel
