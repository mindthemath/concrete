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
    <section className="bg-neutral-800 border-2 border-neutral-600 rounded-xl p-8 shadow-xl shadow-black/20">
      <h2 className="text-3xl font-bold mb-8 text-center uppercase tracking-widest border-b-2 border-neutral-500 pb-4 bg-neutral-700 px-6 py-3 rounded-t-lg">
        Input Parameters
      </h2>
      
      <form onSubmit={handleSubmit} className="space-y-8">
        <div className="space-y-6 mb-16">
          <label className="block text-lg font-mono uppercase tracking-wide text-neutral-300 bg-neutral-700 p-3 rounded">
            Desired Compressive Strength (MPa)
          </label>
          <input
            type="number"
            value={desiredStrength}
            onChange={(e) => setDesiredStrength(Number(e.target.value))}
            min="10"
            max="100"
            step="0.1"
            className="w-full p-5 bg-neutral-900 border-2 border-neutral-500 rounded-lg focus:border-blue-400 focus:outline-none text-xl font-mono text-neutral-100 shadow-inner"
            placeholder="Enter desired strength (MPa)"
            required
            disabled={loading}
          />
          <div className="space-y-3">
            <label className="block text-sm font-mono uppercase tracking-wide text-neutral-300 bg-neutral-700 p-2 rounded">
              Target Tolerance (± MPa)
            </label>
            <input
              type="number"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              min="0"
              max="30"
              step="0.1"
              className="w-full p-4 bg-neutral-900 border-2 border-neutral-500 rounded-lg focus:border-blue-400 focus:outline-none text-lg font-mono text-neutral-100 shadow-inner"
              placeholder="Acceptable deviation in MPa"
              required
              disabled={loading}
            />
          </div>
          <div className="text-sm text-neutral-400 font-mono bg-neutral-900 p-3 rounded">
            Typical range: 20-60 MPa for structural concrete applications
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16 mt-8">
          <div className="space-y-6">
            <label className="block text-lg font-mono uppercase tracking-wide text-neutral-300 bg-neutral-700 p-3 rounded">
              Mortar Selection
            </label>
            <select
              value={mortarId}
              onChange={(e) => setMortarId(e.target.value)}
              className="w-full p-5 bg-neutral-900 border-2 border-neutral-500 rounded-lg focus:border-blue-400 focus:outline-none text-xl font-mono text-neutral-100 shadow-inner appearance-none"
              required
              disabled={loading}
            >
              {mortars.map((mortar) => (
                <option key={mortar.id} value={mortar.id} className="font-mono">
                  {mortar.name}
                </option>
              ))}
            </select>
            <div className="text-xs text-neutral-500 font-mono">8 mortar formulations available</div>
          </div>

          <div className="space-y-6">
            <label className="block text-lg font-mono uppercase tracking-wide text-neutral-300 bg-neutral-700 p-3 rounded">
              Region Selection
            </label>
            <select
              value={regionId}
              onChange={(e) => setRegionId(e.target.value)}
              className="w-full p-5 bg-neutral-900 border-2 border-neutral-500 rounded-lg focus:border-blue-400 focus:outline-none text-xl font-mono text-neutral-100 shadow-inner appearance-none"
              required
              disabled={loading}
            >
              {regions.map((region) => (
                <option key={region.id} value={region.id} className="font-mono">
                  {region.name}
                </option>
              ))}
            </select>
            <div className="text-xs text-neutral-500 font-mono">Regional rock availability database</div>
          </div>

        </div>

        <div className="space-y-4 mt-8 mb-6">
          <button
            type="submit"
            disabled={loading}
            className={`
              w-full py-5 px-8 uppercase font-mono tracking-widest text-xl border-[3px]
              rounded-lg focus:outline-none transition-all duration-200 shadow-lg
              ${loading
                ? 'bg-neutral-700 border-neutral-600 cursor-not-allowed text-neutral-400 hover:bg-neutral-700'
                : 'bg-green-600 hover:bg-green-700 border-green-500 hover:border-green-400 text-white font-bold'
              }
              ${!loading && 'hover:shadow-xl hover:shadow-green-500/25 transform hover:-translate-y-0.5'}
            `}
          >
            {loading ? (
              <>
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Running Inverse ML Model...
                </span>
              </>
            ) : (
              'Analyze Optimal Mix Design'
            )}
          </button>
          <div className="text-center text-sm text-neutral-500 font-mono pt-4 border-t border-neutral-700">
            <p>Model will evaluate available rocks against strength requirements</p>
          </div>
        </div>
      </form>
    </section>
  )
}

export default InputPanel
