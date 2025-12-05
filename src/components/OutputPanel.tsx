import React from 'react'
import { ApiResponse, CustomMortar } from '../types'
import mortarDb from '../../data/mortar.json'
import rockDb from '../../data/rock.json'
import { useTheme } from '../themes/ThemeContext'

interface MortarDbEntry {
  name: string
  description?: string
  properties?: {
    splitting_strength_mpa: number
    shrinkage_inches: number
    flexural_strength_mpa: number
    slump_inches: number
    compressive_strength_mpa: number
    poissons_ratio: number
  }
  meta?: {
    gwp: number | null
    product_name_long: string
    manufacturer: string | null
    cost_per_pound: number | null
  } | null
}

interface RockDbEntry {
  name: string
  description?: string
  properties?: {
    compressive_strength_mpa: number
    size_inches: number
    density_lb_ft3: number
    specific_gravity: number
  }
  meta?: {
    source?: string
    type?: string
    common_uses?: string[]
  }
}

type MortarDb = Record<string, MortarDbEntry>
type RockDb = Record<string, RockDbEntry>

interface OutputPanelProps {
  result: ApiResponse | null
  loading: boolean
  error: string | null
  mortarMode: 'default' | 'custom'
  customMortars: CustomMortar[]
}

const OutputPanel: React.FC<OutputPanelProps> = ({ result, loading, error, mortarMode, customMortars }) => {
  const { themeId } = useTheme()
  const isConstruction = themeId === 'construction'
  
  // Placeholder/mock data for demonstration
  const mockResult: ApiResponse = {
    predictions: [
      {
        rock_id: 'RK001',
        predicted_compressive_strength_mpa: 35.2,
        rock_ratio: 0.4,
        mortar_id: 'M001',
      },
      {
        rock_id: 'RK003',
        predicted_compressive_strength_mpa: 42.8,
        rock_ratio: 0.5,
        mortar_id: 'M001',
      }
    ],
    status: 'success'
  }

  const displayResult = result || (loading ? null : mockResult)

  if (loading) {
    return (
      <section className="border p-4 min-h-[300px] flex items-center justify-center" style={{ borderColor: 'var(--theme-panel-border)', backgroundColor: 'var(--theme-panel-bg)', color: 'var(--theme-panel-text)' }}>
        <div className="text-center">
          <h2 className="text-xl mb-4">Model Output</h2>
          <div className="flex items-center justify-center mb-2">
            <svg className="animate-spin h-8 w-8" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          </div>
          <p className="text-sm">Analyzing Concrete Mixtures...</p>
        </div>
      </section>
    )
  }

  if (error) {
    return (
      <section className="border p-4" style={{ borderColor: 'var(--theme-panel-border)', backgroundColor: 'var(--theme-panel-bg)', color: 'var(--theme-panel-text)' }}>
        <h2 className="text-xl mb-4">Model Output</h2>
        <div className="text-center py-4">
          <div className="mb-4 p-3 border border-red-600 inline-block text-red-700" style={{ borderColor: '#ef4444', color: '#991b1b' }}>
            <h3 className="text-lg mb-2">Analysis Error</h3>
            <p className="text-sm">{error}</p>
          </div>
          <p className="text-xs">Please verify input parameters and try again</p>
        </div>
      </section>
    )
  }

  if (!displayResult?.predictions?.length) {
    return (
      <section className="border p-4" style={{ borderColor: 'var(--theme-panel-border)', backgroundColor: 'var(--theme-panel-bg)', color: 'var(--theme-panel-text)' }}>
        <h2 className="text-xl mb-4">{displayResult?.mocked ? 'Model Output (Mocked)' : 'Model Output'}</h2>
        <div className="text-center py-10">
          <div className="text-4xl mb-3">∅</div>
          <p className="text-base mb-2">No Viable Mixtures Found</p>
          <div className="text-sm max-w-md mx-auto" style={{ color: 'var(--theme-panel-text)', opacity: 0.8 }}>
            <p>• No rocks in the selected region meet strength requirements</p>
            <p>• Consider changing target strength or selecting different parameters</p>
          </div>
        </div>
      </section>
    )
  }

  return (
    <section className="border p-4" style={{ borderColor: 'var(--theme-panel-border)', backgroundColor: 'var(--theme-panel-bg)', color: 'var(--theme-panel-text)' }}>
      <h2 className="text-xl mb-4">
      Predicted Mix Designs 
      {displayResult?.mocked ? ' (Mocked)' : ''}
      </h2>

      <div className="space-y-4">
        {displayResult.predictions.map((prediction, index) => (
          <div
            key={`${prediction.rock_id}-${index}`}
            className={`border p-3 ${isConstruction ? 'theme-panel-construction' : ''}`}
            style={{ borderColor: 'var(--theme-panel-border)' }}
          >
            <div className="flex justify-between items-start mb-3 pb-2 border-b gap-4" style={{ borderColor: 'var(--theme-panel-border)' }}>
              <div className="flex-1 min-w-0">
                {(() => {
                  const rockName = (rockDb as RockDb)[prediction.rock_id]?.name || `Rock ${prediction.rock_id}`
                  const mortarName = (mortarDb as MortarDb)[prediction.mortar_id]?.name || prediction.mortar_id
                  return (
                    <>
                      <h3 className="text-lg">
                          {mortarName} + {Math.round(prediction.rock_ratio * 100)}% {rockName}  
                      </h3>
                    </>
                  )
                })()}
              </div>
              <div className="flex-shrink-0">
                <div className="border px-3 py-2" style={{ borderColor: 'var(--theme-panel-border)' }}>
                  <span className="block text-2xl">
                    {prediction.predicted_compressive_strength_mpa.toFixed(1)}
                  </span>
                  <span className="text-xs">MPa</span>
                </div>
              </div>
            </div>

            <div className="pl-3 my-2 relative" style={{ 
              background: 'linear-gradient(to right, rgba(255, 255, 0, 0.7) 0%, rgba(255, 255, 0, 0.7) 6px, transparent 6px)'
            }}>

              {/* Mortar meta table */}
              {(() => {
                const mortarIdRaw = String(prediction.mortar_id)
                // If mortar id is an integer index and custom mortars present, use that
                let mortarMeta: CustomMortar['meta'] | MortarDbEntry['meta'] | null = null
                let mortarName: string | null = null

                if (mortarMode === 'custom' && Array.isArray(customMortars)) {
                  const idx = Number(mortarIdRaw)
                  if (!Number.isNaN(idx) && customMortars[idx]) {
                    mortarMeta = customMortars[idx].meta
                    mortarName = customMortars[idx].name
                  }
                }

                // Fallback to database lookup
                if (!mortarMeta && (mortarDb as MortarDb)[mortarIdRaw]) {
                  const db = (mortarDb as MortarDb)[mortarIdRaw]
                  mortarMeta = db.meta || null
                  mortarName = db.meta?.product_name_long || db.name
                }

                if (!mortarMeta) {
                  return (
                    <div className="text-xs" style={{ color: 'var(--theme-panel-text)', opacity: 0.7 }}>Mortar information not available</div>
                  )
                }

                return (
                  <div className="mt-3">
                    <table className="w-full text-sm">
                      <tbody>
                        <tr>
                          <td className="text-xs font-bold pr-3 py-1">Name</td>
                          <td className="py-1" style={{ color: 'var(--theme-panel-text)' }}>{mortarName}</td>
                        </tr>
                        <tr className="border-t" style={{ borderColor: 'var(--theme-panel-border)' }}>
                          <td className="text-xs font-bold pr-3 py-1">Manufacturer</td>
                          <td className="py-1" style={{ color: 'var(--theme-panel-text)' }}>{mortarMeta.manufacturer ?? 'N/A'}</td>
                        </tr>
                        <tr className="border-t" style={{ borderColor: 'var(--theme-panel-border)' }}>
                          <td className="text-xs font-bold pr-3 py-1">GWP</td>
                          <td className="py-1" style={{ color: 'var(--theme-panel-text)' }}>{mortarMeta.gwp ?? 'N/A'}</td>
                        </tr>
                        <tr className="border-t" style={{ borderColor: 'var(--theme-panel-border)' }}>
                          <td className="text-xs font-bold pr-3 py-1">Cost / lb</td>
                          <td className="py-1" style={{ color: 'var(--theme-panel-text)' }}>{mortarMeta.cost_per_pound ?? 'N/A'}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                )
              })()}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

export default OutputPanel
