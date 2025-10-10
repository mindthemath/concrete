import React from 'react'
import { ApiResponse } from '../types'
import mortarDb from '../../data/mortar.json'
import rockDb from '../../data/rock.json'
import regionDb from '../../data/region.json'

interface OutputPanelProps {
  result: ApiResponse | null
  loading: boolean
  error: string | null
}

const OutputPanel: React.FC<OutputPanelProps> = ({ result, loading, error }) => {
  // Placeholder/mock data for demonstration
  const mockResult: ApiResponse = {
    predictions: [
      {
        rock_id: 'RK001',
        predicted_compressive_strength_mpa: 35.2,
        recommended_rock_ratio: 0.4,
        mortar_id: 'M001',
        region_id: 'R001'
      },
      {
        rock_id: 'RK003',
        predicted_compressive_strength_mpa: 42.8,
        recommended_rock_ratio: 0.3,
        mortar_id: 'M001',
        region_id: 'R001'
      }
    ],
    eligible_rocks: 2,
    total_rocks_in_region: 4,
    status: 'success'
  }

  const displayResult = result || (loading ? null : mockResult)

  if (loading) {
    return (
      <section className="border border-gray-600 p-4 min-h-[300px] flex items-center justify-center bg-white text-black">
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
      <section className="border border-red-600 p-4 bg-white text-black">
        <h2 className="text-xl mb-4">Model Output</h2>
        <div className="text-center py-4">
          <div className="mb-4 p-3 border border-red-600 inline-block text-red-700">
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
      <section className="border border-gray-600 p-4 bg-white text-black">
        <h2 className="text-xl mb-4">{displayResult?.mocked ? 'Model Output (Mocked)' : 'Model Output'}</h2>
        <div className="text-center py-10">
          <div className="text-4xl mb-3">∅</div>
          <p className="text-base mb-2">No Viable Mixtures Found</p>
          <div className="text-sm max-w-md mx-auto text-gray-800">
            <p>• No rocks in the selected region meet strength requirements</p>
            <p>• Consider increasing target strength or selecting different parameters</p>
            <p>• 0/{displayResult?.total_rocks_in_region || 0} rocks eligible</p>
          </div>
        </div>
      </section>
    )
  }

  return (
    <section className="border border-gray-600 p-4 bg-white text-black">
      <h2 className="text-xl mb-4">{displayResult?.mocked ? 'Predicted Mix Designs (Mocked)' : 'Predicted Mix Designs'}</h2>

      <div className="border border-gray-500 p-3 mb-4">
        <div className="text-sm text-center">
          <span>Results: {displayResult.predictions.length}</span>
        </div>
        <p className="text-xs mt-2 text-center">
          {(() => {
            const regionId = displayResult.predictions[0].region_id
            const available = (regionDb as any)[regionId]?.available_rocks || []
            const k = Array.isArray(available) ? available.length : 0
            return `Model evaluated ${k} rock types against various mortars` + (displayResult.mocked ? ' | Mocked Results' : '')
          })()}
        </p>
      </div>

      <div className="space-y-4">
        {displayResult.predictions.map((prediction, index) => (
          <div
            key={`${prediction.rock_id}-${index}`}
            className="border border-gray-600 p-3"
          >
            <div className="flex flex-col md:flex-row justify-between items-start mb-3 pb-2 border-b border-gray-400">
              <div className="mb-2 md:mb-0">
                {(() => {
                  const rockName = (rockDb as any)[prediction.rock_id]?.name || `Rock ${prediction.rock_id}`
                  const mortarName = (mortarDb as any)[prediction.mortar_id]?.name || prediction.mortar_id
                  return (
                    <>
                      <h3 className="text-lg">{mortarName} + {rockName}</h3>
                      <div className="flex items-center gap-3 text-sm text-gray-800 mt-1">
                        <span>{prediction.mortar_id}</span>
                        <span>•</span>
                        <span>{prediction.rock_id}</span>
                        <span>•</span>
                        <span>{prediction.region_id}</span>
                      </div>
                    </>
                  )
                })()}
              </div>
              <div className="text-right">
                <div className="inline-block border border-gray-600 px-3 py-2">
                  <span className="block text-2xl">
                    {prediction.predicted_compressive_strength_mpa.toFixed(1)}
                  </span>
                  <span className="text-xs">MPa Predicted</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-2 mb-3">
              <div className="p-3 border border-gray-400">
                <div className="text-xl mb-1">
                  {Math.round(prediction.recommended_rock_ratio * 100)}%
                </div>
                <div className="text-xs">Optimal Rock Ratio</div>
              </div>
            </div>

            <div className="border-l-4 border-green-600 pl-3 py-2">
              <div className="mb-2">
                <div className="text-sm font-medium">Prediction Summary</div>
              </div>

              {/* Mortar meta table */}
              {(() => {
                const mortarIdRaw = String(prediction.mortar_id)
                // If mortar id is an integer index and custom mortars present, use that
                let mortarMeta: any = null
                let mortarName: string | null = null

                if (displayResult?.mortar_mode === 'custom' && Array.isArray(displayResult.custom_mortars)) {
                  const idx = Number(mortarIdRaw)
                  if (!Number.isNaN(idx) && displayResult.custom_mortars[idx]) {
                    mortarMeta = displayResult.custom_mortars[idx].meta
                    mortarName = displayResult.custom_mortars[idx].name
                  }
                }

                // Fallback to database lookup
                if (!mortarMeta && (mortarDb as any)[mortarIdRaw]) {
                  const db = (mortarDb as any)[mortarIdRaw]
                  mortarMeta = db.meta
                  mortarName = db.meta?.product_name_long || db.name
                }

                if (!mortarMeta) {
                  return (
                    <div className="text-xs text-gray-700">Mortar information not available</div>
                  )
                }

                return (
                  <div className="mt-3">
                    <div className="text-sm font-medium mb-2">Mortar</div>
                    <table className="w-full text-xs">
                      <tbody>
                        <tr>
                          <td className="font-medium pr-3 py-1">Name</td>
                          <td className="text-gray-800 py-1">{mortarName}</td>
                        </tr>
                        <tr className="border-t">
                          <td className="font-medium pr-3 py-1">Manufacturer</td>
                          <td className="text-gray-800 py-1">{mortarMeta.manufacturer}</td>
                        </tr>
                        <tr className="border-t">
                          <td className="font-medium pr-3 py-1">GWP</td>
                          <td className="text-gray-800 py-1">{mortarMeta.gwp}</td>
                        </tr>
                        <tr className="border-t">
                          <td className="font-medium pr-3 py-1">Cost / lb</td>
                          <td className="text-gray-800 py-1">{mortarMeta.cost_per_pound}</td>
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
