import React from 'react'
import { ApiResponse } from '../types'

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
      <section className="bg-neutral-800 border-2 border-neutral-600 rounded-xl p-8 shadow-xl shadow-black/20 min-h-[500px] flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-3xl font-bold mb-8 uppercase tracking-widest border-b-2 border-neutral-500 pb-4 bg-neutral-700 px-6 py-3 rounded-t-lg inline-block">
            Model Output
          </h2>
          <div className="space-y-6">
            <div className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-12 w-12 text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            </div>
            <div>
              <p className="text-2xl font-mono text-neutral-200 mb-2">Analyzing Concrete Mixtures</p>
              <p className="text-lg text-neutral-400 font-mono">Phase 1: ML Inverse Design Model</p>
            </div>
            <div className="space-y-1 text-sm text-neutral-500 font-mono">
              <p>• Evaluating {mockResult.total_rocks_in_region} rock types in selected region</p>
              <p>• Testing mortar-rock compatibility matrix</p>
              <p>• Optimizing for compressive strength target</p>
            </div>
          </div>
        </div>
      </section>
    )
  }

  if (error) {
    return (
      <section className="bg-red-900/20 border-2 border-red-600/50 rounded-xl p-8 shadow-xl shadow-red-900/20 min-h-[500px]">
        <h2 className="text-3xl font-bold mb-8 text-center uppercase tracking-widest border-b-2 border-red-500/50 pb-4 bg-red-900/30 px-6 py-3 rounded-t-lg">
          Model Output
        </h2>
        <div className="text-center py-8">
          <div className="text-red-400 mb-6 p-4 bg-red-900/30 rounded-lg inline-block">
            <h3 className="text-2xl font-mono uppercase tracking-wide mb-4 text-red-300">
              Analysis Error
            </h3>
            <p className="text-lg text-neutral-200 font-mono mb-4 bg-red-900/20 p-4 rounded">{error}</p>
          </div>
          <p className="text-sm text-neutral-400 font-mono">
            Please verify input parameters and try again
          </p>
        </div>
      </section>
    )
  }

  if (!displayResult?.predictions?.length) {
    return (
      <section className="bg-neutral-800 border-2 border-neutral-600 rounded-xl p-8 shadow-xl shadow-black/20 min-h-[500px]">
        <h2 className="text-3xl font-bold mb-8 text-center uppercase tracking-widest border-b-2 border-neutral-500 pb-4 bg-neutral-700 px-6 py-3 rounded-t-lg">
          {displayResult?.mocked ? 'Model Output (Mocked)' : 'Model Output'}
        </h2>
        <div className="text-center py-16">
          <div className="text-6xl font-mono text-neutral-500 mb-6">∅</div>
          <p className="text-xl text-neutral-300 font-mono mb-4">
            No Viable Mixtures Found
          </p>
          <div className="space-y-2 text-neutral-400 font-mono max-w-md mx-auto">
            <p>• No rocks in the selected region meet strength requirements</p>
            <p>• Consider increasing target strength or selecting different parameters</p>
            <p>• 0/{displayResult?.total_rocks_in_region || 0} rocks eligible</p>
          </div>
        </div>
      </section>
    )
  }

  return (
    <section className="bg-neutral-800 border-2 border-neutral-600 rounded-xl p-8 shadow-xl shadow-black/20">
      <h2 className="text-3xl font-bold mb-8 text-center uppercase tracking-widest border-b-2 border-neutral-500 pb-4 bg-neutral-700 px-6 py-3 rounded-t-lg">
        {displayResult?.mocked ? 'Predicted Mix Designs (Mocked)' : 'Predicted Mix Designs'}
      </h2>
      
      <div className="bg-neutral-900/50 border border-neutral-600 rounded-lg p-6 mb-8">
        <div className="flex flex-col md:flex-row justify-between items-center text-sm font-mono text-neutral-300 space-y-2 md:space-y-0">
          <div className="flex items-center space-x-6">
            <span className="flex items-center">
              <div className="w-3 h-3 bg-green-400 rounded-full mr-2"></div>
              Eligible Rocks: {displayResult.eligible_rocks}
            </span>
            <span className="flex items-center">
              <div className="w-20 h-1 bg-neutral-600 rounded-full mr-2"></div>
              Total Available: {displayResult.total_rocks_in_region}
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-full bg-neutral-700 rounded-full h-3">
              <div 
                className="bg-green-500 h-3 rounded-full transition-all duration-700" 
                style={{ width: `${(displayResult.eligible_rocks / displayResult.total_rocks_in_region) * 100}%` }}
              ></div>
            </div>
            <span className="text-sm">{Math.round((displayResult.eligible_rocks / displayResult.total_rocks_in_region) * 100)}%</span>
          </div>
        </div>
        <p className="text-xs text-neutral-500 mt-2 text-center font-mono">
          Model evaluated {displayResult.total_rocks_in_region} rock types against {displayResult.predictions[0].mortar_id} mortar
          {displayResult.mocked && ' | Mocked Results'}
        </p>
      </div>

      <div className="space-y-6">
        {displayResult.predictions.map((prediction, index) => (
          <div
            key={`${prediction.rock_id}-${index}`}
            className="bg-neutral-900 border-2 border-neutral-600 rounded-lg p-6 hover:border-green-500/50 hover:bg-neutral-900/50 transition-all duration-200 shadow-lg"
          >
            <div className="flex flex-col md:flex-row justify-between items-start mb-6 pb-4 border-b border-neutral-700">
              <div className="mb-4 md:mb-0">
                <h3 className="text-2xl font-mono uppercase tracking-wide text-neutral-100">
                  Rock {prediction.rock_id}
                </h3>
                <div className="flex items-center space-x-4 text-sm text-neutral-400 font-mono mt-2">
                  <span>{prediction.mortar_id}</span>
                  <span>•</span>
                  <span>{prediction.region_id}</span>
                </div>
              </div>
              <div className="text-right">
                <div className="inline-block bg-green-900/30 border-2 border-green-500/50 px-4 py-2 rounded-lg">
                  <span className="block text-3xl font-bold text-green-400">
                    {prediction.predicted_compressive_strength_mpa.toFixed(1)}
                  </span>
                  <span className="text-sm text-green-300 font-mono uppercase tracking-wide">
                    MPa Predicted
                  </span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 mb-6">
              <div className="bg-neutral-800 p-4 rounded-lg border border-neutral-600">
                <div className="text-3xl font-bold text-blue-400 mb-1">
                  {Math.round(prediction.recommended_rock_ratio * 100)}%
                </div>
                <div className="text-xs uppercase tracking-wide text-neutral-400 font-mono">
                  Optimal Rock Ratio
                </div>
              </div>
            </div>

            <div className="bg-neutral-900/30 border-l-4 border-green-500 pl-4 py-3 rounded">
              <p className="text-sm text-neutral-300 font-mono leading-relaxed">
                <span className="text-green-400 font-bold">RECOMMENDED:</span> This mixture design meets the{' '}
                <span className="text-green-400 font-bold">{prediction.predicted_compressive_strength_mpa.toFixed(1)} MPa</span>{' '}
                compressive strength requirement using{' '}
                <span className="text-blue-400 font-bold">{Math.round(prediction.recommended_rock_ratio * 100)}%</span>{' '}
                <span className="text-neutral-200 font-bold">{prediction.rock_id}</span> aggregate with the{' '}
                <span className="text-neutral-200 font-bold">{prediction.mortar_id}</span> mortar formulation.
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Model confidence removed per requirements */}
    </section>
  )
}

export default OutputPanel
