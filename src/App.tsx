import { useState } from 'react'
import InputPanel from './components/InputPanel'
import OutputPanel from './components/OutputPanel'
import { Mortar, Region, ApiResponse } from './types'

const SAMPLE_MORTARS: Mortar[] = [
  { id: 'M001', name: 'Standard Portland M001' },
  { id: 'M002', name: 'High Early Strength M002' },
  { id: 'M003', name: 'Sulfate Resistant M003' },
  { id: 'M004', name: 'Low Heat M004' },
  { id: 'M005', name: 'Blended Cement M005' },
  { id: 'M006', name: 'White Cement M006' },
  { id: 'M007', name: 'Rapid Hardening M007' },
  { id: 'M008', name: 'Expansive Cement M008' },
]

const SAMPLE_REGIONS: Region[] = [
  { id: 'R001', name: 'Rocky Mountain Region' },
  { id: 'R002', name: 'Pacific Northwest' },
  { id: 'R003', name: 'Great Plains' },
  { id: 'R004', name: 'Appalachian Region' },
  { id: 'R005', name: 'Southwest Desert' },
]

function App() {
  const [result, setResult] = useState<ApiResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (formData: {
    desiredStrength: number
    threshold: number
    mortarId: string
    regionId: string
  }) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Replace with actual API endpoint
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          desired_compressive_strength_mpa: formData.desiredStrength,
          tolerance_mpa: formData.threshold,
          mortar_id: formData.mortarId,
          region_id: formData.regionId,
        }),
      })

      if (!response.ok) {
        // If API not available, produce a clearly-marked mock result
        const mocked: ApiResponse = mockResults(
          formData.desiredStrength,
          formData.threshold,
          formData.mortarId,
          formData.regionId
        )
        setResult(mocked)
        return
      }

      const data: ApiResponse = await response.json()
      setResult({
        ...data,
        target_strength_mpa: formData.desiredStrength,
        tolerance_mpa: formData.threshold,
        mocked: false,
      })
    } catch (err) {
      // Network/404: show a mock so we can see the flow
      const mocked: ApiResponse = mockResults(
        formData.desiredStrength,
        formData.threshold,
        formData.mortarId,
        formData.regionId
      )
      setResult(mocked)
      setError(null)
    } finally {
      setLoading(false)
    }
  }

  function mockResults(target: number, tol: number, mortarId: string, regionId: string): ApiResponse {
    const within = (v: number) => Math.abs(v - target) <= tol
    const randomRock = (i: number) => `RK${(i + 1).toString().padStart(3, '0')}`
    const strengths = [target - tol * 0.6, target + tol * 0.2, target + tol * 0.9]
      .map((v) => Number(v.toFixed(1)))
    const randomRatio = () => {
      const val = 0.4 + Math.random() * 0.2 // 0.4 - 0.6
      return Math.round(val * 100) / 100
    }
    const predictions = strengths.map((s, i) => ({
      rock_id: randomRock(i),
      predicted_compressive_strength_mpa: s,
      recommended_rock_ratio: randomRatio(),
      mortar_id: mortarId,
      region_id: regionId,
    }))
    const eligible = predictions.filter((p) => within(p.predicted_compressive_strength_mpa)).length
    return {
      predictions,
      eligible_rocks: eligible,
      total_rocks_in_region: 6,
      status: 'success',
      target_strength_mpa: target,
      tolerance_mpa: tol,
      mocked: true,
    }
  }

  return (
    <div className="min-h-screen bg-neutral-900 text-gray-100">
      <div className="container mx-auto px-6 py-10 max-w-[1700px]">
        <header className="text-center mb-12">
          <h1 className="text-4xl md:text-6xl font-mono uppercase tracking-widest border-b-4 border-neutral-600 pb-6 inline-block bg-neutral-800 px-8 py-4 rounded-lg shadow-lg shadow-black/25">
            Concrete Strength Analyzer
          </h1>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 xl:gap-12">
          <InputPanel
            mortars={SAMPLE_MORTARS}
            regions={SAMPLE_REGIONS}
            onSubmit={handleSubmit}
            loading={loading}
          />

          <OutputPanel
            result={result}
            loading={loading}
            error={error}
          />
        </div>

        <footer className="mt-16 text-center text-neutral-500 text-sm border-t-2 border-neutral-800 pt-8 bg-neutral-900 rounded-lg">
          <p className="font-mono">Mind the Math, LLC | September 2025</p>
          <p className="mt-2 text-xs font-mono">
            For demonstration purposes only | Based on 96 experimental samples
          </p>
        </footer>
      </div>
    </div>
  )
}

export default App
