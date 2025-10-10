import { useState } from 'react'
import InputPanel from './components/InputPanel'
import OutputPanel from './components/OutputPanel'
import { Region, ApiResponse, CustomMortar } from './types'

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
    regionId: string
    mortarMode: 'default' | 'custom'
    customMortars: CustomMortar[]
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
          region_id: formData.regionId,
          mortar_mode: formData.mortarMode,
          custom_mortars: formData.customMortars,
        }),
      })

      if (!response.ok) {
        // If API not available, produce a clearly-marked mock result
        const mocked: ApiResponse = mockResults(
          formData.desiredStrength,
          formData.threshold,
          formData.regionId,
          formData.mortarMode,
          formData.customMortars
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
        mortar_mode: formData.mortarMode,
      })
    } catch (err) {
      // Network/404: show a mock so we can see the flow
      const mocked: ApiResponse = mockResults(
        formData.desiredStrength,
        formData.threshold,
        formData.regionId,
        formData.mortarMode,
        formData.customMortars
      )
      setResult(mocked)
      setError(null)
    } finally {
      setLoading(false)
    }
  }

  function mockResults(
    target: number,
    tol: number,
    regionId: string,
    mortarMode: 'default' | 'custom',
    customMortars: CustomMortar[]
  ): ApiResponse {
    const within = (v: number) => Math.abs(v - target) <= tol
    const randomRock = (i: number) => `RK${(i + 1).toString().padStart(3, '0')}`
    const strengths = [target - tol * 0.6, target + tol * 0.2, target + tol * 0.9].map((v) =>
      Number(v.toFixed(1))
    )
    const mortarLabel =
      mortarMode === 'default'
        ? 'DB_MORTAR'
        : (customMortars[0]?.name || 'CUSTOM_MORTAR').replace(/\s+/g, '_').toUpperCase()
    const randomRatio = () => {
      const val = 0.4 + Math.random() * 0.2 // 0.4 - 0.6
      return Math.round(val * 100) / 100
    }
    const predictions = strengths.map((s, i) => ({
      rock_id: randomRock(i),
      predicted_compressive_strength_mpa: s,
      recommended_rock_ratio: randomRatio(),
      mortar_id: mortarLabel,
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
      mortar_mode: mortarMode,
      custom_mortars: customMortars,
    }
  }

  return (
    <div className="min-h-screen bg-white text-black">
      <div className="mx-auto px-4 py-6 max-w-[1100px]">
        <header className="mb-6 border-b border-gray-500">
          <h1 className="text-3xl font-normal tracking-normal pb-3">
            Concrete Strength Predictor
          </h1>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <InputPanel
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

        <footer className="mt-10 text-center text-gray-700 text-sm border-t border-gray-500 pt-4">
          <p>Mind the Math, LLC | September 2025</p>
          <p className="mt-1 text-xs">For demonstration purposes only | Based on 96 experimental samples</p>
        </footer>
      </div>
    </div>
  )
}

export default App
