import { useState } from 'react'
import InputPanel from './components/InputPanel'
import OutputPanel from './components/OutputPanel'
import { Region, ApiResponse, CustomMortar } from './types'
import regionData from '../data/region.json'
import concreteDb from '../data/concrete.json'
import { useTheme } from './themes/ThemeContext'
import type { ThemeId } from './themes/types'

interface RegionDataEntry {
  name: string
  description?: string
  available_rocks?: string[]
  primary_rock?: string
  climate?: string
}

interface ConcreteDbEntry {
  mortar_id: string
  rock_id: string
  rock_ratio: number
  concrete_compressive_strength_mpa: number
}

type RegionData = Record<string, RegionDataEntry>
type ConcreteDb = Record<string, ConcreteDbEntry>

const SAMPLE_REGIONS: Region[] = Object.keys(regionData).map((key) => ({
  id: key,
  name: (regionData as RegionData)[key].name,
}))

function App() {
  const { themeId, setTheme } = useTheme()
  const [result, setResult] = useState<ApiResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [mortarMode, setMortarMode] = useState<'default' | 'custom'>('default')
  const [customMortars, setCustomMortars] = useState<CustomMortar[]>([])

  const handleSubmit = async (formData: {
    desiredStrength: number
    regionId: string
    mortarMode: 'default' | 'custom'
    customMortars: CustomMortar[]
  }) => {
    setLoading(true)
    setError(null)
    setResult(null)
    setMortarMode(formData.mortarMode)
    setCustomMortars(formData.customMortars)

    try {
      // Replace with actual API endpoint
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          desired_compressive_strength_mpa: formData.desiredStrength,
          region_id: formData.regionId,
          mortar_mode: formData.mortarMode,
          custom_mortars: formData.customMortars,
        }),
      })

      if (!response.ok) {
        // If API not available, produce a clearly-marked mock result
        const mocked: ApiResponse = mockResults(
          formData.desiredStrength,
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
        mocked: false,
      })
    } catch {
      // Network/404: show a mock so we can see the flow
      const mocked: ApiResponse = mockResults(
        formData.desiredStrength,
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
    regionId: string,
    mortarMode: 'default' | 'custom',
    customMortars: CustomMortar[]
  ): ApiResponse {
    // Parameters kept for API compatibility, not used in mock filtering
    void mortarMode
    void customMortars
    const availableRocks: string[] = (regionData as RegionData)[regionId]?.available_rocks || []
    const entries = Object.values(concreteDb as ConcreteDb) as ConcreteDbEntry[]

    const filtered = entries
      .filter(
        (e) =>
          e.concrete_compressive_strength_mpa >= target &&
          (availableRocks.length === 0 || availableRocks.includes(e.rock_id))
      )
      .sort((a, b) => a.concrete_compressive_strength_mpa - b.concrete_compressive_strength_mpa)

    const predictions = filtered.map((e) => ({
      rock_id: e.rock_id,
      predicted_compressive_strength_mpa: Number(e.concrete_compressive_strength_mpa.toFixed(1)),
      rock_ratio: e.rock_ratio,
      mortar_id: e.mortar_id,
    }))

    return {
      predictions,
      status: 'success',
      mocked: true,
    }
  }

  return (
    <div className="min-h-screen">
      <div className="mx-auto px-4 py-6 max-w-[1100px]" style={{ color: 'var(--theme-text)' }}>
        <header className="mb-6 pb-3" style={{ borderBottom: '1px solid var(--theme-header-border)' }}>
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-normal tracking-normal" style={{ color: 'var(--theme-header-title)' }}>
              Concrete Strength Predictor
            </h1>
            <div className="relative">
              <select
                className="theme-select appearance-none pr-8 pl-3 py-1.5 text-sm cursor-pointer"
                value={themeId}
                onChange={(e) => {
                  const newTheme = e.target.value as ThemeId
                  if (newTheme === 'simple-light' || newTheme === 'simple-dark' || newTheme === 'neutral' || newTheme === 'notebook') {
                    setTheme(newTheme)
                  }
                }}
              >
                <option value="simple-light">SimpleLight</option>
                <option value="simple-dark">SimpleDark</option>
                <option value="neutral">Neutral</option>
                <option value="notebook">Notebook</option>
              </select>
              <div className="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color: 'var(--theme-input-text)' }}>
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </div>
            </div>
          </div>
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
            mortarMode={mortarMode}
            customMortars={customMortars}
          />
        </div>

        <footer className="mt-10 text-center text-sm pt-4" style={{ borderTop: '1px solid var(--theme-footer-border)', color: 'var(--theme-footer-text)' }}>
          <p>Mind the Math, LLC | November 2025</p>
          <p className="mt-1 text-xs">For demonstration purposes only | Based on 96 experimental samples</p>
        </footer>
      </div>
    </div>
  )
}

export default App
