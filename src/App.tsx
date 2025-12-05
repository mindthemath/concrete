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

const SAMPLE_REGIONS: Region[] = Object.keys(regionData).map(key => ({
  id: key,
  name: (regionData as RegionData)[key].name,
}))

// Environment variables with Vite prefix
// Note: API_KEY is not used in production - backend is protected via CORS restrictions
const API_ENDPOINT = import.meta.env.VITE_API_ENDPOINT || 'http://localhost:9600/predict'
const API_TIMEOUT = Number(import.meta.env.VITE_API_TIMEOUT) || 5000 // Default 5 seconds

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
      // Build headers
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      }
      
      // Note: API keys are not secure in static apps, so we rely on CORS restrictions
      // on the backend instead. For local development, you can add auth if needed.

      // Create AbortController for timeout
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT)

      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          desired_compressive_strength_mpa: formData.desiredStrength,
          region_id: formData.regionId,
          custom_mortars: formData.customMortars,
        }),
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(
          `API request failed: ${response.status} ${response.statusText} - ${errorText}`
        )
      }

      const data: ApiResponse = await response.json()
      setResult({
        ...data,
        mocked: false,
      })
    } catch (err) {
      // For timeout errors, just show mock results without error message
      // For other errors, show error but still provide mock results as fallback
      if (err instanceof Error && err.name === 'AbortError') {
        // Timeout: silently fallback to mock results
        const mocked: ApiResponse = mockResults(
          formData.desiredStrength,
          formData.regionId,
          formData.mortarMode,
          formData.customMortars
        )
        setResult(mocked)
        setError(null) // Clear any previous errors
      } else {
        // Other errors: show error message but still provide mock results
        const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred'
        setError(errorMessage)

        // Fallback to mock results for development
      const mocked: ApiResponse = mockResults(
        formData.desiredStrength,
        formData.regionId,
        formData.mortarMode,
        formData.customMortars
      )
      setResult(mocked)
      }
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
        e =>
          e.concrete_compressive_strength_mpa >= target &&
          (availableRocks.length === 0 || availableRocks.includes(e.rock_id))
      )
      .sort((a, b) => a.concrete_compressive_strength_mpa - b.concrete_compressive_strength_mpa)

    const predictions = filtered.map(e => ({
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
        <header
          className="mb-6 pb-3"
          style={{ borderBottom: '1px solid var(--theme-header-border)' }}
        >
          <div className="flex items-center justify-between">
            <h1 
              className="text-2xl font-normal tracking-normal relative inline-block px-4 py-2" 
              style={{ 
                color: 'var(--theme-header-title)',
                backgroundColor:
                  themeId === 'notebook'
                    ? '#000000'
                    : themeId === 'construction'
                      ? 'var(--theme-header-bg)'
                      : 'transparent',
                border:
                  themeId === 'notebook'
                    ? '1px solid #ffffff'
                    : themeId === 'construction'
                      ? '2px solid #000000'
                      : 'none',
                textTransform:
                  themeId === 'notebook' || themeId === 'construction' ? 'uppercase' : 'none',
                width:
                  themeId === 'notebook' || themeId === 'construction' ? 'fit-content' : 'auto',
              }}
            >
              Concrete Strength Predictor
            </h1>
            <div className="relative">
              <select
                className="theme-select appearance-none pr-8 pl-3 py-1.5 text-sm cursor-pointer"
                value={themeId}
                onChange={e => {
                  const newTheme = e.target.value as ThemeId
                  if (
                    ['simple-light', 'simple-dark', 'neutral', 'notebook', 'construction'].includes(
                      newTheme
                    )
                  ) {
                    setTheme(newTheme)
                  }
                }}
              >
                <option value="simple-light">light</option>
                <option value="simple-dark">dark</option>
                <option value="neutral">neutral</option>
                <option value="notebook">notebook</option>
                <option value="construction">construction</option>
              </select>
              <div
                className="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none"
                style={{ color: 'var(--theme-input-text)' }}
              >
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
          <InputPanel regions={SAMPLE_REGIONS} onSubmit={handleSubmit} loading={loading} />

          <OutputPanel
            result={result}
            loading={loading}
            error={error}
            mortarMode={mortarMode}
            customMortars={customMortars}
          />
        </div>

        <footer
          className="mt-10 text-center text-sm pt-4"
          style={{
            borderTop: '1px solid var(--theme-footer-border)',
            color: 'var(--theme-footer-text)',
          }}
        >
          <div 
            className="inline-block px-4 py-2"
            style={{
              backgroundColor:
                themeId === 'notebook'
                  ? '#000000'
                  : themeId === 'construction'
                    ? '#000000'
                    : 'transparent',
              border:
                themeId === 'notebook'
                  ? '1px solid #ffffff'
                  : themeId === 'construction'
                    ? '2px solid #fbbf24'
                    : 'none',
            }}
          >
            <p>Mind the Math, LLC | December 2025</p>
            <p className="mt-1 text-xs">
              For demonstration purposes only | Based on 96 experimental samples
            </p>
          </div>
        </footer>
      </div>
    </div>
  )
}

export default App
