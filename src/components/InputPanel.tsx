import React, { useEffect, useState } from 'react'
import { CustomMortar, Region } from '../types'

type MortarMode = 'default' | 'custom'

type CustomMortarForm = {
  key: string
  name: string
  description: string
  properties: Record<keyof CustomMortar['properties'], string>
  meta: Record<keyof CustomMortar['meta'], string>
}

const createEmptyCustomMortar = (): CustomMortarForm => ({
  key: `custom-${Date.now()}-${Math.random().toString(16).slice(2)}`,
  name: '',
  description: '',
  properties: {
    splitting_strength_mpa: '',
    shrinkage_inches: '',
    flexural_strength_mpa: '',
    slump_inches: '',
    compressive_strength_mpa: '',
    poissons_ratio: '',
  },
  meta: {
    gwp: '',
    product_name_long: '',
    manufacturer: '',
    cost_per_pound: '',
  },
})

interface InputPanelProps {
  regions: Region[]
  onSubmit: (data: {
    desiredStrength: number
    regionId: string
    mortarMode: MortarMode
    customMortars: CustomMortar[]
  }) => void
  loading: boolean
}

const InputPanel: React.FC<InputPanelProps> = ({ regions, onSubmit, loading }) => {
  const [desiredStrength, setDesiredStrength] = useState<string>('30')
  const [regionId, setRegionId] = useState<string>(regions[0]?.id || '')
  const [mortarMode, setMortarMode] = useState<MortarMode>('default')
  const [customMortars, setCustomMortars] = useState<CustomMortarForm[]>([])
  const [validationError, setValidationError] = useState<string | null>(null)

  const customMortarCount = customMortars.length

  useEffect(() => {
    if (regions.length > 0 && !regions.some((region) => region.id === regionId)) {
      setRegionId(regions[0].id)
    }
  }, [regions, regionId])

  useEffect(() => {
    if (mortarMode === 'custom' && customMortarCount === 0) {
      setCustomMortars([createEmptyCustomMortar()])
    }
  }, [mortarMode, customMortarCount])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setValidationError(null)

    if (!regionId) {
      setValidationError('Select a region to continue.')
      return
    }

    const parsedDesiredStrength = Number(desiredStrength)
    if (!Number.isFinite(parsedDesiredStrength) || parsedDesiredStrength <= 0) {
      setValidationError('Desired compressive strength must be a valid positive number.')
      return
    }

    let formattedCustomMortars: CustomMortar[] = []

    if (mortarMode === 'custom') {
      if (customMortarCount === 0) {
        setValidationError('Add at least one custom mortar to continue.')
        return
      }

      const requireTextField = (label: string, value: string, mortarIndex: number) => {
        const trimmed = value.trim()
        if (!trimmed) {
          throw new Error(`${label} is required for custom mortar #${mortarIndex + 1}.`)
        }
        return trimmed
      }

      const parseNumericField = (label: string, value: string, mortarIndex: number) => {
        const trimmed = value.trim()
        if (!trimmed) {
          throw new Error(`${label} is required for custom mortar #${mortarIndex + 1}.`)
        }
        const parsed = Number(trimmed)
        if (!Number.isFinite(parsed)) {
          throw new Error(`${label} must be a number for custom mortar #${mortarIndex + 1}.`)
        }
        return parsed
      }

      try {
        formattedCustomMortars = customMortars.map((mortar, index) => ({
          name: requireTextField('Name', mortar.name, index),
          description: requireTextField('Description', mortar.description, index),
          properties: {
            splitting_strength_mpa: parseNumericField(
              'Splitting strength (MPa)',
              mortar.properties.splitting_strength_mpa,
              index
            ),
            shrinkage_inches: parseNumericField(
              'Shrinkage (in)',
              mortar.properties.shrinkage_inches,
              index
            ),
            flexural_strength_mpa: parseNumericField(
              'Flexural strength (MPa)',
              mortar.properties.flexural_strength_mpa,
              index
            ),
            slump_inches: parseNumericField('Slump (in)', mortar.properties.slump_inches, index),
            compressive_strength_mpa: parseNumericField(
              'Compressive strength (MPa)',
              mortar.properties.compressive_strength_mpa,
              index
            ),
            poissons_ratio: parseNumericField(
              'Poisson\'s ratio',
              mortar.properties.poissons_ratio,
              index
            ),
          },
          meta: {
            gwp: parseNumericField('GWP', mortar.meta.gwp, index),
            product_name_long: requireTextField(
              'Product name',
              mortar.meta.product_name_long,
              index
            ),
            manufacturer: requireTextField('Manufacturer', mortar.meta.manufacturer, index),
            cost_per_pound: parseNumericField(
              'Cost per pound',
              mortar.meta.cost_per_pound,
              index
            ),
          },
        }))
      } catch (err) {
        setValidationError(err instanceof Error ? err.message : 'Unable to process custom mortars.')
        return
      }
    }

    onSubmit({
      desiredStrength: parsedDesiredStrength,
      regionId,
      mortarMode,
      customMortars: formattedCustomMortars,
    })
  }

  const handleMortarModeChange = (mode: MortarMode) => {
    setValidationError(null)
    setMortarMode(mode)
    if (mode === 'default') {
      setCustomMortars([])
    }
  }

  const handleCustomMortarFieldChange = (
    index: number,
    field: 'name' | 'description',
    value: string
  ) => {
    setCustomMortars((prev) =>
      prev.map((mortar, i) => (i === index ? { ...mortar, [field]: value } : mortar))
    )
  }

  const handleCustomMortarPropertyChange = (
    index: number,
    property: keyof CustomMortar['properties'],
    value: string
  ) => {
    setCustomMortars((prev) =>
      prev.map((mortar, i) =>
        i === index
          ? { ...mortar, properties: { ...mortar.properties, [property]: value } }
          : mortar
      )
    )
  }

  const handleCustomMortarMetaChange = (
    index: number,
    metaField: keyof CustomMortar['meta'],
    value: string
  ) => {
    setCustomMortars((prev) =>
      prev.map((mortar, i) =>
        i === index ? { ...mortar, meta: { ...mortar.meta, [metaField]: value } } : mortar
      )
    )
  }

  const handleAddCustomMortar = () => {
    setCustomMortars((prev) => [...prev, createEmptyCustomMortar()])
  }

  const handleRemoveCustomMortar = (key: string) => {
    setCustomMortars((prev) => prev.filter((mortar) => mortar.key !== key))
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
            onChange={(e) => setDesiredStrength(e.target.value)}
            min="10"
            max="100"
            step="0.1"
            className="basic-input"
            placeholder="Enter desired strength (MPa)"
            required
            disabled={loading}
          />
          <div className="text-xs text-gray-700">
            Typical range: 20-60 MPa for structural concrete applications
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="block text-sm">Mortar Selection</label>
            <select
              value={mortarMode}
              onChange={(e) => handleMortarModeChange(e.target.value as MortarMode)}
              className="basic-input"
              disabled={loading}
            >
              <option value="default">Default: Existing Database</option>
              <option value="custom">Custom: Input Mortar Properties</option>
            </select>
            <div className="text-xs text-gray-700">
              Default uses the reference mortar database; custom lets you define new formulations.
            </div>
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

        {mortarMode === 'custom' && (
          <div className="space-y-4 border border-gray-500 bg-gray-50 p-4">
            <p className="text-sm text-gray-800">
              Provide properties for each mortar formulation to include in the analysis.
            </p>

            {customMortars.map((mortar, index) => (
              <div key={mortar.key} className="space-y-4 border border-gray-400 bg-white p-4">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
                  <h3 className="text-base font-medium text-gray-800">Custom Mortar {index + 1}</h3>
                  {customMortarCount > 1 && (
                    <button
                      type="button"
                      onClick={() => handleRemoveCustomMortar(mortar.key)}
                      className="text-xs text-red-600 hover:underline"
                      disabled={loading}
                    >
                      Remove
                    </button>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm">Mortar Name</label>
                    <input
                      type="text"
                      value={mortar.name}
                      onChange={(e) => handleCustomMortarFieldChange(index, 'name', e.target.value)}
                      className="basic-input"
                      placeholder="e.g., Custom Blend A"
                      disabled={loading}
                    />
                  </div>
                  <div>
                    <label className="block text-sm">Manufacturer</label>
                    <input
                      type="text"
                      value={mortar.meta.manufacturer}
                      onChange={(e) =>
                        handleCustomMortarMetaChange(index, 'manufacturer', e.target.value)
                      }
                      className="basic-input"
                      placeholder="e.g., ABC Materials Co."
                      disabled={loading}
                    />
                  </div>
                  <div>
                    <label className="block text-sm">Product Name (Long)</label>
                    <input
                      type="text"
                      value={mortar.meta.product_name_long}
                      onChange={(e) =>
                        handleCustomMortarMetaChange(index, 'product_name_long', e.target.value)
                      }
                      className="basic-input"
                      placeholder="Full catalog name"
                      disabled={loading}
                    />
                  </div>
                  <div>
                    <label className="block text-sm">Environmental Intensity (GWP)</label>
                    <input
                      type="number"
                      value={mortar.meta.gwp}
                      onChange={(e) => handleCustomMortarMetaChange(index, 'gwp', e.target.value)}
                      className="basic-input"
                      min="0"
                      step="0.01"
                      placeholder="e.g., 0.75"
                      disabled={loading}
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm">Product Description</label>
                  <textarea
                    value={mortar.description}
                    onChange={(e) => handleCustomMortarFieldChange(index, 'description', e.target.value)}
                    className="basic-input min-h-[80px]"
                    placeholder="Notes on curing time, setting behavior, etc."
                    disabled={loading}
                  />
                </div>

                <div>
                  <h4 className="text-sm font-medium text-gray-800">Mechanical Properties</h4>
                  <div className="mt-2 grid grid-cols-1 md:grid-cols-3 gap-3">
                    <div>
                      <label className="block text-xs text-gray-700">Splitting Strength (MPa)</label>
                      <input
                        type="number"
                        value={mortar.properties.splitting_strength_mpa}
                        onChange={(e) =>
                          handleCustomMortarPropertyChange(index, 'splitting_strength_mpa', e.target.value)
                        }
                        className="basic-input"
                        min="0"
                        step="0.1"
                        disabled={loading}
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-700">Shrinkage (in)</label>
                      <input
                        type="number"
                        value={mortar.properties.shrinkage_inches}
                        onChange={(e) =>
                          handleCustomMortarPropertyChange(index, 'shrinkage_inches', e.target.value)
                        }
                        className="basic-input"
                        min="0"
                        step="0.001"
                        disabled={loading}
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-700">Flexural Strength (MPa)</label>
                      <input
                        type="number"
                        value={mortar.properties.flexural_strength_mpa}
                        onChange={(e) =>
                          handleCustomMortarPropertyChange(index, 'flexural_strength_mpa', e.target.value)
                        }
                        className="basic-input"
                        min="0"
                        step="0.1"
                        disabled={loading}
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-700">Slump (in)</label>
                      <input
                        type="number"
                        value={mortar.properties.slump_inches}
                        onChange={(e) =>
                          handleCustomMortarPropertyChange(index, 'slump_inches', e.target.value)
                        }
                        className="basic-input"
                        min="0"
                        step="0.1"
                        disabled={loading}
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-700">Compressive Strength (MPa)</label>
                      <input
                        type="number"
                        value={mortar.properties.compressive_strength_mpa}
                        onChange={(e) =>
                          handleCustomMortarPropertyChange(index, 'compressive_strength_mpa', e.target.value)
                        }
                        className="basic-input"
                        min="0"
                        step="0.1"
                        disabled={loading}
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-700">Poisson&apos;s Ratio</label>
                      <input
                        type="number"
                        value={mortar.properties.poissons_ratio}
                        onChange={(e) =>
                          handleCustomMortarPropertyChange(index, 'poissons_ratio', e.target.value)
                        }
                        className="basic-input"
                        min="0"
                        max="0.5"
                        step="0.01"
                        disabled={loading}
                      />
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm">Cost per Pound (USD)</label>
                    <input
                      type="number"
                      value={mortar.meta.cost_per_pound}
                      onChange={(e) =>
                        handleCustomMortarMetaChange(index, 'cost_per_pound', e.target.value)
                      }
                      className="basic-input"
                      min="0"
                      step="0.01"
                      disabled={loading}
                    />
                  </div>
                </div>
              </div>
            ))}

            <button
              type="button"
              onClick={handleAddCustomMortar}
              className="basic-button w-full md:w-auto"
              disabled={loading}
            >
              Add Another Mortar
            </button>
          </div>
        )}

        {validationError && (
          <div className="border border-red-500 bg-red-50 text-sm text-red-700 px-3 py-2">
            {validationError}
          </div>
        )}

        <div className="space-y-2 mt-4">
          <button type="submit" disabled={loading} className="basic-button w-full">
            {loading ? (
              <span className="inline-flex items-center justify-center">
                <svg
                  className="animate-spin h-4 w-4 mr-2"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Running Inverse ML Model...
              </span>
            ) : (
              'Analyze Optimal Mix Design'
            )}
          </button>
          <div className="text-center text-xs text-gray-700 pt-2 border-t border-gray-400">
            <p>Model will evaluate rocks available in the region against strength requirements</p>
          </div>
        </div>
      </form>
    </section>
  )
}

export default InputPanel
