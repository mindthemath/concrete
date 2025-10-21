export interface Region {
  id: string
  name: string
}

export interface Mortar {
  id: string
  name: string
}

export interface CustomMortar {
  name: string
  description: string
  properties: {
    splitting_strength_mpa: number
    shrinkage_inches: number
    flexural_strength_mpa: number
    slump_inches: number
    compressive_strength_mpa: number
    poissons_ratio: number
  }
  meta: {
    gwp: number
    product_name_long: string
    manufacturer: string
    cost_per_pound: number
  }
}

export interface Prediction {
  rock_id: string
  mortar_id: string
  rock_ratio: number // range: 0.3 - 0.5
  predicted_compressive_strength_mpa: number
}

export interface ApiResponse {
  predictions: Prediction[]
  status: 'success' | 'error' | 'partial'
  mocked?: boolean
}

export interface ApiRequest {
  region_id: string
  desired_compressive_strength_mpa: number
  tolerance_mpa: number
  custom_mortars: CustomMortar[]
}
