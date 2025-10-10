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
  predicted_compressive_strength_mpa: number
  recommended_rock_ratio: number // 0.3, 0.4, or 0.5
  mortar_id: string
  region_id: string
}

export interface ApiResponse {
  predictions: Prediction[]
  eligible_rocks: number
  total_rocks_in_region: number
  status: 'success' | 'error' | 'partial'
  // Optional metadata for UI display
  target_strength_mpa?: number
  tolerance_mpa?: number
  mocked?: boolean
  mortar_mode?: 'default' | 'custom'
  custom_mortars?: CustomMortar[]
  mortar_properties?: {
    splitting_strength_mpa: number
    shrinkage_inches: number
    flexural_strength_mpa: number
    slump_inches: number
    compressive_strength_mpa: number
    poissons_ratio: number
  }
  rock_properties?: Record<string, {
    compressive_strength_mpa: number
    size_inches: number
    density_lb_ft3: number
    specific_gravity: number
  }>
}

export interface ApiRequest {
  region_id: string
  desired_compressive_strength_mpa: number
  tolerance_mpa: number
  custom_mortars: CustomMortar[]
}
