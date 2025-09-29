export interface Mortar {
  id: string
  name: string
}

export interface Region {
  id: string
  name: string
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
