import { Theme } from './types'

/**
 * Neutral Theme
 * 
 * Neutral gray theme - same clean, minimal aesthetic
 * with a medium gray background for a neutral appearance.
 */
export const neutralTheme: Theme = {
  id: 'neutral',
  name: 'Neutral',
  description: 'Neutral gray theme with minimal aesthetic',
  
  root: {
    backgroundColor: '#e5e7eb', // gray-200
    color: '#000000', // black
    fontFamily: 'Arial, Helvetica, sans-serif',
  },
  
  components: {
    header: {
      borderColor: '#6b7280', // gray-500 (same as other themes)
      titleColor: '#000000', // black
    },
    input: {
      backgroundColor: '#f3f4f6', // gray-100
      borderColor: '#6b7280', // gray-500 (same as other themes)
      textColor: '#000000', // black
      focusBorderColor: '#2563eb', // blue-600 (same as other themes)
    },
    button: {
      backgroundColor: '#d1d5db', // gray-300
      borderColor: '#6b7280', // gray-500 (same as other themes)
      textColor: '#000000', // black
      hoverBackgroundColor: '#9ca3af', // gray-400
    },
    panel: {
      backgroundColor: '#f3f4f6', // gray-100
      borderColor: '#6b7280', // gray-500 (same as other themes)
      textColor: '#000000', // black
    },
    footer: {
      borderColor: '#6b7280', // gray-500 (same as other themes)
      textColor: '#4b5563', // gray-700 (same as light mode)
    },
  },
}

