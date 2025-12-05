import { Theme } from './types'

/**
 * Construction Theme
 * 
 * Placeholder for a construction/industrial-themed design.
 * This will have a different visual style while maintaining
 * the same element structure and layout.
 */
export const constructionTheme: Theme = {
  id: 'construction',
  name: 'Construction',
  description: 'Industrial construction-themed design',
  
  root: {
    backgroundColor: '#fef3c7', // yellow-100 (placeholder)
    color: '#78350f', // yellow-900 (placeholder)
    fontFamily: 'Arial, Helvetica, sans-serif',
  },
  
  components: {
    header: {
      borderColor: '#d97706', // amber-600 (placeholder)
      titleColor: '#78350f', // yellow-900 (placeholder)
    },
    input: {
      backgroundColor: '#ffffff',
      borderColor: '#d97706', // amber-600 (placeholder)
      textColor: '#78350f', // yellow-900 (placeholder)
      focusBorderColor: '#b45309', // amber-700 (placeholder)
    },
    button: {
      backgroundColor: '#fbbf24', // amber-400 (placeholder)
      borderColor: '#d97706', // amber-600 (placeholder)
      textColor: '#78350f', // yellow-900 (placeholder)
      hoverBackgroundColor: '#f59e0b', // amber-500 (placeholder)
    },
    panel: {
      backgroundColor: '#fffbeb', // amber-50 (placeholder)
      borderColor: '#d97706', // amber-600 (placeholder)
      textColor: '#78350f', // yellow-900 (placeholder)
    },
    footer: {
      borderColor: '#d97706', // amber-600 (placeholder)
      textColor: '#92400e', // yellow-800 (placeholder)
    },
  },
}

