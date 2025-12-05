import { Theme } from './types'

/**
 * SimpleDark Theme
 * 
 * Dark mode version of SimpleLight - same clean, minimal aesthetic
 * but with inverted colors for dark mode.
 */
export const simpleDarkTheme: Theme = {
  id: 'simple-dark',
  name: 'SimpleDark',
  description: 'Dark mode version of SimpleLight theme',
  
  root: {
    backgroundColor: '#1f2937', // gray-800
    color: '#f9fafb', // gray-50
    fontFamily: 'Arial, Helvetica, sans-serif',
  },
  
  components: {
    header: {
      borderColor: '#4b5563', // gray-600
      titleColor: '#f9fafb', // gray-50
    },
    input: {
      backgroundColor: '#374151', // gray-700
      borderColor: '#4b5563', // gray-600
      textColor: '#f9fafb', // gray-50
      focusBorderColor: '#3b82f6', // blue-500
    },
    button: {
      backgroundColor: '#4b5563', // gray-600
      borderColor: '#6b7280', // gray-500
      textColor: '#f9fafb', // gray-50
      hoverBackgroundColor: '#6b7280', // gray-500
    },
    panel: {
      backgroundColor: '#374151', // gray-700
      borderColor: '#4b5563', // gray-600
      textColor: '#f9fafb', // gray-50
    },
    footer: {
      borderColor: '#4b5563', // gray-600
      textColor: '#9ca3af', // gray-400
    },
  },
}

