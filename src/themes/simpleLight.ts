import { Theme } from './types'

/**
 * SimpleLight Theme
 *
 * The current default theme - clean, minimal, light mode styling.
 * This represents the existing "early web" aesthetic.
 */
export const simpleLightTheme: Theme = {
  id: 'simple-light',
  name: 'SimpleLight',
  description: 'Clean, minimal light theme with early web aesthetic',

  root: {
    backgroundColor: '#ffffff',
    color: '#000000',
    fontFamily: 'Arial, Helvetica, sans-serif',
  },

  components: {
    header: {
      borderColor: '#6b7280', // gray-500
      titleColor: '#000000',
    },
    input: {
      backgroundColor: '#ffffff',
      borderColor: '#9ca3af', // gray-400
      textColor: '#000000',
      focusBorderColor: '#2563eb', // blue-600
    },
    button: {
      backgroundColor: '#e5e7eb', // gray-200
      borderColor: '#6b7280', // gray-500
      textColor: '#000000',
      hoverBackgroundColor: '#d1d5db', // gray-300
    },
    panel: {
      backgroundColor: '#ffffff',
      borderColor: '#6b7280', // gray-500
      textColor: '#000000',
    },
    footer: {
      borderColor: '#6b7280', // gray-500
      textColor: '#4b5563', // gray-700
    },
  },
}
