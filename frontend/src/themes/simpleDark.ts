import { Theme } from './types'

/**
 * SimpleDark Theme
 *
 * Dark mode version of SimpleLight - same clean, minimal aesthetic
 * with true black background and white text, maintaining simplicity.
 */
export const simpleDarkTheme: Theme = {
  id: 'simple-dark',
  name: 'SimpleDark',
  description: 'Dark mode version of SimpleLight theme',

  root: {
    backgroundColor: '#000000', // true black
    color: '#ffffff', // white
    fontFamily: 'Arial, Helvetica, sans-serif',
  },

  components: {
    header: {
      borderColor: '#6b7280', // gray-500 (same as light mode)
      titleColor: '#ffffff', // white
    },
    input: {
      backgroundColor: '#000000', // black
      borderColor: '#6b7280', // gray-500 (same as light mode)
      textColor: '#ffffff', // white
      focusBorderColor: '#2563eb', // blue-600 (same as light mode)
    },
    button: {
      backgroundColor: '#1f1f1f', // very dark gray (slightly lighter than black)
      borderColor: '#6b7280', // gray-500 (same as light mode)
      textColor: '#ffffff', // white
      hoverBackgroundColor: '#2f2f2f', // slightly lighter dark gray
    },
    panel: {
      backgroundColor: '#000000', // black
      borderColor: '#6b7280', // gray-500 (same as light mode)
      textColor: '#ffffff', // white
    },
    footer: {
      borderColor: '#6b7280', // gray-500 (same as light mode)
      textColor: '#9ca3af', // gray-400 (same as light mode for subtle text)
    },
  },
}
