import { Theme } from './types'

/**
 * Notebook Theme
 * 
 * Classic composition notebook design with black speckled background,
 * ruled paper panels, and handwritten font.
 */
export const notebookTheme: Theme = {
  id: 'notebook',
  name: 'Notebook',
  description: 'Classic composition notebook design',
  
  root: {
    backgroundColor: '#1a1a1a', // Black background like composition notebook
    // Background image handled via CSS class .theme-bg-notebook
    color: '#ffffff',
    fontFamily: ' "Comic Sans MS", "Kalam", cursive', // Clean handwritten feel
  },
  
  components: {
    header: {
      borderColor: '#ffffff',
      titleColor: '#ffffff',
      borderWidth: '0 0 2px 0',
      borderStyle: 'solid',
    },
    input: {
      backgroundColor: '#ffffff',
      borderColor: '#8c8c8c',
      textColor: '#000000',
      focusBorderColor: '#ff9900', // Safety orange
      borderStyle: 'solid',
      borderWidth: '1px',
    },
    button: {
      backgroundColor: 'linear-gradient(to bottom, #ffd700 0%, #e6c200 100%)', // Safety yellow gradient
      borderColor: '#998100',
      textColor: '#000000',
      hoverBackgroundColor: 'linear-gradient(to bottom, #ffeb3b 0%, #ffd700 100%)',
      borderStyle: 'outset',
      borderWidth: '2px',
      shadow: '1px 1px 0px #000000',
      activeShadow: 'inset 1px 1px 2px rgba(0,0,0,0.4)',
      transform: 'uppercase',
      fontWeight: 'bold',
    },
    panel: {
      backgroundColor: '#ffffff',
      borderColor: '#e0e0e0',
      textColor: '#000000',
      borderStyle: 'solid',
      borderWidth: '1px',
      shadow: '2px 2px 4px rgba(0,0,0,0.15)',
    },
    footer: {
      borderColor: '#ffffff',
      textColor: '#ffffff',
    },
  },
}

