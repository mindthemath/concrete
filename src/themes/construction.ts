import { Theme } from './types'

/**
 * Construction Theme
 * 
 * Industrial construction-themed design with early 00's vibes.
 * Features high contrast, safety yellow/orange, and skeuomorphic elements.
 */
export const constructionTheme: Theme = {
  id: 'construction',
  name: 'Construction',
  description: 'Industrial construction-themed design (Early 00s Style)',
  
  root: {
    backgroundColor: '#a8a8a0', // Darker base for cinder block
    // Background image handled via CSS class .theme-bg-construction
    color: '#1a1a1a',
    fontFamily: '"Verdana", "Geneva", sans-serif',
  },
  
  components: {
    header: {
      borderColor: '#000000',
      titleColor: '#000000',
      borderWidth: '0 0 3px 0',
      borderStyle: 'solid',
    },
    input: {
      backgroundColor: '#ffffff',
      borderColor: '#8c8c8c',
      textColor: '#000000',
      focusBorderColor: '#ff9900', // Safety orange
      borderStyle: 'inset',
      borderWidth: '2px',
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
      backgroundColor: '#f0f0e8',
      borderColor: '#aaaaaa',
      textColor: '#000000',
      borderStyle: 'outset',
      borderWidth: '2px',
      shadow: '2px 2px 4px rgba(0,0,0,0.2)',
    },
    footer: {
      borderColor: '#555555',
      textColor: '#444444',
    },
  },
}
