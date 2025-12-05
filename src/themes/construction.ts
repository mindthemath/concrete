import { Theme } from './types'

/**
 * Construction Theme
 *
 * Industrial design inspired by construction tools, heavy machinery,
 * and safety signage. High contrast, rugged, and durable feel.
 */
export const constructionTheme: Theme = {
  id: 'construction',
  name: 'Construction',
  description: 'Industrial heavy-duty design',

  root: {
    backgroundColor: '#2d2d2d', // Dark industrial grey
    color: '#ffffff',
    fontFamily: '"Chakra Petch", "Rajdhani", "Roboto Mono", monospace', // Industrial technical font
  },

  components: {
    header: {
      backgroundColor: '#fbbf24', // Amber/Yellow
      borderColor: '#000000',
      titleColor: '#000000',
      borderWidth: '0 0 4px 0',
      borderStyle: 'solid',
    },
    input: {
      backgroundColor: '#ffffff',
      borderColor: '#4b5563',
      textColor: '#000000',
      focusBorderColor: '#f59e0b', // Safety orange
      borderStyle: 'solid',
      borderWidth: '2px',
      shadow: 'inset 2px 2px 4px rgba(0,0,0,0.2)', // Depressed look
    },
    button: {
      backgroundColor: 'linear-gradient(180deg, #fcd34d 0%, #f59e0b 100%)', // Safety yellow/orange gradient
      borderColor: '#78350f',
      textColor: '#000000',
      hoverBackgroundColor: 'linear-gradient(180deg, #fbbf24 0%, #d97706 100%)',
      borderStyle: 'solid',
      borderWidth: '2px 2px 4px 2px', // 3D effect: thicker bottom border
      shadow: '0 2px 0 #78350f',
      activeShadow: 'inset 0 2px 4px rgba(0,0,0,0.3)',
      transform: 'uppercase',
      fontWeight: '800', // Extra bold
    },
    panel: {
      backgroundColor: '#d4d4d4', // Light concrete grey
      borderColor: '#404040',
      textColor: '#171717',
      borderStyle: 'solid',
      borderWidth: '3px',
      shadow: '4px 4px 0px #171717', // Hard shadow
    },
    footer: {
      borderColor: '#fbbf24',
      textColor: '#fbbf24',
    },
  },
}
