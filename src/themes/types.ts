/**
 * Theme system types and interfaces
 * 
 * Themes will define complete styling for the application,
 * including colors, typography, spacing, and component styles.
 */

export type ThemeId = 'simple-light' | 'simple-dark' | 'neutral' | 'notebook'

export interface Theme {
  id: ThemeId
  name: string
  description: string
  
  // Root-level styles
  root: {
    backgroundColor: string
    backgroundImage?: string
    color: string
    fontFamily?: string
  }
  
  // Component styles
  components: {
    header: {
      borderColor: string
      titleColor: string
      borderWidth?: string
      borderStyle?: string
      backgroundColor?: string
    }
    input: {
      backgroundColor: string
      borderColor: string
      textColor: string
      focusBorderColor: string
      placeholderColor?: string
      borderWidth?: string
      borderStyle?: string
      shadow?: string
    }
    button: {
      backgroundColor: string // Can be a color or gradient
      borderColor: string
      textColor: string
      hoverBackgroundColor: string
      borderWidth?: string
      borderStyle?: string
      shadow?: string
      activeShadow?: string
      transform?: string // uppercase, etc
      fontWeight?: string
    }
    panel: {
      backgroundColor: string
      borderColor: string
      textColor: string
      borderWidth?: string
      borderStyle?: string
      shadow?: string
    }
    footer: {
      borderColor: string
      textColor: string
    }
  }
  
  // Utility classes that will be applied via CSS variables or classes
  classes?: Record<string, string>
}
