/**
 * Theme system types and interfaces
 * 
 * Themes will define complete styling for the application,
 * including colors, typography, spacing, and component styles.
 */

export type ThemeId = 'simple-light' | 'simple-dark' | 'neutral' | 'construction'

export interface Theme {
  id: ThemeId
  name: string
  description: string
  
  // Root-level styles
  root: {
    backgroundColor: string
    color: string
    fontFamily?: string
  }
  
  // Component styles
  components: {
    header: {
      borderColor: string
      titleColor: string
    }
    input: {
      backgroundColor: string
      borderColor: string
      textColor: string
      focusBorderColor: string
      placeholderColor?: string
    }
    button: {
      backgroundColor: string
      borderColor: string
      textColor: string
      hoverBackgroundColor: string
    }
    panel: {
      backgroundColor: string
      borderColor: string
      textColor: string
    }
    footer: {
      borderColor: string
      textColor: string
    }
  }
  
  // Utility classes that will be applied via CSS variables or classes
  classes?: Record<string, string>
}

