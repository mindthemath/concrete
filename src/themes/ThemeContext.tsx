import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { ThemeId, Theme, getTheme, defaultThemeId } from './index'

interface ThemeContextType {
  theme: Theme
  themeId: ThemeId
  setTheme: (themeId: ThemeId) => void
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}

interface ThemeProviderProps {
  children: ReactNode
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const [themeId, setThemeIdState] = useState<ThemeId>(() => {
    // Load from localStorage or use default
    const saved = localStorage.getItem('theme')
    if (saved && (saved === 'simple-light' || saved === 'simple-dark' || saved === 'neutral' || saved === 'construction')) {
      return saved as ThemeId
    }
    return defaultThemeId
  })

  const theme = getTheme(themeId)

  const setTheme = (newThemeId: ThemeId) => {
    setThemeIdState(newThemeId)
    localStorage.setItem('theme', newThemeId)
  }

  // Apply theme styles to document root
  useEffect(() => {
    const root = document.documentElement
    const { root: rootStyles, components } = theme

    // Apply root styles
    root.style.setProperty('--theme-bg', rootStyles.backgroundColor)
    root.style.setProperty('--theme-text', rootStyles.color)
    root.style.setProperty('--theme-font', rootStyles.fontFamily || 'Arial, Helvetica, sans-serif')

    // Apply component styles
    root.style.setProperty('--theme-header-border', components.header.borderColor)
    root.style.setProperty('--theme-header-title', components.header.titleColor)
    
    root.style.setProperty('--theme-input-bg', components.input.backgroundColor)
    root.style.setProperty('--theme-input-border', components.input.borderColor)
    root.style.setProperty('--theme-input-text', components.input.textColor)
    root.style.setProperty('--theme-input-focus', components.input.focusBorderColor)
    
    root.style.setProperty('--theme-button-bg', components.button.backgroundColor)
    root.style.setProperty('--theme-button-border', components.button.borderColor)
    root.style.setProperty('--theme-button-text', components.button.textColor)
    root.style.setProperty('--theme-button-hover', components.button.hoverBackgroundColor)
    
    root.style.setProperty('--theme-panel-bg', components.panel.backgroundColor)
    root.style.setProperty('--theme-panel-border', components.panel.borderColor)
    root.style.setProperty('--theme-panel-text', components.panel.textColor)
    
    root.style.setProperty('--theme-footer-border', components.footer.borderColor)
    root.style.setProperty('--theme-footer-text', components.footer.textColor)

    // Apply to body
    document.body.style.backgroundColor = rootStyles.backgroundColor
    document.body.style.color = rootStyles.color
    if (rootStyles.fontFamily) {
      document.body.style.fontFamily = rootStyles.fontFamily
    }
  }, [theme])

  return (
    <ThemeContext.Provider value={{ theme, themeId, setTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

