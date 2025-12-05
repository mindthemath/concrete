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
    if (saved && (saved === 'simple-light' || saved === 'simple-dark' || saved === 'neutral' || saved === 'notebook')) {
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
    // root.style.setProperty('--theme-bg-image', rootStyles.backgroundImage || 'none') // Removed: handled via class
    root.style.setProperty('--theme-text', rootStyles.color)
    root.style.setProperty('--theme-font', rootStyles.fontFamily || 'Arial, Helvetica, sans-serif')

    // Apply theme-specific classes to body
    document.body.classList.remove('theme-bg-notebook')
    if (theme.id === 'notebook') {
      document.body.classList.add('theme-bg-notebook')
    }

    // Apply component styles
    root.style.setProperty('--theme-header-border', components.header.borderColor)
    root.style.setProperty('--theme-header-title', components.header.titleColor)
    root.style.setProperty('--theme-header-bg', components.header.backgroundColor || 'transparent')
    root.style.setProperty('--theme-header-border-width', components.header.borderWidth || '1px')
    root.style.setProperty('--theme-header-border-style', components.header.borderStyle || 'solid')
    
    root.style.setProperty('--theme-input-bg', components.input.backgroundColor)
    root.style.setProperty('--theme-input-border', components.input.borderColor)
    root.style.setProperty('--theme-input-text', components.input.textColor)
    root.style.setProperty('--theme-input-focus', components.input.focusBorderColor)
    root.style.setProperty('--theme-input-border-width', components.input.borderWidth || '1px')
    root.style.setProperty('--theme-input-border-style', components.input.borderStyle || 'solid')
    root.style.setProperty('--theme-input-shadow', components.input.shadow || 'none')
    
    root.style.setProperty('--theme-button-bg', components.button.backgroundColor)
    root.style.setProperty('--theme-button-border', components.button.borderColor)
    root.style.setProperty('--theme-button-text', components.button.textColor)
    root.style.setProperty('--theme-button-hover', components.button.hoverBackgroundColor)
    root.style.setProperty('--theme-button-border-width', components.button.borderWidth || '1px')
    root.style.setProperty('--theme-button-border-style', components.button.borderStyle || 'solid')
    root.style.setProperty('--theme-button-shadow', components.button.shadow || 'none')
    root.style.setProperty('--theme-button-active-shadow', components.button.activeShadow || 'none')
    root.style.setProperty('--theme-button-transform', components.button.transform || 'none')
    root.style.setProperty('--theme-button-weight', components.button.fontWeight || 'normal')
    
    root.style.setProperty('--theme-panel-bg', components.panel.backgroundColor)
    root.style.setProperty('--theme-panel-border', components.panel.borderColor)
    root.style.setProperty('--theme-panel-text', components.panel.textColor)
    root.style.setProperty('--theme-panel-border-width', components.panel.borderWidth || '1px')
    root.style.setProperty('--theme-panel-border-style', components.panel.borderStyle || 'solid')
    root.style.setProperty('--theme-panel-shadow', components.panel.shadow || 'none')
    
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

