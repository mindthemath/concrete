/**
 * Theme system exports
 *
 * This module exports all available themes and provides
 * utilities for theme management.
 */

import { Theme, ThemeId } from './types'
import { simpleLightTheme } from './simpleLight'
import { simpleDarkTheme } from './simpleDark'
import { neutralTheme } from './neutral'
import { notebookTheme } from './notebook'
import { constructionTheme } from './construction'

export const themes: Record<ThemeId, Theme> = {
  'simple-light': simpleLightTheme,
  'simple-dark': simpleDarkTheme,
  neutral: neutralTheme,
  notebook: notebookTheme,
  construction: constructionTheme,
}

export const defaultThemeId: ThemeId = 'neutral'

export function getTheme(themeId: ThemeId): Theme {
  return themes[themeId] || themes[defaultThemeId]
}

export function getAllThemes(): Theme[] {
  return Object.values(themes)
}

// Re-export types and context
export type { Theme, ThemeId } from './types'
export { useTheme, ThemeProvider } from './ThemeContext'
