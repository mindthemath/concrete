/**
 * Theme system exports
 * 
 * This module exports all available themes and provides
 * utilities for theme management.
 */

import { Theme, ThemeId } from './types'
import { simpleLightTheme } from './simpleLight'
import { simpleDarkTheme } from './simpleDark'
import { constructionTheme } from './construction'

export const themes: Record<ThemeId, Theme> = {
  'simple-light': simpleLightTheme,
  'simple-dark': simpleDarkTheme,
  'construction': constructionTheme,
}

export const defaultThemeId: ThemeId = 'simple-light'

export function getTheme(themeId: ThemeId): Theme {
  return themes[themeId] || themes[defaultThemeId]
}

export function getAllThemes(): Theme[] {
  return Object.values(themes)
}

// Re-export types
export type { Theme, ThemeId } from './types'

