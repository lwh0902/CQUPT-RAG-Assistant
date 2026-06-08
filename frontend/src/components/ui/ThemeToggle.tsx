import { Sun, Moon } from 'lucide-react'
import { useThemeStore } from '../../store/theme'

export default function ThemeToggle() {
  const { theme, toggleTheme } = useThemeStore()

  return (
    <button
      onClick={toggleTheme}
      className="flex h-8 w-8 items-center justify-center rounded-lg transition-colors hover:bg-[var(--bg-tertiary)]"
      title={theme === 'light' ? '切换深色模式' : '切换浅色模式'}
    >
      {theme === 'light' ? (
        <Moon className="h-4 w-4 text-[var(--text-secondary)]" />
      ) : (
        <Sun className="h-4 w-4 text-[var(--text-secondary)]" />
      )}
    </button>
  )
}
