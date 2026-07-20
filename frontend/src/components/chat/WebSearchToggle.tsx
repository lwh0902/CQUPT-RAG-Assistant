import { Globe2 } from 'lucide-react'

interface WebSearchToggleProps {
  enabled: boolean
  disabled: boolean
  onChange: (enabled: boolean) => void
}

export default function WebSearchToggle({ enabled, disabled, onChange }: WebSearchToggleProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
      aria-label="联网搜索"
      title="联网搜索"
      disabled={disabled}
      onClick={() => onChange(!enabled)}
      className={`inline-flex min-h-9 items-center gap-1.5 rounded-lg border px-2.5 text-xs transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
        enabled
          ? 'border-emerald-500/60 bg-emerald-500/15 text-emerald-700 dark:text-emerald-300'
          : 'border-[var(--border)] bg-[var(--surface)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]'
      }`}
    >
      <Globe2 className="h-3.5 w-3.5" />
      <span>联网搜索</span>
    </button>
  )
}
