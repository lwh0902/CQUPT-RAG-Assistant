import { Languages } from 'lucide-react'
import { useI18nStore, useT } from '../../i18n'

/** Top-bar language toggle: zh-CN ⇄ en-US, persisted via the i18n store. */
export default function LanguageToggle() {
  const lang = useI18nStore((s) => s.lang)
  const setLang = useI18nStore((s) => s.setLang)
  const t = useT()

  const next = lang === 'zh-CN' ? 'en-US' : 'zh-CN'
  return (
    <button
      type="button"
      onClick={() => setLang(next)}
      className="flex h-9 items-center gap-1.5 rounded-lg px-2 text-xs font-medium text-[var(--text-tertiary)] transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
      aria-label={t('lang.switch')}
      title={t('lang.switch')}
    >
      <Languages className="h-4 w-4" />
      {lang === 'zh-CN' ? 'EN' : '中'}
    </button>
  )
}
