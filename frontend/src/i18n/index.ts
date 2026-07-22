import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'
import { zh, type I18nKey } from './zh'
import { en } from './en'

export type Language = 'zh-CN' | 'en-US'

const dicts: Record<Language, Record<I18nKey, string>> = {
  'zh-CN': zh,
  'en-US': en,
}

/** localStorage may be unavailable/broken (SSR, tests, privacy modes) — no-op then. */
const safeStorage = createJSONStorage(() => {
  try {
    const ls = globalThis.localStorage
    if (ls && typeof ls.setItem === 'function') {
      return ls
    }
  } catch {
    // fall through to in-memory no-op storage
  }
  return {
    getItem: () => null,
    setItem: () => undefined,
    removeItem: () => undefined,
  } as unknown as Storage
})

interface I18nState {
  lang: Language
  setLang: (lang: Language) => void
}

export const useI18nStore = create<I18nState>()(
  persist(
    (set) => ({
      lang: 'zh-CN',
      setLang: (lang) => set({ lang }),
    }),
    { name: 'cqupt-rag-lang', storage: safeStorage },
  ),
)

/** Translate a dictionary key using the current language (falls back to zh). */
export function translate(key: I18nKey, lang: Language = useI18nStore.getState().lang): string {
  return dicts[lang][key] ?? zh[key]
}

export function useT() {
  const lang = useI18nStore((s) => s.lang)
  return (key: I18nKey) => translate(key, lang)
}

export function getLang(): Language {
  return useI18nStore.getState().lang
}
