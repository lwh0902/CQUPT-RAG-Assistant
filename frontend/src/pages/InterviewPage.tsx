import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowLeft, Briefcase, CheckCircle2, Download, Loader2, Trash2 } from 'lucide-react'
import {
  deleteInterviewSession,
  fetchInterviewSession,
  fetchInterviewSessions,
  generateInterviewBank,
  type InterviewSession,
} from '../api/client'
import { buildInterviewMarkdown } from '../components/interview/exportMarkdown'
import { useToastStore } from '../store/toast'
import { useT } from '../i18n'

type Tab = 'mcq' | 'qa'

export default function InterviewPage() {
  const t = useT()
  const [company, setCompany] = useState('')
  const [jdText, setJdText] = useState('')
  const [resumeText, setResumeText] = useState('')
  const [resumeFile, setResumeFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [session, setSession] = useState<InterviewSession | null>(null)
  const [history, setHistory] = useState<InterviewSession[]>([])
  const [tab, setTab] = useState<Tab>('mcq')
  const [revealed, setRevealed] = useState<Set<number>>(new Set())

  const loadHistory = async () => {
    try {
      setHistory(await fetchInterviewSessions())
    } catch {
      /* history is optional */
    }
  }

  useEffect(() => {
    loadHistory()
  }, [])

  const handleGenerate = async () => {
    if (jdText.trim().length < 10) {
      useToastStore.getState().addToast(t('interview.jdRequired'), 'error')
      return
    }
    if (!resumeFile && resumeText.trim().length < 20) {
      useToastStore.getState().addToast(t('interview.resumeRequired'), 'error')
      return
    }
    setLoading(true)
    setSession(null)
    try {
      const form = new FormData()
      form.append('company', company.trim())
      form.append('jd_text', jdText.trim())
      form.append('resume_text', resumeText.trim())
      if (resumeFile) form.append('resume_file', resumeFile)
      const result = await generateInterviewBank(form)
      setSession(result)
      setTab('mcq')
      setRevealed(new Set())
      loadHistory()
    } catch (err: any) {
      const detail = err?.response?.data?.detail
      useToastStore.getState().addToast(detail || t('interview.generateFailed'), 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleExport = () => {
    if (!session) return
    const markdown = buildInterviewMarkdown(session)
    const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `面试题库-${session.company || '通用'}-${(session.created_at || '').slice(0, 10)}.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  const openHistory = async (id: string) => {
    try {
      setSession(await fetchInterviewSession(id))
      setTab('mcq')
      setRevealed(new Set())
    } catch {
      useToastStore.getState().addToast(t('interview.generateFailed'), 'error')
    }
  }

  const removeHistory = async (id: string) => {
    try {
      await deleteInterviewSession(id)
      setHistory((items) => items.filter((item) => item.id !== id))
      if (session?.id === id) setSession(null)
    } catch {
      /* ignore */
    }
  }

  const toggleReveal = (index: number) => {
    setRevealed((prev) => {
      const next = new Set(prev)
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })
  }

  return (
    <div className="min-h-dvh bg-[var(--bg-primary)] text-[var(--text-primary)]">
      <header className="sticky top-0 z-10 flex h-14 items-center gap-3 border-b border-[var(--border)] bg-[var(--bg-primary)]/90 px-4 backdrop-blur">
        <Link
          to="/chat"
          className="flex h-9 items-center gap-1.5 rounded-lg px-2 text-sm text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]"
        >
          <ArrowLeft className="h-4 w-4" />
          {t('interview.back')}
        </Link>
        <h1 className="flex items-center gap-2 text-sm font-medium">
          <Briefcase className="h-4 w-4 text-[var(--accent)]" />
          {t('interview.title')}
        </h1>
        {session && (
          <button
            type="button"
            onClick={handleExport}
            className="ml-auto flex h-9 items-center gap-1.5 rounded-lg border border-[var(--border)] px-3 text-xs text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]"
          >
            <Download className="h-4 w-4" />
            {t('interview.export')}
          </button>
        )}
      </header>

      <main className="mx-auto max-w-3xl px-4 py-6">
        {!session && (
          <div className="space-y-4">
            <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)]/80 p-5 shadow-sm">
              <label className="mb-1 block text-xs text-[var(--text-tertiary)]">{t('interview.company')}</label>
              <input
                value={company}
                onChange={(e) => setCompany(e.target.value)}
                placeholder={t('interview.companyPh')}
                className="mb-4 w-full rounded-lg border border-[var(--border-input)] bg-[var(--bg-input)] px-3 py-2 text-sm focus:border-[var(--accent)] focus:outline-none"
              />
              <label className="mb-1 block text-xs text-[var(--text-tertiary)]">{t('interview.jd')}</label>
              <textarea
                value={jdText}
                onChange={(e) => setJdText(e.target.value)}
                placeholder={t('interview.jdPh')}
                rows={6}
                className="mb-4 w-full resize-none rounded-lg border border-[var(--border-input)] bg-[var(--bg-input)] px-3 py-2 text-sm focus:border-[var(--accent)] focus:outline-none"
              />
              <label className="mb-1 block text-xs text-[var(--text-tertiary)]">{t('interview.resume')}</label>
              <textarea
                value={resumeText}
                onChange={(e) => setResumeText(e.target.value)}
                placeholder={t('interview.resumePh')}
                rows={6}
                className="mb-3 w-full resize-none rounded-lg border border-[var(--border-input)] bg-[var(--bg-input)] px-3 py-2 text-sm focus:border-[var(--accent)] focus:outline-none"
              />
              <div className="mb-4 flex items-center gap-3 text-xs text-[var(--text-tertiary)]">
                <span>{t('interview.orUploadPdf')}</span>
                <input
                  type="file"
                  accept=".pdf,.docx,.txt"
                  onChange={(e) => setResumeFile(e.target.files?.[0] ?? null)}
                  className="text-xs"
                />
                {resumeFile && <span className="text-[var(--text-secondary)]">{resumeFile.name}</span>}
              </div>
              <button
                type="button"
                onClick={handleGenerate}
                disabled={loading}
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-[var(--accent)] py-2.5 text-sm font-medium text-white transition-opacity hover:opacity-90 disabled:opacity-50"
              >
                {loading && <Loader2 className="h-4 w-4 animate-spin" />}
                {loading ? t('interview.generating') : t('interview.generate')}
              </button>
              {loading && (
                <p className="mt-2 text-center text-xs text-[var(--text-tertiary)]">{t('interview.generatingHint')}</p>
              )}
            </div>

            {history.length > 0 && (
              <div className="rounded-2xl border border-[var(--border)] p-4">
                <h2 className="mb-2 text-xs font-medium text-[var(--text-tertiary)]">{t('interview.history')}</h2>
                <ul className="space-y-1">
                  {history.map((item) => (
                    <li key={item.id} className="flex items-center justify-between rounded-lg px-2 py-1.5 text-sm hover:bg-[var(--bg-hover)]">
                      <button type="button" onClick={() => openHistory(item.id)} className="min-w-0 flex-1 truncate text-left text-[var(--text-secondary)]">
                        {item.company || t('interview.noCompany')} · {(item.created_at || '').slice(0, 10)} · {item.mcq_count ?? 0}+{item.qa_count ?? 0}题
                      </button>
                      <button type="button" onClick={() => removeHistory(item.id)} className="p-1 text-[var(--text-tertiary)] hover:text-red-500" aria-label={t('interview.delete')}>
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {session && (
          <div>
            <div className="mb-4 flex items-center justify-between">
              <div className="text-sm text-[var(--text-secondary)]">
                {session.company || t('interview.noCompany')} · {session.mcq.length}+{session.qa.length}题
              </div>
              <button type="button" onClick={() => setSession(null)} className="text-xs text-[var(--text-tertiary)] hover:underline">
                {t('interview.newBank')}
              </button>
            </div>

            <div className="mb-4 flex gap-2">
              {(['mcq', 'qa'] as Tab[]).map((key) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setTab(key)}
                  className={`rounded-lg px-4 py-2 text-sm transition-colors ${
                    tab === key
                      ? 'bg-[var(--accent)] text-white'
                      : 'border border-[var(--border)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]'
                  }`}
                >
                  {key === 'mcq' ? t('interview.mcqTab') : t('interview.qaTab')}
                </button>
              ))}
            </div>

            {tab === 'mcq' && (
              <div className="space-y-4">
                {session.mcq.map((item, index) => (
                  <div key={index} className="rounded-2xl border border-[var(--border)] bg-[var(--surface)]/80 p-4">
                    <p className="mb-3 text-sm font-medium">
                      {index + 1}. {item.question}
                    </p>
                    <ul className="mb-3 space-y-1.5">
                      {['A', 'B', 'C', 'D'].map((key) => (
                        <li
                          key={key}
                          className={`rounded-lg px-3 py-2 text-sm ${
                            revealed.has(index) && item.answer === key
                              ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                              : 'text-[var(--text-secondary)]'
                          }`}
                        >
                          {key}. {item.options[key]}
                          {revealed.has(index) && item.answer === key && (
                            <CheckCircle2 className="ml-1.5 inline h-4 w-4" />
                          )}
                        </li>
                      ))}
                    </ul>
                    <button type="button" onClick={() => toggleReveal(index)} className="text-xs text-[var(--accent)] hover:underline">
                      {revealed.has(index) ? t('interview.hideAnswer') : t('interview.showAnswer')}
                    </button>
                    {revealed.has(index) && (
                      <div className="mt-2 rounded-lg border-l-2 border-red-500 bg-red-500/5 px-3 py-2 text-xs leading-relaxed text-[var(--text-secondary)]">
                        <span className="font-medium text-red-500">{t('interview.analysis')}</span>
                        {item.analysis}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {tab === 'qa' && (
              <div className="space-y-4">
                {session.qa.map((item, index) => (
                  <div key={index} className="rounded-2xl border border-[var(--border)] bg-[var(--surface)]/80 p-4">
                    <p className="mb-3 text-sm font-medium">
                      {index + 1}. {item.question}
                    </p>
                    <div className="mb-3 rounded-lg bg-[var(--bg-secondary)] px-3 py-2.5 text-sm leading-relaxed text-[var(--text-primary)]">
                      <span className="mb-1 block text-xs font-medium text-[var(--accent)]">{t('interview.spokenAnswer')}</span>
                      {item.spoken_answer}
                    </div>
                    <div className="rounded-lg border-l-2 border-red-500 bg-red-500/5 px-3 py-2 text-xs leading-relaxed text-[var(--text-secondary)]">
                      <span className="font-medium text-red-500">{t('interview.analysis')}</span>
                      {item.analysis}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
