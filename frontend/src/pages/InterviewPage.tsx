import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  ArrowLeft,
  Briefcase,
  CheckCircle2,
  Download,
  FileText,
  Globe2,
  Loader2,
  Trash2,
  Upload,
  XCircle,
} from 'lucide-react'
import {
  createWeaknessReport,
  deleteInterviewSession,
  fetchInterviewSession,
  fetchInterviewSessions,
  regenerateMcq,
  streamGenerateInterviewBank,
  type InterviewGenerateStage,
  type InterviewSession,
} from '../api/client'
import { buildInterviewMarkdown } from '../components/interview/exportMarkdown'
import GenerationRunner from '../components/interview/GenerationRunner'
import InterviewTutor from '../components/interview/InterviewTutor'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useToastStore } from '../store/toast'
import { useT } from '../i18n'

type Tab = 'mcq' | 'qa' | 'real'

export default function InterviewPage() {
  const t = useT()
  const [company, setCompany] = useState('')
  const [position, setPosition] = useState('')
  const [jdText, setJdText] = useState('')
  const [resumeText, setResumeText] = useState('')
  const [resumeFile, setResumeFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [genStage, setGenStage] = useState<InterviewGenerateStage | null>(null)
  const [session, setSession] = useState<InterviewSession | null>(null)
  const [history, setHistory] = useState<InterviewSession[]>([])
  const [tab, setTab] = useState<Tab>('mcq')

  // Quiz state (round-1 MCQ)
  const [answers, setAnswers] = useState<Record<number, string>>({})
  const [submitted, setSubmitted] = useState(false)
  const [reportLoading, setReportLoading] = useState(false)
  const [regenLoading, setRegenLoading] = useState(false)
  const [viewRound, setViewRound] = useState(1)

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

  const resetQuiz = () => {
    setAnswers({})
    setSubmitted(false)
    setViewRound(1)
  }

  const handleGenerate = () => {
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
    setGenStage({ stage: 'parse', message: t('interview.stageSearch'), progress: 2 })
    const form = new FormData()
    form.append('company', company.trim())
    form.append('position', position.trim())
    form.append('jd_text', jdText.trim())
    form.append('resume_text', resumeText.trim())
    if (resumeFile) form.append('resume_file', resumeFile)

    streamGenerateInterviewBank(form, {
      onStage: (stage) => setGenStage(stage),
      onDone: (result) => {
        setSession(result)
        setTab(result.real_questions && result.real_questions.length > 0 ? 'real' : 'mcq')
        resetQuiz()
        loadHistory()
        setLoading(false)
        setGenStage(null)
      },
      onError: (message) => {
        useToastStore.getState().addToast(message || t('interview.generateFailed'), 'error')
        setLoading(false)
        setGenStage(null)
      },
    })
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
      resetQuiz()
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

  // ---- quiz logic ----
  const mcqRound1 = useMemo(
    () => (session?.mcq ?? []).filter((item) => (item.round ?? 1) === 1),
    [session],
  )
  const mcqRounds = useMemo(
    () => Array.from(new Set((session?.mcq ?? []).map((item) => item.round ?? 1))).sort(),
    [session],
  )
  const visibleMcq = useMemo(
    () => (session?.mcq ?? []).filter((item) => (item.round ?? 1) === viewRound),
    [session, viewRound],
  )

  const wrongIndices = useMemo(() => {
    if (!submitted || viewRound !== 1) return []
    return mcqRound1
      .map((item, index) => ({ item, index }))
      .filter(({ item, index }) => answers[index] !== item.answer)
      .map(({ index }) => index + 1)
  }, [submitted, viewRound, mcqRound1, answers])

  const currentWrongCount = useMemo(() => {
    if (!submitted) return 0
    return visibleMcq.filter((item, index) => answers[index] !== item.answer).length
  }, [submitted, visibleMcq, answers])

  const score = submitted ? visibleMcq.length - currentWrongCount : 0
  const answeredCount = Object.keys(answers).length

  const handleSubmitQuiz = () => {
    if (answeredCount < visibleMcq.length) {
      useToastStore.getState().addToast(t('interview.answerAll'), 'info')
      return
    }
    setSubmitted(true)
  }

  const handleReport = async () => {
    if (!session || wrongIndices.length === 0) {
      useToastStore.getState().addToast(t('interview.noWrong'), 'info')
      return
    }
    setReportLoading(true)
    try {
      const { report } = await createWeaknessReport(session.id, wrongIndices)
      setSession({ ...session, report_text: report })
    } catch (err: any) {
      useToastStore.getState().addToast(err?.response?.data?.detail || t('interview.generateFailed'), 'error')
    } finally {
      setReportLoading(false)
    }
  }

  const handleRegen = async () => {
    if (!session) return
    setRegenLoading(true)
    try {
      const { round, mcq } = await regenerateMcq(session.id, wrongIndices)
      setSession({ ...session, mcq: [...session.mcq, ...mcq.map((item) => ({ ...item, round }))] })
      setViewRound(round)
      setSubmitted(false)
      setAnswers({})
    } catch (err: any) {
      useToastStore.getState().addToast(err?.response?.data?.detail || t('interview.generateFailed'), 'error')
    } finally {
      setRegenLoading(false)
    }
  }

  const qaGroups = useMemo(() => {
    const groups: Record<string, typeof session.qa> = {}
    for (const item of session?.qa ?? []) {
      const category = item.category || '综合'
      groups[category] = [...(groups[category] ?? []), item]
    }
    return groups
  }, [session])

  const formatSize = (bytes: number) =>
    bytes > 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)}MB` : `${Math.max(1, Math.round(bytes / 1024))}KB`

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
              <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div>
                  <label className="mb-1 block text-xs text-[var(--text-tertiary)]">{t('interview.company')}</label>
                  <input
                    value={company}
                    onChange={(e) => setCompany(e.target.value)}
                    placeholder={t('interview.companyPh')}
                    className="w-full rounded-lg border border-[var(--border-input)] bg-[var(--bg-input)] px-3 py-2 text-sm focus:border-[var(--accent)] focus:outline-none"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs text-[var(--text-tertiary)]">{t('interview.position')}</label>
                  <input
                    value={position}
                    onChange={(e) => setPosition(e.target.value)}
                    placeholder={t('interview.positionPh')}
                    className="w-full rounded-lg border border-[var(--border-input)] bg-[var(--bg-input)] px-3 py-2 text-sm focus:border-[var(--accent)] focus:outline-none"
                  />
                </div>
              </div>
              <label className="mb-1 block text-xs text-[var(--text-tertiary)]">{t('interview.jd')}</label>
              <textarea
                value={jdText}
                onChange={(e) => setJdText(e.target.value)}
                placeholder={t('interview.jdPh')}
                rows={6}
                className="mb-4 w-full resize-none rounded-lg border border-[var(--border-input)] bg-[var(--bg-input)] px-3 py-2 text-sm focus:border-[var(--accent)] focus:outline-none"
              />
              <label className="mb-1 block text-xs text-[var(--text-tertiary)]">{t('interview.resume')}</label>

              {/* Upload: clear button + selected-file chip */}
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-dashed border-[var(--accent)] bg-[var(--accent-light)] px-4 py-2.5 text-sm font-medium text-[var(--accent)] transition-opacity hover:opacity-80">
                  <Upload className="h-4 w-4" />
                  {t('interview.uploadResume')}
                  <input
                    type="file"
                    accept=".pdf,.docx,.txt"
                    className="hidden"
                    onChange={(e) => setResumeFile(e.target.files?.[0] ?? null)}
                  />
                </label>
                {resumeFile ? (
                  <span className="flex items-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] px-3 py-2 text-xs text-[var(--text-secondary)]">
                    <FileText className="h-4 w-4 text-[var(--accent)]" />
                    {resumeFile.name}
                    <span className="text-[var(--text-tertiary)]">{formatSize(resumeFile.size)}</span>
                    <button
                      type="button"
                      onClick={() => setResumeFile(null)}
                      aria-label={t('interview.removeFile')}
                      className="text-[var(--text-tertiary)] hover:text-red-500"
                    >
                      <XCircle className="h-4 w-4" />
                    </button>
                  </span>
                ) : (
                  <span className="text-xs text-[var(--text-tertiary)]">{t('interview.noFile')}</span>
                )}
              </div>

              <textarea
                value={resumeText}
                onChange={(e) => setResumeText(e.target.value)}
                placeholder={t('interview.resumePh')}
                rows={5}
                className="mb-4 w-full resize-none rounded-lg border border-[var(--border-input)] bg-[var(--bg-input)] px-3 py-2 text-sm focus:border-[var(--accent)] focus:outline-none"
              />
              <button
                type="button"
                onClick={handleGenerate}
                disabled={loading}
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-[var(--accent)] py-2.5 text-sm font-medium text-white transition-opacity hover:opacity-90 disabled:opacity-50"
              >
                {loading && <Loader2 className="h-4 w-4 animate-spin" />}
                {loading ? t('interview.generating') : t('interview.generate')}
              </button>
              {loading && <GenerationRunner stage={genStage} />}
            </div>

            {history.length > 0 && (
              <div className="rounded-2xl border border-[var(--border)] p-4">
                <h2 className="mb-2 text-xs font-medium text-[var(--text-tertiary)]">{t('interview.history')}</h2>
                <ul className="space-y-1">
                  {history.map((item) => (
                    <li key={item.id} className="flex items-center justify-between rounded-lg px-2 py-1.5 text-sm hover:bg-[var(--bg-hover)]">
                      <button type="button" onClick={() => openHistory(item.id)} className="min-w-0 flex-1 truncate text-left text-[var(--text-secondary)]">
                        {item.company || t('interview.noCompany')}
                        {item.position ? ` · ${item.position}` : ''} · {(item.created_at || '').slice(0, 10)} · {item.mcq_count ?? 0}+{item.qa_count ?? 0}题
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
            <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
              <div className="text-sm text-[var(--text-secondary)]">
                {session.company || t('interview.noCompany')}
                {session.position ? ` · ${session.position}` : ''} · {session.mcq.length}+{session.qa.length}题
              </div>
              <div className="flex items-center gap-3">
                {session.reference_used && (
                  <span className="flex items-center gap-1 text-xs text-[var(--accent)]" title={(session.references ?? []).map((r) => r.title).join('\n')}>
                    <Globe2 className="h-3.5 w-3.5" />
                    {t('interview.referenceUsed')}
                  </span>
                )}
                <button type="button" onClick={() => setSession(null)} className="text-xs text-[var(--text-tertiary)] hover:underline">
                  {t('interview.newBank')}
                </button>
              </div>
            </div>

            <div className="mb-4 flex flex-wrap gap-2">
              {((session.real_questions && session.real_questions.length > 0
                  ? (['real', 'mcq', 'qa'] as Tab[])
                  : (['mcq', 'qa'] as Tab[])
                )).map((key) => (
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
                  {key === 'mcq'
                    ? t('interview.mcqTab')
                    : key === 'qa'
                      ? t('interview.qaTab')
                      : `${t('interview.tabReal')}${session.real_questions?.length ? `（${session.real_questions.length}）` : ''}`}
                </button>
              ))}
              {tab === 'mcq' && mcqRounds.length > 1 && (
                <div className="ml-auto flex items-center gap-1 text-xs">
                  {mcqRounds.map((round) => (
                    <button
                      key={round}
                      type="button"
                      onClick={() => { setViewRound(round); setSubmitted(false); setAnswers({}) }}
                      className={`rounded-md px-2.5 py-1.5 ${
                        viewRound === round
                          ? 'bg-[var(--bg-secondary)] text-[var(--text-primary)]'
                          : 'text-[var(--text-tertiary)] hover:bg-[var(--bg-hover)]'
                      }`}
                    >
                      {t('interview.roundN').replace('{n}', String(round))}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {tab === 'real' && (
              <div className="space-y-3">
                <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)]/80 p-4">
                  <p className="mb-1 text-sm font-medium">{t('interview.realQuestions')}</p>
                  <p className="text-xs text-[var(--text-tertiary)]">{t('interview.realQuestionsHint')}</p>
                </div>
                {(session.real_questions ?? []).length === 0 ? (
                  <p className="text-sm text-[var(--text-tertiary)]">{t('interview.realQuestionsEmpty')}</p>
                ) : (
                  <ol className="space-y-2">
                    {(session.real_questions ?? []).map((item, index) => (
                      <li key={index} className="rounded-2xl border border-[var(--border)] bg-[var(--surface)]/80 p-4">
                        <p className="text-sm font-medium text-[var(--text-primary)]">
                          {index + 1}. {item.question}
                        </p>
                        {(item.source_title || item.source_url) && (
                          <p className="mt-2 text-[11px] text-[var(--text-tertiary)]">
                            {t('interview.source')}：
                            {item.source_url ? (
                              <a
                                href={item.source_url}
                                target="_blank"
                                rel="noreferrer"
                                className="text-[var(--accent)] hover:underline"
                              >
                                {item.source_title || item.source_url}
                              </a>
                            ) : (
                              item.source_title
                            )}
                          </p>
                        )}
                      </li>
                    ))}
                  </ol>
                )}
              </div>
            )}

            {tab === 'mcq' && (
              <div className="space-y-4">
                {submitted && (
                  <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)]/80 p-4">
                    <p className="text-sm">
                      {t('interview.quizScore')
                        .replace('{score}', String(score))
                        .replace('{total}', String(visibleMcq.length))}
                      {currentWrongCount > 0 && (
                        <span className="ml-2 text-red-500">
                          {t('interview.wrongCount').replace('{n}', String(currentWrongCount))}
                        </span>
                      )}
                    </p>
                    {viewRound === 1 && wrongIndices.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-2">
                        <button
                          type="button"
                          onClick={handleReport}
                          disabled={reportLoading}
                          className="flex items-center gap-1.5 rounded-lg bg-[var(--accent)] px-3 py-1.5 text-xs text-white disabled:opacity-50"
                        >
                          {reportLoading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                          {session.report_text ? t('interview.reportRegen') : t('interview.report')}
                        </button>
                        <button
                          type="button"
                          onClick={handleRegen}
                          disabled={regenLoading}
                          className="flex items-center gap-1.5 rounded-lg border border-[var(--border)] px-3 py-1.5 text-xs text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] disabled:opacity-50"
                        >
                          {regenLoading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                          {t('interview.regenMcq')}
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {viewRound === 1 && session.report_text && (
                  <div className="rounded-2xl border border-amber-500/40 bg-amber-500/5 p-4">
                    <p className="mb-2 text-xs font-medium text-amber-600 dark:text-amber-400">{t('interview.reportTitle')}</p>
                    <div className="markdown-body text-sm leading-relaxed text-[var(--text-secondary)]">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{session.report_text}</ReactMarkdown>
                    </div>
                  </div>
                )}

                {visibleMcq.map((item, index) => {
                  const quizIndex = index
                  const chosen = answers[quizIndex]
                  const isWrong = submitted && chosen !== item.answer
                  return (
                    <div key={`${item.round ?? 1}-${index}`} className="rounded-2xl border border-[var(--border)] bg-[var(--surface)]/80 p-4">
                      <p className="mb-3 text-sm font-medium">
                        {index + 1}. {item.question}
                      </p>
                      <ul className="mb-3 space-y-1.5">
                        {['A', 'B', 'C', 'D'].map((key) => {
                          const isChosen = chosen === key
                          const isAnswer = item.answer === key
                          let cls = 'text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]'
                          if (submitted) {
                            if (isAnswer) cls = 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                            else if (isChosen && isWrong) cls = 'bg-red-500/10 text-red-500'
                          } else if (isChosen) {
                            cls = 'bg-[var(--accent-light)] text-[var(--accent)]'
                          }
                          return (
                            <li key={key}>
                              <button
                                type="button"
                                disabled={submitted}
                                onClick={() => setAnswers((prev) => ({ ...prev, [quizIndex]: key }))}
                                className={`w-full rounded-lg px-3 py-2 text-left text-sm transition-colors disabled:cursor-default ${cls}`}
                              >
                                {key}. {item.options[key]}
                                {submitted && isAnswer && (
                                  <CheckCircle2 className="ml-1.5 inline h-4 w-4" />
                                )}
                                {submitted && isChosen && isWrong && (
                                  <XCircle className="ml-1.5 inline h-4 w-4" />
                                )}
                              </button>
                            </li>
                          )
                        })}
                      </ul>
                      {submitted && (
                        <div className="rounded-lg border-l-2 border-red-500 bg-red-500/5 px-3 py-2 text-xs leading-relaxed text-[var(--text-secondary)]">
                          <span className="font-medium text-red-500">{t('interview.analysis')}</span>
                          {item.analysis}
                        </div>
                      )}
                    </div>
                  )
                })}

                {!submitted && visibleMcq.length > 0 && (
                  <button
                    type="button"
                    onClick={handleSubmitQuiz}
                    className="w-full rounded-xl bg-[var(--accent)] py-2.5 text-sm font-medium text-white hover:opacity-90"
                  >
                    {t('interview.submitQuiz').replace('{n}', `${answeredCount}/${visibleMcq.length}`)}
                  </button>
                )}
              </div>
            )}

            {tab === 'qa' && (
              <div className="space-y-6">
                {Object.entries(qaGroups).map(([category, items]) => (
                  <section key={category}>
                    <h3 className="mb-3 flex items-center gap-2 text-sm font-medium text-[var(--text-primary)]">
                      <span className="rounded-md bg-[var(--accent-light)] px-2 py-0.5 text-xs text-[var(--accent)]">{category}</span>
                      <span className="text-xs text-[var(--text-tertiary)]">{items.length}题</span>
                    </h3>
                    <div className="space-y-4">
                      {items.map((item, index) => (
                        <div key={index} className="rounded-2xl border border-[var(--border)] bg-[var(--surface)]/80 p-4">
                          <p className="mb-3 text-sm font-medium">{item.question}</p>
                          <div className="mb-3 rounded-lg bg-[var(--bg-secondary)] px-3 py-2.5 text-sm leading-relaxed text-[var(--text-primary)]">
                            <span className="mb-1 block text-xs font-medium text-[var(--accent)]">{t('interview.spokenAnswer')}</span>
                            {item.spoken_answer}
                          </div>
                          <div className="rounded-lg border-l-2 border-red-500 bg-red-500/5 px-3 py-2 text-xs leading-relaxed text-[var(--text-secondary)]">
                            <span className="font-medium text-red-500">{t('interview.qaGuide')}</span>
                            {item.analysis}
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>
                ))}
              </div>
            )}
          </div>
        )}
      </main>
      <InterviewTutor />
    </div>
  )
}
