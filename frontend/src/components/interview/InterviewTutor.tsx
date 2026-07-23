import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { History, Loader2, Minus, Plus, Send, Trash2, X } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { streamInterviewTutor } from '../../api/client'
import { useT } from '../../i18n'

type ChatMsg = { role: 'user' | 'assistant'; content: string }
type TutorThread = {
  id: string
  title: string
  updatedAt: number
  messages: ChatMsg[]
}

const STORAGE_THREADS = 'interview_tutor_threads_v1'
const STORAGE_UI = 'interview_tutor_ui_v1'

type UiState = {
  fabX: number
  fabY: number
  panelX: number
  panelY: number
  w: number
  h: number
  open: boolean
}

const DEFAULT_SIZE = { w: 380, h: 500 }
const MIN_SIZE = { w: 300, h: 360 }
const MAX_SIZE = { w: 560, h: 740 }
const FAB_SIZE = 120

function uid() {
  return `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`
}

function loadThreads(): TutorThread[] {
  try {
    const raw = localStorage.getItem(STORAGE_THREADS)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function saveThreads(threads: TutorThread[]) {
  localStorage.setItem(STORAGE_THREADS, JSON.stringify(threads.slice(0, 30)))
}

function defaultUi(): UiState {
  const margin = 24
  const w = DEFAULT_SIZE.w
  const h = DEFAULT_SIZE.h
  const fabX = Math.max(margin, window.innerWidth - FAB_SIZE - margin)
  const fabY = Math.max(margin, window.innerHeight - FAB_SIZE - margin - 24)
  return {
    fabX,
    fabY,
    panelX: Math.max(margin, fabX + FAB_SIZE - w),
    panelY: Math.max(margin, fabY - h + 40),
    w,
    h,
    open: false,
  }
}

function loadUi(): UiState {
  try {
    const raw = localStorage.getItem(STORAGE_UI)
    if (!raw) return defaultUi()
    const parsed = JSON.parse(raw) as Partial<UiState>
    const base = defaultUi()
    return clampUi({
      fabX: num(parsed.fabX, base.fabX),
      fabY: num(parsed.fabY, base.fabY),
      panelX: num(parsed.panelX, base.panelX),
      panelY: num(parsed.panelY, base.panelY),
      w: num(parsed.w, base.w),
      h: num(parsed.h, base.h),
      open: false,
    })
  } catch {
    return defaultUi()
  }
}

function num(v: unknown, fallback: number) {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback
}

function clampUi(ui: UiState): UiState {
  const w = Math.min(MAX_SIZE.w, Math.max(MIN_SIZE.w, ui.w))
  const h = Math.min(MAX_SIZE.h, Math.max(MIN_SIZE.h, ui.h))
  const maxFabX = Math.max(8, window.innerWidth - FAB_SIZE - 8)
  const maxFabY = Math.max(8, window.innerHeight - FAB_SIZE - 8)
  const maxPanelX = Math.max(8, window.innerWidth - w - 8)
  const maxPanelY = Math.max(8, window.innerHeight - h - 8)
  return {
    ...ui,
    w,
    h,
    fabX: Math.min(maxFabX, Math.max(8, ui.fabX)),
    fabY: Math.min(maxFabY, Math.max(8, ui.fabY)),
    panelX: Math.min(maxPanelX, Math.max(8, ui.panelX)),
    panelY: Math.min(maxPanelY, Math.max(8, ui.panelY)),
  }
}

function titleFromMessages(messages: ChatMsg[]): string {
  const firstUser = messages.find((m) => m.role === 'user')?.content?.trim()
  if (!firstUser) return '新对话'
  return firstUser.replace(/\s+/g, ' ').slice(0, 24)
}

function TutorMascot({ size = 96 }: { size?: number }) {
  return (
    <div className="tutor-mascot" style={{ width: size, height: size }} aria-hidden>
      <div className="tutor-mascot-shadow" />
      <div className="tutor-mascot-wing tutor-mascot-wing-l" />
      <div className="tutor-mascot-wing tutor-mascot-wing-r" />
      <div className="tutor-mascot-body">
        <div className="tutor-mascot-hair" />
        <div className="tutor-mascot-face">
          <span className="tutor-mascot-eye l" />
          <span className="tutor-mascot-eye r" />
          <span className="tutor-mascot-blush l" />
          <span className="tutor-mascot-blush r" />
          <span className="tutor-mascot-mouth" />
        </div>
        <div className="tutor-mascot-flower l" />
        <div className="tutor-mascot-flower r" />
        <div className="tutor-mascot-gem" />
      </div>
    </div>
  )
}

export default function InterviewTutor() {
  const t = useT()
  const [ui, setUi] = useState<UiState>(() => loadUi())
  const [threads, setThreads] = useState<TutorThread[]>(() => loadThreads())
  const [activeId, setActiveId] = useState<string>(() => loadThreads()[0]?.id || '')
  const [showHistory, setShowHistory] = useState(false)
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [bubbleVisible, setBubbleVisible] = useState(true)

  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<{ abort: () => void } | null>(null)
  const dragRef = useRef<{
    mode: 'panel' | 'resize' | 'fab'
    startX: number
    startY: number
    orig: UiState
    moved: boolean
  } | null>(null)

  const active = useMemo(
    () => threads.find((th) => th.id === activeId) || null,
    [threads, activeId],
  )
  const messages = active?.messages ?? []

  useEffect(() => {
    saveThreads(threads)
  }, [threads])

  useEffect(() => {
    const { open: _o, ...rest } = ui
    localStorage.setItem(STORAGE_UI, JSON.stringify(rest))
  }, [ui.fabX, ui.fabY, ui.panelX, ui.panelY, ui.w, ui.h])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, ui.open, showHistory])

  useEffect(() => () => abortRef.current?.abort(), [])

  useEffect(() => {
    const onResize = () => setUi((prev) => clampUi(prev))
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  const upsertThread = useCallback((thread: TutorThread) => {
    setThreads((prev) => {
      const rest = prev.filter((item) => item.id !== thread.id)
      return [thread, ...rest].sort((a, b) => b.updatedAt - a.updatedAt)
    })
  }, [])

  const ensureThread = useCallback((): TutorThread => {
    if (active) return active
    const created: TutorThread = {
      id: uid(),
      title: t('interview.tutorNewChat'),
      updatedAt: Date.now(),
      messages: [],
    }
    setActiveId(created.id)
    upsertThread(created)
    return created
  }, [active, t, upsertThread])

  const openPanel = useCallback(() => {
    setUi((prev) => {
      // Open panel near the character.
      const panelX = Math.min(
        Math.max(8, prev.fabX + FAB_SIZE - prev.w),
        Math.max(8, window.innerWidth - prev.w - 8),
      )
      const panelY = Math.min(
        Math.max(8, prev.fabY - prev.h + 48),
        Math.max(8, window.innerHeight - prev.h - 8),
      )
      return clampUi({ ...prev, panelX, panelY, open: true })
    })
    setBubbleVisible(false)
    setShowHistory(false)
    if (!activeId) {
      const created: TutorThread = {
        id: uid(),
        title: t('interview.tutorNewChat'),
        updatedAt: Date.now(),
        messages: [],
      }
      setActiveId(created.id)
      upsertThread(created)
    }
  }, [activeId, t, upsertThread])

  const minimizePanel = () => {
    setUi((prev) => ({ ...prev, open: false }))
    setShowHistory(false)
    setBubbleVisible(true)
  }

  const startNewChat = () => {
    const created: TutorThread = {
      id: uid(),
      title: t('interview.tutorNewChat'),
      updatedAt: Date.now(),
      messages: [],
    }
    setActiveId(created.id)
    upsertThread(created)
    setShowHistory(false)
    setInput('')
  }

  const deleteThread = (id: string) => {
    setThreads((prev) => {
      const next = prev.filter((item) => item.id !== id)
      if (activeId === id) setActiveId(next[0]?.id || '')
      return next
    })
  }

  const send = () => {
    const text = input.trim()
    if (!text || streaming) return
    const thread = ensureThread()
    const nextMessages: ChatMsg[] = [...thread.messages, { role: 'user', content: text }]
    const pending: TutorThread = {
      ...thread,
      title: thread.messages.length === 0 ? titleFromMessages(nextMessages) : thread.title,
      updatedAt: Date.now(),
      messages: [...nextMessages, { role: 'assistant', content: '' }],
    }
    upsertThread(pending)
    setActiveId(pending.id)
    setInput('')
    setStreaming(true)
    setShowHistory(false)

    abortRef.current?.abort()
    abortRef.current = streamInterviewTutor(nextMessages, {
      onToken: (token) => {
        setThreads((prev) =>
          prev.map((item) => {
            if (item.id !== pending.id) return item
            const msgs = [...item.messages]
            const last = msgs[msgs.length - 1]
            if (last?.role === 'assistant') {
              msgs[msgs.length - 1] = { role: 'assistant', content: last.content + token }
            }
            return { ...item, messages: msgs, updatedAt: Date.now() }
          }),
        )
      },
      onDone: () => setStreaming(false),
      onError: (message) => {
        setStreaming(false)
        setThreads((prev) =>
          prev.map((item) => {
            if (item.id !== pending.id) return item
            const msgs = [...item.messages]
            const last = msgs[msgs.length - 1]
            if (last?.role === 'assistant' && !last.content) {
              msgs[msgs.length - 1] = {
                role: 'assistant',
                content: message || t('interview.tutorError'),
              }
            } else {
              msgs.push({ role: 'assistant', content: message || t('interview.tutorError') })
            }
            return { ...item, messages: msgs, updatedAt: Date.now() }
          }),
        )
      },
    })
  }

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const drag = dragRef.current
      if (!drag) return
      const dx = e.clientX - drag.startX
      const dy = e.clientY - drag.startY
      if (Math.hypot(dx, dy) > 4) drag.moved = true

      if (drag.mode === 'fab') {
        setUi((prev) =>
          clampUi({
            ...prev,
            fabX: drag.orig.fabX + dx,
            fabY: drag.orig.fabY + dy,
          }),
        )
      } else if (drag.mode === 'panel') {
        setUi((prev) =>
          clampUi({
            ...prev,
            panelX: drag.orig.panelX + dx,
            panelY: drag.orig.panelY + dy,
          }),
        )
      } else if (drag.mode === 'resize') {
        setUi((prev) =>
          clampUi({
            ...prev,
            w: drag.orig.w + dx,
            h: drag.orig.h + dy,
          }),
        )
      }
    }

    const onUp = () => {
      const drag = dragRef.current
      dragRef.current = null
      document.body.style.userSelect = ''
      document.body.style.cursor = ''
      if (drag?.mode === 'fab' && !drag.moved) {
        openPanel()
      }
    }

    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
  }, [openPanel])

  const beginDrag = (e: React.PointerEvent, mode: 'panel' | 'resize' | 'fab') => {
    e.preventDefault()
    e.stopPropagation()
    dragRef.current = {
      mode,
      startX: e.clientX,
      startY: e.clientY,
      orig: ui,
      moved: false,
    }
    document.body.style.userSelect = 'none'
    document.body.style.cursor = mode === 'resize' ? 'nwse-resize' : 'grabbing'
  }

  if (!ui.open) {
    return (
      <div className="fixed z-40 touch-none" style={{ left: ui.fabX, top: ui.fabY }}>
        {bubbleVisible && (
          <button
            type="button"
            onClick={openPanel}
            className="tutor-bubble absolute bottom-[108px] right-0 w-[158px] rounded-2xl border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-left text-xs leading-snug text-[var(--text-secondary)] shadow-lg"
          >
            {t('interview.tutorBubble')}
            <span className="tutor-bubble-tail" />
          </button>
        )}
        <button
          type="button"
          onPointerDown={(e) => beginDrag(e, 'fab')}
          className="tutor-fab-wrap relative flex w-[120px] flex-col items-center"
          title={t('interview.tutorTitle')}
        >
          <TutorMascot size={104} />
          <span className="mt-0.5 rounded-full border border-[var(--border)] bg-[var(--surface)]/95 px-2.5 py-0.5 text-[11px] font-medium text-[var(--text-secondary)] shadow">
            {t('interview.tutorFab')}
          </span>
        </button>
        <style>{mascotCss}</style>
      </div>
    )
  }

  return (
    <div
      className="fixed z-50 flex flex-col overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--bg-primary)] shadow-2xl"
      style={{ left: ui.panelX, top: ui.panelY, width: ui.w, height: ui.h }}
    >
      <header
        className="flex cursor-grab items-center gap-2 border-b border-[var(--border)] bg-[var(--surface)]/90 px-3 py-2.5 active:cursor-grabbing"
        onPointerDown={(e) => beginDrag(e, 'panel')}
      >
        <TutorMascot size={34} />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">{t('interview.tutorTitle')}</p>
          <p className="truncate text-[11px] text-[var(--text-tertiary)]">
            {active?.title || t('interview.tutorHint')}
          </p>
        </div>
        <button
          type="button"
          title={t('interview.tutorHistory')}
          onPointerDown={(e) => e.stopPropagation()}
          onClick={() => setShowHistory((v) => !v)}
          className={`rounded-lg p-1.5 hover:bg-[var(--bg-hover)] ${
            showHistory ? 'text-[var(--accent)]' : 'text-[var(--text-tertiary)]'
          }`}
        >
          <History className="h-4 w-4" />
        </button>
        <button
          type="button"
          title={t('interview.tutorNewChat')}
          onPointerDown={(e) => e.stopPropagation()}
          onClick={startNewChat}
          className="rounded-lg p-1.5 text-[var(--text-tertiary)] hover:bg-[var(--bg-hover)]"
        >
          <Plus className="h-4 w-4" />
        </button>
        <button
          type="button"
          title={t('interview.tutorMinimize')}
          onPointerDown={(e) => e.stopPropagation()}
          onClick={minimizePanel}
          className="rounded-lg p-1.5 text-[var(--text-tertiary)] hover:bg-[var(--bg-hover)]"
        >
          <Minus className="h-4 w-4" />
        </button>
        <button
          type="button"
          onPointerDown={(e) => e.stopPropagation()}
          onClick={minimizePanel}
          className="rounded-lg p-1.5 text-[var(--text-tertiary)] hover:bg-[var(--bg-hover)]"
        >
          <X className="h-4 w-4" />
        </button>
      </header>

      {showHistory ? (
        <div className="flex-1 overflow-y-auto px-3 py-2">
          <p className="mb-2 px-1 text-[11px] font-medium text-[var(--text-tertiary)]">
            {t('interview.tutorRecent')}
          </p>
          {threads.length === 0 && (
            <p className="px-1 text-xs text-[var(--text-tertiary)]">{t('interview.tutorNoHistory')}</p>
          )}
          <ul className="space-y-1">
            {threads.map((thread) => (
              <li key={thread.id}>
                <div
                  className={`group flex items-center gap-2 rounded-xl px-2.5 py-2 text-sm ${
                    thread.id === activeId
                      ? 'bg-[var(--accent-light)] text-[var(--accent)]'
                      : 'text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]'
                  }`}
                >
                  <button
                    type="button"
                    className="min-w-0 flex-1 truncate text-left"
                    onClick={() => {
                      setActiveId(thread.id)
                      setShowHistory(false)
                    }}
                  >
                    {thread.title || t('interview.tutorNewChat')}
                  </button>
                  <button
                    type="button"
                    className="shrink-0 rounded p-1 opacity-0 hover:text-red-500 group-hover:opacity-100"
                    onClick={() => deleteThread(thread.id)}
                    aria-label="delete"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="flex-1 space-y-3 overflow-y-auto px-3 py-3">
          {messages.length === 0 && (
            <div className="flex items-start gap-2">
              <div className="shrink-0">
                <TutorMascot size={48} />
              </div>
              <div className="rounded-2xl bg-[var(--bg-secondary)] px-3 py-2.5 text-xs leading-relaxed text-[var(--text-secondary)]">
                {t('interview.tutorWelcome')}
              </div>
            </div>
          )}
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`max-w-[92%] rounded-2xl px-3 py-2 text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'ml-auto bg-[var(--accent)] text-white'
                  : 'mr-auto border border-[var(--border)] bg-[var(--surface)] text-[var(--text-primary)]'
              }`}
            >
              {msg.role === 'assistant' ? (
                <div className="markdown-body text-sm">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {msg.content || (streaming ? '…' : '')}
                  </ReactMarkdown>
                </div>
              ) : (
                msg.content
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      )}

      {!showHistory && (
        <div className="border-t border-[var(--border)] p-2.5">
          <div className="flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  send()
                }
              }}
              rows={2}
              placeholder={t('interview.tutorPlaceholder')}
              className="min-h-[44px] flex-1 resize-none rounded-xl border border-[var(--border-input)] bg-[var(--bg-input)] px-3 py-2 text-sm focus:border-[var(--accent)] focus:outline-none"
            />
            <button
              type="button"
              onClick={send}
              disabled={streaming || !input.trim()}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[var(--accent)] text-white disabled:opacity-50"
            >
              {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </button>
          </div>
        </div>
      )}

      <div
        className="absolute bottom-0 right-0 h-5 w-5 cursor-nwse-resize"
        onPointerDown={(e) => beginDrag(e, 'resize')}
        title="resize"
      >
        <div className="absolute bottom-1.5 right-1.5 h-2 w-2 border-b-2 border-r-2 border-[var(--text-tertiary)] opacity-70" />
      </div>

      <style>{mascotCss}</style>
    </div>
  )
}

const mascotCss = `
  .tutor-mascot {
    position: relative;
    display: inline-block;
    animation: tutor-float 2.4s ease-in-out infinite;
    filter: drop-shadow(0 8px 14px rgba(52, 211, 153, 0.28));
  }
  .tutor-mascot-shadow {
    position: absolute;
    left: 50%;
    bottom: 2%;
    width: 48%;
    height: 10%;
    transform: translateX(-50%);
    background: radial-gradient(ellipse, rgba(0,0,0,0.16), transparent 70%);
  }
  .tutor-mascot-body {
    position: absolute;
    inset: 18% 22% 12% 22%;
    border-radius: 48% 48% 42% 42%;
    background: linear-gradient(180deg, #f0fbff 0%, #d9f7ff 45%, #c5f0ff 100%);
    border: 1.5px solid rgba(125, 211, 252, 0.7);
    overflow: hidden;
  }
  .tutor-mascot-hair {
    position: absolute;
    left: 8%;
    right: 8%;
    top: -6%;
    height: 42%;
    border-radius: 50% 50% 40% 40%;
    background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
  }
  .tutor-mascot-hair::after {
    content: '';
    position: absolute;
    left: 42%;
    top: -18%;
    width: 18%;
    height: 34%;
    border-radius: 40%;
    background: #7dd3fc;
    transform: rotate(-12deg);
  }
  .tutor-mascot-face {
    position: absolute;
    inset: 34% 18% 28% 18%;
  }
  .tutor-mascot-eye {
    position: absolute;
    top: 28%;
    width: 14%;
    height: 18%;
    border-radius: 50%;
    background: #0f172a;
  }
  .tutor-mascot-eye.l { left: 18%; }
  .tutor-mascot-eye.r { right: 18%; }
  .tutor-mascot-eye::after {
    content: '';
    position: absolute;
    top: 15%;
    right: 15%;
    width: 35%;
    height: 35%;
    border-radius: 50%;
    background: #fff;
  }
  .tutor-mascot-blush {
    position: absolute;
    top: 52%;
    width: 16%;
    height: 10%;
    border-radius: 50%;
    background: rgba(251, 113, 133, 0.35);
  }
  .tutor-mascot-blush.l { left: 8%; }
  .tutor-mascot-blush.r { right: 8%; }
  .tutor-mascot-mouth {
    position: absolute;
    left: 50%;
    top: 68%;
    width: 16%;
    height: 10%;
    border-radius: 0 0 12px 12px;
    border-bottom: 2px solid #fb7185;
    transform: translateX(-50%);
  }
  .tutor-mascot-flower {
    position: absolute;
    top: 10%;
    width: 18%;
    height: 18%;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, #fef08a, #86efac 60%, #4ade80);
    border: 1px solid rgba(74, 222, 128, 0.5);
  }
  .tutor-mascot-flower.l { left: 6%; }
  .tutor-mascot-flower.r { right: 6%; }
  .tutor-mascot-gem {
    position: absolute;
    left: 50%;
    bottom: 14%;
    width: 16%;
    height: 16%;
    transform: translateX(-50%) rotate(45deg);
    border-radius: 4px;
    background: linear-gradient(135deg, #6ee7b7, #34d399);
    box-shadow: 0 0 8px rgba(52, 211, 153, 0.55);
  }
  .tutor-mascot-wing {
    position: absolute;
    top: 34%;
    width: 28%;
    height: 34%;
    border-radius: 60% 40% 60% 40%;
    background: linear-gradient(135deg, rgba(186, 230, 253, 0.95), rgba(167, 243, 208, 0.75));
    border: 1px solid rgba(125, 211, 252, 0.5);
    animation: tutor-wing 1.6s ease-in-out infinite;
  }
  .tutor-mascot-wing-l { left: 4%; transform-origin: right center; }
  .tutor-mascot-wing-r { right: 4%; transform-origin: left center; animation-delay: 0.1s; }
  @keyframes tutor-float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
  }
  @keyframes tutor-wing {
    0%, 100% { transform: scaleY(1) rotate(0deg); }
    50% { transform: scaleY(0.88) rotate(4deg); }
  }
  .tutor-bubble {
    animation: tutor-bubble-in 0.35s ease-out;
  }
  .tutor-bubble-tail {
    position: absolute;
    right: 36px;
    bottom: -6px;
    width: 10px;
    height: 10px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    transform: rotate(45deg);
  }
  @keyframes tutor-bubble-in {
    from { opacity: 0; transform: translateY(6px) scale(0.96); }
    to { opacity: 1; transform: translateY(0) scale(1); }
  }
  .tutor-fab-wrap {
    cursor: grab;
  }
  .tutor-fab-wrap:active {
    cursor: grabbing;
  }
`
