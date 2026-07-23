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
const STORAGE_UI = 'interview_tutor_ui_v2'

type UiState = {
  fabX: number
  fabY: number
  panelX: number
  panelY: number
  w: number
  h: number
  open: boolean
}

type ResizeCorner = 'nw' | 'ne' | 'sw' | 'se'

const DEFAULT_SIZE = { w: 380, h: 500 }
const MIN_SIZE = { w: 300, h: 360 }
const MAX_SIZE = { w: 640, h: 820 }
const FAB_SIZE = 148

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
  const margin = 20
  const w = DEFAULT_SIZE.w
  const h = DEFAULT_SIZE.h
  const fabX = Math.max(margin, window.innerWidth - FAB_SIZE - margin)
  const fabY = Math.max(margin, window.innerHeight - FAB_SIZE - margin - 16)
  return {
    fabX,
    fabY,
    panelX: Math.max(margin, fabX + FAB_SIZE - w),
    panelY: Math.max(margin, fabY - h + 56),
    w,
    h,
    open: false,
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

function titleFromMessages(messages: ChatMsg[]): string {
  const firstUser = messages.find((m) => m.role === 'user')?.content?.trim()
  if (!firstUser) return '新对话'
  return firstUser.replace(/\s+/g, ' ').slice(0, 24)
}

function Fairy({ size = 120, className = '' }: { size?: number; className?: string }) {
  return (
    <img
      src="/tutor-fairy.svg"
      alt=""
      draggable={false}
      width={size}
      height={size}
      className={`select-none ${className}`}
      style={{ width: size, height: size, objectFit: 'contain' }}
    />
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
    mode: 'panel' | 'fab' | ResizeCorner
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
      const panelX = Math.min(
        Math.max(8, prev.fabX + FAB_SIZE - prev.w),
        Math.max(8, window.innerWidth - prev.w - 8),
      )
      const panelY = Math.min(
        Math.max(8, prev.fabY - prev.h + 64),
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
        return
      }

      if (drag.mode === 'panel') {
        setUi((prev) =>
          clampUi({
            ...prev,
            panelX: drag.orig.panelX + dx,
            panelY: drag.orig.panelY + dy,
          }),
        )
        return
      }

      // 4-corner resize
      const o = drag.orig
      let nextX = o.panelX
      let nextY = o.panelY
      let nextW = o.w
      let nextH = o.h

      if (drag.mode === 'se') {
        nextW = o.w + dx
        nextH = o.h + dy
      } else if (drag.mode === 'sw') {
        nextW = o.w - dx
        nextH = o.h + dy
        nextX = o.panelX + dx
      } else if (drag.mode === 'ne') {
        nextW = o.w + dx
        nextH = o.h - dy
        nextY = o.panelY + dy
      } else if (drag.mode === 'nw') {
        nextW = o.w - dx
        nextH = o.h - dy
        nextX = o.panelX + dx
        nextY = o.panelY + dy
      }

      // Keep opposite edge anchored when hitting min/max.
      const clampedW = Math.min(MAX_SIZE.w, Math.max(MIN_SIZE.w, nextW))
      const clampedH = Math.min(MAX_SIZE.h, Math.max(MIN_SIZE.h, nextH))
      if (drag.mode === 'sw' || drag.mode === 'nw') {
        nextX = o.panelX + (o.w - clampedW)
      }
      if (drag.mode === 'ne' || drag.mode === 'nw') {
        nextY = o.panelY + (o.h - clampedH)
      }

      setUi((prev) =>
        clampUi({
          ...prev,
          panelX: nextX,
          panelY: nextY,
          w: clampedW,
          h: clampedH,
        }),
      )
    }

    const onUp = () => {
      const drag = dragRef.current
      dragRef.current = null
      document.body.style.userSelect = ''
      document.body.style.cursor = ''
      if (drag?.mode === 'fab' && !drag.moved) openPanel()
    }

    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
  }, [openPanel])

  const beginDrag = (e: React.PointerEvent, mode: 'panel' | 'fab' | ResizeCorner) => {
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
    document.body.style.cursor =
      mode === 'panel' || mode === 'fab'
        ? 'grabbing'
        : mode === 'nw' || mode === 'se'
          ? 'nwse-resize'
          : 'nesw-resize'
  }

  const cornerClass =
    'absolute z-10 h-4 w-4 rounded-sm bg-transparent hover:bg-[var(--accent)]/20'

  if (!ui.open) {
    return (
      <div className="fixed z-40 touch-none" style={{ left: ui.fabX, top: ui.fabY }}>
        {bubbleVisible && (
          <button
            type="button"
            onClick={openPanel}
            className="tutor-bubble absolute -top-2 right-0 w-[168px] -translate-y-full rounded-2xl border border-[var(--border)] bg-[var(--surface)] px-3 py-2.5 text-left text-xs leading-snug text-[var(--text-secondary)] shadow-lg"
          >
            {t('interview.tutorBubble')}
            <span className="tutor-bubble-tail" />
          </button>
        )}
        <button
          type="button"
          onPointerDown={(e) => beginDrag(e, 'fab')}
          className="tutor-fab-wrap relative flex w-[148px] flex-col items-center"
          title={t('interview.tutorTitle')}
        >
          <span className="tutor-float inline-flex">
            <Fairy size={132} />
          </span>
          <span className="mt-[-4px] rounded-full border border-[var(--border)] bg-[var(--surface)]/95 px-3 py-1 text-[11px] font-medium text-[var(--text-secondary)] shadow">
            {t('interview.tutorFab')}
          </span>
        </button>
        <style>{floatCss}</style>
      </div>
    )
  }

  return (
    <div
      className="fixed z-50 flex flex-col overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--bg-primary)] shadow-2xl"
      style={{ left: ui.panelX, top: ui.panelY, width: ui.w, height: ui.h }}
    >
      <header
        className="flex cursor-grab items-center gap-2 border-b border-[var(--border)] bg-[var(--surface)]/90 px-3 py-2 active:cursor-grabbing"
        onPointerDown={(e) => beginDrag(e, 'panel')}
      >
        <Fairy size={40} />
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
              <Fairy size={52} className="shrink-0" />
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

      {/* four-corner resize handles */}
      <div
        className={`${cornerClass} left-0 top-0 cursor-nwse-resize`}
        onPointerDown={(e) => beginDrag(e, 'nw')}
      />
      <div
        className={`${cornerClass} right-0 top-0 cursor-nesw-resize`}
        onPointerDown={(e) => beginDrag(e, 'ne')}
      />
      <div
        className={`${cornerClass} bottom-0 left-0 cursor-nesw-resize`}
        onPointerDown={(e) => beginDrag(e, 'sw')}
      />
      <div
        className={`${cornerClass} bottom-0 right-0 cursor-nwse-resize`}
        onPointerDown={(e) => beginDrag(e, 'se')}
      >
        <div className="absolute bottom-1 right-1 h-2 w-2 border-b-2 border-r-2 border-[var(--text-tertiary)] opacity-70" />
      </div>

      <style>{floatCss}</style>
    </div>
  )
}

const floatCss = `
  .tutor-float {
    animation: tutor-float 2.6s ease-in-out infinite;
    filter: drop-shadow(0 10px 16px rgba(56, 189, 248, 0.28));
  }
  @keyframes tutor-float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
  }
  .tutor-bubble {
    animation: tutor-bubble-in 0.35s ease-out;
  }
  .tutor-bubble-tail {
    position: absolute;
    right: 42px;
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
