import { useEffect, useRef, useState } from 'react'
import { Loader2, Send, X } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { streamInterviewTutor } from '../../api/client'
import { useT } from '../../i18n'

type ChatMsg = { role: 'user' | 'assistant'; content: string }

/**
 * Floating cartoon tutor entry + modal chat.
 * No auto-injected question context — user pastes what they don't understand.
 */
export default function InterviewTutor() {
  const t = useT()
  const [open, setOpen] = useState(false)
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMsg[]>([])
  const [streaming, setStreaming] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<{ abort: () => void } | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, open])

  useEffect(() => {
    return () => abortRef.current?.abort()
  }, [])

  const send = () => {
    const text = input.trim()
    if (!text || streaming) return
    const next: ChatMsg[] = [...messages, { role: 'user', content: text }]
    setMessages([...next, { role: 'assistant', content: '' }])
    setInput('')
    setStreaming(true)

    abortRef.current?.abort()
    abortRef.current = streamInterviewTutor(next, {
      onToken: (token) => {
        setMessages((prev) => {
          const copy = [...prev]
          const last = copy[copy.length - 1]
          if (last?.role === 'assistant') {
            copy[copy.length - 1] = { role: 'assistant', content: last.content + token }
          }
          return copy
        })
      },
      onDone: () => setStreaming(false),
      onError: (message) => {
        setStreaming(false)
        setMessages((prev) => {
          const copy = [...prev]
          const last = copy[copy.length - 1]
          if (last?.role === 'assistant' && !last.content) {
            copy[copy.length - 1] = { role: 'assistant', content: message || t('interview.tutorError') }
          } else {
            copy.push({ role: 'assistant', content: message || t('interview.tutorError') })
          }
          return copy
        })
      },
    })
  }

  return (
    <>
      {/* Floating cartoon entry */}
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="tutor-fab fixed bottom-6 right-5 z-40 flex items-center gap-2 rounded-full border border-[var(--border)] bg-[var(--surface)] px-3 py-2 shadow-lg hover:shadow-xl"
        title={t('interview.tutorTitle')}
      >
        <span className="tutor-bounce text-2xl" aria-hidden>
          🦉
        </span>
        <span className="hidden text-xs font-medium text-[var(--text-secondary)] sm:inline">
          {t('interview.tutorFab')}
        </span>
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/35 p-3 sm:items-center">
          <div className="flex h-[min(640px,88dvh)] w-full max-w-md flex-col overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--bg-primary)] shadow-2xl">
            <header className="flex items-center gap-2 border-b border-[var(--border)] px-4 py-3">
              <span className="text-2xl" aria-hidden>
                🦉
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium">{t('interview.tutorTitle')}</p>
                <p className="truncate text-[11px] text-[var(--text-tertiary)]">{t('interview.tutorHint')}</p>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-lg p-1.5 text-[var(--text-tertiary)] hover:bg-[var(--bg-hover)]"
                aria-label="close"
              >
                <X className="h-4 w-4" />
              </button>
            </header>

            <div className="flex-1 space-y-3 overflow-y-auto px-4 py-3">
              {messages.length === 0 && (
                <div className="rounded-xl bg-[var(--bg-secondary)] px-3 py-3 text-xs leading-relaxed text-[var(--text-secondary)]">
                  {t('interview.tutorWelcome')}
                </div>
              )}
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`max-w-[90%] rounded-2xl px-3 py-2 text-sm leading-relaxed ${
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

            <div className="border-t border-[var(--border)] p-3">
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
                  className="flex h-10 w-10 items-center justify-center rounded-xl bg-[var(--accent)] text-white disabled:opacity-50"
                >
                  {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .tutor-bounce {
          display: inline-block;
          animation: tutor-bob 1.2s ease-in-out infinite;
        }
        @keyframes tutor-bob {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-4px); }
        }
        .tutor-fab {
          transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .tutor-fab:hover {
          transform: translateY(-2px) scale(1.02);
        }
      `}</style>
    </>
  )
}
