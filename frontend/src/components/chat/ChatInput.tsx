import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { ArrowUp, Paperclip, Square } from 'lucide-react'
import { useT } from '../../i18n'

interface ChatInputProps {
  onSend: (message: string) => void
  isStreaming: boolean
  onStop: () => void
}

export default function ChatInput({ onSend, isStreaming, onStop }: ChatInputProps) {
  const t = useT()
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    textareaRef.current?.focus()
  }, [isStreaming])

  useEffect(() => {
    if (!textareaRef.current) return

    textareaRef.current.style.height = 'auto'
    const scrollHeight = textareaRef.current.scrollHeight
    const maxHeight = 5 * 24
    textareaRef.current.style.height = `${Math.min(scrollHeight, maxHeight)}px`
  }, [value])

  const handleSubmit = () => {
    const trimmed = value.trim()
    if (!trimmed || isStreaming) return

    onSend(trimmed)
    setValue('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSubmit()
    }
  }

  const canSend = value.trim().length > 0 && !isStreaming

  return (
    <div data-testid="chat-composer" className="rounded-[28px] border border-[var(--border-input)] bg-[var(--bg-input)] px-3 py-2 shadow-[0_12px_32px_rgba(0,0,0,0.08)] transition-colors focus-within:border-[var(--text-tertiary)]">
      <div className="flex items-end gap-2">
        <button
          type="button"
          className="mb-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
          aria-label={t('chat.attachDoc')}
          title={t('chat.attachDoc')}
        >
          <Paperclip className="h-5 w-5" />
        </button>

        <textarea
          ref={textareaRef}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={t('chat.inputPlaceholder')}
          rows={1}
          disabled={isStreaming}
          className="max-h-32 min-h-9 flex-1 resize-none bg-transparent py-2 text-base leading-6 text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)] focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
        />

        {isStreaming ? (
          <button
            type="button"
            onClick={onStop}
            className="mb-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-[var(--text-primary)] text-[var(--bg-primary)] transition-opacity hover:opacity-85"
            aria-label={t('chat.stop')}
            title={t('chat.stop')}
          >
            <Square className="h-3.5 w-3.5 fill-current" />
          </button>
        ) : (
          <button
            type="button"
            onClick={handleSubmit}
            disabled={!canSend}
            className="mb-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-[var(--text-primary)] text-[var(--bg-primary)] transition-all hover:opacity-85 disabled:cursor-not-allowed disabled:bg-[var(--bg-tertiary)] disabled:text-[var(--text-tertiary)]"
            aria-label={t('chat.send')}
            title={t('chat.send')}
          >
            <ArrowUp className="h-5 w-5" />
          </button>
        )}
      </div>
    </div>
  )
}
