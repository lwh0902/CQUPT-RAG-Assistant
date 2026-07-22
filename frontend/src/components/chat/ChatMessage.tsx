import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import ThinkingChain from './ThinkingChain'
import SourceCitation from './SourceCitation'
import QuickFactCard from './QuickFactCard'
import type { ThinkingStep } from '../../store/chat'
import { useChatStore } from '../../store/chat'
import { useT } from '../../i18n'
import type { Message, Source } from '../../api/client'
import { Check, Copy, Globe2, RotateCcw, ThumbsDown, ThumbsUp } from 'lucide-react'
import { useState } from 'react'

interface ChatMessageProps {
  message: Message
  thinkingSteps?: ThinkingStep[]
  streamingContent?: string
  streamingSources?: Source[]
  isStreaming?: boolean
}

export default function ChatMessage({
  message,
  thinkingSteps,
  streamingContent,
  streamingSources,
  isStreaming,
}: ChatMessageProps) {
  const t = useT()
  const [copied, setCopied] = useState(false)
  const regenerateLast = useChatStore((state) => state.regenerateLast)
  const retryLastWithWebSearch = useChatStore((state) => state.retryLastWithWebSearch)

  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-[var(--bg-user-bubble)] px-4 py-2.5 text-sm leading-relaxed text-[var(--text-user-bubble)] sm:max-w-[65%]">
          {message.content}
        </div>
      </div>
    )
  }

  const content = isStreaming ? streamingContent ?? '' : message.content
  const sources = isStreaming ? streamingSources ?? [] : message.sources ?? []

  if (!content && (!thinkingSteps || thinkingSteps.length === 0)) return null

  const handleCopy = () => {
    navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="flex gap-3">
      <div className="min-w-0 flex-1">
        {thinkingSteps && thinkingSteps.length > 0 && (
          <ThinkingChain steps={thinkingSteps} />
        )}
        {content && (
          <div>
            {message.quick_fact && !isStreaming ? (
              <QuickFactCard fact={message.quick_fact} />
            ) : (
            <div className="markdown-body text-sm leading-relaxed text-[var(--text-assistant)]">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {content}
              </ReactMarkdown>
              {isStreaming && (
                <span className="inline-block h-4 w-0.5 animate-pulse bg-[var(--text-assistant)] ml-0.5 align-text-bottom" />
              )}
            </div>
            )}
            {!isStreaming && content && (
              <div className="mt-3 flex items-center gap-1 text-[var(--text-tertiary)]">
                <button
                  onClick={handleCopy}
                  className="flex h-8 w-8 items-center justify-center rounded-lg transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
                  aria-label={t('chat.copy')}
                  title="复制"
                >
                  {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </button>
                <button
                  className="flex h-8 w-8 items-center justify-center rounded-lg transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
                  aria-label={t('chat.like')}
                  title={t('chat.like')}
                >
                  <ThumbsUp className="h-4 w-4" />
                </button>
                <button
                  className="flex h-8 w-8 items-center justify-center rounded-lg transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
                  aria-label={t('chat.dislike')}
                  title={t('chat.dislike')}
                >
                  <ThumbsDown className="h-4 w-4" />
                </button>
                <button
                  onClick={regenerateLast}
                  className="flex h-8 w-8 items-center justify-center rounded-lg transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
                  aria-label={t('chat.regenerate')}
                  title={t('chat.regenerate')}
                >
                  <RotateCcw className="h-4 w-4" />
                </button>
              </div>
            )}
          </div>
        )}
        {sources.length > 0 && (
          <SourceCitation
            sources={sources}
            confidenceLevel={isStreaming ? undefined : message.confidence_level}
            evidenceSummary={isStreaming ? undefined : message.evidence_summary}
            uncertainPoints={isStreaming ? undefined : message.uncertain_points}
          />
        )}
        {(message.retrieval_decision === 'out_of_scope' || message.retrieval_decision === 'insufficient') && (
          <div className="mt-3 border-l-2 border-amber-500 px-3 py-2 text-xs text-[var(--text-secondary)]">
            <p>{message.retrieval_decision === 'insufficient' ? t('chat.refusalInsufficient') : t('chat.refusalOutOfScope')} {t('chat.refusalHint')}</p>
            <button onClick={retryLastWithWebSearch} className="mt-2 inline-flex items-center gap-1.5 text-emerald-700 hover:underline dark:text-emerald-300" aria-label={t('chat.retryWithWeb')}>
              <Globe2 className="h-3.5 w-3.5" />{t('chat.retryWithWeb')}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
