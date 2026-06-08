import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import ThinkingChain from './ThinkingChain'
import SourceCitation from './SourceCitation'
import type { ThinkingStep } from '../../store/chat'
import { useChatStore } from '../../store/chat'
import type { Message, Source } from '../../api/client'
import { Check, Copy, RotateCcw, ThumbsDown, ThumbsUp } from 'lucide-react'
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
  const [copied, setCopied] = useState(false)
  const regenerateLast = useChatStore((state) => state.regenerateLast)

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
            <div className="markdown-body text-sm leading-relaxed text-[var(--text-assistant)]">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {content}
              </ReactMarkdown>
              {isStreaming && (
                <span className="inline-block h-4 w-0.5 animate-pulse bg-[var(--text-assistant)] ml-0.5 align-text-bottom" />
              )}
            </div>
            {!isStreaming && content && (
              <div className="mt-3 flex items-center gap-1 text-[var(--text-tertiary)]">
                <button
                  onClick={handleCopy}
                  className="flex h-8 w-8 items-center justify-center rounded-lg transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
                  aria-label="复制回答"
                  title="复制"
                >
                  {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </button>
                <button
                  className="flex h-8 w-8 items-center justify-center rounded-lg transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
                  aria-label="点赞"
                  title="点赞"
                >
                  <ThumbsUp className="h-4 w-4" />
                </button>
                <button
                  className="flex h-8 w-8 items-center justify-center rounded-lg transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
                  aria-label="点踩"
                  title="点踩"
                >
                  <ThumbsDown className="h-4 w-4" />
                </button>
                <button
                  onClick={regenerateLast}
                  className="flex h-8 w-8 items-center justify-center rounded-lg transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
                  aria-label="重新生成"
                  title="重新生成"
                >
                  <RotateCcw className="h-4 w-4" />
                </button>
              </div>
            )}
          </div>
        )}
        {sources.length > 0 && <SourceCitation sources={sources} />}
      </div>
    </div>
  )
}
