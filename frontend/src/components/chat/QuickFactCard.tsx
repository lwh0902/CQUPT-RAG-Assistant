import { ExternalLink, Zap } from 'lucide-react'
import type { QuickFact } from '../../api/client'

/** Compact human-reviewed fact card rendered inside the assistant message flow. */
export default function QuickFactCard({ fact }: { fact: QuickFact }) {
  return (
    <div className="max-w-md rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] px-4 py-3">
      <div className="flex items-center gap-2 text-sm font-medium text-[var(--text-primary)]">
        <Zap className="h-4 w-4 text-[var(--accent)]" />
        {fact.title}
      </div>
      <p className="mt-2 whitespace-pre-line text-sm leading-relaxed text-[var(--text-secondary)]">
        {fact.answer}
      </p>
      <div className="mt-2 flex items-center gap-2 text-xs text-[var(--text-tertiary)]">
        <span>来源：{fact.source_name}</span>
        {fact.updated_at && <span>· {fact.updated_at} 更新</span>}
        {fact.source_url && (
          <a
            href={fact.source_url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-0.5 text-[var(--accent)] hover:underline"
          >
            官方页面
            <ExternalLink className="h-3 w-3" />
          </a>
        )}
      </div>
    </div>
  )
}
