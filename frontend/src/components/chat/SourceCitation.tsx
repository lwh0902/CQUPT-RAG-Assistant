import { useState } from 'react'
import { ChevronDown, FileText } from 'lucide-react'
import type { Source } from '../../api/client'

interface SourceCitationProps {
  sources: Source[]
}

export default function SourceCitation({ sources }: SourceCitationProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null)

  if (!sources || sources.length === 0) return null

  return (
    <div className="mt-4 space-y-2">
      <p className="text-xs font-medium text-[var(--text-tertiary)]">参考来源</p>
      <div className="flex flex-wrap gap-2">
        {sources.map((source, index) => {
          const expanded = expandedIndex === index
          return (
            <button
              key={`${source.page}-${index}`}
              onClick={() => setExpandedIndex(expanded ? null : index)}
              className="inline-flex items-center gap-1.5 rounded-full border border-[var(--border)] bg-[var(--surface)] px-3 py-1.5 text-xs text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
            >
              <FileText className="h-3.5 w-3.5" />
              <span className="font-medium">{source.document_name ?? '学生手册'} P.{source.page}</span>
              <ChevronDown className={`h-3.5 w-3.5 transition-transform ${expanded ? 'rotate-180' : ''}`} />
            </button>
          )
        })}
      </div>

      {expandedIndex !== null && (
        <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-secondary)] px-4 py-3 text-sm leading-relaxed text-[var(--text-secondary)]">
          <div className="mb-1 text-xs font-medium text-[var(--text-tertiary)]">
            {sources[expandedIndex].document_name ?? '学生手册'} · 第 {sources[expandedIndex].page} 页
          </div>
          {sources[expandedIndex].snippet || sources[expandedIndex].preview || '这条历史消息没有保存原文片段。'}
        </div>
      )}
    </div>
  )
}
