import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { AlertCircle, ExternalLink, FileText, Globe2, Loader2 } from 'lucide-react'
import type { ConfidenceLevel, Source } from '../../api/client'
import { api } from '../../api/client'
import { useHoverPreview } from '../../hooks/useHoverPreview'
import PdfPreviewModal from './PdfPreviewModal'

interface SourceCitationProps {
  sources: Source[]
  confidenceLevel?: ConfidenceLevel
  evidenceSummary?: string
  uncertainPoints?: string[]
}

function buildThumbnailUrl(documentId: string, page: number) {
  return `/documents/${documentId}/page/${page}/thumbnail`
}

function KnowledgeSourceChip({
  source,
  onOpen,
}: {
  source: Source
  onOpen: (source: Source) => void
}) {
  const hover = useHoverPreview({ delay: 500 })
  const [thumbStatus, setThumbStatus] = useState<'loading' | 'loaded' | 'error'>('loading')
  const [blobUrl, setBlobUrl] = useState<string | null>(null)

  const documentId = source.document_id
  const page = source.page ?? 1
  const canPreview = Boolean(documentId && source.page !== undefined)

  useEffect(() => {
    if (!canPreview || !documentId) return

    let cancelled = false
    setBlobUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return null
    })
    setThumbStatus('loading')

    api
      .get(buildThumbnailUrl(documentId, page), { responseType: 'blob' })
      .then((res) => {
        if (cancelled) return
        setBlobUrl(URL.createObjectURL(res.data))
      })
      .catch(() => {
        if (!cancelled) setThumbStatus('error')
      })

    return () => {
      cancelled = true
    }
  }, [canPreview, documentId, page])

  useEffect(() => {
    return () => {
      setBlobUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev)
        return null
      })
    }
  }, [])

  const handleEnter = () => {
    if (!canPreview) return
    hover.onEnter(blobUrl ?? '')
  }

  return (
    <div
      className="relative inline-block"
      onMouseEnter={handleEnter}
      onMouseLeave={hover.onLeave}
    >
      <button
        onClick={() => onOpen(source)}
        className="inline-flex items-center gap-1.5 rounded-full border border-[var(--border)] bg-[var(--surface)] px-3 py-1.5 text-xs text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
      >
        <FileText className="h-3.5 w-3.5" />
        <span className="font-medium">
          {source.document_name ?? '学生手册'} P.{page}
        </span>
      </button>

      <AnimatePresence>
        {hover.visible && canPreview && (
          <motion.div
            className="absolute left-0 top-full z-30 mt-2 rounded-xl bg-[var(--surface)] p-1.5 shadow-xl ring-1 ring-[var(--border)]"
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
          >
            {thumbStatus !== 'loaded' && (
              <div className="flex h-[180px] w-[240px] items-center justify-center text-[var(--text-tertiary)]">
                {thumbStatus === 'loading' ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <div className="flex flex-col items-center gap-1.5 text-center text-[11px]">
                    <AlertCircle className="h-4 w-4" />
                    <span>暂不支持预览，点击查看</span>
                  </div>
                )}
              </div>
            )}
            {blobUrl && (
              <img
                src={blobUrl}
                alt={`${source.document_name ?? '文档'} 第 ${source.page} 页缩略图`}
                onLoad={() => setThumbStatus('loaded')}
                onError={() => setThumbStatus('error')}
                className={`rounded-md shadow-sm transition-opacity duration-200 ${
                  thumbStatus === 'loaded' ? 'block w-[240px] opacity-100' : 'hidden opacity-0'
                }`}
              />
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function WebSourceLink({ source }: { source: Source }) {
  if (!source.url) return null

  return (
    <a
      href={source.url}
      target="_blank"
      rel="noreferrer"
      className="flex min-w-0 items-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-xs text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
    >
      <Globe2 className="h-3.5 w-3.5 shrink-0" />
      <span className="min-w-0 truncate font-medium">{source.title ?? source.site_name ?? source.url}</span>
      <ExternalLink className="h-3.5 w-3.5 shrink-0" />
    </a>
  )
}

const confidenceText: Record<ConfidenceLevel, string> = {
  high: '高',
  medium: '中',
  low: '低',
  unknown: '暂无法判断',
}

export default function SourceCitation({
  sources,
  confidenceLevel,
  evidenceSummary,
  uncertainPoints = [],
}: SourceCitationProps) {
  const [activeSource, setActiveSource] = useState<Source | null>(null)

  if (!sources || sources.length === 0) return null

  const knowledgeSources = sources.filter((source) => source.source_type !== 'web')
  const webSources = sources.filter((source) => source.source_type === 'web')

  return (
    <div className="mt-4 space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <p className="text-xs font-medium text-[var(--text-tertiary)]">参考来源</p>
        {confidenceLevel && <span className="text-xs text-[var(--text-tertiary)]">置信度：{confidenceText[confidenceLevel]}</span>}
      </div>
      {evidenceSummary && <p className="text-xs leading-5 text-[var(--text-tertiary)]">{evidenceSummary}</p>}
      {knowledgeSources.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-xs font-medium text-[var(--text-secondary)]">校内知识库</p>
          <div className="flex flex-wrap gap-2">
            {knowledgeSources.map((source, index) => (
              <KnowledgeSourceChip
                key={`${source.document_id ?? 'unknown'}-${source.page}-${index}`}
                source={source}
                onOpen={setActiveSource}
              />
            ))}
          </div>
        </div>
      )}
      {webSources.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-xs font-medium text-[var(--text-secondary)]">网络来源</p>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {webSources.map((source, index) => <WebSourceLink key={`${source.url}-${index}`} source={source} />)}
          </div>
        </div>
      )}
      {uncertainPoints.length > 0 && (
        <ul className="space-y-1 text-xs leading-5 text-amber-700 dark:text-amber-300">
          {uncertainPoints.map((point) => <li key={point}>{point}</li>)}
        </ul>
      )}
      <PdfPreviewModal source={activeSource} onClose={() => setActiveSource(null)} />
    </div>
  )
}
