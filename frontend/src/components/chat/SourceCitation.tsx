import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { AlertCircle, FileText, Loader2 } from 'lucide-react'
import type { Source } from '../../api/client'
import { api } from '../../api/client'
import { useHoverPreview } from '../../hooks/useHoverPreview'
import PdfPreviewModal from './PdfPreviewModal'

interface SourceCitationProps {
  sources: Source[]
}

function buildThumbnailUrl(documentId: string, page: number) {
  return `/documents/${documentId}/page/${page}/thumbnail`
}

function SourceChip({
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
  const canPreview = Boolean(documentId)

  useEffect(() => {
    if (!canPreview || !documentId) return

    let cancelled = false
    setBlobUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return null
    })
    setThumbStatus('loading')

    api
      .get(buildThumbnailUrl(documentId, source.page), { responseType: 'blob' })
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
  }, [canPreview, documentId, source.page])

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
          {source.document_name ?? '学生手册'} P.{source.page}
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

export default function SourceCitation({ sources }: SourceCitationProps) {
  const [activeSource, setActiveSource] = useState<Source | null>(null)

  if (!sources || sources.length === 0) return null

  return (
    <div className="mt-4 space-y-2">
      <p className="text-xs font-medium text-[var(--text-tertiary)]">参考来源</p>
      <div className="flex flex-wrap gap-2">
        {sources.map((source, index) => (
          <SourceChip
            key={`${source.document_id ?? 'unknown'}-${source.page}-${index}`}
            source={source}
            onOpen={setActiveSource}
          />
        ))}
      </div>
      <PdfPreviewModal source={activeSource} onClose={() => setActiveSource(null)} />
    </div>
  )
}
