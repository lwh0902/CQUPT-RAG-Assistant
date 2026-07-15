import { useEffect, useState } from 'react'
import { AlertCircle, ChevronLeft, ChevronRight, Loader2 } from 'lucide-react'
import Modal from '../ui/Modal'
import { api } from '../../api/client'
import type { Source } from '../../api/client'

interface PdfPreviewModalProps {
  source: Source | null
  onClose: () => void
  navigable?: boolean
  totalPages?: number | null
}

function buildPageUrl(documentId: string, page: number) {
  return `/documents/${documentId}/page/${page}`
}

export default function PdfPreviewModal({
  source,
  onClose,
  navigable = false,
  totalPages,
}: PdfPreviewModalProps) {
  const [imgStatus, setImgStatus] = useState<'loading' | 'loaded' | 'error'>('loading')
  const [blobUrl, setBlobUrl] = useState<string | null>(null)
  const [currentPage, setCurrentPage] = useState<number>(source?.page ?? 1)
  const [pageInput, setPageInput] = useState<string>(String(source?.page ?? 1))

  const documentId = source?.document_id
  const canPreview = Boolean(documentId)
  const effectivePage = navigable ? currentPage : source?.page ?? 1
  const total = navigable ? (totalPages ?? 0) : null

  useEffect(() => {
    if (source) {
      setCurrentPage(source.page)
      setPageInput(String(source.page))
    }
  }, [source])

  useEffect(() => {
    setBlobUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return null
    })
    setImgStatus('loading')

    if (!canPreview || !source || !documentId) {
      return
    }

    let cancelled = false
    api
      .get(buildPageUrl(documentId, effectivePage), { responseType: 'blob' })
      .then((res) => {
        if (cancelled) return
        const url = URL.createObjectURL(res.data)
        setBlobUrl(url)
      })
      .catch(() => {
        if (!cancelled) setImgStatus('error')
      })

    return () => {
      cancelled = true
    }
  }, [source, canPreview, documentId, effectivePage])

  useEffect(() => {
    return () => {
      setBlobUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev)
        return null
      })
    }
  }, [])

  if (!source) return null

  const handleImageLoad = () => setImgStatus('loaded')
  const handleImageError = () => setImgStatus('error')

  const goToPage = (raw: string) => {
    const num = parseInt(raw, 10)
    if (!Number.isFinite(num) || num < 1) return
    const max = total ?? Number.MAX_SAFE_INTEGER
    const clamped = Math.min(num, max)
    setCurrentPage(clamped)
    setPageInput(String(clamped))
  }

  const goPrev = () => {
    if (currentPage > 1) {
      const next = currentPage - 1
      setCurrentPage(next)
      setPageInput(String(next))
    }
  }
  const goNext = () => {
    if (total && currentPage >= total) return
    const next = currentPage + 1
    setCurrentPage(next)
    setPageInput(String(next))
  }

  const showSnippet = !navigable

  return (
    <Modal
      open={source !== null}
      onClose={onClose}
      title={
        <span>
          {source.document_name ?? '学生手册'} · 第 {effectivePage} 页
          {total ? ` / ${total}` : ''}
        </span>
      }
    >
      <div
        className={`grid min-h-[60vh] grid-cols-1 ${
          showSnippet ? 'md:grid-cols-[1.4fr_1fr]' : ''
        }`}
      >
        <div className="relative flex items-center justify-center overflow-auto bg-[var(--bg-secondary)] p-4">
          {!canPreview ? (
            <div className="flex flex-col items-center gap-2 px-6 py-12 text-center text-sm text-[var(--text-tertiary)]">
              <AlertCircle className="h-6 w-6" />
              <p>该文档类型暂不支持原页预览{showSnippet ? '，请参考右侧片段' : ''}。</p>
            </div>
          ) : (
            <>
              {imgStatus === 'loading' && (
                <div className="absolute inset-0 flex items-center justify-center text-[var(--text-tertiary)]">
                  <Loader2 className="h-6 w-6 animate-spin" />
                </div>
              )}
              {imgStatus === 'error' && (
                <div className="flex flex-col items-center gap-2 px-6 py-12 text-center text-sm text-[var(--text-tertiary)]">
                  <AlertCircle className="h-6 w-6" />
                  <p>原页加载失败，请稍后重试。</p>
                </div>
              )}
              {blobUrl && (
                <img
                  key={blobUrl}
                  src={blobUrl}
                  alt={`${source.document_name ?? '文档'} 第 ${effectivePage} 页`}
                  onLoad={handleImageLoad}
                  onError={handleImageError}
                  className={`max-h-[80vh] w-auto max-w-full rounded-md shadow-md transition-opacity duration-200 ${
                    imgStatus === 'loaded' ? 'opacity-100' : 'opacity-0'
                  }`}
                />
              )}
            </>
          )}
        </div>
        {showSnippet && (
          <div className="border-t border-[var(--border)] p-5 md:border-l md:border-t-0">
            <div className="mb-2 text-xs font-medium uppercase tracking-wide text-[var(--text-tertiary)]">
              原文片段
            </div>
            <div className="text-sm leading-relaxed text-[var(--text-secondary)]">
              {source.snippet || source.preview || '该来源未保存原文片段。'}
            </div>
          </div>
        )}
      </div>

      {navigable && total ? (
        <div className="flex items-center justify-center gap-3 border-t border-[var(--border)] px-4 py-3 text-sm">
          <button
            onClick={goPrev}
            disabled={currentPage <= 1}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-[var(--border)] text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-hover)] disabled:cursor-not-allowed disabled:opacity-40"
            aria-label="上一页"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <div className="flex items-center gap-2 text-[var(--text-secondary)]">
            <input
              value={pageInput}
              onChange={(e) => setPageInput(e.target.value.replace(/\D/g, ''))}
              onBlur={(e) => goToPage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  goToPage((e.target as HTMLInputElement).value)
                }
              }}
              className="w-12 rounded-md border border-[var(--border-input)] bg-[var(--bg-input)] px-2 py-1 text-center text-sm text-[var(--text-primary)] focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
            />
            <span>/ {total} 页</span>
          </div>
          <button
            onClick={goNext}
            disabled={currentPage >= total}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-[var(--border)] text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-hover)] disabled:cursor-not-allowed disabled:opacity-40"
            aria-label="下一页"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      ) : null}
    </Modal>
  )
}
