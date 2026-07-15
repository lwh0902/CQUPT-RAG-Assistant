import { useEffect, useState } from 'react'
import { AlertCircle, FileText, FileWarning, Loader2 } from 'lucide-react'
import Modal from '../ui/Modal'
import PdfPreviewModal from '../chat/PdfPreviewModal'
import { listKnowledgeDocuments } from '../../api/client'
import type { KnowledgeDocument, Source } from '../../api/client'

function formatFileSize(bytes: number): string {
  if (!bytes) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

const TYPE_LABEL: Record<string, string> = {
  manual: '学生手册',
  school_policy: '校级制度',
  evaluation_policy: '测评制度',
  special_policy: '专项制度',
}

interface KnowledgeBaseModalProps {
  open: boolean
  onClose: () => void
}

export default function KnowledgeBaseModal({ open, onClose }: KnowledgeBaseModalProps) {
  const [docs, setDocs] = useState<KnowledgeDocument[]>([])
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading')
  const [activeSource, setActiveSource] = useState<Source | null>(null)
  const [activeTotalPages, setActiveTotalPages] = useState<number | null>(null)

  useEffect(() => {
    if (!open) return
    let cancelled = false
    setStatus('loading')
    listKnowledgeDocuments()
      .then((res) => {
        if (cancelled) return
        setDocs(res.documents)
        setStatus('success')
      })
      .catch(() => {
        if (!cancelled) setStatus('error')
      })
    return () => {
      cancelled = true
    }
  }, [open])

  return (
    <>
      <Modal open={open} onClose={onClose} title={<span>知识库资料</span>}>
        <div className="min-h-[50vh] w-[min(720px,90vw)]">
          {status === 'loading' && (
            <div className="flex h-64 items-center justify-center text-[var(--text-tertiary)]">
              <Loader2 className="h-6 w-6 animate-spin" />
            </div>
          )}
          {status === 'error' && (
            <div className="flex h-64 flex-col items-center justify-center gap-2 text-sm text-[var(--text-tertiary)]">
              <AlertCircle className="h-6 w-6" />
              <p>资料列表加载失败，请稍后重试。</p>
            </div>
          )}
          {status === 'success' && (
            <div className="max-h-[70vh] overflow-y-auto p-2">
              <p className="px-3 py-2 text-xs text-[var(--text-tertiary)]">
                共 {docs.length} 份资料 · 点击 PDF 类资料可预览首页
              </p>
              <ul className="space-y-1">
                {docs.map((doc) => {
                  const Icon = doc.previewable ? FileText : FileWarning
                  return (
                    <li key={doc.document_id}>
                      <button
                        onClick={() =>
                          doc.previewable &&
                          (setActiveTotalPages(doc.page_count),
                          setActiveSource({
                            document_id: doc.document_id,
                            document_name: doc.document_name,
                            page: 1,
                          }))
                        }
                        disabled={!doc.previewable}
                        className={`flex w-full items-start gap-3 rounded-xl border border-transparent px-3 py-3 text-left transition-colors ${
                          doc.previewable
                            ? 'hover:border-[var(--border)] hover:bg-[var(--bg-hover)]'
                            : 'cursor-not-allowed opacity-60'
                        }`}
                      >
                        <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[var(--bg-secondary)] text-[var(--text-secondary)]">
                          <Icon className="h-4 w-4" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-sm font-medium text-[var(--text-primary)]">
                            {doc.document_name}
                          </div>
                          <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-[var(--text-tertiary)]">
                            <span>{TYPE_LABEL[doc.document_type] ?? doc.document_type}</span>
                            {doc.page_count ? <span>{doc.page_count} 页</span> : null}
                            <span>{formatFileSize(doc.file_size)}</span>
                            {!doc.previewable && <span>· 暂不支持预览</span>}
                          </div>
                        </div>
                      </button>
                    </li>
                  )
                })}
              </ul>
            </div>
          )}
        </div>
      </Modal>
      <PdfPreviewModal
        source={activeSource}
        onClose={() => {
          setActiveSource(null)
          setActiveTotalPages(null)
        }}
        navigable
        totalPages={activeTotalPages}
      />
    </>
  )
}
