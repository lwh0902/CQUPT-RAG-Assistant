import { ListChecks } from 'lucide-react'
import { useState } from 'react'
import { api } from '../../api/client'
import { useToastStore } from '../../store/toast'
import Modal from '../ui/Modal'

interface SummaryData {
  topic: string
  confirmed_points: string[]
  open_questions: string[]
  next_actions: string[]
}

interface ConversationSummaryProps {
  sessionId: string | null
  messageCount: number
}

export default function ConversationSummary({ sessionId, messageCount }: ConversationSummaryProps) {
  const [summary, setSummary] = useState<SummaryData | null>(null)
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  if (!sessionId || messageCount < 2) return null

  const generate = async () => {
    setLoading(true)
    setError('')
    try {
      const { data } = await api.post<{ summary: SummaryData }>(`/sessions/${sessionId}/summary`)
      setSummary(data.summary)
      setOpen(true)
    } catch {
      const message = '生成对话总结失败，请稍后重试'
      setError(message)
      useToastStore.getState().addToast(message, 'error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <button
        type="button"
        onClick={generate}
        disabled={loading}
        className="flex h-9 w-9 items-center justify-center rounded-lg text-[var(--text-tertiary)] transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)] disabled:opacity-50"
        aria-label="总结当前对话"
        title="总结当前对话"
      >
        <ListChecks className="h-4 w-4" />
      </button>
      {error && <span role="status" className="max-w-40 truncate text-xs text-red-500">{error}</span>}
      <Modal open={open} onClose={() => setOpen(false)} title="当前对话总结" className="summary-modal max-w-lg">
        {summary && (
          <div className="space-y-5 p-5 text-sm text-[var(--text-secondary)]">
            <section><h3 className="font-medium text-[var(--text-primary)]">{summary.topic}</h3></section>
            <SummaryList title="已确认" values={summary.confirmed_points} />
            <SummaryList title="未决问题" values={summary.open_questions} />
            <SummaryList title="下一步" values={summary.next_actions} />
          </div>
        )}
      </Modal>
    </>
  )
}

function SummaryList({ title, values }: { title: string; values: string[] }) {
  if (values.length === 0) return null
  return (
    <section className="space-y-1.5">
      <h4 className="font-medium text-[var(--text-primary)]">{title}</h4>
      <ul className="space-y-1 leading-6">
        {values.map((value) => <li key={value}>{value}</li>)}
      </ul>
    </section>
  )
}
