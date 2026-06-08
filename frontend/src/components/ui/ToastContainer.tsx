import { X, AlertCircle, CheckCircle2, Info } from 'lucide-react'
import { useToastStore } from '../../store/toast'
import type { Toast as ToastType } from '../../store/toast'
import { AnimatePresence, motion } from 'framer-motion'

function ToastItem({ toast }: { toast: ToastType }) {
  const { removeToast } = useToastStore()

  const icon = {
    error: <AlertCircle className="h-4 w-4 text-red-400" />,
    success: <CheckCircle2 className="h-4 w-4 text-emerald-400" />,
    info: <Info className="h-4 w-4 text-indigo-400" />,
  }[toast.type]

  return (
    <motion.div
      initial={{ opacity: 0, y: -12, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -12, scale: 0.95 }}
      className="flex w-full max-w-sm items-center gap-2.5 rounded-xl bg-[var(--surface-raised)] px-4 py-3 text-sm text-[var(--text-primary)] shadow-lg ring-1 ring-[var(--border)]"
    >
      {icon}
      <span className="flex-1">{toast.message}</span>
      <button
        onClick={() => removeToast(toast.id)}
        className="shrink-0 text-[var(--text-tertiary)] transition-colors hover:text-[var(--text-secondary)]"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </motion.div>
  )
}

export default function ToastContainer() {
  const { toasts } = useToastStore()

  return (
    <div className="pointer-events-none fixed inset-x-0 top-4 z-50 flex flex-col items-center gap-2 px-4">
      <AnimatePresence>
        {toasts.map((toast) => (
          <div key={toast.id} className="pointer-events-auto">
            <ToastItem toast={toast} />
          </div>
        ))}
      </AnimatePresence>
    </div>
  )
}
