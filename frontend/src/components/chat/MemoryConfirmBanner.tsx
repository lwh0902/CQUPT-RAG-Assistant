import { useChatStore } from '../../store/chat'

/** Inline confirm strip for mid-confidence memory candidates. */
export default function MemoryConfirmBanner() {
  const pending = useChatStore((s) => s.pendingMemoryActions)
  const confirmPendingMemory = useChatStore((s) => s.confirmPendingMemory)
  const rejectPendingMemory = useChatStore((s) => s.rejectPendingMemory)

  if (!pending.length) return null

  return (
    <div className="mb-2 space-y-2">
      {pending.map((item) => {
        const id = item.candidate_id
        if (!id) return null
        return (
          <div
            key={id}
            className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] px-3 py-2.5 text-sm"
          >
            <p className="min-w-0 text-[var(--text-secondary)]">
              {item.message || `是否记住：${item.memory_key}=${item.memory_value}？`}
            </p>
            <div className="flex shrink-0 items-center gap-2">
              <button
                type="button"
                onClick={() => rejectPendingMemory(id)}
                className="rounded-md px-3 py-1.5 text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]"
              >
                忽略
              </button>
              <button
                type="button"
                onClick={() => confirmPendingMemory(id)}
                className="rounded-md bg-[var(--accent)] px-3 py-1.5 text-white hover:opacity-90"
              >
                记住
              </button>
            </div>
          </div>
        )
      })}
    </div>
  )
}
