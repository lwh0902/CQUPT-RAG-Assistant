import { Brain, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { api } from '../../api/client'
import { useToastStore } from '../../store/toast'
import Modal from '../ui/Modal'
import { useT } from '../../i18n'

interface MemoryItem {
  id: number
  memory_type: 'preference' | 'profile' | string
  memory_key: string
  memory_value: string
  confidence: number
}

interface PendingItem {
  id: string
  memory_type?: string
  memory_key?: string
  memory_value?: string
  confidence?: number
  reason?: string
  status?: string
}

const GROUP_LABELS: Record<string, string> = {
  preference: '回答偏好',
  profile: '个人资料',
  goal: '目标',
  constraint: '约束',
}

export default function MemoryManagerModal() {
  const t = useT()
  const [open, setOpen] = useState(false)
  const [memories, setMemories] = useState<MemoryItem[]>([])
  const [pending, setPending] = useState<PendingItem[]>([])
  const [loaded, setLoaded] = useState(false)
  const [pendingDeletion, setPendingDeletion] = useState<MemoryItem | null>(null)
  const [loadError, setLoadError] = useState(false)

  const loadMemories = async () => {
    setLoadError(false)
    try {
      const { data } = await api.get<{ memories: MemoryItem[]; pending?: PendingItem[] }>('/memories')
      setMemories(data.memories)
      setPending(data.pending ?? [])
      setLoaded(true)
    } catch {
      setLoadError(true)
      useToastStore.getState().addToast('加载记忆失败，请重试', 'error')
    }
  }

  const openModal = async () => {
    setOpen(true)
    if (loaded) return
    await loadMemories()
  }

  const confirmDeletion = async () => {
    if (!pendingDeletion) return
    const memory = pendingDeletion
    try {
      const { data } = await api.delete<{ id: number; status: string }>(`/memories/${memory.id}`)
      if (data.status === 'deleted') {
        setMemories((current) => current.filter((item) => item.id !== memory.id))
      } else {
        useToastStore.getState().addToast('记忆不存在或已被删除', 'info')
      }
    } catch {
      useToastStore.getState().addToast('删除记忆失败，请重试', 'error')
    } finally {
      setPendingDeletion(null)
    }
  }

  const confirmCandidate = async (candidateId: string) => {
    try {
      const { data } = await api.post<{ status: string; message?: string }>(
        `/memories/candidates/${candidateId}/confirm`,
      )
      if (data.status === 'confirmed') {
        useToastStore.getState().addToast(data.message || '已记住', 'success')
        setPending((current) => current.filter((item) => item.id !== candidateId))
        setLoaded(false)
        await loadMemories()
      } else {
        useToastStore.getState().addToast('确认失败，请稍后重试', 'info')
      }
    } catch {
      useToastStore.getState().addToast('确认记忆失败', 'error')
    }
  }

  const rejectCandidate = async (candidateId: string) => {
    try {
      await api.post(`/memories/candidates/${candidateId}/reject`)
      setPending((current) => current.filter((item) => item.id !== candidateId))
    } catch {
      useToastStore.getState().addToast('忽略记忆失败', 'error')
    }
  }

  const groupedMemories = memories.reduce<Record<string, MemoryItem[]>>((groups, memory) => {
    const group = groups[memory.memory_type] ?? []
    group.push(memory)
    groups[memory.memory_type] = group
    return groups
  }, {})

  const isEmpty = memories.length === 0 && pending.length === 0

  return (
    <>
      <button
        type="button"
        onClick={openModal}
        className="flex h-9 w-9 items-center justify-center rounded-lg text-[var(--text-tertiary)] transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
        aria-label={t('memory.manage')}
        title={t('memory.manage')}
      >
        <Brain className="h-4 w-4" />
      </button>
      <Modal open={open} onClose={() => { setOpen(false); setPendingDeletion(null) }} title="记忆管理" className="max-w-lg">
        <div className="space-y-5 p-5">
          {loadError ? (
            <div className="space-y-3 py-8 text-center">
              <p className="text-sm text-red-500">加载记忆失败，请重试</p>
              <button type="button" onClick={loadMemories} className="rounded-md px-3 py-1.5 text-sm text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]" aria-label="重试加载记忆">重试</button>
            </div>
          ) : isEmpty ? (
            <div className="space-y-3 py-6 text-center text-sm text-[var(--text-tertiary)]">
              <p className="font-medium text-[var(--text-secondary)]">暂无已保存的长期记忆</p>
              <p className="mx-auto max-w-sm leading-relaxed">
                目前只会在你明确表达时尝试保存，例如「我是某某专业」「以后请简洁回答」。
                中等把握的内容会先征求你确认。不会自动记住全部聊天。可随时在此删除。
              </p>
            </div>
          ) : (
            <>
              <p className="text-xs leading-relaxed text-[var(--text-tertiary)]">
                仅保存你明确表达的偏好/档案信息；待确认项不会注入回答。删除后不再注入。
              </p>

              {pending.length > 0 && (
                <section className="space-y-2">
                  <h3 className="text-sm font-medium text-[var(--text-primary)]">待确认</h3>
                  <ul className="space-y-2">
                    {pending.map((item) => (
                      <li
                        key={item.id}
                        className="flex min-w-0 flex-wrap items-center justify-between gap-3 rounded-lg border border-dashed border-[var(--border)] px-3 py-2.5 text-sm"
                      >
                        <span className="min-w-0 truncate text-[var(--text-secondary)]">
                          {item.memory_key}={item.memory_value}
                        </span>
                        <div className="flex shrink-0 items-center gap-2">
                          <button
                            type="button"
                            onClick={() => rejectCandidate(item.id)}
                            className="rounded-md px-2 py-1 text-xs text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]"
                          >
                            忽略
                          </button>
                          <button
                            type="button"
                            onClick={() => confirmCandidate(item.id)}
                            className="rounded-md bg-[var(--accent)] px-2 py-1 text-xs text-white"
                          >
                            记住
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              {Object.entries(groupedMemories).map(([type, items]) => (
                <section key={type} className="space-y-2">
                  <h3 className="text-sm font-medium text-[var(--text-primary)]">{GROUP_LABELS[type] ?? '其他记忆'}</h3>
                  <ul className="space-y-2">
                    {items.map((memory) => (
                      <li key={memory.id} className="flex min-w-0 items-center justify-between gap-3 rounded-lg border border-[var(--border)] px-3 py-2.5 text-sm">
                        <span className="min-w-0 truncate text-[var(--text-secondary)]">{memory.memory_value}</span>
                        <button
                          type="button"
                          onClick={() => setPendingDeletion(memory)}
                          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-[var(--text-tertiary)] transition-colors hover:bg-red-500/10 hover:text-red-500"
                          aria-label={`删除记忆：${memory.memory_value}`}
                          title="删除记忆"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </li>
                    ))}
                  </ul>
                </section>
              ))}
            </>
          )}
          {pendingDeletion && (
            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[var(--border)] pt-4 text-sm">
              <p className="text-[var(--text-secondary)]">删除这条记忆？</p>
              <div className="flex items-center gap-2">
                <button type="button" onClick={() => setPendingDeletion(null)} className="rounded-md px-3 py-1.5 text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]">取消</button>
                <button type="button" onClick={confirmDeletion} className="rounded-md bg-red-500 px-3 py-1.5 text-white hover:bg-red-600">确认删除</button>
              </div>
            </div>
          )}
        </div>
      </Modal>
    </>
  )
}
