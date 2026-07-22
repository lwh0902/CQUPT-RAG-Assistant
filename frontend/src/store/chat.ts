import { create } from 'zustand'
import type {
  ChatStreamController,
  ConfidenceLevel,
  MemoryAction,
  Session,
  Message,
  Source,
  StreamMessage,
} from '../api/client'
import { api, streamChat } from '../api/client'
import { useToastStore } from './toast'
import { getLang } from '../i18n'

export interface ThinkingStep {
  step: string
  message: string
  detail?: string
  status: 'pending' | 'in_progress' | 'completed'
}

function normalizeSession(session: Session): Session {
  return {
    ...session,
    id: session.id || session.session_id || '',
    title: session.title || '新建对话',
    created_at: session.created_at || new Date().toISOString(),
  }
}

interface ChatState {
  sessions: Session[]
  currentSessionId: string | null
  messages: Message[]
  isStreaming: boolean
  thinkingSteps: ThinkingStep[]
  currentReply: string
  currentSources: Source[]
  currentConfidenceLevel?: ConfidenceLevel
  currentEvidenceSummary?: string
  currentUncertainPoints: string[]
  webSearchEnabled: boolean
  streamController: ChatStreamController | null
  pendingMemoryActions: MemoryAction[]

  fetchSessions: () => Promise<void>
  createSession: () => Promise<string | null>
  selectSession: (sessionId: string) => Promise<void>
  deleteSession: (sessionId: string) => Promise<void>
  renameSession: (sessionId: string, title: string) => Promise<boolean>
  fetchMessages: (sessionId: string) => Promise<void>
  sendMessage: (content: string) => void
  setWebSearchEnabled: (enabled: boolean) => void
  stopStreaming: () => void
  regenerateLast: () => void
  retryLastWithWebSearch: () => void
  resetStream: () => void
  cleanup: () => void
  dismissPendingMemory: (candidateId: string) => void
  confirmPendingMemory: (candidateId: string) => Promise<void>
  rejectPendingMemory: (candidateId: string) => Promise<void>
}

export const useChatStore = create<ChatState>((set, get) => ({
  sessions: [],
  currentSessionId: null,
  messages: [],
  isStreaming: false,
  thinkingSteps: [],
  currentReply: '',
  currentSources: [],
  currentConfidenceLevel: undefined,
  currentEvidenceSummary: undefined,
  currentUncertainPoints: [],
  webSearchEnabled: false,
  streamController: null,
  pendingMemoryActions: [],

  fetchSessions: async () => {
    try {
      const res = await api.get<{ sessions: Session[] }>('/sessions')
      const sessions = res.data.sessions
        .map(normalizeSession)
        .filter((session) => session.id)
      set({ sessions })
    } catch {
      useToastStore.getState().addToast('获取会话列表失败', 'error')
    }
  },

  createSession: async () => {
    const sessionId = `session_${crypto.randomUUID().replace(/-/g, '')}`
    const newSession: Session = {
      id: sessionId,
      title: '新建对话',
      created_at: new Date().toISOString(),
    }
    set((state) => ({
      sessions: [newSession, ...state.sessions],
      currentSessionId: newSession.id,
      messages: [],
      thinkingSteps: [],
      currentReply: '',
      currentSources: [],
      currentConfidenceLevel: undefined,
      currentEvidenceSummary: undefined,
      currentUncertainPoints: [],
    }))
    return newSession.id
  },

  selectSession: async (sessionId: string) => {
    set({
      currentSessionId: sessionId,
      thinkingSteps: [],
      currentReply: '',
      currentSources: [],
      currentConfidenceLevel: undefined,
      currentEvidenceSummary: undefined,
      currentUncertainPoints: [],
    })
    await get().fetchMessages(sessionId)
  },

  deleteSession: async (sessionId: string) => {
    try {
      await api.delete(`/sessions/${sessionId}`)
      const state = get()
      const remaining = state.sessions.filter((s) => s.id !== sessionId)
      if (sessionId === state.currentSessionId) {
        if (remaining.length > 0) {
          set({
            sessions: remaining,
            currentSessionId: remaining[0].id,
            messages: [],
          })
          await get().fetchMessages(remaining[0].id)
        } else {
          set({ sessions: remaining, currentSessionId: null, messages: [] })
          await get().createSession()
        }
      } else {
        set({ sessions: remaining })
      }
      useToastStore.getState().addToast('已删除会话', 'success')
    } catch {
      useToastStore.getState().addToast('删除会话失败', 'error')
    }
  },

  renameSession: async (sessionId: string, title: string) => {
    const trimmed = title.trim()
    if (!trimmed) return false
    const original = get().sessions.find((s) => s.id === sessionId)?.title ?? ''
    set((state) => ({
      sessions: state.sessions.map((s) =>
        s.id === sessionId ? { ...s, title: trimmed } : s,
      ),
    }))
    try {
      await api.patch(`/sessions/${sessionId}`, { title: trimmed })
      return true
    } catch {
      set((state) => ({
        sessions: state.sessions.map((s) =>
          s.id === sessionId ? { ...s, title: original } : s,
        ),
      }))
      useToastStore.getState().addToast('重命名失败，已还原', 'error')
      return false
    }
  },

  fetchMessages: async (sessionId: string) => {
    if (!sessionId || sessionId === 'undefined') return
    try {
      const res = await api.get<{
        session_id: string
        messages: Message[]
      }>(`/sessions/${sessionId}/messages`)
      set({ messages: res.data.messages })
    } catch {
      useToastStore.getState().addToast('加载消息失败', 'error')
    }
  },

  sendMessage: (content: string) => {
    const state = get()
    const userId = localStorage.getItem('user_id')
    if (!userId) return

    const sessionId = state.currentSessionId || `session_${crypto.randomUUID().replace(/-/g, '')}`
    if (!state.currentSessionId) {
      const newSession: Session = {
        id: sessionId,
        title: '新建对话',
        created_at: new Date().toISOString(),
      }
      set((s) => ({
        sessions: [newSession, ...s.sessions],
        currentSessionId: newSession.id,
      }))
    }

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      created_at: new Date().toISOString(),
    }

    const webSearchEnabled = state.webSearchEnabled
    set((prev) => ({
      messages: [...prev.messages, userMessage],
      isStreaming: true,
      thinkingSteps: [],
      currentReply: '',
      currentSources: [],
      currentConfidenceLevel: undefined,
      currentEvidenceSummary: undefined,
      currentUncertainPoints: [],
      webSearchEnabled: false,
    }))

    get().streamController?.abort()

    const streamController = streamChat(
      {
        session_id: sessionId,
        new_message: content,
        web_search_enabled: webSearchEnabled,
        lang: getLang(),
      },
      {
        onMessage: (data: StreamMessage) => {
          switch (data.type) {
            case 'thinking': {
              set((prev) => {
                const steps = [...prev.thinkingSteps]
                const existingIndex = steps.findIndex((s) => s.step === data.step)
                const newStep: ThinkingStep = {
                  step: data.step ?? '',
                  message: data.message ?? '',
                  detail: data.detail,
                  status: 'in_progress',
                }
                for (let i = 0; i < steps.length; i++) {
                  steps[i] = { ...steps[i], status: 'completed' }
                }
                if (existingIndex >= 0) {
                  steps[existingIndex] = newStep
                } else {
                  steps.push(newStep)
                }
                return { thinkingSteps: steps }
              })
              break
            }
            case 'start': {
              break
            }
            case 'delta': {
              set((prev) => {
                const thinkingSteps = prev.thinkingSteps.some((s) => s.status !== 'completed')
                  ? prev.thinkingSteps.map((s) => ({ ...s, status: 'completed' as const }))
                  : prev.thinkingSteps
                return {
                  thinkingSteps,
                  currentReply: prev.currentReply + (data.content ?? ''),
                }
              })
              break
            }
            case 'end': {
              const assistantMessage: Message = {
                id: crypto.randomUUID(),
                role: 'assistant',
                content: get().currentReply,
                sources: data.sources,
                quick_fact: data.quick_fact,
                confidence_level: data.confidence_level,
                evidence_summary: data.evidence_summary,
                uncertain_points: data.uncertain_points,
                retrieval_decision: data.retrieval_decision,
                created_at: new Date().toISOString(),
              }
              const memoryActions = data.memory_actions ?? []
              for (const action of memoryActions) {
                if (action.action === 'saved' && action.message) {
                  useToastStore.getState().addToast(action.message, 'success')
                }
              }
              const pending = memoryActions.filter(
                (action) => action.action === 'pending' && action.candidate_id,
              )
              set((prev) => ({
                messages: [...prev.messages, assistantMessage],
                isStreaming: false,
                currentSources: data.sources ?? [],
                currentConfidenceLevel: data.confidence_level,
                currentEvidenceSummary: data.evidence_summary,
                currentUncertainPoints: data.uncertain_points ?? [],
                streamController: null,
                pendingMemoryActions: pending.length
                  ? [...pending, ...prev.pendingMemoryActions.filter((item) => !pending.some((p) => p.candidate_id === item.candidate_id))]
                  : prev.pendingMemoryActions,
              }))
              get().fetchSessions()
              break
            }
            case 'error': {
              set({ isStreaming: false, streamController: null })
              const msg = data.message ?? '未知错误'
              if (msg.includes('429') || msg.includes('访问量过大')) {
                useToastStore.getState().addToast('服务繁忙，请稍后再试', 'error')
              } else {
                useToastStore.getState().addToast(msg, 'error')
              }
              break
            }
          }
        },
        onError: (error) => {
          useToastStore.getState().addToast(error.message || '连接中断，请重试', 'error')
          set({ isStreaming: false, streamController: null })
        },
        onClose: () => {
          set((prev) => (prev.isStreaming ? { isStreaming: false, streamController: null } : { streamController: null }))
        },
      }
    )

    set({ streamController })
  },

  setWebSearchEnabled: (enabled: boolean) => set({ webSearchEnabled: enabled }),

  stopStreaming: () => {
    const state = get()
    state.streamController?.abort()

    const partialReply = state.currentReply.trim()
    set((prev) => ({
      messages: partialReply
        ? [
            ...prev.messages,
            {
              id: crypto.randomUUID(),
              role: 'assistant',
              content: `${partialReply}\n\n_已停止生成。_`,
              sources: prev.currentSources,
              created_at: new Date().toISOString(),
            },
          ]
        : prev.messages,
      isStreaming: false,
      thinkingSteps: [],
      currentReply: '',
      streamController: null,
    }))
  },

  regenerateLast: () => {
    const state = get()
    if (state.isStreaming) return

    const lastAssistantIndex = [...state.messages]
      .map((message, index) => ({ message, index }))
      .filter((item) => item.message.role === 'assistant')
      .pop()?.index

    const searchEnd = lastAssistantIndex ?? state.messages.length
    const lastUserEntry = state.messages
      .map((message, index) => ({ message, index }))
      .slice(0, searchEnd)
      .filter((item) => item.message.role === 'user')
      .pop()

    if (!lastUserEntry) return

    set({
      messages: state.messages.slice(0, lastUserEntry.index),
    })
    get().sendMessage(lastUserEntry.message.content)
  },

  retryLastWithWebSearch: () => {
    const state = get()
    if (state.isStreaming) return
    const lastAssistantIndex = [...state.messages]
      .map((message, index) => ({ message, index }))
      .filter((item) => item.message.role === 'assistant')
      .pop()?.index
    const searchEnd = lastAssistantIndex ?? state.messages.length
    const lastUserEntry = state.messages
      .map((message, index) => ({ message, index }))
      .slice(0, searchEnd)
      .filter((item) => item.message.role === 'user')
      .pop()
    if (!lastUserEntry) return

    set({ messages: state.messages.slice(0, lastUserEntry.index) })
    set({ webSearchEnabled: true })
    get().sendMessage(lastUserEntry.message.content)
  },

  resetStream: () => {
    set({
      thinkingSteps: [],
      currentReply: '',
      currentSources: [],
      currentConfidenceLevel: undefined,
      currentEvidenceSummary: undefined,
      currentUncertainPoints: [],
    })
  },

  cleanup: () => {
    get().streamController?.abort()
    set({ streamController: null })
  },

  dismissPendingMemory: (candidateId: string) => {
    set((prev) => ({
      pendingMemoryActions: prev.pendingMemoryActions.filter(
        (item) => item.candidate_id !== candidateId,
      ),
    }))
  },

  confirmPendingMemory: async (candidateId: string) => {
    try {
      const { data } = await api.post<{ status: string; message?: string }>(
        `/memories/candidates/${candidateId}/confirm`,
      )
      if (data.status === 'confirmed') {
        useToastStore.getState().addToast(data.message || '已记住', 'success')
      } else {
        useToastStore.getState().addToast('确认失败，请稍后重试', 'info')
      }
    } catch {
      useToastStore.getState().addToast('确认记忆失败', 'error')
    } finally {
      get().dismissPendingMemory(candidateId)
    }
  },

  rejectPendingMemory: async (candidateId: string) => {
    try {
      await api.post(`/memories/candidates/${candidateId}/reject`)
    } catch {
      useToastStore.getState().addToast('忽略记忆失败', 'error')
    } finally {
      get().dismissPendingMemory(candidateId)
    }
  },
}))
