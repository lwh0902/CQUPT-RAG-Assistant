import { create } from 'zustand'
import type { Session, Message, Source, WSMessage } from '../api/client'
import { api, createWebSocket } from '../api/client'
import { useToastStore } from './toast'

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
  ws: WebSocket | null

  fetchSessions: () => Promise<void>
  createSession: () => Promise<string | null>
  selectSession: (sessionId: string) => Promise<void>
  deleteSession: (sessionId: string) => Promise<void>
  fetchMessages: (sessionId: string) => Promise<void>
  sendMessage: (content: string) => void
  stopStreaming: () => void
  regenerateLast: () => void
  resetStream: () => void
  cleanup: () => void
}

export const useChatStore = create<ChatState>((set, get) => ({
  sessions: [],
  currentSessionId: null,
  messages: [],
  isStreaming: false,
  thinkingSteps: [],
  currentReply: '',
  currentSources: [],
  ws: null,

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
    }))
    return newSession.id
  },

  selectSession: async (sessionId: string) => {
    set({
      currentSessionId: sessionId,
      thinkingSteps: [],
      currentReply: '',
      currentSources: [],
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

    set((prev) => ({
      messages: [...prev.messages, userMessage],
      isStreaming: true,
      thinkingSteps: [],
      currentReply: '',
      currentSources: [],
    }))

    if (get().ws) get().ws!.close()

    const ws = createWebSocket(
      (data: WSMessage) => {
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
              created_at: new Date().toISOString(),
            }
            set((prev) => ({
              messages: [...prev.messages, assistantMessage],
              isStreaming: false,
              currentSources: data.sources ?? [],
            }))
            // Refresh session list to sync title
            get().fetchSessions()
            break
          }
          case 'error': {
            set({ isStreaming: false })
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
      () => {
        useToastStore.getState().addToast('连接中断，请重试', 'error')
        set({ isStreaming: false })
      },
      () => {
        set((prev) => (prev.isStreaming ? { isStreaming: false } : {}))
      }
    )

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          user_id: userId,
          session_id: sessionId,
          new_message: content,
        })
      )
    }

    set({ ws })
  },

  stopStreaming: () => {
    const state = get()
    state.ws?.close()

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
      ws: null,
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

  resetStream: () => {
    set({ thinkingSteps: [], currentReply: '', currentSources: [] })
  },

  cleanup: () => {
    const { ws } = get()
    if (ws) ws.close()
    set({ ws: null })
  },
}))
