import axios from 'axios'

const CSRF_COOKIE_NAME = 'csrf_token'
const CSRF_HEADER_NAME = 'X-CSRF-Token'

function readCookie(name: string): string | null {
  if (typeof document === 'undefined') return null
  const prefix = `${encodeURIComponent(name)}=`
  const found = document.cookie
    .split(';')
    .map((part) => part.trim())
    .find((part) => part.startsWith(prefix))
  if (!found) return null
  return decodeURIComponent(found.slice(prefix.length))
}

export const api = axios.create({
  baseURL: '/api',
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }

  const method = (config.method || 'get').toUpperCase()
  if (!['GET', 'HEAD', 'OPTIONS'].includes(method)) {
    const csrf = readCookie(CSRF_COOKIE_NAME)
    if (csrf) {
      config.headers[CSRF_HEADER_NAME] = csrf
    }
  }
  return config
})

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const status = error.response?.status
    const original = error.config as typeof error.config & { _retry?: boolean }
    if (status === 401 && original && !original._retry && !String(original.url || '').includes('/auth/refresh')) {
      original._retry = true
      try {
        const csrf = readCookie(CSRF_COOKIE_NAME)
        const refreshed = await axios.post(
          '/api/auth/refresh',
          {},
          {
            withCredentials: true,
            headers: csrf ? { [CSRF_HEADER_NAME]: csrf } : undefined,
          }
        )
        const accessToken =
          refreshed.data?.access_token || refreshed.data?.token || null
        if (accessToken) {
          localStorage.setItem('token', accessToken)
          original.headers = original.headers || {}
          original.headers.Authorization = `Bearer ${accessToken}`
        }
        return api(original)
      } catch {
        localStorage.removeItem('token')
        localStorage.removeItem('user_id')
        localStorage.removeItem('phone')
        if (window.location.pathname !== '/login') {
          window.location.href = '/login'
        }
      }
    }
    return Promise.reject(error)
  }
)

export interface CheckPhoneResponse {
  exists: boolean
}

export interface LoginResponse {
  token: string
  access_token?: string
  user_id: string
  expires_in?: number
}

export interface RegisterResponse {
  token: string
  access_token?: string
  user_id: string
  expires_in?: number
}

export interface Session {
  id: string
  session_id?: string
  title: string
  first_message?: string
  created_at: string
  preview?: string
  message_count?: number
}

export interface SessionsResponse {
  sessions: Session[]
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  confidence_level?: ConfidenceLevel
  evidence_summary?: string
  uncertain_points?: string[]
  created_at: string
  retrieval_decision?: 'supported' | 'web_only' | 'out_of_scope' | 'insufficient'
}

export interface Source {
  id?: string
  source_type?: 'knowledge_base' | 'web'
  title?: string
  url?: string
  site_name?: string
  document_id?: string
  document_name?: string
  page?: number
  snippet?: string
  preview?: string
  published_at?: string
  retrieved_at?: string
}

export type ConfidenceLevel = 'high' | 'medium' | 'low' | 'unknown'

export interface SessionMessagesResponse {
  session_id: string
  messages: Message[]
}

export interface SessionSearchResult {
  session_id: string
  title: string
  preview: string
  matched_in_title: boolean
  matched_in_message: boolean
  message_count: number
}

export interface SessionSearchResponse {
  query: string
  results: SessionSearchResult[]
}

export interface KnowledgeDocument {
  document_id: string
  document_name: string
  document_type: string
  topic: string
  authority_level: number
  file_name: string
  file_size: number
  file_exists: boolean
  previewable: boolean
  page_count: number | null
}

export interface KnowledgeDocumentsResponse {
  documents: KnowledgeDocument[]
}

export interface MemoryAction {
  action: 'saved' | 'pending' | 'rejected' | string
  memory_type?: string
  memory_key?: string
  memory_value?: string
  candidate_id?: string | null
  memory_id?: number | null
  message?: string
}

export interface StreamMessage {
  type: string
  step?: string
  message?: string
  detail?: string
  content?: string
  sources?: Source[]
  confidence?: number
  confidence_level?: ConfidenceLevel
  evidence_summary?: string
  uncertain_points?: string[]
  retrieval_decision?: 'supported' | 'web_only' | 'out_of_scope' | 'insufficient'
  memory_actions?: MemoryAction[]
}

export interface ChatStreamHandlers {
  onMessage: (data: StreamMessage) => void
  onError?: (error: Error) => void
  onClose?: () => void
}

export interface ChatStreamController {
  abort: () => void
}

function parseSseChunk(buffer: string): { events: string[]; rest: string } {
  const normalized = buffer.replace(/\r\n/g, '\n')
  const parts = normalized.split('\n\n')
  const rest = parts.pop() ?? ''
  const events: string[] = []
  for (const part of parts) {
    const dataLines = part
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
    if (dataLines.length > 0) {
      events.push(dataLines.join('\n'))
    }
  }
  return { events, rest }
}

export function streamChat(
  body: {
    session_id: string
    new_message: string
    web_search_enabled?: boolean
  },
  handlers: ChatStreamHandlers
): ChatStreamController {
  const controller = new AbortController()

  void (async () => {
    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      }
      const token = localStorage.getItem('token')
      if (token) {
        headers.Authorization = `Bearer ${token}`
      }
      const csrf = readCookie(CSRF_COOKIE_NAME)
      if (csrf) {
        headers[CSRF_HEADER_NAME] = csrf
      }

      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers,
        credentials: 'include',
        body: JSON.stringify(body),
        signal: controller.signal,
      })

      if (!response.ok) {
        let message = `请求失败（${response.status}）`
        try {
          const payload = await response.json()
          if (typeof payload?.detail === 'string') {
            message = payload.detail
          } else if (typeof payload?.message === 'string') {
            message = payload.message
          }
        } catch {
          // ignore body parse errors
        }
        if (response.status === 401) {
          localStorage.removeItem('token')
          localStorage.removeItem('user_id')
          localStorage.removeItem('phone')
          window.location.href = '/login'
        }
        throw new Error(message)
      }

      if (!response.body) {
        throw new Error('浏览器不支持流式响应')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parsed = parseSseChunk(buffer)
        buffer = parsed.rest
        for (const raw of parsed.events) {
          try {
            const data = JSON.parse(raw) as StreamMessage
            handlers.onMessage(data)
          } catch {
            console.error('Failed to parse SSE message')
          }
        }
      }

      if (buffer.trim()) {
        const parsed = parseSseChunk(`${buffer}\n\n`)
        for (const raw of parsed.events) {
          try {
            const data = JSON.parse(raw) as StreamMessage
            handlers.onMessage(data)
          } catch {
            console.error('Failed to parse SSE message')
          }
        }
      }

      handlers.onClose?.()
    } catch (error) {
      if (controller.signal.aborted) {
        handlers.onClose?.()
        return
      }
      const err = error instanceof Error ? error : new Error('流式连接失败')
      handlers.onError?.(err)
      handlers.onClose?.()
    }
  })()

  return {
    abort: () => controller.abort(),
  }
}

export async function searchSessions(query: string): Promise<SessionSearchResponse> {
  const { data } = await api.get<SessionSearchResponse>('/sessions/search', {
    params: { q: query },
  })
  return data
}

export async function listKnowledgeDocuments(): Promise<KnowledgeDocumentsResponse> {
  const { data } = await api.get<KnowledgeDocumentsResponse>('/documents')
  return data
}
