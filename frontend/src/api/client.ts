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

export interface QuickFact {
  id: string
  title: string
  answer: string
  source_name?: string
  source_url?: string
  updated_at?: string
}

export interface QuickFactLink {
  id: string
  title: string
  sample_question: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  quick_fact?: QuickFact
  confidence_level?: ConfidenceLevel
  evidence_summary?: string
  uncertain_points?: string[]
  created_at: string
  retrieval_decision?: 'supported' | 'web_only' | 'out_of_scope' | 'insufficient' | 'quick_fact'
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
  retrieval_decision?: 'supported' | 'web_only' | 'out_of_scope' | 'insufficient' | 'quick_fact'
  memory_actions?: MemoryAction[]
  quick_fact?: QuickFact
}

export interface McqItem {
  question: string
  options: Record<string, string>
  answer: string
  analysis: string
  round?: number
}

export interface QaItem {
  question: string
  category?: string
  spoken_answer: string
  analysis: string
}

export interface InterviewReference {
  title: string
  url: string
  snippet?: string
}

export interface RealInterviewQuestion {
  question: string
  source_title?: string
  source_url?: string
}

export interface InterviewSession {
  id: string
  company: string
  position?: string
  jd_text?: string
  resume_text?: string
  resume_filename?: string | null
  reference_used?: boolean
  references?: InterviewReference[]
  real_questions?: RealInterviewQuestion[]
  report_text?: string | null
  created_at?: string
  mcq: McqItem[]
  qa: QaItem[]
  mcq_count?: number
  qa_count?: number
}

export interface InterviewGenerateStage {
  stage: string
  message: string
  progress: number
  ref_count?: number
  chunk?: number
  total_chunks?: number
}

export async function generateInterviewBank(form: FormData): Promise<InterviewSession> {
  const { data } = await api.post<InterviewSession>('/interview/generate', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000,
  })
  return data
}

export type InterviewGenerateHandlers = {
  onStage?: (stage: InterviewGenerateStage) => void
  onDone?: (session: InterviewSession) => void
  onError?: (message: string) => void
}

export function streamGenerateInterviewBank(
  form: FormData,
  handlers: InterviewGenerateHandlers,
): { abort: () => void } {
  const controller = new AbortController()

  void (async () => {
    try {
      const headers: Record<string, string> = {
        Accept: 'text/event-stream',
      }
      const token = localStorage.getItem('token')
      if (token) headers.Authorization = `Bearer ${token}`
      const csrf = readCookie(CSRF_COOKIE_NAME)
      if (csrf) headers[CSRF_HEADER_NAME] = csrf

      const response = await fetch('/api/interview/generate/stream', {
        method: 'POST',
        headers,
        credentials: 'include',
        body: form,
        signal: controller.signal,
      })

      if (!response.ok) {
        let message = `请求失败（${response.status}）`
        try {
          const payload = await response.json()
          if (typeof payload?.detail === 'string') message = payload.detail
        } catch {
          /* ignore */
        }
        if (response.status === 401) {
          localStorage.removeItem('token')
          window.location.href = '/login'
        }
        throw new Error(message)
      }
      if (!response.body) throw new Error('浏览器不支持流式响应')

      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''
      let currentEvent = 'message'
      let finished = false

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop() ?? ''
        for (const part of parts) {
          const lines = part.split('\n')
          let eventName = currentEvent
          const dataLines: string[] = []
          for (const line of lines) {
            if (line.startsWith('event:')) eventName = line.slice(6).trim()
            else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
          }
          if (!dataLines.length) continue
          const raw = dataLines.join('\n')
          let payload: any = {}
          try {
            payload = JSON.parse(raw)
          } catch {
            payload = { detail: raw }
          }
          if (eventName === 'stage') handlers.onStage?.(payload as InterviewGenerateStage)
          else if (eventName === 'done') {
            finished = true
            const session = {
              ...payload,
              mcq: payload.mcq ?? [],
              qa: payload.qa ?? [],
              references: payload.references ?? [],
              real_questions: payload.real_questions ?? [],
            } as InterviewSession
            handlers.onDone?.(session)
          } else if (eventName === 'error') {
            finished = true
            handlers.onError?.(payload.detail || '题库生成失败')
          }
        }
      }
      if (!finished) handlers.onError?.('连接中断，题库可能未生成完成')
    } catch (err: any) {
      if (err?.name === 'AbortError') return
      handlers.onError?.(err?.message || '题库生成失败')
    }
  })()

  return { abort: () => controller.abort() }
}

export type TutorHandlers = {
  onToken?: (text: string) => void
  onDone?: () => void
  onError?: (message: string) => void
}

export function streamInterviewTutor(
  messages: { role: string; content: string }[],
  handlers: TutorHandlers,
): { abort: () => void } {
  const controller = new AbortController()

  void (async () => {
    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      }
      const token = localStorage.getItem('token')
      if (token) headers.Authorization = `Bearer ${token}`
      const csrf = readCookie(CSRF_COOKIE_NAME)
      if (csrf) headers[CSRF_HEADER_NAME] = csrf

      const response = await fetch('/api/interview/tutor/stream', {
        method: 'POST',
        headers,
        credentials: 'include',
        body: JSON.stringify({ messages }),
        signal: controller.signal,
      })
      if (!response.ok) {
        let message = `请求失败（${response.status}）`
        try {
          const payload = await response.json()
          if (typeof payload?.detail === 'string') message = payload.detail
        } catch {
          /* ignore */
        }
        throw new Error(message)
      }
      if (!response.body) throw new Error('浏览器不支持流式响应')

      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop() ?? ''
        for (const part of parts) {
          const lines = part.split('\n')
          let eventName = 'message'
          const dataLines: string[] = []
          for (const line of lines) {
            if (line.startsWith('event:')) eventName = line.slice(6).trim()
            else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
          }
          if (!dataLines.length) continue
          let payload: any = {}
          try {
            payload = JSON.parse(dataLines.join('\n'))
          } catch {
            payload = {}
          }
          if (eventName === 'token') handlers.onToken?.(payload.content || '')
          else if (eventName === 'done') handlers.onDone?.()
          else if (eventName === 'error') handlers.onError?.(payload.detail || '讲解失败')
        }
      }
      handlers.onDone?.()
    } catch (err: any) {
      if (err?.name === 'AbortError') return
      handlers.onError?.(err?.message || '讲解失败')
    }
  })()

  return { abort: () => controller.abort() }
}

export async function fetchInterviewSessions(): Promise<InterviewSession[]> {
  const { data } = await api.get<{ sessions: InterviewSession[] }>('/interview/sessions')
  return (data.sessions ?? []).map((s) => ({ ...s, mcq: s.mcq ?? [], qa: s.qa ?? [] }))
}

export async function fetchInterviewSession(id: string): Promise<InterviewSession> {
  const { data } = await api.get<InterviewSession>(`/interview/sessions/${id}`)
  return data
}

export async function createWeaknessReport(
  id: string,
  wrongIndices: number[],
): Promise<{ report: string }> {
  const { data } = await api.post<{ report: string }>(
    `/interview/sessions/${id}/report`,
    { wrong_indices: wrongIndices },
    { timeout: 120000 },
  )
  return data
}

export async function regenerateMcq(
  id: string,
  wrongIndices: number[],
): Promise<{ round: number; mcq: McqItem[] }> {
  const { data } = await api.post<{ round: number; mcq: McqItem[] }>(
    `/interview/sessions/${id}/regenerate-mcq`,
    { wrong_indices: wrongIndices },
    { timeout: 120000 },
  )
  return data
}

export async function deleteInterviewSession(id: string): Promise<void> {
  await api.delete(`/interview/sessions/${id}`)
}

export interface ChatStreamHandlers {
  onMessage: (data: StreamMessage) => void
  onError?: (error: Error) => void
  onClose?: () => void
}

export interface ChatStreamController {
  abort: () => void
}

export async function fetchQuickFacts(): Promise<QuickFactLink[]> {
  const { data } = await api.get<{ facts: QuickFactLink[] }>('/quick-facts')
  return data.facts ?? []
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
    lang?: string
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
