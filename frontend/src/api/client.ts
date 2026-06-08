import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user_id')
      localStorage.removeItem('phone')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export interface CheckPhoneResponse {
  exists: boolean
}

export interface LoginResponse {
  token: string
  user_id: string
}

export interface RegisterResponse {
  token: string
  user_id: string
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
  created_at: string
}

export interface Source {
  document_name?: string
  page: number
  snippet?: string
  preview?: string
}

export interface SessionMessagesResponse {
  session_id: string
  messages: Message[]
}

export interface WSMessage {
  type: string
  step?: string
  message?: string
  detail?: string
  content?: string
  sources?: Source[]
}

export function createWebSocket(
  onMessage: (data: WSMessage) => void,
  onError?: (error: Event) => void,
  onClose?: (event: CloseEvent) => void
): WebSocket {
  const token = localStorage.getItem('token')
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${protocol}//${window.location.host}/ws/chat${token ? `?token=${token}` : ''}`
  const ws = new WebSocket(wsUrl)

  ws.onmessage = (event) => {
    try {
      const data: WSMessage = JSON.parse(event.data as string)
      onMessage(data)
    } catch {
      console.error('Failed to parse WebSocket message')
    }
  }

  ws.onerror = (event) => {
    onError?.(event)
  }

  ws.onclose = (event) => {
    onClose?.(event)
  }

  return ws
}
