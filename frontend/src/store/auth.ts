import { create } from 'zustand'
import { api } from '../api/client'

interface AuthState {
  token: string | null
  userId: string | null
  phone: string | null
  isAuthenticated: boolean
  checkPhone: (phone: string) => Promise<boolean>
  login: (phone: string, password: string) => Promise<void>
  register: (phone: string, password: string) => Promise<void>
  logout: () => Promise<void>
  restoreSession: () => void
}

function persistAuth(token: string, userId: string, phone: string) {
  localStorage.setItem('token', token)
  localStorage.setItem('user_id', userId)
  localStorage.setItem('phone', phone)
}

function clearPersistedAuth() {
  localStorage.removeItem('token')
  localStorage.removeItem('user_id')
  localStorage.removeItem('phone')
}

export const useAuthStore = create<AuthState>((set) => ({
  token: null,
  userId: null,
  phone: null,
  isAuthenticated: false,

  restoreSession: () => {
    const token = localStorage.getItem('token')
    const userId = localStorage.getItem('user_id')
    const phone = localStorage.getItem('phone')
    if (token && userId) {
      set({ token, userId, phone, isAuthenticated: true })
    }
  },

  checkPhone: async (phone: string) => {
    const res = await api.post<{ exists: boolean }>('/auth/check-phone', { phone })
    return res.data.exists
  },

  login: async (phone: string, password: string) => {
    const res = await api.post<{
      token: string
      access_token?: string
      user_id: string
    }>('/auth/login', {
      phone,
      password,
    })
    const token = res.data.access_token || res.data.token
    const { user_id } = res.data
    persistAuth(token, user_id, phone)
    set({ token, userId: user_id, phone, isAuthenticated: true })
  },

  register: async (phone: string, password: string) => {
    const res = await api.post<{
      token: string
      access_token?: string
      user_id: string
    }>('/auth/register', {
      phone,
      password,
    })
    const token = res.data.access_token || res.data.token
    const { user_id } = res.data
    persistAuth(token, user_id, phone)
    set({ token, userId: user_id, phone, isAuthenticated: true })
  },

  logout: async () => {
    try {
      await api.post('/auth/logout')
    } catch {
      // Clear local state even if the network call fails.
    }
    clearPersistedAuth()
    set({ token: null, userId: null, phone: null, isAuthenticated: false })
  },
}))
