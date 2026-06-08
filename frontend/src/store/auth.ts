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
  logout: () => void
  restoreSession: () => void
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
    const res = await api.post<{ token: string; user_id: string }>('/auth/login', {
      phone,
      password,
    })
    const { token, user_id } = res.data
    localStorage.setItem('token', token)
    localStorage.setItem('user_id', user_id)
    localStorage.setItem('phone', phone)
    set({ token, userId: user_id, phone, isAuthenticated: true })
  },

  register: async (phone: string, password: string) => {
    const res = await api.post<{ token: string; user_id: string }>('/auth/register', {
      phone,
      password,
    })
    const { token, user_id } = res.data
    localStorage.setItem('token', token)
    localStorage.setItem('user_id', user_id)
    localStorage.setItem('phone', phone)
    set({ token, userId: user_id, phone, isAuthenticated: true })
  },

  logout: () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user_id')
    localStorage.removeItem('phone')
    set({ token: null, userId: null, phone: null, isAuthenticated: false })
  },
}))
