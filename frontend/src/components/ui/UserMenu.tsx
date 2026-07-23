import { useEffect, useRef, useState } from 'react'
import { Check, Copy, KeyRound, LogOut, Monitor, Moon, Settings, Sun } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { api } from '../../api/client'
import { useAuthStore } from '../../store/auth'
import { useThemeStore, type Theme } from '../../store/theme'
import { useToastStore } from '../../store/toast'

const THEME_OPTIONS: Array<{
  value: Theme
  label: string
  icon: typeof Sun
}> = [
  { value: 'light', label: '浅色模式', icon: Sun },
  { value: 'dark', label: '深色模式', icon: Moon },
]

export default function UserMenu() {
  const navigate = useNavigate()
  const menuRef = useRef<HTMLDivElement>(null)
  const { phone, logout, canManageInvites, refreshCapabilities } = useAuthStore()
  const { theme, setTheme, useSystemTheme } = useThemeStore()
  const [open, setOpen] = useState(false)
  const [inviteBusy, setInviteBusy] = useState(false)
  const [lastCode, setLastCode] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    void refreshCapabilities()

    const handlePointerDown = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) {
        setOpen(false)
      }
    }

    window.addEventListener('pointerdown', handlePointerDown)
    return () => window.removeEventListener('pointerdown', handlePointerDown)
  }, [open, refreshCapabilities])

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  const handleCreateInvite = async () => {
    if (inviteBusy) return
    setInviteBusy(true)
    try {
      const { data } = await api.post<{ code: string; expires_at: string; ttl_days: number }>(
        '/auth/invite-codes',
      )
      setLastCode(data.code)
      try {
        await navigator.clipboard.writeText(data.code)
        useToastStore.getState().addToast(`邀请码 ${data.code} 已复制（${data.ttl_days} 天有效）`, 'success')
      } catch {
        useToastStore.getState().addToast(`邀请码：${data.code}（${data.ttl_days} 天有效）`, 'success')
      }
    } catch (err: any) {
      const detail = err?.response?.data?.detail
      useToastStore.getState().addToast(typeof detail === 'string' ? detail : '生成邀请码失败', 'error')
    } finally {
      setInviteBusy(false)
    }
  }

  return (
    <div ref={menuRef} className="relative">
      {open && (
        <div className="absolute bottom-12 left-0 right-0 z-30 overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--surface-raised)] p-1.5 shadow-[0_16px_48px_rgba(0,0,0,0.16)]">
          <div className="px-3 py-2">
            <p className="text-xs text-[var(--text-tertiary)]">当前账号</p>
            <p className="truncate text-sm font-medium text-[var(--text-primary)]">{phone}</p>
          </div>

          <div className="my-1 h-px bg-[var(--border)]" />

          {canManageInvites && (
            <>
              <button
                type="button"
                onClick={handleCreateInvite}
                disabled={inviteBusy}
                className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-sm text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-hover)] disabled:opacity-60"
              >
                <KeyRound className="h-4 w-4 text-[var(--text-secondary)]" />
                <span className="flex-1">{inviteBusy ? '生成中…' : '生成邀请码'}</span>
              </button>
              {lastCode && (
                <button
                  type="button"
                  onClick={async () => {
                    try {
                      await navigator.clipboard.writeText(lastCode)
                      useToastStore.getState().addToast('已复制邀请码', 'success')
                    } catch {
                      /* ignore */
                    }
                  }}
                  className="mb-1 flex w-full items-center gap-2 rounded-xl px-3 py-1.5 text-left text-xs text-[var(--accent)] hover:bg-[var(--bg-hover)]"
                >
                  <Copy className="h-3.5 w-3.5" />
                  <span className="font-mono tracking-wider">{lastCode}</span>
                </button>
              )}
              <div className="my-1 h-px bg-[var(--border)]" />
            </>
          )}

          <div className="px-2 py-1.5 text-xs font-medium text-[var(--text-tertiary)]">
            外观
          </div>
          <div className="space-y-0.5">
            {THEME_OPTIONS.map((option) => {
              const Icon = option.icon
              const selected = theme === option.value
              return (
                <button
                  key={option.value}
                  onClick={() => setTheme(option.value)}
                  className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-sm text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-hover)]"
                >
                  <Icon className="h-4 w-4 text-[var(--text-secondary)]" />
                  <span className="flex-1">{option.label}</span>
                  {selected && <Check className="h-4 w-4 text-[var(--accent)]" />}
                </button>
              )
            })}
          </div>

          <button
            onClick={useSystemTheme}
            className="mt-0.5 flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-sm text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
          >
            <Monitor className="h-4 w-4" />
            <span>跟随系统</span>
          </button>

          <div className="my-1 h-px bg-[var(--border)]" />

          <button
            onClick={handleLogout}
            className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-sm text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-hover)]"
          >
            <LogOut className="h-4 w-4 text-[var(--text-secondary)]" />
            <span>退出登录</span>
          </button>
        </div>
      )}

      <button
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2 rounded-xl px-2 py-2 text-left transition-colors hover:bg-[var(--bg-hover)]"
      >
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#0b7a55] text-sm font-semibold text-white">
          {(phone || 'C').slice(-2)}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium text-[var(--text-primary)]">{phone}</span>
          <span className="flex items-center gap-1 text-xs text-[var(--text-tertiary)]">
            <Settings className="h-3 w-3" />
            账号与外观
          </span>
        </span>
      </button>
    </div>
  )
}
