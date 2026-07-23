import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { Eye, EyeOff, Phone, Lock, ArrowRight, Ticket } from 'lucide-react'
import { useAuthStore } from '../../store/auth'
import { Button } from '@/components/ui/neon-button'

type Mode = 'login' | 'register'

function normalizeInvite(raw: string) {
  return raw.toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 6)
}

export default function LoginForm() {
  const [mode, setMode] = useState<Mode>('login')
  const [phone, setPhone] = useState('')
  const [password, setPassword] = useState('')
  const [inviteCode, setInviteCode] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const { login, register } = useAuthStore()
  const navigate = useNavigate()

  const switchMode = (next: Mode) => {
    setMode(next)
    setError('')
    setPassword('')
    setInviteCode('')
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError('')

    if (phone.trim().length !== 11) {
      setError('请输入正确的11位手机号')
      return
    }
    if (!password.trim()) {
      setError(mode === 'register' ? '请设置密码' : '请输入密码')
      return
    }
    if (mode === 'register') {
      const code = normalizeInvite(inviteCode)
      if (code.length !== 6) {
        setError('注册需要填写 6 位邀请码')
        return
      }
    }

    setIsLoading(true)
    try {
      if (mode === 'register') {
        await register(phone.trim(), password.trim(), normalizeInvite(inviteCode))
      } else {
        await login(phone.trim(), password.trim())
      }
      navigate('/chat')
    } catch (err: any) {
      const detail = err?.response?.data?.detail
      if (typeof detail === 'string' && detail) {
        setError(detail)
      } else {
        setError(mode === 'register' ? '注册失败，请检查邀请码后重试' : '手机号或密码错误')
      }
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-5">
      {/* Login / Register toggle */}
      <div className="grid grid-cols-2 rounded-xl border border-white/10 bg-white/[0.03] p-1">
        <button
          type="button"
          onClick={() => switchMode('login')}
          className={`rounded-lg py-2 text-sm font-medium transition-colors ${
            mode === 'login'
              ? 'bg-emerald-500/20 text-emerald-300 shadow-sm ring-1 ring-emerald-500/30'
              : 'text-neutral-400 hover:text-neutral-200'
          }`}
        >
          登录
        </button>
        <button
          type="button"
          onClick={() => switchMode('register')}
          className={`rounded-lg py-2 text-sm font-medium transition-colors ${
            mode === 'register'
              ? 'bg-emerald-500/20 text-emerald-300 shadow-sm ring-1 ring-emerald-500/30'
              : 'text-neutral-400 hover:text-neutral-200'
          }`}
        >
          注册
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        <div>
          <label className="mb-1.5 block text-xs font-medium text-[var(--text-secondary)]">
            手机号
          </label>
          <div className="group relative">
            <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3.5">
              <Phone className="h-[18px] w-[18px] text-[var(--text-tertiary)]" />
            </div>
            <input
              type="tel"
              value={phone}
              onChange={(e) => {
                const val = e.target.value.replace(/\D/g, '')
                if (val.length <= 11) setPhone(val)
              }}
              placeholder="请输入11位手机号"
              className="w-full rounded-lg border border-[var(--border-input)] bg-[var(--bg-input)] py-2.5 pl-11 pr-3 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)] transition-colors focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
            />
          </div>
        </div>

        <div>
          <label className="mb-1.5 block text-xs font-medium text-[var(--text-secondary)]">
            {mode === 'register' ? '设置密码' : '密码'}
          </label>
          <div className="group relative">
            <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3.5">
              <Lock className="h-[18px] w-[18px] text-[var(--text-tertiary)]" />
            </div>
            <input
              type={showPassword ? 'text' : 'password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={mode === 'register' ? '请设置登录密码' : '请输入密码'}
              className="w-full rounded-lg border border-[var(--border-input)] bg-[var(--bg-input)] py-2.5 pl-11 pr-11 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)] transition-colors focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute inset-y-0 right-0 flex items-center pr-3.5 text-[var(--text-tertiary)] transition-colors hover:text-[var(--text-secondary)]"
            >
              {showPassword ? <EyeOff className="h-[18px] w-[18px]" /> : <Eye className="h-[18px] w-[18px]" />}
            </button>
          </div>
        </div>

        {mode === 'register' && (
          <div>
            <label className="mb-1.5 block text-xs font-medium text-[var(--text-secondary)]">
              邀请码（6 位）
            </label>
            <div className="group relative">
              <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3.5">
                <Ticket className="h-[18px] w-[18px] text-[var(--text-tertiary)]" />
              </div>
              <input
                type="text"
                value={inviteCode}
                onChange={(e) => setInviteCode(normalizeInvite(e.target.value))}
                placeholder="请输入邀请码"
                autoComplete="off"
                className="w-full rounded-lg border border-[var(--border-input)] bg-[var(--bg-input)] py-2.5 pl-11 pr-3 text-sm tracking-widest text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)] placeholder:tracking-normal transition-colors focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
              />
            </div>
            <p className="mt-1.5 text-[11px] text-[var(--text-tertiary)]">
              需有效邀请码才能注册（7 天内有效，一次性）
            </p>
          </div>
        )}

        {error && (
          <div className="rounded-lg bg-red-500/10 px-4 py-2.5 text-sm text-red-500">
            {error}
          </div>
        )}

        <Button
          type="submit"
          variant="solid"
          size="lg"
          disabled={isLoading}
          className="w-full justify-center gap-2 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isLoading ? (
            <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
          ) : (
            <>
              {mode === 'register' ? '注册并登录' : '登录'}
              <ArrowRight className="h-4 w-4" />
            </>
          )}
        </Button>

        <p className="text-center text-xs text-neutral-500">
          {mode === 'login' ? (
            <>
              还没有账号？
              <button
                type="button"
                onClick={() => switchMode('register')}
                className="ml-1 text-emerald-400 hover:underline"
              >
                去注册
              </button>
            </>
          ) : (
            <>
              已有账号？
              <button
                type="button"
                onClick={() => switchMode('login')}
                className="ml-1 text-emerald-400 hover:underline"
              >
                去登录
              </button>
            </>
          )}
        </p>
      </form>
    </div>
  )
}
