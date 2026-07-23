import { useState, useCallback, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { Eye, EyeOff, Phone, Lock, ArrowRight, Ticket } from 'lucide-react'
import { useAuthStore } from '../../store/auth'
import { Button } from '@/components/ui/neon-button'

export default function LoginForm() {
  const [phone, setPhone] = useState('')
  const [password, setPassword] = useState('')
  const [inviteCode, setInviteCode] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [phoneExists, setPhoneExists] = useState<boolean | null>(null)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const { checkPhone, login, register } = useAuthStore()
  const navigate = useNavigate()

  const handlePhoneBlur = useCallback(async () => {
    const trimmed = phone.trim()
    if (trimmed.length === 11) {
      try {
        const exists = await checkPhone(trimmed)
        setPhoneExists(exists)
      } catch {
        setError('无法验证手机号，请检查网络')
      }
    }
  }, [phone, checkPhone])

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError('')

    if (phone.trim().length !== 11) {
      setError('请输入正确的11位手机号')
      return
    }
    if (!password.trim()) {
      setError('请输入密码')
      return
    }
    if (phoneExists === false) {
      const code = inviteCode.trim().toUpperCase().replace(/[^A-Z0-9]/g, '')
      if (code.length !== 6) {
        setError('注册需要填写 6 位邀请码')
        return
      }
    }

    setIsLoading(true)
    try {
      if (phoneExists === false) {
        await register(
          phone.trim(),
          password.trim(),
          inviteCode.trim().toUpperCase().replace(/[^A-Z0-9]/g, ''),
        )
      } else {
        await login(phone.trim(), password.trim())
      }
      navigate('/chat')
    } catch (err: any) {
      const detail = err?.response?.data?.detail
      if (typeof detail === 'string' && detail) {
        setError(detail)
      } else {
        setError(phoneExists === false ? '注册失败，请检查邀请码后重试' : '手机号或密码错误')
      }
    } finally {
      setIsLoading(false)
    }
  }

  const passwordLabel = phoneExists === false ? '设置密码' : '输入密码'
  const submitLabel = phoneExists === false ? '注册并登录' : '登录'

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      {/* Phone */}
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
              if (val.length <= 11) {
                setPhone(val)
                if (val.length !== 11) setPhoneExists(null)
              }
            }}
            onBlur={handlePhoneBlur}
            placeholder="请输入11位手机号"
            className="w-full rounded-lg border border-[var(--border-input)] bg-[var(--bg-input)] py-2.5 pl-11 pr-3 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)] transition-colors focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
          />
        </div>
      </div>

      {/* Password */}
      <div>
        <label className="mb-1.5 block text-xs font-medium text-[var(--text-secondary)]">
          {passwordLabel}
        </label>
        <div className="group relative">
          <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3.5">
            <Lock className="h-[18px] w-[18px] text-[var(--text-tertiary)]" />
          </div>
          <input
            type={showPassword ? 'text' : 'password'}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="请输入密码"
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

      {/* Invite code — only when registering a new phone */}
      {phoneExists === false && (
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
              onChange={(e) => {
                const val = e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, '')
                if (val.length <= 6) setInviteCode(val)
              }}
              placeholder="请输入邀请码"
              autoComplete="off"
              className="w-full rounded-lg border border-[var(--border-input)] bg-[var(--bg-input)] py-2.5 pl-11 pr-3 text-sm tracking-widest text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)] placeholder:tracking-normal transition-colors focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
            />
          </div>
          <p className="mt-1.5 text-[11px] text-[var(--text-tertiary)]">
            新用户注册需要有效邀请码（7 天内、一次性）
          </p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="rounded-lg bg-red-500/10 px-4 py-2.5 text-sm text-red-500">
          {error}
        </div>
      )}

      {/* Submit */}
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
            {submitLabel}
            <ArrowRight className="h-4 w-4" />
          </>
        )}
      </Button>
    </form>
  )
}
