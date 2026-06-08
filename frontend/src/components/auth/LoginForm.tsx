import { useState, useCallback, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { Eye, EyeOff, Phone, Lock, ArrowRight } from 'lucide-react'
import { useAuthStore } from '../../store/auth'

export default function LoginForm() {
  const [phone, setPhone] = useState('')
  const [password, setPassword] = useState('')
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

    setIsLoading(true)
    try {
      if (phoneExists === false) {
        await register(phone.trim(), password.trim())
      } else {
        await login(phone.trim(), password.trim())
      }
      navigate('/chat')
    } catch {
      setError(phoneExists === false ? '注册失败，请重试' : '手机号或密码错误')
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

      {/* Error */}
      {error && (
        <div className="rounded-lg bg-red-500/10 px-4 py-2.5 text-sm text-red-500">
          {error}
        </div>
      )}

      {/* Submit */}
      <button
        type="submit"
        disabled={isLoading}
        className="flex w-full items-center justify-center gap-2 rounded-lg bg-[var(--accent)] py-2.5 text-sm font-medium text-[var(--text-on-accent)] transition-colors hover:bg-[var(--accent-hover)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:ring-offset-2 focus:ring-offset-[var(--bg-primary)] disabled:cursor-not-allowed disabled:opacity-60"
      >
        {isLoading ? (
          <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
        ) : (
          <>
            {submitLabel}
            <ArrowRight className="h-4 w-4" />
          </>
        )}
      </button>
    </form>
  )
}
