import { GraduationCap } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Spotlight } from '@/components/ui/spotlight'
import { SplineScene } from '@/components/ui/splite'
import LoginForm from '@/components/auth/LoginForm'

export default function LoginPage() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-black p-4 sm:p-6">
      <Card className="relative flex min-h-[calc(100vh-2rem)] w-full max-w-[calc(100vw-2rem)] flex-col overflow-hidden border-white/[0.08] bg-black/[0.96] sm:min-h-[calc(100vh-3rem)] sm:max-w-[calc(100vw-3rem)]">
        <Spotlight
          className="-top-40 left-0 md:-top-20 md:left-60"
          fill="white"
        />

        <div className="flex min-h-0 flex-1 flex-col md:flex-row">
          {/* 左侧：登录表单 */}
          <div className="dark relative z-10 flex flex-1 flex-col justify-center p-6 sm:p-8 md:p-12">
            <div className="mb-6 flex items-center gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-500/15 ring-1 ring-emerald-500/30">
                <GraduationCap className="h-5 w-5 text-emerald-400" />
              </div>
              <h1 className="bg-gradient-to-b from-neutral-50 to-neutral-400 bg-clip-text text-xl font-bold text-transparent sm:text-2xl">
                重邮极客 Agent
              </h1>
            </div>
            <p className="mb-6 text-sm text-neutral-400 sm:mb-8">
              校园智能问答 · 登录后开始对话
            </p>
            <LoginForm />
          </div>

          {/* 右侧：3D 场景（移动端隐藏） */}
          <div className="relative hidden min-h-0 flex-1 md:block">
            <SplineScene
              scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
              className="h-full w-full"
            />
          </div>
        </div>
      </Card>
    </div>
  )
}
