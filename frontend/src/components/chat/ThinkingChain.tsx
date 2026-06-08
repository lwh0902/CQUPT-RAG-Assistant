import type { ThinkingStep } from '../../store/chat'

interface ThinkingChainProps {
  steps: ThinkingStep[]
}

const stepLabels: Record<string, string> = {
  analyzing: '正在分析问题',
  retrieving: '正在检索知识库',
  retrieved: '正在整理参考资料',
  generating: '正在生成回答',
}

function getCurrentStep(steps: ThinkingStep[]) {
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    if (steps[index].status === 'in_progress') return steps[index]
  }

  return steps[steps.length - 1]
}

export default function ThinkingChain({ steps }: ThinkingChainProps) {
  const currentStep = getCurrentStep(steps)

  if (!currentStep) return null

  const label =
    stepLabels[currentStep.step] ??
    currentStep.message ??
    '正在处理请求'

  return (
    <div className="mb-3 flex items-center gap-3 text-sm text-[var(--text-secondary)]">
      <span className="flex w-8 items-center justify-center gap-1" aria-hidden="true">
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-current [animation-delay:-0.24s]" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-current [animation-delay:-0.12s]" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-current" />
      </span>
      <span className="text-base leading-7">{label}</span>
    </div>
  )
}
