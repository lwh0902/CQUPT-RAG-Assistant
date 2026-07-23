import { useEffect, useMemo, useState } from 'react'
import { useT } from '../../i18n'

/**
 * Cute indeterminate loading animation for interview-bank generation.
 * The backend takes ~2-3 minutes (reference search → MCQ → review → QA chunks),
 * so we simulate progress over ~100s and rotate stage text. The animal runs on
 * the track and bobs while "generating".
 */

const STAGE_KEYS = [
  'interview.stageSearch',
  'interview.stageMcq',
  'interview.stageReview',
  'interview.stageQa',
  'interview.stageGuide',
  'interview.stageFinal',
] as const

function stageIndexFor(progress: number): number {
  if (progress < 12) return 0
  if (progress < 35) return 1
  if (progress < 50) return 2
  if (progress < 78) return 3
  if (progress < 92) return 4
  return 5
}

export default function GenerationRunner() {
  const t = useT()
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const startedAt = Date.now()
    const timer = window.setInterval(() => {
      setElapsed((Date.now() - startedAt) / 1000)
    }, 200)
    return () => window.clearInterval(timer)
  }, [])

  // Simulated: fast to 25% (search), then ease toward 92% over ~100s.
  const progress = useMemo(() => {
    if (elapsed < 5) return (elapsed / 5) * 25
    return 25 + (92 - 25) * (1 - Math.exp(-(elapsed - 5) / 55))
  }, [elapsed])

  const stageKey = STAGE_KEYS[stageIndexFor(progress)]

  return (
    <div className="mt-4 rounded-2xl border border-[var(--border)] bg-[var(--surface)]/80 px-5 py-4">
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="font-medium text-[var(--text-secondary)]">{t(stageKey)}</span>
        <span className="tabular-nums text-[var(--text-tertiary)]">{Math.floor(progress)}%</span>
      </div>

      {/* Track with running animal */}
      <div className="relative h-10">
        <div className="absolute inset-x-0 bottom-2 h-2.5 overflow-hidden rounded-full bg-[var(--bg-secondary)]">
          <div
            className="runner-fill h-full rounded-full"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div
          className="runner-animal absolute bottom-2 text-2xl transition-[left] duration-300 ease-linear"
          style={{ left: `calc(${progress}% - 14px)` }}
          aria-hidden
        >
          🐹
        </div>
        <div
          className="runner-sparkle absolute bottom-3 text-sm transition-[left] duration-300 ease-linear"
          style={{ left: `calc(${progress}% - 34px)` }}
          aria-hidden
        >
          ✨
        </div>
      </div>

      <p className="text-center text-[11px] text-[var(--text-tertiary)]">
        {t('interview.generatingHint')}
      </p>

      <style>{`
        .runner-fill {
          background: linear-gradient(90deg, var(--accent), #34d399, var(--accent));
          background-size: 200% 100%;
          animation: runner-stripes 1.2s linear infinite;
          transition: width 0.3s ease-out;
        }
        @keyframes runner-stripes {
          from { background-position: 0% 0; }
          to { background-position: 200% 0; }
        }
        .runner-animal {
          animation: runner-bob 0.5s ease-in-out infinite alternate;
          transform-origin: bottom center;
        }
        @keyframes runner-bob {
          from { transform: translateY(0) scaleX(1); }
          to { transform: translateY(-6px) scaleX(1); }
        }
        .runner-sparkle {
          animation: runner-twinkle 0.9s ease-in-out infinite alternate;
        }
        @keyframes runner-twinkle {
          from { opacity: 0.2; transform: scale(0.8); }
          to { opacity: 1; transform: scale(1.15); }
        }
      `}</style>
    </div>
  )
}
