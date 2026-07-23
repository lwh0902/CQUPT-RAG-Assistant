import { useT } from '../../i18n'
import type { InterviewGenerateStage } from '../../api/client'

/**
 * Real-stage loading animation for interview-bank generation.
 * Progress + message come from backend SSE; the hamster runs on the track.
 */

type Props = {
  stage?: InterviewGenerateStage | null
}

export default function GenerationRunner({ stage }: Props) {
  const t = useT()
  const progress = Math.max(0, Math.min(100, stage?.progress ?? 2))
  const message = stage?.message || t('interview.stageSearch')

  return (
    <div className="mt-4 rounded-2xl border border-[var(--border)] bg-[var(--surface)]/80 px-5 py-4">
      <div className="mb-1 flex items-center justify-between gap-3 text-xs">
        <span className="min-w-0 flex-1 truncate font-medium text-[var(--text-secondary)]">{message}</span>
        <span className="shrink-0 tabular-nums text-[var(--text-tertiary)]">{Math.floor(progress)}%</span>
      </div>

      <div className="relative h-10">
        <div className="absolute inset-x-0 bottom-2 h-2.5 overflow-hidden rounded-full bg-[var(--bg-secondary)]">
          <div className="runner-fill h-full rounded-full" style={{ width: `${progress}%` }} />
        </div>
        <div
          className="runner-animal absolute bottom-2 text-2xl transition-[left] duration-500 ease-out"
          style={{ left: `calc(${progress}% - 14px)` }}
          aria-hidden
        >
          🐹
        </div>
        <div
          className="runner-sparkle absolute bottom-3 text-sm transition-[left] duration-500 ease-out"
          style={{ left: `calc(${Math.max(progress - 4, 0)}% - 20px)` }}
          aria-hidden
        >
          ✨
        </div>
      </div>

      <p className="text-center text-[11px] text-[var(--text-tertiary)]">{t('interview.generatingHint')}</p>

      <style>{`
        .runner-fill {
          background: linear-gradient(90deg, var(--accent), #34d399, var(--accent));
          background-size: 200% 100%;
          animation: runner-stripes 1.2s linear infinite;
          transition: width 0.45s ease-out;
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
          from { transform: translateY(0); }
          to { transform: translateY(-6px); }
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
