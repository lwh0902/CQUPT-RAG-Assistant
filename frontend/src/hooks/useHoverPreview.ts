import { useEffect, useRef, useState } from 'react'

interface HoverPreviewState {
  visible: boolean
  url: string | null
}

interface UseHoverPreviewOptions {
  /** Delay in ms before showing the preview. */
  delay?: number
}

/**
 * Debounced hover preview hook.
 *
 * Call `onEnter(url)` when the pointer enters a target. After `delay` ms the
 * preview becomes visible. Call `onLeave()` immediately on pointer leave to
 * cancel a pending show or hide an already-visible preview.
 */
export function useHoverPreview({ delay = 500 }: UseHoverPreviewOptions = {}) {
  const [state, setState] = useState<HoverPreviewState>({
    visible: false,
    url: null,
  })
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pendingUrlRef = useRef<string | null>(null)

  const clearTimer = () => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }

  const onEnter = (url: string) => {
    pendingUrlRef.current = url
    clearTimer()
    timerRef.current = setTimeout(() => {
      setState({ visible: true, url: pendingUrlRef.current })
      timerRef.current = null
    }, delay)
  }

  const onLeave = () => {
    clearTimer()
    pendingUrlRef.current = null
    setState({ visible: false, url: null })
  }

  useEffect(() => () => clearTimer(), [])

  return { ...state, onEnter, onLeave }
}
