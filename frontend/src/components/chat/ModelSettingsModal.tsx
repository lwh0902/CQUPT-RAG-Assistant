import { SlidersHorizontal } from 'lucide-react'
import { useState } from 'react'
import { api } from '../../api/client'
import { useToastStore } from '../../store/toast'
import Modal from '../ui/Modal'

export default function ModelSettingsModal() {
  const [open, setOpen] = useState(false)
  const [temperature, setTemperature] = useState(0.3)
  const [topP, setTopP] = useState(0.8)
  const [loaded, setLoaded] = useState(false)

  const openModal = async () => {
    setOpen(true)
    if (loaded) return
    try {
      const { data } = await api.get<{ temperature: number; top_p: number }>('/settings/model')
      setTemperature(data.temperature)
      setTopP(data.top_p)
      setLoaded(true)
    } catch {
      setLoaded(true)
      useToastStore.getState().addToast('加载模型设置失败，已使用默认值', 'error')
    }
  }

  const save = async (nextTemperature = temperature, nextTopP = topP) => {
    try {
      const { data } = await api.put<{ temperature: number; top_p: number }>('/settings/model', {
        temperature: nextTemperature,
        top_p: nextTopP,
      })
      setTemperature(data.temperature)
      setTopP(data.top_p)
    } catch {
      useToastStore.getState().addToast('模型设置保存失败，请重试', 'error')
    }
  }

  return (
    <>
      <button
        type="button"
        onClick={openModal}
        className="ml-auto flex h-9 w-9 items-center justify-center rounded-lg text-[var(--text-tertiary)] transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
        aria-label="模型设置"
        title="模型设置"
      >
        <SlidersHorizontal className="h-4 w-4" />
      </button>
      <Modal open={open} onClose={() => setOpen(false)} title="模型设置" className="max-w-md">
        <div className="space-y-6 p-5">
          <p className="text-xs leading-5 text-[var(--text-tertiary)]">
            制度问答建议保持偏低设置，回答会更稳、更贴材料；调高后更灵活，但更容易发挥。
          </p>
          <label className="block space-y-2 text-sm text-[var(--text-primary)]">
            <span className="flex items-center justify-between"><span>温度</span><span className="text-[var(--text-secondary)]">{temperature.toFixed(1)}</span></span>
            <input
              aria-label="温度"
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={temperature}
              onChange={(event) => setTemperature(Number(event.target.value))}
              onPointerUp={() => save()}
              className="w-full accent-emerald-500"
            />
            <span className="block text-xs leading-5 text-[var(--text-tertiary)]">
              控制回答稳不稳：越低越严谨、少发挥；越高越活泼，也可能更不准。
            </span>
          </label>
          <label className="block space-y-2 text-sm text-[var(--text-primary)]">
            <span className="flex items-center justify-between"><span>多样性</span><span className="text-[var(--text-secondary)]">{topP.toFixed(1)}</span></span>
            <input
              aria-label="多样性"
              type="range"
              min="0.1"
              max="1"
              step="0.1"
              value={topP}
              onChange={(event) => setTopP(Number(event.target.value))}
              onPointerUp={() => save()}
              className="w-full accent-emerald-500"
            />
            <span className="block text-xs leading-5 text-[var(--text-tertiary)]">
              控制用词宽不宽：越低越集中、更单一；越高表达更丰富，但可能更绕。
            </span>
          </label>
        </div>
      </Modal>
    </>
  )
}
