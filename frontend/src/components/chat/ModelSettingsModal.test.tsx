import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { expect, test, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  get: vi.fn().mockResolvedValue({ data: { temperature: 0.3, top_p: 0.8 } }),
  put: vi.fn().mockResolvedValue({ data: { temperature: 0.6, top_p: 0.8 } }),
}))

vi.mock('../../api/client', () => ({ api: { get: mocks.get, put: mocks.put } }))

import ModelSettingsModal from './ModelSettingsModal'

test('loads and saves account model settings from slider controls', async () => {
  render(<ModelSettingsModal />)
  fireEvent.click(screen.getByRole('button', { name: '模型设置' }))

  await screen.findByLabelText('温度')
  fireEvent.change(screen.getByLabelText('温度'), { target: { value: '0.6' } })
  fireEvent.pointerUp(screen.getByLabelText('温度'))

  await waitFor(() => {
    expect(mocks.put).toHaveBeenCalledWith('/settings/model', { temperature: 0.6, top_p: 0.8 })
  })
})
