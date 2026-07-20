import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  get: vi.fn().mockResolvedValue({
    data: {
      memories: [
        { id: 1, memory_type: 'preference', memory_key: 'answer_style', memory_value: '简洁', confidence: 0.95 },
        { id: 2, memory_type: 'profile', memory_key: 'major', memory_value: '软件工程', confidence: 0.95 },
      ],
    },
  }),
  delete: vi.fn().mockResolvedValue({ data: { id: 1, status: 'deleted' } }),
}))

vi.mock('../../api/client', () => ({ api: { get: mocks.get, delete: mocks.delete } }))

import MemoryManagerModal from './MemoryManagerModal'

afterEach(cleanup)

test('shows active memories by type and removes a confirmed memory', async () => {
  render(<MemoryManagerModal />)
  fireEvent.click(screen.getByRole('button', { name: '记忆管理' }))

  expect(await screen.findByText('回答偏好')).toBeInTheDocument()
  expect(screen.getByText('简洁')).toBeInTheDocument()
  expect(screen.getByText('软件工程')).toBeInTheDocument()

  fireEvent.click(screen.getByRole('button', { name: '删除记忆：简洁' }))
  expect(screen.getByText('删除这条记忆？')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '确认删除' }))

  await waitFor(() => {
    expect(mocks.delete).toHaveBeenCalledWith('/memories/1')
  })
  expect(screen.queryByText('简洁')).not.toBeInTheDocument()
  expect(screen.getByText('软件工程')).toBeInTheDocument()
})

test('shows retryable feedback when memories cannot be loaded', async () => {
  mocks.get.mockRejectedValueOnce(new Error('network'))
  render(<MemoryManagerModal />)
  fireEvent.click(screen.getByRole('button', { name: '记忆管理' }))

  expect(await screen.findByText('加载记忆失败，请重试')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: '重试加载记忆' })).toBeInTheDocument()
})
