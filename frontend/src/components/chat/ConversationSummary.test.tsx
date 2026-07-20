import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  post: vi.fn().mockResolvedValue({
    data: {
      summary: {
        topic: '奖学金申请',
        confirmed_points: ['需要准备材料'],
        open_questions: [],
        next_actions: ['提交申请'],
      },
    },
  }),
}))

vi.mock('../../api/client', () => ({ api: { post: mocks.post } }))

import ConversationSummary from './ConversationSummary'

afterEach(cleanup)

test('generates a transient visible summary only when a session has messages', async () => {
  render(<ConversationSummary sessionId="session-1" messageCount={2} />)
  fireEvent.click(screen.getByRole('button', { name: '总结当前对话' }))

  expect(await screen.findByText('奖学金申请')).toBeInTheDocument()
  expect(mocks.post).toHaveBeenCalledWith('/sessions/session-1/summary')
})

test('shows feedback when summary generation fails', async () => {
  mocks.post.mockRejectedValueOnce(new Error('network'))
  render(<ConversationSummary sessionId="session-1" messageCount={2} />)
  fireEvent.click(screen.getByRole('button', { name: '总结当前对话' }))

  expect(await screen.findByText('生成对话总结失败，请稍后重试')).toBeInTheDocument()
})
