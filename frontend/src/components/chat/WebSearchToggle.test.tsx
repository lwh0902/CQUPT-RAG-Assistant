import { fireEvent, render, screen } from '@testing-library/react'
import { expect, test, vi } from 'vitest'
import WebSearchToggle from './WebSearchToggle'

test('web search toggle reports its next one-shot state', () => {
  const onChange = vi.fn()
  render(<WebSearchToggle enabled={false} onChange={onChange} disabled={false} />)

  fireEvent.click(screen.getByRole('switch', { name: '联网搜索' }))

  expect(onChange).toHaveBeenCalledWith(true)
})
