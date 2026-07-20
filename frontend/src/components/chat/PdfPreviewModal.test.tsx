import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  get: vi.fn().mockResolvedValue({ data: new Blob(['page']) }),
}))

vi.mock('../../api/client', () => ({ api: { get: mocks.get } }))

import PdfPreviewModal from './PdfPreviewModal'

afterEach(() => {
  mocks.get.mockClear()
})

test('changes to the next page after an upward swipe on a navigable preview', async () => {
  render(
    <PdfPreviewModal
      source={{ document_id: 'manual', document_name: '学生手册', page: 1 }}
      totalPages={3}
      navigable
      onClose={() => {}}
    />,
  )

  await waitFor(() => expect(mocks.get).toHaveBeenCalledWith('/documents/manual/page/1', { responseType: 'blob' }))
  mocks.get.mockClear()

  const preview = screen.getByLabelText('PDF 页面预览')
  const pointerDown = new Event('pointerdown', { bubbles: true })
  Object.defineProperty(pointerDown, 'clientY', { value: 500 })
  fireEvent(preview, pointerDown)
  const pointerUp = new Event('pointerup', { bubbles: true })
  Object.defineProperty(pointerUp, 'clientY', { value: 300 })
  fireEvent(preview, pointerUp)

  await waitFor(() => expect(mocks.get).toHaveBeenCalledWith('/documents/manual/page/2', { responseType: 'blob' }))
  expect(screen.getByText(/第 2 页/)).toBeInTheDocument()
})
