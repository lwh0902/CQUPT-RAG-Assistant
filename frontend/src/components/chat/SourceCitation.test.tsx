import { render, screen } from '@testing-library/react'
import { expect, test } from 'vitest'
import SourceCitation from './SourceCitation'

test('groups knowledge-base and web sources with a real external link', () => {
  render(
    <SourceCitation
      confidenceLevel="high"
      evidenceSummary="依据校内资料与学校官网生成。"
      uncertainPoints={[]}
      sources={[
        {
          source_type: 'knowledge_base',
          document_name: '学生手册.pdf',
          document_id: 'student-manual',
          page: 12,
          snippet: '奖学金资料',
        },
        {
          source_type: 'web',
          title: '学校官网通知',
          site_name: 'cqupt.edu.cn',
          url: 'https://cqupt.edu.cn/notice/1',
          snippet: '最新通知',
        },
      ]}
    />,
  )

  expect(screen.getByText('校内知识库')).toBeInTheDocument()
  expect(screen.getByText('网络来源')).toBeInTheDocument()
  expect(screen.getByRole('link', { name: /学校官网通知/ })).toHaveAttribute(
    'href',
    'https://cqupt.edu.cn/notice/1',
  )
  expect(screen.getByText('置信度：高')).toBeInTheDocument()
})
