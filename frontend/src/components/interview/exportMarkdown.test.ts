import { expect, test } from 'vitest'
import { buildInterviewMarkdown } from './exportMarkdown'
import type { InterviewSession } from '../../api/client'

const session: InterviewSession = {
  id: 's1',
  company: '字节跳动',
  created_at: '2026-07-22T10:00:00',
  mcq: [
    {
      question: 'Python GIL 的作用？',
      options: { A: '加速', B: '限制单线程执行字节码', C: '内存管理', D: '回收' },
      answer: 'B',
      analysis: 'GIL 限制同一时刻只有一个线程执行字节码。',
    },
  ],
  qa: [
    {
      question: '介绍你的检索架构',
      spoken_answer: '向量加 BM25 混合召回再精排。',
      analysis: '考察系统设计能力。',
    },
  ],
}

test('markdown export contains both banks with answers and analysis', () => {
  const md = buildInterviewMarkdown(session)
  expect(md).toContain('# 面试题库 · 字节跳动')
  expect(md).toContain('## 一、选择题（1 题）')
  expect(md).toContain('**答案：B**')
  expect(md).toContain('解析：GIL')
  expect(md).toContain('## 二、简答题（1 题）')
  expect(md).toContain('**20s 口语答案：**')
  expect(md).toContain('**题目讲解：**考察系统设计能力。')
})
