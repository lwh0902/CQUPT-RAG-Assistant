import type { InterviewSession } from '../../api/client'

/** Build the exportable markdown document for one interview question bank. */
export function buildInterviewMarkdown(session: InterviewSession): string {
  const lines: string[] = []
  const company = session.company || '目标公司'
  lines.push(`# 面试题库 · ${company}`)
  lines.push('')
  if (session.created_at) {
    lines.push(`生成时间：${session.created_at.slice(0, 10)}`)
    lines.push('')
  }

  const real = session.real_questions ?? []
  if (real.length > 0) {
    lines.push(`## 真实面经题（${real.length} 题，无答案）`)
    lines.push('')
    real.forEach((item, index) => {
      lines.push(`${index + 1}. ${item.question}`)
      if (item.source_title || item.source_url) {
        lines.push(`   来源：${item.source_title || ''}${item.source_url ? ` ${item.source_url}` : ''}`)
      }
    })
    lines.push('')
  }

  lines.push(`## 一、选择题（${session.mcq.length} 题）`)
  lines.push('')
  session.mcq.forEach((item, index) => {
    lines.push(`${index + 1}. ${item.question}`)
    for (const key of ['A', 'B', 'C', 'D']) {
      if (item.options[key]) lines.push(`   ${key}. ${item.options[key]}`)
    }
    lines.push(`   **答案：${item.answer}**`)
    lines.push(`   解析：${item.analysis}`)
    lines.push('')
  })

  lines.push(`## 二、简答题（${session.qa.length} 题）`)
  lines.push('')
  session.qa.forEach((item, index) => {
    const category = item.category ? `【${item.category}】` : ''
    lines.push(`${index + 1}. ${category}${item.question}`)
    lines.push(`   **口语答案：**${item.spoken_answer}`)
    lines.push(`   **题目讲解：**${item.analysis}`)
    lines.push('')
  })

  return lines.join('\n')
}
