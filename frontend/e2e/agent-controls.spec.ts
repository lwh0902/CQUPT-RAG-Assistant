import { expect, test } from '@playwright/test'

const viewports = [
  { name: 'mobile-small', width: 320, height: 568 },
  { name: 'mobile', width: 390, height: 844 },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'desktop', width: 1440, height: 900 },
]

for (const viewport of viewports) {
  test(`agent controls are usable at ${viewport.name}`, async ({ page }) => {
    await page.setViewportSize({ width: viewport.width, height: viewport.height })
    await page.addInitScript(() => {
      localStorage.setItem('token', 'test-token')
      localStorage.setItem('user_id', 'user-1')
      localStorage.setItem('phone', '18128161378')
    })

    await page.route('**/api/sessions', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({ json: { sessions: [{ id: 'session-1', title: '奖学金咨询', created_at: '2026-07-15T00:00:00Z' }] } })
      } else {
        await route.continue()
      }
    })
    await page.route('**/api/sessions/session-1/messages', (route) => route.fulfill({
      json: {
        session_id: 'session-1',
        messages: [
          { id: '1', role: 'user', content: '奖学金怎么申请？', created_at: '2026-07-15T00:00:00Z' },
          {
            id: '2',
            role: 'assistant',
            content: '需要准备材料。',
            created_at: '2026-07-15T00:01:00Z',
            confidence_level: 'high',
            evidence_summary: '依据校内资料与学校官网生成。',
            sources: [
              { source_type: 'knowledge_base', document_name: '学生手册.pdf', document_id: 'manual', page: 12, snippet: '奖学金资料' },
              { source_type: 'web', title: '学校官网通知', url: 'https://cqupt.edu.cn/notice/1', site_name: 'cqupt.edu.cn', snippet: '最新通知' },
            ],
          },
        ],
      },
    }))
    await page.route('**/api/settings/model', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({ json: { temperature: 0.3, top_p: 0.8 } })
      } else {
        await route.fulfill({ json: route.request().postDataJSON() })
      }
    })
    await page.route('**/api/sessions/session-1/summary', (route) => route.fulfill({
      json: { summary: { topic: '奖学金申请', confirmed_points: ['需要准备材料'], open_questions: [], next_actions: ['提交申请'] } },
    }))
    await page.route('**/api/memories', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          json: {
            memories: [
              { id: 1, memory_type: 'preference', memory_key: 'answer_style', memory_value: '简洁', confidence: 0.95 },
              { id: 2, memory_type: 'profile', memory_key: 'major', memory_value: '软件工程', confidence: 0.95 },
            ],
          },
        })
      } else {
        await route.continue()
      }
    })
    await page.route('**/api/memories/1', (route) => route.fulfill({ json: { id: 1, status: 'deleted' } }))
    await page.route('**/documents/**', (route) => route.fulfill({ status: 404 }))

    await page.goto('/chat')
    await expect(page.getByRole('switch', { name: '联网搜索' })).toBeVisible()
    await page.getByRole('switch', { name: '联网搜索' }).click()
    await expect(page.getByRole('switch', { name: '联网搜索' })).toHaveAttribute('aria-checked', 'true')

    await page.getByRole('button', { name: '模型设置' }).click()
    await expect(page.getByRole('dialog')).toBeVisible()
    await expect(page.getByLabel('温度')).toBeVisible()
    await page.getByRole('button', { name: '关闭' }).click()

    await page.getByRole('button', { name: '记忆管理' }).click()
    await expect(page.getByRole('dialog')).toContainText('回答偏好')
    await expect(page.getByText('软件工程')).toBeVisible()
    await page.getByRole('button', { name: '删除记忆：简洁' }).click()
    await page.getByRole('button', { name: '确认删除' }).click()
    await expect(page.getByText('简洁')).not.toBeVisible()
    await page.getByRole('button', { name: '关闭' }).click()

    await expect(page.getByText('校内知识库')).toBeVisible()
    await expect(page.getByRole('link', { name: /学校官网通知/ })).toHaveAttribute('href', 'https://cqupt.edu.cn/notice/1')
    await expect(page.getByText('置信度：高')).toBeVisible()

    await page.getByRole('button', { name: '总结当前对话' }).click()
    await expect(page.getByRole('dialog')).toContainText('奖学金申请')
    await page.getByRole('button', { name: '关闭' }).click()

    await expect(page.locator('[data-testid="chat-composer"]')).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    await page.waitForTimeout(250)
    await page.screenshot({ path: `test-results/${viewport.name}.png`, fullPage: true })
  })
}
