import { expect, test } from 'vitest'
import { zh } from './zh'
import { en } from './en'
import { translate, useI18nStore } from './index'

test('zh and en dictionaries have identical keys', () => {
  const zhKeys = Object.keys(zh).sort()
  const enKeys = Object.keys(en).sort()
  expect(enKeys).toEqual(zhKeys)
})

test('translate falls back to zh and respects language', () => {
  expect(translate('chat.send', 'zh-CN')).toBe('发送消息')
  expect(translate('chat.send', 'en-US')).toBe('Send message')
})

test('language store defaults to zh-CN and switches', () => {
  expect(useI18nStore.getState().lang).toBe('zh-CN')
  useI18nStore.getState().setLang('en-US')
  expect(useI18nStore.getState().lang).toBe('en-US')
  useI18nStore.getState().setLang('zh-CN')
})
