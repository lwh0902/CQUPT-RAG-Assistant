import { useEffect, useRef, useState } from 'react'
import Sidebar from '../components/chat/Sidebar'
import ChatMessage from '../components/chat/ChatMessage'
import ChatInput from '../components/chat/ChatInput'
import WebSearchToggle from '../components/chat/WebSearchToggle'
import ModelSettingsModal from '../components/chat/ModelSettingsModal'
import ConversationSummary from '../components/chat/ConversationSummary'
import MemoryManagerModal from '../components/chat/MemoryManagerModal'
import MemoryConfirmBanner from '../components/chat/MemoryConfirmBanner'
import { useChatStore } from '../store/chat'
import { BookOpen, FileQuestion, GraduationCap, PanelLeft, Scale } from 'lucide-react'
import cquptBg from '../assets/brand/cqupt-bg.png'
import cquptLogo from '../assets/brand/cqupt-logo.png'

const SUGGESTIONS = [
  { icon: GraduationCap, text: '奖学金申请条件是什么？' },
  { icon: BookOpen, text: '补考和重修的规定有哪些？' },
  { icon: Scale, text: '学生违纪处分有哪些类型？' },
  { icon: FileQuestion, text: '学籍异动如何办理？' },
]

export default function ChatPage() {
  const {
    sessions,
    currentSessionId,
    messages,
    isStreaming,
    thinkingSteps,
    currentReply,
    webSearchEnabled,
    fetchSessions,
    sendMessage,
    setWebSearchEnabled,
    stopStreaming,
    cleanup,
  } = useChatStore()

  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => window.innerWidth < 768)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    const init = async () => {
      await fetchSessions()
      if (cancelled) return
      const { sessions } = useChatStore.getState()
      if (sessions.length > 0) {
        const targetId = sessions[0].id
        useChatStore.setState({ currentSessionId: targetId })
        await useChatStore.getState().fetchMessages(targetId)
      }
    }
    init()
    return () => { cancelled = true; cleanup() }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, currentReply, thinkingSteps])

  const currentSession = sessions.find((s) => s.id === currentSessionId)
  const headerTitle = currentSession?.title ?? '新建对话'

  const isEmpty = messages.length === 0 && !isStreaming
  const backgroundOpacity = isEmpty ? 0.9 : 0.14
  const darkBackgroundOpacity = isEmpty ? 0.05 : 0.025

  return (
    <div className="flex h-dvh overflow-hidden bg-[var(--bg-primary)]">
      <Sidebar collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed(!sidebarCollapsed)} />

      <main className="relative flex min-w-0 flex-1 flex-col">
        <header className="relative z-10 flex h-14 shrink-0 items-center px-3 sm:px-5">
          {sidebarCollapsed && (
            <button
              onClick={() => setSidebarCollapsed(false)}
              className="mr-2 flex h-9 w-9 items-center justify-center rounded-xl text-[var(--text-tertiary)] transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-secondary)]"
              aria-label="展开侧边栏"
            >
              <PanelLeft className="h-4 w-4" />
            </button>
          )}
          <h2 className="truncate rounded-xl px-3 py-2 text-sm font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]">
            {headerTitle}
          </h2>
          <ConversationSummary sessionId={currentSessionId} messageCount={messages.length} />
          <MemoryManagerModal />
          <ModelSettingsModal />
        </header>

        <div
          className="pointer-events-none absolute inset-0 bg-cover bg-center transition-opacity duration-300 dark:hidden"
          style={{
            backgroundImage: `url(${cquptBg})`,
            opacity: backgroundOpacity,
          }}
        />
        <div
          className="pointer-events-none absolute inset-0 hidden bg-cover bg-center transition-opacity duration-300 dark:block"
          style={{
            backgroundImage: `url(${cquptBg})`,
            opacity: darkBackgroundOpacity,
          }}
        />
        <div className="pointer-events-none absolute inset-0 bg-[var(--bg-primary)]/45 dark:bg-[var(--bg-primary)]/92" />

        <div className="relative z-10 flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl px-4 pb-36 pt-8 sm:px-6">
            <div className="space-y-6">
              {isEmpty && (
                <div className="flex min-h-[calc(100dvh-260px)] flex-col items-center justify-center text-center">
                  <div className="mb-5 flex h-24 w-24 items-center justify-center overflow-hidden rounded-3xl bg-white/95 p-2 shadow-sm dark:bg-white/90">
                    <img src={cquptLogo} alt="重庆邮电大学" className="h-full w-full object-contain" />
                  </div>
                  <h1 className="text-3xl font-semibold tracking-normal text-[var(--text-primary)]">
                    有什么可以帮你？
                  </h1>
                  <p className="mt-3 mb-8 text-sm text-[var(--text-secondary)]">
                    基于校园知识库检索学生手册、学籍、奖助和办事流程
                  </p>

                  <div className="grid w-full max-w-2xl grid-cols-1 gap-3 sm:grid-cols-2">
                    {SUGGESTIONS.map((s) => (
                      <button
                        key={s.text}
                        onClick={() => sendMessage(s.text)}
                        className="flex min-h-14 items-center gap-3 rounded-2xl border border-[var(--border)] bg-[var(--surface)]/85 px-4 py-3 text-left text-sm text-[var(--text-secondary)] shadow-sm backdrop-blur transition-all hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)] active:scale-[0.99] dark:bg-[#2f2f2f]/90 dark:hover:bg-[#343434]"
                      >
                        <s.icon className="h-4 w-4 shrink-0 text-[var(--text-tertiary)]" />
                        <span className="leading-snug">{s.text}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {!isEmpty && messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                />
              ))}

              {isStreaming && (
                <ChatMessage
                  message={{
                    id: 'streaming-placeholder',
                    role: 'assistant',
                    content: '',
                    created_at: new Date().toISOString(),
                  }}
                  thinkingSteps={currentReply ? undefined : thinkingSteps}
                  streamingContent={currentReply}
                  streamingSources={useChatStore.getState().currentSources}
                  isStreaming={true}
                />
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>
        </div>

        <div className="pointer-events-none absolute inset-x-0 bottom-0 z-20 px-4 pb-4 pt-10 sm:px-6">
          <div className="pointer-events-auto mx-auto max-w-3xl">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <WebSearchToggle
                enabled={webSearchEnabled}
                disabled={isStreaming}
                onChange={setWebSearchEnabled}
              />
            </div>
            <MemoryConfirmBanner />
            <ChatInput onSend={sendMessage} isStreaming={isStreaming} onStop={stopStreaming} />
            <p className="mt-2 text-center text-xs text-[var(--text-tertiary)]">
              CQUPT RAG 可能会犯错，请核查重要信息。
            </p>
          </div>
        </div>
      </main>
    </div>
  )
}
