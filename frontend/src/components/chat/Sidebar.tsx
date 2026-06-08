import { BookOpen, MoreHorizontal, PanelLeftClose, Plus, Search, Trash2 } from 'lucide-react'
import { useChatStore } from '../../store/chat'
import { useState } from 'react'
import UserMenu from '../ui/UserMenu'
import cquptLogo from '../../assets/brand/cqupt-logo.png'

interface SidebarProps {
  collapsed: boolean
  onToggle: () => void
}

export default function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const { sessions, currentSessionId, createSession, selectSession, deleteSession } = useChatStore()
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [menuSessionId, setMenuSessionId] = useState<string | null>(null)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)

  const handleNewChat = async () => { await createSession() }

  return (
    <aside
      className="flex h-full flex-col border-r border-[var(--border)] bg-[var(--bg-sidebar)] transition-all duration-300 ease-in-out"
      style={{ width: collapsed ? '0px' : undefined, minWidth: collapsed ? '0px' : '260px', maxWidth: collapsed ? '0px' : '260px', overflow: 'hidden' }}
    >
      <div className="flex h-14 shrink-0 items-center justify-between px-3">
        <div className="flex min-w-0 items-center gap-2">
          <img src={cquptLogo} alt="CQUPT" className="h-8 w-8 shrink-0 rounded-lg object-contain" />
          <span className="truncate text-base font-semibold text-[var(--text-primary)]">CQUPT RAG</span>
        </div>
        <button
          onClick={onToggle}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-[var(--text-tertiary)] transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-secondary)]"
          aria-label="收起侧边栏"
        >
          <PanelLeftClose className="h-4 w-4" />
        </button>
      </div>

      <div className="shrink-0 space-y-1 px-2 pb-3">
        <button
          onClick={handleNewChat}
          className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-hover)]"
        >
          <Plus className="h-5 w-5 shrink-0 text-[var(--text-secondary)]" />
          <span>新建对话</span>
        </button>
        <button className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-hover)]">
          <Search className="h-5 w-5 shrink-0 text-[var(--text-secondary)]" />
          <span>搜索聊天</span>
        </button>
        <button className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-hover)]">
          <BookOpen className="h-5 w-5 shrink-0 text-[var(--text-secondary)]" />
          <span>知识库</span>
        </button>
      </div>

      <div className="px-4 pb-1 text-xs font-semibold text-[var(--text-primary)]">最近</div>

      <div className="flex-1 overflow-y-auto px-2 py-1">
        {sessions.length === 0 ? (
          <div className="px-3 py-8 text-sm text-[var(--text-tertiary)]">暂无会话</div>
        ) : (
          <div className="space-y-0.5">
            {sessions.map((session) => (
              <div
                key={session.id}
                className="group relative"
                onMouseEnter={() => setHoveredId(session.id)}
                onMouseLeave={() => setHoveredId(null)}
              >
                <button
                  onClick={() => selectSession(session.id)}
                  className={`flex w-full items-center gap-2 rounded-xl px-3 py-2.5 text-left text-sm transition-colors ${
                    session.id === currentSessionId
                      ? 'bg-[var(--bg-hover)] text-[var(--text-primary)]'
                      : 'text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]'
                  }`}
                >
                  <p className="min-w-0 flex-1 truncate leading-snug">
                    {session.title}
                  </p>
                  {(hoveredId === session.id || menuSessionId === session.id) && (
                    <span
                      onClick={(event) => {
                        event.stopPropagation()
                        setMenuSessionId((value) => value === session.id ? null : session.id)
                        setConfirmDeleteId(null)
                      }}
                      className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-[var(--text-tertiary)] hover:bg-[var(--bg-tertiary)] hover:text-[var(--text-primary)]"
                      role="button"
                      aria-label="会话操作"
                    >
                      <MoreHorizontal className="h-4 w-4" />
                    </span>
                  )}
                </button>

                {menuSessionId === session.id && (
                  <div className="absolute right-2 top-10 z-20 w-36 overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--surface-raised)] p-1 shadow-[0_12px_32px_rgba(0,0,0,0.16)]">
                    {confirmDeleteId === session.id ? (
                      <div className="p-2">
                        <p className="mb-2 text-xs text-[var(--text-secondary)]">确认删除该会话？</p>
                        <div className="flex gap-1.5">
                          <button
                            onClick={(event) => {
                              event.stopPropagation()
                              setConfirmDeleteId(null)
                            }}
                            className="flex-1 rounded-lg px-2 py-1.5 text-xs text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]"
                          >
                            取消
                          </button>
                          <button
                            onClick={async (event) => {
                              event.stopPropagation()
                              await deleteSession(session.id)
                              setMenuSessionId(null)
                              setConfirmDeleteId(null)
                            }}
                            className="flex-1 rounded-lg bg-red-500 px-2 py-1.5 text-xs text-white hover:bg-red-600"
                          >
                            删除
                          </button>
                        </div>
                      </div>
                    ) : (
                      <button
                        onClick={(event) => {
                          event.stopPropagation()
                          setConfirmDeleteId(session.id)
                        }}
                        className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm text-red-500 transition-colors hover:bg-red-500/10"
                      >
                        <Trash2 className="h-4 w-4" />
                        <span>删除</span>
                      </button>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="shrink-0 border-t border-[var(--border)] p-3">
        <UserMenu />
      </div>
    </aside>
  )
}
