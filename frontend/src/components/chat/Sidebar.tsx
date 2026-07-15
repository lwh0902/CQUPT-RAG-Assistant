import { BookOpen, FileText, MoreHorizontal, PanelLeftClose, Pencil, Plus, Search, Trash2, X } from 'lucide-react'
import { useChatStore } from '../../store/chat'
import { useEffect, useRef, useState } from 'react'
import UserMenu from '../ui/UserMenu'
import cquptLogo from '../../assets/brand/cqupt-logo.png'
import { searchSessions } from '../../api/client'
import type { SessionSearchResult } from '../../api/client'
import KnowledgeBaseModal from './KnowledgeBaseModal'

interface SidebarProps {
  collapsed: boolean
  onToggle: () => void
}

export default function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const { sessions, currentSessionId, createSession, selectSession, deleteSession, renameSession } = useChatStore()
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [menuSessionId, setMenuSessionId] = useState<string | null>(null)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)

  const [searchOpen, setSearchOpen] = useState(false)
  const [searchInput, setSearchInput] = useState('')
  const [searchResults, setSearchResults] = useState<SessionSearchResult[] | null>(null)
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchError, setSearchError] = useState('')
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const [kbOpen, setKbOpen] = useState(false)

  const handleNewChat = async () => { await createSession() }

  const handleSearchChange = (value: string) => {
    setSearchInput(value)
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current)
    if (!value.trim()) {
      setSearchResults(null)
      setSearchError('')
      setSearchLoading(false)
      return
    }
    setSearchLoading(true)
    setSearchError('')
    searchTimerRef.current = setTimeout(async () => {
      try {
        const res = await searchSessions(value.trim())
        setSearchResults(res.results)
      } catch {
        setSearchError('搜索失败，请重试')
        setSearchResults([])
      } finally {
        setSearchLoading(false)
      }
    }, 300)
  }

  useEffect(() => {
    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current)
    }
  }, [])

  const showingSearch = searchOpen && searchResults !== null

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
        <button
          onClick={() => setSearchOpen((v) => !v)}
          className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors ${
            searchOpen
              ? 'bg-[var(--bg-hover)] text-[var(--text-primary)]'
              : 'text-[var(--text-primary)] hover:bg-[var(--bg-hover)]'
          }`}
        >
          <Search className="h-5 w-5 shrink-0 text-[var(--text-secondary)]" />
          <span>搜索聊天</span>
        </button>
        {searchOpen && (
          <div className="relative px-1 pt-1">
            <input
              autoFocus
              value={searchInput}
              onChange={(e) => handleSearchChange(e.target.value)}
              placeholder="输入关键词搜索历史会话"
              className="w-full rounded-lg border border-[var(--border-input)] bg-[var(--bg-input)] py-2 pl-3 pr-8 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)] focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
            />
            {searchInput && (
              <button
                onClick={() => handleSearchChange('')}
                className="absolute right-2 top-1/2 flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded text-[var(--text-tertiary)] hover:text-[var(--text-primary)]"
                aria-label="清空搜索"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        )}
        <button
          onClick={() => setKbOpen(true)}
          className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-hover)]"
        >
          <BookOpen className="h-5 w-5 shrink-0 text-[var(--text-secondary)]" />
          <span>知识库</span>
        </button>
      </div>

      <div className="px-4 pb-1 text-xs font-semibold text-[var(--text-primary)]">
        {showingSearch ? '搜索结果' : '最近'}
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-1">
        {showingSearch ? (
          <SearchResults
            results={searchResults ?? []}
            loading={searchLoading}
            error={searchError}
            onSelect={(sessionId) => {
              selectSession(sessionId)
              setSearchOpen(false)
              setSearchInput('')
              setSearchResults(null)
            }}
          />
        ) : sessions.length === 0 ? (
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
                {editingId === session.id ? (
                  <SessionTitleEditor
                    initialTitle={session.title}
                    onCancel={() => setEditingId(null)}
                    onSave={async (title) => {
                      const ok = await renameSession(session.id, title)
                      if (ok) setEditingId(null)
                    }}
                  />
                ) : (
                  <button
                    onClick={() => selectSession(session.id)}
                    onDoubleClick={() => {
                      setEditingId(session.id)
                      setMenuSessionId(null)
                    }}
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
                )}

                {menuSessionId === session.id && editingId !== session.id && (
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
                      <>
                        <button
                          onClick={(event) => {
                            event.stopPropagation()
                            setEditingId(session.id)
                            setMenuSessionId(null)
                          }}
                          className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
                        >
                          <Pencil className="h-4 w-4" />
                          <span>重命名</span>
                        </button>
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
                      </>
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

      <KnowledgeBaseModal open={kbOpen} onClose={() => setKbOpen(false)} />
    </aside>
  )
}

function SearchResults({
  results,
  loading,
  error,
  onSelect,
}: {
  results: SessionSearchResult[]
  loading: boolean
  error: string
  onSelect: (sessionId: string) => void
}) {
  if (loading) {
    return (
      <div className="flex items-center justify-center gap-2 px-3 py-8 text-sm text-[var(--text-tertiary)]">
        <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-[var(--text-tertiary)] border-t-transparent" />
        搜索中…
      </div>
    )
  }
  if (error) {
    return <div className="px-3 py-8 text-sm text-red-500">{error}</div>
  }
  if (results.length === 0) {
    return (
      <div className="px-3 py-8 text-center text-sm text-[var(--text-tertiary)]">
        <FileText className="mx-auto mb-2 h-6 w-6 opacity-50" />
        没有匹配的会话
      </div>
    )
  }
  return (
    <div className="space-y-0.5">
      {results.map((r) => (
        <button
          key={r.session_id}
          onClick={() => onSelect(r.session_id)}
          className="block w-full rounded-xl px-3 py-2.5 text-left transition-colors hover:bg-[var(--bg-hover)]"
        >
          <div className="flex items-center gap-2">
            <p className="min-w-0 flex-1 truncate text-sm font-medium text-[var(--text-primary)]">
              {r.title}
            </p>
            {r.matched_in_title && (
              <span className="shrink-0 rounded bg-[var(--accent)]/15 px-1.5 py-0.5 text-[10px] text-[var(--accent)]">
                标题
              </span>
            )}
          </div>
          {r.preview && (
            <p className="mt-1 line-clamp-2 text-xs text-[var(--text-tertiary)]">{r.preview}</p>
          )}
        </button>
      ))}
    </div>
  )
}

function SessionTitleEditor({
  initialTitle,
  onSave,
  onCancel,
}: {
  initialTitle: string
  onSave: (title: string) => void
  onCancel: () => void
}) {
  const [value, setValue] = useState(initialTitle)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const el = inputRef.current
    if (!el) return
    el.focus()
    el.select()
  }, [])

  const commit = () => {
    const trimmed = value.trim()
    if (!trimmed || trimmed === initialTitle) {
      onCancel()
      return
    }
    onSave(trimmed)
  }

  return (
    <div className="rounded-xl bg-[var(--bg-hover)] px-2 py-1.5">
      <input
        ref={inputRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onBlur={commit}
        onClick={(e) => e.stopPropagation()}
        onDoubleClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault()
            commit()
          } else if (e.key === 'Escape') {
            e.preventDefault()
            onCancel()
          }
        }}
        maxLength={100}
        className="w-full rounded-md border border-[var(--accent)] bg-[var(--bg-input)] px-2 py-1.5 text-sm text-[var(--text-primary)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
      />
    </div>
  )
}
