import type { ChatSession, DocumentInfo } from "../api";

interface Props {
  sessions: ChatSession[];
  activeSessionId: string | null;
  documents: DocumentInfo[];
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  onOpenUpload: () => void;
}

export default function Sidebar({
  sessions,
  activeSessionId,
  documents,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onOpenUpload,
}: Props) {
  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-r border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-900">
      <div className="flex items-center gap-2 px-4 py-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-sm font-semibold text-white">
          PC
        </div>
        <span className="font-semibold tracking-tight">Personal Chatter</span>
      </div>

      <div className="px-3">
        <button
          onClick={onNewSession}
          className="w-full rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white transition hover:bg-indigo-500 active:bg-indigo-700"
        >
          + New chat
        </button>
      </div>

      <nav className="scroll-thin mt-4 flex-1 overflow-y-auto px-2">
        {sessions.length === 0 && (
          <p className="px-2 py-4 text-sm text-neutral-400">No conversations yet.</p>
        )}
        <ul className="space-y-0.5">
          {sessions.map((session) => (
            <li key={session.id}>
              <div
                className={`group flex items-center justify-between rounded-lg px-2.5 py-2 text-sm transition ${
                  session.id === activeSessionId
                    ? "bg-indigo-50 text-indigo-700 dark:bg-indigo-500/10 dark:text-indigo-300"
                    : "text-neutral-700 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-800"
                }`}
              >
                <button
                  onClick={() => onSelectSession(session.id)}
                  className="min-w-0 flex-1 truncate text-left"
                  title={session.title}
                >
                  {session.title}
                </button>
                <button
                  onClick={() => onDeleteSession(session.id)}
                  className="ml-2 hidden shrink-0 text-neutral-400 hover:text-red-500 group-hover:block"
                  aria-label="Delete conversation"
                  title="Delete conversation"
                >
                  ×
                </button>
              </div>
            </li>
          ))}
        </ul>
      </nav>

      <div className="border-t border-neutral-200 p-3 dark:border-neutral-800">
        <button
          onClick={onOpenUpload}
          className="flex w-full items-center justify-between rounded-lg px-2.5 py-2 text-sm text-neutral-600 transition hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-800"
        >
          <span>📄 Documents</span>
          <span className="rounded-full bg-neutral-200 px-2 py-0.5 text-xs text-neutral-600 dark:bg-neutral-700 dark:text-neutral-300">
            {documents.length}
          </span>
        </button>
      </div>
    </aside>
  );
}
