import { useCallback, useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";
import UploadDropzone from "./components/UploadDropzone";
import {
  type ChatMessage,
  type ChatSession,
  type DocumentInfo,
  createSession,
  deleteSession,
  getMessages,
  listDocuments,
  listSessions,
  streamMessage,
  uploadDocuments,
} from "./api";

export default function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);

  const [streamingContent, setStreamingContent] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const refreshSessions = useCallback(() => listSessions().then(setSessions), []);

  const refreshDocuments = useCallback((sessionId: string | null) => {
    if (!sessionId) {
      setDocuments([]);
      return Promise.resolve();
    }
    return listDocuments(sessionId).then(setDocuments);
  }, []);

  useEffect(() => {
    refreshSessions().catch(console.error);
  }, [refreshSessions]);

  // Each chat session has its own document set -- switching sessions
  // means both its message history and its documents change.
  useEffect(() => {
    if (!activeSessionId) {
      setMessages([]);
      setDocuments([]);
      return;
    }
    const onError = (err: unknown) => setChatError(err instanceof Error ? err.message : String(err));
    getMessages(activeSessionId).then(setMessages).catch(onError);
    refreshDocuments(activeSessionId).catch(onError);
  }, [activeSessionId, refreshDocuments]);

  async function ensureActiveSession(): Promise<string> {
    if (activeSessionId) return activeSessionId;
    const session = await createSession();
    setSessions((prev) => [session, ...prev]);
    setActiveSessionId(session.id);
    return session.id;
  }

  async function handleNewSession() {
    const session = await createSession();
    setSessions((prev) => [session, ...prev]);
    setActiveSessionId(session.id);
    setUploadOpen(true); // a brand-new session has no documents yet
  }

  async function handleDeleteSession(id: string) {
    await deleteSession(id);
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeSessionId === id) setActiveSessionId(null);
  }

  async function handleOpenUpload() {
    try {
      await ensureActiveSession();
      setUploadOpen(true);
    } catch (err) {
      setChatError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleUpload(files: File[]) {
    if (files.length === 0) return;
    setUploading(true);
    setUploadError(null);
    try {
      const sessionId = await ensureActiveSession();
      await uploadDocuments(sessionId, files);
      await refreshDocuments(sessionId);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  }

  // Returns whether the message was actually dispatched, so the input
  // box (see ChatWindow) knows whether it's safe to clear the draft --
  // otherwise a failure here (e.g. couldn't even create a session) would
  // silently lose whatever the user typed.
  async function handleSend(content: string): Promise<boolean> {
    let sessionId: string;
    try {
      sessionId = await ensureActiveSession();
    } catch (err) {
      setChatError(err instanceof Error ? err.message : String(err));
      return false;
    }

    setChatError(null);
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        session_id: sessionId,
        role: "user",
        content,
        sources: [],
        created_at: new Date().toISOString(),
      },
    ]);
    setIsStreaming(true);
    setStreamingContent("");

    let full = "";
    await streamMessage(sessionId, content, {
      onToken: (text) => {
        full += text;
        setStreamingContent(full);
      },
      onDone: (sources) => {
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now() + 1,
            session_id: sessionId,
            role: "assistant",
            content: full.trim(),
            sources,
            created_at: new Date().toISOString(),
          },
        ]);
        setStreamingContent(null);
        setIsStreaming(false);
        refreshSessions().catch(console.error);
      },
      onError: (err) => {
        setChatError(err.message);
        setStreamingContent(null);
        setIsStreaming(false);
      },
    });
    return true;
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        documents={documents}
        onSelectSession={setActiveSessionId}
        onNewSession={handleNewSession}
        onDeleteSession={handleDeleteSession}
        onOpenUpload={handleOpenUpload}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        {chatError && (
          <div className="border-b border-red-200 bg-red-50 px-4 py-2 text-sm text-red-600 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
            {chatError}
          </div>
        )}
        <ChatWindow
          messages={messages}
          streamingContent={streamingContent}
          isStreaming={isStreaming}
          hasDocuments={documents.length > 0}
          onSend={handleSend}
          onOpenUpload={handleOpenUpload}
        />
      </main>

      <UploadDropzone
        open={uploadOpen}
        documents={documents}
        uploading={uploading}
        error={uploadError}
        onClose={() => setUploadOpen(false)}
        onUpload={handleUpload}
      />
    </div>
  );
}
