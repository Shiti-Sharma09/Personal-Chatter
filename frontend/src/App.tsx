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
  const refreshDocuments = useCallback(() => listDocuments().then(setDocuments), []);

  useEffect(() => {
    refreshSessions();
    refreshDocuments();
  }, [refreshSessions, refreshDocuments]);

  useEffect(() => {
    if (!activeSessionId) {
      setMessages([]);
      return;
    }
    getMessages(activeSessionId).then(setMessages);
  }, [activeSessionId]);

  useEffect(() => {
    if (documents.length === 0) setUploadOpen(true);
  }, [documents.length]);

  async function handleNewSession() {
    const session = await createSession();
    setSessions((prev) => [session, ...prev]);
    setActiveSessionId(session.id);
  }

  async function handleDeleteSession(id: string) {
    await deleteSession(id);
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeSessionId === id) setActiveSessionId(null);
  }

  async function handleUpload(files: File[]) {
    if (files.length === 0) return;
    setUploading(true);
    setUploadError(null);
    try {
      await uploadDocuments(files);
      await refreshDocuments();
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  }

  async function handleSend(content: string) {
    let sessionId = activeSessionId;
    if (!sessionId) {
      const session = await createSession();
      setSessions((prev) => [session, ...prev]);
      setActiveSessionId(session.id);
      sessionId = session.id;
    }

    setChatError(null);
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        session_id: sessionId!,
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
            session_id: sessionId!,
            role: "assistant",
            content: full.trim(),
            sources,
            created_at: new Date().toISOString(),
          },
        ]);
        setStreamingContent(null);
        setIsStreaming(false);
        refreshSessions();
      },
      onError: (err) => {
        setChatError(err.message);
        setStreamingContent(null);
        setIsStreaming(false);
      },
    });
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
        onOpenUpload={() => setUploadOpen(true)}
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
          onOpenUpload={() => setUploadOpen(true)}
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
