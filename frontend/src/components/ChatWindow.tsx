import { useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../api";
import MessageBubble from "./MessageBubble";

interface Props {
  messages: ChatMessage[];
  streamingContent: string | null;
  isStreaming: boolean;
  hasDocuments: boolean;
  onSend: (content: string) => void;
  onOpenUpload: () => void;
}

export default function ChatWindow({ messages, streamingContent, isStreaming, hasDocuments, onSend, onOpenUpload }: Props) {
  const [draft, setDraft] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  function submit() {
    const content = draft.trim();
    if (!content || isStreaming) return;
    onSend(content);
    setDraft("");
  }

  if (!hasDocuments) {
    return (
      <div className="flex h-full flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
        <div className="text-4xl">📄</div>
        <h2 className="text-lg font-medium">Upload a document to get started</h2>
        <p className="max-w-sm text-sm text-neutral-500">
          Personal Chatter answers questions grounded in documents you upload — nothing leaves your machine.
        </p>
        <button
          onClick={onOpenUpload}
          className="mt-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-indigo-500"
        >
          Upload documents
        </button>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-1 flex-col">
      <div className="scroll-thin flex-1 space-y-4 overflow-y-auto px-6 py-6">
        {messages.length === 0 && !isStreaming && (
          <p className="mt-10 text-center text-sm text-neutral-400">Ask a question about your documents.</p>
        )}
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        {isStreaming && (
          <MessageBubble message={{ role: "assistant", content: streamingContent ?? "", sources: [] }} isStreaming />
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-neutral-200 px-4 py-3 dark:border-neutral-800">
        <div className="flex items-end gap-2 rounded-xl border border-neutral-200 bg-white px-3 py-2 shadow-sm focus-within:border-indigo-400 dark:border-neutral-700 dark:bg-neutral-900">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="Ask a question about your documents…"
            rows={1}
            className="max-h-40 flex-1 resize-none bg-transparent text-sm outline-none placeholder:text-neutral-400"
          />
          <button
            onClick={submit}
            disabled={!draft.trim() || isStreaming}
            className="shrink-0 rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-neutral-300 dark:disabled:bg-neutral-700"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
