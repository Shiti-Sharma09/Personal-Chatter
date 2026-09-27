import type { ChatMessage } from "../api";

interface Props {
  message: Pick<ChatMessage, "role" | "content" | "sources">;
  isStreaming?: boolean;
}

export default function MessageBubble({ message, isStreaming }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`flex max-w-[75%] gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
        <div
          className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
            isUser
              ? "bg-indigo-600 text-white"
              : "bg-neutral-200 text-neutral-700 dark:bg-neutral-700 dark:text-neutral-200"
          }`}
        >
          {isUser ? "You" : "AI"}
        </div>

        <div>
          <div
            className={`whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm leading-relaxed shadow-sm ${
              isUser
                ? "rounded-tr-sm bg-indigo-600 text-white"
                : "rounded-tl-sm bg-white text-neutral-900 dark:bg-neutral-800 dark:text-neutral-100"
            }`}
          >
            {message.content}
            {isStreaming && <span className="ml-0.5 inline-block w-1.5 animate-pulse">▍</span>}
          </div>

          {message.sources.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {message.sources.map((source) => (
                <span
                  key={source}
                  className="rounded-full border border-neutral-200 bg-neutral-50 px-2 py-0.5 text-xs text-neutral-500 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-400"
                >
                  {source}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
