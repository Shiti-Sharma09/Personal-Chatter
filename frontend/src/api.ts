export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatMessage {
  id: number;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  sources: string[];
  created_at: string;
}

export interface DocumentInfo {
  name: string;
  size_bytes: number;
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function listSessions(): Promise<ChatSession[]> {
  return fetch("/api/sessions").then((r) => asJson(r));
}

export function createSession(title?: string): Promise<ChatSession> {
  return fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: title ?? null }),
  }).then((r) => asJson(r));
}

export function deleteSession(sessionId: string): Promise<void> {
  return fetch(`/api/sessions/${sessionId}`, { method: "DELETE" }).then((r) => {
    if (!r.ok) throw new Error(`Failed to delete session: ${r.status}`);
  });
}

export function getMessages(sessionId: string): Promise<ChatMessage[]> {
  return fetch(`/api/sessions/${sessionId}/messages`).then((r) => asJson(r));
}

export function listDocuments(): Promise<DocumentInfo[]> {
  return fetch("/api/documents").then((r) => asJson(r));
}

export async function uploadDocuments(files: File[]): Promise<{ ingested: string[] }> {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  const res = await fetch("/api/documents/upload", { method: "POST", body: form });
  return asJson(res);
}

type StreamEvent = { type: "token"; text: string } | { type: "done"; message_id: number; sources: string[] };

export interface StreamHandlers {
  onToken: (text: string) => void;
  onDone: (sources: string[]) => void;
  onError: (error: Error) => void;
}

/** POSTs a message and consumes the resulting text/event-stream response,
 * since the browser's EventSource API can't send a POST body. */
export async function streamMessage(sessionId: string, content: string, handlers: StreamHandlers): Promise<void> {
  try {
    const res = await fetch(`/api/sessions/${sessionId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });

    if (!res.ok || !res.body) {
      const body = await res.json().catch(() => null);
      throw new Error(body?.detail ?? `Request failed: ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";

      for (const raw of events) {
        const line = raw.trim();
        if (!line.startsWith("data:")) continue;
        const payload = JSON.parse(line.slice("data:".length).trim()) as StreamEvent;
        if (payload.type === "token") handlers.onToken(payload.text);
        else handlers.onDone(payload.sources);
      }
    }
  } catch (err) {
    handlers.onError(err instanceof Error ? err : new Error(String(err)));
  }
}
