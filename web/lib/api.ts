import type { ChatEvent, IngestStatus, Source } from "@/lib/types";

/** Base URL of the query API. Empty means the Next.js mock routes under /api. */
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";
/** Chat streams straight from the backend when set: Vercel's function layer held the whole streamed answer until it
 * finished, so the chat sat on "..." for the full answer time. Everything else still goes through BASE. */
const STREAM_BASE = process.env.NEXT_PUBLIC_STREAM_URL || BASE;

export async function startIngest(files: File[]): Promise<{ jobId: string }> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f, f.name));
  const res = await fetch(`${BASE}/api/ingest`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`ingest failed: ${res.status}`);
  return res.json();
}

export async function getIngest(jobId: string): Promise<IngestStatus> {
  const res = await fetch(`${BASE}/api/ingest?job=${encodeURIComponent(jobId)}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`ingest status failed: ${res.status}`);
  return res.json();
}

export interface ChatRequest {
  question: string;
  conversationId: string;
  source: Source;
  mode?: string;
}

/** Streams newline-delimited JSON events from the chat endpoint. Resolves when the stream ends. */
export async function streamChat(req: ChatRequest, onEvent: (e: ChatEvent) => void, signal?: AbortSignal): Promise<void> {
  const res = await fetch(`${STREAM_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok || !res.body) {
    onEvent({ type: "error", message: `chat failed: ${res.status}` });
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (line) onEvent(JSON.parse(line) as ChatEvent);
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer) as ChatEvent);
}

export function newConversationId(prefix: "pg" | "ds"): string {
  const hex = Math.random().toString(16).slice(2, 6);
  return `conv_${prefix}_${hex}`;
}
