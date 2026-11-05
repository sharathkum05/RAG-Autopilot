"use client";

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { Database, FileSearch, Layers, ListOrdered, Search, Sparkles, type LucideProps } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ChatComposer, type ComposerAction } from "@/components/ui/v0-ai-chat";
import { MessageLoading } from "@/components/ui/message-loading";
import { ToolGroup, formatElapsed, type NestedTool } from "@/components/ui/tool-group";
import { EvidencePanel } from "@/components/chat/evidence-panel";
import { AnswerText } from "@/components/chat/answer-text";
import { streamChat } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { AssistantMeta, ChatMessage, ChatStep, Confidence, Source } from "@/lib/types";

const CONF: Record<Confidence, string> = { high: "bg-success", medium: "bg-warning", low: "bg-destructive" };
const STEP_ICON: Record<string, React.ComponentType<LucideProps>> = {
  "Encoding question": Layers, "Visual search": FileSearch, "Text search": Search, Fusing: ListOrdered, Reranking: ListOrdered, Answering: Sparkles,
};
const toIcon = (I: React.ComponentType<LucideProps>) => function Icon({ className }: { className?: string }) { return <I className={className} />; };

function ConfidenceBadge({ level }: { level: Confidence }) {
  return (
    <Badge variant="outline" className="gap-1.5 font-normal text-muted-foreground">
      <span className={cn("size-1.5 rounded-full", CONF[level])} aria-hidden="true" />
      confidence {level}
    </Badge>
  );
}

/** The live retrieval step group shown above an answer. */
export function RetrievalSteps({ steps, pending, startedAt, chunks }: { steps: ChatStep[]; pending: boolean; startedAt?: number; chunks?: number }) {
  const [now, setNow] = useState(startedAt ?? 0);   // ticks while pending; Date.now() only inside the interval
  useEffect(() => { if (!pending) return; const t = setInterval(() => setNow(Date.now()), 250); return () => clearInterval(t); }, [pending]);
  if (!steps.length) return null;
  const tools: NestedTool[] = steps.map((s) => ({ icon: toIcon(STEP_ICON[s.name] ?? Database), title: s.name, subtitle: s.detail, isError: s.state === "error", elapsed: s.ms != null ? formatElapsed(s.ms) : undefined }));
  const total = steps.reduce((a, s) => a + (s.ms ?? 0), 0);
  const elapsed = pending && startedAt ? formatElapsed(now - startedAt) : total ? formatElapsed(total) : undefined;
  return (
    <ToolGroup
      state={pending ? "pending" : "completed"}
      nestedTools={tools}
      shimmerLabel="Retrieving"
      completeLabel={chunks != null ? `Retrieved ${chunks} chunk${chunks === 1 ? "" : "s"}` : "Retrieved"}
      interruptedLabel="Retrieval interrupted"
      elapsedTime={elapsed}
      defaultOpen={pending}
    />
  );
}

function AssistantMessage({ msg, conversationId, onAsk }: { msg: ChatMessage; conversationId: string; onAsk: (q: string) => void }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState<number | null>(null);
  const meta = msg.meta;
  const jump = (i: number) => {
    setOpen(true); setActive(i);
    requestAnimationFrame(() => document.getElementById(`chunk-${msg.id}-${i}`)?.scrollIntoView({ block: "nearest", behavior: "smooth" }));
    setTimeout(() => setActive(null), 1600);
  };
  return (
    <div className="grid min-w-0 gap-2.5">
      <div className="flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
        <span className="flex size-5 items-center justify-center rounded-md bg-primary text-[11px] font-semibold text-primary-foreground">M</span>
        <span>Matt</span>
        <span>{conversationId}</span>
      </div>
      {msg.steps?.length ? <RetrievalSteps steps={msg.steps} pending={Boolean(msg.pending)} startedAt={msg.startedAt} chunks={meta?.chunks.length} /> : null}
      {msg.pending && !msg.text ? (msg.steps?.length ? null : <MessageLoading />) : <AnswerText text={msg.text} onHover={setActive} onClick={jump} />}
      {meta ? (
        <>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[12.5px] text-muted-foreground">
            <ConfidenceBadge level={meta.confidence} />
            <span>{meta.chunks.length ? `${meta.chunks.length} chunk${meta.chunks.length > 1 ? "s" : ""} cited` : "no chunk above the floor"}</span>
            <span>retrieval {meta.retrievalMs} ms</span>
            <span>TTR {(meta.ttrMs / 1000).toFixed(1)} s</span>
          </div>
          {meta.chunks.length ? <EvidencePanel messageId={msg.id} chunks={meta.chunks} open={open} onOpenChange={setOpen} activeIndex={active} /> : null}
          {meta.clarify?.length ? (
            <div className="flex flex-wrap gap-1.5">
              {meta.clarify.map((q) => <button key={q} type="button" onClick={() => onAsk(q)} className="rounded-full border border-border bg-card px-3 py-1 text-xs text-muted-foreground hover:border-foreground hover:text-foreground">{q}</button>)}
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

export interface ChatViewProps {
  source: Source;
  conversationId: string;
  welcome?: ChatMessage;
  suggestions?: string[];
  actions?: ComposerAction[];
  placeholder?: string;
  mode?: string;
  model?: string;
  header?: ReactNode;
  disabled?: boolean;
  onAttach?: () => void;
  notice?: ReactNode;
  hint?: ReactNode;
}

export function ChatView({ source, conversationId, welcome, suggestions, actions, placeholder, mode, model, header, disabled, onAttach, notice, hint }: ChatViewProps) {
  const [messages, setMessages] = useState<ChatMessage[]>(welcome ? [welcome] : []);
  const [busy, setBusy] = useState(false);
  const threadRef = useRef<HTMLDivElement>(null);
  const idRef = useRef(0);

  useEffect(() => { const el = threadRef.current; if (el) el.scrollTop = el.scrollHeight; }, [messages]);

  const patchLast = useCallback((fn: (m: ChatMessage) => ChatMessage) => { setMessages((ms) => ms.map((m, i) => (i === ms.length - 1 ? fn(m) : m))); }, []);

  const send = useCallback(
    async (text: string) => {
      if (busy) return;
      const uid = `u${++idRef.current}`, aid = `a${++idRef.current}`;
      setMessages((ms) => [...ms, { id: uid, role: "user", text }, { id: aid, role: "assistant", text: "", pending: true, steps: [], startedAt: Date.now() }]);
      setBusy(true);
      try {
        await streamChat({ question: text, conversationId, source, mode }, (e) => {
          if (e.type === "step") patchLast((m) => { const steps = [...(m.steps ?? [])]; const i = steps.findIndex((s) => s.name === e.step.name); if (i >= 0) steps[i] = e.step; else steps.push(e.step); return { ...m, steps }; });
          else if (e.type === "token") patchLast((m) => ({ ...m, text: m.text + e.text }));
          else if (e.type === "evidence") patchLast((m) => ({ ...m, meta: e.meta as AssistantMeta, pending: false, text: m.text.trimEnd() }));
          else if (e.type === "error") patchLast((m) => ({ ...m, text: e.message, pending: false }));
        });
      } catch (err) {
        patchLast((m) => ({ ...m, text: err instanceof Error ? err.message : "Request failed", pending: false }));
      } finally {
        setBusy(false);
      }
    },
    [busy, conversationId, source, mode, patchLast],
  );

  const fresh = messages.length <= 1;

  return (
    <div className="flex h-[calc(100svh-3.5rem)] min-h-0 flex-col">
      <div ref={threadRef} className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto grid w-full max-w-3xl grid-cols-[minmax(0,1fr)] gap-7 px-4 py-8">
          {header}
          {messages.map((m) =>
            m.role === "user" ? (
              <div key={m.id} className="grid max-w-[85%] justify-self-end gap-1.5">
                <div className="flex justify-end gap-2 font-mono text-[11px] text-muted-foreground"><span>{conversationId}</span><span>you</span></div>
                <div className="rounded-2xl rounded-br-sm border border-border bg-secondary px-4 py-2.5 text-[15px] leading-7">{m.text}</div>
              </div>
            ) : (
              <AssistantMessage key={m.id} msg={m} conversationId={conversationId} onAsk={send} />
            ),
          )}
        </div>
      </div>
      <div className="shrink-0 bg-gradient-to-t from-background via-background to-transparent px-4 pb-4 pt-2">
        <div className="mx-auto w-full max-w-3xl">
          <ChatComposer placeholder={placeholder} disabled={busy || disabled} onSend={send} onAttach={onAttach} model={model} notice={notice}
            suggestions={fresh && !disabled ? suggestions : undefined} actions={fresh ? actions : undefined} />
          {hint ? <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 px-1 text-xs text-muted-foreground">{hint}</div> : null}
        </div>
      </div>
    </div>
  );
}
