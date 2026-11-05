"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Database, FileImage, FileText, FileType, Presentation, Route, Scissors, Sparkles, type LucideProps } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AppShell } from "@/components/app-shell";
import { ChatComposer } from "@/components/ui/v0-ai-chat";
import { ToolGroup, formatElapsed, type NestedTool } from "@/components/ui/tool-group";
import { useIngest } from "@/components/chat/upload-stage";
import { ChatView } from "@/components/chat/chat-view";
import type { ChatMessage, IngestFile, IngestStatus } from "@/lib/types";

const ACCEPT_ALL = ".pdf,.docx,.pptx,.png,.jpg,.jpeg,.md,.txt";
const STEP_ICON: Record<string, React.ComponentType<LucideProps>> = { Router: Route, Parse: FileText, Chunk: Scissors, "Embed text": FileType, "Embed pages": FileImage, Index: Database };
const toIcon = (I: React.ComponentType<LucideProps>) => function Icon({ className }: { className?: string }) { return <I className={className} />; };

function readModel(): string {
  try { const d = localStorage.getItem("matt.defaults"); if (d) return JSON.parse(d).answer_model ?? "claude-opus-5-5"; } catch { /* ignore */ }
  return "claude-opus-5-5";
}

/** Indexing shown as a Claude-style step group, fed live by the ingest job. */
function IndexingSteps({ status, startedAt }: { status: IngestStatus; startedAt: number }) {
  const [now, setNow] = useState(startedAt);   // ticks while indexing; Date.now() only inside the interval
  useEffect(() => { if (status.done) return; const t = setInterval(() => setNow(Date.now()), 250); return () => clearInterval(t); }, [status.done]);
  const names = status.files.map((f) => f.name).join(", ");
  const tools: NestedTool[] = status.steps
    .filter((s) => s.state !== "pending")
    .map((s) => ({ icon: toIcon(STEP_ICON[s.name] ?? Database), title: s.name, subtitle: s.name === "Router" ? `${names} → ${status.files.map((f) => f.type).join(", ")}` : s.detail, elapsed: s.ms != null ? formatElapsed(s.ms) : undefined }));
  if (status.done) tools.push({ icon: toIcon(Sparkles), title: "Ready", subtitle: `${status.files.reduce((a, f) => a + (f.chunks ?? 0), 0)} chunks · ${status.files.length} document${status.files.length > 1 ? "s" : ""} searchable` });
  const total = status.steps.reduce((a, s) => a + (s.ms ?? 0), 0);
  return (
    <ToolGroup
      state={status.done ? "completed" : "pending"}
      nestedTools={tools}
      shimmerLabel={`Indexing ${names}`}
      completeLabel={`Indexed ${names}`}
      interruptedLabel="Indexing interrupted"
      elapsedTime={status.done ? formatElapsed(total) : formatElapsed(now - startedAt)}
      defaultOpen
    />
  );
}

export default function PlaygroundPage() {
  const [ready, setReady] = useState<{ conversationId: string; files: IngestFile[] } | null>(null);
  const [model, setModel] = useState("claude-opus-5-5");
  const [notice, setNotice] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const onDone = useCallback((conversationId: string, files: IngestFile[]) => setReady({ conversationId, files }), []);
  const { status, error, begin, reset } = useIngest(onDone);

  useEffect(() => { const t = setTimeout(() => setModel(readModel()), 0); return () => clearTimeout(t); }, []);

  const pick = (accept = ACCEPT_ALL) => { const el = inputRef.current; if (!el) return; el.accept = accept; el.click(); };
  const onFiles = (list: FileList) => { setStartedAt(Date.now()); setNotice(null); begin(list); };
  const actions = [
    { icon: <FileText className="h-4 w-4" />, label: "Upload a PDF", onClick: () => pick(".pdf") },
    { icon: <FileType className="h-4 w-4" />, label: "Word document", onClick: () => pick(".docx") },
    { icon: <Presentation className="h-4 w-4" />, label: "PowerPoint deck", onClick: () => pick(".pptx") },
    { icon: <FileImage className="h-4 w-4" />, label: "Scanned pages", onClick: () => pick(".png,.jpg,.jpeg") },
    { icon: <FileText className="h-4 w-4" />, label: "Markdown or text", onClick: () => pick(".md,.txt") },
  ];

  const fileInput = <input ref={inputRef} type="file" multiple accept={ACCEPT_ALL} hidden onChange={(e) => { if (e.target.files?.length) onFiles(e.target.files); e.target.value = ""; }} />;

  const subtitle = ready ? `${ready.files.length} document${ready.files.length > 1 ? "s" : ""} indexed` : status ? "indexing" : "upload a document to start";
  const topActions = ready ? (
    <>
      <Badge variant="outline" className="font-mono font-normal text-muted-foreground">conversation <span className="ml-1 text-foreground">{ready.conversationId}</span></Badge>
      <Button variant="ghost" size="sm" onClick={() => { setReady(null); reset(); }}>New upload</Button>
    </>
  ) : null;

  // Hero: nothing uploaded yet.
  if (!status) {
    return (
      <AppShell title="Playground" subtitle={subtitle}>
        {fileInput}
        <div
          className="flex flex-1 items-center justify-center px-4 py-10"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); if (e.dataTransfer.files.length) onFiles(e.dataTransfer.files); }}
        >
          <div className="flex w-full max-w-4xl flex-col items-center space-y-8">
            <h1 className="text-balance text-center text-4xl font-bold tracking-tight text-foreground">How can I help you today</h1>
            <ChatComposer
              placeholder="Ask Matt a question about your document..."
              onSend={() => setNotice("Upload a document first, then ask away. Use the paperclip or one of the buttons below.")}
              onAttach={() => pick()}
              model={model}
              actions={actions}
              notice={notice}
              autoFocus
            />
            {error ? <p className="text-sm text-destructive">{error}</p> : null}
          </div>
        </div>
      </AppShell>
    );
  }

  // Thread: indexing steps at the top, then chat once ready.
  return (
    <AppShell title="Playground" subtitle={subtitle} actions={topActions}>
      {fileInput}
      <ChatView
        key={ready?.conversationId ?? status.jobId}
        source="playground"
        conversationId={ready?.conversationId ?? "—"}
        welcome={ready ? welcomeFor(ready.files) : undefined}
        placeholder={ready ? "Ask about your document" : "Indexing… the box unlocks when it finishes"}
        mode="fused"
        model={model}
        header={<IndexingSteps status={status} startedAt={startedAt} />}
        disabled={!ready}
        onAttach={() => pick()}
        hint={ready ? <><span>Answers only from your documents; each claim cites a chunk.</span><span>Enter to send, Shift+Enter for a new line</span></> : undefined}
      />
    </AppShell>
  );
}

function welcomeFor(files: IngestFile[]): ChatMessage {
  const names = files.map((f) => f.name).join(", ");
  return { id: "welcome", role: "assistant", text: `Indexed ${names}. Ask anything from ${files.length > 1 ? "these documents" : "this document"}; each answer cites the chunk and page it came from.` };
}
