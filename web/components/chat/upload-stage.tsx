"use client";

import { useEffect, useRef, useState } from "react";
import { Upload } from "lucide-react";
import { getIngest, startIngest } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { IngestFile, IngestStatus } from "@/lib/types";

const FORMATS = ["PDF", "DOCX", "PPTX", "PNG · JPG", "MD · TXT"];

/** Drives a mock or real ingest job: call begin(files), then read status until done. */
export function useIngest(onDone: (conversationId: string, files: IngestFile[]) => void) {
  const [status, setStatus] = useState<IngestStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const doneRef = useRef(false);

  const begin = async (list: FileList | File[]) => {
    const files = Array.from(list);
    if (!files.length) return;
    setError(null);
    doneRef.current = false;
    try {
      const { jobId } = await startIngest(files);
      setStatus({ jobId, files: files.map((f) => ({ name: f.name, size: f.size, type: (f.name.split(".").pop() ?? "").toUpperCase().slice(0, 4) })), steps: [], progress: 0, done: false });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    }
  };

  useEffect(() => {
    if (!status || status.done) return;
    const t = setInterval(async () => {
      try {
        const s = await getIngest(status.jobId);
        setStatus(s);
        if (s.done && s.conversationId && !doneRef.current) {
          doneRef.current = true;
          clearInterval(t);
          setTimeout(() => onDone(s.conversationId!, s.files), 400);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Status failed");
        clearInterval(t);
      }
    }, 350);
    return () => clearInterval(t);
  }, [status, onDone]);

  const reset = () => { setStatus(null); setError(null); doneRef.current = false; };
  return { status, error, begin, reset };
}

export function Dropzone({ onFiles, compact }: { onFiles: (files: FileList) => void; compact?: boolean }) {
  const [over, setOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); inputRef.current?.click(); } }}
      onDragEnter={(e) => { e.preventDefault(); setOver(true); }}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={(e) => { e.preventDefault(); setOver(false); }}
      onDrop={(e) => { e.preventDefault(); setOver(false); if (e.dataTransfer.files.length) onFiles(e.dataTransfer.files); }}
      className={cn(
        "grid cursor-pointer justify-items-center gap-2.5 rounded-xl border border-dashed bg-card text-center transition-colors",
        compact ? "px-5 py-7" : "px-6 py-12",
        over ? "border-foreground bg-secondary" : "border-input hover:border-ring",
      )}
    >
      <span className="flex size-10 items-center justify-center rounded-lg bg-secondary text-muted-foreground"><Upload className="size-4.5" /></span>
      <div className="text-sm font-medium">Drop a document here, or click to choose</div>
      <div className="flex flex-wrap justify-center gap-1.5">
        {FORMATS.map((f) => <span key={f} className="rounded-full border border-border px-2 py-0.5 font-mono text-[11px] text-muted-foreground">{f}</span>)}
      </div>
      <input ref={inputRef} type="file" multiple accept=".pdf,.docx,.pptx,.png,.jpg,.jpeg,.md,.txt" hidden onChange={(e) => { if (e.target.files?.length) onFiles(e.target.files); e.target.value = ""; }} />
    </div>
  );
}

export function ProcessingCard({ status, error }: { status: IngestStatus; error: string | null }) {
  return (
    <div className="grid gap-3 rounded-xl border border-border bg-card p-4">
      <div className="grid gap-2">
        {status.files.map((f) => (
          <div key={f.name} className="grid grid-cols-[40px_minmax(0,1fr)_auto] items-center gap-3">
            <span className="rounded-md bg-secondary py-1.5 text-center font-mono text-[10px] tracking-wider text-muted-foreground">{f.type}</span>
            <div className="min-w-0">
              <div className="truncate text-sm font-medium">{f.name}</div>
              <div className="text-xs text-muted-foreground">{(f.size / 1e6).toFixed(1)} MB{f.chunks ? ` · ${f.chunks} chunks` : ""}</div>
            </div>
            <span className={cn("rounded-full border border-border px-2 py-0.5 font-mono text-[11px]", status.done ? "text-success" : "text-info")}>{status.done ? "indexed" : "processing"}</span>
          </div>
        ))}
      </div>
      <div className="h-1 overflow-hidden rounded-full bg-secondary">
        <div className="h-full rounded-full bg-primary transition-[width] duration-300" style={{ width: `${Math.round(status.progress * 100)}%` }} />
      </div>
      <ol className="grid gap-1.5">
        {status.steps.map((s) => (
          <li key={s.name} className={cn("grid grid-cols-[16px_minmax(0,1fr)_auto] items-center gap-3 text-[13px]", s.state === "pending" ? "text-muted-foreground" : "text-foreground")}>
            <span className={cn("size-2.5 justify-self-center rounded-full border", s.state === "done" ? "border-success bg-success" : s.state === "active" ? "border-info bg-info/20 ring-2 ring-info/30" : "border-border")} />
            <span>{s.name} <span className="text-muted-foreground">· {s.detail}</span></span>
            <span className="font-mono text-[11px] text-muted-foreground">{s.ms != null ? (s.ms >= 1000 ? `${(s.ms / 1000).toFixed(1)} s` : `${s.ms} ms`) : ""}</span>
          </li>
        ))}
      </ol>
      {error ? <p className="text-sm text-destructive">{error}</p> : null}
    </div>
  );
}
