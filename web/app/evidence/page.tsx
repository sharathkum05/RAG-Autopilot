"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { ChevronRight, ExternalLink, Trash2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { AppShell } from "@/components/app-shell";
import { MessageLoading } from "@/components/ui/message-loading";
import { AnswerText } from "@/components/chat/answer-text";
import { EvidencePanel } from "@/components/chat/evidence-panel";
import { cn } from "@/lib/utils";
import type { Confidence, EvidenceChunk } from "@/lib/types";
import type { IndexChunk, IndexDocument, IndexPage } from "@/lib/index-data";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";
const j = async <T,>(url: string, init?: RequestInit): Promise<T> => { const r = await fetch(`${BASE}${url}`, init); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); };

interface Workspace {
  id: string; kind: "dataset" | "playground"; title: string; documents: (IndexDocument & { file_type?: string | null; status: string })[];
  turns: number; cost_usd: number; avg_ttr_ms: number | null; last_at: string | null; created_at: string | null; mode: string | null; model: string | null;
}
interface Turn {
  id: number; conversation_id: string; question: string; answer: string | null; confidence: Confidence | null; evidence: EvidenceChunk[]; timings_ms: Record<string, number>;
  ttr_ms: number | null; tokens_in: number | null; tokens_out: number | null; cost_usd: number; trace_id: string | null; trace_url: string | null; status: string; created_at: string;
  model: string | null; mode: string | null; diagnosis?: Diagnosis;
}
interface Finding { level: "bad" | "warn" | "info"; title: string; detail: string }
interface Diagnosis { verdict: "ok" | "warn" | "bad"; headline: string | null; confidence_why: string; findings: Finding[]; full_ranking: boolean; uncited: EvidenceChunk[] }

const CONF: Record<string, string> = { high: "bg-success", medium: "bg-warning", low: "bg-destructive" };
const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);
const when = (iso: string | null) => (iso ? iso.replace("T", " ").slice(0, 16) : "");
const MODE_LABEL: Record<string, string> = { bm25: "BM25 only", text_hybrid: "Hybrid RAG without reranker", text_hybrid_rerank: "Hybrid RAG with reranker", visual: "Visual only (ColQwen)", fused: "Fused (visual + text)", fused_rerank: "Fused + rerank" };
const fmtMode = (m: string | null) => (m ? MODE_LABEL[m] ?? m.replace("_", " + ") : "");

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <span className="inline-flex items-baseline gap-1.5 rounded-full border border-border bg-card px-2.5 py-0.5 font-mono text-[11px] text-muted-foreground">
      {label} <b className="font-medium text-foreground">{value}</b>{sub ? <span>{sub}</span> : null}
    </span>
  );
}

function Kind({ kind }: { kind: string }) {
  return <Badge variant="outline" className={cn("font-mono text-[10px] uppercase", kind === "table" && "text-info", kind === "figure" && "text-warning")}>{kind}</Badge>;
}

// ---------------------------------------------------------------------------- left rail: which conversation

function WorkspaceList({ items, selected, onSelect, onDelete }: { items: Workspace[]; selected: string | null; onSelect: (id: string) => void; onDelete: (id: string) => Promise<void> }) {
  const groups: [string, Workspace[]][] = [["Dataset", items.filter((w) => w.kind === "dataset")], ["Playground conversations", items.filter((w) => w.kind === "playground")]];
  const [confirm, setConfirm] = useState<string | null>(null);     // conversation waiting for a yes
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  return (
    <div className="grid content-start gap-4">
      {groups.map(([label, ws]) => (
        <div key={label} className="grid gap-1.5">
          <div className="px-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{label}</div>
          {ws.length ? ws.map((w) => (
            <div key={w.id} className={cn("grid gap-0.5 rounded-lg border border-border bg-card px-3 py-2 hover:border-foreground/40", selected === w.id && "border-foreground ring-1 ring-foreground/40")}>
              <div className="flex items-start gap-1.5">
                <button type="button" onClick={() => onSelect(w.id)} className="grid min-w-0 flex-1 gap-0.5 text-left">
                  <span className="truncate text-[13px] font-medium">{w.title}</span>
                  <span className="flex flex-wrap gap-x-2 font-mono text-[11px] text-muted-foreground">
                    <span>{w.id}</span>
                    <span>{w.turns} question{w.turns === 1 ? "" : "s"}</span>
                    {w.avg_ttr_ms != null ? <span>TTR {fmtMs(w.avg_ttr_ms)}</span> : null}
                    {w.cost_usd ? <span>${w.cost_usd.toFixed(3)}</span> : null}
                  </span>
                </button>
                {w.kind === "playground" ? (
                  <button type="button" aria-label="Delete conversation" title="Delete this conversation and its indexed documents" onClick={() => { setErr(null); setConfirm(confirm === w.id ? null : w.id); }}
                    className="-mr-1 mt-0.5 shrink-0 rounded p-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"><Trash2 className="size-3.5" /></button>
                ) : null}
              </div>
              {confirm === w.id ? (
                <div className="mt-1 grid gap-1.5 rounded-md border border-destructive/40 bg-destructive/5 px-2 py-1.5 text-[11.5px]">
                  <span className="text-muted-foreground">Delete this conversation, its {w.turns} question{w.turns === 1 ? "" : "s"} and its indexed document{w.documents.length === 1 ? "" : "s"}? Vectors and page images go too.</span>
                  <span className="flex gap-2">
                    <button type="button" disabled={busy === w.id} onClick={async () => { setBusy(w.id); try { await onDelete(w.id); setConfirm(null); } catch (e) { setErr(e instanceof Error ? e.message : "delete failed"); } finally { setBusy(null); } }}
                      className="rounded bg-destructive px-2 py-0.5 font-medium text-white disabled:opacity-50">{busy === w.id ? "Deleting…" : "Delete"}</button>
                    <button type="button" onClick={() => setConfirm(null)} className="text-muted-foreground hover:text-foreground">Cancel</button>
                  </span>
                  {err && confirm === w.id ? <span className="font-mono text-destructive">{err}</span> : null}
                </div>
              ) : null}
            </div>
          )) : <div className="rounded-lg border border-dashed border-border px-3 py-2 text-xs text-muted-foreground">Nothing indexed yet. Upload a document in the Playground and it shows up here.</div>}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------- header: what is indexed in it

function Summary({ ws }: { ws: Workspace }) {
  const pages = ws.documents.reduce((a, d) => a + d.pages, 0), chunks = ws.documents.reduce((a, d) => a + d.chunks, 0);
  return (
    <div className="grid gap-3 rounded-xl border border-border bg-card p-4">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <div className="text-[15px] font-medium">{ws.title}</div>
        <span className="font-mono text-[11px] text-muted-foreground">{ws.id} · {ws.kind}</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        <Stat label="documents" value={String(ws.documents.length)} />
        <Stat label="pages" value={String(pages)} />
        <Stat label="chunks" value={String(chunks)} />
        <Stat label="questions" value={String(ws.turns)} />
        {ws.avg_ttr_ms != null ? <Stat label="mean TTR" value={fmtMs(ws.avg_ttr_ms)} /> : null}
        <Stat label="spent" value={`$${ws.cost_usd.toFixed(4)}`} />
        {ws.mode ? <Stat label="retrieval" value={fmtMode(ws.mode)} /> : null}
        {ws.model ? <Stat label="model" value={ws.model} /> : null}
      </div>
      {ws.documents.length ? (
        <div className="grid gap-1 text-xs">
          {ws.documents.map((d) => (
            <div key={d.id} className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-muted-foreground">
              <span className="font-medium text-foreground">{d.name}</span>
              <span className="font-mono">{d.file_type ?? d.source}</span>
              <span>{d.pages} pages</span><span>{d.chunks} chunks</span>
              <span className={cn("font-mono", d.status === "indexed" ? "text-success" : d.status === "failed" ? "text-destructive" : "text-warning")}>{d.status}</span>
              <span className="font-mono">{when(d.indexed_at)}</span>
            </div>
          ))}
        </div>
      ) : <div className="text-xs text-muted-foreground">No document indexed in this workspace yet.</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------- queries: every question, its evidence, latency and cost

const LEVEL: Record<string, string> = { ok: "bg-success", bad: "bg-destructive", warn: "bg-warning", info: "bg-muted-foreground" };

/**
 * Backtracking for one answer: the confidence rule spelled out for this turn, what looks wrong (from pipeline/diagnose.py),
 * and the retrieved pages Claude did not cite, so a bad answer can be traced to the stage that caused it.
 */
function Backtrack({ t, onExplain }: { t: Turn; onExplain: (q: string) => void }) {
  const d = t.diagnosis;
  if (!d) return null;
  return (
    <div className="grid gap-2.5 rounded-lg border border-border bg-secondary/30 p-3 text-[13px]">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Backtrack</span>
        <span className={cn("size-1.5 rounded-full", LEVEL[d.verdict])} aria-hidden="true" />
        <span className="font-medium">{d.headline}</span>
      </div>
      <div className="text-muted-foreground"><span className="text-foreground">Why confidence is {t.confidence ?? "n/a"}: </span>{d.confidence_why.replace(/^(High|Medium|Low): /, "")}</div>
      {d.findings.length ? (
        <ul className="grid gap-1.5">
          {d.findings.map((f, i) => (
            <li key={i} className="grid grid-cols-[10px_minmax(0,1fr)] items-baseline gap-2">
              <span className={cn("size-1.5 translate-y-[-1px] rounded-full", LEVEL[f.level])} aria-hidden="true" />
              <span><b className="font-medium">{f.title}.</b> <span className="text-muted-foreground">{f.detail}</span></span>
            </li>
          ))}
        </ul>
      ) : null}
      {d.uncited.length ? (
        <Collapsible className="grid gap-1.5">
          <CollapsibleTrigger className="group inline-flex w-fit items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground">
            <ChevronRight className="size-3 transition-transform group-data-[panel-open]:rotate-90" />
            Retrieved but not cited ({d.uncited.length})
          </CollapsibleTrigger>
          <CollapsibleContent className="grid gap-1.5">
            {d.uncited.map((c) => (
              <div key={c.id} className="grid gap-1 rounded-md border border-border bg-card px-2.5 py-2 text-xs">
                <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 font-mono text-[11px] text-muted-foreground">
                  <b className="text-foreground">#{c.scores.fusedRank ?? "?"}</b><span>p.{c.pno}</span><span>{c.id}</span><Kind kind={c.kind} />
                  {c.scores.visualRank != null ? <span>visual #{c.scores.visualRank}</span> : null}
                  {c.scores.textRank != null ? <span>text #{c.scores.textRank}</span> : null}
                  {c.scores.rerank != null ? <span>rerank {c.scores.rerank.toFixed(2)}</span> : null}
                </div>
                <div className="line-clamp-2 text-muted-foreground">{c.snippet}</div>
              </div>
            ))}
          </CollapsibleContent>
        </Collapsible>
      ) : null}
      <div className="text-xs text-muted-foreground">
        Next: <button type="button" onClick={() => onExplain(t.question)} className="underline underline-offset-2 hover:text-foreground">Explain this retrieval</button> re-runs every stage for this question;
        {t.trace_url ? <> the <a href={t.trace_url} target="_blank" rel="noreferrer" className="underline underline-offset-2 hover:text-foreground">Langfuse trace</a> has each stage&apos;s ranked pages.</> : " traces need Langfuse keys."}
      </div>
    </div>
  );
}

function TurnRow({ t, onExplain, role }: { t: Turn; onExplain: (q: string) => void; role?: "baseline" | "improvement" }) {
  const [open, setOpen] = useState(false);
  const [evOpen, setEvOpen] = useState(true);
  const [active, setActive] = useState<number | null>(null);
  const timings = Object.entries(t.timings_ms ?? {});
  const retrieval = timings.reduce((a, [, v]) => a + v, 0);
  const ttr = t.ttr_ms ?? retrieval;
  const stages: [string, number][] = [...timings, ["Answering", Math.max(0, ttr - retrieval)]];
  const cited = t.evidence?.length ?? 0;
  return (
    <Collapsible open={open} onOpenChange={setOpen} className={cn("rounded-xl border border-border bg-card", role === "improvement" && "border-success/40")}>
      <CollapsibleTrigger className="group grid w-full grid-cols-[14px_minmax(0,1fr)_auto] items-center gap-3 px-3.5 py-3 text-left">
        <ChevronRight className="size-3.5 text-muted-foreground transition-transform group-data-[panel-open]:rotate-90" />
        <div className="min-w-0">
          {role ? (
            <div className="flex flex-wrap items-center gap-2 text-[13px]">
              <Badge variant="outline" className={cn("font-mono text-[10px] uppercase", role === "improvement" && "border-success/50 text-success")}>{role}</Badge>
              <span className="font-medium">{fmtMode(t.mode)}</span>
            </div>
          ) : <div className="truncate text-[14px] font-medium">{t.question}</div>}
          <div className="mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
            <span className="inline-flex items-center gap-1.5"><span className={cn("size-1.5 rounded-full", CONF[t.confidence ?? ""] ?? "bg-muted-foreground")} aria-hidden="true" />confidence {t.confidence ?? "n/a"}</span>
            <span>{cited} chunk{cited === 1 ? "" : "s"} cited</span>
            {t.model ? <span className="font-mono">{t.model}</span> : null}
            {t.mode && !role ? <span>{fmtMode(t.mode)}</span> : null}
            {!role ? <span className="font-mono">{when(t.created_at)}</span> : null}
            {t.diagnosis && t.diagnosis.verdict !== "ok" ? (
              <span className={cn("inline-flex items-center gap-1.5", t.diagnosis.verdict === "bad" ? "text-destructive" : "text-warning")}>
                <span className={cn("size-1.5 rounded-full", LEVEL[t.diagnosis.verdict])} aria-hidden="true" />{t.diagnosis.headline}
              </span>
            ) : null}
          </div>
        </div>
        <div className="grid justify-items-end gap-0.5 font-mono text-[11px] text-muted-foreground">
          <span>TTR <b className="font-medium text-foreground">{fmtMs(ttr)}</b></span>
          <span>retrieval {fmtMs(retrieval)}</span>
          <span>${t.cost_usd.toFixed(4)} · {t.tokens_in ?? 0} in · {t.tokens_out ?? 0} out</span>
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="grid grid-cols-[minmax(0,1fr)] gap-3 border-t border-border px-3.5 pb-3.5 pt-3">
          <div className="flex flex-wrap gap-1.5">
            {stages.map(([k, v]) => <Stat key={k} label={k} value={fmtMs(v)} sub={ttr ? `${Math.round((v / ttr) * 100)}%` : undefined} />)}
          </div>
          <Backtrack t={t} onExplain={onExplain} />
          <AnswerText text={t.answer ?? ""} onHover={setActive} onClick={(i) => { setEvOpen(true); setActive(i); setTimeout(() => setActive(null), 1600); }} />
          {cited ? <EvidencePanel messageId={`turn-${t.id}`} chunks={t.evidence} open={evOpen} onOpenChange={setEvOpen} activeIndex={active} /> : null}
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" size="sm" onClick={() => onExplain(t.question)}>Explain this retrieval</Button>
            {t.trace_url ? (
              <a href={t.trace_url} target="_blank" rel="noreferrer" className="inline-flex h-8 items-center gap-1.5 rounded-md border border-border bg-card px-3 text-xs font-medium hover:bg-secondary">
                Trace in Langfuse <ExternalLink className="size-3.5" />
              </a>
            ) : null}
          </div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

/** A ViDoRe Pipeline compare run: the same question through baseline and improvement, shown as one card with both answers. */
function CompareRow({ base, impr, onExplain }: { base: Turn; impr: Turn; onExplain: (q: string) => void }) {
  return (
    <div className="grid gap-2.5 rounded-xl border border-border bg-card p-3.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <Badge variant="outline" className="font-mono text-[10px] uppercase">compare</Badge>
        <span className="min-w-0 flex-1 truncate text-[14px] font-medium">{base.question}</span>
        <span className="font-mono text-[11px] text-muted-foreground">{when(base.created_at)}</span>
      </div>
      <div className="grid grid-cols-[minmax(0,1fr)] gap-2 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <TurnRow t={base} onExplain={onExplain} role="baseline" />
        <TurnRow t={impr} onExplain={onExplain} role="improvement" />
      </div>
    </div>
  );
}

/** Compare runs store one conversation per column: conv_cmp_<hex>_base and conv_cmp_<hex>_impr. The shared prefix pairs them. */
const compareKey = (conversationId: string) => (conversationId.startsWith("conv_cmp_") ? conversationId.replace(/_(base|impr)$/, "") : null);

function Queries({ ws, onExplain }: { ws: Workspace; onExplain: (q: string) => void }) {
  const [turns, setTurns] = useState<Turn[] | null>(null);
  useEffect(() => { let on = true; j<{ turns: Turn[] }>(`/api/evidence/turns?workspace=${encodeURIComponent(ws.id)}`).then((r) => { if (on) setTurns(r.turns); }); return () => { on = false; }; }, [ws.id]);
  if (!turns) return <MessageLoading />;
  if (!turns.length) return <div className="rounded-lg border border-dashed border-border p-4 text-sm text-muted-foreground">No questions in this conversation yet. Ask one in the {ws.kind === "dataset" ? "Dataset" : "Playground"} tab and it appears here with its cited chunks, per-stage latency and cost.</div>;
  const items: ({ kind: "one"; t: Turn } | { kind: "pair"; base: Turn; impr: Turn })[] = [];
  const seen = new Set<number>();
  for (const t of turns) {
    if (seen.has(t.id)) continue;
    const key = compareKey(t.conversation_id);
    const mate = key ? turns.find((u) => u.id !== t.id && !seen.has(u.id) && u.question === t.question && compareKey(u.conversation_id) === key) : undefined;
    seen.add(t.id);
    if (key && mate) {
      seen.add(mate.id);
      const [base, impr] = t.conversation_id.endsWith("_base") ? [t, mate] : [mate, t];
      items.push({ kind: "pair", base, impr });
    } else items.push({ kind: "one", t });
  }
  return <div className="grid gap-2">{items.map((it) => (it.kind === "pair" ? <CompareRow key={`cmp-${it.base.id}`} base={it.base} impr={it.impr} onExplain={onExplain} /> : <TurnRow key={it.t.id} t={it.t} onExplain={onExplain} />))}</div>;
}

// ---------------------------------------------------------------------------- chunks: how each page was cut

function ChunkCard({ c, terms, active, onHover }: { c: IndexChunk; terms?: string[]; active?: boolean; onHover?: (id: string | null) => void }) {
  const text = c.text.length > 900 ? c.text.slice(0, 900) + "…" : c.text;
  const parts = terms?.length ? text.split(new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "ig")) : [text];
  return (
    <div className={cn("grid gap-2 rounded-lg border border-border bg-card p-3 transition-colors", active && "border-foreground")} onMouseEnter={() => onHover?.(c.id)} onMouseLeave={() => onHover?.(null)}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
        <span className="font-mono text-foreground">{c.id}</span>
        <Kind kind={c.kind} />
        <span>page {c.pno}</span>
        <span>{c.n_tokens} tokens</span>
        {c.heading ? <span className="truncate">§ {c.heading}</span> : null}
        <span className="ml-auto font-mono text-[11px]">matt_chunks · {c.vector_id}</span>
      </div>
      <p className={cn("max-h-40 overflow-y-auto whitespace-pre-wrap text-[13px] leading-relaxed", c.kind === "table" && "font-mono text-[12px]")}>
        {parts.map((p, i) => (terms?.some((t) => t.toLowerCase() === p.toLowerCase()) ? <mark key={i} className="rounded-sm bg-evidence/35 text-foreground">{p}</mark> : <span key={i}>{p}</span>))}
      </p>
    </div>
  );
}

/** The page image with every chunk's region drawn on it; the hovered chunk is filled. */
function PageWithBoxes({ page, chunks, hover }: { page: IndexPage; chunks: IndexChunk[]; hover: string | null }) {
  const ratio = page.width && page.height ? `${page.width} / ${page.height}` : "300 / 388";
  return (
    <div className="relative w-full overflow-hidden rounded-md border border-border bg-white" style={{ aspectRatio: ratio }}>
      {page.image ? (
        <Image src={page.image} alt={`page ${page.pno}`} fill sizes="260px" className="object-contain" unoptimized priority />
      ) : (
        <div className="absolute inset-0 grid place-items-center px-4 text-center text-xs text-neutral-500">No page render for this file type on this machine (text-only parse), so there is no image and no visual vectors.</div>
      )}
      {chunks.map((c) => {
        const b = c.bbox;
        if (!b || b.length < 4) return null;
        const norm = b.every((v) => v <= 1.0001);
        const w = norm ? 1 : page.width || 1, h = norm ? 1 : page.height || 1;
        return (
          <span key={c.id} aria-hidden="true"
            className={cn("absolute rounded-[2px] border border-evidence/70 transition-colors", hover === c.id ? "bg-evidence/35 border-evidence" : "bg-evidence/10")}
            style={{ left: `${(b[0] / w) * 100}%`, top: `${(b[1] / h) * 100}%`, width: `${((b[2] - b[0]) / w) * 100}%`, height: `${((b[3] - b[1]) / h) * 100}%` }} />
        );
      })}
    </div>
  );
}

function Chunks({ ws }: { ws: Workspace }) {
  const [doc, setDoc] = useState<IndexDocument>(ws.documents[0]);
  const [pages, setPages] = useState<IndexPage[] | null>(null);
  const [sel, setSel] = useState<IndexPage | null>(null);
  const [chunks, setChunks] = useState<IndexChunk[]>([]);
  const [hover, setHover] = useState<string | null>(null);
  useEffect(() => { let on = true; j<{ pages: IndexPage[] }>(`/api/index/pages?doc=${encodeURIComponent(doc.id)}&workspace=${encodeURIComponent(ws.id)}`).then((r) => { if (on) { setPages(r.pages); setSel(r.pages[0] ?? null); } }); return () => { on = false; }; }, [doc.id, ws.id]);
  useEffect(() => { if (!sel) return; let on = true; j<{ chunks: IndexChunk[] }>(`/api/index/chunks?page=${encodeURIComponent(sel.id)}`).then((r) => { if (on) setChunks(r.chunks); }); return () => { on = false; }; }, [sel]);
  return (
    <div className="grid gap-5">
      {ws.documents.length > 1 ? (
        <div className="flex flex-wrap gap-1.5">
          {ws.documents.map((d) => <button key={d.id} type="button" onClick={() => setDoc(d)} className={cn("rounded-full border border-border px-3 py-1 text-xs hover:border-foreground", doc.id === d.id && "border-foreground bg-secondary")}>{d.name}</button>)}
        </div>
      ) : null}
      <div className="grid gap-6 lg:grid-cols-[300px_minmax(0,1fr)]">
        <div className="grid content-start gap-3">
          <div className="max-h-[60vh] overflow-y-auto rounded-lg border border-border bg-card p-2">
            {pages ? (
              <div className="grid grid-cols-5 gap-1.5">
                {pages.map((p) => (
                  <button key={p.id} type="button" onClick={() => setSel(p)} title={`page ${p.pno} · ${p.n_chunks} chunks`} className={cn("group grid gap-0.5 rounded-md p-1 text-center hover:bg-secondary", sel?.id === p.id && "bg-secondary ring-1 ring-foreground")}>
                    <span className="relative aspect-[300/388] w-full overflow-hidden rounded-sm border border-border bg-white">{p.image ? <Image src={p.image} alt="" fill sizes="56px" className="object-cover" unoptimized /> : null}</span>
                    <span className="font-mono text-[10px] text-muted-foreground">{p.pno}</span>
                  </button>
                ))}
              </div>
            ) : <MessageLoading />}
          </div>
        </div>
        {sel ? (
          <div className="grid gap-4 md:grid-cols-[260px_minmax(0,1fr)]">
            <div className="grid content-start gap-2">
              <PageWithBoxes page={sel} chunks={chunks} hover={hover} />
              <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs text-muted-foreground">
                <dt>page</dt><dd className="font-mono text-foreground">{sel.pno} · {sel.id}</dd>
                <dt>size</dt><dd className="font-mono text-foreground">{sel.width && sel.height ? `${sel.width} × ${sel.height}` : "no render"}</dd>
                <dt>words</dt><dd className="font-mono text-foreground">{sel.n_words}</dd>
                <dt>chunks</dt><dd className="font-mono text-foreground">{sel.n_chunks}</dd>
                <dt>visual vectors</dt><dd className="font-mono text-foreground">{sel.n_visual_vectors ? `${sel.n_visual_vectors} × 128 (matt_pages)` : "none"}</dd>
              </dl>
            </div>
            <div className="grid content-start gap-2">
              <div className="text-xs text-muted-foreground">{chunks.length} chunk{chunks.length === 1 ? "" : "s"} cut from this page. Hover one to see its region on the page.</div>
              {chunks.map((c) => <ChunkCard key={c.id} c={c} active={hover === c.id} onHover={setHover} />)}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------- search chunks

function Search({ ws }: { ws: Workspace }) {
  const [q, setQ] = useState("");
  const [res, setRes] = useState<{ terms: string[]; total: number; results: (IndexChunk & { matches: number })[] } | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async (query: string) => { if (!query.trim()) return; setBusy(true); try { setRes(await j(`/api/index/search?q=${encodeURIComponent(query)}&workspace=${encodeURIComponent(ws.id)}`)); } finally { setBusy(false); } };
  return (
    <div className="grid gap-3">
      <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); run(q); }}>
        <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Find chunks containing…" className="max-w-md" />
        <Button type="submit" disabled={busy}>Search</Button>
        {res ? <span className="self-center text-xs text-muted-foreground">{res.total} chunk{res.total === 1 ? "" : "s"} contain {res.terms.map((t) => `"${t}"`).join(" and ")}</span> : null}
      </form>
      <div className="grid gap-2">{res?.results.map((c) => <ChunkCard key={c.id} c={c} terms={res.terms} />)}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------- explain a query: every candidate list

interface Hit { id: string; page: string | number; pno: number; score: number; chunk?: string; preview?: string }
interface Fused { page: string | number; pno: number; score: number; from: Record<string, number>; preview?: string; rerank?: number }
interface Explain { question: string; terms: string[]; timings_ms: Record<string, number>; paths: { visual: Hit[]; dense: Hit[]; bm25: Hit[] }; text_hybrid: Fused[]; fused: Fused[]; reranked: Fused[]; final: Fused[] }

function HitList({ title, sub, hits, fmt, finalPages }: { title: string; sub: string; hits: { page: string | number; pno: number; score: number; preview?: string; from?: Record<string, number>; rerank?: number }[]; fmt: (h: { score: number; from?: Record<string, number>; rerank?: number }) => string; finalPages: Set<string | number> }) {
  return (
    <div className="grid content-start gap-1.5 rounded-lg border border-border bg-card p-3">
      <div className="text-sm font-medium">{title}</div>
      <div className="-mt-1 text-xs text-muted-foreground">{sub}</div>
      {hits.length ? hits.map((h, i) => (
        <div key={String(h.page)} className={cn("grid grid-cols-[22px_minmax(0,1fr)_auto] items-center gap-2 rounded-md border border-border px-2 py-1.5 text-xs", finalPages.has(h.page) && "border-success/60")}>
          <span className="font-mono text-muted-foreground">#{i + 1}</span>
          <span className="truncate"><span className="font-medium">p.{h.pno}</span>{h.preview ? <span className="text-muted-foreground"> · {h.preview}</span> : null}</span>
          <span className="font-mono text-[11px]">{fmt(h)}</span>
        </div>
      )) : <div className="text-xs text-muted-foreground">no candidates</div>}
    </div>
  );
}

const explainRequest = (question: string, workspace: string) => j<Explain>("/api/retrieve/explain", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, k: 5, workspace }) });

function ExplainView({ ws, initial }: { ws: Workspace; initial: string }) {
  const [q, setQ] = useState(initial);
  const [res, setRes] = useState<Explain | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async (question: string) => { if (!question.trim()) return; setBusy(true); try { setRes(await explainRequest(question, ws.id)); } finally { setBusy(false); } };
  useEffect(() => { if (!initial.trim()) return; let on = true; explainRequest(initial, ws.id).then((r) => { if (on) setRes(r); }); return () => { on = false; }; }, [initial, ws.id]);
  const finalPages = new Set<string | number>(res?.final.map((h) => h.page) ?? []);
  return (
    <div className="grid gap-3">
      <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); run(q); }}>
        <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask a question and watch retrieval happen" className="max-w-xl" />
        <Button type="submit" disabled={busy}>Explain</Button>
      </form>
      {(busy || (initial.trim() && !res)) ? <MessageLoading /> : null}
      {res ? (
        <div className="grid gap-3">
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            {Object.entries(res.timings_ms).map(([k, v]) => <span key={k}>{k} <b className="font-mono text-foreground">{v} ms</b></span>)}
            <span>terms <b className="font-mono text-foreground">{res.terms.join(" ")}</b></span>
            <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full border border-success/60" /> in final top {res.final.length}</span>
          </div>
          <div className="grid gap-3 md:grid-cols-3">
            <HitList title="Visual · ColQwen" sub="MaxSim over page multivectors, matt_pages" hits={res.paths.visual} fmt={(h) => h.score.toFixed(1)} finalPages={finalPages} />
            <HitList title="Dense · bge-m3" sub="cosine over chunks, rolled up to pages" hits={res.paths.dense} fmt={(h) => h.score.toFixed(3)} finalPages={finalPages} />
            <HitList title="Sparse · BM25" sub="exact terms over chunks, rolled up to pages" hits={res.paths.bm25} fmt={(h) => h.score.toFixed(2)} finalPages={finalPages} />
          </div>
          <div className="grid gap-3 md:grid-cols-3">
            <HitList title="Text hybrid" sub="RRF of dense + BM25 (k = 60)" hits={res.text_hybrid} fmt={(h) => `${h.score.toFixed(4)} · d#${h.from?.dense ?? "–"} b#${h.from?.bm25 ?? "–"}`} finalPages={finalPages} />
            <HitList title="Fused" sub="RRF of visual + text hybrid" hits={res.fused} fmt={(h) => `${h.score.toFixed(4)} · v#${h.from?.visual ?? "–"} t#${h.from?.text ?? "–"}`} finalPages={finalPages} />
            <HitList title="Reranked" sub="bge-reranker-v2-m3 on the fused top 10" hits={res.reranked} fmt={(h) => (h.rerank ?? 0).toFixed(3)} finalPages={finalPages} />
          </div>
        </div>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------- page

export default function EvidencePage() {
  const [items, setItems] = useState<Workspace[] | null>(null);
  const [selId, setSelId] = useState<string | null>(null);
  const [tab, setTab] = useState("queries");
  const [explainQ, setExplainQ] = useState("");
  useEffect(() => {
    let on = true;
    j<{ workspaces: Workspace[] }>("/api/evidence/workspaces").then((r) => { if (!on) return; setItems(r.workspaces); setSelId((cur) => cur ?? r.workspaces.find((w) => w.turns > 0)?.id ?? r.workspaces[0]?.id ?? null); });
    return () => { on = false; };
  }, []);
  const ws = items?.find((w) => w.id === selId) ?? null;
  const remove = async (id: string) => {
    await j<{ deleted: Record<string, number | string> }>(`/api/evidence/workspaces/${encodeURIComponent(id)}`, { method: "DELETE" });
    const rest = (items ?? []).filter((w) => w.id !== id);
    setItems(rest);
    if (selId === id) { setSelId(rest.find((w) => w.turns > 0)?.id ?? rest[0]?.id ?? null); setTab("queries"); setExplainQ(""); }
  };
  const subtitle = ws ? `${ws.title} · ${ws.turns} question${ws.turns === 1 ? "" : "s"}` : items ? "what each answer was built from" : "loading";
  return (
    <AppShell title="Evidence" subtitle={subtitle}>
      <div className="grid gap-5 p-4 sm:p-6 lg:grid-cols-[280px_minmax(0,1fr)]">
        {items ? <WorkspaceList items={items} selected={selId} onSelect={(id) => { setSelId(id); setTab("queries"); setExplainQ(""); }} onDelete={remove} /> : <MessageLoading />}
        {ws ? (
          <div className="grid min-w-0 content-start gap-4">
            <Summary ws={ws} />
            <Tabs value={tab} onValueChange={(v) => setTab(String(v))}>
              <TabsList>
                <TabsTrigger value="queries">Queries</TabsTrigger>
                <TabsTrigger value="chunks">Chunks</TabsTrigger>
                <TabsTrigger value="search">Search chunks</TabsTrigger>
                <TabsTrigger value="explain">Explain a query</TabsTrigger>
              </TabsList>
              <TabsContent value="queries" className="pt-4"><Queries key={ws.id} ws={ws} onExplain={(q) => { setExplainQ(q); setTab("explain"); }} /></TabsContent>
              <TabsContent value="chunks" className="pt-4">{ws.documents.length ? <Chunks key={ws.id} ws={ws} /> : <div className="text-sm text-muted-foreground">No document indexed in this workspace yet.</div>}</TabsContent>
              <TabsContent value="search" className="pt-4"><Search key={ws.id} ws={ws} /></TabsContent>
              <TabsContent value="explain" className="pt-4"><ExplainView key={`${ws.id}|${explainQ}`} ws={ws} initial={explainQ} /></TabsContent>
            </Tabs>
          </div>
        ) : null}
      </div>
    </AppShell>
  );
}
