"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Check, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { AppShell } from "@/components/app-shell";
import { ChatComposer } from "@/components/ui/v0-ai-chat";
import { MessageLoading } from "@/components/ui/message-loading";
import { RetrievalSteps } from "@/components/chat/chat-view";
import { AnswerText } from "@/components/chat/answer-text";
import { EvidencePanel, PageThumb } from "@/components/chat/evidence-panel";
import { streamChat } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { AssistantMeta, ChatStep, Confidence, EvidenceChunk } from "@/lib/types";

/**
 * ViDoRe Pipeline: the baseline (text hybrid) and the improvement (text hybrid + cross-encoder rerank) side by side.
 * Two short setup cards name the configs; compare mode runs one question through both configs at once
 * and ticks the pages the benchmark labels as relevant, so the difference is visible on real pages, not only in a table.
 */
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";
const j = async <T,>(url: string): Promise<T> => { const r = await fetch(`${BASE}${url}`); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); };

type MKey = "ndcg@5" | "recall@5" | "mrr@10" | "hit@5";
type Role = "baseline" | "improvement";
type Mode = "compare" | Role;
interface Lat { mean: number; p50: number; p95: number }
interface Metrics extends Record<MKey, number> { n_queries: number; latency_ms: Record<string, Lat> }
interface RunRow { id: string; config: string; metrics: Metrics; hardware: string; created_at: string }
interface Stages { visual: boolean; dense: boolean; bm25: boolean; rerank: boolean }
interface Info { label: string; detail: string; stages: Stages; role: Role | null }
interface Summary { configs: string[]; latest: RunRow[]; baseline: string; improvement: string; info?: Record<string, Info> }
interface BenchQuery { query_id: number; query: string; answer: string; content_type: string[]; relevant: [number, number][]; ranks: { baseline: number | null; improvement: number | null } }
interface Column { config: string; role: Role; steps: ChatStep[]; text: string; meta?: AssistantMeta; pending: boolean; startedAt: number }
interface Turn { id: string; question: string; bench?: BenchQuery; columns: Column[] }

const CONF: Record<Confidence, string> = { high: "bg-success", medium: "bg-warning", low: "bg-destructive" };

// ------------------------------------------------------------------ setup cards

function SetupCard({ role, info }: { role: Role; info?: Info }) {
  return (
    <div className={cn("rounded-xl border border-border bg-card px-4 py-3.5", role === "improvement" && "border-success/40")}>
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="outline" className={cn("font-mono text-[10px] uppercase", role === "improvement" && "border-success/50 text-success")}>{role}</Badge>
        <span className="font-medium">{info?.label ?? role}</span>
      </div>
      <p className="mt-1.5 text-[12.5px] text-muted-foreground">{info?.detail}</p>
    </div>
  );
}

// ------------------------------------------------------------------ compare turns

function PageStrip({ chunks, relevant, cited }: { chunks: EvidenceChunk[]; relevant: Set<number>; cited: Set<number> }) {
  return (
    <div className="flex gap-2 overflow-x-auto pb-1">
      {chunks.map((c, i) => {
        const rel = relevant.has(c.pno);
        return (
          <div key={`${c.id}-${i}`} className="w-[76px] shrink-0" title={`page ${c.pno}${c.scores.visualRank != null ? ` · visual #${c.scores.visualRank}` : ""}${c.scores.textRank != null ? ` · text #${c.scores.textRank}` : ""}`}>
            <div className={cn("relative rounded", rel && "ring-2 ring-success ring-offset-1 ring-offset-card")}>
              <PageThumb chunk={c} />
              <span className="absolute left-1 top-1 rounded bg-foreground/80 px-1 font-mono text-[10px] text-background">#{i + 1}</span>
              {rel ? <span className="absolute right-1 top-1 grid size-4 place-items-center rounded-full bg-success text-white" aria-label="labelled relevant"><Check className="size-3" /></span> : null}
            </div>
            <div className="mt-1 truncate text-center font-mono text-[10px] text-muted-foreground">p.{c.pno}{cited.has(c.pno) ? " · cited" : ""}</div>
          </div>
        );
      })}
    </div>
  );
}

function ColumnView({ col, info, relevant, messageId }: { col: Column; info?: Info; relevant: Set<number>; messageId: string }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState<number | null>(null);
  const meta = col.meta;
  const retrieved = meta?.retrieved ?? [];
  const firstRel = retrieved.findIndex((c) => relevant.has(c.pno));
  const cited = new Set(meta?.chunks.map((c) => c.pno) ?? []);
  const jump = (i: number) => {
    setOpen(true); setActive(i);
    requestAnimationFrame(() => document.getElementById(`chunk-${messageId}-${i}`)?.scrollIntoView({ block: "nearest", behavior: "smooth" }));
    setTimeout(() => setActive(null), 1600);
  };
  return (
    <div className={cn("grid min-w-0 content-start gap-2.5 rounded-xl border border-border bg-card p-3.5", col.role === "improvement" && "border-success/40")}>
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <Badge variant="outline" className={cn("font-mono text-[10px] uppercase", col.role === "improvement" && "border-success/50 text-success")}>{col.role}</Badge>
        <span className="font-medium text-foreground">{info?.label ?? col.config}</span>
        {meta ? <span className="ml-auto font-mono text-[11px]">retrieval {meta.retrievalMs} ms · TTR {(meta.ttrMs / 1000).toFixed(1)} s</span> : null}
      </div>
      {col.steps.length ? <RetrievalSteps steps={col.steps} pending={col.pending} startedAt={col.startedAt} chunks={meta?.chunks.length} /> : null}
      {retrieved.length ? (
        <div className="grid gap-1.5">
          <div className="flex flex-wrap items-center gap-x-2 text-[11.5px] text-muted-foreground">
            <span className="font-medium text-foreground">Retrieved pages</span>
            {relevant.size ? (
              <span className={cn("inline-flex items-center gap-1", firstRel >= 0 ? "text-success" : "text-destructive")}>
                <span className={cn("size-1.5 rounded-full", firstRel >= 0 ? "bg-success" : "bg-destructive")} aria-hidden="true" />
                {firstRel >= 0 ? `first labelled page at rank ${firstRel + 1}` : "no labelled page in the top 5"}
              </span>
            ) : null}
          </div>
          <PageStrip chunks={retrieved} relevant={relevant} cited={cited} />
        </div>
      ) : null}
      {col.pending && !col.text ? (col.steps.length ? null : <MessageLoading />) : <AnswerText text={col.text} onHover={setActive} onClick={jump} />}
      {meta ? (
        <>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px] text-muted-foreground">
            <span className="inline-flex items-center gap-1.5"><span className={cn("size-1.5 rounded-full", CONF[meta.confidence])} aria-hidden="true" />confidence {meta.confidence}</span>
            <span>{meta.chunks.length ? `${meta.chunks.length} chunk${meta.chunks.length > 1 ? "s" : ""} cited` : "nothing cited"}</span>
            {meta.cost != null ? <span className="font-mono">${meta.cost.toFixed(3)}</span> : null}
          </div>
          {meta.chunks.length ? <EvidencePanel messageId={messageId} chunks={meta.chunks} open={open} onOpenChange={setOpen} activeIndex={active} /> : null}
        </>
      ) : null}
    </div>
  );
}

function TurnView({ turn, info }: { turn: Turn; info: Record<string, Info> }) {
  const relevant = useMemo(() => new Set(turn.bench?.relevant.map(([p]) => p) ?? []), [turn.bench]);
  return (
    <div className="grid gap-3">
      <div className="grid max-w-[85%] justify-self-end gap-1.5">
        <div className="flex flex-wrap justify-end gap-x-2 font-mono text-[11px] text-muted-foreground">
          {turn.bench ? <span>benchmark #{turn.bench.query_id} · {turn.bench.content_type.join(", ")} · labelled {turn.bench.relevant.map(([p]) => `p.${p}`).join(" ")}</span> : null}
          <span>you</span>
        </div>
        <div className="rounded-2xl rounded-br-sm border border-border bg-secondary px-4 py-2.5 text-[15px] leading-7">{turn.question}</div>
      </div>
      <div className={cn("grid gap-3", turn.columns.length > 1 && "lg:grid-cols-2")}>
        {turn.columns.map((c) => <ColumnView key={c.config} col={c} info={info[c.config]} relevant={relevant} messageId={`${turn.id}-${c.config}`} />)}
      </div>
      {turn.bench ? <div className="rounded-lg border border-dashed border-border px-3.5 py-2 text-[12.5px] text-muted-foreground"><span className="font-medium text-foreground">Reference answer </span>{turn.bench.answer}</div> : null}
    </div>
  );
}

// ------------------------------------------------------------------ benchmark picker + mode toggle

function Rank({ label, r }: { label: string; r: number | null }) {
  return <span className={cn("rounded border px-1.5 py-0.5", r == null ? "border-destructive/40 text-destructive" : r <= 3 ? "border-success/40 text-success" : "border-border text-muted-foreground")}>{label} {r == null ? "miss" : `#${r}`}</span>;
}

function BenchPicker({ queries, onPick, onClose }: { queries: BenchQuery[]; onPick: (q: BenchQuery) => void; onClose: () => void }) {
  const [filter, setFilter] = useState("");
  const [only, setOnly] = useState<"all" | "table" | "gap">("all");
  const list = useMemo(() => queries.filter((q) => {
    if (filter && !q.query.toLowerCase().includes(filter.toLowerCase())) return false;
    if (only === "table" && !q.content_type.includes("Table")) return false;
    if (only === "gap") { const b = q.ranks.baseline, i = q.ranks.improvement; if (!(i != null && (b == null || i < b))) return false; }
    return true;
  }), [queries, filter, only]);
  return (
    <div className="absolute inset-x-0 bottom-full z-10 mb-2 rounded-xl border border-border bg-card shadow-lg">
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-3 py-2">
        <input autoFocus value={filter} onChange={(e) => setFilter(e.target.value)} placeholder={`Filter the ${queries.length} benchmark queries`} className="min-w-[12rem] flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground" />
        {(["all", "table", "gap"] as const).map((k) => (
          <button key={k} type="button" onClick={() => setOnly(k)} className={cn("rounded-full border px-2 py-0.5 text-[11px]", only === k ? "border-foreground text-foreground" : "border-border text-muted-foreground hover:text-foreground")}>
            {k === "all" ? "all" : k === "table" ? "table questions" : "improved beats baseline"}
          </button>
        ))}
        <button type="button" onClick={onClose} aria-label="close" className="text-muted-foreground hover:text-foreground"><X className="size-4" /></button>
      </div>
      <div className="max-h-72 overflow-y-auto">
        {list.map((q) => (
          <button key={q.query_id} type="button" onClick={() => onPick(q)} className="grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border-b border-border px-3 py-2 text-left last:border-b-0 hover:bg-secondary/60">
            <div className="min-w-0">
              <div className="truncate text-[13px]">{q.query}</div>
              <div className="mt-0.5 flex flex-wrap gap-x-2.5 text-[11px] text-muted-foreground"><span className="font-mono">#{q.query_id}</span><span>{q.content_type.join(", ")}</span><span>relevant {q.relevant.map(([p]) => `p.${p}`).join(" ")}</span></div>
            </div>
            <div className="flex gap-1.5 font-mono text-[10.5px]"><Rank label="base" r={q.ranks.baseline} /><Rank label="impr" r={q.ranks.improvement} /></div>
          </button>
        ))}
        {!list.length ? <div className="px-3 py-4 text-center text-xs text-muted-foreground">{queries.length ? "nothing matches" : "no retrieval eval run stored yet"}</div> : null}
      </div>
    </div>
  );
}

function ModeToggle({ mode, onChange, disabled }: { mode: Mode; onChange: (m: Mode) => void; disabled?: boolean }) {
  const opts: [Mode, string][] = [["compare", "Compare"], ["baseline", "Baseline"], ["improvement", "Improved"]];
  return (
    <div className="inline-flex rounded-lg border border-border bg-card p-0.5 text-xs">
      {opts.map(([m, l]) => (
        <button key={m} type="button" disabled={disabled} onClick={() => onChange(m)} className={cn("rounded-md px-2.5 py-1 transition-colors", mode === m ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground")}>{l}</button>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ page

export default function PipelinePage() {
  const [s, setS] = useState<Summary | null>(null);
  const [bench, setBench] = useState<BenchQuery[]>([]);
  const [mode, setMode] = useState<Mode>("compare");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [picker, setPicker] = useState(false);
  const [preset, setPreset] = useState<number | null>(null);      // ?q=<query_id>, from the Retrieval Evals "open in compare" link
  const idRef = useRef(0);
  const threadRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    try { const v = new URLSearchParams(window.location.search).get("q"); if (v != null && v !== "") setPreset(Number(v)); } catch { /* ignore */ }
    j<Summary>("/api/evals").then(setS).catch(() => setS({ configs: [], latest: [], baseline: "text_hybrid", improvement: "text_hybrid_rerank" }));
    j<{ queries: BenchQuery[] }>("/api/evals/queries").then((r) => setBench(r.queries ?? [])).catch(() => {});
  }, []);
  useEffect(() => { const el = threadRef.current; if (el && turns.length) el.scrollTop = el.scrollHeight; }, [turns]);

  const patch = useCallback((turnId: string, config: string, fn: (c: Column) => Column) => {
    setTurns((ts) => ts.map((t) => (t.id === turnId ? { ...t, columns: t.columns.map((c) => (c.config === config ? fn(c) : c)) } : t)));
  }, []);

  const run = useCallback(async (question: string, b?: BenchQuery) => {
    if (busy || !s || !question.trim()) return;
    setPicker(false); setPreset(null);
    const configs = mode === "compare" ? [s.baseline, s.improvement] : mode === "baseline" ? [s.baseline] : [s.improvement];
    const id = `t${++idRef.current}`, hex = Math.random().toString(16).slice(2, 6);
    setTurns((ts) => [...ts, { id, question, bench: b, columns: configs.map((c) => ({ config: c, role: c === s.baseline ? "baseline" : "improvement", steps: [], text: "", pending: true, startedAt: Date.now() })) }]);
    setBusy(true);
    await Promise.all(configs.map(async (config) => {
      const p = (fn: (c: Column) => Column) => patch(id, config, fn);
      try {
        // One conversation per column, so the Evidence tab keeps the two runs apart.
        await streamChat({ question, conversationId: `conv_cmp_${hex}_${config === s.baseline ? "base" : "impr"}`, source: "compare", mode: config }, (e) => {
          if (e.type === "step") p((c) => { const steps = [...c.steps]; const i = steps.findIndex((x) => x.name === e.step.name); if (i >= 0) steps[i] = e.step; else steps.push(e.step); return { ...c, steps }; });
          else if (e.type === "token") p((c) => ({ ...c, text: c.text + e.text }));
          else if (e.type === "evidence") p((c) => ({ ...c, meta: e.meta, pending: false, text: c.text.trimEnd() }));
          else if (e.type === "error") p((c) => ({ ...c, text: e.message, pending: false }));
        });
      } catch (err) {
        p((c) => ({ ...c, text: err instanceof Error ? err.message : "Request failed", pending: false }));
      } finally {
        p((c) => ({ ...c, pending: false }));
      }
    }));
    setBusy(false);
  }, [busy, s, mode, patch]);

  const info = s?.info ?? {};
  const presetQ = preset != null ? bench.find((b) => b.query_id === preset) : undefined;

  return (
    <AppShell
      title="ViDoRe Pipeline"
      subtitle="baseline vs improvement"
      actions={<ModeToggle mode={mode} onChange={setMode} disabled={busy} />}
    >
      <div className="flex h-[calc(100svh-3.5rem)] min-h-0 flex-col">
        <div ref={threadRef} className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto grid w-full max-w-5xl gap-6 px-4 py-6">
            <div className="grid gap-3">
              <div className="grid gap-3 lg:grid-cols-2">
                <SetupCard role="baseline" info={s ? info[s.baseline] : undefined} />
                <SetupCard role="improvement" info={s ? info[s.improvement] : undefined} />
              </div>
            </div>
            {presetQ ? (
              <div className="flex flex-wrap items-center gap-3 rounded-xl border border-border bg-card px-4 py-3 text-[13px]">
                <div className="min-w-0 flex-1">
                  <div className="font-mono text-[11px] text-muted-foreground">benchmark #{presetQ.query_id} · {presetQ.content_type.join(", ")} · labelled {presetQ.relevant.map(([p]) => `p.${p}`).join(" ")} · stored ranks base {presetQ.ranks.baseline ?? "miss"} / impr {presetQ.ranks.improvement ?? "miss"}</div>
                  <div className="mt-0.5">{presetQ.query}</div>
                </div>
                <button type="button" disabled={busy || !s} onClick={() => run(presetQ.query, presetQ)} className="rounded-full bg-foreground px-3.5 py-1.5 text-xs font-medium text-background disabled:opacity-50">Run compare</button>
                <button type="button" onClick={() => setPreset(null)} className="text-xs text-muted-foreground hover:text-foreground">dismiss</button>
              </div>
            ) : null}
            {turns.length ? turns.map((t) => <TurnView key={t.id} turn={t} info={info} />) : (
              <div className="rounded-xl border border-dashed border-border px-4 py-6 text-center text-[13px] text-muted-foreground">
                Ask anything from the textbook, or pick a benchmark query below. In compare mode the question runs through both setups at once, and the pages the benchmark labels as relevant get a green tick in each list.
              </div>
            )}
          </div>
        </div>
        <div className="shrink-0 bg-gradient-to-t from-background via-background to-transparent px-4 pb-4 pt-2">
          <div className="relative mx-auto grid w-full max-w-5xl gap-2">
            {picker ? <BenchPicker queries={bench} onPick={(q) => run(q.query, q)} onClose={() => setPicker(false)} /> : null}
            <ChatComposer
              placeholder={mode === "compare" ? "Ask the textbook; runs baseline and improvement side by side" : `Ask the textbook (${mode === "baseline" ? "baseline" : "improved"} only)`}
              disabled={busy} onSend={(t) => run(t)} model="claude-opus-5-5"
              actions={[{ label: `Benchmark queries${bench.length ? ` (${bench.length})` : ""}`, onClick: () => setPicker((v) => !v) }]}
            />
            <div className="flex flex-wrap gap-x-4 gap-y-1 px-1 text-xs text-muted-foreground">
              <span>{mode === "compare" ? "Compare runs two retrievals and two Claude answers per question, roughly $0.10." : "Single mode runs one retrieval and one answer."}</span>
              <span>Green ring: page labelled relevant in the benchmark. Numbers: final rank.</span>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
