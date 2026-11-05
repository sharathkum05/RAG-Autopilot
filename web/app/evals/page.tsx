"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ChevronRight } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { AppShell } from "@/components/app-shell";
import { MessageLoading } from "@/components/ui/message-loading";
import { cn } from "@/lib/utils";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";
const j = async <T,>(url: string): Promise<T> => { const r = await fetch(`${BASE}${url}`); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); };

interface Lat { mean: number; p50: number; p95: number }
interface Metrics { "ndcg@5": number; "ndcg@10": number; "recall@5": number; "recall@10": number; "mrr@10": number; "precision@5": number; "hit@5": number; n_queries: number; k: number; latency_ms: Record<string, Lat> }
interface RunRow { id: string; workspace: string; dataset: string; config: string; metrics: Metrics; hardware: string; created_at: string }
interface Top { page: number; score: number; visual_rank: number | null; text_rank: number | null; rerank: number | null }
interface PerQuery { query_id: number; query: string; answer: string; content_type: string[]; relevant: [number, number][]; retrieved: number[]; first_relevant_rank: number | null; "hit@5": boolean; ms: number; timings_ms: Record<string, number>; top: Top[] }
interface Summary { configs: string[]; latest: RunRow[]; runs: RunRow[]; baseline: string; improvement: string; info?: Record<string, { label: string; detail: string }> }

const LABEL: Record<string, string> = { bm25: "BM25 only", text_hybrid: "Hybrid RAG without reranker", text_hybrid_rerank: "Hybrid RAG with reranker", visual: "Visual only (ColQwen)", fused: "Fused (visual + text)", fused_rerank: "Fused + rerank" };

function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return <th className={cn("px-3 py-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground", right ? "text-right" : "text-left")}>{children}</th>;
}
function Td({ children, right, mono, className }: { children: React.ReactNode; right?: boolean; mono?: boolean; className?: string }) {
  return <td className={cn("px-3 py-2 align-top", right && "text-right", mono && "font-mono text-[12.5px]", className)}>{children}</td>;
}

const label = (s: Summary, c: string) => s.info?.[c]?.label ?? LABEL[c] ?? c;

function Latency({ s }: { s: Summary }) {
  const stages = Array.from(new Set(s.latest.flatMap((r) => Object.keys(r.metrics.latency_ms).filter((k) => k !== "total"))));
  return (
    <div className="overflow-x-auto rounded-xl border border-border bg-card">
      <table className="w-full text-[13px]">
        <thead className="border-b border-border"><tr><Th>Config</Th>{stages.map((st) => <Th key={st} right>{st}</Th>)}<Th right>total p50</Th><Th right>total p95</Th></tr></thead>
        <tbody>
          {s.latest.map((r) => (
            <tr key={r.id} className="border-b border-border last:border-b-0">
              <Td className="font-medium">{label(s, r.config)}</Td>
              {stages.map((st) => { const l = r.metrics.latency_ms[st]; return <Td key={st} right mono className={l ? "" : "text-muted-foreground"}>{l ? `${l.mean} ms` : "–"}</Td>; })}
              <Td right mono>{r.metrics.latency_ms.total.p50} ms</Td>
              <Td right mono>{r.metrics.latency_ms.total.p95} ms</Td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function QueryRow({ q }: { q: PerQuery }) {
  const [open, setOpen] = useState(false);
  const relevant = new Set(q.relevant.map(([p]) => p));
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-lg border border-border bg-card">
      <CollapsibleTrigger className="group grid w-full grid-cols-[14px_minmax(0,1fr)_auto] items-center gap-3 px-3 py-2.5 text-left">
        <ChevronRight className="size-3.5 text-muted-foreground transition-transform group-data-[panel-open]:rotate-90" />
        <div className="min-w-0">
          <div className="truncate text-[13.5px]">{q.query}</div>
          <div className="mt-0.5 flex flex-wrap gap-x-3 text-xs text-muted-foreground">
            <span className="inline-flex items-center gap-1.5"><span className={cn("size-1.5 rounded-full", q["hit@5"] ? "bg-success" : "bg-destructive")} aria-hidden="true" />{q["hit@5"] ? "relevant page in top 5" : q.first_relevant_rank ? `first relevant at rank ${q.first_relevant_rank}` : "no relevant page in top 10"}</span>
            <span>relevant {q.relevant.map(([p]) => `p.${p}`).join(" ")}</span>
            <span>retrieved {q.retrieved.slice(0, 5).map((p) => `p.${p}`).join(" ")}</span>
            {q.content_type.length ? <span>{q.content_type.join(", ")}</span> : null}
          </div>
        </div>
        <span className="font-mono text-[11px] text-muted-foreground">{q.ms} ms</span>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="grid gap-3 border-t border-border px-3 pb-3 pt-3 text-[13px]">
          <div><span className="text-muted-foreground">Expected answer: </span>{q.answer}</div>
          <Link href={`/pipeline?q=${q.query_id}`} className="w-fit rounded-full border border-border px-3 py-1 text-xs text-muted-foreground hover:border-foreground hover:text-foreground">Open in ViDoRe Pipeline compare →</Link>
          <table className="w-full text-xs">
            <thead><tr><Th>rank</Th><Th>page</Th><Th>relevant</Th><Th right>score</Th><Th right>visual #</Th><Th right>text #</Th><Th right>rerank</Th></tr></thead>
            <tbody>
              {q.top.map((t, i) => (
                <tr key={i} className={cn("border-t border-border", relevant.has(t.page) && "bg-success/5")}>
                  <Td mono>{i + 1}</Td><Td mono>p.{t.page}</Td><Td>{relevant.has(t.page) ? <span className="text-success">yes (grade {q.relevant.find(([p]) => p === t.page)?.[1]})</span> : <span className="text-muted-foreground">no</span>}</Td>
                  <Td right mono>{t.score.toFixed(4)}</Td><Td right mono>{t.visual_rank ?? "–"}</Td><Td right mono>{t.text_rank ?? "–"}</Td><Td right mono>{t.rerank != null ? t.rerank.toFixed(3) : "–"}</Td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex flex-wrap gap-x-3 text-xs text-muted-foreground">{Object.entries(q.timings_ms).map(([k, v]) => <span key={k}>{k} <b className="font-mono text-foreground">{v} ms</b></span>)}</div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

function Queries({ runs, s }: { runs: RunRow[]; s: Summary }) {
  const [sel, setSel] = useState<string>(runs[0]?.id ?? "");
  const [data, setData] = useState<(RunRow & { per_query: PerQuery[] }) | null>(null);
  const [failuresOnly, setFailuresOnly] = useState(true);
  useEffect(() => { if (!sel) return; let on = true; j<RunRow & { per_query: PerQuery[] }>(`/api/evals/run?id=${encodeURIComponent(sel)}`).then((r) => { if (on) setData(r); }); return () => { on = false; }; }, [sel]);
  const rows = (data?.per_query ?? []).filter((q) => !failuresOnly || !q["hit@5"]);
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-1.5">
        {runs.map((r) => <button key={r.id} type="button" onClick={() => setSel(r.id)} className={cn("rounded-full border border-border px-3 py-1 text-xs hover:border-foreground", sel === r.id && "border-foreground bg-secondary")}>{label(s, r.config)}</button>)}
        <label className="ml-auto inline-flex items-center gap-2 text-xs text-muted-foreground"><input type="checkbox" checked={failuresOnly} onChange={(e) => setFailuresOnly(e.target.checked)} className="accent-foreground" />failures only (no relevant page in the top 5)</label>
      </div>
      {data && data.id === sel ? (
        <>
          <div className="text-xs text-muted-foreground">{rows.length} of {data.per_query.length} queries{failuresOnly ? " missed the top 5" : ""}. A miss here is a retrieval failure: the answer page exists in the index but did not rank. Generation failures (right page, wrong answer) are judged separately once answers are generated for these queries.</div>
          <div className="grid gap-1.5">{rows.map((q) => <QueryRow key={q.query_id} q={q} />)}</div>
        </>
      ) : <MessageLoading />}
    </div>
  );
}

export default function EvalsPage() {
  const [s, setS] = useState<Summary | null>(null);
  useEffect(() => { let on = true; j<Summary>("/api/evals").then((r) => { if (on) setS(r); }); return () => { on = false; }; }, []);
  const n = s?.latest[0]?.metrics.n_queries;
  return (
    <AppShell title="Retrieval Evals" subtitle={s?.latest.length ? `${s.latest[0].dataset} · ${n} queries · graded page labels` : s ? "no runs yet" : "loading"}>
      <div className="grid gap-6 p-4 sm:p-6">
        {!s ? <MessageLoading /> : !s.latest.length ? (
          <div className="rounded-lg border border-dashed border-border p-4 text-sm text-muted-foreground">No eval runs yet. Run <code className="font-mono">modal run modal_app.py::evals</code> to score every retrieval config on the textbook queries.</div>
        ) : (
          <>
            <section className="grid gap-2">
              <h2 className="text-sm font-medium">Latency per stage (mean, ms)</h2>
              <Latency s={s} />
            </section>
            <section className="grid gap-2">
              <h2 className="text-sm font-medium">Per-query results and failures</h2>
              <Queries runs={s.latest} s={s} />
            </section>
          </>
        )}
      </div>
    </AppShell>
  );
}
