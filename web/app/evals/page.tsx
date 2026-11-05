"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ChevronRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
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
const WHAT: Record<string, string> = {
  bm25: "sparse exact-term match over chunks", text_hybrid: "bge-m3 dense + BM25, RRF inside Qdrant", text_hybrid_rerank: "text-hybrid top 20 re-scored by bge-reranker-v2-m3", visual: "ColQwen2.5 MaxSim over page images",
  fused: "RRF of the visual and text lists", fused_rerank: "fused top 20 re-scored by bge-reranker-v2-m3",
};
const pctf = (v: number) => `${(v * 100).toFixed(1)}%`;
const when = (iso: string) => iso.replace("T", " ").slice(0, 16);
const delta = (v: number, base: number) => { const d = (v - base) * 100; return d === 0 ? "" : `${d > 0 ? "+" : ""}${d.toFixed(1)}`; };

function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return <th className={cn("px-3 py-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground", right ? "text-right" : "text-left")}>{children}</th>;
}
function Td({ children, right, mono, className }: { children: React.ReactNode; right?: boolean; mono?: boolean; className?: string }) {
  return <td className={cn("px-3 py-2 align-top", right && "text-right", mono && "font-mono text-[12.5px]", className)}>{children}</td>;
}

const label = (s: Summary, c: string) => s.info?.[c]?.label ?? LABEL[c] ?? c;
const what = (s: Summary, c: string) => s.info?.[c]?.detail ?? WHAT[c] ?? "";
const kind = (ct: string[]) => (ct.includes("Table") ? "Table" : ct.includes("Infographic") || ct.includes("Image") ? "Figure" : "Text");

function Stat({ label, before, after, fmt, lowerIsBetter }: { label: string; before: number; after: number; fmt: (v: number) => string; lowerIsBetter?: boolean }) {
  const d = after - before, good = lowerIsBetter ? d < 0 : d > 0;
  return (
    <div className="rounded-lg border border-border bg-card px-3 py-2.5">
      <div className="text-[10.5px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 flex flex-wrap items-baseline gap-x-2 font-mono text-[13px]"><span className="text-muted-foreground">{fmt(before)}</span><span className="text-muted-foreground">→</span><span className="text-[15px] font-medium">{fmt(after)}</span>
        {d !== 0 ? <span className={cn("text-[11px]", good ? "text-success" : "text-destructive")}>{d > 0 ? "+" : ""}{fmt(d)}</span> : null}</div>
    </div>
  );
}

/** Baseline → improvement: headline deltas, per-query wins and losses, and where the gain lands by content type. */
function Deltas({ s }: { s: Summary }) {
  const base = s.latest.find((r) => r.config === s.baseline), impr = s.latest.find((r) => r.config === s.improvement);
  const [pq, setPq] = useState<{ base: PerQuery[]; impr: PerQuery[] } | null>(null);
  const baseId = base?.id, imprId = impr?.id;
  useEffect(() => {
    if (!baseId || !imprId) return;
    let on = true;
    Promise.all([j<RunRow & { per_query: PerQuery[] }>(`/api/evals/run?id=${encodeURIComponent(baseId)}`), j<RunRow & { per_query: PerQuery[] }>(`/api/evals/run?id=${encodeURIComponent(imprId)}`)])
      .then(([b, i]) => { if (on) setPq({ base: b.per_query ?? [], impr: i.per_query ?? [] }); }).catch(() => {});
    return () => { on = false; };
  }, [baseId, imprId]);
  if (!base || !impr) return null;
  const mb = base.metrics, mi = impr.metrics;
  let wins = 0, ties = 0, losses = 0;
  const byKind: Record<string, { n: number; bHit: number; iHit: number; bRR: number; iRR: number }> = {};
  if (pq) {
    const bm = new Map(pq.base.map((q) => [q.query_id, q]));
    for (const qi of pq.impr) {
      const qb = bm.get(qi.query_id);
      if (!qb) continue;
      const rb = qb.first_relevant_rank ?? 99, ri = qi.first_relevant_rank ?? 99;
      if (ri < rb) wins++; else if (ri > rb) losses++; else ties++;
      const g = (byKind[kind(qi.content_type)] ??= { n: 0, bHit: 0, iHit: 0, bRR: 0, iRR: 0 });
      g.n++; g.bHit += qb["hit@5"] ? 1 : 0; g.iHit += qi["hit@5"] ? 1 : 0;
      g.bRR += qb.first_relevant_rank ? 1 / qb.first_relevant_rank : 0; g.iRR += qi.first_relevant_rank ? 1 / qi.first_relevant_rank : 0;
    }
  }
  const ms = (v: number) => `${Math.round(v)} ms`;
  return (
    <div className="grid gap-3">
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        <Stat label="nDCG@5" before={mb["ndcg@5"]} after={mi["ndcg@5"]} fmt={pctf} />
        <Stat label="Recall@5" before={mb["recall@5"]} after={mi["recall@5"]} fmt={pctf} />
        <Stat label="MRR@10" before={mb["mrr@10"]} after={mi["mrr@10"]} fmt={pctf} />
        <Stat label="retrieval p50" before={mb.latency_ms.total.p50} after={mi.latency_ms.total.p50} fmt={ms} lowerIsBetter />
      </div>
      {pq ? (
        <div className="grid gap-3 lg:grid-cols-[auto_minmax(0,1fr)]">
          <div className="rounded-lg border border-border bg-card px-3.5 py-2.5 text-[13px]">
            <div className="text-[10.5px] uppercase tracking-wide text-muted-foreground">Per query, first labelled page</div>
            <div className="mt-1 flex flex-wrap gap-x-4 font-mono"><span className="text-success">{wins} better</span><span className="text-muted-foreground">{ties} same</span><span className="text-destructive">{losses} worse</span></div>
            <div className="mt-1 text-xs text-muted-foreground">{label(s, s.improvement)} vs {label(s, s.baseline)}, rank of the first labelled page in the top 10</div>
          </div>
          <div className="overflow-x-auto rounded-lg border border-border bg-card">
            <table className="w-full text-[13px]">
              <thead className="border-b border-border"><tr><Th>Question needs</Th><Th right>queries</Th><Th right>hit@5 base → impr</Th><Th right>MRR base → impr</Th></tr></thead>
              <tbody>
                {(["Table", "Figure", "Text"] as const).filter((k) => byKind[k]).map((k) => { const g = byKind[k]; const dh = (g.iHit - g.bHit) / g.n, dr = (g.iRR - g.bRR) / g.n; return (
                  <tr key={k} className="border-b border-border last:border-b-0">
                    <Td className="font-medium">{k === "Table" ? "a table" : k === "Figure" ? "a figure or infographic" : "running text"}</Td>
                    <Td right mono>{g.n}</Td>
                    <Td right mono>{pctf(g.bHit / g.n)} → {pctf(g.iHit / g.n)} <span className={cn(dh > 0 ? "text-success" : dh < 0 ? "text-destructive" : "text-muted-foreground")}>{delta(g.iHit / g.n, g.bHit / g.n) || "±0"}</span></Td>
                    <Td right mono>{pctf(g.bRR / g.n)} → {pctf(g.iRR / g.n)} <span className={cn(dr > 0 ? "text-success" : dr < 0 ? "text-destructive" : "text-muted-foreground")}>{delta(g.iRR / g.n, g.bRR / g.n) || "±0"}</span></Td>
                  </tr>); })}
              </tbody>
            </table>
          </div>
        </div>
      ) : <MessageLoading />}
    </div>
  );
}

function Comparison({ s }: { s: Summary }) {
  const base = s.latest.find((r) => r.config === s.baseline)?.metrics;
  return (
    <div className="overflow-x-auto rounded-xl border border-border bg-card">
      <table className="w-full text-[13px]">
        <thead className="border-b border-border">
          <tr><Th>Retrieval config</Th><Th right>nDCG@5</Th><Th right>Δ vs baseline</Th><Th right>Recall@5</Th><Th right>MRR@10</Th><Th right>hit@5</Th><Th right>mean</Th><Th right>p95</Th><Th>hardware</Th></tr>
        </thead>
        <tbody>
          {s.latest.map((r) => {
            const m = r.metrics, role = r.config === s.baseline ? "baseline" : r.config === s.improvement ? "improvement" : null;
            return (
              <tr key={r.id} className={cn("border-b border-border last:border-b-0", role === "improvement" && "bg-success/5")}>
                <Td>
                  <div className="flex items-center gap-2"><span className="font-medium">{label(s, r.config)}</span>{role ? <Badge variant="outline" className="font-mono text-[10px]">{role}</Badge> : null}</div>
                  <div className="text-xs text-muted-foreground">{what(s, r.config)}</div>
                </Td>
                <Td right mono className="font-medium text-foreground">{pctf(m["ndcg@5"])}</Td>
                <Td right mono className={cn(base && m["ndcg@5"] > base["ndcg@5"] ? "text-success" : base && m["ndcg@5"] < base["ndcg@5"] ? "text-destructive" : "text-muted-foreground")}>{base ? delta(m["ndcg@5"], base["ndcg@5"]) : ""}</Td>
                <Td right mono>{pctf(m["recall@5"])}</Td>
                <Td right mono>{pctf(m["mrr@10"])}</Td>
                <Td right mono>{pctf(m["hit@5"])}</Td>
                <Td right mono>{m.latency_ms.total.mean} ms</Td>
                <Td right mono>{m.latency_ms.total.p95} ms</Td>
                <Td className="text-xs text-muted-foreground">{r.hardware} · {when(r.created_at)}</Td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

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
              <h2 className="text-sm font-medium">Baseline → improvement</h2>
              <p className="max-w-[80ch] text-xs text-muted-foreground">{label(s, s.baseline)} is the standard text RAG; {label(s, s.improvement)} adds the ColQwen page-image list before fusion. Same queries, same labels, same index.</p>
              <Deltas s={s} />
            </section>
            <section className="grid gap-2">
              <h2 className="text-sm font-medium">Retrieval quality by config</h2>
              <p className="max-w-[80ch] text-xs text-muted-foreground">Each query has one or more labelled evidence pages (grade 1 or 2). nDCG@5 rewards ranking the graded pages early, Recall@5 is the share of labelled pages found in the top 5, MRR@10 is the reciprocal rank of the first labelled page, hit@5 is the share of queries with any labelled page in the top 5. Latency is retrieval only, measured on the listed hardware.</p>
              <Comparison s={s} />
            </section>
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
