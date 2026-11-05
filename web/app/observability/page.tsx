"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronRight, ExternalLink } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { MessageLoading } from "@/components/ui/message-loading";
import { cn } from "@/lib/utils";

/**
 * Observability: every traced turn (Playground, Dataset, compare, answer evals) from Langfuse, with the stored turns as a
 * fallback. Where the time and the money go: per-stage latency, tokens and cost, a daily strip, and a waterfall per trace.
 */
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";
const j = async <T,>(url: string): Promise<T> => { const r = await fetch(`${BASE}${url}`, { cache: "no-store" }); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); };

interface Block { n: number; cost_usd: number; tokens_in: number; tokens_out: number; errors: number; latency_ms: { p50: number; p95: number; mean: number }; retrieval_ms: { p50: number; p95: number }; answer_ms: { p50: number; p95: number } }
interface Stage { id: string; name: string; type: string; start_ms: number; ms: number; model: string | null; tokens_in: number; tokens_out: number; cost_usd: number; level: string | null; status: string | null; input?: unknown; output?: unknown; metadata?: unknown }
interface Trace { id: string; name: string; start: string; latency_ms: number; retrieval_ms: number; answer_ms: number; question: string | null; workspace: string | null; config: string | null; source: string; conversation_id: string | null; eval_run: string | null; query_id: number | null; model: string | null; tokens_in: number; tokens_out: number; cost_usd: number; confidence: string | null; cited_pages: string[] | null; error: string | null; level: string | null; scores: Record<string, number>; stages: Stage[]; url: string | null; input?: unknown; output?: unknown }
interface Summary { all: Block; by_source: Record<string, Block>; by_config: Record<string, Block>; by_day: { day: string; n: number; cost_usd: number; p50_ms: number }[]; stages: Record<string, { n: number; p50: number; p95: number; mean: number }> }
interface Overview { source: "langfuse" | "supabase" | "none"; window: string; since: string; error: string | null; summary: Summary | null; traces: Trace[]; total: number }
interface Sample { t: string; cpu: number; ram_gb: number; gpu_util: number; gpu_mem_gb: number }
interface System { source: string; sampled_at: string; container: { id: string; gpu: string; region: string; cloud?: string; uptime_s: number; started_at?: string; model_load_s?: number | null; cold_starts_today: number; state: string; scale_to_zero_after_s: number }; cpu: { percent: number; cores: number }; memory: { used_gb: number; total_gb: number }; gpu: { name?: string; util_percent: number; memory_used_gb: number; memory_total_gb: number; temperature_c: number }; models_loaded: { name: string; memory_gb: number | null }[]; requests: { last_hour: number; p50_ms: number; p95_ms: number; errors: number }; history?: Sample[] }

const WINDOWS: [string, string][] = [["24h", "24 h"], ["7d", "7 days"], ["30d", "30 days"], ["all", "all"]];
const SOURCE_LABEL: Record<string, string> = { playground: "Playground", dataset: "Dataset", compare: "Pipeline compare", answer_eval: "Evals pipeline", chat: "Chat" };
const STAGE_TONE: Record<string, string> = { "Encoding question": "bg-muted-foreground/70", "Visual search": "bg-evidence", "Text search": "bg-foreground/70", Fusing: "bg-warning", Reranking: "bg-warning", Answering: "bg-success" };
const s1 = (ms: number) => `${(ms / 1000).toFixed(1)} s`;
const when = (iso: string) => iso.replace("T", " ").slice(5, 16);
const money = (v: number) => `$${v.toFixed(v >= 1 ? 2 : 3)}`;

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-border bg-card px-3 py-2.5">
      <div className="text-[10.5px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-0.5 font-mono text-[17px] font-medium">{value}</div>
      {sub ? <div className="text-[11px] text-muted-foreground">{sub}</div> : null}
    </div>
  );
}
function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return <th className={cn("px-3 py-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground", right ? "text-right" : "text-left")}>{children}</th>;
}
function Td({ children, right, mono, className }: { children: React.ReactNode; right?: boolean; mono?: boolean; className?: string }) {
  return <td className={cn("px-3 py-2 align-top", right && "text-right", mono && "font-mono text-[12.5px]", className)}>{children}</td>;
}

/** Bars per day: turns on the left axis as bar height, cost as the label. One series, so no legend. */
function DailyStrip({ days }: { days: Summary["by_day"] }) {
  const max = Math.max(1, ...days.map((d) => d.n));
  return (
    <div className="flex h-24 items-end gap-1.5 overflow-x-auto rounded-xl border border-border bg-card px-3 pb-2 pt-3">
      {days.map((d) => (
        <div key={d.day} className="group flex min-w-[44px] flex-1 flex-col items-center gap-1" title={`${d.day}: ${d.n} turns · ${money(d.cost_usd)} · p50 ${s1(d.p50_ms)}`}>
          <div className="font-mono text-[9.5px] text-muted-foreground">{d.n}</div>
          <div className="w-full rounded-t-[4px] bg-foreground/80 transition-colors group-hover:bg-foreground" style={{ height: `${Math.max(2, (d.n / max) * 52)}px` }} />
          <div className="font-mono text-[9.5px] text-muted-foreground">{d.day.slice(5)}</div>
        </div>
      ))}
      {!days.length ? <div className="w-full text-center text-xs text-muted-foreground">no turns in this window</div> : null}
    </div>
  );
}

function Bar({ label, value, max, unit, series }: { label: string; value: number; max: number; unit: string; series?: number[] }) {
  const frac = max > 0 ? Math.min(1, value / max) : 0;
  const smax = Math.max(max, ...(series ?? []), 0.001);
  return (
    <div className="rounded-lg border border-border bg-card px-3 py-2.5">
      <div className="flex items-baseline justify-between"><span className="text-[10.5px] uppercase tracking-wide text-muted-foreground">{label}</span><span className="font-mono text-[12.5px]">{value}{unit}{max ? <span className="text-muted-foreground"> / {max}{unit}</span> : null}</span></div>
      <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-secondary"><div className={cn("h-full rounded-full", frac > 0.9 ? "bg-destructive" : frac > 0.7 ? "bg-warning" : "bg-foreground")} style={{ width: `${frac * 100}%` }} /></div>
      {series && series.length > 1 ? (
        <div className="mt-1.5 flex h-6 items-end gap-px" title="last hour of this container, one sample every 10 s">
          {series.map((v, i) => <div key={i} className="flex-1 rounded-t-[1px] bg-muted-foreground/50" style={{ height: `${Math.max(4, (v / smax) * 100)}%` }} />)}
        </div>
      ) : null}
    </div>
  );
}

/** The container behind the query API: GPU and memory sampled in-process (NVML on Modal), plus the container's own log tail. */
function Runtime() {
  const [sys, setSys] = useState<System | null>(null);
  const [logs, setLogs] = useState<{ lines: string[]; container: string } | null>(null);
  const [logOpen, setLogOpen] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let on = true;
    const tick = () => {
      j<System>("/api/system").then((s) => { if (on) { setSys(s); setErr(null); } }).catch((e: Error) => { if (on) setErr(e.message); });
      if (logOpen) j<{ lines: string[]; container: string }>("/api/system/logs?limit=160").then((l) => { if (on) setLogs(l); }).catch(() => {});
    };
    tick();
    const id = setInterval(tick, 10000);
    return () => { on = false; clearInterval(id); };
  }, [logOpen]);
  const h = sys?.history ?? [];
  const tail = (k: keyof Sample) => h.slice(-90).map((x) => Number(x[k]));
  return (
    <div className="grid gap-2">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-xs text-muted-foreground">
        {sys ? (
          <>
            <span className="inline-flex items-center gap-1.5"><span className="size-1.5 rounded-full bg-success" aria-hidden="true" />{sys.container.id === "local" ? "local backend" : `Modal container ${sys.container.id}`}</span>
            <span>{sys.gpu.name ?? sys.container.gpu}</span>
            {sys.container.region !== "local" ? <span>{sys.container.cloud ? `${sys.container.cloud} · ` : ""}{sys.container.region}</span> : null}
            <span>{sys.container.state} · up {Math.floor(sys.container.uptime_s / 60)} min</span>
            {sys.container.model_load_s != null ? <span>models loaded in {sys.container.model_load_s} s at boot</span> : null}
            {sys.container.scale_to_zero_after_s ? <span>scales to zero after {Math.round(sys.container.scale_to_zero_after_s / 60)} min idle</span> : null}
            <span>{sys.models_loaded.map((m) => m.name).join(" · ") || "no models loaded yet"}</span>
            <span>requests/h {sys.requests.last_hour} · p50 {sys.requests.p50_ms} ms · p95 {sys.requests.p95_ms} ms · errors {sys.requests.errors}</span>
            <span>sampled {sys.sampled_at.slice(11, 19)} UTC</span>
          </>
        ) : err ? <span className="text-destructive">{err}</span> : <span>waking the backend…</span>}
      </div>
      {sys ? (
        <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
          <Bar label="GPU memory" value={sys.gpu.memory_used_gb} max={sys.gpu.memory_total_gb} unit=" GB" series={tail("gpu_mem_gb")} />
          <Bar label="GPU utilisation" value={sys.gpu.util_percent} max={100} unit="%" series={tail("gpu_util")} />
          <Bar label="RAM" value={sys.memory.used_gb} max={sys.memory.total_gb} unit=" GB" series={tail("ram_gb")} />
          <Bar label={`CPU · ${sys.cpu.cores} cores`} value={sys.cpu.percent} max={100} unit="%" series={tail("cpu")} />
        </div>
      ) : null}
      <Collapsible open={logOpen} onOpenChange={setLogOpen} className="rounded-xl border border-border bg-card">
        <CollapsibleTrigger className="group flex w-full items-center gap-2 px-3.5 py-2.5 text-left text-[13px] text-muted-foreground hover:text-foreground">
          <ChevronRight className="size-3.5 shrink-0 transition-transform group-data-[panel-open]:rotate-90" />
          <span className="font-medium text-foreground">Container log</span>
          <span className="text-xs">the backend's own log lines, kept in memory for this container's lifetime</span>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <pre className="max-h-72 overflow-auto border-t border-border px-3.5 py-2 font-mono text-[11px] leading-5 text-muted-foreground">{logs ? (logs.lines.length ? logs.lines.join("\n") : "no log lines yet") : "loading…"}</pre>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

function Waterfall({ t }: { t: Trace }) {
  const total = Math.max(1, t.latency_ms);
  return (
    <div className="grid gap-1">
      {t.stages.map((s) => (
        <div key={s.id} className="grid grid-cols-[150px_minmax(0,1fr)_72px] items-center gap-2 text-[11.5px]">
          <div className="truncate"><span className={cn("mr-1.5 inline-block size-2 rounded-sm align-middle", STAGE_TONE[s.name] ?? "bg-muted-foreground/60")} aria-hidden="true" />{s.name}{s.type === "generation" && s.model ? <span className="text-muted-foreground"> · {s.model}</span> : null}</div>
          <div className="relative h-3.5 rounded bg-secondary"><div className={cn("absolute inset-y-0 rounded", STAGE_TONE[s.name] ?? "bg-muted-foreground/60")} style={{ left: `${(s.start_ms / total) * 100}%`, width: `${Math.max(0.6, (s.ms / total) * 100)}%` }} /></div>
          <div className="text-right font-mono text-muted-foreground">{s.ms >= 1000 ? s1(s.ms) : `${s.ms} ms`}</div>
        </div>
      ))}
      <div className="mt-1 flex flex-wrap gap-x-4 font-mono text-[11px] text-muted-foreground"><span>total {s1(t.latency_ms)}</span><span>retrieval {t.retrieval_ms} ms</span><span>answer {s1(t.answer_ms)}</span>{t.tokens_in ? <span>{t.tokens_in} in · {t.tokens_out} out</span> : null}<span>{money(t.cost_usd)}</span></div>
    </div>
  );
}

function TraceDetail({ id, fallback }: { id: string; fallback: Trace }) {
  const [t, setT] = useState<Trace | null>(null);
  useEffect(() => { let on = true; setT(null); j<Trace>(`/api/observability/trace?id=${encodeURIComponent(id)}`).then((r) => { if (on) setT(r); }).catch(() => { if (on) setT(fallback); }); return () => { on = false; }; }, [id, fallback]);
  const tr = t ?? fallback;
  const answer = tr.stages.find((s) => s.name === "Answering");
  const out = (tr.output ?? null) as { answer?: string } | null;
  return (
    <div className="grid gap-3 rounded-xl border border-border bg-card p-4">
      <div className="flex flex-wrap items-start gap-x-3 gap-y-1">
        <div className="min-w-0 flex-1">
          <div className="text-[13.5px]">{tr.question ?? "(no question recorded)"}</div>
          <div className="mt-0.5 flex flex-wrap gap-x-3 font-mono text-[11px] text-muted-foreground"><span>{when(tr.start)}</span><span>{SOURCE_LABEL[tr.source] ?? tr.source}</span>{tr.config ? <span>{tr.config}</span> : null}{tr.conversation_id ? <span>{tr.conversation_id}</span> : null}{tr.eval_run ? <span>{tr.eval_run} · q{tr.query_id}</span> : null}{tr.confidence ? <span>confidence {tr.confidence}</span> : null}</div>
        </div>
        {tr.url ? <a href={tr.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 rounded-full border border-border px-2.5 py-1 text-[11px] text-muted-foreground hover:border-foreground hover:text-foreground">open in Langfuse <ExternalLink className="size-3" /></a> : null}
      </div>
      {!t ? <MessageLoading /> : null}
      <Waterfall t={tr} />
      {Object.keys(tr.scores).length ? <div className="flex flex-wrap gap-1.5 text-[11px]">{Object.entries(tr.scores).map(([k, v]) => <span key={k} className="rounded border border-border px-1.5 py-0.5 font-mono">{k} {v.toFixed(2)}</span>)}</div> : null}
      {tr.error ? <div className="rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 font-mono text-[11.5px] text-destructive">{tr.error}</div> : null}
      {out?.answer ? <div className="rounded-lg border border-border bg-secondary/40 p-3 text-[12.5px] leading-relaxed whitespace-pre-wrap">{out.answer.slice(0, 1200)}{out.answer.length > 1200 ? "…" : ""}</div> : null}
      {answer && typeof answer.input === "object" && answer.input ? <div className="text-[11px] text-muted-foreground">pages given to the model: {((answer.input as { pages?: string[] }).pages ?? []).join(" · ")}</div> : null}
    </div>
  );
}

export default function ObservabilityPage() {
  const [window_, setWindow] = useState("7d");
  const [o, setO] = useState<Overview | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [source, setSource] = useState("all");
  useEffect(() => { let on = true; setO(null); j<Overview>(`/api/observability?window=${window_}`).then((r) => { if (on) { setO(r); setSel((s) => s ?? r.traces[0]?.id ?? null); } }).catch(() => { if (on) setO({ source: "none", window: window_, since: "", error: "backend unreachable", summary: null, traces: [], total: 0 }); }); return () => { on = false; }; }, [window_]);
  const sources = useMemo(() => Object.keys(o?.summary?.by_source ?? {}), [o]);
  const traces = (o?.traces ?? []).filter((t) => source === "all" || t.source === source);
  const selected = o?.traces.find((t) => t.id === sel) ?? null;
  const a = o?.summary?.all;
  return (
    <AppShell title="Observability" subtitle={o ? `${o.total} traces · ${o.source === "langfuse" ? "Langfuse" : o.source === "supabase" ? "stored turns (Langfuse off)" : "no data"} · last ${WINDOWS.find(([k]) => k === o.window)?.[1]}` : "loading"}
      actions={<div className="inline-flex rounded-lg border border-border bg-card p-0.5 text-xs">{WINDOWS.map(([k, l]) => <button key={k} type="button" onClick={() => { setWindow(k); setSel(null); }} className={cn("rounded-md px-2.5 py-1", window_ === k ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground")}>{l}</button>)}</div>}>
      <div className="grid gap-6 p-4 sm:p-6">
        {!o ? <MessageLoading /> : (
          <>
            {o.error ? <div className="rounded-lg border border-dashed border-border px-3 py-2 text-xs text-muted-foreground">{o.error}</div> : null}
            <section className="grid gap-2">
              <h2 className="text-sm font-medium">Runtime: the container behind the query API</h2>
              <Runtime />
            </section>
            {a ? (
              <section className="grid gap-2">
                <h2 className="text-sm font-medium">Where the time and the money go</h2>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
                  <Tile label="Turns" value={String(a.n)} sub={`${a.errors} errors`} />
                  <Tile label="Cost" value={money(a.cost_usd)} sub={a.n ? `${money(a.cost_usd / a.n)} per turn` : undefined} />
                  <Tile label="Tokens" value={`${(a.tokens_in / 1000).toFixed(1)}k`} sub={`${(a.tokens_out / 1000).toFixed(1)}k out`} />
                  <Tile label="TTR p50" value={s1(a.latency_ms.p50)} sub={`p95 ${s1(a.latency_ms.p95)}`} />
                  <Tile label="Retrieval p50" value={`${a.retrieval_ms.p50} ms`} sub={`p95 ${a.retrieval_ms.p95} ms`} />
                  <Tile label="Answer p50" value={s1(a.answer_ms.p50)} sub={`p95 ${s1(a.answer_ms.p95)} · ${a.latency_ms.p50 ? Math.round((a.answer_ms.p50 / a.latency_ms.p50) * 100) : 0}% of TTR`} />
                </div>
              </section>
            ) : null}
            {o.summary ? (
              <section className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                <div className="grid gap-2">
                  <h2 className="text-sm font-medium">Turns per day</h2>
                  <DailyStrip days={o.summary.by_day} />
                </div>
                <div className="grid gap-2">
                  <h2 className="text-sm font-medium">Per stage, across the window</h2>
                  <div className="overflow-x-auto rounded-xl border border-border bg-card">
                    <table className="w-full text-[12.5px]">
                      <thead className="border-b border-border"><tr><Th>Stage</Th><Th right>runs</Th><Th right>p50</Th><Th right>p95</Th></tr></thead>
                      <tbody>{Object.entries(o.summary.stages).sort((x, y) => y[1].p50 - x[1].p50).map(([k, v]) => (
                        <tr key={k} className="border-b border-border last:border-b-0"><Td><span className={cn("mr-1.5 inline-block size-2 rounded-sm align-middle", STAGE_TONE[k] ?? "bg-muted-foreground/60")} aria-hidden="true" />{k}</Td><Td right mono>{v.n}</Td><Td right mono>{v.p50 >= 1000 ? s1(v.p50) : `${v.p50} ms`}</Td><Td right mono>{v.p95 >= 1000 ? s1(v.p95) : `${v.p95} ms`}</Td></tr>
                      ))}</tbody>
                    </table>
                  </div>
                </div>
              </section>
            ) : null}
            {o.summary ? (
              <section className="grid gap-2">
                <h2 className="text-sm font-medium">By source and by config</h2>
                <div className="grid gap-3 lg:grid-cols-2">
                  {[["source", o.summary.by_source, SOURCE_LABEL], ["config", o.summary.by_config, {}]].map(([title, groups, labels]) => (
                    <div key={title as string} className="overflow-x-auto rounded-xl border border-border bg-card">
                      <table className="w-full text-[12.5px]">
                        <thead className="border-b border-border"><tr><Th>{title as string}</Th><Th right>turns</Th><Th right>cost</Th><Th right>TTR p50</Th><Th right>retrieval p50</Th><Th right>errors</Th></tr></thead>
                        <tbody>{Object.entries(groups as Record<string, Block>).map(([k, b]) => (
                          <tr key={k} className="border-b border-border last:border-b-0"><Td className="font-medium">{(labels as Record<string, string>)[k] ?? k}</Td><Td right mono>{b.n}</Td><Td right mono>{money(b.cost_usd)}</Td><Td right mono>{s1(b.latency_ms.p50)}</Td><Td right mono>{b.retrieval_ms.p50} ms</Td><Td right mono className={b.errors ? "text-destructive" : ""}>{b.errors}</Td></tr>
                        ))}</tbody>
                      </table>
                    </div>
                  ))}
                </div>
              </section>
            ) : null}
            <section className="grid gap-2">
              <div className="flex flex-wrap items-center gap-1.5">
                <h2 className="mr-2 text-sm font-medium">Traces</h2>
                {["all", ...sources].map((k) => <button key={k} type="button" onClick={() => setSource(k)} className={cn("rounded-full border border-border px-3 py-1 text-xs hover:border-foreground", source === k && "border-foreground bg-secondary")}>{k === "all" ? "all" : SOURCE_LABEL[k] ?? k}</button>)}
              </div>
              <div className="grid gap-3 xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
                <div className="max-h-[36rem] overflow-auto rounded-xl border border-border bg-card">
                  <table className="w-full text-[12.5px]">
                    <thead className="sticky top-0 border-b border-border bg-card"><tr><Th>when</Th><Th>question</Th><Th>source · config</Th><Th right>TTR</Th><Th right>retrieval</Th><Th right>cost</Th></tr></thead>
                    <tbody>
                      {traces.map((t) => (
                        <tr key={t.id} onClick={() => setSel(t.id)} className={cn("cursor-pointer border-b border-border last:border-b-0 hover:bg-secondary/50", sel === t.id && "bg-secondary/70")}>
                          <Td mono className="whitespace-nowrap text-muted-foreground">{when(t.start)}</Td>
                          <Td className="max-w-[28ch] truncate">{t.error ? <span className="mr-1 text-destructive">!</span> : null}{t.question ?? <span className="text-muted-foreground">(no question)</span>}</Td>
                          <Td className="whitespace-nowrap text-muted-foreground">{SOURCE_LABEL[t.source] ?? t.source}{t.config ? ` · ${t.config}` : ""}</Td>
                          <Td right mono>{s1(t.latency_ms)}</Td><Td right mono>{t.retrieval_ms} ms</Td><Td right mono>{money(t.cost_usd)}</Td>
                        </tr>
                      ))}
                      {!traces.length ? <tr><Td className="text-center text-muted-foreground">no traces</Td></tr> : null}
                    </tbody>
                  </table>
                </div>
                {selected ? <TraceDetail id={selected.id} fallback={selected} /> : <div className="rounded-xl border border-dashed border-border p-6 text-center text-xs text-muted-foreground">select a trace to see its waterfall</div>}
              </div>
            </section>
          </>
        )}
      </div>
    </AppShell>
  );
}
