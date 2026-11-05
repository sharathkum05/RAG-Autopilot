"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { AppShell } from "@/components/app-shell";
import { MessageLoading } from "@/components/ui/message-loading";
import { cn } from "@/lib/utils";

/**
 * Answer Evals: the baseline and the improvement answer the same seeded sample of benchmark queries end to end
 * (retrieve → Claude), and two independent eval pipelines grade every answer: the deterministic checks in
 * pipeline/checks.py and the frozen four-dimension LLM judge in pipeline/judge.py. One comparison table, then a tab
 * per pipeline with its rubric, its cost, and every answer next to its reference.
 */
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";
const j = async <T,>(url: string): Promise<T> => { const r = await fetch(`${BASE}${url}`); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); };

interface Lat { mean: number; p50: number; p95: number }
interface Group { n: number; pass_rate: number; mean_score: number }
interface CheckStat { applicable: number; mean: number | null; pass_rate: number }
interface JudgeSummary { n: number; mean_correctness: number | null; agreement: number | null; model: string | null; dims?: Record<string, number> | null; mean_overall?: number | null; cost_usd?: number; tokens_in?: number; tokens_out?: number; ms_mean?: number | null }
interface Summary { n_queries: number; pass_rate: number; mean_score: number; refusals: number; errors: number; checks: Record<string, CheckStat>; by_query_type: Record<string, Group>; by_kind: Record<string, Group>; by_format: Record<string, Group>; judge: JudgeSummary | null; cost_usd: number; tokens_in: number; tokens_out: number; latency_ms: Record<string, Lat>; wall_s: number; sample?: { n: number; seed: number; query_ids: number[] } }
interface Rubric { weights: Record<string, number>; threshold: number; descriptions: Record<string, string> }
interface RunRow { id: string; config: string; answer_model: string; judge_model: string | null; n_pages: number; rubric: Rubric; summary: Summary; hardware: string; created_at: string }
interface Judge { model?: string; correctness?: number; scores?: Record<string, number>; overall?: number; reason?: string; error?: string; tokens_in?: number; tokens_out?: number; ms?: number }
interface PerQuery { query_id: number; query: string; query_types: string[]; query_format: string; content_type: string[]; kind: string; reference: string; references: string[]; relevant: [number, number][]; retrieved: number[]; shown: number[]; cited: number[]; answer: string; error: string | null; checks: Record<string, number | null>; score: number; passed: boolean; refused: boolean; detail: { anchors?: string[]; found?: string[]; missing?: string[]; polarity?: { reference: string | null; answer: string | null }; numbers?: { reference: string[]; found: string[] }; labelled_cited?: number[] }; judge: Judge | null; ms: { retrieval: number; answer: number; total: number }; cost_usd: number; trace_id: string | null }
interface FullRun extends RunRow { per_query: PerQuery[] }
interface JudgeRubric { model: string; scale: string; sees: string; dimensions: Record<string, string>; frozen: string; price_per_1m_tokens: { input: number; output: number } }
interface Overview { configs: string[]; latest: RunRow[]; runs: RunRow[]; baseline: string; improvement: string; info?: Record<string, { label: string; detail: string }>; judge_rubric?: JudgeRubric; error?: string | null }

type Role = "baseline" | "improvement";
type Tab = "checks" | "judge";
const DIMS = ["relevance", "groundedness", "correctness", "coherence"] as const;
const CHECK_ORDER = ["grounded", "valid_citations", "anchors", "content", "type", "rouge_l"];
const CHECK_LABEL: Record<string, string> = { grounded: "Grounded citation", valid_citations: "Valid citations", anchors: "Anchor coverage", content: "Content words", type: "Type check", rouge_l: "ROUGE-L (info)" };
const pctf = (v: number) => `${(v * 100).toFixed(1)}%`;
const f2 = (v: number) => v.toFixed(2);
const f3 = (v: number) => v.toFixed(3);
const secs = (ms: number) => `${(ms / 1000).toFixed(1)} s`;
const when = (iso: string) => iso.replace("T", " ").slice(0, 16);
const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return <th className={cn("px-3 py-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground", right ? "text-right" : "text-left")}>{children}</th>;
}
function Td({ children, right, mono, className }: { children: React.ReactNode; right?: boolean; mono?: boolean; className?: string }) {
  return <td className={cn("px-3 py-2 align-top", right && "text-right", mono && "font-mono text-[12.5px]", className)}>{children}</td>;
}
function RoleBadge({ role }: { role: Role }) {
  return <Badge variant="outline" className={cn("font-mono text-[10px] uppercase", role === "improvement" && "border-success/50 text-success")}>{role}</Badge>;
}
function Toggle<T extends string>({ value, onChange, opts }: { value: T; onChange: (v: T) => void; opts: [T, string][] }) {
  return (
    <div className="inline-flex rounded-lg border border-border bg-card p-0.5 text-xs">
      {opts.map(([k, l]) => <button key={k} type="button" onClick={() => onChange(k)} className={cn("rounded-md px-2.5 py-1 transition-colors", value === k ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground")}>{l}</button>)}
    </div>
  );
}

// ------------------------------------------------------------------ baseline → improvement cards

function DeltaCard({ label, b, i, fmt, better }: { label: string; b: number | null | undefined; i: number | null | undefined; fmt: (v: number) => string; better?: "up" | "down" }) {
  const ok = b != null && i != null;
  const same = ok && fmt(b as number) === fmt(i as number);          // equal at display precision
  const d = ok ? (i as number) - (b as number) : 0;
  const good = better && ok && !same ? (better === "up" ? d > 0 : d < 0) : null;
  return (
    <div className="rounded-lg border border-border bg-card px-3 py-2.5">
      <div className="text-[10.5px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 flex items-baseline gap-2 font-mono text-[13px]">
        <span className="text-muted-foreground">{b != null ? fmt(b) : "–"}</span><span className="text-muted-foreground">→</span>
        <span className="text-[15px] font-medium">{i != null ? fmt(i) : "–"}</span>
        {ok && !same ? <span className={cn("text-[11px]", good == null ? "text-muted-foreground" : good ? "text-success" : "text-destructive")}>{d > 0 ? "+" : ""}{fmt === pctf ? `${(d * 100).toFixed(1)} pts` : fmt(d)}</span> : null}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ pipeline 1: deterministic checks

function ChecksTable({ base, impr, labels }: { base: RunRow; impr: RunRow; labels: [string, string] }) {
  const w = base.rubric.weights, d = base.rubric.descriptions;
  const cell = (s: Summary, c: string) => { const st = s.checks[c]; return st ? `${st.mean != null ? st.mean.toFixed(2) : "–"} · full ${pctf(st.pass_rate)}` : "–"; };
  return (
    <div className="overflow-x-auto rounded-xl border border-border bg-card">
      <table className="w-full text-[13px]">
        <thead className="border-b border-border"><tr><Th>check</Th><Th right>weight</Th><Th>what it tests</Th><Th right>{labels[0]}</Th><Th right>{labels[1]}</Th></tr></thead>
        <tbody>
          {CHECK_ORDER.filter((c) => base.summary.checks[c] || impr.summary.checks[c]).map((c) => (
            <tr key={c} className="border-b border-border">
              <Td className="font-medium">{CHECK_LABEL[c] ?? c}</Td><Td right mono>{w[c] != null ? w[c] : "info"}</Td>
              <Td className="text-[12px] text-muted-foreground">{d[c]}</Td>
              <Td right mono className="text-muted-foreground">{cell(base.summary, c)}</Td><Td right mono>{cell(impr.summary, c)}</Td>
            </tr>
          ))}
          <tr><Td className="font-medium">Refusal gate</Td><Td right mono>gate</Td><Td className="text-[12px] text-muted-foreground">{d.refusal}</Td><Td right mono className="text-muted-foreground">{base.summary.refusals} declines</Td><Td right mono>{impr.summary.refusals} declines</Td></tr>
        </tbody>
      </table>
      <div className="border-t border-border px-3 py-2 text-[11px] text-muted-foreground">Each check is a pure function of the answer, its citations, the pages shown to the model, the reference answers and the page labels. No model is involved. Query score = weighted mean over the checks that apply; pass at {base.rubric.threshold}. Cells show mean · share of queries at full marks.</div>
    </div>
  );
}

function Groups({ title, base, impr, labels, names }: { title: string; base: Record<string, Group>; impr: Record<string, Group>; labels: [string, string]; names?: Record<string, string> }) {
  const keys = Array.from(new Set([...Object.keys(base), ...Object.keys(impr)])).sort((a, b) => (impr[b]?.n ?? base[b]?.n ?? 0) - (impr[a]?.n ?? base[a]?.n ?? 0));
  return (
    <div className="overflow-x-auto rounded-xl border border-border bg-card">
      <table className="w-full text-[13px]">
        <thead className="border-b border-border"><tr><Th>{title}</Th><Th right>queries</Th><Th right>{labels[0]} pass</Th><Th right>{labels[1]} pass</Th></tr></thead>
        <tbody>
          {keys.map((k) => { const b = base[k], i = impr[k]; const d = b && i ? i.pass_rate - b.pass_rate : 0; return (
            <tr key={k} className="border-b border-border last:border-b-0">
              <Td className="font-medium">{names?.[k] ?? k}</Td><Td right mono>{i?.n ?? b?.n}</Td>
              <Td right mono className="text-muted-foreground">{b ? pctf(b.pass_rate) : "–"}</Td>
              <Td right mono className={cn(d > 1e-9 ? "text-success" : d < -1e-9 ? "text-destructive" : "")}>{i ? pctf(i.pass_rate) : "–"}</Td>
            </tr>); })}
        </tbody>
      </table>
    </div>
  );
}

/** The answer with the reference's anchors marked: found in green, missing listed in red. */
function Marked({ text, found }: { text: string; found: string[] }) {
  const keys = found.map((k) => k.replace(/\($/, "")).filter(Boolean).sort((a, b) => b.length - a.length);
  if (!keys.length) return <>{text}</>;
  const re = new RegExp(`(${keys.map((k) => (/^[a-z0-9]+$/i.test(k) ? `(?<![a-z0-9])${esc(k)}(?![a-z0-9])` : esc(k))).join("|")})`, "gi");
  return <>{text.split(re).map((part, i) => (i % 2 === 1 ? <mark key={i} className="rounded-sm bg-success/25 px-0.5 text-foreground">{part}</mark> : part))}</>;
}

function CheckChip({ name, v }: { name: string; v: number | null }) {
  if (v == null) return null;
  const tone = v >= 0.999 ? "border-success/40 text-success" : v <= 0.001 ? "border-destructive/40 text-destructive" : "border-warning/50 text-warning";
  return <span className={cn("rounded border px-1.5 py-0.5 font-mono text-[10.5px]", tone)}>{(CHECK_LABEL[name] ?? name).replace(" (info)", "")} {name === "grounded" || name === "valid_citations" ? (v >= 0.999 ? "✓" : "✗") : v.toFixed(2)}</span>;
}

function ScoreChip({ name, v }: { name: string; v: number }) {
  return <span className={cn("rounded border px-1.5 py-0.5 font-mono text-[10.5px]", v >= 4 ? "border-success/40 text-success" : v <= 2 ? "border-destructive/40 text-destructive" : "border-warning/50 text-warning")}>{name} {v}</span>;
}

function AnswerVsReference({ q }: { q: PerQuery }) {
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <div className="rounded-lg border border-border bg-secondary/40 p-3">
        <div className="mb-1 text-[11px] uppercase tracking-wide text-muted-foreground">Answer · {q.cited.length ? `cited ${q.cited.map((p) => `p.${p}`).join(" ")}` : "no citation"}</div>
        <div className="whitespace-pre-wrap leading-relaxed"><Marked text={q.answer || q.error || "(empty)"} found={q.detail.found ?? []} /></div>
      </div>
      <div className="rounded-lg border border-border bg-secondary/40 p-3">
        <div className="mb-1 text-[11px] uppercase tracking-wide text-muted-foreground">Reference · labelled {q.relevant.map(([p]) => `p.${p}`).join(" ")}</div>
        <div className="leading-relaxed">{q.reference}</div>
        {q.references.length ? <div className="mt-2 text-xs text-muted-foreground">{q.references.length} more reference variant{q.references.length > 1 ? "s" : ""} accepted</div> : null}
      </div>
    </div>
  );
}

function QueryRow({ q, tab }: { q: PerQuery; tab: Tab }) {
  const [open, setOpen] = useState(false);
  const rel = new Set(q.relevant.map(([p]) => p));
  const jd = q.judge;
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-lg border border-border bg-card">
      <CollapsibleTrigger className="group grid w-full grid-cols-[14px_minmax(0,1fr)_auto] items-center gap-3 px-3 py-2.5 text-left">
        <ChevronRight className="size-3.5 text-muted-foreground transition-transform group-data-[panel-open]:rotate-90" />
        <div className="min-w-0">
          <div className="truncate text-[13.5px]">{q.query}</div>
          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            {tab === "checks" ? (
              <>
                <span className={cn("rounded-full px-2 py-0.5 font-mono text-[10.5px]", q.passed ? "bg-success/15 text-success" : "bg-destructive/15 text-destructive")}>{q.refused ? "refused" : q.passed ? "pass" : "fail"} {q.score.toFixed(2)}</span>
                {CHECK_ORDER.filter((c) => c !== "rouge_l").map((c) => <CheckChip key={c} name={c} v={q.checks[c] ?? null} />)}
              </>
            ) : jd && jd.overall != null ? (
              <>
                <span className={cn("rounded-full px-2 py-0.5 font-mono text-[10.5px]", jd.overall >= 4 ? "bg-success/15 text-success" : jd.overall <= 2 ? "bg-destructive/15 text-destructive" : "bg-warning/15 text-warning")}>overall {jd.overall.toFixed(2)} / 5</span>
                {DIMS.map((d) => (jd.scores?.[d] != null ? <ScoreChip key={d} name={d} v={jd.scores[d]} /> : null))}
              </>
            ) : <span className="font-mono text-[10.5px] text-muted-foreground">{jd?.error ? "judge error" : "not judged"}</span>}
            <span className="text-[11px] text-muted-foreground">{q.query_types.join(", ")} · {q.kind}</span>
          </div>
        </div>
        <span className="font-mono text-[11px] text-muted-foreground">{secs(q.ms.total)}</span>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="grid gap-3 border-t border-border px-3 pb-3 pt-3 text-[13px]">
          <AnswerVsReference q={q} />
          {tab === "checks" ? (
            <div className="flex flex-wrap items-center gap-1.5 text-[11.5px]">
              <span className="text-muted-foreground">anchors</span>
              {(q.detail.found ?? []).map((a) => <span key={`f${a}`} className="rounded border border-success/40 px-1.5 py-0.5 font-mono text-success">{a}</span>)}
              {(q.detail.missing ?? []).map((a) => <span key={`m${a}`} className="rounded border border-destructive/40 px-1.5 py-0.5 font-mono text-destructive line-through">{a}</span>)}
              {!(q.detail.anchors ?? []).length ? <span className="text-muted-foreground">none in the reference, content words used instead</span> : null}
              {q.detail.polarity ? <span className="ml-2 text-muted-foreground">polarity ref {q.detail.polarity.reference ?? "?"} / answer {q.detail.polarity.answer ?? "?"}</span> : null}
              {q.detail.numbers ? <span className="ml-2 text-muted-foreground">numbers {q.detail.numbers.found.length}/{q.detail.numbers.reference.length}</span> : null}
            </div>
          ) : (
            <div className="grid gap-1 text-[12px]">
              {jd?.reason ? <div><span className="text-muted-foreground">judge: </span>{jd.reason}</div> : null}
              {jd?.error ? <div className="text-destructive">judge error: {jd.error}</div> : null}
              {jd ? <div className="font-mono text-[11px] text-muted-foreground">{jd.model} · {jd.tokens_in ?? 0} in / {jd.tokens_out ?? 0} out tokens · {jd.ms ?? 0} ms · checks said {q.passed ? "pass" : "fail"} {q.score.toFixed(2)}</div> : null}
            </div>
          )}
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11.5px] text-muted-foreground">
            <span>shown to the model {q.shown.map((p) => <span key={p} className={cn("font-mono", rel.has(p) && "text-success")}>p.{p} </span>)}</span>
            <span>retrieved {q.retrieved.slice(0, 5).map((p) => `p.${p}`).join(" ")}</span>
            <span>retrieval {q.ms.retrieval} ms · answer {q.ms.answer} ms · ${q.cost_usd.toFixed(3)}</span>
          </div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

function Queries({ run, tab }: { run: FullRun; tab: Tab }) {
  const [flagged, setFlagged] = useState(true);
  const [type, setType] = useState<string>("all");
  const types = useMemo(() => Array.from(new Set(run.per_query.flatMap((q) => q.query_types))).sort(), [run]);
  const isFlag = (q: PerQuery) => (tab === "checks" ? !q.passed : (q.judge?.overall ?? 5) < 4);
  const rows = run.per_query.filter((q) => (!flagged || isFlag(q)) && (type === "all" || q.query_types.includes(type)));
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-1.5">
        {["all", ...types].map((t) => <button key={t} type="button" onClick={() => setType(t)} className={cn("rounded-full border border-border px-3 py-1 text-xs hover:border-foreground", type === t && "border-foreground bg-secondary")}>{t}</button>)}
        <label className="ml-auto inline-flex items-center gap-2 text-xs text-muted-foreground"><input type="checkbox" checked={flagged} onChange={(e) => setFlagged(e.target.checked)} className="accent-foreground" />{tab === "checks" ? "failures only" : "below 4 / 5 only"}</label>
      </div>
      <div className="text-xs text-muted-foreground">{rows.length} of {run.per_query.length} queries. Expand a row for the answer next to the reference{tab === "checks" ? ", anchors it hit in green, missed struck through" : " and the judge's one-line reason"}.</div>
      <div className="grid gap-1.5">{rows.map((q) => <QueryRow key={q.query_id} q={q} tab={tab} />)}</div>
    </div>
  );
}

// ------------------------------------------------------------------ pipeline 2: LLM as judge

function JudgeRubricCard({ rubric, base, impr }: { rubric?: JudgeRubric; base: RunRow; impr: RunRow }) {
  const jb = base.summary.judge, ji = impr.summary.judge;
  const model = ji?.model ?? jb?.model ?? rubric?.model ?? "judge";
  return (
    <div className="grid gap-3 rounded-xl border border-border bg-card p-3">
      <div className="grid gap-1 text-[12.5px]">
        <div><span className="font-medium">Model </span><span className="font-mono">{model}</span><span className="text-muted-foreground"> · {rubric?.frozen ?? "fixed model, fixed prompt, JSON output"}</span></div>
        <div><span className="font-medium">Sees </span><span className="text-muted-foreground">{rubric?.sees ?? "the question, the reference answer, the page excerpts shown to the model, the answer"}</span></div>
        <div><span className="font-medium">Scale </span><span className="text-muted-foreground">{rubric?.scale ?? "integer 1 to 5 per dimension; overall = mean of the four"}</span></div>
      </div>
      <div className="grid gap-1.5 border-t border-border pt-2.5">
        {DIMS.map((d) => (
          <div key={d} className="grid grid-cols-[110px_minmax(0,1fr)] gap-3 text-[12.5px]"><div className="font-medium capitalize">{d}</div><div className="text-muted-foreground">{rubric?.dimensions?.[d] ?? ""}</div></div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-2 text-[12px] sm:grid-cols-4">
        {[["Judged answers", `${jb?.n ?? 0} · ${ji?.n ?? 0}`], ["Judge cost, both runs", `$${((jb?.cost_usd ?? 0) + (ji?.cost_usd ?? 0)).toFixed(3)}`], ["Per judgement", `${jb?.ms_mean ?? ji?.ms_mean ?? 0} ms${rubric ? ` · $${rubric.price_per_1m_tokens.input} / $${rubric.price_per_1m_tokens.output} per 1M tokens` : ""}`], ["Agrees with the checks", `${jb?.agreement != null ? pctf(jb.agreement) : "–"} · ${ji?.agreement != null ? pctf(ji.agreement) : "–"}`]].map(([l, v]) => (
          <div key={l} className="rounded-lg border border-border px-3 py-2"><div className="text-[10.5px] uppercase tracking-wide text-muted-foreground">{l}</div><div className="mt-0.5 font-mono text-[12.5px]">{v}</div></div>
        ))}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ page

export default function AnswerEvalsPage() {
  const [o, setO] = useState<Overview | null>(null);
  const [full, setFull] = useState<Record<string, FullRun>>({});
  const [tab, setTab] = useState<Tab>("checks");
  const [which, setWhich] = useState<Role>("improvement");
  useEffect(() => { let on = true; j<Overview>("/api/answer-evals").then((r) => { if (on) setO(r); }).catch(() => setO({ configs: [], latest: [], runs: [], baseline: "text_hybrid", improvement: "text_hybrid_rerank" })); return () => { on = false; }; }, []);
  const base = o?.latest.find((r) => r.config === o.baseline), impr = o?.latest.find((r) => r.config === o.improvement);
  useEffect(() => {
    let on = true;
    for (const r of [base, impr]) if (r && !full[r.id]) j<FullRun>(`/api/answer-evals/run?id=${encodeURIComponent(r.id)}`).then((x) => { if (on) setFull((f) => ({ ...f, [x.id]: x })); }).catch(() => {});
    return () => { on = false; };
  }, [base?.id, impr?.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const label = (c?: string) => (c && o?.info?.[c]?.label) || c || "";
  const labels: [string, string] = [label(o?.baseline), label(o?.improvement)];
  const ref = base ?? impr;
  const sample = ref?.summary.sample;
  const sel = which === "baseline" ? base : impr;
  const selFull = sel ? full[sel.id] : undefined;
  const B = base?.summary, I = impr?.summary;
  const dim = (s: Summary | undefined, d: string) => s?.judge?.dims?.[d];
  return (
    <AppShell title="Evals Pipeline" subtitle={ref ? `${sample?.n ?? ref.summary.n_queries} benchmark queries answered end to end · two eval pipelines: deterministic checks and ${ref.judge_model ?? "LLM"} judge` : o ? "no runs yet" : "loading"}>
      <div className="grid gap-6 p-4 sm:p-6">
        {!o ? <MessageLoading /> : !base || !impr || !B || !I ? (
          <div className="grid gap-2 rounded-lg border border-dashed border-border p-4 text-sm text-muted-foreground">
            <div>Need one answer-eval run for each of <code className="font-mono">{o.baseline}</code> and <code className="font-mono">{o.improvement}</code>. Run <code className="font-mono">modal run modal_app.py::answer_evals --config {o.improvement} --sample 30</code> and the same for the baseline.</div>
            {o.error ? <div className="font-mono text-xs text-destructive">{o.error}</div> : null}
          </div>
        ) : (
          <>
            <section className="grid gap-2">
              <h2 className="text-sm font-medium">Two eval pipelines over the same answers</h2>
              <p className="max-w-[90ch] text-xs text-muted-foreground">
                <RoleBadge role="baseline" /> {labels[0]} and <RoleBadge role="improvement" /> {labels[1]} answered the same {sample ? `seeded sample of ${sample.n} benchmark queries (seed ${sample.seed})` : `${ref?.summary.n_queries} benchmark queries`} exactly as the app does: retrieve, read the top {ref?.n_pages} pages, answer with {ref?.answer_model}, cite. Every answer is then graded twice, independently. {impr.hardware} · {when(impr.created_at)}.
              </p>
              <div className="grid gap-3 lg:grid-cols-2">
                <div className="rounded-xl border border-border bg-card p-3 text-[12.5px]">
                  <div className="flex items-center gap-2"><Badge variant="outline" className="font-mono text-[10px]">1</Badge><span className="font-medium">Deterministic checks</span><span className="text-muted-foreground">pipeline/checks.py</span></div>
                  <p className="mt-1.5 text-muted-foreground">Six pure functions of the answer, its citations, the pages shown, the reference answers and the page labels. Weighted into a 0 to 1 score, pass at {base.rubric.threshold}. No model, no randomness: the same answer always scores the same.</p>
                </div>
                <div className="rounded-xl border border-border bg-card p-3 text-[12.5px]">
                  <div className="flex items-center gap-2"><Badge variant="outline" className="font-mono text-[10px]">2</Badge><span className="font-medium">LLM as judge</span><span className="text-muted-foreground">pipeline/judge.py · {o.judge_rubric?.model ?? impr.judge_model}</span></div>
                  <p className="mt-1.5 text-muted-foreground">A frozen model reads the question, the reference, the page excerpts the answering model saw, and the answer. It scores relevance, groundedness, correctness and coherence from 1 to 5; overall is their mean.</p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <DeltaCard label="Time to answer p50" b={B.latency_ms.total.p50} i={I.latency_ms.total.p50} fmt={secs} better="down" />
                <DeltaCard label="Retrieval p50" b={B.latency_ms.retrieval?.p50} i={I.latency_ms.retrieval?.p50} fmt={(v) => `${Math.round(v)} ms`} better="down" />
                <DeltaCard label="Answer cost per query" b={B.cost_usd / Math.max(1, B.n_queries)} i={I.cost_usd / Math.max(1, I.n_queries)} fmt={(v) => `$${v.toFixed(3)}`} better="down" />
                <DeltaCard label="Judge cost per run" b={B.judge?.cost_usd} i={I.judge?.cost_usd} fmt={(v) => `$${v.toFixed(3)}`} better="down" />
              </div>
            </section>
            <section className="grid gap-3">
              <Toggle<Tab> value={tab} onChange={setTab} opts={[["checks", "Deterministic checks"], ["judge", "LLM as judge"]]} />
              {tab === "checks" ? (
                <>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    <DeltaCard label="Pass rate" b={B.pass_rate} i={I.pass_rate} fmt={pctf} better="up" />
                    <DeltaCard label="Mean score" b={B.mean_score} i={I.mean_score} fmt={f3} better="up" />
                    <DeltaCard label="Grounded" b={B.checks.grounded?.pass_rate} i={I.checks.grounded?.pass_rate} fmt={pctf} better="up" />
                    <DeltaCard label="Anchor coverage" b={B.checks.anchors?.mean} i={I.checks.anchors?.mean} fmt={f3} better="up" />
                  </div>
                  <ChecksTable base={base} impr={impr} labels={labels} />
                  <div className="grid gap-3 lg:grid-cols-3">
                    <Groups title="Question type" base={B.by_query_type} impr={I.by_query_type} labels={labels} />
                    <Groups title="Question needs" base={B.by_kind} impr={I.by_kind} labels={labels} names={{ table: "a table", figure: "a figure or infographic", text: "running text" }} />
                    <Groups title="Query format" base={B.by_format} impr={I.by_format} labels={labels} />
                  </div>
                </>
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
                    <DeltaCard label="Overall / 5" b={B.judge?.mean_overall} i={I.judge?.mean_overall} fmt={f2} better="up" />
                    {DIMS.map((d) => <DeltaCard key={d} label={d} b={dim(B, d)} i={dim(I, d)} fmt={f2} better="up" />)}
                  </div>
                  <JudgeRubricCard rubric={o.judge_rubric} base={base} impr={impr} />
                </>
              )}
              <div className="flex flex-wrap items-center gap-3">
                <h2 className="text-sm font-medium">Every answer against its reference</h2>
                <Toggle<Role> value={which} onChange={setWhich} opts={[["baseline", labels[0]], ["improvement", labels[1]]]} />
              </div>
              {selFull ? <Queries key={`${selFull.id}-${tab}`} run={selFull} tab={tab} /> : <MessageLoading />}
            </section>
          </>
        )}
      </div>
    </AppShell>
  );
}
