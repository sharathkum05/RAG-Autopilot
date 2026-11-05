"use client";

import { useCallback, useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { AppShell } from "@/components/app-shell";
import { cn } from "@/lib/utils";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";
const j = async <T,>(url: string, init?: RequestInit): Promise<T> => { const r = await fetch(`${BASE}${url}`, init); const body = await r.json(); if (!r.ok) throw new Error(body.error ?? `${url}: ${r.status}`); return body; };

interface Secrets { writable?: boolean; services: { id: string; name: string; purpose: string; stored: string; editable?: boolean; keys: { name: string; set: boolean; masked: string | null }[] }[] }
interface Config { pipeline: { stage: string; model: string; detail: string; runs_on: string }[]; stores: { name: string; role: string; detail: string; location: string }[]; retrieval: { modes: string[]; default: string }; answer_models: string[] }
interface System { source: string; sampled_at: string; container: { id: string; gpu: string; region: string; uptime_s: number; cold_starts_today: number; state: string; scale_to_zero_after_s: number }; cpu: { percent: number; cores: number }; memory: { used_gb: number; total_gb: number }; gpu: { util_percent: number; memory_used_gb: number; memory_total_gb: number; temperature_c: number }; models_loaded: { name: string; memory_gb: number }[]; requests: { last_hour: number; p50_ms: number; p95_ms: number; errors: number } }

function Section({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <section className="grid gap-3">
      <div><h2 className="text-sm font-semibold">{title}</h2>{sub ? <p className="text-xs text-muted-foreground">{sub}</p> : null}</div>
      {children}
    </section>
  );
}

function Meter({ label, value, max, unit, hist }: { label: string; value: number; max: number; unit: string; hist: number[] }) {
  const pct = Math.min(100, (value / max) * 100);
  const w = 72, h = 20, mx = Math.max(max, ...hist, 1);
  const d = hist.map((v, i) => `${i ? "L" : "M"}${((i / Math.max(hist.length - 1, 1)) * w).toFixed(1)} ${(h - (v / mx) * (h - 2) - 1).toFixed(1)}`).join(" ");
  return (
    <div className="grid gap-1.5 rounded-lg border border-border bg-card p-3">
      <div className="flex items-baseline justify-between gap-2 text-xs text-muted-foreground"><span>{label}</span><span className="font-mono text-foreground">{value}{unit}<span className="text-muted-foreground"> / {max}{unit}</span></span></div>
      <div className="h-1.5 overflow-hidden rounded-full bg-secondary"><div className={cn("h-full rounded-full", pct > 90 ? "bg-destructive" : pct > 70 ? "bg-warning" : "bg-primary")} style={{ width: `${pct}%` }} /></div>
      <svg viewBox={`0 0 ${w} ${h}`} className="h-5 w-[72px] overflow-visible" aria-hidden="true"><path d={d} fill="none" stroke="currentColor" strokeWidth="1.5" className="text-muted-foreground" /></svg>
    </div>
  );
}

export default function SettingsPage() {
  const [secrets, setSecrets] = useState<Secrets | null>(null);
  const [tests, setTests] = useState<Record<string, { ok: boolean; message: string; latency_ms: number } | "busy">>({});
  const [config, setConfig] = useState<Config | null>(null);
  const [byok, setByok] = useState<{ set: boolean; last4: string | null; warning: string | null } | null>(null);
  const [keyInput, setKeyInput] = useState("");
  const [keyMsg, setKeyMsg] = useState<string | null>(null);
  const [sys, setSys] = useState<System | null>(null);
  const [hist, setHist] = useState<{ cpu: number[]; gpu: number[]; gmem: number[]; ram: number[] }>({ cpu: [], gpu: [], gmem: [], ram: [] });
  const [defaults, setDefaults] = useState({ answer_model: "claude-opus-5-5", mode: "fused" });

  useEffect(() => {
    j<Secrets>("/api/secrets").then(setSecrets);
    j<Config>("/api/config").then(setConfig);
    j<typeof byok & object>("/api/settings/key").then((b) => setByok(b as { set: boolean; last4: string | null; warning: string | null }));
    // Read after mount (microtask) so the server-rendered selects hydrate first, then take the stored defaults.
    Promise.resolve().then(() => { try { const d = localStorage.getItem("matt.defaults"); if (d) setDefaults(JSON.parse(d)); } catch { /* ignore */ } });
  }, []);

  const poll = useCallback(async () => {
    const s = await j<System>("/api/system");
    setSys(s);
    setHist((h) => ({ cpu: [...h.cpu, s.cpu.percent].slice(-24), gpu: [...h.gpu, s.gpu.util_percent].slice(-24), gmem: [...h.gmem, s.gpu.memory_used_gb].slice(-24), ram: [...h.ram, s.memory.used_gb].slice(-24) }));
  }, []);
  useEffect(() => { const first = setTimeout(poll, 0); const t = setInterval(poll, 5000); return () => { clearTimeout(first); clearInterval(t); }; }, [poll]);

  const test = async (id: string) => { setTests((t) => ({ ...t, [id]: "busy" })); const r = await j<{ ok: boolean; message: string; latency_ms: number }>("/api/secrets/test", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ service: id }) }); setTests((t) => ({ ...t, [id]: r })); };
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const save = async (id: string) => {
    setSaveMsg(null); setTests((t) => ({ ...t, [id]: "busy" }));
    try {
      const r = await j<{ persisted: boolean; message: string; test: { ok: boolean; message: string; latency_ms: number } }>("/api/secrets", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ service: id, values: draft }) });
      setTests((t) => ({ ...t, [id]: r.test })); setSaveMsg(r.message); setEditing(null); setDraft({});
      j<Secrets>("/api/secrets").then(setSecrets);
    } catch (e) { setSaveMsg(e instanceof Error ? e.message : "Could not save"); setTests((t) => { const c = { ...t }; delete c[id]; return c; }); }
  };
  const saveKey = async () => { setKeyMsg(null); try { const r = await j<{ set: boolean; last4: string }>("/api/settings/key", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key: keyInput }) }); setByok((b) => ({ set: r.set, last4: r.last4, warning: b?.warning ?? null })); setKeyInput(""); setKeyMsg("Saved for this browser session."); } catch (e) { setKeyMsg(e instanceof Error ? e.message : "Could not save"); } };
  const clearKey = async () => { await j("/api/settings/key", { method: "DELETE" }); setByok((b) => ({ set: false, last4: null, warning: b?.warning ?? null })); setKeyMsg("Cleared."); };
  const saveDefaults = (d: typeof defaults) => { setDefaults(d); try { localStorage.setItem("matt.defaults", JSON.stringify(d)); } catch { /* ignore */ } };

  return (
    <AppShell title="Settings" subtitle="secrets · models · runtime">
      <div className="mx-auto grid w-full max-w-5xl gap-8 p-4 sm:p-6">
        <Section title="Secrets" sub={secrets?.writable ? "Keys you save here are tested live, applied immediately and written to the backend's .env file. Nothing is stored in the browser." : "Keys you save here are tested live and applied to the running instance. On a hosted deployment, set the same variables as Modal secrets to persist them."}>
          {saveMsg ? <p className="text-xs text-muted-foreground">{saveMsg}</p> : null}
          <div className="overflow-x-auto rounded-lg border border-border bg-card">
            <table className="w-full text-sm">
              <thead className="bg-secondary/60 text-left font-mono text-[11px] uppercase tracking-wider text-muted-foreground"><tr><th className="px-3 py-2">Service</th><th className="px-3 py-2">Keys</th><th className="px-3 py-2">Stored</th><th className="px-3 py-2">Status</th><th className="px-3 py-2"></th></tr></thead>
              <tbody>
                {secrets?.services.map((s) => { const t = tests[s.id]; const allSet = s.keys.every((k) => k.set); const open = editing === s.id; return (
                  <tr key={s.id} className="border-t border-border align-top">
                    <td className="px-3 py-2.5"><div className="font-medium">{s.name}</div><div className="text-xs text-muted-foreground">{s.purpose}</div></td>
                    <td className="px-3 py-2.5">
                      <div className="grid gap-1.5">{s.keys.map((k) => open ? (
                        <label key={k.name} className="grid gap-0.5 font-mono text-xs"><span>{k.name}</span>
                          <Input type={/KEY|SECRET/.test(k.name) ? "password" : "text"} autoComplete="off" value={draft[k.name] ?? ""} onChange={(e) => setDraft((d) => ({ ...d, [k.name]: e.target.value }))} placeholder={k.masked ?? "not set"} className="h-7 w-72 max-w-full font-mono text-xs" />
                        </label>
                      ) : (
                        <div key={k.name} className="flex items-center gap-2 font-mono text-xs"><span className={cn("size-1.5 rounded-full", k.set ? "bg-success" : "bg-destructive")} />{k.name}<span className="text-muted-foreground">{k.masked ?? "not set"}</span></div>
                      ))}</div>
                    </td>
                    <td className="px-3 py-2.5 text-xs text-muted-foreground">{s.stored}</td>
                    <td className="px-3 py-2.5 text-xs">{t === "busy" ? <span className="text-muted-foreground">testing…</span> : t ? <span className={t.ok ? "text-success" : "text-destructive"}>{t.message} · {t.latency_ms} ms</span> : <span className="text-muted-foreground">{allSet ? "configured" : "missing keys"}</span>}</td>
                    <td className="px-3 py-2.5 text-right"><div className="flex justify-end gap-1.5">
                      {open ? (<><Button size="sm" onClick={() => save(s.id)} disabled={t === "busy" || !Object.values(draft).some((v) => v.trim())}>Save</Button><Button size="sm" variant="ghost" onClick={() => { setEditing(null); setDraft({}); }}>Cancel</Button></>)
                        : (<><Button size="sm" variant="outline" onClick={() => { setEditing(s.id); setDraft({}); }}>Edit</Button><Button size="sm" variant="outline" onClick={() => test(s.id)} disabled={t === "busy"}>Test</Button></>)}
                    </div></td>
                  </tr>); })}
              </tbody>
            </table>
          </div>
        </Section>

        <Section title="Your Anthropic key" sub="Optional. Use your own key for this browser session; it is encrypted into an httpOnly cookie, never stored in a database, and forwarded to the backend per request.">
          <div className="grid gap-2 rounded-lg border border-border bg-card p-3 sm:grid-cols-[minmax(0,1fr)_auto_auto]">
            <Input type="password" value={keyInput} onChange={(e) => setKeyInput(e.target.value)} placeholder={byok?.set ? `key ending in ${byok.last4} is active` : "sk-ant-…"} autoComplete="off" />
            <Button onClick={saveKey} disabled={!keyInput.trim()}>Save</Button>
            <Button variant="outline" onClick={clearKey} disabled={!byok?.set}>Clear</Button>
            <div className="text-xs text-muted-foreground sm:col-span-3">{keyMsg ?? (byok?.set ? `Active: ••••${byok.last4}` : "No session key; the server's key is used.")}{byok?.warning ? <span className="ml-2 text-warning">{byok.warning}</span> : null}</div>
          </div>
        </Section>

        <Section title="Runtime" sub={sys ? `Modal container ${sys.container.id} · ${sys.container.gpu} · ${sys.container.region} · ${sys.container.state} · up ${Math.floor(sys.container.uptime_s / 60)} min · ${sys.container.cold_starts_today} cold starts today · sampled ${sys.sampled_at.slice(11, 19)} UTC${sys.source === "mock" ? " · mock data" : ""}` : "loading"}>
          {sys ? (
            <div className="grid gap-3">
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <Meter label="CPU" value={sys.cpu.percent} max={100} unit="%" hist={hist.cpu} />
                <Meter label="RAM" value={sys.memory.used_gb} max={sys.memory.total_gb} unit=" GB" hist={hist.ram} />
                <Meter label="GPU utilisation" value={sys.gpu.util_percent} max={100} unit="%" hist={hist.gpu} />
                <Meter label="GPU memory" value={sys.gpu.memory_used_gb} max={sys.gpu.memory_total_gb} unit=" GB" hist={hist.gmem} />
              </div>
              <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted-foreground">
                {sys.models_loaded.map((m) => <span key={m.name}>{m.name} <b className="font-mono text-foreground">{m.memory_gb} GB</b></span>)}
                <span>requests / h <b className="font-mono text-foreground">{sys.requests.last_hour}</b></span>
                <span>p50 <b className="font-mono text-foreground">{sys.requests.p50_ms} ms</b></span>
                <span>p95 <b className="font-mono text-foreground">{sys.requests.p95_ms} ms</b></span>
                <span>errors <b className="font-mono text-foreground">{sys.requests.errors}</b></span>
                <span>GPU {sys.gpu.temperature_c} °C</span>
              </div>
            </div>
          ) : null}
        </Section>

        <Section title="Models and stores" sub="Served by the pipeline itself, so this list cannot drift from what runs.">
          <div className="overflow-x-auto rounded-lg border border-border bg-card">
            <table className="w-full text-sm">
              <thead className="bg-secondary/60 text-left font-mono text-[11px] uppercase tracking-wider text-muted-foreground"><tr><th className="px-3 py-2">Stage</th><th className="px-3 py-2">Model</th><th className="px-3 py-2">Detail</th><th className="px-3 py-2">Runs on</th></tr></thead>
              <tbody>{config?.pipeline.map((p) => <tr key={p.stage} className="border-t border-border"><td className="px-3 py-2 font-medium">{p.stage}</td><td className="px-3 py-2 font-mono text-xs">{p.model}</td><td className="px-3 py-2 text-xs text-muted-foreground">{p.detail}</td><td className="px-3 py-2 text-xs">{p.runs_on}</td></tr>)}</tbody>
            </table>
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {config?.stores.map((s) => <div key={s.name} className="rounded-lg border border-border bg-card p-3 text-sm"><div className="flex items-center justify-between"><span className="font-medium">{s.name}</span><Badge variant="outline" className="font-mono text-[10px]">{s.role}</Badge></div><div className="mt-1 text-xs text-muted-foreground">{s.detail}</div><div className="mt-1 font-mono text-[11px] text-muted-foreground">{s.location}</div></div>)}
          </div>
        </Section>

        <Section title="Defaults" sub="Applied to new conversations in the Playground and the Dataset.">
          <div className="grid gap-3 rounded-lg border border-border bg-card p-3 sm:grid-cols-2">
            <label className="grid gap-1 text-xs text-muted-foreground">Answer model
              <select value={defaults.answer_model} onChange={(e) => saveDefaults({ ...defaults, answer_model: e.target.value })} className="rounded-md border border-input bg-background px-2 py-1.5 text-sm text-foreground">
                {(config?.answer_models ?? [defaults.answer_model]).map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </label>
            <label className="grid gap-1 text-xs text-muted-foreground">Retrieval mode
              <select value={defaults.mode} onChange={(e) => saveDefaults({ ...defaults, mode: e.target.value })} className="rounded-md border border-input bg-background px-2 py-1.5 text-sm text-foreground">
                {(config?.retrieval.modes ?? [defaults.mode]).map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </label>
          </div>
        </Section>
      </div>
    </AppShell>
  );
}
