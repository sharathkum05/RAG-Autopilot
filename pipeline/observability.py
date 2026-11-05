"""Observability data for the UI: traces grouped from Langfuse observations (one trace per turn or eval answer, a stage per
span, the Claude call as a generation with tokens and cost), with the stored `turns` rows as a fallback when Langfuse is
not configured. Short in-memory cache so the tab does not hammer the API."""
from __future__ import annotations

import json
import statistics
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from pipeline.tracing import langfuse

WINDOWS = {"24h": 1, "7d": 7, "30d": 30, "all": 3650}
FIELDS = "basic,core,io,metadata,usage,model"
ROOT_NAMES = {"turn", "answer_eval"}            # our traces; the SDK's auto-instrumented Anthropic calls land in traces of their own
CACHE_TTL = 20.0
_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()


def _ts(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, datetime):
        return v.timestamp()
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def pct(xs: list[float], p: float) -> int:
    if not xs:
        return 0
    ys = sorted(xs)
    return int(ys[min(len(ys) - 1, int(round(p * (len(ys) - 1))))])


def _obj(v) -> dict:
    """Langfuse returns input/output/metadata as JSON text or as objects depending on the field group; always hand back a dict."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def _num(v) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def fetch_observations(since: datetime, trace_id: str | None = None, max_pages: int = 10) -> list[dict]:
    api = langfuse().api
    out: list[dict] = []
    cursor = None
    for _ in range(max_pages):
        kw = {"limit": 100, "cursor": cursor, "fields": FIELDS}
        if trace_id:
            kw["trace_id"] = trace_id
        else:
            kw["from_start_time"] = since
        r = api.observations.get_many(**kw)
        out.extend(o.dict(by_alias=True) for o in (r.data or []))
        cursor = getattr(r.meta, "cursor", None) if r.meta else None
        if not cursor or not r.data:
            break
    return out


def fetch_scores(trace_id: str) -> dict[str, float]:
    """{score name: value} for one trace (the answer evals push answer_check / judge_correctness). The list endpoint does not
    return trace ids, so scores are fetched per trace, only for the detail view."""
    try:
        r = langfuse().api.scores.get_many(trace_id=trace_id, limit=20)
        out: dict[str, float] = {}
        for sc in r.data or []:
            d = sc.dict(by_alias=True)
            if d.get("value") is not None:
                out[d.get("name", "score")] = _num(d.get("value"))
        return out
    except Exception:  # noqa: BLE001
        return {}


def _stage(o: dict, t0: float) -> dict:
    s, e = _ts(o.get("startTime")), _ts(o.get("endTime")) or _ts(o.get("startTime"))
    usage = o.get("usageDetails") or {}
    cost = o.get("costDetails") or {}
    return {"id": o.get("id"), "name": o.get("name"), "type": (o.get("type") or "").lower(), "start_ms": int((s - t0) * 1000), "ms": int(max(0.0, e - s) * 1000),
            "model": o.get("model") or None, "tokens_in": int(_num(usage.get("input"))), "tokens_out": int(_num(usage.get("output"))),
            "cost_usd": _num(cost.get("total")) or _num(o.get("totalCost")), "level": o.get("level"), "status": o.get("statusMessage"),
            "input": _obj(o.get("input")) or o.get("input"), "output": _obj(o.get("output")) or o.get("output"), "metadata": _obj(o.get("metadata"))}


def build_traces(obs: list[dict], with_io: bool = False) -> list[dict]:
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for o in obs:
        if o.get("traceId"):
            by_trace[o["traceId"]].append(o)
    lf = langfuse()
    traces: list[dict] = []
    for tid, items in by_trace.items():
        items.sort(key=lambda o: _ts(o.get("startTime")))
        root = next((o for o in items if not o.get("parentObservationId")), items[0])
        if root.get("name") not in ROOT_NAMES:
            continue
        t0 = _ts(items[0].get("startTime"))
        t1 = max(_ts(o.get("endTime")) or _ts(o.get("startTime")) for o in items)
        inp, out, meta = _obj(root.get("input")), _obj(root.get("output")), _obj(root.get("metadata"))
        # the SDK's auto-instrumented Anthropic call duplicates our "Answering" generation; keep ours
        stages = [_stage(o, t0) for o in items if o is not root and o.get("name") != "anthropic.messages.create"]
        answer = next((s for s in stages if s["name"] == "Answering"), None)
        cost = _num(meta.get("cost_usd")) or (answer["cost_usd"] if answer else 0.0)
        retrieval_ms = sum(s["ms"] for s in stages if s["type"] == "span" and s["name"] != "Answering")
        if not with_io:
            for s in stages:
                s.pop("input", None); s.pop("output", None); s.pop("metadata", None)
        url = None
        try:
            url = lf.get_trace_url(trace_id=tid) if lf else None
        except Exception:  # noqa: BLE001
            pass
        traces.append({
            "id": tid, "name": root.get("name"), "start": _iso(t0), "latency_ms": int((t1 - t0) * 1000), "retrieval_ms": retrieval_ms, "answer_ms": answer["ms"] if answer else 0,
            "question": inp.get("question"), "workspace": inp.get("workspace"), "config": inp.get("config"), "source": meta.get("source") or ("answer_eval" if root.get("name") == "answer_eval" else "chat"),
            "conversation_id": meta.get("conversation_id"), "eval_run": meta.get("eval_run"), "query_id": meta.get("query_id"),
            "model": (answer or {}).get("model") or None, "tokens_in": (answer or {}).get("tokens_in", 0), "tokens_out": (answer or {}).get("tokens_out", 0), "cost_usd": round(cost, 5),
            "confidence": out.get("confidence"), "cited_pages": out.get("cited_pages"), "error": out.get("error"), "level": root.get("level"),
            "scores": fetch_scores(tid) if with_io else {}, "stages": stages, "url": url,
            "input": inp if with_io else None, "output": out if with_io else None,
        })
    traces.sort(key=lambda t: t["start"], reverse=True)
    return traces


def summarise(traces: list[dict]) -> dict:
    def block(ts: list[dict]) -> dict:
        lat = [t["latency_ms"] for t in ts]
        return {"n": len(ts), "cost_usd": round(sum(t["cost_usd"] for t in ts), 4), "tokens_in": sum(t["tokens_in"] for t in ts), "tokens_out": sum(t["tokens_out"] for t in ts),
                "errors": sum(1 for t in ts if t.get("error") or t.get("level") == "ERROR"),
                "latency_ms": {"p50": pct(lat, 0.5), "p95": pct(lat, 0.95), "mean": int(statistics.mean(lat)) if lat else 0},
                "retrieval_ms": {"p50": pct([t["retrieval_ms"] for t in ts], 0.5), "p95": pct([t["retrieval_ms"] for t in ts], 0.95)},
                "answer_ms": {"p50": pct([t["answer_ms"] for t in ts if t["answer_ms"]], 0.5), "p95": pct([t["answer_ms"] for t in ts if t["answer_ms"]], 0.95)}}
    by_source: dict[str, list[dict]] = defaultdict(list)
    by_config: dict[str, list[dict]] = defaultdict(list)
    by_day: dict[str, list[dict]] = defaultdict(list)
    stage_ms: dict[str, list[float]] = defaultdict(list)
    for t in traces:
        by_source[t["source"] or "chat"].append(t)
        by_config[t["config"] or "?"].append(t)
        by_day[t["start"][:10]].append(t)
        for s in t["stages"]:
            stage_ms[s["name"]].append(s["ms"])
    return {"all": block(traces), "by_source": {k: block(v) for k, v in sorted(by_source.items())}, "by_config": {k: block(v) for k, v in sorted(by_config.items())},
            "by_day": [{"day": d, "n": len(v), "cost_usd": round(sum(t["cost_usd"] for t in v), 4), "p50_ms": pct([t["latency_ms"] for t in v], 0.5)} for d, v in sorted(by_day.items())],
            "stages": {k: {"n": len(v), "p50": pct(v, 0.5), "p95": pct(v, 0.95), "mean": int(statistics.mean(v))} for k, v in stage_ms.items() if v}}


def supabase_traces(since: datetime, limit: int = 300) -> list[dict]:
    """Fallback from the stored turns when Langfuse is off: stage offsets are reconstructed from the per-stage timings."""
    from pipeline.stores import supabase

    sb = supabase()
    rows = sb.table("turns").select("*").gte("created_at", since.isoformat()).order("created_at", desc=True).limit(limit).execute().data or []
    conv_ids = sorted({r["conversation_id"] for r in rows})
    convs = {c["id"]: c for c in (sb.table("conversations").select("id,source,mode,answer_model").in_("id", conv_ids).execute().data or [])} if conv_ids else {}
    traces: list[dict] = []
    for r in rows:
        c = convs.get(r["conversation_id"], {})
        timings = r.get("timings_ms") or {}
        stages, at = [], 0
        for name, ms in timings.items():
            stages.append({"id": f"{r['id']}-{name}", "name": name, "type": "span", "start_ms": at, "ms": int(ms or 0), "model": None, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "level": None, "status": None})
            at += int(ms or 0)
        ttr = int(r.get("ttr_ms") or 0)
        ans_ms = max(0, ttr - at)
        stages.append({"id": f"{r['id']}-answer", "name": "Answering", "type": "generation", "start_ms": at, "ms": ans_ms, "model": c.get("answer_model"), "tokens_in": int(r.get("tokens_in") or 0), "tokens_out": int(r.get("tokens_out") or 0), "cost_usd": float(r.get("cost_usd") or 0), "level": None, "status": None})
        traces.append({"id": r.get("trace_id") or f"turn-{r['id']}", "name": "turn", "start": r["created_at"], "latency_ms": ttr, "retrieval_ms": at, "answer_ms": ans_ms, "question": r["question"], "workspace": None,
                       "config": c.get("mode"), "source": c.get("source") or "chat", "conversation_id": r["conversation_id"], "eval_run": None, "query_id": None, "model": c.get("answer_model"),
                       "tokens_in": int(r.get("tokens_in") or 0), "tokens_out": int(r.get("tokens_out") or 0), "cost_usd": float(r.get("cost_usd") or 0), "confidence": r.get("confidence"), "cited_pages": None,
                       "error": None if r.get("status", "ok") == "ok" else r.get("status"), "level": None, "scores": {}, "stages": stages, "url": None, "input": None, "output": None})
    return traces


def overview(window: str = "7d", limit: int = 150) -> dict:
    window = window if window in WINDOWS else "7d"
    key = f"{window}:{limit}"
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]
    since = datetime.now(timezone.utc) - timedelta(days=WINDOWS[window])
    source, error = "langfuse", None
    traces: list[dict] = []
    if langfuse() is not None:
        try:
            traces = build_traces(fetch_observations(since))
        except Exception as e:  # noqa: BLE001
            source, error = "supabase", f"Langfuse: {type(e).__name__}: {str(e)[:160]}"
    else:
        source, error = "supabase", "Langfuse keys are not set; showing the stored turns"
    if source == "supabase":
        try:
            traces = supabase_traces(since)
        except Exception as e:  # noqa: BLE001
            error = f"{error} · Supabase: {type(e).__name__}: {str(e)[:120]}"
    payload = {"source": source, "window": window, "since": since.isoformat(), "error": error, "summary": summarise(traces), "traces": traces[:limit], "total": len(traces)}
    with _lock:
        _cache[key] = (time.time(), payload)
    return payload


def trace_detail(trace_id: str) -> dict | None:
    if langfuse() is None:
        return None
    obs = fetch_observations(datetime.now(timezone.utc) - timedelta(days=3650), trace_id=trace_id, max_pages=2)
    traces = build_traces(obs, with_io=True)
    return traces[0] if traces else None
