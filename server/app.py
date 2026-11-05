"""FastAPI server: the routes the UI calls. Run locally with `uv run uvicorn server.app:app --port 8000`; the same app
is mounted on Modal for the deployed version. Stores are always the hosted ones (Qdrant Cloud, Supabase, Langfuse)."""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterator

import psutil
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from pipeline.config import settings
from pipeline.ingest import run_ingest
from pipeline.retrieval import BASELINE, CONFIG_INFO, CONFIGS, DATASET_DEFAULT, IMPROVEMENT, PLAYGROUND_DEFAULT, explain, retrieve
from pipeline.stores import qdrant_ping, sb_get, sb_select, sb_upsert, supabase, supabase_ping
from pipeline.tracing import end, flush, langfuse, langfuse_ping, start_trace, trace_id_of

app = FastAPI(title="Matt", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=settings().cors_origins, allow_methods=["*"], allow_headers=["*"])
STARTED = time.time()
_reqs: list[tuple[float, int, bool]] = []   # (t, ms, ok)
DATASET_WORKSPACE = "python_textbook"


# ---------------------------------------------------------------------------- runtime sampling + log tail
# Modal has no pull API for container metrics, so the container samples itself: NVML on CUDA, psutil for CPU/RAM, every 10 s
# into a ring buffer that lives as long as the container does. The log tail captures the app's and uvicorn's log records.
import collections
import logging
import os

_samples: collections.deque = collections.deque(maxlen=360)
_log_lines: collections.deque = collections.deque(maxlen=400)


def _gpu_sample() -> dict[str, Any]:
    from pipeline.parsing import resolve_device
    dev = resolve_device()
    out: dict[str, Any] = {"name": {"cuda": "NVIDIA", "mps": "Apple GPU (MPS)", "cpu": "none"}[dev], "util_percent": 0, "memory_used_gb": 0.0, "memory_total_gb": 0.0, "temperature_c": 0}
    try:
        if dev == "cuda":
            import pynvml  # type: ignore
            pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
            u = pynvml.nvmlDeviceGetUtilizationRates(h); m = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            out = {"name": name.decode() if isinstance(name, bytes) else str(name), "util_percent": int(u.gpu), "memory_used_gb": round(m.used / 1e9, 2), "memory_total_gb": round(m.total / 1e9, 1), "temperature_c": int(pynvml.nvmlDeviceGetTemperature(h, 0))}
        elif dev == "mps":
            # unified memory; torch.mps calls from a second thread can crash, so report the host figure only
            out.update({"memory_total_gb": round(psutil.virtual_memory().total / 1e9, 1)})
    except Exception:  # noqa: BLE001
        pass
    return out


def _sample() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    g = _gpu_sample()
    return {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cpu": psutil.cpu_percent(interval=None), "ram_gb": round(vm.used / 1e9, 1), "gpu_util": g["util_percent"], "gpu_mem_gb": g["memory_used_gb"]}


def _sampler() -> None:
    time.sleep(8)          # let the server finish importing torch on the main thread first
    while True:
        try:
            _samples.append(_sample())
        except Exception:  # noqa: BLE001
            pass
        time.sleep(10)


class _RingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _log_lines.append(f"{datetime.fromtimestamp(record.created, tz=timezone.utc).strftime('%H:%M:%S')} {record.levelname[:4]} {record.name}: {record.getMessage()}"[:400])
        except Exception:  # noqa: BLE001
            pass


_ring = _RingHandler()
for _name in ("", "uvicorn", "uvicorn.access"):        # root catches "matt" and anything that propagates; uvicorn's loggers do not propagate
    logging.getLogger(_name).addHandler(_ring)
log = logging.getLogger("matt")
log.setLevel(logging.INFO)
threading.Thread(target=_sampler, daemon=True, name="runtime-sampler").start()
log.info("backend started (pid %s, device sampling every 10 s)", os.getpid())


def ndjson(e: dict) -> bytes:
    return (json.dumps(e, ensure_ascii=False) + "\n").encode()


# ---------------------------------------------------------------------------- ingest

@app.post("/api/ingest")
async def ingest(files: list[UploadFile] = File(...), workspace: str | None = Form(None)):
    if not files:
        raise HTTPException(400, "no files")
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    ws = workspace or f"conv_pg_{job_id[4:8]}"
    payload = [(f.filename or "upload", await f.read()) for f in files]
    sb_upsert("conversations", {"id": ws, "source": "playground", "workspace": ws, "mode": PLAYGROUND_DEFAULT, "answer_model": settings().answer_model})
    threading.Thread(target=run_ingest, args=(job_id, ws, payload), daemon=True).start()
    return {"jobId": job_id, "workspace": ws}


@app.get("/api/ingest")
def ingest_status(job: str):
    row = sb_get("jobs", job)
    if not row:
        raise HTTPException(404, "unknown job")
    done = row["status"] in ("done", "failed")
    return {"jobId": job, "files": row["files"], "steps": row["steps"], "progress": row.get("progress") or 0, "done": done,
            "conversationId": row["workspace"] if row["status"] == "done" else None, "error": row.get("error")}


# ---------------------------------------------------------------------------- chat

class ChatIn(BaseModel):
    question: str
    conversationId: str
    source: str = "playground"
    mode: str | None = None
    model: str | None = None


@app.post("/api/chat")
def chat(body: ChatIn, x_anthropic_key: str | None = Header(default=None)):
    s = settings()
    on_dataset = body.source in ("dataset", "compare")      # "compare": the ViDoRe Pipeline tab running baseline and improvement side by side
    default = DATASET_DEFAULT if on_dataset else PLAYGROUND_DEFAULT
    config = (body.mode or default).replace(" + ", "_").replace(" ", "_")
    config = config if config in CONFIGS else default
    workspace = DATASET_WORKSPACE if on_dataset else body.conversationId
    model = body.model or s.answer_model
    sb_upsert("conversations", {"id": body.conversationId, "source": body.source, "workspace": workspace, "mode": config, "answer_model": model})

    def gen() -> Iterator[bytes]:
        t_start = time.perf_counter()
        events: list[dict] = []
        yield ndjson({"type": "start", "conversationId": body.conversationId})
        trace = start_trace("turn", input={"question": body.question, "workspace": workspace, "config": config}, metadata={"conversation_id": body.conversationId, "source": body.source})
        try:
            # Retrieval runs in a worker thread so step events stream out as each stage completes.
            result = None
            err: str | None = None
            import queue
            q: queue.Queue = queue.Queue()

            def worker():
                nonlocal result, err
                try:
                    result = retrieve(body.question, workspace, config, trace=trace,
                                      on_step=lambda n, d, st, ms: q.put({"type": "step", "step": {"name": n, "detail": d, "state": st, "ms": ms}}))
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            while True:
                item = q.get()
                if item is None:
                    break
                yield ndjson(item)
            if err or result is None:
                end(trace, output={"error": err}, level="ERROR")
                yield ndjson({"type": "error", "message": err or "retrieval failed"})
                yield ndjson({"type": "done"})
                return

            retrieval_ms = sum(result.timings_ms.values())
            if not result.evidence:
                meta = {"confidence": "low", "chunks": [], "retrieved": [], "config": config, "retrievalMs": retrieval_ms, "ttrMs": int((time.perf_counter() - t_start) * 1000), "clarify": []}
                yield ndjson({"type": "token", "text": "I couldn't find anything in the indexed documents that answers this. Try a more specific question, or upload the document that contains it."})
                yield ndjson({"type": "evidence", "meta": meta})
                yield ndjson({"type": "done"})
                end(trace, output={"answer": None, "evidence": 0})
                return

            from pipeline.answer import stream_answer
            yield ndjson({"type": "step", "step": {"name": "Answering", "detail": f"{model} · top {settings().answer_pages} pages as citable documents", "state": "active"}})
            final = None
            try:
                for kind, payload in stream_answer(body.question, result.evidence, model=model, api_key=x_anthropic_key, trace=trace):
                    if kind == "token":
                        yield ndjson({"type": "token", "text": payload})
                    else:
                        final = payload
            except Exception as e:  # noqa: BLE001
                end(trace, output={"error": f"answer failed: {e}"}, level="ERROR")
                yield ndjson({"type": "error", "message": f"answer failed: {type(e).__name__}: {e}"})
                yield ndjson({"type": "done"})
                return
            yield ndjson({"type": "step", "step": {"name": "Answering", "detail": f"{final.model} · {final.tokens_in} in · {final.tokens_out} out", "state": "done", "ms": final.ttr_ms}})

            cited_ev = [result.evidence[i] for i in final.cited] or result.evidence[:1]
            margin = (result.evidence[0].score - result.evidence[1].score) if len(result.evidence) > 1 else 1.0
            agree = result.evidence[0].visual_rank is not None and result.evidence[0].text_rank is not None
            confidence = "high" if (final.cited and (agree or margin > 0.05)) else ("medium" if final.cited else "low")
            ttr_ms = int((time.perf_counter() - t_start) * 1000)
            meta = {"confidence": confidence, "chunks": [e.to_chunk(i) for i, e in enumerate(cited_ev)], "retrievalMs": retrieval_ms,
                    "retrieved": [e.to_chunk(i) for i, e in enumerate(result.evidence)], "config": config,   # the full ranked list, for the compare view
                    "ttrMs": ttr_ms, "note": final.note, "cost": final.cost_usd, "model": final.model}
            yield ndjson({"type": "evidence", "meta": meta})
            yield ndjson({"type": "done"})
            log.info("turn %s %s/%s retrieval %d ms answer %d ms ttr %d ms $%.3f %s", body.conversationId, body.source, config, retrieval_ms, final.ttr_ms, ttr_ms, final.cost_usd, body.question[:80].replace("\n", " "))
            end(trace, output={"answer": final.text[:2000], "confidence": confidence, "cited_pages": [f"{e.doc_name} p.{e.page_no}" for e in cited_ev]},
                metadata={"timings_ms": result.timings_ms, "ttr_ms": ttr_ms, "cost_usd": final.cost_usd})
            try:
                sb_upsert("turns", {"conversation_id": body.conversationId, "question": body.question, "answer": final.text, "confidence": confidence,
                                    # every retrieved chunk, cited ones first in citation order, so the Evidence tab can
                                    # show what was found but not used and explain the confidence (pipeline/diagnose.py)
                                    # (cited=False throughout when Claude cited nothing and the top page was only shown as a fallback)
                                    "evidence": [dict(c, cited=bool(final.cited)) for c in meta["chunks"]]
                                                + [dict(e.to_chunk(i), cited=False) for i, e in enumerate(result.evidence) if e not in cited_ev],
                                    "timings_ms": result.timings_ms, "ttr_ms": ttr_ms, "tokens_in": final.tokens_in,
                                    "tokens_out": final.tokens_out, "cost_usd": final.cost_usd, "trace_id": trace_id_of(trace), "status": "ok" if not final.note else "warn"})
            except Exception:
                pass
            _reqs.append((time.time(), retrieval_ms, True))
        finally:
            flush()

    # Modal's proxy compresses (zstd/gzip) any response a browser accepts compressed, and buffers the whole stream to do
    # it, so the steps and the typed answer all landed at once at the end. Event streams and already-encoded responses
    # are passed through, so the NDJSON body is sent under the event-stream type with an explicit identity encoding.
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache, no-transform", "Content-Encoding": "identity", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------- index + explain

@app.get("/api/index/documents")
def index_documents(workspace: str = DATASET_WORKSPACE):
    docs = sb_select("documents", workspace=workspace)
    s = settings()
    return {"documents": [{"id": d["id"], "name": d["name"], "source": d["source"], "pages": d["pages"], "chunks": d["chunks"], "indexed_at": d.get("indexed_at") or d["created_at"],
                           "collections": {"pages": s.pages_collection, "chunks": s.chunks_collection}, "status": d["status"]} for d in docs]}


@app.get("/api/index/pages")
def index_pages(doc: str | None = None, id: str | None = None, workspace: str = DATASET_WORKSPACE):
    if id:
        row = sb_get("pages", id)
        if not row:
            raise HTTPException(404, "unknown page")
        return _page_out(row)
    if not doc:
        docs = sb_select("documents", workspace=workspace)
        doc = docs[0]["id"] if docs else None
        if not doc:
            return {"pages": []}
    rows = supabase().table("pages").select("*").eq("document_id", doc).order("page_no").execute().data or []
    return {"pages": [_page_out(r) for r in rows]}


def _page_out(r: dict) -> dict:
    return {"id": r["id"], "pno": r["page_no"], "width": r.get("width") or 0, "height": r.get("height") or 0, "image": r.get("image_url") or "",
            "n_chunks": r.get("n_chunks") or 0, "n_visual_vectors": r.get("n_visual_vectors") or 0, "n_words": len((r.get("text") or "").split()), "document_id": r["document_id"]}


@app.get("/api/index/chunks")
def index_chunks(page: str | None = None, ids: str | None = None):
    q = supabase().table("chunks").select("*")
    if page:
        q = q.eq("page_id", page).order("id")
    elif ids:
        q = q.in_("id", ids.split(","))
    else:
        q = q.limit(50)
    rows = q.execute().data or []
    return {"chunks": [_chunk_out(r) for r in rows]}


def _chunk_out(r: dict) -> dict:
    return {"id": r["id"], "page": r["page_id"], "pno": r["page_no"], "kind": r["kind"], "heading": " › ".join(r.get("heading_path") or []), "text": r["text"],
            "n_tokens": r.get("n_tokens") or 0, "vector_id": r["vector_id"], "bbox": r.get("bbox")}


@app.get("/api/index/search")
def index_search(q: str, workspace: str = DATASET_WORKSPACE, limit: int = 40):
    terms = [t for t in q.lower().split() if len(t) > 1]
    if not terms:
        return {"query": q, "terms": [], "total": 0, "results": []}
    query = supabase().table("chunks").select("*").eq("document_id", "__none__")
    docs = [d["id"] for d in sb_select("documents", workspace=workspace)]
    query = supabase().table("chunks").select("*").in_("document_id", docs)
    for t in terms:
        query = query.ilike("text", f"%{t}%")
    rows = query.limit(400).execute().data or []
    scored = sorted(rows, key=lambda r: -sum(r["text"].lower().count(t) for t in terms))
    return {"query": q, "terms": terms, "total": len(scored), "results": [{**_chunk_out(r), "matches": sum(r["text"].lower().count(t) for t in terms)} for r in scored[:limit]]}


class ExplainIn(BaseModel):
    question: str
    k: int = 5
    workspace: str = DATASET_WORKSPACE


@app.post("/api/retrieve/explain")
def retrieve_explain(body: ExplainIn):
    trace = start_trace("explain", input={"question": body.question, "workspace": body.workspace})
    try:
        out = explain(body.question, body.workspace, body.k)
        end(trace, output={"final": out.get("final")})
    except Exception as e:
        end(trace, output={"error": str(e)}, level="ERROR")
        raise
    finally:
        flush()
    return out


# ---------------------------------------------------------------------------- evidence (per-conversation inspection)

@app.get("/api/evidence/workspaces")
def evidence_workspaces():
    """Every workspace that can be inspected: the dataset plus each Playground conversation that indexed a document,
    with its documents and per-question totals (count, cost, mean TTR)."""
    from collections import defaultdict
    s = settings()
    db = supabase()
    docs = db.table("documents").select("id,workspace,name,source,file_type,pages,chunks,status,created_at,indexed_at").order("created_at", desc=True).limit(500).execute().data or []
    convs = db.table("conversations").select("*").order("created_at", desc=True).limit(500).execute().data or []
    turns = db.table("turns").select("conversation_id,ttr_ms,cost_usd,created_at").order("created_at", desc=True).limit(2000).execute().data or []
    conv_ws = {c["id"]: c["workspace"] for c in convs}
    ws_docs: dict[str, list] = defaultdict(list)
    for d in docs:
        ws_docs[d["workspace"]].append(d)
    ws_turns: dict[str, list] = defaultdict(list)
    for t in turns:
        ws = conv_ws.get(t["conversation_id"])
        if ws:
            ws_turns[ws].append(t)
    ws_conv: dict[str, dict] = {}
    for c in convs:
        ws_conv.setdefault(c["workspace"], c)          # newest conversation per workspace
    out = []
    for ws in set(ws_docs) | {DATASET_WORKSPACE}:
        ds, ts = ws_docs.get(ws, []), ws_turns.get(ws, [])
        kind = "dataset" if ws == DATASET_WORKSPACE else "playground"
        if kind == "playground" and not ds:
            continue
        conv = ws_conv.get(ws)
        ttrs = [t["ttr_ms"] for t in ts if t.get("ttr_ms")]
        stamps = [t["created_at"] for t in ts] + [d["created_at"] for d in ds] + ([conv["created_at"]] if conv else [])
        out.append({
            "id": ws, "kind": kind,
            "title": ", ".join(d["name"] for d in ds) if ds else ("Introduction to Python Programming" if kind == "dataset" else ws),
            "documents": [{"id": d["id"], "name": d["name"], "source": d["source"], "file_type": d.get("file_type"), "pages": d["pages"], "chunks": d["chunks"],
                           "status": d["status"], "indexed_at": d.get("indexed_at") or d["created_at"], "collections": {"pages": s.pages_collection, "chunks": s.chunks_collection}} for d in ds],
            "turns": len(ts), "cost_usd": round(sum(float(t.get("cost_usd") or 0) for t in ts), 5), "avg_ttr_ms": int(sum(ttrs) / len(ttrs)) if ttrs else None,
            "last_at": max(stamps) if stamps else None, "created_at": conv["created_at"] if conv else (ds[-1]["created_at"] if ds else None),
            "mode": conv["mode"] if conv else None, "model": conv.get("answer_model") if conv else None,
        })
    dataset = [w for w in out if w["kind"] == "dataset"]
    playground = sorted([w for w in out if w["kind"] == "playground"], key=lambda w: w["last_at"] or "", reverse=True)
    return {"workspaces": dataset + playground}


@app.delete("/api/evidence/workspaces/{workspace}")
def evidence_delete_workspace(workspace: str):
    """Deletes a Playground conversation: its turns and conversation rows, its documents (pages and chunks cascade),
    its ingest jobs, its vectors in both Qdrant collections and its page images in Storage. The dataset is protected."""
    from pipeline.stores import delete_document_vectors, delete_workspace_images, sb_delete_document

    if workspace == DATASET_WORKSPACE:
        raise HTTPException(400, "the dataset workspace cannot be deleted")
    db = supabase()
    docs = db.table("documents").select("id").eq("workspace", workspace).execute().data or []
    convs = db.table("conversations").select("id").eq("workspace", workspace).execute().data or []
    if not docs and not convs:
        raise HTTPException(404, "unknown workspace")
    for d in docs:
        delete_document_vectors(d["id"])
    try:
        images = delete_workspace_images(workspace)
    except Exception as e:  # noqa: BLE001  (a Storage hiccup must not keep the rows alive)
        images, err = 0, f"{type(e).__name__}: {str(e)[:120]}"
    else:
        err = None
    turns = 0
    if convs:
        ids = [c["id"] for c in convs]
        turns = len(db.table("turns").delete().in_("conversation_id", ids).execute().data or [])
        db.table("conversations").delete().in_("id", ids).execute()
    for d in docs:
        sb_delete_document(d["id"])
    db.table("jobs").delete().eq("workspace", workspace).execute()
    return {"deleted": {"workspace": workspace, "documents": len(docs), "conversations": len(convs), "turns": turns, "images": images}, "storage_error": err}


@app.get("/api/evidence/turns")
def evidence_turns(workspace: str, limit: int = 100):
    """Every question asked in a workspace, newest first, with the chunks it cited, per-stage timings, tokens, cost and the Langfuse trace link."""
    db = supabase()
    convs = db.table("conversations").select("id,mode,answer_model").eq("workspace", workspace).execute().data or []
    if not convs:
        return {"turns": []}
    by_id = {c["id"]: c for c in convs}
    rows = db.table("turns").select("*").in_("conversation_id", list(by_id)).order("created_at", desc=True).limit(limit).execute().data or []
    lf = langfuse()

    def trace_url(tid: str | None) -> str | None:
        if not (tid and lf):
            return None
        try:
            return lf.get_trace_url(trace_id=tid)
        except Exception:  # noqa: BLE001
            return None

    from pipeline.diagnose import diagnose

    out = []
    for r in rows:
        mode = by_id[r["conversation_id"]].get("mode")
        chunks = r.get("evidence") or []
        out.append({**r, "cost_usd": float(r.get("cost_usd") or 0), "evidence": [c for c in chunks if c.get("cited", True)], "timings_ms": r.get("timings_ms") or {},
                    "trace_url": trace_url(r.get("trace_id")), "model": by_id[r["conversation_id"]].get("answer_model"), "mode": mode,
                    "diagnosis": diagnose(r, mode)})
    return {"turns": out}


# ---------------------------------------------------------------------------- evals

@app.get("/api/evals")
def evals_summary(workspace: str = DATASET_WORKSPACE):
    """Latest eval run per config (metrics only) plus the list of all runs."""
    rows = supabase().table("eval_runs").select("id,workspace,dataset,config,metrics,hardware,created_at").eq("workspace", workspace).order("created_at", desc=True).limit(200).execute().data or []
    latest: dict[str, dict] = {}
    for r in rows:
        latest.setdefault(r["config"], r)
    order = list(CONFIGS)
    return {"configs": order, "latest": [latest[c] for c in order if c in latest], "runs": rows, "baseline": BASELINE, "improvement": IMPROVEMENT, "info": CONFIG_INFO}


@app.get("/api/evals/queries")
def evals_queries(workspace: str = DATASET_WORKSPACE):
    """The benchmark queries with their labelled pages, plus the rank at which the latest baseline and improvement runs first hit one."""
    rows = supabase().table("eval_runs").select("config,per_query,created_at").eq("workspace", workspace).in_("config", [BASELINE, IMPROVEMENT]) \
        .order("created_at", desc=True).limit(20).execute().data or []
    latest: dict[str, list[dict]] = {}
    for r in rows:
        latest.setdefault(r["config"], r.get("per_query") or [])
    ranks = {c: {int(p["query_id"]): p.get("first_relevant_rank") for p in pq} for c, pq in latest.items()}
    source = latest.get(IMPROVEMENT) or latest.get(BASELINE) or []
    queries = [{"query_id": int(p["query_id"]), "query": p["query"], "answer": p.get("answer"), "content_type": p.get("content_type") or [], "relevant": p.get("relevant") or [],
                "ranks": {"baseline": ranks.get(BASELINE, {}).get(int(p["query_id"])), "improvement": ranks.get(IMPROVEMENT, {}).get(int(p["query_id"]))}} for p in source]
    return {"workspace": workspace, "baseline": BASELINE, "improvement": IMPROVEMENT, "queries": sorted(queries, key=lambda q: q["query_id"])}


@app.get("/api/evals/run")
def evals_run(id: str):
    """One run with its per-query detail (for the failure analysis)."""
    row = sb_get("eval_runs", id)
    if not row:
        raise HTTPException(404, "unknown run")
    return row


@app.get("/api/answer-evals")
def answer_evals_summary(workspace: str = DATASET_WORKSPACE):
    """Latest answer-eval run per config (summary + rubric, no per-query rows) plus the list of all runs."""
    try:
        rows = supabase().table("answer_eval_runs").select("id,workspace,dataset,config,answer_model,judge_model,n_pages,rubric,summary,hardware,created_at") \
            .eq("workspace", workspace).order("created_at", desc=True).limit(50).execute().data or []
        error = None
    except Exception as e:  # noqa: BLE001  (table not created yet)
        rows, error = [], f"{type(e).__name__}: {str(e)[:200]}"
    latest: dict[str, dict] = {}
    for r in rows:
        latest.setdefault(r["config"], r)
    from pipeline.judge import rubric as judge_rubric

    return {"configs": list(CONFIGS), "latest": [latest[c] for c in CONFIGS if c in latest], "runs": rows, "baseline": BASELINE, "improvement": IMPROVEMENT, "info": CONFIG_INFO,
            "judge_rubric": judge_rubric(), "error": error}


@app.get("/api/answer-evals/run")
def answer_evals_run(id: str):
    """One answer-eval run with every query: answer, reference, checks, judge, timings."""
    row = sb_get("answer_eval_runs", id)
    if not row:
        raise HTTPException(404, "unknown run")
    return row


# ---------------------------------------------------------------------------- observability

@app.get("/api/observability")
def observability(window: str = "7d", limit: int = 150):
    """Traces in the window (Langfuse, or the stored turns when Langfuse is off) with cost, tokens, latency per stage and a summary."""
    from pipeline.observability import overview
    return overview(window=window, limit=max(1, min(limit, 500)))


@app.get("/api/observability/trace")
def observability_trace(id: str):
    """One trace with every stage's input, output and metadata (the waterfall detail)."""
    from pipeline.observability import trace_detail
    t = trace_detail(id)
    if not t:
        raise HTTPException(404, "trace not found (or Langfuse is not configured)")
    return t


# ---------------------------------------------------------------------------- config, secrets, system

@app.get("/api/config")
def config():
    s = settings()
    from pipeline.parsing import resolve_device
    dev = resolve_device()
    where = {"cuda": "Modal L4", "mps": "local · Apple GPU", "cpu": "local · CPU"}[dev]
    return {
        "pipeline": [
            {"stage": "Parse", "model": "Docling + EasyOCR", "detail": "layout model, TableFormer (fast), OCR where a page has no text layer", "runs_on": where},
            {"stage": "Chunk", "model": "structural chunker", "detail": f"sections ≈{s.chunk_words} words, tables whole (header repeated when split by {s.table_rows_per_part} rows), figures = caption, heading path prepended for embedding", "runs_on": where},
            {"stage": "Text embedding", "model": s.text_model, "detail": "1024-d dense, normalised; max 1024 tokens", "runs_on": where},
            {"stage": "Sparse", "model": f"BM25 ({s.bm25_model})", "detail": "sparse vectors with IDF modifier", "runs_on": "CPU"},
            {"stage": "Visual embedding", "model": s.colqwen_model, "detail": "≈780 × 128-d multivectors per page + one pooled vector" + ("" if s.visual_enabled else " · disabled"), "runs_on": where},
            {"stage": "Fuse", "model": f"RRF (k={s.rrf_k})", "detail": f"dense + BM25 lists (inside Qdrant); the fused configs add the ColQwen page list; top {s.fused_k} → top {s.final_k}", "runs_on": "CPU"},
            {"stage": "Rerank (improvement)", "model": s.rerank_model, "detail": f"cross-encoder on (question, page text) over the text-hybrid top {s.fused_k} → top {s.final_k}; the baseline stops at RRF", "runs_on": where},
            {"stage": "Answer", "model": s.answer_model, "detail": "page text as citable documents + page images; native citations; server-side refusal fallback", "runs_on": "Anthropic API"},
        ],
        "stores": [
            {"name": "Qdrant", "role": "vector store", "detail": f"{s.pages_collection} (colqwen multivector MaxSim + pooled) · {s.chunks_collection} (dense + bm25 sparse)", "location": s.qdrant_url.split("//")[-1][:40] if s.qdrant_url else "not configured"},
            {"name": "Supabase Postgres", "role": "metadata", "detail": "documents, pages, chunks, jobs, conversations, turns, eval runs", "location": s.supabase_url.split("//")[-1][:40] if s.supabase_url else "not configured"},
            {"name": "Supabase Storage", "role": "page images", "detail": f"bucket {s.supabase_bucket}, display copies ≤{s.page_image_width}px", "location": "Supabase"},
            {"name": "Langfuse", "role": "traces", "detail": "one trace per turn: retrieval spans + answer generation with tokens and cost", "location": s.langfuse_host.split("//")[-1]},
        ],
        "retrieval": {"modes": list(CONFIGS), "default": DATASET_DEFAULT, "defaults": {"dataset": DATASET_DEFAULT, "playground": PLAYGROUND_DEFAULT}, "baseline": BASELINE, "improvement": IMPROVEMENT, "info": CONFIG_INFO, "top_k_per_path": s.top_k_per_path, "fused_k": s.fused_k, "final_k": s.final_k, "rrf_k": s.rrf_k},
        "answer_models": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
        "device": dev,
    }


SERVICES = {
    "anthropic": ["ANTHROPIC_API_KEY"], "qdrant": ["QDRANT_URL", "QDRANT_API_KEY"], "supabase": ["SUPABASE_URL", "SUPABASE_SERVICE_KEY"],
    "langfuse": ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"], "huggingface": ["HF_TOKEN"], "openai": ["OPENAI_API_KEY"],
}


def _env(name: str) -> str:
    return getattr(settings(), name.lower(), "") or ""


@app.get("/api/secrets")
def secrets():
    from pipeline.secrets_store import env_writable
    meta = {"anthropic": ("Anthropic", "answer generation"), "qdrant": ("Qdrant", "vector store"), "supabase": ("Supabase", "metadata and page images"), "langfuse": ("Langfuse", "traces, cost, TTR"),
            "huggingface": ("Hugging Face", "model downloads (optional: faster, no rate limits; needed only for gated models)"),
            "openai": ("OpenAI", "answer-eval judge (optional second opinion next to the deterministic checks)")}
    stored = "backend .env (writable)" if env_writable() else "Modal secret (read-only here)"
    return {"writable": env_writable(),
            "services": [{"id": k, "name": meta[k][0], "purpose": meta[k][1], "stored": stored, "editable": env_writable(),
                          "keys": [{"name": e, "set": bool(_env(e)), "masked": ("•" * 8 + _env(e)[-4:]) if _env(e) else None} for e in envs]} for k, envs in SERVICES.items()]}


class SecretsIn(BaseModel):
    service: str
    values: dict[str, str]


@app.post("/api/secrets")
def secrets_save(body: SecretsIn):
    """Apply new keys at runtime (and write .env when running locally), then test the service with them."""
    from pipeline.secrets_store import SERVICES as SVC, apply, env_writable
    if not env_writable():   # hosted (Modal): no login on this API, so keys can't be changed from the outside
        raise HTTPException(403, "Keys are read-only on the hosted backend; change them in the Modal secret and redeploy.")
    if body.service not in SVC:
        raise HTTPException(400, "unknown service")
    try:
        result = apply({k: v for k, v in body.values.items() if k in SVC[body.service]})
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    test = secrets_test(TestIn(service=body.service))
    note = ("saved to .env and applied" if result["persisted"] else "applied to this running instance only; set the same variables in Modal secrets to persist")
    return {**result, "writable": env_writable(), "test": test, "message": note}


class TestIn(BaseModel):
    service: str


@app.post("/api/secrets/test")
def secrets_test(body: TestIn):
    t0 = time.perf_counter()
    try:
        if body.service == "qdrant":
            info = qdrant_ping(); msg = f"ok · collections {', '.join(info['collections']) or 'none yet'} · {info['counts']}"
        elif body.service == "supabase":
            info = supabase_ping(); msg = f"ok · {info['documents']} documents"
        elif body.service == "langfuse":
            langfuse_ping(); msg = "ok · authenticated"
        elif body.service == "anthropic":
            import anthropic
            anthropic.Anthropic(api_key=settings().anthropic_api_key).models.list(limit=1); msg = "ok · key accepted"
        elif body.service == "huggingface":
            from huggingface_hub import whoami
            tok = settings().hf_token
            if not tok:
                raise RuntimeError("HF_TOKEN is not set (optional: downloads work unauthenticated, just slower)")
            who = whoami(token=tok); msg = f"ok · {who.get('name', 'authenticated')} · models cached locally"
        elif body.service == "openai":
            from openai import OpenAI
            key = settings().openai_api_key
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set (optional: the answer evals run the deterministic checks without a judge)")
            OpenAI(api_key=key).models.list(); msg = f"ok · key accepted · judge model {settings().judge_model}"
        else:
            raise HTTPException(400, "unknown service")
        return {"service": body.service, "ok": True, "message": msg, "latency_ms": int((time.perf_counter() - t0) * 1000)}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        return {"service": body.service, "ok": False, "message": f"{type(e).__name__}: {str(e)[:160]}", "latency_ms": int((time.perf_counter() - t0) * 1000)}


@app.get("/api/system")
def system():
    vm = psutil.virtual_memory()
    gpu = _gpu_sample()
    recent = [r for r in _reqs if r[0] > time.time() - 3600]
    lat = sorted(r[1] for r in recent)
    pct = lambda p: lat[min(len(lat) - 1, int(len(lat) * p))] if lat else 0  # noqa: E731
    loaded = []
    for mod, name in (("pipeline.embed_text", "bge-m3"), ("pipeline.embed_visual", "colqwen2.5-v0.2"), ("pipeline.retrieval", "bge-reranker-v2-m3")):
        import sys
        m = sys.modules.get(mod)
        fn = {"pipeline.embed_text": "text_embedder", "pipeline.embed_visual": "visual_embedder", "pipeline.retrieval": "reranker"}[mod]
        if m is not None and getattr(m, fn).cache_info().currsize:
            loaded.append({"name": name, "memory_gb": None})
    task = os.environ.get("MODAL_TASK_ID")
    load_s = os.environ.get("MATT_MODEL_LOAD_S")
    return {"source": "live", "sampled_at": datetime.now(timezone.utc).isoformat(),
            "container": {"id": task[-12:] if task else "local", "gpu": gpu["name"], "region": os.environ.get("MODAL_REGION", "local"), "cloud": os.environ.get("MODAL_CLOUD_PROVIDER"),
                          "uptime_s": int(time.time() - STARTED), "started_at": datetime.fromtimestamp(STARTED, tz=timezone.utc).isoformat(timespec="seconds"),
                          "model_load_s": float(load_s) if load_s else None, "cold_starts_today": 0, "state": "warm",
                          "scale_to_zero_after_s": int(os.environ.get("MATT_SCALEDOWN_S", "600")) if task else 0},
            "cpu": {"percent": psutil.cpu_percent(interval=None), "cores": psutil.cpu_count()},
            "memory": {"used_gb": round(vm.used / 1e9, 1), "total_gb": round(vm.total / 1e9, 1)}, "gpu": gpu, "models_loaded": loaded,
            "requests": {"last_hour": len(recent), "p50_ms": pct(0.5), "p95_ms": pct(0.95), "errors": sum(1 for r in recent if not r[2])},
            "history": list(_samples)}


@app.get("/api/system/logs")
def system_logs(limit: int = 200):
    """The container's own log tail (uvicorn + app records), newest last."""
    lines = list(_log_lines)[-max(1, min(limit, 400)):]
    task = os.environ.get("MODAL_TASK_ID")
    return {"lines": lines, "container": task[-12:] if task else "local"}


@app.get("/api/health")
def health():
    return {"ok": True, "uptime_s": int(time.time() - STARTED)}


@app.exception_handler(Exception)
async def on_error(_req, exc: Exception):
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
