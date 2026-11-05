"""Answer evaluation on the ViDoRe V3 Python textbook (English queries): retrieve with one config, answer with Claude exactly
as the app does, then score every answer with the deterministic checks in `pipeline.checks` and, when OPENAI_API_KEY is
set, a frozen judge. Writes one `answer_eval_runs` row (summary + per-query detail) for the Answer Evals tab.

Run:  uv run python scripts/run_answer_evals.py "raw dataset v3/python_textbook" [--config fused] [--sample 30 --seed 42] [--workers 4] [--no-judge]
      uv run python scripts/run_answer_evals.py --judge-run aeval_fused_xxxxxx      # add judge scores to a stored run without re-answering (cents)
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from pipeline.checks import run_checks, rubric
from pipeline.config import settings
from pipeline.ingest import now_iso
from pipeline.judge import DIMENSIONS, price_of
from pipeline.retrieval import CONFIGS, IMPROVEMENT, retrieve
from pipeline.stores import sb_upsert
from pipeline.tracing import end, flush, langfuse, start_trace, trace_id_of

WORKSPACE = "python_textbook"
DATASET = "vidore_v3_computer_science / Introduction_to_Python_Programming / english"


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(round(p * (len(ys) - 1))))]


def kind_of(content_type: list[str]) -> str:
    ct = set(content_type or [])
    return "table" if "Table" in ct else "figure" if ct & {"Infographic", "Image"} else "text"


def judge_summary(rows: list[dict], fallback_model: str | None) -> dict:
    """Judge block of a run summary: the four 1-5 means and their overall, plus the 0-1 correctness and agreement kept from before."""
    judged = [r for r in rows if r.get("judge") and "correctness" in r["judge"]]
    scored = [r for r in judged if r["judge"].get("scores")]
    model = (judged[0]["judge"].get("model") if judged else None) or fallback_model
    pin, pout = price_of(model)
    return {"n": len(judged),
            "cost_usd": round(sum(r["judge"].get("tokens_in", 0) * pin / 1e6 + r["judge"].get("tokens_out", 0) * pout / 1e6 for r in judged), 4),
            "tokens_in": sum(r["judge"].get("tokens_in", 0) for r in judged), "tokens_out": sum(r["judge"].get("tokens_out", 0) for r in judged),
            "ms_mean": int(statistics.mean(r["judge"].get("ms", 0) for r in judged)) if judged else None,
            "mean_correctness": round(statistics.mean(r["judge"]["correctness"] for r in judged), 4) if judged else None,
            "agreement": round(sum(1 for r in judged if (r["judge"]["correctness"] >= 0.5) == r["passed"]) / len(judged), 4) if judged else None,
            "model": model,
            "dims": {d: round(statistics.mean(r["judge"]["scores"][d] for r in scored), 3) for d in DIMENSIONS} if scored else None,
            "mean_overall": round(statistics.mean(r["judge"]["overall"] for r in scored), 3) if scored else None}


def summarise(rows: list[dict], use_judge: bool, wall_s: int) -> dict:
    n = len(rows)
    checks = sorted({k for r in rows for k in r["checks"]})

    def group(key):
        g: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            for k in (r[key] if isinstance(r[key], list) else [r[key]]) or ["other"]:
                g[k].append(r)
        return {k: {"n": len(v), "pass_rate": round(sum(x["passed"] for x in v) / len(v), 4), "mean_score": round(statistics.mean(x["score"] for x in v), 4)} for k, v in sorted(g.items())}

    judged = [r for r in rows if r.get("judge") and "correctness" in r["judge"]]
    return {
        "n_queries": n, "pass_rate": round(sum(r["passed"] for r in rows) / max(1, n), 4), "mean_score": round(statistics.mean(r["score"] for r in rows), 4) if rows else 0,
        "refusals": sum(r["refused"] for r in rows), "errors": sum(1 for r in rows if r.get("error")),
        "checks": {c: {"applicable": sum(1 for r in rows if r["checks"].get(c) is not None),
                       "mean": round(statistics.mean(r["checks"][c] for r in rows if r["checks"].get(c) is not None), 4) if any(r["checks"].get(c) is not None for r in rows) else None,
                       "pass_rate": round(sum(1 for r in rows if (r["checks"].get(c) or 0) >= 0.999) / max(1, sum(1 for r in rows if r["checks"].get(c) is not None)), 4)} for c in checks},
        "by_query_type": group("query_types"), "by_kind": group("kind"), "by_format": group("query_format"),
        "judge": judge_summary(rows, None) if (use_judge or judged) else None,
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 4), "tokens_in": sum(r["tokens_in"] for r in rows), "tokens_out": sum(r["tokens_out"] for r in rows),
        "latency_ms": {k: {"mean": int(statistics.mean(v)), "p50": int(pct(v, 0.5)), "p95": int(pct(v, 0.95))} for k, v in
                       {"retrieval": [r["ms"]["retrieval"] for r in rows], "answer": [r["ms"]["answer"] for r in rows], "total": [r["ms"]["total"] for r in rows]}.items() if v},
        "wall_s": wall_s,
    }


def rescore_existing(run_id: str) -> None:
    """Recompute the deterministic checks for a stored run from its stored answers (no retrieval, no generation, no cost)."""
    from pipeline.stores import sb_get

    row = sb_get("answer_eval_runs", run_id)
    if not row:
        sys.exit(f"unknown run {run_id}")
    rows = row.get("per_query") or []
    before = sum(r["passed"] for r in rows)
    for r in rows:
        rel = sorted({p for p, _ in r.get("relevant", [])})
        chk = run_checks(r.get("answer") or "", r.get("cited") or [], r.get("shown") or [], rel, [r["reference"]] + list(r.get("references") or []), r.get("query_types") or [])
        r.update(chk.as_dict())
    summary = summarise(rows, use_judge=bool(row.get("judge_model")), wall_s=int((row.get("summary") or {}).get("wall_s") or 0))
    sb_upsert("answer_eval_runs", {**row, "per_query": rows, "summary": summary, "rubric": rubric()})
    print(f"{run_id}: rescored {len(rows)} answers · pass {before} → {sum(r['passed'] for r in rows)} · mean {summary['mean_score']:.3f} · refusals {summary['refusals']} · grounded {summary['checks']['grounded']['pass_rate']}", flush=True)


def judge_existing(run_id: str, workers: int = 4) -> None:
    """Second pass: score a stored run's answers with the frozen judge and update the row's per_query and summary."""
    from pipeline.judge import judge_answer
    from pipeline.stores import sb_get

    s = settings()
    if not s.openai_api_key:
        sys.exit("OPENAI_API_KEY is not set")
    row = sb_get("answer_eval_runs", run_id)
    if not row:
        sys.exit(f"unknown run {run_id}")
    rows = row.get("per_query") or []
    print(f"{run_id}: judging {len(rows)} stored answers with {s.judge_model}", flush=True)
    from pipeline.stores import page_texts
    from scripts.ingest_dataset import DOC_ID

    def one(r: dict) -> dict:
        if not r.get("answer"):
            return r
        shown = [int(p) for p in (r.get("shown") or [])]
        texts = page_texts([f"{DOC_ID}:{p}" for p in shown])
        ctx = "\n\n".join(f"[{i + 1}] page {p}\n{(texts.get(f'{DOC_ID}:{p}') or '')[:4000]}" for i, p in enumerate(shown))
        try:
            r["judge"] = judge_answer(r["query"], r["reference"], r["answer"], ctx)
        except Exception as e:  # noqa: BLE001
            r["judge"] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
        return r

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(one, rows))
    summary = dict(row.get("summary") or {})
    summary["judge"] = judge_summary(rows, s.judge_model)
    sb_upsert("answer_eval_runs", {**row, "per_query": rows, "summary": summary, "judge_model": s.judge_model})
    print(f"judge overall {summary['judge']['mean_overall']} / 5  dims {summary['judge']['dims']}  agreement with checks {summary['judge']['agreement']}  ({summary['judge']['n']} judged)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="")
    ap.add_argument("--judge-run", default="", help="id of a stored answer_eval_runs row: judge its answers in place, no retrieval or generation")
    ap.add_argument("--rescore-run", default="", help="id of a stored run: recompute the deterministic checks from the stored answers (after a rubric change)")
    ap.add_argument("--config", default=IMPROVEMENT)
    ap.add_argument("--limit", type=int, default=0, help="first N queries by query_id")
    ap.add_argument("--sample", type=int, default=0, help="a seeded random subset of N queries (the same N for every config)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-pages", type=int, default=3)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--hardware", default="")
    ap.add_argument("--dry-run", action="store_true", help="score and print, do not write the answer_eval_runs row")
    args = ap.parse_args()
    if args.judge_run:
        return judge_existing(args.judge_run, args.workers)
    if args.rescore_run:
        return rescore_existing(args.rescore_run)
    if not args.root:
        sys.exit("root (the dataset folder) is required unless --judge-run is given")
    if args.config not in CONFIGS:
        sys.exit(f"unknown config {args.config}; one of {', '.join(CONFIGS)}")
    from pipeline.answer import stream_answer
    from pipeline.judge import judge_answer
    from pipeline.parsing import resolve_device

    s = settings()
    root = Path(args.root)
    corpus = pd.read_parquet(root / "corpus.parquet")
    page_of = {int(r.corpus_id): int(r.page_number_in_doc) for r in corpus.itertuples()}
    queries = pd.read_parquet(root / "queries.parquet")
    queries = queries[queries.language == "english"].sort_values("query_id")
    if args.limit:
        queries = queries.head(args.limit)
    if args.sample:
        queries = queries.sample(n=min(args.sample, len(queries)), random_state=args.seed).sort_values("query_id")
    qrels_df = pd.read_parquet(root / "qrels.parquet")
    qrels_df = qrels_df[qrels_df.query_id.isin(queries.query_id)]
    relevant: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in qrels_df.itertuples():
        relevant[int(r.query_id)].append((page_of[int(r.corpus_id)], int(r.score)))
    dev = resolve_device()
    hardware = args.hardware or {"cuda": "Modal GPU", "mps": "local Apple GPU", "cpu": "local CPU"}[dev]
    use_judge = not args.no_judge and bool(s.openai_api_key)
    run_id = f"aeval_{args.config}_{uuid.uuid4().hex[:6]}"
    lf = langfuse()
    encode_lock = threading.Lock()      # one retrieval at a time: concurrent forward passes crash on MPS; the answers still overlap
    # Load the models once on the main thread before the pool starts: a first load from several threads at once crashes torch.
    from pipeline.embed_text import text_embedder
    text_embedder()
    if CONFIGS[args.config][0]:
        from pipeline.embed_visual import visual_embedder
        visual_embedder()
    if CONFIGS[args.config][3]:
        from pipeline.retrieval import reranker
        reranker()
    print(f"{run_id}: {len(queries)} queries · config {args.config} · answers {s.answer_model} · judge {'on (' + s.judge_model + ')' if use_judge else 'off'} · {hardware}", flush=True)

    def one(q) -> dict:
        qid = int(q.query_id)
        qtypes = [str(t) for t in (q.query_types if q.query_types is not None else [])]
        ctype = [str(t) for t in (q.content_type if q.content_type is not None else [])]
        refs = [str(q.answer)] + [str(a) for a in (q.raw_answers if q.raw_answers is not None else []) if str(a) != str(q.answer)]
        rel_pages = sorted({p for p, _ in relevant.get(qid, [])})
        trace = start_trace("answer_eval", input={"question": q.query, "workspace": WORKSPACE, "config": args.config},
                            metadata={"eval_run": run_id, "query_id": qid, "query_types": qtypes, "source": "answer_eval"})
        t0 = time.perf_counter()
        with encode_lock:
            res = retrieve(q.query, WORKSPACE, args.config, k=10, trace=trace)
        t_ret = int((time.perf_counter() - t0) * 1000)
        shown = [e.page_no for e in res.evidence[: args.n_pages]]
        text, final = "", None
        err = None
        try:
            for kind, payload in stream_answer(q.query, res.evidence, n_pages=args.n_pages, trace=trace):
                if kind == "token":
                    text += payload
                else:
                    final = payload
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        text = (final.text if final and final.text else text).strip()
        cited = [res.evidence[i].page_no for i in (final.cited if final else []) if i < len(res.evidence)]
        chk = run_checks(text, cited, shown, rel_pages, refs, qtypes)
        jd = None
        if use_judge and text:
            ctx = "\n\n".join(f"[{i + 1}] page {e.page_no}\n{(e.page_text or '')[:4000]}" for i, e in enumerate(res.evidence[: args.n_pages]))
            try:
                jd = judge_answer(q.query, str(q.answer), text, ctx)
            except Exception as e:  # noqa: BLE001
                jd = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
        total_ms = int((time.perf_counter() - t0) * 1000)
        end(trace, output={"answer": text[:2000], "score": chk.score, "passed": chk.passed, "cited_pages": cited, "error": err},
            metadata={"timings_ms": res.timings_ms, "retrieval_ms": t_ret, "ttr_ms": total_ms, "cost_usd": final.cost_usd if final else 0})
        tid = trace_id_of(trace)
        if lf is not None and tid:
            try:
                lf.create_score(trace_id=tid, name="answer_check", value=float(chk.score), comment="deterministic rubric" + (" · refused" if chk.refused else ""))
                if jd and "correctness" in jd:
                    lf.create_score(trace_id=tid, name="judge_correctness", value=float(jd["correctness"]), comment=jd.get("reason"))
                    for dim, val in (jd.get("scores") or {}).items():
                        lf.create_score(trace_id=tid, name=f"judge_{dim}", value=float(val))
                    if jd.get("overall") is not None:
                        lf.create_score(trace_id=tid, name="judge_overall", value=float(jd["overall"]))
            except Exception:  # noqa: BLE001
                pass
        return {
            "query_id": qid, "query": q.query, "query_types": qtypes, "query_format": str(q.query_format), "content_type": ctype, "kind": kind_of(ctype),
            "reference": str(q.answer), "references": refs[1:], "relevant": sorted(relevant.get(qid, [])), "retrieved": [e.page_no for e in res.evidence],
            "shown": shown, "cited": cited, "answer": text, "error": err, **chk.as_dict(), "judge": jd,
            "ms": {"retrieval": t_ret, "answer": final.ttr_ms if final else 0, "total": total_ms}, "timings_ms": res.timings_ms,
            "tokens_in": final.tokens_in if final else 0, "tokens_out": final.tokens_out if final else 0, "cost_usd": final.cost_usd if final else 0.0,
            "model": final.model if final else None, "trace_id": tid,
        }

    t_run = time.time()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, row in enumerate(pool.map(one, list(queries.itertuples())), start=1):
            rows.append(row)
            if i % 5 == 0 or i == len(queries):
                print(f"[{args.config}] {i}/{len(queries)} · last score {row['score']:.2f} {'pass' if row['passed'] else 'FAIL'} · {row['ms']['total']} ms", flush=True)
    flush()

    summary = summarise(rows, use_judge=use_judge, wall_s=int(time.time() - t_run))
    if args.sample:
        summary["sample"] = {"n": len(rows), "seed": args.seed, "query_ids": [r["query_id"] for r in rows]}
    row = {"id": run_id, "workspace": WORKSPACE, "dataset": DATASET, "config": args.config, "answer_model": s.answer_model, "judge_model": s.judge_model if use_judge else None,
           "n_pages": args.n_pages, "rubric": rubric(), "summary": summary, "per_query": rows, "hardware": hardware, "created_at": now_iso()}
    if args.dry_run:
        import json
        out = Path("/tmp") / f"{run_id}.json"
        try:
            out = Path(__import__("os").environ.get("MATT_SCRATCH", "/tmp")) / f"{run_id}.json"
            out.write_text(json.dumps(row, indent=1, default=str))
            print(f"dry run: wrote {out}", flush=True)
        except OSError:
            pass
    else:
        sb_upsert("answer_eval_runs", row)
    print(f"{args.config:>12}  pass {summary['pass_rate']:.3f}  mean score {summary['mean_score']:.3f}  grounded {summary['checks'].get('grounded', {}).get('pass_rate')}  "
          f"judge {summary['judge']['mean_overall'] if summary['judge'] else '–'}/5 {summary['judge']['dims'] if summary['judge'] else ''}  ${summary['cost_usd']:.2f}  TTR p50 {summary['latency_ms']['total']['p50']} ms  ({summary['wall_s']}s, {hardware})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
