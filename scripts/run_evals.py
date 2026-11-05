"""Retrieval evaluation on the ViDoRe V3 Python textbook (English queries) for each retrieval config.

Scores every config against the benchmark's graded page labels (ranx: nDCG@5, Recall@5/10, MRR@10), records per-stage
latency, and writes one `eval_runs` row per config (metrics + per-query detail) so the Evals tab can show the comparison
and the failures.

Run:  uv run python scripts/run_evals.py "raw dataset v3/python_textbook" [--configs bm25,text_hybrid,visual,fused,fused_rerank] [--limit N] [--k 10]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import pandas as pd

from pipeline.config import settings
from pipeline.ingest import now_iso
from pipeline.retrieval import CONFIGS, retrieve
from pipeline.stores import sb_upsert

WORKSPACE = "python_textbook"
DATASET = "vidore_v3_computer_science / Introduction_to_Python_Programming / english"


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(round(p * (len(ys) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--hardware", default="")
    args = ap.parse_args()
    from ranx import Qrels, Run, evaluate

    from pipeline.parsing import resolve_device

    root = Path(args.root)
    s = settings()
    corpus = pd.read_parquet(root / "corpus.parquet")
    page_of = {int(r.corpus_id): int(r.page_number_in_doc) for r in corpus.itertuples()}
    queries = pd.read_parquet(root / "queries.parquet")
    queries = queries[queries.language == "english"].sort_values("query_id")
    if args.limit:
        queries = queries.head(args.limit)
    qrels_df = pd.read_parquet(root / "qrels.parquet")
    qrels_df = qrels_df[qrels_df.query_id.isin(queries.query_id)]
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for r in qrels_df.itertuples():
        qrels[str(r.query_id)][str(page_of[int(r.corpus_id)])] = int(r.score)
    dev = resolve_device()
    hardware = args.hardware or {"cuda": "Modal GPU", "mps": "local Apple GPU", "cpu": "local CPU"}[dev]
    configs = [c for c in args.configs.split(",") if c in CONFIGS]

    for config in configs:
        needs_visual = CONFIGS[config][0]
        if needs_visual and not s.visual_enabled:
            print(f"skip {config}: visual path disabled", flush=True)
            continue
        run: dict[str, dict[str, float]] = {}
        per_query: list[dict] = []
        stage_ms: dict[str, list[int]] = defaultdict(list)
        totals: list[int] = []
        t_cfg = time.time()
        for i, q in enumerate(queries.itertuples(), start=1):
            qid = str(q.query_id)
            t0 = time.perf_counter()
            res = retrieve(q.query, WORKSPACE, config, k=args.k)
            ms = int((time.perf_counter() - t0) * 1000)
            totals.append(ms)
            for name, v in res.timings_ms.items():
                stage_ms[name].append(v)
            ranked = [e.page_no for e in res.evidence]
            run[qid] = {str(p): float(args.k - r) for r, p in enumerate(ranked)}       # rank order only; ranx needs descending scores
            rel = qrels.get(qid, {})
            first_rel = next((r + 1 for r, p in enumerate(ranked) if str(p) in rel), None)
            per_query.append({
                "query_id": int(q.query_id), "query": q.query, "answer": q.answer, "content_type": list(q.content_type) if q.content_type is not None else [],
                "relevant": sorted((int(p), sc) for p, sc in rel.items()), "retrieved": ranked,
                "first_relevant_rank": first_rel, "hit@5": bool(first_rel and first_rel <= 5), "ms": ms, "timings_ms": res.timings_ms,
                "top": [{"page": e.page_no, "score": round(e.score, 4), "visual_rank": e.visual_rank, "text_rank": e.text_rank, "rerank": e.rerank_score} for e in res.evidence[:5]],
            })
            if i % 10 == 0 or i == len(queries):
                print(f"[{config}] {i}/{len(queries)} · {ms} ms", flush=True)
        metrics = evaluate(Qrels(dict(qrels)), Run(run), ["ndcg@5", "ndcg@10", "recall@5", "recall@10", "mrr@10", "precision@5"])
        metrics = {k: round(float(v), 4) for k, v in metrics.items()}
        metrics["hit@5"] = round(sum(p["hit@5"] for p in per_query) / max(1, len(per_query)), 4)
        metrics["n_queries"] = len(per_query)
        metrics["k"] = args.k
        metrics["latency_ms"] = {"total": {"mean": int(statistics.mean(totals)), "p50": int(pct(totals, 0.5)), "p95": int(pct(totals, 0.95))},
                                 **{n: {"mean": int(statistics.mean(v)), "p50": int(pct(v, 0.5)), "p95": int(pct(v, 0.95))} for n, v in stage_ms.items()}}
        row = {"id": f"eval_{config}_{uuid.uuid4().hex[:6]}", "workspace": WORKSPACE, "dataset": DATASET, "config": config, "metrics": metrics,
               "per_query": per_query, "hardware": hardware, "created_at": now_iso()}
        sb_upsert("eval_runs", row)
        print(f"{config:>13}  nDCG@5 {metrics['ndcg@5']:.3f}  Recall@5 {metrics['recall@5']:.3f}  MRR@10 {metrics['mrr@10']:.3f}  hit@5 {metrics['hit@5']:.3f}  "
              f"mean {metrics['latency_ms']['total']['mean']} ms  ({time.time() - t_cfg:.0f}s, {hardware})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
