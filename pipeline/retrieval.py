"""Steps 6-8: retrieve (visual + text), fuse (RRF), rerank. Also the "explain" view with every candidate list."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

from pipeline.config import settings
from pipeline.embed_text import text_embedder
from pipeline.stores import chunks_by_ids, page_texts, pages_by_ids, search_text, search_visual
from pipeline.tracing import child, end

CONFIGS: dict[str, tuple[bool, bool, bool, bool]] = {   # visual, dense, bm25, rerank
    "bm25": (False, False, True, False),
    "text_hybrid": (False, True, True, False),
    "text_hybrid_rerank": (False, True, True, True),
    "visual": (True, False, False, False),
    "fused": (True, True, True, False),
    "fused_rerank": (True, True, True, True),
}
# The two configs the writeup, the ViDoRe Pipeline tab and the eval tabs compare, judged on answers (deterministic checks +
# LLM judge over the same 30 benchmark queries). The page-image path (`fused`) was implemented and measured too: better
# retrieval ranks, a tie on answers, so it stays the Playground default for slides and scans, not the dataset improvement.
BASELINE = "text_hybrid"          # standard text RAG: Docling chunks, bge-m3 dense + BM25 sparse, RRF
IMPROVEMENT = "text_hybrid_rerank"  # + bge-reranker-v2-m3 cross-encoder over the text-hybrid top 20
DATASET_DEFAULT = IMPROVEMENT     # what the Dataset tab and the answer evals run
PLAYGROUND_DEFAULT = "fused"      # uploads are slides and scans; the page-image path earns its cost there
CONFIG_INFO: dict[str, dict] = {
    "bm25": {"label": "BM25 only", "detail": "sparse exact-term match over chunks"},
    "text_hybrid": {"label": "Hybrid RAG without reranker", "detail": "bge-m3 dense + BM25 sparse over Docling chunks, RRF inside Qdrant"},
    "text_hybrid_rerank": {"label": "Hybrid RAG with reranker", "detail": "text-hybrid top 20 re-scored by bge-reranker-v2-m3 cross-encoder, top 5 shown"},
    "visual": {"label": "Visual only (ColQwen)", "detail": "ColQwen2.5 MaxSim over page images"},
    "fused": {"label": "Fused (visual + text)", "detail": "RRF of the ColQwen page list and the text-hybrid list"},
    "fused_rerank": {"label": "Fused + rerank", "detail": "fused top 20 re-scored by bge-reranker-v2-m3"},
}
for _name, (_v, _d, _b, _r) in CONFIGS.items():
    CONFIG_INFO[_name]["stages"] = {"visual": _v, "dense": _d, "bm25": _b, "rerank": _r}
    CONFIG_INFO[_name]["role"] = "baseline" if _name == BASELINE else "improvement" if _name == IMPROVEMENT else None

StepFn = Callable[[str, str, str, int | None], None]   # name, detail, state, ms


@dataclass
class Evidence:
    page_id: str
    document_id: str
    doc_name: str
    page_no: int
    image_url: str | None
    score: float
    rank: int
    visual_rank: int | None = None
    visual_score: float | None = None
    text_rank: int | None = None
    text_score: float | None = None
    rerank_score: float | None = None
    best_chunk: dict | None = None
    page_text: str = ""

    def to_chunk(self, index: int) -> dict:
        """The evidence chunk shape the UI renders."""
        bc = self.best_chunk or {}
        return {
            "id": bc.get("id") or self.page_id, "doc": self.doc_name, "page": self.page_no, "pno": self.page_no,
            "kind": bc.get("kind", "text"), "boxes": [bc["bbox"]] if bc.get("bbox") else [], "image": self.image_url or "",
            "snippet": (bc.get("text") or self.page_text or "")[:700], "highlight": None,
            "scores": {"visualRank": self.visual_rank, "visualScore": self.visual_score, "textRank": self.text_rank, "textScore": self.text_score, "fusedRank": self.rank, "rerank": self.rerank_score,
                       "final": round(float(self.score), 4)},   # the score the final ranking sorted by (rerank score, or fused score)
        }


@dataclass
class RetrievalResult:
    evidence: list[Evidence]
    timings_ms: dict[str, int] = field(default_factory=dict)
    paths: dict[str, list] = field(default_factory=dict)


def rrf(lists: list[list[tuple[str, float]]], k: int) -> list[tuple[str, float, dict[int, int]]]:
    score: dict[str, float] = defaultdict(float)
    where: dict[str, dict[int, int]] = defaultdict(dict)
    for li, lst in enumerate(lists):
        for rank, (pid, _) in enumerate(lst):
            score[pid] += 1.0 / (k + rank + 1)
            where[pid][li] = rank + 1
    return sorted(((pid, s, where[pid]) for pid, s in score.items()), key=lambda x: -x[1])


@lru_cache
def reranker():
    from sentence_transformers import CrossEncoder

    from pipeline.parsing import resolve_device

    dev = resolve_device()
    return CrossEncoder(settings().rerank_model, max_length=1024, device=dev if dev != "mps" else "cpu")


def _no_step(*_a) -> None:
    return None


def retrieve(question: str, workspace: str, config: str = "fused_rerank", k: int | None = None, on_step: StepFn = _no_step, trace=None) -> RetrievalResult:
    s = settings()
    visual_on, dense_on, bm25_on, rerank_on = CONFIGS[config]
    visual_on = visual_on and s.visual_enabled
    k = k or s.final_k
    timings: dict[str, int] = {}
    te = text_embedder()

    def ranked(pairs, n: int = 20) -> list[dict]:
        """A stage's ranked pages for its trace span: rank, page number, score (page ids are `<doc>:<page>`)."""
        return [{"rank": r, "page": int(pid.rsplit(":", 1)[-1]) if pid.rsplit(":", 1)[-1].isdigit() else None, "page_id": pid, "score": round(float(sc), 4)}
                for r, (pid, sc, *_) in enumerate(list(pairs)[:n], start=1)]

    def timed(name: str, detail: str, fn, summary=None):
        """Runs one stage under its own span. `summary(out)` turns the stage's result into the span output (its ranked
        pages), so a trace shows where a correct page entered or dropped out of the list."""
        on_step(name, detail, "active", None)
        t0 = time.perf_counter()
        obs = child(trace, name, input={"detail": detail})
        out, ok = None, False
        try:
            out = fn()
            ok = True
        finally:
            ms = int((time.perf_counter() - t0) * 1000)
            span_out = None
            if summary is not None and ok and obs is not None:
                try:
                    span_out = summary(out)
                except Exception:  # noqa: BLE001  tracing must never break retrieval
                    span_out = None
            end(obs, output=span_out, metadata={"ms": ms})
        timings[name] = ms
        on_step(name, detail, "done", ms)
        return out

    # 1. encode
    enc: dict = {}
    def _encode():
        if dense_on:
            enc["dense"] = te.dense_query(question)
        if bm25_on:
            enc["sparse"] = te.sparse_query(question)
        if visual_on:
            from pipeline.embed_visual import visual_embedder
            enc["qv"], enc["qp"] = visual_embedder().query(question)
    timed("Encoding question", " · ".join(x for x, on in (("ColQwen2.5", visual_on), ("bge-m3", dense_on), ("BM25", bm25_on)) if on), _encode)

    # 2. visual path → pages
    visual_hits: list[tuple[str, float, dict]] = []
    if visual_on:
        visual_hits = timed("Visual search", f"{s.pages_collection} · MaxSim · top {s.top_k_per_path}", lambda: search_visual(enc["qv"], enc["qp"], workspace, s.top_k_per_path),
                            summary=lambda hits: {"pages": ranked(hits)})

    # 3. text path → chunks → pages
    text_pages: list[tuple[str, float]] = []
    best_chunk: dict[str, dict] = {}
    if dense_on or bm25_on:
        def _text():
            hits = search_text(enc.get("dense") if dense_on else None, enc.get("sparse") if bm25_on else None, workspace, 4 * s.top_k_per_path)
            for cid, score, payload in hits:
                pid = payload["page_id"]
                if pid not in best_chunk:
                    best_chunk[pid] = {"id": cid, "score": score, "payload": payload}
                    text_pages.append((pid, score))
                if len(text_pages) >= s.top_k_per_path:
                    break
        timed("Text search", f"{s.chunks_collection} · {' + '.join(x for x, on in (('dense', dense_on), ('BM25', bm25_on)) if on)} · top {s.top_k_per_path}", _text,
              summary=lambda _: {"pages": [dict(row, chunk=best_chunk[pid]["id"]) for row, (pid, _sc) in zip(ranked(text_pages), text_pages)]})

    # 4. fuse
    lists = [l for l in ([(p, sc) for p, sc, _ in visual_hits], text_pages) if l]
    def _fuse():
        if len(lists) > 1:
            return rrf(lists, s.rrf_k)
        return [(pid, sc, {0: r + 1}) for r, (pid, sc) in enumerate(lists[0])] if lists else []
    fused = timed("Fusing", f"reciprocal rank fusion → {s.fused_k} pages", _fuse, summary=lambda out: {"pages": ranked(out, s.fused_k)})[: s.fused_k]

    vis_rank = {pid: (r + 1, sc) for r, (pid, sc, _) in enumerate(visual_hits)}
    txt_rank = {pid: (r + 1, sc) for r, (pid, sc) in enumerate(text_pages)}
    page_ids = [pid for pid, _, _ in fused]

    # 5. rerank
    texts = page_texts(page_ids) if rerank_on else {}
    order: list[tuple[str, float]] = [(pid, sc) for pid, sc, _ in fused]
    rr_score: dict[str, float] = {}
    if rerank_on and page_ids:
        def _rerank():
            scores = reranker().predict([(question, (texts.get(pid) or "")[:4000]) for pid in page_ids])
            for pid, sc in zip(page_ids, scores):
                rr_score[pid] = float(sc)
            return sorted(((pid, rr_score[pid]) for pid in page_ids), key=lambda x: -x[1])
        order = timed("Reranking", f"{s.rerank_model.split('/')[-1]} → top {k}", _rerank, summary=lambda out: {"pages": ranked(out), "kept": k})

    # 6. evidence records
    top = order[:k]
    meta = pages_by_ids([pid for pid, _ in top])
    chunk_rows = chunks_by_ids([best_chunk[pid]["id"] for pid, _ in top if pid in best_chunk])
    evidence: list[Evidence] = []
    for rank, (pid, sc) in enumerate(top, start=1):
        m = meta.get(pid, {})
        bc = chunk_rows.get(best_chunk[pid]["id"]) if pid in best_chunk else None
        evidence.append(Evidence(
            page_id=pid, document_id=m.get("document_id", pid.split(":")[0]), doc_name=m.get("doc_name") or m.get("document_id", ""), page_no=int(m.get("page_no", 0)),
            image_url=m.get("image_url"), score=float(sc), rank=rank,
            visual_rank=vis_rank.get(pid, (None, None))[0], visual_score=vis_rank.get(pid, (None, None))[1],
            text_rank=txt_rank.get(pid, (None, None))[0], text_score=txt_rank.get(pid, (None, None))[1],
            rerank_score=rr_score.get(pid), best_chunk=bc, page_text=m.get("text") or "",
        ))
    # document names
    from pipeline.stores import sb_select
    names = {d["id"]: d["name"] for d in sb_select("documents", workspace=workspace)}
    for e in evidence:
        e.doc_name = names.get(e.document_id, e.doc_name)
    return RetrievalResult(evidence=evidence, timings_ms=timings,
                           paths={"visual": [(p, sc) for p, sc, _ in visual_hits[:10]], "text": text_pages[:10], "fused": [(p, sc) for p, sc, _ in fused[:10]]})


def explain(question: str, workspace: str, k: int = 5) -> dict:
    """Every candidate list with scores, for the Index tab's Explain view."""
    s = settings()
    te = text_embedder()
    t0 = time.perf_counter()
    timings: dict[str, int] = {}

    def lap(name: str) -> None:
        nonlocal t0
        timings[name] = int((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()

    dense_q, sparse_q = te.dense_query(question), te.sparse_query(question)
    qv = qp = None
    if s.visual_enabled:
        from pipeline.embed_visual import visual_embedder
        qv, qp = visual_embedder().query(question)
    lap("encode")
    visual = search_visual(qv, qp, workspace, 10) if qv is not None else []
    lap("visual")
    dense_hits = search_text(dense_q, None, workspace, 40); lap("dense")
    bm25_hits = search_text(None, sparse_q, workspace, 40); lap("bm25")

    def pages_of(hits):
        seen, out = set(), []
        for cid, sc, payload in hits:
            pid = payload["page_id"]
            if pid in seen:
                continue
            seen.add(pid); out.append({"id": cid, "page": pid, "pno": payload.get("page_no"), "score": round(sc, 4), "chunk": cid, "preview": (payload.get("preview") or "")[:160]})
            if len(out) >= 10:
                break
        return out

    dense_pages, bm25_pages = pages_of(dense_hits), pages_of(bm25_hits)
    visual_pages = [{"id": pid, "page": pid, "pno": payload.get("page_no"), "score": round(sc, 2)} for pid, sc, payload in visual]
    text_hybrid = rrf([[(h["page"], h["score"]) for h in dense_pages], [(h["page"], h["score"]) for h in bm25_pages]], s.rrf_k)[:10]
    fused = rrf([[(h["page"], h["score"]) for h in visual_pages], [(pid, sc) for pid, sc, _ in text_hybrid]], s.rrf_k)[:10]
    lap("fusion")
    texts = page_texts([pid for pid, _, _ in fused])
    rr = reranker().predict([(question, (texts.get(pid) or "")[:4000]) for pid, _, _ in fused]) if fused else []
    lap("rerank")
    pno = {h["page"]: h["pno"] for h in dense_pages + bm25_pages + visual_pages}
    preview = {h["page"]: h.get("preview") for h in dense_pages + bm25_pages}
    reranked = sorted([{"page": pid, "pno": pno.get(pid), "score": round(sc, 4), "from": {("visual" if i == 0 else "text"): r for i, r in where.items()}, "preview": preview.get(pid), "rerank": round(float(r), 4)} for (pid, sc, where), r in zip(fused, rr)], key=lambda x: -x["rerank"])
    return {
        "question": question, "terms": question.lower().split(), "k": k,
        "timings_ms": {**timings, "total": sum(timings.values())},
        "paths": {"visual": visual_pages, "dense": dense_pages, "bm25": bm25_pages},
        "text_hybrid": [{"page": pid, "pno": pno.get(pid), "score": round(sc, 4), "from": {("dense" if i == 0 else "bm25"): r for i, r in where.items()}, "preview": preview.get(pid)} for pid, sc, where in text_hybrid],
        "fused": [{"page": pid, "pno": pno.get(pid), "score": round(sc, 4), "from": {("visual" if i == 0 else "text"): r for i, r in where.items()}, "preview": preview.get(pid)} for pid, sc, where in fused],
        "reranked": reranked, "final": reranked[:k],
    }
