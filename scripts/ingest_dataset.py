"""Load the preloaded dataset (ViDoRe V3 Python textbook subset) into the hosted stores for the Dataset tab.

Uses the OCR markdown shipped with the benchmark for the text path (no OCR needed), uploads page images to Supabase
Storage, and embeds pages with ColQwen when VISUAL_ENABLED. Resumable: pages already present are skipped.

Run:  uv run python scripts/ingest_dataset.py "raw dataset v3/python_textbook" [--limit N] [--no-visual]
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

from pipeline.chunking import chunk_markdown
from pipeline.config import settings
from pipeline.embed_text import text_embedder
from pipeline.ingest import now_iso
from pipeline.parsing import to_display_copy
from pipeline.stores import ensure_collections, point_id, sb_get, sb_select, sb_upsert, sb_update, upload_page_image, upsert_chunks, upsert_pages

WORKSPACE = "python_textbook"
DOC_ID = "python_textbook"
DOC_NAME = "Introduction to Python Programming"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-visual", action="store_true")
    args = ap.parse_args()
    s = settings()
    root = Path(args.root)
    corpus = pd.read_parquet(root / "corpus.parquet").sort_values("page_number_in_doc")
    if args.limit:
        corpus = corpus.head(args.limit)
    visual_on = s.visual_enabled and not args.no_visual

    ensure_collections()
    sb_upsert("conversations", {"id": "dataset", "source": "dataset", "workspace": WORKSPACE, "mode": "fused_rerank", "answer_model": s.answer_model})
    if not sb_get("documents", DOC_ID):
        sb_upsert("documents", {"id": DOC_ID, "workspace": WORKSPACE, "name": DOC_NAME, "source": "vidore_v3", "file_type": "pdf", "pages": len(corpus), "status": "indexing", "created_at": now_iso()})
    done = {r["id"] for r in sb_select("pages", document_id=DOC_ID) if r.get("n_chunks")}
    te = text_embedder()
    ve = None
    if visual_on:
        from pipeline.embed_visual import visual_embedder
        ve = visual_embedder()

    total_chunks = 0
    t_start = time.time()
    for i, r in enumerate(corpus.itertuples(), start=1):
        pno = int(r.page_number_in_doc)
        pid = f"{DOC_ID}:{pno}"
        if pid in done:
            continue
        t0 = time.time()
        img = Image.open(root.parent / r.image_path if not Path(r.image_path).is_absolute() else r.image_path).convert("RGB")
        md = r.markdown or ""
        chunks = chunk_markdown(DOC_ID, DOC_NAME, pno, md, s.chunk_words, s.table_rows_per_part)
        url = upload_page_image(f"{WORKSPACE}/{DOC_ID}/{pno}.jpg", to_display_copy(img, s.page_image_width))
        n_vis = 0
        multi = pooled = None
        if ve is not None:
            multi, pooled = ve.pages([img])
            n_vis = int(multi[0].shape[0])
        # page row first: chunks reference it through a foreign key
        sb_upsert("pages", {"id": pid, "document_id": DOC_ID, "page_no": pno, "width": int(r.width), "height": int(r.height), "image_url": url, "text": md, "n_chunks": len(chunks), "n_visual_vectors": n_vis, "parser": "vidore_v3_markdown"})
        if chunks:
            dense = te.dense([c.embed_text for c in chunks]); sparse = te.sparse([c.text for c in chunks])
            sb_upsert("chunks", [{"id": c.id, "document_id": DOC_ID, "page_id": pid, "page_no": pno, "kind": c.kind, "heading_path": c.heading_path, "bbox": c.bbox, "text": c.text, "n_tokens": c.n_tokens, "vector_id": point_id(c.id)} for c in chunks])
            upsert_chunks([{"id": point_id(c.id), "dense": dense[j], "sparse": sparse[j], "payload": {"workspace": WORKSPACE, "document_id": DOC_ID, "page_id": pid, "page_no": pno, "chunk_id": c.id, "kind": c.kind, "preview": c.text[:160]}} for j, c in enumerate(chunks)])
        if multi is not None:
            upsert_pages([{"id": point_id(pid), "multivector": multi[0], "pooled": pooled[0], "payload": {"workspace": WORKSPACE, "document_id": DOC_ID, "page_id": pid, "page_no": pno, "doc_name": DOC_NAME}}])
        total_chunks += len(chunks)
        print(f"[{i}/{len(corpus)}] p.{pno}: {len(chunks)} chunks, {n_vis} visual vectors, {time.time() - t0:.1f}s", flush=True)
    n = sum(r.get("n_chunks") or 0 for r in sb_select("pages", document_id=DOC_ID))
    sb_update("documents", DOC_ID, {"status": "indexed", "chunks": n, "pages": len(corpus), "indexed_at": now_iso()})
    print(f"done: {len(corpus)} pages, {n} chunks total, {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
