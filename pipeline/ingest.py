"""Ingestion job: route → parse → chunk → embed text → embed pages → index. State lives in Supabase (jobs table),
page images in Supabase Storage, vectors in Qdrant. Idempotent per document id."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from pipeline.chunking import Chunk, chunk_markdown, chunk_page
from pipeline.config import settings
from pipeline.embed_text import text_embedder
from pipeline.parsing import ParsedDocument, parse_office, parse_text, parse_with_docling, to_display_copy
from pipeline.routing import detect
from pipeline.stores import delete_document_vectors, ensure_collections, point_id, sb_delete_document, sb_update, sb_upsert, upload_page_image, upsert_chunks, upsert_pages

STEPS = [("Router", "file type from magic bytes"), ("Parse", "Docling · layout · OCR · tables"), ("Chunk", "sections · tables · figures"),
         ("Embed text", "bge-m3 + BM25"), ("Embed pages", "ColQwen2.5"), ("Index", "qdrant + page store")]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Job:
    def __init__(self, job_id: str, workspace: str, files: list[tuple[str, bytes]]):
        self.id, self.workspace, self.files = job_id, workspace, files
        self.steps = [{"name": n, "detail": d, "state": "pending"} for n, d in STEPS]
        self.file_rows = [{"name": n, "size": len(b), "type": (n.rsplit(".", 1)[-1].upper()[:4] if "." in n else "FILE")} for n, b in files]
        self.t0 = time.perf_counter()
        sb_upsert("jobs", {"id": job_id, "workspace": workspace, "status": "running", "files": self.file_rows, "steps": self.steps, "progress": 0})

    def step(self, i: int, state: str, detail: str | None = None, ms: int | None = None) -> None:
        st = self.steps[i]
        st["state"] = state
        if detail:
            st["detail"] = detail
        if ms is not None:
            st["ms"] = ms
        done = sum(1 for s in self.steps if s["state"] == "done")
        sb_update("jobs", self.id, {"steps": self.steps, "progress": done / len(self.steps)})

    def finish(self, error: str | None = None) -> None:
        sb_update("jobs", self.id, {"status": "failed" if error else "done", "error": error, "files": self.file_rows, "steps": self.steps, "progress": 1.0 if not error else None, "finished_at": now_iso()})


def _timed(job: Job, i: int, fn, detail: str | None = None):
    job.step(i, "active")
    t0 = time.perf_counter()
    out = fn()
    job.step(i, "done", detail, int((time.perf_counter() - t0) * 1000))
    return out


def run_ingest(job_id: str, workspace: str, files: list[tuple[str, bytes]], shipped_markdown: dict[str, dict[int, str]] | None = None) -> None:
    """shipped_markdown: optional {file_name: {page_no: markdown}} to skip OCR (ViDoRe V3 ships OCR text)."""
    s = settings()
    job = Job(job_id, workspace, files)
    try:
        ensure_collections()
        # 1. route
        kinds = _timed(job, 0, lambda: [detect(n, b) for n, b in files], " · ".join(f"{n} → {detect(n, b).kind}" for n, b in files))
        for (name, _), k in zip(files, kinds):
            if k.kind == "unsupported":
                raise ValueError(f"{name}: unsupported file type")

        # 2. parse
        docs: list[tuple[str, str, ParsedDocument]] = []   # (document_id, name, parsed)
        def _parse():
            for (name, data), k in zip(files, kinds):
                doc_id = f"doc_{uuid.uuid5(uuid.NAMESPACE_URL, workspace + '/' + name).hex[:10]}"
                title = name.rsplit(".", 1)[0]
                if k.kind in ("pdf", "image"):
                    parsed = parse_with_docling(name, data, title)
                elif k.kind in ("docx", "pptx"):
                    parsed = parse_office(name, k.kind, data, title)
                else:
                    parsed = parse_text(name, data, title, k.kind)
                docs.append((doc_id, name, parsed))
        _timed(job, 1, _parse, lambda_detail := None)
        job.step(1, "done", f"{sum(len(p.pages) for _, _, p in docs)} pages · " + ", ".join(sorted({pg.parser for _, _, p in docs for pg in p.pages})))

        # Replace any previous version of these documents.
        for doc_id, name, parsed in docs:
            delete_document_vectors(doc_id)
            sb_delete_document(doc_id)
            sb_upsert("documents", {"id": doc_id, "workspace": workspace, "name": name, "source": "upload", "file_type": name.rsplit(".", 1)[-1].lower() if "." in name else None,
                                    "pages": len(parsed.pages), "status": "indexing", "created_at": now_iso()})

        # 3. chunk
        all_chunks: list[Chunk] = []
        page_rows: list[dict] = []
        def _chunk():
            for doc_id, name, parsed in docs:
                md_pages = (shipped_markdown or {}).get(name, {})
                for pg in parsed.pages:
                    if pg.page_no in md_pages:
                        chunks = chunk_markdown(doc_id, parsed.title, pg.page_no, md_pages[pg.page_no], s.chunk_words, s.table_rows_per_part)
                        pg.text = md_pages[pg.page_no]
                    elif pg.elements:
                        chunks = chunk_page(doc_id, parsed.title, pg.page_no, pg.elements, s.chunk_words, s.table_rows_per_part)
                    else:
                        chunks = chunk_markdown(doc_id, parsed.title, pg.page_no, pg.text, s.chunk_words, s.table_rows_per_part)
                    all_chunks.extend(chunks)
                    page_rows.append({"id": f"{doc_id}:{pg.page_no}", "document_id": doc_id, "page_no": pg.page_no, "width": pg.width, "height": pg.height,
                                      "text": pg.text, "n_chunks": len(chunks), "parser": pg.parser})
        _timed(job, 2, _chunk, None)
        job.step(2, "done", f"{len(all_chunks)} chunks from {len(page_rows)} pages")

        # 4. text embeddings
        te = text_embedder()
        dense = _timed(job, 3, lambda: te.dense([c.embed_text for c in all_chunks]), f"bge-m3 + BM25 · {len(all_chunks)} chunks")
        sparse = te.sparse([c.text for c in all_chunks])

        # 5. page embeddings (visual path) + display copies
        visual: dict[str, tuple] = {}
        def _pages():
            images = [(f"{doc_id}:{pg.page_no}", pg.image) for doc_id, _, parsed in docs for pg in parsed.pages if pg.image is not None]
            if not images:
                return "no page images (text only)"
            # upload display copies
            for pid, img in images:
                url = upload_page_image(f"{workspace}/{pid.replace(':', '/')}.jpg", to_display_copy(img, s.page_image_width))
                for row in page_rows:
                    if row["id"] == pid:
                        row["image_url"] = url
            if not s.visual_enabled:
                return f"{len(images)} page images stored · ColQwen off"
            from pipeline.embed_visual import visual_embedder
            multi, pooled = visual_embedder().pages([img for _, img in images])
            for (pid, _), mv, pv in zip(images, multi, pooled):
                visual[pid] = (mv, pv)
                for row in page_rows:
                    if row["id"] == pid:
                        row["n_visual_vectors"] = int(mv.shape[0])
            return f"ColQwen2.5 · {len(images)} pages · ~{int(sum(m.shape[0] for m, _ in visual.values()) / max(len(visual), 1))} vectors/page"
        detail5 = _timed(job, 4, _pages)
        job.step(4, "done", detail5)

        # 6. index
        def _index():
            doc_names = {doc_id: parsed.title for doc_id, _, parsed in docs}
            for row in page_rows:
                row["doc_name"] = doc_names.get(row["document_id"])
            sb_upsert("pages", [{k: v for k, v in r.items() if k != "doc_name"} for r in page_rows])
            sb_upsert("chunks", [{"id": c.id, "document_id": c.document_id, "page_id": f"{c.document_id}:{c.page_no}", "page_no": c.page_no, "kind": c.kind,
                                  "heading_path": c.heading_path, "bbox": c.bbox, "text": c.text, "n_tokens": c.n_tokens, "vector_id": point_id(c.id)} for c in all_chunks])
            upsert_chunks([{"id": point_id(c.id), "dense": dense[i], "sparse": sparse[i],
                            "payload": {"workspace": workspace, "document_id": c.document_id, "page_id": f"{c.document_id}:{c.page_no}", "page_no": c.page_no, "chunk_id": c.id, "kind": c.kind, "preview": c.text[:160]}}
                           for i, c in enumerate(all_chunks)])
            if visual:
                upsert_pages([{"id": point_id(pid), "multivector": mv, "pooled": pv,
                               "payload": {"workspace": workspace, "document_id": pid.split(":")[0], "page_id": pid, "page_no": int(pid.split(":")[1]), "doc_name": doc_names.get(pid.split(":")[0])}}
                              for pid, (mv, pv) in visual.items()])
            for doc_id, _, parsed in docs:
                n = sum(1 for c in all_chunks if c.document_id == doc_id)
                sb_update("documents", doc_id, {"status": "indexed", "chunks": n, "indexed_at": now_iso()})
        _timed(job, 5, _index, f"{s.chunks_collection} · {s.pages_collection} · Supabase")
        for fr, (doc_id, _, _) in zip(job.file_rows, docs):
            fr["chunks"] = sum(1 for c in all_chunks if c.document_id == doc_id)
        job.finish()
    except Exception as e:  # noqa: BLE001
        for st in job.steps:
            if st["state"] == "active":
                st["state"] = "error"
        job.finish(error=f"{type(e).__name__}: {e}")
        raise
