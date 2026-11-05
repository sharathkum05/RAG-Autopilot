"""Build the Index inspector's mock data from the Python textbook subset: a thumbnail per page and chunks cut from
the benchmark's own OCR markdown with the same rules the pipeline uses (heading sections ~300 words, tables whole).
Run:  uv run python scripts/build_index_json.py
"""
import json, re
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path("raw dataset v3/python_textbook"); OUT = Path("web/public")
corpus = pd.read_parquet(ROOT / "corpus.parquet").sort_values("page_number_in_doc")
HEAD = re.compile(r"^(#+\s+.+|\d+(\.\d+)*\s+[A-Z][^|]{2,60})$")
CHUNK_WORDS = 300

def chunk_page(cid, pno, md):
    lines = [l.rstrip() for l in md.splitlines()]
    chunks, buf, heading, table = [], [], "", []
    def flush():
        nonlocal buf
        text = " ".join(buf).strip()
        if text: chunks.append({"kind": "text", "heading": heading, "text": text})
        buf = []
    def flush_table():
        nonlocal table
        if table:
            chunks.append({"kind": "table", "heading": heading, "text": "\n".join(table)})
            table = []
    for l in lines:
        if "|" in l and l.count("|") >= 2:
            if buf: flush()
            table.append(l); continue
        if table: flush_table()
        s = l.strip()
        if not s: continue
        if HEAD.match(s) and len(s.split()) <= 10:
            flush(); heading = s.lstrip("# ").strip(); buf.append(s); continue
        if sum(len(b.split()) for b in buf) + len(s.split()) > CHUNK_WORDS and buf: flush()
        buf.append(s)
    flush(); flush_table()
    out = []
    for i, c in enumerate(chunks):
        if c["kind"] == "table" and len(c["text"].split("\n")) < 2: continue
        words = len(c["text"].split())
        out.append({"id": f"{cid}-c{i:02d}", "page": int(cid), "pno": int(pno), "kind": c["kind"], "heading": c["heading"],
                    "text": c["text"][:2400], "n_tokens": int(words * 1.3), "vector_id": f"{cid}-c{i:02d}"})
    return out

pages, chunks = [], []
for r in corpus.itertuples():
    thumb = OUT / "pages/python/all" / f"{r.corpus_id:05d}.jpg"
    if not thumb.exists():
        im = Image.open(ROOT.parent / r.image_path if not Path(r.image_path).is_absolute() else r.image_path).convert("RGB")
        im = im.resize((300, int(im.height * 300 / im.width)), Image.LANCZOS); im.save(thumb, "JPEG", quality=55, optimize=True)
    pc = chunk_page(r.corpus_id, r.page_number_in_doc, r.markdown or "")
    chunks.extend(pc)
    pages.append({"id": int(r.corpus_id), "pno": int(r.page_number_in_doc), "width": int(r.width), "height": int(r.height),
                  "image": f"/pages/python/all/{r.corpus_id:05d}.jpg", "n_chunks": len(pc), "n_visual_vectors": 768,
                  "n_words": len((r.markdown or "").split())})
doc = {"id": "python_textbook", "name": "Introduction to Python Programming", "source": "vidore/vidore_v3_computer_science",
       "pages": len(pages), "chunks": len(chunks), "indexed_at": "2026-11-04T01:12:00Z",
       "collections": {"pages": "matt_pages", "chunks": "matt_chunks"}}
(OUT / "data/python_textbook.json").write_text(json.dumps({"document": doc, "pages": pages, "chunks": chunks}, ensure_ascii=False))
print(f"pages={len(pages)} chunks={len(chunks)} kinds={pd.Series([c['kind'] for c in chunks]).value_counts().to_dict()}")
print("json MB:", round((OUT / 'data/python_textbook.json').stat().st_size / 1e6, 2), "| thumbs MB:", round(sum(p.stat().st_size for p in (OUT / 'pages/python/all').glob('*.jpg')) / 1e6, 1))
