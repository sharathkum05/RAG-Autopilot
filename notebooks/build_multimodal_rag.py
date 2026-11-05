"""Builds multimodal_rag.ipynb: the Colab prototype, one cell per pipeline step.

Run:  uv run --with nbformat python notebooks/build_multimodal_rag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = Path(__file__).resolve().parent.parent / "multimodal_rag.ipynb"

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("md", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# --------------------------------------------------------------------------- #
md(r"""
# Multimodal RAG over ViDoRe v2 — Colab prototype

End-to-end prototype of the evidence-retrieval pipeline on one ViDoRe v2 dataset
(`vidore/economics_reports_v2`: 452 page images from 5 World Bank reports, 58 English queries,
graded page-level relevance). Same architecture as the full Modal build; this notebook proves every
stage on real data and produces the first **measured** numbers.

| Step | Cell | What happens |
|---|---|---|
| 1 | Dependencies | Docling, ColQwen2.5, bge-m3, BM25, Qdrant, ranx |
| 2 | Configuration | paths, models, chunk sizes, dry-run switch |
| 3 | Load dataset | page images → PNG files + manifest; queries, qrels |
| 4 | Parse | Docling layout + OCR + table structure → canonical page JSON |
| 5 | Chunk | section / table / figure chunks with page + bbox metadata |
| 6 | Text embeddings | bge-m3 dense + BM25 sparse per chunk |
| 7 | Visual embeddings | ColQwen2.5 multivectors per page (+ pooled vector) |
| 8 | Index | Qdrant `pages` and `chunks` collections |
| 9 | Retrieve | visual, text hybrid, RRF fusion, optional rerank → evidence records |
| 10 | Evaluate | nDCG / recall / MRR per config + latency p50/p95 |
| 11 | Demo query | top-5 evidence with page thumbnails |
| 12 | Failure analysis | retrieval miss vs ranking miss, per query |
| 13 | Answer (optional) | Claude reads the evidence, cites pages, TTR measured |
| 14 | Persist | zip the work folder (or copy to Drive) |

**Runtime:** pick an **L4** or **A100** GPU (Runtime → Change runtime type). A T4 works with fp16 but is slower.
**Dry run:** set `SAMPLE_PAGES = 40` in the configuration cell to validate the whole notebook in a few minutes, then set it back to `None` for the full run.
**Persistence:** a Colab runtime can disappear. Run the optional Drive-mount cell to keep the work folder, or download the zip from the last cell.
""")

code(r"""
# @title 1 · Dependencies
%pip install -q docling easyocr colpali-engine sentence-transformers fastembed qdrant-client datasets ranx pandas pyarrow tqdm
# Colab ships torchao 0.10, which peft (a colpali-engine dependency) refuses to import next to. Nothing here uses it.
%pip uninstall -y -q torchao

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less GPU fragmentation; must precede CUDA init
import importlib, torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
for m in ["docling", "colpali_engine", "sentence_transformers", "fastembed", "qdrant_client", "datasets", "ranx"]:
    mod = importlib.import_module(m)
    print(f"{m:22s} {getattr(mod, '__version__', '')}")
""")

code(r"""
# @title (optional) Mount Google Drive so the work folder survives the runtime
MOUNT_DRIVE = False  # @param {type:"boolean"}
# Tick the box and run this cell in the browser to keep the work folder on Drive (opens a Google sign-in).
# Leave it unticked for a dry run or when running from the CLI; the work folder then lives on the VM disk.
from pathlib import Path
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    Path("/content/drive/MyDrive/vidore-prototype").mkdir(parents=True, exist_ok=True)
    print("Drive mounted; work folder: /content/drive/MyDrive/vidore-prototype")
else:
    print("Drive not mounted; the work folder lives on the VM disk (download the zip from the last cell to keep it)")
""")

code(r"""
# @title 2 · Configuration
import os, re, json, time, math, hashlib, pickle
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, torch
from tqdm.auto import tqdm

DATASET = "vidore/economics_reports_v2"     # ViDoRe v2, BEIR layout: corpus / queries / qrels / docs
_drive = Path("/content/drive/MyDrive/vidore-prototype")
ROOT = (_drive if _drive.exists() else Path(os.environ.get("VIDORE_ROOT", "/content/work"))) / DATASET.split("/")[-1]

SAMPLE_PAGES = int(os.environ.get("VIDORE_SAMPLE_PAGES", "0")) or None   # None = all 452 pages; 40 = quick dry run
LANGUAGES = ["english"]                      # queries exist in english/french/spanish/german (58 each)

CHUNK_WORDS = 300             # ~400 bge-m3 tokens
TABLE_ROWS_PER_PART = 25      # long tables are split by rows, header row repeated
COLQWEN_MODEL = "vidore/colqwen2.5-v0.2"
TEXT_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
BM25_MODEL = "Qdrant/bm25"

TOP_K_PER_PATH, FUSED_K, FINAL_K, RRF_K = 50, 20, 5, 60

# Qdrant: a cloud cluster if QDRANT_URL is set (Colab secret or env), else local mode on disk under ROOT.
def _secret(name):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)
QDRANT_URL, QDRANT_API_KEY = _secret("QDRANT_URL"), _secret("QDRANT_API_KEY")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if DEVICE == "cuda" else torch.float32
for sub in ["pages", "parsed", "emb", "qdrant", "runs"]:
    (ROOT / sub).mkdir(parents=True, exist_ok=True)

TIMINGS = defaultdict(list)   # stage -> list of seconds; every measured number comes from here
print(f"root     = {ROOT}\ndevice   = {DEVICE} ({DTYPE})\nqdrant   = {'cloud ' + QDRANT_URL if QDRANT_URL else 'local mode on disk'}\nsample   = {SAMPLE_PAGES or 'all pages'}")
""")

code(r"""
# @title 3 · Load the dataset → page PNGs, manifest, queries, qrels
from datasets import load_dataset
from PIL import Image

corpus = load_dataset(DATASET, "corpus", split="test")
queries = load_dataset(DATASET, "queries", split="test").to_pandas()
qrels = load_dataset(DATASET, "qrels", split="test").to_pandas()
docs = load_dataset(DATASET, "docs", split="test").to_pandas()
if SAMPLE_PAGES:   # dry run: keep the pages that answer the most queries, so the eval still has labelled queries
    keep = set(qrels[qrels["score"] > 0]["corpus-id"].value_counts().index[:SAMPLE_PAGES].tolist())
    corpus = corpus.select([i for i, cid in enumerate(corpus["corpus-id"]) if cid in keep])

rows = []
for ex in tqdm(corpus, desc="writing pages"):
    cid, did, img = int(ex["corpus-id"]), ex["doc-id"], ex["image"]
    path = ROOT / "pages" / f"{cid:04d}.png"
    if not path.exists():
        img.save(path)
    rows.append({"corpus_id": cid, "doc_id": did, "image_path": str(path), "width": img.width, "height": img.height,
                 "page_hash": hashlib.sha256(path.read_bytes()).hexdigest()[:16]})
manifest = pd.DataFrame(rows).sort_values("corpus_id").reset_index(drop=True)
manifest["page_in_doc"] = manifest.groupby("doc_id").cumcount() + 1          # 1-based page number inside its report
manifest.to_parquet(ROOT / "manifest.parquet")
queries.to_parquet(ROOT / "queries.parquet"); qrels.to_parquet(ROOT / "qrels.parquet"); docs.to_parquet(ROOT / "docs.parquet")

# Query set for this run: chosen languages, and only queries with at least one relevant page among the loaded pages.
qrels_run = qrels[qrels["corpus-id"].isin(manifest.corpus_id) & (qrels["score"] > 0)]
qset = queries[queries.language.isin(LANGUAGES) & queries["query-id"].isin(qrels_run["query-id"])].reset_index(drop=True)

print(f"pages: {len(manifest)} across {manifest.doc_id.nunique()} documents")
print(manifest.groupby("doc_id").size().to_string())
print(f"\nqueries in run: {len(qset)} ({', '.join(LANGUAGES)}) | relevant pages per query: "
      f"mean {qrels_run[qrels_run['query-id'].isin(qset['query-id'])].groupby('query-id').size().mean():.1f}")
assert len(qset) > 0, "no labelled queries for the loaded pages; raise SAMPLE_PAGES or set it to None"
print("\nexample:", qset.iloc[0]["query"])
display(Image.open(manifest.iloc[0].image_path).resize((400, int(400 * manifest.iloc[0].height / manifest.iloc[0].width))))
""")

code(r"""
# @title 4 · Parse every page with Docling (layout + OCR + tables) → canonical page JSON
from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (PdfPipelineOptions, EasyOcrOptions, TableFormerMode,
                                                AcceleratorOptions, AcceleratorDevice)
from docling_core.types.doc import TableItem, PictureItem, SectionHeaderItem

opts = PdfPipelineOptions(do_ocr=True, do_table_structure=True)
opts.ocr_options = EasyOcrOptions(force_full_page_ocr=True, lang=["en"])
opts.table_structure_options.mode = TableFormerMode.ACCURATE
opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA if DEVICE == "cuda" else AcceleratorDevice.CPU)
converter = DocumentConverter(format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)})

def _bbox_norm(prov, page_w, page_h):
    b = prov.bbox.to_top_left_origin(page_h)
    return [round(b.l / page_w, 4), round(b.t / page_h, 4), round(b.r / page_w, 4), round(b.b / page_h, 4)]

def parse_page(row) -> dict:
    # One ViDoRe page = one image = one Docling document with a single page.
    doc = converter.convert(row.image_path).document
    page = next(iter(doc.pages.values()))
    W, H = page.size.width, page.size.height
    heading_path, elements = [], []
    for item, _level in doc.iterate_items():
        if not getattr(item, "prov", None):
            continue
        label = item.label.value if hasattr(item.label, "value") else str(item.label)
        el = {"id": f"{row.corpus_id}-e{len(elements):03d}", "type": label,
              "bbox": _bbox_norm(item.prov[0], W, H), "heading_path": list(heading_path)}
        if isinstance(item, SectionHeaderItem):
            lvl = int(getattr(item, "level", 1) or 1)
            heading_path = heading_path[: max(0, lvl - 1)] + [item.text.strip()]
            el.update(text=item.text, level=lvl)
        elif isinstance(item, TableItem):
            el.update(markdown=item.export_to_markdown(doc), n_rows=item.data.num_rows, n_cols=item.data.num_cols,
                      caption=item.caption_text(doc))
        elif isinstance(item, PictureItem):
            el.update(caption=item.caption_text(doc))
        else:
            el.update(text=getattr(item, "text", "") or "")
        elements.append(el)
    return {"corpus_id": int(row.corpus_id), "doc_id": row.doc_id, "page_in_doc": int(row.page_in_doc),
            "image_path": row.image_path, "width": int(row.width), "height": int(row.height),
            "parser": "docling+easyocr", "elements": elements, "text": doc.export_to_markdown()}

parsed, errors = {}, {}
for row in tqdm(manifest.itertuples(), total=len(manifest), desc="parsing"):
    out = ROOT / "parsed" / f"{row.corpus_id:04d}.json"
    if out.exists():                                  # resumable: skip pages already parsed
        parsed[row.corpus_id] = json.loads(out.read_text()); continue
    t0 = time.perf_counter()
    try:
        page = parse_page(row)
        out.write_text(json.dumps(page))
        parsed[row.corpus_id] = page
    except Exception as e:                            # record and move on; the visual path still covers this page
        errors[row.corpus_id] = repr(e)
    TIMINGS["parse_page"].append(time.perf_counter() - t0)

import gc
del converter; gc.collect(); torch.cuda.empty_cache()     # Docling's OCR/layout/table models are not needed after this cell
n_el = sum(len(p["elements"]) for p in parsed.values())
kinds = pd.Series([e["type"] for p in parsed.values() for e in p["elements"]]).value_counts()
print(f"parsed {len(parsed)} pages, {n_el} elements, {len(errors)} errors | "
      f"{np.mean(TIMINGS['parse_page']):.2f} s/page" if TIMINGS["parse_page"] else "all pages were already parsed")
print(kinds.to_string())
if errors: print("errors:", errors)
assert parsed, "no page parsed; check the errors above"
ex = next(iter(parsed.values()))
print(f"\npage {ex['corpus_id']} ({ex['doc_id']} p.{ex['page_in_doc']}), first elements:")
for e in ex["elements"][:6]:
    print(" ", e["type"], e["bbox"], (e.get("text") or e.get("markdown") or e.get("caption") or "")[:90].replace("\n", " "))
""")

code(r"""
# @title 5 · Chunking → chunks (section / table / figure) with page + bbox metadata
_SENT = re.compile(r"(?<=[.!?])\s+")

def _union(boxes):
    if not boxes: return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]

def _split_table(markdown: str):
    lines = [l for l in markdown.splitlines() if l.strip()]
    if len(lines) <= 2 + TABLE_ROWS_PER_PART:
        return [markdown]
    header, body = lines[:2], lines[2:]
    return ["\n".join(header + body[i:i + TABLE_ROWS_PER_PART]) for i in range(0, len(body), TABLE_ROWS_PER_PART)]

def chunk_page(p: dict) -> list[dict]:
    chunks, buf, ids, boxes, hp, carry = [], [], [], [], [], ""
    def flush():
        nonlocal buf, ids, boxes, carry
        if buf:
            text = " ".join(buf).strip()
            chunks.append({"kind": "text", "text": text, "element_ids": ids, "bbox": _union(boxes), "heading_path": hp})
            sents = _SENT.split(text)
            carry = sents[-1] if len(sents) > 1 and len(sents[-1].split()) < 60 else ""   # 1-sentence overlap
        buf, ids, boxes = [], [], []
    for el in p["elements"]:
        t = el["type"]
        if t == "table" and el.get("markdown"):
            flush()
            for i, part in enumerate(_split_table(el["markdown"])):
                lead = (el.get("caption") or "").strip()
                chunks.append({"kind": "table", "text": (lead + "\n" if lead else "") + part, "element_ids": [el["id"]],
                               "bbox": el["bbox"], "heading_path": el["heading_path"], "part": i})
        elif t == "picture":
            flush()
            if (el.get("caption") or "").strip():
                chunks.append({"kind": "figure", "text": el["caption"].strip(), "element_ids": [el["id"]],
                               "bbox": el["bbox"], "heading_path": el["heading_path"]})
        else:
            text = (el.get("text") or "").strip()
            if not text: continue
            if t in ("section_header", "title"):
                flush(); hp = el["heading_path"] + [text]
            words = len(text.split())
            if buf and sum(len(b.split()) for b in buf) + words > CHUNK_WORDS:
                flush()
            if not buf:
                hp = el["heading_path"] if t not in ("section_header", "title") else hp
                if carry: buf.append(carry); carry = ""
            buf.append(text); ids.append(el["id"]); boxes.append(el["bbox"])
    flush()
    for i, c in enumerate(chunks):
        c.update(chunk_id=f"{p['corpus_id']}-c{i:02d}", corpus_id=p["corpus_id"], doc_id=p["doc_id"], page_in_doc=p["page_in_doc"],
                 n_words=len(c["text"].split()),
                 embed_text=(p["doc_id"] + (" › " + " › ".join(c["heading_path"]) if c["heading_path"] else "") + "\n" + c["text"]))
    return chunks

chunks = pd.DataFrame([c for p in parsed.values() for c in chunk_page(p)])
chunks.to_parquet(ROOT / "chunks.parquet")
pages_text = {cid: p["text"] for cid, p in parsed.items()}        # whole-page text, used by the reranker
print(f"{len(chunks)} chunks from {len(parsed)} pages ({len(chunks)/max(len(parsed),1):.1f} per page)")
print(chunks.kind.value_counts().to_string())
print(chunks.n_words.describe()[["mean", "50%", "max"]].round(0).to_string())
print("\nexample chunk:\n", chunks.iloc[0].embed_text[:500])
""")

code(r"""
# @title 6 · Text embeddings: bge-m3 dense + BM25 sparse per chunk
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

text_model = SentenceTransformer(TEXT_MODEL, device=DEVICE)
text_model.max_seq_length = 1024
bm25 = SparseTextEmbedding(BM25_MODEL)

t0 = time.perf_counter()
dense = text_model.encode(chunks.embed_text.tolist(), batch_size=32, normalize_embeddings=True, show_progress_bar=True)
TIMINGS["embed_text_total"].append(time.perf_counter() - t0)
t0 = time.perf_counter()
sparse = list(bm25.embed(chunks.text.tolist(), batch_size=256))
TIMINGS["embed_bm25_total"].append(time.perf_counter() - t0)

np.save(ROOT / "emb" / "chunks_dense.npy", dense)
with open(ROOT / "emb" / "chunks_bm25.pkl", "wb") as f:
    pickle.dump([(s.indices, s.values) for s in sparse], f)
torch.cuda.empty_cache()
print(f"dense {dense.shape} in {TIMINGS['embed_text_total'][-1]:.1f}s | bm25 {len(sparse)} vectors in {TIMINGS['embed_bm25_total'][-1]:.1f}s")
""")

code(r"""
# @title 7 · Visual embeddings: ColQwen2.5 multivectors per page (+ one pooled vector)
# Guard against Colab's preinstalled torchao 0.10 (peft refuses it). Safe to run repeatedly; no runtime restart needed.
import importlib, importlib.util, subprocess, sys
if importlib.util.find_spec("torchao") is not None:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    sys.modules.pop("torchao", None)
    importlib.invalidate_caches()
try:
    import peft.import_utils as _piu
    _piu.is_torchao_available.cache_clear()
except Exception:
    pass

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

col_model = ColQwen2_5.from_pretrained(COLQWEN_MODEL, torch_dtype=DTYPE, device_map=DEVICE).eval()
col_proc = ColQwen2_5_Processor.from_pretrained(COLQWEN_MODEL)

def _unpad(emb, mask):
    return [e[m].float().cpu().numpy().astype(np.float32) for e, m in zip(emb, mask.bool())]

multivecs, pooled, BATCH = {}, {}, 2
for i in tqdm(range(0, len(manifest), BATCH), desc="colqwen pages"):
    rows = manifest.iloc[i:i + BATCH]
    imgs = [Image.open(p).convert("RGB") for p in rows.image_path]
    t0 = time.perf_counter()
    batch = col_proc.process_images(imgs).to(col_model.device)
    with torch.inference_mode():
        emb = col_model(**batch)
    for cid, v in zip(rows.corpus_id, _unpad(emb, batch["attention_mask"])):
        multivecs[int(cid)] = v
        m = v.mean(0); pooled[int(cid)] = m / (np.linalg.norm(m) + 1e-9)
    TIMINGS["embed_visual_batch"].extend([(time.perf_counter() - t0) / len(rows)] * len(rows))
    del emb, batch
torch.cuda.empty_cache()

bad = [cid for cid, v in multivecs.items() if not np.isfinite(v).all()]
np.savez(ROOT / "emb" / "pages_colqwen.npz", **{str(k): v for k, v in multivecs.items()})
np.save(ROOT / "emb" / "pages_pooled.npy", np.stack([pooled[c] for c in manifest.corpus_id]))
toks = [v.shape[0] for v in multivecs.values()]
print(f"{len(multivecs)} pages | vectors/page mean {np.mean(toks):.0f} (max {max(toks)}) × {toks and multivecs[next(iter(multivecs))].shape[1]} dims "
      f"| {np.mean(TIMINGS['embed_visual_batch']):.2f} s/page | non-finite pages: {bad}")

def encode_query_visual(q: str) -> np.ndarray:
    batch = col_proc.process_queries([q]).to(col_model.device)
    with torch.inference_mode():
        emb = col_model(**batch)
    return _unpad(emb, batch["attention_mask"])[0]
""")

code(r"""
# @title 8 · Index into Qdrant: `pages` (multivector + pooled) and `chunks` (dense + BM25 sparse)
from qdrant_client import QdrantClient, models

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120) if QDRANT_URL else QdrantClient(path=str(ROOT / "qdrant"))
for name in ("pages", "chunks"):
    if client.collection_exists(name):
        client.delete_collection(name)

client.create_collection("pages", vectors_config={
    "colqwen": models.VectorParams(size=128, distance=models.Distance.DOT,
                                   multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                                   hnsw_config=models.HnswConfigDiff(m=0)),          # scanned after the pooled prefetch
    "pooled": models.VectorParams(size=128, distance=models.Distance.DOT)})
client.create_collection("chunks",
    vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.DOT)},
    sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)})

t0 = time.perf_counter()
page_pts = [models.PointStruct(id=int(r.corpus_id),
                               vector={"colqwen": multivecs[int(r.corpus_id)].tolist(), "pooled": pooled[int(r.corpus_id)].tolist()},
                               payload={"corpus_id": int(r.corpus_id), "doc_id": r.doc_id, "page_in_doc": int(r.page_in_doc),
                                        "image_path": r.image_path, "page_hash": r.page_hash})
            for r in manifest.itertuples()]
for i in range(0, len(page_pts), 16):
    client.upsert("pages", page_pts[i:i + 16])

chunk_pts = [models.PointStruct(id=i,
                                vector={"dense": dense[i].tolist(),
                                        "bm25": models.SparseVector(indices=sparse[i].indices.tolist(), values=sparse[i].values.tolist())},
                                payload={"chunk_id": c.chunk_id, "corpus_id": int(c.corpus_id), "doc_id": c.doc_id,
                                         "page_in_doc": int(c.page_in_doc), "kind": c.kind, "text": c.text[:1500],
                                         "heading_path": list(c.heading_path), "bbox": c.bbox})
             for i, c in enumerate(chunks.itertuples())]
for i in range(0, len(chunk_pts), 128):
    client.upsert("chunks", chunk_pts[i:i + 128])
TIMINGS["index_total"].append(time.perf_counter() - t0)
print(f"pages: {client.count('pages').count} | chunks: {client.count('chunks').count} | indexed in {TIMINGS['index_total'][-1]:.1f}s")
""")

code(r"""
# @title 9 · Retrieval: visual, text hybrid, RRF fusion, optional rerank → evidence records
from sentence_transformers import CrossEncoder
# The reranker goes on the GPU only if at least 4 GB is free; otherwise CPU (1-2 s per query, fine for the prototype).
_free_gb = torch.cuda.mem_get_info()[0] / 1e9 if DEVICE == "cuda" else 0
reranker = CrossEncoder(RERANK_MODEL, max_length=1024, device=DEVICE if _free_gb > 4 else "cpu")
print(f"reranker on {reranker.device if hasattr(reranker, 'device') else ('cuda' if _free_gb > 4 else 'cpu')} (free GPU memory {_free_gb:.1f} GB)")
page_meta = manifest.set_index("corpus_id").to_dict("index")

def encode_query_dense(q):  return text_model.encode([q], normalize_embeddings=True)[0]
def encode_query_bm25(q):
    s = next(bm25.query_embed(q)); return models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())

def search_visual(q, k=TOP_K_PER_PATH):
    # ColQwen late interaction: the pooled single-vector prefetch narrows to 4k candidates, MaxSim ranks them.
    qv = encode_query_visual(q)
    qp = qv.mean(0); qp = qp / (np.linalg.norm(qp) + 1e-9)
    res = client.query_points("pages", prefetch=models.Prefetch(query=qp.tolist(), using="pooled", limit=max(4 * k, 100)),
                              query=qv.tolist(), using="colqwen", limit=k, with_payload=False)
    return [(p.id, p.score) for p in res.points]

def search_text(q, k=TOP_K_PER_PATH, dense_on=True, bm25_on=True):
    # Chunk-level search (dense and/or BM25, fused server-side by RRF), aggregated to pages by their best chunk.
    pre = []
    if dense_on: pre.append(models.Prefetch(query=encode_query_dense(q).tolist(), using="dense", limit=200))
    if bm25_on:  pre.append(models.Prefetch(query=encode_query_bm25(q), using="bm25", limit=200))
    if len(pre) == 1:
        res = client.query_points("chunks", query=pre[0].query, using=pre[0].using, limit=400, with_payload=["corpus_id", "chunk_id"])
    else:
        res = client.query_points("chunks", prefetch=pre, query=models.FusionQuery(fusion=models.Fusion.RRF), limit=400,
                                  with_payload=["corpus_id", "chunk_id"])
    pages, best_chunk = {}, {}
    for p in res.points:                       # points arrive best-first; first hit per page wins
        cid = p.payload["corpus_id"]
        if cid not in pages:
            pages[cid] = p.score; best_chunk[cid] = p.payload["chunk_id"]
        if len(pages) >= k: break
    return [(cid, s) for cid, s in pages.items()], best_chunk

def rrf(ranked_lists, k=RRF_K):
    scores = defaultdict(float)
    for lst in ranked_lists:
        for rank, (cid, _s) in enumerate(lst):
            scores[cid] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])

CONFIGS = {   # name -> (visual, dense, bm25, rerank)
    "bm25":         (False, False, True,  False),
    "text_hybrid":  (False, True,  True,  False),   # baseline
    "visual":       (True,  False, False, False),
    "fused":        (True,  True,  True,  False),   # the one improvement
    "fused_rerank": (True,  True,  True,  True),    # second experiment
}

def retrieve(q: str, config: str = "fused", k: int = FINAL_K) -> list[dict]:
    visual_on, dense_on, bm25_on, rerank_on = CONFIGS[config]
    t = {}; lists = []; vis = {}; txt = {}; best_chunk = {}
    if visual_on:
        t0 = time.perf_counter(); v = search_visual(q); t["visual"] = time.perf_counter() - t0
        vis = {cid: (r, s) for r, (cid, s) in enumerate(v)}; lists.append(v)
    if dense_on or bm25_on:
        t0 = time.perf_counter(); tx, best_chunk = search_text(q, dense_on=dense_on, bm25_on=bm25_on); t["text"] = time.perf_counter() - t0
        txt = {cid: (r, s) for r, (cid, s) in enumerate(tx)}; lists.append(tx)
    fused = rrf(lists) if len(lists) > 1 else lists[0]
    cand = fused[:FUSED_K]
    if rerank_on:
        t0 = time.perf_counter()
        rr = reranker.predict([(q, pages_text.get(cid, "")[:4000]) for cid, _ in cand])
        cand = sorted(((cid, float(s)) for (cid, _), s in zip(cand, rr)), key=lambda x: -x[1])
        t["rerank"] = time.perf_counter() - t0
    out = []
    for rank, (cid, score) in enumerate(cand[:k]):
        m = page_meta[cid]
        out.append({"rank": rank + 1, "score": float(score), "corpus_id": int(cid), "doc_id": m["doc_id"], "page_in_doc": m["page_in_doc"],
                    "image_path": m["image_path"],
                    "visual_rank": vis.get(cid, (None, None))[0], "visual_score": vis.get(cid, (None, None))[1],
                    "text_rank": txt.get(cid, (None, None))[0], "text_score": txt.get(cid, (None, None))[1],
                    "best_chunk": best_chunk.get(cid), "timings": t})
    return out

_ = retrieve(qset.iloc[0]["query"], "fused_rerank")       # warm-up: loads kernels, fills caches
for e in retrieve(qset.iloc[0]["query"], "fused", k=3):
    print(e["rank"], e["doc_id"], "p.", e["page_in_doc"], "| visual", e["visual_rank"], "| text", e["text_rank"], "| chunk", e["best_chunk"])
""")

code(r"""
# @title 10 · Evaluate every config: nDCG / recall / MRR + latency p50/p95
from ranx import Qrels, Run, evaluate

qrels_obj = Qrels({str(qid): {str(int(cid)): int(s) for cid, s in grp[["corpus-id", "score"]].values}
                   for qid, grp in qrels_run[qrels_run["query-id"].isin(qset["query-id"])].groupby("query-id")})
METRICS = ["ndcg@5", "ndcg@10", "recall@5", "recall@10", "precision@5", "mrr@10"]
results, runs = {}, {}
for name in CONFIGS:
    run_dict, lat = {}, []
    for qid, qtext in tqdm(list(zip(qset["query-id"], qset["query"])), desc=name, leave=False):
        t0 = time.perf_counter()
        ev = retrieve(qtext, name, k=10)
        lat.append(time.perf_counter() - t0)
        run_dict[str(qid)] = {str(e["corpus_id"]): float(e["score"]) for e in ev}
    runs[name] = run_dict
    m = evaluate(qrels_obj, Run(run_dict, name=name), METRICS)
    lat = lat[1:] or lat                                              # drop the first (cold) query
    m.update({"p50_ms": 1000 * float(np.percentile(lat, 50)), "p95_ms": 1000 * float(np.percentile(lat, 95)), "n_queries": len(qset)})
    results[name] = m

table = pd.DataFrame(results).T
stamp = time.strftime("%Y%m%d-%H%M%S")
table.to_csv(ROOT / "runs" / f"results_{stamp}.csv")
json.dump({"dataset": DATASET, "pages": len(manifest), "languages": LANGUAGES, "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
           "results": results, "runs": runs, "timings_s": {k: float(np.mean(v)) for k, v in TIMINGS.items()}},
          open(ROOT / "runs" / f"run_{stamp}.json", "w"), indent=1)
display(table.round(3))
print("measured on", len(manifest), "pages;", len(qset), "queries;", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu")
""")

code(r"""
# @title 11 · Demo query: top-5 evidence with page thumbnails
import matplotlib.pyplot as plt
QUESTION = qset.iloc[3]["query"]      # replace with your own question
ev = retrieve(QUESTION, "fused_rerank", k=5)
rel = set(qrels_run[qrels_run["query-id"] == int(qset.iloc[3]["query-id"])]["corpus-id"])
print("Q:", QUESTION, "\n")
fig, axes = plt.subplots(1, len(ev), figsize=(4 * len(ev), 6))
for ax, e in zip(np.atleast_1d(axes), ev):
    ax.imshow(Image.open(e["image_path"])); ax.axis("off")
    ax.set_title(f"#{e['rank']} {e['doc_id'][:22]}\np.{e['page_in_doc']}  vis {e['visual_rank']}  txt {e['text_rank']}"
                 f"\n{'RELEVANT' if e['corpus_id'] in rel else 'not labelled'}", fontsize=9)
plt.tight_layout(); plt.show()
if ev and ev[0]["best_chunk"] is not None:
    print("best chunk of #1:\n", chunks.set_index("chunk_id").loc[ev[0]["best_chunk"]].text[:600])
""")

code(r"""
# @title 12 · Failure analysis: where did the first relevant page land, and which path found it?
def first_relevant_rank(run, qid, rel):
    ranked = sorted(run[qid].items(), key=lambda x: -x[1])
    for r, (cid, _) in enumerate(ranked, 1):
        if int(cid) in rel: return r
    return None

rows = []
for qid_int, qtext in zip(qset["query-id"], qset["query"]):
    qid = str(qid_int); rel = set(qrels_run[qrels_run["query-id"] == qid_int]["corpus-id"])
    rows.append({"query_id": qid, "query": qtext[:80],
                 "text_hybrid": first_relevant_rank(runs["text_hybrid"], qid, rel),
                 "visual": first_relevant_rank(runs["visual"], qid, rel),
                 "fused": first_relevant_rank(runs["fused"], qid, rel),
                 "fused_rerank": first_relevant_rank(runs["fused_rerank"], qid, rel)})
fa = pd.DataFrame(rows)
def diagnose(row):
    if row.fused is not None and row.fused <= 5: return "ok"
    if row.text_hybrid is None and row.visual is None: return "R1 miss: no path found a relevant page in top 10"
    if row.fused is None: return "R2 fusion: a path found it, fusion dropped it"
    return "R2 ranking: found but below top 5"
fa["diagnosis"] = fa.apply(diagnose, axis=1)
print(fa.diagnosis.value_counts().to_string(), "\n")
display(fa[fa.diagnosis != "ok"].sort_values("fused", na_position="first").head(15))
fa.to_csv(ROOT / "runs" / f"failures_{stamp}.csv", index=False)
""")

code(r"""
# @title 13 · (optional) Answer with Claude from the top evidence, with page citations and measured TTR
# Needs an ANTHROPIC_API_KEY in Colab secrets (key icon in the left sidebar). Skipped otherwise.
import base64, io
API_KEY = _secret("ANTHROPIC_API_KEY")
if not API_KEY:
    print("ANTHROPIC_API_KEY not set; skipping the answer step.")
else:
    %pip install -q anthropic
    import anthropic
    ac = anthropic.Anthropic(api_key=API_KEY)

    def answer(question: str, config: str = "fused_rerank", n_pages: int = 3, model: str = "claude-opus-5-5"):
        t_start = time.perf_counter()
        ev = retrieve(question, config, k=n_pages)
        content = []
        for e in ev:                                   # page text as a citable document, page image for reading charts/tables
            content.append({"type": "document", "source": {"type": "text", "media_type": "text/plain",
                                                            "data": pages_text.get(e["corpus_id"], "")[:20000] or "(no OCR text)"},
                            "title": f"{e['doc_id']} p.{e['page_in_doc']}", "citations": {"enabled": True}})
            img = Image.open(e["image_path"]).convert("RGB"); img.thumbnail((1200, 1200))
            buf = io.BytesIO(); img.save(buf, format="PNG")
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                         "data": base64.b64encode(buf.getvalue()).decode()}})
        content.append({"type": "text", "text": question})
        resp = ac.beta.messages.create(
            model=model, max_tokens=4096, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            system="Answer only from the provided pages. Cite the page for every claim. If the pages do not contain the answer, say so.",
            messages=[{"role": "user", "content": content}])
        ttr = time.perf_counter() - t_start
        text, cited = "", []
        for block in resp.content:
            if block.type == "text":
                text += block.text
                for c in (block.citations or []):
                    if c.document_title not in cited: cited.append(c.document_title)
        return {"answer": text, "cited_pages": cited, "evidence": ev, "ttr_s": ttr, "retrieval_s": sum(ev[0]["timings"].values()),
                "stop_reason": resp.stop_reason, "model": resp.model}

    out = answer(QUESTION)
    print("Q:", QUESTION, f"\n\nTTR {out['ttr_s']:.1f}s (retrieval {out['retrieval_s']:.2f}s) | model {out['model']} | stop {out['stop_reason']}\n")
    print(out["answer"], "\n\ncited:", out["cited_pages"])
""")

code(r"""
# @title 14 · Persist the work folder (zip for download, or it is already on Drive)
import shutil
if str(ROOT).startswith("/content/drive"):
    print("Work folder is on Google Drive:", ROOT)
else:
    zip_path = shutil.make_archive("/content/vidore_prototype", "zip", ROOT.parent)
    print("zip:", zip_path, f"({os.path.getsize(zip_path)/1e6:.0f} MB)")
    print("download with:  colab download -s multimodal-rag /content/vidore_prototype.zip ./vidore_prototype.zip")
print("\nmeasured stage timings (mean seconds):")
print(pd.Series({k: float(np.mean(v)) for k, v in TIMINGS.items()}).round(3).to_string())
""")


def build() -> None:
    nb = new_notebook()
    nb.metadata.update({
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"name": "multimodal rag", "provenance": [], "gpuType": "L4"},
    })
    for kind, src in CELLS:
        nb.cells.append(new_markdown_cell(src) if kind == "md" else new_code_cell(src))
    nbformat.validate(nb)
    nbformat.write(nb, OUT)
    print(f"wrote {OUT} with {len(nb.cells)} cells")


if __name__ == "__main__":
    build()
