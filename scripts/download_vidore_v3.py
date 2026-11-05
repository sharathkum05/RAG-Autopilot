"""Download the eight public ViDoRe V3 datasets as raw files (plain format, not the MTEB mirror).

Layout per dataset:
    <out>/<domain>/pages/<corpus_id:04d>.png     page images
    <out>/<domain>/corpus.parquet                 corpus_id, doc_id, page_number_in_doc, markdown (OCR text shipped
                                                  with the benchmark), width, height, image_path
    <out>/<domain>/queries.parquet                query_id, query, language, query_types, answer, raw_answers, ...
    <out>/<domain>/qrels.parquet                  query_id, corpus_id, score, content_type, bounding_boxes
    <out>/<domain>/documents_metadata.parquet     file_name, doc_id, url, doc_type, doc_language, doc_year, ...

Run:  uv run python scripts/download_vidore_v3.py "raw dataset v3" [domain ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

DOMAINS = ["computer_science", "energy", "finance_en", "finance_fr", "hr", "industrial", "pharmaceuticals", "physics"]


def download(domain: str, out: Path) -> dict:
    name = f"vidore/vidore_v3_{domain}"
    root = out / domain
    (root / "pages").mkdir(parents=True, exist_ok=True)
    summary = {"dataset": name}

    for cfg in ("queries", "qrels", "documents_metadata"):
        df = load_dataset(name, cfg, split="test").to_pandas()
        df.to_parquet(root / f"{cfg}.parquet")
        summary[cfg] = len(df)

    corpus = load_dataset(name, "corpus", split="test")
    rows, t0 = [], time.time()
    for ex in tqdm(corpus, desc=domain, unit="page", mininterval=5):
        cid, img = int(ex["corpus_id"]), ex["image"]
        path = root / "pages" / f"{cid:05d}.png"
        if not path.exists():
            img.save(path)
        rows.append({"corpus_id": cid, "doc_id": ex["doc_id"], "page_number_in_doc": int(ex["page_number_in_doc"]),
                     "markdown": ex.get("markdown") or "", "width": img.width, "height": img.height,
                     "image_path": str(path.relative_to(out))})
    df = pd.DataFrame(rows).sort_values(["doc_id", "page_number_in_doc"]).reset_index(drop=True)
    df.to_parquet(root / "corpus.parquet")
    summary.update(pages=len(df), documents=int(df.doc_id.nunique()),
                   page_mb=round(sum(p.stat().st_size for p in (root / "pages").glob("*.png")) / 1e6, 1),
                   seconds=round(time.time() - t0, 1))
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "raw dataset v3").resolve()
    domains = sys.argv[2:] or DOMAINS
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for d in domains:
        summaries.append(download(d, out))
        (out / "manifest.json").write_text(json.dumps(summaries, indent=1))   # updated after every dataset
    print(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    main()
