"""Download the ViDoRe v2 benchmark datasets as raw files.

Layout per dataset:
    <out>/<dataset>/pages/<corpus_id:04d>.png     page images, as shipped (no source PDFs exist)
    <out>/<dataset>/corpus.parquet                 corpus_id, doc_id, page_in_doc, width, height, image_path
    <out>/<dataset>/queries.parquet                query-id, query, language (+ any extra columns)
    <out>/<dataset>/qrels.parquet                  query-id, corpus-id, score (+ answer columns where present)
    <out>/<dataset>/docs.parquet                   doc-id, summary (where present)

Run:  uv run python scripts/download_vidore_v2.py "raw dataset v2"
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm

DATASETS = [
    "vidore/economics_reports_v2",
    "vidore/esg_reports_v2",
    "vidore/esg_reports_human_labeled_v2",
    "vidore/biomedical_lectures_v2",
]
# Datasets whose corpus is the same page set as another one: identical pages are hard-linked, not stored twice.
TWINS = {"esg_reports_human_labeled_v2": "esg_reports_v2"}


def download(name: str, out: Path) -> dict:
    short = name.split("/")[-1]
    root = out / short
    (root / "pages").mkdir(parents=True, exist_ok=True)
    configs = get_dataset_config_names(name)
    summary = {"dataset": name, "configs": configs}

    for cfg in configs:
        if cfg == "corpus":
            continue
        df = load_dataset(name, cfg, split="test").to_pandas()
        df.to_parquet(root / f"{cfg}.parquet")
        summary[cfg] = len(df)

    corpus = load_dataset(name, "corpus", split="test")
    twin_dir = out / TWINS[short] / "pages" if short in TWINS else None   # same corpus already on disk?
    rows, linked, t0 = [], 0, time.time()
    for ex in tqdm(corpus, desc=short, unit="page", mininterval=5):
        cid, img = int(ex["corpus-id"]), ex["image"]
        did = ex.get("doc-id") or (Path(ex["pdf_url"]).stem if ex.get("pdf_url") else f"{ex.get('company', 'doc')}_{ex.get('date', '')}")
        path = root / "pages" / f"{cid:04d}.png"
        if not path.exists():
            buf = io.BytesIO(); img.save(buf, format="PNG"); data = buf.getvalue()
            twin = twin_dir / path.name if twin_dir else None
            if twin and twin.exists() and twin.read_bytes() == data:
                os.link(twin, path); linked += 1          # byte-identical page: hard link, no second copy
            else:
                path.write_bytes(data)
        row = {"corpus_id": cid, "doc_id": did, "width": img.width, "height": img.height,
               "image_path": str(path.relative_to(out))}
        row.update({k: ex[k] for k in ("pdf_url", "company", "date") if k in ex})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("corpus_id").reset_index(drop=True)
    df["page_in_doc"] = df.groupby("doc_id").cumcount() + 1
    df.to_parquet(root / "corpus.parquet")
    summary.update(pages=len(df), documents=int(df.doc_id.nunique()), hard_linked=linked,
                   page_mb=round(sum(p.stat().st_size for p in (root / "pages").glob("*.png")) / 1e6, 1),
                   seconds=round(time.time() - t0, 1))
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "raw dataset v2").resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = [n if "/" in n else f"vidore/{n}" for n in sys.argv[2:]] or DATASETS
    summaries = []
    for name in names:
        summaries.append(download(name, out))
        (out / "manifest.json").write_text(json.dumps(summaries, indent=1))   # updated after every dataset
    print(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    main()
