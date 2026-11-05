"""Carve one document out of a downloaded ViDoRe V3 domain into its own raw-dataset folder.

Keeps: that document's pages (hard-linked, no second copy), corpus rows (with the shipped OCR markdown),
queries whose relevant pages all lie in the document (all languages of those query ids), their qrels
(with bounding boxes), and the document's metadata row.

Run:  uv run python scripts/make_subset_v3.py "raw dataset v3/computer_science" Introduction_to_Python_Programming "raw dataset v3/python_textbook"
"""
import json, os, sys
from pathlib import Path
import pandas as pd

src, doc, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
(out / "pages").mkdir(parents=True, exist_ok=True)
corpus = pd.read_parquet(src / "corpus.parquet"); qrels = pd.read_parquet(src / "qrels.parquet")
queries = pd.read_parquet(src / "queries.parquet"); meta = pd.read_parquet(src / "documents_metadata.parquet")

c = corpus[corpus.doc_id == doc].sort_values("page_number_in_doc").reset_index(drop=True)
in_doc = set(c.corpus_id)
per_q = qrels.groupby("query_id").corpus_id.agg(lambda s: set(s) <= in_doc)
keep_q = set(per_q[per_q].index)
q = queries[queries.query_id.isin(keep_q)].reset_index(drop=True)
qr = qrels[qrels.query_id.isin(keep_q)].reset_index(drop=True)

linked = 0
for r in c.itertuples():
    dst = out / "pages" / Path(r.image_path).name
    if not dst.exists():
        try: os.link(src.parent / r.image_path, dst); linked += 1
        except OSError: dst.write_bytes((src.parent / r.image_path).read_bytes())
c = c.assign(image_path=[str((out / "pages" / Path(p).name).relative_to(out.parent)) for p in c.image_path])
c.to_parquet(out / "corpus.parquet"); q.to_parquet(out / "queries.parquet"); qr.to_parquet(out / "qrels.parquet")
meta[meta.doc_id == doc].to_parquet(out / "documents_metadata.parquet")
summary = {"source": str(src), "document": doc, "pages": len(c), "queries_all_languages": len(q),
           "queries_english": int((q.language.str.lower() == "english").sum()), "qrels": len(qr),
           "labelled_pages": int(qr.corpus_id.nunique()), "pages_hard_linked": linked,
           "languages": sorted(q.language.unique().tolist())}
(out / "manifest.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))
