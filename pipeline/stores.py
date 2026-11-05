"""Hosted stores: Qdrant Cloud for vectors, Supabase Postgres for metadata and job state, Supabase Storage for page images.
Nothing here writes to local disk."""
from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from pipeline.config import settings

NS = uuid.UUID("5a4b3c2d-1e0f-4a9b-8c7d-6e5f4a3b2c1d")


def point_id(key: str) -> str:
    """Qdrant wants UUIDs; derive one deterministically from our string ids."""
    return str(uuid.uuid5(NS, key))


# --------------------------------------------------------------------------- Qdrant

@lru_cache
def qdrant() -> QdrantClient:
    s = settings()
    if not s.qdrant_url:
        raise RuntimeError("QDRANT_URL is not set")
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None, timeout=120)


def ensure_collections() -> None:
    s, c = settings(), qdrant()
    if not c.collection_exists(s.pages_collection):
        c.create_collection(s.pages_collection, vectors_config={
            "colqwen": models.VectorParams(size=s.visual_dims, distance=models.Distance.DOT, on_disk=True,
                                           multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                                           hnsw_config=models.HnswConfigDiff(m=0)),
            "pooled": models.VectorParams(size=s.visual_dims, distance=models.Distance.DOT),
        })
        c.create_payload_index(s.pages_collection, "workspace", models.PayloadSchemaType.KEYWORD)
        c.create_payload_index(s.pages_collection, "document_id", models.PayloadSchemaType.KEYWORD)
    if not c.collection_exists(s.chunks_collection):
        c.create_collection(s.chunks_collection,
                            vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.DOT)},
                            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)})
        c.create_payload_index(s.chunks_collection, "workspace", models.PayloadSchemaType.KEYWORD)
        c.create_payload_index(s.chunks_collection, "document_id", models.PayloadSchemaType.KEYWORD)


def _ws_filter(workspace: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="workspace", match=models.MatchValue(value=workspace))])


def upsert_pages(points: list[dict[str, Any]]) -> None:
    """points: {id, multivector (n×128), pooled (128), payload}"""
    s, c = settings(), qdrant()
    for i in range(0, len(points), 8):
        batch = [models.PointStruct(id=p["id"], vector={"colqwen": np.asarray(p["multivector"]).tolist(), "pooled": np.asarray(p["pooled"]).tolist()}, payload=p["payload"]) for p in points[i:i + 8]]
        c.upsert(s.pages_collection, batch, wait=True)


def upsert_chunks(points: list[dict[str, Any]]) -> None:
    """points: {id, dense (1024), sparse (indices, values), payload}"""
    s, c = settings(), qdrant()
    for i in range(0, len(points), 128):
        batch = [models.PointStruct(id=p["id"], vector={"dense": np.asarray(p["dense"]).tolist(), "bm25": models.SparseVector(indices=p["sparse"][0], values=p["sparse"][1])}, payload=p["payload"]) for p in points[i:i + 128]]
        c.upsert(s.chunks_collection, batch, wait=True)


def delete_document_vectors(document_id: str) -> None:
    s, c = settings(), qdrant()
    f = models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))])
    for col in (s.pages_collection, s.chunks_collection):
        if c.collection_exists(col):
            c.delete(col, points_selector=models.FilterSelector(filter=f))


def search_visual(qv: np.ndarray, qp: np.ndarray, workspace: str, k: int) -> list[tuple[str, float, dict]]:
    """Pooled prefetch narrows to 4k candidates; MaxSim over the multivectors ranks them. Returns (page_id, score, payload)."""
    s, c = settings(), qdrant()
    res = c.query_points(s.pages_collection, prefetch=models.Prefetch(query=qp.tolist(), using="pooled", limit=max(4 * k, 100), filter=_ws_filter(workspace)),
                         query=qv.tolist(), using="colqwen", limit=k, with_payload=True, query_filter=_ws_filter(workspace))
    return [(p.payload["page_id"], float(p.score), p.payload) for p in res.points]


def search_text(dense: np.ndarray | None, sparse: tuple[list[int], list[float]] | None, workspace: str, limit: int) -> list[tuple[str, float, dict]]:
    """Chunk-level search: dense and/or BM25, fused server-side by RRF when both are on. Returns (chunk_id, score, payload) best-first."""
    s, c = settings(), qdrant()
    pre = []
    if dense is not None:
        pre.append(models.Prefetch(query=dense.tolist(), using="dense", limit=limit, filter=_ws_filter(workspace)))
    if sparse is not None:
        pre.append(models.Prefetch(query=models.SparseVector(indices=sparse[0], values=sparse[1]), using="bm25", limit=limit, filter=_ws_filter(workspace)))
    if not pre:
        return []
    if len(pre) == 1:
        res = c.query_points(s.chunks_collection, query=pre[0].query, using=pre[0].using, limit=limit, with_payload=True, query_filter=_ws_filter(workspace))
    else:
        res = c.query_points(s.chunks_collection, prefetch=pre, query=models.FusionQuery(fusion=models.Fusion.RRF), limit=limit, with_payload=True)
    return [(p.payload["chunk_id"], float(p.score), p.payload) for p in res.points]


def qdrant_ping() -> dict:
    s, c = settings(), qdrant()
    cols = {x.name for x in c.get_collections().collections}
    counts = {col: c.count(col).count for col in (s.pages_collection, s.chunks_collection) if col in cols}
    return {"collections": sorted(cols), "counts": counts}


# --------------------------------------------------------------------------- Supabase

@lru_cache
def supabase():
    from supabase import create_client

    s = settings()
    if not s.supabase_url or not s.supabase_service_key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY are not set")
    return create_client(s.supabase_url, s.supabase_service_key)


def sb_upsert(table: str, rows: list[dict] | dict) -> None:
    if not rows:
        return
    supabase().table(table).upsert(rows).execute()


def sb_select(table: str, **eq: Any) -> list[dict]:
    q = supabase().table(table).select("*")
    for k, v in eq.items():
        q = q.eq(k, v)
    return q.execute().data or []


def sb_get(table: str, id: str) -> dict | None:
    rows = supabase().table(table).select("*").eq("id", id).limit(1).execute().data
    return rows[0] if rows else None


def sb_update(table: str, id: str, patch: dict) -> None:
    supabase().table(table).update(patch).eq("id", id).execute()


def sb_delete_document(document_id: str) -> None:
    supabase().table("documents").delete().eq("id", document_id).execute()   # cascades to pages and chunks


def delete_workspace_images(workspace: str) -> int:
    """Removes every page JPEG under `<workspace>/` in the pages bucket (two levels: workspace/document/page.jpg). Returns the count."""
    s = settings()
    bucket = supabase().storage.from_(s.supabase_bucket)
    paths: list[str] = []
    for entry in bucket.list(workspace) or []:
        name = entry.get("name")
        if not name:
            continue
        if entry.get("id") is None:                                 # a folder (one per document)
            paths += [f"{workspace}/{name}/{f['name']}" for f in (bucket.list(f"{workspace}/{name}") or []) if f.get("name")]
        else:
            paths.append(f"{workspace}/{name}")
    for i in range(0, len(paths), 100):
        bucket.remove(paths[i:i + 100])
    return len(paths)


def upload_page_image(path: str, data: bytes) -> str:
    """Uploads a JPEG to the pages bucket and returns its public URL."""
    s = settings()
    bucket = supabase().storage.from_(s.supabase_bucket)
    bucket.upload(path, data, {"content-type": "image/jpeg", "upsert": "true"})
    return bucket.get_public_url(path)


def supabase_ping() -> dict:
    docs = supabase().table("documents").select("id", count="exact").limit(1).execute()
    return {"documents": docs.count}


def page_texts(page_ids: list[str]) -> dict[str, str]:
    if not page_ids:
        return {}
    rows = supabase().table("pages").select("id,text").in_("id", page_ids).execute().data or []
    return {r["id"]: r.get("text") or "" for r in rows}


def pages_by_ids(page_ids: list[str]) -> dict[str, dict]:
    if not page_ids:
        return {}
    rows = supabase().table("pages").select("*").in_("id", page_ids).execute().data or []
    return {r["id"]: r for r in rows}


def chunks_by_ids(chunk_ids: list[str]) -> dict[str, dict]:
    if not chunk_ids:
        return {}
    rows = supabase().table("chunks").select("*").in_("id", chunk_ids).execute().data or []
    return {r["id"]: r for r in rows}
