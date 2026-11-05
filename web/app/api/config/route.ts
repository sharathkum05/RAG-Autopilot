import { proxy } from "@/lib/proxy";
import { NextResponse } from "next/server";

/** What the pipeline runs with. The backend serves the same shape from its own settings so this view cannot drift. */
export async function GET(req: Request) {
  const forwarded = await proxy(req, "/api/config");
  if (forwarded) return forwarded;
  return NextResponse.json({
    pipeline: [
      { stage: "Parse", model: "Docling 2.146 + EasyOCR 1.7", detail: "layout model, TableFormer (accurate), full-page OCR for image pages", runs_on: "Modal L4" },
      { stage: "Chunk", model: "structural chunker", detail: "sections ≈400 tokens, tables whole (header repeated when split by rows), figures = caption, heading path prepended for embedding", runs_on: "Modal CPU" },
      { stage: "Text embedding", model: "BAAI/bge-m3", detail: "1024-d dense, normalised; max 1024 tokens", runs_on: "Modal L4" },
      { stage: "Sparse", model: "BM25 (Qdrant/bm25 via FastEmbed)", detail: "sparse vectors with IDF modifier", runs_on: "Modal CPU" },
      { stage: "Visual embedding", model: "vidore/colqwen2.5-v0.2", detail: "≈780 × 128-d multivectors per page + one pooled vector", runs_on: "Modal L4 · bf16" },
      { stage: "Rerank", model: "BAAI/bge-reranker-v2-m3", detail: "cross-encoder on (question, page text), top 20 → top 5", runs_on: "Modal L4" },
      { stage: "Answer", model: "claude-opus-5-5", detail: "page text as citable documents + page images; native citations; server-side refusal fallback", runs_on: "Anthropic API" },
    ],
    stores: [
      { name: "Qdrant", role: "vector store", detail: "matt_pages (colqwen multivector MaxSim + pooled) · matt_chunks (dense + bm25 sparse)", location: "Qdrant Cloud" },
      { name: "Supabase Postgres", role: "metadata", detail: "documents, pages, chunks, conversations, turns, eval runs, timings", location: "Supabase" },
      { name: "Supabase Storage", role: "page images", detail: "originals + display copies, served to the UI", location: "Supabase" },
      { name: "Langfuse", role: "traces", detail: "one trace per turn: encode, retrieve, fuse, rerank, answer spans; tokens and cost", location: "Langfuse Cloud" },
    ],
    retrieval: { modes: ["bm25", "text_hybrid", "text_hybrid_rerank", "visual", "fused", "fused_rerank"], default: "text_hybrid_rerank", top_k_per_path: 50, fused_k: 20, final_k: 5, rrf_k: 60 },
    answer_models: ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
  });
}
