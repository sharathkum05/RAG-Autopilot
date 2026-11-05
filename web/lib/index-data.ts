import { readFile } from "node:fs/promises";
import path from "node:path";

/** Shapes served by the index endpoints. The backend returns the same shapes from Supabase + Qdrant. */
export interface IndexDocument {
  id: string;
  name: string;
  source: string;
  pages: number;
  chunks: number;
  indexed_at: string;
  collections: { pages: string; chunks: string };
}
export interface IndexPage {
  id: number;
  pno: number;
  width: number;
  height: number;
  image: string;
  n_chunks: number;
  n_visual_vectors: number;
  n_words: number;
}
export interface IndexChunk {
  id: string;
  page: number;
  pno: number;
  kind: "text" | "table" | "figure";
  heading: string;
  text: string;
  n_tokens: number;
  vector_id: string;
  /** [x1, y1, x2, y2], normalised 0..1 by the backend (absent in mock data) */
  bbox?: number[] | null;
}
export interface IndexData {
  document: IndexDocument;
  pages: IndexPage[];
  chunks: IndexChunk[];
}

let cache: Promise<IndexData> | null = null;

/** Mock store: one JSON file built by scripts/build_index_json.py. Read once per server process. */
export function loadIndex(): Promise<IndexData> {
  if (!cache) {
    cache = readFile(path.join(process.cwd(), "public", "data", "python_textbook.json"), "utf8").then((s) => JSON.parse(s) as IndexData);
  }
  return cache;
}

export const tokenize = (s: string) => (s.toLowerCase().match(/[a-z0-9_.()\\]+/g) ?? []).filter((t) => t.length > 1);

/** Deterministic 0..1 noise so mock dense/visual scores are stable per (query, item). */
export function noise(seed: string): number {
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) { h ^= seed.charCodeAt(i); h = Math.imul(h, 16777619); }
  return ((h >>> 0) % 10000) / 10000;
}
