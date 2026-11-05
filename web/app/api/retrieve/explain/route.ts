import { proxy } from "@/lib/proxy";
import { NextResponse } from "next/server";
import { loadIndex, noise, tokenize, type IndexChunk } from "@/lib/index-data";

/**
 * Mock "explain a query": real BM25-style scoring over the chunk text, stable pseudo scores for the dense and visual
 * paths, then the exact fusion and rerank steps the pipeline uses, so the mechanics on screen are the real ones.
 * The backend returns the same shape with true scores from Qdrant and the reranker.
 */
const RRF_K = 60;
interface Hit { id: string; page: number; pno: number; score: number; chunk?: string; preview?: string }

function rrf(lists: { name: string; hits: Hit[] }[]) {
  const acc = new Map<number, { page: number; pno: number; score: number; from: Record<string, number>; chunk?: string; preview?: string }>();
  for (const { name, hits } of lists) {
    hits.forEach((h, rank) => {
      const c = 1 / (RRF_K + rank + 1);
      const cur = acc.get(h.page) ?? { page: h.page, pno: h.pno, score: 0, from: {}, chunk: h.chunk, preview: h.preview };
      cur.score += c; cur.from[name] = rank + 1; cur.chunk ??= h.chunk; cur.preview ??= h.preview;
      acc.set(h.page, cur);
    });
  }
  return [...acc.values()].sort((a, b) => b.score - a.score);
}
const toPages = (hits: Hit[], k: number) => { const seen = new Set<number>(); const out: Hit[] = []; for (const h of hits) { if (!seen.has(h.page)) { seen.add(h.page); out.push(h); if (out.length >= k) break; } } return out; };
const preview = (c: IndexChunk) => (c.heading ? c.heading + " · " : "") + c.text.slice(0, 160).replace(/\s+/g, " ");

export async function POST(req: Request) {
  const forwarded = await proxy(req, "/api/retrieve/explain");
  if (forwarded) return forwarded;
  const { question = "", k = 5 } = (await req.json()) as { question?: string; k?: number };
  const { chunks, pages } = await loadIndex();
  const terms = tokenize(question);
  const t0 = Date.now();
  if (!terms.length) return NextResponse.json({ error: "empty question" }, { status: 400 });

  // BM25-lite: tf * idf over chunk text.
  const N = chunks.length;
  const df = new Map<string, number>();
  const toks = chunks.map((c) => tokenize(c.text));
  for (const w of terms) df.set(w, toks.filter((t) => t.includes(w)).length);
  const idf = (w: string) => Math.log(1 + (N - (df.get(w) ?? 0) + 0.5) / ((df.get(w) ?? 0) + 0.5));
  const bm25All: Hit[] = chunks.map((c, i) => {
    const len = toks[i].length || 1;
    let s = 0;
    for (const w of terms) { const tf = toks[i].filter((t) => t === w).length; if (tf) s += idf(w) * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * len / 180)); }
    return { id: c.id, page: c.page, pno: c.pno, score: +s.toFixed(3), chunk: c.id, preview: preview(c) };
  }).filter((h) => h.score > 0).sort((a, b) => b.score - a.score);
  const bm25Max = bm25All[0]?.score ?? 1;

  // Dense (mock): lexical overlap blended with stable noise, scaled to a cosine range.
  const lexById = new Map(bm25All.map((h) => [h.id, h.score / bm25Max]));
  const dense: Hit[] = chunks.map((c) => {
    const s = 0.25 + 0.45 * (lexById.get(c.id) ?? 0) + 0.3 * noise(question + "|d|" + c.id);
    return { id: c.id, page: c.page, pno: c.pno, score: +s.toFixed(3), chunk: c.id, preview: preview(c) };
  }).sort((a, b) => b.score - a.score);

  // Visual (mock): page-level MaxSim-like score from the page's best lexical chunk plus noise.
  const bestLexByPage = new Map<number, number>();
  for (const h of bm25All) bestLexByPage.set(h.page, Math.max(bestLexByPage.get(h.page) ?? 0, h.score / bm25Max));
  const visual: Hit[] = pages.map((p) => ({ id: `page-${p.id}`, page: p.id, pno: p.pno, score: +(12 + 7 * (bestLexByPage.get(p.id) ?? 0) + 4 * noise(question + "|v|" + p.id)).toFixed(2) }))
    .sort((a, b) => b.score - a.score);

  const K = 10;
  const bm25Pages = toPages(bm25All, K), densePages = toPages(dense, K), visualPages = visual.slice(0, K);
  const textHybrid = rrf([{ name: "dense", hits: densePages }, { name: "bm25", hits: bm25Pages }]).slice(0, K);
  const fused = rrf([{ name: "visual", hits: visualPages }, { name: "text", hits: textHybrid.map((h) => ({ id: `page-${h.page}`, page: h.page, pno: h.pno, score: h.score, chunk: h.chunk, preview: h.preview })) }]);
  const fmax = fused[0]?.score ?? 1;
  const reranked = fused.slice(0, K).map((h, i) => ({ ...h, rerank: +(0.35 + 0.45 * (h.score / fmax) + 0.2 * noise(question + "|r|" + h.page) - 0.02 * i).toFixed(3) })).sort((a, b) => b.rerank - a.rerank);

  return NextResponse.json({
    question, terms, k, timings_ms: { encode: 38, visual: 97, dense: 44, bm25: 12, fusion: 1, rerank: 388, total: Date.now() - t0 + 580 },
    paths: { visual: visualPages, dense: densePages, bm25: bm25Pages },
    text_hybrid: textHybrid, fused: fused.slice(0, K), reranked, final: reranked.slice(0, k),
  });
}
