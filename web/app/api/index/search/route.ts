import { proxy } from "@/lib/proxy";
import { NextResponse } from "next/server";
import { loadIndex, tokenize } from "@/lib/index-data";

/** Plain text filter over chunk text: every query term must appear. Ranked by term frequency. */
export async function GET(req: Request) {
  const forwarded = await proxy(req, "/api/index/search");
  if (forwarded) return forwarded;
  const { chunks } = await loadIndex();
  const q = (new URL(req.url).searchParams.get("q") ?? "").trim();
  const terms = tokenize(q);
  if (!terms.length) return NextResponse.json({ query: q, total: 0, results: [] });
  const scored = chunks
    .map((c) => { const t = c.text.toLowerCase(); const hits = terms.map((w) => t.split(w).length - 1); return { c, hits, score: hits.reduce((a, b) => a + b, 0) }; })
    .filter((r) => r.hits.every((h) => h > 0))
    .sort((a, b) => b.score - a.score);
  return NextResponse.json({ query: q, terms, total: scored.length, results: scored.slice(0, 40).map((r) => ({ ...r.c, matches: r.score })) });
}
