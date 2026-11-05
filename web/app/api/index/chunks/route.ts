import { proxy } from "@/lib/proxy";
import { NextResponse } from "next/server";
import { loadIndex } from "@/lib/index-data";

export async function GET(req: Request) {
  const forwarded = await proxy(req, "/api/index/chunks");
  if (forwarded) return forwarded;
  const { chunks } = await loadIndex();
  const q = new URL(req.url).searchParams;
  const page = q.get("page");
  const ids = q.get("ids");
  if (page) return NextResponse.json({ chunks: chunks.filter((c) => c.page === Number(page)) });
  if (ids) { const set = new Set(ids.split(",")); return NextResponse.json({ chunks: chunks.filter((c) => set.has(c.id)) }); }
  return NextResponse.json({ total: chunks.length, chunks: chunks.slice(0, 50) });
}
