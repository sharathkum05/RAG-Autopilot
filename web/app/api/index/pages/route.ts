import { proxy } from "@/lib/proxy";
import { NextResponse } from "next/server";
import { loadIndex } from "@/lib/index-data";

export async function GET(req: Request) {
  const forwarded = await proxy(req, "/api/index/pages");
  if (forwarded) return forwarded;
  const { pages } = await loadIndex();
  const id = new URL(req.url).searchParams.get("id");
  if (id) {
    const page = pages.find((p) => p.id === Number(id));
    return page ? NextResponse.json(page) : NextResponse.json({ error: "unknown page" }, { status: 404 });
  }
  return NextResponse.json({ pages });
}
