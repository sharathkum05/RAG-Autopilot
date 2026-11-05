import { proxy } from "@/lib/proxy";
import { NextResponse } from "next/server";

export async function GET(req: Request) {
  const forwarded = await proxy(req, "/api/evals/queries");
  if (forwarded) return forwarded;
  return NextResponse.json({ workspace: "python_textbook", baseline: "text_hybrid", improvement: "text_hybrid_rerank", queries: [] });
}
