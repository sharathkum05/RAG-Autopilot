import { proxy } from "@/lib/proxy";
import { NextResponse } from "next/server";

export async function GET(req: Request) {
  const forwarded = await proxy(req, "/api/answer-evals");
  if (forwarded) return forwarded;
  return NextResponse.json({ configs: [], latest: [], runs: [], baseline: "text_hybrid", improvement: "text_hybrid_rerank", info: {} });
}
