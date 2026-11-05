import { proxy } from "@/lib/proxy";
import { NextResponse } from "next/server";

export async function GET(req: Request) {
  const forwarded = await proxy(req, "/api/observability");
  if (forwarded) return forwarded;
  return NextResponse.json({ source: "none", window: "7d", summary: null, traces: [] });
}
