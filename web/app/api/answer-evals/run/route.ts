import { proxy } from "@/lib/proxy";
import { NextResponse } from "next/server";

export async function GET(req: Request) {
  const forwarded = await proxy(req, "/api/answer-evals/run");
  if (forwarded) return forwarded;
  return NextResponse.json({ error: "no backend" }, { status: 404 });
}
