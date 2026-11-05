import { proxy } from "@/lib/proxy";
import { NextResponse } from "next/server";
import { loadIndex } from "@/lib/index-data";

export async function GET(req: Request) {
  const forwarded = await proxy(req, "/api/index/documents");
  if (forwarded) return forwarded;
  const { document } = await loadIndex();
  return NextResponse.json({ documents: [document] });
}
