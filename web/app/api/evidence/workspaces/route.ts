import { proxy } from "@/lib/proxy";
import { NextResponse } from "next/server";

export async function GET(req: Request) {
  const forwarded = await proxy(req, "/api/evidence/workspaces");
  if (forwarded) return forwarded;
  return NextResponse.json({ workspaces: [] });
}
