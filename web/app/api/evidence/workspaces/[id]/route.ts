import { proxy } from "@/lib/proxy";
import { NextResponse } from "next/server";

/** Delete a Playground conversation and everything indexed for it. Only the backend can do this; the mock has nothing to delete. */
export async function DELETE(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const forwarded = await proxy(req, `/api/evidence/workspaces/${encodeURIComponent(id)}`);
  if (forwarded) return forwarded;
  return NextResponse.json({ error: "no backend configured" }, { status: 501 });
}
